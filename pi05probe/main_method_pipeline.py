"""Per-task LIBERO main method: fixed render-proxy EOT, then paired real rollouts.

Uses the seven existing visibility-selected calibrations. Completed compatible
EOT artifacts are evaluated first; each remaining task is trained and evaluated
before its GPU slot takes another task. No nominal/ablation jobs are scheduled.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import threading

import numpy as np
import multitask_pipeline as P
import visible_pos_pipeline as V
from main_method_job_runtime import install as install_job_runtime

SOURCE = P.ROOT/'out/visible_pos'
DEST = P.ROOT/'out/main_method'
TASKS = tuple(P.TRAIN_TASKS+P.HELDOUT_TASKS)
CONDITIONS = ('clean', 'random', 'eot')
INDICES = tuple(range(40, 50))
FIT = P.ROOT/'out/dual_view/fit'
RANDOM = P.ROOT/'out/dual_view/manifest_both/random.png'
_HASHES = {}
_HASH_LOCK = threading.Lock()
_OWNS_OUTPUT = False
EOT_RANGES = dict(angle_deg=[-15., 15.], dx_fraction=[-.05, .05],
                  dy_fraction=[-.05, .05], scale=[.9, 1.1], gain=[.9, 1.1])


def file_sha(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _HASH_LOCK:
        if key in _HASHES:
            return _HASHES[key]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    value = digest.hexdigest()
    with _HASH_LOCK:
        _HASHES[key] = value
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_path(path):
    # These are the established local and NFS mount names of this project.
    value = str(path)
    for prefix in ['/shared/user65/workspace/chence/WAMattack',
                   '/home/user1/workspace/chence/WAMattack']:
        if value == prefix or value.startswith(prefix+'/'):
            return str(P.ROOT.parent)+value[len(prefix):]
    return str(Path(path).resolve())


def calibration(task, selection):
    folder = SOURCE/'tasks'/task
    meta = read_json(folder/'capture.json')
    require(meta.get('complete') is True and meta.get('task') == task, 'incomplete calibration: '+task)
    require(np.allclose(meta['cell_world'][:2], selection[task]['world'], atol=1e-7, rtol=0),
            'calibration differs from frozen selected position: '+task)
    require(V.scan_cache_reason(SOURCE/'scan'/task) is None, 'invalid visibility scan: '+task)
    for view in ['base', 'wrist']:
        with np.load(folder/(view+'.npz'), allow_pickle=False) as data:
            require(float(data['patch_m']) == .13 and np.array_equal(data['cell_world'], meta['cell_world']),
                    'calibration geometry mismatch: '+task+'/'+view)
            ep, train, valid = data['ep_id'], data['train_mask'], data['validation_mask']
            require(len(ep) == meta['frames'] and np.array_equal(train, ep < 10)
                    and np.array_equal(valid, ep >= 10) and set(ep) <= set(range(12)),
                    'invalid calibration train/validation split: '+task+'/'+view)
    return dict(folder=str(folder), cell_world=meta['cell_world'], frames=meta['frames'],
                train_frames=meta['train_frames'], validation_frames=meta['validation_frames'],
                capture_sha256=file_sha(folder/'capture.json'), init_states_sha256=meta['init_states_sha256'],
                scan_sha256=file_sha(SOURCE/'scan'/task/'visibility.json'),
                calibration_sha256={v: file_sha(folder/(v+'.npz')) for v in ['base', 'wrist']})


def training_reason(task, folder, provenance):
    """None means a finished artifact matches the fixed main-method recipe."""
    try:
        config, result = read_json(folder/'config.json'), read_json(folder/'results.json')
        expected = dict(schema='wam-multitask-texture-train-v1', complete=True, mode='both', tasks=[task], iters=500, batch=8,
                        per_task_per_batch=8, lr=.01, seed_batch=20261005,
                        noise_seed=20260901, init='constant gray .5', patch_m=.13,
                        proxy_rmse_max=.02)
        for key, value in expected.items():
            if config.get(key) != value:
                return 'training config differs: '+key
        if canonical_path(config['checkpoint']) != str(P.ROOT.parent/'checkpoints/pi05_libero_pytorch'):
            return 'checkpoint path differs'
        if config.get('calibration_sha256', {}).get(task) != provenance['calibration_sha256']:
            return 'training calibration hashes differ'
        if config.get('cell_world_by_task', {}).get(task) != provenance['cell_world']:
            return 'training position differs'
        if config.get('fit_sha256') != {v: file_sha(FIT/(v+'.json')) for v in ['base', 'wrist']}:
            return 'proxy fit hashes differ'
        for name in ['multitask_train.py', 'dual_view_proxy.py']:
            if config.get('source_sha256', {}).get(name) != file_sha(P.ROOT/name):
                return 'training source differs: '+name
        if config.get('source_eot_sha256') != file_sha(P.ROOT.parent/'robotwinprobe/texture_eot.py'):
            return 'EOT source differs'
        eot = config.get('eot') or {}
        if eot.get('seed') != 2026102901 or eot.get('ranges') != EOT_RANGES:
            return 'missing fixed EOT recipe'
        if config.get('train_frames', {}).get(task) != provenance['train_frames'] or config.get('validation_frames', {}).get(task) != provenance['validation_frames']:
            return 'training frame split differs'
        if result.get('complete') is not True or result.get('iterations') != 500 or result.get('gradient_frame_forwards') != 4000:
            return 'training result is incomplete'
        if result.get('texture_sha256') != file_sha(folder/'texture_0500.png'):
            return 'final texture hash differs'
        for view in ['base', 'wrist']:
            check = config.get('proxy_check', {}).get(task+'/'+view, {})
            error = check.get('mask_RMSE_max')
            if view == 'base' and check.get('frames', 0) <= 0:
                return 'missing base proxy validation'
            if error is not None and (not np.isfinite(error) or error > .02):
                return 'proxy validation failed: '+view
    except (OSError, ValueError, KeyError, TypeError) as error:
        return 'missing or unreadable training artifact: '+str(error)
    return None


def train_candidates(task):
    return [SOURCE/'train'/task/'eot', DEST/'train'/task/'eot']


def compatible_training(task, provenance):
    return next((p for p in train_candidates(task) if training_reason(task, p, provenance) is None), None)


def archive_interrupted(folder):
    if not folder.exists():
        return
    for name in ['config.json', 'results.json']:
        if (folder/name).exists() and read_json(folder/name).get('complete') is True:
            raise ValueError('preserving completed incompatible training artifact: '+str(folder))
    target = DEST/'recovery'/('interrupted_'+V.now().replace(':', '').replace('.', ''))/folder.parent.name/folder.name
    target.parent.mkdir(parents=True, exist_ok=True)
    folder.rename(target)
    V.event('interrupted_training_archived', path=str(target), original=str(folder))


def mark(task, status, **extra):
    with V._STATE_LOCK:
        V._STATE.setdefault('tasks', {}).setdefault(task, {}).update(status=status, updated=V.now(), **extra)
        V.write_json(DEST/'progress.json', dict(V._STATE, updated=V.now()))
    print('[task]', task, status, extra, flush=True)


def ensure_training(task, gpu, provenance):
    old = compatible_training(task, provenance)
    if old is not None:
        mark(task, 'trained_reused', train_run=str(old), texture_sha256=file_sha(old/'texture_0500.png'))
        return old
    old_folder, new_folder = train_candidates(task)
    # Preserve any completed incompatible experiment. Only interrupted runs move.
    folder = old_folder
    if any((old_folder/name).exists() and read_json(old_folder/name).get('complete') is True
           for name in ['config.json', 'results.json']):
        folder = new_folder
    archive_interrupted(folder)
    mark(task, 'training', train_run=str(folder))
    with V.allocation('main_train_'+task, gpu, 32000):
        V.run([P.MODEL, 'multitask_train.py', '--tasks-root', str(SOURCE/'tasks'), '--tasks', task,
               '--out', str(folder), '--eot', '--iters', '500', '--batch', '8', '--proxy-rmse-max', '0.02'],
              'train_'+task+'_eot.log', {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu['uuid'], 'OMP_NUM_THREADS': '4'}, 14400)
    reason = training_reason(task, folder, provenance)
    require(reason is None, 'new training failed provenance validation: '+str(reason))
    mark(task, 'trained', train_run=str(folder), texture_sha256=file_sha(folder/'texture_0500.png'))
    return folder


def make_manifest(task, train, provenance, selection):
    textures = dict(clean=None, random=str(RANDOM), eot=str(train/'texture_0500.png'))
    return dict(source_task=task, stage='main_method_per_task_eot', train_run=str(train),
                geometry=dict(patch_m=.13, cell_world=provenance['cell_world']), selection=selection[task],
                textures=textures, texture_sha256={k: None if v is None else file_sha(v) for k, v in textures.items()},
                conditions=list(CONDITIONS), indices=list(INDICES),
                train_config_sha256=file_sha(train/'config.json'), train_results_sha256=file_sha(train/'results.json'),
                calibration=provenance, metric='task-local clean-success paired ASR; clean/random/eot, init40-49')


def write_immutable(path, value):
    if path.exists():
        require(read_json(path) == value, 'refusing to overwrite changed frozen metadata: '+str(path))
    else:
        V.write_json(path, value)


def episode_summary(path, task, manifest, require_complete=False):
    """Never merge attempts or present a partial denominator as final ASR."""
    data = read_json(path)
    metadata, groups = data['metadata'], data['per_episode']
    require(metadata['task'] == task and metadata['indices'] == list(INDICES), 'wrong rollout task/test indices')
    require(metadata['manifest_sha256'] == file_sha(manifest), 'rollout manifest hash changed')
    frozen = read_json(manifest)
    require(metadata.get('clean_reference') is None and metadata['geometry'] == frozen['geometry'],
            'unpaired clean reference or wrong rollout geometry')
    require(set(groups) == set(CONDITIONS), 'wrong rollout conditions')
    require(metadata.get('init_states_sha256') == frozen['calibration']['init_states_sha256']
            and metadata.get('max_steps') == 300 and metadata.get('settling_steps') == 10
            and metadata.get('replan') == 5, 'rollout dataset or episode budget differs')
    for name in CONDITIONS:
        texture = metadata.get('textures', {}).get(name)
        require((texture is None and frozen['textures'][name] is None)
                or (isinstance(texture, dict) and texture.get('sha256') == frozen['texture_sha256'][name]),
                'rollout texture provenance differs: '+name)
    rows = {}
    for name in CONDITIONS:
        rows[name] = {}
        for row in groups[name]:
            ep = row['init_index']
            require(ep in INDICES and ep not in rows[name] and type(row['success']) is bool
                    and row['env_seed'] == 10000+ep, 'invalid or duplicate rollout episode')
            replay = Path(path).parent/row['replay_record']
            require(replay.parent == Path(path).parent and replay.is_file(), 'missing/unsafe rollout replay')
            with np.load(replay, allow_pickle=False) as z:
                require(int(z['init_index']) == ep and int(z['env_seed']) == 10000+ep, 'replay episode mismatch')
                noise = z['noise_seeds']
                require(np.array_equal(noise, 20261006+ep*1000+np.arange(len(noise)))
                        and len(noise) == row['inferences'], 'replay noise pairing mismatch')
                state = z['initial_state']
                require(hashlib.sha256(state.tobytes()).hexdigest() == row['initial_state_sha256'], 'replay state hash mismatch')
                row = dict(row, fixture_body_pos=z['fixture_body_pos'], fixture_body_quat=z['fixture_body_quat'])
            rows[name][ep] = row
    for name in CONDITIONS[1:]:
        require(set(rows[name]) <= set(rows['clean']), 'attack episode lacks its task-local clean reference')
        for ep in set(rows[name]) & set(rows['clean']):
            a, b = rows['clean'][ep], rows[name][ep]
            n = min(len(a['fixture_body_pos']), len(b['fixture_body_pos']))
            require(a['initial_state_sha256'] == b['initial_state_sha256']
                    and np.array_equal(a['fixture_body_pos'][:n], b['fixture_body_pos'][:n])
                    and np.array_equal(a['fixture_body_quat'][:n], b['fixture_body_quat'][:n]), 'clean/attack physics pairing mismatch')
    complete = all(set(rows[c]) == set(INDICES) for c in CONDITIONS)
    require(not require_complete or complete, 'rollout has incomplete condition/episode coverage')
    clean_success = {ep for ep, row in rows['clean'].items() if row['success']}
    summary = {}
    for name in CONDITIONS:
        paired = clean_success & set(rows[name])
        failures = sum(not rows[name][ep]['success'] for ep in paired)
        summary[name] = dict(n=len(rows[name]), successes=sum(r['success'] for r in rows[name].values()),
                             clean_success_n=len(clean_success), paired_clean_success_n=len(paired), paired_failures=failures,
                             paired_ASR=failures/len(paired) if complete and paired else None)
    return dict(task=task, complete=complete, indices=list(INDICES), conditions=list(CONDITIONS),
                episodes=str(path), episodes_sha256=file_sha(path), summary=summary)


def evaluate(task, train, gpu, port, provenance, selection):
    folder = DEST/'eval'/task
    manifest = folder/'manifest.json'
    write_immutable(manifest, make_manifest(task, train, provenance, selection))
    attempts = sorted(folder.glob('attempt_*'))
    for out in attempts:
        if not (out/'episodes.json').exists():
            continue
        try:
            result = episode_summary(out/'episodes.json', task, manifest, True)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        V.write_json(out/'complete.json', dict(complete=True, validated=V.now(), episodes_sha256=result['episodes_sha256']))
        V.write_json(DEST/'results'/f'{task}.json', result)
        mark(task, 'complete', result=str(DEST/'results'/f'{task}.json'))
        return
    first = max([int(p.name.split('_')[-1]) for p in attempts]+[-1])+1
    for number in range(first, first+3):
        out = folder/f'attempt_{number:03d}'
        mark(task, 'evaluating', attempt=number, train_run=str(train))
        try:
            with V.allocation('main_eval_'+task+'_attempt'+str(number), gpu, 32000), V.server(gpu, port):
                V.run([P.SIM, 'cross_task_transfer_rollout.py', '--manifest', str(manifest), '--task', task,
                       '--conditions', ','.join(CONDITIONS), '--indices', ','.join(map(str, INDICES)),
                       '--port', str(port), '--out', str(out.relative_to(P.ROOT/'out'))],
                      f'eval_{task}_{number:03d}.log', V.sim_env(gpu), 10800)
            result = episode_summary(out/'episodes.json', task, manifest, True)
            V.write_json(out/'complete.json', dict(complete=True, validated=V.now(), episodes_sha256=result['episodes_sha256']))
            V.write_json(DEST/'results'/f'{task}.json', result)
            mark(task, 'complete', result=str(DEST/'results'/f'{task}.json'))
            return
        except V.ResourceBlocked:
            raise
        except Exception as error:
            V.event('evaluation_attempt_failed', task=task, attempt=number, error=repr(error))
            if (out/'episodes.json').exists():
                try:
                    V.write_json(out/'partial_summary.json', episode_summary(out/'episodes.json', task, manifest))
                except (OSError, ValueError, KeyError, TypeError) as validation_error:
                    V.write_json(out/'invalid.json', dict(error=repr(validation_error), updated=V.now()))
    raise RuntimeError('three fresh paired rollout attempts failed: '+task)


def protocol(provenance, selection):
    return dict(schema='libero-per-task-main-method-v1', method='both-camera render-proxy texture-plane EOT',
                tasks=list(TASKS), conditions=list(CONDITIONS), indices=list(INDICES),
                training=dict(iters=500, batch=8, lr=.01, batch_seed=20261005, noise_seed=20260901,
                              eot_seed=2026102901, proxy_rmse_max=.02),
                metric='per task and condition: attacked failure among the same init40-49 clean successes; undefined if zero clean successes',
                rollout=dict(max_steps=300, settling_steps=10, replan=5, env_seed='10000+init_index',
                             action_noise='20261006+init_index*1000+replan_index', cameras='both actual MuJoCo rendered inputs'),
                selection=selection, selection_sha256=file_sha(SOURCE/'selection.json'), calibration=provenance,
                random_texture_sha256=file_sha(RANDOM),
                source_sha256={n: file_sha(P.ROOT/n) for n in ['multitask_train.py', 'dual_view_proxy.py',
                    'cross_task_transfer_rollout.py', 'render_proxy_policy_server.py']},
                source_eot_sha256=file_sha(P.ROOT.parent/'robotwinprobe/texture_eot.py'))


def main():
    global DEST, _OWNS_OUTPUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpus', required=True, help='JSON list of {host,index,uuid,egl_index}; mappings checked before each allocation')
    ap.add_argument('--tasks', default=','.join(TASKS), help='subset of the fixed seven tasks to execute now')
    ap.add_argument('--port-base', type=int, default=8461)
    ap.add_argument('--out', type=Path, default=DEST)
    args = ap.parse_args()
    DEST = args.out.resolve()
    require(DEST.is_relative_to(P.ROOT/'out') if hasattr(DEST, 'is_relative_to') else str(DEST).startswith(str(P.ROOT/'out')+'/'),
            '--out must be below pi05probe/out')
    gpus, requested = json.loads(args.gpus), args.tasks.split(',')
    require(requested and len(set(requested)) == len(requested) and set(requested) <= set(TASKS), 'unknown or duplicate task')
    require(gpus and len({g['uuid'] for g in gpus}) == len(gpus)
            and len({int(g['index']) for g in gpus}) == len(gpus) and all('egl_index' in g for g in gpus),
            'expected unique GPUs with explicit native EGL mappings')
    for name in ['logs', 'services', 'results', 'eval', 'train', 'recovery', 'runs']:
        (DEST/name).mkdir(parents=True, exist_ok=True)
    V.DEST = DEST
    install_job_runtime(V)
    with (DEST/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _OWNS_OUTPUT = True
        selection = read_json(SOURCE/'selection.json')
        provenance = {t: calibration(t, selection) for t in TASKS}
        write_immutable(DEST/'protocol.json', protocol(provenance, selection))
        run_id = V.now().replace(':', '').replace('.', '')
        V.write_json(DEST/'runs'/(run_id+'.json'), dict(started=V.now(), host=V.HOST, pid=os.getpid(), gpus=gpus,
                     requested_tasks=requested, orchestrator_sha256=file_sha(__file__), protocol_sha256=file_sha(DEST/'protocol.json')))
        previous = read_json(DEST/'progress.json') if (DEST/'progress.json').exists() else {}
        V._STATE.update(tasks=previous.get('tasks', {}), requested_tasks=requested)
        V.stage('running', controller='main_method_pipeline.py')
        ready = [t for t in requested if compatible_training(t, provenance[t]) is not None]
        pending = queue.Queue()
        for task in ready+[t for t in requested if t not in ready]:
            pending.put(task)
        def worker(slot, gpu):
            while not V._ABORT.is_set():
                try:
                    task = pending.get_nowait()
                except queue.Empty:
                    return
                try:
                    train = ensure_training(task, gpu, provenance[task])
                    evaluate(task, train, gpu, args.port_base+slot, provenance[task], selection)
                except BaseException as error:
                    mark(task, 'blocked' if isinstance(error, V.ResourceBlocked) else 'failed', error=repr(error))
                    V._ABORT.set()
                    raise
                finally:
                    pending.task_done()
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(worker, i, g) for i, g in enumerate(gpus)]
            for future in as_completed(futures):
                future.result()
        results = {t: read_json(DEST/'results'/f'{t}.json') for t in TASKS if (DEST/'results'/f'{t}.json').exists()}
        V.write_json(DEST/'summary.json', dict(updated=V.now(), tasks=results,
                     complete=len(results) == len(TASKS) and all(r['complete'] for r in results.values())))
        V.stage('complete' if len(results) == len(TASKS) else 'requested_tasks_complete',
                completed_tasks=sorted(results), requested_tasks=requested)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        V._ABORT.set()
        if _OWNS_OUTPUT and V._STATE.get('stage') not in ['blocked', 'failed']:
            V.stage('failed', last_phase=V._STATE.get('stage'), error=repr(error))
        raise
