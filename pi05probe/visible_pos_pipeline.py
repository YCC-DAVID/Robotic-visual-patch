"""#60: per-task visible patch position + single-task render-proxy training (nominal / EOT).

1. clean pools (inits 0-11): reuse #59 captures for the 4 #59 train tasks; capture the 3 others.
2. visibility scan on TRAINING frames (inits 0-9) over 470 legal 13cm table positions.
3. selection rule (fixed before any result): among positions whose patch has >=300 base px
   on >=90% of training frames, pick max mean wrist px over training frames.
4. dual-view calibration at the selected position (same states).
5. per task: single-task patch, E26 recipe 500x8, nominal and #51/#52 texture-plane EOT.
6. real MuJoCo closed loop, inits 40-49, clean / random / nominal / eot at that task's position.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import numpy as np
import multitask_pipeline as P

DEST = P.ROOT/'out/visible_pos'
TASKS = P.TRAIN_TASKS+P.HELDOUT_TASKS
CONDITIONS = ['clean', 'random', 'nominal', 'eot']
BASE_MIN, BASE_FRAC = 300, .9


SCAN_SCHEMA = 'visibility-reset-safe-v2'
HOST = socket.gethostname().split('.')[0]
_STATE_LOCK = threading.RLock()
_GPU_LOCKS = {}
_ALLOCATION_CHECK_LOCK = threading.Lock()
_ABORT = threading.Event()
_LOCAL = threading.local()
_STATE = dict(stage='initializing', active_jobs={})


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp.%d.%d' % (os.getpid(), threading.get_ident()))
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def stage(name, **kw):
    with _STATE_LOCK:
        _STATE.update(stage=name, updated=now(), controller_pid=os.getpid(), host=HOST, **kw)
        write_json(DEST/'progress.json', _STATE)
    print('[stage]', name, kw, flush=True)


def event(name, **kw):
    record = dict(event=name, updated=now(), controller_pid=os.getpid(), host=HOST, **kw)
    with _STATE_LOCK:
        folder = DEST/'resources'
        folder.mkdir(parents=True, exist_ok=True)
        with (folder/'allocations.jsonl').open('a') as f:
            f.write(json.dumps(record)+'\n')
    print('[resource]', json.dumps(record), flush=True)


class ResourceBlocked(RuntimeError):
    pass


def preflight_script():
    paths = [P.ROOT.parent/'skills/gpu-preflight/scripts/check_gpus.py',
             Path('/home/user1/.codex/skills/gpu-preflight/scripts/check_gpus.py')]
    for path in paths:
        if path.is_file():
            return path
    raise ResourceBlocked('gpu-preflight script missing: '+str(paths))


def inventory(job, suffix, hosts=None, min_free_mib=0):
    cmd = [sys.executable, str(preflight_script()), '--json', '--min-free-mib', str(min_free_mib)]
    if hosts:
        cmd += ['--hosts', *hosts]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ResourceBlocked('gpu-preflight could not complete: '+str(error)) from error
    # Exit 2 may describe an unrelated UNKNOWN node/card; inspect the selected GPU explicitly.
    if result.returncode not in (0, 2):
        raise ResourceBlocked('gpu-preflight failed: '+result.stderr[-2000:])
    try:
        report = json.loads(result.stdout)
    except ValueError as error:
        raise ResourceBlocked('invalid gpu-preflight output: '+result.stdout[-1000:]) from error
    write_json(DEST/'resources'/(job+'_'+suffix+'.json'), dict(command=cmd, **report))
    return report


def checked_gpu(report, gpu):
    if HOST not in [f'nnmc{i}' for i in range(60, 66)]+['nnmc71', 'nnmc76']:
        raise ResourceBlocked('pipeline must run on a checked compute node, got '+HOST)
    if gpu.get('host', HOST).split('.')[0] != HOST:
        raise ResourceBlocked('GPU host does not match pipeline host: '+str(gpu))
    if HOST == 'nnmc63' and int(gpu['index']) == 0:
        raise ResourceBlocked('nnmc63 GPU0 excluded for historical ECC fault')
    node = next((n for n in report['nodes'] if n['host'] == HOST), None)
    if not node or node.get('status') != 'ok':
        raise ResourceBlocked('selected node is UNKNOWN: '+HOST)
    found = next((g for g in node['gpus'] if g['uuid'] == gpu['uuid']
                  and int(g['index']) == int(gpu['index'])), None)
    if not found or not found.get('idle') or not found.get('eligible') or found.get('faulty'):
        raise ResourceBlocked('selected GPU is occupied, faulty or insufficient: '+str(found or gpu))
    return found


def check_allocation(job, gpu, min_free_mib):
    # Confirm distinct queued allocations in sequence, even on different GPUs.
    with _ALLOCATION_CHECK_LOCK:
        checked_gpu(inventory(job, 'cluster', min_free_mib=min_free_mib), gpu)
        if 'egl_index' in gpu:
            result = subprocess.run([sys.executable, str(P.ROOT/'egl_device_audit.py'),
                '--uuid', gpu['uuid'], '--egl-index', str(gpu['egl_index'])],
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu['uuid']},
                capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise ResourceBlocked('EGL UUID mapping check failed: '+result.stderr[-2000:])
            write_json(DEST/'resources'/(job+'_egl.json'), json.loads(result.stdout))
        return checked_gpu(inventory(job, 'selected', hosts=[HOST], min_free_mib=min_free_mib), gpu)


@contextmanager
def allocation(label, gpu, min_free_mib=2048):
    """One integrated GPU task; cached jobs never enter this context."""
    key = gpu['uuid']
    with _STATE_LOCK:
        lock = _GPU_LOCKS.setdefault(key, threading.Lock())
    with lock:
        if _ABORT.is_set():
            raise RuntimeError('pipeline canceled after another task failed')
        lockdir = Path('/tmp/wamattack-gpu-allocation-locks')
        lockdir.mkdir(parents=True, exist_ok=True)
        job = label+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        with (lockdir/(HOST+'_'+key+'.lock')).open('a') as handle:
            acquired = False
            try:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError as error:
                    raise ResourceBlocked('another controller holds GPU allocation lock: '+key) from error
                selected = check_allocation(job, gpu, min_free_mib)
                _LOCAL.job = job
                _LOCAL.gpu = gpu
                with _STATE_LOCK:
                    _STATE['active_jobs'][job] = dict(label=label, gpu=gpu, checked_at=now())
                    write_json(DEST/'progress.json', dict(_STATE, updated=now()))
                event('allocated', job=job, label=label, gpu=selected,
                      note='snapshot checks and cooperative lock; no cluster scheduler reservation')
                yield
            except ResourceBlocked as error:
                _ABORT.set()
                stage('blocked', last_phase=_STATE.get('stage'), error=str(error), job=job, gpu=gpu)
                event('blocked', job=job, label=label, gpu=gpu, error=str(error))
                raise
            finally:
                if getattr(_LOCAL, 'job', None) == job:
                    with _STATE_LOCK:
                        _STATE['active_jobs'].pop(job, None)
                        write_json(DEST/'progress.json', dict(_STATE, updated=now()))
                    event('released', job=job, gpu=gpu)
                    del _LOCAL.job, _LOCAL.gpu
                if acquired:
                    fcntl.flock(handle, fcntl.LOCK_UN)


def sim_env(gpu):
    # EGL enumeration can differ from nvidia-smi/CUDA order (nnmc71).
    # An explicit mapping is verified by UUID before it is put in the launch record.
    egl = str(gpu.get('egl_index', gpu['index']))
    cuda = gpu['uuid'] if 'egl_index' in gpu else str(gpu['index'])
    # robosuite 1.4 has an old substring assertion, including for UUID visibility.
    if egl not in cuda:
        raise RuntimeError('robosuite cannot accept this CUDA/EGL visibility mapping: '+str(gpu))
    return {**os.environ, 'CUDA_VISIBLE_DEVICES': cuda,
            'MUJOCO_EGL_DEVICE_ID': egl, 'OMP_NUM_THREADS': '2'}


def stop_process(proc):
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def process_event(proc, cmd, log, env):
    event('process_started', job=_LOCAL.job, gpu=_LOCAL.gpu, pid=proc.pid,
          command=[str(x) for x in cmd], cwd=str(P.ROOT), log=str(log),
          cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),
          mujoco_egl_device_id=env.get('MUJOCO_EGL_DEVICE_ID'))


def run(cmd, log, env, timeout):
    logpath = DEST/'logs'/log
    with logpath.open('w') as f:
        proc = subprocess.Popen(cmd, cwd=P.ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        process_event(proc, cmd, logpath, env)
        try:
            code = proc.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, cmd)
        finally:
            stop_process(proc)
            event('process_finished', job=_LOCAL.job, pid=proc.pid, returncode=proc.returncode)


@contextmanager
def server(gpu, port):
    folder = DEST/'services'
    logpath = folder/f'policy_server_{port}.log'
    cmd = [P.MODEL, 'render_proxy_policy_server.py', '--port', str(port)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu['uuid'], 'OMP_NUM_THREADS': '4'}
    with logpath.open('w') as log:
        proc = subprocess.Popen(cmd, cwd=P.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        process_event(proc, cmd, logpath, env)
        write_json(folder/f'owned_service_{port}.json', dict(pid=proc.pid, port=port, gpu=gpu, job=_LOCAL.job, started=now()))
        try:
            start = time.monotonic()
            while '[serve] ready' not in logpath.read_text():
                if proc.poll() is not None or time.monotonic()-start >= 900:
                    raise RuntimeError('policy server failed or timed out: '+str(logpath))
                time.sleep(3)
            yield
        finally:
            stop_process(proc)
            event('process_finished', job=_LOCAL.job, pid=proc.pid, returncode=proc.returncode)


def scan_cache_reason(out):
    if not (out/'visibility.npz').is_file() or not (out/'visibility.json').is_file():
        return 'missing visibility data or metadata'
    try:
        meta = json.loads((out/'visibility.json').read_text())
        if meta.get('schema') != SCAN_SCHEMA or meta.get('validated') is not True or meta.get('smoke') is not False:
            return 'scan is not a validated full run of '+SCAN_SCHEMA
        source = meta.get('scanner_sha256', meta.get('source_sha256'))
        if isinstance(source, dict):
            source = source.get('visibility_scan.py')
        if source != P.sha(P.ROOT/'visibility_scan.py'):
            return 'visibility scanner source SHA256 changed'
        with np.load(out/'visibility.npz', allow_pickle=False) as data:
            frames, episodes = data['frames'], data['ep_id']
            cand, base, wrist = data['candidates'], data['base_px'], data['wrist_px']
            nf, nc = len(frames), len(cand)
            if (nf == 0 or nc == 0 or frames.shape != (nf,) or episodes.shape != (nf,)
                    or cand.shape != (nc, 2) or base.shape != (nf, nc) or wrist.shape != (nf, nc)
                    or meta.get('frames') != nf or meta.get('candidates') != nc):
                return 'visibility data and metadata dimensions disagree'
            if (np.any(episodes < 0) or np.any(episodes >= 10) or np.any(np.diff(frames) <= 0)
                    or not np.isfinite(cand).all() or not np.isfinite(base).all() or not np.isfinite(wrist).all()
                    or np.any(base < 0) or np.any(wrist < 0) or np.any(base > 224**2) or np.any(wrist > 224**2)):
                return 'invalid training frames, positions or visibility values'
        checks = meta.get('spot_check')
        if not checks or len({c.get('candidate') for c in checks}) < min(2, nc):
            return 'missing independent fresh-scene candidate checks'
        training_episodes = set(int(e) for e in episodes)
        for check in checks:
            if set(check.get('episodes', [])) != training_episodes:
                return 'fresh-scene spot checks do not cover every training episode'
            j = check.get('candidate')
            if not isinstance(j, int) or not 0 <= j < nc or not np.array_equal(check.get('world'), cand[j]):
                return 'fresh-scene check candidate does not match scan data'
        for check in checks:
            if check.get('max_abs_diff') != 0 or check.get('mask_disagreement_pixels_max') != 0:
                return 'fresh-scene area or full-mask mismatch'
        if meta.get('mask_disagreement_pixels_max', 0) != 0:
            return 'nonzero mask disagreement summary'
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
        return 'unreadable scan metadata: '+str(error)
    return None


def archive_scan(out, reason):
    if out.exists():
        recovery = DEST/'recovery'/('invalid_scan_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
        recovery.mkdir(parents=True, exist_ok=True)
        target = recovery/out.name
        out.rename(target)
        write_json(recovery/'reason.json', dict(task=out.name, reason=reason, archived=now(), path=str(target)))
        event('scan_archived', task=out.name, reason=reason, path=str(target))


def pool_path(task):
    old = P.DEST/'tasks'/task/'base.npz'
    return old if task in P.TRAIN_TASKS else DEST/'pools'/task/'pool.npz'


def make_pool(task, gpu, port):
    if pool_path(task).exists(): return
    with allocation('pool_'+task, gpu, 32000), server(gpu, port):
        run([P.SIM, 'multitask_capture.py', '--task', task, '--port', str(port), '--out', str(DEST/'pools'/task), '--pool-only'],
            f'pool_{task}.log', sim_env(gpu), 7200)


def scan(task, gpu):
    out = DEST/'scan'/task
    reason = scan_cache_reason(out)
    if reason is None:
        return
    archive_scan(out, reason)
    with allocation('scan_'+task, gpu):
        run([P.SIM, 'visibility_scan.py', '--task', task, '--pool', str(pool_path(task)), '--out', str(out)],
            f'scan_{task}.log', sim_env(gpu), 14400)
    reason = scan_cache_reason(out)
    if reason is not None:
        raise RuntimeError('visibility validation failed for '+task+': '+reason)


def select(task):
    reason = scan_cache_reason(DEST/'scan'/task)
    if reason is not None:
        raise RuntimeError('refusing invalid visibility scan for '+task+': '+reason)
    z = np.load(DEST/'scan'/task/'visibility.npz')
    base, wrist, cand = z['base_px'], z['wrist_px'], z['candidates']
    ok = (base >= BASE_MIN).mean(0) >= BASE_FRAC
    assert ok.any(), task
    score = np.where(ok, wrist.mean(0), -1)
    j = int(np.argmax(score))
    old = int(np.argmin(np.abs(cand-[0.045, 0.117]).sum(1)))
    return dict(task=task, candidate=j, world=cand[j].tolist(), wrist_mean=float(wrist[:, j].mean()),
                wrist_max=int(wrist[:, j].max()), base_mean=float(base[:, j].mean()), eligible=int(ok.sum()),
                site101_nearest_candidate=cand[old].tolist(), site101_nearest_wrist_mean=float(wrist[:, old].mean()))


def calibrate(task, sel, gpu):
    out = DEST/'tasks'/task
    if (out/'capture.json').exists():
        meta = json.loads((out/'capture.json').read_text())
        if not meta.get('complete') or not np.allclose(meta['cell_world'][:2], sel['world'], atol=1e-7, rtol=0):
            raise RuntimeError('cached calibration does not match selected position: '+task)
        return
    with allocation('calibrate_'+task, gpu):
        run([P.SIM, 'multitask_capture.py', '--task', task, '--port', '0', '--out', str(out), '--pool-from', str(pool_path(task)),
             '--cell-world=%.4f,%.4f' % tuple(sel['world'])], f'calib_{task}.log', sim_env(gpu), 7200)


def train(task, mode, gpu):
    out = DEST/'train'/task/mode
    if (out/'results.json').exists(): return
    if out.exists():  # interrupted run: keep it, retry into a fresh folder name
        out.rename(out.with_name(mode+'_interrupted_%d' % len(list(out.parent.glob(mode+'_interrupted_*')))))
    with allocation('train_'+task+'_'+mode, gpu, 32000):
        run([P.MODEL, 'multitask_train.py', '--tasks-root', str(DEST/'tasks'), '--tasks', task, '--out', str(out), '--proxy-rmse-max', '0.02']
            + (['--eot'] if mode == 'eot' else []), f'train_{task}_{mode}.log',
            {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu['uuid'], 'OMP_NUM_THREADS': '4'}, 14400)


def evaluate(task, gpu, port):
    folder = DEST/'eval'/task
    sel = json.loads((DEST/'selection.json').read_text())[task]
    textures = {'clean': None, 'random': P.ROOT/'out/dual_view/manifest_both/random.png',
                'nominal': DEST/'train'/task/'nominal/texture_0500.png', 'eot': DEST/'train'/task/'eot/texture_0500.png'}
    manifest = folder/'manifest.json'; folder.mkdir(parents=True, exist_ok=True)
    P.dump(manifest, dict(source_task=task, stage='visible_pos_eval',
        geometry={'patch_m': .13, 'cell_world': json.loads((DEST/'tasks'/task/'capture.json').read_text())['cell_world']},
        textures={k: None if p is None else str(p) for k, p in textures.items()},
        texture_sha256={k: None if p is None else P.sha(p) for k, p in textures.items()}, selection=sel))
    for attempt in range(3):
        out = folder/('rollout' if attempt == 0 else f'rollout_attempt{attempt}')
        if (out/'complete.json').exists(): return
        try:
            with allocation('evaluate_'+task+'_attempt'+str(attempt), gpu, 32000), server(gpu, port):
                run([P.SIM, 'cross_task_transfer_rollout.py', '--manifest', str(manifest), '--task', task,
                     '--conditions', ','.join(CONDITIONS), '--indices', ','.join(map(str, P.INDICES)), '--port', str(port),
                     '--out', str(out.relative_to(P.ROOT/'out'))], f'eval_{task}_{attempt}.log', sim_env(gpu), 10800)
            P.dump(out/'complete.json', dict(complete=True, attempt=attempt)); return
        except (subprocess.CalledProcessError, AssertionError) as error:
            print('[retry]', task, attempt, error, flush=True)
    raise RuntimeError('evaluation failed 3 times: '+task)


def guarded(function, *args):
    if _ABORT.is_set():
        raise RuntimeError('pipeline canceled after another task failed')
    try:
        return function(*args)
    except BaseException as error:
        _ABORT.set()
        if _STATE.get('stage') not in ['blocked', 'failed']:
            stage('failed', last_phase=_STATE.get('stage'), error=repr(error))
        raise


def main():
    gpus = json.loads(sys.argv[1])
    if not gpus or len({g['uuid'] for g in gpus}) != len(gpus) or len({int(g['index']) for g in gpus}) != len(gpus):
        raise ValueError('expected unique GPUs with UUID and physical index')
    for name in ['logs', 'pools', 'scan', 'tasks', 'train', 'eval', 'services']: (DEST/name).mkdir(parents=True, exist_ok=True)
    if not (DEST/'protocol.json').exists():
        P.dump(DEST/'protocol.json', dict(experiment='#60 per-task visible position, single-task nominal vs EOT', tasks=TASKS,
            selection=f'training frames only; eligible if base px>={BASE_MIN} on >={BASE_FRAC:.0%} frames; argmax mean wrist px; '
                      '470 legal 13cm positions, 3cm grid, keepout+1cm',
            recipe='E26 both-camera 500x8, seed 20261005, noise 20260901, gray init; eot = robotwinprobe/texture_eot.py plan seed 2026102901',
            conditions=CONDITIONS, indices=P.INDICES, gpus=gpus, host=os.uname().nodename))
    pick = lambda i: gpus[i % len(gpus)]
    stage('pools')
    with ThreadPoolExecutor(len(gpus)) as ex:
        for f in [ex.submit(guarded, make_pool, t, pick(i), 8381+i) for i, t in enumerate(P.HELDOUT_TASKS)]: f.result()
    stage('scan')
    with ThreadPoolExecutor(len(gpus)) as ex:
        for f in [ex.submit(guarded, scan, t, pick(i)) for i, t in enumerate(TASKS)]: f.result()
    selection = {t: select(t) for t in TASKS}
    if (DEST/'selection.json').exists(): assert json.loads((DEST/'selection.json').read_text()) == selection
    else: P.dump(DEST/'selection.json', selection)
    print('[selection]', json.dumps(selection), flush=True)
    stage('calibrate')
    with ThreadPoolExecutor(len(gpus)) as ex:
        for f in [ex.submit(guarded, calibrate, t, selection[t], pick(i)) for i, t in enumerate(TASKS)]: f.result()
    stage('train')
    jobs = [(t, m) for t in TASKS for m in ['nominal', 'eot']]
    with ThreadPoolExecutor(len(gpus)) as ex:
        for f in [ex.submit(guarded, train, t, m, pick(i)) for i, (t, m) in enumerate(jobs)]: f.result()
    stage('evaluate', episodes=len(TASKS)*len(CONDITIONS)*10)
    with ThreadPoolExecutor(len(gpus)) as ex:
        for f in [ex.submit(guarded, evaluate, t, pick(i), 8391+i) for i, t in enumerate(TASKS)]: f.result()
    stage('complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        _ABORT.set()
        if _STATE.get('stage') not in ['blocked', 'failed']:
            stage('failed', last_phase=_STATE.get('stage'), error=repr(error))
        raise
