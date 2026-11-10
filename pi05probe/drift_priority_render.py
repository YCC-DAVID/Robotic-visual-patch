"""Priority native-render screening with frozen DRIFT science and one trained patch.

The original complete-suite jobs are untouched. This independent run reuses their
recorded clean/digital references and runs a prespecified Spatial subset through
the exact frozen native-render episode function. No process is ever signalled.
"""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from drift_pi0_common import ROOT, dump, now, read, sha


def modules(model):
    prefix = 'drift_pi0' if model == 'pi0' else 'drift_paper'
    return prefix, importlib.import_module(prefix + '_common')


def scope(args):
    tasks = tuple(int(x) for x in args.tasks.split(','))
    inits = tuple(int(x) for x in args.inits.split(','))
    assert tasks and len(set(tasks)) == len(tasks) and all(0 <= x < 10 for x in tasks)
    assert inits and len(set(inits)) == len(inits) and all(0 <= x < 10 for x in inits)
    return dict(suite=args.suite, task_ids=list(tasks), init_indices=list(inits),
                rendered_conditions=['drift_render', 'random_render'],
                rendered_episodes=2 * len(tasks) * len(inits))


def prepare(args):
    """CPU-only snapshot; references retain their original JSON and NPZ hashes."""
    prefix, common = modules(args.model)
    source, root = Path(args.source_root).resolve(), Path(args.root).resolve()
    assert source != root
    root.mkdir(parents=True, exist_ok=True)
    for name in ['logs', 'resources', 'services', 'train', 'eval', 'source_snapshot']:
        (root / name).mkdir(exist_ok=True)
    source_protocol = read(source / 'protocol.json')
    expected = 'pi0_libero' if args.model == 'pi0' else 'pi05_libero'
    assert source_protocol['model_config'] == expected
    assert Path(args.checkpoint).resolve() == Path(source_protocol['checkpoint']).resolve(), 'checkpoint must match source run'
    for name, digest in source_protocol['source_sha256'].items():
        assert sha(ROOT / 'pi05probe' / name) == digest, ('frozen science changed', name)
    assert read(source / 'geometry/complete.json')['complete'] is True
    assert read(source / 'train/complete.json')['complete'] is True
    patch_stem = 'drift_pi0_32px' if args.model == 'pi0' else 'drift_pi05_64px'
    frozen = dict(schema='drift-priority-native-render-subset-v1', model=args.model,
                  created=now(), source_root=str(source), checkpoint=str(Path(args.checkpoint).resolve()),
                  scope=scope(args), source_protocol_sha256=sha(source / 'protocol.json'),
                  source_sha256=source_protocol['source_sha256'],
                  source_checkpoint_sha256=source_protocol['checkpoint_sha256'],
                  patch_npy_sha256=sha(source / 'train' / (patch_stem + '.npy')),
                  patch_png_sha256=sha(source / 'train' / (patch_stem + '.png')),
                  geometry_sha256=sha(source / 'geometry/geometry.json'),
                  random_png_sha256=sha(source / 'random.png'),
                  runner_sha256=sha(__file__), patch_stem=patch_stem,
                  training='none; exact fixed final upstream-trained patch reused',
                  observation='native MuJoCo RGB in both cameras, frozen preprocessing; no input overlay',
                  limitation='Rigid visual plate, not proof of a flush physical sticker',
                  references='Unmodified source clean/digital JSON; original NPZ record paths and hashes retained')
    if (root / 'protocol.json').exists():
        old = read(root / 'protocol.json')
        for key in frozen:
            if key != 'created':
                assert old[key] == frozen[key], ('priority protocol changed', key)
        frozen = old
    else:
        dump(root / 'protocol.json', frozen)
    files = [Path('random.png'), Path('protocol.json')]
    files += [Path('train') / (patch_stem + ext) for ext in ['.npy', '.png']]
    files += [Path('train/complete.json')]
    for rel in files:
        dest = root / ('source_protocol.json' if str(rel) == 'protocol.json' else rel)
        if dest.exists():
            assert sha(dest) == sha(source / rel), ('snapshot changed', dest)
        else:
            shutil.copy2(source / rel, dest)
    for p in sorted((source / 'geometry').iterdir()):
        if p.is_file():
            dest = root / 'geometry' / p.name
            dest.parent.mkdir(exist_ok=True)
            if dest.exists():
                assert sha(dest) == sha(p), ('geometry snapshot changed', dest)
            else:
                shutil.copy2(p, dest)
    for name, digest in frozen['source_sha256'].items():
        dest = root / 'source_snapshot' / name
        if not dest.exists():
            shutil.copy2(ROOT / 'pi05probe' / name, dest)
        assert sha(dest) == digest
    runner_snapshot = root / 'source_snapshot' / Path(__file__).name
    if not runner_snapshot.exists():
        shutil.copy2(__file__, runner_snapshot)
    assert sha(runner_snapshot) == frozen['runner_sha256'], 'runner snapshot changed'
    s = frozen['scope']
    manifest = []
    for tid in s['task_ids']:
        for condition in ['clean', 'random_digital', 'drift_digital', 'drift_render', 'random_render']:
            for init_id in s['init_indices']:
                rel = Path('eval') / s['suite'] / ('task%02d' % tid) / condition / ('init%02d.json' % init_id)
                src = source / rel
                if not src.exists():
                    assert condition != 'clean', ('clean reference not ready', src)
                    continue
                row = read(src)
                assert row['runtime_error'] is False and sha(row['record']) == row['record_sha256']
                dest = root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    if condition.endswith('_render'):
                        # The full run may finish this same pair after the
                        # priority run. Keep our already validated observation;
                        # its record path differs even when inference matches.
                        verify_pair(root, read(dest))
                    else:
                        assert sha(dest) == sha(src)
                else:
                    shutil.copy2(src, dest)
                manifest.append(dict(path=str(rel), source_json_sha256=sha(src),
                    snapshot_json_sha256=sha(dest), reused_source_json=sha(src) == sha(dest),
                    record_sha256=read(dest)['record_sha256']))
    dump(root / 'reference_manifest.json', manifest)
    return frozen


def verify_pair(root, row):
    import numpy as np
    prefix, _ = modules(read(root / 'protocol.json')['model'])
    validator = importlib.import_module(prefix + '_validate')
    clean = read(root / 'eval' / row['suite'] / ('task%02d' % row['task_id']) /
                 'clean' / ('init%02d.json' % row['init_index']))
    assert row['runtime_error'] is False and clean['runtime_error'] is False
    assert row['initial_state_sha256'] == clean['initial_state_sha256']
    assert row['env_seed'] == clean['env_seed'] and row['task'] == clean['task']
    assert sha(row['record']) == row['record_sha256']
    assert sha(clean['record']) == clean['record_sha256']
    with np.load(clean['record'], allow_pickle=False) as c, np.load(row['record'], allow_pickle=False) as r:
        assert np.array_equal(c['initial_state'], r['initial_state'])
        assert validator.same_layout(c['fixture_body_pos'], c['fixture_body_quat'],
                                     r['fixture_body_pos'], r['fixture_body_quat'], True)
        n = min(len(c['noise_seeds']), len(r['noise_seeds']))
        assert n > 0 and np.array_equal(c['noise_seeds'][:n], r['noise_seeds'][:n])
        assert len(r['actions']) == row['queries'] == len(r['noise_seeds'])
        assert np.isfinite(r['actions']).all() and np.isfinite(r['sim_states']).all()
        assert r['visible_pixels224'].shape == (row['queries'], 2)


def summarize(root, verify=True):
    root = Path(root)
    p = read(root / 'protocol.json')
    s = p['scope']
    conditions = ['clean', 'random_digital', 'drift_digital', 'random_render', 'drift_render']
    by_task, aggregate = {}, {c: [] for c in conditions}
    complete = True
    for tid in s['task_ids']:
        groups = {}
        for condition in conditions:
            rows = []
            for init_id in s['init_indices']:
                path = root / 'eval' / s['suite'] / ('task%02d' % tid) / condition / ('init%02d.json' % init_id)
                if not path.exists():
                    continue
                row = read(path)
                assert row['task_id'] == tid and row['init_index'] == init_id
                assert row['suite'] == s['suite'] and row['condition'] == condition
                assert row['runtime_error'] is False and type(row['success']) is bool
                if verify:
                    assert sha(row['record']) == row['record_sha256']
                    if condition.endswith('_render'):
                        verify_pair(root, row)
                rows.append(row)
            if condition in ['clean', 'drift_render', 'random_render']:
                complete = complete and len(rows) == len(s['init_indices'])
            groups[condition] = rows
            aggregate[condition].extend(rows)
        by_task[str(tid)] = metrics(groups)
    result = dict(complete=complete, updated=now(), scope=s, model=p['model'],
                  conditions=metrics(aggregate), tasks=by_task,
                  render_episodes=sum(len(aggregate[c]) for c in s['rendered_conditions']),
                  target_render_episodes=s['rendered_episodes'],
                  patch_png_sha256=p['patch_png_sha256'], geometry_sha256=p['geometry_sha256'],
                  runtime_errors_counted=False, validation='hashes, matched physics/layout/noise, finite actions' if verify else 'not run')
    dump(root / 'priority_summary.json', result)
    return result


def metrics(groups):
    clean = {(r['task_id'], r['init_index']): r for r in groups['clean']}
    result = {}
    for condition, rows in groups.items():
        paired = [(r, clean[(r['task_id'], r['init_index'])]) for r in rows]
        clean_successes = sum(c['success'] for r, c in paired)
        successes = sum(r['success'] for r, c in paired)
        result[condition] = dict(n=len(rows), successes=successes,
            SR=successes / len(rows) if rows else None,
            matched_clean_successes=clean_successes,
            relative_ASR=(clean_successes - successes) / clean_successes if clean_successes else None,
            paired_ASR=sum(c['success'] and not r['success'] for r, c in paired) / clean_successes if clean_successes else None)
    return result


def worker(args):
    prefix, common = modules(args.model)
    common.bootstrap(sim=True)
    common.configure_libero(args.root)
    sim = importlib.import_module(prefix + '_sim')
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy
    from PIL import Image
    import numpy as np
    root = Path(args.root)
    p = read(root / 'protocol.json')
    s = p['scope']
    suite = benchmark.get_benchmark_dict()[s['suite']]()
    patch_paths = dict(drift=root / 'train' / (p['patch_stem'] + '.png'), random=root / 'random.png')
    assert sha(patch_paths['drift']) == p['patch_png_sha256']
    assert sha(root / 'geometry/geometry.json') == p['geometry_sha256']
    geometry = read(root / 'geometry/geometry.json')
    patches = {k: np.asarray(Image.open(v).convert('RGB')) for k, v in patch_paths.items()}
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    try:
        for tid in s['task_ids']:
            task, inits = suite.get_task(tid), suite.get_task_init_states(tid)
            for condition in s['rendered_conditions']:
                kind = condition.split('_')[0]
                dest = root / 'eval' / s['suite'] / ('task%02d' % tid) / condition
                dest.mkdir(parents=True, exist_ok=True)
                env = sim.make_env(task, sim.plane_processor(geometry, patch_paths[kind]))
                try:
                    for init_id in s['init_indices']:
                        output = dest / ('init%02d' % init_id)
                        if output.with_suffix('.json').exists():
                            verify_pair(root, read(output.with_suffix('.json')))
                            continue
                        row, _ = sim.run_episode(env, task, inits[init_id], init_id, s['suite'], tid,
                            client, condition, patches[kind], output.with_suffix('.npz'), video=init_id == s['init_indices'][0])
                        verify_pair(root, row)
                        dump(output.with_suffix('.json'), row)
                        value = summarize(root, verify=False)
                        common.stage(root / 'eval', 'native_render_evaluating', suite=s['suite'], task=tid,
                                     condition=condition, init_index=init_id, success=row['success'],
                                     render_episodes=value['render_episodes'], target_render_episodes=s['rendered_episodes'])
                finally:
                    env.close()
    finally:
        client._ws.close()


def run(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'priority.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p = prepare(args)
        if summarize(root)['complete']:
            dump(root / 'complete.json', dict(complete=True, updated=now(), scope=p['scope']))
            return
        gpu = json.loads(args.gpu)
        assert gpu['host'] == socket.gethostname().split('.')[0]
        import visible_pos_pipeline as V
        from main_method_job_runtime import install
        V.DEST = root
        V.P.MODEL, V.P.SIM = args.model_python, args.sim_python
        install(V)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='4',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.70',
                   OPENPI_DATA_HOME=str(Path(args.checkpoint).parents[2]))
        sim_env = dict(env, MUJOCO_EGL_DEVICE_ID=str(gpu['egl_index']), OMP_NUM_THREADS='2')
        prefix, _ = modules(args.model)
        # V.allocation performs full-fleet preflight, driver/EGL UUID audit, then
        # immediate selected-node idle confirmation before either child starts.
        with V.allocation('drift_priority_' + args.model + '_native_render', gpu, 24000):
            V.stage('native_render_running', model=args.model, gpu=gpu, scope=p['scope'])
            log = root / 'logs/server_render.log'
            cmd = [args.model_python, '-u', str(ROOT / 'pi05probe' / (prefix + '_model.py')),
                   'serve', '--root', str(root), '--checkpoint', args.checkpoint, '--port', str(args.port)]
            with log.open('w') as stream:
                service = subprocess.Popen(cmd, cwd=ROOT / 'pi05probe', env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                V.process_event(service, cmd, log, env)
                failure = None
                try:
                    start = time.monotonic()
                    while '[finite-service] listening' not in log.read_text():
                        if service.poll() is not None:
                            raise RuntimeError('policy service failed: ' + str(log))
                        if time.monotonic() - start > 1200:
                            raise RuntimeError('policy readiness deadline; process left untouched')
                        time.sleep(.5)
                    V.run([args.sim_python, '-u', str(Path(__file__).resolve()), '--worker',
                           '--model', args.model, '--root', str(root), '--source-root', args.source_root,
                           '--checkpoint', args.checkpoint, '--port', str(args.port)],
                          'native_render.log', sim_env, 172800)
                except BaseException as error:
                    failure = error
                    raise
                finally:
                    code = service.wait()
                    V.event('priority_service_natural_exit', pid=service.pid, returncode=code)
                    if code and failure is None:
                        raise subprocess.CalledProcessError(code, cmd)
        result = summarize(root)
        assert result['complete'], 'priority scope incomplete'
        dump(root / 'complete.json', dict(complete=True, updated=now(), scope=p['scope'],
             render_episodes=result['render_episodes'], priority_summary_sha256=sha(root / 'priority_summary.json')))
        V.stage('complete', render_episodes=result['render_episodes'], model=args.model)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['pi0', 'pi05'], required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--gpu', default='{}')
    p.add_argument('--suite', default='libero_spatial', choices=['libero_spatial', 'libero_goal', 'libero_object', 'libero_10'])
    p.add_argument('--tasks', default='0,1,2')
    p.add_argument('--inits', default=','.join(map(str, range(10))))
    p.add_argument('--port', type=int, default=18701)
    p.add_argument('--model-python', default='/shared/user65/miniconda3/envs/openpi-server/bin/python')
    p.add_argument('--sim-python', default='/shared/user65/miniconda3/envs/openpi-libero/bin/python')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--summary-only', '--summarize-only', dest='summary_only', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    try:
        if args.worker:
            worker(args)
        elif args.summary_only:
            print(json.dumps(summarize(args.root), ensure_ascii=False))
        elif args.prepare_only:
            prepare(args)
            print(json.dumps(summarize(args.root), ensure_ascii=False))
        else:
            run(args)
    except BaseException as error:
        dump(Path(args.root) / ('worker_error.json' if args.worker else 'error.json'),
             dict(updated=now(), pid=os.getpid(), error=repr(error)))
        if not args.worker and error.__class__.__name__ == 'ResourceBlocked':
            raise SystemExit(75)
        raise
