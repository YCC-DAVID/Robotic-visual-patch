"""Fixed DRIFT pose, our differentiable-render/EOT training, native evaluation.

Queue and supervisors are CPU-only. Every GPU phase repeats fleet preflight,
driver/EGL mapping and an immediate idle-device check. No stop operations.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from drift_paper_common import ROOT, dump, read, sha, now

MODEL = '/shared/user65/miniconda3/envs/openpi-server/bin/python'
SIM = '/shared/user65/miniconda3/envs/openpi-libero/bin/python'
SOURCES = ('drift_aligned_experiment.py', 'drift_aligned_capture.py',
           'drift_aligned_proxy.py', 'drift_aligned_train.py', 'drift_aligned_eval.py')


def prepare(args):
    root, source = Path(args.root).resolve(), Path(args.source_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ['logs', 'services', 'resources', 'baseline', 'train', 'geometry', 'source_snapshot', 'launches']:
        (root / name).mkdir(exist_ok=True)
    original = read(source / 'protocol.json')
    assert original['model_config'] == 'pi05_libero'
    for name, digest in original['source_sha256'].items():
        assert sha(ROOT / 'pi05probe' / name) == digest
    protocol = dict(schema='wam-render-eot-at-drift-pose-v1', created=now(),
        model='pi05_libero', backend='JAX', checkpoint=original['checkpoint'],
        checkpoint_sha256=original['checkpoint_sha256'], source_root=str(source),
        source_protocol_sha256=sha(source / 'protocol.json'), pool_sha256=sha(source / 'pool.npy'),
        geometry_sha256=sha(source / 'geometry/geometry.json'),
        drift_patch_png_sha256=sha(source / 'train/drift_pi05_64px.png'),
        random_png_sha256=sha(source / 'random.png'),
        fixed_placement=dict(wrist_top_left_yx=[160, 80], patch_hw=[64, 64],
                             geometry='exact original fixed wrist plate; no pose or scale search'),
        training=dict(method='our differentiable native-render proxy plus texture-plane EOT, adapted to DRIFT protocol',
            pool='same Spatial10 x 3 train initial states 10-12, all 642 original query frames',
            steps=500, batch=8, alpha=.01, init='gray0.5', patch_hw=[64, 64], seed=0,
            noise='same fresh official JAX noise schedule as DRIFT', objective='squared first t=1 velocity divergence, all 10x32 coordinates',
            views='one shared texture, both native camera proxies',
            eot=dict(seed=2026102901, angle_deg=[-15, 15], dx_fraction=[-.05, .05],
                     dy_fraction=[-.05, .05], scale=[.9, 1.1], gain=[.9, 1.1],
                     scope='one draw per sampled frame, shared by both cameras; physical geometry fixed'),
            proxy='native black/white photometry, observed UV+occlusion, trilinear mipmap, uint8 STE, native256 to224 bilinear antialias',
            proxy_gate='independent random texture native vs proxy, visible interior RMSE <=0.02; sparse base pixels reported explicitly',
            selection='fixed final step500; no test-set selection or per-task retraining'),
        evaluation=dict(suite='libero_spatial', task_ids=[0, 1, 2], init_indices=list(range(10)),
            conditions=['clean', 'drift_render', 'random_render', 'ours_render'], episodes=120,
            policy='same one model process, matched per-query noise and initial states in all conditions',
            input='two native MuJoCo RGB cameras; same frozen preprocessing',
            visibility_diagnostics='segmentation runs in a separate same-state environment; frozen RGB-repeat assertion retained',
            metric='relative ASR=(cleanSR-attackSR)/cleanSR, paired ASR separately',
            limitation='single training seed, three tasks, fixed wrist plate; not the four-suite final result'),
        adaptations_from_original_ours=['JAX original weights instead of Torch conversion',
            '64px instead of128, DRIFT seed0 batch and fresh noise instead of fixed training noise',
            'one universal Spatial-trained patch instead of main experiment per-task patches',
            'original wrist plate rather than original13cm table patch; same render-proxy and texture EOT idea'],
        frozen_original_source_sha256=original['source_sha256'],
        source_sha256={name: sha(ROOT / 'pi05probe' / name) for name in SOURCES},
        eot_reference_sha256=sha(ROOT / 'robotwinprobe/texture_eot.py'))
    if (root / 'protocol.json').exists():
        old = read(root / 'protocol.json')
        for key, value in protocol.items():
            if key != 'created':
                assert old[key] == value, ('frozen protocol changed', key)
        protocol = old
    else:
        dump(root / 'protocol.json', protocol)
    for name, digest in protocol['source_sha256'].items():
        dest = root / 'source_snapshot' / name
        if not dest.exists():
            shutil.copy2(ROOT / 'pi05probe' / name, dest)
        assert sha(dest) == digest
    for src, dest in [(source / 'random.png', root / 'random.png'),
                      (source / 'train/drift_pi05_64px.png', root / 'baseline/drift_pi05_64px.png')]:
        if not dest.exists():
            shutil.copy2(src, dest)
        assert sha(src) == sha(dest)
    for src in (source / 'geometry').iterdir():
        if src.is_file():
            dest = root / 'geometry' / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            assert sha(dest) == sha(src)
    return protocol


def controller(args):
    root = Path(args.root)
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare(args)
        gpu = json.loads(args.gpu)
        assert gpu['host'] == socket.gethostname().split('.')[0]
        import visible_pos_pipeline as V
        from main_method_job_runtime import install
        V.DEST = root
        V.P.MODEL, V.P.SIM = MODEL, SIM
        install(V)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='4',
                   XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.70')
        sim_env = dict(env, MUJOCO_EGL_DEVICE_ID=str(gpu['egl_index']), OMP_NUM_THREADS='2')
        phases = [('capture', SIM, 'drift_aligned_capture.py', sim_env, 14400),
                  ('train', MODEL, 'drift_aligned_train.py', env, 21600),
                  ('eval', SIM, 'drift_aligned_eval.py', sim_env, 21600)]
        for phase, python, script, child_env, timeout in phases:
            if completed_phase(root, phase, protocol):
                continue
            with V.allocation('drift_aligned_' + phase, gpu, 24000):
                V.stage('running_' + phase, gpu=gpu)
                cmd = [python, '-u', str(ROOT / 'pi05probe' / script), '--root', str(root)]
                if phase in ('capture', 'train'):
                    cmd += ['--source-root', args.source_root]
                if phase == 'train':
                    cmd += ['--checkpoint', protocol['checkpoint']]
                    if (root / 'train/config.json').exists():
                        cmd += ['--resume']
                if phase != 'eval':
                    V.run(cmd, phase + '.log', child_env, timeout)
                else:
                    server_cmd = [MODEL, '-u', str(ROOT / 'pi05probe/drift_paper_model.py'),
                        'serve', '--root', str(root), '--checkpoint', protocol['checkpoint'], '--port', str(args.port)]
                    log = root / 'logs/server_eval.log'
                    with log.open('w') as stream:
                        service = subprocess.Popen(server_cmd, env=env, cwd=ROOT / 'pi05probe',
                            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        V.process_event(service, server_cmd, log, env)
                        try:
                            start = time.monotonic()
                            while '[finite-service] listening' not in log.read_text():
                                if service.poll() is not None:
                                    raise RuntimeError('evaluation model service exited before ready')
                                if time.monotonic() - start > 1200:
                                    raise RuntimeError('model readiness deadline; process left untouched')
                                time.sleep(.5)
                            V.run(cmd + ['--port', str(args.port)], 'eval.log', child_env, timeout)
                        finally:
                            code = service.wait()
                            V.event('model_service_natural_exit', pid=service.pid, returncode=code)
                            if code:
                                raise subprocess.CalledProcessError(code, server_cmd)
                assert completed_phase(root, phase, protocol), ('missing validated completion', phase)
        from drift_aligned_eval import summarize
        result = summarize(root)
        assert result['complete'] and result['validation'] == 'passed'
        dump(root / 'complete.json', dict(complete=True, finished=now(),
             protocol_sha256=sha(root / 'protocol.json'), summary_sha256=sha(root / 'summary.json')))
        V.stage('complete', episodes=120)


def completed_phase(root, phase, protocol):
    path = root / phase / 'complete.json'
    if not path.exists():
        return False
    value = read(path)
    assert value['complete'] is True
    if phase == 'capture':
        manifest_path = root / 'capture/manifest.json'
        assert value['manifest_sha256'] == sha(manifest_path)
        manifest = read(manifest_path)
        assert manifest['complete'] and manifest['frames'] == 642 and len(manifest['records']) == 30
        assert manifest['geometry_sha256'] == protocol['geometry_sha256']
        assert manifest['protocol_sha256'] == sha(root / 'capture/protocol.json')
        for row in manifest['records']:
            assert row['complete'] and sha(row['file']) == row['sha256']
    elif phase == 'train':
        import numpy as np
        from PIL import Image
        assert value['completed_steps'] == 500 and value['gradient_frame_forwards'] == 4000
        assert value['finite_losses'] and value['nonzero_finite_gradients']
        for key, name in [('patch_sha256', 'ours_pi05_64px.npy'), ('png_sha256', 'ours_pi05_64px.png'),
                          ('plans_sha256', 'plans.npz'), ('proxy_gate_sha256', 'proxy_gate.json')]:
            assert value[key] == sha(root / 'train' / name)
        assert value['source_pool_sha256'] == protocol['pool_sha256']
        patch = np.load(root / 'train/ours_pi05_64px.npy', allow_pickle=False)
        assert patch.shape == (64, 64, 3) and patch.dtype == np.uint8
        assert np.array_equal(patch, np.asarray(Image.open(root / 'train/ours_pi05_64px.png').convert('RGB')))
    elif phase == 'eval':
        from drift_aligned_eval import summarize
        result = summarize(root, verify=True)
        assert result['complete'] and result['episodes'] == 120 and result['validation'] == 'passed'
    else:
        raise AssertionError(phase)
    return True


def supervise(args):
    root = Path(args.root)
    attempt = Path(args.attempt)
    cmd = [MODEL, '-u', str(Path(__file__).resolve()), '--controller', '--root', args.root,
           '--source-root', args.source_root, '--gpu', args.gpu, '--port', str(args.port)]
    with (attempt / 'controller.log').open('w') as stream:
        p = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1'),
                             cwd=ROOT / 'pi05probe', stdin=subprocess.DEVNULL, stdout=stream,
                             stderr=subprocess.STDOUT, start_new_session=True)
        dump(attempt / 'launch.json', dict(pid=p.pid, supervisor_pid=os.getpid(), gpu=json.loads(args.gpu),
                                          command=cmd, started=now()))
        code = p.wait()
    dump(attempt / 'exit.json', dict(pid=p.pid, returncode=code, finished=now(), exit_policy='natural'))


REMOTE_LAUNCH = r'''
import json,os,subprocess
from pathlib import Path
attempt=Path(payload['attempt'])
cmd=payload['command']
with (attempt/'supervisor.log').open('w') as stream:
    p=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1'),
        cwd=payload['cwd'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid,host=os.uname().nodename)))
'''


def queue(args):
    root = Path(args.root).resolve()
    prepare(args)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            if (root / 'complete.json').exists():
                dump(root / 'queue_progress.json', dict(stage='complete', pid=os.getpid(), updated=now()))
                return
            check = subprocess.run([sys.executable, str(ROOT / 'skills/gpu-preflight/scripts/check_gpus.py'),
                '--json', '--min-free-mib', '24000'], capture_output=True, text=True, timeout=100)
            if check.returncode not in (0, 2):
                dump(root / 'queue_progress.json', dict(stage='waiting_for_preflight', pid=os.getpid(),
                     updated=now(), error=check.stderr[-1000:]))
                time.sleep(30)
                continue
            report = json.loads(check.stdout)
            dump(root / 'resources/latest_fleet.json', report)
            cards = [dict(g, host=n['host']) for n in report['nodes'] if n.get('status') == 'ok'
                for g in n.get('gpus', []) if g.get('idle') and g.get('eligible') and not g.get('faulty')
                and ('L40S' in g['name'] or 'H100' in g['name'])
                and not (n['host'] == 'nnmc63' and g['index'] == 0)]
            cards.sort(key=lambda g: ('L40S' not in g['name'], g['host'], g['index']))
            launched = None
            for card in cards:
                audit = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', card['host'],
                    'env', 'CUDA_VISIBLE_DEVICES=' + card['uuid'], MODEL, str(ROOT / 'pi05probe/egl_device_audit.py'),
                    '--uuid', card['uuid']], capture_output=True, text=True, timeout=40)
                if audit.returncode:
                    continue
                gpu = {key: card[key] for key in ['host', 'index', 'uuid', 'name']}
                gpu['egl_index'] = json.loads(audit.stdout)['egl_index']
                attempt = root / 'launches' / str(time.time_ns())
                attempt.mkdir()
                command = [MODEL, '-u', str(Path(__file__).resolve()), '--supervise', '--root', str(root),
                    '--source-root', args.source_root, '--gpu', json.dumps(gpu), '--attempt', str(attempt),
                    '--port', str(args.port)]
                payload = dict(command=command, attempt=str(attempt), cwd=str(ROOT / 'pi05probe'))
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', card['host'], 'python3', '-'],
                    input='payload = ' + repr(payload) + '\n' + REMOTE_LAUNCH,
                    capture_output=True, text=True, timeout=40)
                if result.returncode:
                    dump(attempt / 'launch_error.json', dict(error=result.stderr[-2000:]))
                    continue
                launched = dict(attempt=str(attempt), gpu=gpu, supervisor=json.loads(result.stdout))
                dump(root / 'queue_progress.json', dict(stage='controller_launched', pid=os.getpid(), updated=now(), **launched))
                break
            if launched is None:
                dump(root / 'queue_progress.json', dict(stage='waiting_for_idle_gpu', pid=os.getpid(),
                     updated=now(), no_stop_operations=True, phases=['capture', 'train', 'eval']))
                time.sleep(30)
                continue
            while not (attempt / 'exit.json').exists():
                time.sleep(30)
            info = read(attempt / 'exit.json')
            if info['returncode'] == 75:
                continue
            dump(root / 'queue_progress.json', dict(stage='complete' if info['returncode'] == 0 else 'failed',
                 pid=os.getpid(), updated=now(), **launched, exit=info))
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--gpu', default='{}')
    parser.add_argument('--port', type=int, default=18871)
    parser.add_argument('--attempt')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--controller', action='store_true')
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    try:
        if args.prepare_only:
            prepare(args)
        elif args.supervise:
            supervise(args)
        elif args.controller:
            controller(args)
        else:
            queue(args)
    except BaseException as error:
        kind = error.__class__.__name__
        dump(Path(args.root) / 'error.json', dict(updated=now(), pid=os.getpid(), error=repr(error)))
        if args.controller and kind == 'ResourceBlocked':
            raise SystemExit(75)
        raise
