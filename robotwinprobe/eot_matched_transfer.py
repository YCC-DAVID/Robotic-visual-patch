"""Evaluate the frozen position-EOT texture on the original transfer cohort.

Owns its model server; delegates actual rendering to the unchanged transfer
rollout. Existing transfer/pilot results are read-only references.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT/'robotwinprobe'
TRANSFER = PROBE/'out/render_transfer'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(dest):
    clean_data = json.loads((TRANSFER/'clean/results.json').read_text())
    clean = {e['seed']: e for e in clean_data['episodes'] if 10200 <= e['seed'] < 10210}
    assert clean_data['complete'] and len(clean) == 10 and sum(e['success'] for e in clean.values()) == 8
    result = json.loads((dest/'position_render/results.json').read_text())
    assert result['complete'] and {e['seed'] for e in result['episodes']} == set(clean)
    for e in result['episodes']:
        seed = e['seed']
        with np.load(dest/f'position_render/seed{seed}.npz') as actual, np.load(TRANSFER/f'clean/seed{seed}.npz') as ref:
            assert np.array_equal(actual['states14'][0], ref['states14'][0]), (seed, 'state')
            for cam in ['head_camera', 'left_camera', 'right_camera']:
                assert np.array_equal(actual['initial_clean_'+cam], ref[cam][0]), (seed, cam)
            assert np.array_equal(actual['noise_seeds'], 20261006+seed*1000+np.arange(len(actual['noise_seeds'])))
    rows = []
    for name, path in [('proxy_full_render', TRANSFER/'proxy_full_render/results.json'),
                       ('proxy_band32_render', TRANSFER/'proxy_band32_render/results.json'),
                       ('position_eot_render', dest/'position_render/results.json')]:
        d = json.loads(path.read_text())
        failed = sorted(e['seed'] for e in d['episodes'] if clean[e['seed']]['success'] and not e['success'])
        rows.append(dict(condition=name, episodes=10, task_successes=sum(e['success'] for e in d['episodes']),
                         clean_successes=8, attack_successes=len(failed), ASR=len(failed)/8,
                         failed_seeds=failed, source=str(path.relative_to(ROOT))))
    summary = dict(complete=True, seeds=sorted(clean), rows=rows,
                   checks=dict(initial_state_exact=True, initial_clean_all_three_cameras_exact=True,
                               same_explicit_noise_rule=True, frozen_step500_texture=True),
                   scope='Original position and lighting only; this comparison does not measure position robustness.')
    dump(dest/'summary.json', summary)
    sources = [TRANSFER/'clean/results.json', dest/'position_render/results.json',
               dest/'position_render/protocol.json', dest/'protocol.json']
    sources += [ROOT/r['source'] for r in rows[:2]]
    dump(dest/'audit.json', dict(**summary['checks'], complete=True,
         source_hashes=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for p in sources]))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(PROBE/'out/eot_transfer_matched_20261009'))
    ap.add_argument('--gpu', default='7'); ap.add_argument('--port', type=int, default=8257)
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args(); dest = Path(args.out).resolve(); dest.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        report(dest); return
    texture = PROBE/'out/eot_robustness/train_position/texture_0500.png'
    original = json.loads((PROBE/'out/eot_robustness/eval/nominal/position/protocol.json').read_text())
    assert digest(texture) == original['texture_sha256']
    protocol = dict(task='place_container_plate', task_config='demo_clean', test_seeds=list(range(10200,10210)),
                    texture=str(texture), texture_sha256=digest(texture), checkpoint_iteration=500,
                    center=[0,0,.741], size_m=.20, lighting='original', replan=32,
                    noise_rule='20261006 + seed * 1000 + query', retraining=False,
                    clean_reference=str(TRANSFER/'clean/results.json'),
                    baseline_reference=str(TRANSFER/'proxy_full_render/results.json'))
    if (dest/'protocol.json').exists():
        assert json.loads((dest/'protocol.json').read_text()) == protocol
    else:
        dump(dest/'protocol.json', protocol)
    if (dest/'summary.json').exists():
        report(dest); return
    snap = dest/'source_snapshot'; snap.mkdir(exist_ok=True)
    for name in ['eot_matched_transfer.py', 'render_rollout.py', 'render_geometry.py', 'scene_smoke.py', 'policy_server.py']:
        target = snap/name
        if target.exists(): assert digest(target) == digest(PROBE/name)
        else: target.write_bytes((PROBE/name).read_bytes())
    with socket.socket() as s:
        s.bind(('127.0.0.1', args.port))
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    py = '/home/user1/miniconda3/envs/openpi-server/bin/python'
    meta = dest/'server.json'
    if meta.exists(): meta.unlink()
    def status(stage, **kwargs):
        dump(dest/'status.json', dict(stage=stage, pid=os.getpid(), gpu=args.gpu,
                                     port=args.port, updated_unix=time.time(), **kwargs))
    server = worker = None
    def interrupted(sig, frame): raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        status('starting_policy_server')
        with (dest/'server.log').open('a') as log:
            server = subprocess.Popen([py, '-u', str(PROBE/'policy_server.py'), '--port', str(args.port),
                                       '--metadata', str(meta)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        dump(dest/'owned_server.json', dict(pid=server.pid, stopped=False))
        deadline = time.monotonic()+360
        while not meta.exists():
            if server.poll() is not None: raise RuntimeError(f'server exited {server.returncode}')
            if time.monotonic() > deadline: raise TimeoutError('server readiness')
            time.sleep(1)
        assert json.loads(meta.read_text()) == json.loads((TRANSFER/'worker_full/server.json').read_text())
        status('evaluating_10_original_test_seeds', server_pid=server.pid)
        command = ['bash', str(PROBE/'run_sim.sh'), str(PROBE/'render_rollout.py'), '--texture', str(texture),
                   '--condition', 'render', '--out', str(dest/'position_render'), '--port', str(args.port),
                   '--seeds', ','.join(map(str,range(10200,10210))), '--resume']
        dump(dest/'rollout_command.json', command)
        with (dest/'rollout.log').open('a') as log:
            worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if worker.wait() != 0: raise RuntimeError(f'rollout exited {worker.returncode}; see rollout.log')
        report(dest)
        status('complete', complete=True)
    except BaseException as exc:
        status('failed', complete=False, error=repr(exc)); raise
    finally:
        for process in [worker, server]:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
        if server is not None:
            dump(dest/'owned_server.json', dict(pid=server.pid, stopped=True, returncode=server.returncode))


if __name__ == '__main__': main()
