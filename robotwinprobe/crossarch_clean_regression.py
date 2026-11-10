"""Replay the historical ten clean seeds through the current pi backend.

Default: prepare an isolated comparison. --run requires an available GPU.
Historical episodes and the main cross-architecture protocol are never replaced.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

from crossarch_common import DEFAULT, ROOT, PY, CAMERAS, PORTS, dump, sha, request


def compare(root, legacy, expected):
    rows = []
    for old in expected:
        seed = old['seed']
        path = root / 'closedloop/pi/clean' / f'seed{seed}.json'
        row = dict(seed=seed, historical_success=old['success'], rerun_complete=False)
        if path.exists():
            new = json.loads(path.read_text())
            if 'success' not in new:
                row['error'] = new.get('error', 'Missing success field')
            else:
                with np.load(legacy / old['record']) as before, np.load(path.with_suffix('.npz')) as after:
                    h = hashlib.sha256(before['states14'][0].astype(np.float32).tobytes())
                    for camera in CAMERAS:
                        h.update(before[camera][0].tobytes())
                    row.update(rerun_complete=True, current_success=new['success'],
                        success_matches=old['success'] == new['success'],
                        initial_state_and_rgb_match=h.hexdigest() == new['initial_clean_sha256'],
                        prompt_matches=old['prompt'] == str(after['prompt']),
                        first_action_max_abs_error=float(np.max(np.abs(before['actions'][0] - after['actions'][0]))),
                        full_state_sequence_equal=bool(np.array_equal(before['states14'], after['states14'])),
                        full_action_sequence_equal=bool(np.array_equal(before['actions'], after['actions'])),
                        full_head_sequence_equal=bool(np.array_equal(before['head_camera'], after['head'])),
                        historical_steps=old['steps'], current_steps=new['steps'])
        rows.append(row)
    dump(root / 'comparison.json', dict(complete=all(r['rerun_complete'] for r in rows),
        historical_successes=sum(r['historical_success'] for r in rows),
        current_completed=sum(r['rerun_complete'] for r in rows),
        current_successes=sum(r.get('current_success', False) for r in rows), rows=rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    root = DEFAULT / 'clean_regression_10200_10209'
    legacy = ROOT / 'robotwinprobe/out/render_transfer/clean'
    root.mkdir(parents=True, exist_ok=True)
    # Share the main runner's lock because the pi backend uses a fixed local port.
    with (DEFAULT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = [r for r in json.loads((legacy / 'results.json').read_text())['episodes']
                    if 10200 <= r['seed'] <= 10209]
        assert sorted(r['seed'] for r in expected) == list(range(10200, 10210))
        protocol = json.loads((DEFAULT / 'protocol.json').read_text())
        protocol['evaluation']['seeds'] = list(range(10200, 10210))
        protocol['evaluation']['conditions'] = ['clean']
        protocol['purpose'] = 'Historical clean regression only; excluded from formal attack evaluation'
        manifest = dict(legacy=str(legacy), legacy_results_sha256=sha(legacy / 'results.json'),
            legacy_protocol_sha256=sha(legacy / 'protocol.json'),
            seeds=list(range(10200, 10210)), expected_successes=sum(r['success'] for r in expected),
            noise_rule='20261006 + seed*1000 + query', checkpoint=protocol['policies']['pi'],
            historical_weight_hash_available=False,
            current_recorded_weight_hash=json.loads((DEFAULT / 'provenance.json').read_text())[
                str(ROOT / 'checkpoints/pi05_robotwin2_pytorch/model.safetensors')])
        for name, value in [('protocol.json', protocol), ('regression_manifest.json', manifest)]:
            path = root / name
            if path.exists():
                assert json.loads(path.read_text()) == value, f'Existing regression input changed: {path}'
            else:
                dump(path, value)
        compare(root, legacy, expected)
        if not args.run:
            print(f'Prepared {root}; historical clean {manifest["expected_successes"]}/10; no rollout launched.')
            return
        gpu = subprocess.run(['nvidia-smi', '-i', args.gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True)
        if gpu.returncode or int(gpu.stdout.strip()) < 16000:
            reason = gpu.stderr.strip() or gpu.stdout.strip() or 'Need at least 16000 MiB free GPU memory'
            dump(root / 'status.json', dict(stage='blocked_gpu', reason=reason))
            raise SystemExit('Clean regression not launched: ' + reason)
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', PORTS['pi']))
        weight = ROOT / 'checkpoints/pi05_robotwin2_pytorch/model.safetensors'
        assert sha(weight) == manifest['current_recorded_weight_hash'], 'Current weight differs from recorded crossarch weight'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   HF_HUB_OFFLINE='1', PYTHONHASHSEED='0', JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)
        ready = root / 'pi_ready.json'
        ready.unlink(missing_ok=True)
        with (logs / 'backend.log').open('a') as log:
            backend = subprocess.Popen([PY['pi'], '-u', str(ROOT / 'robotwinprobe/crossarch_backend.py'),
                '--root', str(root), '--model', 'pi'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            dump(root / 'status.json', dict(stage='starting_backend', pid=os.getpid(), backend_pid=backend.pid))
            deadline = time.monotonic() + 600
            while not ready.exists():
                if backend.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Backend failed or timed out; see logs/backend.log')
                time.sleep(1)
            assert request('pi', {'command': 'ping'})['model'] == 'pi'
            for old in expected:
                seed = old['seed']
                dump(root / 'status.json', dict(stage='clean_regression', seed=seed, pid=os.getpid()))
                with (logs / f'seed{seed}.log').open('a') as log:
                    subprocess.run(['bash', str(ROOT / 'robotwinprobe/run_sim.sh'),
                        str(ROOT / 'robotwinprobe/crossarch_rollout.py'), '--root', str(root),
                        '--model', 'pi', '--condition', 'clean', '--seed', str(seed)],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                compare(root, legacy, expected)
            dump(root / 'status.json', dict(stage='complete'))
        except Exception as error:
            dump(root / 'status.json', dict(stage='stopped_on_error', error=repr(error)))
            raise
        finally:
            if backend.poll() is None:
                backend.terminate()
                try:
                    backend.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    backend.kill()
                    backend.wait()


if __name__ == '__main__':
    main()
