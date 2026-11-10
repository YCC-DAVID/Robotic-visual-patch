"""Locate #57 replay mismatch by repeating the SAME fixed step-1 ImageWAM batch.

No candidate optimization, extra states, alternative noise, or closed-loop runs.
This does not accept a mismatch or relax the six-point completion gate.
"""
import os
from pathlib import Path
import sys
ROOT = Path(os.environ.get('WAMATTACK_PROJECT_ROOT', '/shared/user65/workspace/chence/WAMattack'))
sys.path.insert(0, str(ROOT / 'robotwinprobe'))
import pi_imagewam_gradient_replay as D
import argparse
import fcntl
import json
import socket
import subprocess
import time
import traceback
import numpy as np


def run(gpu):
    lock = (D.DIAG / 'diagnostic.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = D.cpu_prepare()
    previous = D.read(D.DIAG / 'execution.json')
    assert previous['backends_exited']
    attempt = 'reproduction_check_' + str(time.time_ns())
    work = D.prepare_work(plan, attempt)
    D.dump(work / 'previous_execution.json', previous)
    D.dump(work / 'scope.json', dict(step=1, source='imagewam', repetitions=2,
        purpose='test backward repeatability after exact historical-gradient reproduction failed',
        new_states=0, new_draws=0, new_noise=0, new_candidate_updates=0,
        original_gate_unchanged=True))
    execution = dict(host=socket.gethostname().split('.')[0], pipeline_pid=os.getpid(),
        command=sys.argv, work=str(work), backends={}, started_at=time.time(),
        diagnostic_lock=str(D.DIAG / 'diagnostic.lock'), purpose='same-step reproduction discrepancy localization')
    def save():
        D.dump(D.DIAG / 'execution.json', execution)
        D.dump(work / 'execution.json', execution)
    proc = None
    save()
    try:
        D.status('reproduction_check_start_imagewam', work=str(work), gpu_job_started=False)
        card = D.preflight(work, gpu, 'repeat_load_imagewam')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=card['uuid'], PYTHONNOUSERSITE='1',
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', WAMATTACK_PROJECT_ROOT=str(ROOT),
            WAMATTACK_GPU_PREFLIGHT='/shared/user65/.codex/skills/gpu-preflight/scripts/check_gpus.py',
            WAMATTACK_LIBERO_CONFIG_PATH=str(work / 'libero_config'), HF_HUB_OFFLINE='1',
            JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
        command = [D.PY['imagewam'], '-u', str(ROOT / ('robotwinprobe/pi_imagewam_gradient_trace_server.py' if os.environ.get('WAMATTACK_REPLAY_TRACE') == '1' else 'robotwinprobe/pi_imagewam_gradient_replay.py')),
                   'server', '--work', str(work), '--kind', 'imagewam']
        with (work / 'logs/imagewam_backend.log').open('a') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        execution['backends']['imagewam'] = dict(card, pid=proc.pid, command=command)
        save()
        deadline = time.monotonic() + 600
        while not (work / 'ready.json').exists():
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(2)
        assert D.read(work / 'ready.json')['pid'] == proc.pid
        D.status('reproduction_check_native_audit', work=str(work), gpu_job_started=True)
        assert D.rpc('imagewam', 'audit')['passed']
        if os.environ.get('WAMATTACK_REPLAY_TRACE') == '1':
            (work / 'trace_enabled').write_text('same original step1 only\n')
        point = plan['points'][0]
        assert point['step'] == 1
        with np.load(point['input_npz']) as z:
            q = z['q'].copy()
        results = []
        for repeat in range(2):
            D.status('reproduction_check_same_step', step=1, repeat=repeat + 1, work=str(work), gpu_job_started=True)
            out = D.rpc('imagewam', 'batch', texture=q, indices=point['batch']['indices'],
                seeds=point['batch']['seeds'], draws=point['draws'], iteration=1,
                gradient=True, source_model='imagewam')
            path = work / f'repeat_{repeat + 1}.npz'
            np.savez_compressed(path, q=q, gradient=out['gradient'], losses=np.asarray(out['losses']))
            results.append(dict(gradient_sha256=D.ash(out['gradient']), arrays_sha256=D.sha(path),
                raw_loss=float(np.mean(out['losses'])), provenance=out['gradient_provenance']))
        with np.load(work / 'repeat_1.npz') as a, np.load(work / 'repeat_2.npz') as b:
            ga, gb = a['gradient'], b['gradient']
            result = dict(step=1, same_process=True, same_original_q_batch_noise_draw=True,
                exact_forward_losses_equal=np.array_equal(a['losses'], b['losses']),
                gradients_bitwise_equal=np.array_equal(ga, gb),
                gradient_sign_agreement=float(np.mean(np.sign(ga) == np.sign(gb))),
                gradient_relative_l2=float(np.linalg.norm(ga.astype(np.float64) - gb.astype(np.float64)) / np.linalg.norm(ga.astype(np.float64))),
                gradient_cosine=float(np.sum(ga.astype(np.float64) * gb.astype(np.float64)) /
                    (np.linalg.norm(ga.astype(np.float64)) * np.linalg.norm(gb.astype(np.float64)))),
                max_abs_gradient_difference=float(np.max(np.abs(ga - gb))), results=results,
                original_step1=point['history']['provenance']['imagewam'],
                original_raw_loss=point['history']['raw']['imagewam'],
                exact_replay_gate_passed=False, threshold_relaxed=False,
                scope='mismatch evidence only; does not complete or accept six-point replay',
                work=str(work), completed_at=time.time())
        D.dump(work / 'reproduction_variability.json', result)
        D.dump(D.DIAG / 'reproduction_variability.json', result)
        print(json.dumps(result), flush=True)
        D.status('reproduction_discrepancy_preserved', work=str(work), gpu_job_started=True,
                 reason='exact historical gradient gate remains unsatisfied')
    except Exception as exc:
        error = dict(error=repr(exc), traceback=traceback.format_exc(), work=str(work), time=time.time())
        D.dump(work / 'error.json', error)
        D.dump(D.DIAG / 'errors' / (attempt + '.json'), error)
        D.status('runtime_error', work=str(work), gpu_job_started=proc is not None, error=repr(exc))
        raise
    finally:
        if proc is not None:
            if proc.poll() is None:
                try:
                    D.rpc('imagewam', 'shutdown')
                except Exception:
                    proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            execution['backends']['imagewam'].update(exit_code=proc.returncode, exited_at=time.time())
        execution.update(backends_exited=proc is None or proc.poll() is not None, finished_at=time.time())
        save()
        assert D.preserved() == plan['original_hashes']
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True)
    run(ap.parse_args().gpu)
