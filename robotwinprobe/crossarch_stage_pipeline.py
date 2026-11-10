"""Owned, resumable stage-specialist training and 280 paired pilot rollouts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import traceback

import numpy as np

from crossarch_common import ROOT, PY, STAGES, PORTS, request, dump, sha
from crossarch_capture import records
from crossarch_stage_common import DEFAULT, SOURCE, MODELS, prepare
from crossarch_stage_report import report
from crossarch_report import report as parent_report


def run(a):
    p = prepare(a.root)
    lock = (a.root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent_lock = (SOURCE / 'pipeline.lock').open('a')
    fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    logs = a.root / 'logs'
    logs.mkdir(exist_ok=True)
    owned = []
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1',
               PYTHONHASHSEED='0', JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    gpus = dict(pi=a.pi_gpu, wam=a.wam_gpu)

    def status(stage, **extra):
        dump(a.root / 'status.json', dict(stage=stage, pid=os.getpid(), updated=time.time(), **extra))
        progress = {}
        for model in MODELS:
            for phase in STAGES:
                path = a.root / 'patches' / model / phase / 'progress.json'
                if path.exists():
                    progress[model + '_' + phase] = json.loads(path.read_text())
        if progress:
            dump(a.root / 'training_progress.json', dict(iteration=sum(v['iteration'] for v in progress.values()),
                total=4000, patches_completed=sum(v['complete'] for v in progress.values()),
                patches_total=8, patches=progress))
        report(a.root)
        parent_report(SOURCE)

    def launch(name, argv, gpu):
        with (logs / (name + '.log')).open('a') as f:
            proc = subprocess.Popen(argv, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        owned.append(proc)
        return proc

    def call(name, argv, gpu=None):
        status(name)
        proc = launch(name, argv, a.sim_gpu if gpu is None else gpu)
        while proc.poll() is None:
            time.sleep(10)
            status(name, child_pid=proc.pid)
        if proc.returncode:
            raise RuntimeError(f'{name} exited {proc.returncode}; see {logs / (name + ".log")}')

    def simulator(script, *args):
        return ['bash', str(ROOT / 'robotwinprobe/run_sim.sh'), str(ROOT / 'robotwinprobe' / script),
                '--root', str(a.root), *map(str, args)]

    try:
        snapshot = a.root / 'source_snapshots' / time.strftime('%Y%m%d_%H%M%S')
        snapshot.mkdir(parents=True, exist_ok=True)
        if (a.root / 'status.json').exists():
            shutil.copy2(a.root / 'status.json', snapshot / 'previous_status.json')
        for source in (ROOT / 'robotwinprobe').glob('crossarch_*.py'):
            shutil.copy2(source, snapshot / source.name)
        dump(snapshot / 'execution.json', dict(gpus=gpus, sim_gpu=a.sim_gpu,
            protocol_sha256=sha(a.root / 'protocol.json')))
        for model in MODELS:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', PORTS[model]))
            status('start_' + model)
            ready = a.root / (model + '_ready.json')
            ready.unlink(missing_ok=True)
            proc = launch(model + '_backend', [PY[model], '-u', str(ROOT / 'robotwinprobe/crossarch_backend.py'),
                '--root', str(a.root), '--model', model], gpus[model])
            deadline = time.monotonic() + 600
            while not ready.exists():
                if proc.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(model + ' backend failed or timed out')
                time.sleep(5)
            assert request(model, dict(command='ping'))['model'] == model
        # Smoke tests are outside the ten evaluation seeds. No trained texture is needed.
        for model in MODELS:
            for spec in ('clean', 'random_grasp'):
                call(f'smoke_{model}_{spec}', simulator('crossarch_stage_rollout.py', '--model', model,
                    '--condition', spec, '--seed', 23000, '--smoke'))
            clean = json.loads((a.root / 'smoke' / model / 'clean/seed23000.json').read_text())
            attack = json.loads((a.root / 'smoke' / model / 'random_grasp/seed23000.json').read_text())
            if 'grasp' in clean['query_phases']:
                assert attack['active_queries'] > 0 and attack['prefix_verified_queries'] > 0
        dump(a.root / 'audit/stage_exposure_smoke.json', dict(passed=True, seed=23000,
            models=list(MODELS), rule='same boundary replanning; clean prefix exact; actual target-stage exposure'))
        if not (a.root / 'audit/proxy_review.json').exists():
            validation = [r['path'] for seed in p['data']['validation_seeds'] for r in records(a.root, seed)]
            review = {}
            for model in MODELS:
                status('proxy_validation_' + model)
                result = request(model, dict(command='validate', records=validation))
                review[model] = {k: v for k, v in result.items() if k != 'rows'}
                assert result['pixel_pass'], model + ' image fidelity failed'
                from render_geometry import patterns
                q = patterns()['random'].transpose(2, 0, 1)[None].astype(np.float32) / 255
                gradient = request(model, dict(command='gradient', record=validation[0], texture=q, seed=42))
                assert np.isfinite(gradient['gradient']).all() and np.linalg.norm(gradient['gradient']) > 0
                review[model]['finite_nonzero_texture_gradient'] = True
            dump(a.root / 'audit/proxy_review.json', dict(accepted_for_stage_experiment=True, models=review,
                policy=p['proxy_review'], diagnostics_are_not_original_gate_passes=True))
        for seed in p['data']['train_seeds']:
            call(f'capture_{seed}', simulator('crossarch_capture.py', '--mode', 'capture', '--seed', seed))
        for phase in STAGES:
            workers = []
            for model in MODELS:
                workers.append(launch(f'train_{model}_{phase}', [PY['cpu'], '-u',
                    str(ROOT / 'robotwinprobe/crossarch_stage_train.py'), '--root', str(a.root),
                    '--model', model, '--stage', phase], gpus[model]))
            while any(proc.poll() is None for proc in workers):
                for proc in workers:
                    if proc.poll() not in (None, 0):
                        raise RuntimeError('Stage patch trainer exited; inspect train logs')
                status('patch_training', specialist_stage=phase, child_pids=[proc.pid for proc in workers])
                time.sleep(10)
            assert all(proc.returncode == 0 for proc in workers)
        for seed in p['evaluation']['seeds']:
            # Finish all phase/exposure groups on each seed before moving to the next.
            for model in MODELS:
                for spec in p['evaluation']['conditions']:
                    call(f'eval_{model}_{spec}_{seed}', simulator('crossarch_stage_rollout.py', '--model', model,
                        '--condition', spec, '--seed', seed))
        status('complete')
    except Exception as error:
        status('stopped_on_error', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        for proc in reversed(owned):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DEFAULT)
    ap.add_argument('--pi-gpu', default='2')
    ap.add_argument('--wam-gpu', default='4')
    ap.add_argument('--sim-gpu', default='4')
    run(ap.parse_args())
