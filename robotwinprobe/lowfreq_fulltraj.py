"""Band-limited (8 cycles/patch) velocity14 patch trained on whole-trajectory states, then development evaluation.

Only change versus lowfreq8_20261028 (approach-start states): the training states. Draws reuse the
stage_joint_20261026 'joint' plan (24 expert trajectories, approach/grasp/transport/place, 2 per stage per batch).
Clean/Random development episodes are copied from lowfreq8 (same protocol hash) so all conditions stay paired.
"""
import argparse, json, os, shutil, socket, subprocess, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from e_method_common import load, dump, request, results, sha, cohort
from e_method_train import bandlimit

SOURCE = BASE/'out/lowfreq8_20261028'
STAGE = BASE/'out/stage_joint_20261026'
NAME = 'velocity14_lowfreq8_fulltraj'
CUTOFF = 8.
PY = '/shared/user65/miniconda3/envs/openpi-server/bin/python'
SIM_PY = '/shared/user65/miniconda3/envs/RoboTwin/bin/python'


def local(path):
    """Map a recorded absolute path to this node's view of the same NFS tree."""
    rel = str(path).split('/robotwinprobe/', 1)[1]
    return str(BASE/rel)


def setup(root):
    if (root/'protocol.json').exists():
        assert sha(root/'protocol.json') == sha(SOURCE/'protocol.json'); return
    root.mkdir(parents=True)
    for f in ('protocol.json', 'collection_manifest.json'): shutil.copy2(SOURCE/f, root/f)
    shutil.copytree(SOURCE/'patches/random', root/'patches/random')
    episodes = cohort(root, 'development')/'episodes'; episodes.mkdir()
    src = SOURCE/'cohorts/development/episodes'
    for f in sorted(src.iterdir()):
        if '_clean_' in f.name or '_random_' in f.name: shutil.copy2(f, episodes/f.name)
    draws = json.loads((STAGE/'training_draws.json').read_text())['joint']
    draws = [[dict(record=local(d['record']), seed=d['seed'], stage=d['stage']) for d in batch] for batch in draws]
    dump(root/'training_draws.json', dict(source=str(STAGE/'training_draws.json'), source_sha256=sha(STAGE/'training_draws.json'),
                                         cutoff_cycles_per_patch=CUTOFF, kind='velocity14', name=NAME, draws=draws))


def train(root):
    p = load(root)['method_study']; plan = json.loads((root/'training_draws.json').read_text())
    draws = plan['draws']; assert len(draws) >= p['steps'] and all(len(b) == p['batch'] for b in draws)
    dest = root/'training'/NAME; dest.mkdir(parents=True, exist_ok=True); ckpt = dest/'state.npz'
    q = np.full((1, 3, 128, 128), .5, np.float32); history = []; start = 0
    if ckpt.exists():
        with np.load(ckpt) as z: q = z['texture'].copy(); start = int(z['iteration']); history = json.loads(str(z['history']))
    for it in range(start, p['steps']):
        gradient = np.zeros_like(q); stage_l0 = {}
        for d in draws[it]:
            r = request(dict(command='method_gradient', kind='velocity14', texture=q, record=local(d['record']), seed=d['seed']), root)
            gradient += r['gradient']/p['batch']; stage_l0.setdefault(d['stage'], []).append(r['first14_mse'])
        assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
        q = bandlimit(q+p['lr']*np.sign(gradient), CUTOFF)
        row = dict(iteration=it+1, first14_mse=float(np.mean([v for vs in stage_l0.values() for v in vs])),
                   per_stage={k: float(np.mean(v)) for k, v in stage_l0.items()}, gradient_norm=float(np.linalg.norm(gradient)))
        history.append(row)
        tmp = ckpt.with_suffix('.tmp.npz'); np.savez_compressed(tmp, texture=q, iteration=it+1, history=json.dumps(history)); tmp.replace(ckpt)
        if it == 0 or (it+1) % 25 == 0:
            out = root/'patches'/NAME; out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.rint(q[0].transpose(1, 2, 0)*255).astype(np.uint8)).save(out/f'texture_{it+1:04d}.png')
        dump(dest/'progress.json', row); print('[train]', row, flush=True)
    dump(dest/'complete.json', dict(complete=True, texture_sha256=sha(root/'patches'/NAME/'texture_0500.png')))


def main(args):
    root = args.root.resolve(); setup(root)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               HF_HUB_OFFLINE='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', OPENPI_DATA_HOME='/shared/user65/.cache/openpi',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', ROBOTWIN_PYTHON=SIM_PY)
    env.pop('PYTHONPATH', None)
    status = lambda stage, **kw: dump(root/'run_status.json', dict(stage=stage, host=socket.gethostname(), gpu=args.gpu, pid=os.getpid(), **kw))
    logs = root/'logs'; logs.mkdir(exist_ok=True); backend = None
    try:
        status('loading_backend'); (root/'backend_ready.json').unlink(missing_ok=True)
        with (logs/'backend.log').open('a') as f:
            backend = subprocess.Popen([PY, '-u', str(BASE/'e_method_backend.py'), '--root', str(root)], cwd=BASE, env=env,
                                       stdout=f, stderr=subprocess.STDOUT)
        while not (root/'backend_ready.json').exists():
            if backend.poll() is not None: raise RuntimeError('backend failed; see logs/backend.log')
            time.sleep(5)
        status('training', backend_pid=backend.pid); train(root)
        conditions = ('clean', 'random', NAME)
        for seed in load(root)['method_study']['development_seeds']:
            status('evaluating', seed=seed, backend_pid=backend.pid)
            with (logs/f'seed{seed}.log').open('a') as f:
                subprocess.run(['bash', str(BASE/'run_sim.sh'), str(BASE/'e_method_sim.py'), '--root', str(root), '--mode', 'evaluate',
                                '--split', 'development', '--seed', str(seed), '--conditions', ','.join(conditions)],
                               env=env, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=BASE.parent)
        dump(root/'development_results.json', results(root, 'development', conditions))
        status('development_complete')
    except BaseException as e:
        status('failed', error=repr(e)); raise
    finally:
        if backend is not None and backend.poll() is None: backend.terminate(); backend.wait(timeout=30)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=BASE/'out/lowfreq8_fulltraj_20261029')
    ap.add_argument('--gpu', required=True)
    main(ap.parse_args())
