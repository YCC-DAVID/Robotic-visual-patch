"""Train an open-gripper target on grasp/transport, then paired development evaluation."""
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
NAME = 'action_gripper_open_lowfreq8'
CUTOFF = 8.
PY = '/shared/user65/miniconda3/envs/openpi-server/bin/python'
SIM_PY = '/shared/user65/miniconda3/envs/RoboTwin/bin/python'


def local(path):
    """Map a recorded absolute path to this node's view of the same NFS tree."""
    rel = str(path).split('/robotwinprobe/', 1)[1]
    return str(BASE/rel)


def setup(root):
    root.mkdir(parents=True,exist_ok=True)
    protocol=load(SOURCE)
    protocol['backend_port']=18772
    protocol['experiment']='directional_gripper_open'
    protocol['directional_gripper']=dict(target='normalized +1 (physical open)',
        loss='negative MSE to open on all 32 predicted active-gripper actions after full10flow',
        stages=['grasp','transport'],stage_samples_per_batch=4,training_seed=202610270,
        active_arm='only nonzero six-joint arm; assert opposite arm parked at zero',
        data_source=str(STAGE),frequency_cutoff=8,initialization='gray',
        evaluation='fresh paired clean/random/optimized on fixed32 development; no test used')
    if (root/'protocol.json').exists():assert load(root)==protocol
    else:dump(root/'protocol.json',protocol)
    dest=root/'patches/random';dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(SOURCE/'patches/random/texture.png',dest/'texture.png')
    original=json.loads((STAGE/'training_draws.json').read_text())['joint']
    pools={stage:{} for stage in ('grasp','transport')}
    for batch in original:
        for d in batch:
            if d['stage'] not in pools:continue
            path=local(d['record'])
            with np.load(path) as z:state=z['state14']
            norms=[float(np.linalg.norm(state[:6])),float(np.linalg.norm(state[7:13]))]
            assert min(norms)<1e-5 and max(norms)>.1
            pools[d['stage']][path]=dict(record=path,stage=d['stage'],active_start=7*int(norms[1]>norms[0]))
    rng=np.random.default_rng(202610270);draws=[]
    for _ in range(500):
        batch=[]
        for stage,pool in pools.items():
            records=list(pool.values())
            for _ in range(4):batch.append(dict(records[int(rng.integers(len(records)))],seed=int(rng.integers(1,2**30))))
        draws.append(batch)
    plan=dict(source=str(STAGE/'training_draws.json'),source_sha256=sha(STAGE/'training_draws.json'),
              cutoff_cycles_per_patch=CUTOFF,kind='action_gripper_open',name=NAME,draws=draws)
    if (root/'training_draws.json').exists():assert json.loads((root/'training_draws.json').read_text())==plan
    else:dump(root/'training_draws.json',plan)


def train(root):
    p = load(root)['method_study']; plan = json.loads((root/'training_draws.json').read_text())
    draws = plan['draws']; assert len(draws) >= p['steps'] and all(len(b) == p['batch'] for b in draws)
    dest = root/'training'/NAME; dest.mkdir(parents=True, exist_ok=True); ckpt = dest/'state.npz'
    q = np.full((1, 3, 128, 128), .5, np.float32); history = []; start = 0
    if ckpt.exists():
        with np.load(ckpt) as z:
            assert str(z['protocol_sha256'])==sha(root/'protocol.json') and str(z['draws_sha256'])==sha(root/'training_draws.json')
            q = z['texture'].copy(); start = int(z['iteration']); history = json.loads(str(z['history']))
    for it in range(start, p['steps']):
        gradient = np.zeros_like(q); stage_l0 = {}
        for d in draws[it]:
            r = request(dict(command='method_gradient', kind='action_gripper_open', texture=q, record=local(d['record']), seed=d['seed'], active_start=d['active_start']), root)
            gradient += r['gradient']/p['batch']; stage_l0.setdefault(d['stage'], []).append(r['loss'])
        assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 0
        q = bandlimit(q+p['lr']*np.sign(gradient), CUTOFF)
        row = dict(iteration=it+1,total=p['steps'], open_target_loss=float(np.mean([v for vs in stage_l0.values() for v in vs])),
                   per_stage={k: float(np.mean(v)) for k, v in stage_l0.items()}, gradient_norm=float(np.linalg.norm(gradient)))
        history.append(row)
        tmp = ckpt.with_suffix('.tmp.npz'); np.savez_compressed(tmp, texture=q, iteration=it+1, history=json.dumps(history),protocol_sha256=sha(root/'protocol.json'),draws_sha256=sha(root/'training_draws.json')); tmp.replace(ckpt)
        if it == 0 or (it+1) % 25 == 0:
            out = root/'patches'/NAME; out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.rint(q[0].transpose(1, 2, 0)*255).astype(np.uint8)).save(out/f'texture_{it+1:04d}.png')
        dump(dest/'progress.json', row); print('[train]', row, flush=True)
    dump(dest/'complete.json', dict(complete=True, protocol_sha256=sha(root/'protocol.json'),draws_sha256=sha(root/'training_draws.json'),texture_sha256=sha(root/'patches'/NAME/'texture_0500.png')))


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
            backend = subprocess.Popen([PY, '-u', str(BASE/'gripper_open_backend.py'), '--root', str(root)], cwd=BASE, env=env,
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
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=BASE/'out/gripper_open_lowfreq8_20261029')
    ap.add_argument('--gpu', required=True)
    main(ap.parse_args())
