"""E-MT: same-checkpoint multi-task patch (pi0.5 LIBERO, LIBERO-goal).

capture (4 train tasks, inits 0-11) -> joint training (500x8, E26 recipe)
-> real MuJoCo closed loop on 4 seen + 3 unseen tasks, inits 40-49,
conditions clean / random / single-task E26 patch / multi-task patch.
All tasks/conditions fixed here before any result exists.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'out/multitask_patch'
ENVROOT = os.environ.get('WAMATTACK_ENVROOT', '/home/user1/miniconda3/envs')
MODEL = ENVROOT+'/openpi-server/bin/python'
SIM = ENVROOT+'/openpi-libero/bin/python'
TRAIN_TASKS = ['put_the_bowl_on_the_plate', 'put_the_bowl_on_top_of_the_cabinet',
               'put_the_wine_bottle_on_the_rack', 'turn_on_the_stove']
HELDOUT_TASKS = ['put_the_bowl_on_the_stove', 'put_the_cream_cheese_in_the_bowl',
                 'open_the_middle_drawer_of_the_cabinet']
CONDITIONS = ['clean', 'random', 'single_task', 'multi_task']
INDICES = list(range(40, 50))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')


def stage(name, **kw):
    dump(DEST/'progress.json', dict(stage=name, updated=datetime.datetime.now().isoformat(), **kw))
    print('[stage]', name, kw, flush=True)


class Server:
    def __init__(self, folder, gpu, port):
        self.folder, self.gpu, self.port = Path(folder), gpu, port
        self.folder.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        self.logpath = self.folder/f'policy_server_{self.port}.log'
        self.log = self.logpath.open('w')
        self.proc = subprocess.Popen([MODEL, 'render_proxy_policy_server.py', '--port', str(self.port)], cwd=ROOT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': self.gpu['uuid'], 'OMP_NUM_THREADS': '4'},
            stdout=self.log, stderr=subprocess.STDOUT)
        dump(self.folder/f'owned_service_{self.port}.json', dict(pid=self.proc.pid, port=self.port, gpu=self.gpu))
        start = time.monotonic()
        while '[serve] ready' not in self.logpath.read_text():
            assert self.proc.poll() is None and time.monotonic()-start < 900, ('server failed', self.logpath)
            time.sleep(3)
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None: self.proc.terminate()
        try: self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait(timeout=10)
        self.log.close()


def sim_env(gpu): return {**os.environ, 'MUJOCO_EGL_DEVICE_ID': str(gpu['index']), 'OMP_NUM_THREADS': '2'}


def capture(task, gpu, port):
    out = DEST/'tasks'/task
    if (out/'capture.json').exists(): return
    with Server(DEST/'services'/'capture', gpu, port):
        with (DEST/'logs'/f'capture_{task}.log').open('w') as f:
            subprocess.run([SIM, 'multitask_capture.py', '--task', task, '--port', str(port), '--out', str(out)],
                           cwd=ROOT, env=sim_env(gpu), stdout=f, stderr=subprocess.STDOUT, check=True, timeout=7200)


def evaluate(task, gpu, port):
    folder = DEST/'eval'/task
    if (folder/'rollout/episodes.json').exists(): return
    with Server(DEST/'services'/'eval', gpu, port):
        with (DEST/'logs'/f'eval_{task}.log').open('w') as f:
            subprocess.run([SIM, 'cross_task_transfer_rollout.py', '--manifest', str(DEST/'manifest.json'), '--task', task,
                            '--conditions', ','.join(CONDITIONS), '--indices', ','.join(map(str, INDICES)),
                            '--port', str(port), '--out', str((folder/'rollout').relative_to(ROOT/'out'))],
                           cwd=ROOT, env=sim_env(gpu), stdout=f, stderr=subprocess.STDOUT, check=True, timeout=10800)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', required=True, help='json list of {index,uuid} on this host, preflight-checked')
    args = ap.parse_args()
    gpus = json.loads(args.gpus)
    for name in ['logs', 'tasks', 'eval', 'services']: (DEST/name).mkdir(parents=True, exist_ok=True)
    e26 = ROOT/'out/dual_view'
    protocol = dict(experiment='same-checkpoint multi-task patch', model='pi05_libero (one checkpoint, all LIBERO suites)',
        train_tasks=TRAIN_TASKS, heldout_tasks=HELDOUT_TASKS, conditions=CONDITIONS, test_indices=INDICES,
        capture='per train task clean pi0.5 inits 0-11 (0-9 train, 10-11 validation); E22 pool + E26 dual-view rules',
        training='E26 both-camera recipe: gray init, 500 iters x batch 8 (2/task), sign lr .01, noise 20260901, '
                 'first-step velocity SSE vs same-noise clean; per-frame task prompt; same compute as single-task E26',
        geometry='13cm, position 101 (E26), same for every task; no EOT',
        controls=dict(random=str(e26/'manifest_both/random.png'), single_task=str(e26/'train_both/texture_0500.png'),
                      note='single_task = E26 patch trained only on put_the_bowl_on_the_plate (5/9 source, 0 on E28 targets)'),
        metric='per task: clean success AND attacked failure / clean successes (paired, task-local clean)',
        rollout='cross_task_transfer_rollout.py: env seed 10000+init, noise 20261006+init*1000+replan, 300 steps, replan 5',
        gpus=gpus, host=os.uname().nodename,
        source_sha256={n: sha(ROOT/n) for n in ['multitask_pipeline.py', 'multitask_capture.py', 'multitask_train.py',
                       'cross_task_transfer_rollout.py', 'render_proxy_policy_server.py', 'dual_view_proxy.py']})
    if (DEST/'protocol.json').exists():
        old = json.loads((DEST/'protocol.json').read_text())
        assert {k: v for k, v in old.items() if k not in ['gpus', 'host']} == {k: v for k, v in protocol.items() if k not in ['gpus', 'host']}, 'protocol changed'
    else:
        dump(DEST/'protocol.json', protocol)
    # 1. capture
    stage('capture', tasks=TRAIN_TASKS)
    with ThreadPoolExecutor(max_workers=len(TRAIN_TASKS)) as ex:
        for f in [ex.submit(capture, t, gpus[i % len(gpus)], 8301+i) for i, t in enumerate(TRAIN_TASKS)]: f.result()
    # 2. train
    train_out = DEST/'train'
    if not (train_out/'results.json').exists():
        stage('train')
        with (DEST/'logs/train.log').open('w') as f:
            subprocess.run([MODEL, 'multitask_train.py', '--tasks-root', str(DEST/'tasks'), '--tasks', ','.join(TRAIN_TASKS),
                            '--out', str(train_out)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=14400,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpus[0]['uuid'], 'OMP_NUM_THREADS': '4'})
    # 3. evaluate
    textures = {'clean': None, 'random': e26/'manifest_both/random.png', 'single_task': e26/'train_both/texture_0500.png',
                'multi_task': train_out/'texture_0500.png'}
    manifest = dict(source_task='multi:'+'+'.join(TRAIN_TASKS), stage='multitask_eval',
                    geometry=json.loads((e26/'manifest_both/manifest.json').read_text())['geometry'],
                    textures={k: None if p is None else str(p) for k, p in textures.items()},
                    texture_sha256={k: None if p is None else sha(p) for k, p in textures.items()})
    if (DEST/'manifest.json').exists(): assert json.loads((DEST/'manifest.json').read_text()) == manifest
    else: dump(DEST/'manifest.json', manifest)
    tasks = TRAIN_TASKS+HELDOUT_TASKS
    stage('evaluate', tasks=tasks, episodes=len(tasks)*len(CONDITIONS)*len(INDICES))
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        for f in [ex.submit(evaluate, t, gpus[i % len(gpus)], 8311+i) for i, t in enumerate(tasks)]: f.result()
    stage('complete')


if __name__ == '__main__':
    main()
