"""E31: frozen proxy-trained texture, world XY translation only."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'out/position_sensitivity'
MODEL = '/home/user1/miniconda3/envs/openpi-server/bin/python'
SIM = '/home/user1/miniconda3/envs/openpi-libero/bin/python'
TASK = 'put_the_bowl_on_the_plate'
POSITIONS = [('origin', 0, 0)] + [
    (axis+('_plus' if sign > 0 else '_minus')+str(cm)+'cm',
     sign*cm/100 if axis == 'x' else 0,
     sign*cm/100 if axis == 'y' else 0)
    for cm in [1, 3] for axis in ['x', 'y'] for sign in [-1, 1]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2)+'\n')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    assert not (DEST/'protocol.json').exists(), 'preserve previous runs'
    old = json.loads((ROOT/'out/dual_view/manifest_both/manifest.json').read_text())
    textures = {'random': old['textures']['random'],
                'distance': str(ROOT/'out/direction_loss/train_distance/texture_0500.png')}
    assert sha(textures['distance']) == old['texture_sha256']['optimized']
    clean = ROOT/'out/render_proxy_scale/clean_baseline/episodes.json'
    sources = ['position_sensitivity_pipeline.py', 'position_sensitivity_rollout.py',
               'render_proxy_policy_server.py', 'scene_patch.py', 'drift_eval_digital.py', 'drift_resize_order.py']
    protocol = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'task': TASK, 'question': 'How sensitive is one frozen proxy-trained physical texture to world XY translations?',
        'positions': [{'name': n, 'offset_xy_m': [x, y]} for n, x, y in POSITIONS],
        'source_geometry': old['geometry'], 'conditions': list(textures),
        'texture_checkpoint': 'E29 distance500, byte-identical to E26 full dual-view500',
        'train_new_patch': False, 'texture_adaptation': False,
        'indices': list(range(40, 50)), 'new_episodes': 180,
        'clean_reference': str(clean), 'clean_reference_sha256': sha(clean),
        'fixed': 'task, patch size13cm, orientation, texture bytes, init state, environment seed, per-replan noise, 300step horizon',
        'environment_seed_rule': '10000+init', 'action_noise_rule': '20261006+init*1000+replan',
        'primary_metric': 'Task failures among the nine clean-success initial states',
        'visibility': 'actual object-ID segmentation per input; summarize each episode then positions',
        'limits': ['One source texture/seed/task; exploratory reused10states',
                   'Physical translation changes projection, sampling, visibility and object relationship together',
                   'No retraining or position selection using outcomes; not a pure image-pixel translation test'],
        'source_sha256': {n: sha(ROOT/n) for n in sources}}
    write(DEST/'protocol.json', protocol)
    snap = DEST/'source_snapshot'; snap.mkdir()
    for name in sources:
        shutil.copy2(ROOT/name, snap/name)
    for name, dx, dy in POSITIONS:
        folder = DEST/name; folder.mkdir()
        center = list(old['geometry']['cell_world']); center[0] += dx; center[1] += dy
        assert abs(center[0])+.065 < .5 and abs(center[1])+.065 < .6
        manifest = {'stage': 'frozen_position_sensitivity', 'source_task': TASK,
                    'geometry': {'patch_m': .13, 'cell_world': center},
                    'offset_xy_m': [dx, dy], 'textures': textures,
                    'texture_sha256': {k: sha(v) for k, v in textures.items()}, 'frequency': 'full128'}
        write(folder/'manifest.json', manifest)
    write(DEST/'progress.json', {'stage': 'running', 'planned_new_episodes': 180})

    def worker(worker_id, jobs):
        gpu = worker_id+2; port = 8172+worker_id
        service = DEST/('worker'+str(worker_id)); service.mkdir()
        with (service/'policy_server.log').open('w') as log:
            server = subprocess.Popen([MODEL, 'render_proxy_policy_server.py', '--port', str(port)], cwd=ROOT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4'},
                stdout=log, stderr=subprocess.STDOUT)
            write(service/'owned_service.json', {'pid': server.pid, 'gpu': gpu, 'port': port})
            try:
                start = time.monotonic()
                while '[serve] ready' not in (service/'policy_server.log').read_text():
                    assert server.poll() is None and time.monotonic()-start < 900, 'server startup failed'
                    time.sleep(2)
                for name, _, _ in jobs:
                    folder = DEST/name; started = time.monotonic()
                    with (folder/'rollout.log').open('w') as output:
                        subprocess.run([SIM, 'position_sensitivity_rollout.py', '--manifest', str(folder/'manifest.json'),
                            '--task', TASK, '--clean-reference', str(clean), '--conditions', ','.join(textures),
                            '--indices', ','.join(map(str, range(40, 50))), '--port', str(port),
                            '--out', 'position_sensitivity/'+name+'/rollout'], cwd=ROOT,
                            env={**os.environ, 'MUJOCO_EGL_DEVICE_ID': str(gpu), 'OMP_NUM_THREADS': '2'},
                            stdout=output, stderr=subprocess.STDOUT, check=True, timeout=7200)
                    write(folder/'complete.json', {'complete': True, 'new_episodes': 20, 'elapsed_s': time.monotonic()-started})
                    print('[done]', name, flush=True)
            finally:
                if server.poll() is None: server.terminate()
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired: server.kill(); server.wait(timeout=10)
                write(service/'service_stopped.json', {'pid': server.pid, 'stopped': True})
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, i, POSITIONS[i::3]) for i in range(3)]
            for future in futures: future.result()
        write(DEST/'progress.json', {'stage': 'measurements_complete', 'new_episodes': 180})
    except BaseException as error:
        write(DEST/'progress.json', {'stage': 'failed', 'error': repr(error)})
        raise


if __name__ == '__main__':
    main()
