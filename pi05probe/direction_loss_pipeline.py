"""E29 fixed-budget directional loss training and true-render evaluation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'out/direction_loss'
MODEL = '/home/user1/miniconda3/envs/openpi-server/bin/python'
SIM = '/home/user1/miniconda3/envs/openpi-libero/bin/python'
MODES = ['distance', 'orthogonal', 'reverse']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def command(args, log, env):
    with (DEST/log).open('w') as f:
        subprocess.run(args, cwd=ROOT, env={**os.environ, **env}, stdout=f,
                       stderr=subprocess.STDOUT, check=True, timeout=5400)


def state(stage, **extra):
    (DEST/'progress.json').write_text(json.dumps({'stage': stage, **extra}, indent=2)+'\n')
    print('[stage]', stage, flush=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    assert not (DEST/'protocol.json').exists(), 'preserve prior trials'
    protocol = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'question': 'Does angular early-flow optimization outperform squared distance at equal budget?',
        'objectives_maximized': {'distance': 'sum((v_adv-v_clean)^2)',
                                 'orthogonal': '1-cos(v_adv,v_clean)^2',
                                 'reverse': '-cos(v_adv,v_clean)'},
        'vector': 'first flow step t=1, flatten10x32 v itself, not delta-v or physical motion',
        'norm_guard': 'stop if adversarial/clean norm <= 0.001; never label zero as orthogonal',
        'iterations': 500, 'batch': 8, 'lr': .01, 'seed_batch': 20261005,
        'noise_seed': 20260901, 'initialization': 'constant gray0.5',
        'parameterization': 'one shared full Q128, both calibrated camera proxies, no EOT',
        'geometry': '13cm@101', 'checkpoint_selection': 'fixed500; no selection from test',
        'train_episodes': list(range(10)), 'validation_episodes': [10,11],
        'test_episodes': list(range(40,50)), 'new_closedloop_count': 30,
        'monitoring': 'same8train+32val, iterations0/100/200/300/400/500, proxy same fixed training noise',
        'diagnostics': '32 matched states,3noise, both true render and proxy; full10-step denoise and final action angles',
        'clean_and_random_controls': 'Reuse E26 source-task clean and full random with identical geometry/seeds',
        'preflight': 'Old fullboth render v angle mean4.0729deg over10x32; angular target may be unreachable. Orthogonal/reverse gradients have same sign direction while cos>0 for a single frame; batch weighting differs.',
        'limitations': ['Single task and training seed', 'Previously used test states: exploratory, not blind',
                       '500 fixed-budget comparison is not a convergence guarantee',
                       'Early denoising vector angle is not robot Cartesian motion angle'],
        'source_sha256': {n: sha(ROOT/n) for n in ['direction_objective.py','direction_loss_train.py',
            'direction_loss_capture.py','direction_loss_actions.py','direction_loss_pipeline.py',
            'dual_view_proxy.py','dual_view_probe.py','render_scale_rollout.py','render_proxy_policy_server.py']},
        'calibration_fit_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            [ROOT/'out/dual_view'/folder/(view+ext) for folder,ext in [('calibration','.npz'),('fit','.json')]
             for view in ['base','wrist']]}}
    (DEST/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    state('training'); started = time.monotonic()
    def train(mode, gpu):
        command([MODEL,'direction_loss_train.py','--objective',mode,'--camera-mode','both',
                 '--calibration','out/dual_view/calibration/base.npz','--fit','out/dual_view/fit/base.json',
                 '--wrist-calibration','out/dual_view/calibration/wrist.npz','--wrist-fit','out/dual_view/fit/wrist.json',
                 '--out',str(DEST/('train_'+mode))], 'train_'+mode+'.log',
                {'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'})
        print('[trained]', mode, flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(train, mode, gpu) for mode, gpu in zip(MODES,[2,3,4])]
        for f in futures:
            f.result()
    textures = {mode: str(DEST/('train_'+mode)/'texture_0500.png') for mode in MODES}
    cfg = json.loads((DEST/'train_distance/config.json').read_text())
    manifest = {'stage': 'exploratory_loss_ablation', 'geometry': {'patch_m': cfg['patch_m'],
                'cell_world': cfg['cell_world']}, 'textures': textures,
                'texture_sha256': {k: sha(v) for k,v in textures.items()}, 'frequency': 'full128'}
    (DEST/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    state('true_render_evaluation')
    with (DEST/'policy_server.log').open('w') as log:
        server = subprocess.Popen([MODEL,'render_proxy_policy_server.py','--port','8168'], cwd=ROOT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'5','OMP_NUM_THREADS':'4'}, stdout=log, stderr=subprocess.STDOUT)
        (DEST/'owned_service.json').write_text(json.dumps({'pid':server.pid,'port':8168})+'\n')
        try:
            wait = time.monotonic()
            while '[serve] ready' not in (DEST/'policy_server.log').read_text():
                assert server.poll() is None and time.monotonic()-wait < 600, 'server failed to start'
                time.sleep(2)
            def rollout():
                command([SIM,'render_scale_rollout.py','--manifest',str(DEST/'manifest.json'),
                    '--conditions',','.join(MODES),'--clean-reference',str(ROOT/'out/render_proxy_scale/clean_baseline/episodes.json'),
                    '--indices','40,41,42,43,44,45,46,47,48,49','--port','8168','--out','direction_loss/rollout'],
                    'rollout.log', {'MUJOCO_EGL_DEVICE_ID':'5','OMP_NUM_THREADS':'2'})
            def validation():
                command([SIM,'direction_loss_capture.py'],'validation_capture.log',{'MUJOCO_EGL_DEVICE_ID':'6','OMP_NUM_THREADS':'2'})
                command([MODEL,'dual_view_probe.py','--out','direction_loss/validation','--modes','both'],
                    'validation_probe.log', {'CUDA_VISIBLE_DEVICES':'6','OMP_NUM_THREADS':'4'})
                command([MODEL,'direction_loss_actions.py'],'actions.log',{'CUDA_VISIBLE_DEVICES':'6','OMP_NUM_THREADS':'4'})
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = [ex.submit(rollout), ex.submit(validation)]
                for f in futures:
                    f.result()
            state('measurements_complete', new_rollouts=30, elapsed_s=time.monotonic()-started)
        finally:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait(timeout=10)
            (DEST/'service_stopped.json').write_text(json.dumps({'stopped_pid':server.pid})+'\n')


if __name__ == '__main__':
    main()
