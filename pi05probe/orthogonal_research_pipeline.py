"""E30 direct orthogonality:500-to1000 continuation and true-render evaluation."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'out/orthogonal_research'
MODEL = '/home/user1/miniconda3/envs/openpi-server/bin/python'
SIM = '/home/user1/miniconda3/envs/openpi-libero/bin/python'
RUNS = [('distance_pgd','distance','pgd','distance'),
        ('orthogonal_pgd','orthogonal','pgd','orthogonal'),
        ('orthogonal_adam','orthogonal','adam','orthogonal'),
        ('orthogonal_target','target','pgd','orthogonal')]
MODES = [r[0] for r in RUNS]


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
        'question': 'Can training make v_patch itself perpendicular to v_clean and induce closed-loop failures?',
        'runs': RUNS, 'vector': 'first flow t=1, all10x32 actual v_patch versus v_clean; never delta-v or epsilon-v',
        'objectives': {'distance': 'max sum((v_patch-v_clean)^2)',
                       'orthogonal': 'min cos(v_patch,v_clean)^2',
                       'target': 'min ||v_patch-v_target||^2 / ||v_clean||^2; fixed v_target perpendicular and equal-norm to clean'},
        'target_direction': 'normalize transverse component of initial500 velocity, scale by clean norm, frozen separately per state before updates',
        'initial_checkpoints': {r[0]: str(ROOT/'out/direction_loss'/('train_'+r[3])/'checkpoint_0500.npz') for r in RUNS},
        'initial_checkpoint_sha256': {r[0]: sha(ROOT/'out/direction_loss'/('train_'+r[3])/'checkpoint_0500.npz') for r in RUNS},
        'new_iterations': 500, 'total_lineage_updates': 1000, 'batch': 8, 'lr': .01,
        'seed_batch': 20261005, 'batch_sequence': 'same RNG, skip first500 exact saved draws; use next500',
        'noise_seed': 20260901, 'checkpoint_selection': 'fixed1000 before outcomes, no best-checkpoint selection',
        'adam': {'betas':[.9,.999], 'eps':1e-8, 'weight_decay':0, 'initial_moments':'zero at500'},
        'norm_guard': 'abort if norm_adv/norm_clean <=0.001, report norm alongside angle',
        'near_orthogonal_criterion': '80 to100degrees, nondegenerate norm; not exact90 proof',
        'parameterization': 'same shared full Q128, both calibrated physical-view proxies, no EOT',
        'geometry': '13cm@101', 'train_episodes': list(range(10)), 'validation_episodes':[10,11],
        'test_episodes':list(range(40,50)), 'new_closedloop_count':40,
        'monitoring':'fixed8train+32val every100 new updates;3noise actual render and full actions at1000',
        'reused_controls':'E29 distance/orthogonal500 and E26 clean/random, identical geometry/noise/initial physics',
        'limitations':['One training seed and source task', 'Previously used test states: exploratory',
                       'A fitted orthogonal target need not be reachable through a bounded physical patch',
                       '1000 updates and two optimizers cannot establish a theoretical impossibility'],
        'source_sha256': {n: sha(ROOT/n) for n in ['direction_objective.py','orthogonal_research_objective.py','orthogonal_research_train.py',
            'orthogonal_research_capture.py','orthogonal_research_actions.py','orthogonal_research_pipeline.py',
            'dual_view_proxy.py','dual_view_probe.py','render_scale_rollout.py','render_proxy_policy_server.py']},
        'calibration_fit_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
            [ROOT/'out/dual_view'/folder/(view+ext) for folder,ext in [('calibration','.npz'),('fit','.json')]
             for view in ['base','wrist']]}}
    (DEST/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    state('training'); started = time.monotonic()
    def train(mode, loss, optimizer, parent, gpu):
        command([MODEL,'orthogonal_research_train.py','--objective',loss,'--optimizer',optimizer,
                 '--initial-checkpoint',str(ROOT/'out/direction_loss'/('train_'+parent)/'checkpoint_0500.npz'),'--camera-mode','both',
                 '--calibration','out/dual_view/calibration/base.npz','--fit','out/dual_view/fit/base.json',
                 '--wrist-calibration','out/dual_view/calibration/wrist.npz','--wrist-fit','out/dual_view/fit/wrist.json',
                 '--out',str(DEST/('train_'+mode))], 'train_'+mode+'.log',
                {'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'})
        print('[trained]', mode, flush=True)
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(train, *run, gpu) for run, gpu in zip(RUNS,[2,3,4,5])]
        for f in futures:
            f.result()
    textures = {mode: str(DEST/('train_'+mode)/'texture_1000.png') for mode in MODES}
    cfg = json.loads((DEST/'train_distance_pgd/config.json').read_text())
    manifest = {'stage': 'exploratory_direct_orthogonality', 'geometry': {'patch_m': cfg['patch_m'],
                'cell_world': cfg['cell_world']}, 'textures': textures,
                'texture_sha256': {k: sha(v) for k,v in textures.items()}, 'frequency': 'full128'}
    (DEST/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    state('true_render_evaluation')
    with (DEST/'policy_server.log').open('w') as log:
        server = subprocess.Popen([MODEL,'render_proxy_policy_server.py','--port','8170'], cwd=ROOT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'6','OMP_NUM_THREADS':'4'}, stdout=log, stderr=subprocess.STDOUT)
        (DEST/'owned_service.json').write_text(json.dumps({'pid':server.pid,'port':8170})+'\n')
        try:
            wait = time.monotonic()
            while '[serve] ready' not in (DEST/'policy_server.log').read_text():
                assert server.poll() is None and time.monotonic()-wait < 600, 'server failed to start'
                time.sleep(2)
            def rollout():
                command([SIM,'render_scale_rollout.py','--manifest',str(DEST/'manifest.json'),
                    '--conditions',','.join(MODES),'--clean-reference',str(ROOT/'out/render_proxy_scale/clean_baseline/episodes.json'),
                    '--indices','40,41,42,43,44,45,46,47,48,49','--port','8170','--out','orthogonal_research/rollout'],
                    'rollout.log', {'MUJOCO_EGL_DEVICE_ID':'6','OMP_NUM_THREADS':'2'})
            def validation():
                command([SIM,'orthogonal_research_capture.py'],'validation_capture.log',{'MUJOCO_EGL_DEVICE_ID':'7','OMP_NUM_THREADS':'2'})
                command([MODEL,'dual_view_probe.py','--out','orthogonal_research/validation','--modes','both'],
                    'validation_probe.log', {'CUDA_VISIBLE_DEVICES':'7','OMP_NUM_THREADS':'4'})
                command([MODEL,'orthogonal_research_actions.py'],'actions.log',{'CUDA_VISIBLE_DEVICES':'7','OMP_NUM_THREADS':'4'})
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = [ex.submit(rollout), ex.submit(validation)]
                for f in futures:
                    f.result()
            state('measurements_complete', new_rollouts=40, elapsed_s=time.monotonic()-started)
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
