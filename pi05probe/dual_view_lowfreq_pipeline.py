"""Matched 2x2 low-frequency investigation; reuse frozen block32/both corner."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'out/dual_view_lowfreq'
MODEL = '/home/user1/miniconda3/envs/openpi-server/bin/python'
SIM = '/home/user1/miniconda3/envs/openpi-libero/bin/python'
RUNS = [('bilinear_base', 'bilinear32', 'base', 2),
        ('bilinear_both', 'bilinear32', 'both', 3),
        ('block_base', 'block32', 'base', 4)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(args, log, env=None):
    with (DEST / log).open('w') as f:
        subprocess.run(args, cwd=ROOT, env={**os.environ, **(env or {})},
                       stdout=f, stderr=subprocess.STDOUT, check=True, timeout=4800)


def state(stage, **extra):
    (DEST / 'progress.json').write_text(json.dumps({'stage': stage, **extra}, indent=2)+'\n')
    print('[stage]', stage, flush=True)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    assert not (DEST/'protocol.json').exists(), 'use a fresh output; do not overwrite trials'
    protocol = {
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'question': 'Does adding wrist hurt low-frequency attack, controlling parameterization and imaging proxy?',
        'new_runs': RUNS, 'reused_corner': 'dual_view_both_low/train',
        'iterations': 500, 'batch': 8, 'lr': .01, 'seed': 20261005,
        'noise_seed': 20260901, 'checkpoint_selection': 'fixed500',
        'geometry': '13cm@101', 'train_episodes': list(range(10)),
        'validation_episodes': [10,11], 'test_episodes': list(range(40,50)),
        'new_closedloop_count': 30,
        'random_controls': 'Reuse exact same physical bilinear random E24 and block random E26; same PNG/geometry/noise/states.',
        'diagnostics': '32 states, early velocity/KV per-view and both proxy/render; complete10-step action differences with common noise; optimizer-coordinate gradient statistics.',
        'limitations': ['Single seed/task', 'Previously used test states; exploratory factorial',
                       'OldE24 differs in imaging implementation; not a matched wrist ablation',
                       'Gradient dominance/conflict and action difference are diagnostics, not causal proof of task failure'],
        'source_sha256': {n:sha(ROOT/n) for n in ['dual_view_lowfreq_train.py','dual_view_proxy.py']},
        'calibration_fit_sha256': {str(p.relative_to(ROOT)):sha(p) for p in
                                  [ROOT/'out/dual_view'/folder/(view+ext)
                                   for folder,ext in [('calibration','.npz'),('fit','.json')]
                                   for view in ['base','wrist']]}}
    (DEST/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    state('training')
    def train(run, low, camera, gpu):
        command([MODEL, 'dual_view_lowfreq_train.py', '--calibration', 'out/dual_view/calibration/base.npz',
                 '--fit', 'out/dual_view/fit/base.json', '--wrist-calibration', 'out/dual_view/calibration/wrist.npz',
                 '--wrist-fit', 'out/dual_view/fit/wrist.json', '--out', str(DEST/('train_'+run)),
                 '--camera-mode', camera, '--low-mode', low], 'train_'+run+'.log',
                {'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4'})
        print('[trained]', run, flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures=[ex.submit(train,*run) for run in RUNS]
        for f in futures:f.result()
    textures={run:str(DEST/('train_'+run)/'texture_0500.png') for run,_,_,_ in RUNS}
    cfg=json.loads((DEST/'train_bilinear_base/config.json').read_text())
    manifest={'stage':'exploratory_matched_retest','geometry':{'patch_m':cfg['patch_m'],'cell_world':cfg['cell_world']},
              'textures':textures,'texture_sha256':{k:sha(v) for k,v in textures.items()},
              'frequency':'bilinear32 vs block32, see condition and protocol'}
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    state('true_render_evaluation')
    log=(DEST/'policy_server.log').open('w')
    server=subprocess.Popen([MODEL,'render_proxy_policy_server.py','--port','8162'],cwd=ROOT,
                            env={**os.environ,'CUDA_VISIBLE_DEVICES':'5','OMP_NUM_THREADS':'4'},
                            stdout=log,stderr=subprocess.STDOUT)
    (DEST/'owned_service.json').write_text(json.dumps({'pid':server.pid,'port':8162})+'\n')
    try:
        start=time.monotonic()
        while '[serve] ready' not in (DEST/'policy_server.log').read_text():
            assert server.poll() is None and time.monotonic()-start<600
            time.sleep(2)
        def rollout():
            command([SIM,'render_scale_rollout.py','--manifest',str(DEST/'manifest.json'),
                     '--conditions',','.join(textures),'--clean-reference',str(ROOT/'out/render_proxy_scale/clean_baseline/episodes.json'),
                     '--indices','40,41,42,43,44,45,46,47,48,49','--port','8162','--out','dual_view_lowfreq/rollout'],
                    'rollout.log',{'MUJOCO_EGL_DEVICE_ID':'5','OMP_NUM_THREADS':'2'})
        def validation():
            command([SIM,'dual_view_lowfreq_capture.py'],'validation_capture.log',{'MUJOCO_EGL_DEVICE_ID':'6','OMP_NUM_THREADS':'2'})
            command([MODEL,'dual_view_probe.py','--out','dual_view_lowfreq/validation'],'validation_probe.log',
                    {'CUDA_VISIBLE_DEVICES':'6','OMP_NUM_THREADS':'4'})
            command([MODEL,'dual_view_lowfreq_actions.py'],'actions.log',{'CUDA_VISIBLE_DEVICES':'6','OMP_NUM_THREADS':'4'})
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures=[ex.submit(rollout),ex.submit(validation)]
            for f in futures:f.result()
        state('complete',new_rollouts=30)
    finally:
        if server.poll() is None:server.terminate()
        try:server.wait(timeout=30)
        except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
        log.close()
        (DEST/'service_stopped.json').write_text(json.dumps({'stopped_pid':server.pid})+'\n')


if __name__=='__main__':
    main()
