"""Bounded E26 positive-control replay and fresh 13cm pi patch retraining.
No FastWAM/joint reruns until the pi source attack is re-established.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from crossarch_common import ROOT, dump, sha

PROBE=ROOT/'pi05probe'
SOURCE=PROBE/'out/dual_view'
DEFAULT=PROBE/'out/pi13_restore_20261028'
PY='/home/user1/miniconda3/envs/openpi-server/bin/python'
SIM='/home/user1/miniconda3/envs/openpi-libero/bin/python'
REPORT=ROOT/'result/45_libero_pi13_restore.md'


def report(root):
    state=json.loads((root/'status.json').read_text())
    lines=['# LIBERO π：13cm旧阳性对照与重新训练','',f'状态：`{state["stage"]}`。',
        '恢复旧E26配置：13cm@101，两路CameraProxy，155帧，固定noise20260901、batch seed20261005，首步10×32速度差SSE；灰色Q128，500×8，sign梯度上升.01。',
        '真实render闭环采用旧协议：等待10步、最多300动作、每5步重规划，init40–49及原动作noise规则。',
        '先新测clean、Random、旧成功纹理；阳性对照复现后，独立从灰色重训并评估固定第500步纹理。',
        '这是同配方复现，不是独立训练seed的稳健性验证；不据此分离10cm零结果的各个原因。','',
        '| 条件 | 完成回合 | 任务成功 | 配对攻击成功 |','|---|---:|---:|---:|']
    for folder,label in [('controls','旧纹理对照'),('retrained','新训纹理')]:
        path=root/folder/'episodes.json'
        if not path.exists():continue
        data=json.loads(path.read_text())
        for c,row in data['summary'].items():
            if folder=='retrained' and c=='clean':continue
            lines.append(f'| {label}/{c} | {row["n"]}/10 | {row["successes"]} | '+('—' if c=='clean' else str(row.get('paired_ASR_failures'))+'/'+str(row.get('paired_ASR_clean_success_n')))+' |')
    progress=root/'train/progress.json'
    if progress.exists():lines+=['','训练：'+str(json.loads(progress.read_text())['completed_iters'])+'/500。']
    if state.get('error'):lines+=['',state['error']]
    lines+=['',f'记录：`{root}`。']
    REPORT.write_text('\n'.join(lines)+'\n')


def status(root,stage,**kw):
    dump(root/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw));report(root)


def main(root,gpu,port):
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'logs').mkdir(exist_ok=True)
    cfg=json.loads((SOURCE/'train_both/config.json').read_text())
    for n,h in cfg['source_sha256'].items():assert sha(PROBE/n)==h,('historical code differs',n)
    for pathkey,hashkey in [('calibration_path','calibration_sha256'),('fit_path','fit_sha256'),('wrist_calibration_path','wrist_calibration_sha256'),('wrist_fit_path','wrist_fit_sha256')]:
        assert sha(cfg[pathkey])==cfg[hashkey],pathkey
    frozen=json.loads((SOURCE/'manifest_both/manifest.json').read_text())
    for c,path in frozen['textures'].items():
        if path:assert sha(path)==frozen['texture_sha256'][c]
    protocol=dict(source_config=cfg,source_manifest=frozen,gpu=gpu,port=port,
        training='fresh gray Q; exactly E26 hyperparameters and sampling/noise',
        evaluation=dict(indices=list(range(40,50)),wait=10,steps=300,replan=5),
        positive_control_gate='fresh clean and old patch outcomes must match historical per-initial-state outcomes before training')
    p=root/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:dump(p,protocol)
    snapshot=root/'source_snapshots';snapshot.mkdir(exist_ok=True)
    names=['dual_view_train.py','dual_view_proxy.py','drift_patch_train.py','render_scale_rollout.py','render_proxy_policy_server.py','drift_eval_digital.py']
    for n in names:
        target=snapshot/n
        if not target.exists():shutil.copy2(PROBE/n,target)
        assert sha(target)==sha(PROBE/n)
    dump(snapshot/'manifest.json',{n:sha(snapshot/n) for n in names})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',JAX_PLATFORMS='cpu',HF_HUB_OFFLINE='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    service=None
    def job(stage,argv):
        status(root,stage)
        with (root/'logs'/(stage+'.log')).open('a') as log:
            subprocess.run(argv,cwd=PROBE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    def stop():
        nonlocal service
        if service is not None and service.poll() is None:
            service.terminate()
            try:service.wait(timeout=30)
            except subprocess.TimeoutExpired:service.kill();service.wait()
        service=None
    def start(tag):
        nonlocal service
        status(root,'start_policy_'+tag)
        path=root/'logs'/('policy_'+tag+'.log')
        with path.open('w') as log:
            service=subprocess.Popen([PY,'-u',str(PROBE/'render_proxy_policy_server.py'),'--port',str(port)],cwd=PROBE,env=env,stdout=log,stderr=subprocess.STDOUT)
        began=time.monotonic()
        while '[serve] ready' not in path.read_text():
            if service.poll() is not None or time.monotonic()-began>600:raise RuntimeError('policy startup failed: '+str(path))
            time.sleep(2)
    def rollout(folder,manifest,conditions,reference=None):
        dest=root/folder
        if (dest/'episodes.json').exists():
            data=json.loads((dest/'episodes.json').read_text())
            assert all(len(data['per_episode'].get(c,[]))==10 for c in conditions.split(',')), 'partial rollout retained; requires explicit resume handling'
            return
        cmd=[SIM,'-u',str(PROBE/'render_scale_rollout.py'),'--manifest',str(manifest),'--conditions',conditions,'--indices',','.join(map(str,range(40,50))),'--port',str(port),'--out',str(dest)]
        if reference:cmd+=['--clean-reference',str(reference)]
        job('rollout_'+folder,cmd)
    try:
        start('controls')
        rollout('controls',SOURCE/'manifest_both/manifest.json','clean,random,optimized')
        stop()
        status(root,'positive_control_check')
        fresh=json.loads((root/'controls/episodes.json').read_text())['per_episode']
        old=json.loads((SOURCE/'rollout_both/episodes.json').read_text())['per_episode']
        comparisons={}
        for c in ['clean','optimized']:
            a={x['init_index']:x['success'] for x in fresh[c]};b={x['init_index']:x['success'] for x in old[c]}
            comparisons[c]=dict(new=a,historical=b,exact=a==b)
        dump(root/'audit/positive_control.json',comparisons)
        assert all(x['exact'] for x in comparisons.values()),'old positive control did not reproduce; stop before training'
        finished=root/'train/results.json'
        if not finished.exists():
            cmd=[PY,'-u',str(PROBE/'dual_view_train.py'),'--calibration',cfg['calibration_path'],'--fit',cfg['fit_path'],
                '--wrist-calibration',cfg['wrist_calibration_path'],'--wrist-fit',cfg['wrist_fit_path'],
                '--camera-mode','both','--out',str(root/'train'),'--iters','500','--batch','8','--lr','.01','--seed',str(cfg['seed_batch'])]
            checkpoints=sorted((root/'train').glob('checkpoint_*.npz'))
            if checkpoints:cmd+=['--resume',str(checkpoints[-1])]
            job('training',cmd)
        else:assert json.loads(finished.read_text())['complete']
        with np.load(SOURCE/'train_both/checkpoint_0500.npz') as a,np.load(root/'train/checkpoint_0500.npz') as b:
            assert np.array_equal(a['batches'],b['batches']) and np.array_equal(a['noise'],b['noise'])
            dump(root/'audit/training_comparison.json',dict(batches_exact=True,noise_exact=True,
                texture_exact=bool(np.array_equal(a['Q'],b['Q'])),texture_mean_abs_difference=float(np.abs(a['Q']-b['Q']).mean()),
                historical_png_sha256=sha(SOURCE/'train_both/texture_0500.png'),new_png_sha256=sha(root/'train/texture_0500.png')))
        manifest=root/'manifest/manifest.json'
        if not manifest.exists():job('manifest',[PY,str(PROBE/'render_scale_manifest.py'),'--run',str(root/'train'),'--out',str(root/'manifest')])
        start('retrained')
        rollout('retrained',manifest,'optimized',root/'controls/episodes.json')
        stop();status(root,'complete')
    except Exception as e:
        status(root,'runtime_error',error=repr(e),traceback=traceback.format_exc());raise
    finally:stop()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--gpu',type=int,default=4);ap.add_argument('--port',type=int,default=8164)
    a=ap.parse_args();main(a.root,a.gpu,a.port)
