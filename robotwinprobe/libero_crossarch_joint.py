"""Matched 10cm LIBERO bidirectional transfer, followed by white-box joint training.

All conditions use one world texture visible through both native cameras.
Single-model and joint textures have fixed 500-update budgets; no ASR selection.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
from crossarch_common import ROOT, AUTH, dump, sha
from fastwam_libero_trial import CALIBRATION, PY as FAST_PY, SIM
from fastwam_libero_both import DEFAULT as SOURCE
from wam_orthogonal_control import call

DEFAULT=ROOT/'robotwinprobe/out/libero_pi_fastwam_crossarch_10cm_20261028'
PI_PY='/home/user1/miniconda3/envs/openpi-server/bin/python'
REPORT=ROOT/'result/44_libero_pi_fastwam_crossarch.md'
CONDITIONS=['clean','random','pi','fast','joint']


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    original=json.loads((SOURCE/'protocol.json').read_text())
    p=dict(experiment='libero_pi_fastwam_transfer_joint_10cm',source=str(SOURCE),
        source_protocol_sha256=sha(SOURCE/'protocol.json'),calibration_sha256=sha(CALIBRATION),
        patch=original['patch'],training=copy.deepcopy(original['training']),
        conditions=CONDITIONS,indices=original['evaluation']['indices'],
        transfer='fixed final source texture; no target gradient or target outcome selection',
        joint='new gray texture; maximize .5*Lpi/Spi + .5*Lfast/Sfast; fixed positive scales from Random on first training batch',
        cameras='shared world texture, base WorldTextureProxy and per-state wrist CameraProxy',
        objective='first-flow effective7 velocity MSE; pi horizon10, FastWAM horizon32; absolute losses not cross-model comparable',
        baseline='all episodes newly evaluated, each model native horizon/replan, same physics and initial-state indices',
        selection='final update500 only, no validation or ASR checkpoint selection')
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:dump(path,p)
    (root/'textures').mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    for name,src in [('random',SOURCE/'random.png'),('fast',SOURCE/'patch/texture_0500.png')]:
        dest=root/'textures'/(name+'.png')
        if not dest.exists():shutil.copy2(src,dest)
        assert sha(dest)==sha(src)
    plan=root/'training_plan.json'
    if not plan.exists():shutil.copy2(SOURCE/'training_plan.json',plan)
    assert sha(plan)==sha(SOURCE/'training_plan.json')
    for kind,port in [('pi',18642),('fast',18643)]:
        r=root/'models'/kind;r.mkdir(parents=True,exist_ok=True)
        cfg=copy.deepcopy(original)
        cfg.update(experiment=p['experiment'],model_kind=kind,control_port=port,parent_root=str(root),
            checkpoint=str(ROOT/'checkpoints/pi05_libero_pytorch') if kind=='pi' else original['checkpoint'])
        cfg['evaluation'].update(conditions=CONDITIONS,baseline_reuse=False,
            horizon=10 if kind=='pi' else 32,replan=5 if kind=='pi' else 10,
            texture_paths={n:str(root/'textures'/(n+'.png')) for n in CONDITIONS if n!='clean'})
        cfg['training']=copy.deepcopy(p['training']);cfg['training']['objective']=p['objective'];cfg['training']['cameras']=p['cameras']
        dest=r/'protocol.json'
        if dest.exists():assert json.loads(dest.read_text())==cfg
        else:dump(dest,cfg)
        if kind=='fast':
            for name in ['prompts.npz','checkpoints/libero_uncond_2cam224.pt','checkpoints/libero_uncond_2cam224_dataset_stats.json']:
                dest=r/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():os.link(SOURCE/name,dest)
    return p


def report(root):
    s=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'prepared'}
    lines=['# LIBERO π0.5 / FastWAM：10cm双相机迁移与联训','',f'状态：`{s["stage"]}`。',
        '单任务put_the_bowl_on_the_plate，10×10cm、Q128、固定桌面位置；同一纹理同时经过主摄和腕部成像。',
        'π单模型新训500×8；FastWAM使用已完成的10cm双相机最终纹理。双向迁移不使用目标模型梯度。',
        '联训另从灰色起步500×8，最大化两模型归一化L0的等权和；固定尺度取首个训练batch的Random响应。',
        'L0只计算有效7维首步速度MSE；π动作跨度10、每5步重规划；FastWAM跨度32、每10步重规划。两模型均原生10步去噪。',
        '全部闭环重新评估40–49初态；ASR分母分别为各模型clean成功集。单任务10初态为探索性结果。','',
        '| 目标模型 | 纹理来源 | 已完成 | clean成功配对失败 |','|---|---|---:|---:|']
    outcomes={}
    for kind in ['pi','fast']:
        rr={}
        for c in CONDITIONS:
            rr[c]={r['init_index']:r for f in (root/'models'/kind/'episodes'/c).glob('*.json') for r in [json.loads(f.read_text())] if 'success' in r}
        outcomes[kind]=rr;clean={i for i,r in rr['clean'].items() if r['success']}
        for c in CONDITIONS:
            paired=clean & rr[c].keys()
            lines.append(f'| {kind} | {c} | {len(rr[c])}/10 | '+('—' if c=='clean' else f'{sum(not rr[c][i]["success"] for i in paired)}/{len(paired)}')+' |')
    for mode in ['pi','joint']:
        f=root/'training'/mode/'progress.json'
        lines+=['',mode+'更新：'+(str(json.loads(f.read_text())['iteration'])+'/500' if f.exists() else '未开始')]
    if s.get('error'):lines+=['',s['error']]
    lines+=['',f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(lines)+'\n')
    dump(root/'outcomes.json',outcomes)


def status(root,stage,**kw):
    dump(root/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw));report(root)


def server(root,kind):
    import torch
    from multiprocessing.connection import Listener
    if kind=='pi':from libero_pi_adapter import Backend
    else:from fastwam_adapter import Backend
    sys.path.insert(0,str(ROOT/'pi05probe'))
    from render_world_proxy import WorldTextureProxy
    from dual_view_proxy import CameraProxy
    torch.set_num_threads(4);r=root/'models'/kind;b=Backend(r)
    with np.load(CALIBRATION) as z:
        d={k:z[k].copy() for k in ['state8','clean_img256','clean_wrist','train_mask','validation_mask']};prompt=str(z['prompt'])
    base=WorldTextureProxy(CALIBRATION,device='cuda',fit=SOURCE/'proxy_fit/fit.json')
    wrist=CameraProxy(SOURCE/'camera_proxy/calibration/wrist.npz',device='cuda',fit=SOURCE/'camera_proxy/fit/wrist.json')
    cache={}
    def views(k):return [torch.tensor(d[n][k].copy(),device='cuda').permute(2,0,1)[None].float()/255 for n in ['clean_img256','clean_wrist']]
    def response(k,seed,images):
        v,_=b.response(b.prepare(d['state8'][k],prompt,images),seed,full=False)
        return v[...,:7].float()
    def objective(req):
        k=req['index'];seed=req['seed'];key=(k,seed)
        if key not in cache:
            with torch.no_grad():cache[key]=response(k,seed,views(k)).cpu()
        q=torch.tensor(req['texture'],device='cuda',requires_grad=req['command']=='gradient')
        value=(response(k,seed,[base(q,k,native=True),wrist(q,k,native=True)])-cache[key].cuda()).square().mean()
        assert torch.isfinite(value)
        out=dict(loss=float(value.detach()))
        if req['command']=='gradient':
            g,=torch.autograd.grad(value,q)
            assert torch.isfinite(g).all() and float(g.norm())>0
            assert all(p.grad is None for p in b.model.parameters())
            out.update(gradient=g.cpu().numpy(),gradient_norm=float(g.norm()))
        return out
    def audit():
        rows=[]
        with torch.no_grad():
            for k in np.flatnonzero(d['validation_mask'])[:2]:
                prepared=b.prepare(d['state8'][k],prompt,views(k));_,a=b.response(prepared,20261201,full=True)
                if kind=='pi':native=b.native_full(prepared,20261201)
                else:
                    image,proprio,(context,mask)=prepared
                    native=b.model.infer_action(prompt=None,input_image=image,proprio=proprio,context=context,context_mask=mask,
                        action_horizon=32,num_inference_steps=10,seed=20261201,rand_device='cpu')['action']
                    native=native[None].float().cpu()
                error=float((a.float().cpu()-native).abs().max());assert error==0,('native parity',kind,error)
                rows.append(dict(index=int(k),max_abs=error))
        k=int(np.flatnonzero(d['train_mask'])[0]);q=torch.full((1,3,128,128),.5,device='cuda',requires_grad=True)
        ib,iw=base(q,k,native=True),wrist(q,k,native=True)
        with torch.no_grad():ref=response(k,20261201,views(k))
        loss=(response(k,20261201,[ib,iw])-ref).square().mean()
        cb,cw=torch.autograd.grad(loss,(ib,iw),retain_graph=True)
        gb,=torch.autograd.grad(ib,q,grad_outputs=cb,retain_graph=True)
        gw,=torch.autograd.grad(iw,q,grad_outputs=cw,retain_graph=True)
        total,=torch.autograd.grad((ib,iw),q,grad_outputs=(cb,cw))
        assert float(gb.norm())>0 and float(gw.norm())>0
        error=float((gb+gw-total).norm()/total.norm());assert error<1e-4
        out=dict(passed=True,native=rows,base_gradient_norm=float(gb.norm()),wrist_gradient_norm=float(gw.norm()),camera_vjp_relative_error=error)
        dump(r/'audit/native_gradient.json',out);return out
    p=json.loads((r/'protocol.json').read_text())
    with Listener(('127.0.0.1',p['control_port']),authkey=AUTH) as listener:
        dump(r/'ready.json',dict(pid=os.getpid(),ready=True));running=True
        while running:
            with listener.accept() as c:
                try:
                    req=c.recv();cmd=req['command']
                    if cmd=='shutdown':running=False;out={'stopped':True}
                    elif cmd=='audit':out=audit()
                    elif cmd in ['gradient','loss']:
                        with torch.set_grad_enabled(cmd=='gradient'):out=objective(req)
                    elif cmd=='infer':out=b.infer(req)
                    elif cmd=='trim':torch.cuda.empty_cache();out={'trimmed':True}
                    else:raise ValueError(cmd)
                    c.send(out)
                except Exception as e:
                    traceback.print_exc();c.send(dict(error=repr(e),traceback=traceback.format_exc()))


def train(root,mode):
    plan=json.loads((root/'training_plan.json').read_text());dest=root/'training'/mode;dest.mkdir(parents=True,exist_ok=True)
    scales={'pi':1.} if mode=='pi' else json.loads((root/'scales.json').read_text())['scales']
    q=np.full((1,3,128,128),.5,np.float32);history=[];checkpoint=dest/'state.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as z:q=z['texture'].copy();history=json.loads(str(z['history']))
    for it in range(len(history),500):
        g=np.zeros_like(q);raw={kind:[] for kind in scales}
        for kind,scale in scales.items():
            for k,seed in zip(plan[it]['indices'],plan[it]['seeds']):
                row=call(root/'models'/kind,'gradient',index=k,seed=seed,texture=q)
                raw[kind].append(row['loss']);g+=row['gradient']/(8*len(scales)*scale)
        assert np.isfinite(g).all() and np.linalg.norm(g)>0
        q=np.clip(q+.01*np.sign(g),0,1)
        losses={k:float(np.mean(v)) for k,v in raw.items()}
        history.append(dict(iteration=it+1,raw=losses,objective=float(np.mean([losses[k]/scales[k] for k in scales])),updated=time.time()))
        temp=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(temp,texture=q,history=json.dumps(history));temp.replace(checkpoint)
        dump(dest/'progress.json',dict(complete=it==499,**history[-1]))
        if it==0 or (it+1)%25==0:
            Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png');report(root)
        print('[crossarch train]',mode,it+1,losses,flush=True)
    shutil.copy2(dest/'texture_0500.png',root/'textures'/(mode+'.png'))


def pipeline(root,gpu):
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=open(root/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(root);services={}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    snapshot=root/'source_snapshots'/str(os.getpid());snapshot.mkdir(parents=True,exist_ok=True)
    names=['libero_crossarch_joint.py','libero_pi_adapter.py','fastwam_libero_trial.py','fastwam_adapter.py']
    for name in names:shutil.copy2(ROOT/'robotwinprobe'/name,snapshot/name)
    dump(snapshot/'manifest.json',{n:sha(snapshot/n) for n in names})
    def start(kind):
        if kind in services:return
        status(root,'start_'+kind);r=root/'models'/kind;ready=r/'ready.json'
        if ready.exists():ready.unlink()
        log=(root/'logs'/(kind+'_backend.log')).open('a')
        services[kind]=subprocess.Popen([PI_PY if kind=='pi' else FAST_PY,'-u',str(Path(__file__).resolve()),'server','--root',str(root),'--kind',kind],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT);log.close()
        began=time.monotonic()
        while not ready.exists():
            if services[kind].poll() is not None or time.monotonic()-began>600:raise RuntimeError(kind+' backend failed; inspect logs')
            time.sleep(2)
        status(root,'audit_'+kind);call(r,'audit');call(r,'trim')
    def evaluate(kind,conditions):
        for condition in conditions:
            for index in range(40,50):
                dest=root/'models'/kind/'episodes'/condition/f'init{index}.json'
                if dest.exists() and 'success' in json.loads(dest.read_text()):continue
                status(root,f'evaluate_{kind}_{condition}_{index}')
                with (root/'logs'/f'{kind}_{condition}_{index}.log').open('a') as log:
                    subprocess.run([SIM,'-u',str(ROOT/'robotwinprobe/fastwam_libero_trial.py'),'rollout','--root',str(root/'models'/kind),'--condition',condition,'--index',str(index)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if condition=='clean':
                rows=[json.loads(f.read_text()) for f in (root/'models'/kind/'episodes/clean').glob('*.json')]
                assert sum(r.get('success',False) for r in rows)>=6,kind+' clean gate'
    try:
        # Verify fixed public FastWAM source, independently of target outcomes.
        assert sha(root/'models/fast/checkpoints/libero_uncond_2cam224.pt')=='1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579'
        start('pi');evaluate('pi',['clean','random','fast'])
        status(root,'train_pi');train(root,'pi');call(root/'models/pi','trim')
        evaluate('pi',['pi']);call(root/'models/pi','trim')
        start('fast');evaluate('fast',['clean','random','fast','pi']);call(root/'models/fast','trim')
        scale_path=root/'scales.json'
        if not scale_path.exists():
            status(root,'fixed_training_random_scales');first=json.loads((root/'training_plan.json').read_text())[0]
            q=np.asarray(Image.open(root/'textures/random.png').convert('RGB'),dtype=np.float32).transpose(2,0,1)[None]/255
            rows={kind:[call(root/'models'/kind,'loss',index=k,seed=seed,texture=q)['loss'] for k,seed in zip(first['indices'],first['seeds'])] for kind in ['pi','fast']}
            scales={k:float(np.mean(v)) for k,v in rows.items()}
            assert all(np.isfinite(v) and v>0 for v in scales.values())
            dump(scale_path,dict(scales=scales,rows=rows,source='first training batch, Random, before joint optimization',plan_sha256=sha(root/'training_plan.json')))
        status(root,'train_joint');train(root,'joint')
        for kind in ['pi','fast']:call(root/'models'/kind,'trim');evaluate(kind,['joint'])
        status(root,'closed_loop_complete')
    except Exception as e:
        status(root,'runtime_error',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        for kind,proc in services.items():
            if proc.poll() is None:
                try:call(root/'models'/kind,'shutdown')
                except Exception:proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','pipeline','server','report'])
    ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--kind',choices=['pi','fast']);ap.add_argument('--gpu',type=int,default=4)
    a=ap.parse_args()
    if a.command=='pipeline':pipeline(a.root,a.gpu)
    elif a.command=='server':server(a.root,a.kind)
    else:globals()[a.command](a.root)
