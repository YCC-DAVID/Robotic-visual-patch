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
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image
from crossarch_common import ROOT, AUTH, dump, sha
from fastwam_libero_trial import CALIBRATION as CAL10, PY as FAST_PY, SIM
from fastwam_libero_both import DEFAULT as SOURCE
from wam_orthogonal_control import call

# Keep protocol path strings stable across the NFS mount aliases on nnmc65/61.
ROOT=Path('/home/user1/workspace/chence/WAMattack')
SOURCE=ROOT/'robotwinprobe/out/fastwam_libero_both_10cm_20261027'
CAL10=ROOT/'pi05probe/out/fastwam_10cm_calibration_20261027/calibration.npz'
DEFAULT=ROOT/'robotwinprobe/out/libero_pi_fastwam_crossarch_13cm_20261028'
E26=ROOT/'pi05probe/out/dual_view'
RESTORED=ROOT/'pi05probe/out/pi13_restore_20261028'
CALIBRATION=E26/'calibration/base.npz'
PI_PY=os.environ.get('WAMATTACK_PI_PYTHON','/home/user1/miniconda3/envs/openpi-server/bin/python')
FAST_PY=os.environ.get('WAMATTACK_FAST_PYTHON',FAST_PY)
SIM=os.environ.get('WAMATTACK_SIM_PYTHON',SIM)
REPORT=ROOT/'result/46_libero_pi_fastwam_13cm.md'
CONDITIONS=['clean','random','pi','fast','joint']


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    original=json.loads((SOURCE/'protocol.json').read_text())
    with np.load(E26/'train_both/checkpoint_0500.npz') as z:
        batches=z['batches'].tolist()
    plan=[dict(indices=batch,seeds=[20260901]*8) for batch in batches]
    patch=dict(size_m=.13,resolution=128,position='existing fixed table site101',initialization=.5)
    p=dict(experiment='libero_pi_fastwam_13cm_restore_transfer_joint',patch=patch,
        source_pi=str(RESTORED),source_pi_sha256=sha(RESTORED/'train/texture_0500.png'),
        calibration_sha256=sha(CALIBRATION),
        training=dict(steps=500,batch=8,lr=.01,noise_seed=20260901,batch_seed=20261005),
        cameras='same world texture through two per-state CameraProxy branches; original E26 224 input forward',
        loss=dict(pi='original first10x32 velocity SSE',fast='first32x7 velocity MSE'),
        joint='.5*Lpi/Spi+.5*Lfast/Sfast; fixed training-Random scales; fresh gray texture500x8',
        evaluation='each model own clean denominator; pi wait10/max300/replan5, FastWAM wait30/max400/replan10; fixed common initial indices40-49',
        source_selection='fixed update500, no ASR selection; pi13 exact restored positive source',
        diagnosis='10cm legacy pi recipe retraining; old evaluation for new failed10cm, restored10cm and shrunk13cm textures',
        transfer='freeze source PNG and no target gradients; joint is separately labeled dual-model whitebox')
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:dump(path,p)
    dump(root/'training_plan.json',plan)
    (root/'textures').mkdir(exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    for name,source in [('pi',RESTORED/'train/texture_0500.png'),('random',E26/'manifest_both/random.png')]:
        dest=root/'textures'/(name+'.png')
        if not dest.exists():shutil.copy2(source,dest)
        assert sha(dest)==sha(source)
    for kind,port in [('pi',18662),('fast',18663),('pi10',18662)]:
        r=root/'models'/kind;r.mkdir(parents=True,exist_ok=True)
        cfg=copy.deepcopy(original);is_pi=kind!='fast'
        cfg.update(experiment=p['experiment'],model_kind='pi' if is_pi else 'fast',control_port=port,parent_root=str(root),
            calibration=str(CALIBRATION if kind!='pi10' else CAL10),patch=copy.deepcopy(patch),
            checkpoint=str(ROOT/'checkpoints/pi05_libero_pytorch') if is_pi else original['checkpoint'])
        cfg['evaluation'].update(conditions=CONDITIONS,baseline_reuse=False,
            wait=10 if is_pi else 30,steps=300 if is_pi else 400,
            horizon=10 if is_pi else 32,replan=5 if is_pi else 10,
            texture_paths={n:str(root/'textures'/(n+'.png')) for n in CONDITIONS if n!='clean'})
        if kind=='pi10':
            cfg['patch']['size_m']=.10
            cfg['evaluation']['conditions']=['clean','random','legacy10','new10','shrink13']
            cfg['evaluation']['texture_paths']=dict(random=str(SOURCE/'random.png'),legacy10=str(root/'textures/legacy10.png'),
                new10=str(ROOT/'robotwinprobe/out/libero_pi_fastwam_crossarch_10cm_20261028/textures/pi.png'),shrink13=str(root/'textures/pi.png'))
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
    lines=['# LIBERO π0.5 / FastWAM：13cm迁移、联训与10cm问题定位','',f'状态：`{s["stage"]}`。',
        '单任务put_the_bowl_on_the_plate，13×13cm、Q128、固定桌面位置；同一纹理同时经过主摄和腕部成像。',
        'π使用精确复现5/9的13cm最终纹理；FastWAM在相同13cm成像下补训500×8。双向迁移不使用目标模型梯度。',
        '联训另从灰色起步500×8，最大化两模型归一化L0的等权和；固定尺度取首个训练batch的Random响应。',
        'π保持原10×32首步速度SSE、等待10步/最多300步；FastWAM为32×7首步速度MSE、等待30步/最多400步。每5/10动作重规划，两模型均原生10步去噪。',
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
    for mode in ['fast','joint','legacy10']:
        f=root/'training'/mode/'progress.json'
        lines+=['',mode+'更新：'+(str(json.loads(f.read_text())['iteration'])+'/500' if f.exists() else '未开始')]
    lines+=['','10cm定位：同一旧闭环协议（等待10步、上限300步）下比较原新训纹理、13cm纹理缩小曝光、恢复旧配方的新训10cm纹理。','', '| 10cm条件 | 完成 | 配对攻击成功 |','|---|---:|---:|']
    diag={c:{r['init_index']:r for f in (root/'models/pi10/episodes'/c).glob('*.json') for r in [json.loads(f.read_text())] if 'success' in r} for c in ['clean','random','new10','shrink13','legacy10']}
    clean={i for i,r in diag['clean'].items() if r['success']}
    for c,rows in diag.items():
        paired=clean & rows.keys()
        lines.append(f'| {c} | {len(rows)}/10 | '+('—' if c=='clean' or not paired else f'{sum(not rows[i]["success"] for i in paired)}/{len(paired)}')+' |')
    if s.get('error'):lines+=['',s['error']]
    lines+=['',f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(lines)+'\n')
    dump(root/'outcomes.json',outcomes)


def status(root,stage,**kw):
    dump(root/'status.json',dict(stage=stage,pid=os.getpid(),host=socket.gethostname(),updated=time.time(),**kw));report(root)


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
        d={k:z[k].copy() for k in ['state8','clean_base','clean_wrist','train_mask','validation_mask']};prompt=str(z['prompt'])
    base=CameraProxy(CALIBRATION,device='cuda',fit=E26/'fit/base.json')
    wrist=CameraProxy(E26/'calibration/wrist.npz',device='cuda',fit=E26/'fit/wrist.json')
    with np.load(CAL10) as z:d10={k:z[k].copy() for k in ['state8','clean_base','clean_wrist','train_mask','validation_mask']}
    base10=wrist10=None
    if kind=='pi':
        base10=CameraProxy(SOURCE/'camera_proxy/calibration/base.npz',device='cuda',fit=SOURCE/'camera_proxy/fit/base.json')
        wrist10=CameraProxy(SOURCE/'camera_proxy/calibration/wrist.npz',device='cuda',fit=SOURCE/'camera_proxy/fit/wrist.json')
    cache={}
    def views(k,data=None):
        data=d if data is None else data
        return [torch.tensor(data[n][k].copy(),device='cuda').permute(2,0,1)[None].float()/255 for n in ['clean_base','clean_wrist']]
    def response(k,seed,images,data=None):
        data=d if data is None else data
        v,_=b.response(b.prepare(data['state8'][k],prompt,images),seed,full=False)
        return v.float() if kind=='pi' else v[...,:7].float()
    def objective(req):
        k=req['index'];seed=req['seed'];variant=req.get('variant','13cm')
        data,cb,cw=(d10,base10,wrist10) if variant=='legacy10' else (d,base,wrist)
        key=(variant,k,seed)
        if key not in cache:
            with torch.no_grad():cache[key]=response(k,seed,views(k,data),data).cpu()
        q=torch.tensor(req['texture'],device='cuda',requires_grad=req['command']=='gradient')
        delta=response(k,seed,[cb(q,k),cw(q,k)],data)-cache[key].cuda()
        value=delta.square().sum() if kind=='pi' else delta.square().mean()
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
        ib,iw=base(q,k),wrist(q,k)
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
        dump(r/'ready.json',dict(pid=os.getpid(),host=socket.gethostname(),ready=True));running=True
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
    scales=({'pi':1.} if mode=='legacy10' else {'fast':1.}) if mode!='joint' else json.loads((root/'scales.json').read_text())['scales']
    q=np.full((1,3,128,128),.5,np.float32);history=[];checkpoint=dest/'state.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as z:q=z['texture'].copy();history=json.loads(str(z['history']))
    for it in range(len(history),500):
        g=np.zeros_like(q);raw={kind:[] for kind in scales}
        for kind,scale in scales.items():
            for k,seed in zip(plan[it]['indices'],plan[it]['seeds']):
                row=call(root/'models'/kind,'gradient',index=k,seed=seed,texture=q,variant='legacy10' if mode=='legacy10' else '13cm')
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


def pipeline(root,gpu,fast_gpu):
    root=root.absolute();root.mkdir(parents=True,exist_ok=True)
    lock=open(root/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(root);services={}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    snapshot=root/'source_snapshots'/str(os.getpid());snapshot.mkdir(parents=True,exist_ok=True)
    names=['libero_crossarch_13cm.py','libero_pi_adapter.py','fastwam_libero_trial.py','fastwam_adapter.py']
    for name in names:shutil.copy2(ROOT/'robotwinprobe'/name,snapshot/name)
    dump(snapshot/'manifest.json',{n:sha(snapshot/n) for n in names})
    def preflight(physical,required,tag):
        script=os.environ.get('WAMATTACK_GPU_PREFLIGHT','/home/user1/.codex/skills/gpu-preflight/scripts/check_gpus.py')
        result=subprocess.run([sys.executable,script,'--hosts',socket.gethostname(),'--json','--min-free-mib',str(required)],capture_output=True,text=True)
        info=json.loads(result.stdout)
        node=next(n for n in info['nodes'] if n['host']==socket.gethostname())
        assert node['status']=='ok',node
        card=next(g for g in node['gpus'] if g['index']==physical)
        own={p.pid for p in services.values() if p.poll() is None}
        reuse=bool(own) and set(card['pids']).issubset(own) and card['free_mib']>=required
        allowed=card['eligible'] or reuse
        dump(root/'audit'/('preflight_'+tag+'.json'),dict(checked_at=info['checked_at'],host=node['host'],gpu=card,owned_backend_pids=sorted(own),same_job_card_reuse=reuse,allowed=allowed))
        assert allowed,('GPU is occupied by another task or lacks memory',node['host'],physical,card)
        return card['uuid']
    def start(kind):
        if kind in services:return
        status(root,'start_'+kind);r=root/'models'/kind;ready=r/'ready.json'
        if ready.exists():ready.unlink()
        physical=gpu if kind=='pi' else fast_gpu
        required=14500 if kind=='pi' else 17000
        uuid=preflight(physical,required,'start_'+kind)
        if kind=='pi':env['CUDA_VISIBLE_DEVICES']=uuid
        service_env=dict(env,CUDA_VISIBLE_DEVICES=uuid)
        with (root/'logs'/(kind+'_backend.log')).open('a') as log:
            services[kind]=subprocess.Popen([PI_PY if kind=='pi' else FAST_PY,'-u',str(Path(__file__).resolve()),'server','--root',str(root),'--kind',kind],env=service_env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
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
                preflight(gpu,512,f'render_{kind}_{condition}_{index}')
                status(root,f'evaluate_{kind}_{condition}_{index}')
                with (root/'logs'/f'{kind}_{condition}_{index}.log').open('a') as log:
                    subprocess.run([SIM,'-u',str(ROOT/'robotwinprobe/fastwam_libero_trial.py'),'rollout','--root',str(root/'models'/kind),'--condition',condition,'--index',str(index)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if condition=='clean':
                rows=[json.loads(f.read_text()) for f in (root/'models'/kind/'episodes/clean').glob('*.json')]
                assert sum(r.get('success',False) for r in rows)>=6,kind+' clean gate'
    try:
        status(root,'checkpoint_identity')
        assert sha(root/'models/fast/checkpoints/libero_uncond_2cam224.pt')=='1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579'
        start('pi');evaluate('pi',['clean','random','pi'])
        expected=json.loads((RESTORED/'retrained/episodes.json').read_text())['per_episode']
        for condition,old in [('clean','clean'),('pi','optimized')]:
            actual={i:json.loads((root/'models/pi/episodes'/condition/f'init{i}.json').read_text())['success'] for i in range(40,50)}
            assert actual=={r['init_index']:r['success'] for r in expected[old]},'pi13 native positive-control mismatch'
        dump(root/'audit/pi_positive.json',dict(passed=True,ASR='5/9',source=str(RESTORED)))
        with np.load(E26/'train_both/checkpoint_0500.npz') as z:expected_initial=float(z['history'][0])
        first=json.loads((root/'training_plan.json').read_text())[0];gray=np.full((1,3,128,128),.5,np.float32)
        values=[call(root/'models/pi','loss',index=k,seed=seed,texture=gray)['loss'] for k,seed in zip(first['indices'],first['seeds'])]
        actual=float(np.mean(values));relative=abs(actual-expected_initial)/max(abs(expected_initial),1e-30)
        dump(root/'audit/pi_initial_objective.json',dict(expected=expected_initial,actual=actual,relative_error=relative,passed=relative<1e-5))
        assert relative<1e-5,('pi objective differs from restored E26',expected_initial,actual)

        call(root/'models/pi','trim')
        start('fast');evaluate('fast',['clean','random','pi'])
        status(root,'train_fast13');train(root,'fast');call(root/'models/fast','trim')
        evaluate('fast',['fast']);evaluate('pi',['fast'])
        if not (root/'scales.json').exists():
            status(root,'fixed_training_random_scales');first=json.loads((root/'training_plan.json').read_text())[0]
            q=np.asarray(Image.open(root/'textures/random.png').convert('RGB'),dtype=np.float32).transpose(2,0,1)[None]/255
            rows={kind:[call(root/'models'/kind,'loss',index=k,seed=seed,texture=q)['loss'] for k,seed in zip(first['indices'],first['seeds'])] for kind in ['pi','fast']}
            scales={k:float(np.mean(v)) for k,v in rows.items()};assert all(np.isfinite(v) and v>0 for v in scales.values())
            dump(root/'scales.json',dict(scales=scales,rows=rows,source='first E26 training batch Random; no heldout or ASR normalization'))
        status(root,'train_joint13');train(root,'joint')
        for kind in ['pi','fast']:call(root/'models'/kind,'trim');evaluate(kind,['joint'])
        status(root,'train_legacy10_diagnosis');train(root,'legacy10');call(root/'models/pi','trim')
        evaluate('pi10',['clean','random','new10','shrink13','legacy10'])
        status(root,'closed_loop_and_diagnosis_complete')
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
    ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--kind',choices=['pi','fast']);ap.add_argument('--gpu',type=int,default=4);ap.add_argument('--fast-gpu',type=int,default=2)
    a=ap.parse_args()
    if a.command=='pipeline':pipeline(a.root,a.gpu,a.fast_gpu)
    elif a.command=='server':server(a.root,a.kind)
    else:globals()[a.command](a.root)
