"""LIBERO 13cm π0.5/FastWAM joint texture with worst-case (max-min) step allocation.

Same loss, batches, noise, sign-PGD lr and 500x8 budget as the equal-weight joint
texture in libero_pi_fastwam_crossarch_13cm_20261028; only the fusion rule changes.
Each step both models give their batch L0 and gradient; attainment r_m is the batch
L0 divided by the single-model specialist's L0 on the same batch at the same
iteration; the step follows sign(grad) of argmin_m r_m.  No weight sweep, no EOT.
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
from multiprocessing.connection import Client,Listener

import numpy as np
from PIL import Image
from crossarch_common import AUTH,dump,sha
ROOT=Path(os.environ.get('WAMATTACK_PROJECT_ROOT','/home/user1/workspace/chence/WAMattack'))
OLD='/home/user1/workspace/chence/WAMattack'
DEFAULT=ROOT/'robotwinprobe/out/libero13_joint_maxmin_20261029'
SOURCE=ROOT/'robotwinprobe/out/libero_pi_fastwam_crossarch_13cm_20261028'
E26=ROOT/'pi05probe/out/dual_view'
CALIBRATION=E26/'calibration/base.npz'
PI_SPECIALIST=ROOT/'pi05probe/out/pi13_restore_20261028/train/checkpoint_0500.npz'
ENVROOT=Path('/shared/user65/miniconda3/envs')
PY={'pi':str(ENVROOT/'openpi-server/bin/python'),'fast':str(ENVROOT/'fastwam/bin/python')}
SIM=str(ENVROOT/'openpi-libero/bin/python')
PORT={'pi':18722,'fast':18723}
REPORT=ROOT/'result/50_libero_joint_maxmin.md'
CONDITIONS=['clean','maxmin']
KINDS=['pi','fast']
STEPS=500
LR=.01
HELDOUT_TEXTURES=['random','pi','fast','joint','maxmin']


def rebase(value):
    return json.loads(json.dumps(value).replace(OLD,str(ROOT)))


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    for name in ['logs','audit','textures','training']:(root/name).mkdir(exist_ok=True)
    p=dict(experiment='libero13_joint_maxmin',source=str(SOURCE),
        patch=dict(size_m=.13,resolution=128,position='existing fixed table site101',initialization=.5),
        training=dict(steps=STEPS,batch=8,lr=LR,update='sign-PGD',plan=str(SOURCE/'training_plan.json'),plan_sha256=sha(SOURCE/'training_plan.json')),
        loss=dict(pi='first10x32 velocity SSE',fast='first32x7 velocity MSE'),
        fusion='attainment r_m=L_m(batch,Q_t)/L_m^specialist(batch,t); step sign(grad of argmin r_m); tie->pi; both gradients logged',
        references=dict(pi=str(PI_SPECIALIST),pi_sha256=sha(PI_SPECIALIST),fast=str(SOURCE/'training/fast/state.npz'),fast_sha256=sha(SOURCE/'training/fast/state.npz')),
        evaluation='fresh clean (must equal source clean per init) then maxmin, init40-49, each model own clean denominator; same rollout protocol as source',
        heldout='first 8 calibration validation frames, noise 20260901, loss only for random/pi/fast/joint/maxmin textures',
        scope='single strategy, no weight grid, no EOT, frozen models')
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:dump(path,p)
    with np.load(PI_SPECIALIST) as z:ref_pi=[float(x) for x in z['history']]
    with np.load(SOURCE/'training/fast/state.npz') as z:ref_fast=[h['raw']['fast'] for h in json.loads(str(z['history']))]
    assert len(ref_pi)==len(ref_fast)==STEPS
    dump(root/'reference.json',dict(pi=ref_pi,fast=ref_fast))
    for kind in KINDS:
        r=root/'models'/kind;r.mkdir(parents=True,exist_ok=True)
        cfg=rebase(json.loads((SOURCE/'models'/kind/'protocol.json').read_text()))
        cfg.update(experiment=p['experiment'],control_port=PORT[kind],parent_root=str(root))
        cfg['evaluation'].update(conditions=CONDITIONS,baseline_reuse=False,texture_paths=dict(maxmin=str(root/'textures/maxmin.png')))
        dest=r/'protocol.json'
        if dest.exists():assert json.loads(dest.read_text())==cfg
        else:dump(dest,cfg)
        if kind=='fast':
            for name in ['prompts.npz','checkpoints/libero_uncond_2cam224.pt','checkpoints/libero_uncond_2cam224_dataset_stats.json']:
                dest=r/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():os.link(SOURCE/'models/fast'/name,dest)
    config_dir=root/'libero_config';config_dir.mkdir(exist_ok=True)
    (config_dir/'config.yaml').write_text((ROOT/'pi05probe/libero_config/config.yaml').read_text().replace(OLD,str(ROOT)))
    return p


def outcomes(base,kind):
    rows={}
    for c in (base/'models'/kind/'episodes').glob('*'):
        rows[c.name]={v['init_index']:v['success'] for f in c.glob('init*.json') for v in [json.loads(f.read_text())] if 'success' in v}
    return rows


def report(root):
    s=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'prepared'}
    lines=['# LIBERO 13cm：π0.5/FastWAM最大最小联训','',f'状态：`{s["stage"]}`；节点：`{s.get("host","unknown")}`。','',
        '与等权联训（result/46）相同：13cm、Q128、固定位置、同500×8批次与噪声、sign-PGD步长0.01、同L0损失，无EOT。',
        '只改融合规则：每步算两模型在本批的L0与各自专用纹理在同批同步数的L0之比（达成率），按达成率较低的模型的梯度符号更新。',
        '动机：等权联训末50步达成率FastWAM约101%、π约61%，与闭环π 2/9（专用5/9）、FastWAM 9/10（专用8/10）一致。','']
    f=root/'training/progress.json'
    if f.exists():
        h=json.loads(f.read_text())
        lines+=[f'训练：{h["iteration"]}/{STEPS}；末50步平均达成率 π {h["tail_ratio"]["pi"]:.2f}、FastWAM {h["tail_ratio"]["fast"]:.2f}；步数分配 π {h["chosen"]["pi"]}、FastWAM {h["chosen"]["fast"]}。','']
    f=root/'heldout.json'
    if f.exists():
        a=json.loads(f.read_text())
        lines+=['验证帧（8帧）平均L0，括号内为相对各自专用纹理的比例：','','| 纹理 | π | FastWAM |','|---|---:|---:|']
        for t in HELDOUT_TEXTURES:
            lines.append(f'| {t} | '+' | '.join(f'{a[k][t]:.4g}（{a[k][t]/a[k][k]:.2f}）' for k in KINDS)+' |')
        lines.append('')
    lines+=['| 目标 | 纹理 | 完成 | 配对失败/clean成功 |','|---|---|---:|---:|']
    for kind in KINDS:
        new=outcomes(root,kind);old=outcomes(SOURCE,kind)
        clean={i for i,v in new.get('clean',{}).items() if v}
        for label,rows in [('clean',new.get('clean',{})),('专用（46）',old.get(kind,{})),('等权联训（46）',old.get('joint',{})),('最大最小（本次）',new.get('maxmin',{}))]:
            pair=clean & rows.keys()
            score='—' if label=='clean' or not pair else f'{sum(not rows[i] for i in pair)}/{len(pair)}'
            lines.append(f'| {kind} | {label} | {len(rows)}/10 | {score} |')
    if s.get('error'):lines+=['',s['error']]
    lines+=['','单任务10初态，探索性；与46号同一clean分母逐初态核对。',f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(lines)+'\n')


def status(root,stage,**kw):
    dump(root/'status.json',dict(stage=stage,pid=os.getpid(),host=socket.gethostname(),updated=time.time(),**kw));report(root)


def server(root,kind):
    import torch
    torch.set_num_threads(4)
    torch.cuda.init()
    if kind=='pi':from libero_pi_adapter import Backend
    else:from fastwam_adapter import Backend
    sys.path.insert(0,str(ROOT/'pi05probe'))
    from dual_view_proxy import CameraProxy
    r=root/'models'/kind;b=Backend(r)
    with np.load(CALIBRATION) as z:
        d={n:z[n].copy() for n in ['state8','clean_base','clean_wrist','train_mask','validation_mask']};prompt=str(z['prompt'])
    base=CameraProxy(CALIBRATION,device='cuda',fit=E26/'fit/base.json')
    wrist=CameraProxy(E26/'calibration/wrist.npz',device='cuda',fit=E26/'fit/wrist.json')
    cache={}
    def views(k):return [torch.tensor(d[n][k].copy(),device='cuda').permute(2,0,1)[None].float()/255 for n in ['clean_base','clean_wrist']]
    def response(k,seed,images):
        v,_=b.response(b.prepare(d['state8'][k],prompt,images),seed,full=False)
        return v.float() if kind=='pi' else v[...,:7].float()
    def objective(texture,k,seed,gradient):
        if (k,seed) not in cache:
            with torch.no_grad():cache[(k,seed)]=response(k,seed,views(k)).cpu()
        q=torch.tensor(texture,device='cuda',requires_grad=gradient)
        delta=response(k,seed,[base(q,k),wrist(q,k)])-cache[(k,seed)].cuda()
        value=delta.square().sum() if kind=='pi' else delta.square().mean()
        assert torch.isfinite(value)
        if not gradient:return float(value),None
        g,=torch.autograd.grad(value,q)
        assert torch.isfinite(g).all() and float(g.norm())>0
        assert all(p.grad is None for p in b.model.parameters())
        return float(value.detach()),g.cpu().numpy()
    def batch(req):
        losses=[];total=None
        for k,seed in zip(req['indices'],req['seeds']):
            with torch.set_grad_enabled(req['gradient']):value,g=objective(req['texture'],int(k),int(seed),req['gradient'])
            losses.append(value)
            if g is not None:total=g if total is None else total+g
        out=dict(losses=losses)
        if total is not None:out['gradient']=total/len(losses)
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
                        action_horizon=32,num_inference_steps=10,seed=20261201,rand_device='cpu')['action'][None].float().cpu()
                error=float((a.float().cpu()-native).abs().max());assert error==0,('native parity',kind,error)
                rows.append(dict(index=int(k),max_abs=error))
        dump(r/'native_audit.json',dict(passed=True,rows=rows));return {'passed':True}
    with Listener(('127.0.0.1',PORT[kind]),authkey=AUTH) as listener:
        dump(r/'ready.json',dict(pid=os.getpid(),host=socket.gethostname(),ready=True));running=True
        while running:
            with listener.accept() as c:
                try:
                    req=c.recv();cmd=req['command']
                    if cmd=='shutdown':running=False;out={'stopped':True}
                    elif cmd=='audit':out=audit()
                    elif cmd=='batch':out=batch(req)
                    elif cmd=='validation_indices':out=dict(indices=[int(k) for k in np.flatnonzero(d['validation_mask'])])
                    elif cmd=='infer':out=b.infer(req)
                    elif cmd=='trim':torch.cuda.empty_cache();out={'trimmed':True}
                    else:raise ValueError(cmd)
                    c.send(out)
                except Exception as e:
                    traceback.print_exc();c.send(dict(error=repr(e),traceback=traceback.format_exc()))


SERIAL_RPC=False


def parallel_rpc(command,**kw):
    if SERIAL_RPC:
        results={}
        for kind in KINDS:
            with Client(('127.0.0.1',PORT[kind]),authkey=AUTH) as c:
                c.send(dict(command=command,**kw));results[kind]=c.recv()
    else:
        conns={}
        try:
            for kind in KINDS:
                c=Client(('127.0.0.1',PORT[kind]),authkey=AUTH);c.send(dict(command=command,**kw));conns[kind]=c
            results={kind:c.recv() for kind,c in conns.items()}
        finally:
            for c in conns.values():c.close()
    for kind,r in results.items():
        if 'error' in r:raise RuntimeError(kind+': '+r['error']+'\n'+r.get('traceback',''))
    return results


def load_png(path):
    return np.asarray(Image.open(path).convert('RGB'),dtype=np.float32).transpose(2,0,1)[None]/255


def train(root):
    plan=json.loads((SOURCE/'training_plan.json').read_text());ref=json.loads((root/'reference.json').read_text())
    dest=root/'training';checkpoint=dest/'state.npz'
    q=np.full((1,3,128,128),.5,np.float32);history=[]
    if checkpoint.exists():
        with np.load(checkpoint) as z:q=z['texture'].copy();history=json.loads(str(z['history']))
    for it in range(len(history),STEPS):
        out=parallel_rpc('batch',texture=q,indices=plan[it]['indices'],seeds=plan[it]['seeds'],gradient=True)
        raw={k:float(np.mean(out[k]['losses'])) for k in KINDS}
        if it==0:
            for k in KINDS:
                rel=abs(raw[k]-ref[k][0])/ref[k][0]
                dump(root/'audit'/f'initial_objective_{k}.json',dict(actual=raw[k],reference=ref[k][0],relative_error=rel,passed=rel<1e-5))
                assert rel<1e-5,(k,'initial objective differs from specialist reference',raw[k],ref[k][0])
        ratio={k:raw[k]/ref[k][it] for k in KINDS}
        chosen='pi' if ratio['pi']<=ratio['fast'] else 'fast'
        g={k:out[k]['gradient'] for k in KINDS}
        cos=float(np.sum(g['pi']*g['fast'])/(np.linalg.norm(g['pi'])*np.linalg.norm(g['fast'])))
        agree=float(np.mean(np.sign(g['pi'])==np.sign(g['fast'])))
        q=np.clip(q+LR*np.sign(g[chosen]),0,1).astype(np.float32)
        history.append(dict(iteration=it+1,raw=raw,ratio=ratio,chosen=chosen,cosine=cos,sign_agreement=agree,updated=time.time()))
        temp=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(temp,texture=q,history=json.dumps(history));temp.replace(checkpoint)
        tail=history[-50:]
        dump(dest/'progress.json',dict(complete=it==STEPS-1,iteration=it+1,tail_ratio={k:float(np.mean([h['ratio'][k] for h in tail])) for k in KINDS},
            chosen={k:sum(h['chosen']==k for h in history) for k in KINDS},last=history[-1]))
        if it==0 or (it+1)%25==0:
            Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png');report(root)
        print('[maxmin]',it+1,raw,{k:round(v,3) for k,v in ratio.items()},chosen,flush=True)
    final=root/'textures/maxmin.png'
    if not final.exists():shutil.copy2(dest/f'texture_{STEPS:04d}.png',final)


def heldout(root):
    if (root/'heldout.json').exists():return
    idx=parallel_rpc('validation_indices')['pi']['indices'][:8]
    paths={t:SOURCE/'textures'/(t+'.png') for t in HELDOUT_TEXTURES if t!='maxmin'};paths['maxmin']=root/'textures/maxmin.png'
    out={k:{} for k in KINDS}
    for t,path in paths.items():
        res=parallel_rpc('batch',texture=load_png(path),indices=idx,seeds=[20260901]*len(idx),gradient=False)
        for k in KINDS:out[k][t]=float(np.mean(res[k]['losses']))
    out['indices']=idx;out['texture_sha256']={t:sha(p) for t,p in paths.items()}
    dump(root/'heldout.json',out)


def preflight(root,gpu,own,tag,required=1024):
    script=os.environ.get('WAMATTACK_GPU_PREFLIGHT','/shared/user65/.codex/skills/gpu-preflight/scripts/check_gpus.py')
    result=subprocess.run([sys.executable,script,'--hosts',socket.gethostname(),'--json'],capture_output=True,text=True)
    info=json.loads(result.stdout);node=info['nodes'][0];assert node['status']=='ok',node
    card=next(g for g in node['gpus'] if g['index']==gpu)
    reuse=bool(own) and set(card['pids']).issubset(set(own)) and card['free_mib']>=required
    allowed=card['eligible'] or reuse
    dump(root/'audit'/('preflight_'+tag+'.json'),dict(checked_at=info['checked_at'],host=node['host'],gpu=card,own_pids=list(own),same_job_reuse=reuse,allowed=allowed))
    assert allowed,('GPU not available',card)
    assert not(socket.gethostname()=='nnmc63' and gpu==0),'ECC excluded GPU'
    return card['uuid']


def pipeline(root,gpus):
    global SERIAL_RPC
    SERIAL_RPC=gpus[0]==gpus[1]
    root=root.absolute();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(root);services={};uuids={};envs={};rollouts=[]
    snapshot=root/'source_snapshots'/str(os.getpid());snapshot.mkdir(parents=True)
    for name in ['libero_joint_maxmin.py','libero_pi_adapter.py','fastwam_adapter.py','fastwam_libero_trial.py']:
        shutil.copy2(ROOT/'robotwinprobe'/name,snapshot/name)
    dump(snapshot/'manifest.json',{f.name:sha(f) for f in snapshot.glob('*.py')})
    def own_on(gpu):return [proc.pid for name,proc in services.items() if gpus[KINDS.index(name)]==gpu]
    try:
        for kind,gpu in zip(KINDS,gpus):
            status(root,'start_'+kind)
            uuids[kind]=preflight(root,gpu,own_on(gpu),'start_'+kind,18000)
            envs[kind]=dict(os.environ,WAMATTACK_PROJECT_ROOT=str(ROOT),WAMATTACK_LIBERO_CONFIG_PATH=str(root/'libero_config'),CUDA_VISIBLE_DEVICES=uuids[kind],MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
            ready=root/'models'/kind/'ready.json'
            if ready.exists():ready.unlink()
            with (root/'logs'/(kind+'_backend.log')).open('a') as log:
                services[kind]=subprocess.Popen([PY[kind],'-u',str(Path(__file__).absolute()),'server','--root',str(root),'--kind',kind],cwd=ROOT,env=envs[kind],stdout=log,stderr=subprocess.STDOUT)
        began=time.monotonic()
        while not all((root/'models'/kind/'ready.json').exists() for kind in services):
            if any(proc.poll() is not None for proc in services.values()) or time.monotonic()-began>600:raise RuntimeError('backend initialization failed; see logs')
            time.sleep(2)
        dump(root/'execution.json',dict(host=socket.gethostname(),pipeline_pid=os.getpid(),backends={k:dict(pid=v.pid,gpu=gpus[i],uuid=uuids[k]) for i,(k,v) in enumerate(services.items())}))
        status(root,'native_audit');parallel_rpc('audit')
        status(root,'train_maxmin');train(root);parallel_rpc('trim')
        status(root,'heldout_loss');heldout(root);parallel_rpc('trim')
        def paired_render(condition,index):
            jobs=[]
            for i,kind in enumerate(KINDS):
                path=root/'models'/kind/'episodes'/condition/f'init{index}.json'
                if path.exists():assert 'success' in json.loads(path.read_text());continue
                preflight(root,gpus[i],own_on(gpus[i]),f'render_{kind}_{condition}_{index}')
                log=(root/'logs'/f'{kind}_{condition}_{index}.log').open('a')
                # robosuite asserts the EGL id is a substring of CUDA_VISIBLE_DEVICES; the sim does no
                # CUDA compute, so give it the same physical card by index instead of UUID.
                sim_env=dict(envs[kind],CUDA_VISIBLE_DEVICES=str(gpus[i]))
                proc=subprocess.Popen([SIM,'-u',str(ROOT/'robotwinprobe/fastwam_libero_trial.py'),'rollout','--root',str(root/'models'/kind),'--condition',condition,'--index',str(index)],cwd=ROOT,env=sim_env,stdout=log,stderr=subprocess.STDOUT)
                rollouts.append(proc)
                if SERIAL_RPC:
                    rc=proc.wait();log.close();rollouts.remove(proc)
                    if rc:raise RuntimeError(f'{kind} {condition} init{index} rollout failed')
                else:jobs.append((kind,proc,log))
            for kind,proc,log in jobs:
                rc=proc.wait();log.close();rollouts.remove(proc)
                if rc:raise RuntimeError(f'{kind} {condition} init{index} rollout failed')
        for condition in CONDITIONS:
            for index in range(40,50):
                status(root,f'evaluate_{condition}_{index}');paired_render(condition,index)
            if condition=='clean':
                for kind in KINDS:
                    assert outcomes(root,kind)['clean']==outcomes(SOURCE,kind)['clean'],(kind,'clean baseline changed')
        status(root,'complete')
    except Exception as e:
        status(root,'runtime_error',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        for proc in rollouts:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for kind,proc in services.items():
            if proc.poll() is None:
                try:
                    with Client(('127.0.0.1',PORT[kind]),authkey=AUTH) as c:c.send({'command':'shutdown'});c.recv()
                except Exception:proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['pipeline','server','prepare','report']);ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--kind',choices=KINDS);ap.add_argument('--gpus',type=int,nargs=2,default=[0,1]);a=ap.parse_args()
    if a.command=='pipeline':pipeline(a.root,a.gpus)
    elif a.command=='server':server(a.root,a.kind)
    elif a.command=='prepare':prepare(a.root);status(a.root,'prepared')
    else:report(a.root)
