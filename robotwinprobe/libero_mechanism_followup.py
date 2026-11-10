"""One small gradient/propagation diagnosis, then fixed color/structure ablations.
Two independent policy processes may run on separate initially idle GPUs.
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
from crossarch_common import AUTH,dump,sha
from gradient_fusion import common_ascent
ROOT=Path(os.environ.get('WAMATTACK_PROJECT_ROOT','/home/user1/workspace/chence/WAMattack'))
DEFAULT=ROOT/'robotwinprobe/out/libero13_gradient_diagnosis_20261028'
SOURCE=ROOT/'robotwinprobe/out/libero_pi_fastwam_crossarch_13cm_20261028'
COLOR=ROOT/'robotwinprobe/out/libero13_color_structure_20261028'
E26=ROOT/'pi05probe/out/dual_view'
ENVROOT=Path('/shared/user65/miniconda3/envs')
PY={'pi':str(ENVROOT/'openpi-server/bin/python'),'fast':str(ENVROOT/'fastwam/bin/python')}
SIM=str(ENVROOT/'openpi-libero/bin/python')
PORT={'pi':18712,'fast':18713}
REPORT=ROOT/'result/49_libero_mechanism_followup.md'
CONDITIONS=['clean','random','original','gray','shuffle','gray_shuffle','rot90']
SERIAL_RPC=False


def prepare(root):
    p=json.loads((root/'protocol.json').read_text())
    (root/'logs').mkdir(exist_ok=True)
    controls=json.loads((COLOR/'texture_controls.json').read_text())
    def rebase(value):
        return value.replace('/home/user1/workspace/chence/WAMattack',str(ROOT),1)
    for value in controls['variants'].values():assert sha(rebase(value['path']))==value['sha256']
    for kind in ['pi','fast']:
        for target in [root,COLOR]:
            r=target/'models'/kind;r.mkdir(parents=True,exist_ok=True)
            cfg=copy.deepcopy(json.loads((SOURCE/'models'/kind/'protocol.json').read_text()))
            cfg.update(experiment='libero13_mechanism_then_color_structure',control_port=PORT[kind],parent_root=str(target),calibration=str(E26/'calibration/base.npz'))
            cfg['checkpoint']=rebase(cfg['checkpoint'])
            paths={name:rebase(row['path']) for name,row in controls['variants'].items()}
            paths['random']=str(SOURCE/'textures/random.png')
            cfg['evaluation'].update(conditions=CONDITIONS,texture_paths=paths,baseline_reuse=False)
            dest=r/'protocol.json'
            if dest.exists():assert json.loads(dest.read_text())==cfg
            else:dump(dest,cfg)
        r=root/'models'/kind
        if kind=='fast':
            for name in ['prompts.npz','checkpoints/libero_uncond_2cam224.pt','checkpoints/libero_uncond_2cam224_dataset_stats.json']:
                dest=r/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():os.link(SOURCE/'models/fast'/name,dest)
    config_dir=root/'libero_config';config_dir.mkdir(exist_ok=True)
    original=(ROOT/'pi05probe/libero_config/config.yaml').read_text()
    (config_dir/'config.yaml').write_text(original.replace('/home/user1/workspace/chence/WAMattack',str(ROOT)))
    return p


def report(root):
    state=json.loads((root/'status.json').read_text())
    lines=['# LIBERO：loss传递／梯度融合小诊断与后续结构消融','',f'状态：`{state["stage"]}`；节点：`{state.get("host","unknown")}`。',
        '固定4训练+4验证状态、共同新noise、最终联合纹理；先诊断，再做颜色/结构对照，无新训练。',
        '诊断使用成像代理，不能当作新增真实render ASR。完整10步后，π前5、FastWAM前10动作的连续7维输出差作为执行前缀代理。FastWAM夹爪硬阈值另行统计，不伪造其梯度。']
    f=root/'fusion_summary.json'
    if f.exists():
        a=json.loads(f.read_text());lines+=['','最大最小一阶增益：'+str(a['common_ascent']['minimum_gain'])+'；存在共同正方向：'+str(a['common_ascent']['common_positive'])+'。']
    lines+=['','| 目标 | 条件 | 完成 | 配对失败/clean成功 |','|---|---|---:|---:|']
    for kind in ['pi','fast']:
        rr={c:{v['init_index']:v for f in (COLOR/'models'/kind/'episodes'/c).glob('*.json') for v in [json.loads(f.read_text())] if 'success' in v} for c in CONDITIONS}
        eligible={i for i,v in rr['clean'].items() if v['success']}
        for c in CONDITIONS:
            pair=eligible & rr[c].keys();score='—' if c=='clean' or not pair else f'{sum(not rr[c][i]["success"] for i in pair)}/{len(pair)}'
            lines.append(f'| {kind} | {c} | {len(rr[c])}/10 | {score} |')
    if state.get('error'):lines+=['',state['error']]
    lines+=['','块打乱保留RGB直方图和局部块，改变多种空间统计，不能唯一归因旋涡；单shuffle seed和单任务10初态均为探索性结果。']
    REPORT.write_text('\n'.join(lines)+'\n')


def status(root,stage,**kw):
    dump(root/'status.json',dict(stage=stage,pid=os.getpid(),host=socket.gethostname(),updated=time.time(),**kw));report(root)


def server(root,kind):
    import torch
    torch.set_num_threads(4)
    torch.cuda.init()  # Register the real job context before slow CPU checkpoint loading.
    if kind=='pi':from libero_pi_adapter import Backend
    else:from fastwam_adapter import Backend
    sys.path.insert(0,str(ROOT/'pi05probe'))
    from dual_view_proxy import CameraProxy
    b=Backend(root/'models'/kind)
    with np.load(E26/'calibration/base.npz') as z:
        d={n:z[n].copy() for n in ['state8','clean_base','clean_wrist']};prompt=str(z['prompt'])
    base=CameraProxy(E26/'calibration/base.npz',device='cuda',fit=E26/'fit/base.json')
    wrist=CameraProxy(E26/'calibration/wrist.npz',device='cuda',fit=E26/'fit/wrist.json')
    prefix=5 if kind=='pi' else 10
    if kind=='pi':
        from openpi.transforms import Unnormalize
        transforms=b.policy._output_transform.transforms
        assert [type(t).__name__ for t in transforms]==['Unnormalize','LiberoOutputs'],[type(t).__name__ for t in transforms]
        unnorm=next(t for t in transforms if isinstance(t,Unnormalize));stats=unnorm.norm_stats['actions']
        def continuous(x):
            a=x[0,:,:7].float().cpu()
            if unnorm.use_quantiles:
                lo=torch.as_tensor(np.array(stats.q01[:7],copy=True));hi=torch.as_tensor(np.array(stats.q99[:7],copy=True))
                a=(a+1.)/2.*(hi-lo+1e-6)+lo
            else:
                a=a*(torch.as_tensor(np.array(stats.std[:7],copy=True))+1e-6)+torch.as_tensor(np.array(stats.mean[:7],copy=True))
            return a.to('cuda')
        def native_continuous(x):
            return np.asarray(b.policy._output_transform({'actions':x[0].float().cpu().numpy(),'state':np.zeros(32,np.float32)})['actions'])
    else:
        def continuous(x):
            a=b.proc.normalizer.normalizers['action']['default'].backward(x[0].float().cpu())
            return torch.cat([a[:,:6],-(2*a[:,6:7]-1)],-1).cuda()
        def native_continuous(x):
            a=b.denorm(b.proc,x[0]).copy();a[:,-1]=-(2*a[:,-1]-1);return a
    def views(k):return [torch.tensor(d[n][k].copy(),device='cuda').permute(2,0,1)[None].float()/255 for n in ['clean_base','clean_wrist']]
    cache={}
    def teacher(k,seed):
        key=(k,seed)
        if key not in cache:
            with torch.no_grad():
                v,x=b.response(b.prepare(d['state8'][k],prompt,views(k)),seed,full=True)
                cache[key]=(v.float().cpu(),continuous(x).cpu())
        return [x.cuda() for x in cache[key]]
    def measure(q_np,k,seed,gradient=False,trace=False):
        refv,refa=teacher(k,seed)
        q=torch.tensor(q_np,device='cuda',requires_grad=gradient)
        ib,iw=base(q,k),wrist(q,k)
        v,x=b.response(b.prepare(d['state8'][k],prompt,[ib,iw]),seed,full=True)
        a=continuous(x);dv=v.float()-refv
        l0=dv.square().sum() if kind=='pi' else dv[...,:7].square().mean()
        delta=a[:prefix]-refa[:prefix];le=delta.square().mean()
        out=dict(index=int(k),L0=float(l0.detach()),execution_loss=float(le.detach()),arm_loss=float(delta[:,:6].square().mean().detach()),gripper_surrogate_loss=float(delta[:,6].square().mean().detach()))
        if kind=='fast':out['gripper_binary_change_fraction']=float((a[:prefix,6].sign()!=refa[:prefix,6].sign()).float().mean().detach())
        if gradient:
            g0,=torch.autograd.grad(l0,q,retain_graph=True)
            if trace:
                cb,cw=torch.autograd.grad(le,(ib,iw),retain_graph=True)
                gb,=torch.autograd.grad(ib,q,grad_outputs=cb,retain_graph=True)
                gw,=torch.autograd.grad(iw,q,grad_outputs=cw,retain_graph=True)
                gcamera,=torch.autograd.grad((ib,iw),q,grad_outputs=(cb,cw),retain_graph=True)
            ge,=torch.autograd.grad(le,q)
            assert torch.isfinite(g0).all() and torch.isfinite(ge).all() and float(g0.norm())>0 and float(ge.norm())>0
            assert all(p.grad is None for p in b.model.parameters())
            out.update(g_l0=g0.cpu().numpy(),g_exec=ge.cpu().numpy())
            if trace:
                error=float((gb+gw-gcamera).norm()/gcamera.norm())
                assert error<1e-4 and float(gb.norm())>0 and float(gw.norm())>0
                out['camera_trace']=dict(base_norm=float(gb.norm()),wrist_norm=float(gw.norm()),relative_sum_error=error,full10_differentiable=True,cross_backward_relative_difference=float((ge-gcamera).norm()/ge.norm()))
        return out
    def audit():
        p=json.loads((root/'protocol.json').read_text());rows=[]
        with torch.no_grad():
            for k in [p['training_indices'][0],p['heldout_indices'][0]]:
                prepared=b.prepare(d['state8'][k],prompt,views(k));_,a=b.response(prepared,p['noise_seed'],full=True)
                if kind=='pi':native=b.native_full(prepared,p['noise_seed'])
                else:
                    image,proprio,(context,mask)=prepared
                    native=b.model.infer_action(prompt=None,input_image=image,proprio=proprio,context=context,context_mask=mask,action_horizon=32,num_inference_steps=10,seed=p['noise_seed'],rand_device='cpu')['action'][None].float().cpu()
                error=float((a.float().cpu()-native).abs().max());assert error==0,('native full10 mismatch',error)
                ref=native_continuous(a);mapped=continuous(a).cpu().numpy()
                transform_error=float(np.max(np.abs(mapped-ref)));assert np.allclose(mapped,ref,rtol=1e-6,atol=1e-8),transform_error
                rows.append(dict(index=k,native_normalized_max_abs=error,action_transform_max_abs=transform_error))
        dump(root/'models'/kind/'native_audit.json',dict(passed=True,rows=rows));return {'passed':True}
    def batch(req,gradient):
        rows=[];g0=[];ge=[]
        for i,k in enumerate(req['indices']):
            with torch.set_grad_enabled(gradient):row=measure(req['texture'],k,req['seed'],gradient,trace=gradient and i==0)
            if gradient:g0.append(row.pop('g_l0'));ge.append(row.pop('g_exec'))
            rows.append(row)
            dump(root/'models'/kind/'progress.json',dict(stage=req['tag'],completed=i+1,total=len(req['indices'])))
        if gradient:np.savez_compressed(root/'models'/kind/'gradients.npz',g_l0=g0,g_exec=ge,indices=req['indices'])
        dump(root/'models'/kind/(req['tag']+'.json'),rows)
        torch.cuda.empty_cache()
        return dict(rows=rows)
    with Listener(('127.0.0.1',PORT[kind]),authkey=AUTH) as listener:
        dump(root/'models'/kind/'ready.json',dict(pid=os.getpid(),host=socket.gethostname(),ready=True));running=True
        while running:
            with listener.accept() as c:
                try:
                    req=c.recv();cmd=req['command']
                    if cmd=='shutdown':running=False;out={'stopped':True}
                    elif cmd=='audit':out=audit()
                    elif cmd in ['probe_batch','measure_batch']:out=batch(req,cmd=='probe_batch')
                    elif cmd=='infer':out=b.infer(req)
                    elif cmd=='trim':torch.cuda.empty_cache();out={'trimmed':True}
                    else:raise ValueError(cmd)
                    c.send(out)
                except Exception as e:
                    traceback.print_exc();c.send(dict(error=repr(e),traceback=traceback.format_exc()))


def parallel_rpc(command,**kw):
    if SERIAL_RPC:
        results={}
        for kind in ['pi','fast']:
            with Client(('127.0.0.1',PORT[kind]),authkey=AUTH) as c:
                c.send(dict(command=command,**kw));r=c.recv()
                if 'error' in r:raise RuntimeError(kind+': '+r['error']+'\n'+r.get('traceback',''))
                results[kind]=r
        return results
    conns={}
    try:
        for kind in ['pi','fast']:
            c=Client(('127.0.0.1',PORT[kind]),authkey=AUTH);c.send(dict(command=command,**kw));conns[kind]=c
        results={kind:c.recv() for kind,c in conns.items()}
        for kind,r in results.items():
            if 'error' in r:raise RuntimeError(kind+': '+r['error']+'\n'+r.get('traceback',''))
        return results
    finally:
        for c in conns.values():c.close()


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
    root=root.absolute();root.mkdir(exist_ok=True,parents=True)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=prepare(root);services={};uuids={};envs={};rollouts=[]
    (root/'audit').mkdir(exist_ok=True)
    snapshot=root/'source_snapshots'/str(os.getpid());snapshot.mkdir(parents=True)
    for name in ['libero_mechanism_followup.py','libero_pi_adapter.py','fastwam_adapter.py','gradient_fusion.py','fastwam_libero_trial.py']:
        shutil.copy2(ROOT/'robotwinprobe'/name,snapshot/name)
    dump(snapshot/'manifest.json',{f.name:sha(f) for f in snapshot.glob('*.py')})
    try:
        for kind,gpu in zip(['pi','fast'],gpus):
            status(root,'start_'+kind)
            same_card_own=[proc.pid for name,proc in services.items() if gpus[['pi','fast'].index(name)]==gpu]
            uuids[kind]=preflight(root,gpu,same_card_own,'start_'+kind,18000)
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
        status(root,'native_full10_audit');parallel_rpc('audit')
        with np.load(ROOT/p['source']) as z:q=z[p['texture_key']].copy()
        indices=p['training_indices']+p['heldout_indices'];ntrain=len(p['training_indices'])
        status(root,'parallel_gradient_probe')
        before=parallel_rpc('probe_batch',texture=q,indices=indices,seed=p['noise_seed'],tag='before')
        grads={}
        for kind in services:
            with np.load(root/'models'/kind/'gradients.npz') as z:grads[kind]={key:z[key][:ntrain].mean(0) for key in ['g_l0','g_exec']}
        with (SOURCE/'scales.json').open() as f:scales=json.load(f)['scales']
        original=.5*grads['pi']['g_l0']/scales['pi']+.5*grads['fast']['g_l0']/scales['fast']
        old_delta=np.clip(q+.01*np.sign(original),0,1)-q
        fusion=common_ascent(q,np.stack([grads[k]['g_exec'] for k in ['pi','fast']]))
        def cosine(a,b):
            den=np.linalg.norm(a)*np.linalg.norm(b)
            return float(np.sum(a*b)/den) if den else None
        info=dict(common_ascent={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fusion.items() if k!='delta'},
            cosine_between_models=dict(first_flow=cosine(grads['pi']['g_l0'],grads['fast']['g_l0']),execution=cosine(grads['pi']['g_exec'],grads['fast']['g_exec'])),
            per_model={k:dict(first_flow_norm=float(np.linalg.norm(g['g_l0'])),execution_norm=float(np.linalg.norm(g['g_exec'])),first_flow_vs_execution_cosine=cosine(g['g_l0'],g['g_exec']),old_update_predicted_execution_gain=float(np.sum(g['g_exec']*old_delta)),new_update_predicted_execution_gain=float(np.sum(g['g_exec']*fusion['delta']))) for k,g in grads.items()})
        dump(root/'fusion_summary.json',info)
        for tag,delta in [('old_l0_fusion_step',old_delta),('common_execution_step',fusion['delta'])]:
            status(root,tag)
            parallel_rpc('measure_batch',texture=(q+delta).astype(np.float32),indices=indices,seed=p['noise_seed'],tag=tag)
        parallel_rpc('trim');status(root,'small_diagnosis_complete')
        # The latest user asks to resume color/structure after the small diagnosis.
        dump(COLOR/'status.json',dict(stage='running_after_mechanism_diagnosis',host=socket.gethostname(),parent_root=str(root),pid=os.getpid()))
        def paired_render(condition,index):
            jobs=[]
            for i,kind in enumerate(['pi','fast']):
                path=COLOR/'models'/kind/'episodes'/condition/f'init{index}.json'
                if path.exists():assert 'success' in json.loads(path.read_text());continue
                own=[proc.pid for name,proc in services.items() if gpus[['pi','fast'].index(name)]==gpus[i]]
                preflight(root,gpus[i],own,f'render_{kind}_{condition}_{index}')
                log=(root/'logs'/f'{kind}_{condition}_{index}.log').open('a')
                proc=subprocess.Popen([SIM,'-u',str(ROOT/'robotwinprobe/fastwam_libero_trial.py'),'rollout','--root',str(COLOR/'models'/kind),'--condition',condition,'--index',str(index)],cwd=ROOT,env=envs[kind],stdout=log,stderr=subprocess.STDOUT)
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
                status(root,f'parallel_color_{condition}_{index}');paired_render(condition,index)
            if condition=='original':
                for kind in ['pi','fast']:
                    for new,old in [('clean','clean'),('random','random'),('original','joint')]:
                        actual={i:json.loads((COLOR/'models'/kind/'episodes'/new/f'init{i}.json').read_text())['success'] for i in range(40,50)}
                        expected={i:json.loads((SOURCE/'models'/kind/'episodes'/old/f'init{i}.json').read_text())['success'] for i in range(40,50)}
                        assert actual==expected,(kind,new,'baseline behavior changed before ablations')
        status(root,'complete')
        dump(COLOR/'status.json',dict(stage='complete',host=socket.gethostname(),parent_root=str(root),new_episodes=140))
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
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['pipeline','server']);ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--kind',choices=['pi','fast']);ap.add_argument('--gpus',type=int,nargs=2,default=[1,3]);a=ap.parse_args()
    if a.command=='pipeline':pipeline(a.root,a.gpus)
    else:server(a.root,a.kind)
