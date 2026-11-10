"""FastWAM LIBERO 10cm world-texture pilot, separate from digital DRIFT."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from PIL import Image

from crossarch_common import ROOT,AUTH,dump,sha
from wam_orthogonal_control import call

DEFAULT=ROOT/'robotwinprobe/out/fastwam_libero_10cm_20261027'
CALIBRATION=ROOT/'pi05probe/out/fastwam_10cm_calibration_20261027/calibration.npz'
PY='/home/user1/miniconda3/envs/fastwam/bin/python'
SIM='/home/user1/miniconda3/envs/openpi-libero/bin/python'
REPORT=ROOT/'result/41_fastwam_libero_10cm.md'


def prepare(root):
    with np.load(CALIBRATION) as z:prompt=str(z['prompt'])
    p=dict(experiment='fastwam_libero_physical_10cm',domain='libero',task='put_the_bowl_on_the_plate',
        suite='libero_goal',control_port=18612,prompt_tasks=[prompt],
        checkpoint='yuanty/fastwam@8eaceeb24c3cc92ff2a9c9a9d266a4941b836705/libero_uncond_2cam224.pt',
        calibration=str(CALIBRATION),calibration_sha256=sha(CALIBRATION),
        patch=dict(size_m=.10,resolution=128,position='existing fixed table site101',initialization=.5),
        training=dict(steps=500,batch=8,lr=.01,seed=20261027,objective='first-flow effective7 velocity MSE',
            update='projected sign-gradient ascent',source='155 saved training states; 32 separate validation states from existing pi0.5 trajectories',
            cameras='base image rendered through calibrated world-texture proxy; wrist input clean in training',
            eot='none; geometry fixed, no additional phase/color/LOD randomization'),
        evaluation=dict(indices=list(range(40,50)),conditions=['clean','random','optimized'],
            steps=400,wait=30,replan=10,horizon=32,flow_steps=10,
            cameras='both native camera renders, including physical patch if visible; no digital image paste',
            env_seed='10000 + init_index before reset',noise_seed='20261006 + init_index*1000 + query',
            gripper='native FastWAM LIBERO: denorm, 2*x-1, invert sign, binarize',
            baseline_reuse=False,clean_minimum=6),
        proxy_response='diagnostic only; real-render paired outcomes primary',
        drift_scope='not a DRIFT reproduction: DRIFT digitally overlays wrist images at 32/64px, not a 10cm table plane')
    root.mkdir(parents=True,exist_ok=True)
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:dump(path,p)
    return p


def status(root,stage,**extra):
    dump(root/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra));report(root)


def report(root):
    if not (root/'protocol.json').exists():return
    p=json.loads((root/'protocol.json').read_text())
    s=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'prepared'}
    rows={name:[json.loads(f.read_text()) for f in (root/'episodes'/name).glob('*.json')] for name in p['evaluation']['conditions']}
    clean={r['init_index']:r for r in rows['clean'] if 'success' in r}
    text=['# FastWAM LIBERO：10×10cm真实render试验','',f'状态：`{s["stage"]}`。',
        '公开LIBERO策略权重，冻结模型；put_the_bowl_on_the_plate/libero_goal，10cm/Q128世界桌面贴纸。',
        '训练首步速度差MSE，500×8、符号梯度上升0.01、灰色起步；固定位置，无EOT。',
        '训练仅base代理，wrist保持clean；闭环两路均使用真实MuJoCo渲染。',
        '已有状态池155训练/32验证；重新进行10cm成像标定，不复用13cm/26cm几何。',
        '评估40–49初态，clean/Random/新patch各10回合；32步动作、10步去噪、每10动作重规划、30步等待、400步上限。',
        '模拟器使用现有MuJoCo3.2.3环境；公开策略原发布数据版本不同，因此这是本地探索性试验，不宣称复现官方基准。',
        'DRIFT的32/64像素腕相机数字贴图与此物理桌面攻击分别解释。','',
        '| 条件 | 完成 | 任务成功 | 配对失败/clean成功 |','|---|---:|---:|---:|']
    for name,rr in rows.items():
        valid=[r for r in rr if 'success' in r];eligible=[r for r in valid if clean.get(r['init_index'],{}).get('success')]
        paired='—' if name=='clean' else (f'{sum(not r["success"] for r in eligible)}/{len(eligible)}' if valid else '未开始')
        text.append(f'| {name} | {len(valid)}/10 | {sum(r["success"] for r in valid) if valid else "—"} | {paired} |')
    progress=root/'patch/progress.json'
    text+=['','训练更新：'+(str(json.loads(progress.read_text())['iteration'])+'/500' if progress.exists() else '未开始')]
    l0=root/'l0_validation/summary.json'
    if l0.exists():
        a=json.loads(l0.read_text());text+=['','## 最终纹理同状态L0验证','',
            '32个留出状态（验证episode10/11）×3份新共同noise，每条件96条。仅评估固定500步最终纹理，未重训、未重跑闭环。',
            'L0为首步32×7有效速度差MSE；另存SSE，SSE=MSE×224。render_both使用原生两路实际渲染，最接近闭环输入；base-only路径用于隔离训练相机范围。',
            '| 条件 | 条数 | L0 MSE均值 | L0 MSE中位数 |','|---|---:|---:|---:|']
        for name,row in a['conditions'].items():
            text.append(f'| {name} | {row["n"]} | {row["L0_MSE"]["mean"]:.7g} | {row["L0_MSE"]["median"]:.7g} |')
        text+=['','L0描述固定状态的首步输出偏移，不等于任务失败率；不能用此离线均值归因某个闭环失败初态。']
    if s.get('error'):text+=['',str(s['error'])]
    text+=['',f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(text)+'\n')


def serve(root):
    import torch
    from multiprocessing.connection import Listener
    from fastwam_adapter import Backend
    sys.path.insert(0,str(ROOT/'pi05probe'))
    from render_world_proxy import WorldTextureProxy
    torch.set_num_threads(4);b=Backend(root)
    with np.load(CALIBRATION) as z:
        d={k:z[k].copy() for k in ['clean_img256','clean_wrist','state8','random256','train_mask','validation_mask']}
        prompt=str(z['prompt'])
    proxy=WorldTextureProxy(CALIBRATION,device='cuda',fit=root/'proxy_fit/fit.json');cache={}
    def views(k,texture=None,real=False):
        if texture is None:base=torch.tensor(d['clean_img256'][k],device='cuda').permute(2,0,1)[None].float()/255
        elif real:base=torch.tensor(d['random256'][k],device='cuda').permute(2,0,1)[None].float()/255
        else:base=proxy(texture,k,native=True)
        wrist=torch.tensor(d['clean_wrist'][k],device='cuda').permute(2,0,1)[None].float()/255
        return [base,wrist]
    def teacher(k,seed):
        key=(k,seed)
        if key not in cache:
            with torch.no_grad():v,_=b.response(b.prepare(d['state8'][k],prompt,views(k)),seed,full=False)
            cache[key]=v.detach().float().cpu()
        return cache[key].cuda()
    def gradient(req):
        k=req['index'];seed=req['seed'];q=torch.tensor(req['texture'],device='cuda',requires_grad=True)
        clean=teacher(k,seed)
        v,_=b.response(b.prepare(d['state8'][k],prompt,views(k,q)),seed,full=False)
        value=(v.float()-clean).square().mean();g,=torch.autograd.grad(value,q)
        assert torch.isfinite(g).all() and float(g.norm())>0 and all(p.grad is None for p in b.model.parameters())
        return dict(loss=float(value.detach()),gradient=g.cpu().numpy(),gradient_norm=float(g.norm()))
    def audit():
        rows=[]
        with torch.no_grad():
            for k in np.flatnonzero(d['validation_mask'])[:4]:
                prepared=b.prepare(d['state8'][k],prompt,views(int(k)))
                v,a=b.response(prepared,20261201,full=True)
                image,proprio,(context,mask)=prepared
                native=b.model.infer_action(prompt=None,input_image=image,proprio=proprio,context=context,
                    context_mask=mask,action_horizon=32,num_inference_steps=10,seed=20261201,rand_device='cpu')['action']
                rows.append(dict(index=int(k),native_max_abs=float((a[0].float().cpu()-native).abs().max())))
        assert all(r['native_max_abs']==0 for r in rows)
        result=dict(passed=True,rows=rows,scope='native full32x7 action parity, four heldout clean states')
        dump(root/'audit/native.json',result);return result
    p=json.loads((root/'protocol.json').read_text())
    with Listener(('127.0.0.1',p['control_port']),authkey=AUTH) as listener:
        dump(root/'ready.json',dict(ready=True,pid=os.getpid()))
        running=True
        while running:
            with listener.accept() as c:
                try:
                    req=c.recv();cmd=req['command']
                    if cmd=='shutdown':result={'stopped':True};running=False
                    elif cmd=='audit':result=audit()
                    elif cmd=='gradient':result=gradient(req)
                    elif cmd=='infer':result=b.infer(req)
                    else:raise ValueError(cmd)
                    c.send(result)
                except Exception as error:
                    traceback.print_exc();c.send(dict(error=repr(error),traceback=traceback.format_exc()))


def rollout(root,condition,index):
    sys.path.insert(0,str(ROOT/'pi05probe'))
    import drift_eval_digital as D
    import scene_patch as sp
    import torch,yaml
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    p=json.loads((root/'protocol.json').read_text());dest=root/'episodes'/condition;dest.mkdir(parents=True,exist_ok=True)
    output=dest/f'init{index}.json'
    if output.exists():
        prior=json.loads(output.read_text());assert 'success' in prior and prior['protocol_sha256']==sha(root/'protocol.json');return
    suite=benchmark.get_benchmark_dict()['libero_goal']()
    task=next(suite.get_task(i) for i in range(suite.n_tasks) if Path(suite.get_task(i).bddl_file).stem==p['task'])
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    inits=np.array(torch.load(str(Path(get_libero_path('init_states'))/'libero_goal'/(p['task']+'.pruned_init'))))
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    if condition!='clean':
        cfg=yaml.safe_load((ROOT/'pi05probe/config/scene.yaml').read_text());cfg['patch']['size_wh']=[p['patch']['size_m']]*2
        with np.load(p.get('calibration',str(CALIBRATION))) as z:world=z['cell_world'].copy()
        by_init=p['evaluation'].get('texture_paths_by_init',{}).get(condition,{})
        texture=root/by_init.get(str(index),p['evaluation'].get('texture_paths',{}).get(condition,
            'random.png' if condition=='random' else 'patch/texture_0500.png'))
        env.env.set_xml_processor(sp.make_xml_processor(cfg,world,str(texture)))
    began=time.monotonic()
    try:
        env.seed(10000+index);env.reset();obs=env.set_init_state(inits[index])
        for _ in range(p['evaluation']['wait']):obs,_,_,_=env.step(D.DUMMY)
        initial=env.get_sim_state().copy();body_pos=env.env.sim.model.body_pos.copy();body_quat=env.env.sim.model.body_quat.copy()
        initial_hash=hashlib.sha256(initial.tobytes()).hexdigest()
        if condition!='clean':
            source=root/'episodes/clean'/f'init{index}.json';clean=json.loads(source.read_text());assert initial_hash==clean['initial_state_sha256']
            with np.load(source.with_suffix('.npz')) as z:
                n=min(len(body_pos),len(z['body_pos']))
                assert np.array_equal(body_pos[:n],z['body_pos'][:n]) and np.array_equal(body_quat[:n],z['body_quat'][:n])
        queue=[];queries=0;success=False;states=[];actions=[];images=[];wrists=[]
        for step in range(p['evaluation']['steps']):
            if not queue:
                obs=env.regenerate_obs_from_state(env.get_sim_state().copy())
                base=np.ascontiguousarray(obs['agentview_image'][::-1,::-1]);wrist=np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1])
                state=np.concatenate([obs['robot0_eef_pos'],D.quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']]).astype(np.float32)
                seed=20261006+index*1000+queries
                result=call(root,'infer',state=state,prompt=p['prompt_tasks'][0],images={'base':base,'wrist':wrist},seed=seed)
                chunk=np.asarray(result['actions']);assert chunk.shape==(p['evaluation']['horizon'],7) and np.isfinite(chunk).all()
                if p.get('model_kind','fastwam')!='pi':
                    chunk[:,-1]=np.sign(-(2*chunk[:,-1]-1))
                states.append(state);actions.append(chunk);images.append(base);wrists.append(wrist)
                queue=list(chunk[:p['evaluation']['replan']]);queries+=1
            obs,_,done,_=env.step(queue.pop(0))
            if done:success=True;break
        np.savez_compressed(output.with_suffix('.npz'),states=states,actions=actions,base=images,wrist=wrists,
            initial_state=initial,body_pos=body_pos,body_quat=body_quat)
        dump(output,dict(init_index=index,condition=condition,success=success,steps=step+1,queries=queries,
            protocol_sha256=sha(root/'protocol.json'),initial_state_sha256=initial_hash,elapsed_s=time.monotonic()-began))
        print('[FastWAM LIBERO]',index,condition,success,step+1,flush=True)
    except Exception as error:
        dump(output,dict(init_index=index,condition=condition,error=repr(error),traceback=traceback.format_exc()));raise
    finally:env.close()


def train(root):
    p=json.loads((root/'protocol.json').read_text());cfg=p['training']
    with np.load(CALIBRATION) as z:pool=np.flatnonzero(z['train_mask'])
    rng=np.random.default_rng(cfg['seed'])
    draws=[dict(indices=rng.choice(pool,cfg['batch'],replace=False).tolist(),seeds=rng.integers(1,2**30,size=cfg['batch']).tolist()) for _ in range(cfg['steps'])]
    dump(root/'training_plan.json',draws);dest=root/'patch';dest.mkdir(exist_ok=True)
    q=np.full((1,3,128,128),.5,np.float32);history=[]
    checkpoint=dest/'training_state.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as z:q=z['texture'].copy();history=json.loads(str(z['history']))
    for iteration in range(len(history),cfg['steps']):
        values=[];g=np.zeros_like(q)
        for k,seed in zip(draws[iteration]['indices'],draws[iteration]['seeds']):
            r=call(root,'gradient',index=k,seed=seed,texture=q);g+=r['gradient']/cfg['batch'];values.append(r['loss'])
        assert np.isfinite(g).all() and np.linalg.norm(g)>0
        q=np.clip(q+cfg['lr']*np.sign(g),0,1);history.append(float(np.mean(values)))
        temp=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(temp,texture=q,history=json.dumps(history),iteration=iteration+1);temp.replace(checkpoint)
        if iteration==0 or (iteration+1)%25==0:Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(dest/f'texture_{iteration+1:04d}.png')
        dump(dest/'progress.json',dict(iteration=iteration+1,total=cfg['steps'],loss=history[-1],complete=iteration+1==cfg['steps']))
        if iteration==0 or (iteration+1)%25==0:report(root)
        print('[FastWAM LIBERO patch]',iteration+1,history[-1],flush=True)
    final=dest/'texture_0500.png'
    if not final.exists():Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(final)


def pipeline(root,gpu):
    root.mkdir(parents=True,exist_ok=True);lock=open(root/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=prepare(root);(root/'logs').mkdir(exist_ok=True);server=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
    def own(cmd):return [PY,'-u',str(Path(__file__).resolve()),cmd,'--root',str(root)]
    def job(name,argv):
        status(root,name)
        with open(root/'logs'/(name+'.log'),'a') as log:subprocess.run(argv,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        status(root,'checkpoint_identity')
        expected='1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579'
        actual=sha(root/'checkpoints/libero_uncond_2cam224.pt');assert actual==expected
        dump(root/'audit/checkpoint_identity.json',dict(passed=True,sha256=actual))
        with np.load(CALIBRATION) as z:Image.fromarray(z['random__texture']).save(root/'random.png')
        if not (root/'prompts.npz').exists():
            job('encode_prompts',own('prompts'))
        status(root,'start_backend');ready=root/'ready.json'
        if ready.exists():ready.unlink()
        with open(root/'logs/backend.log','a') as log:server=subprocess.Popen(own('serve'),env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        start=time.monotonic()
        while not ready.exists():
            if server.poll() is not None or time.monotonic()-start>600:raise RuntimeError('LIBERO backend failed to initialize')
            time.sleep(2)
        status(root,'native_audit');call(root,'audit')
        for index in p['evaluation']['indices']:job('clean_'+str(index),[SIM,'-u',str(Path(__file__).resolve()),'rollout','--root',str(root),'--condition','clean','--index',str(index)])
        rows=[json.loads(f.read_text()) for f in (root/'episodes/clean').glob('*.json')]
        if sum(r.get('success',False) for r in rows)<6:
            status(root,'stopped_clean_gate',error='FastWAM clean below 6/10; no attack training');return
        job('training',own('train'))
        for condition in ('random','optimized'):
            for index in p['evaluation']['indices']:job(condition+'_'+str(index),[SIM,'-u',str(Path(__file__).resolve()),'rollout','--root',str(root),'--condition',condition,'--index',str(index)])
        status(root,'complete')
    except Exception as error:
        status(root,'runtime_error',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if server is not None and server.poll() is None:
            try:call(root,'shutdown')
            except Exception:server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','pipeline','serve','train','rollout','report','prompts'])
    ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--gpu',type=int,default=7)
    ap.add_argument('--condition');ap.add_argument('--index',type=int)
    a=ap.parse_args()
    if a.command=='pipeline':pipeline(a.root,a.gpu)
    elif a.command=='rollout':rollout(a.root,a.condition,a.index)
    elif a.command=='prompts':
        from fastwam_adapter import prompts
        prompts(a.root)
    else:globals()[a.command](a.root)
