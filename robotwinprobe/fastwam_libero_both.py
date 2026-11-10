"""Same-state/noise/budget FastWAM trial adding wrist-image texture gradients."""
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

from crossarch_common import ROOT,AUTH,dump,sha
from wam_orthogonal_control import call
from fastwam_libero_trial import DEFAULT as SOURCE,CALIBRATION,PY,SIM

DEFAULT=ROOT/'robotwinprobe/out/fastwam_libero_both_10cm_20261027'
REPORT=ROOT/'result/42_fastwam_libero_both_10cm.md'


def link(a,b):
    b.parent.mkdir(parents=True,exist_ok=True)
    if not b.exists():os.link(a,b)


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(json.loads((SOURCE/'protocol.json').read_text()))
    p.update(experiment='fastwam_libero_physical_10cm_both',control_port=18622,baseline_root=str(SOURCE),
        authorization='User explicitly requires wrist input after base-only trial; same world texture must affect both views through actual projection',
        isolation='same base proxy, original155 training states, batch/noise plan, Q128, gray initialization,500x8,.01; add wrist physical imaging path only')
    p['training']['cameras']='one shared Q through unchanged base WorldTextureProxy and per-state calibrated wrist CameraProxy; both branches backpropagate to Q'
    p['evaluation'].update(conditions=['clean','random','base_only','both'],
        texture_paths=dict(random='random.png',base_only='baseline/base_only_0500.png',both='patch/texture_0500.png'),
        baseline_reuse=True,expected_new_episodes=10,
        reference='copy verified same FastWAM clean/Random/base-only episodes with source hashes; no other model baseline reuse')
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:dump(path,p)
    for name in ['prompts.npz','random.png']:
        link(SOURCE/name,root/name)
    for name in ['libero_uncond_2cam224.pt','libero_uncond_2cam224_dataset_stats.json']:
        link(SOURCE/'checkpoints'/name,root/'checkpoints'/name)
    link(SOURCE/'patch/texture_0500.png',root/'baseline/base_only_0500.png')
    # The base camera forward and gradient are intentionally unchanged.
    for name in ['fit.json','pixel_check.json']:
        link(SOURCE/'proxy_fit'/name,root/'proxy_fit'/name)
    references=[]
    for new,old in [('clean','clean'),('random','random'),('base_only','optimized')]:
        for index in p['evaluation']['indices']:
            source=SOURCE/'episodes'/old/f'init{index}.json';row=json.loads(source.read_text());assert 'success' in row
            row.update(condition=new,protocol_sha256=sha(path),reference_reused=True,source_record=str(source),
                source_record_sha256=sha(source),source_protocol_sha256=json.loads(source.read_text())['protocol_sha256'])
            target=root/'episodes'/new/source.name
            if target.exists():assert json.loads(target.read_text())==row
            else:dump(target,row)
            link(source.with_suffix('.npz'),target.with_suffix('.npz'))
            references.append(dict(target=str(target),source=str(source),sha256=sha(source)))
    dump(root/'audit/references.json',references)
    plan=root/'training_plan.json'
    if not plan.exists():shutil.copy2(SOURCE/'training_plan.json',plan)
    assert json.loads(plan.read_text())==json.loads((SOURCE/'training_plan.json').read_text())
    return p


def report(root):
    if not (root/'protocol.json').exists():return
    p=json.loads((root/'protocol.json').read_text())
    s=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else dict(stage='calibration')
    lines=['# FastWAM LIBERO：10cm同纹理双相机训练','',f'状态：`{s["stage"]}`。',
        '修正此前仅优化base、wrist保持clean的训练范围。保持单任务/位置/10cm/Q128/灰色初始化/首步速度MSE/500×8/.01不变。',
        '复用原155训练状态和逐次完全相同的batch/noise计划。base使用原WorldTextureProxy及原fit；新增逐帧腕相机CameraProxy，单一Q同时生成两路成像并接收两路梯度。',
        '腕相机位姿随状态变化，187状态逐帧重新标定，不复用13cm或首帧H。物体遮挡保留；仅在真实可见位置影响输入。',
        '原FastWAM clean/Random/base-only共30条记录透明复用，保留来源hash；新增both10条真实MuJoCo闭环。',
        '历史π0.5相机消融及13cm双相机5/9不是FastWAM结果，不用于改写本轮ASR。','',
        '| 条件 | 完成 | 任务成功 | 配对失败/clean成功 |','|---|---:|---:|---:|']
    clean={}
    for path in (root/'episodes/clean').glob('*.json'):
        r=json.loads(path.read_text())
        if 'success' in r:clean[r['init_index']]=r['success']
    for condition in p['evaluation']['conditions']:
        rr=[json.loads(f.read_text()) for f in (root/'episodes'/condition).glob('*.json')]
        rr=[r for r in rr if 'success' in r];eligible=[r for r in rr if clean.get(r['init_index'])]
        paired='—' if condition=='clean' else (f'{sum(not r["success"] for r in eligible)}/{len(eligible)}' if rr else '未开始')
        lines.append(f'| {condition} | {len(rr)}/10 | {sum(r["success"] for r in rr) if rr else "—"} | {paired} |')
    f=root/'patch/progress.json';lines+=['','both更新：'+(str(json.loads(f.read_text())['iteration'])+'/500' if f.exists() else '未开始')]
    fit=root/'camera_proxy/fit/checks.json'
    if fit.exists():
        a=json.loads(fit.read_text());lines+=['',f'腕相机留出Random/平滑图案mask RMSE均值：{a["wrist"]["mask_RMSE_mean"]:.6g}。']
    f=root/'audit/gradient.json'
    if f.exists():
        a=json.loads(f.read_text());lines+=['',f'共享Q梯度预检：{a["passed"]}；采样状态{a["index"]}，base/wrist梯度范数分别{a["base_gradient_norm"]:.6g}/{a["wrist_gradient_norm"]:.6g}。']
    f=root/'l0_validation/summary.json'
    if f.exists():
        a=json.loads(f.read_text());lines+=['','32个留出状态×3份新共同noise；实际两路render下的首步32×7速度差MSE：',
            '| 纹理 | L0均值 | L0中位数 |','|---|---:|---:|']
        for name,label in [('random_render_both','Random'),('optimized_render_both','新both纹理')]:
            row=a['conditions'][name]['L0_MSE'];lines.append(f'| {label} | {row["mean"]:.7g} | {row["median"]:.7g} |')
    if s.get('error'):lines+=['',str(s['error'])]
    lines+=['',f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(lines)+'\n')


def status(root,stage,**kw):
    dump(root/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**kw));report(root)


def server(root):
    import torch
    from multiprocessing.connection import Listener
    from fastwam_adapter import Backend
    sys.path.insert(0,str(ROOT/'pi05probe'))
    from render_world_proxy import WorldTextureProxy
    from dual_view_proxy import CameraProxy
    torch.set_num_threads(4);b=Backend(root)
    with np.load(CALIBRATION) as z:
        d={k:z[k].copy() for k in ['state8','clean_img256','clean_wrist','train_mask','validation_mask']};prompt=str(z['prompt'])
    with np.load(root/'camera_proxy/calibration/wrist.npz') as z:visible=z['mask'].sum((1,2))
    base=WorldTextureProxy(CALIBRATION,device='cuda',fit=root/'proxy_fit/fit.json')
    wrist=CameraProxy(root/'camera_proxy/calibration/wrist.npz',device='cuda',fit=root/'camera_proxy/fit/wrist.json')
    cache={}
    def clean_views(k):
        return [torch.tensor(d[name][k].copy(),device='cuda').permute(2,0,1)[None].float()/255 for name in ['clean_img256','clean_wrist']]
    def teacher(k,seed):
        key=(k,seed)
        if key not in cache:
            with torch.no_grad():v,_=b.response(b.prepare(d['state8'][k],prompt,clean_views(k)),seed,full=False)
            cache[key]=v.float().cpu()
        return cache[key].cuda()
    def objective(k,seed,qb,qw):
        reference=teacher(k,seed)
        v,_=b.response(b.prepare(d['state8'][k],prompt,[base(qb,k,native=True),wrist(qw,k,native=True)]),seed,full=False)
        return (v.float()-reference).square().mean()
    def gradient(req):
        k=req['index'];seed=req['seed'];q=torch.tensor(req['texture'],device='cuda',requires_grad=True)
        value=objective(k,seed,q,q);g,=torch.autograd.grad(value,q)
        assert torch.isfinite(value) and torch.isfinite(g).all() and float(g.norm())>0
        assert all(p.grad is None for p in b.model.parameters())
        return dict(loss=float(value.detach()),gradient=g.cpu().numpy(),gradient_norm=float(g.norm()))
    def audit():
        rows=[]
        with torch.no_grad():
            for k in np.flatnonzero(d['validation_mask'])[:4]:
                image,proprio,(context,mask)=prepared=b.prepare(d['state8'][k],prompt,clean_views(k))
                _,a=b.response(prepared,20261201,full=True)
                native=b.model.infer_action(prompt=None,input_image=image,proprio=proprio,context=context,context_mask=mask,
                    action_horizon=32,num_inference_steps=10,seed=20261201,rand_device='cpu')['action']
                error=float((a[0].float().cpu()-native).abs().max());assert error==0
                rows.append(dict(index=int(k),max_abs=error))
        dump(root/'audit/native.json',dict(passed=True,rows=rows))
        k=int(np.flatnonzero(d['train_mask'] & (visible>=100))[0]);seed=20261201
        q=torch.full((1,3,128,128),.5,device='cuda',requires_grad=True)
        ib,iw=base(q,k,native=True),wrist(q,k,native=True)
        reference=teacher(k,seed)
        v,_=b.response(b.prepare(d['state8'][k],prompt,[ib,iw]),seed,full=False)
        value=(v.float()-reference).square().mean()
        # Obtain model cotangents once. Re-running BF16/checkpointed model
        # backward for separate leaves is not an exact wiring identity test.
        cb,cw=torch.autograd.grad(value,(ib,iw),retain_graph=True)
        gb,=torch.autograd.grad(ib,q,grad_outputs=cb,retain_graph=True)
        gw,=torch.autograd.grad(iw,q,grad_outputs=cw,retain_graph=True)
        combined,=torch.autograd.grad((ib,iw),q,grad_outputs=(cb,cw))
        assert torch.isfinite(gb).all() and torch.isfinite(gw).all() and float(gb.norm())>0 and float(gw.norm())>0
        total=gb.cpu().numpy()+gw.cpu().numpy();relative=float(np.linalg.norm(total-combined.cpu().numpy())/max(np.linalg.norm(total),1e-30))
        assert relative<1e-4,relative
        actual=gradient(dict(index=k,seed=seed,texture=q.detach().cpu().numpy()))
        repeat_relative=float(np.linalg.norm(actual['gradient']-total)/max(np.linalg.norm(total),1e-30))
        result=dict(passed=True,index=k,visible224=int(visible[k]),base_gradient_norm=float(gb.norm()),wrist_gradient_norm=float(gw.norm()),
            shared_gradient_relative_error=relative,chain_rule_scope='one model cotangent evaluation, joint versus separate camera VJPs',
            cross_backward_relative_difference=repeat_relative,model_frozen=True)
        dump(root/'audit/gradient.json',result);return result
    p=json.loads((root/'protocol.json').read_text())
    with Listener(('127.0.0.1',p['control_port']),authkey=AUTH) as listener:
        dump(root/'ready.json',dict(ready=True,pid=os.getpid()));running=True
        while running:
            with listener.accept() as c:
                try:
                    r=c.recv();cmd=r['command']
                    if cmd=='shutdown':out={'stopped':True};running=False
                    elif cmd=='audit':out=audit()
                    elif cmd=='gradient':out=gradient(r)
                    elif cmd=='infer':out=b.infer(r)
                    else:raise ValueError(cmd)
                    c.send(out)
                except Exception as error:
                    traceback.print_exc();c.send(dict(error=repr(error),traceback=traceback.format_exc()))


def train(root):
    import fastwam_libero_trial as base
    base.report=report
    base.train(root)
    assert json.loads((root/'training_plan.json').read_text())==json.loads((SOURCE/'training_plan.json').read_text())


def pipeline(root,gpu):
    root.mkdir(parents=True,exist_ok=True);lock=open(root/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=prepare(root);(root/'logs').mkdir(exist_ok=True);service=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
    def own(cmd):return [PY,'-u',str(Path(__file__).resolve()),cmd,'--root',str(root)]
    def job(name,argv):
        status(root,name)
        with (root/'logs'/(name+'.log')).open('a') as log:subprocess.run(argv,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        if not (root/'camera_proxy/calibration/wrist.npz').exists():
            job('camera_capture',[SIM,'-u',str(ROOT/'pi05probe/dual_view_capture.py'),'--calibration',str(CALIBRATION),'--out',str(root/'camera_proxy/calibration')])
        if not (root/'camera_proxy/fit/checks.json').exists():
            job('camera_fit',[PY,'-u',str(ROOT/'pi05probe/dual_view_fit.py'),'--root',str(root/'camera_proxy'),'--device','cuda'])
        snapshot=root/'source_snapshots'/str(os.getpid());snapshot.mkdir(parents=True,exist_ok=True)
        names=['robotwinprobe/fastwam_libero_both.py','robotwinprobe/fastwam_libero_trial.py','robotwinprobe/fastwam_adapter.py',
               'pi05probe/dual_view_capture.py','pi05probe/dual_view_fit.py','pi05probe/dual_view_proxy.py','pi05probe/render_world_proxy.py']
        for name in names:shutil.copy2(ROOT/name,snapshot/Path(name).name)
        dump(snapshot/'manifest.json',{Path(n).name:sha(snapshot/Path(n).name) for n in names})
        expected=json.loads((SOURCE/'audit/checkpoint_identity.json').read_text())['sha256']
        assert sha(root/'checkpoints/libero_uncond_2cam224.pt')==expected
        status(root,'start_backend');ready=root/'ready.json'
        if ready.exists():ready.unlink()
        with (root/'logs/backend.log').open('a') as log:service=subprocess.Popen(own('server'),env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        start=time.monotonic()
        while not ready.exists():
            if service.poll() is not None or time.monotonic()-start>600:raise RuntimeError('Both-view backend failed')
            time.sleep(2)
        status(root,'gradient_native_audit');call(root,'audit')
        job('training',own('train'))
        for index in p['evaluation']['indices']:
            job('both_'+str(index),[SIM,'-u',str(ROOT/'robotwinprobe/fastwam_libero_trial.py'),'rollout','--root',str(root),'--condition','both','--index',str(index)])
        call(root,'shutdown');service.wait(timeout=30);service=None
        if not (root/'l0_validation/inputs.npz').exists():
            job('l0_capture',[SIM,'-u',str(ROOT/'robotwinprobe/fastwam_libero_l0.py'),'capture','--root',str(root)])
        if not (root/'l0_validation/summary.json').exists():
            job('l0_measure',[PY,'-u',str(ROOT/'robotwinprobe/fastwam_libero_l0.py'),'measure','--root',str(root)])
        status(root,'complete')
    except Exception as error:
        status(root,'runtime_error',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if service is not None and service.poll() is None:
            try:call(root,'shutdown')
            except Exception:service.terminate()
            try:service.wait(timeout=30)
            except subprocess.TimeoutExpired:service.kill();service.wait()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','pipeline','server','train','report'])
    ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--gpu',type=int,default=7)
    a=ap.parse_args()
    if a.command=='pipeline':pipeline(a.root,a.gpu)
    else:globals()[a.command](a.root)
