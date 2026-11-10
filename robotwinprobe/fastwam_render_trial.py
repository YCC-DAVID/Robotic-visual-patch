"""Isolated FastWAM render-patch trial. No changes to archived ImageWAM runs."""
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

from crossarch_common import ROOT, CAMERAS, WIRE, AUTH, dump, sha
from crossarch_stage_common import DEFAULT as CALIBRATION
from wam_orthogonal_control import call

DEFAULT = ROOT / 'robotwinprobe/out/fastwam_render_patch_20261027'
PRIOR = ROOT / 'robotwinprobe/out/wam_steps_to_patch_20261027'
PY = '/home/user1/miniconda3/envs/fastwam/bin/python'
OBJECTIVES = ('first_flow', 'final_action')
REPORT = ROOT / 'result/39_fastwam_render_patch.md'


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    existing=root/'protocol.json'
    if existing.exists():
        saved=json.loads(existing.read_text())
        if saved.get('experiment')=='fastwam_render_patch_exploratory':
            assert saved['authorization']['continue_with_proxy_diagnostic_only']
            return saved
    p = copy.deepcopy(json.loads((PRIOR / 'protocol.json').read_text()))
    p.update(experiment='fastwam_render_patch', model='FastWAM Wan2.2 5B + ActionDiT',
             source_run=str(PRIOR), control_port=18582,
             checkpoint='yuanty/fastwam@8eaceeb24c3cc92ff2a9c9a9d266a4941b836705/robotwin_uncond_3cam_384.pt')
    for key in ('intervention_masks', 'step_search', 'confirmed_evidence', 'confirmed_manifest_sha256'):
        p.pop(key, None)
    p['policies'] = {'fastwam': str(root / 'checkpoints/robotwin_uncond_3cam_384.pt')}
    p['patch_trial'].update(scope='two four-phase-sampled textures for one task, full exposure',
        training_start='only after native parity, clean baseline, and proxy diagnostics pass')
    p['evaluation'].update(horizon={'fastwam':32}, replan={'fastwam':24},
        baseline_reuse='NONE: new FastWAM clean and Random episodes', expected_reused_baselines=0,
        conditions=['clean','random_full',*OBJECTIVES], expected_total_episodes=40,
        scope='exploratory same-model comparison, ten previously used simulator initial states')
    p['gates'] = dict(clean_minimum=6, clean_count=10, native_max_abs=0,
        proxy_cosine_median_minimum=.8, proxy_relative_L0_median_maximum=.5,
        proxy_scope='16 validation states x Random/Checker x two new action noise seeds; aggregate diagnostic',
        failure_rule='stop before training if any gate fails; preserve data, no threshold relaxation')
    path=root/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == p
    else:
        dump(path,p)
    target=root/'training_plan.json'
    if not target.exists():
        shutil.copy2(PRIOR/'training_plan.json',target)
    return p


def status(root, stage, **extra):
    dump(root/'status.json',dict(stage=stage, pid=os.getpid(),updated=time.time(),**extra))


def server(root):
    import torch
    from multiprocessing.connection import Listener
    from fastwam_adapter import Backend
    from wam_flow_patch import gradient, diagnostic
    torch.set_num_threads(4)
    backend=Backend(root)
    cache={}
    p=json.loads((root/'protocol.json').read_text())
    with Listener(('127.0.0.1',p['control_port']), authkey=AUTH) as listener:
        dump(root/'ready.json',dict(pid=os.getpid(),ready=True))
        running=True
        while running:
            with listener.accept() as connection:
                try:
                    req=connection.recv();command=req['command']
                    if command=='shutdown':
                        result={'stopped':True};running=False
                    elif command=='ping':result={'ready':True}
                    elif command=='infer_image':result=backend.infer(req)
                    elif command=='patch_gradient':result=gradient(backend,req,cache)
                    elif command=='patch_diagnostic':result=diagnostic(backend,req,cache)
                    elif command=='audit':result=audit(root,backend)
                    else:raise ValueError(command)
                    connection.send(result)
                except Exception as error:
                    traceback.print_exc()
                    connection.send(dict(error=repr(error),traceback=traceback.format_exc()))


def audit(root, backend):
    import torch
    from crossarch_proxy import images
    from wam_flow_patch import teacher
    p=json.loads((root/'protocol.json').read_text())
    native=[]; proxy=[];cache={}
    for index, record in enumerate(p['diagnosis']['records']):
        with np.load(record['path']) as z:
            state=z['state14'].copy();prompt=str(z['prompt'])
            for texture in ('random','checker'):
                raw={w:z[c+'_'+texture].copy() for c,w in zip(CAMERAS,WIRE)}
                for seed in p['diagnosis']['action_noise_seeds']:
                    with torch.no_grad():
                        clean=teacher(backend,record['path'],seed,cache,True)
                        prepared=backend.raw_prepared(state,prompt,raw)
                        vr,ar=backend.response(prepared,seed,full=True)
                        if texture=='random' and seed==p['diagnosis']['action_noise_seeds'][0]:
                            image,proprio,(context,mask)=prepared
                            oracle=backend.model.infer_action(prompt=None,input_image=image,proprio=proprio,
                                action_horizon=32,context=context,context_mask=mask,num_inference_steps=10,
                                seed=seed,rand_device='cpu',tiled=False)['action']
                            native.append(dict(record=record['id'],max_abs=float((ar[0].float().cpu()-oracle).abs().max())))
                        q=torch.tensor(z['texture_'+texture] if 'texture_'+texture in z else pattern(texture),device='cuda')
                        if q.ndim==3:q=q.permute(2,0,1)[None].float()/255
                        vp,ap=backend.response(backend.prepare(state,prompt,images(q,record['path'])),seed,full=True)
                    dv=(vr.float().cpu()-clean['v']).numpy().ravel()
                    dp=(vp.float().cpu()-clean['v']).numpy().ravel()
                    cosine=float(dp@dv/max(np.linalg.norm(dp)*np.linalg.norm(dv),1e-30))
                    real=float(np.mean(dv**2));approx=float(np.mean(dp**2))
                    proxy.append(dict(record=record['id'],texture=texture,seed=seed,
                        cosine=cosine,relative_L0_error=abs(approx-real)/max(real,1e-30),render_L0=real,proxy_L0=approx))
        print('[FastWAM audit]',index+1,len(p['diagnosis']['records']),flush=True)
    median_cos=float(np.median([r['cosine'] for r in proxy]))
    median_error=float(np.median([r['relative_L0_error'] for r in proxy]))
    result=dict(native_rows=native,proxy_rows=proxy,native_passed=all(r['max_abs']==0 for r in native),
        proxy_cosine_median=median_cos,proxy_relative_L0_error_median=median_error,
        proxy_passed=median_cos>=p['gates']['proxy_cosine_median_minimum'] and
                     median_error<=p['gates']['proxy_relative_L0_median_maximum'])
    dump(root/'audit/native_proxy.json',result)
    return result


def pattern(name):
    from render_geometry import patterns
    return patterns()[name]


def rollout(root,condition,seed):
    from types import SimpleNamespace
    import crossarch_stage_rollout as runner
    original=runner.condition
    runner.condition=lambda name: dict(texture=name,stage=None,mode='full') if name in OBJECTIVES else original(name)
    runner.request=lambda model,payload:call(root,'infer_image',**{k:v for k,v in payload.items() if k!='command'})
    runner.run(SimpleNamespace(root=root,model='fastwam',condition=condition,seed=seed,smoke=False))


def train(root):
    p=json.loads((root/'protocol.json').read_text());cfg=p['patch_trial']
    draws=json.loads((root/'training_plan.json').read_text())
    for objective in OBJECTIVES:
        dest=root/'patches/fastwam'/objective;dest.mkdir(parents=True,exist_ok=True)
        checkpoint=dest/'training_state.npz';q=np.full((1,3,128,128),.5,np.float32);history=[];compute_s=0
        if checkpoint.exists():
            with np.load(checkpoint) as z:
                q=z['texture'].copy();history=json.loads(str(z['history']));compute_s=float(z['compute_s'])
        for iteration in range(len(history),cfg['steps']):
            gradient=np.zeros_like(q);values=[]
            for draw in draws[iteration]:
                r=call(root,'patch_gradient',objective=objective,texture=q,record=draw['record'],seed=draw['seed'])
                gradient+=r['gradient']/cfg['batch'];values.append(r['loss']);compute_s+=r['compute_elapsed_s']
            norm=float(np.linalg.norm(gradient));assert np.isfinite(gradient).all() and norm>0
            q=np.clip(q+cfg['lr']*np.sign(gradient),0,1);history.append(float(np.mean(values)))
            temp=checkpoint.with_suffix('.tmp.npz')
            np.savez_compressed(temp,texture=q,history=json.dumps(history),iteration=iteration+1,compute_s=compute_s)
            temp.replace(checkpoint)
            if iteration==0 or (iteration+1)%25==0:
                Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(dest/f'texture_{iteration+1:04d}.png')
            dump(dest/'progress.json',dict(iteration=iteration+1,total=cfg['steps'],loss=history[-1],
                gradient_norm=norm,compute_s=compute_s,complete=iteration+1==cfg['steps']))
            print('[FastWAM patch]',objective,iteration+1,history[-1],flush=True)
        final=dest/'texture_0500.png'
        rgb=np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)
        if final.exists():assert np.array_equal(np.asarray(Image.open(final).convert('RGB')),rgb)
        else:Image.fromarray(rgb).save(final)
    dump(root/'training_complete.json',dict(complete=True))


def capture(root,seed):
    import wam_flux_diagnosis as renderer
    def textures():
        return {'random':pattern('random'), **{name:np.asarray(Image.open(
            root/'patches/fastwam'/name/'texture_0500.png').convert('RGB')) for name in OBJECTIVES}}
    renderer.texture_bank=textures
    renderer.capture(root,seed)


def validate(root):
    p=json.loads((root/'protocol.json').read_text())
    textures={'random':pattern('random'), **{name:np.asarray(Image.open(
        root/'patches/fastwam'/name/'texture_0500.png').convert('RGB')) for name in OBJECTIVES}}
    dest=root/'patch_validation';dest.mkdir(exist_ok=True)
    for record in p['diagnosis']['records']:
        for name,rgb in textures.items():
            for path in ('proxy','render'):
                for seed in p['diagnosis']['action_noise_seeds']:
                    output=dest/(record['id']+'_'+name+'_'+path+'_'+str(seed)+'.json')
                    if output.exists():continue
                    r=call(root,'patch_diagnostic',record=record['path'],seed=seed,path=path,
                        texture=rgb.transpose(2,0,1)[None].astype(np.float32)/255,
                        capture=str(root/'captures'/(record['id']+'.npz')),texture_name=name)
                    np.savez_compressed(output.with_suffix('.npz'),delta_v=r.pop('delta_v'),delta_action=r.pop('delta_action'))
                    dump(output,dict(record=record['id'],stage=record['stage'],texture=name,path=path,noise_seed=seed,**r))


def gradient_probe(root):
    import torch
    from fastwam_adapter import Backend
    from wam_flow_patch import gradient
    from crossarch_proxy import images
    torch.set_num_threads(4)
    backend=Backend(root);cache={}
    draw=json.loads((root/'training_plan.json').read_text())[0][0]
    q=np.full((1,3,128,128),.5,np.float32);rows=[]
    # Repeat native parity after enabling autograd through the two inference wrappers.
    with np.load(draw['record']) as z:
        prepared=backend.prepare(z['state14'],str(z['prompt']),images(None,draw['record']))
    with torch.no_grad():
        _,action=backend.response(prepared,draw['seed'],full=True)
        image,proprio,(context,mask)=prepared
        oracle=backend.model.infer_action(prompt=None,input_image=image,proprio=proprio,context=context,
            context_mask=mask,action_horizon=32,num_inference_steps=10,seed=draw['seed'],rand_device='cpu')['action']
    difference=float((action[0].float().cpu()-oracle).abs().max());assert difference==0
    for objective in OBJECTIVES:
        req=dict(objective=objective,texture=q,record=draw['record'],seed=draw['seed'])
        base=gradient(backend,req,cache);g=base.pop('gradient')
        row=dict(objective=objective,base=base,probes=[])
        for sign in (1,-1):
            r=gradient(backend,dict(req,texture=np.clip(q+sign*.01*np.sign(g),0,1)),cache)
            r.pop('gradient');row['probes'].append(dict(sign=sign,**r))
        rows.append(row)
        print('[ascent probe]',objective,'base',base['loss'],'plus/minus',[r['loss'] for r in row['probes']],flush=True)
    dump(root/'audit/ascent_direction_probe.json',dict(native_max_abs=difference,rows=rows,
        scope='One fixed training state/noise; plus/minus one 0.01 sign-gradient step. Diagnostic only, no saved patch or training updates.'))
    note=root/'audit/gradient_wrapper_diagnosis.json'
    if note.exists():
        d=json.loads(note.read_text());d['verification']='finite nonzero texture gradients and exact native parity confirmed; see ascent_direction_probe.json'
        dump(note,d)


def report(root):
    p=json.loads((root/'protocol.json').read_text());s=json.loads((root/'status.json').read_text())
    lines=['# FastWAM：真实渲染patch试验','',f'状态：`{s["stage"]}`。',
        f'报告快照时间（Unix秒）：{time.time():.0f}。阶段切换时刷新；逐次训练更新见patches/fastwam/*/progress.json。','',
        '公开RoboTwin策略权重，冻结模型；任务place_container_plate/demo_clean。与ImageWAM运行独立。',
        '原生三相机384×320合成图、32步动作块、24步最大重规划、10步flow；各组均在阶段边界重规划。',
        'clean和Random均要求独立测量，不复用其他模型的成功率；实际完成情况见下表。', '',
        '预定两张纹理：首步速度差MSE、完整10步最终动作差MSE；梯度上升，500×8、10cm/Q128、灰色初始化。',
        '复用已封存的状态/EOT标定和相同抽样计划；不宣称跨架构迁移。','']
    audit_path=root/'audit/native_proxy.json'
    if audit_path.exists():
        a=json.loads(audit_path.read_text());lines.extend([
            f'原生动作逐值一致：{a["native_passed"]}；代理方向余弦中位数：{a["proxy_cosine_median"]:.4f}；L0相对误差中位数：{a["proxy_relative_L0_error_median"]:.1%}。',
            f'代理诊断参考：方向余弦≥0.8、L0相对误差≤0.5；达到参考值：{a["proxy_passed"]}。',
            '代理偏差保留为诊断，按用户新要求继续真实render探索。' if p['gates'].get('proxy_is_diagnostic_only') else '未达代理门槛时暂停训练。'])
        lines+=['','| 阶段 | 代理/真实速度差方向余弦中位数 | L0相对误差中位数 |','|---|---:|---:|']
        stages={r['id']:r['stage'] for r in p['diagnosis']['records']}
        for stage in ('approach','grasp','transport','place'):
            rows=[r for r in a['proxy_rows'] if stages[r['record']]==stage]
            lines.append(f'| {stage} | {np.median([r["cosine"] for r in rows]):.4f} | {np.median([r["relative_L0_error"] for r in rows]):.1%} |')
    lines+=['','| 条件 | 完成 | 任务成功 | clean成功后的配对失败 |','|---|---:|---:|---:|']
    clean={}
    for f in (root/'episodes/fastwam/clean').glob('seed*.json'):
        r=json.loads(f.read_text())
        if 'success' in r:clean[r['seed']]=r
    for condition in p['evaluation']['conditions']:
        rows=[json.loads(f.read_text()) for f in (root/'episodes/fastwam'/condition).glob('seed*.json')]
        rows=[r for r in rows if 'success' in r]
        eligible=[r for r in rows if clean.get(r['seed'],{}).get('success')]
        paired='—' if condition=='clean' else (f'{sum(not r["success"] for r in eligible)}/{len(eligible)}' if rows else '未开始')
        successes=str(sum(r['success'] for r in rows)) if rows else '—'
        lines.append(f'| {condition} | {len(rows)}/10 | {successes} | {paired} |')
    for name in OBJECTIVES:
        path=root/'patches/fastwam'/name/'progress.json'
        lines.append(f'\n{name}更新：'+(str(json.loads(path.read_text())['iteration'])+'/500' if path.exists() else '未开始'))
    measurements=[json.loads(f.read_text()) for f in (root/'patch_validation').glob('*.json')]
    lines+=['',f'最终纹理代理/真实验证：{len(measurements)}/192。']
    if measurements:
        lines+=['','| 纹理 | 路径 | 条数 | 首步L0中位数 | 最终动作MSE中位数 | 关节RMSE/rad |',
                '|---|---|---:|---:|---:|---:|']
        for texture in ('random',*OBJECTIVES):
            for path in ('proxy','render'):
                rows=[r for r in measurements if r['texture']==texture and r['path']==path]
                if rows:lines.append(f'| {texture} | {path} | {len(rows)} | '+ ' | '.join(
                    f'{np.median([r[k] for r in rows]):.6g}' for k in ('first_L0','final_action_MSE','joint_action_RMSE_rad'))+' |')
    ascent=root/'audit/ascent_direction_probe.json'
    if ascent.exists():
        a=json.loads(ascent.read_text());lines+=['',f'可微接口修正后原生完整动作最大绝对误差：{a["native_max_abs"]}。',
            '固定单个训练状态/noise的一步梯度方向检查（仅诊断，未启动训练）：',
            '| 目标 | 灰色起点 | 沿梯度+0.01 | 沿梯度-0.01 |','|---|---:|---:|---:|']
        for r in a['rows']:lines.append(f'| {r["objective"]} | {r["base"]["loss"]:.6g} | {r["probes"][0]["loss"]:.6g} | {r["probes"][1]["loss"]:.6g} |')
    if s.get('error'):lines+=['',f'停止原因：{s["error"]}']
    lines+=['',f'原始目录：`{root}`。','运行错误与方法门槛失败分开记录；未开始的实验不记为攻击失败。',
        ('代理偏差作为诊断保留；有效性以实际render闭环及同尺寸Random配对计数为准，不宣称跨架构迁移。'
         if p['gates'].get('proxy_is_diagnostic_only') else
         '代理门槛未通过时，本轮只构成适配诊断，不构成训练后patch ASR或跨架构迁移结论。')]
    Path(p.get('report_path',str(REPORT))).write_text('\n'.join(lines)+'\n')


def pipeline(root,gpu):
    lock=open(root/'pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=prepare(root);server_process=None
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',
             HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    (root/'logs').mkdir(exist_ok=True)
    def job(name,argv):
        status(root,name);report(root)
        with open(root/'logs'/(name+'.log'),'a') as log:
            subprocess.run(argv,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    def own(*args):return [PY,'-u',str(Path(__file__).resolve()),*args,'--root',str(root)]
    try:
        status(root,'checkpoint_identity');report(root)
        release=json.loads((root/'release_metadata.json').read_text())
        expected=next(r['lfs']['sha256'] for r in release['siblings'] if r['rfilename']=='robotwin_uncond_3cam_384.pt')
        actual=sha(root/'checkpoints/robotwin_uncond_3cam_384.pt');assert actual==expected
        dump(root/'audit/checkpoint_identity.json',dict(actual_sha256=actual,expected_sha256=expected,passed=True,
            code_revision=subprocess.check_output(['git','-C',str(ROOT/'third_party/FastWAM'),'rev-parse','HEAD'],text=True).strip()))
        snapshot=root/'source_snapshots'/str(os.getpid());snapshot.mkdir(parents=True,exist_ok=True)
        for name in ('fastwam_adapter.py','fastwam_render_trial.py','wam_flow_patch.py','crossarch_proxy.py','crossarch_stage_rollout.py'):
            shutil.copy2(ROOT/'robotwinprobe'/name,snapshot/name)
        if not (root/'prompts.npz').exists():job('encode_prompts',own('prompts'))
        status(root,'start_backend');report(root)
        ready=root/'ready.json'
        if ready.exists():ready.unlink()
        with open(root/'logs/backend.log','a') as log:
            server_process=subprocess.Popen(own('server'),env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        started=time.monotonic()
        while not ready.exists():
            if server_process.poll() is not None:raise RuntimeError('FastWAM backend exited; inspect logs/backend.log')
            if time.monotonic()-started>600:raise RuntimeError('FastWAM backend readiness timeout')
            time.sleep(2)
        status(root,'native_proxy_audit');report(root)
        audit_path=root/'audit/native_proxy.json'
        a=json.loads(audit_path.read_text()) if audit_path.exists() else call(root,'audit')
        report(root)
        if not a['native_passed']:raise RuntimeError('Native numerical parity failed')
        # Complete clean characterization before spending the patch training budget.
        for seed in p['evaluation']['seeds']:
            job('clean_'+str(seed),['bash',str(ROOT/'robotwinprobe/run_sim.sh'),
                str(Path(__file__).resolve()),'rollout','--root',str(root),'--condition','clean','--seed',str(seed)])
        rows=[json.loads(f.read_text()) for f in (root/'episodes/fastwam/clean').glob('seed*.json')]
        clean_success=sum(r.get('success',False) for r in rows)
        failures=[]
        if clean_success<p['gates']['clean_minimum']:failures.append(f'clean {clean_success}/10 below 6/10')
        if not a['proxy_passed'] and not p['gates'].get('proxy_is_diagnostic_only',False):
            failures.append('proxy response fidelity gate failed')
        if failures:
            status(root,'stopped_at_method_gate',error='; '.join(failures));report(root);return
        # Gradient preflight on four phases and both objectives, before the first update.
        draws=json.loads((root/'training_plan.json').read_text())
        grad_rows=[]
        for objective in OBJECTIVES:
            for draw in draws[0][::2]:
                r=call(root,'patch_gradient',objective=objective,texture=np.full((1,3,128,128),.5,np.float32),
                       record=draw['record'],seed=draw['seed'])
                grad_rows.append({k:v for k,v in r.items() if k!='gradient'}|dict(stage=draw['stage']))
        dump(root/'audit/gradient_preflight.json',dict(passed=True,rows=grad_rows))
        job('patch_training',own('train'))
        for seed in sorted({r['seed'] for r in p['diagnosis']['records']}):
            job('capture_final_'+str(seed),['bash',str(ROOT/'robotwinprobe/run_sim.sh'),
                str(Path(__file__).resolve()),'capture','--root',str(root),'--seed',str(seed)])
        job('validate_final',own('validate'))
        for condition in ('random_full',*OBJECTIVES):
            for seed in p['evaluation']['seeds']:
                job(condition+'_'+str(seed),['bash',str(ROOT/'robotwinprobe/run_sim.sh'),
                    str(Path(__file__).resolve()),'rollout','--root',str(root),'--condition',condition,'--seed',str(seed)])
        status(root,'complete');report(root)
    except Exception as error:
        status(root,'runtime_error',error=repr(error),traceback=traceback.format_exc());report(root);raise
    finally:
        if server_process is not None and server_process.poll() is None:
            try:call(root,'shutdown')
            except Exception:server_process.terminate()
            try:server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:server_process.kill();server_process.wait()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','prompts','server','train','rollout','capture','validate','gradient_probe','pipeline','report'])
    ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--gpu',type=int,default=3)
    ap.add_argument('--condition');ap.add_argument('--seed',type=int)
    a=ap.parse_args()
    if a.command=='prepare':prepare(a.root)
    elif a.command=='prompts':
        from fastwam_adapter import prompts
        prompts(a.root)
    elif a.command=='rollout':rollout(a.root,a.condition,a.seed)
    elif a.command=='capture':capture(a.root,a.seed)
    elif a.command=='pipeline':pipeline(a.root,a.gpu)
    else:globals()[a.command](a.root)
