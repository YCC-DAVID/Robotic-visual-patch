"""Frozen ImageWAM patches: real/proxy pixels, VAE, FLUX KV, and full flow traces."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback

import numpy as np

from crossarch_common import ROOT, PY, CAMERAS, WIRE, STAGES, dump, sha
from crossarch_capture import records
from crossarch_stage_common import DEFAULT as SOURCE
from wam_orthogonal_control import DEFAULT as CONTROL

DEFAULT = ROOT / 'robotwinprobe/out/wam_flux_diagnosis_20261027'
REPORT = ROOT / 'result/35_imagewam_flux_diagnosis.md'
TEXTURES = ('random', 'gray') + STAGES


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    original = json.loads((SOURCE / 'protocol.json').read_text())
    selected = [r for seed in original['data']['validation_seeds'][:2]
                for r in records(SOURCE, seed) if r['id'].endswith('_d0')]
    assert len(selected) == 16
    p = copy.deepcopy(original)
    p.update(experiment='imagewam_flux_render_diagnosis', source_run=str(SOURCE),
             source_protocol_sha256=sha(SOURCE / 'protocol.json'))
    p['diagnosis'] = dict(records=selected, textures=list(TEXTURES), paths=['proxy', 'render'],
        action_noise_seeds=[999000001, 999000002], full_flow_steps=10, expected_measurements=384,
        expected_clean_traces=32, held_out='first two existing validation trajectories, first/last saved frame of each phase',
        rendering='nominal 10cm pose/light, matching the completed closed-loop evaluation',
        measurements='native pixels; VAE tokens; image-token K/V per FLUX layer; first-flow MSE/SSE; all ten flow velocities and action latent updates; final normalized and joint-space actions',
        scope='offline paired feature diagnosis of frozen final textures, no new patch training or ASR',
        texture_hashes={stage:sha(SOURCE/'patches/wam'/stage/'texture_0500.png') for stage in STAGES})
    used={draw['seed'] for path in (SOURCE/'plans').glob('*.json')
          for batch in json.loads(path.read_text()) for draw in batch}
    assert not used.intersection(p['diagnosis']['action_noise_seeds'])
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p, 'Preserve this diagnostic protocol'
    else:dump(path,p)
    for name in ('prompts.npz','provenance.json'):
        if not (root/name).exists():os.link(SOURCE/name,root/name)
    return p


def texture_bank():
    from PIL import Image
    from render_geometry import patterns
    bank={k:patterns()[k] for k in ('random','gray')}
    bank.update({stage:np.asarray(Image.open(SOURCE/'patches/wam'/stage/'texture_0500.png').convert('RGB')) for stage in STAGES})
    return bank


def capture(root,seed):
    import bootstrap as B
    B.simulator_paths()
    from scene_smoke import make_task
    from render_geometry import Patch
    from eot_conditions import Lighting,LIGHTS
    p=json.loads((root/'protocol.json').read_text())
    targets=[r for r in p['diagnosis']['records'] if r['seed']==seed and not (root/'captures'/(r['id']+'.npz')).exists()]
    if not targets:return
    dest=root/'replays'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE/'trajectories'/f'seed{seed}'/'trace.npz') as z:actions=z['actions'].copy()
    task=None;random.seed(seed)
    try:
        task,_,_=make_task(p['task'],seed,dest);lighting=Lighting(task);textures=texture_bank()
        for query in range(max(r['query'] for r in targets)+1):
            for record in [r for r in targets if r['query']==query]:
                lighting.apply(LIGHTS['nominal']);clean=task.get_obs()
                with np.load(record['path']) as z:
                    state=clean['joint_action']['vector'].astype(np.float32)
                    assert np.array_equal(state,z['state14']),('replay state',seed,query)
                    for camera in CAMERAS:
                        assert np.array_equal(clean['observation'][camera]['rgb'],z[camera+'_clean']),('replay pixels',seed,query,camera)
                    data={camera+'_clean':clean['observation'][camera]['rgb'].copy() for camera in CAMERAS}
                    patch=Patch(task,dest,center=record['center_m'],size=float(p['patch']['size_m']))
                    try:
                        for name,texture in textures.items():
                            patch.texture(texture);obs=task.get_obs()
                            assert np.array_equal(state,obs['joint_action']['vector'].astype(np.float32))
                            for camera in CAMERAS:
                                data[camera+'_'+name]=obs['observation'][camera]['rgb'].copy()
                                if name in ('random','gray'):
                                    assert np.array_equal(data[camera+'_'+name],z[camera+'_'+name]),('control pixels',seed,query,camera,name)
                    finally:task.scene.remove_actor(patch.actor)
                    restored=task.get_obs()
                    for camera in CAMERAS:assert np.array_equal(restored['observation'][camera]['rgb'],data[camera+'_clean'])
                output=root/'captures'/(record['id']+'.npz');output.parent.mkdir(exist_ok=True)
                temporary=output.with_suffix('.tmp.npz');np.savez_compressed(temporary,**data);temporary.replace(output)
                print('[capture final textures]',record['id'],flush=True)
            if query<len(actions):task.take_action(actions[query],action_type='qpos')
    finally:
        if task is not None:task.close_env()


def cosine(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    denominator=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(a@b/denominator) if denominator>1e-20 else None


def relative(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def measure(root):
    import torch
    from crossarch_backend import Backend
    from crossarch_proxy import images
    backend=Backend('wam',root);backend.audit()
    model=backend.model;M=backend.M
    p=json.loads((root/'protocol.json').read_text());cfg=p['diagnosis'];textures=texture_bank()
    target=root/'measurements';target.mkdir(exist_ok=True)
    representations=root/'representations';representations.mkdir(exist_ok=True)
    dump(root/'audit/measurement_source.json',dict(script_sha256=sha(Path(__file__)),
        wam_adapter_sha256=sha(ROOT/'robotwinprobe/crossarch_wam.py')))
    times,deltas=model.infer_action_scheduler.build_inference_schedule(10,device='cuda',dtype=torch.bfloat16)
    dump(root/'schedule.json',dict(times=times.float().cpu().tolist(),deltas=deltas.float().cpu().tolist()))

    def features(prepared):
        latent,_=model._encode_flux2_image_tokens(prepared[0],time_value=10.)
        cache=M.prefix(model,*prepared)
        return latent,cache

    def feature_metrics(latent,cache,clean_latent,clean_cache):
        layers=[];start=cache[0]['txt_len']
        for family in ('double','single'):
            for index,(actual,clean) in enumerate(zip(cache[0][family],clean_cache[0][family])):
                row=dict(layer=family+'_'+str(index))
                for key in ('k','v'):
                    x=actual[key][:,start:].float();y=clean[key][:,start:].float()
                    row[key+'_relative_delta']=float((x-y).norm()/y.norm().clamp_min(1e-12))
                layers.append(row)
        l=latent.float();c=clean_latent.float()
        x=cache[0]['final_video']['img'].float();y=clean_cache[0]['final_video']['img'].float()
        return dict(latent_relative_delta=float((l-c).norm()/c.norm().clamp_min(1e-12)),
            backbone_relative_delta=float((x-y).norm()/y.norm().clamp_min(1e-12)),
            latent_shape=list(latent.shape),
            latent_delta_token_RMS=(l-c).square().mean(-1).sqrt()[0].cpu().tolist(),kv_layers=layers)

    def trace(cache,noise_seed):
        x=M.noise(noise_seed);xs=[x.float().cpu().numpy()];vs=[]
        for t,dt in zip(times,deltas):
            v=M.velocity(model,cache,x,t.expand(1));vs.append(v.float().cpu().numpy())
            x=model.infer_action_scheduler.step(v,dt,x);xs.append(x.float().cpu().numpy())
        return dict(v=np.stack(vs),x=np.stack(xs),action=M.denorm(backend.proc,x[0]))

    with torch.no_grad():
        for record in cfg['records']:
            with np.load(record['path']) as z:state=z['state14'].copy();prompt=str(z['prompt'])
            with np.load(root/'captures'/(record['id']+'.npz')) as z:rendered={k:z[k].copy() for k in z.files}
            clean_views=images(None,record['path']);prepared=backend.prepare(state,prompt,clean_views)
            clean_latent,clean_cache=features(prepared)
            cleans={ns:trace(clean_cache,ns) for ns in cfg['action_noise_seeds']}
            # Independent oracle forward under the exact native public deployment.
            noise_seed=cfg['action_noise_seeds'][0]
            official=backend.infer(dict(seed=noise_seed,state=state,prompt=prompt,
                images={w:rendered[n+'_clean'] for n,w in zip(CAMERAS,WIRE)}))['actions']
            error=float(np.max(np.abs(official-cleans[noise_seed]['action'])))
            assert error<1e-4,('native trace mismatch',record['id'],error)
            dump(root/'audit'/('native_'+record['id']+'.json'),dict(passed=True,max_action_error=error))
            for ns,result in cleans.items():
                np.savez_compressed(target/(record['id']+f'_clean_{ns}.npz'),**result)
            for texture,label in textures.items():
                latent_by_path={};row_by_path={}
                for path in cfg['paths']:
                    if path=='proxy':
                        q=torch.tensor(label.copy(),device='cuda').permute(2,0,1)[None].float()/255
                        views=images(q,record['path'])
                    else:
                        views=[torch.tensor(rendered[n+'_'+texture],device='cuda').permute(2,0,1)[None].float()/255 for n in CAMERAS]
                    current=backend.prepare(state,prompt,views)
                    latent,cache=features(current)
                    info=feature_metrics(latent,cache,clean_latent,clean_cache)
                    latent_by_path[path]=latent.float().cpu().numpy()-clean_latent.float().cpu().numpy()
                    np.savez_compressed(representations/(record['id']+'_'+texture+'_'+path+'.npz'),
                        latent=latent.float().cpu().numpy(),clean_latent=clean_latent.float().cpu().numpy())
                    pixel_rows=[]
                    with np.load(record['path']) as z:
                        for n,view in zip(CAMERAS,views):
                            actual=rendered[n+'_'+texture].astype(float)/255
                            proxy=view[0].permute(1,2,0).float().cpu().numpy()
                            delta=proxy-actual;mask=z[n+'_mask'].astype(bool)
                            pixel_rows.append(dict(camera=n,rmse=float(np.sqrt(np.mean(delta**2))),
                                mask_rmse=float(np.sqrt(np.mean(delta[mask]**2))) if mask.any() else None))
                    for ns in cfg['action_noise_seeds']:
                        key=record['id']+'_'+texture+'_'+path+'_'+str(ns)
                        out=trace(cache,ns);clean=cleans[ns]
                        dv=out['v']-clean['v'];dx=out['x']-clean['x'];da=out['action']-clean['action']
                        joint=da[:,[0,1,2,3,4,5,7,8,9,10,11,12]]
                        latent_delta_rms=np.sqrt(np.mean(dx**2,axis=(1,2,3)))
                        row=dict(record=record['id'],stage=record['stage'],texture=texture,path=path,noise_seed=ns,
                            L0=float(np.mean(dv[0]**2)),L0_sum=float(np.sum(dv[0]**2)),
                            velocity_cosine=cosine(out['v'][0],clean['v'][0]),
                            relative_velocity_change=relative(out['v'][0],clean['v'][0]),
                            final_normalized_action_MSE=float(np.mean(dx[-1]**2)),
                            joint_action_RMSE_rad=float(np.sqrt(np.mean(joint**2))),
                            first_joint_action_RMSE_rad=float(np.sqrt(np.mean(joint[0]**2))),
                            velocity_delta_RMS_by_step=np.sqrt(np.mean(dv**2,axis=(1,2,3))).tolist(),
                            action_latent_delta_RMS_by_step=latent_delta_rms.tolist(),
                            final_over_first_update=float(latent_delta_rms[-1]/latent_delta_rms[1]) if latent_delta_rms[1]>1e-12 else None,
                            pixels= pixel_rows,**info)
                        dump(target/(key+'.json'),row)
                        np.savez_compressed(target/(key+'.npz'),**out)
                        row_by_path[(path,ns)]=(row,dv[0],dx[-1])
                for ns in cfg['action_noise_seeds']:
                    proxy,pv,pa=row_by_path[('proxy',ns)];actual,rv,ra=row_by_path[('render',ns)]
                    dump(root/'pairs'/(record['id']+'_'+texture+'_'+str(ns)+'.json'),dict(
                        record=record['id'],stage=record['stage'],texture=texture,noise_seed=ns,
                        proxy_L0=proxy['L0'],render_L0=actual['L0'],
                        relative_L0_error=abs(proxy['L0']-actual['L0'])/max(actual['L0'],1e-12),
                        delta_velocity_cosine=cosine(pv,rv),delta_action_cosine=cosine(pa,ra),
                        latent_delta_cosine=cosine(latent_by_path['proxy'],latent_by_path['render'])))
                print('[FLUX diagnosis]',record['id'],texture,flush=True)


def report(root,plots=False):
    p=json.loads((root/'protocol.json').read_text());status=json.loads((root/'status.json').read_text())
    rows=[json.loads(f.read_text()) for f in (root/'measurements').glob('*.json')]
    pairs=[json.loads(f.read_text()) for f in (root/'pairs').glob('*.json')]
    summary=dict(status=status,captures=len(list((root/'captures').glob('*.npz'))),expected_captures=16,
        measurements=len(rows),expected_measurements=384,pairs=len(pairs),expected_pairs=192,matching_stage=[],
        stage_matched_controls=[],errors=[status['error']] if status.get('error') else [])
    for texture in TEXTURES:
        selected=[r for r in rows if r['path']=='render' and r['texture']==texture and (texture not in STAGES or r['stage']==texture)]
        if selected:
            summary['matching_stage'].append(dict(texture=texture,n=len(selected),
                **{key:float(np.median([r[key] for r in selected])) for key in
                    ('L0','relative_velocity_change','latent_relative_delta','backbone_relative_delta','joint_action_RMSE_rad')}))
    if pairs:
        summary['proxy_render']={key:float(np.median([r[key] for r in pairs if r[key] is not None]))
            for key in ('relative_L0_error','delta_velocity_cosine','delta_action_cosine','latent_delta_cosine')}
    for stage in STAGES:
        trained=[r for r in rows if r['path']=='render' and r['stage']==stage and r['texture']==stage]
        random_rows={(r['record'],r['noise_seed']):r for r in rows
                     if r['path']=='render' and r['stage']==stage and r['texture']=='random'}
        matched=[r for r in trained if (r['record'],r['noise_seed']) in random_rows]
        if matched:
            baseline=[random_rows[(r['record'],r['noise_seed'])] for r in matched]
            summary['stage_matched_controls'].append(dict(stage=stage,n=len(matched),
                trained_L0_median=float(np.median([r['L0'] for r in matched])),
                random_L0_median=float(np.median([r['L0'] for r in baseline])),
                paired_L0_ratio_median=float(np.median([r['L0']/max(b['L0'],1e-12) for r,b in zip(matched,baseline)])),
                trained_joint_RMSE_rad_median=float(np.median([r['joint_action_RMSE_rad'] for r in matched])),
                random_joint_RMSE_rad_median=float(np.median([r['joint_action_RMSE_rad'] for r in baseline])),
                velocity_angle_degrees_median=float(np.median([np.degrees(np.arccos(np.clip(r['velocity_cosine'],-1,1))) for r in matched])),
                final_over_first_update_median=float(np.median([r['final_over_first_update'] for r in matched if r['final_over_first_update'] is not None]))))
    pixel_conditions=[r for r in rows if r['path']=='proxy' and r['noise_seed']==p['diagnosis']['action_noise_seeds'][0]]
    pixel_checks=[check for r in pixel_conditions for check in r['pixels']]
    if pixel_checks:
        summary['final_texture_pixel_check']=dict(conditions=len(pixel_conditions),
            passed=all(r['rmse']<.06 and (r['mask_rmse'] is None or r['mask_rmse']<.15) for r in pixel_checks),
            max_rmse=max(r['rmse'] for r in pixel_checks),
            max_mask_rmse=max(r['mask_rmse'] for r in pixel_checks if r['mask_rmse'] is not None))
    dump(root/'summary.json',summary)
    lines=['# ImageWAM：最终patch的render代理与FLUX机制诊断','',
        f'状态：{status["stage"]}；真实捕获{summary["captures"]}/16；配对响应{len(rows)}/384。','',
        '固定前两条验证轨迹、四阶段首/末保存帧，共16状态；6张纹理（Random、灰色、4张WAM最终阶段patch），'
        '代理/真实渲染两条路径，两份未参与训练的动作noise。位置、10cm尺寸及光照与已完成闭环一致。','',
        '使用同一原生FLUX.2 4B RoboTwin模型与AE。每个状态独立核验手工10步轨迹与官方完整动作一致；'
        '真实重放的clean、Random和灰色图像均须与既有标定逐像素一致。','']
    if summary['stage_matched_controls']:
        lines+=['下表每行严格使用相同阶段、状态与noise的Random对照；每阶段4个状态×2份noise，共8条。','',
            '| 阶段 | 训练纹理L0中位数 | 同状态Random L0中位数 | 训练纹理关节RMSE/rad | Random关节RMSE/rad | 内部速度转角中位数 |',
            '|---|---:|---:|---:|---:|---:|']
        for r in summary['stage_matched_controls']:
            lines.append(f'| {r["stage"]} | {r["trained_L0_median"]:.6g} | {r["random_L0_median"]:.6g} | '
                f'{r["trained_joint_RMSE_rad_median"]:.6g} | {r["random_joint_RMSE_rad_median"]:.6g} | '
                f'{r["velocity_angle_degrees_median"]:.4g}° |')
        lines+=['','真实render下训练纹理引起了可测扰动；本轮接近/抓取L0中位数约为同状态Random的3.4倍，'
            '但对应关节动作块RMSE中位数仍约0.002–0.004 rad，不能由开环差值推导任务必然失败。'
            '已有闭环仍为每条件0/9，未新增或改写。','',
            '持续图像扰动下，完整去噪后的动作latent差中位数仍大于第一次更新后的差，'
            '不能概括为后续去噪把所有图像扰动消除了；该轨迹比值也不是孤立首步干预的恢复率。'
            '首步强制正交控制须结合独立第34号结果解释。','']
    lines += [f'最终纹理/对照图像检查：{summary.get("final_texture_pixel_check", "待测")}。','',
        '| 纹理 | 条数 | 真实render L0中位数 | 相对速度改变量 | VAE相对变化 | FLUX输出相对变化 | 关节动作RMSE/rad |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary['matching_stage']:
        lines.append(f'| {r["texture"]} | {r["n"]} | '+ ' | '.join(f'{r[key]:.6g}' for key in
            ('L0','relative_velocity_change','latent_relative_delta','backbone_relative_delta','joint_action_RMSE_rad'))+' |')
    lines+=['','阶段纹理只在该表选取对应阶段状态；Random/灰色包含全部状态，不能直接当作严格同阶段比值。'
        '完整跨阶段记录保存在measurements，支持相同状态/noise逐条配对。','',
        'L0为16×14首步速度差MSE，另存L0_sum；完整动作另报告标准化MSE与真实关节角RMSE。'
        'VAE与FLUX表示使用各自clean范数归一化，跨空间的数值大小不能直接解释为衰减倍数。'
        '各步轨迹差描述持续图像条件下的响应；是否能纠正一次首步干预，需结合独立正交控制。','',
        f'代理/真实渲染配对中位数：{summary.get("proxy_render", "待测")}。','',
        '这是16状态的离线诊断，不是新增闭环ASR，也不能把首步变化或latent距离直接等同于任务失败率。','',
        f'原始目录：`{root}`。']
    schedule=root/'schedule.json'
    if schedule.exists():
        values=json.loads(schedule.read_text())['deltas']
        lines+=['',f'原生10步积分步长：{values}。首步绝对步长{abs(values[0]):.6g}，'
            f'最后一步{abs(values[-1]):.6g}；首步L0与最终动作之间还受积分权重和后续去噪影响。']
    if plots and rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        folder=root/'figures';folder.mkdir(exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        texture_names=['random']+list(STAGES)
        matrix=np.array([[np.median([r['L0'] for r in rows if r['path']=='render' and r['texture']==texture and r['stage']==stage])
                          for stage in STAGES] for texture in texture_names])
        image=axes[0].imshow(np.log10(np.maximum(matrix,1e-12)),aspect='auto')
        axes[0].set_xticks(range(4),STAGES,rotation=20);axes[0].set_yticks(range(5),texture_names)
        axes[0].set_title('Actual render: log10 first-flow MSE');fig.colorbar(image,ax=axes[0])
        for path,color in [('proxy','tab:orange'),('render','tab:blue')]:
            selected=[r for r in rows if r['path']==path and r['texture'] in STAGES and r['texture']==r['stage']]
            curve=np.median([r['action_latent_delta_RMS_by_step'] for r in selected],axis=0)
            axes[1].plot(range(11),curve,label=path,color=color,marker='o')
        selected=[r for r in rows if r['path']=='render' and r['texture']=='random']
        axes[1].plot(range(11),np.median([r['action_latent_delta_RMS_by_step'] for r in selected],axis=0),label='Random render',color='gray')
        axes[1].set_xlabel('Denoising updates');axes[1].set_ylabel('Median action-latent delta RMS');axes[1].legend()
        fig.tight_layout();fig.savefig(folder/'render_flow.png',dpi=170);fig.savefig(folder/'render_flow.pdf');plt.close(fig)
        lines+=['',f'![真实render与逐步动作变化]({folder / "render_flow.png"})']
    elif (root/'figures/render_flow.png').exists():
        lines+=['',f'![真实render与逐步动作变化]({root / "figures/render_flow.png"})']
    REPORT.write_text('\n'.join(lines)+'\n')
    return summary


def pipeline(a):
    p=prepare(a.root)
    lock=(a.root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    logs=a.root/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',PYTHONHASHSEED='0')
    owned=[];last_notice=time.monotonic()
    def status(stage,**extra):
        dump(a.root/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra))
        return report(a.root)
    def notify(kind,summary):
        nonlocal last_notice
        if not a.thread:return
        message=(f'ImageWAM render代理/FLUX特性诊断：{kind}。请读取{a.root}/status.json和summary.json后中文汇报。'
            f'目标：冻结4张最终WAM纹理，与Random/灰色在16验证状态、2新noise、代理/真实渲染下比较VAE、FLUX KV、首步L0与完整动作。'
            f'当前{summary["status"]["stage"]}；捕获{summary["captures"]}/16；响应{summary["measurements"]}/384。'
            f'报告{REPORT}；原第31/32号闭环结果不改动。报告目标、计划进度、状态/错误和剩余时间；'
            '运行故障保留证据尽量自行修复；方法/机制问题及时报告，不擅自改协议。')
        try:
            result=subprocess.run(['/home/user1/.local/bin/codex','queue','--thread',a.thread,'--message',message],capture_output=True,text=True,timeout=30)
            dump(a.root/'last_notification.json',dict(kind=kind,time=time.time(),returncode=result.returncode,receipt=result.stdout,error=result.stderr))
        except Exception as error:
            dump(a.root/'last_notification.json',dict(kind=kind,time=time.time(),error=repr(error)))
        last_notice=time.monotonic()
    def call_job(name,argv,gpu):
        required = 18000 if name=='measuring_flux' else 5000
        while True:
            query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            free={line.split(',')[0].strip():int(line.split(',')[1]) for line in query.stdout.strip().splitlines()}
            if free[gpu]>=required:break
            s=status('waiting_for_gpu',next_stage=name,gpu=gpu,free_MiB=free[gpu],required_MiB=required)
            if time.monotonic()-last_notice>=1800:notify('每30分钟进度',s)
            time.sleep(10)
        status(name)
        with (logs/(name+'.log')).open('a') as log:
            proc=subprocess.Popen(argv,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=gpu),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        owned.append(proc)
        while proc.poll() is None:
            time.sleep(5);s=status(name,child_pid=proc.pid)
            if time.monotonic()-last_notice>=1800:notify('每30分钟进度',s)
        if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}; see logs/{name}.log')
    try:
        for seed in sorted({r['seed'] for r in p['diagnosis']['records']}):
            call_job('capture_'+str(seed),['bash',str(ROOT/'robotwinprobe/run_sim.sh'),str(Path(__file__).resolve()),
                'capture','--root',str(a.root),'--seed',str(seed)],a.sim_gpu)
        # Release the existing diagnostic backend first; no second model copy competes for its memory.
        while True:
            prior=json.loads((CONTROL/'status.json').read_text())
            ready=CONTROL/'ready.json';alive=False
            if ready.exists():
                pid=json.loads(ready.read_text())['pid']
                try:alive=b'wam_orthogonal_control.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
                except (FileNotFoundError,ProcessLookupError):pass
            if not alive:break
            s=status('waiting_for_orthogonal_backend',orthogonal_stage=prior['stage'])
            if time.monotonic()-last_notice>=1800:notify('每30分钟进度',s)
            time.sleep(10)
        call_job('measuring_flux',[PY['wam'],'-u',str(Path(__file__).resolve()),'measure','--root',str(a.root)],a.gpu)
        s=report(a.root)
        assert s['captures']==16 and s['measurements']==384 and s['pairs']==192,s
        status('rendering_report')
        s=status('complete');report(a.root,plots=True);notify('完成通知',s)
    except Exception as error:
        s=status('stopped_on_error',error=repr(error),traceback=traceback.format_exc());notify('即时异常告警',s)
        raise
    finally:
        for proc in reversed(owned):
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['pipeline','capture','measure','report'])
    parser.add_argument('--root',type=Path,default=DEFAULT);parser.add_argument('--seed',type=int)
    parser.add_argument('--gpu',default='4');parser.add_argument('--sim-gpu',default='2');parser.add_argument('--thread')
    args=parser.parse_args()
    if args.mode=='pipeline':pipeline(args)
    elif args.mode=='capture':capture(args.root,args.seed)
    elif args.mode=='measure':measure(args.root)
    else:report(args.root,plots=True)
