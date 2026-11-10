"""Run and publish paired digital-transform diagnostics and closed loops.

Uses only its own model service. Existing experiments remain read-only.
"""
import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from digital_transform_common import (ROOT, PROBE, TRANSFER, METHODS, CONDITIONS,
    dump, sha, prepare, textures, composite, center, rotated_grid)

PY = '/home/user1/miniconda3/envs/openpi-server/bin/python'
EXPERIMENT = PROBE/'digital_transform_experiment.py'


def cpu_audit(dest):
    import bootstrap as B
    B.simulator_paths()
    import torch
    from render_geometry import CAMERAS, WIRE
    from render_proxy import grid_for
    from render_rollout import digital_images
    torch.set_num_threads(4)
    checked=0; max_error=0.
    for seed in [10110,10111]:
        with np.load(TRANSFER/f'calibration/seed{seed}.npz') as z:
            for i in range(len(z['state14'])):
                obs={'observation':{n:{'rgb':z[n+'_clean'][i]} for n in CAMERAS}}
                geo={n:{k:z[n+'_'+k][i] for k in ['intrinsic','extrinsic','mask']} for n in CAMERAS}
                for rgb in textures().values():
                    actual=composite(obs,geo,rgb,CONDITIONS[0])
                    old=digital_images(obs,geo,rgb,[0,0,.741],.20)
                    assert all(np.array_equal(actual[w],old[w]) for w in WIRE)
                    checked+=3
                for n in CAMERAS:
                    k,e=geo[n]['intrinsic'],geo[n]['extrinsic']
                    e4=np.eye(4);e4[:3]=e[:3];pose=np.linalg.inv(e4)
                    yy,xx=np.indices((224,224),dtype=np.float64)
                    # Independent CV projection oracle; retain original half-pixel convention.
                    native_x=(xx+.5)/.7
                    native_y=(yy-28+.5)/.7
                    rays=np.stack([native_x,native_y,np.ones_like(xx)],-1)@np.linalg.inv(k).T
                    direction=rays@pose[:3,:3].T
                    valid=np.abs(direction[...,2])>1e-5
                    world=pose[:3,3]+direction*((.741-pose[2,3])/np.where(valid,direction[...,2],1))[...,None]
                    for c in CONDITIONS[:7]:
                        grid=rotated_grid(grid_for(k,e,center(c),.20,224,224,.5),c['yaw_deg'])
                        a=np.deg2rad(c['yaw_deg']);delta=world-np.asarray(center(c))
                        local_x=delta@np.array([np.cos(a),np.sin(a),0])
                        local_y=delta@np.array([-np.sin(a),np.cos(a),0])
                        expected=np.stack([local_x/.10,-local_y/.10],-1)
                        roi=valid&(np.abs(expected).max(-1)<2)
                        if roi.any():max_error=max(max_error,float(np.abs(grid[roi]-expected[roi]).max()))
    assert checked==54 and max_error<1e-4,(checked,max_error)
    dump(dest/'cpu_audit.json',dict(complete=True,nominal_camera_images_compared=checked,
        original_compositor_exact=True,world_yaw_ray_projection_max_UV_error=max_error,
        audit_scope='Saved9states: nominalRGB parity; independent ray/world-plane transform check. Live masks pending capture.'))


def old_episodes(method):
    name = {'clean':'clean','full':'digital_full_digital','random':'random_digital'}[method]
    j = json.loads((TRANSFER/name/'results.json').read_text())
    assert j['complete']
    return {r['seed']:r for r in j['episodes'] if 10200<=r['seed']<10210}


def records(dest, c, method):
    if c['name']=='nominal' or (method=='clean' and c['gain']==1):
        return old_episodes(method)
    path = dest/'eval'/c['name']/method
    return {r['seed']:r for f in sorted(path.glob('seed*.json')) for r in [json.loads(f.read_text())]}


def figures(dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    with np.load(dest/'inputs.npz') as z:
        images=z['images'][0]; checker=z['checker']; masks=z['masks'][0]
    fig, axes=plt.subplots(3,len(CONDITIONS),figsize=(24,8))
    for ci,c in enumerate(CONDITIONS):
        for mi,method in enumerate(METHODS):
            ax=axes[mi,ci]; ax.imshow(images[ci,mi,0]); ax.axis('off')
            if mi==0:ax.set_title(c['label'],fontproperties=font,fontsize=12)
            if ci==0:ax.text(-.08,.5,method,rotation=90,ha='right',va='center',transform=ax.transAxes)
    fig.suptitle('RoboTwin 数字贴图：同一状态／head相机／冻结20 cm纹理',fontproperties=font,fontsize=19)
    fig.tight_layout(); fig.savefig(dest/'input_grid.png',dpi=140); plt.close(fig)
    selected=[0,2,5,6]; fig,axes=plt.subplots(3,len(selected),figsize=(12,9))
    for col,ci in enumerate(selected):
        for vi in range(3):
            ax=axes[vi,col]; ax.imshow(checker[ci,vi]); ax.axis('off')
            if masks[ci,vi].any():ax.contour(masks[ci,vi],levels=[.5],colors=['white'],linewidths=.4)
            if vi==0:ax.set_title(CONDITIONS[ci]['label'],fontproperties=font)
            if col==0:ax.text(-.05,.5,['head','left wrist','right wrist'][vi],rotation=90,
                             va='center',ha='right',transform=ax.transAxes)
    fig.suptitle('几何核对：同一规则世界棋盘，内容和可见mask同步变换',fontproperties=font,fontsize=15)
    fig.tight_layout();fig.savefig(dest/'geometry_checker.png',dpi=160);plt.close(fig)


def report(dest):
    rows=[]; completed_groups=0
    baseline=old_episodes('full'); nominal_clean=old_episodes('clean')
    for c in CONDITIONS:
        rr={m:records(dest,c,m) for m in METHODS}
        ready=all(set(v)==set(range(10200,10210)) for v in rr.values())
        if ready:completed_groups+=1
        for method in ['full','random']:
            row=dict(condition=c['name'],label=c['label'],method=method,complete=ready,
                     available_episodes=len(rr[method]),clean_episodes=len(rr['clean']))
            if ready:
                eligible=[s for s,r in rr['clean'].items() if r['success']]
                failed=[s for s in eligible if not rr[method][s]['success']]
                common_clean=[s for s in eligible if nominal_clean[s]['success']]
                original_failures=[s for s in common_clean if not baseline[s]['success']]
                retained=[s for s in original_failures if not rr[method][s]['success']]
                row.update(clean_successes=len(eligible),attack_successes=len(failed),
                    ASR=len(failed)/len(eligible) if eligible else None,failed_seeds=sorted(failed),
                    task_successes=sum(r['success'] for r in rr[method].values()),
                    original_full_failures_on_common_clean=original_failures,
                    retained_original_full_failures=retained)
            rows.append(row)
    complete=completed_groups==len(CONDITIONS)
    data=dict(complete=complete,completed_condition_groups=completed_groups,total_condition_groups=len(CONDITIONS),
              updated_utc=datetime.now(timezone.utc).isoformat(),rows=rows)
    dump(dest/'summary.json',data)
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (dest/'asr.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    rel='../'+str(dest.relative_to(ROOT))
    status_file=dest/'status.json'
    stage=json.loads(status_file.read_text()).get('stage') if status_file.exists() else 'prepared_not_started'
    stage_label={'prepared_not_started':'已准备，尚未启动','waiting_for_gpu':'已排队，等待空闲GPU',
        'capturing_transformed_digital_inputs':'正在采集变换后的数字输入',
        'measuring_L0_on_fresh_noise':'正在测量L0','starting_owned_policy_server':'正在启动闭环策略服务',
        'closed_loop_evaluation':'正在进行闭环评测','failed':'运行中断，详见进度文件','complete':'全部完成'}.get(stage,stage)
    lines=['# 10　直接贴图的变换敏感性', '', '[总览](README.md) · [原数字／render对照](01_digital_to_render.md)', '',
        '**目的：冻结旧数字Full patch，检查位置、旋转和光度变化后是否仍能攻击。**', '',
        'RoboTwin `place_container_plate / demo_clean`，原20 cm Q128、第500步Full纹理；所有相机共享同一矩形。此轮不重新训练，也不把代理或EOT纹理混入比较。', '',
        '| 单独改变的因素 | 测试设置 |', '|---|---|',
        '| 位置 | 世界X/Y分别±1 cm，原朝向 |',
        '| 旋转 | 世界平面内绕贴纸中心±10°，原中心 |',
        '| 光度 | 整幅224×224模型输入sRGB乘0.85／1.15并裁剪；clean同步处理 |', '',
        '位置和旋转后重新计算透视与可见mask，纹理内容随矩形一起变换。实际仿真器只提供场景背景、相机和遮挡信息，模型看到的patch RGB始终来自数字合成。光度条件是输入亮度敏感性，不等于真实光源变化。所有变换在一集内固定。', '',
        '训练初态10100–10107；开环验证10110／10111共9状态；闭环复用10200–10209这10个训练外初态及相同动作noise。它们已有过评测，因此本轮是配对诊断，不是全新盲测。', '',
        '## 闭环ASR', '',
        f'状态：**{"全部完成" if complete else stage_label}**；完整条件组{completed_groups}/{len(CONDITIONS)}（包含复用的原位置组）。每个新变换均测Full与同一张Random，亮度变化另跑对应clean，共180条新闭环。', '',
        '| 条件 | 同条件clean成功／10 | Full ASR | Random ASR |', '|---|---:|---:|---:|']
    for c in CONDITIONS:
        a,b=[r for r in rows if r['condition']==c['name']]
        if a['complete']:
            lines.append(f"| {c['label']} | {a['clean_successes']}/10 | {a['attack_successes']}/{a['clean_successes']} | {b['attack_successes']}/{b['clean_successes']} |")
        else:
            lines.append(f"| {c['label']} | 待该组配对完成 | 待测完（{a['available_episodes']}/10） | 待测完（{b['available_episodes']}/10） |")
    lines += ['', 'ASR＝同条件clean能成功而加patch失败的比例。几何变换复用原clean；亮度条件的分母由相同增益下的新clean决定。各组跑齐后才发布ASR，不以部分结果选条件。原位置历史Full为3/8、Random为0/8。CSV还保存原有攻击失败初态在变换后的保留情况。', '',
              '## 同状态L0诊断', '']
    if (dest/'probe.json').exists():
        metrics=json.loads((dest/'probe.json').read_text())['rows']
        lines += ['9个状态来自两条验证轨迹，每状态使用三份训练外noise（20261021／22／23）。同一条件的Full、Random与clean保持状态和noise一致；L0是t=1全部32×32速度分量平方差之和，再对noise和状态平均。', '',
                  '| 条件 | Full L0均值 | Full中位数 | 相对原位置均值 | Random L0均值 |', '|---|---:|---:|---:|---:|']
        for c in CONDITIONS:
            a,b=[r for r in metrics if r['condition']==c['name']]
            lines.append(f"| {c['label']} | {a['L0_mean']:.5f} | {a['L0_median']:.5f} | {a['relative_to_nominal']:.2%} | {b['L0_mean']:.5f} |")
        lines += ['', '这些L0使用新的三份noise，与09对应的既有L0不混列。L0变化是开环诊断；是否仍有攻击效果以本页配对闭环ASR为准。']
    else:lines += ['尚未完成；不将L0诊断当成闭环ASR。']
    if (dest/'input_grid.png').exists():
        lines += ['', '## 实际数字输入与几何核对', '', f'![三种输入的同状态变换]({rel}/input_grid.png)', '',
                  f'![世界矩形的透视旋转核对]({rel}/geometry_checker.png)', '',
                  '图示来自验证seed10110/query0；几何图中的棋盘仅用于核对透视、旋转和mask，不参与攻击评测。']
    lines += ['', '## 如何判断', '',
        '若新初态上的原数字路径仍有效，而小变换后Full相对Random的优势和闭环ASR消失，支持对固定成像条件过度适配。若这些变换下仍有效而真实render失效，则应继续定位数字变换没有覆盖的过滤、光度或采样差异。单个方向或10初态的计数变化不足以概括所有任务；这轮也没有单独测试1 px图像平移。', '',
        f'数据：[冻结协议]({rel}/protocol.json) · [进度]({rel}/status.json) · [闭环汇总]({rel}/summary.json) · [CSV]({rel}/asr.csv)。']
    (ROOT/'result/10_digital_transform_sensitivity.md').write_text('\n'.join(lines)+'\n')
    return complete


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default=str(PROBE/'out/digital_transform_20261009'))
    ap.add_argument('--gpu',default='5');ap.add_argument('--port',type=int,default=8265)
    ap.add_argument('--report-only',action='store_true');ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args();dest=Path(args.root).resolve();prepare(dest)
    if args.report_only:
        if (dest/'inputs.npz').exists():figures(dest)
        report(dest);return
    def status(stage,**extra):
        dump(dest/'status.json',dict(stage=stage,pid=os.getpid(),gpu=args.gpu,port=args.port,
             updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
    if args.prepare_only:
        status('prepared_not_started');report(dest);return
    snap=dest/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in ['digital_transform_common.py','digital_transform_experiment.py','digital_transform_pipeline.py',
                 'render_rollout.py','render_geometry.py','render_proxy.py','scene_smoke.py','policy_server.py','render_model.py']:
        src=PROBE/name;target=snap/name
        if target.exists():assert sha(target)==sha(src),'source changed during resume: '+name
        else:target.write_bytes(src.read_bytes())
    dump(snap/'manifest.json',{p.name:sha(p) for p in snap.glob('*.py')})
    env=os.environ.copy();env.pop('PYTHONPATH',None)
    env.update(CUDA_VISIBLE_DEVICES=args.gpu,PYTHONNOUSERSITE='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    server=worker=None
    def interrupted(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
    signal.signal(signal.SIGTERM,interrupted)
    def run(cmd,logname):
        nonlocal worker
        with (dest/logname).open('a') as f:
            worker=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        code=worker.wait()
        if code:raise RuntimeError(f'{logname}: exit {code}')
    try:
        cpu_audit(dest)
        # Wait without claiming a GPU is idle while another training job uses it.
        status('waiting_for_gpu');report(dest)
        while True:
            raw=subprocess.check_output(['nvidia-smi','-i',args.gpu,
                '--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            free,util=map(int,raw.strip().split(','))
            if free>=26000 and util<=20:break
            status('waiting_for_gpu',free_memory_mib=free,utilization_percent=util)
            time.sleep(30)
        status('capturing_transformed_digital_inputs');report(dest)
        run(['bash',str(PROBE/'run_sim.sh'),str(EXPERIMENT),'--root',str(dest),'--stage','capture'],'capture.log')
        figures(dest);report(dest)
        status('measuring_L0_on_fresh_noise');report(dest)
        run([PY,'-u',str(EXPERIMENT),'--root',str(dest),'--stage','probe'],'probe.log')
        report(dest)
        with socket.socket() as s:s.bind(('127.0.0.1',args.port))
        meta=dest/'server.json'
        if meta.exists():meta.unlink()
        status('starting_owned_policy_server')
        with (dest/'server.log').open('a') as f:
            server=subprocess.Popen([PY,'-u',str(PROBE/'policy_server.py'),'--port',str(args.port),
                '--metadata',str(meta)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        dump(dest/'owned_server.json',dict(pid=server.pid,stopped=False))
        deadline=time.monotonic()+360
        while not meta.exists():
            if server.poll() is not None:raise RuntimeError('policy server exited')
            if time.monotonic()>deadline:raise TimeoutError('server startup')
            time.sleep(1)
        assert json.loads(meta.read_text())==json.loads((TRANSFER/'worker_full/server.json').read_text())
        for c in CONDITIONS[1:]:
            methods=['clean','full','random'] if c['gain']!=1 else ['full','random']
            for method in methods:
                status('closed_loop_evaluation',condition=c['name'],method=method)
                run(['bash',str(PROBE/'run_sim.sh'),str(EXPERIMENT),'--root',str(dest),'--stage','eval',
                    '--method',method,'--condition',c['name'],'--port',str(args.port)],
                    f"eval_{c['name']}_{method}.log")
                report(dest)
        assert report(dest)
        status('complete',complete=True)
    except BaseException as exc:
        status('failed',complete=False,error=repr(exc));report(dest);raise
    finally:
        for process in [worker,server]:
            if process is not None and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        if server is not None:dump(dest/'owned_server.json',dict(pid=server.pid,stopped=True,returncode=server.returncode))


if __name__=='__main__':main()
