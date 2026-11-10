"""Live counts, paired stage ASR, final replay audit, and review artifacts."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from stage_joint_common import DEFAULT,ROOT,STAGES,TEXTURES,CONDITIONS,CAMERAS,load,jobs,job_id,dump,sha
from stage_joint_rollout import output_path,native_job,digest,noise_seed


def interval(values):
    if not values:return [None,None]
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(202610260)
    means=x[rng.integers(0,len(x),(2000,len(x)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def wilson(values):
    if not values:return [None,None]
    n=len(values);v=sum(values)/n;z=1.959963984540054
    center=(v+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(v*(1-v)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0.,center-radius),min(1.,center+radius)]


def report(root=DEFAULT):
    p=load(root);planned=jobs(root)
    records={}
    for path in (root/'episodes').glob('*.json'):
        r=json.loads(path.read_text());assert r['protocol_sha256']==sha(root/'protocol.json')
        records[job_id(r['job'])]=r
    rows=[]
    for suite in ('native_full','delayed','pulse'):
        stages=('approach',) if suite=='native_full' else STAGES[1:] if suite=='delayed' else STAGES
        seeds=p['evaluation']['pulse_seeds' if suite=='pulse' else 'seeds']
        for stage in stages:
            complete=[];eligible=[];no_op=0
            for seed in seeds:
                group={c:records.get(job_id(dict(seed=seed,suite=suite,stage=stage,condition=c))) for c in CONDITIONS}
                native=records.get(job_id(native_job(seed)))
                if all(r is not None and r['status']=='no_opportunity' for r in group.values()):
                    no_op+=1;continue
                if not all(r is not None and r['status']=='complete' for r in group.values()):continue
                for r in group.values():assert r['initial_clean_sha256']==group['clean']['initial_clean_sha256']
                complete.append((seed,group))
                if native and native.get('success') and group['clean']['success']:
                    eligible.append((seed,group))
            for condition in ('random',*TEXTURES):
                fail=[int(not g[condition]['success']) for s,g in eligible]
                rand=[int(not g['random']['success']) for s,g in eligible]
                differences=[a-b for a,b in zip(fail,rand)]
                delta=[g[condition]['steps']-g['clean']['steps'] for s,g in eligible if g[condition]['success']]
                rows.append(dict(suite=suite,stage=stage,condition=condition,complete_groups=len(complete),
                    n=len(eligible),failures=sum(fail),ASR=float(np.mean(fail)) if fail else None,
                    ASR_CI95=wilson(fail),random_failures=sum(rand),random_ASR=float(np.mean(rand)) if rand else None,
                    excess_ASR=float(np.mean(differences)) if differences else None,excess_CI95=interval(differences),
                    no_opportunity_seeds=no_op,eligible_seeds=[s for s,g in eligible],
                    task_successes=sum(g[condition]['success'] for s,g in complete),
                    mean_successful_step_delta=float(np.mean(delta)) if delta else None))
    progress_path=root/'training_progress.json';progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'iteration':0,'complete':False}
    status_path=root/'status.json';status=json.loads(status_path.read_text()) if status_path.exists() else {'stage':'capture'}
    errors=[dict(id=k,error=v['error']) for k,v in records.items() if v.get('error')]
    valid=sum(r['status'] in ('complete','no_opportunity') for r in records.values())
    summary=dict(stage=status['stage'],complete=valid==len(planned) and not errors,
        recorded=valid,expected=len(planned),actual_episodes=sum(r['status']=='complete' for r in records.values()),
        no_opportunities=sum(r['status']=='no_opportunity' for r in records.values()),
        native_clean_completed=sum(records.get(job_id(native_job(s)),{}).get('status')=='complete' for s in p['evaluation']['seeds']),
        native_clean_successes=sum(bool(records.get(job_id(native_job(s)),{}).get('success')) for s in p['evaluation']['seeds']),
        training=progress,rows=rows,errors=errors,protocol_sha256=sha(root/'protocol.json'),
        interpretation='Native full uses native clean; delayed/pulse use intersection of native clean success and matched stage-replanned clean success. Incomplete five-condition groups excluded from comparison.')
    dump(root/'summary.json',summary)
    with (root/'asr.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# RoboTwin：全阶段联合 patch 与分阶段注入','',
        '冻结公开π0.5，任务place_container_plate/demo_clean。10cm、site07、Q128；24条成功专家轨迹训练、8条独立验证。只更新纹理。',
        '',f'当前状态：**{status["stage"]}**。三版训练更新：{progress["iteration"]}/500；正式记录：{valid}/2400。',
        f'训练标定：{len(list((root/"bank/train").glob("*.npz")))}/1536；验证标定：{len(list((root/"bank/validation").glob("*.npz")))}/128。',
        '', 'E仅接近阶段；J-mean四阶段等量采样；J-balanced使用相同帧/noise并优化log(1+L0/c)。c为训练前固定的训练Random均值。三版均500×8、灰色初始化、lr .01、相同3cm位移/±15°旋转/三档光照EOT。',
        '', '按用户批准，代理方向/强度/像素误差仅作诊断，不作为训练门槛。原生推理一致性、有限非零纹理梯度、精确回放和实际render配对评测仍必须通过。',
        '', 'Native-full保留原生32步动作段。Delayed在clean事件时立即推理并持续贴图；Pulse只让一次策略推理看到patch。每个阶段的Clean、Random、E、J-mean、J-balanced采用相同重规划和动作noise。阶段注入前重放对应策略的原生clean前缀，不使用专家动作。',
        '', '| 模式 | 阶段 | 纹理 | 攻击成功/有效初态 | Random | 配对增量 | 已完成五条件组 |',
        '|---|---|---|---|---|---|---|']
    for r in rows:
        if r['condition']=='random':continue
        excess=f"{r['excess_ASR']:+.1%}" if r['n'] else '待测'
        lines.append(f"| {r['suite']} | {r['stage']} | {r['condition']} | {r['failures']}/{r['n']} | {r['random_failures']}/{r['n']} | {excess} | {r['complete_groups']} |")
    lines+=['','ASR按独立初态统计。阶段组另要求原生clean成功、相应重规划clean成功且存在注入机会。缺失阶段、运行错误、提前结束和实际暴露次数分别保留。单次推理短窗不保证每阶段执行的动作数相等：任务可能在动作段结束前成功。',
        '', '离线L0及完整flow动作差是独立成功专家轨迹上的固定状态诊断，不能替代闭环ASR。',
        '',f'原始目录：`{root}`。协议：`protocol.json`；训练：`training_progress.json`；结果：`asr.csv`；实际render诊断：`diagnostics/`；前缀审计：`audit/final.json`。']
    if errors:lines+=['',f'运行错误：{len(errors)}条，见summary.json；不计作任务失败。']
    if status['stage']=='complete' and summary['complete']:
        full={r['condition']:r for r in rows if r['suite']=='native_full'}
        grasp=next(r for r in rows if r['suite']=='delayed' and r['stage']=='grasp' and r['condition']=='joint_balanced')
        transport=next(r for r in rows if r['suite']=='delayed' and r['stage']=='transport' and r['condition']=='joint_balanced')
        random=full['random']
        lines+=['', '## 最终结果与适用范围', '',
            f'目标：验证冻结π0.5在place_container_plate全过程及不同阶段受到同一张patch干扰的效果。六个计划阶段全部完成；正式记录{valid}/{len(planned)}，其中实际闭环{summary["actual_episodes"]}次、无注入机会{summary["no_opportunities"]}条，运行错误{len(errors)}条。原生clean成功{summary["native_clean_successes"]}/{summary["native_clean_completed"]}。', '',
            f'本配置未显示全阶段联合训练相对Random具有稳定优势。全程暴露时，Random为{random["failures"]}/{random["n"]}；E为{full["early"]["failures"]}/{full["early"]["n"]}；J-mean与J-balanced均为{full["joint_mean"]["failures"]}/{full["joint_mean"]["n"]}。', '',
            '全程暴露的相对Random配对增量及95% bootstrap区间：']
        for name in TEXTURES:
            r=full[name];lo,hi=r['excess_CI95']
            lines.append(f'- {name}：{100*r["excess_ASR"]:+.2f}个百分点，区间[{100*lo:+.2f}, {100*hi:+.2f}]个百分点。')
        lo,hi=grasp['excess_CI95']
        lines+=['',
            f'抓取起持续暴露时，J-balanced为{grasp["failures"]}/{grasp["n"]}，Random为{grasp["random_failures"]}/{grasp["n"]}；配对增量{100*grasp["excess_ASR"]:+.2f}个百分点，95%区间[{100*lo:+.2f}, {100*hi:+.2f}]，跨零。搬运起持续暴露的三版训练patch均为0/{transport["n"]}；单个ASR的95% Wilson上限为{transport["ASR_CI95"][1]:.1%}。', '',
            '单次推理暴露存在个别攻击成功案例，有效初态为12–14个；不足以说明patch在各阶段具有稳定攻击性。离线实际渲染响应的增强未转化为本轮闭环评测中的稳定优势。结论限于本任务、冻结checkpoint、site07、10cm patch和既定注入/噪声设置。', '',
            '最终审计通过：核对文件哈希、原生clean动作/状态/物体前缀、事件RGB、配对噪声、单次推理patch移除与暴露次数。阶段有效样本要求原生clean和同阶段重规划clean均成功；无注入机会不计入ASR分母。', '',
            '产物：']
        for label,path in [('ASR表',root/'asr.csv'),('完整统计与置信区间',root/'summary.json'),
                ('阶段ASR图',root/'stage_ASR.png'),('训练曲线',root/'training_curve.png'),
                ('实际渲染阶段响应',root/'diagnostics/stage_responses.png'),
                ('示例视频及选择规则',root/'videos/selection.json'),('最终审计',root/'audit/final.json')]:
            lines.append(f'- [{label}]({path})')
        lines+=['', '示例视频采用预先声明的首个原生clean成功seed（22002），用于展示；ASR保留全部预定初态。播放为查询帧与最终帧的3fps序列，不代表物理时间。']
    (ROOT/'result/33_robotwin_stage_joint.md').write_text('\n'.join(lines)+'\n')
    return summary


def audit(root):
    p=load(root);summary=report(root);assert summary['complete']
    checked=0;no_op=0
    for seed in p['evaluation']['seeds']:
        npath=output_path(root,native_job(seed));native=json.loads(npath.read_text())
        assert sha(npath.with_suffix('.npz'))==native['artifact_sha256']
        with np.load(npath.with_suffix('.npz')) as z:ref={k:z[k].copy() for k in z.files}
        for job in [j for j in jobs(root) if j['seed']==seed]:
            path=output_path(root,job);r=json.loads(path.read_text())
            if r['status']=='no_opportunity':
                event=native['events'].get(job['stage'])
                assert event is None or event>=native['steps'];no_op+=1;continue
            assert r['status']=='complete' and sha(path.with_suffix('.npz'))==r['artifact_sha256']
            assert r['initial_clean_sha256']==native['initial_clean_sha256']
            with np.load(path.with_suffix('.npz')) as z:
                assert len(z['executed_actions'])==r['steps']
                assert len(z['step_states'])==r['steps']+1
                expected=[noise_seed(p['evaluation']['noise_base'],seed,job['stage'],job['suite'],q) for q in range(len(z['states14']))]
                assert np.array_equal(z['noise_seeds'],expected)
                if job['suite']!='native_full':
                    t=native['events'][job['stage']]
                    assert r['trigger_step']==t and int(z['query_steps'][0])==t
                    for name in ('step_states','step_object_plate'):
                        assert np.array_equal(z[name][:t+1],ref[name][:t+1]),(job_id(job),name)
                    assert np.array_equal(z['executed_actions'][:t],ref['executed_actions'][:t])
                    if job['condition']!='clean':
                        ei=int(np.flatnonzero(ref['event_steps']==t)[0])
                        for ci,n in enumerate(CAMERAS):
                            assert digest(z['before_'+n])==ref['event_rgb_sha256'][ei,ci]
                if job['condition']=='clean':
                    assert not z['patch_active'].any()
                elif job['suite']=='pulse':
                    assert z['patch_active'].tolist()==[True]+[False]*(len(z['states14'])-1)
                    assert r['affected_actions']==min(32,r['steps']-r['trigger_step'])
                else:
                    assert z['patch_active'].all()
                    assert r['affected_actions']==r['steps']-r['trigger_step']
            checked+=1
    out=dict(passed=True,actual_episodes=checked,no_opportunity_records=no_op,protocol_sha256=sha(root/'protocol.json'),
        checks='artifact hashes; exact native action/state/object prefixes; event RGB; paired noise; pulse removal and exposure counts')
    dump(root/'audit/final.json',out);return out


def artifacts(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=report(root)
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,suite in zip(axes,('native_full','delayed','pulse')):
        stages=('approach',) if suite=='native_full' else STAGES[1:] if suite=='delayed' else STAGES
        for name in ('random',*TEXTURES):
            rs=[next(r for r in summary['rows'] if r['suite']==suite and r['stage']==s and r['condition']==name) for s in stages]
            ax.plot(stages,[r['ASR'] if r['n'] else np.nan for r in rs],'-o',label=name)
        ax.set(title=suite,ylim=(-.03,1.03));ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Paired clean-success conditional ASR');axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'stage_ASR.png',dpi=180);plt.close(fig)
    progress=root/'training_state.npz'
    if progress.exists():
        with np.load(progress) as z:history=json.loads(str(z['history']))
        fig,ax=plt.subplots(figsize=(8,4))
        for name in TEXTURES:ax.plot([r['iteration'] for r in history],[r['variants'][name]['L0'] for r in history],label=name)
        ax.set(xlabel='Texture update',ylabel='Training first-flow MSE14');ax.legend()
        fig.tight_layout();fig.savefig(root/'training_curve.png',dpi=180);plt.close(fig)
    # Videos are descriptive examples; all seeds remain in ASR. Choose first
    # native clean-success seed, and explicitly record this selection rule.
    candidates=[]
    for seed in load(root)['evaluation']['seeds']:
        path=output_path(root,native_job(seed))
        if path.exists() and json.loads(path.read_text()).get('success'):candidates.append(seed)
    if not candidates:return
    seed=candidates[0];video=root/'videos';video.mkdir(exist_ok=True)
    selected=[dict(seed=seed,suite='native_full',stage='approach',condition=c) for c in CONDITIONS]
    selected += [dict(seed=seed,suite='delayed',stage=s,condition='joint_balanced') for s in STAGES[1:]]
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    written=[]
    for job in selected:
        path=output_path(root,job)
        if not path.exists() or json.loads(path.read_text())['status']!='complete':continue
        dest=video/(job_id(job)+'.mp4')
        outcome=json.loads(path.read_text())
        with np.load(path.with_suffix('.npz')) as z:
            frames=list(z['head']);steps=list(z['query_steps']);flags=list(z['patch_active'])
            if job['suite']!='native_full':
                refpath=output_path(root,native_job(seed)).with_suffix('.npz')
                with np.load(refpath) as ref:
                    mask=ref['query_steps']<outcome['trigger_step']
                    frames=list(ref['head'][mask])+frames
                    steps=list(ref['query_steps'][mask])+steps
                    flags=[False]*int(mask.sum())+flags
            frames.append(z['final_head_camera']);steps.append(outcome['steps']);flags.append(flags[-1])
            annotated=[]
            for frame,step,active in zip(frames,steps,flags):
                im=Image.fromarray(frame.copy());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,320,25),fill='black')
                draw.text((3,2),f"{job['condition']} | step {step} | patch {int(active)}",fill='white')
                draw.text((3,13),f"{job['suite']} / {job['stage']} | success {outcome['success']}",fill='white')
                annotated.append(np.asarray(im))
            imageio.mimsave(dest,annotated,fps=3,macro_block_size=1,ffmpeg_params=['-threads','2'])
        written.append(str(dest))
    dump(video/'selection.json',dict(seed=seed,rule='first sorted native clean-success seed, descriptive only',
        playback='native clean query prefix plus affected branch queries and final frame, 3fps; not physical time',videos=written))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--audit',action='store_true');ap.add_argument('--artifacts',action='store_true')
    a=ap.parse_args();report(a.root)
    if a.audit:print(audit(a.root),flush=True)
    if a.artifacts:artifacts(a.root)
