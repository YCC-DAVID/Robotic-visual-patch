"""Auditable progress, paired ASR/SR tables, transfer matrices and texture CV."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
import checkpoint_study_common as S


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def interval(x):
    x=np.asarray(x,float)
    if len(x)==0:return None,None,None
    rng=np.random.default_rng(202610300)
    means=x[rng.integers(0,len(x),(10000,len(x)))].mean(1)
    return float(x.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))


def cv(root,out,textures):
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import rgb_to_hsv
    arrays={};spectra={};edges={};rows=[]
    bins=np.linspace(0,np.sqrt(.5),33)
    for item in textures:
        a=np.asarray(Image.open(item['path']).convert('RGB'),float)/255;arrays[item['id']]=a
        hsv=rgb_to_hsv(a);gray=a@np.array([.2126,.7152,.0722])
        centered=a-a.mean((0,1),keepdims=True)
        window=np.outer(np.hanning(a.shape[0]),np.hanning(a.shape[1]))
        power=(abs(np.fft.fft2(centered*window[...,None],axes=(0,1)))**2).sum(2)
        f=np.fft.fftfreq(a.shape[0]);r=np.hypot(f[:,None],f[None,:]);total=power.sum()
        radial=np.array([power[(r>=lo)&(r<hi)].sum() for lo,hi in zip(bins[:-1],bins[1:])])
        radial=radial/max(radial.sum(),1e-12);spectra[item['id']]=radial
        gy,gx=np.gradient(gray);edges[item['id']]=np.hypot(gx,gy)
        counts=np.bincount(np.rint(gray*255).astype(np.uint8).ravel(),minlength=256);pr=counts[counts>0]/counts.sum()
        rows.append(dict(id=item['id'],source=item.get('model'),patch_seed=item.get('patch_seed'),
            mean_R=float(a[...,0].mean()),mean_G=float(a[...,1].mean()),mean_B=float(a[...,2].mean()),
            saturation=float(hsv[...,1].mean()),gray_hist_entropy_bits=float(-(pr*np.log2(pr)).sum()),
            ac_power=float(total),low_frequency_power_fraction=float(power[(r>0)&(r<=.125)].sum()/max(total,1e-12)),
            high_frequency_power_fraction=float(power[r>=.25].sum()/max(total,1e-12)),
            edge_mean=float(edges[item['id']].mean())))
    pairs=[]
    def corr(a,b):
        a=a.ravel()-a.mean();b=b.ravel()-b.mean();d=np.linalg.norm(a)*np.linalg.norm(b)
        return float(np.dot(a,b)/d) if d>1e-12 else None
    for a,b in itertools.combinations(arrays,2):
        pairs.append(dict(left=a,right=b,pixel_corr=corr(arrays[a],arrays[b]),
            edge_corr=corr(edges[a],edges[b]),
            rgb_mean_distance=float(np.linalg.norm(arrays[a].mean((0,1))-arrays[b].mean((0,1)))),
            radial_cosine=float(np.dot(spectra[a],spectra[b])/max(np.linalg.norm(spectra[a])*np.linalg.norm(spectra[b]),1e-12))))
    write_csv(out/'texture_metrics.csv',rows);write_csv(out/'texture_pairs.csv',pairs)
    fig,axes=plt.subplots(4,4,figsize=(12,12))
    for ax in axes.flat:ax.axis('off')
    for ax,item in zip(axes.flat,textures):ax.imshow(arrays[item['id']]);ax.set_title(item['id'],fontsize=8)
    fig.tight_layout();fig.savefig(out/'textures.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for key,values in spectra.items():ax.plot((bins[:-1]+bins[1:])/2,values,label=key)
    ax.set(xlabel='Radial spatial frequency (cycles / texture pixel)',ylabel='Fraction of centered, Hann-windowed RGB power per bin')
    ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(out/'spectra.png',dpi=160);plt.close(fig)


def transfer_analysis(out,results,textures,models):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trained=[t for t in textures if t.get('model')]
    matrix=np.full((len(trained),len(models)),np.nan)
    matrix_rows=[];asymmetry=[]
    for i,t in enumerate(trained):
        row=dict(patch=t['id'],source=t['model'],patch_seed=t['patch_seed'])
        for j,m in enumerate(models):
            c=results.get((m['id'],'clean'));a=results.get((m['id'],t['id']))
            value=float((~a)[c].mean()) if c is not None and a is not None and c.any() else None
            row[m['id']]=value
            if value is not None:matrix[i,j]=value
        matrix_rows.append(row)
    write_csv(out/'ASR_cross_matrix.csv',matrix_rows)
    if trained:
        fig,ax=plt.subplots(figsize=(9,9));im=ax.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap='viridis')
        ax.set_xticks(range(len(models)),[m['id'] for m in models],rotation=30,ha='right')
        ax.set_yticks(range(len(trained)),[t['id'] for t in trained],fontsize=8)
        ax.set(xlabel='Target checkpoint',ylabel='Patch trained on source checkpoint',title='Actual-render paired ASR')
        for (i,j),v in np.ndenumerate(matrix):
            if np.isfinite(v):ax.text(j,i,f'{v:.0%}',ha='center',va='center',fontsize=8,color='white' if v<.5 else 'black')
        fig.colorbar(im,ax=ax,label='ASR');fig.tight_layout();fig.savefig(out/'ASR_cross_matrix.png',dpi=160);plt.close(fig)
    groups={'independent_seed':['M0_s20000','M1_s20000','M2_s20000'],
            'same_run_steps':['M0_s05000','M0_s10000','M0_s20000']}
    for group,ids in groups.items():
        for a,b in itertools.combinations(ids,2):
            if (a,'clean') not in results or (b,'clean') not in results:continue
            common=results[a,'clean']&results[b,'clean']
            pa=[t['id'] for t in trained if t['model']==a];pb=[t['id'] for t in trained if t['model']==b]
            if len(pa)!=3 or len(pb)!=3 or not all((b,t) in results for t in pa) or not all((a,t) in results for t in pb):continue
            forward=np.array([~results[b,t] for t in pa],float)[:,common]
            reverse=np.array([~results[a,t] for t in pb],float)[:,common]
            mean,lo,hi=interval((forward-reverse).mean(0))
            asymmetry.append(dict(group=group,source_a=a,source_b=b,paired_common_clean_N=int(common.sum()),
                a_to_b_ASR=float(forward.mean()) if common.any() else None,
                b_to_a_ASR=float(reverse.mean()) if common.any() else None,
                signed_ASR_difference=mean,CI_low=lo,CI_high=hi,
                CI_unit='initial-state bootstrap, three paired patch seeds held fixed'))
    write_csv(out/'transfer_asymmetry.csv',asymmetry)


def diagnostic_summary(root,out):
    rows=[]
    for path in sorted((root/'diagnostics').glob('*/responses.csv')):
        data=list(csv.DictReader(path.open()))
        for condition in sorted({r['condition'] for r in data}):
            subset=[r for r in data if r['condition']==condition]
            # Equal weight to initial-state seeds; queries/noises are repeated
            # measures rather than independent samples.
            row=dict(model=path.parent.name,condition=condition,measurements=len(subset),seeds=len({r['seed'] for r in subset}))
            for metric in ['L0_32','L0_14','action_MSE14','joint_RMSE_rad','gripper_RMSE']:
                per_seed=[np.mean([float(r[metric]) for r in subset if r['seed']==seed]) for seed in sorted({r['seed'] for r in subset})]
                mean,lo,hi=interval(per_seed)
                row.update({metric+'_mean':mean,metric+'_median':float(np.median(per_seed)),metric+'_CI_low':lo,metric+'_CI_high':hi})
            rows.append(row)
    write_csv(out/'diagnostic_summary.csv',rows)
    return rows


def main(args):
    root=args.root;out=root/'comparison';out.mkdir(parents=True,exist_ok=True)
    p=S.prepare_protocol(root);models=S.checkpoint_specs(root)
    lines=['# RoboTwin π0.5 多 checkpoint / 多 patch 受控对照','',
        '本轮把模型 seed、同一次训练的 checkpoint step、patch seed 分开；旧外部 RoboTwin checkpoint 不参与模型微调起点。','',
        '## 核查与可复用部分','',
        '本地已有官方 π0.5 base、RoboTwin 三相机推理/闭环代码、10cm真实渲染标定与EOT状态库；'
        '已有外部 RoboTwin 微调权重不满足本轮只改变seed的控制要求，因此不作为新模型起点。'
        '没有找到可直接用于受控微调的本地专家训练集，现按固定seed顺序采集并保存原始图像、关节状态和成功/失败记录。','',
        '训练复用本地 OpenPI JAX 代码和已验证的LoRA导出实现，保持三路Aloha输入及14维动作编码。'
        '启动核查时GPU3/5/6可用，均为46GiB L40S；其余GPU上的已有任务未停止。正式微调前再次检查显存占用。'
        '三步预检的稳定更新约2.4秒/步，20k更新约13–14小时/模型；三个模型并行，后续闭环另计。','',
        '## 固定设置','',
        '| 控制项 | 设置 |','|---|---|',
        '| 任务 / 数据 | place_container_plate / demo_clean；100 条成功专家轨迹训练，20 条独立验证 |',
        '| 起点 / 配方 | 官方 π0.5 base；冻结视觉、语言 LoRA rank16、完整 action expert；batch16、micro4、20k updates、EMA .99 |',
        '| 模型 seed | 202610230、202610231、202610232；最终 checkpoint 组成独立 seed 组 |',
        '| step 组 | 仅 seed202610230 的 5k / 10k / 20k；不当作三个独立模型训练 |',
        '| patch | 每 checkpoint 3 seed；10cm、128×128、灰色初始化、500×8、lr .01 |',
        '| render / EOT | 同一三相机世界纹理；固定代理；3cm 位移、±15°旋转、三种光照；无 flip |',
        '| 闭环 | 100 个预先固定、未按专家可行性筛选的初态；15×5 patch 交叉；每目标模型另测 clean / Random |','',
        '## 当前进度','', '| 阶段 | 已完成 |','|---|---|']
    status_path=root/'pipeline_status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text())
        labels={'failed':'已停止：异常','collecting_expert_data':'采集固定专家数据',
                'waiting_for_training_gpus':'等待训练GPU显存释放','matched_finetuning':'正式模型微调',
                'fingerprinting_inputs':'核对起点和代码指纹','complete':'全部完成'}
        lines.append(f'| 流水线状态 | {labels.get(status["stage"],status["stage"])} |')
        if status.get('error'):lines.append(f'| 停止原因 | {status["error"]} |')
        if status.get('occupied_MiB'):lines.append(f'| 等待中的GPU占用(MiB) | {status["occupied_MiB"]} |')
    recovery=root/'recovery_20261025/note.json'
    if recovery.exists():
        note=json.loads(recovery.read_text())
        lines.append(f'| 2026-10-25 恢复记录 | {note["summary"]} |')
    for split in ('train','validation'):
        path=root/'expert'/f'{split}_manifest.json'
        if path.exists():
            x=json.loads(path.read_text());lines.append(f'| {split} 专家数据 | {len(x["selected"])}/{x["target"]}；尝试 {len(x["attempted"])} 个初态 |')
    for name in ('smoke','M0','M1','M2'):
        path=root/'training'/name/'status.json'
        x=json.loads(path.read_text()) if path.exists() else dict(stage='未启动')
        lines.append(f'| {name} 微调 | {x.get("stage")}；step {x.get("step","—")} |')
    for label,name in [('专家qpos回放','expert_replay/audit.json'),('JAX / Torch 导出一致性','smoke_parity/parity.json'),
                       ('clean / Random 配对接口','smoke_interface.json')]:
        path=root/name
        if path.exists():lines.append(f'| {label} | {"通过" if json.loads(path.read_text()).get("passed") else "未通过"} |')
    lines+=['','smoke 只有3次模型更新、2次patch更新及32步接口执行，仅验证链路，不是本轮策略能力或攻击效果。']
    textures=json.loads((root/'texture_manifest.json').read_text()) if (root/'texture_manifest.json').exists() else []
    trained_textures=any(t.get('model') for t in textures)
    if trained_textures:cv(root,out,textures)
    conditions=['clean']+[t['id'] for t in textures]
    results={};missing=[]
    for m in models:
        for c in conditions:
            path=root/'closedloop'/m['id']/c/'results.json'
            if path.exists() and (d:=json.loads(path.read_text()))['complete']:
                assert [e['seed'] for e in d['episodes']]==p['evaluation']['seeds']
                results[m['id'],c]=np.array([e['success'] for e in d['episodes']],bool)
            else:missing.append((m['id'],c))
    rows=[];common=None
    if all((m['id'],'clean') in results for m in models):
        common=np.logical_and.reduce([results[m['id'],'clean'] for m in models])
    for (m,c),success in results.items():
        if (m,'clean') not in results:continue
        clean=results[m,'clean'];eligible=clean;sr,lo,hi=interval(success)
        asr,al,ah=interval((~success)[eligible]) if c!='clean' else (None,None,None)
        excess=interval(((~success).astype(int)-(~results[m,'Random']).astype(int))[eligible]) if (m,'Random') in results and c!='clean' else (None,None,None)
        rows.append(dict(target=m,condition=c,N=len(success),clean_successes=int(clean.sum()),
            task_successes=int(success.sum()),task_SR=sr,SR_CI_low=lo,SR_CI_high=hi,
            attack_successes=int((clean&~success).sum()) if c!='clean' else None,ASR=asr,ASR_CI_low=al,ASR_CI_high=ah,
            excess_ASR_vs_Random=excess[0],excess_CI_low=excess[1],excess_CI_high=excess[2],
            common_clean_N=int(common.sum()) if common is not None else None,
            common_clean_ASR=float((~success)[common].mean()) if c!='clean' and common is not None and common.any() else None))
    write_csv(out/'metrics.csv',rows)
    if textures:transfer_analysis(out,results,textures,models)
    diagnostic_rows=diagnostic_summary(root,out)
    S.dump(out/'coverage.json',dict(complete=len(textures)==16 and not missing,
        completed_groups=len(results),missing=missing,expected_groups=85,
        common_clean_N=int(common.sum()) if common is not None else None))
    lines+=['',f'闭环已完成 {len(results)}/85 组；每组目标 100 个初态。尚未完成的组不填推测的 ASR。','',
        '## 指标口径','',
        '- ASR = clean 成功且 patch 失败的配对数 / 该目标模型 clean 成功数。',
        '- 整体 SR = 任务成功数 / 100；clean 本来失败的初态仍在分母中，因此 SR 不等于 1−ASR。',
        '- 另报告所有目标模型共同 clean 成功初态上的 ASR、相对 Random 的配对 ASR 增量及初态 bootstrap 95% 区间。',
        '- L0 是同一状态、图像几何、动作噪声下的首步速度平方差，分别统计完整32维和有效14维；不是任务失败率。',
        '- 完整动作差经过全部10步 flow；同时报告有效14维归一化 MSE、12个关节的弧度 RMSE 和2个夹爪的 RMSE。','',
        '## 解释边界','',
        '主交叉矩阵固定 patch 位置与光照，用于隔离 checkpoint 迁移；本轮所有 patch 的训练 EOT 完全相同。'
        '代理使用已有、与本轮模型训练和闭环测试初态分离的固定图像状态库。它来自旧策略轨迹，可能限制新策略轨迹的覆盖，但所有 checkpoint 受到相同限制。','',
        '同色、同频谱或同向 Δv 都不能单独证明找到了共同攻击结构。纹理指标与真实渲染诊断用于定位候选；确认结构因果作用还需要替换/保留区域对照与闭环复测。',
        'Random 为同一固定随机纹理，控制遮挡和未优化噪声；不能把一张 Random 外推成所有随机纹理。','',
        f'原始目录：`{root}`。配置：`protocol.json`；训练：`training/*/config.json`；诊断：`diagnostics/*/responses.csv`。']
    if rows:
        lines+=['','## 当前闭环结果','','| 目标 | 条件 | SR | ASR | clean 成功数 |','|---|---|---:|---:|---:|']
        for r in rows:
            a='—' if r['ASR'] is None else f'{r["ASR"]:.1%}'
            lines.append(f'| {r["target"]} | {r["condition"]} | {r["task_SR"]:.1%} | {a} | {r["clean_successes"]} |')
    if any(t.get('model') for t in textures):lines+=['','![交叉矩阵](ASR_cross_matrix.png)']
    if trained_textures:lines+=['','![纹理](textures.png)','','![频谱](spectra.png)']
    if diagnostic_rows:
        lines+=['','## 真实渲染固定状态诊断','',
            'L0 与完整动作差先在每个初态内平均，再对初态等权；完整数值与区间见 comparison/diagnostic_summary.csv。','',
            '| 目标 | 纹理 | L0有效14维 | 完整动作MSE14 | 关节RMSE(rad) |','|---|---|---:|---:|---:|']
        for r in diagnostic_rows:
            lines.append(f'| {r["model"]} | {r["condition"]} | {r["L0_14_mean"]:.6g} | {r["action_MSE14_mean"]:.6g} | {r["joint_RMSE_rad_mean"]:.6g} |')
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    report=S.ROOT/'result/28_robotwin_checkpoint_study.md'
    relative='../robotwinprobe/out/'+root.name+'/comparison/'
    body='\n'.join(lines)+'\n'
    for name in ['textures.png','spectra.png','ASR_cross_matrix.png']:body=body.replace(f'({name})',f'({relative}{name})')
    report.write_text(body)
    print('[report]',report,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=S.DEFAULT);main(ap.parse_args())
