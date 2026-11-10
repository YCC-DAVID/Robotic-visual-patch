"""CPU report for backbone-only random/digital/render attention and CAM."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'clean':'Clean','optimized_digital':'Optimized / digital',
          'optimized_render':'Optimized / rendered base','random_digital':'Random / digital',
          'random_render':'Random / rendered base'}
COLORS = ['#333333','#bb3135','#207bad','#bf8434','#6b9b62']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',default='backbone_patch_cam')
    args = ap.parse_args()
    dest = ROOT/'pi05probe/out'/args.out
    data = json.loads((dest/'metrics.json').read_text())
    assert data['complete']
    maps = np.load(dest/'maps.npz',allow_pickle=False)
    pool = np.load(data['inputs_source'],allow_pickle=False)
    names = maps['conditions'].tolist()
    steps = maps['steps'].tolist()
    nframes = len(steps)
    old = json.loads((Path(data['inputs_source']).parent/'metrics.json').read_text())
    assert old['complete'] and old['inputs_sha256']==data['inputs_sha256']
    old_names = {'optimized_digital':'digital','optimized_render':'render_base',
                 'random_digital':'random_digital','random_render':'random_render_base'}
    lookup = {(r['step'],r['patch']):r for r in old['records']}
    summary = {'conditions':{},'frames':nframes,'mean_patch_area_fraction':float(maps['mask'].mean()),
               'checks':dict(data['checks']),'scale_note':'One shared color scale per map type across selected states and conditions.'}
    checks = []

    def mean_present(xs):
        values = [v for v in xs if v is not None]
        return float(np.mean(values)) if values else None

    def fraction(h,coverage):
        den = float(np.sum(h))
        return float(np.sum(h*coverage)/den) if den>1e-20 else None

    for ci,name in enumerate(names):
        rows = [r for r in data['records'] if r['condition']==name]
        layer = [r['layers'][-1] for r in rows]
        item = {'V_direction_score_mean':float(np.mean([r['V_direction_score'] for r in rows])),
                'feature_base_relative_norm_mean':float(np.mean([r['feature_base_relative_norm'] for r in rows])),
                'feature_delta_cosine_to_digital_mean':mean_present([r['feature_delta_cosine_to_digital'] for r in rows]),
                'last_layer_K_relative_norm':float(np.mean([r['kv']['K']['relative_norm_all_valid'] for r in layer])),
                'last_layer_V_relative_norm':float(np.mean([r['kv']['V']['relative_norm_all_valid'] for r in layer])),
                'last_layer_noun_attention_base_mass':float(np.mean([r['noun_attention_base_mass'] for r in layer])),
                'last_layer_noun_attention_patch_mass':float(np.mean([r['noun_attention_patch_mass'] for r in layer])),
                'noun_attention_patch_fraction_within_base_mean':mean_present(
                    [r['noun_attention_patch_fraction_within_base'] for r in layer]),
                'V_delta_patch_fraction_within_base_mean':mean_present(
                    [r['V_delta_patch_fraction_within_base'] for r in layer]),
                'CAM_patch_fraction_within_base_mean':mean_present([r['CAM_patch_fraction_within_base'] for r in rows]),
                'CAM_positive_base_sum_mean':float(np.mean([r['CAM_positive_base_sum'] for r in rows])),
                'CAM_nonzero_base_frames':sum(r['CAM_patch_fraction_within_base'] is not None for r in rows),
                'pixel_gradient_patch_fraction_within_base_mean':mean_present(
                    [r['pixel_gradient_patch_fraction_within_base'] for r in rows]),
                'noun_attention_patch_enrichment_mean':float(np.mean([
                    r['layers'][-1]['noun_attention_patch_fraction_within_base']/r['mask_area_fraction'] for r in rows])),
                'CAM_patch_enrichment_mean':mean_present([
                    r['CAM_patch_fraction_within_base']/r['mask_area_fraction']
                    if r['CAM_patch_fraction_within_base'] is not None else None for r in rows]),
                'noun_patch_fraction_by_word':{}}
        for ni,noun in enumerate(['bowl','plate']):
            fractions = []
            for fi in range(nframes):
                coverage = maps['mask'][fi].reshape(16,14,16,14).mean((1,3))
                fractions.append(fraction(maps['attention_nouns'][fi,ci,-1,ni,0],coverage))
            item['noun_patch_fraction_by_word'][noun] = mean_present(fractions)
        item['L0_from_E15_mean'] = 0.0
        item['L0_exec7_from_E15_mean'] = 0.0
        if name!='clean':
            values = [lookup[(int(s),'drift')]['conditions'][old_names[name]] for s in steps]
            item['L0_from_E15_mean'] = float(np.mean([v['L0'] for row in values for v in row['velocity']]))
            item['L0_exec7_from_E15_mean'] = float(np.mean([v['L0_exec7'] for row in values for v in row['velocity']]))
            for row,ref in zip(rows,values):
                for kind in ['K','V']:
                    now = row['layers'][-1]['kv'][kind]['relative_norm_all_valid']
                    prior = ref['kv_vs_clean'][-1]['groups']['all_valid'][kind]['relative_norm']
                    checks.append(abs(now-prior)<1e-6)
        summary['conditions'][name] = item
        middle = [r['layers'][9] for r in rows]
        item['layer10_noun_attention_patch_fraction_within_base_mean'] = mean_present(
            [r['noun_attention_patch_fraction_within_base'] for r in middle])
    assert checks and all(checks),'Backbone KV differs from E15 controls'
    summary['checks']['last_layer_KV_matches_E15_all_patched_states'] = True

    base_images = np.stack([pool['clean_base'][:nframes],pool['drift__digital'][:nframes],
                           pool['drift__optimized_render_base'][:nframes],pool['drift__random_digital'][:nframes],
                           pool['drift__random_render_base'][:nframes]],axis=1)
    wrist_images = np.repeat(pool['clean_wrist'][:nframes,None],len(names),axis=1)
    attention = maps['attention_nouns'][:,:,-1].mean(2)
    rollout = maps['attention_rollout'][:,:,-1]
    kv_energy = maps['kv_delta_V_energy'][:,:,-1]
    kv_density = kv_energy/np.maximum(kv_energy.sum((-1,-2),keepdims=True),1e-20)
    cam = maps['gradcam_positive']
    selected = [steps.index(s) for s in [10,40,60,85] if s in steps]
    if not selected:
        selected = list(range(min(4,nframes)))
    scales = {}

    def montage(values,view,filename,title,difference=False):
        shown = values[selected,:,view]
        scale = float(np.percentile(np.abs(shown),99.5))
        if scale<=0:
            scale = 1.0
        scales[filename] = {'vmax':scale,'percentile':99.5,'view':view,'steps':[steps[i] for i in selected]}
        images = base_images if view==0 else wrist_images
        fig,axes = plt.subplots(len(names),len(selected),figsize=(3.2*len(selected),3.05*len(names)),squeeze=False)
        for ci,name in enumerate(names):
            for j,fi in enumerate(selected):
                ax = axes[ci,j]
                ax.imshow(images[fi,ci])
                hm = values[fi,ci,view]
                overlay = np.repeat(np.repeat(hm,14,axis=0),14,axis=1)
                heat = ax.imshow(overlay,cmap='coolwarm' if difference else 'inferno',
                                 vmin=-scale if difference else 0,vmax=scale,alpha=.55,interpolation='nearest')
                if view==0:
                    ax.contour(maps['mask'][fi].astype(float),levels=[.5],colors=['#29d5dd'],linewidths=.8)
                ax.set_title(LABELS[name]+' / step '+str(steps[fi]),fontsize=9)
                ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(title+'\nShared scale; cyan = patch pixel mask; 16x16 token maps',fontsize=12)
        fig.subplots_adjust(left=.02,right=.93,bottom=.02,top=.94,hspace=.12,wspace=.05)
        cax=fig.add_axes([.945,.15,.012,.65]);fig.colorbar(heat,cax=cax)
        fig.savefig(dest/filename,dpi=130,bbox_inches='tight');plt.close(fig)

    montage(attention,0,'fig_attention_base.png','Backbone last-layer task nouns (bowl + plate) -> base attention')
    montage(attention-attention[:,0:1],0,'fig_attention_change.png','Task-noun attention change relative to same-state clean',True)
    montage(rollout,0,'fig_attention_rollout.png','Residual attention rollout from task nouns -> base tokens')
    montage(cam,0,'fig_cam_base.png','Grad-CAM of digital-full V-direction score / base')
    montage(cam,1,'fig_cam_wrist.png','Grad-CAM of the same V-direction score / unchanged wrist input')
    montage(kv_energy,0,'fig_kv_energy.png','Last-layer V delta energy relative to clean / base token positions')
    montage(kv_density,0,'fig_kv_spatial_density.png','Within-base normalized V delta distribution / location, not magnitude')
    depth_frames = [steps.index(s) for s in [10,60] if s in steps] or [0]
    depth_pairs = [(fi,li) for fi in depth_frames for li in [9,17]]
    depth = maps['attention_nouns'][:,:,:,:,0].mean(3)
    scale = float(np.percentile(np.stack([depth[fi,:,li] for fi,li in depth_pairs]),99.5))
    fig,axes = plt.subplots(len(names),len(depth_pairs),figsize=(3.2*len(depth_pairs),3.05*len(names)),squeeze=False)
    for ci,name in enumerate(names):
        for j,(fi,li) in enumerate(depth_pairs):
            ax=axes[ci,j];ax.imshow(base_images[fi,ci])
            overlay=np.repeat(np.repeat(depth[fi,ci,li],14,0),14,1)
            heat=ax.imshow(overlay,cmap='inferno',vmin=0,vmax=scale,alpha=.55,interpolation='nearest')
            ax.contour(maps['mask'][fi].astype(float),levels=[.5],colors=['#29d5dd'],linewidths=.8)
            ax.set_title(LABELS[name]+' / step {} / layer {}'.format(steps[fi],li+1),fontsize=8)
            ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle('Task-noun attention at middle and final prefix layers\nShared scale; layer 10 selected descriptively after viewing layer curves')
    fig.subplots_adjust(left=.02,right=.93,bottom=.02,top=.94,hspace=.12,wspace=.05)
    cax=fig.add_axes([.945,.15,.012,.65]);fig.colorbar(heat,cax=cax)
    fig.savefig(dest/'fig_attention_depth.png',dpi=130,bbox_inches='tight');plt.close(fig)
    scales['fig_attention_depth.png']={'vmax':scale,'percentile':99.5,'pairs':[(steps[fi],li+1) for fi,li in depth_pairs]}
    fig,axes = plt.subplots(2,2,figsize=(13,9))
    for ci,name in enumerate(names):
        rows=[r for r in data['records'] if r['condition']==name]
        noun_patch=np.mean([[l['noun_attention_patch_fraction_within_base'] for l in r['layers']] for r in rows],axis=0)
        base_mass=np.mean([[l['noun_attention_base_mass'] for l in r['layers']] for r in rows],axis=0)
        v_norm=np.mean([[l['kv']['V']['relative_norm_all_valid'] for l in r['layers']] for r in rows],axis=0)
        axes[0,0].plot(np.arange(len(noun_patch))+1,noun_patch,label=LABELS[name],color=COLORS[ci])
        axes[0,1].plot(np.arange(len(base_mass))+1,base_mass,label=LABELS[name],color=COLORS[ci])
        axes[1,0].plot(np.arange(len(v_norm))+1,v_norm,label=LABELS[name],color=COLORS[ci])
        values=[r['CAM_patch_fraction_within_base'] for r in rows]
        axes[1,1].plot(steps,[np.nan if v is None else v for v in values],label=LABELS[name],color=COLORS[ci])
    titles=['Task-noun attention patch fraction within base','Task-noun total attention mass to base',
            'V relative change vs clean / all valid tokens','Positive Grad-CAM patch fraction within base']
    axes[0,0].axhline(summary['mean_patch_area_fraction'],color='grey',linestyle=':',label='Pixel area fraction')
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.legend(fontsize=7)
        ax.set_xlabel('Environment step' if ax is axes[1,1] else 'Prefix layer (1-based)')
    fig.suptitle('Same-state backbone controls: 26 cm, position 101; one clean trajectory')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(dest/'fig_backbone_curves.png',dpi=150);plt.close(fig)
    summary['figure_scales'] = scales
    (dest/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    lines=['# Backbone attention／CAM 五组配对对照','',
           '26 cm full，位置101；{}个同状态回放。随机数字与随机渲染复用 E15 的同一图案与输入。'.format(nframes),
           '本次只执行 backbone；表中的 L0 复用同输入 E15 的三动作噪声结果。',
           '渲染条件仅替换base，wrist保持clean，用于隔离base成像链。','',
           '| 条件 | 视觉编码后base特征相对差 | 最后层K相对差 | 最后层V相对差 | 沿数字V扰动投影 | E15 L0 |',
           '|---|---:|---:|---:|---:|---:|']
    for name in names:
        s=summary['conditions'][name]
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(name,
            s['feature_base_relative_norm_mean'],s['last_layer_K_relative_norm'],s['last_layer_V_relative_norm'],
            s['V_direction_score_mean'],s['L0_from_E15_mean']))
    lines += ['','## 位置读数','',
              'attention由bowl／plate的query向base image key求均值。patch占base像素平均 {:.2%}。'.format(summary['mean_patch_area_fraction']),
              '按14×14 token内patch的像素覆盖比例加权；以下比例在base内部归一化，绝对base attention mass另列。',
              'CAM使用同一线性V方向目标：clean分数0、数字优化分数1，clean CAM可非零，它表示敏感性而非clean含攻击。','',
              '正CAM总量为零的帧记为缺失，CAM比例只在分母非零的帧取均值；各条件有效帧数见summary.json。','',
              '| 条件 | base总attention mass | patch attention比例 | bowl／plate各自patch比例 | 正CAM的patch比例 | 像素梯度patch比例 |',
              '|---|---:|---:|---:|---:|---:|']
    def pct(x):
        return '{:.2%}'.format(x) if x is not None else '无正CAM／零分母'
    for name in names:
        s=summary['conditions'][name]; noun=s['noun_patch_fraction_by_word']
        lines.append('| {} | {:.4f} | {} | {} / {} | {} | {} |'.format(name,
            s['last_layer_noun_attention_base_mass'],pct(s['noun_attention_patch_fraction_within_base_mean']),
            pct(noun['bowl']),pct(noun['plate']),pct(s['CAM_patch_fraction_within_base_mean']),
            pct(s['pixel_gradient_patch_fraction_within_base_mean'])))
    lines += ['','## 图与原始数据','',
              '- [原始名词attention](fig_attention_base.png)、[相对clean的attention变化](fig_attention_change.png)、[中层／末层位置](fig_attention_depth.png)、[attention rollout](fig_attention_rollout.png)。',
              '- [base Grad-CAM](fig_cam_base.png)、[wrist Grad-CAM](fig_cam_wrist.png)：目标为同状态数字full的V扰动方向，不是action loss。',
              '- [V变化能量](fig_kv_energy.png)、[V变化位置分布](fig_kv_spatial_density.png)：后一张图内部归一化，只比较位置。',
              '- [逐层与时序曲线](fig_backbone_curves.png)、[汇总](summary.json)、[指标](metrics.json)、[热图数组](maps.npz)。','',
              '叠图按图种共用色标，99.5百分位截断上限写入summary.json；16×16图用nearest放大，未伪造像素级分辨率。',
              'CAM在视觉projector的256个特征token上按空间平均梯度作通道权重，再ReLU。它是选择目标下的梯度解释；attention／rollout与CAM均不等于位置的闭环攻击效果。',
              '特征和KV都已包含跨token混合，输出token位置不是独立输入像素。单一已贴位置不能产生所有候选点的攻击效力排序。','',
              '检查：backbone KV与E15匹配；grad/no-grad KV一致；clean分数0、digital分数1；wrist视觉编码逐位不变；未执行action expert。']
    (dest/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['conditions'],indent=2))


if __name__=='__main__':
    main()
