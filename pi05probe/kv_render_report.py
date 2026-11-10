"""Summarize and plot the completed KV experiment (CPU, openpi-libero)."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='kv_render_probe')
    args = ap.parse_args()
    dest = ROOT/'pi05probe/out'/args.out
    data = json.loads((dest/'metrics.json').read_text())
    assert data['complete'], 'Probe not complete'
    records = data['records']
    tags = list(dict.fromkeys(r['patch'] for r in records))
    names = ['digital','render_base','render_both','random_digital','random_render_base','random_render_both']
    colors = {'digital':'#b33739','render_base':'#2878a1','render_both':'#dd9230',
              'random_digital':'#777777','random_render_base':'#65a7a0','random_render_both':'#b08eaf'}
    summary = {}
    for tag in tags:
        rows = [r for r in records if r['patch']==tag]
        summary[tag] = {'conditions':{},'alignment':{},'kv_swaps':{},
                        'optimized_vs_random':{},'optimized_random_alignment':{}}
        for name in names:
            values = [r['conditions'][name] for r in rows]
            item = {'L0_mean':float(np.mean([v['L0'] for d in values for v in d['velocity']])),
                    'L0_real7_mean':float(np.mean([v['L0_real7'] for d in values for v in d['velocity']])),
                    'L0_exec7_mean':float(np.mean([v['L0_exec7'] for d in values for v in d['velocity']])),
                    'action_exec7_median':float(np.median([d['action_exec7_norm'] for d in values]))}
            item['last_layer'] = {}
            for group in ['all_valid','base_0_rgb','left_wrist_0_rgb','language_valid','base_patch_tokens']:
                if group not in values[0]['kv_vs_clean'][-1]['groups']:
                    continue
                item['last_layer'][group] = {kind:float(np.mean([d['kv_vs_clean'][-1]['groups'][group][kind]['relative_norm']
                                                             for d in values])) for kind in ['K','V']}
            item['expert_last_layer_output_relative_norm'] = float(np.mean(
                [d['expert_attention_vs_clean'][-1]['attention_output_relative_norm'] for d in values]))
            summary[tag]['conditions'][name] = item
        for name in ['render_base','render_both']:
            item = {}
            for kind in ['K','V']:
                item[kind] = {key:float(np.mean([r['alignment'][name][-1]['groups']['all_valid'][kind][key]
                                                for r in rows]))
                              for key in ['delta_cosine','norm_ratio','parallel_gain','orthogonal_ratio']}
            summary[tag]['alignment'][name] = item
        for name in ['digital','render_base','render_both']:
            values = [r['optimized_vs_random'][name] for r in rows]
            summary[tag]['optimized_vs_random'][name] = {
                'L0_mean':float(np.mean([v['L0'] for d in values for v in d['velocity']])),
                'L0_exec7_mean':float(np.mean([v['L0_exec7'] for d in values for v in d['velocity']])),
                'last_layer_relative_norm':{kind:float(np.mean([
                    d['kv'][-1]['groups']['all_valid'][kind]['relative_norm'] for d in values]))
                    for kind in ['K','V']}}
        for name in ['render_base','render_both']:
            summary[tag]['optimized_random_alignment'][name] = {
                kind:{key:float(np.mean([r['optimized_random_alignment'][name][-1]['groups']['all_valid'][kind][key]
                                        for r in rows]))
                      for key in ['delta_cosine','norm_ratio','parallel_gain','orthogonal_ratio']}
                for kind in ['K','V']}
        for name in rows[0]['kv_swaps']:
            summary[tag]['kv_swaps'][name] = {'L0_mean':float(np.mean([v['L0'] for r in rows for v in r['kv_swaps'][name]])),
                                             'L0_exec7_mean':float(np.mean([v['L0_exec7'] for r in rows for v in r['kv_swaps'][name]]))}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig, axes = plt.subplots(len(tags),4,figsize=(18,4*len(tags)),squeeze=False)
    for i,tag in enumerate(tags):
        rows = [r for r in records if r['patch']==tag]
        for name in names:
            values = [r['conditions'][name] for r in rows]
            for j,kind in enumerate(['K','V']):
                curve = np.mean([[layer['groups']['all_valid'][kind]['relative_norm'] for layer in v['kv_vs_clean']]
                                 for v in values],axis=0)
                axes[i,j].plot(np.arange(len(curve))+1,curve,label=name,color=colors[name],
                               linestyle='--' if name.startswith('random') else '-')
        for name in ['render_base','render_both']:
            for kind,style in [('K','--'),('V','-')]:
                curve = np.mean([[layer['groups']['all_valid'][kind]['delta_cosine'] for layer in r['alignment'][name]]
                                 for r in rows],axis=0)
                axes[i,2].plot(np.arange(len(curve))+1,curve,label=name+'/'+kind,color=colors[name],linestyle=style)
        for name in ['digital','render_base','render_both','random_digital']:
            curve = np.mean([[layer['attention_output_relative_norm'] for layer in r['conditions'][name]['expert_attention_vs_clean']]
                             for r in rows],axis=0)
            axes[i,3].plot(np.arange(len(curve))+1,curve,label=name,color=colors[name])
        for j,title in enumerate(['K relative change vs clean','V relative change vs clean',
                                   'Cosine of delta: digital vs render','Expert attention output change']):
            axes[i,j].set_title(tag+' / '+title,fontsize=10)
            axes[i,j].set_xlabel('Transformer layer (1-based)')
            axes[i,j].grid(alpha=.2)
            axes[i,j].legend(fontsize=7)
        axes[i,2].set_ylim(-1,1)
    fig.suptitle('Matched prefix KV and first-step expert response\nMeans over 16 states of one clean trajectory; KV excludes padded tokens')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(dest/'fig_kv_layers.png',dpi=150);plt.close(fig)
    fig, axes = plt.subplots(len(tags),2,figsize=(12,3.5*len(tags)),squeeze=False)
    for i,tag in enumerate(tags):
        rows=[r for r in records if r['patch']==tag]
        steps=[r['step'] for r in rows]
        for name in names:
            axes[i,0].plot(steps,[np.mean([v['L0_exec7'] for v in r['conditions'][name]['velocity']]) for r in rows],
                           label=name,color=colors[name],linestyle='--' if name.startswith('random') else '-')
            axes[i,1].plot(steps,[r['conditions'][name]['action_exec7_norm'] for r in rows],
                           label=name,color=colors[name],linestyle='--' if name.startswith('random') else '-')
        for j,title in enumerate(['First-step velocity difference / executed 5x7','Final action difference / executed 5x7']):
            axes[i,j].set_title(tag+' / '+title,fontsize=10);axes[i,j].set_xlabel('Environment step (clean replay)')
            axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=7)
    fig.suptitle('Open-loop response across task stages; not ASR or robot displacement')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(dest/'fig_kv_velocity_time.png',dpi=150);plt.close(fig)
    fig, axes = plt.subplots(len(tags),2,figsize=(12,3.5*len(tags)),squeeze=False)
    for i,tag in enumerate(tags):
        rows=[r for r in records if r['patch']==tag]
        for name in ['digital','render_base']:
            for kind,style in [('K','--'),('V','-')]:
                curve=np.mean([[layer['groups']['all_valid'][kind]['relative_norm']
                                for layer in r['optimized_vs_random'][name]['kv']] for r in rows],axis=0)
                axes[i,0].plot(np.arange(len(curve))+1,curve,label=name+'/'+kind,color=colors[name],linestyle=style)
        for kind,style in [('K','--'),('V','-')]:
            curve=np.mean([[layer['groups']['all_valid'][kind]['delta_cosine']
                            for layer in r['optimized_random_alignment']['render_base']] for r in rows],axis=0)
            axes[i,1].plot(np.arange(len(curve))+1,curve,label=kind,color='#2878a1',linestyle=style)
        for j,title in enumerate(['Optimized vs random: KV relative change',
                                  'Delta cosine: digital vs render (optimized - random)']):
            axes[i,j].set_title(tag+' / '+title,fontsize=10);axes[i,j].set_xlabel('Transformer layer (1-based)')
            axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=7)
        axes[i,1].set_ylim(-1,1)
    fig.suptitle('Pattern identity control: one shared random pattern, not a frequency-matched optimization trial')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(dest/'fig_kv_pattern_identity.png',dpi=150);plt.close(fig)
    lines = ['# KV／渲染诊断结果','',
             '单条 clean 成功轨迹的16个同状态回放；4张已有图案；第一去噪步配对3个动作噪声。',
             '下表是开环量，不是闭环成功率。动作是模型归一化空间；KV相对差按有效token统计。','',
             '| 图案／条件 | L0均值 | 执行5×7的L0 | 最终执行5×7动作差中位数 | 最后层K相对差 | 最后层V相对差 |',
             '|---|---:|---:|---:|---:|---:|']
    for tag in tags:
        for name in names:
            s=summary[tag]['conditions'][name]
            lines.append('| {} / {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(tag,name,
                         s['L0_mean'],s['L0_exec7_mean'],s['action_exec7_median'],
                         s['last_layer']['all_valid']['K'],s['last_layer']['all_valid']['V']))
    lines += ['', '## 数字扰动方向在渲染后的变化','',
              '这里比较 ΔKVdigital 与 ΔKVrender（各自减去同一个clean），而非两个大KV张量直接求相关。',
              'cos反映方向；parallel gain是渲染扰动沿数字扰动方向的投影比例；不等于攻击保留率。','',
              '| 图案／条件 | ΔK cosine | ΔV cosine | K parallel gain | V parallel gain |',
              '|---|---:|---:|---:|---:|']
    for tag in tags:
        for name,a in summary[tag]['alignment'].items():
            lines.append('| {} / {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(tag,name,
                         a['K']['delta_cosine'],a['V']['delta_cosine'],a['K']['parallel_gain'],a['V']['parallel_gain']))
    lines += ['','## 优化图案相对随机图案的对照','',
              'KV方向比较改用 Δdigital = KVopt,digital − KVrandom,digital，Δrender 同理。',
              'L0(opt,random) 是两个速度向量的平方差，不是两份 L0(clean) 相减。',
              '随机图案只有一个，band32的随机对照未匹配频谱；此处用于区分图案身份，不作为训练效益的严格因果估计。','',
              '| 图案 | 数字 L0(opt,random) | 渲染base L0(opt,random) | ΔK cosine | ΔV cosine |',
              '|---|---:|---:|---:|---:|']
    for tag in tags:
        s=summary[tag]; a=s['optimized_random_alignment']['render_base']
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(tag,
                     s['optimized_vs_random']['digital']['L0_mean'],
                     s['optimized_vs_random']['render_base']['L0_mean'],a['K']['delta_cosine'],a['V']['delta_cosine']))
    lines += ['','## KV交换（诊断，不是可实现输入）','',
              '| 图案 | digital K + render V 的L0 | render K + digital V 的L0 |',
              '|---|---:|---:|']
    for tag in tags:
        s=summary[tag]['kv_swaps']
        lines.append('| {} | {:.4f} | {:.4f} |'.format(tag,s['digital_K_render_V']['L0_mean'],s['render_K_digital_V']['L0_mean']))
    lines += ['','[逐层图](fig_kv_layers.png) · [时序图](fig_kv_velocity_time.png) · [随机对照图](fig_kv_pattern_identity.png) · [原始指标](metrics.json)',
              '', '有效性检查：clean渲染重复一致；KV重复一致；第一步速度重复一致；缓存路径与原DRIFT速度函数逐位一致。',
              '局限：一个位置、一条开发轨迹；不是泛化评测。render_base保留clean wrist，render_both使用真实渲染两视角。',
              'KV含多层条件传播；全局距离和交换实验不能单独确立具体像素或具体层的因果机制。']
    (dest/'results.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
