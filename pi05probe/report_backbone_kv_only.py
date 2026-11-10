"""Separate E18's saved backbone KV metrics from action-head diagnostics.

No model invocation. All layers use 520 valid prefix tokens. Raw KV distance
and clean-referenced perturbation alignment are deliberately different fields.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'pi05probe/out/render_ablation'
DEST=SOURCE/'kv_only'


def main():
    data=json.loads((SOURCE/'metrics.json').read_text())
    assert data['complete']
    DEST.mkdir(exist_ok=True)
    rows=[]
    for r in data['records']:
        rows.append({'pattern':r['pattern'],'condition':r['condition'],'step':r['step'],
                     'layers':r['kv']})
    def mean(values):
        values=[v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    summary={}
    for pattern in ['full','band32','random']:
        summary[pattern]={}
        for cond in data['conditions']:
            selected=[r for r in rows if r['pattern']==pattern and r['condition']==cond]
            summary[pattern][cond]=[]
            for i in range(18):
                summary[pattern][cond].append({'layer':i+1,**{key:{metric:mean([r['layers'][i][key][metric] for r in selected])
                    for metric in ['relative_clean','relative_to_digital','norm_ratio','delta_cosine','parallel_gain','residual_ratio']}
                    for key in ['K','V']}})
    meta={'complete':True,'model_executed':False,'action_head_metrics_included':False,
          'valid_prefix_tokens':520,'layers':18,'states':16,'records':rows,
          'definitions':{'relative_clean':'||X-C|| / ||C||',
                         'relative_to_digital':'||X-D|| / ||D||',
                         'delta_cosine':'cos(X-C, D-C)',
                         'residual_ratio':'||(X-C)-(D-C)|| / ||D-C||'},
          'notes':'K post-RoPE, V prefix cache; Frobenius norms over valid token/head/channel dimensions. '
                  'Summaries average per-state values, not distance between averaged tensors. '
                  'This extracts saved E18 measurements, no new model inference or action-head comparison.'}
    (DEST/'metrics.json').write_text(json.dumps(meta,allow_nan=False)+'\n')
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    colors={'full':'#b94143','band32':'#d79022','random':'#3b72a3'}
    for col,key in enumerate(['K','V']):
        for pattern in colors:
            chosen=[r for r in rows if r['pattern']==pattern and r['condition']=='render']
            for rr,metric in enumerate(['relative_to_digital','delta_cosine']):
                values=np.array([[r['layers'][i][key][metric] for i in range(18)] for r in chosen])
                xx=np.arange(1,19)
                axes[rr,col].plot(xx,values.mean(0),label=pattern,color=colors[pattern])
                axes[rr,col].fill_between(xx,values.min(0),values.max(0),alpha=.12,color=colors[pattern])
        axes[0,col].set_title(key+' / raw render-to-digital distance')
        axes[0,col].set_ylabel('||R-D|| / ||D||')
        axes[1,col].set_title(key+' / clean-referenced perturbation alignment')
        axes[1,col].set_ylabel('cos(R-C, D-C)')
        for ax in axes[:,col]:
            ax.set_xlabel('Prefix backbone layer');ax.grid(alpha=.2);ax.legend()
        # Relative distances can exceed 1; only cosine is bounded.
        axes[0,col].set_ylim(bottom=0)
        axes[1,col].set_ylim(-.05,1.05)
    fig.suptitle('Backbone KV only / 16 replay states / 520 valid tokens\nLines: per-state mean; shading: state min–max, not confidence intervals')
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(DEST/'fig_kv_only.png',dpi=150,bbox_inches='tight');plt.close(fig)
    lines=['# Backbone KV独立比较','',
           '仅复算已保存的KV指标，无新模型调用。下表为第18层、520有效prefix token、16状态逐状态指标的平均。',
           'D=数字贴图的KV，R=原版真实render的KV，C=clean的KV。K/V分别计算Frobenius范数。','',
           '| 图案 | K距离 ∥R−D∥/∥D∥ | V距离 ∥R−D∥/∥D∥ | K扰动方向cos | V扰动方向cos |',
           '|---|---:|---:|---:|---:|']
    for p in colors:
        v=summary[p]['render'][-1]
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(p,v['K']['relative_to_digital'],v['V']['relative_to_digital'],v['K']['delta_cosine'],v['V']['delta_cosine']))
    lines+=['','## 相对clean的距离','',
            '| 图案 | 数字ΔK/clean K | renderΔK/clean K | 数字ΔV/clean V | renderΔV/clean V |',
            '|---|---:|---:|---:|---:|']
    for p in colors:
        d,r=summary[p]['digital'][-1],summary[p]['render'][-1]
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(p,d['K']['relative_clean'],r['K']['relative_clean'],d['V']['relative_clean'],r['V']['relative_clean']))
    lines+=['','## 同一成像消融只看V的直接距离','',
            '均使用数字V作为分母；render是原图，其他renderer干预固定mask背景。',
            '| 条件 | full ∥X−D∥/∥D∥ | band32 ∥X−D∥/∥D∥ | random ∥X−D∥/∥D∥ |',
            '|---|---:|---:|---:|']
    for cond in data['conditions']:
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} |'.format(cond,*[summary[p][cond][-1]['V']['relative_to_digital'] for p in colors]))
    lines+=['','[18层曲线](fig_kv_only.png) · [逐状态指标](metrics.json) · [全部汇总](summary.json)',
            '原始KV距离与ΔKV的方向是不同指标。距离0.725不是72.5%的语义信息丢失，也不能据此直接推算行为。']
    (DEST/'results.md').write_text('\n'.join(lines)+'\n')
    provenance={'date':'2026-10-05','source_sha256':hashlib.sha256((SOURCE/'metrics.json').read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'model_executed':False,'checks':{'records':len(rows),'layers':18,'no_action_fields':True}}
    (DEST/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('\n'.join(lines[:22]))


if __name__=='__main__':main()
