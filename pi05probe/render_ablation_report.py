"""E18 tables and figures; no policy or simulator calls."""
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_pixel_diagnosis import paired_stats, tv, roi_square, spectrum

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'pi05probe/out/render_ablation'


def main():
    data=json.loads((DEST/'metrics.json').read_text())
    assert data['complete']
    pool=np.load(DEST/'inputs.npz',allow_pickle=False)
    conditions=data['conditions']
    labels={'digital':'Digital', 'digital_resize256':'P 224→256→224',
      'digital_export128':'P→texture128→224','digital_export128_align':'+ alignment',
      'digital_lowpass32':'Same P low-pass32','render':'True render',
      'render_roi':'Render in fixed ROI','render_nomipmap':'No mipmap (linear)',
      'render_nearest':'Nearest texture','render_unlit':'Unlit',
      'render_unlit_nomipmap':'Unlit + no mipmap','render_unlit_nearest':'Unlit + nearest',
      'render_resize_nearest':'Nearest final resize','render_native224':'Native render224',
      'render_noaa':'No MSAA','render_unlit_nearest_noaa':'Unlit + nearest + no AA',
      'render_unlit_nearest_noaa_native224':'Unlit/nearest/no AA/native224'}
    summary={}; pixel_rows=[]
    for pattern in ['full','band32','random']:
        summary[pattern]={}
        for cond in conditions:
            rows=[r for r in data['records'] if r['pattern']==pattern and r['condition']==cond]
            def mean(values):
                values=[v for v in values if v is not None]
                return float(np.mean(values)) if values else None
            item={'L0':mean([r['velocity']['L0'] for r in rows]),
                  'L0_real7':mean([r['velocity']['L0_real7'] for r in rows]),
                  'L0_exec7':mean([r['velocity']['L0_exec7'] for r in rows]),
                  'feature_relative_clean':mean([r['feature']['relative_clean'] for r in rows]),
                  'feature_delta_cosine':mean([r['feature']['delta_cosine'] for r in rows]),
                  'last_V_delta_cosine':mean([r['kv'][-1]['V']['delta_cosine'] for r in rows]),
                  'last_V_relative_clean':mean([r['kv'][-1]['V']['relative_clean'] for r in rows]),
                  'last_V_parallel_gain':mean([r['kv'][-1]['V']['parallel_gain'] for r in rows]),
                  'last_V_residual_ratio':mean([r['kv'][-1]['V']['residual_ratio'] for r in rows])}
            if pattern!='random':
                item['L0_vs_random']=mean([r['velocity_vs_random']['L0'] for r in rows])
                item['last_pattern_V_delta_cosine']=mean([r['optimized_random_kv_alignment'][-1]['V']['delta_cosine'] for r in rows])
            pix=[]
            for k,step in enumerate(pool['steps']):
                mask=cv2.erode(pool['mask'][k].astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
                digital=pool[pattern+'__digital'][k].astype(np.float64)/255
                im=pool[pattern+'__'+cond][k].astype(np.float64)/255
                stats=paired_stats(digital,im,mask)
                x,y,side=roi_square(mask)
                sd=spectrum(digital[y:y+side,x:x+side]); ss=spectrum(im[y:y+side,x:x+side])
                pix.append({'step':int(step),'MAE':stats['MAE'],'correlation':stats['correlation_mean_channels'],
                            'TV_retention':tv(im,mask)/tv(digital,mask),
                            'HF_retention':ss['high_frequency_energy']/sd['high_frequency_energy']})
            item['pixel_MAE']=mean([r['MAE'] for r in pix]);item['pixel_corr']=mean([r['correlation'] for r in pix])
            item['TV_retention']=mean([r['TV_retention'] for r in pix]);item['HF_retention']=mean([r['HF_retention'] for r in pix])
            summary[pattern][cond]=item
            pixel_rows.append({'pattern':pattern,'condition':cond,'frames':pix})
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (DEST/'pixel_metrics.json').write_text(json.dumps(pixel_rows,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(16,8))
    y=np.arange(len(conditions))
    for ax,pattern in zip(axes,['full','band32']):
        for name,color,offset in [(pattern,'#bd4145',-.15),('random','#466d9c',.15)]:
            ax.scatter([summary[name][c]['L0'] for c in conditions],y+offset,s=28,label=name,color=color)
        ax.set_yticks(y);ax.set_yticklabels([labels[c] for c in conditions],fontsize=8)
        ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Mean early-flow L0 (log scale)');ax.legend()
        ax.set_title(pattern+' / paired conditions');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Same 16 states × 3 action noises / base only / no training or rollout\nRenderer interventions use fixed ROI; raw render is a separate anchor')
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(DEST/'fig_l0_ablation.png',dpi=150,bbox_inches='tight');plt.close(fig)
    chosen=['digital_resize256','digital_lowpass32','render','render_unlit','render_nomipmap','render_unlit_nearest_noaa_native224']
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for col,pattern in enumerate(['full','band32']):
        for cond in chosen:
            rows=[r for r in data['records'] if r['pattern']==pattern and r['condition']==cond]
            for row,field in [(0,'delta_cosine'),(1,'parallel_gain')]:
                arr=np.array([[r['kv'][i]['V'][field] for i in range(18)] for r in rows])
                axes[row,col].plot(np.arange(1,19),arr.mean(0),label=labels[cond])
                axes[row,col].set_title(pattern+' / ΔV '+field);axes[row,col].set_xlabel('Prefix layer')
                axes[row,col].grid(alpha=.2)
        axes[0,col].legend(fontsize=7);axes[0,col].set_ylim(-.1,1.05)
    fig.suptitle('Perturbation referenced to clean; digital direction is the reference\nSimilarity or projection alone is not attack efficacy')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(DEST/'fig_kv_ablation.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,pattern in zip(axes,['full','band32']):
        for cond in ['digital','digital_resize256','render','render_unlit','render_unlit_nearest_noaa_native224']:
            rows=[r for r in data['records'] if r['pattern']==pattern and r['condition']==cond]
            ax.plot([r['step'] for r in rows],[r['velocity_vs_random']['L0'] for r in rows],label=labels[cond],marker='.',markersize=3)
        ax.set_yscale('log');ax.set_title(pattern+' / early L0(optimized, random)')
        ax.set_xlabel('Environment step');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(DEST/'fig_l0_time.png',dpi=150,bbox_inches='tight');plt.close(fig)
    chosen=['digital','digital_resize256','render','render_unlit','render_nearest','render_unlit_nearest_noaa_native224']
    mask=pool['mask'][0];ys,xs=np.where(mask);y0,y1=ys.min()-3,ys.max()+4;x0,x1=xs.min()-3,xs.max()+4
    fig,axes=plt.subplots(3,len(chosen),figsize=(16,8))
    for i,pattern in enumerate(['full','band32','random']):
        for j,cond in enumerate(chosen):
            axes[i,j].imshow(pool[pattern+'__'+cond][0,y0:y1,x0:x1],interpolation='nearest')
            axes[i,j].set_title(pattern+' / '+labels[cond],fontsize=8);axes[i,j].axis('off')
    fig.suptitle('Step 10 / same patch mask and background for interventions')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(DEST/'fig_inputs.png',dpi=150,bbox_inches='tight');plt.close(fig)
    lines=['# E18 full／band32／random 成像消融','',
      '同一clean轨迹16状态、3个配对动作噪声，26 cm固定位置。L0是早期flow速度平方差，非ASR。',
      '除原始render外，所有渲染干预都只替换同一hard mask内像素，背景／wrist不变。','',
      '| 条件 | full L0 | band32 L0 | random L0 | full末层ΔV cosine | band32末层ΔV cosine |',
      '|---|---:|---:|---:|---:|---:|']
    for cond in conditions:
        f,b,r=[summary[p][cond] for p in ['full','band32','random']]
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.3f} | {:.3f} |'.format(cond,f['L0'],b['L0'],r['L0'],f['last_V_delta_cosine'],b['last_V_delta_cosine']))
    lines+=['','## 相对同条件 random 的响应','',
       '这里是 optimized 与 random 两个速度向量之差的平方范数，不是两个 L0 标量相减。',
       '| 条件 | full L0(opt,rand) | band32 L0(opt,rand) | full图案ΔV cosine | band32图案ΔV cosine |',
       '|---|---:|---:|---:|---:|']
    for cond in conditions:
        f,b=[summary[p][cond] for p in ['full','band32']]
        lines.append('| {} | {:.4f} | {:.4f} | {:.3f} | {:.3f} |'.format(cond,f['L0_vs_random'],b['L0_vs_random'],f['last_pattern_V_delta_cosine'],b['last_pattern_V_delta_cosine']))
    lines+=['','[输入图](fig_inputs.png) · [L0消融](fig_l0_ablation.png) · [逐层KV](fig_kv_ablation.png) · [相对random的时序](fig_l0_time.png)',
            '[全部汇总](summary.json) · [逐状态模型指标](metrics.json) · [像素指标](pixel_metrics.json) · [渲染设置与校验](capture.json)']
    (DEST/'results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:25]))


if __name__=='__main__':main()
