"""E19 cumulative additions: separate KV and flow reports, no model calls."""
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_additive_capture import renderer_condition

DEST = Path(__file__).resolve().parent/'out/render_additive'
PATTERNS = ['full','band32','random']
STAGES = ['Digital','Minimal render','+ bilinear','+ mipmap','+ MSAA4',
          '+ raster256 / NN224','+ PIL resize','+ lighting']
EDGE_LABELS = ['Projection / sampling','Bilinear','Mipmap','MSAA4',
               'Raster256 / NN224','PIL resize','Lighting']
PATH_LABELS = {
    'sampling_first':['Digital','Minimal','Bilinear','Mipmap','AA4','256 / NN','PIL','Lighting'],
    'lighting_first':['Digital','Minimal','Lighting','Bilinear','Mipmap','AA4','256 / NN','PIL'],
    'aa_first':['Digital','Minimal','AA4','Bilinear','Mipmap','256 / NN','PIL','Lighting']}


def mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    data = json.loads((DEST/'metrics.json').read_text())
    assert data['complete']
    pool = np.load(DEST/'inputs.npz',allow_pickle=False)
    plan = data['plan']; paths = plan['paths']; main_path = paths['sampling_first']
    summary = {}; transitions = {}
    for pattern in PATTERNS:
        summary[pattern] = {}
        for cond in data['conditions']:
            rows = [r for r in data['records'] if r['pattern']==pattern and r['condition']==cond]
            item = {field:mean([r['velocity'][field] for r in rows])
                    for field in ['L0','L0_real7','L0_exec7']}
            item['feature_relative_clean'] = mean([r['feature']['relative_clean'] for r in rows])
            item['layers'] = [dict(layer=i,**{label:{
                field:mean([r['kv'][i][label][field] for r in rows])
                for field in ['relative_clean','relative_to_digital','delta_cosine','norm_ratio']}
                for label in ['K','V']}) for i in range(18)]
            if pattern!='random':
                item['L0_vs_random'] = mean([r['velocity_vs_random']['L0'] for r in rows])
            # Explicitly identify switches producing identical pixels. A zero
            # effect in this view need not generalize to other projected scales.
            item['identical_to_digital_frames'] = int(sum(
                np.array_equal(a,b) for a,b in zip(pool[pattern+'__'+cond],pool[pattern+'__digital'])))
            summary[pattern][cond] = item
        transitions[pattern] = {}
        for path_name,path in paths.items():
            out = []
            for edge,(ca,cb) in enumerate(zip(path,path[1:])):
                rows = [r for r in data['transitions'] if r['pattern']==pattern
                        and r['path']==path_name and r['edge']==edge]
                out.append({'from':ca,'to':cb,
                    'identical_pixel_frames':int(sum(np.array_equal(a,b) for a,b in
                         zip(pool[pattern+'__'+ca],pool[pattern+'__'+cb]))),
                    'pixel_MAE_u8':float(np.mean([np.abs(a.astype(float)-b.astype(float))[mask].mean()
                         for a,b,mask in zip(pool[pattern+'__'+ca],pool[pattern+'__'+cb],pool['mask'])])),
                    'velocity_change_L0':mean([r['velocity_change']['L0'] for r in rows]),
                    'layers':[dict(layer=i,**{label:{field:mean([r['kv'][i][label][field] for r in rows])
                        for field in ['relative_to_previous','relative_to_digital','raw_cosine','clean_delta_cosine']}
                        for label in ['K','V']}) for i in range(18)]})
            transitions[pattern][path_name] = out
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    (DEST/'transition_summary.json').write_text(json.dumps(transitions,indent=2,allow_nan=False)+'\n')

    fig,axes = plt.subplots(2,2,figsize=(15,9))
    for col,pattern in enumerate(['full','band32']):
        for label,color in [('K','#c85145'),('V','#3479a8')]:
            vals = [summary[pattern][c]['layers'][-1][label]['relative_to_digital'] for c in main_path]
            axes[0,col].plot(range(8),vals,marker='o',label=label,color=color)
        axes[0,col].plot(range(8),[summary['random'][c]['layers'][-1]['V']['relative_to_digital']
            for c in main_path],'--',color='#888888',label='random V')
        axes[0,col].set_title(pattern+' / last KV distance from digital')
        axes[0,col].set_ylabel('||KV(stage) - KV(digital)|| / ||KV(digital)||')
        edges = transitions[pattern]['sampling_first']
        axes[1,col].bar(range(7),[e['layers'][-1]['V']['relative_to_digital'] for e in edges],color='#3479a8')
        axes[1,col].set_title(pattern+' / last V change at each addition')
        axes[1,col].set_ylabel('||V(next) - V(previous)|| / ||V(digital)||')
        axes[0,col].set_xticks(range(8)); axes[0,col].set_xticklabels(STAGES,rotation=35,ha='right',fontsize=8)
        axes[1,col].set_xticks(range(7)); axes[1,col].set_xticklabels(EDGE_LABELS,rotation=35,ha='right',fontsize=8)
        axes[0,col].legend(); axes[0,col].grid(alpha=.2);axes[1,col].grid(axis='y',alpha=.2)
    fig.suptitle('Backbone only: 16 paired states / fixed base ROI / post-RoPE K\nAdjacent differences are measured directly, not differences of scalar distances')
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(DEST/'fig_kv_additive.png',dpi=150,bbox_inches='tight');plt.close(fig)

    fig,axes = plt.subplots(3,2,figsize=(14,12))
    for row,(name,path) in enumerate(paths.items()):
        for col,pattern in enumerate(['full','band32']):
            ax = axes[row,col]
            ax.plot(range(8),[summary[pattern][c]['L0'] for c in path],'-o',label='opt vs clean',color='#c85145')
            ax.plot(range(8),[summary['random'][c]['L0'] for c in path],'--o',label='random vs clean',color='#888888')
            ax.plot(range(8),[summary[pattern][c]['L0_vs_random'] for c in path],'-s',label='opt vs random',color='#3479a8')
            ax.set_xticks(range(8));ax.set_xticklabels(PATH_LABELS[name],rotation=35,ha='right',fontsize=8)
            ax.set_yscale('log');ax.set_ylabel('Mean early-flow squared difference (log)')
            ax.set_title(pattern+' / '+name.replace('_',' '));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Action head: t=1, 3 paired action noises / no integration or rollout\nOrders denote which effects are enabled, not reordered renderer operations')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(DEST/'fig_flow_additive.png',dpi=150,bbox_inches='tight');plt.close(fig)

    mask = pool['mask'][0]; ys,xs = np.where(mask)
    y0,y1=max(0,ys.min()-3),min(224,ys.max()+4)
    x0,x1=max(0,xs.min()-3),min(224,xs.max()+4)
    fig,axes=plt.subplots(3,8,figsize=(20,8))
    for row,pattern in enumerate(PATTERNS):
        for col,cond in enumerate(main_path):
            axes[row,col].imshow(pool[pattern+'__'+cond][0,y0:y1,x0:x1],interpolation='nearest')
            axes[row,col].set_title(pattern+' / '+STAGES[col],fontsize=8);axes[row,col].axis('off')
    fig.suptitle('Step 10 / identical mask and clean background / sampling-first additions')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(DEST/'fig_inputs_additive.png',dpi=150,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for ax,pattern in zip(axes,PATTERNS):
        arr=np.array([[summary[pattern][renderer_condition(u,f,a,z)]['L0']
            for f in ['nearest','linear','default'] for z in [224,256]]
            for u,a in [(True,0),(True,4),(False,0),(False,4)]])
        im=ax.imshow(np.log10(arr),cmap='viridis',aspect='auto')
        for i in range(4):
            for j in range(6): ax.text(j,i,'{:.3f}'.format(arr[i,j]),ha='center',va='center',color='white',fontsize=9)
        ax.set_xticks(range(6));ax.set_xticklabels(['NN / 224','NN / 256','linear / 224','linear / 256',
            'mipmap / 224','mipmap / 256'],rotation=35,ha='right',fontsize=8)
        ax.set_yticks(range(4));ax.set_yticklabels(['Unlit / AA0','Unlit / AA4','Lit / AA0','Lit / AA4'])
        ax.set_title(pattern+' / mean early L0')
        fig.colorbar(im,ax=ax,label='log10 L0 (per-panel scale)',shrink=.8)
    fig.suptitle('24 renderer configurations: lighting × filtering × AA × raster size\n256 includes production PIL resize to 224; all share the bilinear128 texture export')
    fig.tight_layout(rect=[0,0,1,.9]);fig.savefig(DEST/'fig_factorial.png',dpi=150,bbox_inches='tight');plt.close(fig)

    lines=['# E19：逐项添加真实成像效果','',
        '33条件×3图案×16相同状态＝1584输入；每组3个配对动作噪声。固定26 cm、位置、mask和clean背景。',
        '下文先列backbone KV，再列action head的early L0；没有积分最终动作、新训练或闭环。','',
        '## Backbone：主链末层KV直接距离','',
        '距离定义为 `||KV(stage)-KV(digital)|| / ||KV(digital)||`；K是post-RoPE，有效prefix共520 tokens。',
        '各状态先算归一化距离，再取16状态均值。不是先平均KV再做差。','',
        '| 阶段 | full K | full V | band32 K | band32 V | random V |',
        '|---|---:|---:|---:|---:|---:|']
    for stage,cond in zip(STAGES,main_path):
        f,b,r=[summary[p][cond]['layers'][-1] for p in PATTERNS]
        lines.append('| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |'.format(stage,
            f['K']['relative_to_digital'],f['V']['relative_to_digital'],b['K']['relative_to_digital'],
            b['V']['relative_to_digital'],r['V']['relative_to_digital']))
    lines+=['','### 相邻阶段直接比较','',
        '这里直接对相邻两组KV张量做差；共同分母是数字V范数，不是上表相邻两数相减。','',
        '| 添加项 | full ΔV/数字V | band32 ΔV/数字V | random ΔV/数字V | band32相同像素帧/16 |',
        '|---|---:|---:|---:|---:|---:|']
    for i,label in enumerate(EDGE_LABELS):
        f,b,r=[transitions[p]['sampling_first'][i] for p in PATTERNS]
        lines.append('| {} | {:.5f} | {:.5f} | {:.5f} | {} |'.format(label,
            f['layers'][-1]['V']['relative_to_digital'],b['layers'][-1]['V']['relative_to_digital'],
            r['layers'][-1]['V']['relative_to_digital'],b['identical_pixel_frames']))
    lines+=['','![Backbone逐项添加](fig_kv_additive.png)','',
        '## Action head：主链early L0','',
        'L0为`t=1`的速度平方差，含10×32维，对3噪声与16状态平均。不是KV差、最终动作或ASR。',
        '`opt,random`是配对速度向量的平方差，不是两个相对clean的L0标量相减。','',
        '| 阶段 | full L0 | band32 L0 | random L0 | full L0(opt,random) | band32 L0(opt,random) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for stage,cond in zip(STAGES,main_path):
        f,b,r=[summary[p][cond] for p in PATTERNS]
        lines.append('| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |'.format(stage,
            f['L0'],b['L0'],r['L0'],f['L0_vs_random'],b['L0_vs_random']))
    lines+=['','![三种添加顺序](fig_flow_additive.png)','',
        '## 导出／几何采样的独立对照','',
        'NN/NL/LN分别指纹理导出与逆投影的nearest/linear组合；这是OpenCV近似往返，不是MuJoCo中间buffer。',
        '原版所有真实渲染条件仍使用同一个bilinear128导出。224→256→224只做数字P的重采样。','',
        '| 条件 | full L0 | band32 L0 | random L0 | full末层V距离 | band32末层V距离 |',
        '|---|---:|---:|---:|---:|---:|']
    for cond in data['conditions'][:6]:
        f,b,r=[summary[p][cond] for p in PATTERNS]
        lines.append('| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |'.format(cond,
            f['L0'],b['L0'],r['L0'],f['layers'][-1]['V']['relative_to_digital'],b['layers'][-1]['V']['relative_to_digital']))
    lines+=['','## 全部条件与交互','',
        '完整交叉条件便于检查添加顺序依赖。最近邻最简渲染也有UV导出、透视投影与点采样；',
        '因此数字→最简渲染的落差不能单独归于光照、mipmap或AA。',
        '零效应只表示当前可见尺度／状态下该开关未改变结果，不能推广至其他投影尺寸。','',
        '![交叉条件](fig_factorial.png)','',
        '| 条件 | full L0 | band32 L0 | random L0 | full V距离 | band32 V距离 |',
        '|---|---:|---:|---:|---:|---:|']
    for cond in data['conditions'][6:]:
        f,b,r=[summary[p][cond] for p in PATTERNS]
        lines.append('| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |'.format(cond,
            f['L0'],b['L0'],r['L0'],f['layers'][-1]['V']['relative_to_digital'],b['layers'][-1]['V']['relative_to_digital']))
    lines+=['','![同状态输入](fig_inputs_additive.png)','',
        '[33条件逐层汇总](summary.json) · [三条链相邻KV差](transition_summary.json) · ',
        '[逐帧模型指标](metrics.json) · [捕获与锚点](capture.json) · [添加顺序](plan.json) · [来源校验](provenance.json)',
        '', '限制：仅一条clean轨迹、一个random图案与一份已训练full/band32。16帧相关，不是16独立episode。',
        'KV和L0单位不同，不能按数值比例声称action head把KV差放大了多少。']
    (DEST/'results.md').write_text('\n'.join(lines)+'\n')
    for pattern in PATTERNS:
        print(pattern,{c:round(summary[pattern][c]['L0'],4) for c in main_path})


if __name__=='__main__':main()
