"""E19 nearest-vs-linear texture export, from two paired real-render runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_additive_capture import renderer_condition

OUT=Path(__file__).resolve().parent/'out'
DEST=OUT/'render_additive_nearest_export'
PATTERNS=['full','band32','random']


def main():
    original=json.loads((OUT/'render_additive/metrics.json').read_text())
    nearest=json.loads((DEST/'metrics.json').read_text())
    assert original['complete'] and nearest['complete']
    # All paired states and the unchanged digital controls must agree exactly.
    orig_pool=np.load(OUT/'render_additive/inputs.npz',allow_pickle=False)
    nn_pool=np.load(DEST/'inputs.npz',allow_pickle=False)
    assert np.array_equal(orig_pool['steps'],nn_pool['steps'])
    for key in ['mask','clean_base','clean_wrist','state8','prompt']:
        assert np.array_equal(orig_pool[key],nn_pool[key]),key
    for pattern in PATTERNS:
        for c in nn_pool['conditions'][:6]:
            assert np.array_equal(orig_pool[pattern+'__'+c],nn_pool[pattern+'__'+c]),(pattern,c)
    conditions=nn_pool['conditions'].tolist()
    summary={}
    for pattern in PATTERNS:
        summary[pattern]={}
        for name,data in [('linear_export',original),('nearest_export',nearest)]:
            summary[pattern][name]={}
            for cond in conditions:
                rows=[r for r in data['records'] if r['pattern']==pattern and r['condition']==cond]
                def mean(values): return float(np.mean(values))
                item={f:mean([r['velocity'][f] for r in rows]) for f in ['L0','L0_real7','L0_exec7']}
                item['last_V_relative_digital']=mean([r['kv'][-1]['V']['relative_to_digital'] for r in rows])
                item['last_K_relative_digital']=mean([r['kv'][-1]['K']['relative_to_digital'] for r in rows])
                item['last_V_delta_cosine']=mean([r['kv'][-1]['V']['delta_cosine'] for r in rows])
                if pattern!='random': item['L0_vs_random']=mean([r['velocity_vs_random']['L0'] for r in rows])
                summary[pattern][name][cond]=item
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    chosen=['digital','digital_export128_NN','digital_export128',
            renderer_condition(True,'nearest',0,224),renderer_condition(True,'linear',0,224),
            renderer_condition(False,'nearest',0,224),renderer_condition(False,'default',4,256)]
    labels=['Digital','NN roundtrip','LL roundtrip','Minimal / NN','Minimal / bilinear',
            'Minimal / lit NN','Original pipeline']
    fig,axes=plt.subplots(2,2,figsize=(15,8))
    for col,pattern in enumerate(['full','band32']):
        for name,color in [('linear_export','#3479a8'),('nearest_export','#c85145')]:
            s=summary[pattern][name]
            axes[0,col].plot(range(len(chosen)),[s[c]['last_V_relative_digital'] for c in chosen],'-o',label=name,color=color)
            axes[1,col].plot(range(len(chosen)),[s[c]['L0_vs_random'] for c in chosen],'-o',label=name,color=color)
        axes[0,col].set_ylabel('||V(condition)-V(digital)|| / ||V(digital)||')
        axes[0,col].set_title(pattern+' / backbone last V')
        axes[1,col].set_ylabel('Early L0(opt,random), log scale')
        axes[1,col].set_title(pattern+' / action head');axes[1,col].set_yscale('log')
        for row in range(2):
            axes[row,col].set_xticks(range(len(chosen)));axes[row,col].set_xticklabels(labels,rotation=30,ha='right',fontsize=8)
            axes[row,col].legend(fontsize=8);axes[row,col].grid(alpha=.2)
    fig.suptitle('Only change the real texture-export interpolator: linear vs nearest\nDigital controls are unchanged; conditions on x-axis are separate branches, not cumulative stages')
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(DEST/'fig_export_comparison.png',dpi=150,bbox_inches='tight');plt.close(fig)
    mask=orig_pool['mask'][0];ys,xs=np.where(mask)
    y0,y1=max(0,ys.min()-3),min(224,ys.max()+4);x0,x1=max(0,xs.min()-3),min(224,xs.max()+4)
    imgs=[('digital','Digital'),(renderer_condition(True,'nearest',0,224),'Minimal NN sampling'),
          (renderer_condition(True,'linear',0,224),'Minimal bilinear sampling'),
          (renderer_condition(False,'default',4,256),'Complete pipeline')]
    fig,axes=plt.subplots(4,4,figsize=(12,10))
    for row,(pattern,name,pool) in enumerate([('full','linear export',orig_pool),('full','nearest export',nn_pool),
                                            ('band32','linear export',orig_pool),('band32','nearest export',nn_pool)]):
        for col,(cond,label) in enumerate(imgs):
            axes[row,col].imshow(pool[pattern+'__'+cond][0,y0:y1,x0:x1],interpolation='nearest')
            axes[row,col].set_title(pattern+' / '+name+'\n'+label,fontsize=8);axes[row,col].axis('off')
    fig.suptitle('Step 10 / actual MuJoCo images in the same fixed ROI')
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(DEST/'fig_export_inputs.png',dpi=150,bbox_inches='tight');plt.close(fig)
    lines=['# E19补充：纹理导出插值的真实渲染对照','',
           'nearest导出分支：11条件×3图案×16状态＝528输入，每组3配对动作噪声。',
           '与主实验逐状态配对，只改128纹理导出插值；数字控制、背景、mask、wrist、state、语言完全相同。','',
           '## Backbone：末层V相对数字图的直接距离','',
           '| 条件 | full linear导出 | full nearest导出 | band32 linear导出 | band32 nearest导出 |',
           '|---|---:|---:|---:|---:|']
    for cond,label in zip(chosen,labels):
        f,b=[summary[p] for p in ['full','band32']]
        lines.append('| {} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |'.format(label,
             f['linear_export'][cond]['last_V_relative_digital'],f['nearest_export'][cond]['last_V_relative_digital'],
             b['linear_export'][cond]['last_V_relative_digital'],b['nearest_export'][cond]['last_V_relative_digital']))
    lines+=['','## Action head：early L0','',
            '| 条件 | full linear | full nearest | band32 linear | band32 nearest | random linear | random nearest |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for cond,label in zip(chosen,labels):
        vals=[summary[p][e][cond]['L0'] for p in PATTERNS for e in ['linear_export','nearest_export']]
        lines.append('| {} | '.format(label)+' | '.join('{:.5f}'.format(v) for v in vals)+' |')
    lines+=['','### 相对同条件random的配对速度差','',
            '| 条件 | full linear | full nearest | band32 linear | band32 nearest |',
            '|---|---:|---:|---:|---:|']
    for cond,label in zip(chosen,labels):
        vals=[summary[p][e][cond]['L0_vs_random'] for p in ['full','band32'] for e in ['linear_export','nearest_export']]
        lines.append('| {} | '.format(label)+' | '.join('{:.5f}'.format(v) for v in vals)+' |')
    lines+=['','![导出与采样比较](fig_export_comparison.png)','',
            '横轴是独立分支，不是一条逐项添加链；尤其NN／LL数字往返都没有进入MuJoCo。',
            '完整管线的反应若仍弱，不能把最近邻导出当作已经解决真实渲染鲁棒性的训练方法。','',
            '![真实输入对照](fig_export_inputs.png)','',
            '[汇总](summary.json) · [模型指标](metrics.json) · [渲染设置](capture.json) · [校验](provenance.json)',
            '', 'KV先按状态计算再平均；L0是速度差，非最终action或ASR。一条轨迹、一个random，未训练或闭环。']
    (DEST/'results.md').write_text('\n'.join(lines)+'\n')
    for pattern in PATTERNS:
        for name in ['linear_export','nearest_export']:
            print(pattern,name,{c:round(summary[pattern][name][c]['L0'],4) for c in chosen})


if __name__=='__main__':main()
