"""E20: recompute response curves from saved paired velocities, no model calls."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEST=Path(__file__).resolve().parent/'out/kv_interpolation'
SCOPES={'all':(10,32),'real7':(10,7),'exec7':(5,7)}
COLORS={'all':'#be4b46','real7':'#3479a8','exec7':'#548848'}


def main():
    data=json.loads((DEST/'metrics.json').read_text());assert data['complete']
    arrays=np.load(DEST/'velocities.npz',allow_pickle=False)
    v=arrays['velocity'][:,:,:,:,0].astype(np.float64)
    clean=arrays['clean_velocity'][:,:,0].astype(np.float64)
    alphas=arrays['alphas'];patterns=arrays['patterns'].tolist()
    n,p,a,s,h,d=v.shape
    assert (n,p,a,s,h,d)==(16,3,23,3,10,32)
    summary={};rows={}
    for pattern in patterns:
        rows[pattern]={alpha:[r for r in data['records'] if r['pattern']==pattern and r['alpha']==alpha]
                       for alpha in alphas}
        pi=patterns.index(pattern);summary[pattern]=[]
        for ai,alpha in enumerate(alphas):
            records=rows[pattern][alpha]
            item={'alpha':float(alpha),
                'kv_global':{label:{field:float(np.mean([r['kv']['global'][label][field] for r in records]))
                    for field in ['relative_to_digital','fraction_of_endpoint_distance','projection_fraction',
                                  'off_path_fraction','changed_element_fraction']}
                    for label in ['K','V','KV']},'scopes':{}}
            for scope,(hh,dd) in SCOPES.items():
                current=v[:,pi,ai,:,:hh,:dd];digital=v[:,pi,0,:,:hh,:dd];render=v[:,pi,-1,:,:hh,:dd]
                base=clean[:,:,:hh,:dd];endpoint=render-digital;change=current-digital
                den=np.maximum((endpoint**2).sum(axis=(-2,-1)),1e-20)
                progress=np.sqrt((change**2).sum(axis=(-2,-1))/den)
                projection=(change*endpoint).sum(axis=(-2,-1))/den
                bend=np.sqrt(((change-alpha*endpoint)**2).sum(axis=(-2,-1))/den)
                digital_l0=float(np.mean(((digital-base)**2).sum(axis=(-2,-1))))
                result={'L0_clean':float(np.mean(((current-base)**2).sum(axis=(-2,-1)))),
                    'L0_digital':float(np.mean((change**2).sum(axis=(-2,-1)))),
                    'L0_render':float(np.mean(((current-render)**2).sum(axis=(-2,-1)))),
                    'velocity_fraction_mean':float(progress.mean()),'velocity_fraction_median':float(np.median(progress)),
                    'velocity_fraction_pooled':float(np.sqrt(np.sum(change**2)/np.sum(endpoint**2))),
                    'velocity_fraction_min':float(progress.min()),'velocity_fraction_max':float(progress.max()),
                    'projection_fraction_mean':float(projection.mean()),'linear_deviation_mean':float(bend.mean()),
                    'linear_velocity_L0_clean':float(np.mean(((digital+alpha*endpoint-base)**2).sum(axis=(-2,-1))))}
                result['L0_clean_retention']=result['L0_clean']/max(digital_l0,1e-20)
                if pattern!='random':
                    random=v[:,patterns.index('random'),ai,:,:hh,:dd]
                    result['L0_random']=float(np.mean(((current-random)**2).sum(axis=(-2,-1))))
                item['scopes'][scope]=result
            summary[pattern].append(item)
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for col,pattern in enumerate(['full','band32','random']):
        ss=summary[pattern]
        axes[0,col].plot(alphas,[r['scopes']['all']['L0_clean'] for r in ss],'-o',color=COLORS['all'],label='actual head',markersize=3)
        axes[0,col].plot(alphas,[r['scopes']['all']['linear_velocity_L0_clean'] for r in ss],'--',color='#888888',label='linear velocity reference')
        axes[0,col].set_title(pattern+' / early L0 vs clean');axes[0,col].set_yscale('log')
        axes[0,col].set_ylabel('Mean squared velocity difference (log)')
        for scope in SCOPES:
            axes[1,col].plot([r['kv_global']['KV']['fraction_of_endpoint_distance'] for r in ss],
                [r['scopes'][scope]['velocity_fraction_pooled'] for r in ss],'-o',color=COLORS[scope],label=scope,markersize=3)
        axes[1,col].plot([0,1],[0,1],'--',color='#888888',label='linear reference')
        axes[1,col].set_title(pattern+' / progress along endpoint gap')
        axes[1,col].set_xlabel('Actual ||KVα-KVD|| / ||KVR-KVD||')
        axes[1,col].set_ylabel('Pooled ||vα-vD|| / ||vR-vD||')
        axes[0,col].set_xlabel('KV interpolation α (digital=0, render=1)')
        for row in range(2):axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
    fig.suptitle('All 18 layers K+V / fixed noise, robot state and t=1\n16 states × 3 paired action noises; mixed caches, no integration or rollout')
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(DEST/'fig_response.png',dpi=150,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(2,3,figsize=(17,8))
    small=alphas<=.1
    for col,pattern in enumerate(['full','band32','random']):
        ss=[r for r in summary[pattern] if r['alpha']<=.1]
        for scope in SCOPES:
            axes[0,col].plot(alphas[small],[r['scopes'][scope]['L0_clean_retention'] for r in ss],'-o',
                color=COLORS[scope],label=scope,markersize=3)
            axes[1,col].plot(alphas[small],[r['scopes'][scope]['velocity_fraction_pooled'] for r in ss],'-o',
                color=COLORS[scope],label=scope,markersize=3)
        axes[0,col].set_title(pattern+' / digital response retention');axes[0,col].set_ylabel('Mean L0α / mean L0D')
        axes[1,col].set_title(pattern+' / early endpoint progress');axes[1,col].set_ylabel('Pooled ||vα-vD|| / ||vR-vD||')
        axes[1,col].plot(alphas[small],alphas[small],'--',color='#888888',label='linear reference')
        for row in range(2):
            axes[row,col].set_xlabel('Nominal α (0 to 0.1)');axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
    fig.suptitle('Small KV changes: dense sampling near digital endpoint\nall=10×32, real7=10×7, exec7=5×7 velocities; no final actions')
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(DEST/'fig_small_alpha.png',dpi=150,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(17,4.5))
    for ax,pattern in zip(axes,['full','band32','random']):
        arr=np.array([[rows[pattern][alpha][i]['velocity_vs_clean']['L0'] for alpha in alphas] for i in range(n)])
        retention=arr/arr[:,0,None]
        im=ax.imshow(retention,aspect='auto',origin='lower',cmap='viridis')
        ticks=[0,4,7,10,13,16,18,22]
        ax.set_xticks(ticks);ax.set_xticklabels(['{:g}'.format(alphas[i]) for i in ticks],rotation=30)
        ax.set_yticks(range(n));ax.set_yticklabels(arrays['steps'].tolist(),fontsize=8)
        ax.set_xlabel('KV α (nonuniform grid)');ax.set_ylabel('Environment step');ax.set_title(pattern)
        fig.colorbar(im,ax=ax,label='L0α / L0D per state',shrink=.8)
    fig.suptitle('State-dependent response retention / per-panel color scales\nAbsolute steps from one clean trajectory; not event-aligned independent episodes')
    fig.tight_layout(rect=[0,0,1,.88]);fig.savefig(DEST/'fig_state_response.png',dpi=150,bbox_inches='tight');plt.close(fig)

    selected=[0.,.001,.0025,.005,.01,.02,.05,.1,.25,.5,.75,.9,1.]
    def at(pattern,alpha):return summary[pattern][list(alphas).index(alpha)]
    lines=['# E20：同noise下数字→render的KV插值','',
        '16状态×3图案×23插值点＝1104组缓存干预，每组3个相同动作噪声，共3312次主曲线head前向。',
        '额外clean／原生端点用于校验。每次head都从同一ε出发，t=1；不把前一α的速度带入下一α。',
        '所有18层K/V一起插值。固定机器人状态、prefix mask、wrist与语言；原图像仅用于生成两端KV。','',
        '## 输入：实际KV改变量','',
        '下表为18层有效token的K+V拼接范数，`||KVα-KVD||/||KVD||`。',
        '插值在FP32算术后转回原缓存dtype，0/1端点直接使用原缓存；实际路径误差见summary。','',
        '| α | full KV相对数字差 | band32 KV相对数字差 | random KV相对数字差 |',
        '|---|---:|---:|---:|']
    for alpha in selected:
        lines.append('| {:g} | '.format(alpha)+' | '.join('{:.6f}'.format(at(p,alpha)['kv_global']['KV']['relative_to_digital'])
            for p in ['full','band32','random'])+' |')
    lines+=['','## 输出：head第一步速度差','',
        'L0为相对clean的10×32速度平方差，先均值3噪声再均值16状态。不是最终action或ASR。','',
        '| α | full L0 | band32 L0 | random L0 | full保留数字L0 | band32保留数字L0 |',
        '|---|---:|---:|---:|---:|---:|']
    for alpha in selected:
        f,b,r=[at(p,alpha)['scopes']['all'] for p in ['full','band32','random']]
        lines.append('| {:g} | {:.5f} | {:.5f} | {:.5f} | {:.2%} | {:.2%} |'.format(alpha,
            f['L0_clean'],b['L0_clean'],r['L0_clean'],f['L0_clean_retention'],b['L0_clean_retention']))
    lines+=['','### 速度变化走过端点差的多少','',
        '下表使用`√(Σ||vα-vD||² / Σ||vR-vD||²)`，将16状态×3噪声的速度向量拼接后归一化。',
        '逐配对比例的均值／中位数也保留在summary；小端点差会放大逐配对比值，两个口径不能混用。',
        '它不是L0保留率；线性head对应约α。实际head的偏离可表明沿这条KV方向的非线性，',
        '不能直接解释为所有KV方向均敏感，也不能把KV与L0原值相除称放大倍数。','',
        '| α | full 全维 | full真实7维 | full执行5×7 | band32 全维 | band32执行5×7 | random 全维 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for alpha in selected:
        f,b,r=[at(p,alpha)['scopes'] for p in ['full','band32','random']]
        vals=[f['all'],f['real7'],f['exec7'],b['all'],b['exec7'],r['all']]
        lines.append('| {:g} | '.format(alpha)+' | '.join('{:.4f}'.format(x['velocity_fraction_pooled']) for x in vals)+' |')
    lines+=['','### 极小α的绝对速度变化','',
        '`L0(vα,vD)`直接比较本α与数字速度，不是相对clean的两个L0标量相减。','',
        '| α | full L0(vα,vD) | band32 L0(vα,vD) | random L0(vα,vD) |',
        '|---|---:|---:|---:|']
    for alpha in [.001,.01,.1]:
        lines.append('| {:g} | '.format(alpha)+' | '.join('{:.6f}'.format(at(p,alpha)['scopes']['all']['L0_digital'])
             for p in ['full','band32','random'])+' |')
    lines+=['','![完整响应曲线](fig_response.png)','', '![靠近数字端](fig_small_alpha.png)','',
        '## 状态变化与边界','', '![逐状态响应](fig_state_response.png)','',
        '这条曲线是人工缓存干预，未保证每个中间KV都有对应可实现的输入图像。',
        '单轨迹16相关状态、一个random、三噪声；曲线不构成泛化／闭环成功率证据。',
        '真实7维和执行前5×7速度单独保留；各图的轴范围可能不同。','',
        '[23点汇总](summary.json) · [逐帧逐层指标](metrics.json) · [可复算速度与噪声](velocities.npz) · [来源校验](provenance.json)']
    (DEST/'results.md').write_text('\n'.join(lines)+'\n')
    for pattern in ['full','band32','random']:
        print(pattern)
        for alpha in [0.,.001,.01,.05,.1,.25,.5,1.]:
            r=at(pattern,alpha);s=r['scopes']['all']
            print(alpha,'KV/D',round(r['kv_global']['KV']['relative_to_digital'],6),'L0',round(s['L0_clean'],5),
                'retention',round(s['L0_clean_retention'],5),'pooled velocity progress',round(s['velocity_fraction_pooled'],5))


if __name__=='__main__':main()
