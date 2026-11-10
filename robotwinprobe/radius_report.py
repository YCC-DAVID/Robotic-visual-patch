"""Paired ASR and seed-cluster intervals; completed common cells only."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from radius_plan import dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);args=ap.parse_args();root=Path(args.root).resolve()
    cohort=json.loads((root/'cohort.json').read_text());seeds=cohort['seeds']
    methods=['baseline','r01','r03','r05','random']
    textures=json.loads((root/'textures.json').read_text())
    hashes={m:hashlib.sha256(Path(textures[m]).read_bytes()).hexdigest() for m in methods}
    clean={s:json.loads((root/'eval/clean'/f'seed{s}_p00.json').read_text()) for s in seeds}
    allrows={m:{} for m in methods}
    for m in methods:
        for s in seeds:
            for pi in range(11):
                p=root/'eval'/m/f'seed{s}_p{pi:02d}.json'
                if not p.exists():continue
                r=json.loads(p.read_text());c=clean[s]
                assert r['texture_sha256']==hashes[m],(m,s,pi,'texture changed')
                for k in ['initial_rgb_sha256','initial_state_sha256','first_noise_seed','prompt']:
                    assert r[k]==c[k],(m,s,pi,k)
                allrows[m][(s,pi)]=r
    common=set.intersection(*(set(r) for r in allrows.values()))
    for cell in common:
        assert all(allrows[m][cell]['center']==allrows['baseline'][cell]['center'] for m in methods)
    summary=[]
    bins=[('origin',lambda p,d:p==0),('disk_0_5cm',lambda p,d:p>0),
          ('0_1cm',lambda p,d:p>0 and d<=1),('1_3cm',lambda p,d:p>0 and 1<d<=3),
          ('3_5cm',lambda p,d:p>0 and 3<d<=5+1e-9)]
    for name,select in bins:
        cells=sorted(k for k in common if select(k[1],allrows['baseline'][k]['radius_cm']))
        for m in methods:
            byseed=[];baseline_byseed=[]
            for s in seeds:
                keys=[k for k in cells if k[0]==s]
                den=len(keys)*int(clean[s]['success'])
                num=sum(not allrows[m][k]['success'] for k in keys) if clean[s]['success'] else 0
                byseed.append([num,den])
                baseline_byseed.append(sum(not allrows['baseline'][k]['success'] for k in keys) if clean[s]['success'] else 0)
            a=np.array(byseed);num,den=a.sum(0);rng=np.random.RandomState(20261012)
            indices=rng.randint(0,len(seeds),(2000,len(seeds)))
            boot=a[indices].sum(1);valid=boot[:,1]>0
            ci=np.quantile(boot[valid,0]/boot[valid,1],[.025,.975]).tolist() if valid.any() else None
            diff=a[:,0]-np.array(baseline_byseed)
            delta_ci=np.quantile(diff[indices].sum(1)[valid]/boot[valid,1],[.025,.975]).tolist() if valid.any() else None
            summary.append({'region':name,'method':m,'attacks':int(num),'eligible':int(den),
                'asr':float(num/den) if den else None,'seed_cluster_bootstrap_95':ci,
                'paired_delta_vs_baseline':float(diff.sum()/den) if den else None,
                'paired_delta_cluster_95':delta_ci,
                'task_successes':sum(allrows[m][k]['success'] for k in cells),'episodes':len(cells)})
    complete=len(common)==len(seeds)*11
    dump(root/'summary.json',{'complete':complete,'common_paired_cells':len(common),
        'total_paired_cells':len(seeds)*11,'clean_successes':sum(c['success'] for c in clean.values()),
        'initial_states':len(seeds),'rows':summary,
        'limits':'one training seed; cluster intervals quantify initial-state sampling only; partial reports use shared completed cells'})
    lines=['# 连续位置EOT：共同测试位置的结果','',f'完成：{complete}；共同单元 {len(common)}/{len(seeds)*11}。',
        '', '| 测试区域 | 方法 | 攻击成功/clean成功位置对 | ASR | 场景聚类95%区间 |','|---|---|---:|---:|---|']
    for r in summary:
        rate='—' if r['asr'] is None else f"{100*r['asr']:.1f}%"
        ci='—' if r['seed_cluster_bootstrap_95'] is None else '–'.join(f'{100*x:.1f}%' for x in r['seed_cluster_bootstrap_95'])
        lines.append(f"| {r['region']} | {r['method']} | {r['attacks']}/{r['eligible']} | {rate} | {ci} |")
    (root/'results.md').write_text('\n'.join(lines)+'\n')
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,5,figsize=(20,4),constrained_layout=True)
        edges=np.linspace(-5,5,6)
        for ax,m in zip(axes,methods):
            den=np.zeros((5,5));num=np.zeros((5,5))
            for s,pi in common:
                if pi==0 or not clean[s]['success']:continue
                r=allrows[m][(s,pi)];x,y=np.array(r['center'][:2])*100
                ix=np.clip(np.searchsorted(edges,x,side='right')-1,0,4);iy=np.clip(np.searchsorted(edges,y,side='right')-1,0,4)
                den[iy,ix]+=1;num[iy,ix]+=not r['success']
            rates=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
            im=ax.imshow(rates,origin='lower',extent=(-5,5,-5,5),vmin=0,vmax=1,cmap='magma')
            for iy in range(5):
                for ix in range(5):
                    if den[iy,ix]:ax.text(edges[ix]+1,edges[iy]+1,f'{int(num[iy,ix])}/{int(den[iy,ix])}',ha='center',va='center',fontsize=7,color='cyan')
            ax.set_title(m);ax.set_xlabel('World X offset (cm)');ax.set_ylabel('World Y offset (cm)')
        fig.colorbar(im,ax=axes,label='Paired ASR');fig.savefig(root/'position_asr.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
