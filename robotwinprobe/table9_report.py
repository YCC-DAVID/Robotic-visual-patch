"""Compare learned texture structure and true-render cross-placement responses."""
import argparse
import json
from pathlib import Path

import numpy as np
from radius_plan import dump
from table9_plan import sha


def spectrum(rgb):
    q=rgb.astype(np.float64)/255
    centered=q-q.mean((0,1),keepdims=True)
    power=(abs(np.fft.fft2(centered,axes=(0,1)))**2).mean(2)
    fy,fx=np.meshgrid(np.fft.fftfreq(q.shape[0]),np.fft.fftfreq(q.shape[1]),indexing='ij')
    radius=np.sqrt(fx*fx+fy*fy);edges=np.linspace(0,np.sqrt(.5),33)
    radial=np.array([power[(radius>=a)&(radius<b)].sum() for a,b in zip(edges[:-1],edges[1:])])
    radial/=max(power.sum(),1e-12)
    return (edges[:-1]+edges[1:])/2,radial,float(power[radius>=.25].sum()/max(power.sum(),1e-12))


def report(root):
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=root/'comparison';dest.mkdir(exist_ok=True)
    sites=[f'site{i:02d}' for i in range(9)]
    paths=[root/'sites'/s/'train/texture_0500.png' for s in sites]
    assert all(p.exists() for p in paths),'All nine fixed final checkpoints are needed'
    images=[np.asarray(Image.open(p).convert('RGB')) for p in paths]
    fig,axs=plt.subplots(3,3,figsize=(9,9))
    for ax,s,q in zip(axs.flat,sites,images):ax.imshow(q);ax.set_title(s);ax.axis('off')
    fig.suptitle('Nine independent learned textures | 10 cm | final step 500')
    fig.tight_layout();fig.savefig(dest/'nine_textures.png',dpi=160);plt.close(fig)
    rows=[];spectra=[]
    fig,ax=plt.subplots(figsize=(9,5))
    for s,q in zip(sites,images):
        x,y,high=spectrum(q);spectra.append(y);ax.plot(x,y,label=s)
        rows.append(dict(site=s,mean_rgb=(q.mean((0,1))/255).tolist(),std_rgb=(q.std((0,1))/255).tolist(),
            high_frequency_power_fraction_r_ge_025=high,
            mean_adjacent_pixel_absolute_difference=float((abs(np.diff(q.astype(float),axis=0)).mean()+abs(np.diff(q.astype(float),axis=1)).mean())/2/255)))
    ax.set(xlabel='Radial frequency (cycles / texture pixel)',ylabel='Fraction of centered RGB power per radial bin',
        title='World texture spectra; before rendering')
    ax.legend(ncol=3);fig.tight_layout();fig.savefig(dest/'radial_spectra.png',dpi=160);plt.close(fig)
    flat=np.stack([q.astype(float).ravel() for q in images]);flat-=flat.mean(1,keepdims=True)
    norms=np.linalg.norm(flat,axis=1);corr=(flat@flat.T)/np.maximum(norms[:,None]*norms[None,:],1e-12)
    a=np.stack(spectra);spec_cos=(a@a.T)/np.maximum(np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(a,axis=1)[None,:],1e-12)
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for ax,m,title in zip(axs,[corr,spec_cos],['Pixel correlation (fixed UV alignment)','Radial power-profile cosine similarity']):
        im=ax.imshow(m,vmin=-1 if m is corr else 0,vmax=1,cmap='coolwarm' if m is corr else 'viridis')
        ax.set_xticks(range(9),sites,rotation=60);ax.set_yticks(range(9),sites);ax.set_title(title);fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(dest/'texture_similarity.png',dpi=160);plt.close(fig)
    probes=[];complete=True
    for s in sites:
        p=root/'sites'/s/'cross_probe.json'
        if not p.exists():complete=False;continue
        d=json.loads(p.read_text());assert d['complete'];probes.extend(d['rows'])
    matrices={};diagonal=[]
    if complete:
        methods=sites+['random']
        fig,axs=plt.subplots(1,2,figsize=(15,6))
        for ax,condition in zip(axs,['nominal','joint']):
            matrix=np.array([[np.mean([r['L0_mean'] for r in probes if r['method']==m and r['target_site']==s and r['condition']==condition]) for s in sites] for m in methods])
            assert np.isfinite(matrix).all()
            matrices[condition]=matrix.tolist()
            im=ax.imshow(np.log10(np.maximum(matrix,1e-8)),cmap='magma',aspect='auto')
            ax.set_xticks(range(9),sites,rotation=60);ax.set_yticks(range(10),methods)
            ax.set(xlabel='Test placement',ylabel='Training origin / Random',title=condition+' | true-render L0')
            fig.colorbar(im,ax=ax,label='log10(mean L0)')
            for s in sites:
                trained=[r['L0_mean'] for r in probes if r['method']==s and r['target_site']==s and r['condition']==condition]
                random=[r['L0_mean'] for r in probes if r['method']=='random' and r['target_site']==s and r['condition']==condition]
                diagonal.append(dict(site=s,condition=condition,L0_mean=float(np.mean(trained)),L0_median=float(np.median(trained)),
                    random_L0_mean=float(np.mean(random)),random_L0_median=float(np.median(random))))
        fig.tight_layout();fig.savefig(dest/'cross_placement_L0.png',dpi=160);plt.close(fig)
    dump(dest/'summary.json',dict(textures_complete=True,cross_probe_complete=complete,texture_hashes={s:sha(p) for s,p in zip(sites,paths)},
        texture_stats=rows,pixel_correlation=corr.tolist(),radial_profile_cosine=spec_cos.tolist(),cross_matrices=matrices,diagonal=diagonal))
    lines=['# 九个位置独立训练：纹理与真实渲染比较','',
        '每个位置独立优化一张 10×10 cm Q128 patch，各训练 500 次更新；任一场景只有一张。三路相机观察同一个世界纹理。',
        '', '![九张纹理](nine_textures.png)', '', '![径向功率谱](radial_spectra.png)', '', '![纹理相似度](texture_similarity.png)',
        '', '频谱横轴是 cycles/texture pixel，纵轴为去均值后 RGB 功率在径向频率区间内的占比。像素相关性依赖固定 UV 对齐；频谱相似并不能证明具有相同攻击结构。',
        '',f'真实渲染跨位置探针完成：{complete}。']
    if complete:
        lines+=['','![真实渲染跨位置L0](cross_placement_L0.png)','',
            '固定最终第 500 步纹理，在 9 个位置分别测试。名义工况和新的联合 EOT 工况，各使用同一组 9 个验证状态及 3 份未参与训练的动作 noise，Random 与训练纹理严格配对。9 帧来自 2 条轨迹；L0 是速度场差异，不是任务 ASR。',
            '', '|位置|工况|训练纹理 L0 均值|中位数|Random 均值|中位数|','|---|---|---:|---:|---:|---:|']
        for r in diagonal:lines.append(f"|{r['site']}|{r['condition']}|{r['L0_mean']:.5g}|{r['L0_median']:.5g}|{r['random_L0_mean']:.5g}|{r['random_L0_median']:.5g}|")
    lines+=['','几何匹配和左右腕可见面积见[几何核验](../geometry/report.md)。仅这一轮同初始化实验不能区分位置效应与训练随机性的一般规律；训练抽样在九组间保持一致。']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(textures_complete=True,cross_probe_complete=complete,report=str(dest/'report.md'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    report(Path(p.parse_args().root).resolve())
