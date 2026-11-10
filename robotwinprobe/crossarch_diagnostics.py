"""Held-out real-render full-stage L0 and complete normalized action differences."""
import csv
import json
import numpy as np
from crossarch_common import DEFAULT,CAMERAS,WIRE,STAGES,prepare,request,dump

def texture_analysis(root):
    from PIL import Image
    from render_geometry import patterns
    import colorsys
    textures={'random':patterns()['random']}
    textures.update({k:np.asarray(Image.open(root/'patches'/k/'texture_0500.png').convert('RGB')) for k in ('pi','wam','joint')})
    yy,xx=np.indices((128,128));freq=np.sqrt((yy-64)**2+(xx-64)**2)/128
    window=np.outer(np.hanning(128),np.hanning(128))
    stats={};profiles={}
    for k,rgb in textures.items():
        x=rgb.astype(np.float64)/255;centered=x-x.mean((0,1),keepdims=True)
        hsv=np.array([colorsys.rgb_to_hsv(*v) for v in x.reshape(-1,3)])
        power=(np.abs(np.fft.fftshift(np.fft.fft2(centered*window[...,None],axes=(0,1)),axes=(0,1)))**2).sum(-1)
        bins=np.linspace(0,np.sqrt(.5),41)
        radial=[float(power[(freq>=a)&(freq<b)].mean()) if np.any((freq>=a)&(freq<b)) else 0 for a,b in zip(bins,bins[1:])]
        profiles[k]=radial
        stats[k]=dict(mean_RGB=x.mean((0,1)).tolist(),mean_saturation=float(hsv[:,1].mean()),
            low_frequency_power_fraction=float(power[freq<=.125].sum()/max(power.sum(),1e-15)),
            TV=float((np.abs(np.diff(x,axis=0)).mean()+np.abs(np.diff(x,axis=1)).mean())/2))
    pairwise=[]
    keys=list(textures)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            def center(k):
                x=textures[k].astype(float);return (x-x.mean((0,1),keepdims=True)).flatten()
            xa,xb=center(a),center(b)
            pairwise.append(dict(a=a,b=b,centered_correlation=float(np.dot(xa,xb)/(np.linalg.norm(xa)*np.linalg.norm(xb)))))
    dump(root/'diagnostics/texture_metrics.json',dict(textures=stats,pairwise=pairwise,
        radial_power=profiles,frequency_cycles_per_texel=((bins[:-1]+bins[1:])/2).tolist(),
        definition='Per-channel mean removal, separable Hann window, RGB-summed 2D FFT power; low band radius <= 0.125 cycles/texel. Correlation at fixed UV alignment.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(12,3))
    for ax,(k,rgb) in zip(axes,textures.items()):ax.imshow(rgb);ax.set_title(k);ax.axis('off')
    fig.tight_layout();fig.savefig(root/'diagnostics/textures.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4))
    for k,y in profiles.items():ax.semilogy((bins[:-1]+bins[1:])/2,np.maximum(y,1e-12),label=k)
    ax.set(xlabel='Radial spatial frequency (cycles / texture texel)',ylabel='Mean power per Fourier bin (arbitrary units)');ax.legend()
    fig.tight_layout();fig.savefig(root/'diagnostics/spectra.png',dpi=180);plt.close(fig)

def diagnose(root=DEFAULT):
    p=prepare(root);dest=root/'diagnostics';dest.mkdir(parents=True,exist_ok=True)
    texture_analysis(root)
    for model in ('pi','wam'):
        rows=[]
        for path in sorted((root/'post').glob('*.npz')):
            target=dest/(model+'_'+path.stem+'.json')
            if target.exists():rows.extend(json.loads(target.read_text()));continue
            local=[]
            with np.load(path) as z:
                for seed in [202610250,202610251,202610252]:
                    def response(condition):
                        return request(model,dict(command='diagnostic',state=z['state14'].copy(),prompt=str(z['prompt']),seed=seed,
                            images={w:z[n+'_'+condition].copy() for n,w in zip(CAMERAS,WIRE)}))
                    clean=response('clean')
                    for condition in ('random','pi','wam','joint'):
                        attacked=response(condition);dv=attacked['v']-clean['v'];da=attacked['action']-clean['action']
                        local.append(dict(model=model,record=path.stem,stage=str(z['stage']),condition=condition,noise_seed=seed,
                            L0_mse14=float(np.mean(dv[...,:14]**2)),L0_sse14=float(np.sum(dv[...,:14]**2)),
                            L0_full_mse=float(np.mean(dv**2)),action_mse14=float(np.mean(da[...,:14]**2)),
                            yaw_deg=float(z['yaw_deg']),light=str(z['light'])))
            dump(target,local);rows.extend(local)
        with (dest/(model+'_responses.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        summary=[]
        for condition in ('random','pi','wam','joint'):
            for stage in STAGES:
                selected=[r for r in rows if r['condition']==condition and r['stage']==stage and r['yaw_deg']==0]
                summary.append(dict(condition=condition,stage=stage,count=len(selected),
                    L0_mse14=float(np.mean([r['L0_mse14'] for r in selected])),
                    action_mse14=float(np.mean([r['action_mse14'] for r in selected]))))
        dump(dest/(model+'_summary.json'),summary)

if __name__=='__main__':diagnose()
