"""Paired digital/render pixel diagnosis, held-out surrogate fits, no model.

All comparisons reuse E15's identical simulator states. Includes optimized and
random patterns for full and band32 at 26 cm. Surrogate fits describe mixtures
of raster/photometric effects; they do not isolate renderer causal mechanisms.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SIGMAS = [0.,.25,.5,.75,1.,1.5,2.]


def paired_stats(a,b,mask):
    x,y = a[mask],b[mask]
    err = y-x
    corrs = []
    for c in range(3):
        xc,yc = x[:,c]-x[:,c].mean(),y[:,c]-y[:,c].mean()
        den = np.linalg.norm(xc)*np.linalg.norm(yc)
        corrs.append(float(xc@yc/den) if np.linalg.norm(xc)>1e-10 and np.linalg.norm(yc)>1e-10 else None)
    return {'MAE':float(np.abs(err).mean()),'RMSE':float(np.sqrt((err**2).mean())),
            'mean_rgb_digital':x.mean(0).tolist(),'mean_rgb_render':y.mean(0).tolist(),
            'std_rgb_digital':x.std(0).tolist(),'std_rgb_render':y.std(0).tolist(),
            'correlation_mean_channels':float(np.mean([c for c in corrs if c is not None])) if any(c is not None for c in corrs) else None}


def tv(image,mask):
    horizontal = mask[:,1:]&mask[:,:-1]
    vertical = mask[1:]&mask[:-1]
    dx = np.abs(image[:,1:]-image[:,:-1])[horizontal]
    dy = np.abs(image[1:]-image[:-1])[vertical]
    return float((dx.sum()+dy.sum())/max(dx.size+dy.size,1))


def affine_fit(x,targets,masks,indices):
    xx = np.concatenate([x[masks[i]] for i in indices])
    yy = np.concatenate([targets[i][masks[i]] for i in indices])
    xmean,ymean = xx.mean(0),yy.mean(0)
    centered = xx-xmean
    gain = (centered*(yy-ymean)).sum(0)/np.maximum((centered**2).sum(0),1e-12)
    bias = ymean-gain*xmean
    pred = np.clip(x*gain+bias,0,1)
    rmses = [float(np.sqrt(((pred[masks[i]]-targets[i][masks[i]])**2).mean())) for i in indices]
    return pred,{'gain_rgb':gain.tolist(),'bias_rgb':bias.tolist(),'train_RMSE':float(np.mean(rmses))}


def roi_square(mask):
    distance = cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    for side in [24,16,8]:
        sums = cv2.matchTemplate(mask.astype(np.float32),np.ones((side,side),np.float32),cv2.TM_CCORR)
        yy,xx = np.where(sums>side*side-.05)
        if len(xx):
            best = np.argmax(distance[yy+side//2,xx+side//2])
            return int(xx[best]),int(yy[best]),side
    raise ValueError('No square ROI fully inside patch mask')


def spectrum(crop):
    n = crop.shape[0]
    window = np.outer(np.hanning(n),np.hanning(n))[...,None]
    centered = crop-crop.mean((0,1),keepdims=True)
    power = np.abs(np.fft.fft2(centered*window,axes=(0,1)))**2
    power = power.sum(-1)
    f = np.fft.fftfreq(n)
    radius = np.sqrt(f[:,None]**2+f[None,:]**2)
    bins = np.linspace(0,np.sqrt(.5),15)
    curve = [float(power[(radius>=lo)&(radius<hi)].sum()) for lo,hi in zip(bins[:-1],bins[1:])]
    return {'frequency_centers':((bins[:-1]+bins[1:])/2).tolist(),
            'radial_energy':curve,'high_frequency_energy':float(power[radius>.25].sum()),
            'total_centered_energy':float(power.sum()),'window':'Hann, mean-subtracted RGB',
            'high_frequency_threshold_cycles_per_pixel':.25}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs',default='kv_render_probe')
    ap.add_argument('--out',default='render_pixel_diagnosis')
    args = ap.parse_args()
    source = ROOT/'pi05probe/out'/args.inputs
    dest = ROOT/'pi05probe/out'/args.out
    dest.mkdir(parents=True,exist_ok=True)
    pool = np.load(source/'inputs.npz',allow_pickle=False)
    steps = pool['steps'].tolist()
    n = len(steps)
    train,test = list(range(n//2)),list(range(n//2,n))
    random = np.random.RandomState(20260902).randint(0,256,(224,224,3)).astype(np.uint8)
    assert np.array_equal(pool['drift__random_render_base'],pool['band32__random_render_base'])
    cases,examples = {},{}
    for tag in ['drift','band32']:
        patch = np.load(ROOT/'pi05probe/out'/('drift_patch_'+tag+'.npz'),allow_pickle=True)
        optimized = np.clip(patch['P'].transpose(1,2,0)*255,0,255).astype(np.uint8)
        quad = patch['quad224'].astype(np.float32)[:,::-1]
        H = cv2.getPerspectiveTransform(quad,np.array([[0,0],[127,0],[127,127],[0,127]],np.float32))
        mask = pool[tag+'__mask'].astype(bool)
        interiors = np.array([cv2.erode(m.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool) for m in mask])
        assert all(m.sum()>100 for m in interiors)
        for label,pattern in [('optimized',optimized),('random',random)]:
            name=tag+'/'+label
            digital = pool[tag+('__digital' if label=='optimized' else '__random_digital')]
            rendered = pool[tag+'__'+label+'_render_base'].astype(np.float64)/255
            assert np.array_equal(digital,np.where(mask[...,None],pattern,pool['clean_base']))
            pattern = pattern.astype(np.float64)/255
            digital = digital.astype(np.float64)/255
            texture = np.asarray(Image.open(source/'textures'/(tag+'_'+label+'.png')))[::-1].copy()
            roundtrip = cv2.warpPerspective(texture,np.linalg.inv(H),(224,224),flags=cv2.INTER_LINEAR).astype(np.float64)/255
            rows = []
            for i,step in enumerate(steps):
                inner,boundary = interiors[i],mask[i]&~interiors[i]
                dilated = cv2.dilate(mask[i].astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
                x,y,side = roi_square(inner)
                specd = spectrum(digital[i,y:y+side,x:x+side])
                specr = spectrum(rendered[i,y:y+side,x:x+side])
                row={'step':step,'mask_px':int(mask[i].sum()),'interior_px':int(inner.sum()),
                     'all_patch':paired_stats(digital[i],rendered[i],mask[i]),
                     'interior4':paired_stats(digital[i],rendered[i],inner),
                     'boundary4':paired_stats(digital[i],rendered[i],boundary),
                     'outside_patch':paired_stats(digital[i],rendered[i],~mask[i]),
                     'far_outside8':paired_stats(digital[i],rendered[i],~dilated),
                     'TV_digital':tv(digital[i],inner),'TV_render':tv(rendered[i],inner),
                     'spectrum_roi_xywh':[x,y,side,side],'spectrum_digital':specd,'spectrum_render':specr}
                rows.append(row)
            # Fit each family on first-half states; evaluate on second-half
            # states. This is held-out replay data, not independent episodes.
            fits,predictions = {},{}
            candidates=[]
            for sigma in SIGMAS:
                blurred = pattern if sigma==0 else cv2.GaussianBlur(pattern,(0,0),sigma)
                pred,fit = affine_fit(blurred,rendered,interiors,train)
                fit['sigma_px'] = sigma
                candidates.append((fit['train_RMSE'],pred,fit,blurred))
            _,best_pred,best_fit,best_blur = min(candidates,key=lambda x:x[0])
            direct_pred,direct_fit = affine_fit(pattern,rendered,interiors,train)
            rt_pred,rt_fit = affine_fit(roundtrip,rendered,interiors,train)
            shifted=[]
            for dx in np.arange(-2.,2.01,.5):
                for dy in np.arange(-2.,2.01,.5):
                    moved=cv2.warpAffine(roundtrip,np.float32([[1,0,dx],[0,1,dy]]),(224,224),flags=cv2.INTER_LINEAR)
                    pred,fit = affine_fit(moved,rendered,interiors,train)
                    fit.update(dx_px=dx,dy_px=dy)
                    shifted.append((fit['train_RMSE'],pred,fit))
            _,shift_pred,shift_fit = min(shifted,key=lambda x:x[0])
            constant_rgb = np.concatenate([rendered[i][interiors[i]] for i in train]).mean(0)
            predictions={'identity':pattern,'constant_rgb':np.broadcast_to(constant_rgb,pattern.shape),
                         'color_affine_only':direct_pred,
                         'best_gaussian_without_color':best_blur,'gaussian_plus_affine':best_pred,
                         'export_128_roundtrip':roundtrip,'roundtrip_plus_affine':rt_pred,
                         'roundtrip_shift_affine':shift_pred}
            parameters={'identity':{},'constant_rgb':{'rgb':constant_rgb.tolist()},'color_affine_only':direct_fit,
                        'best_gaussian_without_color':{'sigma_px':best_fit['sigma_px']},
                        'gaussian_plus_affine':best_fit,'export_128_roundtrip':{},
                        'roundtrip_plus_affine':rt_fit,'roundtrip_shift_affine':shift_fit}
            for family,pred in predictions.items():
                validation=[paired_stats(pred,rendered[i],interiors[i]) for i in test]
                fits[family]={'parameters':parameters[family],
                    'heldout_RMSE':float(np.mean([v['RMSE'] for v in validation])),
                    'heldout_MAE':float(np.mean([v['MAE'] for v in validation])),
                    'heldout_correlation':float(np.mean([v['correlation_mean_channels'] for v in validation if v['correlation_mean_channels'] is not None])) if any(v['correlation_mean_channels'] is not None for v in validation) else None}
            aggregate={'interior_MAE':float(np.mean([r['interior4']['MAE'] for r in rows])),
                       'interior_correlation':float(np.mean([r['interior4']['correlation_mean_channels'] for r in rows])),
                       'interior_mean_digital':float(np.mean([r['interior4']['mean_rgb_digital'] for r in rows])),
                       'interior_mean_render':float(np.mean([r['interior4']['mean_rgb_render'] for r in rows])),
                       'TV_retention':float(np.mean([r['TV_render']/r['TV_digital'] for r in rows])),
                       'high_frequency_absolute_energy_retention':float(np.mean([
                           r['spectrum_render']['high_frequency_energy']/r['spectrum_digital']['high_frequency_energy'] for r in rows])),
                       'high_frequency_fraction_digital':float(np.mean([
                           r['spectrum_digital']['high_frequency_energy']/r['spectrum_digital']['total_centered_energy'] for r in rows])),
                       'high_frequency_fraction_render':float(np.mean([
                           r['spectrum_render']['high_frequency_energy']/r['spectrum_render']['total_centered_energy'] for r in rows])),
                       'outside_patch_MAE':float(np.mean([r['outside_patch']['MAE'] for r in rows])),
                       'far_outside8_MAE':float(np.mean([r['far_outside8']['MAE'] for r in rows]))}
            cases[name]={'aggregate':aggregate,'frames':rows,'surrogate_fits':fits,
                         'gaussian_fit_grid':[{k:v for k,v in fit.items()} for _,_,fit,_ in candidates]}
            fi=steps.index(10) if 10 in steps else 0
            examples[name]=(digital[fi],rendered[fi],np.where(mask[fi,...,None],shift_pred,digital[fi]),mask[fi])
            print(name,json.dumps(aggregate),'best_sigma',best_fit['sigma_px'],'shift',shift_fit['dx_px'],shift_fit['dy_px'],flush=True)
    metadata={'complete':True,'frames':n,'steps':steps,'train_steps':[steps[i] for i in train],
              'heldout_steps':[steps[i] for i in test],'input_sha256':hashlib.sha256((source/'inputs.npz').read_bytes()).hexdigest(),
              'cases':cases,'model_executed':False,
              'random_control_same_in_full_and_band32':True,
              'notes':'Pure pixel diagnostics. Interior eroded 4px; FFT uses a fully interior square. '
                      'Affine, blur and shift fits confound renderer factors and are not causal identifications. '
                      'Export roundtrip inverts the saved exporter homography and is not the MuJoCo rasterizer. '
                      'Train/test are two portions of one replay trajectory, not independent episodes.'}
    (dest/'metrics.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(len(cases),5,figsize=(16,3.2*len(cases)),squeeze=False)
    for i,(name,(di,re,pr,mask)) in enumerate(examples.items()):
        ys,xs=np.where(mask);y0,y1=max(0,ys.min()-3),min(224,ys.max()+4);x0,x1=max(0,xs.min()-3),min(224,xs.max()+4)
        images=[di,re,np.sqrt(((re-di)**2).mean(-1)),pr,np.sqrt(((re-pr)**2).mean(-1))]
        for j,(im,title) in enumerate(zip(images,['Digital','MuJoCo render','RMS difference','Roundtrip + shift + affine fit','Remaining RMS difference'])):
            axes[i,j].imshow(im[y0:y1,x0:x1],interpolation='nearest',**({'cmap':'magma','vmin':0,'vmax':.35} if j in [2,4] else {}))
            axes[i,j].set_title(name+' / '+title,fontsize=8);axes[i,j].axis('off')
    fig.suptitle('Same-state patch ROI at step 10 / error scale 0..0.35 in RGB [0,1]\nSurrogate is a pixel fit, not an attack result; fitting excludes the 4px boundary')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'fig_pixel_pairs.png',dpi=150,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(len(cases),2,figsize=(12,3.1*len(cases)),squeeze=False)
    for i,(name,case) in enumerate(cases.items()):
        rows=case['frames']
        for kind,color in [('digital','#b8373b'),('render','#2177a5')]:
            spectra=[r['spectrum_'+kind] for r in rows]
            freq=spectra[0]['frequency_centers'];energy=np.mean([s['radial_energy'] for s in spectra],0)
            axes[i,0].plot(freq,energy,label=kind,color=color)
        axes[i,0].set_yscale('log');axes[i,0].set_title(name+' / windowed RGB spectrum')
        axes[i,0].set_xlabel('Radial frequency (cycles / pixel)');axes[i,0].set_ylabel('Unnormalized spectral energy');axes[i,0].legend()
        axes[i,1].plot(steps,[r['TV_render']/r['TV_digital'] for r in rows],label='TV retention')
        axes[i,1].plot(steps,[r['spectrum_render']['high_frequency_energy']/r['spectrum_digital']['high_frequency_energy'] for r in rows],label='HF energy retention (>0.25 cy/px)')
        axes[i,1].set_title(name+' / retained local contrast');axes[i,1].set_xlabel('Environment step');axes[i,1].legend(fontsize=8)
        for ax in axes[i]:ax.grid(alpha=.2)
    fig.suptitle('Digital vs render texture statistics / same interior ROI per state\nSpectral attenuation mixes resampling, shading and alignment; not an identified optical filter')
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(dest/'fig_frequency.png',dpi=150,bbox_inches='tight');plt.close(fig)
    lines=['# 同状态数字／渲染像素诊断','',
           '26 cm full／band32，优化／random；16个状态，复用E15输入。没有模型推理或训练。',
           'RGB范围[0,1]。内部排除4px边界；TV与频谱在内部统计；HF指径向频率>0.25 cycles/pixel。','',
           '| 图案 | 内部MAE | 内部相关 | 平均RGB数字→渲染 | TV保留 | HF绝对能量保留 |',
           '|---|---:|---:|---:|---:|---:|']
    for name,c in cases.items():
        a=c['aggregate'];lines.append('| {} | {:.4f} | {:.3f} | {:.3f}→{:.3f} | {:.1%} | {:.1%} |'.format(name,
            a['interior_MAE'],a['interior_correlation'],a['interior_mean_digital'],a['interior_mean_render'],a['TV_retention'],a['high_frequency_absolute_energy_retention']))
    lines+=['','## 简单变换对真实渲染的拟合','',
            '在step10–45拟合，step50–85检查。它们是同一条轨迹的两段，不是独立测试episode。',
            '每个图案各自拟合参数；不可把拟合增益/等效模糊当作独立光照/滤波参数。','',
            '| 图案 | 不变RMSE | 常量RGB基线 | 仅颜色affine | Gaussian+affine | 128往返+affine | 128往返+平移+affine |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name,c in cases.items():
        fits=c['surrogate_fits'];lines.append('| {} | {} |'.format(name,' | '.join('{:.4f}'.format(fits[f]['heldout_RMSE']) for f in ['identity','constant_rgb','color_affine_only','gaussian_plus_affine','roundtrip_plus_affine','roundtrip_shift_affine'])))
    lines+=['','[配对图／误差图](fig_pixel_pairs.png) · [频谱／TV图](fig_frequency.png) · [逐帧指标与参数](metrics.json)',
            '', '较小像素拟合误差只能帮助挑选模型对照，不能说明攻击已恢复，也不能单独确定成像主因。',
            '下一步对数字输入加入单项变换、对渲染输入恢复指定分量，各自同时报告像素、视觉特征／KV、early L0并配random。']
    (dest/'results.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
