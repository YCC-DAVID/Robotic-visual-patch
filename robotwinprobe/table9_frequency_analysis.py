"""Reproduce G1 texture metrics and compare nine-site spectra with paired ASR.

CPU-only, read-only with respect to training and rollout artifacts. Frequency
fractions describe the original UV texture, not the renderer's camera output.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from scipy.stats import spearmanr

import bootstrap as B
from radius_plan import dump
from table9_plan import DEFAULT, sha


def correlation(a,b):
    a=a.ravel()-a.mean();b=b.ravel()-b.mean()
    denominator=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/denominator) if denominator>0 else None


def neighbor_rgb(rgb,lag):
    pairs=[]
    for c in range(3):
        pairs.extend([correlation(rgb[:,:-lag,c],rgb[:,lag:,c]),
                      correlation(rgb[:-lag,:,c],rgb[lag:,:,c])])
    return float(np.mean(pairs)) if all(v is not None for v in pairs) else None


def spectrum(rgb,windowed=True):
    h,w,_=rgb.shape
    window=np.outer(np.hanning(h),np.hanning(w)) if windowed else np.ones((h,w))
    mean=(rgb*window[...,None]).sum((0,1))/window.sum()
    centered=(rgb-mean)*window[...,None]
    power=np.square(abs(np.fft.fft2(centered,axes=(0,1)))).sum(-1)
    power[0,0]=0
    radius=np.hypot(np.fft.fftfreq(h)[:,None],np.fft.fftfreq(w)[None,:])
    bands={'low':radius<.125,'mid':(radius>=.125)&(radius<.25),'high':radius>=.25}
    total=power.sum()
    # Parseval normalization: RGB/channel spatial variance, Hann-energy corrected.
    normalizer=h*w*np.square(window).sum()*3
    out={'total_ac_power':float(total/normalizer)}
    for name,mask in bands.items():
        band=power[mask].sum()
        out[name+'_fraction']=float(band/total) if total>0 else None
        out[name+'_absolute_power']=float(band/normalizer)
    return out


def metrics(rgb):
    hi=rgb.max(-1);lo=rgb.min(-1)
    saturation=np.divide(hi-lo,hi,out=np.zeros_like(hi),where=hi>0)
    windows=sliding_window_view(rgb,(7,7),axis=(0,1))
    local=float(np.sqrt(windows.var(axis=(-2,-1)).mean(axis=-1)).mean())
    return dict(corr_rgb_1px=neighbor_rgb(rgb,1),corr_rgb_4px=neighbor_rgb(rgb,4),
        mean_saturation=float(saturation.mean()),local_rgb_rms_7x7=local,
        mean_rgb=rgb.mean((0,1)).tolist(),hann=spectrum(rgb),unwindowed=spectrum(rgb,False))


def flatten(row):
    m=row['native128'];h=m['hann']
    return dict(id=row['id'],group=row['group'],path=row['path'],sha256=row['sha256'],
        corr_rgb_1px=m['corr_rgb_1px'],corr_rgb_4px=m['corr_rgb_4px'],
        low_fraction=h['low_fraction'],mid_fraction=h['mid_fraction'],high_fraction=h['high_fraction'],
        low_absolute_power=h['low_absolute_power'],total_ac_power=h['total_ac_power'],
        mean_saturation=m['mean_saturation'],local_rgb_rms_7x7=m['local_rgb_rms_7x7'])


def analyze(root):
    ev=json.loads((root/'summary.json').read_text());assert ev['complete']
    by={(r['site'],r['condition'],r['method']):r for r in ev['rows']}
    paths=[('random','case_random',root/'random.png')]
    paths.extend((f'site{i:02d}','case_trained',DEFAULT/'sites'/f'site{i:02d}'/'train/texture_0500.png') for i in range(9))
    reference=B.ROOT/'result/g1_print_10cm_20261013/textures_128'
    paths.extend((name,'g1_reference',reference/(filename+'.png'))
                 for name,filename in [('G1_P00','P00'),('G1_P07','P07')])
    rows=[]
    for name,group,path in paths:
        rgb=np.asarray(Image.open(path).convert('RGB'),dtype=np.float64)/255
        assert rgb.shape==(128,128,3)
        down=rgb.reshape(16,8,16,8,3).mean((1,3))
        rows.append(dict(id=name,group=group,path=str(path),sha256=sha(path),
                         native128=metrics(rgb),block_average16=metrics(down)))
    # Validate that the other conversation's values use exactly this definition.
    references={r['id']:r for r in rows if r['group']=='g1_reference'}
    assert abs(references['G1_P00']['native128']['hann']['low_fraction']-.0513)<.0001
    assert abs(references['G1_P07']['native128']['hann']['low_fraction']-.8045)<.0001
    assert abs(references['G1_P07']['native128']['corr_rgb_1px']-.898)<.0006
    cross=json.loads((DEFAULT/'comparison/summary.json').read_text())
    for row in rows:
        if row['group']!='case_trained':continue
        site=row['id'];i=int(site[4:]);outcomes={}
        for condition in ['nominal','position','joint']:
            trained=by[(site,condition,'trained')];control=by[(site,condition,'random')]
            assert trained['clean_success_pairs']==control['clean_success_pairs']
            outcomes[condition]=dict(ASR=trained['ASR'],random_ASR=control['ASR'],
                ASR_excess=trained['ASR']-control['ASR'],attacks=trained['attack_successes'],
                random_attacks=control['attack_successes'],denominator=trained['clean_success_pairs'],
                task_SR=trained['task_SR'],random_task_SR=control['task_SR'],
                origin_retention=trained['origin_retention'])
        probes={}
        for condition,matrix in cross['cross_matrices'].items():
            matrix=np.asarray(matrix)
            probes[condition]=dict(own_L0=float(matrix[i,i]),own_random_L0=float(matrix[9,i]),
                same_nine_placements_L0=float(matrix[i].mean()),
                same_nine_placements_excess=float((matrix[i]-matrix[9]).mean()))
        row.update(outcomes=outcomes,open_loop=probes)
    trained=[r for r in rows if r['group']=='case_trained']
    correlations=[]
    for predictor in ['low_fraction','low_absolute_power','corr_rgb_1px','corr_rgb_4px',
                      'mean_saturation','local_rgb_rms_7x7']:
        x=[r['native128']['hann'][predictor] if predictor in ['low_fraction','low_absolute_power']
           else r['native128'][predictor] for r in trained]
        for condition in ['nominal','position','joint']:
            for outcome in ['ASR','ASR_excess']:
                y=[r['outcomes'][condition][outcome] for r in trained]
                correlations.append(dict(predictor=predictor,endpoint='closed_loop',condition=condition,
                    outcome=outcome,spearman_rho=float(spearmanr(x,y).statistic),textures=9))
        for condition in ['nominal','joint']:
            for outcome in ['own_L0','same_nine_placements_L0']:
                y=[r['open_loop'][condition][outcome] for r in trained]
                correlations.append(dict(predictor=predictor,endpoint='open_loop',condition=condition,
                    outcome=outcome,spearman_rho=float(spearmanr(x,y).statistic),textures=9))
    dest=root/'frequency_analysis';dest.mkdir(exist_ok=True)
    protocol=dict(texture_resolution=[128,128],color_space='normalized sRGB RGB; no linearization',
        neighbor_correlation='Pearson within each RGB channel, averaged across channels and horizontal/vertical directions',
        spectrum='per-channel Hann-weighted mean removed, separable Hann window, RGB FFT powers summed, DC set to zero',
        frequencies=dict(low='radius < 0.125 cycles/texture pixel',mid='0.125 <= radius < 0.25',high='radius >= 0.25'),
        absolute_power='FFT band power / (height * width * sum(Hann^2) * 3); window-energy corrected per-channel AC variance',
        local_contrast='mean across valid 7x7 windows of sqrt(mean-channel spatial variance)',
        saturation='HSV S=(max RGB-min RGB)/max RGB, zero for black; mean across pixels',
        downsample='illustrative 8x8 block average to 16x16, not real renderer output',
        association='Spearman across nine trained textures only; Random and G1 reference excluded from correlation',
        limitations=['Different own placements confound closed-loop comparisons.',
            'Closed-loop cohort has 10 initial states, 4 nominal clean successes; repeated placements are not independent initial states.',
            'Feature/outcome comparisons are exploratory; no causal inference or significance claims.',
            'Texture spectrum differs from rendered image spectrum; smoothing can increase low-frequency fraction without increasing attack.',
            'Relative spectral fraction and absolute band power are reported separately.'])
    dump(dest/'summary.json',dict(protocol=protocol,rows=rows,correlations=correlations,
        reference_reproduction_passed=True,closedloop_summary_sha256=sha(root/'summary.json'),
        cross_probe_summary_sha256=sha(DEFAULT/'comparison/summary.json'),script_sha256=sha(__file__)))
    flat=[flatten(r) for r in rows]
    with (dest/'texture_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    combined=[]
    for r in trained:
        m=flatten(r)
        for condition,outcome in r['outcomes'].items():
            combined.append(dict(**m,condition=condition,**outcome))
    with (dest/'metrics_asr.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(combined[0]));writer.writeheader();writer.writerows(combined)
    plot(dest,rows,correlations)
    for r in rows:
        m=flatten(r)
        print(json.dumps(m),flush=True)
    for c in correlations:
        if c['predictor']=='low_fraction':print(json.dumps(c),flush=True)
    print('[frequency analysis complete]',dest,flush=True)


def plot(dest,rows,correlations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    trained=[r for r in rows if r['group']=='case_trained']
    fig,axs=plt.subplots(2,3,figsize=(15,9),sharex=True)
    x=np.array([r['native128']['hann']['low_fraction']*100 for r in trained])
    for col,condition in enumerate(['nominal','position','joint']):
        for rowidx,outcome in enumerate(['ASR','ASR_excess']):
            ax=axs[rowidx,col]
            y=np.array([r['outcomes'][condition][outcome]*100 for r in trained])
            ax.scatter(x,y,s=45)
            for index,(xx,yy,r) in enumerate(zip(x,y,trained)):
                offset=-12 if index%2 else 5
                ax.annotate(r['id'],(xx,yy),xytext=(-4 if xx>65 else 4,offset),
                            ha='right' if xx>65 else 'left',textcoords='offset points',fontsize=8)
            ax.margins(y=.14)
            rho=next(c['spearman_rho'] for c in correlations if c['predictor']=='low_fraction' and c['endpoint']=='closed_loop'
                     and c['condition']==condition and c['outcome']==outcome)
            ax.set_title(f'{condition} | Spearman rho={rho:+.2f}')
            ax.set_ylabel('Trained ASR (%)' if outcome=='ASR' else 'Trained - matched Random (pp)')
            if rowidx:ax.axhline(0,color='gray',linestyle='--',linewidth=1)
            ax.grid(alpha=.2)
            ax.set_xlabel('Low-frequency power fraction (%) | f < 0.125')
    fig.suptitle('Nine 10 cm trained textures | different own placements | 10 initial states\nDescriptive associations; nominal ASR denominator = 4',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(dest/'frequency_vs_asr.png',dpi=160);plt.close(fig)
    case=[r for r in rows if r['group'] in ['case_random','case_trained']]
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    bottom=np.zeros(len(case));colors=['#3288bd','#fee08b','#d53e4f']
    for band,color in zip(['low','mid','high'],colors):
        values=np.array([r['native128']['hann'][band+'_fraction']*100 for r in case])
        axs[0].bar(range(len(case)),values,bottom=bottom,label=band,color=color);bottom+=values
    axs[0].set_ylabel('AC power fraction (%)');axs[0].set_title('Original UV texture, Hann window');axs[0].legend()
    axs[1].bar(range(len(case)),[r['native128']['hann']['low_absolute_power'] for r in case],color=colors[0])
    axs[1].set_ylabel('Window-energy corrected RGB variance');axs[1].set_title('Absolute low-frequency power')
    for ax in axs:ax.set_xticks(range(len(case)),[r['id'] for r in case],rotation=60)
    fig.tight_layout();fig.savefig(dest/'spectral_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default=str(B.OUT/'table9_closedloop_10cm_20261013'))
    analyze(Path(ap.parse_args().root).resolve())
