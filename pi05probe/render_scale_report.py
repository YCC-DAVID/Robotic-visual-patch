"""Aggregate completed sweep records; never turn pending jobs into results."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'out/render_proxy_scale'


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    dest=BASE/'report';dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    baseline=read(BASE/'clean_baseline/episodes.json')
    for run in sorted(BASE.glob('train_*')):
        name=run.name[6:];config=read(run/'config.json');result=read(run/'results.json')
        if not config:continue
        cal=Path(config['calibration_path']);capture=read(cal.parent/'capture.json')
        pilot=read(BASE/('pilot_'+name)/'episodes.json')
        confirmation=read(BASE/('confirm_'+name)/'episodes.json')
        metrics=read(BASE/('validation_'+name)/'metrics.json')
        row={'name':name,'size_cm':config['patch_m']*100,'position':name.split('_')[1],
             'frequency':'band32' if config['parameter_shape'][-1]==32 else 'full',
             'mode':config['mode'],'training_complete':config['complete'],
             'visible_px224_train_median':capture.get('visible_pixels224_train_median') if capture else None,
             'validation_proxy_L0':metrics['summary']['optimized_proxy']['L0'] if metrics else (result.get('final_validation_proxy_L0') if result else None),
             'training_validation_proxy_L0':result.get('final_validation_proxy_L0') if result else None,
             'pilot':pilot['summary'] if pilot else None,'confirmation':confirmation['summary'] if confirmation else None,
             'combined_paired_failures':sum(stage['summary']['optimized']['paired_ASR_failures'] for stage in [pilot,confirmation] if stage),
             'combined_paired_clean_success_n':sum(stage['summary']['optimized']['paired_ASR_clean_success_n'] for stage in [pilot,confirmation] if stage),
             'true_render_validation':metrics['summary'] if metrics else None}
        rows.append(row)
    planned=read(BASE/'protocol.json')['planned_training_runs']
    finished=len(rows)==len(planned) and all(r['training_complete'] and r['true_render_validation'] and
        all(r[stage] and all(r[stage][c]['n']==5 for c in ['random','optimized']) for stage in ['pilot','confirmation']) for r in rows)
    paired=[r for r in rows if all(r[stage] and r[stage]['optimized']['n']==5 for stage in ['pilot','confirmation'])]
    observed=[r for r in paired if r['combined_paired_failures']>sum(r[stage]['random']['paired_ASR_failures'] for stage in ['pilot','confirmation'])]
    strong=[r for r in paired if all(r[stage]['optimized']['paired_ASR']>=.5 and
        r[stage]['optimized']['paired_ASR_failures']>r[stage]['random']['paired_ASR_failures'] for stage in ['pilot','confirmation'])]
    summary={'complete':bool(finished),'rows':rows,'clean':baseline['summary'] if baseline else None,
             'smallest_observed_cm':min(r['size_cm'] for r in observed) if observed else None,
             'smallest_strong_screen_cm':min(r['size_cm'] for r in strong) if strong else None,
             'training_runs':len(rows),'gradient_frame_forwards':sum(4000 for r in rows if r['training_complete']),
             'unique_test_init_indices':list(range(40,50)),
             'actual_physical_episodes':10+sum(r[stage][c]['n'] for r in rows for stage in ['pilot','confirmation'] for c in ['random','optimized'] if r[stage]),
             'notes':'Robot successes, not attack successes. Development40-44 and confirmation45-49 are separate. Single task, training seed and five initial states per stage. A size that fails this budget is not theoretically impossible.'}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    text=['# Proxy frequency, size and position results','',
          'Every row is independently trained with 500x8 frames. Both views use actual MuJoCo render in closed loop. '+('All11 runs are complete.' if finished else 'Pending results are shown as pending.'),'',
          '| Run | Validation render L0 | Proxy L0 | Pilot clean/random/optimized successes | Paired failures / clean successes | Confirmation clean/random/optimized |',
          '|---|---:|---:|---|---|---|']
    for r in rows:
        metrics=r['true_render_validation'];real=metrics['optimized_render']['L0'] if metrics else None
        def counts(stage):
            s=r[stage]
            return ' / '.join(str(s[c]['successes'])+'/'+str(s[c]['n']) for c in ['clean','random','optimized']) if s else 'pending'
        s=r['pilot'];paired=str(s['optimized']['paired_ASR_failures'])+'/'+str(s['optimized']['paired_ASR_clean_success_n']) if s else 'pending'
        text.append('| {} | {} | {} | {} | {} | {} |'.format(r['name'], 'pending' if real is None else '{:.5f}'.format(real),
            'pending' if r['validation_proxy_L0'] is None else '{:.5f}'.format(r['validation_proxy_L0']),counts('pilot'),paired,counts('confirmation')))
    text+=['','Completed model diagnostics use the same three noises and uploaded PNG for proxy/render L0. Training-only proxy L0 is separately retained in JSON. L0 uses all 10x32 dimensions of the first t=1 velocity. Real7 and execution5x7, direct proxy/render joint KV distances and paired velocity differences remain in per-run metrics.json.','',
           'band32 means latent32 bilinear-upsampled to world Q128; not a strict Fourier cutoff and not identical to historical image-space band32. Random is matched to the latent parameterization.','',
           'Smallest tested size with observed paired interference:{}cm. Smallest tested strong size (at least half paired failures in both stages):{}cm. Development screening and follow-up confirmation were expanded as documented in protocol.json, so this is an exploratory pilot minimum, not a theoretical limit. Position101/161 differ in projected area and task context.'.format(summary['smallest_observed_cm'],summary['smallest_strong_screen_cm'])]
    (dest/'results.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for run in sorted(BASE.glob('train_*')):
        checkpoints=sorted(run.glob('checkpoint_*.npz'))
        if not checkpoints:continue
        history=np.load(checkpoints[-1],allow_pickle=False)['history']
        if len(history)>=10:
            smooth=np.convolve(history,np.ones(10)/10,mode='valid')
            ax.plot(np.arange(10,len(history)+1),smooth,label=run.name[6:])
    ax.set(xlabel='PGD updates',ylabel='Training early velocity L0 (10-point mean)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(dest/'training_curves.png',dpi=180);plt.close(fig)
    evaluated=[r for r in rows if r['pilot'] and r['pilot']['optimized']['n']==5]
    if evaluated:
        fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(evaluated));width=.25
        for j,c in enumerate(['clean','random','optimized']):
            ax.bar(x+(j-1)*width,[r['pilot'][c]['successes']/r['pilot'][c]['n'] for r in evaluated],width,label=c)
        ax.set_xticks(x);ax.set_xticklabels([r['name'] for r in evaluated],rotation=20,ha='right',fontsize=8)
        ax.set(ylim=(0,1.12),ylabel='Robot task success (five development episodes)');ax.legend();fig.tight_layout()
        fig.savefig(dest/'pilot_success.png',dpi=180);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
        x=np.arange(len(evaluated));width=.35
        for ax,stage,title in zip(axes,['pilot','confirmation'],['Development init40-44','Confirmation init45-49']):
            for j,c in enumerate(['random','optimized']):
                values=[]
                for r in evaluated:
                    s=r[stage]
                    values.append(s[c]['paired_ASR'] if s and s[c]['n']==5 else np.nan)
                ax.bar(x+(j-.5)*width,values,width,label=c)
                for i,r in enumerate(evaluated):
                    s=r[stage]
                    if s and s[c]['n']==5 and c=='optimized':
                        ax.text(x[i]+width/2,values[i]+.025,'{}/{}'.format(s[c]['paired_ASR_failures'],s[c]['paired_ASR_clean_success_n']),ha='center',fontsize=8)
            ax.set(ylim=(0,1.16),ylabel='Paired task failures / clean successes',title=title)
            ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
        axes[-1].set_xticks(x);axes[-1].set_xticklabels([r['name'] for r in evaluated],rotation=25,ha='right',fontsize=8)
        fig.tight_layout();fig.savefig(dest/'paired_failures.png',dpi=180);plt.close(fig)
    captured=[]
    for run in sorted(BASE.glob('train_*')):
        name=run.name[6:];path=BASE/('validation_'+name)/'inputs.npz'
        if path.exists():captured.append((name,path))
    if captured:
        from PIL import Image
        fig,axes=plt.subplots(len(captured),3,figsize=(8,2.6*len(captured)),squeeze=False)
        for row,(name,path) in enumerate(captured):
            data=np.load(path,allow_pickle=False);mask=data['mask'][0];yy,xx=np.where(mask)
            a,b=max(0,int(yy.min())-3),min(224,int(yy.max())+4)
            c,d=max(0,int(xx.min())-3),min(224,int(xx.max())+4)
            for col,image in enumerate([data['optimized__texture'],data['optimized__render'][0],data['optimized__render'][0,a:b,c:d]]):
                axes[row,col].imshow(image);axes[row,col].axis('off')
            axes[row,0].set_title(name,fontsize=9);axes[row,1].set_title('Actual render, same clean state');axes[row,2].set_title('Visible patch crop: {} px'.format(int(mask.sum())))
        fig.tight_layout();fig.savefig(dest/'textures_and_renders.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
