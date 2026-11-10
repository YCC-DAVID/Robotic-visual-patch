"""CPU-only descriptive summary and PCA view of the representation pilot."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'representation_distribution'
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    summary={}
    for ax,model in zip(axes,['F_s10000','U_s10000']):
        d=out/model
        assert json.loads((d/'status.json').read_text())['complete']
        rows=json.loads((d/'results.json').read_text())['rows']
        summary[model]={}
        for condition in sorted({r['condition'] for r in rows}):
            selected=[r for r in rows if r['condition']==condition]
            vision=[r for r in selected if r['feature'].startswith('vision')]
            actions={r['frame']:r['action_MSE14'] for r in selected}
            summary[model][condition]=dict(frames=len(actions),
                vision_calibrated_distance_median=float(np.median([r['distance_over_calibration_median'] for r in vision])),
                vision_calibrated_percentile_median=float(np.median([r['calibration_percentile'] for r in vision])),
                vision_relative_shift_median=float(np.median([r['relative_feature_shift'] for r in vision])),
                action_MSE14_median=float(np.median(list(actions.values()))))
        with np.load(d/'features.npz') as z:
            bank=np.concatenate([z[f'bank__vision{i}'] for i in range(3)],axis=1)
            meta=json.loads((d/'bank_manifest.json').read_text())['samples']
            ref=np.array([m['split']=='reference' for m in meta])
            mean=bank[ref].mean(0);scale=np.maximum(bank[ref].std(0),1e-4)
            x=(bank[ref]-mean)/scale
            _,singular,v=np.linalg.svd(x,full_matrices=False)
            components=v[:2].T
            project=lambda a:((a-mean)/scale)@components
            b=project(bank);ax.scatter(b[ref,0],b[ref,1],c='gray',s=16,alpha=.5,label='Clean expert reference')
            ax.scatter(b[~ref,0],b[~ref,1],c='silver',marker='x',s=18,label='Clean expert calibration')
            frames=sorted({k.split('__')[0] for k in z.files if not k.startswith('bank__')})
            for label,conditions,color,marker in [
                ('Clean query',['clean'],'black','o'),('Random',['Random'],'orange','^'),
                ('F patches',['F_P0','F_P1','F_P2'],'crimson','.'),
                ('U patches',['U_P0','U_P1','U_P2'],'royalblue','.')]:
                pts=np.stack([np.concatenate([z[f'{f}__{c}__vision{i}'] for i in range(3)])
                              for f in frames for c in conditions])
                p=project(pts);ax.scatter(p[:,0],p[:,1],c=color,marker=marker,s=22,alpha=.75,label=label)
            ev=(singular[:2]**2).sum()/(singular**2).sum()
            ax.set_title(f'{model}: reference PCA, 2D variance {ev:.1%}')
            ax.set_xlabel('PC1 (within-model axes)');ax.set_ylabel('PC2');ax.legend(fontsize=7)
    fig.suptitle('Exploratory pooled vision features; 2D proximity is not a density test')
    fig.tight_layout();fig.savefig(out/'vision_pca.png',dpi=170);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
