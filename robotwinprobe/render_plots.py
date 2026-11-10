"""Standalone publication/review figures from saved arrays; base Python."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/wam_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import json


def main():
    root=Path(__file__).resolve().parent/'out/render_transfer'
    labels=['digital_full','digital_band32','proxy_full','proxy_band32']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for label in labels:
        points=[]
        for p in sorted((root/f'train_{label}').glob('checkpoint_*.npz')):
            with np.load(p) as d:points.append((len(d['history']),float(np.mean(d['history'][-25:])),float(np.mean(d['validation']))))
        a=np.array(points)
        if len(a):
            axes[0].plot(a[:,0],a[:,1],label=label);axes[1].plot(a[:,0],a[:,2],label=label)
    for ax,title in zip(axes,['Training objective (last25 update mean)','Held-out objective (9 frames)']):
        ax.set_title(title);ax.set_xlabel('PGD update');ax.set_ylabel('First-flow squared distance');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'fig_training.png',dpi=160);plt.close(fig)
    source=root/'learned_figure_data.npz'
    if source.exists():
        d=np.load(source)
        for label in labels:
            mask=d[label+'_mask'];ys,xs=np.where(mask);sy=slice(max(0,ys.min()-2),min(224,ys.max()+3));sx=slice(max(0,xs.min()-2),min(224,xs.max()+3))
            fig,axes=plt.subplots(2,5,figsize=(13,5.5))
            axes[0,0].imshow(d[label+'_texture'],interpolation='nearest');axes[0,0].set_title('Uploaded world Q128')
            axes[1,0].text(.05,.8,'Head camera\nseed10110 / query0\n\nSame perspective\nand visible region\n\nCrop enlarged with\nnearest neighbour',va='top');axes[1,0].axis('off')
            for j,title in enumerate(['Clean','Perspective digital','Actual SAPIEN RT','Candidate proxy']):
                image=d[label+'_inputs'][j];axes[0,j+1].imshow(image,interpolation='nearest');axes[0,j+1].set_title(title)
                axes[1,j+1].imshow(image[sy,sx],interpolation='nearest')
            for ax in axes.flat:ax.axis('off')
            fig.suptitle(label+' | fixed final checkpoint500');fig.tight_layout();fig.savefig(root/f'fig_inputs_{label}.png',dpi=160);plt.close(fig)
    p=root/'rollout_summary.json'
    if p.exists():
        summary=json.loads(p.read_text());rows=[r for r in summary['rows'] if r['complete']]
        fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(rows));values=[r['attack_successes'] for r in rows]
        ax.bar(x,values,color=['#888888' if r['condition'].startswith('random') else '#2563eb' for r in rows])
        ax.set_xticks(x,[r['condition'] for r in rows],rotation=30,ha='right');ax.set_ylim(0,summary['clean_test_successes']+1)
        ax.set_ylabel('Clean successes changed into task failures')
        for i,v in enumerate(values):ax.text(i,v+.1,f"{v}/{summary['clean_test_successes']}",ha='center')
        ax.set_title('Paired closed-loop attacks | one task, one training seed');fig.tight_layout();fig.savefig(root/'fig_rollout.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
