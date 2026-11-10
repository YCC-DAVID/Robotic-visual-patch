"""Audit paired prefixes and plot delayed-onset results and insertion frames."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from radius_plan import dump
from stage_injection import STAGES, digest, load_protocol, report, sha

LABELS = ['Initial','Approach','Lifted','Above plate']


def wilson(k,n):
    if not n:
        return [None,None]
    z = 1.959963984540054
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-radius,center+radius]


def audit(root):
    p = load_protocol(root)
    summary = report(root)
    onset = []
    checked = 0
    prior_comparisons = []
    for path,h in p['config'].get('prior_evidence_sha256',{}).items():
        assert sha(path) == h, 'prior selection evidence changed'
    for seed in p['config']['seeds']:
        path = root/'episodes'/f'seed{seed}_clean.json'
        if not path.exists():
            continue
        clean = json.loads(path.read_text())
        assert sha(path.with_suffix('.npz')) == clean['artifact_sha256']
        with np.load(path.with_suffix('.npz')) as d:
            reference = {k:d[k] for k in d.files}
        evidence = p['config'].get('prior_evidence_sha256',{})
        if evidence:
            for method,prior_name in [('clean',f'clean_{seed}_nominal.json'),
                    ('trained',f'site07_{seed}_nominal_0_trained.json'),
                    ('random',f'site07_{seed}_nominal_0_random.json')]:
                prior_path = next(Path(v) for v in evidence if Path(v).name==prior_name)
                prior = json.loads(prior_path.read_text())
                current_path = root/'episodes'/(f'seed{seed}_clean.json' if method=='clean'
                                              else f'seed{seed}_{method}_initial.json')
                if current_path.exists():
                    current=json.loads(current_path.read_text())
                    prior_comparisons.append(dict(seed=seed,method=method,
                        prior_success=prior['success'],current_success=current['success'],
                        prior_steps=prior['steps'],current_steps=current['steps'],
                        replicated_outcome=current['success']==prior['success']))
        for stage in STAGES:
            paired = []
            for method in ['trained','random']:
                branch_path = root/'episodes'/f'seed{seed}_{method}_{stage}.json'
                if not branch_path.exists():
                    continue
                row = json.loads(branch_path.read_text())
                q = clean['schedule'][stage]
                if row['status'] != 'complete':
                    assert q is None and row['status'] == 'no_injection_opportunity'
                    continue
                assert row['trigger_query'] == q and row['injection']['query'] == q
                assert row['injection']['step'] == int(reference['query_steps'][q])
                assert row['clean_prefix_verified']
                assert sha(branch_path.with_suffix('.npz')) == row['artifact_sha256']
                with np.load(branch_path.with_suffix('.npz')) as d:
                    for name in ['states14','object_tcp_positions','query_steps','noise_seeds']:
                        assert np.array_equal(d[name][:q+1],reference[name][:q+1]), (seed,stage,method,name)
                    for name in ['actions','rgb_sha256']:
                        assert np.array_equal(d[name][:q],reference[name][:q]), (seed,stage,method,name)
                    assert not d['patch_active'][:q].any()
                    assert d['patch_active'][q]
                    duration = p['config']['duration_queries']
                    expected = np.arange(len(d['patch_active'])) >= q
                    if duration:
                        expected &= np.arange(len(expected)) < q+duration
                    assert np.array_equal(d['patch_active'],expected)
                    cameras = ['head_camera','left_camera','right_camera']
                    for ci,name in enumerate(cameras):
                        assert digest(d['before_'+name]) == reference['rgb_sha256'][q,ci]
                    paired.append((row['injection'],d['visible_pixels'][q].tolist()))
                    delta = d['actions'][q].astype(np.float64)-reference['actions'][q].astype(np.float64)
                    joint_indices = [0,1,2,3,4,5,7,8,9,10,11,12]
                    onset.append(dict(seed=seed,stage=stage,method=method,clean_success=clean['success'],
                        success=row['success'],episode_steps=row['steps'],clean_episode_steps=clean['steps'],
                        step_delta=row['steps']-clean['steps'],query=q,event_step=row['injection']['clean_event_step'],
                        effective_step=row['injection']['step'],delay_actions=row['injection']['delay_actions'],
                        active_queries=row['active_queries'],head_visible_px=int(d['visible_pixels'][q,0]),
                        left_visible_px=int(d['visible_pixels'][q,1]),right_visible_px=int(d['visible_pixels'][q,2]),
                        action_joint_rms_rad=float(np.sqrt(np.mean(delta[:,joint_indices]**2))),
                        action_gripper_rms=float(np.sqrt(np.mean(delta[:,[6,13]]**2))),
                        duplicate_onset_stages=','.join(s for s,v in clean['schedule'].items() if v==q and s!=stage)))
                checked += 1
            if len(paired)==2:
                assert paired[0] == paired[1], (seed,stage,'trained/Random onset or mask mismatch')
    if onset:
        with (root/'onsets.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(onset[0]))
            writer.writeheader();writer.writerows(onset)
    dump(root/'audit.json',dict(passed=True,checked_branches=checked,complete=summary['complete'],
        checks='artifact hashes; exact clean state/pose/RGB/action prefix; same event, query and mask; finite window',
        prior_outcome_replication=prior_comparisons,
        onset_rows=onset))
    return summary,onset


def plot(root,summary,onset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,cohort,title in zip(axes,['stage_reachable','all_stages_reachable'],
                              ['Reachable at each stage','Same seeds reachable at all stages']):
        values = [next(r for r in summary['rows'] if r['stage']==s and r['cohort']==cohort) for s in STAGES]
        for method,color,dx in [('trained','#b73535',-.06),('random','#406f9e',.06)]:
            for i,r in enumerate(values):
                n=r['n']
                if not n:
                    continue
                k=r[method+'_failures'];lo,hi=wilson(k,n);v=k/n
                ax.errorbar(i+dx,v,yerr=[[v-lo],[hi-v]],fmt='o',color=color,capsize=3,
                    label=method.capitalize() if i==0 else None)
            xs=[i+dx for i,r in enumerate(values) if r['n']]
            ys=[r[method+'_ASR'] for r in values if r['n']]
            ax.plot(xs,ys,color=color,alpha=.6)
        ax.set_xticks(range(4),[f'{label}\nn={r["n"]}' for label,r in zip(LABELS,values)])
        ax.set_ylim(-.04,1.04);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        if any(r['n'] for r in values):
            ax.legend()
    axes[0].set_ylabel('Task failure among paired clean successes (ASR)')
    fig.suptitle(f'Frozen 10 cm patch, delayed onset | {summary["recorded"]}/{summary["expected"]} records')
    fig.text(.5,.015,'95% Wilson intervals; pilot. Later onset also reduces exposure. Coincident stages share one query.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(root/'stage_asr.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for method,color,dx in [('trained','#b73535',-.08),('random','#406f9e',.08)]:
        for i,stage in enumerate(STAGES):
            vals=[r['action_joint_rms_rad'] for r in onset
                  if r['method']==method and r['stage']==stage and r['clean_success']]
            if vals:
                ax.scatter(np.full(len(vals),i+dx),vals,color=color,alpha=.65,
                           label=method.capitalize() if i==0 else None)
                ax.plot([i+dx-.05,i+dx+.05],[np.mean(vals)]*2,color=color,linewidth=3)
    ax.set_xticks(range(4),LABELS);ax.set_ylabel('First affected chunk joint RMS change (rad)')
    ax.set_title('Same state and noise: patched action chunk vs clean')
    ax.grid(axis='y',alpha=.2)
    if any(r['clean_success'] for r in onset):
        ax.legend()
    fig.tight_layout();fig.savefig(root/'action_deviation.png',dpi=180);plt.close(fig)
    candidates = sorted({r['seed'] for r in onset if r['clean_success']})
    if not candidates:
        candidates = sorted({r['seed'] for r in onset})
    if not candidates:
        return
    seed=max(candidates,key=lambda s:len({r['stage'] for r in onset if r['seed']==s}))
    fig,axes=plt.subplots(4,3,figsize=(11,10))
    for i,(stage,label) in enumerate(zip(STAGES,LABELS)):
        path=root/'episodes'/f'seed{seed}_trained_{stage}.npz'
        if not path.exists():
            for ax in axes[i]:
                ax.axis('off');ax.set_title(label+': no injection opportunity')
            continue
        row=next(r for r in onset if r['seed']==seed and r['stage']==stage and r['method']=='trained')
        with np.load(path) as d:
            for ax,name,title in zip(axes[i],['before_head_camera','after_head_camera','final_head_camera'],
                                     ['Before insertion','After insertion','Final']):
                ax.imshow(d[name]);ax.axis('off')
                ax.set_title(f'{label} @ step {row["effective_step"]}\n{title}')
    fig.suptitle(f'Seed {seed}: true-render trained patch (head camera)')
    fig.tight_layout();fig.savefig(root/'insertion_frames.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True)
    ap.add_argument('--plot',action='store_true')
    args=ap.parse_args();root=Path(args.root).resolve()
    summary,onset=audit(root)
    if args.plot:
        plot(root,summary,onset)
