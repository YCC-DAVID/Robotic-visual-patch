"""Non-gating proxy diagnostics and dense true-render held-out trajectory responses."""
import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np

from stage_joint_common import (DEFAULT,STAGES,CONDITIONS,TEXTURES,CAMERAS,WIRE,
    load,request,dump,sha)


def proxy(root):
    from crossarch_capture import records
    p=load(root);out=root/'audit/proxy_review.json'
    if out.exists():return
    paths=[r['path'] for s in p['data']['validation_seeds'] for r in records(root,s)]
    assert all(Path(v).exists() for v in paths)
    response=request(dict(command='validate',records=paths),root)
    summary={k:v for k,v in response.items() if k!='rows'}
    summary.update(training_gate=False,proceed_independent_of_proxy_score=True,
        protocol_sha256=sha(root/'protocol.json'),records=len(paths),
        note='Actual interface and finite texture gradients remain mandatory; proxy response thresholds are diagnostics only, as approved.')
    dump(out,summary)
    print('[proxy diagnostic]',summary,flush=True)


def dense(root):
    import bootstrap as B
    B.simulator_paths()
    from PIL import Image
    from scene_smoke import make_task
    from render_geometry import Patch,patterns,camera_objects
    p=load(root);texture_manifest=json.loads((root/'texture_manifest.json').read_text())
    for name in TEXTURES:
        assert sha(root/'patches'/name/'texture_0500.png')==texture_manifest['textures'][name]
    textures={'random':patterns()['random'],**{n:np.asarray(Image.open(root/'patches'/n/'texture_0500.png').convert('RGB')) for n in TEXTURES}}
    noise_seeds=[202610270,202610271,202610272]
    all_rows=[]
    for seed in p['data']['validation_seeds']:
        output=root/'diagnostics'/f'seed{seed}.json'
        if output.exists():
            saved=json.loads(output.read_text());assert saved['protocol_sha256']==sha(root/'protocol.json')
            all_rows.extend(saved['rows']);continue
        dest=root/'trajectories'/f'seed{seed}'
        meta=json.loads((dest/'complete.json').read_text())
        with np.load(dest/'trace.npz') as z:
            actions=z['actions'].copy();states=z['state14'].copy();prompt=str(z['prompt'])
        with np.load(dest/'frames.npz') as z:
            reference={int(q):{n:z[n][i].copy() for n in CAMERAS} for i,q in enumerate(z['query'])}
        selected=set(range(0,len(states),p['data']['validation_stride']))|{len(states)-1}
        for lo,hi in meta['spans'].values():selected|={lo,hi-1}
        selected=sorted(selected)
        def phase(step):
            return next(s for s,(lo,hi) in meta['spans'].items() if lo<=step<hi)
        random.seed(seed);task,_,_=make_task(p['task'],seed,dest)
        rows=[];saved_frames={};selected_steps=[]
        try:
            for step in range(len(states)):
                if step in selected:
                    clean=task.get_obs();state=clean['joint_action']['vector'].astype(np.float32)
                    assert np.array_equal(state,states[step]),('dense state',seed,step)
                    if step in reference:
                        for n in CAMERAS:
                            assert np.array_equal(clean['observation'][n]['rgb'],reference[step][n])
                    raw={w:clean['observation'][n]['rgb'].copy() for n,w in zip(CAMERAS,WIRE)}
                    baseline=[request(dict(command='diagnostic',state=state,prompt=prompt,images=raw,seed=ns),root) for ns in noise_seeds]
                    saved_frames.setdefault('clean_head',[]).append(raw[WIRE[0]])
                    selected_steps.append(step)
                    patch=Patch(task,dest,center=p['patch']['center_m'],size=p['patch']['size_m'])
                    try:
                        for name,texture in textures.items():
                            patch.texture(texture);obs=task.get_obs()
                            assert np.array_equal(state,obs['joint_action']['vector'].astype(np.float32))
                            raw={w:obs['observation'][n]['rgb'].copy() for n,w in zip(CAMERAS,WIRE)}
                            cams=camera_objects(task)
                            visible=[int((cams[n].get_picture('Segmentation')[...,1]==patch.actor.per_scene_id).sum()) for n in CAMERAS]
                            saved_frames.setdefault(name+'_head',[]).append(raw[WIRE[0]])
                            for ns,c in zip(noise_seeds,baseline):
                                r=request(dict(command='diagnostic',state=state,prompt=prompt,images=raw,seed=ns),root)
                                dv=r['v']-c['v'];da=r['action']-c['action']
                                rows.append(dict(seed=seed,step=step,stage=phase(step),condition=name,noise_seed=ns,
                                    L0_mse14=float(np.mean(dv[...,:14]**2)),L0_full_mse=float(np.mean(dv**2)),
                                    normalized_action_mse14=float(np.mean(da[...,:14]**2)),
                                    head_visible_px=visible[0],left_visible_px=visible[1],right_visible_px=visible[2],
                                    terminal=step==len(states)-1))
                    finally:task.scene.remove_actor(patch.actor)
                    restored=task.get_obs()
                    for n in CAMERAS:assert np.array_equal(restored['observation'][n]['rgb'],clean['observation'][n]['rgb'])
                    dump(root/'diagnostics/progress.json',dict(seed=seed,step=step,rows=len(rows),complete=False))
                if step<len(actions):
                    task.take_action(actions[step],action_type='qpos')
            assert task.eval_success,'Dense expert replay failed'
            output.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(output.with_suffix('.npz'),steps=selected_steps,**saved_frames)
            dump(output,dict(seed=seed,protocol_sha256=sha(root/'protocol.json'),rows=rows,
                samples=len(selected),complete=True,texture_manifest_sha256=sha(root/'texture_manifest.json')))
            all_rows.extend(rows)
        finally:task.close_env()
    write_summary(root,all_rows)


def write_summary(root,rows):
    out=root/'diagnostics';out.mkdir(exist_ok=True)
    with (out/'responses.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    controls={(r['seed'],r['step'],r['noise_seed']):r for r in rows if r['condition']=='random'}
    summary=[]
    for name in ('random',*TEXTURES):
        for stage in STAGES:
            values=[r for r in rows if r['condition']==name and r['stage']==stage]
            ratios=[r['L0_mse14']/max(controls[r['seed'],r['step'],r['noise_seed']]['L0_mse14'],1e-12) for r in values]
            summary.append(dict(condition=name,stage=stage,n=len(values),independent_trajectories=len({r['seed'] for r in values}),
                mean_L0=float(np.mean([r['L0_mse14'] for r in values])),
                mean_normalized_action_mse14=float(np.mean([r['normalized_action_mse14'] for r in values])),
                mean_head_visible_px=float(np.mean([r['head_visible_px'] for r in values])),
                median_Random_ratio=float(np.median(ratios)),fraction_over_5Random=float(np.mean(np.array(ratios)>=5))))
    coverage=[]
    for name in TEXTURES:
        per_frame=[];longest=0
        for seed in sorted({r['seed'] for r in rows}):
            points=sorted({r['step'] for r in rows if r['seed']==seed})
            start=None
            for step in points:
                patch_values=[r['L0_mse14'] for r in rows if r['seed']==seed and r['step']==step and r['condition']==name]
                random_values=[r['L0_mse14'] for r in rows if r['seed']==seed and r['step']==step and r['condition']=='random']
                ratio=float(np.mean(patch_values)/max(np.mean(random_values),1e-12));per_frame.append(ratio)
                if ratio<2:
                    if start is None:start=step
                    longest=max(longest,step-start)
                else:start=None
        coverage.append(dict(condition=name,frame_count=len(per_frame),Random_ratio_p10=float(np.quantile(per_frame,.1)),
            fraction_over_5Random=float(np.mean(np.array(per_frame)>=5)),longest_low_response_sampled_span_actions=longest,
            weakest_stage_L0=min(r['mean_L0'] for r in summary if r['condition']==name)))
    dump(out/'summary.json',dict(complete=True,rows=summary,coverage=coverage,
        unit='independent held-out expert trajectories; time/noise samples are correlated',
        scope='offline true-render paired responses, not closed-loop ASR',
        sampled_span_note='low-response intervals join observed samples at stride8/boundaries; not proof of continuously low response between samples',
        protocol_sha256=sha(root/'protocol.json')))
    dump(out/'progress.json',dict(complete=True,rows=len(rows)))


def plot_files(root):
    with (root/'diagnostics/responses.csv').open() as f:
        rows=list(csv.DictReader(f))
    for r in rows:
        for key in ('seed','step','noise_seed','head_visible_px','left_visible_px','right_visible_px'):r[key]=int(r[key])
        for key in ('L0_mse14','L0_full_mse','normalized_action_mse14'):r[key]=float(r[key])
    summary=json.loads((root/'diagnostics/summary.json').read_text())['rows']
    plot(root,rows,summary)


def plot(root,rows,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=('random',*TEXTURES);fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name in names:
        values=[next(r for r in summary if r['condition']==name and r['stage']==s) for s in STAGES]
        axes[0].plot(STAGES,[r['mean_L0'] for r in values],'-o',label=name)
        axes[1].plot(STAGES,[r['mean_normalized_action_mse14'] for r in values],'-o',label=name)
    axes[0].set(ylabel='True-render L0, effective14 MSE')
    axes[1].set(ylabel='Complete10-flow normalized action MSE14')
    for ax in axes[:2]:ax.set_yscale('symlog',linthresh=1e-7)
    for name in names:
        stage_means=[r['mean_L0'] for r in summary if r['condition']==name]
        axes[2].bar(name,min(stage_means))
    axes[2].set(ylabel='Weakest stage mean L0');axes[2].tick_params(axis='x',rotation=25)
    axes[0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'diagnostics/stage_responses.png',dpi=180);plt.close(fig)
    seeds=sorted({r['seed'] for r in rows});fig,axes=plt.subplots(len(seeds),1,figsize=(12,2.6*len(seeds)),squeeze=False)
    for ax,seed in zip(axes[:,0],seeds):
        for name in names:
            subset=[r for r in rows if r['seed']==seed and r['condition']==name]
            steps=sorted({r['step'] for r in subset})
            ax.plot(steps,[np.mean([r['L0_mse14'] for r in subset if r['step']==step]) for step in steps],label=name)
        ax.set(ylabel=f'Seed {seed}\nL0 MSE14',xlabel='Expert action step')
        ax.set_yscale('symlog',linthresh=1e-7)
        if seed==seeds[0]:ax.legend(ncol=4,fontsize=8)
    fig.tight_layout();fig.savefig(root/'diagnostics/trajectory_L0.png',dpi=150);plt.close(fig)
    for field,label,filename in [('normalized_action_mse14','Complete10-flow action MSE14','trajectory_actions.png'),
            ('head_visible_px','Visible native pixels','trajectory_visibility.png')]:
        fig,axes=plt.subplots(len(seeds),1,figsize=(12,2.3*len(seeds)),squeeze=False)
        for ax,seed in zip(axes[:,0],seeds):
            fields=('head_visible_px','left_visible_px','right_visible_px') if 'visible' in field else (field,)
            conditions=('joint_balanced',) if 'visible' in field else names
            for name in conditions:
                selected=[r for r in rows if r['seed']==seed and r['condition']==name]
                steps=sorted({r['step'] for r in selected})
                for key in fields:
                    ax.plot(steps,[np.mean([r[key] for r in selected if r['step']==step]) for step in steps],
                            label=key if 'visible' in field else name)
            ax.set(ylabel=f'Seed {seed}\n{label}',xlabel='Expert action step')
            if 'visible' not in field:ax.set_yscale('symlog',linthresh=1e-7)
            if seed==seeds[0]:ax.legend(ncol=4,fontsize=8)
        fig.tight_layout();fig.savefig(root/'diagnostics'/filename,dpi=150);plt.close(fig)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--mode',choices=['proxy','dense','plot'],required=True)
    a=ap.parse_args();{'proxy':proxy,'dense':dense,'plot':plot_files}[a.mode](a.root.resolve())
