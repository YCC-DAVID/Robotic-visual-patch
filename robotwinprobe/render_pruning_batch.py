"""Measure fixed cumulative prefixes together; select the first rejection in order.

The order is frozen before measurement, so batching does not change the original
80%-of-full stopping rule. Also report Random and excess-over-Random responses.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from radius_plan import dump
from table9_plan import sha


def prepare(scan):
    source=json.loads((scan/'protocol.json').read_text())
    summary=json.loads((scan/'summary.json').read_text())
    full=np.asarray(Image.open(scan/'textures/full.png').convert('RGB'))
    random_rgb=np.asarray(Image.open(scan/'textures/random.png').convert('RGB'))
    background=np.random.RandomState(20261130).randint(0,256,full.shape,dtype=np.uint8)
    dest=scan/'pruning_batch';textures=dest/'textures';textures.mkdir(parents=True,exist_ok=True)
    variants=[dict(name='full',kind='full'),dict(name='random',kind='random')]
    images=dict(full=full,random=random_rgb)
    current=full.copy();order=summary['cumulative_replacement_order']
    for step,wi in enumerate(order,1):
        w=source['windows'][wi]
        current[w['y']:w['y']+16,w['x']:w['x']+16]=background[w['y']:w['y']+16,w['x']:w['x']+16]
        name=f'step{step:02d}';images[name]=current.copy()
        variants.append(dict(name=name,kind='cumulative',step=step,window=wi))
    for count in [1,4]:
        mask=np.zeros((128,128),bool)
        for wi in summary['top_windows'][:count]:
            w=source['windows'][wi];mask[w['y']:w['y']+16,w['x']:w['x']+16]=True
        name=f'keep_top{count}';images[name]=np.where(mask[...,None],full,random_rgb)
        variants.append(dict(name=name,kind='retention_control',area_fraction=count/64))
    mask=np.zeros((128,128),bool)
    for wi in summary['top_windows'][:4]:
        w=source['windows'][wi];mask[w['y']:w['y']+16,w['x']:w['x']+16]=True
    images['remove_top4']=np.where(mask[...,None],random_rgb,full)
    variants.append(dict(name='remove_top4',kind='retention_control',area_fraction=4/64))
    for name,rgb in images.items():
        path=textures/(name+'.png')
        if path.exists():assert np.array_equal(np.asarray(Image.open(path)),rgb),name
        else:Image.fromarray(rgb).save(path)
    p=dict(source,variants=variants,
        texture_sha256={name:sha(textures/(name+'.png')) for name in images},
        intervention='world-texture cumulative replacement; fixed order, no reoptimization',
        selection_order='ascending mean absolute per-sample normalized change from frozen scan',
        order=order,replacement_seed=20261130,
        threshold='mean candidate L0 >= 0.8 * mean full L0; first rejection stops and rolls back',
        execution='all fixed prefixes measured in one batch; later prefixes never used to skip a rejection',
        secondary_analysis='excess-over-Random retention reported diagnostically; original stopping rule unchanged',
        source_summary_sha256=sha(scan/'summary.json'))
    path=dest/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:dump(path,p)
    return dest,p


def summarize(scan,dest,p):
    metrics=[];footprints=[]
    for r in p['samples']:
        with np.load(dest/'metrics'/f"sample{r['id']:03d}.npz") as d:
            assert str(d['protocol_sha256'])==sha(dest/'protocol.json')
            metrics.append(d['L0'].mean(-1));footprints.append(d['footprints'])
    l0=np.stack(metrics).astype(float)
    full=l0[:,0].mean();random=l0[:,1].mean();excess=full-random
    raw_random_ratio=random/max(full,1e-12)
    trace=[];accepted=[];stopped=False
    for step,wi in enumerate(p['order'],1):
        candidate=l0[:,step+1];ratio=float(candidate.mean()/max(full,1e-12))
        ok=ratio>=.8
        if not stopped and ok:accepted.append(wi)
        elif not ok:stopped=True
        trace.append(dict(step=step,window=wi,raw_L0_ratio=ratio,accepted=ok and not stopped,
            after_first_rejection=stopped and ok,
            candidate_L0=float(candidate.mean()),
            excess_over_random_ratio=float((candidate.mean()-random)/excess) if abs(excess)>1e-12 else None,
            per_sample_raw_ratio=(candidate/np.maximum(l0[:,0],1e-12)).tolist(),
            per_sample_L0=candidate.tolist()))
    # Accepted prefix length is final even if a later independently measured prefix recovers.
    n=len(accepted)
    selected_index=n+1 if n else 0
    texture='full' if not n else f'step{n:02d}'
    Image.open(dest/'textures'/(texture+'.png')).save(dest/'retained.png')
    if n<64:Image.open(dest/'textures'/f'step{n+1:02d}.png').save(dest/'first_rejected.png')
    mask=np.ones((128,128),bool)
    for wi in accepted:
        w=p['windows'][wi];mask[w['y']:w['y']+16,w['x']:w['x']+16]=False
    Image.fromarray((mask*255).astype(np.uint8)).save(dest/'retained_mask.png')
    total_visible=np.stack(footprints).sum()
    removed_visible=np.stack(footprints)[:,accepted].sum() if accepted else 0.
    controls=[]
    for vi,v in enumerate(p['variants']):
        if v['kind']!='retention_control':continue
        controls.append(dict(name=v['name'],L0=float(l0[:,vi].mean()),
            raw_L0_ratio=float(l0[:,vi].mean()/max(full,1e-12)),
            excess_over_random_ratio=float((l0[:,vi].mean()-random)/excess) if abs(excess)>1e-12 else None))
    result=dict(complete=True,accepted_windows=accepted,accepted_count=n,
        original_texture_fraction=float(mask.mean()),retained_visible_sampling_fraction=float(1-removed_visible/max(total_visible,1e-12)),
        full_L0=float(full),random_L0=float(random),random_raw_L0_ratio=float(raw_random_ratio),
        raw_threshold_non_discriminating=bool(raw_random_ratio>=.8),
        final_raw_ratio=float(l0[:,selected_index].mean()/max(full,1e-12)),
        excess_full_minus_random=float(excess),controls=controls,
        physical_outline_m=p['size_m'],endpoint='actual-render open-loop L0; pruned ASR not measured',
        interpretation='When Random itself passes the raw 80% gate, raw L0 preservation cannot establish preserved attack or a sufficient structure',
        protocol_sha256=sha(dest/'protocol.json'))
    dump(dest/'trace.json',trace);dump(dest/'selection.json',result)
    return result


def run(runner,scan):
    dest,p=prepare(scan)
    runner.status('cumulative_render_replacement',scan=str(scan),variants=len(p['variants']),samples=len(p['samples']))
    if not all((dest/'capture'/f"sample{r['id']:03d}.npz").exists() for r in p['samples']):
        runner.run('pruning_batch_capture',runner.sim('robust_occlusion.py','capture','--root',dest))
    if not all((dest/'metrics'/f"sample{r['id']:03d}.npz").exists() for r in p['samples']):
        runner.run('pruning_batch_measure',runner.model('robust_occlusion.py','measure','--root',dest))
    return summarize(scan,dest,p)
