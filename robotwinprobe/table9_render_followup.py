"""Persistently resume closed loops, then scan and prune a supported EOT patch."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import bootstrap as B
from radius_plan import dump
from radius_pipeline import Runner
from table9_plan import sha
from table9_closedloop import prepare as closedloop_prepare
from robust_occlusion import prepare as scan_prepare

PLOT_PYTHON='/home/user1/miniconda3/envs/openpi-libero/bin/python'


def mean_l0(root, samples, candidate=2):
    full, changed = [], []
    for r in samples:
        with np.load(root/'metrics'/f"sample{r['id']:03d}.npz") as d:
            full.append(float(d['L0'][0].mean()))
            changed.append(float(d['L0'][candidate].mean()))
    full=np.array(full);changed=np.array(changed)
    return dict(full_mean=float(full.mean()), candidate_mean=float(changed.mean()),
        ratio=float(changed.mean()/max(full.mean(),1e-12)),
        per_sample_full=full.tolist(), per_sample_candidate=changed.tolist(),
        per_sample_ratio=(changed/np.maximum(full,1e-12)).tolist())


def cumulative(run, scan):
    p=json.loads((scan/'protocol.json').read_text())
    summary=json.loads((scan/'summary.json').read_text())
    full=np.asarray(Image.open(scan/'textures/full.png').convert('RGB'))
    random_rgb=np.asarray(Image.open(scan/'textures/random.png').convert('RGB'))
    background=np.random.RandomState(20261130).randint(0,256,full.shape,dtype=np.uint8)
    current=full.copy();accepted=[];trace=[]
    dest=scan/'pruning';dest.mkdir(exist_ok=True)
    for step,wi in enumerate(summary['cumulative_replacement_order'],1):
        w=p['windows'][wi];candidate=current.copy()
        candidate[w['y']:w['y']+16,w['x']:w['x']+16]=background[w['y']:w['y']+16,w['x']:w['x']+16]
        root=dest/f'step{step:02d}';textures=root/'textures';textures.mkdir(parents=True,exist_ok=True)
        protocol=dict(p)
        protocol.update(variants=[dict(name='full',kind='full'),dict(name='random',kind='random'),
            dict(name='candidate',kind='cumulative')], intervention='cumulative fixed-noise replacement; no texture reoptimization',
            selection_order='ascending mean absolute per-sample normalized change from frozen scan',
            cumulative_windows=accepted+[wi], replacement_seed=20261130,
            threshold='mean candidate L0 >= 0.8 * mean full L0; first rejection stops and rolls back')
        for name,rgb in [('full',full),('random',random_rgb),('candidate',candidate)]:
            path=textures/(name+'.png')
            if path.exists():assert np.array_equal(np.asarray(Image.open(path)),rgb)
            else:Image.fromarray(rgb).save(path)
        protocol['texture_sha256']={name:sha(textures/(name+'.png')) for name in ['full','random','candidate']}
        if (root/'protocol.json').exists():assert json.loads((root/'protocol.json').read_text())==protocol
        else:dump(root/'protocol.json',protocol)
        run.status('cumulative_render_replacement',step=step,window=wi,accepted=len(accepted),scan=str(scan))
        if not all((root/'capture'/f"sample{r['id']:03d}.npz").exists() for r in p['samples']):
            run.run(f'prune{step:02d}_capture',run.sim('robust_occlusion.py','capture','--root',root))
        if not all((root/'metrics'/f"sample{r['id']:03d}.npz").exists() for r in p['samples']):
            run.run(f'prune{step:02d}_measure',run.model('robust_occlusion.py','measure','--root',root))
        result=mean_l0(root,p['samples']);ok=result['ratio']>=.8
        trace.append(dict(step=step,window=wi,accepted=ok,**result))
        dump(dest/'trace.json',trace)
        if not ok:
            Image.fromarray(candidate).save(dest/'first_rejected.png')
            break
        current=candidate;accepted.append(wi)
    Image.fromarray(current).save(dest/'retained.png')
    mask=np.ones((128,128),dtype=bool)
    for wi in accepted:
        w=p['windows'][wi];mask[w['y']:w['y']+16,w['x']:w['x']+16]=False
    Image.fromarray((mask*255).astype(np.uint8)).save(dest/'retained_mask.png')
    for rep,seed in enumerate([20261131,20261132],1):
        bg=np.random.RandomState(seed).randint(0,256,full.shape,dtype=np.uint8)
        Image.fromarray(np.where(mask[...,None],full,bg)).save(dest/f'retained_random{rep}.png')
    dump(dest/'selection.json',dict(complete=True,accepted_windows=accepted,original_texture_fraction=float(mask.mean()),
        final_ratio=next((t['ratio'] for t in reversed(trace) if t['accepted']),1.0),
        endpoint='actual-render open-loop L0; pruned ASR not yet measured',
        physical_outline_m=p['size_m'],notes='removed area can include occluded texels; no minimum-area or standalone-structure claim'))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--gpu',default='6')
    ap.add_argument('--port',type=int,default=8397)
    ap.add_argument('--launch',action='store_true')
    args=ap.parse_args();root=Path(args.root).resolve()
    closedloop_prepare(root)
    if args.launch:
        with (root/'followup_daemon.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),
                '--gpu',args.gpu,'--port',str(args.port)],cwd=B.ROOT,start_new_session=True,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
        dump(root/'followup_launch.json',dict(pid=proc.pid,gpu=args.gpu,port=args.port,detached=True,launched_unix=time.time()))
        print(json.dumps(dict(pid=proc.pid,detached=True)),flush=True)
        return
    lock=(root/'followup.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=Runner(root,args.gpu,'followup')
    try:
        summary_path=root/'summary.json'
        if not (summary_path.exists() and json.loads(summary_path.read_text())['complete']):
            run.run('resume_closed_loops',run.model('table9_closedloop_pipeline.py','--root',root,
                '--gpu',args.gpu,'--port',args.port,'--workers',2))
        summary=json.loads(summary_path.read_text());assert summary['complete']
        by={(r['site'],r['condition'],r['method']):r for r in summary['rows']}
        p=json.loads((root/'protocol.json').read_text());candidates=[]
        for s in p['sites']:
            rows=[by[(s['id'],c,'trained')] for c in ['nominal','position','joint']]
            controls=[by[(s['id'],c,'random')] for c in ['nominal','position','joint']]
            if rows[0]['clean_success_pairs']<3 or any(r['ASR'] is None for r in rows+controls):continue
            excess=[r['ASR']-c['ASR'] for r,c in zip(rows,controls)]
            if min(excess)>0:
                candidates.append(dict(site=s['id'],center=s['center_m'],excess=excess,score=float(np.mean(excess))))
        candidates.sort(key=lambda c:(-c['score'],c['site']))
        dump(root/'render_ablation_selection.json',dict(candidates=candidates,
            rule='minimum three clean-success original pairs; positive trained-minus-Random ASR in original, position and joint conditions; rank mean excess',
            limitation='pilot screening, not statistical proof of robustness; downstream ASR uses separate initial states',
            closedloop_summary_sha256=sha(summary_path)))
        if not candidates:
            run.status('no_supported_candidate',complete=False,
                reason='No patch beats matched Random in all three closed-loop conditions; avoid interpreting weak attack signals as structure')
            return
        selected=candidates[0];scan=root/'render_ablation'/selected['site']
        source=Path(p['source'])/'sites'/selected['site']/'train/texture_0500.png'
        scan_prepare(scan,source,selected['center'],.10,.03)
        run.status('render_sliding_replacement',site=selected['site'],scan=str(scan))
        if not (scan/'summary.json').exists():
            run.run('render_scan_capture',run.sim('robust_occlusion.py','capture','--root',scan))
            run.run('render_scan_measure',run.model('robust_occlusion.py','measure','--root',scan))
        if not (scan/'heatmaps.png').exists():
            run.run('render_scan_report',[PLOT_PYTHON,'-u',str(B.PROBE/'robust_occlusion.py'),'report','--root',str(scan)])
        from render_pruning_batch import run as pruning_run
        pruning_run(run,scan)
        run.status('render_replacement_complete',complete=True,scan=str(scan),
            next_stage='independent closed-loop ASR of retained, changed-background and first-rejected textures')
    except BaseException as exc:
        run.status('failed',complete=False,error=repr(exc));raise
    finally:
        run.close()


if __name__=='__main__':main()
