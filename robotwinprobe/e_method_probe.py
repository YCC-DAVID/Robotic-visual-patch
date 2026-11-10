"""Post-round development-only diagnosis using saved exact real-render inputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from e_method_common import DEFAULT,SCREEN,CAMERAS,WIRE,dump,load,request,sha


def metrics(clean,patched,active_start):
    dv=np.asarray(patched['v'],np.float64)-np.asarray(clean['v'],np.float64)
    da=np.asarray(patched['action'],np.float64)-np.asarray(clean['action'],np.float64)
    active=slice(active_start,active_start+7)
    full=float(np.sum(dv*dv));effective=float(np.sum(dv[...,:14]**2))
    return dict(first14_mse=float(np.mean(dv[...,:14]**2)),first32_mse=float(np.mean(dv**2)),
        active_first7_mse=float(np.mean(dv[...,active]**2)),
        discarded_velocity_energy_fraction=(full-effective)/max(full,1e-30),
        final14_mse=float(np.mean(da[...,:14]**2)),
        active_final7_mse=float(np.mean(da[...,active]**2)),
        active_gripper_mse=float(np.mean(da[...,active_start+6]**2)))


def run(root):
    p=load(root);seeds=p['method_study']['development_seeds']
    folder=root/'cohorts/development/episodes';out=root/'transfer_diagnostic';out.mkdir(exist_ok=True)
    conditions=[c for c in SCREEN if c!='clean'];began=time.monotonic()
    config=dict(scope='development initial state only; real saved SAPIEN RGB; not new closed-loop ASR',
        seeds=seeds,conditions=conditions,extra_noise_seeds=[202610280,202610281],
        protocol_sha256=sha(root/'protocol.json'),test_used=False)
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==config
    else:dump(path,config)
    all_rows=[]
    for seed in seeds:
        target=out/f'seed{seed}.json'
        if target.exists():
            saved=json.loads(target.read_text());assert saved['diagnostic_protocol_sha256']==sha(path)
            all_rows.extend(saved['rows']);continue
        rows=[];clean_cache={};sources={};reference=None
        clean_outcome=json.loads((folder/f'native_full_approach_clean_s{seed}.json').read_text())
        for condition in conditions:
            source=folder/f'native_full_approach_{condition}_s{seed}.json'
            outcome=json.loads(source.read_text());artifact=source.with_suffix('.npz')
            assert outcome['status']=='complete' and sha(artifact)==outcome['artifact_sha256']
            assert outcome['initial_clean_sha256']==clean_outcome['initial_clean_sha256']
            sources[str(artifact)]=outcome['artifact_sha256']
            with np.load(artifact) as z:
                state=z['states14'][0].copy()
                clean_images={w:z['before_'+c].copy() for c,w in zip(CAMERAS,WIRE)}
                patched_images={w:z['after_'+c].copy() for c,w in zip(CAMERAS,WIRE)}
                native_noise=int(z['noise_seeds'][0])
                active_start=7 if z['object_plate'][0,0,0]>0 else 0
            if reference is None:reference=(state,clean_images)
            assert np.array_equal(state,reference[0])
            for camera in WIRE:assert np.array_equal(clean_images[camera],reference[1][camera])
            for noise in [native_noise,*config['extra_noise_seeds']]:
                if noise not in clean_cache:
                    clean_cache[noise]=request(dict(command='diagnostic',state=state,prompt=outcome['prompt'],
                        images=clean_images,seed=noise),root)
                patched=request(dict(command='diagnostic',state=state,prompt=outcome['prompt'],
                    images=patched_images,seed=noise),root)
                rows.append(dict(seed=seed,condition=condition,noise_seed=noise,
                    native_noise=noise==native_noise,clean_success=clean_outcome['success'],
                    closed_loop_attack_success=bool(clean_outcome['success'] and not outcome['success']) if noise==native_noise else None,
                    active_start=active_start,**metrics(clean_cache[noise],patched,active_start)))
        dump(target,dict(seed=seed,rows=rows,sources=sources,diagnostic_protocol_sha256=sha(path)))
        all_rows.extend(rows)
        dump(out/'progress.json',dict(completed_seeds=len({r['seed'] for r in all_rows}),total_seeds=len(seeds),
            rows=len(all_rows),elapsed_s=time.monotonic()-began,complete=False))
        print('[transfer diagnostic]',seed,len(rows),flush=True)
    summaries=[]
    for condition in conditions:
        subset=[r for r in all_rows if r['condition']==condition]
        per_seed={s:[r for r in subset if r['seed']==s] for s in seeds}
        values={key:[float(np.mean([r[key] for r in group])) for group in per_seed.values()]
                for key in ['first14_mse','first32_mse','active_first7_mse','discarded_velocity_energy_fraction',
                            'final14_mse','active_final7_mse','active_gripper_mse']}
        original=[r for r in subset if r['native_noise'] and r['clean_success']]
        x=np.log10(np.maximum(values['first14_mse'],1e-20));y=np.log10(np.maximum(values['final14_mse'],1e-20))
        correlation=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
        summaries.append(dict(condition=condition,independent_seeds=len(per_seed),
            median_over_seeds={k:float(np.median(v)) for k,v in values.items()},
            log_first14_final14_pearson=correlation,native_clean_successes=len(original),
            groups={label:dict(n=len(group),mean_initial_L0=float(np.mean([r['first14_mse'] for r in group])) if group else None,
                      mean_initial_final_action_mse=float(np.mean([r['final14_mse'] for r in group])) if group else None)
                for label,group in [('attack_success',[r for r in original if r['closed_loop_attack_success']]),
                                    ('attack_failure',[r for r in original if not r['closed_loop_attack_success']])]}))
    dump(out/'summary.json',dict(complete=True,rows=summaries,total_rows=len(all_rows),scope=config['scope'],
        aggregation='average three noises within each seed, then median across32 seeds; initial-only association is not causation',
        test_used=False,diagnostic_protocol_sha256=sha(path)))
    dump(out/'progress.json',dict(completed_seeds=len(seeds),total_seeds=len(seeds),rows=len(all_rows),complete=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    run(parser.parse_args().root.resolve())
