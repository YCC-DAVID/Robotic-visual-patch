"""User-authorized 10 cm FastWAM trial, conditional independent 20 cm trial."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from crossarch_common import ROOT, dump, sha
from crossarch_stage_common import DEFAULT as CALIBRATION
from fastwam_render_trial import DEFAULT as DIAGNOSIS, PY, OBJECTIVES

DEFAULT=ROOT/'robotwinprobe/out/fastwam_size_sequence_20261027'


def link(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():os.link(source,target)


def prepare_trial(sequence,size):
    root=sequence/f'{size}cm';source=json.loads((DIAGNOSIS/'protocol.json').read_text())
    root.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(source)
    p.update(experiment='fastwam_render_patch_exploratory',source_run=str(DIAGNOSIS),
             control_port=18602 if size==10 else 18603,
             report_path=str(ROOT/'result'/f'40_fastwam_{size}cm_patch.md'))
    p['authorization']=dict(continue_with_proxy_diagnostic_only=True,
        instruction='Try FastWAM; if not successful try 20x20 cm; also investigate LIBERO 10x10 cm or DRIFT-like patch.',
        recorded_at='2026-10-27',scope='Actual rendered image patch only; frozen policy; no internal velocity interventions')
    p['gates']['proxy_is_diagnostic_only']=True
    p['gates']['failure_rule']='Native parity, finite nonzero gradients, clean capability remain mandatory. Preserve proxy mismatch as diagnostic; actual-render paired outcomes are primary.'
    p['patch']['size_m']=size/100
    p['patch_trial']['training_start']='after native parity, independent FastWAM clean baseline and finite texture gradients; proxy cosine retained as diagnostic'
    p['patch_trial']['initialization']=f'gray .5, Q128,{size}cm; identical 3cm/15degree/three-light EOT'
    p['policies']={'fastwam':str(root/'checkpoints/robotwin_uncond_3cam_384.pt')}
    p['evaluation']['baseline_reuse']='Reuse the verified FastWAM clean-only ten episodes from initial diagnosis; new Random and trained-patch episodes for each size'
    p['evaluation']['expected_reused_baselines']=10
    p['evaluation']['expected_new_patch_episodes']=30
    if size==20:
        for record in p['diagnosis']['records']:
            record['path']=str(root/'bank/validation'/Path(record['path']).name)
    path=root/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:dump(path,p)
    for name in ('prompts.npz','release_metadata.json'):
        link(DIAGNOSIS/name,root/name)
    for name in ('robotwin_uncond_3cam_384.pt','robotwin_uncond_3cam_384_dataset_stats.json'):
        link(DIAGNOSIS/'checkpoints'/name,root/'checkpoints'/name)
    if size==10:
        for name in ('native_proxy.json','ascent_direction_probe.json','gradient_wrapper_diagnosis.json'):
            # Audit evidence is copied, never edited through a hard link.
            dest=root/'audit'/name;dest.parent.mkdir(exist_ok=True)
            if not dest.exists():shutil.copy2(DIAGNOSIS/'audit'/name,dest)
    else:
        for seed in p['data']['train_seeds']+p['data']['validation_seeds']:
            for name in ('complete.json','trace.npz','frames.npz','stages.png'):
                link(CALIBRATION/'trajectories'/f'seed{seed}'/name,root/'trajectories'/f'seed{seed}'/name)
    plan=json.loads((DIAGNOSIS/'training_plan.json').read_text())
    if size==20:
        for draws in plan:
            for draw in draws:draw['record']=str(root/'bank/train'/Path(draw['record']).name)
    if (root/'training_plan.json').exists():assert json.loads((root/'training_plan.json').read_text())==plan
    else:dump(root/'training_plan.json',plan)
    references=[]
    for seed in p['evaluation']['seeds']:
        source_record=DIAGNOSIS/'episodes/fastwam/clean'/f'seed{seed}.json'
        row=json.loads(source_record.read_text())
        row.update(protocol_sha256=sha(path),reference_reused=True,source_record=str(source_record),
                   source_record_sha256=sha(source_record),source_protocol_sha256=json.loads(source_record.read_text())['protocol_sha256'])
        target=root/'episodes/fastwam/clean'/source_record.name
        if target.exists():assert json.loads(target.read_text())==row
        else:dump(target,row)
        link(source_record.with_suffix('.npz'),target.with_suffix('.npz'))
        references.append(dict(source=str(source_record),sha256=sha(source_record)))
    dump(root/'audit/reused_clean.json',dict(scope='same FastWAM checkpoint, native inputs, state/noise, replanning; clean contains no patch, independent of patch size',records=references))
    return root


def capture_bank(root,seed):
    import crossarch_capture as capture
    capture.prepare=lambda _:json.loads((root/'protocol.json').read_text())
    capture.capture(root,seed)


def outcome(root):
    p=json.loads((root/'protocol.json').read_text())
    clean={seed:json.loads((root/'episodes/fastwam/clean'/f'seed{seed}.json').read_text())['success'] for seed in p['evaluation']['seeds']}
    counts={};discordances={}
    all_rows={name:{seed:json.loads((root/'episodes/fastwam'/name/f'seed{seed}.json').read_text())
              for seed in p['evaluation']['seeds']} for name in ('random_full',*OBJECTIVES)}
    for name,rows in all_rows.items():
        assert all('success' in r for r in rows.values()),'Runtime errors cannot become task failures'
        counts[name]=sum(clean[seed] and not r['success'] for seed,r in rows.items())
        if name!='random_full':
            discordances[name]=dict(patch_fails_random_succeeds=[seed for seed,r in rows.items() if clean[seed] and not r['success'] and all_rows['random_full'][seed]['success']],
                random_fails_patch_succeeds=[seed for seed,r in rows.items() if clean[seed] and r['success'] and not all_rows['random_full'][seed]['success']])
    no_gain=all(counts[name]<=counts['random_full'] for name in OBJECTIVES)
    result=dict(eligible=sum(clean.values()),paired_failures=counts,discordances=discordances,
        upgrade_to_20cm=no_gain,criterion='Both trained patches fail to exceed matched Random paired failure count; exploratory count criterion, not a significance claim.')
    dump(root/'size_decision.json',result)
    return result


def run(sequence,gpu):
    sequence.mkdir(parents=True,exist_ok=True)
    lock=open(sequence/'sequence.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    specification=dict(order=[10,20],conditional_20cm=True,
        trigger='both 10cm trained patches have no paired failure-count gain over 10cm Random',
        patch_objectives=list(OBJECTIVES),updates_per_patch=500,batch=8,
        only_change='physical patch size; new photometric/occlusion calibration and new same-size Random required',
        libero='Separate 10cm physical-render trial or DRIFT-style wrist-image digital diagnostic; no equivalence claim',
        user_instruction='If FastWAM is unsuccessful, try 20x20; LIBERO can use 10x10 or DRIFT-like settings')
    if (sequence/'sequence_protocol.json').exists():assert json.loads((sequence/'sequence_protocol.json').read_text())==specification
    else:dump(sequence/'sequence_protocol.json',specification)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
    try:
        for size in (10,20):
            if size==20 and not outcome(sequence/'10cm')['upgrade_to_20cm']:
                dump(sequence/'status.json',dict(stage='complete_10cm_positive_no_size_escalation',pid=os.getpid(),updated=time.time()));return
            root=prepare_trial(sequence,size)
            if size==20:
                p=json.loads((root/'protocol.json').read_text())
                for seed in p['data']['train_seeds']+p['data']['validation_seeds']:
                    dump(sequence/'status.json',dict(stage='calibration_20cm',seed=seed,pid=os.getpid(),updated=time.time()))
                    log=root/'logs'/f'calibration_{seed}.log';log.parent.mkdir(exist_ok=True)
                    with log.open('a') as f:
                        subprocess.run(['bash',str(ROOT/'robotwinprobe/run_sim.sh'),str(Path(__file__).resolve()),
                            'capture','--root',str(root),'--seed',str(seed)],env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
            dump(sequence/'status.json',dict(stage=f'trial_{size}cm',pid=os.getpid(),updated=time.time(),trial_root=str(root)))
            with (root/'pipeline.log').open('a') as log:
                subprocess.run([PY,'-u',str(ROOT/'robotwinprobe/fastwam_render_trial.py'),'pipeline','--root',str(root),'--gpu',str(gpu)],
                    env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            s=json.loads((root/'status.json').read_text())
            if s['stage']!='complete':
                raise RuntimeError(f'{size}cm trial did not finish: {s}')
            decision=outcome(root)
            print('[FastWAM size result]',size,decision,flush=True)
        dump(sequence/'status.json',dict(stage='complete',pid=os.getpid(),updated=time.time()))
    except Exception as error:
        dump(sequence/'status.json',dict(stage='runtime_error',pid=os.getpid(),updated=time.time(),error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['run','prepare','capture'])
    ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--gpu',type=int,default=3)
    ap.add_argument('--size',type=int,default=10);ap.add_argument('--seed',type=int)
    a=ap.parse_args()
    if a.command=='run':run(a.root,a.gpu)
    elif a.command=='prepare':prepare_trial(a.root,a.size)
    else:capture_bank(a.root,a.seed)
