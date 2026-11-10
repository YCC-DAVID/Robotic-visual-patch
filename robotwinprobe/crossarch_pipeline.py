"""Resumable public-policy cross-architecture experiment; each simulator in isolation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from crossarch_common import DEFAULT,ROOT,WAM_WEIGHTS,AE,PY,STAGES,dump,sha,prepare,request
from crossarch_report import report

def reviewed_clean_gate(root,pilot_counts):
    """Resume a diagnostic hold only with a complete, identical historical replay."""
    regression=root/'clean_regression_10200_10209'
    comparison=json.loads((regression/'comparison.json').read_text())
    identity=json.loads((regression/'weight_identity.json').read_text())
    keys=('success_matches','initial_state_and_rgb_match','prompt_matches',
          'full_state_sequence_equal','full_action_sequence_equal','full_head_sequence_equal')
    assert comparison['complete'] and comparison['current_completed']==10
    assert {r['seed'] for r in comparison['rows']}==set(range(10200,10210))
    assert all(all(r[k] for k in keys) for r in comparison['rows'])
    assert identity['matches_current_crossarch_provenance']
    assert pilot_counts=={'pi':5,'wam':8},'Review applies only to the inspected pilot'
    for model in ('pi','wam'):
        assert json.loads((root/f'audit/{model}.json').read_text())['passed']
    review=dict(accepted=True,original_gate_minimum=6,pilot_successes=pilot_counts,
        original_gate_passed=False,legacy_clean_successes=comparison['current_successes'],
        legacy_comparison_sha256=sha(regression/'comparison.json'),
        weight_identity_sha256=sha(regression/'weight_identity.json'),
        reason='Historical ten-seed clean replay is identical in states, images, actions and outcomes; public RoboTwin weight identity verified. User requested patch training after this review. Retain the original pilot and all formal evaluation seeds.',
        proxy_fidelity_gate_unchanged=True,updated=time.time())
    dump(root/'audit/clean_gate_review.json',review)

def run(root,pilot_only=False,pi_gpu='0',wam_gpu='1',sim_gpu=None,resume_reviewed_clean=False):
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=prepare(root);logs=root/'logs';logs.mkdir(exist_ok=True);workers=[]
    base=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',
              PYTHONHASHSEED='0',JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    def status(stage,**extra):
        dump(root/'status.json',dict(stage=stage,pid=os.getpid(),updated=time.time(),**extra));report(root)
    sim_gpu=pi_gpu if sim_gpu is None else sim_gpu
    def call(name,argv,gpu=None):
        gpu=sim_gpu if gpu is None else gpu
        status(name)
        with (logs/(name+'.log')).open('a') as f:
            proc=subprocess.Popen(argv,cwd=ROOT,env=dict(base,CUDA_VISIBLE_DEVICES=gpu),stdout=f,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                time.sleep(10);status(name,child_pid=proc.pid)
            if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}; see {logs/(name+".log")}')
    sim=lambda script,*args:['bash',str(ROOT/'robotwinprobe/run_sim.sh'),str(ROOT/'robotwinprobe'/script),'--root',str(root),*map(str,args)]
    try:
        # Keep the original source/status evidence when resuming a stopped run.
        snapshot=root/'source_snapshots'/time.strftime('%Y%m%d_%H%M%S');snapshot.mkdir(parents=True,exist_ok=True)
        if (root/'status.json').exists():shutil.copy2(root/'status.json',snapshot/'previous_status.json')
        if (root/'audit/clean_pilot.json').exists():shutil.copy2(root/'audit/clean_pilot.json',snapshot/'previous_clean_pilot.json')
        for source in (ROOT/'robotwinprobe').glob('crossarch_*.py'):shutil.copy2(source,snapshot/source.name)
        dump(snapshot/'execution.json',dict(pi_gpu=pi_gpu,wam_gpu=wam_gpu,sim_gpu=sim_gpu,
            resume_reviewed_clean=resume_reviewed_clean,pilot_only=pilot_only))
        if not (root/'provenance.json').exists():
            status('hashing_public_assets')
            files=[WAM_WEIGHTS/'model.pt',WAM_WEIGHTS/'config.yaml',WAM_WEIGHTS/'dataset_stats.json',AE,
                ROOT/'checkpoints/pi05_robotwin2_pytorch/model.safetensors']
            dump(root/'provenance.json',{str(f):sha(f) for f in files})
        if not (root/'prompts.npz').exists():call('prompts',[PY['wam'],str(ROOT/'robotwinprobe/crossarch_wam.py')],gpu=wam_gpu)
        for model,gpu in [('pi',pi_gpu),('wam',wam_gpu)]:
            if pilot_only and all((root/f'pilot/{model}/clean/seed{s}.json').exists() for s in p['evaluation']['pilot_seeds']):continue
            status('start_'+model)
            ready=root/f'{model}_ready.json'
            if ready.exists():ready.unlink()
            f=(logs/(model+'_backend.log')).open('a')
            proc=subprocess.Popen([PY[model],'-u',str(ROOT/'robotwinprobe/crossarch_backend.py'),'--root',str(root),'--model',model],
                cwd=ROOT,env=dict(base,CUDA_VISIBLE_DEVICES=gpu),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append(proc);f.close();deadline=time.monotonic()+600
            while not ready.exists():
                if proc.poll() is not None:raise RuntimeError(model+' backend failed')
                if time.monotonic()>deadline:raise TimeoutError(model+' startup')
                time.sleep(5)
            assert request(model,{'command':'ping'})['model']==model
        for model in ('pi','wam'):
            for seed in p['evaluation']['pilot_seeds']:
                existing=root/f'pilot/{model}/clean/seed{seed}.json'
                if existing.exists():
                    assert 'success' in json.loads(existing.read_text()),'Unresolved episode error: '+str(existing)
                    continue
                call(f'pilot_{model}_{seed}',sim('crossarch_rollout.py','--model',model,'--condition','clean','--seed',seed,'--pilot'))
        pilot_counts={model:sum(json.loads((root/f'pilot/{model}/clean/seed{s}.json').read_text())['success']
            for s in p['evaluation']['pilot_seeds']) for model in ('pi','wam')}
        dump(root/'audit/clean_pilot.json',dict(successes=pilot_counts,total_per_model=10,gate_minimum=6,
            passed=all(n>=6 for n in pilot_counts.values()),note='Finish independent model checks before applying the joint training gate.'))
        if pilot_only:
            status('clean_pilot_complete_training_held',successes=pilot_counts);return
        if not all(n>=6 for n in pilot_counts.values()):
            assert resume_reviewed_clean,str(pilot_counts)+' clean pilot below 6/10; inspect interface before training'
            reviewed_clean_gate(root,pilot_counts)
            status('clean_hold_reviewed',successes=pilot_counts)
        for seed in p['data']['validation_seeds']+p['data']['train_seeds']:
            call(f'trace_{seed}',sim('crossarch_capture.py','--mode','trace','--seed',seed))
        for seed in p['data']['validation_seeds']:
            call(f'capture_{seed}',sim('crossarch_capture.py','--mode','capture','--seed',seed))
        from crossarch_capture import records
        validation=[]
        for seed in p['data']['validation_seeds'][:2]:
            rs=records(root,seed)
            for s in STAGES:
                validation.extend(r['path'] for r in [x for x in rs if x['stage']==s][:2])
        for model in ('pi','wam'):
            status('proxy_validation_'+model)
            result=request(model,dict(command='validate',records=validation))
            assert result['passed'],model+' proxy failed fidelity gate; inspect audit before training'
        for seed in p['data']['train_seeds']:
            call(f'capture_{seed}',sim('crossarch_capture.py','--mode','capture','--seed',seed))
        call('patch_training',[PY['cpu'],'-u',str(ROOT/'robotwinprobe/crossarch_train.py')])
        for seed in p['data']['validation_seeds']:
            call(f'post_{seed}',sim('crossarch_capture.py','--mode','post','--seed',seed))
        call('diagnostics',[PY['cpu'],'-u',str(ROOT/'robotwinprobe/crossarch_diagnostics.py')])
        for model in ('pi','wam'):
            for condition in p['evaluation']['conditions']:
                for seed in p['evaluation']['seeds']:
                    call(f'eval_{model}_{condition}_{seed}',sim('crossarch_rollout.py','--model',model,'--condition',condition,'--seed',seed))
        status('complete')
    except Exception as e:
        status('stopped_on_error',error=repr(e),traceback=traceback.format_exc());raise
    finally:
        # Only processes created by this runner are terminated.
        for proc in workers:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=20)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--pilot-only',action='store_true')
    ap.add_argument('--pi-gpu',default='0');ap.add_argument('--wam-gpu',default='1')
    ap.add_argument('--sim-gpu');ap.add_argument('--resume-reviewed-clean',action='store_true')
    a=ap.parse_args();run(a.root,a.pilot_only,a.pi_gpu,a.wam_gpu,a.sim_gpu,a.resume_reviewed_clean)
