"""Owned, resumable queue for nine independent small-patch experiments."""
import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import bootstrap as B
from radius_plan import dump
from radius_pipeline import Runner
from table9_plan import prepare,sha

CPU_PY='/home/user1/miniconda3/bin/python'


def wait_gpu(run,minimum_mib):
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={int(x.split(',')[0]):int(x.split(',')[1]) for x in raw.strip().splitlines()}
        if free[int(run.gpu)]>=minimum_mib:return
        run.status('waiting_gpu_memory',required_mib=minimum_mib,free_mib=free[int(run.gpu)])
        time.sleep(30)


def run_stage(run,label,cmd,model=False):
    wait_gpu(run,30000 if model else 4000)
    run.run(label,cmd)


def trained(root,site):
    p=root/'sites'/site/'train/progress.json'
    return p.exists() and json.loads(p.read_text()).get('complete',False)


def site_train(run,site):
    root=run.root;prepare(root)
    if trained(root,site):return
    for split in ['validation','train']:
        marker=root/'sites'/site/f'{split}_capture_complete.json'
        if not marker.exists():
            run_stage(run,f'{site}_capture_{split}',run.sim('table9_capture.py','--root',root,'--site',site,'--split',split))
        if split=='validation':
            marker=root/'sites'/site/'proxy_validation.json'
            if not marker.exists():
                run_stage(run,f'{site}_validate',run.model('table9_train.py','--root',root,'--site',site,'--stage','validate'),model=True)
            assert json.loads(marker.read_text())['passed'], 'Failed proxy fidelity; inspect before continuing'
    run_stage(run,f'{site}_train',run.model('table9_train.py','--root',root,'--site',site,'--stage','train'),model=True)
    assert trained(root,site)


def site_post(run,site):
    root=run.root
    assert all(trained(root,f'site{i:02d}') for i in range(9))
    if not (root/'sites'/site/'post_capture_complete.json').exists():
        run_stage(run,f'{site}_post_capture',run.sim('table9_capture.py','--root',root,'--site',site,'--split','post'))
    run_stage(run,f'{site}_cross_probe',run.model('table9_probe.py','--root',root,'--site',site),model=True)


def snapshot(root):
    dest=root/'source_snapshot';dest.mkdir(exist_ok=True)
    paths=list(B.PROBE.glob('table9_*.py'))+[B.PROBE/n for n in ['bootstrap.py','radius_plan.py','radius_pipeline.py','render_model.py','render_proxy.py','render_geometry.py','scene_smoke.py','eot_conditions.py','digital_transform_common.py','run_sim.sh']]
    hashes={}
    for p in paths:
        out=dest/p.name
        if out.exists():assert out.read_bytes()==p.read_bytes(),f'Source changed since queue was frozen: {p}'
        else:out.write_bytes(p.read_bytes())
        hashes[p.name]=sha(p)
    dump(dest/'sha256.json',hashes)


def check_snapshot(root):
    expected=json.loads((root/'source_snapshot/sha256.json').read_text())
    for name,h in expected.items():assert sha(B.PROBE/name)==h,f'Source changed while queued: {name}'


def geometry(run):
    root=run.root
    while not (root/'geometry/verify_capture_complete.json').exists():
        launch=json.loads((root/'geometry_verify_launch.json').read_text())
        proc=Path('/proc')/str(launch['pid'])/'cmdline'
        alive=proc.exists() and b'table9_geometry_capture.py' in proc.read_bytes()
        if not alive:
            run_stage(run,'resume_geometry_verify',run.sim('table9_geometry_capture.py','--root',root,'--verify'))
            break
        progress=json.loads((root/'geometry/progress.json').read_text())
        run.status('geometry_verification_running',geometry_pid=launch['pid'],progress=progress)
        time.sleep(30)
    run.run('geometry_report',[CPU_PY,str(B.PROBE/'table9_geometry_verify.py'),'--root',str(root)])
    prepare(root)


def queue(run,gpus,stage):
    pending=[f'site{i:02d}' for i in range(9)];active={};errors=[]
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:continue
            site=pending.pop(0)
            if stage=='train' and trained(run.root,site):continue
            if stage=='post' and (run.root/'sites'/site/'cross_probe.json').exists():continue
            env=run.env.copy();env['CUDA_VISIBLE_DEVICES']=str(gpu)
            cmd=run.model('table9_pipeline.py','--root',run.root,'--stage',stage,'--site',site,'--gpu',gpu)
            p=run.launch(f'{stage}_{site}',cmd,env)
            active[gpu]=(p,site)
        for gpu,(p,site) in list(active.items()):
            if p.poll() is not None:
                if p.returncode:errors.append(dict(site=site,exit_code=p.returncode,log=f'logs/{stage}_{site}.log'))
                del active[gpu]
        run.status('independent_'+stage,active={gpu:dict(pid=p.pid,site=s) for gpu,(p,s) in active.items()},
            pending=pending,errors=errors,completed_training_sites=[f'site{i:02d}' for i in range(9) if trained(run.root,f'site{i:02d}')])
        if pending or active:time.sleep(30)
    assert not errors,f'{stage} failed at: {errors}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['full','train','post'],default='full')
    p.add_argument('--site');p.add_argument('--gpu',default='2');p.add_argument('--gpus',default='2,5');args=p.parse_args()
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    name='pipeline' if args.stage=='full' else f'{args.stage}_{args.site}'
    lock=open(root/(name+'.lock'),'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=Runner(root,args.gpu,name)
    run.env.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MPLCONFIGDIR='/tmp/table9_mpl')
    try:
        if args.stage=='full':
            gpus=args.gpus.split(',');assert len(gpus)==len(set(gpus))
            snapshot(root);geometry(run)
            queue(run,gpus,'train')
            run.run('texture_comparison',[CPU_PY,str(B.PROBE/'table9_report.py'),'--root',str(root)])
            queue(run,gpus,'post')
            run.run('final_comparison',[CPU_PY,str(B.PROBE/'table9_report.py'),'--root',str(root)])
        elif args.stage=='train':
            check_snapshot(root);site_train(run,args.site)
        else:
            check_snapshot(root);site_post(run,args.site)
        run.status('complete',single_patch_per_scene=True)
    except BaseException:
        run.status('failed',error=traceback.format_exc());raise
    finally:run.close()


if __name__=='__main__':main()
