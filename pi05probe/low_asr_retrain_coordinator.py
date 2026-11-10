"""CPU-only runner for audited corrective reruns; no process stop operations."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parent.parent
DEST=ROOT/'out/main_results/low_asr_retrain'
ENVROOT=Path(os.environ.get('WAMATTACK_ENVROOT','/shared/user65/miniconda3/envs'))


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def main(args):
    host=socket.gethostname().split('.')[0]
    libero,robo=json.loads(args.libero_gpu),json.loads(args.robotwin_gpu)
    assert libero['host']==robo['host']==host
    shared_gpu=libero['uuid']==robo['uuid']
    DEST.mkdir(parents=True,exist_ok=True)
    with (DEST/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        specs={
            'libero':dict(command=[str(ENVROOT/'openpi-server/bin/python'),'-u',
                str(ROOT/'pi05probe/low_asr_retrain_pipeline.py'),'--gpu',json.dumps(libero)],
                progress=ROOT/'pi05probe/out/low_asr_retrain_v1/progress.json',
                summary=ROOT/'pi05probe/out/low_asr_retrain_v1/summary.json',gpu=libero),
            'robotwin':dict(command=[str(ENVROOT/'openpi-server/bin/python'),'-u',
                str(ROOT/'robotwinprobe/low_asr_retry_pipeline.py'),'--gpu',json.dumps(robo)],
                progress=ROOT/'robotwinprobe/out/low_asr_retry_20261108/progress.json',
                summary=ROOT/'robotwinprobe/out/low_asr_retry_20261108/summary.json',gpu=robo)}
        state=dict(started=now(),host=host,pid=os.getpid(),stage='running',
                   scope='five LIBERO low-ASR tasks + RoboTwin place_container_plate',
                   baseline='original 7+1 main result retained',
                   scheduling='RoboTwin then LIBERO, one GPU' if shared_gpu else 'separate GPUs',jobs={})
        procs,streams,next_start={},{},{}
        def start(name):
            spec=specs[name]
            streams[name]=(DEST/(name+'.log')).open('a')
            proc=subprocess.Popen(spec['command'],cwd=ROOT/'pi05probe',
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=streams[name],
                stderr=subprocess.STDOUT,start_new_session=True)
            procs[name]=proc
            previous=state['jobs'].get(name,{})
            state['jobs'][name]=dict(pid=proc.pid,gpu=spec['gpu'],command=spec['command'],
                log=str(DEST/(name+'.log')),status='running',started=now(),
                resource_retry_count=previous.get('resource_retry_count',0))
        for name in specs:
            if shared_gpu and name=='libero':
                state['jobs'][name]=dict(status='waiting_for_robotwin',gpu=libero,resource_retry_count=0)
                procs[name]=None
            else:start(name)
        dump(DEST/'launch.json',state)
        try:
            while True:
                for name,spec in specs.items():
                    job=state['jobs'][name];proc=procs[name]
                    if job['status'] in ['complete','failed']:continue
                    if job['status']=='waiting_for_robotwin':
                        if state['jobs']['robotwin']['status'] in ['complete','failed']:start(name)
                        continue
                    if job['status']=='waiting_for_idle_gpu':
                        if time.monotonic()>=next_start[name]:start(name)
                        continue
                    code=proc.poll()
                    if code is None:continue
                    streams[name].close();job.update(exit_code=code,finished=now())
                    progress=json.loads(spec['progress'].read_text()) if spec['progress'].is_file() else {}
                    if code!=0 and progress.get('stage')=='blocked':
                        job.update(status='waiting_for_idle_gpu',resource_retry_count=job['resource_retry_count']+1,
                            note='worker will repeat fleet, UUID mapping and idle selected-card checks before any GPU work')
                        next_start[name]=time.monotonic()+60
                    elif code==0 and spec['summary'].is_file() and json.loads(spec['summary'].read_text()).get('complete'):
                        job.update(status='complete',summary=str(spec['summary']))
                    else:
                        job.update(status='failed',detail=progress)
                terminal=all(j['status'] in ['complete','failed'] for j in state['jobs'].values())
                if terminal:state['stage']='complete' if all(j['status']=='complete' for j in state['jobs'].values()) else 'incomplete'
                dump(DEST/'progress.json',dict(state,updated=now()))
                if terminal:break
                time.sleep(10)
        finally:
            # No signal, stop request, or cleanup of an active child is authorized.
            for stream in streams.values():stream.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--libero-gpu',required=True);ap.add_argument('--robotwin-gpu',required=True)
    main(ap.parse_args())
