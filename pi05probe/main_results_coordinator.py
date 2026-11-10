"""Main ASR batch: existing LIBERO EOT first, RoboTwin, remaining LIBERO tasks.

This CPU controller never reserves a GPU. Each benchmark worker performs its
own complete preflight and UUID check before every integrated GPU task.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'out/main_results'
FIRST='put_the_bowl_on_the_plate'
REMAINING=['put_the_bowl_on_top_of_the_cabinet','put_the_wine_bottle_on_the_rack',
           'turn_on_the_stove','put_the_bowl_on_the_stove','put_the_cream_cheese_in_the_bowl',
           'open_the_middle_drawer_of_the_cabinet']


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path): return json.loads(Path(path).read_text())


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_alive(pid):
    try:return b'main_method_pipeline.py' in (Path('/proc')/str(pid)/'cmdline').read_bytes()
    except OSError:return False


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpus',required=True)
    ap.add_argument('--first-controller-pid',required=True,type=int)
    args=ap.parse_args();gpus=json.loads(args.gpus)
    host=socket.gethostname().split('.')[0]
    assert gpus and all(g.get('host')==host for g in gpus)
    envroot=Path(os.environ.get('WAMATTACK_ENVROOT','/home/user1/miniconda3/envs'))
    sim=str(envroot/'openpi-libero/bin/python');model=str(envroot/'openpi-server/bin/python')
    env={**os.environ,'OPENPI_DATA_HOME':os.environ.get('WAMATTACK_OPENPI_DATA_HOME','/shared/user65/.cache/openpi'),
         'ROBOTWIN_PYTHON':str(envroot/'RoboTwin/bin/python'),'HF_HUB_OFFLINE':'1','PYTHONNOUSERSITE':'1'}
    first=ROOT/'pi05probe/out/main_method/results'/f'{FIRST}.json'
    phases=[
        dict(name='robotwin_main',argv=[model,'-u',str(ROOT/'robotwinprobe/main_method_pipeline.py'),
             '--gpus',json.dumps(gpus),'--tasks','place_container_plate'],cwd=str(ROOT),
             status=str(ROOT/'robotwinprobe/out/main_method/progress.json')),
        dict(name='libero_remaining',argv=[sim,'-u',str(ROOT/'pi05probe/main_method_pipeline.py'),
             '--gpus',json.dumps(gpus),'--tasks',','.join(REMAINING)],cwd=str(ROOT/'pi05probe'),
             status=str(ROOT/'pi05probe/out/main_method/progress.json'))]
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=dict(schema='main-asr-first-batch-v1',created=now(),host=host,gpus=gpus,
            main_method='13cm full-band render proxy + texture-plane EOT, 500x8, frozen pi0.5 per benchmark',
            conditions=['clean','random','method(EOT)'],
            metric='per task: method failure among matched clean successes; errors excluded and reported',
            libero_tasks=[FIRST,*REMAINING],libero_indices=list(range(40,50)),
            robotwin_first_batch_tasks=['place_container_plate'],robotwin_seeds=list(range(26000,26010)),
            robotwin_coverage='other49 tasks pending scope clarification and new task-specific data/proxy setup',
            robotwin_training_coverage='49 nominal approach-stage records over24 training seeds; disclosed separately',
            first_result=str(first),first_controller_pid=args.first_controller_pid,phases=phases,
            coordinator_source_sha256=sha(__file__))
        dump(OUT/'plan.json',plan)
        state=dict(started=now(),host=host,pid=os.getpid(),gpus=gpus,completed_phases=[],failed_phases=[])
        def status(stage,**kw):
            state.update(stage=stage,updated=now(),**kw);dump(OUT/'progress.json',state)
            print('[stage]',stage,kw,flush=True)
        child=None
        def stop_child():
            if child is None or child.poll() is not None:return
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
        def interrupted(sig,frame): raise KeyboardInterrupt('signal '+str(sig))
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            status('waiting_for_first_libero_ASR',first_controller_pid=args.first_controller_pid)
            deadline=time.monotonic()+10800
            while not first.exists():
                progress=read(ROOT/'pi05probe/out/main_method/progress.json')
                if progress.get('stage') in ['failed','blocked']:
                    raise RuntimeError('first LIBERO evaluation stopped: '+json.dumps(progress))
                if not first_alive(args.first_controller_pid):
                    raise RuntimeError('first controller is absent before validated ASR output')
                if time.monotonic()>deadline:raise TimeoutError('first LIBERO ASR')
                time.sleep(5)
            assert read(first)['complete'] is True
            # The validated result is written shortly before the original server exits.
            while first_alive(args.first_controller_pid):
                if time.monotonic()>deadline:raise TimeoutError('first LIBERO controller exit')
                time.sleep(2)
            state['completed_phases'].append('libero_first')
            for phase in phases:
                log=OUT/(phase['name']+'.log')
                source=Path(phase['argv'][2]);assert source.is_file(),source
                status('launching_'+phase['name'],phase=phase['name'],command=phase['argv'],log=str(log),
                       phase_source_sha256=sha(source))
                with log.open('a') as stream:
                    child=subprocess.Popen(phase['argv'],cwd=phase['cwd'],env=env,
                        stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                    status('running_'+phase['name'],child_pid=child.pid)
                    code=child.wait()
                if code:
                    detail=read(phase['status']) if Path(phase['status']).exists() else {}
                    status('blocked' if detail.get('stage')=='blocked' else 'failed',error=detail,
                           exit_code=code,failed_phase=phase['name'])
                    state['failed_phases'].append(dict(phase=phase['name'],exit_code=code,detail=detail))
                    child=None
                    continue  # an independent benchmark failure must not stop the other
                child=None;state['completed_phases'].append(phase['name'])
            status('first_batch_incomplete' if state['failed_phases'] else 'first_batch_complete',child_pid=None,
                   note='LIBERO7 + RoboTwin first ready task; other RoboTwin tasks are not claimed evaluated')
        except BaseException as error:
            if state.get('stage') not in ['failed','blocked']:status('failed',error=repr(error))
            raise
        finally:stop_child()


if __name__=='__main__': main()
