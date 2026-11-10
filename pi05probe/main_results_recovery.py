"""Retry the repaired RoboTwin main method after the active LIBERO worker exits."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'out/main_results'


def read(path):return json.loads(Path(path).read_text())


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def alive(pid,expected):
    try:return expected.encode() in (Path('/proc')/str(pid)/'cmdline').read_bytes()
    except OSError:return False


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--gpus',required=True)
    ap.add_argument('--libero-pid',required=True,type=int)
    ap.add_argument('--coordinator-pid',required=True,type=int)
    args=ap.parse_args();gpus=json.loads(args.gpus);host=socket.gethostname().split('.')[0]
    assert gpus and all(g['host']==host for g in gpus)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'robotwin_recovery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=dict(host=host,pid=os.getpid(),gpus=gpus,libero_pid=args.libero_pid,
            coordinator_pid=args.coordinator_pid,note='dependency queue only; no GPU reserved while waiting')
        def status(stage,**kw):
            state.update(stage=stage,updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
            dump(OUT/'recovery_progress.json',state);print('[stage]',stage,kw,flush=True)
        child=None
        def interrupted(sig,frame):raise KeyboardInterrupt('signal '+str(sig))
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        try:
            status('queued_after_active_libero')
            deadline=time.monotonic()+8*3600
            while alive(args.libero_pid,'main_method_pipeline.py') or alive(args.coordinator_pid,'main_results_coordinator.py'):
                if time.monotonic()>deadline:raise TimeoutError('LIBERO dependency wait')
                time.sleep(5)
            envroot=Path(os.environ.get('WAMATTACK_ENVROOT','/shared/user65/miniconda3/envs'))
            cmd=[str(envroot/'openpi-server/bin/python'),'-u',str(ROOT/'robotwinprobe/main_method_pipeline.py'),
                 '--gpus',json.dumps(gpus),'--tasks','place_container_plate']
            status('starting_repaired_robotwin',command=cmd,
                   note='worker will run all-node and immediate selected-card preflight before each task')
            log=OUT/'robotwin_repaired.log'
            with log.open('a') as f:
                child=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,'OPENPI_DATA_HOME':'/shared/user65/.cache/openpi'},
                    stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                status('running_repaired_robotwin',child_pid=child.pid,log=str(log))
                code=child.wait()
            if code:
                detail=read(ROOT/'robotwinprobe/out/main_method/progress.json')
                status('blocked' if detail.get('stage')=='blocked' else 'failed',exit_code=code,error=detail)
                return
            libero=read(ROOT/'pi05probe/out/main_method/summary.json')
            robotwin=read(ROOT/'robotwinprobe/out/main_method/summary.json')
            summary=dict(libero=libero,robotwin=robotwin,first_batch_complete=bool(libero['complete'] and robotwin['complete']),
                coverage='LIBERO seven current tasks and RoboTwin place_container_plate; no claim for other49 tasks')
            dump(OUT/'summary.json',summary)
            status('first_batch_complete' if summary['first_batch_complete'] else 'first_batch_incomplete',summary=str(OUT/'summary.json'))
        except BaseException as error:
            status('failed',error=repr(error));raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


if __name__=='__main__':main()
