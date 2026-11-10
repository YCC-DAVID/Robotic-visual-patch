"""Owned policy service and simulator processes for delayed patch injection."""
import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

import bootstrap as B
from radius_pipeline import Runner
from stage_injection import load_protocol, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True)
    ap.add_argument('--gpu',default='7')
    ap.add_argument('--port',type=int,default=8416)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--smoke-only',action='store_true')
    args = ap.parse_args()
    assert args.workers > 0
    root = Path(args.root).resolve()
    load_protocol(root)
    lock = (root/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if report(root)['complete']:
        return
    free = subprocess.check_output(['nvidia-smi','-i',args.gpu,
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(free.strip()) < 32000:
        raise RuntimeError('GPU needs at least 32000 MiB free before starting')
    run = Runner(root,args.gpu,'pipeline')
    run.env.update(OMP_NUM_THREADS='4',MPLCONFIGDIR=str(root/'mplcache'))
    try:
        run.status('starting_server')
        run.server(args.port)
        run.status('paired_stage_injection',workers=args.workers)
        workers = []
        for i in range(1 if args.smoke_only else args.workers):
            cmd = run.sim('stage_injection.py','rollout','--root',root,'--port',args.port,
                          '--worker',i,'--workers',args.workers)
            if args.smoke_only:
                cmd += ['--limit-seeds','1']
            workers.append(run.launch(f'worker{i}',cmd))
        last_report = time.monotonic()
        while workers:
            for proc in workers[:]:
                if proc.poll() is not None:
                    if proc.returncode:
                        raise RuntimeError(f'simulator {proc.pid} failed; inspect logs')
                    workers.remove(proc)
            if time.monotonic()-last_report >= 60:
                report(root)
                run.status('paired_stage_injection',active_workers=[p.pid for p in workers])
                last_report = time.monotonic()
            if workers:
                time.sleep(5)
        summary = report(root)
        assert args.smoke_only or summary['complete']
        run.status('smoke_complete' if args.smoke_only else 'complete',complete=summary['complete'])
    except BaseException as exc:
        run.status('failed',complete=False,error=repr(exc))
        raise
    finally:
        run.close()


if __name__ == '__main__':
    main()
