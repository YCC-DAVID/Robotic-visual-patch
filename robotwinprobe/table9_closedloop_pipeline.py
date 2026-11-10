"""Owned GPU server and two simulator workers for the nine-site paired pilot."""
import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

import bootstrap as B
from radius_pipeline import Runner
from table9_closedloop import prepare


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--gpu',default='7')
    ap.add_argument('--port',type=int,default=8397)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--smoke-only',action='store_true')
    args=ap.parse_args();root=Path(args.root).resolve()
    prepare(root)
    lock=(root/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    free=subprocess.check_output(['nvidia-smi','-i',args.gpu,
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if int(free.strip())<32000:
        raise RuntimeError('GPU needs at least 32000 MiB free before starting')
    run=Runner(root,args.gpu,'pipeline')
    run.env.update(OMP_NUM_THREADS='4',MPLCONFIGDIR=str(root/'mplcache'))
    snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['table9_closedloop.py','table9_closedloop_pipeline.py','policy_server.py',
                 'render_geometry.py','eot_conditions.py','scene_smoke.py','run_sim.sh']:
        data=(B.PROBE/name).read_bytes();target=snapshot/name
        if target.exists():assert target.read_bytes()==data, 'source changed; preserve earlier execution'
        else:target.write_bytes(data)
    try:
        run.status('starting_server')
        run.server(args.port)
        run.run('smoke',run.sim('table9_closedloop.py','rollout','--root',root,
            '--port',args.port,'--limit',3))
        run.run('smoke_pairing_report',run.model('table9_closedloop.py','report','--root',root))
        if args.smoke_only:
            run.status('smoke_complete',complete=False)
            return
        run.status('paired_closed_loops',workers=args.workers)
        workers=[run.launch(f'worker{i}',run.sim('table9_closedloop.py','rollout','--root',root,
            '--port',args.port,'--worker',i,'--workers',args.workers)) for i in range(args.workers)]
        last_report=time.monotonic()
        while workers:
            for proc in workers[:]:
                if proc.poll() is not None:
                    if proc.returncode:
                        raise RuntimeError(f'worker {proc.pid} failed; inspect logs')
                    workers.remove(proc)
            if time.monotonic()-last_report>=300:
                run.run('partial_report',run.model('table9_closedloop.py','report','--root',root))
                run.status('paired_closed_loops',workers=[p.pid for p in workers])
                last_report=time.monotonic()
            if workers:time.sleep(5)
        run.run('final_report',run.model('table9_closedloop.py','report','--root',root))
        assert json.loads((root/'summary.json').read_text())['complete']
        run.status('complete',complete=True,summary=str(root/'summary.json'))
    except BaseException as exc:
        run.status('failed',complete=False,error=repr(exc))
        raise
    finally:
        run.close()


if __name__=='__main__':main()
