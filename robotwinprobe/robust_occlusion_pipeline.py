"""Run the true-render EOT occlusion scan with a smoke check before full capture."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PYTHON = '/home/user1/miniconda3/envs/openpi-server/bin/python'
PLOT_PYTHON = '/home/user1/miniconda3/envs/openpi-libero/bin/python'
SIM_PYTHON = '/home/user1/miniconda3/envs/RoboTwin/bin/python'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--gpu', default='7')
    ap.add_argument('--smoke-only', action='store_true')
    args = ap.parse_args()
    dest = Path(args.root).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    lock = (dest/'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', PYTHONNOUSERSITE='1',
               MPLCONFIGDIR=str(dest/'mplcache'))
    free = subprocess.check_output(['nvidia-smi','-i',args.gpu,
        '--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True)
    if int(free.strip()) < 32000:
        raise RuntimeError(f'GPU {args.gpu} has {free.strip()} MiB free; need 32000 before starting')
    script = ROOT/'robotwinprobe/robust_occlusion.py'
    started = time.time()
    def status(stage, **extra):
        data = dict(stage=stage, pid=os.getpid(), gpu=args.gpu,
                    updated_unix=time.time(), elapsed_s=time.time()-started, **extra)
        tmp = dest/'pipeline_status.tmp'
        tmp.write_text(json.dumps(data, indent=2)+'\n')
        tmp.replace(dest/'pipeline_status.json')
        print('[stage]', stage, flush=True)
    def run(stage, smoke=False):
        python = SIM_PYTHON if stage == 'capture' else PLOT_PYTHON if stage == 'report' else MODEL_PYTHON
        cmd = [python,'-u',str(script),stage,'--root',str(dest)]
        if smoke:
            cmd += ['--limit','1']
        status(('smoke_' if smoke else '')+stage, command=cmd)
        with (dest/f"{'smoke_' if smoke else ''}{stage}.log").open('a') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        run('prepare')
        run('capture', smoke=True)
        run('measure', smoke=True)
        if args.smoke_only:
            status('smoke_complete', complete=False)
            return
        run('capture')
        run('measure')
        run('report')
        status('scan_complete', complete=True, closed_loop_complete=False,
               next_stage='cumulative low-influence replacement and true-render closed-loop verification')
    except BaseException as exc:
        status('failed', error=repr(exc), complete=False)
        raise


if __name__ == '__main__':
    main()
