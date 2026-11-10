"""Run isolated speed probes sequentially; do not alter training settings."""
import argparse,json,os,subprocess,time
from pathlib import Path
import checkpoint_study_common as S

def main(args):
    out=args.root/'speed_probe_20261027';out.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpus,JAX_PLATFORMS='cuda,cpu',
        PYTHONHASHSEED='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_SHM_DISABLE='1')
    results=[]
    for name,micro,shm in [('baseline_m2',2,'1'),('micro4',4,'1'),('shm_m2',2,'0')]:
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free={a.strip():int(b) for a,b in (line.split(',') for line in raw.splitlines())}
            if all(free[g]>=34000 for g in args.gpus.split(',')):break
            S.dump(out/'suite_status.json',dict(stage='waiting_for_memory',complete=False,pid=os.getpid(),case=name,free_MiB=free))
            time.sleep(30)
        dest=out/name;dest.mkdir(exist_ok=True)
        with (dest/'run.log').open('a') as log:
            child=subprocess.Popen([S.PY_TRAIN,'robotwinprobe/vision_freeze_speed_probe.py','--root',str(args.root),
                '--out',str(dest),'--micro-batch',str(micro)],cwd=S.ROOT,env=env|{'NCCL_SHM_DISABLE':shm},stdout=log,stderr=subprocess.STDOUT)
            S.dump(out/'suite_status.json',dict(stage='running',complete=False,pid=os.getpid(),child_pid=child.pid,case=name,gpus=args.gpus))
            try:code=child.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                child.terminate();child.wait(timeout=60);code=124
        results.append(dict(case=name,returncode=code,result=json.loads((dest/'result.json').read_text()) if (dest/'result.json').exists() else None))
    S.dump(out/'suite_status.json',dict(stage='complete',complete=True,pid=os.getpid(),results=results))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--gpus',default='0,1');main(p.parse_args())
