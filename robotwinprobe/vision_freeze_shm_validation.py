"""Queue isolated SHM roundtrips on spare GPUs; never switch live trainers."""
import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import checkpoint_study_common as S


def main(root,candidates=('2','3','4','7')):
    out=root/'shm_validation_20261027';out.mkdir(exist_ok=True)
    for name in ['data_smoke','smoke_assets']:
        target=out/name
        if not target.exists():target.symlink_to(root/name,target_is_directory=True)
    if not (out/'protocol.json').exists():
        (out/'protocol.json').write_bytes((root/'protocol.json').read_bytes())
    def state(stage,**extra):
        S.dump(out/'validation_status.json',dict(stage=stage,complete=False,pid=os.getpid(),updated_unix=time.time(),**extra))
    while True:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        free={a.strip():int(b) for a,b in (line.split(',') for line in raw.splitlines())}
        # Defaults exclude local production; remote jobs provide explicit GPUs.
        available=[g for g in candidates if free.get(g,0)>=34000]
        if len(available)>=2:break
        state('waiting_for_spare_gpus',required_each_MiB=34000,free_MiB={g:free.get(g) for g in candidates})
        time.sleep(30)
    pair=next(iter(itertools.combinations(available,2)));gpus=','.join(pair)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,JAX_PLATFORMS='cuda,cpu',PYTHONHASHSEED='0',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',NCCL_SHM_DISABLE='0')
    for variant in ['U','F']:
        argv=[S.PY_TRAIN,'robotwinprobe/vision_freeze_ft.py','--root',str(out),'--name',variant,
            '--seed','202610230','--steps','4','--smoke']
        if variant=='U':argv+=['--unfreeze-vision']
        for mode,cmd in [('train',argv),('restore',argv+['--resume','--verify-only'])]:
            with (out/f'{variant}_{mode}.log').open('a') as log:
                child=subprocess.Popen(cmd,cwd=S.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                state(f'{variant}_{mode}',gpus=list(pair),child_pid=child.pid)
                code=child.wait()
            assert code==0,f'{variant}_{mode} exited {code}'
    configs=[];initial=[];restores={}
    for v in ['F','U']:
        d=out/'training'/v
        config=json.loads((d/'config.json').read_text())['recipe'];config.pop('freeze_vision');configs.append(config)
        initial.append(json.loads((d/'initial_audit.json').read_text()))
        restores[v]=json.loads((d/'restore_audit.json').read_text())
        assert restores[v]['passed']
    assert configs[0]==configs[1]
    assert initial[0]['vision_hashes']==initial[1]['vision_hashes']
    assert all(abs(initial[0]['monitor'][k]-initial[1]['monitor'][k])<1e-6 for k in initial[0]['monitor'])
    S.dump(out/'validation_status.json',dict(stage='passed',complete=True,pid=os.getpid(),gpus=list(pair),
        paired_initial_passed=True,restores=restores,production_switched=False,
        next_action='Review before applying SHM at a saved production checkpoint; no live process modified'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate-gpus',default='2,3,4,7');args=p.parse_args()
    try:main(args.root,tuple(args.candidate_gpus.split(',')))
    except Exception as exc:
        S.dump(args.root/'shm_validation_20261027/validation_status.json',dict(stage='failed',complete=False,
            pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc()))
        raise
