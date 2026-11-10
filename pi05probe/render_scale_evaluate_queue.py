"""Three bounded evaluation workers; wait for frozen training outputs.

Model/renderer subprocesses run only after training completes. Worker ports
and GPUs are explicit; all data writes use separate per-run directories.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'out/render_proxy_scale'
MODEL='/home/user1/miniconda3/envs/openpi-server/bin/python'
RENDER='/home/user1/miniconda3/envs/openpi-libero/bin/python'
RUNS=['26_101_band32','26_101_band32_eot','13_101_full','9_101_full','6_101_full','9_161_full','6_161_full']


def complete(path):
    try:return json.loads(path.read_text()).get('complete') is True
    except (FileNotFoundError,json.JSONDecodeError):return False


def command(argv,log,env):
    with (BASE/'logs'/log).open('w') as output:
        subprocess.run(argv,cwd=ROOT,env={**os.environ,**env},stdout=output,stderr=subprocess.STDOUT,check=True)


def evaluate(name,port,gpu):
    run=BASE/('train_'+name);out=BASE/('eval_'+name)
    assert complete(run/'config.json') and complete(run/'results.json')
    command([MODEL,'render_scale_manifest.py','--run',str(run),'--out',str(out)],'manifest_'+name+'.log',{'OMP_NUM_THREADS':'4'})
    manifest=out/'manifest.json'
    command([RENDER,'render_scale_rollout.py','--manifest',str(manifest),'--conditions','random,optimized',
             '--clean-reference',str(BASE/'clean_baseline/episodes.json'),'--indices','40,41,42,43,44',
             '--port',str(port),'--out','render_proxy_scale/pilot_'+name], 'pilot_'+name+'.log',
            {'OMP_NUM_THREADS':'2','MUJOCO_EGL_DEVICE_ID':str(gpu)})
    print('[closed-loop done]',name,flush=True)
    command([RENDER,'render_scale_validation_capture.py','--manifest',str(manifest),
             '--out','render_proxy_scale/validation_'+name], 'validation_capture_'+name+'.log',
            {'OMP_NUM_THREADS':'2','MUJOCO_EGL_DEVICE_ID':str(gpu)})
    command([MODEL,'render_scale_validation_probe.py','--out','render_proxy_scale/validation_'+name],
            'validation_probe_'+name+'.log',{'OMP_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':str(gpu)})
    print('[all evaluations done]',name,flush=True)
    return name


def main():
    pending=set(RUNS);running={};free=[(8144,4),(8145,5),(8146,6)];finished=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        while pending or running:
            for future,(name,resource) in list(running.items()):
                if future.done():
                    future.result();finished.append(name);free.append(resource);del running[future]
            baseline=False
            try:
                baseline=len(json.loads((BASE/'clean_baseline/episodes.json').read_text())['per_episode']['clean'])==10
            except (FileNotFoundError,json.JSONDecodeError):pass
            for name in RUNS:
                if name in pending and free and baseline and complete(BASE/('train_'+name)/'config.json'):
                    resource=free.pop(0);pending.remove(name)
                    future=executor.submit(evaluate,name,*resource);running[future]=(name,resource)
                    print('[evaluate]',name,resource,flush=True)
            (BASE/'queue_progress.json').write_text(json.dumps({'pending':sorted(pending),
                'running':[r[0] for r in running.values()],'finished':finished},indent=2)+'\n')
            if pending or running:time.sleep(3)
    print('[done]',finished,flush=True)


if __name__=='__main__':main()
