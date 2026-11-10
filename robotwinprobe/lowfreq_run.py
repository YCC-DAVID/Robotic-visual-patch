"""Wait for free GPU memory, then run an isolated fixed-budget low-frequency study."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out/lowfreq8_20261028'
PY='/home/user1/miniconda3/envs/openpi-server/bin/python'
def status(stage,**kw):
    (OUT/'run_status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))
backend=None
try:
    status('waiting_for_gpu',required_free_MiB=26000)
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
        available=[int(r.split(',')[0]) for r in rows.splitlines() if int(r.split(',')[1])>=26000]
        if available:break
        time.sleep(30)
    gpu=available[0];env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    status('loading_backend',gpu=gpu)
    (OUT/'backend_ready.json').unlink(missing_ok=True)
    with (OUT/'backend.log').open('a') as log:
        backend=subprocess.Popen([PY,'-u',str(ROOT/'e_method_backend.py'),'--root',str(OUT)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        ready=OUT/'backend_ready.json'
        while not ready.exists():
            if backend.poll() is not None:raise RuntimeError('backend failed; see backend.log')
            time.sleep(5)
        status('training',gpu=gpu,cutoff=8)
        with (OUT/'train.log').open('a') as trainlog:
            subprocess.run([PY,'-u',str(ROOT/'e_method_train.py'),'--root',str(OUT),'--name','velocity14_lowfreq8','--kind','velocity14','--training-seed','202610270','--cutoff','8'],cwd=ROOT,env=env,stdout=trainlog,stderr=subprocess.STDOUT,check=True)
        status('training_complete_needs_real_render_evaluation',gpu=gpu)
except Exception as exc:
    status('failed',error=repr(exc));raise
finally:
    if backend is not None and backend.poll() is None:
        backend.terminate();backend.wait(timeout=30)
