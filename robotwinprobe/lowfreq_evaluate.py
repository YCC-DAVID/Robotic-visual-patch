"""Fixed development-only evaluation, with one preflight-selected GPU for this job."""
import argparse,json,os,subprocess,time,socket,sys
from pathlib import Path
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from e_method_common import load,dump,results
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);a=p.parse_args()
root=BASE/'out/lowfreq8_20261028';env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OPENPI_DATA_HOME='/shared/user65/.cache/openpi')
py='/shared/user65/miniconda3/envs/openpi-server/bin/python'
logs=root/'evaluation_logs';logs.mkdir(exist_ok=True)
backend=None
try:
 dump(root/'evaluation_status.json',dict(stage='loading_backend',host=socket.gethostname(),gpu=a.gpu,pid=os.getpid()))
 (root/'backend_ready.json').unlink(missing_ok=True)
 with (logs/'backend.log').open('a') as f:backend=subprocess.Popen([py,'-u',str(BASE/'e_method_backend.py'),'--root',str(root)],env=env,stdout=f,stderr=subprocess.STDOUT)
 deadline=time.monotonic()+600
 while not (root/'backend_ready.json').exists():
  if backend.poll() is not None:raise RuntimeError('backend failed')
  if time.monotonic()>deadline:raise TimeoutError('backend ready')
  time.sleep(2)
 for seed in load(root)['method_study']['development_seeds']:
  dump(root/'evaluation_status.json',dict(stage='evaluating',host=socket.gethostname(),gpu=a.gpu,pid=os.getpid(),seed=seed))
  argv=['bash',str(BASE/'run_sim.sh'),str(BASE/'e_method_sim.py'),'--root',str(root),'--mode','evaluate','--split','development','--seed',str(seed),'--conditions','clean,random,velocity14_lowfreq8']
  with (logs/f'seed{seed}.log').open('a') as f:subprocess.run(argv,env=dict(env,ROBOTWIN_PYTHON='/shared/user65/miniconda3/envs/RoboTwin/bin/python'),stdout=f,stderr=subprocess.STDOUT,check=True,cwd=BASE.parent)
 dump(root/'development_results.json',results(root,'development',('clean','random','velocity14_lowfreq8')))
 dump(root/'evaluation_status.json',dict(stage='development_complete',host=socket.gethostname(),gpu=a.gpu))
except BaseException as e:
 dump(root/'evaluation_status.json',dict(stage='failed',error=repr(e)));raise
finally:
 if backend is not None and backend.poll() is None:backend.terminate();backend.wait(timeout=30)
