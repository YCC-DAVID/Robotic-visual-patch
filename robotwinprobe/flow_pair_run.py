"""One preflight-assigned GPU for model plus sequential replay, fixed22 development cases."""
import argparse,os,json,subprocess,socket,time
from pathlib import Path
from e_method_common import load,dump,sha
from flow_pair_analyze import analyze
BASE=Path(__file__).resolve().parent
PY='/shared/user65/miniconda3/envs/openpi-server/bin/python'

def setup(root):
    source=BASE/'out/lowfreq8_20261028';p=load(source);p['backend_port']=18773
    p['flow_diagnostic']=dict(source=str(source),condition='velocity14_lowfreq8',
        scope='all22 clean-success development seeds; both task-failure and task-success groups; same attacked state/noise; exact replay',
        no_new_training=True)
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():assert load(root)==p
    else:dump(root/'protocol.json',p)
    return source

def main(a):
    root=BASE/'out/flow_pair_lowfreq8_20261029';source=setup(root)
    seeds=[s for s in load(source)['method_study']['development_seeds'] if json.loads((source/'cohorts/development/episodes'/f'native_full_approach_clean_s{s}.json').read_text())['success']]
    logs=root/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OPENPI_DATA_HOME='/shared/user65/.cache/openpi',ROBOTWIN_PYTHON='/shared/user65/miniconda3/envs/RoboTwin/bin/python')
    env.pop('PYTHONPATH',None)
    status=lambda stage,**kw:dump(root/'run_status.json',dict(stage=stage,host=socket.gethostname(),gpu=a.gpu,pid=os.getpid(),**kw))
    backend=None
    try:
        status('loading_backend');(root/'backend_ready.json').unlink(missing_ok=True)
        with (logs/'backend.log').open('a') as f:backend=subprocess.Popen([PY,'-u',str(BASE/'flow_pair_backend.py'),'--root',str(root)],env=env,cwd=BASE.parent,stdout=f,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+600
        while not (root/'backend_ready.json').exists():
            if backend.poll() is not None:raise RuntimeError('backend failed')
            if time.monotonic()>deadline:raise TimeoutError('backend startup')
            time.sleep(2)
        for seed in seeds:
            if (root/'replay'/f'seed{seed}'/'complete.json').exists():continue
            status('replaying',seed=seed,total_seeds=len(seeds),backend_pid=backend.pid)
            with (logs/f'seed{seed}.log').open('a') as f:subprocess.run(['bash',str(BASE/'run_sim.sh'),str(BASE/'flow_pair_replay.py'),'--root',str(root),'--seed',str(seed)],env=env,cwd=BASE.parent,stdout=f,stderr=subprocess.STDOUT,check=True)
            analyze(root)
        analyze(root);status('complete',total_seeds=len(seeds))
    except BaseException as e:status('failed',error=repr(e));raise
    finally:
        if backend is not None and backend.poll() is None:backend.terminate();backend.wait(timeout=30)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);main(p.parse_args())
