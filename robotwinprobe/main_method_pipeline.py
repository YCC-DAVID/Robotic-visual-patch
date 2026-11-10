"""Fixed first-ready RoboTwin main method: train once, then paired true ASR."""
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import bootstrap as B
from main_method_common import DEFAULT,TASK,prepare,training_spec,records,sha,dump

ENVROOT=Path(os.environ.get('WAMATTACK_ENVROOT','/shared/user65/miniconda3/envs'))
MODEL=str(ENVROOT/'openpi-server/bin/python')
SIM=str(ENVROOT/'RoboTwin/bin/python')
sys.path.insert(0,str(B.ROOT/'pi05probe'))
import visible_pos_pipeline as V
from main_method_job_runtime import install as install_job_runtime, robotwin_server


def model_env(gpu):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONNOUSERSITE='1',HF_HUB_OFFLINE='1',
        OPENPI_DATA_HOME='/shared/user65/.cache/openpi',JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')


def valid_training(out,spec):
    paths=[out/n for n in ['config.json','results.json','texture_0500.png']]
    if not all(p.is_file() for p in paths):return False
    config=json.loads(paths[0].read_text());result=json.loads(paths[1].read_text())
    return (config==spec and result.get('complete') is True and result.get('iterations')==500
            and result.get('texture_sha256')==sha(paths[2]) and result.get('config_sha256')==sha(paths[0]))


@contextmanager
def server(root,gpu,port):
    with robotwin_server(root,gpu,port,MODEL,model_env,V):
        yield


def manifest(root,train):
    from PIL import Image
    from render_geometry import patterns
    texture=root/'random.png'
    if not texture.exists():Image.fromarray(patterns()['random']).save(texture)
    value=dict(task=TASK,protocol_sha256=sha(root/'protocol.json'),training_result_sha256=sha(train/'results.json'),
        textures=dict(clean=None,random=dict(path=str(texture),sha256=sha(texture)),
                      method=dict(path=str(train/'texture_0500.png'),sha256=sha(train/'texture_0500.png'))))
    path=root/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==value,'evaluation texture/provenance changed'
    else:dump(path,value)


def main(args):
    root=Path(args.out).resolve();tasks=args.tasks.split(',')
    V.DEST=root
    install_job_runtime(V)
    prepare(root,tasks)
    for folder in ['logs','services','resources','train']: (root/folder).mkdir(exist_ok=True)
    if args.prepare_only:
        print(json.dumps(dict(prepared=True,root=str(root),tasks=tasks,training_states=49,evaluation_episodes=30)))
        return
    gpus=json.loads(args.gpus)
    if not gpus:raise ValueError('no allocated GPU')
    # This first ready batch is one task. Further tasks need their own calibration.
    gpu=gpus[0]
    V.stage('prepare',tasks=tasks)
    check=root/'scene_check'/'passed.json'
    if not check.exists() or json.loads(check.read_text())['binding']['authorized_uuid']!=gpu['uuid']:
        V.stage('scene_check',task=TASK)
        with V.allocation('robotwin_main_scene_check_'+TASK,gpu,32000):
            env=dict(model_env(gpu),ROBOTWIN_PYTHON=SIM)
            V.run(['bash',str(B.PROBE/'run_sim.sh'),str(B.PROBE/'main_method_scene_check.py'),
                '--out',str(root/'scene_check'),'--gpu-uuid',gpu['uuid']],
                'scene_check.log',env,1800)
        assert json.loads(check.read_text())['passed']
    train=root/'train'/TASK;spec=training_spec(records())
    if not valid_training(train,spec):
        if train.exists():
            archived=root/'recovery'/('train_interrupted_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
            archived.parent.mkdir(exist_ok=True);train.rename(archived)
        V.stage('train',task=TASK)
        with V.allocation('robotwin_main_train_'+TASK,gpu,32000):
            V.run([MODEL,'-u',str(B.PROBE/'main_method_train.py'),'--out',str(train)],'train.log',model_env(gpu),14400)
        if not valid_training(train,spec):raise RuntimeError('trained artifact/provenance incomplete')
    manifest(root,train)
    complete=root/'eval'/TASK/'complete.json'
    if complete.exists():
        value=json.loads(complete.read_text())
        assert value.get('complete') and value['protocol_sha256']==sha(root/'protocol.json') and value['manifest_sha256']==sha(root/'manifest.json')
        assert json.loads((root/'summary.json').read_text())['complete']
    else:
        V.stage('evaluate',task=TASK,episodes=30)
        with V.allocation('robotwin_main_evaluate_'+TASK,gpu,32000),server(root,gpu,args.port):
            env=dict(model_env(gpu),ROBOTWIN_PYTHON=SIM)
            V.run(['bash',str(B.PROBE/'run_sim.sh'),str(B.PROBE/'main_method_rollout.py'),
                '--root',str(root),'--port',str(args.port),'--gpu-uuid',gpu['uuid']],
                'rollout.log',env,14400)
        assert json.loads((root/'summary.json').read_text())['complete']
    V.stage('complete',task=TASK,summary=str(root/'summary.json'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--gpus',default='[]')
    ap.add_argument('--tasks',default=TASK);ap.add_argument('--out',default=str(DEFAULT))
    ap.add_argument('--port',type=int,default=8452);ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    Path(args.out).mkdir(parents=True,exist_ok=True)
    with (Path(args.out)/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main(args)
        except BaseException as error:
            V.stage('blocked' if isinstance(error,V.ResourceBlocked) else 'failed',error=repr(error))
            raise
