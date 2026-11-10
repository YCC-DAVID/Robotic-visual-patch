"""Public RoboTwin policies, immutable full-trajectory patch protocol."""
import json
from pathlib import Path
from checkpoint_study_common import dump, sha, ROOT, CAMERAS, WIRE, MODEL

DEFAULT = ROOT/'robotwinprobe/out/crossarch_joint_20261025'
EXPERT = ROOT/'robotwinprobe/out/checkpoint_study_20261023/expert'
WAM_SOURCE = Path('/shared/user64/workspace/yuhao/imagewam/ImageWAM')
WAM_WEIGHTS = WAM_SOURCE.parent/'checkpoints/imagewam_release/robotwin/flux2_klein_4b'
AE = WAM_SOURCE.parent/'checkpoints/flux2/FLUX.2-dev/ae.safetensors'
QWEN = Path('/shared/user64/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c')
PY = {'pi': '/home/user1/miniconda3/envs/openpi-server/bin/python',
      'wam': str(WAM_SOURCE/'.venv/bin/python'), 'cpu':'/home/user1/miniconda3/bin/python'}
STAGES = ('approach', 'grasp', 'transport', 'place')
PORTS = {'pi': 18551, 'wam': 18552}
AUTH = b'robotwin-crossarch-local-20261025'

def prepare(root=DEFAULT):
    path=root/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    seeds=[]
    for p in sorted(EXPERT.glob('seed*/status.json')):
        row=json.loads(p.read_text())
        if row.get('success') and (p.parent/'demonstration.npz').exists(): seeds.append(row['seed'])
    train=[s for s in seeds if 20000<=s<21000][:24]
    val=[s for s in seeds if 21000<=s<22000][:8]
    assert len(train)==24 and len(val)==8
    selection=json.loads((ROOT/'robotwinprobe/out/table9_joint_eot_10cm_20261012/geometry/selection.json').read_text())
    center=next(s['center_m'] for s in selection['centers'] if s['id']=='site05')
    p=dict(version=1,task='place_container_plate',task_config='demo_clean',policy_training=False,
        policies={'pi':str(ROOT/'checkpoints/pi05_robotwin2_pytorch'),'wam':str(WAM_WEIGHTS)},
        data=dict(train_seeds=train,validation_seeds=val,source=str(EXPERT),
            frame_selection='4 evenly spaced actual replay states per event stage; validation 2 per stage',
            stage_rules='active gripper <0.85; object lift >3cm; container within 6cm xy of plate; retain final release',
            no_attack_outcome_filter=True),
        patch=dict(size_m=.10,resolution=128,center_m=center,steps=500,batch=8,lr=.01,
            optimizer='projected sign-gradient ascent; clip [0,1]',initialization='gray .5',
            objectives=['pi','wam','joint'],stage_batch='two per stage per update',seed=20261025,
            eot=dict(position_disk_m=.03,yaw_degrees=15,lights=['nominal','dim_warm','bright_cool'],
                     flip=False,recorded_draws_per_training_frame=4),
            loss='effective14 first-flow mean squared velocity difference; joint equal mean of L0 / fixed per-model Random mean',
            proxy='shared native SAPIEN RGB: perspective, linear RGB footprint averaging, five-gray per-pixel response, global response, occlusion; model-native resize afterwards'),
        evaluation=dict(seeds=list(range(22000,22100)),pilot_seeds=list(range(23000,23010)),
            conditions=['clean','random','pi','wam','joint'],flow_steps=10,
            horizon={'pi':32,'wam':16},replan={'pi':32,'wam':16},
            primary='actual render, fixed nominal pose/light; all trajectories closed loop',
            ASR='clean-success and attacked-failure / target-model clean successes',
            SR='task successes / 100 fixed initial seeds',common_clean_subset=True,
            runtime_error='not task failure; stop and preserve traceback',
            diagnostics='held-out expert states, all four stages, three paired noise seeds; effective14 mean and sum L0 plus full dimensions; full action differences'),
        source_hashes={str(EXPERT/f'seed{s}/demonstration.npz'):sha(EXPERT/f'seed{s}/demonstration.npz') for s in train+val})
    dump(path,p)
    return p

def request(model,payload):
    from multiprocessing.connection import Client
    with Client(('127.0.0.1',PORTS[model]),authkey=AUTH) as c:
        c.send(payload); result=c.recv()
    if 'error' in result: raise RuntimeError(result['error']+'\n'+result.get('traceback',''))
    return result
