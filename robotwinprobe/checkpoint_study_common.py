"""Immutable recipe and shared interfaces for the controlled checkpoint study."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'robotwinprobe/out/checkpoint_study_20261023'
SERVER = ROOT / 'realprobe/third_party/openpi-fintune'
BASE = Path('/home/user1/.cache/openpi/openpi-assets/checkpoints/pi05_base/params')
ASSET = 'robotwin_checkpoint_study'
CAMERAS = ('head_camera', 'left_camera', 'right_camera')
WIRE = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')
MODEL = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')
PY_TRAIN = '/shared/user64/workspace/yuhao/pi/openpi-fintune/.venv/bin/python'
PY_MODEL = '/home/user1/miniconda3/envs/openpi-server/bin/python'
PY_CPU = '/home/user1/miniconda3/bin/python'
PATCH_REFERENCE = ROOT / 'robotwinprobe/out/table9_joint_eot_10cm_20261012'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def training_paths():
    for path in (SERVER/'src', SERVER/'packages/openpi-client/src', SERVER/'scripts', ROOT/'realprobe'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('WANDB_MODE','disabled')
    os.environ.setdefault('OPENPI_DATA_HOME','/home/user1/.cache/openpi')
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')


def model_config():
    from openpi.models.pi0_config import Pi0Config
    return Pi0Config(pi05=True, action_horizon=32, action_dim=32,
        discrete_state_input=True, paligemma_variant='gemma_2b_lora',
        action_expert_variant='gemma_300m', pytorch_compile_mode=None)


def data_factory(root):
    from openpi.training import config as C
    return C.LeRobotAlohaDataConfig(repo_id=ASSET,
        assets=C.AssetsConfig(assets_dir=str(Path(root)/'assets'), asset_id=ASSET),
        adapt_to_pi=True, use_delta_joint_actions=True)


def prepare_protocol(root):
    root = Path(root)
    path = root/'protocol.json'
    if path.exists():
        return json.loads(path.read_text())
    selection = json.loads((PATCH_REFERENCE/'geometry/selection.json').read_text())
    site = next(s for s in selection['centers'] if s['id']=='site05')
    protocol = dict(version=1, task='place_container_plate', task_config='demo_clean',
        embodiment='aloha-agilex', architecture='pi0.5', initialization=str(BASE),
        external_finetuned_checkpoint_used=False,
        model_seeds=[202610230,202610231,202610232], model_steps=20000,
        temporal_group=dict(model_seed=202610230, steps=[5000,10000,20000],
                            separate_from_independent_seeds=True),
        train=dict(expert_successes=100, expert_seed_start=20000,
            validation_successes=20, validation_seed_start=21000,
            batch=16, micro_batch=4, learning_rate=2.5e-5, warmup=500,
            decay_steps=15000, final_learning_rate=2.5e-6,
            optimizer='AdamW b1=.9 b2=.95 eps=1e-8 weight_decay=1e-4 clip=1',
            frozen_vision=True, language_lora_rank=16, full_action_expert=True,
            ema=.99, discrete_state_input=True, normalization='training demonstrations only, common to all seeds',
            actions='Aloha adaptation; delta 12 arm joints, absolute 2 grippers; padding to 32; horizon 32',
            image_augmentation='head crop95%/rotation+-5deg; all views color jitter .3/.4/.5'),
        patch=dict(size_m=.10, texture_resolution=128, site='site05', center_m=site['center_m'],
            selection_reason='preselected geometry/exposure; not selected using this study attack scores',
            seeds=[202610240,202610241,202610242], steps=500, batch=8, lr=.01,
            initialization='constant gray .5', band='full',
            eot='shared recorded bank: 3cm disk, yaw +-15deg, three lighting conditions, no flip',
            proxy_bank=str(PATCH_REFERENCE), shared_world_texture_all_cameras=True,
            loss='maximize first-flow t=1 squared velocity difference, paired clean state/noise',
            matched_draws_across_checkpoints=True),
        evaluation=dict(seeds=list(range(22000,22100)), expert_filter=False,
            groups='clean + fixed Random + every one of 15 textures on all 5 distinct checkpoints',
            expected_closed_loops=8500, replan=32, flow_steps=10,
            mode='actual SAPIEN render, same nominal 10cm patch pose for primary transfer matrix',
            ASR='clean-success & attacked-failure / clean-success, per target checkpoint',
            common_clean_subset='also report intersection of clean successes across all five checkpoints',
            SR='task successes / all attempted evaluation initial states',
            errors='runtime errors retained separately, never silently counted as task failure',
            intervals='paired bootstrap over independent initial-state seed',
            diagnostic_L0='shared states, image geometry and noise; normalized full32 and active14',
            action_difference='full 10-step normalized active14 MSE and physical joint/gripper differences'),
        sources={str(p.relative_to(ROOT)):sha(p) for p in [
            ROOT/'third_party/RoboTwin/envs/place_container_plate.py',
            ROOT/'third_party/RoboTwin/env_cfg/task_config/demo_clean.yml',
            ROOT/'robotwinprobe/render_proxy.py', PATCH_REFERENCE/'protocol.json',
            PATCH_REFERENCE/'geometry/selection.json']})
    dump(path, protocol)
    return protocol


def checkpoint_specs(root):
    p = prepare_protocol(root)
    result=[]
    for i, seed in enumerate(p['model_seeds']):
        steps=p['temporal_group']['steps'] if i==0 else [p['model_steps']]
        for step in steps:
            result.append(dict(id=f'M{i}_s{step:05d}', seed=seed, step=step,
                train_id=f'M{i}', path=str(Path(root)/'models'/f'M{i}_s{step:05d}')))
    return result
