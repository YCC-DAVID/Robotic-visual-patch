"""Approved pi0.5 whole-trajectory experiment: immutable data and matched draws."""
import copy
import json
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from checkpoint_study_common import ROOT, CAMERAS, WIRE, MODEL, dump, sha

DEFAULT = ROOT/'robotwinprobe/out/stage_joint_20261026'
SOURCE = ROOT/'robotwinprobe/out/crossarch_joint_20261025'
STAGES = ('approach','grasp','transport','place')
TEXTURES = ('early','joint_mean','joint_balanced')
CONDITIONS = ('clean','random',*TEXTURES)
PORT = 18661
AUTH = b'robotwin-pi-stage-joint-20261026'
PY = '/home/user1/miniconda3/envs/openpi-server/bin/python'


def load(root=DEFAULT):
    return json.loads((Path(root)/'protocol.json').read_text())


def prepare(root=DEFAULT):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():
        return load(root)
    original=json.loads((SOURCE/'protocol.json').read_text())
    p=copy.deepcopy(original)
    selection=json.loads((ROOT/'robotwinprobe/out/table9_joint_eot_10cm_20261012/geometry/selection.json').read_text())
    site=next(s for s in selection['centers'] if s['id']=='site07')
    p.update(version=3,experiment='pi_stage_joint',source_run=str(SOURCE),
        source_protocol_sha256=sha(SOURCE/'protocol.json'),policy_training=False,
        approval='User approved 2026-10-26: three pi0.5 patches; proxy response fidelity diagnostic only; actual render endpoints.')
    p['policies']={'pi':original['policies']['pi']}
    p['patch'].update(center_m=site['center_m'],site='site07',objectives=list(TEXTURES),
        seed=202610260,steps=500,batch=8,
        stage_batch='early: eight approach samples; joint_mean/joint_balanced: same two draws per stage',
        loss='effective14 first-flow MSE; mean L0/c or mean log1p(L0/c), c fixed training-only Random mean',
        normalization='single global c from 32 training-only state/noise draws, not per-stage',
        proxy_fidelity_gate=False,gradient_gate='finite nonzero texture gradient and native interface parity')
    p['evaluation'].update(seeds=list(range(22000,22100)),pulse_seeds=list(range(22000,22020)),
        pilot_seeds=[23100,23101],conditions=list(CONDITIONS),
        suites=['native_full','delayed','pulse'],replan={'pi':32},horizon={'pi':32},
        noise_base=202610260,
        stage_events='grasp: active gripper <.85; transport: stable lift >.03m for 3 actions; place: XY distance <.06m after lift',
        delayed='three later stages; exact native clean prefix; force one query at clean event; then native32 cadence; patch persists',
        pulse='four stages on fixed first20 seeds; force query at clean event; one affected policy query, remove before next query',
        stage_clean='each delayed/pulse stage has its own identically replanned clean; joint eligibility: native clean success AND stage clean success AND opportunity',
        opportunity='must have observed event before native task termination; absent/terminal event is not an attack failure',
        primary='actual SAPIEN RT; fixed nominal pose/light; paired state and action noise',
        expected_records_upper_bound=2400,
        diagnostics='eight held-out successful expert trajectories; stride8 plus phase boundaries/final; three noises; true render L0 and complete10-flow action differences',
        endpoint='official first task success or step limit; release is part of place, no post-success ASR')
    p['proxy_review']={'response_fidelity':'diagnostic only; user explicitly removed training gate',
        'mandatory':'correct native inference; finite nonzero texture gradients; exact replay; actual-render paired outcomes'}
    p['data']['validation_stride']=8
    dump(root/'protocol.json',p)
    for seed in p['data']['train_seeds']+p['data']['validation_seeds']:
        target=root/'trajectories'/f'seed{seed}';target.mkdir(parents=True,exist_ok=True)
        for name in ('complete.json','trace.npz','frames.npz','stages.png'):
            source=SOURCE/'trajectories'/f'seed{seed}'/name
            assert source.exists(),source
            (target/name).write_bytes(source.read_bytes())
    dump(root/'data_manifest.json',dict(source=str(SOURCE),source_trajectories={
        str((SOURCE/'trajectories'/f'seed{s}'/name)):sha(SOURCE/'trajectories'/f'seed{s}'/name)
        for s in p['data']['train_seeds']+p['data']['validation_seeds'] for name in ('trace.npz','complete.json','frames.npz')},
        independent_split=True,successful_expert_trajectories=True))
    # Fixed Random is identical to prior RoboTwin experiments.
    from PIL import Image
    from render_geometry import patterns
    dest=root/'patches/random';dest.mkdir(parents=True,exist_ok=True)
    Image.fromarray(patterns()['random']).save(dest/'texture.png')
    return p


def request(payload,root=DEFAULT):
    port=load(root).get('backend_port',PORT)
    with Client(('127.0.0.1',port),authkey=AUTH) as c:
        c.send(payload);result=c.recv()
    if 'error' in result:
        raise RuntimeError(result['error']+'\n'+result.get('traceback',''))
    return result


def training_plan(root):
    from crossarch_capture import records
    p=load(root)
    rs=[r for s in p['data']['train_seeds'] for r in records(root,s)]
    by_stage={s:[r['path'] for r in rs if r['stage']==s] for s in STAGES}
    assert all(by_stage.values())
    rng=np.random.default_rng(p['patch']['seed'])
    joint=[[dict(record=str(rng.choice(by_stage[s])),seed=int(rng.integers(1,2**30)),stage=s)
            for s in STAGES for _ in range(2)] for _ in range(500)]
    rng=np.random.default_rng(p['patch']['seed']+1)
    early=[[dict(record=str(rng.choice(by_stage['approach'])),seed=int(rng.integers(1,2**30)),stage='approach')
            for _ in range(8)] for _ in range(500)]
    return {'early':early,'joint':joint}


class PhaseTracker:
    def __init__(self,initial_object):
        self.z0=float(initial_object[2]);self.active=13 if initial_object[0]>0 else 6
        self.index=0;self.high_count=0;self.events={'approach':0}

    @property
    def phase(self):
        return STAGES[self.index]

    def update(self,state,obj,plate,step):
        if self.index==0 and state[self.active]<.85:
            self.index=1;self.events['grasp']=int(step)
        self.high_count=self.high_count+1 if obj[2]>self.z0+.03 else 0
        if self.index==1 and self.high_count>=3:
            self.index=2;self.events['transport']=int(step)
        if self.index==2 and np.linalg.norm(np.asarray(obj)[:2]-np.asarray(plate)[:2])<.06:
            self.index=3;self.events['place']=int(step)


def jobs(root):
    p=load(root);result=[]
    for seed in p['evaluation']['seeds']:
        for c in CONDITIONS:
            result.append(dict(seed=seed,suite='native_full',stage='approach',condition=c))
        for stage in STAGES[1:]:
            for c in CONDITIONS:
                result.append(dict(seed=seed,suite='delayed',stage=stage,condition=c))
        if seed in p['evaluation']['pulse_seeds']:
            for stage in STAGES:
                for c in CONDITIONS:
                    result.append(dict(seed=seed,suite='pulse',stage=stage,condition=c))
    assert len(result)==2400
    return result


def job_id(job):
    return f"{job['suite']}_{job['stage']}_{job['condition']}_s{job['seed']}"
