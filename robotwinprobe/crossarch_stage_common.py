"""Frozen recipe for eight stage specialists and paired exposure tests."""
import copy
import json
import os
from pathlib import Path

import numpy as np

from crossarch_common import DEFAULT as SOURCE, ROOT, STAGES, dump, sha

DEFAULT = SOURCE / 'stage_specialists_20261026'
MODELS = ('pi', 'wam')


def load(root=DEFAULT):
    return json.loads((root / 'protocol.json').read_text())


def prepare(root=DEFAULT):
    root.mkdir(parents=True, exist_ok=True)
    original = json.loads((SOURCE / 'protocol.json').read_text())
    p = copy.deepcopy(original)
    p.update(version=2, experiment='stage_specialists', source_run=str(SOURCE),
             source_protocol_sha256=sha(SOURCE / 'protocol.json'))
    p['patch'].update(objectives=list(MODELS), stages=list(STAGES),
        stage_batch='all eight samples from the one specialist stage; 500x8 per texture',
        loss='effective14 first-flow velocity MSE; single model, no cross-model scaling needed')
    p['evaluation'].update(seeds=list(range(22000, 22010)),
        conditions=['clean', 'random_full'] + ['random_' + s for s in STAGES] +
            [s + '_' + mode for s in STAGES for mode in ('full', 'stage_only')],
        scope='exploratory ten fixed initial seeds, same-model stage specialists; joint and cross-model tests deferred',
        exposures=['full', 'stage_only'],
        replan_rule='native maximum chunk, interrupted at each observed stage boundary in ALL groups, including clean and Random',
        stage_only_rule='insert only when the online monotonic phase equals the specialist stage; remove at its exit; force an immediate policy query at boundaries',
        stage_only_eligibility='clean-success seeds with a clean policy query in the target stage; no exposure opportunities reported separately',
        clean_baseline='new baseline under the shared boundary-replanning rule; historical clean SR is not reused')
    p['proxy_review'] = dict(pixel_gate='all-image RMSE < .06, mask RMSE < .15',
        response_gate='original cosine .8 and relative L0 .5 retained as diagnostics, not training gates',
        original_pi_proxy_sha256=sha(SOURCE / 'audit/pi_proxy.json'),
        justification='User requested a patch per stage and actual closed-loop full/stage-only tests after reviewing the response-direction mismatch; no claim of original gate passing.',
        mandatory='native interface parity; pixel gate; finite nonzero texture gradients; actual-render paired evaluation')
    path = root / 'protocol.json'
    if path.exists():
        assert load(root) == p, 'Protocol changed; preserve this run'
    else:
        dump(path, p)
    # Immutable calibration/replay evidence is hard-linked into an independent run.
    for seed in p['data']['train_seeds'] + p['data']['validation_seeds']:
        target = root / 'trajectories' / f'seed{seed}'
        target.mkdir(parents=True, exist_ok=True)
        for name in ('complete.json', 'trace.npz', 'frames.npz', 'stages.png'):
            source = SOURCE / 'trajectories' / f'seed{seed}' / name
            dest = target / name
            if not dest.exists():
                os.link(source, dest)
    for source in (SOURCE / 'bank/validation').glob('*.npz'):
        dest = root / 'bank/validation' / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            os.link(source, dest)
    for name in ('prompts.npz', 'provenance.json'):
        dest = root / name
        if not dest.exists():
            os.link(SOURCE / name, dest)
    return p


class PhaseTracker:
    """Observed, monotonic events; no expert actions or future labels in rollouts."""
    def __init__(self, initial_object):
        self.z0 = float(initial_object[2])
        self.gripper_index = 13 if initial_object[0] > 0 else 6
        self.index = 0
        self.events = {'approach': 0}

    @property
    def phase(self):
        return STAGES[self.index]

    def update(self, state, obj, plate, step):
        before = self.index
        if self.index == 0 and state[self.gripper_index] < .85:
            self.index = 1
            self.events['grasp'] = int(step)
        if self.index == 1 and obj[2] > self.z0 + .03:
            self.index = 2
            self.events['transport'] = int(step)
        if self.index == 2 and np.linalg.norm(np.asarray(obj)[:2] - np.asarray(plate)[:2]) < .06:
            self.index = 3
            self.events['place'] = int(step)
        return before != self.index


def condition(spec):
    if spec == 'clean':
        return dict(texture=None, stage=None, mode='clean')
    if spec == 'random_full':
        return dict(texture='random', stage=None, mode='full')
    if spec.startswith('random_'):
        stage = spec[len('random_'):]
        assert stage in STAGES
        return dict(texture='random', stage=stage, mode='stage_only')
    for stage in STAGES:
        for mode in ('full', 'stage_only'):
            if spec == stage + '_' + mode:
                return dict(texture=stage, stage=stage, mode=mode)
    raise ValueError(spec)


def exposed(spec, phase):
    return spec['mode'] == 'full' or (spec['mode'] == 'stage_only' and spec['stage'] == phase)
