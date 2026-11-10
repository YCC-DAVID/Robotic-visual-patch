"""Physical full-episode versus target-stage-only patch exposure, shared replanning."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np

import bootstrap as B
from crossarch_common import CAMERAS, WIRE, request, dump, sha
from crossarch_stage_common import DEFAULT, PhaseTracker, condition, exposed, load


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(a):
    B.simulator_paths()
    from scene_smoke import make_task
    from render_geometry import Patch, patterns, camera_objects
    from PIL import Image
    p = load(a.root)
    spec = condition(a.condition)
    group = 'smoke' if a.smoke else 'episodes'
    dest = a.root / group / a.model / a.condition
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / f'seed{a.seed}.json'
    protocol_hash = sha(a.root / 'protocol.json')
    if output.exists():
        prior = json.loads(output.read_text())
        assert 'success' in prior and prior['protocol_sha256'] == protocol_hash, str(output)
        return
    reference = None
    clean_path = a.root / group / a.model / 'clean' / f'seed{a.seed}.json'
    if spec['mode'] != 'clean':
        clean = json.loads(clean_path.read_text())
        assert clean['protocol_sha256'] == protocol_hash
        with np.load(clean_path.with_suffix('.npz')) as z:
            reference = {k: z[k].copy() for k in ('states14', 'actions', 'query_steps', 'rgb_sha256', 'object_plate')}
    task, patch = None, None
    began = time.monotonic()
    random.seed(a.seed)
    try:
        task, _, _ = make_task(p['task'], a.seed, dest)
        prompt = 'Place the ' + ('bowl' if task.actor_name == '002_bowl' else 'cup') + ' on the plate.'
        task.set_instruction(prompt)
        initial = task.get_obs()
        state0 = initial['joint_action']['vector'].astype(np.float32)
        initial_hash = digest(np.concatenate([state0.view(np.uint8).ravel()] +
            [initial['observation'][n]['rgb'].ravel() for n in CAMERAS]))
        if reference is not None:
            assert clean['initial_clean_sha256'] == initial_hash
        tracker = PhaseTracker(task.container.get_pose().p)
        texture = None
        if spec['texture']:
            texture = patterns()['random'] if spec['texture'] == 'random' else np.asarray(
                Image.open(a.root / 'patches' / a.model / spec['texture'] / 'texture_0500.png').convert('RGB'))
        states, chunks, query_steps, hashes, poses, query_phases, active, visible = [], [], [], [], [], [], [], []
        ever_exposed = False
        prefix_verified = 0
        while task.take_action_cnt < task.step_lim and not task.eval_success:
            # Phase is observed after every executed action; boundaries interrupt all conditions.
            phase = tracker.phase
            q = len(states)
            want_patch = exposed(spec, phase)
            obs = task.get_obs()
            state = obs['joint_action']['vector'].astype(np.float32)
            pose = np.asarray([task.container.get_pose().p, task.plate.get_pose().p])
            if spec['mode'] == 'stage_only' and not ever_exposed:
                assert q < len(reference['states14'])
                assert np.array_equal(state, reference['states14'][q]), ('prefix state', q)
                assert np.array_equal(pose, reference['object_plate'][q]), ('prefix pose', q)
                assert task.take_action_cnt == reference['query_steps'][q]
                assert [digest(obs['observation'][n]['rgb']) for n in CAMERAS] == reference['rgb_sha256'][q].tolist()
                prefix_verified += 1
            if want_patch and patch is None:
                patch = Patch(task, dest, center=p['patch']['center_m'], size=p['patch']['size_m'])
                patch.texture(texture)
                after = task.get_obs()
                assert np.array_equal(state, after['joint_action']['vector'].astype(np.float32))
                obs = after
            elif not want_patch and patch is not None:
                task.scene.remove_actor(patch.actor)
                patch = None
                obs = task.get_obs()
            raw = {w: obs['observation'][n]['rgb'] for n, w in zip(CAMERAS, WIRE)}
            noise_seed = 20261006 + a.seed * 1000 + q
            chunk = np.asarray(request(a.model, dict(command='infer', state=state, prompt=prompt, images=raw,
                                                     seed=noise_seed))['actions'])
            assert chunk.shape == (p['evaluation']['horizon'][a.model], 14) and np.isfinite(chunk).all()
            if spec['mode'] == 'stage_only' and not ever_exposed and not want_patch:
                assert np.array_equal(chunk, reference['actions'][q]), ('prefix action', q)
            states.append(state)
            chunks.append(chunk)
            query_steps.append(task.take_action_cnt)
            hashes.append([digest(obs['observation'][n]['rgb']) for n in CAMERAS])
            poses.append(pose)
            query_phases.append(phase)
            active.append(bool(want_patch))
            if patch is None:
                visible.append([0, 0, 0])
            else:
                cams = camera_objects(task)
                visible.append([int((cams[n].get_picture('Segmentation')[..., 1] == patch.actor.per_scene_id).sum()) for n in CAMERAS])
            ever_exposed |= want_patch
            for action in chunk[:p['evaluation']['replan'][a.model]]:
                task.take_action(action, action_type='qpos')
                current = np.asarray(task.robot.get_left_arm_jointState() + task.robot.get_right_arm_jointState())
                changed = tracker.update(current, task.container.get_pose().p, task.plate.get_pose().p, task.take_action_cnt)
                if changed or task.eval_success or task.take_action_cnt >= task.step_lim:
                    break
        assert all(not flag or exposed(spec, phase) for flag, phase in zip(active, query_phases))
        temp = output.with_suffix('.tmp.npz')
        np.savez_compressed(temp, states14=states, actions=chunks, query_steps=query_steps,
            rgb_sha256=hashes, object_plate=poses, query_phases=query_phases, patch_active=active,
            visible_pixels=visible, prompt=prompt)
        temp.replace(output.with_suffix('.npz'))
        dump(output, dict(seed=a.seed, model=a.model, condition=a.condition, protocol_sha256=protocol_hash,
            success=bool(task.eval_success), steps=task.take_action_cnt, queries=len(states),
            initial_clean_sha256=initial_hash, events=tracker.events, query_phases=query_phases,
            active_queries=sum(active), visible_queries=sum(any(v) for v in visible),
            prefix_verified_queries=prefix_verified, target_stage=spec['stage'], exposure_mode=spec['mode'],
            elapsed_s=time.monotonic()-began))
        print('[stage rollout]', a.model, a.condition, a.seed, bool(task.eval_success),
              'exposure queries', sum(active), flush=True)
    except Exception as error:
        dump(output, dict(seed=a.seed, model=a.model, condition=a.condition,
            protocol_sha256=protocol_hash, error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        if task is not None:
            task.close_env()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DEFAULT)
    ap.add_argument('--model', choices=['pi', 'wam'], required=True)
    ap.add_argument('--condition', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    run(ap.parse_args())
