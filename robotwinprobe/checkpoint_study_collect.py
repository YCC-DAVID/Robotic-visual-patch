"""Collect fixed, successful expert demonstrations in the actual evaluation scene.

Each episode runs in a separate process. All accepted and rejected seeds remain
recorded; expert filtering is for imitation data only, never evaluation seeds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import bootstrap as B

DEFAULT = B.OUT / 'checkpoint_study_20261023'
CAMERAS = ('head_camera', 'left_camera', 'right_camera')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def classify_attempt(row):
    """Keep known expert-data rejections distinct from unexpected runtime errors.

    This classification applies only to expert data collection, never to the
    unfiltered closed-loop evaluation seeds. Raw per-seed statuses are retained.
    """
    row = dict(row)
    error = row.get('error', '')
    trace = row.get('traceback', '')
    if (error.startswith('UnStableError(')
            and 'envs.utils.create_actor.UnStableError:' in trace
            and '_init_task_env_' in trace):
        row['scene_rejection'] = row.pop('error')
        row['classification'] = 'upstream scene initialization rejected unstable objects'
    elif (error == "AssertionError('target_pose cannot be None for move action.')"
            and 'choose_grasp_pose' not in trace and 'grasp_actor' in trace):
        row['planning_failure'] = row.pop('error')
        row['classification'] = 'upstream expert has no feasible grasp pose'
    return row


def worker(root, seed):
    B.simulator_paths()
    from scene_smoke import make_task
    import random
    random.seed(seed)
    dest = root / 'expert' / f'seed{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'status.json').exists():
        return
    task = None
    start = time.monotonic()
    try:
        task, args, embodiment = make_task('place_container_plate', seed, dest)
        task.need_plan = True
        task.save_freq = 15
        frames = []

        def capture():
            obs = task.get_obs()
            frames.append(dict(state=obs['joint_action']['vector'].astype(np.float32).copy(),
                **{name:obs['observation'][name]['rgb'].copy() for name in CAMERAS}))

        # Same collection points as the upstream _take_picture; avoid temporary
        # pickle caches and preserve original, uncompressed RGB in an NPZ file.
        task._take_picture = capture
        capture()
        prompt = 'Place the ' + ('bowl' if task.actor_name == '002_bowl' else 'cup') + ' on the plate.'
        info = task.play_once()
        capture()
        legal, excess = task.planned_joints_legal()
        success = bool(task.plan_success and task.check_success() and legal)
        meta = dict(seed=seed, success=success, frames=len(frames), prompt=prompt,
            plan_success=bool(task.plan_success), joints_legal=bool(legal),
            joint_excess=float(excess), embodiment=embodiment, save_freq=15,
            task_config='demo_clean', eval_mode=True, actor_name=task.actor_name,
            container_id=int(task.container_id), elapsed_s=time.monotonic()-start,
            action_label='future recorded joint_action.vector, absolute 14D; chunk starts at current frame')
        if success:
            state = np.stack([f['state'] for f in frames])
            assert state.shape[1] == 14 and np.isfinite(state).all() and len(state) > 32
            arrays = {name:np.stack([f[name] for f in frames]) for name in CAMERAS}
            assert all(a.shape[1:] == (240,320,3) and a.dtype == np.uint8 for a in arrays.values())
            target = dest / 'demonstration.npz'
            temporary = dest / 'demonstration.tmp.npz'
            np.savez_compressed(temporary, state14=state, prompt=prompt, seed=seed, **arrays)
            temporary.replace(target)
            meta['sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
            dump(dest / 'scene_config.json', args)
        dump(dest / 'status.json', meta)
        print('[expert]', json.dumps(meta), flush=True)
    except Exception as exc:
        dump(dest / 'status.json', dict(seed=seed, success=False, error=repr(exc),
            traceback=traceback.format_exc(), elapsed_s=time.monotonic()-start))
        raise
    finally:
        if task is not None:
            task.close_env()


def collect(root, split, count, start_seed):
    status_path = root / 'expert' / (split + '_manifest.json')
    selected = []
    attempted = []
    for seed in range(start_seed, start_seed + count * 5):
        dest = root / 'expert' / f'seed{seed}'
        dest.mkdir(parents=True, exist_ok=True)
        result = dest / 'status.json'
        if not result.exists():
            with (dest / 'collector.log').open('w') as log:
                command = [sys.executable, str(Path(__file__).resolve()), '--root', str(root), '--worker', '--seed', str(seed)]
                try:
                    proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                except subprocess.TimeoutExpired:
                    dump(result, dict(seed=seed, success=False, error='expert timeout 600s'))
            if not result.exists():
                dump(result, dict(seed=seed, success=False, error=f'worker exited {proc.returncode} without result'))
        row = classify_attempt(json.loads(result.read_text()))
        # Upstream choose_grasp_pose may return (None, None) when no IK
        # candidate exists; grasp_actor then constructs Action(None). Keep the
        # original traceback, but classify this specific expert planning failure.
        attempted.append(row)
        # Unexpected runtime failures are not silently reclassified as expert
        # planning failures. Stop for diagnosis; ordinary plan failures continue.
        if 'error' in row:
            dump(status_path, dict(complete=False, split=split, target=count, selected=selected,
                                  attempted=attempted, error=row['error']))
            raise RuntimeError(f'Expert worker failed for seed {seed}: {row["error"]}')
        if row['success']:
            selected.append(row)
        dump(status_path, dict(complete=len(selected)==count, split=split, target=count,
            start_seed=start_seed, selected=selected, attempted=attempted,
            test_seeds_unfiltered=True))
        print('[collection]', split, len(selected), '/', count, 'attempt', seed, flush=True)
        if len(selected) == count:
            return
    raise RuntimeError('Too few successful expert demonstrations in predeclared seed range')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--seed', type=int, default=20000)
    parser.add_argument('--split', choices=['smoke','train','validation'], default='train')
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        worker(root, args.seed)
    else:
        collect(root, args.split, args.count, args.seed)
