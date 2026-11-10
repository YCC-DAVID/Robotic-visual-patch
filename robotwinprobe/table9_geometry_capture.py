"""Capture clean depth / camera geometry, or verify one proposed patch at a time."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import bootstrap as B
from radius_plan import REFERENCE, dump


def capture(root, verify=False):
    B.simulator_paths()
    import sapien
    from scene_smoke import make_task
    from render_geometry import CAMERAS, Patch, camera_objects, patterns

    selected = json.loads((root / 'geometry/selection.json').read_text()) if verify else None
    if verify:
        assert selected['feasible'] and len(selected['centers']) == 9
        common = {(f['seed'],f['query']) for f in selected['training_common_frames']}
        with np.load(root/'geometry/screening.npz') as z:
            screening = z['head_metrics'].copy()
            frame_index = {(int(s),int(q)):i for i,(s,q) in enumerate(zip(z['frame_seeds'],z['frame_queries']))}
        from table9_geometry_select import poses as screen_poses
    assets = root / 'geometry' / ('verify_assets' if verify else 'capture_assets')
    assets.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in list(range(10100, 10108)) + [10110, 10111]:
        target = root / 'geometry' / ('verified' if verify else 'clean') / f'seed{seed}.npz'
        if target.exists():
            continue
        random.seed(seed)
        task, _, _ = make_task('place_container_plate', seed, assets)
        try:
            saved = {}
            with np.load(REFERENCE / 'clean' / f'seed{seed}.npz') as reference:
                for query, actions in enumerate(reference['actions']):
                    if verify and seed < 10110 and (seed,query) not in common:
                        for action in actions:
                            task.take_action(action,action_type='qpos')
                            if task.eval_success or task.take_action_cnt>=task.step_lim:break
                        continue
                    obs = task.get_obs()
                    state = obs['joint_action']['vector'].astype(np.float32)
                    assert np.array_equal(state, reference['states14'][query])
                    for name in CAMERAS:
                        assert np.array_equal(obs['observation'][name]['rgb'], reference[name][query])
                    cameras = camera_objects(task)
                    frame = dict(state14=state.copy(), query=np.array(query))
                    if not verify:
                        for name in CAMERAS:
                            camera = cameras[name]
                            position = camera.get_picture('Position').copy()
                            frame[name + '_clean'] = obs['observation'][name]['rgb'].copy()
                            frame[name + '_depth'] = -position[..., 2]
                            frame[name + '_depth_valid'] = position[..., 3] < 1
                            frame[name + '_intrinsic'] = camera.get_intrinsic_matrix()
                            frame[name + '_extrinsic'] = camera.get_extrinsic_matrix()
                    else:
                        # This is a montage of separate renders, never nine actors in one scene.
                        for site in selected['centers']:
                            sid = site['id']
                            poses = [dict(center=site['center_m'], yaw_deg=0.)]
                            # Nominal plus the most occluded predicted EOT pose;
                            # if all probes are clear, check a translated/rotated boundary.
                            values = screening[site['candidate_index'],frame_index[seed,query],:,0]
                            worst = int(values.argmin())
                            if worst == 0:worst = 4
                            offset,yaw = screen_poses()[worst]
                            poses.append(dict(center=[site['center_m'][0]+offset[0],site['center_m'][1]+offset[1],.741],yaw_deg=yaw))
                            for pi, pose in enumerate(poses):
                                patch = Patch(task, assets, center=pose['center'], size=.10)
                                try:
                                    patch.texture(patterns()['checker'])
                                    a = np.deg2rad(pose['yaw_deg']) / 2
                                    patch.actor.set_pose(sapien.Pose(pose['center'], [np.cos(a), 0, 0, np.sin(a)]))
                                    patched = task.get_obs()
                                    geo = patch.geometry(task)
                                    assert np.array_equal(state, patched['joint_action']['vector'].astype(np.float32))
                                    for name in CAMERAS:
                                        prefix = f'{sid}_p{pi}_{name}'
                                        frame[prefix + '_mask'] = geo[name]['mask'].copy()
                                        if not saved and pi == 0:
                                            frame[prefix + '_rgb'] = patched['observation'][name]['rgb'].copy()
                                    frame[f'{sid}_p{pi}_pose'] = np.r_[pose['center'],pose['yaw_deg']]
                                finally:
                                    task.scene.remove_actor(patch.actor)
                        restored = task.get_obs()
                        for name in CAMERAS:
                            assert np.array_equal(obs['observation'][name]['rgb'], restored['observation'][name]['rgb'])
                    for key, value in frame.items():
                        saved.setdefault(key, []).append(value)
                    rows.append(dict(seed=seed, query=query))
                    dump(root / 'geometry/progress.json', dict(stage='verify' if verify else 'capture_clean',
                        seed=seed, query=query, completed_frames_this_run=len(rows), single_patch_per_scene=True))
                    print('[geometry]', seed, query, 'verify' if verify else 'clean', flush=True)
                    for action in actions:
                        task.take_action(action, action_type='qpos')
                        if task.eval_success or task.take_action_cnt >= task.step_lim:
                            break
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, seed=seed, **{k: np.array(v) for k, v in saved.items()})
            tmp.replace(target)
        finally:
            task.close_env()
    dump(root / 'geometry' / ('verify_capture_complete.json' if verify else 'clean_complete.json'),
        dict(complete=True, seeds=list(range(10100, 10108)) + [10110, 10111],
            initial_clean_state_RGB_exact=True, single_patch_per_scene=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    capture(Path(args.root).resolve(), args.verify)
