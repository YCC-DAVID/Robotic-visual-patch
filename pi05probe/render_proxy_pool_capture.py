"""New clean trajectories with complete replay state and independent split indices."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-render-proxy-pool-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8144)
    ap.add_argument('--indices', default='0,1,2,3,4,5,6,7,8,9,10,11')
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--out', default='render_proxy_eot/data')
    args = ap.parse_args()
    import collections
    import torch
    import yaml
    import cv2
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    from drift_resize_order import resize_rgb
    import scene_patch as sp
    dest = D.OUT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'pool.npz').exists(), 'do not overwrite a completed data pool'
    patch = np.load(D.OUT / 'drift_patch_drift.npz', allow_pickle=False)
    cfg = yaml.safe_load((D.PROBE / 'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(patch['patch_m'])] * 2
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    bddl = D.pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    init_path = D.pathlib.Path(get_libero_path('init_states')) / 'libero_goal' / (D.TASK + '.pruned_init')
    inits = np.array(torch.load(str(init_path)))
    indices = [int(i) for i in args.indices.split(',')]
    assert len(indices) == len(set(indices)) and all(0 <= i < len(inits) for i in indices)
    assert set(indices) <= set(range(12)), 'reserved test indices 40+ must not enter this pool'
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    probe = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    probe.env.set_xml_processor(sp.make_xml_processor(cfg, patch['cell_world'],
        str(D.OUT / 'white_patch_probe/white_texture.png')))
    arrays = {key: [] for key in ['clean_img256', 'clean_img224', 'clean_wrist224', 'clean_state8',
                                  'sim_states', 'ep_id', 'env_seed', 'env_step', 'noise_seeds', 'masks']}
    outcomes, fixtures, cached_observation_differences = [], [], []
    checks = {'repeat_clean_render_equal': True, 'obs_matches_state_replay': True,
              'patched_fixture_seed_sync': True}
    try:
        for ep in indices:
            seed = args.seed + ep
            env.seed(seed); env.reset()
            probe.seed(seed); probe.reset()
            # Visual-only body is appended, so original body indices remain intact.
            clean_positions = env.env.sim.model.body_pos.copy()
            assert np.array_equal(clean_positions, probe.env.sim.model.body_pos[:len(clean_positions)])
            fixtures.append({'episode': ep, 'env_seed': seed,
                             'body_pos': clean_positions.tolist(),
                             'body_quat': env.env.sim.model.body_quat.tolist()})
            obs = env.set_init_state(inits[ep])
            for _ in range(D.NWAIT):
                obs, _, _, _ = env.step(D.DUMMY)
            queue = collections.deque()
            ok, inference, success_step = False, 0, None
            goal = env.env.parsed_problem['goal_state']
            for t in range(D.MAX_STEPS):
                if not queue:
                    state = env.get_sim_state().copy()
                    cached_native = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
                    replay = env.regenerate_obs_from_state(state)
                    native = np.ascontiguousarray(replay['agentview_image'][::-1, ::-1])
                    if not np.array_equal(cached_native, native):
                        checks['obs_matches_state_replay'] = False
                        cached_observation_differences.append({'episode': ep, 'replan': inference,
                            'MAE_u8': float(np.abs(cached_native.astype(float) - native).mean()),
                            'fraction_different_pixels': float(np.any(cached_native != native, axis=-1).mean())})
                    # Always use state-synchronized regenerated RGB and robot observations.
                    # Cached env.step observations need not be pixel-identical to replay.
                    obs = replay
                    if inference == 0:
                        repeat = env.regenerate_obs_from_state(state)
                        assert np.array_equal(replay['agentview_image'], repeat['agentview_image'])
                    # Object-ID mask avoids dependence on the white RGB threshold.
                    probe.regenerate_obs_from_state(state)
                    sim = probe.env.sim
                    ctx = sim._render_context_offscreen
                    gid = sim.model.geom_name2id(cfg['patch']['name'])
                    cid = sim.model.camera_name2id('agentview')
                    ctx.render(256, 256, camera_id=cid, segmentation=True)
                    seg = np.ascontiguousarray(ctx.read_pixels(256, 256, segmentation=True)[::-1, ::-1])
                    import mujoco
                    visible = (seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 1] == gid)
                    coverage = resize_rgb(np.repeat((visible.astype(np.uint8) * 255)[..., None], 3, -1), 224)[..., 0]
                    st = np.concatenate([obs['robot0_eef_pos'], D.quat2axisangle(obs['robot0_eef_quat']),
                                         obs['robot0_gripper_qpos']]).astype(np.float32)
                    img = resize_rgb(native, 224)
                    wrist = resize_rgb(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]), 224)
                    noise_seed = 20261006 + ep * 1000 + inference
                    noise = np.random.RandomState(noise_seed).standard_normal((10, 32)).astype(np.float32)
                    values = [native, img, wrist, st, state, ep, seed, t + D.NWAIT, noise_seed, coverage >= 128]
                    for key, value in zip(arrays, values):
                        arrays[key].append(value)
                    out = client.infer({'observation/image': img, 'observation/wrist_image': wrist,
                                        'observation/state': st, 'prompt': str(task.language), 'noise': noise})
                    queue.extend(out['actions'][:D.REPLAN])
                    inference += 1
                    del ctx, sim
                obs, _, _, _ = env.step(queue.popleft().tolist())
                if all(env.env._eval_predicate(s) for s in goal):
                    ok, success_step = True, t + D.NWAIT
                    break
            outcomes.append({'episode': ep, 'env_seed': seed, 'success': ok,
                             'success_step': success_step, 'frames': inference})
            (dest / 'progress.json').write_text(json.dumps({'episodes': outcomes}, indent=2) + '\n')
            print('[pool] init={} success={} frames={} total_frames={}'.format(ep, ok, inference, len(arrays['ep_id'])), flush=True)
    finally:
        env.close(); probe.close(); client._ws.close()
    arrays = {key: np.array(value) for key, value in arrays.items()}
    arrays.update(prompt=np.array(str(task.language)), quad224=patch['quad224'], cell_xy=patch['cell_xy'],
                  cell_world=patch['cell_world'], patch_m=patch['patch_m'],
                  munion=np.any(arrays['masks'], axis=0), episodes=len(indices), seed=args.seed, replan=D.REPLAN,
                  train_mask=arrays['ep_id'] < 10, validation_mask=arrays['ep_id'] >= 10)
    np.savez_compressed(dest / 'pool.npz', **arrays)
    metadata = {'complete': True, 'task': D.TASK, 'train_indices': list(range(10)),
                'validation_indices': [10, 11], 'reserved_test_indices': list(range(40, 50)),
                'captured_indices': indices, 'episodes': outcomes, 'frames': len(arrays['ep_id']),
                'noise_seed_rule': '20261006 + init_index*1000 + replan_index', 'checks': checks,
                'mask_source': 'object-ID segmentation at256 -> production resize224 -> threshold128',
                'seed_rule': '10000 + init_index; each environment seeded immediately before reset',
                'init_states_sha256': hashlib.sha256(init_path.read_bytes()).hexdigest(),
                'pool_sha256': hashlib.sha256((dest / 'pool.npz').read_bytes()).hexdigest(),
                'fixtures': fixtures, 'cached_observation_differences': cached_observation_differences,
                'inference_observation_source': 'regenerate_obs_from_state immediately before every inference',
                'notes': 'New clean trajectories, full sim states and native256 retained. Test40+ disjoint '
                         'from this pool and historical first40 training; only use train_mask for training. '
                         '10/11 are disjoint from new training but were in historical full training.'}
    (dest / 'manifest.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print('[done]', dest, flush=True)


if __name__ == '__main__':
    main()
