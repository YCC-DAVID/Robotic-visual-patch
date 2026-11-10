"""Paired held-out closed loop using actual texture PNGs in MuJoCo, both cameras."""
import argparse
import collections
import hashlib
import json
import os
import time
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-render-proxy-rollout-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--conditions', default='clean,random_world,old_full')
    ap.add_argument('--indices', default='40,41,42,43,44')
    ap.add_argument('--port', type=int, default=8144)
    ap.add_argument('--out', default='render_proxy_eot/physical_baseline')
    args = ap.parse_args()
    import torch
    import yaml
    import mujoco
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    import scene_patch as sp
    manifest_path = D.pathlib.Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    names = args.conditions.split(',')
    paths = {c: None if manifest['textures'][c] is None else D.pathlib.Path(manifest['textures'][c]).resolve()
             for c in names}
    assert all(p is None or p.is_file() for p in paths.values())
    ids = [int(i) for i in args.indices.split(',')]
    assert len(ids) == len(set(ids)) and set(ids) <= set(range(40, 50)), 'reserved test indices only'
    dest = D.OUT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'episodes.json').exists(), 'use a new output for each completed run'
    patch = np.load(D.OUT / 'drift_patch_drift.npz', allow_pickle=False)
    cfg = yaml.safe_load((D.PROBE / 'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(patch['patch_m'])] * 2
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    bddl = D.pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    init_path = D.pathlib.Path(get_libero_path('init_states')) / 'libero_goal' / (D.TASK + '.pruned_init')
    inits = np.array(torch.load(str(init_path)))
    assert max(ids) < len(inits)
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    results = {c: [] for c in names}
    metadata = {'task': D.TASK, 'indices': ids, 'train_indices': list(range(10)), 'validation_indices': [10, 11],
                'geometry': {'patch_m': float(patch['patch_m']), 'cell_world': patch['cell_world'].tolist()},
                'textures': {c: None if p is None else {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                             for c, p in paths.items()},
                'environment_seed_rule': '10000 + init_index, immediately before reset',
                'action_noise_rule': '20261006 + init_index*1000 + replan_index',
                'init_states_sha256': hashlib.sha256(init_path.read_bytes()).hexdigest(),
                'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'camera_inputs': 'both actual patched render, no digital composition or input masking',
                'inference_observation_source': 'regenerate_obs_from_state immediately before every inference',
                'max_steps': D.MAX_STEPS, 'settling_steps': D.NWAIT, 'replan': D.REPLAN,
                'pilot': True, 'notes': 'Robot task successes; paired ASR defined only among clean-success episodes. '
                                      'Test40+ disjoint from new and historical first40 training. '
                                      'Do not compare this 5-episode pilot directly to old 50-episode rates.'}

    def save():
        summary = {c: {'n': len(rows), 'successes': sum(r['success'] for r in rows),
                       'robot_success_rate': float(np.mean([r['success'] for r in rows])) if rows else None}
                   for c, rows in results.items()}
        clean_rows = {r['init_index']: r['success'] for r in results.get('clean', [])}
        for c, rows in results.items():
            paired = [r for r in rows if clean_rows.get(r['init_index']) is True]
            summary[c]['paired_ASR_clean_success_n'] = len(paired)
            summary[c]['paired_ASR_failures'] = sum(not r['success'] for r in paired)
            summary[c]['paired_ASR'] = (sum(not r['success'] for r in paired) / len(paired)) if paired else None
        (dest / 'episodes.json').write_text(json.dumps({'metadata': metadata, 'summary': summary,
                                                      'per_episode': results}, indent=2, allow_nan=False) + '\n')

    start = time.monotonic()
    try:
        for ep in ids:
            reference_body_pos = None
            reference_initial_state = None
            for name in names:
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
                if paths[name] is not None:
                    env.env.set_xml_processor(sp.make_xml_processor(cfg, patch['cell_world'], str(paths[name])))
                try:
                    env.seed(10000 + ep); env.reset()
                    if paths[name] is not None:
                        texture = np.asarray(Image.open(paths[name]).convert('RGB'))
                        model = env.env.sim.model._model
                        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, cfg['patch']['name'] + '_tex')
                        adr = int(model.tex_adr[tid])
                        actual = model.tex_data[adr:adr+texture.size].reshape(texture.shape)
                        assert np.array_equal(actual, texture), 'PNG orientation differs from model texture upload'
                        del model, actual
                    body_pos = env.env.sim.model.body_pos.copy()
                    if reference_body_pos is None:
                        reference_body_pos = body_pos
                    else:
                        shared_bodies = min(len(reference_body_pos), len(body_pos))
                        assert np.array_equal(reference_body_pos[:shared_bodies], body_pos[:shared_bodies])
                    obs = env.set_init_state(inits[ep])
                    for _ in range(D.NWAIT):
                        obs, _, _, _ = env.step(D.DUMMY)
                    initial_state = env.get_sim_state().copy()
                    if reference_initial_state is None:
                        reference_initial_state = initial_state
                    else:
                        assert np.array_equal(reference_initial_state, initial_state), 'patch changes initial physics'
                    queue = collections.deque()
                    ok, inference, success_step = False, 0, None
                    goal = env.env.parsed_problem['goal_state']
                    noise_seeds, actions, state_inputs, replay_states, images, wrists = [], [], [], [], [], []
                    for t in range(D.MAX_STEPS):
                        if not queue:
                            obs = env.regenerate_obs_from_state(env.get_sim_state().copy())
                            img, wrist = D.model_input(obs)
                            st = np.concatenate([obs['robot0_eef_pos'], D.quat2axisangle(obs['robot0_eef_quat']),
                                                 obs['robot0_gripper_qpos']]).astype(np.float32)
                            noise_seed = 20261006 + ep * 1000 + inference
                            noise = np.random.RandomState(noise_seed).standard_normal((10, 32)).astype(np.float32)
                            result = client.infer({'observation/image': img, 'observation/wrist_image': wrist,
                                                   'observation/state': st, 'prompt': str(task.language), 'noise': noise})
                            action = np.array(result['actions'], copy=True)
                            assert np.isfinite(action).all()
                            actions.append(action); noise_seeds.append(noise_seed); state_inputs.append(st)
                            replay_states.append(env.get_sim_state().copy()); images.append(img); wrists.append(wrist)
                            queue.extend(action[:D.REPLAN]); inference += 1
                        obs, _, _, _ = env.step(queue.popleft().tolist())
                        if all(env.env._eval_predicate(s) for s in goal):
                            ok, success_step = True, t + D.NWAIT
                            break
                    record_path = dest / ('init{:02d}_{}.npz'.format(ep, name))
                    np.savez_compressed(record_path, base224=np.array(images), wrist224=np.array(wrists),
                                        state8=np.array(state_inputs), sim_states=np.array(replay_states),
                                        actions=np.array(actions), noise_seeds=np.array(noise_seeds),
                                        initial_state=initial_state, init_index=ep, env_seed=10000 + ep)
                    results[name].append({'init_index': ep, 'env_seed': 10000 + ep, 'success': ok,
                                          'success_step': success_step, 'inferences': inference,
                                          'replay_record': record_path.name,
                                          'initial_state_sha256': hashlib.sha256(initial_state.tobytes()).hexdigest()})
                    save()
                    print('[rollout] init={} {} success={} step={} calls={} elapsed={:.1f}s'.format(
                        ep, name, ok, success_step, inference, time.monotonic() - start), flush=True)
                finally:
                    env.close()
    finally:
        client._ws.close()
    print('[done]', dest, flush=True)


if __name__ == '__main__':
    main()
