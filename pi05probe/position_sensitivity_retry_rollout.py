"""Frozen texture position transfer; paired physics/noise and actual dual-camera visibility."""
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
    ap.add_argument('--task', required=True)
    ap.add_argument('--clean-reference', default='')
    ap.add_argument('--conditions', default='clean,random,optimized')
    ap.add_argument('--indices', default='40,41,42,43,44')
    ap.add_argument('--port', type=int, default=8144)
    ap.add_argument('--out', default='render_proxy_scale/manual_rollout')
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
    assert args.task == D.TASK, 'position sweep must retain source task'
    assert args.clean_reference and 'clean' not in names, 'reuse source task paired clean baseline'
    assert manifest['geometry']['patch_m'] == 0.13
    for name,path in paths.items():
        assert (None if path is None else hashlib.sha256(path.read_bytes()).hexdigest()) == manifest['texture_sha256'][name]
    ids = [int(i) for i in args.indices.split(',')]
    assert len(ids) == len(set(ids)) and set(ids) <= set(range(40, 50)), 'reserved test indices only'
    dest = D.OUT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'episodes.json').exists(), 'use a new output for each completed run'
    patch = {'patch_m': np.array(manifest['geometry']['patch_m']), 'cell_world': np.array(manifest['geometry']['cell_world'])}
    cfg = yaml.safe_load((D.PROBE / 'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(patch['patch_m'])] * 2
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == args.task)
    bddl = D.pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    init_path = D.pathlib.Path(get_libero_path('init_states')) / 'libero_goal' / (args.task + '.pruned_init')
    inits = np.array(torch.load(str(init_path)))
    assert max(ids) < len(inits)
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    results = {c: [] for c in names}
    clean_reference = None
    if args.clean_reference:
        clean_reference = D.pathlib.Path(args.clean_reference).resolve()
        reference_json = json.loads(clean_reference.read_text())
        assert reference_json['metadata']['task'] == args.task
        assert reference_json['metadata']['init_states_sha256'] == hashlib.sha256(init_path.read_bytes()).hexdigest()
        clean_rows = [r for r in reference_json['per_episode']['clean'] if r['init_index'] in ids]
        assert sorted(r['init_index'] for r in clean_rows) == sorted(ids)
        assert 'clean' not in names
        results['clean'] = clean_rows
        references = {r['init_index']: np.load(clean_reference.parent/r['replay_record'], allow_pickle=False) for r in clean_rows}

    metadata = {'clean_reference': str(clean_reference) if clean_reference else None, 'task': args.task, 'indices': ids, 'train_indices': list(range(10)), 'validation_indices': [10, 11],
                'geometry': {'patch_m': float(patch['patch_m']), 'cell_world': patch['cell_world'].tolist()},
                'textures': {c: None if p is None else {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                             for c, p in paths.items()},
                'environment_seed_rule': '10000 + init_index, immediately before reset',
                'action_noise_rule': '20261006 + init_index*1000 + replan_index',
                'init_states_sha256': hashlib.sha256(init_path.read_bytes()).hexdigest(),
                'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'prompt': str(task.language), 'bddl_sha256': hashlib.sha256(bddl.read_bytes()).hexdigest(),
                'source_task': manifest['source_task'], 'new_target_task_training': False,
                'segmentation_RGB_repeat_checked_each_patched_initial_frame': True,
                'camera_inputs': 'both actual patched render, no digital composition or input masking',
                'inference_observation_source': 'regenerate_obs_from_state immediately before every inference',
                'max_steps': D.MAX_STEPS, 'settling_steps': D.NWAIT, 'replan': D.REPLAN,
                'pilot': set(ids) <= set(range(40,45)), 'frequency': manifest.get('frequency'), 'train_run': manifest.get('train_run'), 'stage': manifest.get('stage'), 'notes': 'Robot task successes; paired ASR defined only among clean-success episodes. '
                                      'Same source task and frozen source texture; only patch world XY center changes. '
                                      'Reused clean reference; matched random at every position; exploratory previously used test states.'}

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
            reference_body_quat = None
            reference_initial_state = None
            if clean_reference:
                reference_body_pos = references[ep]['fixture_body_pos']
                reference_body_quat = references[ep]['fixture_body_quat']
                reference_initial_state = references[ep]['initial_state']
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
                    body_quat = env.env.sim.model.body_quat.copy()
                    if reference_body_pos is None:
                        reference_body_pos = body_pos
                        reference_body_quat = body_quat
                    else:
                        shared_bodies = min(len(reference_body_pos), len(body_pos))
                        assert np.array_equal(reference_body_pos[:shared_bodies], body_pos[:shared_bodies])
                        assert np.array_equal(reference_body_quat[:shared_bodies], body_quat[:shared_bodies])
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
                    visibility=[]
                    for t in range(D.MAX_STEPS):
                        if not queue:
                            obs = env.regenerate_obs_from_state(env.get_sim_state().copy())
                            img, wrist = D.model_input(obs)
                            counts=[-1,-1]
                            if paths[name] is not None:
                                from drift_resize_order import resize_rgb
                                sim=env.env.sim;ctx=sim._render_context_offscreen;gid=sim.model.geom_name2id(cfg['patch']['name'])
                                counts=[]
                                for camera in ['agentview','robot0_eye_in_hand']:
                                    cid=sim.model.camera_name2id(camera)
                                    ctx.render(256,256,camera_id=cid,segmentation=True)
                                    seg=np.ascontiguousarray(ctx.read_pixels(256,256,segmentation=True)[::-1,::-1])
                                    mask=(seg[...,0]==mujoco.mjtObj.mjOBJ_GEOM)&(seg[...,1]==gid)
                                    coverage=resize_rgb(np.repeat((mask.astype(np.uint8)*255)[...,None],3,-1),224)[...,0]>=128
                                    counts.append(int(coverage.sum()))
                                del sim,ctx
                                if inference==0:
                                    repeat_b,repeat_w=D.model_input(env.regenerate_obs_from_state(env.get_sim_state().copy()))
                                    if not (np.array_equal(repeat_b,img) and np.array_equal(repeat_w,wrist)):
                                        np.savez_compressed(dest / ('diagnostic_mismatch_init%d_%s.npz' % (ep, name)), before_base=img, after_base=repeat_b, before_wrist=wrist, after_wrist=repeat_w, state=env.get_sim_state().copy())
                                    assert np.array_equal(repeat_b,img) and np.array_equal(repeat_w,wrist), 'visibility diagnostic changed RGB input'
                            visibility.append(counts)
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
                                        actions=np.array(actions), noise_seeds=np.array(noise_seeds), visible_pixels224=np.array(visibility),
                                        initial_state=initial_state, fixture_body_pos=body_pos, fixture_body_quat=body_quat, init_index=ep, env_seed=10000 + ep)
                    results[name].append({'init_index': ep, 'env_seed': 10000 + ep, 'success': ok,
                                          'success_step': success_step, 'inferences': inference,
                                          'visible_pixels224_median': np.median(visibility,axis=0).tolist(),
                                          'visible_pixels224_max': np.max(visibility,axis=0).tolist(),
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
