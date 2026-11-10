"""Disjoint Spatial collection and paired DRIFT digital/native-render LIBERO rollouts.

Image preprocessing, task limits, five-action execution and warmup follow upstream
DRIFT. A real rigid gripper-mounted plane is calibrated only on a training state.
It is an explicitly documented physical extension, not geometry released by DRIFT.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from drift_paper_common import (ROOT, SUITES, LIMITS, TRAIN_INITS, TEST_INITS, CONDITIONS,
                                bootstrap, configure_libero, dump, read, sha, stage)

PATCH_NAME = 'drift_gripper_patch'


def inputs(obs):
    from openpi_client import image_tools
    return tuple(image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs[k][::-1, ::-1]), 224, 224))
        for k in ['agentview_image', 'robot0_eye_in_hand_image'])


def state8(obs):
    from drift_eval_digital import quat2axisangle
    return np.concatenate([obs['robot0_eef_pos'], quat2axisangle(obs['robot0_eef_quat']),
                           obs['robot0_gripper_qpos']]).astype(np.float32)


def make_env(task, processor=None):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    if processor is not None:
        env.env.set_xml_processor(processor)
    return env


def reset(env, initial, seed):
    env.seed(seed)
    np.random.seed(seed)
    env.reset()
    obs = env.set_init_state(initial)
    for _ in range(10):
        obs, _, _, _ = env.step([0.] * 6 + [-1.])
    return obs


def camera_render(env, camera, segmentation=False, depth=False):
    sim = env.env.sim
    ctx = sim._render_context_offscreen
    cid = sim.model.camera_name2id(camera)
    ctx.render(256, 256, camera_id=cid, segmentation=segmentation)
    value = ctx.read_pixels(256, 256, depth=depth, segmentation=segmentation)
    if depth:
        return tuple(np.ascontiguousarray(v[::-1, ::-1]) for v in value)
    return np.ascontiguousarray(value[::-1, ::-1])


def plane_processor(geometry, texture):
    from PIL import Image
    texture = Path(texture)
    if geometry.get('texture_upload_transform') == 'fliplr':
        # MuJoCo's box +z UV has the opposite horizontal axis to the model ROI.
        # Export coordinates only: the pattern seen in the model keeps its
        # original orientation, verified with the asymmetric four-color probe.
        upload = texture.with_name(texture.stem + '_mujoco_uv.png')
        pixels = np.asarray(Image.open(texture).convert('RGB'))
        Image.fromarray(np.ascontiguousarray(pixels[:, ::-1])).save(upload)
        texture = upload
    def process(xml):
        root = ET.fromstring(xml)
        body = root.find('.//body[@name="' + geometry['parent_body'] + '"]')
        assert body is not None, 'gripper camera parent missing'
        asset = root.find('asset')
        ET.SubElement(asset, 'texture', dict(name=PATCH_NAME + '_tex', type='2d', file=str(texture)))
        ET.SubElement(asset, 'material', dict(name=PATCH_NAME + '_mat', texture=PATCH_NAME + '_tex',
            texrepeat='1 1', texuniform='false', specular='0', shininess='0', reflectance='0'))
        child = ET.SubElement(body, 'body', dict(name=PATCH_NAME + '_body'))
        ET.SubElement(child, 'geom', dict(name=PATCH_NAME, type='box',
            pos=' '.join(map(str, geometry['box_center_parent'])),
            quat=' '.join(map(str, geometry['box_quat_parent'])),
            size=' '.join(map(str, geometry['box_half_sizes'])),
            material=PATCH_NAME + '_mat', group='1', contype='0', conaffinity='0', density='0'))
        return ET.tostring(root, encoding='unicode')
    return process


def visibility(env):
    import mujoco
    sim = env.env.sim
    gid = sim.model.geom_name2id(PATCH_NAME)
    masks = []
    from openpi_client import image_tools
    for camera in ['agentview', 'robot0_eye_in_hand']:
        seg = camera_render(env, camera, segmentation=True)
        mask = (seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 1] == gid)
        mask224 = image_tools.resize_with_pad(np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, -1), 224, 224)[..., 0] >= 128
        masks.append(mask224)
    return masks


def geometry(args):
    """Back-project the exact digital ROI to one rigid plane in the camera parent.

    The depth comes from visible gripper geometry on a training init only; it is
    frozen, never adjusted using attack strength or evaluation images.
    """
    import mujoco
    from PIL import Image
    from libero.libero import benchmark
    root = Path(args.root)
    dest = root / 'geometry'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        assert read(dest / 'complete.json')['complete']
        return
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    task, initial = suite.get_task(0), suite.get_task_init_states(0)[TRAIN_INITS[0]]
    env = make_env(task)
    try:
        obs = reset(env, initial, 700010)
        ref_state = env.get_sim_state().copy()
        clean_b, clean_w = inputs(env.regenerate_obs_from_state(ref_state))
        sim = env.env.sim
        cid = sim.model.camera_name2id('robot0_eye_in_hand')
        parent = int(sim.model.cam_bodyid[cid])
        parent_name = sim.model.body_id2name(parent)
        rcam = np.asarray(sim.data.cam_xmat[cid]).reshape(3, 3)
        rbody = np.asarray(sim.data.body_xmat[parent]).reshape(3, 3)
        cam_origin = rbody.T @ (np.asarray(sim.data.cam_xpos[cid]) - np.asarray(sim.data.body_xpos[parent]))
        cam_rotation = rbody.T @ rcam
        seg = camera_render(env, 'robot0_eye_in_hand', segmentation=True)
        _, depth_buf = camera_render(env, 'robot0_eye_in_hand', depth=True)
        model = sim.model._model
        near, far = float(model.vis.map.znear * model.stat.extent), float(model.vis.map.zfar * model.stat.extent)
        real_depth = near / (1 - depth_buf * (1 - near / far))
        hand_gids = [i for i in range(sim.model.ngeom)
                     if any(word in str(sim.model.geom_id2name(i)) for word in ['hand_visual', 'finger1_visual', 'finger2_visual'])]
        gripper = (seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 1], hand_gids)
        roi = np.zeros((256, 256), bool)
        roi[183:, 91:165] = True
        valid = gripper & roi
        assert valid.sum() >= 10, 'paper ROI has insufficient real gripper support to determine a physical depth'
        # A thin supported plate just in front of the closest gripper surface.
        # It can extend beyond the visible gripper silhouette; report that fact.
        depth = float(np.quantile(real_depth[valid], .02) - .001)
        assert .01 < depth < .30, ('unexpected physical gripper depth', depth)
        f = 128 / np.tan(np.deg2rad(float(sim.model.cam_fovy[cid])) / 2)
        corners = np.array([[80, 160], [144, 160], [144, 224], [80, 224]], float)
        native = corners * (256 / 224) - .5
        camera_vertices = np.c_[-(native[:, 0] - 127.5) * depth / f,
                                -(native[:, 1] - 127.5) * depth / f, np.full(4, -depth)]
        vertices = camera_vertices @ cam_rotation.T + cam_origin
        # PNG row zero maps to OBJ v=1. This is checked with an asymmetric probe.
        mesh = dest / 'gripper_plane.obj'
        lines = ['v ' + ' '.join(map(str, p)) for p in vertices]
        lines += ['vt 0 1', 'vt 1 1', 'vt 1 0', 'vt 0 0', 'vn ' + ' '.join(map(str, cam_rotation[:, 2])),
                  'f 1/1/1 2/2/1 3/3/1', 'f 1/1/1 3/3/1 4/4/1']
        mesh.write_text('\n'.join(lines) + '\n')
        info = dict(parent_body=parent_name, mesh_path=str(mesh), mesh_sha256=sha(mesh),
            source_suite='libero_spatial', source_task=str(task.language), source_init=TRAIN_INITS[0],
            target_model_roi_yxhw=[160, 80, 64, 64], camera_depth_m=depth,
            plane_width_m=float(np.linalg.norm(vertices[1] - vertices[0])),
            plane_height_m=float(np.linalg.norm(vertices[3] - vertices[0])),
            parent_vertices=vertices.tolist(), camera_fovy_deg=float(sim.model.cam_fovy[cid]),
            gripper_support_fraction_roi_native=float(valid.sum() / roi.sum()),
            physical_extension='Rigid zero-mass visual plane on the gripper camera parent; exact nominal ROI projection. '
                'A mounted plate, not a demonstrated flush sticker. Metric pose is not specified by DRIFT. '
                'Normal MuJoCo lighting, filtering and occlusion; no input-space compositing for render conditions.')
        box_quat = np.zeros(4)
        mujoco.mju_mat2Quat(box_quat, np.ascontiguousarray(cam_rotation).reshape(-1))
        info.update(box_center_parent=(vertices.mean(0) - cam_rotation[:, 2] * .0001).tolist(),
                    box_quat_parent=box_quat.tolist(),
                    box_half_sizes=[info['plane_width_m']/2, info['plane_height_m']/2, .0001])
        dump(dest / 'geometry_candidate.json', info)
        bodies = {sim.model.body_id2name(i): (float(sim.model.body_mass[i]), sim.model.body_inertia[i].copy())
                  for i in range(sim.model.nbody)}
        np.savez_compressed(dest / 'reference.npz', state=ref_state, clean_base=clean_b, clean_wrist=clean_w,
                            initial=initial)
        Image.fromarray(clean_w).save(dest / 'clean_wrist.png')
        reference_evolution = []
        for _ in range(10):
            env.step([0.] * 6 + [-1.])
            reference_evolution.append(env.get_sim_state().copy())
        del sim, model
    finally:
        env.close()
    probe = np.zeros((64, 64, 3), np.uint8)
    probe[:32, :32] = [255, 0, 0]
    probe[:32, 32:] = [0, 255, 0]
    probe[32:, :32] = [0, 0, 255]
    probe[32:, 32:] = [255, 255, 0]
    probe[:8, :8] = 255
    tex = dest / 'orientation_probe.png'
    Image.fromarray(probe).save(tex)
    info['texture_upload_transform'] = 'fliplr'
    env = make_env(task, plane_processor(info, tex))
    try:
        reset(env, initial, 700010)
        assert np.array_equal(ref_state, env.get_sim_state()), 'render attachment changes settled physics'
        for expected_state in reference_evolution:
            env.step([0.] * 6 + [-1.])
            assert np.array_equal(expected_state, env.get_sim_state()), 'render attachment changes physics evolution'
        env.regenerate_obs_from_state(ref_state)
        sim = env.env.sim
        debug_gid = sim.model.geom_name2id(PATCH_NAME)
        debug_model = sim.model._model
        debug_vertices = np.array([[-1,-1,1],[-1,1,1],[1,1,1],[1,-1,1]]) * np.asarray(sim.model.geom_size[debug_gid])
        debug_world = debug_vertices @ np.asarray(sim.data.geom_xmat[debug_gid]).reshape(3, 3).T + sim.data.geom_xpos[debug_gid]
        debug_cid = sim.model.camera_name2id('robot0_eye_in_hand')
        debug_cam = (debug_world - sim.data.cam_xpos[debug_cid]) @ np.asarray(sim.data.cam_xmat[debug_cid]).reshape(3, 3)
        dump(dest / 'compiled_geometry.json', dict(camera_vertices=debug_cam.tolist(),
             geom_rgba=sim.model.geom_rgba[debug_gid].tolist(), geom_group=int(sim.model.geom_group[debug_gid]),
             geom_size=sim.model.geom_size[debug_gid].tolist(), geom_position=sim.data.geom_xpos[debug_gid].tolist()))
        del debug_model
        for i in range(sim.model.nbody):
            name = sim.model.body_id2name(i)
            if name in bodies:
                mass, inertia = bodies[name]
                assert mass == float(sim.model.body_mass[i]) and np.array_equal(inertia, sim.model.body_inertia[i]), name
        rgb_b, rgb_w = inputs(env.regenerate_obs_from_state(ref_state))
        masks = visibility(env)
        repeat = inputs(env.regenerate_obs_from_state(ref_state))
        assert np.array_equal(repeat[0], rgb_b) and np.array_equal(repeat[1], rgb_w), 'segmentation changed RGB'
        mask = masks[1]
        target = np.zeros((224, 224), bool)
        target[160:, 80:144] = True
        iou = float((mask & target).sum() / (mask | target).sum())
        Image.fromarray(np.concatenate([clean_w, rgb_w], axis=1)).save(dest / 'clean_vs_native_probe.png')
        Image.fromarray(rgb_b).save(dest / 'native_probe_base.png')
        Image.fromarray(mask.astype(np.uint8) * 255).save(dest / 'probe_mask.png')
        assert iou >= .90, ('physical projection fails to reproduce prescribed ROI', iou, int(mask.sum()))
        expected = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]])
        classifications = []
        for y, x in [(176, 96), (176, 128), (208, 96), (208, 128)]:
            color = np.median(rgb_w[y-4:y+4, x-4:x+4], axis=(0, 1)).astype(float)
            classifications.append(int(np.argmax((color / max(np.linalg.norm(color), 1)) @
                                    (expected / np.linalg.norm(expected, axis=1, keepdims=True)).T)))
        assert classifications == [0, 1, 2, 3], ('texture UV orientation mismatch', classifications)
        Image.fromarray(np.concatenate([clean_w, rgb_w], axis=1)).save(dest / 'clean_vs_native_probe.png')
        Image.fromarray(rgb_b).save(dest / 'native_probe_base.png')
        dump(dest / 'geometry.json', info)
        dump(dest / 'complete.json', dict(complete=True, physics_exact=True, ten_step_physics_evolution_exact=True, original_body_inertias_exact=True,
            rgb_repeat_exact=True, uv_quadrants=classifications, wrist_roi_iou=iou,
            visible_wrist224=int(mask.sum()), visible_base224=int(masks[0].sum()), geometry_sha256=sha(dest / 'geometry.json')))
    finally:
        env.close()


def run_episode(env, task, initial, init_id, suite_name, task_id, client, condition, patch, record,
                collect=False, video=False):
    """Use one noise stream for matching query numbers across conditions."""
    import imageio
    seed = 700000 + SUITES.index(suite_name) * 10000 + task_id * 100 + init_id
    obs = reset(env, initial, seed)
    initial_state = env.get_sim_state().copy()
    body_pos = env.env.sim.model.body_pos.copy()
    body_quat = env.env.sim.model.body_quat.copy()
    actions, noise_ids, states, simstates, hashes, counts, pool, frames = [], [], [], [], [], [], [], []
    plan = collections.deque()
    success, queries, success_step = False, 0, None
    for t in range(LIMITS[suite_name]):
        if not plan:
            # All conditions get a fresh render at exactly the saved simulator state.
            obs = env.regenerate_obs_from_state(env.get_sim_state().copy())
            base, wrist = inputs(obs)
            if condition.endswith('_digital'):
                wrist = wrist.copy()
                wrist[160:224, 80:144] = patch
            st = state8(obs)
            nseed = 100000000 + SUITES.index(suite_name) * 10000000 + task_id * 100000 + init_id * 1000 + queries
            noise = np.random.default_rng(nseed).standard_normal((10, 32)).astype(np.float32)
            request = {'observation/image': base, 'observation/wrist_image': wrist,
                       'observation/state': st, 'prompt': str(task.language)}
            if collect:
                pool.append({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in request.items()})
            request['noise'] = noise
            action = np.asarray(client.infer(request)['actions'])
            assert action.shape[0] >= 5 and action.shape[1] == 7 and np.isfinite(action).all()
            actions.append(action)
            noise_ids.append(nseed)
            states.append(st)
            simstates.append(env.get_sim_state().copy())
            hashes.append([hashlib.sha256(x.tobytes()).hexdigest() for x in [base, wrist]])
            if condition.endswith('_render'):
                masks = visibility(env)
                counts.append([int(m.sum()) for m in masks])
                after = inputs(env.regenerate_obs_from_state(env.get_sim_state().copy()))
                assert np.array_equal(after[0], base) and np.array_equal(after[1], wrist), 'diagnostic altered input'
            if video:
                frames.append(np.concatenate([base, wrist], axis=1))
            plan.extend(action[:5])
            queries += 1
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            success, success_step = True, t + 10
            break
    np.savez_compressed(record, initial_state=initial_state, fixture_body_pos=body_pos,
        fixture_body_quat=body_quat, actions=np.array(actions), noise_seeds=np.array(noise_ids),
        state8=np.array(states), sim_states=np.array(simstates), input_sha256=np.array(hashes),
        visible_pixels224=np.array(counts, dtype=np.int32))
    if frames:
        imageio.mimwrite(str(record.with_suffix('.mp4')), frames, fps=2)
    row = dict(suite=suite_name, task_id=task_id, task=str(task.language), init_index=init_id,
        condition=condition, success=success, success_step=success_step, queries=queries, env_seed=seed,
        initial_state_sha256=hashlib.sha256(initial_state.tobytes()).hexdigest(),
        record=str(record), record_sha256=sha(record), runtime_error=False)
    return row, pool


def collect(args):
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy
    root = Path(args.root)
    dest = root / 'collection'
    dest.mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    all_obs, rows = [], []
    try:
        for tid in range(suite.n_tasks):
            task, inits = suite.get_task(tid), suite.get_task_init_states(tid)
            env = make_env(task)
            try:
                for init_id in TRAIN_INITS:
                    prefix = dest / ('task%02d_init%02d' % (tid, init_id))
                    if prefix.with_suffix('.json').exists():
                        row = read(prefix.with_suffix('.json'))
                        assert sha(prefix.with_suffix('.npz')) == row['record_sha256']
                        pool = np.load(str(prefix) + '_pool.npy', allow_pickle=True).tolist()
                    else:
                        row, pool = run_episode(env, task, inits[init_id], init_id, 'libero_spatial', tid,
                            client, 'clean', None, prefix.with_suffix('.npz'), collect=True)
                        np.save(str(prefix) + '_pool.npy', pool, allow_pickle=True)
                        dump(prefix.with_suffix('.json'), row)
                    all_obs.extend(pool)
                    rows.append(row)
                    stage(root / 'collection', 'collecting', completed_episodes=len(rows), target_episodes=30,
                          frames=len(all_obs), task=tid, init_index=init_id)
            finally:
                env.close()
    finally:
        client._ws.close()
    assert len(rows) == 30 and len(set(o['prompt'] for o in all_obs)) == 10
    data = {k: np.stack([o[k] for o in all_obs])
            for k in ['observation/image', 'observation/wrist_image', 'observation/state']}
    data['prompts'] = [o['prompt'] for o in all_obs]
    np.save(root / 'pool.npy', data)
    dump(dest / 'complete.json', dict(complete=True, episodes=rows, frames=len(all_obs),
        train_indices=list(TRAIN_INITS), pool_sha256=sha(root / 'pool.npy'), success_filter=False,
        note='Same 10 Spatial tasks x 3 episodes as paper; init10-12 keeps the official eval0-9 disjoint. '
             'Frame count is measured, not forced to the reported 696.'))


def evaluate(args):
    from PIL import Image
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy
    root = Path(args.root)
    geometry_info = read(root / 'geometry/geometry.json')
    patch_paths = dict(random=root / 'random.png', drift=root / 'train/drift_pi05_64px.png')
    patches = {k: np.asarray(Image.open(p).convert('RGB')) for k, p in patch_paths.items()}
    assert read(root / 'train/complete.json')['png_sha256'] == sha(patch_paths['drift'])
    conditions = args.conditions.split(',') if args.conditions else list(CONDITIONS)
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    try:
        for suite_name in SUITES:
            suite = benchmark.get_benchmark_dict()[suite_name]()
            for tid in range(suite.n_tasks):
                task, inits = suite.get_task(tid), suite.get_task_init_states(tid)
                for condition in conditions:
                    dest = root / 'eval' / suite_name / ('task%02d' % tid) / condition
                    dest.mkdir(parents=True, exist_ok=True)
                    patch_kind = condition.split('_')[0]
                    texture = patch_paths.get(patch_kind)
                    proc = plane_processor(geometry_info, texture) if condition.endswith('_render') else None
                    env = make_env(task, proc)
                    try:
                        for init_id in TEST_INITS:
                            prefix = dest / ('init%02d' % init_id)
                            if prefix.with_suffix('.json').exists():
                                row = read(prefix.with_suffix('.json'))
                                assert row['runtime_error'] is False and sha(prefix.with_suffix('.npz')) == row['record_sha256']
                                continue
                            row, _ = run_episode(env, task, inits[init_id], init_id, suite_name, tid,
                                client, condition, patches.get(patch_kind), prefix.with_suffix('.npz'), video=init_id == 0)
                            if condition != 'clean':
                                clean_path = root / 'eval' / suite_name / ('task%02d' % tid) / 'clean' / ('init%02d.json' % init_id)
                                clean = read(clean_path)
                                assert clean['initial_state_sha256'] == row['initial_state_sha256'], 'condition changes initial physics'
                            dump(prefix.with_suffix('.json'), row)
                            stage(root / 'eval', 'evaluating', suite=suite_name, task=tid, condition=condition,
                                  init_index=init_id, success=row['success'])
                    finally:
                        env.close()
    finally:
        client._ws.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['geometry', 'collect', 'evaluate'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--port', type=int, default=18671)
    ap.add_argument('--conditions', default='')
    args = ap.parse_args()
    bootstrap(sim=True)
    configure_libero(args.root)
    globals()[args.mode](args)
