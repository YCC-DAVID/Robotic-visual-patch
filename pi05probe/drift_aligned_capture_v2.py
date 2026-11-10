"""Native calibration for our texture training at the frozen DRIFT wrist pose.

No policy is loaded. Only the original Spatial init10--12 simulator states are
replayed. Segmentation lives in a separate environment from photometric passes.
All textures below are fixed calibration patterns, never optimized patches.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from drift_paper_common import ROOT, bootstrap, configure_libero, dump, now, read, sha

CAMERAS = ('agentview', 'robot0_eye_in_hand')
RESOLUTION = 64
RANDOM_SEED = 20261109


def patterns():
    x = np.rint(np.linspace(0, 255, RESOLUTION)).astype(np.uint8)
    ramp = np.empty((RESOLUTION, RESOLUTION, 3), np.uint8)
    ramp[..., 0] = x[None, :]
    ramp[..., 1] = x[:, None]
    ramp[..., 2] = 255
    random = np.random.default_rng(RANDOM_SEED).integers(0, 256, ramp.shape, dtype=np.uint8)
    return dict(black=np.zeros_like(ramp), white=np.full_like(ramp, 255),
                gray=np.full_like(ramp, 128), random=random, ramp=ramp)


def prepare(source, root):
    source, root = Path(source).resolve(), Path(root).resolve()
    dest = root / 'capture'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'shards').mkdir(exist_ok=True)
    (dest / 'source_snapshot').mkdir(exist_ok=True)
    protocol = read(source / 'protocol.json')
    assert protocol['model_config'] == 'pi05_libero'
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT / 'pi05probe' / name) == digest, ('frozen source changed', name)
    collection = read(source / 'collection/complete.json')
    assert collection['complete'] and collection['train_indices'] == [10, 11, 12]
    assert sha(source / 'pool.npy') == collection['pool_sha256']
    geometry = read(source / 'geometry/geometry.json')
    assert geometry['target_model_roi_yxhw'] == [160, 80, 64, 64]
    assert read(source / 'geometry/complete.json')['complete']
    assert geometry['texture_upload_transform'] == 'fliplr'
    rows = collection['episodes']
    assert len(rows) == 30
    assert {(r['task_id'], r['init_index']) for r in rows} == {
        (t, i) for t in range(10) for i in (10, 11, 12)}
    records = []
    offset = 0
    for row in rows:
        assert row['suite'] == 'libero_spatial' and row['runtime_error'] is False
        path = source / 'collection' / ('task%02d_init%02d.npz' % (row['task_id'], row['init_index']))
        assert sha(path) == row['record_sha256']
        records.append(dict(task_id=row['task_id'], init_index=row['init_index'],
                            frames=row['queries'], offset=offset, record=str(path),
                            record_sha256=row['record_sha256'], env_seed=row['env_seed'],
                            prompt=row['task']))
        offset += row['queries']
    assert offset == collection['frames']
    value = dict(schema='drift-aligned-native-calibration-v2', source_root=str(source),
        model_config='pi05_libero', frames=offset, texture_resolution=64,
        views=list(CAMERAS), uv_units='original PNG texel-center x,y; not normalized',
        uv_source='analytic front-face homography; unlit-ramp coordinates on visible side/back faces',
        lod_source='analytic front-face footprint, reference for native mipmap approximation',
        native_resolution=256, model_resolution=224, random_seed=RANDOM_SEED,
        train_indices=[10, 11, 12], eval_indices_excluded=list(range(10)),
        source_protocol_sha256=sha(source / 'protocol.json'),
        source_pool_sha256=sha(source / 'pool.npy'), source_sha256=protocol['source_sha256'],
        geometry_sha256=sha(source / 'geometry/geometry.json'),
        script_sha256=sha(__file__), records=records,
        segmentation='separate environment; photometric context is never segmented',
        fixture_replay='saved body_pos/quaternions plus qpos/qvel; map original bodies by name and retain extra patch pose',
        strict_checks=['source clean RGB224 exact', 'same-pattern repeated RGB256 exact',
                       'simulator state exact before/after', 'original fixtures exact'],
        calibration='fixed black/white/gray/random plus unlit white/UV ramp; no learned pattern fitting')
    if (root / 'reused_shards.json').exists():
        value['reused_shards_manifest_sha256'] = sha(root / 'reused_shards.json')
    path = dest / 'protocol.json'
    if path.exists():
        assert read(path) == value, 'capture protocol changed after freeze'
    else:
        dump(path, value)
    from PIL import Image
    import shutil
    for name, tex in patterns().items():
        path = dest / (name + '_texture.png')
        if path.exists():
            assert np.array_equal(np.asarray(Image.open(path).convert('RGB')), tex)
        else:
            Image.fromarray(tex).save(path)
    np.savez_compressed(dest / 'patterns.npz', **patterns())
    for name in list(protocol['source_sha256']) + [Path(__file__).name]:
        p = ROOT / 'pi05probe' / name
        target = dest / 'source_snapshot' / name
        if target.exists():
            assert sha(target) == sha(p)
        else:
            shutil.copy2(p, target)
    return value


def homography(sim, geometry, camera):
    """Native rotated image pixels -> original PNG texel centers on front face."""
    import cv2
    parent = sim.model.body_name2id(geometry['parent_body'])
    rotation = np.asarray(sim.data.body_xmat[parent]).reshape(3, 3)
    world = np.asarray(geometry['parent_vertices']) @ rotation.T + sim.data.body_xpos[parent]
    cid = sim.model.camera_name2id(camera)
    camera_xyz = (world - sim.data.cam_xpos[cid]) @ np.asarray(sim.data.cam_xmat[cid]).reshape(3, 3)
    f = 128 / np.tan(np.deg2rad(float(sim.model.cam_fovy[cid])) / 2)
    image_xy = 127.5 + f * camera_xyz[:, :2] / camera_xyz[:, 2:]
    tex_xy = np.array([[-.5, -.5], [63.5, -.5], [63.5, 63.5], [-.5, 63.5]])
    return cv2.getPerspectiveTransform(image_xy.astype(np.float32), tex_xy.astype(np.float32)).astype(np.float64)


def lod_from_homography(H):
    yy, xx = np.mgrid[:256, :256]
    den = H[2, 0] * xx + H[2, 1] * yy + H[2, 2]
    den2 = np.maximum(den * den, 1e-20)
    numx = H[0, 0] * xx + H[0, 1] * yy + H[0, 2]
    numy = H[1, 0] * xx + H[1, 1] * yy + H[1, 2]
    dx = np.stack([(H[0, 0] * den - numx * H[2, 0]) / den2,
                   (H[1, 0] * den - numy * H[2, 0]) / den2], axis=-1)
    dy = np.stack([(H[0, 1] * den - numx * H[2, 1]) / den2,
                   (H[1, 1] * den - numy * H[2, 1]) / den2], axis=-1)
    footprint = np.maximum(np.linalg.norm(dx, axis=-1), np.linalg.norm(dy, axis=-1))
    return np.clip(np.log2(np.maximum(footprint, 1)), 0, 6).astype(np.float32)


def difference(a, b):
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return dict(equal=bool(np.array_equal(a, b)), max_u8=int(delta.max()),
                MAE_u8=float(delta.mean()), changed_rgb_values=int(np.count_nonzero(delta)))


def set_state(env, state):
    env.set_state(state)
    env.env.sim.forward()
    assert np.array_equal(env.get_sim_state(), state), 'state replay changed simulator state'


def restore_fixture_layout(clean_env, environments, saved_pos, saved_quat, patch_body):
    """Replay model transforms that LIBERO excludes from its flattened state.

    The clean environment is compiled from the identical source BDDL and frozen
    code, so it defines the source model's body order. Name mapping preserves all
    original bodies even when the extra visual patch changes compiled indices.
    """
    model = clean_env.env.sim.model
    assert saved_pos.shape == model.body_pos.shape
    assert saved_quat.shape == model.body_quat.shape
    names = [model.body_id2name(i) for i in range(model.nbody)]
    assert len(set(names)) == len(names) and patch_body not in names
    reports = []
    for env in environments:
        target = env.env.sim.model
        target_names = [target.body_id2name(i) for i in range(target.nbody)]
        assert len(set(target_names)) == len(target_names)
        expected_extra = set() if env is clean_env else {patch_body}
        assert set(target_names) == set(names) | expected_extra, 'unexpected model bodies during fixture replay'
        lookup = {name: index for index, name in enumerate(target_names)}
        indices = np.asarray([lookup[name] for name in names], dtype=np.int64)
        patch_before = None
        if expected_extra:
            k = lookup[patch_body]
            patch_before = (target.body_pos[k].copy(), target.body_quat[k].copy())
        position_delta = np.abs(target.body_pos[indices] - saved_pos)
        rotation_delta = np.abs(target.body_quat[indices] - saved_quat)
        target.body_pos[indices] = saved_pos
        target.body_quat[indices] = saved_quat
        assert np.array_equal(target.body_pos[indices], saved_pos)
        assert np.array_equal(target.body_quat[indices], saved_quat)
        if patch_before is not None:
            assert np.array_equal(target.body_pos[k], patch_before[0])
            assert np.array_equal(target.body_quat[k], patch_before[1])
        env.env.sim.forward()
        reports.append(dict(original_bodies=len(names), additional_patch_bodies=len(expected_extra),
                            position_max_change_m=float(position_delta.max()),
                            quaternion_max_change=float(rotation_delta.max()),
                            original_body_names=names, exact=True, patch_pose_preserved=True))
    return reports


def render_normal(env, camera):
    sim = env.env.sim
    ctx = sim._render_context_offscreen
    ctx.gl_ctx.make_current()
    cid = sim.model.camera_name2id(camera)
    ctx.render(256, 256, camera_id=cid, segmentation=False)
    return np.ascontiguousarray(ctx.read_pixels(256, 256)[::-1, ::-1])


def render_unlit(env, camera):
    import mujoco
    from drift_paper_sim import PATCH_NAME
    sim = env.env.sim
    ctx, model = sim._render_context_offscreen, sim.model._model
    ctx.gl_ctx.make_current()
    ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    ctx.cam.fixedcamid = sim.model.camera_name2id(camera)
    mujoco.mjv_updateScene(model, sim.data._data, ctx.vopt, ctx.pert, ctx.cam,
                          mujoco.mjtCatBit.mjCAT_ALL, ctx.scn)
    flags = ctx.scn.flags.copy()
    gid = sim.model.geom_name2id(PATCH_NAME)
    try:
        ctx.scn.nlight = 0
        for flag in ['mjRND_SHADOW', 'mjRND_REFLECTION', 'mjRND_SKYBOX', 'mjRND_HAZE', 'mjRND_FOG']:
            if hasattr(mujoco.mjtRndFlag, flag):
                ctx.scn.flags[getattr(mujoco.mjtRndFlag, flag)] = 0
        for geom in ctx.scn.geoms[:ctx.scn.ngeom]:
            geom.emission = 1. if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == gid else 0.
        mujoco.mjr_render(mujoco.MjrRect(0, 0, 256, 256), ctx.scn, ctx.con)
        return np.ascontiguousarray(ctx.read_pixels(256, 256)[::-1, ::-1])
    finally:
        ctx.scn.flags[:] = flags


def upload(env, texture):
    import mujoco
    from drift_paper_sim import PATCH_NAME
    sim = env.env.sim
    ctx, model = sim._render_context_offscreen, sim.model._model
    ctx.gl_ctx.make_current()
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, PATCH_NAME + '_tex')
    assert int(model.tex_width[tid]) == 64 and int(model.tex_height[tid]) == 64
    adr = int(model.tex_adr[tid])
    # Match the frozen plane_processor UV compensation exactly.
    model.tex_data[adr:adr + 64 * 64 * 3] = np.ascontiguousarray(texture[:, ::-1]).reshape(-1)
    mujoco.mjr_uploadTexture(model, ctx.con, tid)


def segmentation(env, camera):
    import mujoco
    from drift_paper_sim import PATCH_NAME
    sim = env.env.sim
    ctx = sim._render_context_offscreen
    ctx.gl_ctx.make_current()
    gid = sim.model.geom_name2id(PATCH_NAME)
    cid = sim.model.camera_name2id(camera)
    ctx.render(256, 256, camera_id=cid, segmentation=True)
    seg = np.ascontiguousarray(ctx.read_pixels(256, 256, segmentation=True)[::-1, ::-1])
    return (seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 1] == gid)


def capture(args):
    source, root = Path(args.source_root).resolve(), Path(args.root).resolve()
    protocol = prepare(source, root)
    dest = root / 'capture'
    cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    egl = os.environ.get('MUJOCO_EGL_DEVICE_ID', '')
    assert cuda.startswith('GPU-') and ',' not in cuda and egl.isdigit(), 'one assigned CUDA UUID and audited EGL index required'
    from egl_device_audit import audit
    mapping = audit(cuda, int(egl))
    dump(dest / 'egl_mapping.json', mapping)
    bootstrap(sim=True)
    configure_libero(dest)
    import drift_paper_sim as S
    from drift_paper_validate import same_layout
    from libero.libero import benchmark
    pool = np.load(source / 'pool.npy', allow_pickle=True).item()
    geometry = read(source / 'geometry/geometry.json')
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    texture_patterns = patterns()
    records, total, complete = [], 0, True
    for episode in protocol['records']:
        tid, init_id = episode['task_id'], episode['init_index']
        prefix = dest / 'shards' / ('task%02d_init%02d' % (tid, init_id))
        if prefix.with_suffix('.json').exists():
            old = read(prefix.with_suffix('.json'))
            assert old['complete'] and sha(prefix.with_suffix('.npz')) == old['sha256']
            records.append(old)
            total += old['frames']
            continue
        if args.max_frames is not None and total >= args.max_frames:
            complete = False
            break
        task = suite.get_task(tid)
        inits = suite.get_task_init_states(tid)
        environments = [S.make_env(task),
                        S.make_env(task, S.plane_processor(geometry, dest / 'random_texture.png')),
                        S.make_env(task, S.plane_processor(geometry, dest / 'random_texture.png'))]
        clean_env, photo_env, mask_env = environments
        record = np.load(episode['record'], allow_pickle=False)
        arrays = {key: [] for key in ['frame_indices', 'task_ids', 'init_indices', 'query_indices',
                  'clean224', 'black256', 'white256', 'gray256', 'random256', 'ramp256',
                  'unlit_white256', 'mask256', 'uv256', 'H_native_to_texture', 'lod256',
                  'uv_measured256', 'frontface256', 'coverage256',
                  'state8', 'sim_states', 'noise_seeds', 'prompt']}
        checks = []
        try:
            for env in environments:
                env.env.sim._render_context_offscreen.gl_ctx.make_current()
                S.reset(env, inits[init_id], episode['env_seed'])
            clean_pos, clean_quat = record['fixture_body_pos'], record['fixture_body_quat']
            fixture_replay = restore_fixture_layout(clean_env, environments, clean_pos, clean_quat,
                                                    S.PATCH_NAME + '_body')
            for env in environments:
                assert same_layout(clean_pos, clean_quat, env.env.sim.model.body_pos,
                                   env.env.sim.model.body_quat, env is not clean_env), 'fixture layout changed'
            for query, state in enumerate(record['sim_states']):
                index = episode['offset'] + query
                assert pool['prompts'][index] == episode['prompt']
                clean_env.env.sim._render_context_offscreen.gl_ctx.make_current()
                obs = clean_env.regenerate_obs_from_state(state)
                clean = np.stack(S.inputs(obs))
                expected = np.stack([pool['observation/image'][index], pool['observation/wrist_image'][index]])
                clean_diff = difference(clean, expected)
                if not clean_diff['equal']:
                    np.savez_compressed(dest / ('source_rgb_error_frame%04d.npz' % index),
                                        expected=expected, actual=clean, sim_state=state)
                assert clean_diff['equal'], ('source clean RGB replay mismatch', index, clean_diff)
                assert np.array_equal(clean_env.get_sim_state(), state)
                for env in [photo_env, mask_env]:
                    set_state(env, state)
                masks = np.stack([segmentation(mask_env, camera) for camera in CAMERAS])
                values = {}
                for name in ['black', 'white', 'gray', 'random']:
                    upload(photo_env, texture_patterns[name])
                    values[name + '256'] = np.stack([render_normal(photo_env, camera) for camera in CAMERAS])
                    assert np.array_equal(photo_env.get_sim_state(), state), ('render changed physics', name, index)
                upload(photo_env, texture_patterns['white'])
                values['unlit_white256'] = np.stack([render_unlit(photo_env, camera) for camera in CAMERAS])
                upload(photo_env, texture_patterns['ramp'])
                values['ramp256'] = np.stack([render_unlit(photo_env, camera) for camera in CAMERAS])
                upload(photo_env, texture_patterns['black'])
                repeat = np.stack([render_normal(photo_env, camera) for camera in CAMERAS])
                repeat_diff = difference(values['black256'], repeat)
                if not repeat_diff['equal']:
                    np.savez_compressed(dest / ('repeat_error_frame%04d.npz' % index),
                                        before=values['black256'], after=repeat, sim_state=state)
                assert repeat_diff['equal'], ('unlit calibration changed normal RGB', index, repeat_diff)
                assert np.array_equal(photo_env.get_sim_state(), state)
                denominator = values['unlit_white256'][..., :2].astype(np.float32)
                ratio = values['ramp256'][..., :2].astype(np.float32) / np.maximum(denominator, 1)
                measured_uv = np.empty(ratio.shape, np.float32)
                ramp_values = texture_patterns['ramp'][0, :, 0].astype(np.float32) / 255
                for dim in range(2):
                    measured_uv[..., dim] = np.interp(np.clip(ratio[..., dim], 0, 1), ramp_values,
                                                      np.arange(64)).astype(np.float32)
                measured_uv[~masks] = 0
                H = np.stack([homography(photo_env.env.sim, geometry, camera) for camera in CAMERAS])
                lod = np.stack([lod_from_homography(h) for h in H])
                yy, xx = np.mgrid[:256, :256]
                pixel = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
                uv = measured_uv.copy()
                frontface = np.zeros(masks.shape, bool)
                uv_audit = []
                sim = photo_env.env.sim
                gid = sim.model.geom_name2id(S.PATCH_NAME)
                normal = np.asarray(sim.data.geom_xmat[gid]).reshape(3, 3)[:, 2]
                for view, camera in enumerate(CAMERAS):
                    projected = pixel @ H[view].T
                    analytic = projected[..., :2] / projected[..., 2:]
                    cid = sim.model.camera_name2id(camera)
                    facing = float(normal @ (sim.data.cam_xpos[cid] - sim.data.geom_xpos[gid])) > 0
                    front = masks[view] & facing & (analytic[..., 0] >= -.5) & (analytic[..., 0] <= 63.5)
                    front &= (analytic[..., 1] >= -.5) & (analytic[..., 1] <= 63.5)
                    frontface[view] = front
                    uv[view, front] = analytic[front].astype(np.float32)
                    interior = front & (analytic[..., 0] >= 2) & (analytic[..., 0] <= 61)
                    interior &= (analytic[..., 1] >= 2) & (analytic[..., 1] <= 61)
                    interior &= values['unlit_white256'][view].min(axis=-1) >= 254
                    error = float(np.sqrt(((analytic[interior] - measured_uv[view, interior]) ** 2).mean())) if interior.any() else None
                    uv_audit.append(dict(view=camera, frontface_pixels=int(front.sum()),
                                         uv_interior_pixels=int(interior.sum()),
                                         analytic_ramp_RMSE_texels=error))
                coverage = values['unlit_white256'][..., 2].astype(np.float32) / 255
                frame = dict(frame_indices=index, task_ids=tid, init_indices=init_id, query_indices=query,
                             clean224=clean, mask256=masks, uv256=uv, H_native_to_texture=H, lod256=lod,
                             uv_measured256=measured_uv, frontface256=frontface, coverage256=coverage,
                             state8=record['state8'][query], sim_states=state,
                             noise_seeds=record['noise_seeds'][query], prompt=episode['prompt'], **values)
                for key in arrays:
                    arrays[key].append(frame[key])
                checks.append(dict(frame_index=index, clean_source_exact=True, repeat_lit_rgb_exact=True,
                                   simulator_state_exact=True,
                                   visible_native=[int(mask.sum()) for mask in masks], uv_audit=uv_audit))
                total += 1
                if query % 10 == 0:
                    dump(dest / 'progress.json', dict(stage='capturing', pid=os.getpid(), updated=now(),
                         completed_frames=total, target_frames=protocol['frames'], task_id=tid,
                         init_index=init_id, query_index=query))
                    print('[capture]', total, '/', protocol['frames'], tid, init_id, query, flush=True)
            values = {key: np.asarray(value) for key, value in arrays.items()}
            tmp = prefix.with_name(prefix.name + '.tmp.npz')
            np.savez_compressed(tmp, **values)
            tmp.replace(prefix.with_suffix('.npz'))
            meta = dict(complete=True, task_id=tid, init_index=init_id, frames=len(values['frame_indices']),
                        offset=episode['offset'], file=str(prefix.with_suffix('.npz')),
                        sha256=sha(prefix.with_suffix('.npz')), source_record_sha256=episode['record_sha256'],
                        checks=checks, fixture_replay=fixture_replay)
            dump(prefix.with_suffix('.json'), meta)
            records.append(meta)
        finally:
            record.close()
            for env in reversed(environments):
                env.env.sim._render_context_offscreen.gl_ctx.make_current()
                env.close()
    complete = complete and total == protocol['frames'] and len(records) == len(protocol['records'])
    manifest = dict(complete=complete, schema=protocol['schema'], updated=now(),
                    frames=total, target_frames=protocol['frames'], records=records,
                    protocol_sha256=sha(dest / 'protocol.json'), geometry_sha256=protocol['geometry_sha256'],
                    script_sha256=protocol['script_sha256'], random_texture_sha256=sha(dest / 'random_texture.png'),
                    cuda_visible_devices=cuda, native_egl_index=int(egl),
                    physics_exact=True, fixture_layout_exact=True, clean_source_rgb_exact=True,
                    normal_rgb_repeat_exact=True, segmentation_isolated=True)
    dump(dest / 'manifest.json', manifest)
    dump(dest / 'progress.json', dict(stage='complete' if complete else 'partial', pid=os.getpid(),
                                    updated=now(), frames=total, target_frames=protocol['frames']))
    if complete:
        dump(dest / 'complete.json', dict(complete=True, frames=total, manifest_sha256=sha(dest / 'manifest.json')))
    print(json.dumps(dict(complete=complete, frames=total)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--max-frames', type=int)
    args = p.parse_args()
    dest = Path(args.root) / 'capture'
    dest.mkdir(parents=True, exist_ok=True)
    try:
        with (dest / 'capture.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if args.prepare_only:
                print(json.dumps(prepare(args.source_root, args.root)))
            else:
                capture(args)
    except BaseException as error:
        dump(dest / 'error.json', dict(updated=now(), error=repr(error), pid=os.getpid()))
        raise
