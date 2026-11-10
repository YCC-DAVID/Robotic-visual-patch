"""Calibrate a fixed physical Q128 renderer using ramps and black/white.

No optimized texture is used to fit geometry or photometry. The input pool
must preserve simulator states. E15 mode supplies existing paired anchors.
"""
import argparse
import hashlib
import json
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-render-world-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='render_proxy_eot/calibration_e15')
    ap.add_argument('--pool', default='', help='NPZ with flat_states, clean_base/wrist, state8, steps, env_seed')
    args = ap.parse_args()
    import cv2, mujoco, yaml
    from PIL import Image
    from OpenGL import GL
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb
    import scene_patch as sp
    dest = D.OUT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    patch = np.load(D.OUT / 'drift_patch_drift.npz', allow_pickle=True)
    old = np.load(D.OUT / 'kv_render_probe/inputs.npz', allow_pickle=False)
    if args.pool:
        pool = np.load(args.pool, allow_pickle=False)
        states = pool['sim_states'] if 'sim_states' in pool.files else pool['flat_states']
        seed = int(pool['env_seed'].flat[0])
        arrays = {k: pool[k] for k in pool.files if k != 'flat_states'}
        aliases = {'clean_base': 'clean_img224', 'clean_wrist': 'clean_wrist224',
                   'state8': 'clean_state8', 'steps': 'env_step', 'mask': 'masks'}
        for target, original in aliases.items():
            if target not in arrays and original in arrays:
                arrays[target] = arrays[original]
        source = args.pool
    else:
        trajectory = D.OUT / ('traj_' + D.TASK + '.npz')
        traj = np.load(trajectory, allow_pickle=False)
        states = [traj['f{:03d}__flatten'.format(traj['ts'].tolist().index(s))] for s in old['steps']]
        seed = int(traj['shared_seed'])
        arrays = {k: old[k] for k in ['steps', 'state8', 'prompt', 'clean_base', 'clean_wrist']}
        arrays['mask'] = old['drift__mask']
        source = str(trajectory)
    cfg = yaml.safe_load((D.PROBE / 'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(patch['patch_m'])]*2
    H = cv2.getPerspectiveTransform(patch['quad224'].astype(np.float32)[:, ::-1],
                                    np.array([[0, 0], [127, 0], [127, 127], [0, 127]], np.float32))
    random = np.random.RandomState(20260902).randint(0, 256, (224, 224, 3)).astype(np.uint8)
    patterns = {'random': random}
    for name, tag in [('full', 'drift'), ('band32', 'band32')]:
        data = np.load(D.OUT / ('drift_patch_' + tag + '.npz'), allow_pickle=True)
        patterns[name] = np.clip(data['P'].transpose(1, 2, 0)*255, 0, 255).astype(np.uint8)
    textures = {k: cv2.warpPerspective(v, H, (128, 128))[::-1].copy() for k, v in patterns.items()}
    for k, v in textures.items():
        arrays[k + '__texture'] = v
        arrays[k + '__pattern'] = patterns[k]
    x = np.rint(np.linspace(0, 255, 128)).astype(np.uint8)
    ramp = np.empty((128, 128, 3), np.uint8)
    ramp[:, :, 0] = x[None, :]
    ramp[:, :, 1] = x[:, None]
    ramp[:, :, 2] = 255
    texpath = dest / 'initial_texture.png'
    Image.fromarray(textures['random']).save(texpath)
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    bddl = D.pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg, patch['cell_world'], str(texpath)))
    captures = {k: [] for k in ['black256', 'white256', 'unlit_white256', 'ramp256',
                               'gray128_256', 'random256', 'full256', 'band32_256']}
    checks = {'old_random_anchors_equal': None if args.pool else True, 'repeat_equal': True,
              'far_background_matches_pool': True, 'far_background_max_u8_error': 0}
    try:
        env.seed(seed); env.reset()
        sim, ctx = env.env.sim, env.env.sim._render_context_offscreen
        model = sim.model._model
        ctx.gl_ctx.make_current()
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, cfg['patch']['name'] + '_tex')
        gid = sim.model.geom_name2id(cfg['patch']['name'])
        adr, length = int(model.tex_adr[tid]), 128*128*3
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(ctx.con.texture[tid]))
        runtime = {'mujoco_version': mujoco.__version__, 'offSamples': int(ctx.con.offSamples),
                   'texture_min_filter': int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER)),
                   'texture_mag_filter': int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER)),
                   'framebuffer_srgb': bool(GL.glIsEnabled(GL.GL_FRAMEBUFFER_SRGB))}
        assert runtime['offSamples'] == 4
        def upload(texture):
            model.tex_data[adr:adr+length] = texture.reshape(-1)
            mujoco.mjr_uploadTexture(model, ctx.con, tid)
        def draw(unlit=False):
            ctx.gl_ctx.make_current()
            ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            ctx.cam.fixedcamid = sim.model.camera_name2id('agentview')
            mujoco.mjv_updateScene(model, sim.data._data, ctx.vopt, ctx.pert, ctx.cam,
                                  mujoco.mjtCatBit.mjCAT_ALL, ctx.scn)
            flags = ctx.scn.flags.copy()
            if unlit:
                ctx.scn.nlight = 0
                for flag in ['mjRND_SHADOW', 'mjRND_REFLECTION', 'mjRND_SKYBOX', 'mjRND_HAZE', 'mjRND_FOG']:
                    if hasattr(mujoco.mjtRndFlag, flag):
                        ctx.scn.flags[getattr(mujoco.mjtRndFlag, flag)] = 0
                for g in ctx.scn.geoms[:ctx.scn.ngeom]:
                    g.emission = 1. if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == gid else 0.
            mujoco.mjr_render(mujoco.MjrRect(0, 0, 256, 256), ctx.scn, ctx.con)
            raw = ctx.read_pixels(256, 256)
            ctx.scn.flags[:] = flags
            return np.ascontiguousarray(raw[::-1, ::-1])
        for k, state in enumerate(states):
            if args.pool and k > 0 and int(arrays['env_seed'][k]) != int(arrays['env_seed'][k-1]):
                # A hard reset can replace the model and EGL context. Release
                # old external references before constructing the next one.
                del ctx, sim, model
                env.seed(int(arrays['env_seed'][k])); env.reset()
                sim, ctx = env.env.sim, env.env.sim._render_context_offscreen
                model = sim.model._model; ctx.gl_ctx.make_current()
                tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, cfg['patch']['name'] + '_tex')
                gid = sim.model.geom_name2id(cfg['patch']['name'])
                adr = int(model.tex_adr[tid])
            env.regenerate_obs_from_state(state)
            for name, texture in textures.items():
                upload(texture); image = draw()
                key = 'band32_256' if name == 'band32' else name + '256'
                captures[key].append(image)
                if not args.pool:
                    sourcekey = 'drift__random_render_base' if name == 'random' else (
                        'band32' if name == 'band32' else 'drift') + '__optimized_render_base'
                    assert np.array_equal(resize_rgb(image, 224), old[sourcekey][k]), (name, k)
            for name, value, unlit in [('black256', 0, False), ('white256', 255, False),
                                       ('unlit_white256', 255, True), ('gray128_256', 128, False)]:
                upload(np.full((128, 128, 3), value, np.uint8))
                image = draw(unlit); captures[name].append(image)
                if name == 'white256' and k == 0:
                    assert np.array_equal(image, draw(unlit))
            upload(ramp); captures['ramp256'].append(draw(True))
            far = ~cv2.dilate(arrays['mask'][k].astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
            black224 = resize_rgb(captures['black256'][-1], 224)
            error = int(np.abs(black224.astype(int)-arrays['clean_base'][k].astype(int))[far].max())
            checks['far_background_max_u8_error'] = max(checks['far_background_max_u8_error'], error)
            assert error == 0, ('new pool replay/background mismatch', k, error)
            if k % 20 == 0:
                print('[calibration]', k, '/', len(states), flush=True)
        del draw, upload, ctx, sim, model
    finally:
        env.close()
    arrays.update({k: np.array(v) for k, v in captures.items()})
    arrays.update(quad224=patch['quad224'], cell_world=patch['cell_world'],
                  cell_xy=patch['cell_xy'], patch_m=patch['patch_m'])
    # Use only ramp calibration, not optimized texture outputs, for geometry.
    xy_in, xy_tex = [], []
    for k in range(min(8, len(states))):
        white = arrays['unlit_white256'][k]
        valid = cv2.erode((white.min(-1) == 255).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        # Restrict to the injected patch's support; other emission-free scene
        # elements are black in the unit-albedo pass.
        y, xx = np.where(valid)
        assert len(y) > 500
        uv = arrays['ramp256'][k][valid, :2].astype(float)
        uv = np.stack([np.interp(uv[:, j], x, np.arange(128)) for j in range(2)], -1)
        xy_in.append(np.stack([xx, y], -1).astype(float))
        xy_tex.append(uv)
    a, b = np.concatenate(xy_in), np.concatenate(xy_tex)
    fitH, _ = cv2.findHomography(a, b, method=0)
    ph = np.concatenate([a, np.ones((len(a), 1))], -1) @ fitH.T
    residual = ph[:, :2]/ph[:, 2:] - b
    arrays['image256_to_texture_pixel_H'] = fitH
    np.savez_compressed(dest / 'calibration.npz', **arrays)
    meta = {'complete': True, 'frames': len(states), 'runtime': runtime, 'checks': checks,
            'ramp_homography_RMSE_texture_pixels': float(np.sqrt((residual**2).mean())),
            'geometry_fit_frames': min(8, len(states)), 'source': source,
            'source_sha256': hashlib.sha256(D.pathlib.Path(source).read_bytes()).hexdigest(),
            'notes': 'Q is physical texture PNG orientation. UV geometry fitted only to known ramps. '
                     'Black/white per-state photometry and boundary support are real renderer captures. '
                     'Gray128 tests affine response. No optimized texture is used for calibration fitting. '
                     'E15 mode reproduces old full/band32/random anchors exactly.'}
    (dest / 'capture.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
    print('[done]', json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
