"""Fixed E15 geometry: colored checks through export, sampling, resize and light."""
import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-color-checker-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='color_checker_probe')
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
    assert not (dest / 'inputs.npz').exists(), 'Use a fresh experiment directory'
    old = np.load(D.OUT / 'kv_render_probe/inputs.npz', allow_pickle=False)
    trajectory = D.OUT / ('traj_' + D.TASK + '.npz')
    traj = np.load(trajectory, allow_pickle=False)
    states = np.array([traj['f{:03d}__flatten'.format(traj['ts'].tolist().index(s))]
                       for s in old['steps']])
    patch = np.load(D.OUT / 'drift_patch_drift.npz', allow_pickle=False)
    cfg = yaml.safe_load((D.PROBE / 'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(patch['patch_m'])] * 2
    H = cv2.getPerspectiveTransform(patch['quad224'].astype(np.float32)[:, ::-1],
                                    np.array([[0, 0], [127, 0], [127, 127], [0, 127]], np.float32))
    palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255],
                        [255, 255, 0], [255, 0, 255]], np.uint8)
    yy, xx = np.mgrid[:224, :224]
    patterns = {'cell{}'.format(s): palette[(yy // s + 2 * (xx // s)) % 5]
                for s in [2, 4, 8, 16]}
    textures = {name: {mode: cv2.warpPerspective(p, H, (128, 128), flags=flag)[::-1].copy()
                      for mode, flag in [('linear', cv2.INTER_LINEAR), ('nearest', cv2.INTER_NEAREST)]}
                for name, p in patterns.items()}
    arrays = {key: old[key] for key in ['steps', 'state8', 'prompt', 'clean_base', 'clean_wrist']}
    arrays.update(mask=old['drift__mask'], palette=palette, patterns=np.array(list(patterns)),
                  sim_states=states, environment_seed=np.array(traj['shared_seed']),
                  patch_m=patch['patch_m'], cell_world=patch['cell_world'],
                  source224_to_export128_H=H)
    arrays['inner_mask'] = np.array([cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
                                    for m in arrays['mask']])
    calibration_path = D.OUT / 'render_proxy_eot/calibration_e15/calibration.npz'
    calibration = np.load(calibration_path, allow_pickle=False)
    arrays['image256_to_texture_pixel_H'] = calibration['image256_to_texture_pixel_H']
    assert np.array_equal(calibration['steps'], arrays['steps'])
    conditions = ['digital', 'digital_resize256_224', 'export_roundtrip_linear',
                  'export_roundtrip_nearest', 'minimal_linear_export', 'minimal_nearest_export',
                  'unlit_nomipmap', 'unlit_default', 'render']
    arrays['conditions'] = np.array(conditions)
    for name, p in patterns.items():
        arrays[name + '__pattern'] = p
        for mode, q in textures[name].items():
            arrays[name + '__texture_' + mode] = q
            Image.fromarray(q).save(dest / (name + '_texture_' + mode + '.png'))
        digital = {'digital': p, 'digital_resize256_224': resize_rgb(resize_rgb(p, 256), 224)}
        for mode, flag in [('linear', cv2.INTER_LINEAR), ('nearest', cv2.INTER_NEAREST)]:
            digital['export_roundtrip_' + mode] = cv2.warpPerspective(
                textures[name][mode][::-1].copy(), np.linalg.inv(H), (224, 224), flags=flag)
        for c, p224 in digital.items():
            arrays[name + '__' + c] = np.where(arrays['mask'][..., None], p224, arrays['clean_base'])
    texpath = dest / 'cell8_texture_linear.png'
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    bddl = D.pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    runtime = {}
    checks = {'all16_original_random_render_anchors_equal': True,
              'repeat_render_equal': True, 'unlit_white_interior_equal255': True,
              'identical_fixture_arrays_across_AA': True, 'same_composition_mask_and_background': True}
    fixture_reference = None
    for aa in [0, 4]:
        inject = sp.make_xml_processor(cfg, patch['cell_world'], str(texpath))
        def xml_processor(xml):
            root = ET.fromstring(inject(xml))
            visual = root.find('visual')
            if visual is None: visual = ET.SubElement(root, 'visual')
            quality = visual.find('quality')
            if quality is None: quality = ET.SubElement(visual, 'quality')
            quality.set('offsamples', str(aa))
            return ET.tostring(root, encoding='unicode')
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.env.set_xml_processor(xml_processor)
        try:
            env.seed(int(traj['shared_seed'])); env.reset()
            model, sim = env.env.sim.model._model, env.env.sim
            ctx = sim._render_context_offscreen
            ctx.gl_ctx.make_current()
            assert int(ctx.con.offSamples) == aa
            fixture = np.concatenate([np.asarray(model.body_pos).flatten(), np.asarray(model.body_quat).flatten()])
            if fixture_reference is None: fixture_reference = fixture
            else: assert np.array_equal(fixture, fixture_reference)
            arrays['fixture_pose_aa' + str(aa)] = fixture
            tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, cfg['patch']['name'] + '_tex')
            gid = sim.model.geom_name2id(cfg['patch']['name'])
            adr = int(model.tex_adr[tid]); glid = int(ctx.con.texture[tid])
            GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, glid)
            default_min = int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER))
            default_mag = int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER))
            runtime[str(aa)] = {'mujoco': mujoco.__version__, 'offSamples': aa,
                               'min_filter': default_min, 'mag_filter': default_mag,
                               'framebuffer_srgb': bool(GL.glIsEnabled(GL.GL_FRAMEBUFFER_SRGB))}
            def upload(q):
                model.tex_data[adr:adr + 128 * 128 * 3] = q.reshape(-1)
                mujoco.mjr_uploadTexture(model, ctx.con, tid)
            def draw(size=256, unlit=False, mode='default'):
                ctx.gl_ctx.make_current()
                GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, glid)
                mini = {'default': default_min, 'nearest': GL.GL_NEAREST, 'linear': GL.GL_LINEAR}[mode]
                mag = GL.GL_NEAREST if mode == 'nearest' else default_mag
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mini)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, mag)
                assert int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER)) == mini
                ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                ctx.cam.fixedcamid = sim.model.camera_name2id('agentview')
                mujoco.mjv_updateScene(model, sim.data._data, ctx.vopt, ctx.pert, ctx.cam,
                                      mujoco.mjtCatBit.mjCAT_ALL, ctx.scn)
                flags = ctx.scn.flags.copy()
                if unlit:
                    ctx.scn.nlight = 0
                    for f in ['mjRND_SHADOW', 'mjRND_REFLECTION', 'mjRND_SKYBOX', 'mjRND_HAZE', 'mjRND_FOG']:
                        if hasattr(mujoco.mjtRndFlag, f): ctx.scn.flags[getattr(mujoco.mjtRndFlag, f)] = 0
                    for g in ctx.scn.geoms[:ctx.scn.ngeom]:
                        g.emission = float(g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == gid)
                mujoco.mjr_render(mujoco.MjrRect(0, 0, size, size), ctx.scn, ctx.con)
                im = np.ascontiguousarray(ctx.read_pixels(size, size)[::-1, ::-1])
                ctx.scn.flags[:] = flags
                return im
            captures = {name: {} for name in patterns}
            for k, state in enumerate(states):
                env.regenerate_obs_from_state(state)
                if aa == 4:
                    upload(calibration['random__texture'])
                    assert np.array_equal(resize_rgb(draw(), 224), old['drift__random_render_base'][k])
                upload(np.full((128, 128, 3), 255, np.uint8))
                white = draw(224 if aa == 0 else 256, True, 'nearest' if aa == 0 else 'default')
                assert resize_rgb(white, 224)[arrays['inner_mask'][k]].min() == 255
                for name in patterns:
                    variants = [('minimal_' + m + '_export', m, 224, True, 'nearest')
                                for m in ['linear', 'nearest']] if aa == 0 else [
                                    ('unlit_nomipmap', 'linear', 256, True, 'linear'),
                                    ('unlit_default', 'linear', 256, True, 'default'),
                                    ('render', 'linear', 256, False, 'default')]
                    for c, export, size, unlit, filt in variants:
                        upload(textures[name][export]); raw = draw(size, unlit, filt)
                        if k == 0: assert np.array_equal(raw, draw(size, unlit, filt))
                        captures[name].setdefault(c, []).append(raw)
                print('[capture] AA', aa, 'step', int(arrays['steps'][k]), flush=True)
            for name, cs in captures.items():
                for c, raw in cs.items():
                    arrays[name + '__' + c + '__native'] = np.array(raw)
                    p224 = np.array([resize_rgb(im, 224) for im in raw])
                    arrays[name + '__' + c] = np.where(arrays['mask'][..., None], p224, arrays['clean_base'])
            del upload, draw, ctx, sim, model
        finally:
            env.close()
    for name in patterns:
        for c in conditions:
            assert np.array_equal(arrays[name + '__' + c][~arrays['mask']], arrays['clean_base'][~arrays['mask']])
    np.savez_compressed(dest / 'inputs.npz', **arrays)
    meta = {'complete': True, 'frames': len(states), 'patterns': list(patterns), 'conditions': conditions,
            'cell_widths_in_source224_pixels': [2, 4, 8, 16], 'palette': palette.tolist(), 'runtime': runtime,
            'checks': checks, 'environment_seed': int(traj['shared_seed']),
            'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                trajectory, D.OUT / 'kv_render_probe/inputs.npz', calibration_path]},
            'notes': 'Five-color asymmetric (row+2*col)%5 tile pattern, source224, original warp exportQ128. '
                     'Fixed26cm and world position; same E15 composition mask/clean background, base-only. '
                     'Cell size changes texture frequency only. Minimal=unlit nearest AA0 native224. '
                     'NN export control changes only export interpolation. No training or closed loop.'}
    (dest / 'capture.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    print('[done]', dest, flush=True)


if __name__ == '__main__':
    main()
