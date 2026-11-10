"""Independent real-render check: replace only original export geometry with frozen UV calibration."""
import argparse
import json
import os
import xml.etree.ElementTree as ET
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-checker-align-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='Apply identical fixed-ramp geometry to historical full')
    args = ap.parse_args()
    import cv2, mujoco, yaml
    from PIL import Image
    from OpenGL import GL
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb
    import scene_patch as sp
    grid_dest = D.OUT / 'color_checker_probe'
    old = np.load(grid_dest / 'inputs.npz', allow_pickle=False)
    dest = D.OUT / ('color_checker_full_alignment' if args.full else 'color_checker_probe')
    dest.mkdir(exist_ok=True)
    filename = 'inputs.npz' if args.full else 'alignment_inputs.npz'
    assert not (dest / filename).exists()
    if args.full:
        old = {k: old[k] for k in old.files}
        patch = np.load(D.OUT / 'drift_patch_drift.npz', allow_pickle=False)
        p = np.clip(patch['P'].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        source = np.load(D.OUT / 'kv_render_probe/inputs.npz', allow_pickle=False)
        nearest = np.load(D.OUT / 'render_additive_nearest_export/inputs.npz', allow_pickle=False)
        old['patterns'] = np.array(['full'])
        old['full__pattern'] = p
        old['full__digital'] = np.where(old['mask'][..., None], p, old['clean_base'])
        assert np.array_equal(old['full__digital'], nearest['full__digital'])
        old['full__minimal_nearest_export'] = nearest['full__unlit_nearest_aa0_native224']
        old['full__render'] = np.where(old['mask'][..., None], source['drift__optimized_render_base'], old['clean_base'])
        for mode, flag in [('nearest', cv2.INTER_NEAREST), ('linear', cv2.INTER_LINEAR)]:
            old['full__texture_' + mode] = cv2.warpPerspective(p, old['source224_to_export128_H'], (128, 128), flags=flag)[::-1].copy()
    scale = 256 / 224
    A = np.array([[scale, 0, (scale - 1) / 2], [0, scale, (scale - 1) / 2], [0, 0, 1]])
    H = old['image256_to_texture_pixel_H'] @ A
    textures = {name: {mode: cv2.warpPerspective(old[name + '__pattern'], H, (128, 128), flags=flag)
                      for mode, flag in [('nearest', cv2.INTER_NEAREST), ('linear', cv2.INTER_LINEAR)]}
                for name in old['patterns']}
    arrays = {'calibrated_source224_to_PNG128_H': H, 'steps': old['steps']}
    if args.full:
        arrays.update({k: old[k] for k in ['steps', 'state8', 'prompt', 'clean_base', 'clean_wrist', 'mask', 'inner_mask', 'patterns']})
        arrays['conditions'] = np.array(['digital', 'old_minimal', 'calibrated_minimal', 'render', 'calibrated_render'])
        arrays['full__digital'] = old['full__digital']
        arrays['full__old_minimal'] = old['full__minimal_nearest_export']
        arrays['full__render'] = old['full__render']
    for name, cs in textures.items():
        for mode, q in cs.items():
            arrays[name + '__calibrated_texture_' + mode] = q
    texpath = dest / 'alignment_initial_texture.png'
    anchor_name = 'full' if args.full else 'cell8'
    Image.fromarray(textures[anchor_name]['nearest']).save(texpath)
    cfg = yaml.safe_load((D.PROBE / 'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(old['patch_m'])] * 2
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    bddl = D.pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    checks = {'all16_original_anchor_replay_equal': True, 'native_anchor_check': not args.full,
              'composited_anchor_check': args.full, 'fixture_arrays_equal_main': True}
    for aa in [0, 4]:
        inject = sp.make_xml_processor(cfg, old['cell_world'], str(texpath))
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
            env.seed(int(old['environment_seed'])); env.reset()
            model, sim = env.env.sim.model._model, env.env.sim
            ctx = sim._render_context_offscreen; ctx.gl_ctx.make_current()
            fixture = np.concatenate([np.asarray(model.body_pos).flatten(), np.asarray(model.body_quat).flatten()])
            assert np.array_equal(fixture, old['fixture_pose_aa' + str(aa)])
            tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, cfg['patch']['name'] + '_tex')
            gid = sim.model.geom_name2id(cfg['patch']['name'])
            adr = int(model.tex_adr[tid]); glid = int(ctx.con.texture[tid])
            def upload(q):
                model.tex_data[adr:adr + q.size] = q.reshape(-1)
                mujoco.mjr_uploadTexture(model, ctx.con, tid)
                if aa == 0:
                    GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, glid)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
                    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            def draw():
                size = 224 if aa == 0 else 256
                ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                ctx.cam.fixedcamid = sim.model.camera_name2id('agentview')
                mujoco.mjv_updateScene(model, sim.data._data, ctx.vopt, ctx.pert, ctx.cam,
                                      mujoco.mjtCatBit.mjCAT_ALL, ctx.scn)
                flags = ctx.scn.flags.copy()
                if aa == 0:
                    ctx.scn.nlight = 0
                    for f in ['mjRND_SHADOW', 'mjRND_REFLECTION', 'mjRND_SKYBOX', 'mjRND_HAZE', 'mjRND_FOG']:
                        if hasattr(mujoco.mjtRndFlag, f): ctx.scn.flags[getattr(mujoco.mjtRndFlag, f)] = 0
                    for g in ctx.scn.geoms[:ctx.scn.ngeom]:
                        g.emission = float(g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == gid)
                mujoco.mjr_render(mujoco.MjrRect(0, 0, size, size), ctx.scn, ctx.con)
                im = np.ascontiguousarray(ctx.read_pixels(size, size)[::-1, ::-1])
                ctx.scn.flags[:] = flags
                return im
            mode = 'nearest' if aa == 0 else 'linear'
            c = 'calibrated_minimal' if aa == 0 else 'calibrated_render'
            anchor = 'minimal_nearest_export' if aa == 0 else 'render'
            outputs = {name: [] for name in old['patterns']}
            for k, state in enumerate(old['sim_states']):
                env.regenerate_obs_from_state(state)
                upload(old[anchor_name + '__texture_' + mode])
                anchor_image = draw()
                if args.full:
                    im = np.where(old['mask'][k, ..., None], resize_rgb(anchor_image, 224), old['clean_base'][k])
                    assert np.array_equal(im, old['full__' + anchor][k])
                else:
                    assert np.array_equal(anchor_image, old['cell8__' + anchor + '__native'][k])
                for name in old['patterns']:
                    upload(textures[name][mode]); outputs[name].append(draw())
            for name, raw in outputs.items():
                arrays[name + '__' + c + '__native'] = np.array(raw)
                images = np.array([resize_rgb(im, 224) for im in raw])
                arrays[name + '__' + c] = np.where(old['mask'][..., None], images, old['clean_base'])
            del upload, draw, ctx, sim, model
        finally: env.close()
        print('[alignment] AA', aa, 'complete', flush=True)
    np.savez_compressed(dest / filename, **arrays)
    (dest / 'alignment_capture.json').write_text(json.dumps({'complete': True, 'checks': checks,
        'notes': 'Real MuJoCo, no chessboard-based fitting. Frozen historical rampH only. '
                 'Minimal comparison keeps nearest export/NN/AA0/unlit/native224, changing exportH only. '
                 'Production comparison keeps linear export/mipmap/AA4/lit/256-to224, changing exportH only. '
                 'Grid is an image-only calibration control. The separate full control uses the same fixed mapping; '
                 'paired model diagnosis is a separate process. No training or closed-loop claims.'}, indent=2) + '\n')
    print('[done]', dest)


if __name__ == '__main__':
    main()
