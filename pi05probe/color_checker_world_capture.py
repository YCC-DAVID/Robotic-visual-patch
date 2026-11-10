"""World-UV checker control: one Q128, forward-projected and truly rendered.

Unlike E23's screen-space export roundtrip, the regular checks are defined
directly in PNG/texture coordinates. No image-to-texture export is applied.
The digital reference uses the frozen, independently measured UV homography;
the checker outputs are never used to fit that mapping. Pixel controls only.
"""
import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET

os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-world-checker-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='color_checker_world')
    args = ap.parse_args()
    import cv2
    import mujoco
    import yaml
    from PIL import Image
    from OpenGL import GL
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb
    import scene_patch as sp

    dest = D.OUT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'inputs.npz').exists(), 'Use a fresh experiment directory'
    old_path = D.OUT / 'color_checker_probe/inputs.npz'
    cal_path = D.OUT / 'render_proxy_eot/calibration_e15/calibration.npz'
    old = np.load(old_path, allow_pickle=False)
    cal = np.load(cal_path, allow_pickle=False)
    assert np.array_equal(old['steps'], cal['steps'])
    H256 = cal['image256_to_texture_pixel_H']
    scale = 256. / 224.
    A = np.array([[scale, 0, (scale - 1) / 2],
                  [0, scale, (scale - 1) / 2], [0, 0, 1]])
    H224 = H256 @ A
    # Q pixel centers -> image pixel centers, including perspective division.
    forward224, forward256 = np.linalg.inv(H224), np.linalg.inv(H256)
    assert np.linalg.norm(forward224[2, :2]) > 1e-6, 'Must include perspective'
    palette = old['palette']
    yy, xx = np.mgrid[:128, :128]
    sizes = [2, 4, 8, 16]
    textures = {'uvcell{}'.format(s): palette[(yy // s + 2 * (xx // s)) % 5]
                for s in sizes}
    arrays = {k: old[k] for k in ['steps', 'state8', 'prompt', 'clean_base',
                                 'clean_wrist', 'mask', 'inner_mask',
                                 'sim_states', 'environment_seed', 'cell_world', 'patch_m']}
    arrays.update(patterns=np.array(list(textures)), palette=palette,
                  source_Q128_cell_sizes=np.array(sizes),
                  image256_to_Q128_H=H256, image224_to_Q128_H=H224,
                  Q128_to_image224_H=forward224)
    # Texel edges, not centers, delimit the physical plane.
    texcorners = np.array([[-.5, -.5], [127.5, -.5], [127.5, 127.5], [-.5, 127.5]])
    points = np.concatenate([texcorners, np.ones((4, 1))], axis=1) @ forward224.T
    arrays['projected_quad224_xy'] = points[:, :2] / points[:, 2:]
    for name, q in textures.items():
        assert q.shape == (128, 128, 3) and q.dtype == np.uint8
        arrays[name + '__Q128'] = q
        Image.fromarray(q).save(dest / (name + '_Q128.png'))
        projected224 = cv2.warpPerspective(q, forward224, (224, 224),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_REPLICATE)
        projected256 = cv2.warpPerspective(q, forward256, (256, 256),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_REPLICATE)
        wrong_cut = resize_rgb(q, 224)
        for condition, image in [('screen_cut_no_projection', wrong_cut),
                                 ('projected_nearest224', projected224),
                                 ('projected_nearest256_resize224', resize_rgb(projected256, 224))]:
            arrays[name + '__' + condition] = np.where(
                arrays['mask'][..., None], image, arrays['clean_base'])

    cfg = yaml.safe_load((D.PROBE / 'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(old['patch_m'])] * 2
    texpath = dest / 'uvcell8_Q128.png'
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    bddl = D.pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    runtime = {}
    checks = {'same_frozen_Q_uploaded_without_inverse_export': True,
              'frozen_ramp_geometry_not_fitted_to_checks': True,
              'fixture_arrays_match_E23': True,
              'original_native_checker_anchors_reproduced': True,
              'repeat_render_equal': True,
              'all_minimal_interior_pixels_on_visible_patch': True,
              'same_composition_mask_and_clean_background': True}
    for aa in [0, 4]:
        inject = sp.make_xml_processor(cfg, old['cell_world'], str(texpath))

        def xml_processor(xml):
            root = ET.fromstring(inject(xml))
            visual = root.find('visual')
            if visual is None:
                visual = ET.SubElement(root, 'visual')
            quality = visual.find('quality')
            if quality is None:
                quality = ET.SubElement(visual, 'quality')
            quality.set('offsamples', str(aa))
            return ET.tostring(root, encoding='unicode')

        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        env.env.set_xml_processor(xml_processor)
        try:
            env.seed(int(old['environment_seed'])); env.reset()
            sim = env.env.sim
            model, ctx = sim.model._model, sim._render_context_offscreen
            ctx.gl_ctx.make_current()
            assert int(ctx.con.offSamples) == aa
            fixture = np.concatenate([np.asarray(model.body_pos).ravel(), np.asarray(model.body_quat).ravel()])
            assert np.array_equal(fixture, old['fixture_pose_aa' + str(aa)])
            arrays['fixture_pose_aa' + str(aa)] = fixture
            tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, cfg['patch']['name'] + '_tex')
            gid = sim.model.geom_name2id(cfg['patch']['name'])
            adr, glid = int(model.tex_adr[tid]), int(ctx.con.texture[tid])
            GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, glid)
            default_min = int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER))
            default_mag = int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER))
            runtime[str(aa)] = {'mujoco': mujoco.__version__, 'offSamples': aa,
                               'min_filter': default_min, 'mag_filter': default_mag}

            def upload(q):
                model.tex_data[adr:adr + q.size] = q.reshape(-1)
                mujoco.mjr_uploadTexture(model, ctx.con, tid)

            def draw(unlit=False):
                size = 224 if aa == 0 else 256
                ctx.gl_ctx.make_current()
                GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, glid)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                                   GL.GL_NEAREST if aa == 0 else default_min)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER,
                                   GL.GL_NEAREST if aa == 0 else default_mag)
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
                    for geom in ctx.scn.geoms[:ctx.scn.ngeom]:
                        geom.emission = float(geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == gid)
                mujoco.mjr_render(mujoco.MjrRect(0, 0, size, size), ctx.scn, ctx.con)
                raw = np.ascontiguousarray(ctx.read_pixels(size, size)[::-1, ::-1])
                ctx.scn.flags[:] = flags
                return raw

            outputs = {name: {} for name in textures}
            for k, state in enumerate(old['sim_states']):
                env.regenerate_obs_from_state(state)
                # Reproduce E23 exactly before changing to a world-regular Q.
                upload(old['cell8__texture_nearest' if aa == 0 else 'cell8__texture_linear'])
                anchor = draw(aa == 0)
                key = 'cell8__minimal_nearest_export__native' if aa == 0 else 'cell8__render__native'
                assert np.array_equal(anchor, old[key][k]), ('anchor', aa, k)
                if aa == 0:
                    upload(np.full((128, 128, 3), 255, np.uint8))
                    assert draw(True)[arrays['inner_mask'][k]].min() == 255
                for name, q in textures.items():
                    upload(q)
                    variants = [('minimal_unlit_nearest224', True)] if aa == 0 else [
                        ('unlit_default256_resize224', True), ('lit_default256_resize224', False)]
                    for condition, unlit in variants:
                        raw = draw(unlit)
                        if k == 0:
                            assert np.array_equal(raw, draw(unlit))
                        outputs[name].setdefault(condition, []).append(raw)
                print('[world-checker] AA', aa, 'step', int(old['steps'][k]), flush=True)
            for name, cases in outputs.items():
                for condition, captures in cases.items():
                    arrays[name + '__' + condition + '__native'] = np.array(captures)
                    images = np.array([resize_rgb(raw, 224) for raw in captures])
                    arrays[name + '__' + condition] = np.where(
                        arrays['mask'][..., None], images, arrays['clean_base'])
            del upload, draw, model, ctx, sim
        finally:
            env.close()

    conditions = ['screen_cut_no_projection', 'projected_nearest224',
                  'projected_nearest256_resize224', 'minimal_unlit_nearest224',
                  'unlit_default256_resize224', 'lit_default256_resize224']
    arrays['conditions'] = np.array(conditions)
    records = []
    for name in textures:
        for condition in ['screen_cut_no_projection', 'minimal_unlit_nearest224',
                          'unlit_default256_resize224', 'lit_default256_resize224']:
            reference = ('projected_nearest224' if condition in
                         ['screen_cut_no_projection', 'minimal_unlit_nearest224'] else
                         'projected_nearest256_resize224')
            for k, step in enumerate(arrays['steps']):
                image = arrays[name + '__' + condition][k]
                target = arrays[name + '__' + reference][k]
                assert np.array_equal(image[~arrays['mask'][k]], arrays['clean_base'][k][~arrays['mask'][k]])
                mask, inner = arrays['mask'][k].astype(bool), arrays['inner_mask'][k].astype(bool)
                error = np.abs(image.astype(float) - target.astype(float))
                records.append({'pattern': name, 'step': int(step), 'condition': condition,
                                'reference': reference, 'mask_MAE_u8': float(error[mask].mean()),
                                'interior_MAE_u8': float(error[inner].mean()),
                                'interior_changed_pixel_fraction': float(np.any(image != target, axis=-1)[inner].mean())})
    summary = {name: {condition: {key: float(np.mean([r[key] for r in records
                  if r['pattern'] == name and r['condition'] == condition]))
                  for key in ['mask_MAE_u8', 'interior_MAE_u8', 'interior_changed_pixel_fraction']}
                  for condition in ['screen_cut_no_projection', 'minimal_unlit_nearest224',
                                    'unlit_default256_resize224', 'lit_default256_resize224']}
               for name in textures}
    np.savez_compressed(dest / 'inputs.npz', **arrays)
    (dest / 'pixel_metrics.json').write_text(json.dumps({'records': records, 'summary': summary}, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mask = arrays['mask'][0].astype(bool)
    by, bx = np.where(mask)
    sy = slice(max(0, int(by.min()) - 5), min(224, int(by.max()) + 6))
    sx = slice(max(0, int(bx.min()) - 5), min(224, int(bx.max()) + 6))
    columns = [('screen_cut_no_projection', 'Same Q resized then cut (wrong geometry)'),
               ('projected_nearest224', 'Q forward perspective projection'),
               ('minimal_unlit_nearest224', 'Real MuJoCo: unlit / nearest / AA0'),
               ('lit_default256_resize224', 'Real MuJoCo: production rendering')]
    fig, axes = plt.subplots(4, 5, figsize=(16, 11), constrained_layout=True)
    for row, (name, q) in enumerate(textures.items()):
        axes[row, 0].imshow(q, interpolation='nearest')
        axes[row, 0].set_ylabel('{} texels / cell\n{:.3f} mm / cell'.format(
            sizes[row], float(old['patch_m']) * 1000 * sizes[row] / 128))
        if row == 0:
            axes[row, 0].set_title('One regular world-UV Q128')
        for col, (condition, title) in enumerate(columns, 1):
            axes[row, col].imshow(arrays[name + '__' + condition][0, sy, sx], interpolation='nearest')
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('World-UV checker control: same texture, world plane, camera, state and visibility mask')
    fig.savefig(dest / 'fig_world_checker.png', dpi=160); plt.close(fig)
    name = 'uvcell8'
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, condition, title in zip(axes, ['screen_cut_no_projection', 'projected_nearest224',
                                         'lit_default256_resize224'],
                                    ['Screen-space cut', 'World-Q perspective reference', 'True scene render']):
        image = (arrays[name + '__lit_default256_resize224__native'][0] if condition == 'lit_default256_resize224'
                 else arrays[name + '__' + condition][0])
        ax.imshow(image, interpolation='nearest'); ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(dest / 'fig_world_scene.png', dpi=160); plt.close(fig)

    meta = {'complete': True, 'frames': len(old['steps']), 'texture_patterns': list(textures),
            'physical_patch_cm': float(old['patch_m']) * 100,
            'cell_width_world_mm': [float(old['patch_m']) * 1000 * s / 128 for s in sizes],
            'runtime': runtime, 'checks': checks, 'summary': summary,
            'projected_quad224_xy': arrays['projected_quad224_xy'].tolist(),
            'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [old_path, cal_path, D.pathlib.Path(__file__)]},
            'notes': 'Regular grid is defined in Q128 world texture, uploaded unchanged. '
                     'Digital reference forward-projects Q with frozen independent ramp H. '
                     'Same E15 16 correlated states and composition mask; raw actual images also saved. '
                     'H is empirically calibrated, not an independent camera-geometry ground truth. '
                     'Minimal uses native224 NN; production reference uses native256 NN then PIL224, '
                     'so production residual includes texture filtering, AA and photometry. '
                     'No new model K/V, velocity, training or closed-loop measurements.'}
    (dest / 'summary.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    lines = ['# Regular world-UV checker: forward perspective control', '',
             'One Q128 is uploaded unchanged and also forward-projected into the digital input. '
             'No inverse export from a screen-space checker is used. Frozen ramp homography predates this test.', '',
             '| Q128 cell width | World cell width, mm | Screen-cut MAE | Minimal-render MAE | Unlit-production MAE | Lit-production MAE |',
             '|---|---:|---:|---:|---:|---:|']
    for name, size in zip(textures, sizes):
        values = [summary[name][c]['interior_MAE_u8'] for c in
                  ['screen_cut_no_projection', 'minimal_unlit_nearest224',
                   'unlit_default256_resize224', 'lit_default256_resize224']]
        lines.append('| {} | {:.3f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} |'.format(
            size, float(old['patch_m']) * 1000 * size / 128, *values))
    lines += ['', 'MAE is mean interior RGB error on 16 correlated states, u8 scale. '
              'Screen-cut and minimal use projected NN224 reference; production uses projected NN256 resized to224. '
              'Cell widths are texture texels/world millimeters, not the source224 pixel widths of E23.', '',
              '![World checker](fig_world_checker.png)', '', '![Whole scene](fig_world_scene.png)', '',
              'The geometry reference uses previously calibrated H; this is a forward-sampling validation, '
              'not an independent camera-intrinsics calibration. No new model or closed-loop result.']
    (dest / 'results.md').write_text('\n'.join(lines) + '\n')
    print('[done]', json.dumps({'out': str(dest), 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
