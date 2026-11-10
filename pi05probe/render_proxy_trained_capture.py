"""True MuJoCo images of new PNG textures at all independent validation states."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-render-proxy-trained-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=int, default=500)
    ap.add_argument('--out', default='render_proxy_eot/trained_validation')
    args = ap.parse_args()
    import mujoco
    import yaml
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb
    import scene_patch as sp
    root = D.OUT / 'render_proxy_eot'
    calibration = np.load(root / 'calibration_pool/calibration.npz', allow_pickle=False)
    indices = np.flatnonzero(calibration['validation_mask'])
    assert set(calibration['ep_id'][indices]) == {10, 11}
    dest = D.OUT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'inputs.npz').exists(), 'use a fresh result directory'
    runs = {'nominal': root / 'train_nominal', 'eot': root / 'train_eot_run1'}
    textures = {c: np.asarray(Image.open(p / ('texture_{:04d}.png'.format(args.checkpoint))).convert('RGB'))
                for c, p in runs.items()}
    assert all(a.shape == (128, 128, 3) for a in textures.values())
    for c, p in runs.items():
        snapshot = np.load(p / ('checkpoint_{:04d}.npz'.format(args.checkpoint)), allow_pickle=False)
        assert len(snapshot['history']) == args.checkpoint
        expected = np.rint(snapshot['Q'].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        assert np.array_equal(expected, textures[c]), 'PNG and optimization checkpoint differ'
        if args.checkpoint == 500:
            result = json.loads((p / 'results.json').read_text())
            assert result['complete'] and result['iterations'] == args.checkpoint
    cfg = yaml.safe_load((D.PROBE / 'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(calibration['patch_m'])] * 2
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    bddl = D.pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg, calibration['cell_world'],
        str(runs['nominal'] / ('texture_{:04d}.png'.format(args.checkpoint)))))
    arrays = {'steps': indices, 'ep_id': calibration['ep_id'][indices],
              'env_step': calibration['env_step'][indices], 'env_seed': calibration['env_seed'][indices],
              'state8': calibration['state8'][indices], 'prompt': calibration['prompt'],
              'clean_base': calibration['clean_base'][indices], 'clean_wrist': calibration['clean_wrist'][indices],
              'mask': calibration['mask'][indices], 'patterns': np.array(list(runs))}
    rendered = {c: [] for c in runs}
    native = {c: [] for c in runs}
    previous_seed = None
    try:
        for k in indices:
            seed = int(calibration['env_seed'][k])
            if seed != previous_seed:
                env.seed(seed); env.reset(); previous_seed = seed
            env.regenerate_obs_from_state(calibration['sim_states'][k])
            sim = env.env.sim
            ctx = sim._render_context_offscreen
            model = sim.model._model
            ctx.gl_ctx.make_current()
            tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, cfg['patch']['name'] + '_tex')
            adr = int(model.tex_adr[tid])
            for c, texture in [('anchor', calibration['random__texture'])] + list(textures.items()):
                model.tex_data[adr:adr+128*128*3] = texture.reshape(-1)
                mujoco.mjr_uploadTexture(model, ctx.con, tid)
                ctx.render(256, 256, camera_id=sim.model.camera_name2id('agentview'))
                image = np.ascontiguousarray(ctx.read_pixels(256, 256)[::-1, ::-1])
                if c == 'anchor':
                    assert np.array_equal(image, calibration['random256'][k]), ('calibration replay mismatch', k)
                else:
                    native[c].append(image); rendered[c].append(resize_rgb(image, 224))
            del ctx, sim, model
            print('[capture]', int(k), 'episode', int(calibration['ep_id'][k]), flush=True)
    finally:
        env.close()
    for c in runs:
        arrays[c + '__render'] = np.array(rendered[c])
        arrays[c + '__native256'] = np.array(native[c])
        arrays[c + '__texture'] = textures[c]
    arrays['old_full__render'] = np.array([resize_rgb(calibration['full256'][k], 224) for k in indices])
    arrays['old_random__render'] = np.array([resize_rgb(calibration['random256'][k], 224) for k in indices])
    np.savez_compressed(dest / 'inputs.npz', **arrays)
    meta = {'complete': True, 'frames': len(indices), 'training_checkpoint_iteration': args.checkpoint,
            'validation_init_indices': [10, 11],
            'source_calibration_sha256': hashlib.sha256((root / 'calibration_pool/calibration.npz').read_bytes()).hexdigest(),
            'textures_sha256': {c: hashlib.sha256((p / ('texture_{:04d}.png'.format(args.checkpoint))).read_bytes()).hexdigest()
                               for c, p in runs.items()},
            'checks': {'all_calibration_random_native_anchors_equal': True},
            'notes': '32 validation states, init10/11 disjoint from new training0-9. Actual base render, '
                     'full image without masking; clean wrist for paired base-only model diagnostic. '
                     'Both patched views are evaluated separately in physical closed loop. '
                     'Old full/random are historical reference patterns from calibration captures.'}
    (dest / 'capture.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
    print('[done]', dest, flush=True)


if __name__ == '__main__':
    main()
