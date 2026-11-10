"""Capture matched clean/digital/MuJoCo inputs on one saved clean trajectory.

Runs in openpi-libero. Writes only to its own output directory; no policy,
training, or closed-loop execution. Each renderer resets with the same fixture
seed before restoring the same saved simulator state.
"""
import argparse
import hashlib
import json

import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patches', default='drift13,drift13_b32,drift,band32')
    ap.add_argument('--out', default='kv_render_probe')
    ap.add_argument('--frames', type=int, default=16)
    args = ap.parse_args()
    import cv2
    import yaml
    from PIL import Image, ImageDraw
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    from drift_resize_order import resize_rgb

    dest = D.OUT / args.out
    dest.mkdir(exist_ok=True)
    (dest / 'textures').mkdir(exist_ok=True)
    traj_path = D.OUT / ('traj_' + D.TASK + '.npz')
    traj = np.load(traj_path, allow_pickle=False)
    selected = np.linspace(0, int(traj['n_frames']) - 1,
                           min(args.frames, int(traj['n_frames'])), dtype=int)
    states = [traj['f{:03d}__flatten'.format(k)] for k in selected]
    seed = int(traj['shared_seed'])
    cfg = yaml.safe_load((D.PROBE / 'config/scene.yaml').read_text())
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    bddl = D.pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file

    def render(texture=None, world=None):
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        if texture is not None:
            env.env.set_xml_processor(sp.make_xml_processor(cfg, world, str(texture)))
        frames = {'base': [], 'wrist': []}
        try:
            env.seed(seed)
            env.reset()
            for state in states:
                obs = env.regenerate_obs_from_state(state)
                for view, cam in [('base', 'agentview'), ('wrist', 'robot0_eye_in_hand')]:
                    im = np.ascontiguousarray(obs[cam + '_image'][::-1, ::-1])
                    frames[view].append(resize_rgb(im, 224))
        finally:
            env.close()
        return {k: np.array(v) for k, v in frames.items()}

    clean = render()
    repeat = render()
    assert all(np.array_equal(clean[k], repeat[k]) for k in clean), 'Clean render not repeatable'
    saved_mae = {}
    for view, suffix in [('base', 'img224'), ('wrist', 'wrist224')]:
        saved = np.array([traj['f{:03d}__{}'.format(k, suffix)] for k in selected])
        saved_mae[view] = float(np.abs(saved.astype(float) - clean[view]).mean())
    arrays = {'clean_base': clean['base'], 'clean_wrist': clean['wrist'],
              'state8': np.array([traj['f{:03d}__state8'.format(k)] for k in selected]),
              'steps': traj['ts'][selected], 'prompt': str(traj['prompt'])}
    meta = {'task': D.TASK, 'trajectory': traj_path.name, 'trajectory_sha256':
            hashlib.sha256(traj_path.read_bytes()).hexdigest(), 'seed': seed,
            'frames': len(selected), 'steps': arrays['steps'].tolist(),
            'clean_repeat_equal': True, 'saved_clean_mae_u8': saved_mae,
            'notes': 'One clean trajectory replay, not a closed-loop or independent-test evaluation.',
            'texture_resolution': 128, 'patches': {}}
    random_image = np.random.RandomState(20260902).randint(0, 256, (224, 224, 3)).astype(np.uint8)
    montage_steps = [0, len(selected)//2, len(selected)-1]
    for tag in args.patches.split(','):
        path = D.OUT / ('drift_patch_' + tag + '.npz')
        p = np.load(path, allow_pickle=True)
        cfg['patch']['size_wh'] = [float(p['patch_m'])] * 2
        world = p['cell_world']
        pattern = np.clip(p['P'].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        quad = p['quad224'].astype(np.float32)[:, ::-1]
        corners = np.array([[0, 0], [127, 0], [127, 127], [0, 127]], np.float32)
        H = cv2.getPerspectiveTransform(quad, corners)
        textures = {}
        for label, image in [('optimized', pattern), ('random', random_image)]:
            # Match the calibrated DRIFT exporter, without overwriting config/.
            texture = cv2.warpPerspective(image, H, (128, 128))[::-1].copy()
            tp = dest / 'textures' / (tag + '_' + label + '.png')
            Image.fromarray(texture).save(tp)
            textures[label] = render(tp, world)
        probe = render(D.PROBE / 'config/probe_texture.png', world)
        mask = np.max(np.abs(probe['base'].astype(np.int16) - clean['base'].astype(np.int16)), -1) > 12
        arrays[tag + '__mask'] = mask
        arrays[tag + '__digital'] = np.where(mask[..., None], pattern, clean['base']).astype(np.uint8)
        arrays[tag + '__random_digital'] = np.where(mask[..., None], random_image, clean['base']).astype(np.uint8)
        for label, images in textures.items():
            for view, image in images.items():
                arrays[tag + '__' + label + '_render_' + view] = image
        meta['patches'][tag] = {'source': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'size_m': float(p['patch_m']), 'world': world.tolist(),
                                'visible_mask_px': mask.reshape(len(mask), -1).sum(1).tolist()}
        canvas = Image.new('RGB', (224 * 4, 250 * len(montage_steps)), 'white')
        draw = ImageDraw.Draw(canvas)
        for row, k in enumerate(montage_steps):
            images = [('clean base', clean['base'][k]), ('digital base', arrays[tag+'__digital'][k]),
                      ('render base', textures['optimized']['base'][k]),
                      ('render wrist', textures['optimized']['wrist'][k])]
            for col, (label, im) in enumerate(images):
                canvas.paste(Image.fromarray(im), (col*224, row*250+24))
                draw.text((col*224+4, row*250+4), '{} / step {}'.format(label, int(arrays['steps'][k])), fill='black')
        canvas.save(dest / ('inputs_' + tag + '.png'))
        print('[capture]', tag, 'size', float(p['patch_m']), 'frames', len(selected), flush=True)
    arrays['patch_tags'] = np.array(args.patches.split(','))
    np.savez_compressed(dest / 'inputs.npz', **arrays)
    (dest / 'capture.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('[written]', dest, flush=True)


if __name__ == '__main__':
    main()
