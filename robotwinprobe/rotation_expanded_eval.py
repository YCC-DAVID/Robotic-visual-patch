"""New clean baseline and paired digital yaw rollouts; no writes to old runs."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import bootstrap as B
from digital_transform_common import dump, sha, condition, clean_images, geometry_at, composite
from rotation_expanded_plan import prepare, episode_dir


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def evaluate(root, job_path, port):
    B.simulator_paths()
    import torch
    from PIL import Image
    from scene_smoke import make_task
    from render_geometry import CAMERAS, WIRE
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    torch.set_num_threads(4)
    protocol = prepare(root)
    jobs = json.loads(job_path.read_text())
    pats = {}
    for method, path in protocol['texture_paths'].items():
        assert sha(path) == protocol['texture_sha256'][method]
        pats[method] = np.asarray(Image.open(path).convert('RGB'))
    assets = root / 'assets' / job_path.stem
    assets.mkdir(parents=True, exist_ok=True)
    client = WebsocketClientPolicy('127.0.0.1', port)
    try:
        for job in jobs:
            seed, method, cname = job['seed'], job['method'], job['condition']
            assert [cname, method] in protocol['groups']
            assert seed in protocol['initial_seeds'] + protocol['extension_seeds']
            out = episode_dir(root, cname, method)
            out.mkdir(parents=True, exist_ok=True)
            stem = out / f'seed{seed}'
            texture_hash = None if method == 'clean' else protocol['texture_sha256'][method]
            if stem.with_suffix('.json').exists():
                saved = json.loads(stem.with_suffix('.json').read_text())
                assert (saved['seed'], saved['condition'], saved['method'], saved['texture_sha256']) == (seed, cname, method, texture_hash)
                assert stem.with_suffix('.npz').exists()
                continue
            started = time.monotonic()
            random.seed(seed)
            task, _, _ = make_task(protocol['task'], seed, assets)
            try:
                prompt = 'Place the ' + ('bowl' if task.actor_name == '002_bowl' else 'cup') + ' on the plate.'
                task.set_instruction(prompt)
                initial = task.get_obs()
                initial_state = initial['joint_action']['vector'].astype(np.float32).copy()
                initial_rgb = {n: initial['observation'][n]['rgb'].copy() for n in CAMERAS}
                if method != 'clean':
                    base = episode_dir(root, 'nominal', 'clean') / f'seed{seed}'
                    clean_row = json.loads(base.with_suffix('.json').read_text())
                    assert clean_row['prompt'] == prompt
                    with np.load(base.with_suffix('.npz')) as ref:
                        assert np.array_equal(initial_state, ref['initial_state14']), (seed, 'state')
                        for n in CAMERAS:
                            assert np.array_equal(initial_rgb[n], ref['initial_clean_' + n]), (seed, n)
                c = condition(cname)
                states, chunks, noises, pixels = [], [], [], []
                rgb = {w: [] for w in WIRE}
                first_masks = None
                while task.take_action_cnt < task.step_lim and not task.eval_success:
                    obs = task.get_obs()
                    state = obs['joint_action']['vector'].astype(np.float32)
                    if method == 'clean':
                        images = clean_images(obs)
                        masks = {w: np.zeros((224, 224), bool) for w in WIRE}
                    else:
                        geo = geometry_at(task, assets, c, pats[method])
                        images, masks = composite(obs, geo, pats[method], c, True)
                        if not states:
                            restored = task.get_obs()
                            assert np.array_equal(restored['joint_action']['vector'].astype(np.float32), state)
                            for n in CAMERAS:
                                assert np.array_equal(restored['observation'][n]['rgb'], obs['observation'][n]['rgb'])
                    if first_masks is None:
                        first_masks = np.stack([masks[w] for w in WIRE])
                    pixels.append([int(masks[w].sum()) for w in WIRE])
                    ns = 20261006 + seed * 1000 + len(states)
                    noise = np.random.RandomState(ns).standard_normal((32, 32)).astype(np.float32)
                    actions = np.asarray(client.infer(dict(state=state, prompt=prompt, images=images, noise=noise))['actions'])
                    assert actions.shape == (32, 14) and np.isfinite(actions).all()
                    states.append(state.copy()); chunks.append(actions.copy()); noises.append(ns)
                    for w in WIRE:
                        rgb[w].append(images[w].copy())
                    for action in actions:
                        task.take_action(action, action_type='qpos')
                        if task.eval_success or task.take_action_cnt >= task.step_lim:
                            break
                assert len(states) > 0
                tmp = stem.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, initial_state14=initial_state, states14=states,
                    actions=chunks, noise_seeds=noises, mask_pixels=pixels, initial_masks=first_masks,
                    **{w: np.asarray(v) for w, v in rgb.items()},
                    **{'initial_clean_' + n: v for n, v in initial_rgb.items()})
                tmp.replace(stem.with_suffix('.npz'))
                row = dict(seed=seed, condition=cname, method=method, texture_sha256=texture_hash,
                    success=bool(task.eval_success), steps=task.take_action_cnt, step_limit=task.step_lim,
                    queries=len(states), prompt=prompt, noise_rule=protocol['noise_rule'],
                    initial_state_sha256=digest(initial_state),
                    initial_rgb_sha256={n: digest(v) for n, v in initial_rgb.items()},
                    paired_initial_exact=(method != 'clean'),
                    patch_removal_restores_initial_clean=(method != 'clean'),
                    wall_seconds=time.monotonic() - started)
                dump(stem.with_suffix('.json'), row)
                print('[episode]', cname, method, seed, row['success'], round(row['wall_seconds'], 1), flush=True)
            finally:
                task.close_env()
    finally:
        client._ws.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--port', type=int, required=True)
    args = ap.parse_args()
    evaluate(Path(args.root).resolve(), Path(args.jobs).resolve(), args.port)
