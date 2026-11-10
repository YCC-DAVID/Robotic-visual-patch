"""Paired local replacement of a frozen position-EOT patch in actual SAPIEN RT.

Separate CPU preparation, simulator capture, and model measurement environments.
All replacements modify world texels before rendering; no digital overlays.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import bootstrap as B
from radius_plan import REFERENCE, disk, dump


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def prepare(root, source=None, center=None, size=.20, radius=.05):
    source = Path(source).resolve() if source else B.OUT/'radius_eot_20261009/train_r03/texture_0500.png'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists():
        p = json.loads((root/'protocol.json').read_text())
        assert p['source_sha256'] == sha(p['source'])
        check(root, p)
        return p
    full = np.asarray(Image.open(source).convert('RGB'))
    assert full.shape == (128, 128, 3)
    windows = [dict(index=y//16*8+x//16, x=x, y=y, size=16)
               for y in range(0, 128, 16) for x in range(0, 128, 16)]
    tiles = [np.random.RandomState(s).randint(0, 256, (16, 16, 3), dtype=np.uint8)
             for s in [20261030, 20261031, 20261032]]
    variants = [dict(name='full', kind='full'), dict(name='random', kind='random')]
    images = {'full': full, 'random': np.random.RandomState(20261007).randint(
        0, 256, full.shape, dtype=np.uint8)}
    for w in windows:
        for rep, tile in enumerate(tiles):
            name = f"w{w['index']:02d}_r{rep}"
            q = full.copy()
            q[w['y']:w['y']+16, w['x']:w['x']+16] = tile
            images[name] = q
            variants.append(dict(name=name, kind='replacement', window=w['index'], rep=rep))
    texture_dir = root/'textures'
    texture_dir.mkdir(exist_ok=True)
    for name, q in images.items():
        Image.fromarray(q).save(texture_dir/(name+'.png'))
    frames = []
    reference_hashes = {}
    for seed in [10110, 10111]:
        path = REFERENCE/'calibration'/f'seed{seed}.npz'
        reference_hashes[str(path)] = sha(path)
        with np.load(path) as d:
            frames.extend(dict(seed=seed, query=int(query), prompt=str(d['prompt']))
                          for query in d['query'])
    assert len(frames) == 9
    # Independent held-out placements, shared by every variant and action noise.
    offsets = radius*disk(np.random.RandomState(20261013), (len(frames), 3))
    base_center = np.array(center if center is not None else [0,0,.741], dtype=float)
    samples = []
    for fi, frame in enumerate(frames):
        for pi, xy in enumerate(np.vstack([np.zeros(2), offsets[fi]])):
            samples.append(dict(id=len(samples), frame=fi, position_index=pi,
                                center=(base_center+np.r_[xy,0]).tolist(), **frame))
    p = dict(model='pi0.5 RoboTwin', source=str(source), source_sha256=sha(source),
             training='frozen proxy/EOT source; see source training protocol',
             imaging_path='actual SAPIEN ray tracing; no digital overlay',
             size_m=size, test_radius_m=radius, base_center=base_center.tolist(), light='nominal', yaw_deg=0, windows=windows, variants=variants,
             samples=samples, reference_sha256=reference_hashes,
             texture_sha256={name:sha(texture_dir/(name+'.png')) for name in images},
             replacement_noise_seeds=[20261030, 20261031, 20261032],
             action_noise_seeds=[20261021, 20261022, 20261023], action_shape=[32, 32],
             sampling=f'nine held-out states from two trajectories; own center plus three new {radius*100:g} cm disk positions per state',
             selection_scope='diagnostic states; no ASR or independent-scene significance claims',
             score='full L0 - replacement L0; mean over replacement seeds and common action noises',
             secondary_scores=['velocity SSE from full', 'projection on original attack delta',
                               'per-state/per-placement response', 'visible sampling weights'],
             followup='cumulative low-influence replacement on multiple placements, followed by matched true-render closed loops')
    dump(root/'protocol.json', p)
    check(root, p)
    return p


def check(root, p):
    for path, h in p['reference_sha256'].items():
        assert sha(path) == h, path
    for name, h in p['texture_sha256'].items():
        assert sha(root/'textures'/(name+'.png')) == h, name
    full = np.asarray(Image.open(root/'textures/full.png'))
    for v in p['variants'][2:]:
        if v['kind'] != 'replacement':
            continue
        w = p['windows'][v['window']]
        changed = np.asarray(Image.open(root/'textures'/(v['name']+'.png'))) != full
        changed[w['y']:w['y']+16, w['x']:w['x']+16] = False
        assert not changed.any(), v['name']


def capture(root, limit):
    B.simulator_paths()
    import sapien
    from scene_smoke import make_task
    from render_geometry import Patch, CAMERAS
    p = prepare(root)
    rs = p['samples'][:limit] if limit else p['samples']
    pending = [r for r in rs if not (root/'capture'/f"sample{r['id']:03d}.npz").exists()]
    textures = [np.asarray(Image.open(root/'textures'/(v['name']+'.png')).convert('RGB'))
                for v in p['variants']]
    for seed in sorted({r['seed'] for r in pending}):
        assets = root/'assets'/str(seed)
        assets.mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        task, _, _ = make_task('place_container_plate', seed, assets)
        try:
            with np.load(REFERENCE/'clean'/f'seed{seed}.npz') as ref:
                selected = [r for r in pending if r['seed'] == seed]
                for query, actions in enumerate(ref['actions']):
                    if query > max(r['query'] for r in selected):
                        break
                    group = [r for r in selected if r['query'] == query]
                    if group:
                        clean = task.get_obs()
                        state = clean['joint_action']['vector'].astype(np.float32)
                        assert np.array_equal(state, ref['states14'][query])
                        for n in CAMERAS:
                            assert np.array_equal(clean['observation'][n]['rgb'], ref[n][query])
                        for r in group:
                            patch = Patch(task, assets, center=r['center'], size=p['size_m'])
                            data = dict(sample_id=r['id'], state14=state, prompt=r['prompt'],
                                        protocol_sha256=sha(root/'protocol.json'),
                                        variant_names=[v['name'] for v in p['variants']])
                            rendered = {n:[] for n in CAMERAS}
                            try:
                                for vi, texture in enumerate(textures):
                                    patch.texture(texture)
                                    obs = task.get_obs()
                                    assert np.array_equal(state, obs['joint_action']['vector'].astype(np.float32))
                                    for n in CAMERAS:
                                        rendered[n].append(obs['observation'][n]['rgb'].copy())
                                    if vi == 0:
                                        geo = patch.geometry(task)
                                        for n in CAMERAS:
                                            data[n+'_mask'] = geo[n]['mask']
                                            data[n+'_grid'] = geo[n]['grid']
                                    if vi % 16 == 0:
                                        dump(root/'capture_status.json', dict(stage='capturing', sample=r['id'],
                                            total_samples=len(p['samples']), variants_done=vi+1, variants=len(textures)))
                                patch.texture(textures[0])
                                repeat = task.get_obs()
                                for n in CAMERAS:
                                    assert np.array_equal(rendered[n][0], repeat['observation'][n]['rgb']), 'full render changed'
                            finally:
                                task.scene.remove_actor(patch.actor)
                            restored = task.get_obs()
                            for n in CAMERAS:
                                assert np.array_equal(clean['observation'][n]['rgb'], restored['observation'][n]['rgb'])
                                data[n+'_clean'] = clean['observation'][n]['rgb']
                                data[n+'_variants'] = np.stack(rendered[n])
                            save_npz(root/'capture'/f"sample{r['id']:03d}.npz", **data)
                            print('[captured]', r['id'], len(p['samples']), flush=True)
                    for action in actions:
                        task.take_action(action, action_type='qpos')
                        if task.eval_success or task.take_action_cnt >= task.step_lim:
                            break
        finally:
            task.close_env()
    dump(root/'capture_status.json', dict(stage='complete' if not limit else 'smoke_complete',
        samples=len(rs), full_samples=len(p['samples'])))


def measure(root, limit):
    B.model_paths()
    import torch
    import render_model as M
    from render_geometry import CAMERAS, WIRE
    p = prepare(root)
    policy = M.load()
    model = policy._model
    device = policy._pytorch_device
    noises = [torch.tensor(np.random.RandomState(s).standard_normal((1, 32, 32)).astype(np.float32), device=device)
              for s in p['action_noise_seeds']]
    def velocity(obs):
        state, pad, kv = M.prefix(model, obs)
        return torch.cat([model.denoise_step(state, pad, kv, n, torch.ones(1, device=device))
                          for n in noises]).float().cpu().numpy()
    rs = p['samples'][:limit] if limit else p['samples']
    with torch.no_grad():
        for r in rs:
            output = root/'metrics'/f"sample{r['id']:03d}.npz"
            if output.exists():
                continue
            path = root/'capture'/f"sample{r['id']:03d}.npz"
            with np.load(path) as d:
                assert str(d['protocol_sha256']) == sha(root/'protocol.json')
                assert list(d['variant_names']) == [v['name'] for v in p['variants']]
                clean = {w:d[n+'_clean'] for n, w in zip(CAMERAS, WIRE)}
                vc = velocity(M.observation(policy, d['state14'], str(d['prompt']), clean))
                velocities = []
                for vi, v in enumerate(p['variants']):
                    raw = {w:d[n+'_variants'][vi] for n, w in zip(CAMERAS, WIRE)}
                    velocities.append(velocity(M.observation(policy, d['state14'], str(d['prompt']), raw)))
                    if vi % 16 == 0:
                        dump(root/'measure_status.json', dict(stage='measuring', sample=r['id'],
                            total_samples=len(p['samples']), variants_done=vi+1, variants=len(p['variants'])))
                velocities = np.stack(velocities)
                delta = velocities-vc
                l0 = np.square(delta).sum((-2, -1))
                weights = []
                for n in CAMERAS:
                    # Bilinear visible texel weights at native camera pixel centers.
                    g = d[n+'_grid'][d[n+'_mask']]
                    xy = np.clip((g+1)*64-.5, 0, 127)
                    low = np.floor(xy).astype(int)
                    frac = xy-low
                    texel_weights = np.zeros((128, 128))
                    for dx, dy in [(0,0), (0,1), (1,0), (1,1)]:
                        x = np.minimum(low[:,0]+dx,127)
                        y = np.minimum(low[:,1]+dy,127)
                        weight = (frac[:,0] if dx else 1-frac[:,0])*(frac[:,1] if dy else 1-frac[:,1])
                        np.add.at(texel_weights, (y,x), weight)
                    weights.append([float(texel_weights[w['y']:w['y']+16,w['x']:w['x']+16].sum()) for w in p['windows']])
                save_npz(output, velocity=velocities, clean_velocity=vc, L0=l0,
                    drop=l0[0]-l0, full_velocity_change=np.square(velocities-velocities[0]).sum((-2,-1)),
                    attack_delta_projection=(delta*delta[0]).sum((-2,-1))/np.maximum(l0[0],1e-12),
                    footprints=np.array(weights).T, capture_sha256=sha(path), protocol_sha256=sha(root/'protocol.json'))
            print('[measured]', r['id'], 'full', l0[0].mean(), flush=True)
    dump(root/'measure_status.json', dict(stage='complete' if not limit else 'smoke_complete',
        samples=len(rs), full_samples=len(p['samples'])))


def report(root):
    p = prepare(root)
    metrics = []
    for r in p['samples']:
        with np.load(root/'metrics'/f"sample{r['id']:03d}.npz") as d:
            assert str(d['protocol_sha256']) == sha(root/'protocol.json')
            metrics.append({k:d[k].copy() for k in ['L0','drop','full_velocity_change','footprints']})
    l0 = np.stack([m['L0'] for m in metrics]).mean(-1)
    drops = np.stack([m['drop'] for m in metrics]).mean(-1)[:,2:].reshape(len(metrics),64,3).mean(-1)
    fractions = drops/np.maximum(l0[:,0,None],1e-12)
    rows = []
    for w in p['windows']:
        i = w['index']
        rows.append(dict(**w, mean_drop=float(drops[:,i].mean()),
            pooled_drop_fraction=float(drops[:,i].mean()/max(l0[:,0].mean(),1e-12)),
            mean_normalized_drop=float(fractions[:,i].mean()),
            mean_absolute_normalized_change=float(np.abs(fractions[:,i]).mean()),
            samples_with_positive_drop=int((drops[:,i]>0).sum()),
            per_sample_drop=drops[:,i].tolist(), per_sample_normalized_drop=fractions[:,i].tolist(),
            visible_weight_mean=np.stack([m['footprints'] for m in metrics])[:,i].mean(0).tolist()))
    order = sorted(range(64), key=lambda i:rows[i]['mean_absolute_normalized_change'])
    dump(root/'summary.json', dict(complete=True, full_L0_mean=float(l0[:,0].mean()),
        random_L0_mean=float(l0[:,1].mean()), per_sample_full_L0=l0[:,0].tolist(),
        per_sample_random_L0=l0[:,1].tolist(), windows=rows, cumulative_replacement_order=order,
        top_windows=sorted(range(64),key=lambda i:rows[i]['mean_drop'],reverse=True)[:8],
        endpoint='actual-render open-loop t=1 velocity; no new ASR measured',
        protocol_sha256=sha(root/'protocol.json')))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1,3,figsize=(15,4))
    for ax, values, title in zip(axs,
            [drops.mean(0), fractions.mean(0), np.stack([m['footprints'] for m in metrics]).sum(-1).mean(0)],
            ['Mean L0 drop', 'Mean per-sample normalized drop', 'Visible native sampling weight']):
        im=ax.imshow(values.reshape(8,8));ax.set_title(title);fig.colorbar(im,ax=ax)
    fig.tight_layout();fig.savefig(root/'heatmaps.png',dpi=160);plt.close(fig)
    print('[scan complete]', root, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prepare','capture','measure','report'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--limit', type=int, default=0, help='smoke only; never reports full completion')
    ap.add_argument('--source')
    ap.add_argument('--center',type=float,nargs=3)
    ap.add_argument('--size',type=float,default=.20)
    ap.add_argument('--radius',type=float,default=.05)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if args.stage == 'prepare':
        prepare(root,args.source,args.center,args.size,args.radius)
    elif args.stage in ['capture','measure']:
        globals()[args.stage](root, args.limit)
    else:
        report(root)
