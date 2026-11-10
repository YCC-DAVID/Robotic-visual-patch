"""New learned textures: nominal proxy versus actual render, three common noises."""
import argparse
import dataclasses
import hashlib
import json
import time
import drift_patch_train as T
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='render_proxy_eot/trained_validation')
    args = ap.parse_args()
    import torch
    import jax
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    from render_world_proxy import WorldTextureProxy
    torch.set_num_threads(4)
    root = T.OUT / 'render_proxy_eot'
    dest = T.OUT / args.out
    pool = np.load(dest / 'inputs.npz', allow_pickle=False)
    calibration_path = root / 'calibration_pool/calibration.npz'
    fit_path = root / 'calibration_pool_fit/fit.json'
    proxy = WorldTextureProxy(calibration_path, 'cpu', fit_path)
    images = {c + '_render': pool[c + '__render'] for c in pool['patterns']}
    pixel = {}
    for c in pool['patterns']:
        q = torch.from_numpy(pool[c + '__texture'].astype(np.float32).transpose(2, 0, 1) / 255)[None]
        with torch.no_grad():
            pred = np.array([np.rint(proxy(q, int(k)).numpy()[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
                             for k in pool['steps']])
        images[c + '_proxy'] = pred
        delta = (pred.astype(float) - pool[c + '__render'].astype(float)) / 255
        pixel[c] = {'full_image_RMSE': np.sqrt(np.square(delta).mean((1, 2, 3))).tolist(),
                    'mask_RMSE': [float(np.sqrt(np.square(d[m]).mean())) for d, m in zip(delta, pool['mask'])]}
    images['old_full_render'] = pool['old_full__render']
    images['old_random_render'] = pool['old_random__render']
    np.savez_compressed(dest / 'model_inputs.npz', **images)
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg, T.TORCH_CKPT)
    model, device = policy._model, policy._pytorch_device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    seeds = [20260901, 20260902, 20260903]
    noise_np = np.array([np.random.RandomState(s).standard_normal((10, 32)).astype(np.float32) for s in seeds])
    noises = [torch.from_numpy(x)[None].to(device) for x in noise_np]

    def forward(image, k):
        x = policy._input_transform({'observation/image': image, 'observation/wrist_image': pool['clean_wrist'][k],
                                    'observation/state': pool['state8'][k], 'prompt': str(pool['prompt'])})
        x = jax.tree.map(lambda a: torch.from_numpy(np.array(a)).to(device)[None], x)
        state, mask, cache = T.prefix_and_cache(model, device, M.Observation.from_dict(x))
        snapshots = [tuple(t.clone() for t in layer) for layer in cache]
        v = torch.stack([model.denoise_step(state, mask, cache, e.clone(), torch.ones(1, device=device))
                         for e in noises]).float()
        assert all(torch.equal(a, b) for x, y in zip(snapshots, cache) for a, b in zip(x, y))
        kv = torch.cat([t[:, :, mask[0].bool()].float().flatten() for layer in cache for t in layer])
        return v, kv, mask

    def velocity(a, b):
        d = a - b
        return {'L0': float(d.square().flatten(1).sum(1).mean()),
                'L0_real7': float(d[..., :7].square().flatten(1).sum(1).mean()),
                'L0_exec7': float(d[:, :, :5, :7].square().flatten(1).sum(1).mean())}

    records, all_v, all_clean = [], [], []
    conditions = list(images)
    started = time.monotonic()
    with torch.no_grad():
        for k, frame in enumerate(pool['steps']):
            vc, kc, mask = forward(pool['clean_base'][k], k)
            if k == 0:
                repeat = forward(pool['clean_base'][k], k)
                assert torch.equal(vc, repeat[0]) and torch.equal(kc, repeat[1])
            packs = {c: forward(images[c][k], k) for c in conditions}
            for c, (v, kv, mk) in packs.items():
                assert torch.equal(mask, mk)
                row = {'frame_index': int(frame), 'init_index': int(pool['ep_id'][k]),
                       'env_step': int(pool['env_step'][k]), 'condition': c,
                       'vs_clean': velocity(v, vc), 'KV_relative_clean': float((kv-kc).norm()/kc.norm())}
                if c.endswith('_proxy'):
                    r = packs[c.replace('_proxy', '_render')]
                    row['vs_render'] = velocity(v, r[0])
                    row['KV_relative_render'] = float((kv-r[1]).norm()/r[1].norm())
                records.append(row)
            all_v.append(np.array([packs[c][0].cpu().numpy() for c in conditions]))
            all_clean.append(vc.cpu().numpy())
            print('[probe]', int(frame), {r['condition']: round(r['vs_clean']['L0'], 5)
                  for r in records[-len(conditions):]}, flush=True)
    np.savez_compressed(dest / 'velocities.npz', velocity=np.array(all_v), clean_velocity=np.array(all_clean),
                        action_noise=noise_np, steps=pool['steps'], conditions=np.array(conditions))
    summary = {}
    for c in conditions:
        rows = [r for r in records if r['condition'] == c]
        summary[c] = {key: float(np.mean([r['vs_clean'][key] for r in rows]))
                      for key in ['L0', 'L0_real7', 'L0_exec7']}
        if c.endswith('_proxy'):
            summary[c]['paired_L0_vs_render'] = float(np.mean([r['vs_render']['L0'] for r in rows]))
            summary[c]['KV_relative_render'] = float(np.mean([r['KV_relative_render'] for r in rows]))
    metadata = {'complete': True, 'frames': len(pool['steps']), 'conditions': conditions,
                'noise_seeds': seeds, 'records': records, 'summary': summary, 'pixel_errors': pixel,
                'elapsed_s': time.monotonic()-started,
                'inputs_sha256': hashlib.sha256((dest / 'inputs.npz').read_bytes()).hexdigest(),
                'checks': {'repeat_clean_equal': True, 'cache_not_mutated': True, 'state_masks_equal': True},
                'notes': '32 independent validation states (two episodes), base-only and full images. '
                         'Same three explicit noises; proxy/render paired directly. No integration or rollout. '
                         'These episodes were in old historical full training but not new training0-9.'}
    (dest / 'metrics.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('[done]', summary, flush=True)


if __name__ == '__main__':
    main()
