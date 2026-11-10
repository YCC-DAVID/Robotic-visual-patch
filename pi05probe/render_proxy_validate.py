"""Paired model validation of a candidate renderer proxy, never training.

For each pattern, compare candidate/true-render tensors directly, with common
clean inputs and action noise. Pixel RMSE alone does not validate this route.
"""
import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
import drift_patch_train as T
import numpy as np


def e17_inputs(dest):
    import torch
    from render_training_proxy import RenderTrainingProxy
    pool = np.load(T.OUT / 'kv_render_probe/inputs.npz', allow_pickle=False)
    patch = np.load(T.OUT / 'drift_patch_drift.npz', allow_pickle=True)
    proxy = RenderTrainingProxy(T.OUT / 'render_pixel_diagnosis/training_proxy_calibration.json',
                                patch['quad224'], 'cpu')
    patterns = {'random': np.random.RandomState(20260902).randint(0, 256, (224, 224, 3)).astype(np.uint8)}
    for name, tag in [('full', 'drift'), ('band32', 'band32')]:
        p = np.load(T.OUT / ('drift_patch_' + tag + '.npz'), allow_pickle=True)
        patterns[name] = np.clip(p['P'].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    arrays = {k: pool[k] for k in ['steps', 'clean_base', 'clean_wrist', 'state8', 'prompt']}
    arrays['mask'] = pool['drift__mask']
    arrays['patterns'] = np.array(list(patterns))
    arrays['conditions'] = np.array(['digital', 'render', 'e17'])
    for name, pattern in patterns.items():
        pred = proxy(torch.from_numpy(pattern.transpose(2, 0, 1).astype(np.float32) / 255)[None])
        pred = np.clip(np.rint(pred.numpy()[0].transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
        tag = 'band32' if name == 'band32' else 'drift'
        source = tag + '__' + ('random_render_base' if name == 'random' else 'optimized_render_base')
        for cond, image in [('digital', pattern), ('render', pool[source]), ('e17', pred)]:
            arrays[name + '__' + cond] = np.where(arrays['mask'][..., None], image, arrays['clean_base'])
    np.savez_compressed(dest / 'inputs.npz', **arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='render_proxy_eot/validation_e17')
    ap.add_argument('--build-e17', action='store_true')
    args = ap.parse_args()
    import torch
    import jax
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    torch.set_num_threads(4)
    dest = T.OUT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    if args.build_e17:
        e17_inputs(dest)
    pool = np.load(dest / 'inputs.npz', allow_pickle=False)
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg, T.TORCH_CKPT)
    model, device = policy._model, policy._pytorch_device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    seeds = [20260901, 20260902, 20260903]
    noise_np = np.array([np.random.RandomState(s).standard_normal((T.AH, T.AD)).astype(np.float32) for s in seeds])
    noises = [torch.from_numpy(e)[None].to(device) for e in noise_np]
    originals = [e.clone() for e in noises]

    def forward(image, k):
        x = policy._input_transform({'observation/image': image,
            'observation/wrist_image': pool['clean_wrist'][k],
            'observation/state': pool['state8'][k], 'prompt': str(pool['prompt'])})
        x = jax.tree.map(lambda a: torch.from_numpy(np.array(a)).to(device)[None], x)
        state, mask, cache = T.prefix_and_cache(model, device, M.Observation.from_dict(x))
        snapshot = [tuple(t.clone() for t in layer) for layer in cache]
        v = torch.stack([model.denoise_step(state, mask, cache, e.clone(), torch.ones(1, device=device))
                         for e in noises]).float()
        assert all(torch.equal(a, b) for x, y in zip(snapshot, cache) for a, b in zip(x, y))
        assert all(torch.equal(a, b) for a, b in zip(originals, noises))
        valid = mask[0].bool()
        kv = [tuple(t[:, :, valid].float() for t in layer) for layer in cache]
        return {'kv': kv, 'velocity': v}, valid

    def distance(a, b):
        d = a - b
        return {'relative': float(d.norm() / b.norm().clamp_min(1e-12)),
                'rms': float(d.square().mean().sqrt()),
                'cosine': float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))}

    def pair(a, b):
        dv = a['velocity'] - b['velocity']
        return {'all18_KV': distance(torch.cat([t.flatten() for l in a['kv'] for t in l]),
                                     torch.cat([t.flatten() for l in b['kv'] for t in l])),
            'layers': [{'layer': i, 'K': distance(ak[0], bk[0]), 'V': distance(ak[1], bk[1])}
                       for i, (ak, bk) in enumerate(zip(a['kv'], b['kv']))],
            'L0': float(dv.square().flatten(1).sum(1).mean()),
            'L0_real7': float(dv[..., :7].square().flatten(1).sum(1).mean()),
            'L0_exec7': float(dv[:, :, :5, :7].square().flatten(1).sum(1).mean()),
            'per_noise_L0': dv.square().flatten(1).sum(1).cpu().tolist()}

    start = time.monotonic()
    records, clean_v, velocities = [], [], []
    conditions = pool['conditions'].tolist()
    patterns = pool['patterns'].tolist()
    with torch.no_grad():
        for k, step in enumerate(pool['steps']):
            clean, valid = forward(pool['clean_base'][k], k)
            if k == 0:
                repeat, valid2 = forward(pool['clean_base'][k], k)
                assert torch.equal(valid, valid2) and torch.equal(clean['velocity'], repeat['velocity'])
            all_packs = []
            for name in patterns:
                packs = {}
                for cond in conditions:
                    packs[cond], mask = forward(pool[name + '__' + cond][k], k)
                    assert torch.equal(valid, mask)
                for cond in conditions:
                    m = pool['mask'][k].astype(bool)
                    image = pool[name + '__' + cond][k].astype(np.float64) / 255
                    target = pool[name + '__render'][k].astype(np.float64) / 255
                    records.append({'step': int(step), 'pattern': name, 'condition': cond,
                        'pixel_RMSE_vs_render': float(np.sqrt((image[m] - target[m]).square().mean()))
                            if hasattr(image[m], 'square') else float(np.sqrt(((image[m]-target[m])**2).mean())),
                        'vs_clean': pair(packs[cond], clean),
                        'vs_digital': pair(packs[cond], packs['digital']),
                        'vs_render': pair(packs[cond], packs['render'])})
                all_packs.append(np.array([packs[c]['velocity'].cpu().numpy() for c in conditions]))
                print('[probe]', int(step), name,
                      {c: round(pair(packs[c], clean)['L0'], 6) for c in conditions}, flush=True)
            velocities.append(np.array(all_packs))
            clean_v.append(clean['velocity'].cpu().numpy())
            (dest / 'metrics.json').write_text(json.dumps({'complete': False, 'records': records}, allow_nan=False))
    np.savez_compressed(dest / 'velocities.npz', velocity=np.array(velocities), clean_velocity=np.array(clean_v),
                        action_noise=noise_np, steps=pool['steps'], patterns=pool['patterns'], conditions=pool['conditions'])
    metadata = {'complete': True, 'records': records, 'frames': len(pool['steps']), 'noise_seeds': seeds,
        'checks': {'repeat_equal': True, 'noise_unchanged': True, 'cache_not_mutated': True},
        'inputs_sha256': hashlib.sha256((dest / 'inputs.npz').read_bytes()).hexdigest(),
        'elapsed_s': time.monotonic() - start, 'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
        'notes': 'Same-mask base-only paired direct tensor differences. Clean wrist, state and language. '
                 'KV over all valid prefix tokens, 18 layers. L0 is early t=1 velocity squared difference, '
                 'not final actions or closed-loop ASR. Calibration/check frames are from one trajectory.'}
    (dest / 'metrics.json').write_text(json.dumps(metadata, allow_nan=False))
    summary = {}
    for name in patterns:
        summary[name] = {}
        for cond in conditions:
            rs = [r for r in records if r['pattern'] == name and r['condition'] == cond]
            summary[name][cond] = {'L0_clean': float(np.mean([r['vs_clean']['L0'] for r in rs])),
                'L0_vs_render': float(np.mean([r['vs_render']['L0'] for r in rs])),
                'KV_vs_render': float(np.mean([r['vs_render']['all18_KV']['relative'] for r in rs])),
                'pixel_RMSE_vs_render': float(np.mean([r['pixel_RMSE_vs_render'] for r in rs]))}
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('[done]', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
