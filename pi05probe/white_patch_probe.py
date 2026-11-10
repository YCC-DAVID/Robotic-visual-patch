"""E21: paired backbone KV and first-flow responses to constant-white inputs."""
import argparse
import dataclasses
import hashlib
import json
import time
import drift_patch_train as T
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='white_patch_probe')
    args = ap.parse_args()
    import torch
    import jax
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    torch.set_num_threads(4)
    dest = T.OUT / args.out
    pool = np.load(dest / 'inputs.npz', allow_pickle=False)
    old_clean = np.load(T.OUT / 'kv_interpolation/velocities.npz', allow_pickle=False)
    assert np.array_equal(pool['steps'], old_clean['steps'])
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg, T.TORCH_CKPT)
    model, device = policy._model, policy._pytorch_device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    seeds = [20260901, 20260902, 20260903]
    noise_np = np.array([np.random.RandomState(s).standard_normal((T.AH, T.AD)).astype(np.float32)
                         for s in seeds])
    noises = [torch.from_numpy(e)[None].to(device) for e in noise_np]
    initial_noises = [e.clone() for e in noises]
    features = []
    handle = model.paligemma_with_expert.paligemma.model.multi_modal_projector.register_forward_hook(
        lambda module, inputs, output: features.append(output.detach().float()))
    checks = {'state_masks_equal': True, 'cache_not_mutated': True,
              'noise_unchanged': True, 'identical_interior_input_kv_velocity_equal': True}
    old_clean_error = 0.

    def forward(image, k):
        features.clear()
        x = policy._input_transform({'observation/image': image,
            'observation/wrist_image': pool['clean_wrist'][k],
            'observation/state': pool['state8'][k], 'prompt': str(pool['prompt'])})
        x = jax.tree.map(lambda a: torch.from_numpy(np.array(a)).to(device)[None], x)
        obs = M.Observation.from_dict(x)
        state, mask, cache = T.prefix_and_cache(model, device, obs)
        feature = features[list(obs.images).index(T.BASE_KEY)]
        snapshot = [tuple(t.clone() for t in layer) for layer in cache]
        velocity = torch.stack([model.denoise_step(state, mask, cache, e.clone(),
                                torch.ones(1, device=device)) for e in noises]).float()
        assert all(torch.equal(a, b) for x, y in zip(snapshot, cache) for a, b in zip(x, y))
        assert all(torch.equal(a, b) for a, b in zip(initial_noises, noises))
        valid = mask[0].bool()
        kv = [tuple(t[:, :, valid].float() for t in layer) for layer in cache]
        return {'feature': feature, 'kv': kv, 'velocity': velocity}, valid

    def distance(a, b):
        delta = a - b
        na, nb, nd = float(a.norm()), float(b.norm()), float(delta.norm())
        return {'relative_to_reference': nd / max(nb, 1e-12),
                'delta_norm': nd, 'delta_rms': float(delta.square().mean().sqrt()),
                'raw_cosine': float((a * b).sum()) / (na * nb) if na * nb > 1e-12 else None}

    def velocity(a, b):
        d = a - b
        return {'L0': float(d.square().flatten(1).sum(1).mean()),
                'L0_real7': float(d[..., :7].square().flatten(1).sum(1).mean()),
                'L0_exec7': float(d[:, :, :5, :7].square().flatten(1).sum(1).mean()),
                'per_noise_L0': d.square().flatten(1).sum(1).cpu().tolist()}

    def pair(a, b):
        layers = [{'layer': i, **{label: distance(ak[j], bk[j])
                  for j, label in enumerate(['K', 'V'])}}
                  for i, (ak, bk) in enumerate(zip(a['kv'], b['kv']))]
        akv = torch.cat([t.flatten() for layer in a['kv'] for t in layer])
        bkv = torch.cat([t.flatten() for layer in b['kv'] for t in layer])
        return {'feature': distance(a['feature'], b['feature']), 'kv': layers,
                'all18_KV': distance(akv, bkv), 'velocity': velocity(a['velocity'], b['velocity'])}

    pairs = [('render_white', 'digital_white'), ('render_white_unlit', 'digital_white'),
             ('render_white', 'digital_matched_rgb'), ('digital_matched_rgb', 'digital_white'),
             ('render_white', 'render_white_unlit'), ('render_white_raw', 'render_white'),
             ('render_white_interior', 'digital_white_interior'),
             ('render_white_unlit_interior', 'digital_white_interior')]
    conditions = pool['conditions'].tolist()
    records, pair_records, all_velocities, clean_velocities = [], [], [], []
    start = time.monotonic()
    try:
        with torch.no_grad():
            for k, step in enumerate(pool['steps']):
                clean, valid = forward(pool['clean_base'][k], k)
                old_clean_error = max(old_clean_error, float(np.max(np.abs(
                    clean['velocity'].cpu().numpy() - old_clean['clean_velocity'][k]))))
                assert old_clean_error <= 1e-4, ('E20 clean anchor differs', old_clean_error)
                if k == 0:
                    repeat, v = forward(pool['clean_base'][k], k)
                    assert torch.equal(valid, v)
                    assert torch.equal(clean['velocity'], repeat['velocity'])
                    assert all(torch.equal(a, b) for x, y in zip(clean['kv'], repeat['kv'])
                               for a, b in zip(x, y))
                    checks['repeat_clean_equal'] = True
                    valid_tokens = int(valid.sum())
                    prefix_tokens = int(valid.numel())
                packs = {}
                for condition in conditions:
                    target, v = forward(pool[condition][k], k)
                    assert torch.equal(valid, v)
                    packs[condition] = target
                    reference = 'digital_white_interior' if condition.endswith('_interior') else 'digital_white'
                    records.append({'step': int(step), 'condition': condition, 'reference': reference,
                        'vs_clean': pair(target, clean), 'vs_digital': pair(target, packs[reference])})
                assert np.array_equal(pool['digital_white_interior'][k], pool['render_white_unlit_interior'][k])
                a, b = packs['digital_white_interior'], packs['render_white_unlit_interior']
                assert torch.equal(a['velocity'], b['velocity'])
                assert all(torch.equal(x, y) for ak, bk in zip(a['kv'], b['kv']) for x, y in zip(ak, bk))
                for ca, cb in pairs:
                    pair_records.append({'step': int(step), 'target': ca, 'reference': cb,
                                         **pair(packs[ca], packs[cb])})
                all_velocities.append(np.array([packs[c]['velocity'].cpu().numpy() for c in conditions]))
                clean_velocities.append(clean['velocity'].cpu().numpy())
                l0 = {r['condition']: round(r['vs_clean']['velocity']['L0'], 6)
                      for r in records[-len(conditions):]}
                print('[probe]', int(step), l0, flush=True)
                (dest / 'metrics.json').write_text(json.dumps(
                    {'complete': False, 'records': records, 'pairs': pair_records}, allow_nan=False))
    finally:
        handle.remove()
    checks['E20_clean_velocity_anchor_max_error'] = old_clean_error
    np.savez_compressed(dest / 'velocities.npz', velocity=np.array(all_velocities),
                        clean_velocity=np.array(clean_velocities), action_noise=noise_np,
                        steps=pool['steps'], conditions=pool['conditions'])
    metadata = {'complete': True, 'frames': len(pool['steps']), 'conditions': conditions,
        'records': records, 'pairs': pair_records, 'checks': checks, 'noise_seeds': seeds,
        'valid_tokens': valid_tokens, 'prefix_tokens': prefix_tokens,
        'elapsed_s': time.monotonic() - start,
        'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30,
        'inputs_sha256': hashlib.sha256((dest / 'inputs.npz').read_bytes()).hexdigest(),
        'notes': 'Same 16 states; base-only, clean wrist/state/language. Post-RoPE K and V at all 18 layers, '
                 'valid prefix tokens only. L0 is squared first-flow velocity difference at t=1, averaged '
                 'over three common noises, not actions or ASR. All condition pairs compare tensors '
                 'directly. Full-mask conditions and eroded-interior conditions use separate digital '
                 'references. Constant-white is a diagnostic control, not a newly trained patch. '
                 'No training, integration or rollout.'}
    (dest / 'metrics.json').write_text(json.dumps(metadata, allow_nan=False))
    print('[done]', len(records), round(time.monotonic() - start), 'seconds', flush=True)


if __name__ == '__main__':
    main()
