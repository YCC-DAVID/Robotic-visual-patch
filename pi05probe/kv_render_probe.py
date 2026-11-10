"""Inspect prefix K/V and expert response using matched captured observations.

Raw cache sizes are normalized within layer/group. Also compare perturbation
directions and use diagnostic K/V swaps. No patch optimization or rollout.
Runs in openpi-server with the same frozen pi05_libero checkpoint.
"""
import argparse
import dataclasses
import hashlib
import json
import time

import drift_patch_train as T
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='kv_render_probe')
    ap.add_argument('--noise-seeds', default='20260901,20260902,20260903')
    ap.add_argument('--frames', type=int, default=0)
    args = ap.parse_args()
    import torch
    import jax
    from transformers.cache_utils import DynamicCache
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M

    dest = T.OUT / args.out
    pool = np.load(dest / 'inputs.npz', allow_pickle=False)
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg, T.TORCH_CKPT)
    model, device = policy._model, policy._pytorch_device
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    seeds = [int(x) for x in args.noise_seeds.split(',')]
    noises = [torch.from_numpy(np.random.RandomState(s).standard_normal((T.AH, T.AD)).astype(np.float32)).to(device)[None]
              for s in seeds]
    records, layout = [], None
    active, attention = {}, {}

    def hook(layer):
        def capture(module, inputs, output):
            if active.get('capture'):
                # Eager Gemma attention returns projected output and probabilities.
                attention[layer] = (output[0].detach().float(), output[1].detach().float())
        return capture

    handles = [layer.self_attn.register_forward_hook(hook(i))
               for i, layer in enumerate(model.paligemma_with_expert.gemma_expert.model.layers)]

    def observation(base, wrist, k):
        x = policy._input_transform({'observation/image': base, 'observation/wrist_image': wrist,
                                     'observation/state': pool['state8'][k], 'prompt': str(pool['prompt'])})
        x = jax.tree.map(lambda a: torch.from_numpy(np.array(a)).to(device)[None], x)
        return M.Observation.from_dict(x)

    def first(pack, noise, capture=False):
        active['capture'] = capture
        attention.clear()
        state, mask, cache = pack
        v = model.denoise_step(state, mask, cache, noise.clone(), torch.ones(1, device=device))
        active['capture'] = False
        return v, dict(attention) if capture else None

    def integrate(pack, noise):
        state, mask, cache = pack
        x = noise.clone()
        for step in range(10):
            t = torch.full((1,), 1.0-step/10, device=device)
            x = x - .1 * model.denoise_step(state, mask, cache, x, t)
        return x

    def norm(x):
        return float(torch.linalg.vector_norm(x.float()))

    def cosine(a, b):
        aa, bb = a.float().reshape(-1), b.float().reshape(-1)
        den = aa.norm()*bb.norm()
        return float(torch.dot(aa, bb)/den) if float(den) > 1e-12 else None

    def change(a, b):
        delta = b.float()-a.float()
        return {'diff_norm': norm(delta), 'relative_norm': norm(delta)/max(norm(a), 1e-12),
                'rms': float(delta.square().mean().sqrt()), 'cosine': cosine(a, b)}

    def velocity_metrics(v, ref):
        d = v-ref
        return {'L0': float(d.square().sum()), 'L0_real7': float(d[..., :7].square().sum()),
                'L0_exec7': float(d[:, :5, :7].square().sum())}

    def kv_compare(clean, target, groups):
        layers = []
        for i in range(len(clean)):
            item = {'layer': i, 'groups': {}}
            for group, mask in groups.items():
                if not bool(mask.any()):
                    continue
                item['groups'][group] = {label: change(clean[i][j][:, :, mask], target[i][j][:, :, mask])
                                         for j, label in enumerate(['K', 'V'])}
            layers.append(item)
        return layers

    def kv_alignment(clean, digital, rendered, groups, rendered_reference=None):
        if rendered_reference is None:
            rendered_reference = clean
        layers = []
        for i in range(len(clean)):
            item = {'layer': i, 'groups': {}}
            for group, mask in groups.items():
                if not bool(mask.any()):
                    continue
                item['groups'][group] = {}
                for j, label in enumerate(['K', 'V']):
                    a = digital[i][j][:, :, mask].float()-clean[i][j][:, :, mask].float()
                    b = rendered[i][j][:, :, mask].float()-rendered_reference[i][j][:, :, mask].float()
                    energy = float(a.square().sum())
                    gain = float((a*b).sum())/energy if energy>1e-12 else None
                    item['groups'][group][label] = {'delta_cosine': cosine(a, b),
                        'norm_ratio': norm(b)/max(norm(a), 1e-12), 'parallel_gain': gain,
                        'orthogonal_ratio': norm(b-gain*a)/max(norm(a), 1e-12) if gain is not None else None}
            layers.append(item)
        return layers

    def attn_compare(clean, target, groups):
        layers = []
        for i in sorted(clean):
            co, ca = clean[i]
            po, pa = target[i]
            masses = {g: float((pa[..., :len(mask)][..., mask].sum(-1)-
                               ca[..., :len(mask)][..., mask].sum(-1)).mean())
                      for g, mask in groups.items() if bool(mask.any())}
            layers.append({'layer': i, 'attention_l1': float((pa-ca).abs().sum(-1).mean()),
                           'attention_output_relative_norm': change(co, po)['relative_norm'],
                           'prefix_mass_difference': masses})
        return layers

    start = time.monotonic()
    nframes = len(pool['steps']) if not args.frames else min(args.frames, len(pool['steps']))
    try:
        with torch.no_grad():
            for k in range(nframes):
                cb, cw = pool['clean_base'][k], pool['clean_wrist'][k]
                co = observation(cb, cw, k)
                cp = T.prefix_and_cache(model, device, co)
                valid = cp[1][0].bool()
                image_names = list(co.images)
                lang_len = co.tokenized_prompt.shape[1]
                image_len = (len(valid)-lang_len)//len(image_names)
                assert image_len*len(image_names)+lang_len == len(valid)
                groups = {'all_valid': valid}
                for i, name in enumerate(image_names):
                    mask = torch.zeros_like(valid)
                    mask[i*image_len:(i+1)*image_len] = valid[i*image_len:(i+1)*image_len]
                    groups[name] = mask
                language = torch.zeros_like(valid)
                language[image_len*len(image_names):] = valid[image_len*len(image_names):]
                groups['language_valid'] = language
                if layout is None:
                    layout = {'cache_type': type(cp[2]).__name__, 'layers': len(cp[2]),
                              'shape_per_layer': [list(cp[2][i][0].shape) for i in range(len(cp[2]))],
                              'groups_valid_tokens': {g: int(m.sum()) for g,m in groups.items()},
                              'image_tokens_per_view': image_len}
                    print('[layout]', json.dumps(layout), flush=True)
                cv, ca = [], None
                for j, noise in enumerate(noises):
                    v, a = first(cp, noise, capture=(j==0))
                    cv.append(v)
                    if j==0:
                        ca = a
                clean_action = integrate(cp, noises[0])
                if k==0:
                    repeat = T.prefix_and_cache(model, device, observation(cb, cw, k))
                    assert all(torch.equal(a,b) for x,y in zip(cp[2],repeat[2]) for a,b in zip(x,y)), 'KV repeat failed'
                    repeat_v, _ = first(repeat, noises[0])
                    assert torch.equal(cv[0], repeat_v), 'Velocity repeat failed'
                    direct = T.velocity_at_steps(model, device, observation(cb, cw, k), noises[0])[0]
                    assert torch.equal(cv[0], direct), 'Cache path differs from original velocity function'
                    del repeat
                for tag in pool['patch_tags'].tolist():
                    images = {
                        'digital': (pool[tag+'__digital'][k], cw),
                        'render_base': (pool[tag+'__optimized_render_base'][k], cw),
                        'render_both': (pool[tag+'__optimized_render_base'][k], pool[tag+'__optimized_render_wrist'][k]),
                        'random_digital': (pool[tag+'__random_digital'][k], cw),
                        'random_render_base': (pool[tag+'__random_render_base'][k], cw),
                        'random_render_both': (pool[tag+'__random_render_base'][k], pool[tag+'__random_render_wrist'][k])}
                    packs, velocities = {}, {}
                    row = {'patch': tag, 'frame': k, 'step': int(pool['steps'][k]), 'conditions': {}}
                    # Tokens are prefill image patches; image_len is checked above.
                    side = int(np.sqrt(image_len))
                    assert side*side == image_len and 224 % side == 0
                    block = 224//side
                    footprint = pool[tag+'__mask'][k].reshape(side,block,side,block).any((1,3)).reshape(-1)
                    patch_mask = torch.zeros_like(valid)
                    base_idx = image_names.index(T.BASE_KEY)
                    patch_mask[base_idx*image_len:(base_idx+1)*image_len] = torch.from_numpy(footprint).to(device)
                    local_groups = dict(groups, base_patch_tokens=patch_mask & valid)
                    for name, (base,wrist) in images.items():
                        obs = observation(base, wrist, k)
                        pack = T.prefix_and_cache(model, device, obs)
                        assert torch.equal(cp[0],pack[0]) and torch.equal(cp[1],pack[1]), 'State/mask differs'
                        item = {'kv_vs_clean': kv_compare(cp[2],pack[2],local_groups), 'velocity': []}
                        velocities[name] = []
                        for j, noise in enumerate(noises):
                            v, att = first(pack,noise,capture=(j==0))
                            velocities[name].append(v)
                            item['velocity'].append(velocity_metrics(v,cv[j]))
                            if j==0:
                                item['expert_attention_vs_clean'] = attn_compare(ca,att,local_groups)
                        action = integrate(pack,noises[0])-clean_action
                        item['action_norm'] = norm(action)
                        item['action_real7_norm'] = norm(action[..., :7])
                        item['action_exec7_norm'] = norm(action[:, :5, :7])
                        item['action_exec_xyz_net_norm'] = norm(action[:, :5, :3].sum(1))
                        row['conditions'][name] = item
                        packs[name] = pack
                    row['alignment'] = {name: kv_alignment(cp[2], packs['digital'][2], packs[name][2],local_groups)
                                        for name in ['render_base','render_both']}
                    # Subtract a matched random patch to separate pattern identity
                    # from the shared presence of a textured object. One random
                    # pattern is a diagnostic control, not a frequency-matched trial.
                    row['optimized_vs_random'] = {}
                    for name in ['digital','render_base','render_both']:
                        reference = 'random_'+name
                        row['optimized_vs_random'][name] = {
                            'kv': kv_compare(packs[reference][2], packs[name][2],local_groups),
                            'velocity': [velocity_metrics(v,ref) for v,ref in
                                         zip(velocities[name],velocities[reference])]}
                    row['optimized_random_alignment'] = {name: kv_alignment(
                        packs['random_digital'][2],packs['digital'][2],packs[name][2],local_groups,
                        rendered_reference=packs['random_'+name][2])
                        for name in ['render_base','render_both']}
                    # Swap across all layers to test K vs V contributions. Diagnostic,
                    # mixed caches are not physically realizable observations.
                    row['kv_swaps'] = {}
                    for name,k_source,v_source in [('digital_K_render_V','digital','render_base'),
                                                  ('render_K_digital_V','render_base','digital')]:
                        pairs = tuple((packs[k_source][2][i][0],packs[v_source][2][i][1])
                                      for i in range(len(cp[2])))
                        cache = DynamicCache.from_legacy_cache(pairs)
                        pack = (cp[0],cp[1],cache)
                        row['kv_swaps'][name] = [velocity_metrics(first(pack,noise)[0],cv[j])
                                                for j,noise in enumerate(noises)]
                    records.append(row)
                    (dest/'metrics.json').write_text(json.dumps({'layout':layout,'noise_seeds':seeds,
                        'frames_completed':k+1,'records':records,'complete':False},allow_nan=False))
                    print('[probe]',tag,'step',int(pool['steps'][k]),'L0',
                          {n:round(np.mean([v['L0'] for v in d['velocity']]),4) for n,d in row['conditions'].items()},
                          'elapsed_s',round(time.monotonic()-start),flush=True)
                    del packs, velocities, row
                del cp, ca, cv, co, attention
                attention = {}
    finally:
        for handle in handles:
            handle.remove()
    metadata = {'layout':layout,'noise_seeds':seeds,'frames_completed':nframes,'records':records,'complete':True,
                'inputs_sha256':hashlib.sha256((dest/'inputs.npz').read_bytes()).hexdigest(),
                'model_checkpoint':str(T.TORCH_CKPT),'open_loop_only':True,
                'notes':'K is post-RoPE prefix cache; valid tokens only. Attention metrics use first noise. '
                        'Actions use normalized model space and first noise; no physical displacement or ASR. '
                        'KV swaps are diagnostic mixed caches across all layers.',
                'checks':{'repeat_KV_equal':True,'repeat_velocity_equal':True,'original_velocity_path_equal':True},
                'elapsed_s':time.monotonic()-start}
    (dest/'metrics.json').write_text(json.dumps(metadata,allow_nan=False))
    print('[done]',len(records),'records',flush=True)


if __name__ == '__main__':
    main()
