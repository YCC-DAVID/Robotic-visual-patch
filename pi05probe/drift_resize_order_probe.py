"""Paired, fixed-noise open-loop evaluation of native-resolution paste order."""
import argparse
import dataclasses
import json
import time

import drift_patch_train as T  # model environment/path setup
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='drift_resize_order_native')
    ap.add_argument('--patches', default='drift,band32')
    ap.add_argument('--out', default='drift_resize_order_probe')
    ap.add_argument('--serve', type=int, default=0, help='Optionally serve this policy afterward with explicit noise support')
    args = ap.parse_args()
    import torch
    import jax
    from PIL import Image, ImageDraw
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    from drift_resize_order import compose_variants, resize_rgb

    pool = np.load(T.OUT / (args.pool + '.npz'), allow_pickle=True)
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg, T.TORCH_CKPT)
    model, device = policy._model, policy._pytorch_device
    for p in model.parameters():
        p.requires_grad_(False)
    noise_np = np.random.RandomState(T.SEED_NOISE).standard_normal((T.AH, T.AD)).astype(np.float32)
    noise = torch.from_numpy(noise_np).to(device)[None]
    prompt = str(pool['prompt'])

    def obs(image, k):
        inputs = policy._input_transform({'observation/image': image,
                                          'observation/wrist_image': pool['clean_wrist224'][k],
                                          'observation/state': pool['clean_state8'][k], 'prompt': prompt})
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None], inputs)
        return M.Observation.from_dict(inputs)

    patch_tags = args.patches.split(',')
    patches = {tag: np.clip(np.load(T.OUT / ('drift_patch_' + tag + '.npz'))['P'].transpose(1, 2, 0)
                            * 255, 0, 255).astype(np.uint8) for tag in patch_tags}
    random_patch = np.random.RandomState(20260902).randint(0, 256, (224, 224, 3), dtype=np.uint8)
    rows = {tag: {} for tag in patch_tags}
    start = time.monotonic()
    with torch.no_grad():
        for k, clean256 in enumerate(pool['clean_img256']):
            clean224 = resize_rgb(clean256, 224)
            baseline = obs(clean224, k)
            vc = T.velocity_at_steps(model, device, baseline, noise.clone())
            ac = T.integrate_actions(model, device, obs(clean224, k), noise.clone())
            if k == 0:
                repeat = T.velocity_at_steps(model, device, obs(clean224, k), noise.clone())
                delta = float((vc - repeat).abs().max())
                assert delta == 0, 'Non-deterministic forward: {}'.format(delta)
                print('[repeat] max delta={}'.format(delta), flush=True)
            for tag, patch in patches.items():
                images = compose_variants(clean256, pool['masks256'][k], pool['masks224'][k], patch)
                randoms = compose_variants(clean256, pool['masks256'][k], pool['masks224'][k], random_patch)
                images.update(rand224=randoms['post224'], rand_pre256=randoms['pre256_native'])
                if k == 0:
                    names = ['clean'] + list(images)
                    canvas = Image.new('RGB', (224 * 4, 250 * 2), 'white')
                    draw = ImageDraw.Draw(canvas)
                    for i, name in enumerate(names):
                        im = clean224 if name == 'clean' else images[name]
                        x, y = (i % 4) * 224, (i // 4) * 250
                        canvas.paste(Image.fromarray(im), (x, y + 24)); draw.text((x + 5, y + 5), name, fill='black')
                    canvas.save(T.OUT / (args.out + '_' + tag + '.png'))
                for name, image in images.items():
                    v = T.velocity_at_steps(model, device, obs(image, k), noise.clone())
                    a = T.integrate_actions(model, device, obs(image, k), noise.clone())
                    item = rows[tag].setdefault(name, {'L0': [], 'action_norm': [], 'action_norm7': [], 'pixel_mae': []})
                    item['L0'].append(float(((v - vc) ** 2).sum()))
                    item['action_norm'].append(float((a - ac).norm()))
                    item['action_norm7'].append(float((a[..., :7] - ac[..., :7]).norm()))
                    item['pixel_mae'].append(float(np.abs(image.astype(np.float32) - images['post224']).mean() / 255))
            print('[probe] {}/{} elapsed={:.1f}s'.format(k + 1, len(pool['ep_id']), time.monotonic() - start), flush=True)
    summary = {}
    lines = ['Paired native-256 paste-order experiment',
             'Frames: {} initial states, after 10 settling steps; noise seed {}'.format(len(pool['ep_id']), T.SEED_NOISE),
             'No illumination/color transform. All variants quantized to uint8 before identical model preprocessing.',
             'L0 and action differences are open-loop metrics, not attack success rates.']
    for tag, conditions in rows.items():
        summary[tag] = {}
        lines += ['', tag, '{:18s} {:>10s} {:>10s} {:>10s} {:>10s}'.format('condition', 'L0 mean', '/post224', 'action med', '7dim med')]
        ref = np.mean(conditions['post224']['L0'])
        for name, item in conditions.items():
            s = {'L0_mean': float(np.mean(item['L0'])), 'L0_median': float(np.median(item['L0'])),
                 'L0_ratio': float(np.mean(item['L0']) / ref),
                 'action_median': float(np.median(item['action_norm'])),
                 'action7_median': float(np.median(item['action_norm7'])),
                 'pixel_mae_mean': float(np.mean(item['pixel_mae']))}
            summary[tag][name] = s
            lines.append('{:18s} {:10.4f} {:10.1%} {:10.4f} {:10.4f}'.format(name, s['L0_mean'], s['L0_ratio'],
                                                                                   s['action_median'], s['action7_median']))
    dest = T.OUT / (args.out + '.json')
    dest.write_text(json.dumps({'summary': summary, 'per_frame': rows, 'pool': args.pool,
                               'noise_seed': T.SEED_NOISE, 'ep_id': pool['ep_id'].tolist(),
                               'notes': lines[:4]}, indent=2))
    (T.OUT / (args.out + '.txt')).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    if args.serve:
        from openpi.serving import websocket_policy_server

        class ExplicitNoisePolicy:
            metadata = policy.metadata

            def infer(self, element):
                element = dict(element)
                input_noise = np.array(element.pop('noise'), copy=True)
                return policy.infer(element, noise=input_noise)

        print('[serve] port={}'.format(args.serve), flush=True)
        websocket_policy_server.WebsocketPolicyServer(ExplicitNoisePolicy(), host='127.0.0.1', port=args.serve).serve_forever()


if __name__ == '__main__':
    main()
