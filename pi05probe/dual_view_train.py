"""E26 same-budget shared-Q base/both/frequency-routed training."""
import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path
import drift_patch_train as T
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--fit', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=.01)
    ap.add_argument('--eot', action='store_true')
    ap.add_argument('--camera-mode', required=True, choices=['base','both','routed'])
    ap.add_argument('--wrist-calibration', required=True)
    ap.add_argument('--wrist-fit', required=True)
    ap.add_argument('--phase-jitter', type=float, default=.25, help='Native renderer pixel sampling phase, not patch placement')
    ap.add_argument('--color-jitter', type=float, default=.05)
    ap.add_argument('--lod-jitter', type=float, default=.10)
    ap.add_argument('--seed', type=int, default=20261005)
    ap.add_argument('--save-every', type=int, default=25)
    ap.add_argument('--resume', default='')
    ap.add_argument('--parameter-size', type=int, choices=[32, 128], default=128)
    args = ap.parse_args()
    import torch
    import torch.nn.functional as F
    import jax
    from PIL import Image
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    from dual_view_proxy import CameraProxy, base_gradient_route
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    source, fit = Path(args.calibration), Path(args.fit)
    dest = Path(args.out); dest.mkdir(parents=True, exist_ok=True)
    assert args.resume or not (dest/'config.json').exists(), 'refuse to overwrite a training run'
    archive = np.load(source, allow_pickle=False)
    keys = ['train_mask', 'validation_mask', 'ep_id', 'patch_m', 'cell_world', 'cell_xy',
            'clean_base', 'clean_wrist', 'state8', 'prompt']
    d = {k: archive[k] for k in keys}
    archive.close()
    train = np.flatnonzero(d['train_mask'])
    validation = np.flatnonzero(d['validation_mask'])
    assert len(train) > 0 and len(validation) > 0
    assert set(d['ep_id'][train].tolist()).isdisjoint(set(d['ep_id'][validation].tolist()))
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg, T.TORCH_CKPT)
    model, device = policy._model, policy._pytorch_device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert not args.eot and args.parameter_size == 128
    proxy = CameraProxy(source, device, fit)
    wrist_proxy = CameraProxy(args.wrist_calibration, device, args.wrist_fit)
    gradient_log = []
    frame_gradients = {}
    noise_np = np.random.RandomState(T.SEED_NOISE).standard_normal((T.AH, T.AD)).astype(np.float32)
    noise = torch.from_numpy(noise_np)[None].to(device)
    rng_batch = np.random.RandomState(args.seed)
    rng_eot = np.random.RandomState(args.seed+1)
    metadata = {'schema': 'wam-world-texture-train-v1', 'complete': False,
        'mode': args.camera_mode, 'iters': args.iters, 'batch': args.batch, 'lr': args.lr,
        'seed_batch': args.seed, 'seed_eot': args.seed+1, 'noise_seed': T.SEED_NOISE,
        'noise_sha256': hashlib.sha256(noise_np.tobytes()).hexdigest(),
        'init': 'constant gray .5', 'parameter_shape': [1, 3, args.parameter_size, args.parameter_size], 'texture_shape': [1, 3, 128, 128],
        'frequency_parameterization': 'full' if args.parameter_size == 128 else 'band32: latent32 bilinear align_corners=False to world Q128',
        'parameter_orientation': 'exact uploaded PNG RGB orientation',
        'wrist_calibration_path': str(Path(args.wrist_calibration).resolve()), 'wrist_calibration_sha256': sha(args.wrist_calibration), 'wrist_fit_sha256': sha(args.wrist_fit), 'wrist_fit_path': str(Path(args.wrist_fit).resolve()),
        'calibration_path': str(source.resolve()), 'calibration_sha256': sha(source),
        'fit_path': str(fit.resolve()), 'fit_sha256': sha(fit),
        'source_sha256': {Path(p).name: sha(p) for p in [__file__, T.PROBE/'dual_view_proxy.py']},
        'checkpoint': str(T.TORCH_CKPT), 'patch_m': float(d['patch_m']), 'cell_world': d['cell_world'].tolist(),
        'train_frames': len(train), 'validation_frames': len(validation),
        'train_episodes': sorted(set(d['ep_id'][train].tolist())),
        'validation_episodes': sorted(set(d['ep_id'][validation].tolist())),
        'eot_ranges': {'phase_native_px': args.phase_jitter if args.eot else 0,
                       'relative_rgb_gain': args.color_jitter if args.eot else 0,
                       'lod_bias': args.lod_jitter if args.eot else 0},
        'notes': 'Shared Q128, per-state analytic float64 geometry and calibrated full imaging. '
                 'base: clean wrist. both: both actual-proxy views. routed: identical joint forward; '
                 'base gradient is the adjoint of avgpool4+bilinear128, wrist unrestricted. '
                 'PGD sign is applied after summing; final Q is NOT strictly bandlimited. '
                 'Same500x8,155train/32val,batches/noise; no EOT; all modes freshly trained.'}
    (dest/'config.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')

    def obs(k):
        x = policy._input_transform({'observation/image': d['clean_base'][k],
            'observation/wrist_image': d['clean_wrist'][k],
            'observation/state': d['state8'][k], 'prompt': str(d['prompt'])})
        x = jax.tree.map(lambda a: torch.from_numpy(np.array(a)).to(device)[None], x)
        return M.Observation.from_dict(x)

    clean = {}
    print('[cache]', len(train)+len(validation), 'clean frames', flush=True)
    with torch.no_grad():
        first = int(train[0])
        a = T.velocity_at_steps(model, device, obs(first), noise.clone())
        b = T.velocity_at_steps(model, device, obs(first), noise.clone())
        assert torch.equal(a, b), 'same-input forward must be deterministic'
        for i, k in enumerate(np.concatenate([train, validation])):
            clean[int(k)] = T.velocity_at_steps(model, device, obs(int(k)), noise.clone()).detach()
            if i % 50 == 0:
                print('[cache]', i, flush=True)
    Q = torch.full((1, 3, args.parameter_size, args.parameter_size), .5, device=device, requires_grad=True)
    def texture():
        return Q if args.parameter_size == 128 else F.interpolate(Q, (128, 128), mode='bilinear', align_corners=False)
    history, batches, transforms = [], [], []
    start_it = 0
    if args.resume:
        previous = np.load(args.resume, allow_pickle=False)
        assert str(previous['calibration_sha256']) == metadata['calibration_sha256']
        Q.data.copy_(torch.from_numpy(previous['latent']).to(device)[None])
        history = previous['history'].tolist()
        batches = previous['batches'].tolist()
        transforms = previous['transforms'].tolist()
        start_it = len(history)
        # Replay RNG draws exactly, including EOT calls per sampled frame.
        for it in range(start_it):
            old = rng_batch.choice(train, min(args.batch, len(train)), replace=False)
            assert np.array_equal(old, batches[it])
            if args.eot:
                for _ in old:
                    rng_eot.uniform(-args.phase_jitter, args.phase_jitter, 2)
                    rng_eot.uniform(-args.lod_jitter, args.lod_jitter)
                    rng_eot.uniform(1-args.color_jitter, 1+args.color_jitter, 3)

    def loss(k, jitter):
        o = obs(k)
        q = texture()
        qb = q * 1
        qw = q * 1
        if torch.is_grad_enabled():
            qb.register_hook(lambda g: frame_gradients.__setitem__('base', g.detach().clone()))
            if args.camera_mode != 'base':
                qw.register_hook(lambda g: frame_gradients.__setitem__('wrist', g.detach().clone()))
        b = proxy(base_gradient_route(qb) if args.camera_mode == 'routed' else qb, k)
        o.images[T.BASE_KEY] = b*2-1
        if args.camera_mode != 'base':
            assert 'left_wrist_0_rgb' in o.images
            o.images['left_wrist_0_rgb'] = wrist_proxy(qw, k)*2-1
        sampled = [0., 0., 0., 1., 1., 1.]
        velocity = T.velocity_at_steps(model, device, o, noise.clone())
        return (velocity-clean[k]).square().sum(), sampled

    def evaluate(ids):
        with torch.no_grad():
            return [float(loss(int(k), False)[0]) for k in ids]

    diag = train[np.linspace(0, len(train)-1, min(8, len(train)), dtype=int)]
    initial = evaluate(diag)
    print('[init]', np.mean(initial), 'train_diag', flush=True)
    started = time.monotonic()

    def save(completed):
        q = texture().detach().cpu().numpy()[0]
        latent = Q.detach().cpu().numpy()[0]
        image = np.clip(np.rint(q.transpose(1, 2, 0)*255), 0, 255).astype(np.uint8)
        Image.fromarray(image).save(dest/f'texture_{completed:04d}.png')
        np.savez_compressed(dest/f'checkpoint_{completed:04d}.npz', Q=q, latent=latent, parameter_size=args.parameter_size, history=np.array(history),
            batches=np.array(batches), transforms=np.array(transforms),
            calibration_sha256=metadata['calibration_sha256'], fit_sha256=metadata['fit_sha256'],
            noise=noise_np, patch_m=d['patch_m'], cell_world=d['cell_world'], cell_xy=d['cell_xy'])
        progress = {'complete': completed == args.iters, 'completed_iters': completed,
                    'elapsed_training_s': time.monotonic()-started,
                    'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30,
                    'last_batch_L0': history[-1] if history else None}
        (dest/'progress.json').write_text(json.dumps(progress, indent=2, allow_nan=False)+'\n')

    for it in range(start_it, args.iters):
        batch = rng_batch.choice(train, min(args.batch, len(train)), replace=False)
        Q.grad = None
        total, drawn = 0., []
        gb = torch.zeros_like(Q); gw = torch.zeros_like(Q)
        for k in batch:
            objective, transform = loss(int(k), args.eot)
            frame_gradients.clear()
            (-objective).backward()
            gb.add_(frame_gradients['base'])
            if 'wrist' in frame_gradients: gw.add_(frame_gradients['wrist'])
            total += float(objective.detach()); drawn.append(transform)
        if it % 25 == 0 or it == args.iters-1:
            gradient_log.append({'iteration': it+1, 'base_norm': float(gb.norm()), 'wrist_norm': float(gw.norm()),
                'cosine': float((gb*gw).sum()/(gb.norm()*gw.norm()).clamp_min(1e-20))})
        assert Q.grad is not None and torch.isfinite(Q.grad).all()
        assert float(Q.grad.norm()) > 0, 'physical texture gradient vanished'
        with torch.no_grad():
            Q.sub_(args.lr*Q.grad.sign()).clamp_(0, 1)
        history.append(total/len(batch)); batches.append(batch.tolist()); transforms.append(drawn)
        if (it+1) % args.save_every == 0 or it+1 == args.iters:
            save(it+1)
        if it == 0 or (it+1) % 5 == 0 or it+1 == args.iters:
            print('[train]', it+1, 'L0', history[-1], 'minutes', (time.monotonic()-started)/60,
                  'GiB', torch.cuda.max_memory_allocated()/2**30, flush=True)
    final_train = evaluate(diag)
    final_validation = evaluate(validation)
    result = {'complete': True, 'iterations': len(history), 'gradient_frame_forwards': sum(map(len, batches)),
              'initial_train_diag_L0': float(np.mean(initial)),
              'final_train_diag_L0': float(np.mean(final_train)),
              'final_validation_proxy_L0': float(np.mean(final_validation)),
              'validation_per_frame': final_validation, 'validation_indices': validation.tolist(),
              'train_diag_indices': diag.tolist(), 'train_diag_final_per_frame': final_train,
              'elapsed_training_s': time.monotonic()-started,
              'texture_sha256': sha(dest/f'texture_{args.iters:04d}.png'),
              'repeat_clean_equal': True, 'notes': 'Heldout proxy L0 only; real-render evaluation pending. band32 means low-resolution world-texture parameters, not a strict Fourier cutoff.'}
    (dest/'view_gradients.json').write_text(json.dumps(gradient_log, indent=2)+'\n')
    (dest/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    metadata['complete'] = True
    (dest/'config.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print('[done]', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
