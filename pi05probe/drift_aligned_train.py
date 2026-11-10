"""Our render-proxy + texture EOT at the frozen DRIFT pi0.5 placement.

The checkpoint, observation pool, 64px parameter count, batch draws, action
noise and PGD budget follow the companion DRIFT run. This is an aligned
adaptation, not an unchanged execution of the older Torch/Q128 trainer.
"""
import argparse
import csv
import importlib
import json
import os
import re
from pathlib import Path
import sys
import time

import numpy as np
from drift_paper_common import ROOT, DRIFT, bootstrap, dump, now, read, sha


def observation_with_images(clean, images, model_module):
    return model_module.Observation(images=images, image_masks=clean.image_masks, state=clean.state,
        tokenized_prompt=clean.tokenized_prompt, tokenized_prompt_mask=clean.tokenized_prompt_mask,
        token_ar_mask=clean.token_ar_mask, token_loss_mask=clean.token_loss_mask)


def train(args):
    bootstrap()
    os.environ['OPENPI_ROOT'] = str(ROOT / 'third_party/openpi')
    sys.path.insert(0, str(DRIFT))
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from PIL import Image
    from openpi.models import model as M
    from openpi.policies import policy_config
    from openpi.training import config
    from utils.attack import preprocess_obs_batch, prefix_kv_cache, suffix_velocity, WRIST_KEY
    from drift_aligned_proxy import CalibrationPool, EOT_RANGES, eot_draws, proxy_gate, render_one

    root, source = Path(args.root).resolve(), Path(args.source_root).resolve()
    dest = root / 'train'
    dest.mkdir(parents=True, exist_ok=True)
    assert args.steps == 500 and args.batch == 8 and args.lr == .01 and args.seed == 0
    assert os.environ.get('CUDA_VISIBLE_DEVICES', '').startswith('GPU-')
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu', 'one preflighted assigned GPU required'
    frozen = read(source / 'protocol.json')
    assert frozen['model_config'] == 'pi05_libero'
    assert Path(args.checkpoint).resolve() == Path(frozen['checkpoint']).resolve()
    assert frozen['training']['patch_hw'] == [64, 64]
    assert read(source / 'geometry/complete.json')['complete'] is True
    for name, digest in frozen['source_sha256'].items():
        assert sha(ROOT / 'pi05probe' / name) == digest
    pool = CalibrationPool(root)
    raw = np.load(source / 'pool.npy', allow_pickle=True).item()
    assert pool.n == len(raw['observation/image']) == len(raw['observation/wrist_image'])
    assert pool.n == 642, 'use the entire frozen 642-frame pi0.5 pool'
    batches = np.random.default_rng(args.seed).integers(0, pool.n, size=(args.steps, args.batch))
    draws = eot_draws(args.steps, args.batch, args.eot_seed)
    plans = dest / 'plans.npz'
    if plans.exists():
        with np.load(plans, allow_pickle=False) as old:
            assert np.array_equal(old['batch_indices'], batches) and np.array_equal(old['eot_draws'], draws)
    else:
        np.savez_compressed(plans, batch_indices=batches, eot_draws=draws)
    metadata = dict(schema='wam-drift-position-render-eot-jax-v1', complete=False,
        model_config='pi05_libero', backend='JAX', checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=frozen['checkpoint_sha256'], source_protocol_sha256=sha(source / 'protocol.json'),
        source_pool_sha256=sha(source / 'pool.npy'), frames=pool.n,
        geometry_sha256=sha(source / 'geometry/geometry.json'),
        patch_hw=[64, 64], wrist_top_left_yx=[160, 80],
        steps=args.steps, batch=args.batch, lr=args.lr, seed=args.seed, initialization='constant gray0.5',
        objective='first t=1 same-noise velocity difference; sum actiondim, mean batch/horizon (positive rescaling of old SSE)',
        action_horizon=10, action_dim=32, action_noise='fresh official JAX seed0 split per PGD step',
        batch_sampling='official default_rng0.integers with replacement; identical ordered original pool',
        texture_eot=dict(ranges=EOT_RANGES, seed=args.eot_seed, one_draw_per_frame=True, shared_between_views=True,
                         source_sha256=sha(ROOT / 'robotwinprobe/texture_eot.py')),
        plans_sha256=sha(plans), png_quantization='floor(clip(Q,0,1)*255), same as DRIFT',
        capture_manifest_sha256=pool.manifest_sha256, capture_protocol_sha256=pool.protocol_sha256,
        proxy_rmse_max=args.proxy_rmse_max, native_texture_samples=1,
        source_sha256={name: sha(ROOT / 'pi05probe' / name) for name in ['drift_aligned_proxy.py', 'drift_aligned_train.py']},
        differences_from_legacy_ours=['official JAX checkpoint instead of Torch conversion',
            'Q64 instead of Q128', 'universal ten-task patch instead of per-task training',
            'DRIFT batch sampling and fresh action noise instead of old stratified sampling/fixed noise'],
        limitation='calibrated differentiable approximation; final attack success requires independent native rollout')
    config_path = dest / 'config.json'
    if config_path.exists():
        old = read(config_path)
        assert args.resume, 'refuse to overwrite training metadata; use --resume'
        assert {k: v for k, v in old.items() if k != 'complete'} == {k: v for k, v in metadata.items() if k != 'complete'}
        assert not old['complete'], 'training already complete'
    else:
        dump(config_path, metadata)

    started = time.monotonic()
    gate = proxy_gate(pool, max_rmse=args.proxy_rmse_max, samples=1)
    dump(dest / 'proxy_gate.json', gate)
    assert gate['complete']
    print('[proxy gate]', json.dumps({k: v for k, v in gate.items() if k != 'views'}), flush=True)
    if args.proxy_check_only:
        return

    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), args.checkpoint)
    model = policy._model
    assert not policy._is_pytorch_model and model.action_horizon == 10 and model.action_dim == 32
    graphdef, model_state = nnx.split(model)

    def observations(indices):
        prompts = raw.get('prompts', raw.get('prompt'))
        selected = [str(prompts[i]) for i in indices] if isinstance(prompts, (list, np.ndarray)) else [str(prompts)] * len(indices)
        return preprocess_obs_batch(policy._input_transform, {
            'observation/image': raw['observation/image'][indices],
            'observation/wrist_image': raw['observation/wrist_image'][indices],
            'observation/state': raw['observation/state'][indices], 'prompts': selected}, len(indices))

    def loss_fn(q, state, key, clean, calibration, transforms):
        m = nnx.merge(graphdef, state)
        key_noise, _ = jax.random.split(key)
        noise = jax.random.normal(key_noise, (args.batch, 10, 32))
        clean_pp = M.preprocess_observation(None, clean, train=False)
        rendered = jax.vmap(lambda c, d: render_one(q, c, d, samples=1))(calibration, transforms)
        images = dict(clean.images)
        images['base_0_rgb'] = rendered[:, 0] * 2. - 1.
        images[WRIST_KEY] = rendered[:, 1] * 2. - 1.
        attacked = observation_with_images(clean, images, M)
        attacked_pp = M.preprocess_observation(None, attacked, train=False)
        mask_c, kv_c = prefix_kv_cache(m, clean_pp)
        mask_a, kv_a = prefix_kv_cache(m, attacked_pp)
        t = jnp.ones((args.batch,))
        clean_v = suffix_velocity(m, clean_pp, noise, t, mask_c, kv_c)
        attacked_v = suffix_velocity(m, attacked_pp, noise, t, mask_a, kv_a)
        divergence = jnp.mean(jnp.sum(jnp.square(attacked_v - clean_v), axis=-1))
        return -divergence, divergence

    gradient = jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=True))
    q = jnp.full((64, 64, 3), .5, dtype=jnp.float32)
    rng = jax.random.key(args.seed)
    losses, grad_norms = [], []
    checkpoints = sorted(p for p in dest.glob('checkpoint_*.npz')
                         if re.fullmatch(r'checkpoint_\d{4}\.npz', p.name))
    if args.resume and checkpoints:
        with np.load(checkpoints[-1], allow_pickle=False) as saved:
            q = jnp.asarray(saved['Q'])
            losses, grad_norms = saved['losses'].tolist(), saved['gradient_norms'].tolist()
            rng = jax.random.wrap_key_data(jnp.asarray(saved['rng_key_data']))
            assert int(saved['completed_steps']) == len(losses) == len(grad_norms)
        print('[resume]', len(losses), flush=True)

    def save(done):
        array = np.asarray(q)
        path = dest / ('checkpoint_%04d.npz' % done)
        temporary = path.with_name(path.stem + '.tmp.npz')
        np.savez_compressed(temporary, Q=array, losses=np.asarray(losses), gradient_norms=np.asarray(grad_norms),
                            completed_steps=np.asarray(done), rng_key_data=np.asarray(jax.random.key_data(rng)))
        temporary.replace(path)
        dump(dest / 'progress.json', dict(complete=False, completed_steps=done, target_steps=args.steps,
             gradient_frame_forwards=done * args.batch, last_divergence=losses[-1] if losses else None,
             elapsed_seconds=time.monotonic() - started, updated=now()))

    for step in range(len(losses), args.steps):
        indices = batches[step]
        clean = observations(indices)
        calibration = {k: jnp.asarray(v) for k, v in pool.batch(indices).items()}
        rng, key = jax.random.split(rng)
        (objective, divergence), g = gradient(q, model_state, key, clean, calibration, jnp.asarray(draws[step]))
        value, norm = float(divergence), float(jnp.linalg.norm(g))
        assert np.isfinite(value) and np.isfinite(norm) and norm > 0., ('invalid gradient', step, value, norm)
        q = jnp.clip(q - args.lr * jnp.sign(g), 0., 1.)
        losses.append(value)
        grad_norms.append(norm)
        if (step + 1) % 25 == 0 or step + 1 == args.steps:
            save(step + 1)
        if step % 20 == 0 or step + 1 == args.steps:
            print('[train]', step + 1, 'divergence', value, 'grad_norm', norm,
                  'minutes', (time.monotonic() - started) / 60., flush=True)

    array = (np.asarray(q) * 255).clip(0, 255).astype(np.uint8)
    out = dest / 'ours_pi05_64px.npy'
    np.save(out, array)
    Image.fromarray(array).save(out.with_suffix('.png'))
    with (dest / 'ours_pi05_64px.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'divergence', 'gradient_norm'])
        writer.writerows(zip(range(args.steps), losses, grad_norms))
    assert len(losses) == 500 and np.isfinite(losses).all() and np.isfinite(grad_norms).all() and np.all(np.asarray(grad_norms) > 0)
    assert np.array_equal(np.load(out, allow_pickle=False), np.asarray(Image.open(out.with_suffix('.png')).convert('RGB')))
    result = dict(complete=True, completed_steps=500, gradient_frame_forwards=4000, updated=now(),
                  patch_sha256=sha(out), png_sha256=sha(out.with_suffix('.png')), plans_sha256=sha(plans),
                  source_pool_sha256=metadata['source_pool_sha256'], proxy_gate_sha256=sha(dest / 'proxy_gate.json'),
                  elapsed_seconds=time.monotonic() - started, finite_losses=True, nonzero_finite_gradients=True)
    dump(dest / 'complete.json', result)
    dump(dest / 'progress.json', result)
    metadata['complete'] = True
    dump(config_path, metadata)
    print('[complete]', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eot-seed', type=int, default=2026102901)
    parser.add_argument('--proxy-rmse-max', type=float, default=.02)
    parser.add_argument('--proxy-check-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    opts = parser.parse_args()
    try:
        train(opts)
    except BaseException as error:
        dump(Path(opts.root) / 'train/error.json', dict(error=repr(error), updated=now(), pid=os.getpid()))
        raise
