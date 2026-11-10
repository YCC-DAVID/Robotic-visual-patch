"""JAX official-checkpoint service and unmodified upstream DRIFT training entry."""
import argparse
import os
from pathlib import Path
import sys

from drift_pi0_common import (ROOT, DRIFT, bootstrap, dump, now, read, sha,
                              MODEL_CONFIG, PATCH_SIZE, PATCH_Y, PATCH_X, ACTION_HORIZON, ACTION_DIM)


def serve(args):
    import jax
    import numpy as np
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from main_method_job_runtime import patch_finite_websocket_service
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu', 'exactly one assigned GPU is required'
    policy = policy_config.create_trained_policy(config.get_config(MODEL_CONFIG), args.checkpoint)
    assert not policy._is_pytorch_model, 'official DRIFT reproduction requires JAX checkpoint'
    assert policy._model.action_horizon == ACTION_HORIZON and policy._model.action_dim == ACTION_DIM

    class ExplicitNoise:
        def infer(self, element):
            element = dict(element)
            noise = np.asarray(element.pop('noise'), dtype=np.float32)
            assert noise.shape == (ACTION_HORIZON, ACTION_DIM) and np.isfinite(noise).all()
            return policy.infer(element, noise=noise)

    wrapper = ExplicitNoise()
    dummy = {'observation/image': np.zeros((224, 224, 3), np.uint8),
             'observation/wrist_image': np.zeros((224, 224, 3), np.uint8),
             'observation/state': np.zeros(8, np.float32), 'prompt': 'pick up the black bowl',
             'noise': np.random.default_rng(0).standard_normal((ACTION_HORIZON, ACTION_DIM)).astype(np.float32)}
    a, b = wrapper.infer(dummy)['actions'], wrapper.infer(dummy)['actions']
    assert np.array_equal(a, b) and np.isfinite(a).all(), 'same-noise repeat inference differs'
    dump(Path(args.root) / 'services' / ('server_' + str(args.port) + '.json'),
         dict(pid=os.getpid(), started=now(), checkpoint=args.checkpoint, backend='JAX', model_config=MODEL_CONFIG,
              action_horizon=ACTION_HORIZON, action_dim=ACTION_DIM,
              devices=[str(d) for d in jax.devices()], repeat_equal=True,
              cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    patch_finite_websocket_service(600)
    websocket_policy_server.WebsocketPolicyServer(wrapper, host='127.0.0.1', port=args.port,
        metadata=dict(policy.metadata, backend='JAX', explicit_noise=True)).serve_forever()


def train(args):
    import runpy
    root = Path(args.root)
    out = root / 'train' / 'drift_pi0_32px.npy'
    assert not out.exists(), 'refuse to overwrite a trained patch'
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ['OPENPI_ROOT'] = str(ROOT / 'third_party/openpi')
    sys.path.insert(0, str(DRIFT))
    sys.argv = [str(DRIFT / 'attack_patch.py'), '--train-config', MODEL_CONFIG,
                '--checkpoint-dir', args.checkpoint, '--libero-obs-path', str(root / 'pool.npy'),
                '--patch-size', str(PATCH_SIZE), '--batch-size', '8', '--pgd-steps', '500',
                '--alpha', '0.01', '--seed', '0', '--top-k-steps', '1',
                '--num-steps', '10', '--wrist-pos-y', str(PATCH_Y), '--wrist-pos-x', str(PATCH_X),
                '--out', str(out)]
    dump(root / 'train' / 'command.json', dict(command=sys.argv, upstream_source_sha256=sha(DRIFT / 'attack_patch.py'),
         pool_sha256=sha(root / 'pool.npy'), backend='JAX', source_modified=False))
    runpy.run_path(str(DRIFT / 'attack_patch.py'), run_name='__main__')
    import numpy as np
    patch = np.load(out, allow_pickle=False)
    assert patch.shape == (PATCH_SIZE, PATCH_SIZE, 3) and patch.dtype == np.uint8
    assert len(out.with_suffix('.csv').read_text().splitlines()) == 501
    import csv
    losses = list(csv.DictReader(out.with_suffix('.csv').open()))
    assert [int(r['step']) for r in losses] == list(range(500))
    assert np.isfinite([float(r['divergence']) for r in losses]).all()
    from PIL import Image
    assert np.array_equal(patch, np.asarray(Image.open(out.with_suffix('.png')).convert('RGB')))
    dump(root / 'train' / 'complete.json', dict(complete=True, iterations=500,
         patch_sha256=sha(out), png_sha256=sha(out.with_suffix('.png')), pool_sha256=sha(root / 'pool.npy')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['serve', 'train'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--port', type=int, default=18681)
    args = ap.parse_args()
    bootstrap()
    (serve if args.mode == 'serve' else train)(args)
