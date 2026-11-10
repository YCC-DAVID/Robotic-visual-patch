"""Local policy service with caller-supplied, reproducible action noise."""
import argparse
import dataclasses
import drift_patch_train as T
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8144)
    args = ap.parse_args()
    import torch
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.serving import websocket_policy_server
    torch.set_num_threads(4)
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg, T.TORCH_CKPT)
    policy._model.eval()
    for p in policy._model.parameters():
        p.requires_grad_(False)

    class ExplicitNoisePolicy:
        metadata = policy.metadata

        def infer(self, element):
            element = dict(element)
            noise = np.array(element.pop('noise'), dtype=np.float32, copy=True)
            assert noise.shape == (T.AH, T.AD) and np.isfinite(noise).all()
            return policy.infer(element, noise=noise)

    wrapper = ExplicitNoisePolicy()
    dummy = {'observation/image': np.zeros((224, 224, 3), np.uint8),
             'observation/wrist_image': np.zeros((224, 224, 3), np.uint8),
             'observation/state': np.zeros(8, np.float32),
             'prompt': 'put the bowl on the plate',
             'noise': np.random.RandomState(20261006).standard_normal((T.AH, T.AD)).astype(np.float32)}
    a = wrapper.infer(dummy)['actions']
    b = wrapper.infer(dummy)['actions']
    assert np.array_equal(a, b), 'repeat action inference differs'
    print('[serve] ready port={} repeat_equal=True'.format(args.port), flush=True)
    websocket_policy_server.WebsocketPolicyServer(wrapper, host='127.0.0.1', port=args.port).serve_forever()


if __name__ == '__main__':
    main()
