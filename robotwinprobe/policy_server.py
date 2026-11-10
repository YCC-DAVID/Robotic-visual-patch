"""RoboTwin-specific pi0.5, three RGB views and14joint coordinates."""
import argparse
import dataclasses
import json
import numpy as np
import bootstrap as B
B.model_paths()


def aloha_images(images):
    """Wire format is native HWC uint8 RGB; AlohaInputs expects CHW RGB."""
    converted = {}
    for name, image in images.items():
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f'{name}: expected HWC uint8 RGB, got {image.shape} {image.dtype}')
        converted[name] = np.ascontiguousarray(image.transpose(2, 0, 1))
    return converted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8180)
    ap.add_argument('--checkpoint', default=str(B.ROOT/'checkpoints/pi05_robotwin2_pytorch'))
    ap.add_argument('--metadata', default=str(B.OUT/'setup/policy.json'))
    args = ap.parse_args()
    import torch
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.serving import websocket_policy_server
    torch.set_num_threads(4)
    cfg = C.get_config('pi05_aloha')
    cfg = dataclasses.replace(cfg,
        model=dataclasses.replace(cfg.model, action_horizon=32, pytorch_compile_mode=None),
        data=C.LeRobotAlohaDataConfig(
            assets=C.AssetsConfig(asset_id='pi0.5_clean_randomize_joint_training'),
            adapt_to_pi=True, use_delta_joint_actions=True))
    policy = PC.create_trained_policy(cfg, args.checkpoint)
    policy._model.eval()
    for parameter in policy._model.parameters(): parameter.requires_grad_(False)

    class ExplicitNoisePolicy:
        metadata = {'model': 'pi05_robotwin2', 'action_horizon': 32, 'action_dim': 14,
                    'views': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']}

        def infer(self, element):
            element = dict(element)
            noise = np.array(element.pop('noise'), dtype=np.float32, copy=True)
            assert noise.shape == (32, 32) and np.isfinite(noise).all()
            assert np.asarray(element['state']).shape == (14,)
            assert set(element['images']) == set(self.metadata['views'])
            element['images'] = aloha_images(element['images'])
            return policy.infer(element, noise=noise)

    wrapped = ExplicitNoisePolicy()
    request = {'images': {n: np.zeros((224, 224, 3), np.uint8) for n in wrapped.metadata['views']},
               'state': np.zeros(14, np.float32), 'prompt': 'place the container on the plate',
               'noise': np.random.RandomState(20261006).normal(size=(32,32)).astype(np.float32)}
    first = wrapped.infer(request)['actions']; second = wrapped.infer(request)['actions']
    assert first.shape == (32, 14) and np.isfinite(first).all()
    assert np.array_equal(first, second), 'explicit noise inference must repeat'
    meta = {**wrapped.metadata, 'checkpoint': args.checkpoint, 'noise_shape': [32,32],
            'adapt_to_pi': True, 'delta_joint_actions': True, 'repeat_equal': True,
            'note': 'Settings follow the checkpoint author model card; clean rollout validation still required.'}
    B.Path(args.metadata).write_text(json.dumps(meta, indent=2)+'\n')
    print('[serve] ready port='+str(args.port), flush=True)
    websocket_policy_server.WebsocketPolicyServer(wrapped, host='127.0.0.1', port=args.port).serve_forever()


if __name__ == '__main__': main()
