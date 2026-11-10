"""Observe native attention backward without replacing kernels or changing inputs."""
import os
from pathlib import Path
import sys
ROOT = Path(os.environ.get('WAMATTACK_PROJECT_ROOT', '/shared/user65/workspace/chence/WAMattack'))
sys.path.insert(0, str(ROOT / 'robotwinprobe'))
import pi_imagewam_gradient_replay as D
import argparse
import hashlib


def server(work):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from imagewam_libero_whitebox_adapter import Backend
    import numpy as np
    work = D.guard_work(work)
    original_grad = torch.autograd.grad
    original_response = Backend.response
    original_prepare = Backend.prepare_views
    counter = [0]

    def signature(value):
        if isinstance(value, torch.Tensor):
            with torch._C._DisableTorchDispatch():
                a = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
                return dict(shape=list(value.shape), dtype=str(value.dtype),
                            sha256=hashlib.sha256(a.tobytes()).hexdigest())
        if isinstance(value, (list, tuple)):
            return [signature(v) for v in value]
        if isinstance(value, dict):
            return {str(k):signature(v) for k,v in value.items()}
        return value if value is None or isinstance(value, (bool, int, float, str)) else repr(value)

    class ObserveAttention(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.events = []
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = str(func)
            record = 'attention' in name and 'backward' in name
            inputs = signature((args, kwargs)) if record else None
            result = func(*args, **kwargs)
            if record:
                self.events.append(dict(operator=name, inputs=inputs, outputs=signature(result)))
            return result

    def traced_grad(*args, **kwargs):
        if not (work / 'trace_enabled').exists():
            return original_grad(*args, **kwargs)
        index = counter[0]
        counter[0] += 1
        observer = ObserveAttention()
        with observer:
            result = original_grad(*args, **kwargs)
        D.dump(work / 'attention_backward_trace' / f'call_{index:03d}.json',
               dict(call=index, events=observer.events, final_gradients=signature(result),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    cudnn_deterministic=torch.backends.cudnn.deterministic,
                    cudnn_benchmark=torch.backends.cudnn.benchmark,
                    torch_version=torch.__version__, cuda_version=torch.version.cuda))
        return result

    def traced_prepare(self, *args, **kwargs):
        result = original_prepare(self, *args, **kwargs)
        self._diagnostic_input_image = result[0]
        return result

    def traced_response(self, image, proprio, seed, full=False, return_intermediates=False):
        active = (work / 'trace_enabled').exists() and torch.is_grad_enabled()
        if not active:
            return original_response(self, image, proprio, seed, full=full, return_intermediates=return_intermediates)
        first, result, ref = original_response(self, image, proprio, seed, full=full, return_intermediates=True)
        index = counter[0]
        for name, tensor in [('ref_tokens', ref), ('input_image', image)]:
            def save_gradient(g, name=name, index=index):
                dest = work / 'intermediate_gradients'
                dest.mkdir(exist_ok=True)
                np.save(dest / f'call_{index:03d}_{name}.npy', g.detach().float().cpu().numpy())
                return g
            tensor.register_hook(save_gradient)
        return (first, result, ref) if return_intermediates else (first, result)

    torch.autograd.grad = traced_grad
    Backend.prepare_views = traced_prepare
    Backend.response = traced_response
    assert D.preserved() == D.read(D.DIAG / 'replay_inputs.json')['original_hashes']
    D.J.PORTS = D.PORTS
    D.J.server(work, 'imagewam')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['server'])
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--kind', choices=['imagewam'], required=True)
    a = ap.parse_args()
    server(a.work)
