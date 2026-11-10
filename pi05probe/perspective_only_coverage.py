"""Post-run sampling support audit; reproduce the model's global matmul setting."""
import json
from pathlib import Path
import numpy as np
import torch
from perspective_only_proxy import PerspectiveOnlyProxy


def main():
    dest = Path(__file__).resolve().parent / 'out/perspective_only'
    cfg = json.loads((dest/'train_13_101_full/config.json').read_text())
    data = np.load(cfg['calibration_path'], allow_pickle=False)
    q_final = np.load(dest/'train_13_101_full/checkpoint_0500.npz')['Q']
    changed = np.any(q_final != .5, axis=0)
    torch.set_float32_matmul_precision('highest')
    reference = PerspectiveOnlyProxy(cfg['calibration_path'], 'cpu', cfg['fit_path'])
    # Pi0Pytorch.__init__ sets high before the training proxy is constructed.
    torch.set_float32_matmul_precision('high')
    proxy = PerspectiveOnlyProxy(cfg['calibration_path'], 'cuda', cfg['fit_path'])
    union = proxy.mask[np.flatnonzero(data['train_mask'])].amax(0, keepdim=True)
    q = torch.full((1, 3, 128, 128), .5, device='cuda', requires_grad=True)
    image = torch.nn.functional.grid_sample(q, proxy.grid, mode='bilinear',
                                           padding_mode='border', align_corners=False)
    (image * union).sum().backward()
    covered = (q.grad[0] > 0).any(0).cpu().numpy()
    assert np.array_equal(covered, changed)
    delta = ((proxy.grid.cpu()-reference.grid)*64)[0][union.cpu()[0, 0] > 0]
    result = {
        'complete': True, 'covered_texels': int(covered.sum()), 'total_texels': covered.size,
        'unchanged_gray_fraction': float((~changed).mean()),
        'changed_support_exactly_matches_actual_training_grid': True,
        'runtime_matmul_precision': 'high (set by Pi0Pytorch before proxy construction)',
        'GPU_vs_CPU_highest_UV_max_abs_texels': float(delta.abs().max()),
        'GPU_vs_CPU_highest_UV_mean_abs_texels': float(delta.abs().mean()),
        'notes': 'Fixed base camera, bilinear224, union of 155 train visibility masks. '
                 'CPU highest support is not the runtime grid. This audit does not '
                 'attribute attack failure to precision or to uncovered texels alone. '
                 'Future sampling ablations should construct geometry in float64 '
                 'before casting to avoid model-global TF32 effects.'}
    (dest/'sampling_support_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
