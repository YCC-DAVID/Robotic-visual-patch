"""Build random-only calibration and check transfer/numerics, on CPU.

Uses E17's random-pattern fit on step 10..45. Reports errors on step 50..85
for random/full/band32 without refitting optimized patterns. No policy calls.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from render_training_proxy import RenderTrainingProxy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'pi05probe/out'


def main():
    torch.set_num_threads(2)
    dest = OUT / 'render_pixel_diagnosis'
    metrics = json.loads((dest / 'metrics.json').read_text())
    pool = np.load(OUT / 'kv_render_probe/inputs.npz', allow_pickle=False)
    patch = np.load(OUT / 'drift_patch_drift.npz', allow_pickle=True)
    params = metrics['cases']['drift/random']['surrogate_fits']['roundtrip_shift_affine']['parameters']
    cfg = {'schema': 'wam-render-proxy-v1', 'texture_resolution': 128,
           'quad224_yx': patch['quad224'].tolist(),
           'patch_m': float(patch['patch_m']), 'calibration_pattern': 'random_seed_20260902',
           'train_steps': metrics['train_steps'], 'heldout_steps': metrics['heldout_steps'],
           'input_sha256': metrics['input_sha256'],
           'fit_metrics_sha256': hashlib.sha256((dest/'metrics.json').read_bytes()).hexdigest(),
           'dx_px': params['dx_px'], 'dy_px': params['dy_px'],
           'gain_rgb': params['gain_rgb'], 'bias_rgb': params['bias_rgb'],
           'notes': 'Fixed 26cm base camera calibration. No policy validation. Continuous bilinear '
                    'proxy omits uint8 quantization, exact rasterizer, lighting variation and soft edges. '
                    'Jitter widths are user-chosen, not measured confidence intervals.'}
    calibration = dest/'training_proxy_calibration.json'
    calibration.write_text(json.dumps(cfg, indent=2, allow_nan=False)+'\n')
    proxy = RenderTrainingProxy(calibration, patch['quad224'], 'cpu')
    n = int(cfg['texture_resolution'])
    H = cv2.getPerspectiveTransform(patch['quad224'].astype(np.float32)[:, ::-1],
                                   np.array([[0,0],[n-1,0],[n-1,n-1],[0,n-1]], np.float32))
    result = {'complete': True, 'model_executed': False, 'transfer': {},
              'notes': 'Random-only parameters; heldout states are from the same trajectory. '
                       'Gradient check tests continuous proxy only, not true renderer gradients.'}
    random = np.random.RandomState(20260902).randint(0,256,(224,224,3)).astype(np.uint8)
    for tag, label in [('drift','random'), ('drift','optimized'), ('band32','optimized')]:
        p = np.load(OUT/('drift_patch_'+tag+'.npz'), allow_pickle=True)
        assert np.allclose(p['quad224'], patch['quad224'], atol=1e-4, rtol=0)
        image = random if label=='random' else np.clip(p['P'].transpose(1,2,0)*255,0,255).astype(np.uint8)
        x = torch.from_numpy(image.astype(np.float32).transpose(2,0,1)/255)[None]
        pred = proxy(x).numpy()[0].transpose(1,2,0)
        continuous = cv2.warpPerspective(image.astype(np.float32)/255, H, (n,n))
        continuous = cv2.warpPerspective(continuous, np.linalg.inv(H), (224,224))
        continuous = cv2.warpAffine(continuous, np.float32([[1,0,cfg['dx_px']],[0,1,cfg['dy_px']]]),(224,224))
        reference = np.clip(continuous*np.array(cfg['gain_rgb']) + np.array(cfg['bias_rgb']),0,1)
        mask = pool[tag+'__mask'].astype(bool)
        interior = np.array([cv2.erode(m.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool) for m in mask])
        numerical_rmse = float(np.sqrt(((pred-reference)[interior.any(0)]**2).mean()))
        assert numerical_rmse < .01, numerical_rmse
        target = pool[tag+'__'+label+'_render_base'].astype(np.float32)/255
        ids = [pool['steps'].tolist().index(s) for s in metrics['heldout_steps']]
        errors = [float(np.sqrt(((pred-target[i])[interior[i]]**2).mean())) for i in ids]
        result['transfer'][tag+'/'+label] = {'heldout_RMSE': float(np.mean(errors)),
                                              'torch_vs_cv2_continuous_RMSE': numerical_rmse}
    # Double precision directional finite difference through geometry + clamp.
    x = (.2 + .6*torch.rand((1,3,224,224), generator=torch.Generator().manual_seed(6), dtype=torch.float64)).requires_grad_()
    direction = torch.randn(x.shape, generator=torch.Generator().manual_seed(7), dtype=x.dtype)
    direction /= direction.norm()
    weight = torch.randn(x.shape, generator=torch.Generator().manual_seed(8), dtype=x.dtype)
    for attr in ['export_grid', 'return_grid', 'identity_grid', 'gain', 'bias']:
        setattr(proxy, attr, getattr(proxy, attr).double())
    score = (proxy(x)*weight).sum()
    gradient, = torch.autograd.grad(score, x)
    analytic = float((gradient*direction).sum())
    h = 1e-4
    numeric = float(((proxy(x+h*direction)-proxy(x-h*direction))*weight).sum()/(2*h))
    relerror = abs(analytic-numeric)/max(abs(analytic),abs(numeric),1e-10)
    assert relerror < 1e-4, relerror
    assert torch.isfinite(gradient).all() and float(gradient.norm())>0
    result['gradient_check'] = {'analytic':analytic, 'numeric':numeric, 'relative_error':relerror,
                                'step':h, 'gradient_norm':float(gradient.norm())}
    float_proxy = RenderTrainingProxy(calibration, patch['quad224'], 'cpu')
    sample = torch.from_numpy(random.transpose(2,0,1).astype(np.float32)/255)[None]
    a = float_proxy(sample, rng=np.random.RandomState(9), color_jitter=.15, shift_jitter_px=.25)
    b = float_proxy(sample, rng=np.random.RandomState(9), color_jitter=.15, shift_jitter_px=.25)
    c = float_proxy(sample, rng=np.random.RandomState(10), color_jitter=.15, shift_jitter_px=.25)
    assert torch.equal(a,b) and not torch.equal(a,c)
    assert torch.isfinite(a).all() and float(a.min())>=0 and float(a.max())<=1
    result['seeded_jitter_repeat_equal'] = True
    result['different_jitter_seed_changes_output'] = True
    try:
        RenderTrainingProxy(calibration, patch['quad224']+1, 'cpu')
    except ValueError:
        result['wrong_geometry_rejected'] = True
    else:
        raise AssertionError('Wrong geometry was accepted')
    (dest/'training_proxy_check.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
