"""Fit sampling phase/LOD on random only; check transfer to optimized textures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from render_world_proxy import WorldTextureProxy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'pi05probe/out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibration', default='render_proxy_eot/calibration_e15/calibration.npz')
    ap.add_argument('--out', default='render_proxy_eot/validation_world')
    ap.add_argument('--samples', type=int, choices=[1, 4], default=4)
    args = ap.parse_args()
    torch.set_num_threads(4)
    source = OUT/args.calibration
    archive = np.load(source, allow_pickle=False)
    d = {k: archive[k] for k in archive.files}
    archive.close()
    dest = OUT/args.out; dest.mkdir(parents=True, exist_ok=True)
    proxy = WorldTextureProxy(source, samples=args.samples)
    q = torch.from_numpy(d['random__texture'].transpose(2, 0, 1).astype(np.float32)/255)[None]
    train = np.flatnonzero(d['train_mask']) if 'train_mask' in d else np.arange(8)
    heldout = np.flatnonzero(d['validation_mask']) if 'validation_mask' in d else np.arange(8, len(d['steps']))
    train = train[:8]
    target = torch.from_numpy(d['random256'][train].transpose(0, 3, 1, 2).astype(np.float32)/255)
    black, gain = proxy.black[train], proxy.gain[train]
    valid = torch.from_numpy((d['unlit_white256'][train].min(-1) == 255).astype(np.float32))[:, None]
    parameters = torch.zeros(3, requires_grad=True)
    optimizer = torch.optim.Adam([parameters], lr=.02)
    history = []
    for it in range(100):
        optimizer.zero_grad()
        texture = proxy.texture_native(q, parameters[0], parameters[1], parameters[2], quantize=False)
        pred = black + gain*texture
        loss = ((pred-target).square()*valid).sum()/(valid.sum()*3)
        loss.backward(); optimizer.step()
        with torch.no_grad():
            parameters[:2].clamp_(-1, 1); parameters[2].clamp_(-1, 1)
        history.append(float(loss))
        if it % 20 == 0:
            print('[fit]', it, float(loss), parameters.detach().tolist(), flush=True)
    cfg = {'complete': True, 'schema': 'wam-world-render-proxy-v2',
           'parameters': dict(zip(['dx_native_px', 'dy_native_px', 'lod_bias'], parameters.detach().tolist())) | {'samples': args.samples},
           'fit_frames': train.tolist(), 'heldout_frames': heldout.tolist(),
           'calibration_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
           'pattern': 'random_seed_20260902_only', 'history': history,
           'notes': 'UV homography comes from independent coordinate ramps. Fit only three shared '
                    'sampling parameters on random texture, never full or band32. Per-state white/black '
                    'light/coverage captures are input-independent calibration, not optimized-pattern fitting.'}
    (dest/'fit.json').write_text(json.dumps(cfg, indent=2, allow_nan=False)+'\n')
    proxy.config.update(cfg['parameters'])
    result = {'complete': True, 'transfer': {}, 'gray128_affine_RMSE_native': []}
    arrays = {k: d[k] for k in ['steps', 'state8', 'prompt', 'clean_base', 'clean_wrist', 'mask']}
    arrays['patterns'] = np.array(['random', 'full', 'band32'])
    arrays['conditions'] = np.array(['digital', 'render', 'world'])
    for name in arrays['patterns']:
        texture = torch.from_numpy(d[name+'__texture'].transpose(2, 0, 1).astype(np.float32)/255)[None]
        raw = d['band32_256' if name == 'band32' else name+'256']
        # Preserve exact production PIL target rather than reimplement target resize.
        from PIL import Image
        rendered = np.array([np.asarray(Image.fromarray(a).resize((224, 224), Image.Resampling.BILINEAR)) for a in raw])
        pred = np.array([np.clip(np.rint(proxy(texture, k).detach().numpy()[0].transpose(1, 2, 0)*255),
                                0, 255).astype(np.uint8) for k in range(len(raw))])
        for condition, image in [('digital', d[name+'__pattern']), ('render', rendered), ('world', pred)]:
            arrays[name+'__'+condition] = np.where(d['mask'][..., None], image, d['clean_base'])
        errors = [float(np.sqrt((((pred[k][d['mask'][k].astype(bool)].astype(float) -
                                 rendered[k][d['mask'][k].astype(bool)])/255)**2).mean())) for k in heldout]
        result['transfer'][str(name)] = {'heldout_mask_RMSE': float(np.mean(errors)),
                                       'per_frame_mask_RMSE': errors}
    qgray = torch.full((1, 3, 128, 128), 128/255)
    for k in heldout:
        gray = proxy(qgray, int(k), native=True).detach().numpy()[0].transpose(1, 2, 0)
        target = d['gray128_256'][k].astype(float)/255
        valid = d['unlit_white256'][k].min(-1) == 255
        result['gray128_affine_RMSE_native'].append(float(np.sqrt(((gray[valid]-target[valid])**2).mean())))
    # A finite difference check uses continuous path (no STE quantization).
    for attr in ['H', 'black', 'gain', 'xy']:
        setattr(proxy, attr, getattr(proxy, attr).double())
    qtest = torch.full((1, 3, 128, 128), .4, dtype=torch.float64, requires_grad=True)
    direction = torch.randn(qtest.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(22))
    direction /= direction.norm()
    score = proxy(qtest, 0, quantize=False).square().mean()
    gradient, = torch.autograd.grad(score, qtest)
    h = .001
    numeric = float((proxy(qtest+h*direction, 0, quantize=False).square().mean()-
                     proxy(qtest-h*direction, 0, quantize=False).square().mean())/(2*h))
    analytic = float((gradient*direction).sum())
    relative = abs(numeric-analytic)/max(abs(numeric), abs(analytic), 1e-10)
    assert relative < .02, (numeric, analytic, relative)
    result['continuous_gradient_check'] = {'analytic': analytic, 'numeric': numeric, 'relative_error': relative,
                                           'note': 'STE uint8 derivative is a surrogate, not checked as renderer derivative.'}
    np.savez_compressed(dest/'inputs.npz', **arrays)
    (dest/'pixel_check.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('[done]', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
