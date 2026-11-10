"""Real-render spectrum: grating MTF, band/E-texture retention, proxy-vs-real image and L0 agreement."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from spectrum_common import DEFAULT, dump
from crossarch_common import CAMERAS
from render_proxy import grid_for

NOISE_SEEDS = (42, 43)


def t(x):
    return torch.tensor(np.asarray(x).copy(), device='cuda').permute(2, 0, 1)[None].float()/255


def resize224(x):
    return F.interpolate(x, (168, 224), mode='bilinear', align_corners=False, antialias=True)


def camera_geometry(z, camera):
    """Eroded patch mask, per-pixel display-space slope at gray, and texture UV at native and model resolution."""
    m = torch.tensor(z[camera+'_mask'].copy(), device='cuda')[None, None].float()
    eroded = (-F.max_pool2d(-m, 3, 1, 1)) > .5
    slope = (t(z[camera+'_gray192'])-t(z[camera+'_gray64']))/(128/255)
    k, e, c, s = z[camera+'_intrinsic'], z[camera+'_extrinsic'], z['center_m'], float(z['size_m'])
    g = torch.tensor(grid_for(k, e, c, s), device='cuda')
    g224 = torch.tensor(grid_for(k, e, c, s, 224, 224)[28:196], device='cuda')
    m224 = resize224(m) > .99
    return dict(mask=eroded, slope=slope, g=g, mask224=m224, slope224=resize224(slope), g224=g224,
                pixels=int(eroded.sum()), pixels224=int(m224.sum()))


def fit_gain(d, slope, g, mask, k, axis, amplitude):
    """Least-squares D ~ slope*(a sin + b cos) of the ideal UV sinusoid; returns |a+ib|/amplitude and R^2."""
    u = (g[..., 0 if axis == 'u' else 1]+1)/2
    basis = [slope*torch.sin(2*np.pi*k*u)[None, None], slope*torch.cos(2*np.pi*k*u)[None, None]]
    sel = mask.expand_as(d)
    a = torch.stack([b[sel] for b in basis], 1).double(); y = d[sel].double()
    coef = torch.linalg.lstsq(a, y[:, None]).solution[:, 0]
    resid = y-a@coef
    r2 = 1-float(resid.square().sum()/y.square().sum().clamp_min(1e-12))
    return float(coef.norm())/amplitude, r2, float(torch.atan2(coef[1], coef[0]))


def agreement(a, b, mask):
    sel = mask.expand_as(a); x, y = a[sel].double(), b[sel].double()
    corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1]) if x.numel() > 2 and x.std() > 0 and y.std() > 0 else float('nan')
    return dict(corr=corr, rel_err=float((x-y).norm()/y.norm().clamp_min(1e-12)),
                energy_ratio=float(x.square().sum()/y.square().sum().clamp_min(1e-12)))


def analyze_record(path, order, meta, tex, backend, min_pixels):
    from crossarch_proxy import images
    z = np.load(path); rows = []
    geos = {c: camera_geometry(z, c) for c in CAMERAS}
    state, prompt = z['state14'].copy(), str(z['prompt'])
    with torch.no_grad():
        clean = [t(z[c+'_clean']) for c in CAMERAS]
        real_gray = {c: t(z[c+'_gray']) for c in CAMERAS}
        proxy_gray = dict(zip(CAMERAS, images(torch.full((1, 3, 128, 128), 128/255, device='cuda'), path)))
        if backend is not None:
            vc = {n: backend.response(backend.prepare(state, prompt, clean), n)[0].float() for n in NOISE_SEEDS}
        for name in order:
            if meta[name]['kind'] == 'lut' and name != 'gray': continue
            q = torch.tensor(tex[name], device='cuda').permute(2, 0, 1)[None].float()/255
            real = [t(z[c+'_'+name]) for c in CAMERAS]
            proxy = images(q, path)
            row = dict(record=path.name, state_seed=int(z['seed']), size_m=float(z['size_m']), texture=name, **meta[name])
            for c, rv, pv in zip(CAMERAS, real, proxy):
                geo = geos[c]
                if geo['pixels'] < min_pixels: continue
                dr, dp = rv-real_gray[c], pv-proxy_gray[c]
                cam = dict(pixels=geo['pixels'], pixels224=geo['pixels224'],
                           native=agreement(dp, dr, geo['mask']),
                           model=agreement(resize224(dp), resize224(dr), geo['mask224']) if geo['pixels224'] >= min_pixels else None)
                cam['real_energy'] = float(dr[geo['mask'].expand_as(dr)].square().mean())
                cam['proxy_energy'] = float(dp[geo['mask'].expand_as(dp)].square().mean())
                if meta[name]['kind'] == 'grating':
                    k, axis, amp = meta[name]['k'], meta[name]['axis'], meta[name]['amplitude']/255
                    for label, d in (('real', dr), ('proxy', dp)):
                        cam[label+'_gain'], cam[label+'_r2'], cam[label+'_phase'] = fit_gain(d, geo['slope'], geo['g'], geo['mask'], k, axis, amp)
                        if geo['pixels224'] >= min_pixels:
                            cam[label+'_gain224'], cam[label+'_r2_224'], _ = fit_gain(resize224(d), geo['slope224'], geo['g224'], geo['mask224'], k, axis, amp)
                row[c] = cam
            if backend is not None:
                for n in NOISE_SEEDS:
                    vr = backend.response(backend.prepare(state, prompt, real), n)[0].float()
                    vp = backend.response(backend.prepare(state, prompt, proxy), n)[0].float()
                    dr, dp = (vr-vc[n])[..., :14].flatten(), (vp-vc[n])[..., :14].flatten()
                    row[f'L0_real_n{n}'] = float(dr.square().mean()); row[f'L0_proxy_n{n}'] = float(dp.square().mean())
                    row[f'dv_cos_n{n}'] = float(F.cosine_similarity(dr, dp, 0))
            rows.append(row)
    return rows


def main(args):
    root = args.root.resolve(); info = json.loads((root/'textures.json').read_text())
    tex = dict(np.load(root/'textures.npz')); order, meta = info['order'], info['meta']
    backend = None
    if not args.no_model:
        from crossarch_backend import Backend
        backend = Backend('pi', root)
    out = root/'analysis'; out.mkdir(exist_ok=True)
    for path in sorted((root/'renders').glob('s*_size*.npz')):
        target = out/(path.stem+'.json')
        if target.exists() and not args.force: continue
        started = time.time()
        rows = analyze_record(path, order, meta, tex, backend, args.min_pixels)
        dump(target, dict(record=str(path), rows=rows, model=backend is not None, noise_seeds=list(NOISE_SEEDS)))
        print('[analyze]', path.name, len(rows), f'{time.time()-started:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=DEFAULT)
    ap.add_argument('--no-model', action='store_true'); ap.add_argument('--force', action='store_true')
    ap.add_argument('--min-pixels', type=int, default=30)
    main(ap.parse_args())
