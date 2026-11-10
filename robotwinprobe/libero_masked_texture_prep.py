"""Build printable 'scattered' patches for closed-loop tests (CPU only).

MuJoCo 3.2.3 ignores per-texel alpha, so the parts of the 13cm patch outside the mask are printed
with a fitted *table camouflage* texture T: the texture whose proxy rendering best matches the clean
table in both cameras over 64 training frames.  Final PNG = M*Q + (1-M)*T, Q = max-min+EOT texture.
Masks: IG(both models) top-f from libero13_mask_crop_20261104 (selection frames only), plus a smooth
random-blob shape control and a table-only (f=0) camouflage check.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/shared/user65/workspace/chence/WAMattack')
sys.path.insert(0, str(ROOT / 'pi05probe')); sys.path.insert(0, str(ROOT / 'robotwinprobe'))
from dual_view_proxy import CameraProxy  # noqa: E402
import libero_mask_crop_probe as P  # noqa: E402

E26 = ROOT / 'pi05probe/out/dual_view'
CROP = ROOT / 'robotwinprobe/out/libero13_mask_crop_20261104'
OUT = ROOT / 'robotwinprobe/out/libero13_scattered_closedloop_20261104'
Q = ROOT / 'robotwinprobe/out/libero13_joint_eot_20261029/textures/eot_nominal.png'


def fit_table(steps=400):
    torch.manual_seed(0)
    z = np.load(E26 / 'calibration/base.npz')
    tr = np.flatnonzero(z['train_mask']); va = np.flatnonzero(z['validation_mask'])
    fr = tr[np.round(np.linspace(0, len(tr) - 1, 64)).astype(int)]
    cams = []
    for cal, fit, key in [('base', 'base', 'clean_base'), ('wrist', 'wrist', 'clean_wrist')]:
        zz = np.load(E26 / f'calibration/{cal}.npz')
        prox = CameraProxy(E26 / f'calibration/{cal}.npz', device='cpu', fit=E26 / f'fit/{fit}.json')
        clean = torch.from_numpy(z[key].astype(np.float32) / 255).permute(0, 3, 1, 2)
        mask = torch.from_numpy(zz['mask'].astype(np.float32))[:, None]
        cams.append((cal, prox, clean, mask))
    logit = torch.zeros((1, 3, 128, 128), requires_grad=True)
    opt = torch.optim.Adam([logit], lr=.05)

    def err(q, frames):
        tot, n = 0., 0.
        for _, prox, clean, mask in cams:
            for k in frames:
                k = int(k); m = mask[k:k + 1]
                if m.sum() < 1:
                    continue
                tot = tot + ((prox(q, k, quantize=False) - clean[k:k + 1]).square() * m).sum(); n += float(m.sum()) * 3
        return tot / max(n, 1)
    for it in range(steps):
        b = np.random.RandomState(it).choice(fr, 8, replace=False)
        loss = err(torch.sigmoid(logit), b); opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0:
            print('fit', it, float(loss) ** .5, flush=True)
    T = torch.sigmoid(logit).detach()
    with torch.no_grad():
        rep = {name: float(err(q, va)) ** .5 for name, q in [
            ('table_fit', T), ('gray', torch.full_like(T, .5)),
            ('maxmin_eot', torch.from_numpy(P.png(Q)))]}
    return T[0].permute(1, 2, 0).numpy(), rep


def save(path, a):
    Image.fromarray(np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)).save(path)


def main():
    (OUT / 'textures').mkdir(parents=True, exist_ok=True)
    T, rep = fit_table()
    save(OUT / 'textures/table_only.png', T)
    q = P.png(Q)[0].transpose(1, 2, 0)
    A = {k: dict(np.load(CROP / f'attrib_{k}.npz')) for k in P.KINDS}
    ig = {k: P.blur(A[k]['maxmin_eot/ig'], 2.) for k in P.KINDS}
    both = sum(ig[k] / np.abs(ig[k]).sum() for k in P.KINDS)
    masks = {f'ig{int(f * 100)}': P.topk(both, f) for f in (.5, .7, .8)}
    masks['blob70'] = P.topk(P.blur(np.random.RandomState(1000).standard_normal((128, 128)), 8.), .7)
    meta = dict(table_fit_rmse_validation=rep, source_texture=str(Q), masks={})
    for name, m in masks.items():
        save(OUT / f'textures/{name}.png', m[..., None] * q + (1 - m[..., None]) * T)
        np.save(OUT / f'textures/{name}_mask.npy', m)
        meta['masks'][name] = dict(area=float(m.mean()), components=P.components(m))
    P.dump(OUT / 'textures/meta.json', meta)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
