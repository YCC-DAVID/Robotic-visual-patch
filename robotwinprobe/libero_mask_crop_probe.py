"""LIBERO 13cm: which part of a trained patch carries the attack?  (proxy L0 only, no rollouts)

Keep only a fraction f of the texture area (binary texture-space mask); outside the mask the
table shows through: image = M*proxy(Q) + (1-M)*clean, with M the mask projected by the same
CameraProxy geometry/mip chain as the texture (zero outside the 13cm footprint).
Masks of equal area are compared: centre square, best of 16 squares, |Q-.5| energy, integrated
gradients (IG) of each model's L0 w.r.t. the mask, the IG-worst region, and 3 smooth random blobs.
Selection (IG, square scan) uses 16 training frames; reported numbers use the 32 validation frames.

  attrib --kind pi|fast   -> attrib_<kind>.npz
  masks                   -> masks.npz, masks.png
  eval   --kind pi|fast   -> eval_<kind>.json
  report                  -> summary.csv, summary.md, fig
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('WAMATTACK_PROJECT_ROOT', '/home/user1/workspace/chence/WAMattack'))
OUT = ROOT / 'robotwinprobe/out/libero13_mask_crop_20261104'
RUN = ROOT / 'robotwinprobe/out/libero13_joint_eot_20261029'
E26 = ROOT / 'pi05probe/out/dual_view'
CAL = E26 / 'calibration/base.npz'
TEX = {
    'maxmin_eot': RUN / 'textures/eot_nominal.png',
    'pi_eot': ROOT / 'robotwinprobe/out/libero13_eot_transfer_20261030/textures/pi_eot.png',
    'fast_eot': ROOT / 'robotwinprobe/out/libero13_eot_transfer_20261030/textures/fast_eot.png',
}
RANDOM = ROOT / 'robotwinprobe/out/libero_pi_fastwam_crossarch_13cm_20261028/textures/random.png'
FRACS = [0.15, 0.3, 0.5]
ALPHAS = [.25, .5, .75, 1.]
N_SEL = 16
GRID = 4  # 4x4 square positions
KINDS = ['pi', 'fast']


def frames():
    with np.load(CAL) as z:
        tr, va = np.flatnonzero(z['train_mask']), np.flatnonzero(z['validation_mask'])
    sel = tr[np.round(np.linspace(0, len(tr) - 1, N_SEL)).astype(int)]
    return [int(k) for k in sel], [20261104 + i for i in range(N_SEL)], [int(k) for k in va], [20260901] * len(va)


def png(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert('RGB'), np.float32).transpose(2, 0, 1)[None] / 255


def dump(path, obj):
    tmp = Path(str(path) + '.tmp'); tmp.write_text(json.dumps(obj, indent=1)); tmp.replace(path)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


class Probe:
    def __init__(self, kind):
        import torch
        torch.set_num_threads(4)
        sys.path.insert(0, str(ROOT / 'robotwinprobe')); sys.path.insert(0, str(ROOT / 'pi05probe'))
        if kind == 'pi':
            from libero_pi_adapter import Backend
        else:
            from fastwam_adapter import Backend
        from dual_view_proxy import CameraProxy
        self.torch, self.kind = torch, kind
        self.b = Backend(RUN / 'models' / kind)
        with np.load(CAL) as z:
            self.d = {n: z[n].copy() for n in ['state8', 'clean_base', 'clean_wrist']}; self.prompt = str(z['prompt'])
        self.proxies = [(CameraProxy(CAL, device='cuda', fit=E26 / 'fit/base.json'), 'clean_base'),
                        (CameraProxy(E26 / 'calibration/wrist.npz', device='cuda', fit=E26 / 'fit/wrist.json'), 'clean_wrist')]
        self.cache = {}

    def clean(self, name, k):
        t = self.torch
        return t.tensor(self.d[name][k], device='cuda').permute(2, 0, 1)[None].float() / 255

    def project_mask(self, proxy, m, k):
        import torch.nn.functional as F
        prm = [proxy.config[x] for x in ['dx_native_px', 'dy_native_px', 'lod_bias']]
        grid, lod = proxy.grid(k, *prm[:2]); grid = grid.float(); lod = (lod + prm[2]).clamp(0, 7).float()
        cur, out = m, 0
        for level in range(8):
            w = (1 - (lod - level).abs()).clamp_min(0)[None, None]
            out = out + F.grid_sample(cur, grid, mode='bilinear', padding_mode='zeros', align_corners=False) * w
            if level < 7:
                cur = F.avg_pool2d(cur, 2)
        return F.interpolate(out, (224, 224), mode='bilinear', align_corners=False, antialias=True).clamp(0, 1)

    def views(self, q, m, k):
        out = []
        for proxy, name in self.proxies:
            img = proxy(q, k)
            if m is not None:
                mi = self.project_mask(proxy, m, k)
                img = mi * img + (1 - mi) * self.clean(name, k)
            out.append(img)
        return out

    def response(self, k, seed, images):
        v, _ = self.b.response(self.b.prepare(self.d['state8'][k], self.prompt, images), seed, full=False)
        return v.float() if self.kind == 'pi' else v[..., :7].float()

    def loss(self, q, m, k, seed):
        t = self.torch
        if (k, seed) not in self.cache:
            with t.no_grad():
                self.cache[(k, seed)] = self.response(k, seed, [self.clean(n, k) for _, n in self.proxies])
        delta = self.response(k, seed, self.views(q, m, k)) - self.cache[(k, seed)]
        return delta.square().sum() if self.kind == 'pi' else delta.square().mean()

    def value(self, q, m, ks, seeds):
        t = self.torch
        with t.no_grad():
            return [float(self.loss(q, m, k, s)) for k, s in zip(ks, seeds)]

    def tq(self, path):
        return self.torch.tensor(png(path), device='cuda')

    def tm(self, m):
        return self.torch.tensor(np.asarray(m, np.float32)[None, None], device='cuda')


def squares(f):
    side = int(round(128 * np.sqrt(f)))
    pos = np.round(np.linspace(0, 128 - side, GRID)).astype(int)
    out = []
    for y in pos:
        for x in pos:
            m = np.zeros((128, 128), np.float32); m[y:y + side, x:x + side] = 1; out.append(((int(y), int(x), side), m))
    return out


def attrib(kind):
    OUT.mkdir(parents=True, exist_ok=True)
    p = Probe(kind); sel, sseed, _, _ = frames(); t = p.torch; res = {}; t0 = time.time()
    for tex, path in TEX.items():
        q = p.tq(path)
        res[f'{tex}/full'] = np.array(p.value(q, p.tm(np.ones((128, 128))), sel, sseed))
        res[f'{tex}/proxy'] = np.array(p.value(q, None, sel, sseed))
        ig = np.zeros((128, 128)); curve = np.zeros((len(ALPHAS), len(sel)))
        for i, (k, s) in enumerate(zip(sel, sseed)):
            for a, alpha in enumerate(ALPHAS):
                m = t.full((1, 1, 128, 128), alpha, device='cuda', requires_grad=True)
                L = p.loss(q, m, k, s)
                g, = t.autograd.grad(L, m)
                ig += g[0, 0].double().cpu().numpy() / len(ALPHAS) / len(sel); curve[a, i] = float(L)
        res[f'{tex}/ig'] = ig; res[f'{tex}/alpha_curve'] = curve
        log(kind, tex, 'IG done', f'{time.time() - t0:.0f}s', 'full', res[f'{tex}/full'].mean(), 'proxy', res[f'{tex}/proxy'].mean())
        for f in FRACS:
            res[f'{tex}/sq{f}'] = np.array([np.mean(p.value(q, p.tm(m), sel, sseed)) for _, m in squares(f)])
        log(kind, tex, 'squares done', f'{time.time() - t0:.0f}s')
    np.savez(OUT / f'attrib_{kind}.npz', **res)


def blur(x, s):
    pad = np.pad(x, 64, mode='reflect')
    kk = np.fft.fftfreq(256); gg = np.exp(-2 * (np.pi * s) ** 2 * (kk[:, None] ** 2 + kk[None, :] ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(pad) * gg))[64:192, 64:192]


def topk(score, f, largest=True):
    n = int(round(f * 128 * 128)); idx = np.argsort(score.ravel(), kind='stable')
    idx = idx[::-1][:n] if largest else idx[:n]
    m = np.zeros(128 * 128, np.float32); m[idx] = 1; return m.reshape(128, 128)


def components(m):
    from collections import deque
    seen = np.zeros_like(m, bool); n = 0
    for y, x in zip(*np.nonzero(m)):
        if seen[y, x]:
            continue
        n += 1; dq = deque([(y, x)]); seen[y, x] = True
        while dq:
            a, b = dq.popleft()
            for c, d in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                if 0 <= c < 128 and 0 <= d < 128 and m[c, d] and not seen[c, d]:
                    seen[c, d] = True; dq.append((c, d))
    return n


def masks():
    A = {k: dict(np.load(OUT / f'attrib_{k}.npz')) for k in KINDS}
    M, meta = {}, {}
    for tex, path in TEX.items():
        dev = np.abs(png(path)[0] - .5).sum(0)
        ig = {k: blur(A[k][f'{tex}/ig'], 2.) for k in KINDS}
        ig['both'] = sum(ig[k] / np.abs(ig[k]).sum() for k in KINDS)
        full = {k: A[k][f'{tex}/full'].mean() for k in KINDS}
        for f in FRACS:
            sq = squares(f)
            score = sum(A[k][f'{tex}/sq{f}'] / full[k] for k in KINDS)
            side = sq[0][0][2]; c = (128 - side) // 2
            cen = np.zeros((128, 128), np.float32); cen[c:c + side, c:c + side] = 1
            cand = {'center_square': cen, 'best_square': sq[int(np.argmax(score))][1],
                    'energy': topk(blur(dev, 2.), f), 'ig_pi': topk(ig['pi'], f), 'ig_fast': topk(ig['fast'], f),
                    'ig_both': topk(ig['both'], f), 'ig_both_worst': topk(ig['both'], f, largest=False)}
            for r in range(3):
                cand[f'random_blob{r}'] = topk(blur(np.random.RandomState(1000 + r).standard_normal((128, 128)), 8.), f)
            for name, m in cand.items():
                key = f'{tex}|{f}|{name}'; M[key] = m
                meta[key] = dict(area=float(m.mean()), components=components(m),
                                 square=sq[int(np.argmax(score))][0] if name == 'best_square' else None)
    np.savez_compressed(OUT / 'masks.npz', **M); dump(OUT / 'masks.json', meta)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    names = ['center_square', 'best_square', 'energy', 'ig_pi', 'ig_fast', 'ig_both', 'ig_both_worst', 'random_blob0']
    fig, ax = plt.subplots(len(TEX) * len(FRACS), len(names) + 1, figsize=(2 * (len(names) + 1), 2 * len(TEX) * len(FRACS)))
    for i, tex in enumerate(TEX):
        img = png(TEX[tex])[0].transpose(1, 2, 0)
        for j, f in enumerate(FRACS):
            r = i * len(FRACS) + j; ax[r, 0].imshow(img); ax[r, 0].set_ylabel(f'{tex}\nf={f}', fontsize=8)
            for c, n in enumerate(names):
                m = M[f'{tex}|{f}|{n}']; ax[r, c + 1].imshow(img * m[..., None] + .85 * (1 - m[..., None]))
                if r == 0: ax[r, c + 1].set_title(n, fontsize=8)
    for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / 'masks.png', dpi=90)


def evaluate(kind):
    p = Probe(kind); _, _, va, vseed = frames(); M = dict(np.load(OUT / 'masks.npz')); res = {}; t0 = time.time()
    path = OUT / f'eval_{kind}.json'
    if path.exists():
        res = json.loads(path.read_text())
    jobs = [('random|1.0|proxy', p.tq(RANDOM), None)]
    for tex, tp in TEX.items():
        q = p.tq(tp)
        jobs += [(f'{tex}|1.0|proxy', q, None), (f'{tex}|1.0|mask_full', q, p.tm(np.ones((128, 128))))]
        jobs += [(key, q, p.tm(m)) for key, m in M.items() if key.startswith(tex + '|')]
    for i, (key, q, m) in enumerate(jobs):
        if key in res:
            continue
        res[key] = p.value(q, m, va, vseed); dump(path, res)
        if i % 10 == 0:
            log(kind, i, '/', len(jobs), key, f'{np.mean(res[key]):.4g}', f'{time.time() - t0:.0f}s')
    log(kind, 'eval complete', len(res))


def report():
    E = {k: json.loads((OUT / f'eval_{k}.json').read_text()) for k in KINDS}
    meta = json.loads((OUT / 'masks.json').read_text())
    rows = []
    for key in E['pi']:
        tex, f, name = key.split('|')
        if tex == 'random':
            continue
        base = {k: np.mean(E[k][f'{tex}|1.0|proxy']) for k in KINDS}
        r = {k: np.mean(E[k][key]) / base[k] for k in KINDS}
        rows.append(dict(texture=tex, area=float(f), mask=name, keep_pi=r['pi'], keep_fast=r['fast'],
                         L_pi=np.mean(E['pi'][key]), L_fast=np.mean(E['fast'][key]),
                         components=meta.get(key, {}).get('components', 1)))
    rnd = {k: np.mean(E[k]['random|1.0|proxy']) for k in KINDS}
    with open(OUT / 'summary.csv', 'w') as fh:
        fh.write('texture,area,mask,keep_pi,keep_fast,L_pi,L_fast,components\n')
        for r in rows:
            fh.write(f"{r['texture']},{r['area']},{r['mask']},{r['keep_pi']:.4f},{r['keep_fast']:.4f},{r['L_pi']:.5g},{r['L_fast']:.5g},{r['components']}\n")
    lines = ['# 掩码裁剪代理测试', '', f'Random 纹理整块 L0：π {rnd["pi"]:.4g}，FastWAM {rnd["fast"]:.4g}', '']
    for tex in TEX:
        lines += [f'## {tex}', '', '| 面积 | 掩码 | π保留 | FastWAM保留 | 连通块 |', '|---|---|---:|---:|---:|']
        for r in rows:
            if r['texture'] == tex:
                lines.append(f"| {r['area']} | {r['mask']} | {r['keep_pi']:.2f} | {r['keep_fast']:.2f} | {r['components']} |")
        lines.append('')
    (OUT / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('command', choices=['attrib', 'masks', 'eval', 'report'])
    ap.add_argument('--kind', choices=KINDS); a = ap.parse_args()
    {'attrib': lambda: attrib(a.kind), 'masks': masks, 'eval': lambda: evaluate(a.kind), 'report': report}[a.command]()
