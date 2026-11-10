"""Aggregate spectrum analysis: head-camera MTF, proxy-vs-real image agreement, L0 agreement; figure + summary."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from spectrum_common import DEFAULT, GRATING_K, NOISE_BANDS, LOWPASS_K, dump

CAM = 'head_camera'
COLORS = {.06: '#2a78d6', .10: '#eb6834', .14: '#1baf7a'}


def l0(row, kind):
    vals = [v for k, v in row.items() if k.startswith(f'L0_{kind}_n')]
    return float(np.mean(vals)) if vals else np.nan


def cos(row):
    vals = [v for k, v in row.items() if k.startswith('dv_cos_n')]
    return float(np.mean(vals)) if vals else np.nan


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def cutoff(ks, gains, level):
    """First k (log-interpolated) where the gain falls below level."""
    for i in range(1, len(ks)):
        if gains[i] < level <= gains[i-1]:
            f = (gains[i-1]-level)/(gains[i-1]-gains[i])
            return float(np.exp(np.log(ks[i-1])+f*(np.log(ks[i])-np.log(ks[i-1]))))
    return None if gains[0] < level else float('inf')


def main(args):
    root = args.root.resolve()
    rows = [r for p in sorted((root/'analysis').glob('*.json')) for r in json.loads(p.read_text())['rows']]
    sizes = sorted({r['size_m'] for r in rows}); summary = dict(camera=CAM, sizes={})
    by = defaultdict(list)
    for r in rows: by[(r['size_m'], r['texture'])].append(r)
    for size in sizes:
        s = dict(states=len(by[(size, 'gray')]))
        heads = [r[CAM] for r in by[(size, 'gray')] if CAM in r]
        s['head_pixels_native'] = float(np.median([h['pixels'] for h in heads]))
        s['head_pixels_model'] = float(np.median([h['pixels224'] for h in heads]))
        for cam in ('left_camera', 'right_camera'):
            s[cam+'_states_visible'] = sum(cam in r for r in by[(size, 'gray')])
        mtf = {}
        for space, suffix in (('native', ''), ('model', '224')):
            for src in ('real', 'proxy'):
                for axis in ('u', 'v', 'mean'):
                    curve = []
                    for k in GRATING_K:
                        axes = ('u', 'v') if axis == 'mean' else (axis,)
                        vals = [r[CAM][f'{src}_gain{suffix}'] for ax in axes for ph in ('sin', 'cos')
                                for r in by[(size, f'grat_k{k}_{ax}_{ph}')] if CAM in r and f'{src}_gain{suffix}' in r[CAM]]
                        curve.append(float(np.median(vals)) if vals else np.nan)
                    mtf[f'{space}_{src}_{axis}'] = curve
        s['mtf'] = mtf
        s['cutoff_k'] = {f'{key}_{lvl}': cutoff(GRATING_K, mtf[key], lvl) for key in mtf if key.endswith('_mean') for lvl in (.5, .1)}
        floor = np.median([l0(r, 'real') for r in by[(size, 'gray')]])
        s['L0_floor_gray_patch'] = float(floor)
        bands = {}
        for lo, hi in NOISE_BANDS:
            rs = [r for i in range(3) for r in by[(size, f'band_{lo}_{hi}_s{i}')]]
            hr = [r[CAM] for r in rs if CAM in r]
            bands[f'{lo}-{hi}'] = dict(
                image_corr_native=float(np.median([h['native']['corr'] for h in hr])),
                image_corr_model=float(np.median([h['model']['corr'] for h in hr if h['model']])),
                energy_proxy_over_real_model=float(np.median([h['model']['energy_ratio'] for h in hr if h['model']])),
                L0_real=float(np.median([l0(r, 'real') for r in rs])), L0_proxy=float(np.median([l0(r, 'proxy') for r in rs])),
                L0_real_over_floor=float(np.median([l0(r, 'real') for r in rs])/floor),
                dv_cos=float(np.median([cos(r) for r in rs])))
        s['bands'] = bands
        elp = {}
        for src in ('velocity14', 'onpolicy_mix', 'random'):
            full = {r['record']: r for r in by[(size, f'e_{src}_lp64')]}
            curve = {}
            for k in LOWPASS_K:
                rs = by[(size, f'e_{src}_lp{k}')]
                curve[k] = dict(L0_real=float(np.median([l0(r, 'real') for r in rs])),
                                L0_proxy=float(np.median([l0(r, 'proxy') for r in rs])),
                                real_frac_of_full=float(np.median([l0(r, 'real')/l0(full[r['record']], 'real') for r in rs])),
                                proxy_frac_of_full=float(np.median([l0(r, 'proxy')/l0(full[r['record']], 'proxy') for r in rs])),
                                dv_cos=float(np.median([cos(r) for r in rs])))
            elp[src] = curve
        s['e_lowpass'] = elp
        nongray = [r for r in rows if r['size_m'] == size and r['kind'] != 'lut']
        s['L0_rank_corr_real_proxy_all'] = spearman(np.array([l0(r, 'real') for r in nongray]), np.array([l0(r, 'proxy') for r in nongray]))
        s['L0_median_proxy_over_real_all'] = float(np.median([l0(r, 'proxy')/l0(r, 'real') for r in nongray]))
        summary['sizes'][f'{size:.2f}'] = s
    dump(root/'summary.json', summary)
    plot(root, summary)


def plot(root, summary):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                         'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': .6})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.4), facecolor='#fcfcfb')
    band_x = [np.sqrt(max(lo, .5)*hi) for lo, hi in NOISE_BANDS]
    for key, s in summary['sizes'].items():
        c = COLORS[float(key)]; lab = f'{float(key)*100:.0f} cm'
        ax[0, 0].plot(GRATING_K, s['mtf']['model_real_mean'], '-o', color=c, lw=2, ms=4, label=lab+' real')
        ax[0, 0].plot(GRATING_K, s['mtf']['model_proxy_mean'], '--', color=c, lw=1.5, label=lab+' proxy')
        b = s['bands']
        ax[0, 1].plot(band_x, [v['image_corr_model'] for v in b.values()], '-o', color=c, lw=2, ms=4, label=lab+' image corr')
        ax[0, 1].plot(band_x, [v['dv_cos'] for v in b.values()], '--s', color=c, lw=1.5, ms=4, label=lab+' Δv cos')
        ax[1, 0].plot(band_x, [v['L0_real_over_floor'] for v in b.values()], '-o', color=c, lw=2, ms=4, label=lab+' real')
        ax[1, 0].plot(band_x, [v['L0_proxy']/s['L0_floor_gray_patch'] for v in b.values()], '--', color=c, lw=1.5, label=lab+' proxy')
        e = s['e_lowpass']
        for src, ls in (('velocity14', '-o'), ('onpolicy_mix', '-^')):
            ax[1, 1].plot(LOWPASS_K, [e[src][k]['L0_real'] for k in LOWPASS_K], ls, color=c, lw=2, ms=4,
                          label=f'{lab} {src} real')
            ax[1, 1].plot(LOWPASS_K, [e[src][k]['L0_proxy'] for k in LOWPASS_K], '--', color=c, lw=1.2, alpha=.7)
    for a in ax.flat: a.set_xscale('log', base=2); a.set_facecolor('#fcfcfb')
    ax[0, 0].set(title='Head camera, model input (224): grating amplitude kept', xlabel='cycles per patch side', ylabel='gain', ylim=(-.02, 1.05))
    ax[0, 0].axhline(.5, color='#52514e', lw=.8, ls=':')
    ax[0, 1].set(title='Proxy vs real agreement, band-limited noise', xlabel='band centre, cycles per patch side', ylabel='correlation / cosine', ylim=(0, 1.02))
    ax[1, 0].set(title='π0.5 first-step L0 of band noise (÷ gray-patch L0)', xlabel='band centre, cycles per patch side', ylabel='L0 / gray-patch L0')
    ax[1, 0].axhline(1, color='#52514e', lw=.8, ls=':')
    ax[1, 1].set(title='E textures low-passed at k (solid real, dashed proxy)', xlabel='low-pass cutoff k (64 = unfiltered)', ylabel='first-step L0', yscale='log')
    for a in ax.flat: a.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(root/'fig_spectrum.png', dpi=150); print(root/'fig_spectrum.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=DEFAULT)
    main(ap.parse_args())
