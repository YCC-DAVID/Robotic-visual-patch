"""Render-path spectrum study: shared test textures (numpy only, used by sim and model envs)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'robotwinprobe/out/spectrum_20261028'
E_ROOT = ROOT/'robotwinprobe/out/e_method_20261027'
N = 128
GRATING_K = (1, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 60)   # cycles per patch side
NOISE_BANDS = ((0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64))
LOWPASS_K = (2, 4, 8, 12, 16, 24, 32, 64)                   # 64 = texture Nyquist = unfiltered
E_SOURCES = {'velocity14': 'patches/velocity14/texture_0500.png',
             'onpolicy_mix': 'patches/velocity_onpolicy_mix/texture_0500.png',
             'random': 'patches/random/texture.png'}
LUT = {'black': 0, 'gray64': 64, 'gray': 128, 'gray192': 192, 'white': 255}
SEEDS = tuple(range(25000, 25006))
SIZES = (.06, .10, .14)


def radial(n=N, pad=1):
    f = np.fft.fftfreq(n*pad)*n          # cycles per patch side
    return np.hypot(f[:, None], f[None, :])


def lowpass(rgb, k):
    """Ideal radial low-pass at k cycles/patch; mirror padding avoids wrap-around edges."""
    x = np.asarray(rgb, np.float64)
    if k >= N//2: return x
    m = np.concatenate([x, x[::-1]], 0); m = np.concatenate([m, m[:, ::-1]], 1)
    keep = radial(N, 2) <= k
    y = np.real(np.fft.ifft2(np.fft.fft2(m, axes=(0, 1))*keep[..., None], axes=(0, 1)))
    return y[:N, :N]


def u8(x): return np.ascontiguousarray(np.clip(np.round(x), 0, 255).astype(np.uint8))


def textures():
    """Ordered dict name -> (128,128,3) uint8 sRGB texture, plus metadata."""
    out, meta = {}, {}
    for name, value in LUT.items():
        out[name] = np.full((N, N, 3), value, np.uint8); meta[name] = dict(kind='lut', value=value)
    col = (np.arange(N)+.5)/N
    for k in GRATING_K:
        for axis in ('u', 'v'):
            for phase in ('sin', 'cos'):
                arg = 2*np.pi*k*col
                wave = np.sin(arg) if phase == 'sin' else np.cos(arg)
                img = np.tile(wave[None, :], (N, 1)) if axis == 'u' else np.tile(wave[:, None], (1, N))
                name = f'grat_k{k}_{axis}_{phase}'
                out[name] = u8(128+64*img[..., None].repeat(3, -1))
                meta[name] = dict(kind='grating', k=k, axis=axis, phase=phase, amplitude=64)
    r = radial()
    for lo, hi in NOISE_BANDS:
        for s in range(3):
            rng = np.random.default_rng(20261028+100*lo+s)
            spec = np.fft.fft2(rng.standard_normal((N, N, 3)), axes=(0, 1))*((r > lo) & (r <= hi))[..., None]
            x = np.real(np.fft.ifft2(spec, axes=(0, 1))); x = x/x.std()*48
            name = f'band_{lo}_{hi}_s{s}'
            out[name] = u8(128+x); meta[name] = dict(kind='band', lo=lo, hi=hi, seed=s)
    for src, rel in E_SOURCES.items():
        base = np.asarray(Image.open(E_ROOT/rel).convert('RGB'), np.float64)
        assert base.shape == (N, N, 3), base.shape
        for k in LOWPASS_K:
            name = f'e_{src}_lp{k}'
            out[name] = u8(lowpass(base, k)); meta[name] = dict(kind='e_lowpass', source=src, k=k)
    return out, meta


def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data, indent=2)+'\n'); tmp.replace(path)
