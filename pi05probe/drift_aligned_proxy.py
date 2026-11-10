"""JAX version of our calibrated texture proxy at the fixed DRIFT footprint.

This approximates native MuJoCo rasterization; the gradient is through a
calibrated texture sampler, never a claimed differentiable MuJoCo renderer.
Texture coordinates use the original, unflipped PNG orientation.
"""
from functools import lru_cache
from pathlib import Path
import json
import re

import numpy as np

EOT_KEYS = ('angle_deg', 'dx_fraction', 'dy_fraction', 'scale', 'gain')
EOT_RANGES = dict(angle_deg=[-15., 15.], dx_fraction=[-.05, .05],
                  dy_fraction=[-.05, .05], scale=[.9, 1.1], gain=[.9, 1.1])
MSAA_OFFSETS = ((.125, .375), (-.375, .125), (.375, -.125), (-.125, -.375))


def ste_u8(x, floor=False):
    import jax
    import jax.numpy as jnp
    clipped = jnp.clip(x, 0., 1.)
    quantized = (jnp.floor(clipped * 255.) if floor else jnp.round(clipped * 255.)) / 255.
    return x + jax.lax.stop_gradient(quantized - x)


def bilinear(texture, xy, border=True):
    """HWC sampling using pixel-center xy, like align_corners=False grid_sample."""
    import jax.numpy as jnp
    h, w = texture.shape[:2]
    x, y = xy[..., 0], xy[..., 1]
    if border:
        x, y = jnp.clip(x, 0., w - 1.), jnp.clip(y, 0., h - 1.)
    x0, y0 = jnp.floor(x).astype(jnp.int32), jnp.floor(y).astype(jnp.int32)
    wx, wy = x - x0, y - y0

    def get(xx, yy):
        value = texture[jnp.clip(yy, 0, h - 1), jnp.clip(xx, 0, w - 1)]
        if not border:
            valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
            value = value * valid[..., None]
        return value

    top = get(x0, y0) * (1. - wx[..., None]) + get(x0 + 1, y0) * wx[..., None]
    bottom = get(x0, y0 + 1) * (1. - wx[..., None]) + get(x0 + 1, y0 + 1) * wx[..., None]
    return top * (1. - wy[..., None]) + bottom * wy[..., None]


def transform_texture(q, draw):
    """Exact inverse-affine convention of robotwinprobe/texture_eot.py."""
    import jax.numpy as jnp
    h, w = q.shape[:2]
    angle, dx, dy, scale, gain = draw
    radians = angle * (jnp.pi / 180.)
    c, s = jnp.cos(radians), jnp.sin(radians)
    yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
    x, y = 2. * (xx + .5) / w - 1., 2. * (yy + .5) / h - 1.
    tx, ty = 2. * dx, 2. * dy
    sx = (c * (x - tx) + s * (y - ty)) / scale
    sy = (-s * (x - tx) + c * (y - ty)) / scale
    xy = jnp.stack([(sx + 1.) * w / 2. - .5, (sy + 1.) * h / 2. - .5], -1)
    return jnp.clip((bilinear(q - .5, xy, border=False) + .5) * gain, 0., 1.)


def eot_draws(steps=500, batch=8, seed=2026102901):
    rng = np.random.default_rng(seed)
    values = np.empty((steps, batch, len(EOT_KEYS)), np.float32)
    for i in range(steps):
        for b in range(batch):
            values[i, b] = [rng.uniform(*EOT_RANGES[key]) for key in EOT_KEYS]
    return values


@lru_cache(maxsize=8)
def pil_bilinear_coefficients(n_in=256, n_out=224):
    """PIL's normalized, antialiased triangle coefficients in float64."""
    scale = n_in / n_out
    filter_scale = max(scale, 1.)
    weights = np.zeros((n_out, n_in), np.float64)
    for out in range(n_out):
        center = (out + .5) * scale
        lower = max(int(center - filter_scale + .5), 0)
        upper = min(int(center + filter_scale + .5), n_in)
        x = (np.arange(lower, upper, dtype=np.float64) - center + .5) / filter_scale
        values = np.maximum(0., 1. - np.abs(x))
        weights[out, lower:upper] = values / values.sum()
    return weights


def resize224(image, quantize=True):
    import jax
    import jax.numpy as jnp
    coefficients = pil_bilinear_coefficients(image.shape[-2], 224)
    weights = jnp.asarray(coefficients, dtype=image.dtype)
    horizontal = jnp.einsum('...hwc,ow->...hoc', image, weights, precision=jax.lax.Precision.HIGHEST)
    vertical = jnp.einsum('...hwc,oh->...owc', horizontal, weights, precision=jax.lax.Precision.HIGHEST)
    if not quantize:
        return vertical
    # PIL RGB8 normalizes its coefficients to 22-bit signed integers, rounding
    # the uint8 intermediate after each axis. Preserve that exact forward and
    # use the continuous triangle-filter derivative as our declared surrogate.
    fixed = jnp.asarray(np.floor(coefficients * (1 << 22) + .5).astype(np.int32))
    pixels = jnp.floor(jnp.clip(image, 0., 1.) * 255. + .5).astype(jnp.int32)
    horizontal_integer = jnp.clip((jnp.einsum('...hwc,ow->...hoc', pixels, fixed) + (1 << 21)) >> 22, 0, 255)
    vertical_integer = jnp.clip((jnp.einsum('...hwc,oh->...owc', horizontal_integer, fixed) + (1 << 21)) >> 22, 0, 255)
    exact = vertical_integer.astype(image.dtype) / 255.
    return vertical + jax.lax.stop_gradient(exact - vertical)


def native_derivatives(H, height=256, width=256):
    """Analytic UV derivatives used for LOD and fixed MSAA sample offsets."""
    import jax
    import jax.numpy as jnp
    yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing='ij')
    points = jnp.stack([xx, yy, jnp.ones_like(xx)], -1).astype(jnp.float32)
    projected = jnp.einsum('hwj,...ij->...hwi', points, H, precision=jax.lax.Precision.HIGHEST)
    z = projected[..., 2:]
    z = jnp.where(jnp.abs(z) < 1e-8, 1e-8, z)
    uv = projected[..., :2] / z
    dx = (H[..., None, None, :2, 0] - uv * H[..., None, None, 2:3, 0]) / z
    dy = (H[..., None, None, :2, 1] - uv * H[..., None, None, 2:3, 1]) / z
    return dx, dy


def render_one(q, calibration, draw=None, native=False, quantize=True, samples=1):
    """Return [2,H,W,3] RGB [0,1]; both views share the same EOT texture."""
    import jax
    import jax.numpy as jnp
    if draw is not None:
        q = transform_texture(q, draw)
    current = ste_u8(q, floor=True) if quantize else q
    uv = calibration['uv256']
    lod = jnp.clip(calibration['lod256'], 0., 6.)
    if samples == 1:
        grids = [uv]
    else:
        assert samples == 4
        dx, dy = native_derivatives(calibration['H_native_to_texture'])
        grids = [uv + dx * x + dy * y for x, y in MSAA_OFFSETS]
    texel = jnp.zeros(uv.shape[:-1] + (3,), dtype=q.dtype)
    for level in range(7):
        level_grids = [((grid + .5) * (current.shape[0] / 64.) - .5) for grid in grids]
        sampled = sum(jax.vmap(lambda coords: bilinear(current, coords))(grid) for grid in level_grids) / len(grids)
        weight = jnp.clip(1. - jnp.abs(lod - level), 0., 1.)
        texel = texel + sampled * weight[..., None]
        if level < 6:
            n = current.shape[0] // 2
            current = current.reshape(n, 2, n, 2, 3).mean(axis=(1, 3))
            if quantize:
                current = ste_u8(current)
    black = calibration['black256'].astype(jnp.float32) / 255.
    white = calibration['white256'].astype(jnp.float32) / 255.
    image = jnp.clip(black + jnp.maximum(white - black, 0.) * texel, 0., 1.)
    if quantize:
        image = ste_u8(image)
    return image if native else resize224(image, quantize=quantize)


class CalibrationPool:
    """CPU resident calibration arrays indexed exactly like the original pool."""
    KEYS = ('black256', 'white256', 'uv256', 'lod256', 'H_native_to_texture')

    def __init__(self, root, require_complete=True):
        from drift_paper_common import sha
        self.root = Path(root)
        capture = self.root / 'capture'
        if require_complete:
            complete = json.loads((capture / 'complete.json').read_text())
            assert complete.get('complete') is True
            assert sha(capture / 'manifest.json') == complete['manifest_sha256']
            manifest = json.loads((capture / 'manifest.json').read_text())
            assert manifest['complete'] and sha(capture / 'protocol.json') == manifest['protocol_sha256']
            protocol = json.loads((capture / 'protocol.json').read_text())
            assert protocol['train_indices'] == [10, 11, 12]
            assert protocol['texture_resolution'] == 64 and protocol['model_config'] == 'pi05_libero'
            assert sha(capture / 'random_texture.png') == manifest['random_texture_sha256']
            for record in manifest['records']:
                assert record['complete'] and sha(record['file']) == record['sha256'], ('capture shard changed', record['file'])
            self.manifest_sha256 = complete['manifest_sha256']
            self.protocol_sha256 = manifest['protocol_sha256']
        self.paths = sorted(p for p in (capture / 'shards').glob('task*_init*.npz')
                            if re.fullmatch(r'task\d{2}_init\d{2}\.npz', p.name))
        assert self.paths, 'no calibration shards'
        if require_complete:
            assert {str(p.resolve()) for p in self.paths} == {str(Path(row['file']).resolve()) for row in manifest['records']}
        index, values = [], {key: [] for key in self.KEYS}
        self.frame_metadata = []
        self.shard_ranges = []
        for path in self.paths:
            with np.load(path, allow_pickle=False) as d:
                ids = d['frame_indices'].astype(np.int64)
                assert len(ids) > 0
                index.extend(ids.tolist())
                for key in self.KEYS:
                    values[key].append(d[key])
                self.shard_ranges.append((path, ids.copy()))
        order = np.argsort(index)
        sorted_ids = np.asarray(index)[order]
        assert np.array_equal(sorted_ids, np.arange(len(sorted_ids))), 'calibration must cover every original pool index exactly once'
        self.data = {key: np.concatenate(chunks, axis=0)[order] for key, chunks in values.items()}
        self.n = len(sorted_ids)
        assert self.data['uv256'].shape == (self.n, 2, 256, 256, 2)
        assert np.isfinite(self.data['uv256']).all() and np.isfinite(self.data['lod256']).all()

    def batch(self, indices):
        return {key: value[indices] for key, value in self.data.items()}

    def validation_rows(self):
        """At most 13 uniformly prespecified frames; random pattern is not trained."""
        selected = set(np.linspace(0, self.n - 1, min(12, self.n), dtype=int).tolist())
        records = {}
        max_base, max_record = -1, None
        texture = None
        for path, ids in self.shard_ranges:
            with np.load(path, allow_pickle=False) as d:
                for local, global_id in enumerate(ids):
                    area = int(d['mask256'][local, 0].sum())
                    if area > max_base:
                        max_base = area
                        max_record = (int(global_id), d['random256'][local].copy(), d['mask256'][local].copy())
                    if int(global_id) in selected:
                        records[int(global_id)] = (d['random256'][local].copy(), d['mask256'][local].copy())
        if max_record is not None:
            records[max_record[0]] = max_record[1:]
        return sorted(records.items())


def proxy_gate(pool, max_rmse=.02, samples=1):
    """Independent random-pattern native RGB agreement before any PGD update."""
    import jax
    import jax.numpy as jnp
    from PIL import Image
    from scipy.ndimage import binary_erosion
    texture = np.asarray(Image.open(pool.root / 'capture/random_texture.png').convert('RGB'), dtype=np.float32) / 255.
    assert texture.shape == (64, 64, 3)
    predict = jax.jit(lambda q, c: render_one(q, c, native=True, samples=samples))
    by_view = {name: [] for name in ['base', 'wrist']}
    for index, (real, mask) in pool.validation_rows():
        pred = np.asarray(predict(jnp.asarray(texture), {k: jnp.asarray(v[index]) for k, v in pool.data.items()}))
        pred224 = np.asarray(resize224(jnp.asarray(pred)))
        for view, name in enumerate(['base', 'wrist']):
            interior = binary_erosion(mask[view], structure=np.ones((5, 5), bool))
            area = int(mask[view].sum())
            n = int(interior.sum())
            error = float(np.sqrt(np.mean((pred[view][interior] - real[view][interior] / 255.) ** 2))) if n else None
            boundary_error = float(np.sqrt(np.mean((pred[view][mask[view]] - real[view][mask[view]] / 255.) ** 2))) if area else None
            real224 = np.asarray(Image.fromarray(real[view]).resize((224, 224), Image.Resampling.BILINEAR)) / 255.
            mask224 = np.asarray(Image.fromarray(mask[view].astype(np.uint8) * 255).resize((224, 224), Image.Resampling.NEAREST)) > 0
            interior224 = binary_erosion(mask224, structure=np.ones((3, 3), bool))
            error224 = float(np.sqrt(np.mean((pred224[view][interior224] - real224[interior224]) ** 2))) if interior224.any() else None
            by_view[name].append(dict(frame=index, visible_pixels=area, interior_pixels=n,
                                      interior_RMSE=error, mask_RMSE=boundary_error,
                                      model224_interior_pixels=int(interior224.sum()), model224_interior_RMSE=error224))
    result = dict(max_allowed_interior_RMSE=max_rmse, sampling_offsets=list(MSAA_OFFSETS) if samples == 4 else [[0., 0.]],
                  samples=samples, views=by_view, complete=False)
    wrist_values = [r['interior_RMSE'] for r in by_view['wrist'] if r['interior_pixels'] >= 20]
    issues = []
    if not wrist_values:
        issues.append('wrist proxy has no usable random validation samples')
    for name in ['base', 'wrist']:
        errors = [r['interior_RMSE'] for r in by_view[name] if r['interior_pixels'] >= 20]
        result[name + '_interior_samples'] = len(errors)
        result[name + '_interior_RMSE_max'] = max(errors) if errors else None
        if errors:
            if not np.isfinite(errors).all() or max(errors) > max_rmse:
                issues.append('proxy/native random mismatch: ' + name)
        model_errors = [r['model224_interior_RMSE'] for r in by_view[name] if r['model224_interior_pixels'] >= 20]
        result[name + '_model224_interior_RMSE_max'] = max(model_errors) if model_errors else None
        if model_errors and (not np.isfinite(model_errors).all() or max(model_errors) > max_rmse):
            issues.append('proxy/model224 random mismatch: ' + name)
    result['base_interior_unavailable'] = result['base_interior_samples'] == 0
    result['issues'] = issues
    result['complete'] = not issues
    return result
