"""Seeded, differentiable texture-plane EOT with a fixed physical footprint."""
import math

import numpy as np


IDENTITY = dict(angle_deg=0., dx_fraction=0., dy_fraction=0., scale=1., gain=1.)
RANGES = dict(angle_deg=[-15., 15.], dx_fraction=[-.05, .05],
              dy_fraction=[-.05, .05], scale=[.9, 1.1], gain=[.9, 1.1])


def sample(rng):
    return {key: float(rng.uniform(*bounds)) for key, bounds in RANGES.items()}


def transform(q, draw):
    """Rotate/translate/scale printed content, fill with gray, then apply gain.

    dx/dy are fractions of the full texture width/height. The inverse affine
    sampling map represents forward p_out = scale * R * p_in + translation.
    This changes content inside the square, never the square's world geometry.
    """
    import torch
    import torch.nn.functional as F
    if draw == IDENTITY:
        return q
    # A photometric-only ablation must not add floating-point interpolation
    # through a nominally identity grid. Full random EOT never takes this path.
    if all(draw[key] == IDENTITY[key] for key in ['angle_deg','dx_fraction','dy_fraction','scale']):
        return (q * draw['gain']).clamp(0, 1)
    angle = math.radians(draw['angle_deg'])
    c, s, scale = math.cos(angle), math.sin(angle), draw['scale']
    tx, ty = 2 * draw['dx_fraction'], 2 * draw['dy_fraction']
    theta = q.new_tensor([[c / scale, s / scale, -(c * tx + s * ty) / scale],
                          [-s / scale, c / scale, (s * tx - c * ty) / scale]])[None]
    grid = F.affine_grid(theta, q.shape, align_corners=False)
    warped = F.grid_sample(q - .5, grid, mode='bilinear', padding_mode='zeros',
                           align_corners=False) + .5
    return (warped * draw['gain']).clamp(0, 1)


def plan(steps=500, batch=8, seed=2026102901):
    rng = np.random.default_rng(seed)
    return [[sample(rng) for _ in range(batch)] for _ in range(steps)]
