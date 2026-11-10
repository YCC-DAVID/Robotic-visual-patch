"""Rectangular world Q -> differentiable perspective -> visible-mask composite.

Geometry-only baseline: bilinear sampling directly at model224, no photometry,
mipmap, learned sampling phase, EOT or native256 downsample in this forward.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class PerspectiveOnlyProxy:
    def __init__(self, calibration, device='cpu', fit=None):
        data = np.load(calibration, allow_pickle=False)
        geometry = json.loads(Path(fit).read_text())
        assert geometry['complete'] and geometry['all_camera_projections_constant']
        self.clean = torch.from_numpy(data['clean_base'].transpose(0, 3, 1, 2).astype(np.float32)/255).to(device)
        self.mask = torch.from_numpy(data['mask'][:, None].astype(np.float32)).to(device)
        H = torch.tensor(np.linalg.inv(geometry['Q128_to_image224_H']), dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(torch.arange(224, device=device), torch.arange(224, device=device), indexing='ij')
        p = torch.stack([xx, yy, torch.ones_like(xx)], -1).float() @ H.T
        uv = p[..., :2]/p[..., 2:]
        self.grid = (2*(uv+.5)/128-1)[None]

    def __call__(self, q, k, rng=None, phase_jitter=0, color_jitter=0, lod_jitter=0):
        assert rng is None and phase_jitter == color_jitter == lod_jitter == 0
        patch = F.grid_sample(q, self.grid, mode='bilinear', padding_mode='border', align_corners=False)
        mask = self.mask[k:k+1]
        return self.clean[k:k+1]*(1-mask)+patch*mask
