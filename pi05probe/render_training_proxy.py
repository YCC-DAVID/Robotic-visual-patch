"""Differentiable, fixed-camera export/resampling proxy; not a MuJoCo renderer.

Fit photometry/alignment on a random calibration pattern, then test transfer to
optimized patterns before training. Continuous bilinear interpolation omits
PNG quantization and OpenCV's interpolation-table rounding. Occlusion/boundary
compositing remain the training pool's masks. Camera motion needs new grids.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def sampling_grid(output_to_input, out_side, in_side):
    yy, xx = np.mgrid[:out_side, :out_side]
    points = np.stack([xx, yy, np.ones_like(xx)], -1) @ output_to_input.T
    xy = points[..., :2] / points[..., 2:3]
    return torch.from_numpy((2 * xy / (in_side - 1) - 1).astype(np.float32))[None]


class RenderTrainingProxy:
    def __init__(self, calibration, quad224, device, stage='full'):
        self.config = json.loads(Path(calibration).read_text())
        cfg = self.config
        if cfg.get('schema') != 'wam-render-proxy-v1':
            raise ValueError('Unsupported calibration schema')
        if stage not in ('resample', 'resample_align', 'full'):
            raise ValueError('Unknown render proxy stage')
        if not np.allclose(quad224, cfg['quad224_yx'], atol=1e-4, rtol=0):
            raise ValueError('Calibration belongs to a different projected patch quad; recapture/refit')
        self.stage = stage
        n = int(cfg['texture_resolution'])
        H = cv2.getPerspectiveTransform(np.asarray(quad224, np.float32)[:, ::-1],
                                       np.array([[0, 0], [n-1, 0], [n-1, n-1], [0, n-1]], np.float32))
        self.export_grid = sampling_grid(np.linalg.inv(H), n, 224).to(device)
        self.return_grid = sampling_grid(H, 224, n).to(device)
        self.identity_grid = sampling_grid(np.eye(3), 224, 224).to(device)
        self.gain = torch.tensor(cfg['gain_rgb'], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.bias = torch.tensor(cfg['bias_rgb'], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def __call__(self, pattern, rng=None, color_jitter=0., shift_jitter_px=0.):
        if pattern.shape[-3:] != (3, 224, 224):
            raise ValueError('Expected B x 3 x 224 x 224 RGB [0,1]')
        if color_jitter < 0 or shift_jitter_px < 0:
            raise ValueError('Jitter ranges must be nonnegative')
        cfg = self.config
        b = pattern.shape[0]
        sample = lambda x, grid: F.grid_sample(x, grid.expand(b, -1, -1, -1),
                                              mode='bilinear', padding_mode='zeros', align_corners=True)
        # UV flip is undone by the calibrated physical projection; both grids
        # operate in the same (unflipped) texture coordinate system.
        rendered = sample(sample(pattern, self.export_grid), self.return_grid)
        if self.stage == 'resample':
            return rendered
        dx, dy = float(cfg['dx_px']), float(cfg['dy_px'])
        if rng is not None and shift_jitter_px:
            dx, dy = np.array([dx, dy]) + rng.uniform(-shift_jitter_px, shift_jitter_px, 2)
        grid = self.identity_grid - pattern.new_tensor([2*dx/223, 2*dy/223])
        rendered = sample(rendered, grid)
        if self.stage == 'resample_align':
            return rendered
        gain, bias = self.gain, self.bias
        if rng is not None and color_jitter:
            gain = gain * pattern.new_tensor(rng.uniform(1-color_jitter, 1+color_jitter, 3)).view(1, 3, 1, 1)
            bias = bias + pattern.new_tensor(rng.uniform(-.1*color_jitter, .1*color_jitter, 3)).view(1, 3, 1, 1)
        return (rendered * gain + bias).clamp(0, 1)
