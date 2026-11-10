"""Differentiable physical-texture proxy calibrated against MuJoCo.

Q128 is in exactly the PNG orientation uploaded to the renderer. Calibrated
UV homography replaces the approximate image-space export/return chain.
Native256 black/white captures provide each state's spatial illumination and
boundary mixture; trilinear mipmapping and production downsampling remain in
the differentiable forward. This is an approximation, not renderer gradients.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def ste_u8(x):
    rounded = (x.clamp(0, 1)*255).round()/255
    return x + (rounded-x).detach()


class WorldTextureProxy:
    def __init__(self, calibration, device='cpu', fit=None, samples=4):
        self.source = str(calibration)
        data = np.load(calibration, allow_pickle=False)
        self.H = torch.tensor(data['image256_to_texture_pixel_H'], dtype=torch.float32, device=device)
        self.black = torch.from_numpy(data['black256'].astype(np.float32).transpose(0, 3, 1, 2)/255).to(device)
        white = torch.from_numpy(data['white256'].astype(np.float32).transpose(0, 3, 1, 2)/255).to(device)
        self.gain = (white-self.black).clamp_min(0)
        fit_config = json.loads(Path(fit).read_text())['parameters'] if fit else {}
        samples = int(fit_config.get('samples', samples))
        self.samples = samples
        if samples not in (1, 4):
            raise ValueError('Only native-center or four-sample approximation implemented')
        # Default NVIDIA four-sample pattern, in native image coordinates after
        # both axes have been flipped. Exact boundary sample coverage is still
        # captured by black/white and remains an approximation for textured edges.
        self.offsets = [(0., 0.)] if samples == 1 else [(.125, .375), (-.375, .125), (.375, -.125), (-.125, -.375)]
        yy, xx = torch.meshgrid(torch.arange(256, device=device), torch.arange(256, device=device), indexing='ij')
        self.xy = torch.stack([xx, yy], -1).float()
        self.config = {'dx_native_px': 0., 'dy_native_px': 0., 'lod_bias': 0., 'samples': samples}
        self.config.update(fit_config)

    def grids(self, dx=0., dy=0.):
        grids, lod = [], None
        for ox, oy in self.offsets:
            xy = self.xy + self.xy.new_tensor([ox, oy]) + torch.stack([
                torch.as_tensor(dx, dtype=self.xy.dtype, device=self.xy.device),
                torch.as_tensor(dy, dtype=self.xy.dtype, device=self.xy.device)])
            p = torch.cat([xy, torch.ones_like(xy[..., :1])], -1) @ self.H.T
            texture_xy = p[..., :2]/p[..., 2:]
            uv = (texture_xy+.5)/128
            grids.append((2*uv-1)[None])
            if lod is None:
                # Perspective derivative in physical texture pixel units.
                z = p[..., 2:]
                du = (self.H[:2, 0] - texture_xy*self.H[2, 0])/z
                dv = (self.H[:2, 1] - texture_xy*self.H[2, 1])/z
                rho = torch.maximum(du.square().sum(-1).sqrt(), dv.square().sum(-1).sqrt())
                lod = rho.clamp_min(1e-6).log2().clamp_min(0)
        return grids, lod

    def texture_native(self, q, dx=0., dy=0., lod_bias=0., quantize=True):
        if q.shape != (1, 3, 128, 128):
            raise ValueError('Physical texture must have shape 1 x 3 x 128 x 128')
        grids, lod = self.grids(dx, dy)
        lod = (lod+lod_bias).clamp(0, 7)
        current = ste_u8(q) if quantize else q
        out = torch.zeros((1, 3, 256, 256), device=q.device, dtype=q.dtype)
        for level in range(8):
            weight = (1-(lod-level).abs()).clamp_min(0)[None, None]
            sampled = sum(F.grid_sample(current, grid, mode='bilinear', padding_mode='border',
                                        align_corners=False) for grid in grids)/len(grids)
            out = out + sampled*weight
            if level < 7:
                current = F.avg_pool2d(current, 2)
                if quantize:
                    current = ste_u8(current)
        return out

    def __call__(self, q, k, rng=None, phase_jitter=.0, color_jitter=.0, lod_jitter=.0,
                 quantize=True, native=False):
        c = self.config
        dx, dy, bias = c['dx_native_px'], c['dy_native_px'], c['lod_bias']
        color = 1.
        if rng is not None:
            dx, dy = np.array([dx, dy]) + rng.uniform(-phase_jitter, phase_jitter, 2)
            bias += rng.uniform(-lod_jitter, lod_jitter)
            color = q.new_tensor(rng.uniform(1-color_jitter, 1+color_jitter, 3)).view(1, 3, 1, 1)
        image = self.black[k:k+1] + self.gain[k:k+1]*color*self.texture_native(q, dx, dy, bias, quantize)
        image = image.clamp(0, 1)
        if quantize:
            image = ste_u8(image)
        if native:
            return image
        # antialias=True matches PIL bilinear filtering on float RGB, with
        # remaining integer rounding checked explicitly on renderer anchors.
        image = F.interpolate(image, (224, 224), mode='bilinear', align_corners=False, antialias=True)
        return ste_u8(image) if quantize else image
