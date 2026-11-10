"""Three binary imaging factors at a fixed calibrated world-to-image mapping.

M: trilinear mipmap filtering versus level-zero bilinear sampling.
A: antialiasing in the final 256->224 bilinear resize (not renderer MSAA).
C: calibrated black+gain photometry versus unlit RGB with measured coverage.
Geometry, native resolution, coverage data and uint8 STE stay fixed.
"""
import numpy as np
import torch
import torch.nn.functional as F
from render_world_proxy import WorldTextureProxy as OriginalProxy, ste_u8

VARIANTS = ['000', '100', '010', '001', '110', '101', '011', '111']


class WorldTextureProxy(OriginalProxy):
    def __init__(self, calibration, device='cpu', fit=None, variant='111'):
        super().__init__(calibration, device, fit)
        assert variant in VARIANTS and self.samples == 1
        self.variant = variant
        self.mipmap, self.antialias, self.photometry = [x == '1' for x in variant]
        with np.load(calibration, allow_pickle=False) as d:
            alpha = (d['unlit_white256'].astype(float)-d['unlit_black256'].astype(float))/255
            assert np.max(np.ptp(alpha, axis=-1)) <= 1/255+1e-7
            assert alpha.min() >= 0 and alpha.max() <= 1
            alpha = alpha.mean(-1)[:, None].astype(np.float32)
            self.alpha = torch.from_numpy(alpha).to(device)
            self.background = torch.from_numpy(d['clean_img256'].astype(np.float32).transpose(0,3,1,2)/255).to(device)

    def texture_native(self, q, dx=0., dy=0., lod_bias=0., quantize=True):
        if self.mipmap:
            return super().texture_native(q, dx, dy, lod_bias, quantize)
        grids, _ = self.grids(dx, dy)
        return F.grid_sample(ste_u8(q) if quantize else q, grids[0], mode='bilinear',
                             padding_mode='border', align_corners=False)

    def __call__(self, q, k, rng=None, phase_jitter=0., color_jitter=0., lod_jitter=0.,
                 quantize=True, native=False):
        assert rng is None, 'factorial is nominal; no random EOT transforms'
        c = self.config
        texture = self.texture_native(q, c['dx_native_px'], c['dy_native_px'], c['lod_bias'], quantize)
        if self.photometry:
            image = self.black[k:k+1] + self.gain[k:k+1]*texture
        else:
            alpha = self.alpha[k:k+1]
            image = (1-alpha)*self.background[k:k+1] + alpha*texture
        image = image.clamp(0,1)
        if quantize: image = ste_u8(image)
        if native: return image
        image = F.interpolate(image, (224,224), mode='bilinear', align_corners=False,
                              antialias=self.antialias)
        return ste_u8(image) if quantize else image
