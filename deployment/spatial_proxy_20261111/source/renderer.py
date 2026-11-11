"""Differentiable native B/G camera proxy with the legacy quantization order.

H maps native IMAGE pixel centres to TEXTURE pixel centres. There is no mask
multiplier: spatial visibility is represented by G itself. No H/grid cache is
used, because H and S can be outputs of trainable modules.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def ste_u8(x: Tensor) -> Tensor:
    rounded = (x.clamp(0, 1) * 255).round() / 255
    return x + (rounded - x).detach()


def texture_corners_to_image_H(corners: Tensor, native_hw: tuple[int, int],
                               texture_hw: tuple[int, int] = (128, 128)) -> tuple[Tensor, Tensor]:
    """Four native image corners -> native IMAGE-to-TEXTURE H via float64 DLT.

    Corner order is texture TL, TR, BR, BL, using pixel centres 0 and N-1.
    Invalid/degenerate quads receive a finite centred-rectangle fallback and a
    false validity flag. The caller must retain corner supervision for invalid
    predictions; the fallback is not an accepted geometric prediction.
    """
    if corners.shape[-2:] != (4, 2):
        raise ValueError("corners must end in [4,2]")
    h, w = native_hw
    th, tw = texture_hw
    if min(h, w, th, tw) < 2:
        raise ValueError("Image and texture sides must be >=2")
    c = corners.to(torch.float64)
    finite = torch.isfinite(c).all(dim=(-2, -1))
    safe = torch.nan_to_num(c)
    edges = safe.roll(-1, -2) - safe
    following = edges.roll(-1, -2)
    cross = edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]
    convex = (cross > 1e-8).all(-1) | (cross < -1e-8).all(-1)
    valid = finite & convex
    fallback = c.new_tensor([[w*.25, h*.25], [w*.75, h*.25],
                             [w*.75, h*.75], [w*.25, h*.75]])
    c = torch.where(valid[..., None, None], safe, fallback)
    # Fixed image/texture normalization avoids the conditioning of native-pixel
    # DLT without detaching the input corners or differentiating an SVD gauge.
    xy = 2 * (c + .5) / c.new_tensor([w, h]) - 1
    uv = c.new_tensor([[0, 0], [tw-1, 0], [tw-1, th-1], [0, th-1]])
    uv = 2 * (uv + .5) / c.new_tensor([tw, th]) - 1
    uv = uv.expand_as(xy)
    x, y = xy.unbind(-1)
    u, v = uv.unbind(-1)
    z, o = torch.zeros_like(x), torch.ones_like(x)
    a = torch.stack([torch.stack([x, y, o, z, z, z, -u*x, -u*y], -1),
                     torch.stack([z, z, z, x, y, o, -v*x, -v*y], -1)], -2)
    a = a.reshape(*c.shape[:-2], 8, 8)
    rhs = torch.stack([u, v], -1).reshape(*c.shape[:-2], 8, 1)
    solution, info = torch.linalg.solve_ex(a, rhs)
    # Convex quads should be solvable. Fail explicitly on numerical failure;
    # returning a silent zero matrix would corrupt both renders and gradients.
    if (info != 0).any() or not torch.isfinite(solution).all():
        raise FloatingPointError("DLT solve failed; inspect corner predictions")
    hn = torch.cat([solution[..., 0], torch.ones_like(solution[..., :1, 0])], -1)
    hn = hn.reshape(*c.shape[:-2], 3, 3)
    ti = c.new_tensor([[2/w, 0, 1/w-1], [0, 2/h, 1/h-1], [0, 0, 1]])
    tt_inv = c.new_tensor([[tw/2, 0, (tw-1)/2], [0, th/2, (th-1)/2], [0, 0, 1]])
    result = tt_inv @ hn @ ti
    return result, valid


def project(H: Tensor, xy: Tensor) -> Tensor:
    p = torch.cat([xy, torch.ones_like(xy[..., :1])], -1) @ H.transpose(-1, -2)
    return p[..., :2] / p[..., 2:]


def native_grid(H: Tensor, S: Tensor, native_hw: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Legacy grid and unbiased LOD in float64, retaining H/S autograd."""
    if H.ndim != 3 or H.shape[-2:] != (3, 3) or S.shape != (len(H), 3):
        raise ValueError("H [N,3,3], S [N,3] required")
    H = H.to(dtype=torch.float64)
    S = S.to(device=H.device, dtype=torch.float64)
    h, w = native_hw
    yy, xx = torch.meshgrid(torch.arange(h, device=H.device),
                           torch.arange(w, device=H.device), indexing="ij")
    xy = torch.stack([xx, yy], -1).double()[None] + S[:, None, None, :2]
    p = torch.cat([xy, torch.ones_like(xy[..., :1])], -1) @ H[:, None].transpose(-1, -2)
    z = p[..., 2:]
    z = torch.where(z.abs() < 1e-9, torch.where(z >= 0, 1e-9, -1e-9), z)
    uv = p[..., :2] / z
    du = (H[:, None, None, :2, 0] - uv * H[:, None, None, 2:3, 0]) / z
    dv = (H[:, None, None, :2, 1] - uv * H[:, None, None, 2:3, 1]) / z
    rho = torch.maximum(du.square().sum(-1).sqrt(), dv.square().sum(-1).sqrt())
    # This ordering is intentional: log2(rho).clamp_min(0), THEN lod_bias.
    lod = rho.clamp_min(1e-6).log2().clamp_min(0)
    grid = 2 * (uv + .5) / 128 - 1
    lod = (lod + S[:, 2, None, None]).clamp(0, 7)
    return grid, lod


def render_camera_proxy(q: Tensor, H: Tensor, B: Tensor, G: Tensor, S: Tensor,
                        *, quantize: bool = True, native: bool = False) -> Tensor:
    """B+G*sample(Q), matching CameraProxy; no additional footprint/alpha mask.

    q: [1 or N,3,128,128]; H: [N,3,3]; B/G: [N,3,Hnative,Wnative];
    S: [N,3] = dx_native_px,dy_native_px,lod_bias. Sampling/compositing is
    float32; geometry is float64 even under an outer autocast context.
    """
    if B.ndim != 4 or B.shape != G.shape or B.shape[1] != 3:
        raise ValueError("B and G must have equal [N,3,H,W] shapes")
    n = len(B)
    if q.shape[1:] != (3, 128, 128) or len(q) not in (1, n):
        raise ValueError("Expected q [1 or N,3,128,128]")
    if len(H) != n or len(S) != n:
        raise ValueError("Batch size mismatch")
    # Explicitly disable mixed precision; model/policy callers may enable it.
    with torch.autocast(device_type=q.device.type, enabled=False):
        grid, lod = native_grid(H, S, B.shape[-2:])
        grid, lod = grid.float(), lod.float()
        q = q.float().expand(n, -1, -1, -1)
        current = ste_u8(q) if quantize else q
        out = torch.zeros_like(B, dtype=torch.float32)
        for level in range(8):
            weight = (1 - (lod - level).abs()).clamp_min(0)[:, None]
            out = out + F.grid_sample(current, grid, mode="bilinear", padding_mode="border",
                                      align_corners=False) * weight
            if level < 7:
                current = F.avg_pool2d(current, 2)
                if quantize:
                    current = ste_u8(current)
        image = (B.float() + G.float() * out).clamp(0, 1)
        if quantize:
            image = ste_u8(image)
        if native:
            return image
        image = F.interpolate(image, (224, 224), mode="bilinear", align_corners=False, antialias=True)
        return ste_u8(image) if quantize else image
