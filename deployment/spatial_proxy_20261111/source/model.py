"""TinyViT-5M + spatial B/G skips + state/query geometry and shared S.

No task/episode IDs or proxy target tensors enter forward(). timm is required;
the architecture never silently falls back to a different encoder.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F
try:
    from .renderer import texture_corners_to_image_H
except ImportError:
    from renderer import texture_corners_to_image_H

QUERY_FIELDS = (("center_xyz", 3), ("size_wh", 2), ("plane_z", 1),
                ("reserved_gauge_DISABLED", 9), ("texture_hw", 2),
                ("orientation_wxyz", 4))
QUERY_DIM = sum(n for _, n in QUERY_FIELDS)


def pack_exported_query(fields: dict, *, device=None) -> tuple[Tensor, Tensor]:
    """Pack one sidecar's known query fields; never reads a target H/B/G/S.

    Returns [active_cameras,21] values and per-scalar bool validity. Exported
    query_known_valid has six *field-group* columns, not 21 scalar columns.
    The historical canonical gauge was derived from true camera H/height. It is
    label-only: this function never reads it, and its reserved nine inputs are
    zero/invalid even if the old sidecar declares that field group known.
    """
    def t(key):
        return torch.as_tensor(fields[key], device=device, dtype=torch.float32)
    center = t("query_center_xyz")
    n = len(center)
    values = [center.reshape(n, 3), t("query_size_wh").reshape(n, 2),
              t("query_plane_z").reshape(1, 1).expand(n, 1),
              center.new_zeros(n, 9),
              t("query_texture_hw").reshape(1, 2).expand(n, 2),
              t("query_orientation_wxyz").reshape(n, 4)]
    group_valid = torch.as_tensor(fields["query_known_valid"], device=device, dtype=torch.bool).clone()
    if group_valid.shape != (n, 6):
        raise ValueError("query_known_valid must have six field-group flags")
    group_valid[:, 3] = False
    valid = torch.cat([group_valid[:, i:i+1].expand(n, width)
                       for i, (_, width) in enumerate(QUERY_FIELDS)], -1)
    return torch.cat(values, -1), valid


class SeparableBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                                 nn.Conv2d(channels, channels, 1), nn.GroupNorm(8, channels), nn.GELU())

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SpatialProxyEstimator(nn.Module):
    """One timestamp, shared encoder across all valid cameras.

    All observations have the same native H,W within a batch. Group different
    resolutions into separate batches. See MODEL_NOTES.md for the input schema.
    """
    def __init__(self, *, num_camera_ids: int = 3, num_imaging_configs: int = 6,
                 pretrained: bool = False, decoder_width: int = 48):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("Install official timm (CPU verified: 0.9.10); TinyViT is required") from e
        if decoder_width % 8:
            raise ValueError("decoder_width must be divisible by 8")
        self.encoder_name = "tiny_vit_5m_224.in1k"
        self.timm_version = timm.__version__
        self.pretrained_requested = bool(pretrained)
        self.encoder = timm.create_model(self.encoder_name, pretrained=pretrained,
                                         features_only=True, out_indices=(0, 1, 2, 3))
        channels = self.encoder.feature_info.channels()
        if channels != [64, 128, 160, 320]:
            raise RuntimeError(f"Unexpected TinyViT features: {channels}")
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.register_buffer("query_scale", torch.tensor([1., 1., 1., .1, .1, 1.,
                                                           *([1.] * 9), 128., 128., 1., 1., 1., 1.]))
        self.camera_embedding = nn.Embedding(num_camera_ids, 16)
        # Two EEFs: xyz + first two rotation columns + gripper, plus two pairs of validity flags.
        self.state_mlp = nn.Sequential(nn.Linear(24, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.query_mlp = nn.Sequential(nn.Linear(QUERY_DIM*2 + 4, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        condition_dim = 128 + 64 + 16
        self.geometry_head = nn.Sequential(nn.Linear(320 + condition_dim, 192), nn.GELU(), nn.Linear(192, 8))
        nn.init.normal_(self.geometry_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.geometry_head[-1].bias)
        self.register_buffer("initial_corners", torch.tensor([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]))
        self.lateral = nn.ModuleList([nn.Conv2d(c, decoder_width, 1) for c in channels])
        self.skip_blocks = nn.ModuleList([SeparableBlock(decoder_width) for _ in channels])
        self.film = nn.Linear(condition_dim, len(channels)*decoder_width*2)
        nn.init.normal_(self.film.weight, std=1e-3)
        nn.init.zeros_(self.film.bias)
        self.rgb_skip = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU())
        self.pixel_fusion = nn.Sequential(nn.Conv2d(decoder_width+16+3, 32, 1), nn.GELU(), SeparableBlock(32))
        self.background_head = nn.Conv2d(32, 3, 1)
        self.gain_head = nn.Conv2d(32, 3, 1)
        nn.init.normal_(self.background_head.weight, std=1e-3)
        nn.init.zeros_(self.background_head.bias)
        nn.init.normal_(self.gain_head.weight, std=1e-3)
        nn.init.constant_(self.gain_head.bias, -4.)
        # Configuration-level S: not conditioned on RGB, state, query, or sample ID.
        self.s_embedding = nn.Embedding(num_imaging_configs, 8)
        self.s_mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 3))
        nn.init.normal_(self.s_mlp[-1].weight, std=1e-3)
        nn.init.constant_(self.s_mlp[-1].bias, .01)

    def metadata(self) -> dict:
        return dict(encoder=self.encoder_name, timm_version=self.timm_version,
                    pretrained_requested=self.pretrained_requested,
                    encoder_parameters=sum(p.numel() for p in self.encoder.parameters()),
                    total_parameters=sum(p.numel() for p in self.parameters()),
                    trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    query_fields=QUERY_FIELDS, query_dim=QUERY_DIM,
                    disabled_input_dimensions=list(range(6, 15)),
                    canonical_gauge_is_model_input=False,
                    H_direction="native_image_pixel_to_texture_pixel", texture_hw=[128, 128],
                    S_scope="shared imaging configuration only", single_timestamp=True)

    def forward(self, *, native_clean: Tensor, camera_valid: Tensor,
                T_world_eef: Tensor, eef_valid: Tensor, gripper: Tensor, gripper_valid: Tensor,
                query: Tensor, query_valid: Tensor, camera_id: Tensor,
                imaging_config_id: Tensor, imaging_config_valid: Tensor,
                image_orientation_xy: Tensor | None = None) -> dict[str, Tensor]:
        if native_clean.ndim != 5 or native_clean.shape[2] != 3:
            raise ValueError("native_clean must be [batch,camera,3,native_h,native_w]")
        b, v, _, h, w = native_clean.shape
        if camera_valid.shape != (b, v) or query.shape != (b, v, QUERY_DIM) or query_valid.shape != query.shape:
            raise ValueError("Invalid camera/query shapes")
        if T_world_eef.shape != (b, 2, 4, 4) or eef_valid.shape != (b, 2):
            raise ValueError("Two explicit EEF slots [B,2,4,4] / [B,2] required")
        if gripper.shape != (b, 2) or gripper_valid.shape != (b, 2):
            raise ValueError("gripper and gripper_valid must be [B,2]")
        for x in (camera_id, imaging_config_id, imaging_config_valid):
            if x.shape != (b, v):
                raise ValueError("ID/validity tensors must be [B,V]")
        if image_orientation_xy is None:
            image_orientation_xy = native_clean.new_ones(b, v, 2)
        if image_orientation_xy.shape != (b, v, 2):
            raise ValueError("image_orientation_xy must be [B,V,2]")
        active = camera_valid.bool().reshape(-1).nonzero(as_tuple=False)[:, 0]
        B = native_clean.new_zeros((b*v, 3, h, w))
        G = torch.zeros_like(B)
        H = torch.eye(3, device=native_clean.device, dtype=torch.float64).expand(b*v, 3, 3).clone()
        corners = native_clean.new_zeros((b*v, 4, 2))
        S = native_clean.new_zeros((b*v, 3))
        h_valid = torch.zeros(b*v, dtype=torch.bool, device=native_clean.device)
        s_valid = torch.zeros_like(h_valid)
        if len(active):
            clean = native_clean.reshape(b*v, 3, h, w)[active]
            if not torch.isfinite(clean).all() or clean.min() < 0 or clean.max() > 1:
                raise ValueError("Valid clean RGB must be finite and in [0,1]")
            cid = camera_id.reshape(-1)[active].long()
            if (cid < 0).any() or (cid >= self.camera_embedding.num_embeddings).any():
                raise ValueError("A valid camera needs a declared camera_id")
            # Inactive/missing views never enter TinyViT BatchNorm or the decoder.
            encoded = F.interpolate(clean, (224, 224), mode="bilinear", align_corners=False, antialias=True)
            features = self.encoder((encoded-self.rgb_mean)/self.rgb_std)
            ef = eef_valid.bool()
            gf = gripper_valid.bool() & ef
            E = torch.where(ef[..., None, None], T_world_eef, torch.zeros_like(T_world_eef))
            grip = torch.where(gf, gripper, torch.zeros_like(gripper))
            state = torch.cat([E[..., :3, 3], E[..., :3, :2].reshape(b, 2, 6), grip[..., None]], -1)
            state = torch.cat([state.reshape(b, 20), ef.to(state.dtype), gf.to(state.dtype)], -1)
            if not torch.isfinite(state).all():
                raise ValueError("Known EEF/gripper inputs must be finite")
            state_feature = self.state_mlp(state)[:, None].expand(-1, v, -1).reshape(b*v, -1)[active]
            qmask = query_valid.bool().reshape(b*v, QUERY_DIM)[active]
            if qmask[:, 6:15].any():
                raise ValueError("Canonical gauge is target-derived and forbidden as a model input; query_valid[6:15] must be false")
            qvalues = query.reshape(b*v, QUERY_DIM)[active]
            qvalues = torch.where(qmask, qvalues, torch.zeros_like(qvalues)) / self.query_scale
            if not torch.isfinite(qvalues).all():
                raise ValueError("Known query inputs must be finite")
            orientation = image_orientation_xy.reshape(b*v, 2)[active]
            if not torch.isfinite(orientation).all():
                raise ValueError("Known image orientation must be finite")
            hw = clean.new_tensor([h/256, w/256])[None].expand(len(active), -1)
            qfeature = self.query_mlp(torch.cat([qvalues, qmask.to(qvalues.dtype), hw, orientation], -1))
            condition = torch.cat([state_feature, qfeature, self.camera_embedding(cid)], -1)
            pooled = features[-1].mean((-2, -1))
            normalized_corners = self.initial_corners + self.geometry_head(torch.cat([pooled, condition], -1)).reshape(-1, 4, 2)
            current_corners = (normalized_corners+1)*clean.new_tensor([w, h])/2-.5
            with torch.autocast(device_type=clean.device.type, enabled=False):
                current_H, current_valid = texture_corners_to_image_H(current_corners, (h, w))
            H = H.index_copy(0, active, current_H)
            corners = corners.index_copy(0, active, current_corners)
            h_valid = h_valid.index_copy(0, active, current_valid)
            film = self.film(condition).reshape(len(active), 4, 2, -1)
            x = None
            for level in reversed(range(4)):
                skip = self.lateral[level](features[level])
                x = skip if x is None else skip+F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
                x = self.skip_blocks[level](x)
                scale, bias = film[:, level].unbind(1)
                x = x * (1+.1*scale[:, :, None, None]) + .1*bias[:, :, None, None]
            x = F.interpolate(x, (h, w), mode="bilinear", align_corners=False)
            pixels = self.pixel_fusion(torch.cat([x, self.rgb_skip(clean), clean], 1))
            B = B.index_copy(0, active, (clean+self.background_head(pixels).tanh()).clamp(0, 1))
            G = G.index_copy(0, active, self.gain_head(pixels).sigmoid().to(G.dtype))
            conf = imaging_config_id.reshape(-1)[active].long()
            known = imaging_config_valid.bool().reshape(-1)[active]
            if ((conf[known] < 0) | (conf[known] >= self.s_embedding.num_embeddings)).any():
                raise ValueError("Known imaging configuration ID outside declared vocabulary")
            safe_conf = torch.where(known, conf, torch.zeros_like(conf))
            current_S = self.s_mlp(self.s_embedding(safe_conf)).tanh() * clean.new_tensor([2., 2., 3.])
            current_S = torch.where(known[:, None], current_S, torch.zeros_like(current_S))
            S = S.index_copy(0, active, current_S)
            s_valid = s_valid.index_copy(0, active, known)
        return dict(H=H.reshape(b, v, 3, 3), H_valid=h_valid.reshape(b, v),
                    corners_native=corners.reshape(b, v, 4, 2),
                    B=B.reshape(b, v, 3, h, w), G=G.reshape(b, v, 3, h, w),
                    S=S.reshape(b, v, 3), S_valid=s_valid.reshape(b, v),
                    camera_valid=camera_valid.bool())
