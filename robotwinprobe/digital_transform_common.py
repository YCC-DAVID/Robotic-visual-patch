"""Frozen digital patch transforms; keep the original perspective compositor.

Only simulator segmentation is used for visibility. Rendered patch RGB never
enters a policy input. Brightness is an explicit sRGB input gain, not relighting.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / 'robotwinprobe'
TRANSFER = PROBE / 'out/render_transfer'
METHODS = ['clean', 'full', 'random']
CONDITIONS = [
    dict(name='nominal', label='原位置', xy_cm=[0, 0], yaw_deg=0, gain=1.),
    dict(name='x_minus1', label='X −1 cm', xy_cm=[-1, 0], yaw_deg=0, gain=1.),
    dict(name='x_plus1', label='X +1 cm', xy_cm=[1, 0], yaw_deg=0, gain=1.),
    dict(name='y_minus1', label='Y −1 cm', xy_cm=[0, -1], yaw_deg=0, gain=1.),
    dict(name='y_plus1', label='Y +1 cm', xy_cm=[0, 1], yaw_deg=0, gain=1.),
    dict(name='yaw_minus10', label='平面旋转 −10°', xy_cm=[0, 0], yaw_deg=-10, gain=1.),
    dict(name='yaw_plus10', label='平面旋转 +10°', xy_cm=[0, 0], yaw_deg=10, gain=1.),
    dict(name='gain085', label='输入亮度 ×0.85', xy_cm=[0, 0], yaw_deg=0, gain=.85),
    dict(name='gain115', label='输入亮度 ×1.15', xy_cm=[0, 0], yaw_deg=0, gain=1.15),
]


def dump(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def textures():
    from PIL import Image
    return {k: np.array(Image.open(p).convert('RGB')) for k, p in texture_paths().items()}


def texture_paths():
    return dict(full=TRANSFER/'train_digital_full/texture_0500.png',
                random=TRANSFER/'controls/random.png')


def condition(name):
    return next(c for c in CONDITIONS if c['name'] == name)


def center(c):
    return [float(c['xy_cm'][0]) / 100, float(c['xy_cm'][1]) / 100, .741]


def rotated_grid(grid, yaw_deg):
    if not yaw_deg:
        return grid
    # Inverse world yaw, expressed in UV coordinates whose vertical axis is -Y.
    a = np.deg2rad(yaw_deg)
    u, v = grid[..., 0], grid[..., 1]
    return np.stack([np.cos(a)*u - np.sin(a)*v,
                     np.sin(a)*u + np.cos(a)*v], -1).astype(np.float32)


def geometry_at(task, assets, c, rgb):
    import sapien
    from render_geometry import Patch
    patch = Patch(task, assets, center=center(c), size=.20)
    try:
        patch.texture(rgb)
        a = np.deg2rad(c['yaw_deg']) / 2
        patch.actor.set_pose(sapien.Pose(center(c), [np.cos(a), 0, 0, np.sin(a)]))
        task.get_obs()  # Refresh segmentation at the new pose without physics.
        return patch.geometry(task)
    finally:
        task.scene.remove_actor(patch.actor)


def gain_images(images, gain):
    if gain == 1:
        return images
    return {k: np.rint(v.astype(np.float32)*gain).clip(0, 255).astype(np.uint8)
            for k, v in images.items()}


def clean_images(obs, gain=1.):
    from openpi_client.image_tools import resize_with_pad
    from render_geometry import CAMERAS, WIRE
    images = {w: resize_with_pad(obs['observation'][n]['rgb'], 224, 224).copy()
              for n, w in zip(CAMERAS, WIRE)}
    return gain_images(images, gain)


def composite(obs, geometry, rgb, c, return_masks=False):
    import torch
    import torch.nn.functional as F
    from render_geometry import CAMERAS, WIRE
    from render_proxy import grid_for
    q = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float()/255
    backgrounds = clean_images(obs)
    images, masks = {}, {}
    for name, wire in zip(CAMERAS, WIRE):
        geo = geometry[name]
        g = grid_for(geo['intrinsic'], geo['extrinsic'], center(c), .20, 224, 224, .5)
        g = rotated_grid(g, c['yaw_deg'])
        t = F.grid_sample(q, torch.from_numpy(g)[None], mode='bilinear',
                          padding_mode='border', align_corners=False)
        m = F.pad(F.interpolate(torch.from_numpy(geo['mask'])[None, None].float(),
                                size=(168, 224), mode='nearest'), (0, 0, 28, 28))
        b = torch.from_numpy(backgrounds[wire]).permute(2, 0, 1)[None].float()/255
        images[wire] = ((b*(1-m)+t*m)[0].permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        masks[wire] = m[0, 0].numpy().astype(bool)
    images = gain_images(images, c['gain'])
    return (images, masks) if return_masks else images


def prepare(dest):
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    protocol = dict(task='place_container_plate', task_config='demo_clean', patch_m=.20,
        origin=[0, 0, .741], texture_parameterization='frozen original digital Full Q128, step500',
        methods=METHODS, conditions=CONDITIONS, train_seeds=list(range(10100, 10108)),
        validation_seeds=[10110, 10111], test_seeds=list(range(10200, 10210)),
        fresh_probe_noise_seeds=[20261021, 20261022, 20261023],
        rollout_noise_rule='20261006 + seed * 1000 + query', replan=32,
        texture_hashes={k:sha(v) for k,v in texture_paths().items()},
        transforms_fixed_within_episode=True, shared_world_patch_all_three_views=True,
        patch_RGB='digital perspective bilinear sampling at model224; no rendered patch RGB',
        geometry='world XY translation and world Z-axis yaw; fresh segmentation at every query',
        brightness='multiply the entire composited sRGB uint8 model input, clip and round; clean matched',
        nominal_reuse='original complete digital full/random and clean rollouts; compositor parity checked',
        endpoint='closed-loop ASR among clean-success episodes under the same brightness',
        interpretation='small paired sensitivity study on existing held-out seeds; not a new blind benchmark',
        new_rollouts=180, training=False)
    f = dest/'protocol.json'
    if f.exists():
        assert json.loads(f.read_text()) == protocol, 'Frozen protocol differs'
    else:
        dump(f, protocol)
    return protocol
