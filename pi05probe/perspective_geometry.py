"""Planar projection from actual MuJoCo camera and box geometry.

The model image reverses both axes of the bottom-up OpenGL readback. Pixel
centers therefore have principal point (size-1)/2, with both camera X and Y
negated. UV orientation is an explicitly recorded discrete corner convention.
"""
import numpy as np


def project_world(sim, points, camera='agentview', size=256):
    cid = sim.model.camera_name2id(camera)
    position = np.asarray(sim.data.cam_xpos[cid]).copy()
    rotation = np.asarray(sim.data.cam_xmat[cid]).reshape(3, 3).copy()
    camera_xyz = (np.asarray(points) - position) @ rotation
    depth = -camera_xyz[:, 2]
    assert (depth > 0).all()
    f = .5 * size / np.tan(np.deg2rad(float(sim.model.cam_fovy[cid])) / 2)
    image = (size - 1) / 2 - f * camera_xyz[:, :2] / depth[:, None]
    return image, depth


def box_top_projection(sim, geom_name, size=256, permutation=(0, 1, 2, 3)):
    import cv2
    gid = sim.model.geom_name2id(geom_name)
    half = np.asarray(sim.model.geom_size[gid]).copy()
    local = np.array([[-half[0], -half[1], half[2]],
                      [half[0], -half[1], half[2]],
                      [half[0], half[1], half[2]],
                      [-half[0], half[1], half[2]]])
    rotation = np.asarray(sim.data.geom_xmat[gid]).reshape(3, 3)
    world = local @ rotation.T + np.asarray(sim.data.geom_xpos[gid])
    image, depth = project_world(sim, world, size=size)
    q_edges = np.array([[-.5, -.5], [127.5, -.5], [127.5, 127.5], [-.5, 127.5]], np.float32)
    forward = cv2.getPerspectiveTransform(q_edges, image[list(permutation)].astype(np.float32))
    return forward, {'world_corners': world.tolist(), 'image_corners_xy': image.tolist(),
                     'camera_depth_m': depth.tolist(), 'uv_to_world_corner_permutation': list(permutation),
                     'image_size': size, 'camera': 'agentview'}


def transform_xy(H, xy):
    p = np.c_[np.asarray(xy), np.ones(len(xy))] @ H.T
    return p[:, :2] / p[:, 2:]


def identify_uv_orientation(sim, geom_name, reference_image_to_uv, image_points):
    """Choose among eight UV axis/flip conventions; no continuous fit."""
    candidates = []
    target = transform_xy(reference_image_to_uv, image_points)
    for flip in [False, True]:
        base = np.arange(4)[::-1] if flip else np.arange(4)
        for shift in range(4):
            order = np.roll(base, shift)
            H, geometry = box_top_projection(sim, geom_name, permutation=order)
            error = transform_xy(np.linalg.inv(H), image_points) - target
            candidates.append({'permutation': order.tolist(), 'uv_RMSE_texels': float(np.sqrt((error**2).mean()))})
    chosen = min(candidates, key=lambda row: row['uv_RMSE_texels'])
    return chosen, candidates
