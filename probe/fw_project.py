#!/usr/bin/env python3
"""FastWAM 7×7 token 网格 → 世界坐标反投影(与 g0i_project 同法,NS=7、32px/格)。
相机/桌面与 π0.5 逐字相同(同一 LIBERO env),只是模型输入 7×7、朝向另验。
用 fw_anchor_cells.npz(78 锚点渲染出的真值落格)暴力搜 8 种二面体朝向定朝向,
再把 7×7 每格中心反投影到 z=0.9 → 每格 world + 合法性(scene_patch 同判据,10cm patch)。

用法: /home/user1/miniconda3/envs/openpi-libero/bin/python probe/fw_project.py
"""
import os, pathlib, sys
ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
PI05 = ROOT / "pi05probe"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(PI05 / "libero_config"))
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(PI05))
sys.path.insert(0, str(ROOT / "third_party" / "openpi" / "third_party" / "libero"))
import numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import transform_utils as TU

RES, NIN, NS = 256, 224, 7
CELL = NIN // NS                 # 32 px/格
Z = 0.900
OUT = ROOT / "probe" / "out"


def cam_transform(sim, cam, H, W):
    cid = sim.model.camera_name2id(cam)
    fovy = sim.model.cam_fovy[cid]
    f = 0.5 * H / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    Kx = np.eye(4); Kx[:3, :3] = K
    R = TU.make_pose(sim.data.cam_xpos[cid], sim.data.cam_xmat[cid].reshape(3, 3))
    R = R @ np.array([[1., 0, 0, 0], [0, -1., 0, 0], [0, 0, -1., 0], [0, 0, 0, 1.]])
    return Kx @ TU.pose_inv(R)


def T_matrix():
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    stem = "put_the_bowl_on_the_plate"
    tid = next(i for i in range(suite.n_tasks)
               if pathlib.Path(suite.get_task(i).bddl_file).stem == stem)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    env.seed(10000); env.reset()
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) / "libero_goal" / f"{stem}.pruned_init")))
    env.set_init_state(inits[0])
    T = cam_transform(env.env.sim, "agentview", RES, RES)
    env.close()
    return T


def world_to_px256(T, xy):
    ph = T @ np.array([xy[0], xy[1], Z, 1.0])
    return ph[1] / ph[2], ph[0] / ph[2]      # (row,col) raw 256


def orient(r, c, code):
    fr, fc, sw = code
    if fr: r = (RES - 1) - r
    if fc: c = (RES - 1) - c
    if sw: r, c = c, r
    return r, c


def world_to_cell(T, xy, code):
    r, c = world_to_px256(T, xy); r, c = orient(r, c, code)
    r224, c224 = r * NIN / RES, c * NIN / RES
    return int(np.clip(r224 // CELL, 0, NS - 1)), int(np.clip(c224 // CELL, 0, NS - 1))


def main():
    T = T_matrix()
    fa = np.load(OUT / "fw_anchor_cells.npz", allow_pickle=True)
    aw = fa["anchor_world"][:, :2]; truth = fa["cell"]
    vis = truth[:, 0] >= 0
    aw, truth = aw[vis], truth[vis]
    best = None
    for fr in (0, 1):
        for fc in (0, 1):
            for sw in (0, 1):
                code = (fr, fc, sw)
                hit = sum(world_to_cell(T, aw[i], code) == tuple(truth[i]) for i in range(len(aw)))
                near = sum(abs(world_to_cell(T, aw[i], code)[0] - truth[i][0]) +
                           abs(world_to_cell(T, aw[i], code)[1] - truth[i][1]) <= 1 for i in range(len(aw)))
                print(f"[朝向] code={code}: 精确 {hit:2d}/{len(aw)}  ±1格 {near:2d}/{len(aw)}")
                if best is None or hit > best[0]:
                    best = (hit, near, code)
    CODE = best[2]
    print(f"\n[best] code={CODE} 精确 {best[0]}/{len(aw)}  ±1 {best[1]}/{len(aw)}")

    def cell_to_world(rc):
        r224, c224 = (rc[0] + 0.5) * CELL, (rc[1] + 0.5) * CELL
        r_or, c_or = r224 * RES / NIN, c224 * RES / NIN
        fr, fc, sw = CODE
        r, c = (c_or, r_or) if sw else (r_or, c_or)
        if fc: c = (RES - 1) - c
        if fr: r = (RES - 1) - r
        px, py = c, r
        a, b, cc = T[0], T[1], T[2]
        A = np.array([[a[0] - px * cc[0], a[1] - px * cc[1]], [b[0] - py * cc[0], b[1] - py * cc[1]]])
        rhs = -np.array([(a[2] - px * cc[2]) * Z + (a[3] - px * cc[3]),
                         (b[2] - py * cc[2]) * Z + (b[3] - py * cc[3])])
        return np.linalg.solve(A, rhs)

    err = [np.linalg.norm(cell_to_world(world_to_cell(T, aw[i], CODE)) - aw[i]) for i in range(len(aw))]
    print(f"[闭环] world→格→world 误差 中位 {np.median(err)*100:.1f}cm 最大 {max(err)*100:.1f}cm")

    import yaml, scene_patch as sp
    cfg = yaml.safe_load((PI05 / "config" / "scene.yaml").read_text())
    plane = sp.Plane.from_cfg(cfg["plane"]); w, h = cfg["patch"]["size_wh"]
    keep = cfg.get("keepout") or {}
    def legal(xy):
        if not plane.contains_uv(xy[0], xy[1], w, h):
            return False
        x0, x1, y0, y1 = xy[0]-w/2, xy[0]+w/2, xy[1]-h/2, xy[1]+h/2
        return not any(x1 >= b["x"][0] and x0 <= b["x"][1] and y1 >= b["y"][0] and y0 <= b["y"][1]
                       for b in keep.values())
    cells, worlds = [], []
    for r in range(NS):
        for c in range(NS):
            xy = cell_to_world((r, c))
            if plane.bounds_u[0] <= xy[0] <= plane.bounds_u[1] and plane.bounds_v[0] <= xy[1] <= plane.bounds_v[1]:
                cells.append((r, c)); worlds.append(xy)
    worlds = np.array(worlds); leg = np.array([legal(xy) for xy in worlds])
    d01 = np.linalg.norm(cell_to_world((3, 3)) - cell_to_world((3, 4)))
    print(f"[网格] 桌面内 {len(cells)}/49 格  合法(10cm){int(leg.sum())}  相邻格间距 ≈{d01*100:.1f}cm")
    np.savez_compressed(OUT / "fw_cell_world.npz", T=T, code=np.array(CODE),
                        cells=np.array(cells), worlds=worlds, legal=leg, cell_spacing=d01)
    print(f"[written] {OUT/'fw_cell_world.npz'}")


if __name__ == "__main__":
    main()
