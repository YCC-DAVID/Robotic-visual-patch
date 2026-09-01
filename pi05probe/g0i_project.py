#!/usr/bin/env python3
"""相机投影/反投影 + 验证:π0.5 侧把 token 格子中心反投影到桌面世界坐标。
先用 61 个已知锚点正投影核对(world→模型输入 224 图的 16×16 格),和 cov 主导格比对。
通过后,把 16×16 每个格子中心反投影到 z=0.9 → 每格世界坐标(供 per-patch 扫描)。

模型输入朝向:render 256 → agentview[::-1,::-1] → resize_with_pad 256→224(方形,纯缩放)。
用法: /home/user1/miniconda3/envs/openpi-libero/bin/python pi05probe/g0i_project.py
"""
import os, pathlib, sys
ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT/"pi05probe"/"libero_config"))
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(ROOT/"pi05probe"))
sys.path.insert(0, str(ROOT/"third_party"/"openpi"/"third_party"/"libero"))
import numpy as np, torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils import transform_utils as TU


def cam_transform(sim, cam, H, W):
    """内联 robosuite.camera_utils.get_camera_transform_matrix(避开 h5py 依赖)。"""
    cid = sim.model.camera_name2id(cam)
    fovy = sim.model.cam_fovy[cid]
    f = 0.5 * H / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    Kx = np.eye(4); Kx[:3, :3] = K
    R = TU.make_pose(sim.data.cam_xpos[cid], sim.data.cam_xmat[cid].reshape(3, 3))
    R = R @ np.array([[1., 0, 0, 0], [0, -1., 0, 0], [0, 0, -1., 0], [0, 0, 0, 1.]])
    return Kx @ TU.pose_inv(R)

RES, NIN, NS = 256, 224, 16          # render 256, 模型输入 224, 16×16 格
CELL = NIN // NS                      # 14 px/格
Z = 0.900
OUT = ROOT/"pi05probe"/"out"


def T_matrix():
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    stem = "put_the_bowl_on_the_plate"
    tid = next(i for i in range(suite.n_tasks)
               if pathlib.Path(suite.get_task(i).bddl_file).stem == stem)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    env.seed(10000); env.reset()
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states"))/"libero_goal"/f"{stem}.pruned_init")))
    env.set_init_state(inits[0])
    T = cam_transform(env.env.sim, "agentview", RES, RES)  # world→pixel(256)
    env.close()
    return T


def world_to_px256(T, xy):
    ph = T @ np.array([xy[0], xy[1], Z, 1.0])
    return ph[1]/ph[2], ph[0]/ph[2]                  # (row, col) in raw render 256


def orient(r, c, code):
    """8 种二面体朝向:code=(fr,fc,swap)。像素 256 空间。"""
    fr, fc, sw = code
    if fr: r = (RES-1) - r
    if fc: c = (RES-1) - c
    if sw: r, c = c, r
    return r, c


def world_to_cell(T, xy, code):
    r, c = world_to_px256(T, xy)
    r, c = orient(r, c, code)
    r224, c224 = r*NIN/RES, c*NIN/RES
    return int(np.clip(r224//CELL, 0, NS-1)), int(np.clip(c224//CELL, 0, NS-1))


CODE = (0, 1, 0)                                      # 验证得:只翻列


def cell_to_world(T, rc):
    """模型输入格 (row,col) 中心 → 反投影到 z=Z 平面的 world(x,y)(朝向 CODE=(0,1,0))。"""
    r224, c224 = (rc[0]+0.5)*CELL, (rc[1]+0.5)*CELL
    r_or, c_or = r224*RES/NIN, c224*RES/NIN           # oriented 256
    r256, c256 = r_or, (RES-1) - c_or                 # 撤销 code=(0,1,0):列翻回
    px, py = c256, r256
    a, b, c = T[0], T[1], T[2]
    # (a-px*c)·[x,y,Z,1]=0 ; (b-py*c)·[x,y,Z,1]=0
    A = np.array([[a[0]-px*c[0], a[1]-px*c[1]], [b[0]-py*c[0], b[1]-py*c[1]]])
    rhs = -np.array([(a[2]-px*c[2])*Z + (a[3]-px*c[3]), (b[2]-py*c[2])*Z + (b[3]-py*c[3])])
    xy = np.linalg.solve(A, rhs)
    return xy


def main():
    T = T_matrix()
    za = np.load(OUT/"grad"/"g0_attn_task.npz", allow_pickle=True)
    aw = za["plate__anchor_world"][:, :2]
    cov = za["plate__cov"]
    dom = cov.argmax(1)
    dom_rc = [(int(d//NS), int(d%NS)) for d in dom]
    best = None
    for fr in (0, 1):
        for fc in (0, 1):
            for sw in (0, 1):
                code = (fr, fc, sw)
                hit = sum(world_to_cell(T, aw[i], code) == dom_rc[i] for i in range(len(aw)))
                near = sum(abs(world_to_cell(T, aw[i], code)[0]-dom_rc[i][0]) +
                           abs(world_to_cell(T, aw[i], code)[1]-dom_rc[i][1]) <= 1 for i in range(len(aw)))
                print(f"[朝向] code={code}: 精确 {hit:2d}/{len(aw)}  ±1格 {near:2d}/{len(aw)}")
                if best is None or hit > best[0]:
                    best = (hit, code)
    print(f"\n[best] code={best[1]} 精确命中 {best[0]}/{len(aw)}")

    # 闭环:world→格→world
    err = [np.linalg.norm(cell_to_world(T, world_to_cell(T, aw[i], CODE)) - aw[i]) for i in range(len(aw))]
    print(f"[闭环] world→格→world 误差 中位 {np.median(err)*100:.1f}cm 最大 {max(err)*100:.1f}cm")

    # 生成 16×16 每格中心的 world,判合法(scene_patch 同一判据)
    import yaml, scene_patch as sp
    cfg = yaml.safe_load((ROOT/"pi05probe"/"config"/"scene.yaml").read_text())
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
            xy = cell_to_world(T, (r, c))
            if plane.bounds_u[0] <= xy[0] <= plane.bounds_u[1] and plane.bounds_v[0] <= xy[1] <= plane.bounds_v[1]:
                cells.append((r, c)); worlds.append(xy)
    worlds = np.array(worlds)
    leg = np.array([legal(xy) for xy in worlds])
    # 相邻格世界间距(定 patch 尺寸参考)
    d01 = np.linalg.norm(cell_to_world(T, (8, 8)) - cell_to_world(T, (8, 9)))
    print(f"[网格] 落在桌面内的格 {len(cells)}/256  其中合法(10cm patch 判据){int(leg.sum())}")
    print(f"[网格] 相邻格世界间距 ≈ {d01*100:.1f} cm(⇒ per-cell patch 尺寸参考)")
    np.savez_compressed(OUT/"grad"/"pi05_cell_world.npz",
                        T=T, code=np.array(CODE), cells=np.array(cells),
                        worlds=worlds, legal=leg, cell_spacing=d01)
    print(f"[written] {OUT/'grad'/'pi05_cell_world.npz'}")


if __name__ == "__main__":
    main()
