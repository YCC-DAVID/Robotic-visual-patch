#!/usr/bin/env python3
"""DRIFT 复现 · 纹理反 warp 管线的朝向/UV 红线检验。

为什么必须查
----------
评测 B 的缺口诊断量到:渲染出来的足迹像素与设计图案 P 的**相关系数 +0.002**(近乎无关),
而总变差 TV 还保留 24.1%。低通滤波会让图案变模糊、TV 下降,但**不会让它与原图零相关**。
⇒ 更像是**纹理朝向 / UV 约定不匹配**(翻转、转置、偏移):内容还在但对不上位。

这不是小事:本项目 §2.3 第 3 条"优化图案在闭环零效果"的老负结果用的是同一套反 warp 管线
(`patch_texture_from_train.py` 同法)。若管线有朝向 bug,那个负结论可能部分是假象。

方法:用**低频、强不对称**的测试纹理(4 象限纯色 + 一个角落标记)。
低频 ⇒ 不怕透视重采样和降采样;强不对称 ⇒ 任何翻转/转置/旋转都会立刻暴露。
渲染后按最近颜色把每个足迹像素判到某个象限,再与"用 drift_texture.py 的同一个单应
把图像像素映回纹理空间"所预测的象限比对,报一致率。

同时穷举 8 种二面体变换(恒等/水平翻/垂直翻/180°/转置/…),看哪一种一致率最高
⇒ 直接读出正确的 UV 约定。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_warp_check.py --patch-m 0.26
"""
import argparse
import os
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"

for p in reversed([OPENPI / "packages" / "openpi-client" / "src", OPENPI / "third_party" / "libero"]):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(PROBE))
os.environ["LIBERO_CONFIG_PATH"] = str(PROBE / "libero_config")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYTHONNOUSERSITE"] = "1"

import yaml  # noqa: E402

RES, RESIZE, NWAIT, TEX = 256, 224, 10, 128
DUMMY = [0.0] * 6 + [-1.0]
TASK = "put_the_bowl_on_the_plate"
Z_TABLE = 0.9

# 4 象限纯色 + 左上角白块标记(打破 180° 对称)
QUAD_COLORS = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]], np.uint8)

# 8 种二面体变换(作用在纹理上),名字 → 函数
DIHEDRAL = {
    "identity": lambda a: a,
    "flipud": lambda a: a[::-1],
    "fliplr": lambda a: a[:, ::-1],
    "rot180": lambda a: a[::-1, ::-1],
    "transpose": lambda a: a.transpose(1, 0, 2) if a.ndim == 3 else a.T,
    "rot90": lambda a: np.rot90(a, 1),
    "rot270": lambda a: np.rot90(a, 3),
    "anti_transpose": lambda a: np.rot90(a, 2).transpose(1, 0, 2) if a.ndim == 3
    else np.rot90(a, 2).T,
}


def model_input(obs):
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))


def project_quad(cw, cell_xy, patch_m):
    """与 drift_pool_prep.project_quad / drift_texture.py 完全同法。"""
    T, code = cw["T"], tuple(int(v) for v in cw["code"])
    cx, cy = cell_xy
    hh = patch_m / 2.0
    q = []
    for x, y in [(cx - hh, cy - hh), (cx + hh, cy - hh), (cx + hh, cy + hh), (cx - hh, cy + hh)]:
        ph = T @ np.array([x, y, Z_TABLE, 1.0])
        r, c = ph[1] / ph[2], ph[0] / ph[2]
        fr, fc, sw = code
        if fr:
            r = (RES - 1) - r
        if fc:
            c = (RES - 1) - c
        if sw:
            r, c = c, r
        q.append([r * RESIZE / RES, c * RESIZE / RES])
    return np.array(q, np.float32)


def make_test_texture():
    t = np.zeros((TEX, TEX, 3), np.uint8)
    h = TEX // 2
    t[:h, :h] = QUAD_COLORS[0]        # 左上 红
    t[:h, h:] = QUAD_COLORS[1]        # 右上 绿
    t[h:, :h] = QUAD_COLORS[2]        # 左下 蓝
    t[h:, h:] = QUAD_COLORS[3]        # 右下 黄
    t[:TEX // 8, :TEX // 8] = 255     # 左上角白块:打破 180° 对称
    return t


def quad_id(uv):
    """纹理坐标 (row,col) ∈[0,TEX) → 象限号 0..3(与 make_test_texture 一致)。"""
    h = TEX / 2.0
    return (uv[:, 0] >= h).astype(int) * 2 + (uv[:, 1] >= h).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-m", type=float, default=0.26)
    ap.add_argument("--cell", default=None, help="世界 x,y;默认取 influence 合法 argmax")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--thresh", type=int, default=12)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    import torch

    cw = np.load(OUT / "grad" / "pi05_cell_world.npz", allow_pickle=True)
    if args.cell:
        cell_xy = tuple(float(v) for v in args.cell.split(","))
    else:
        d = np.load(OUT / "percell_scores_plate.npz", allow_pickle=True)
        infl, legal, aw = d["influence"], d["legal"].astype(bool), d["anchor_world"]
        legi = np.where(legal)[0]
        am = int(legi[int(np.argmax(infl[legi]))])
        cell_xy = (float(aw[am][0]), float(aw[am][1]))
    quad = project_quad(cw, cell_xy, args.patch_m)
    print(f"[cfg] cell=({cell_xy[0]:+.4f},{cell_xy[1]:+.4f})  patch_m={args.patch_m}  "
          f"quad(row,col)=\n{quad}", flush=True)

    tex = make_test_texture()
    texp = PROBE / "config" / "drift_warp_test.png"
    cv2.imwrite(str(texp), tex[:, :, ::-1])
    print(f"[tex] 测试纹理 {texp}(4 象限 红/绿/蓝/黄 + 左上白块)", flush=True)

    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    cfg["patch"]["size_wh"] = [args.patch_m, args.patch_m]
    plane = sp.Plane.from_cfg(cfg["plane"])
    lift = cfg["patch"]["thickness"] / 2.0 + cfg["patch"]["normal_offset"]
    world = plane.to_world(cell_xy[0], cell_xy[1], lift)

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid = next(i for i in range(suite.n_tasks)
               if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                    "libero_goal" / f"{TASK}.pruned_init")))

    cenv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    cenv.seed(args.seed); cenv.reset()
    cenv.set_init_state(inits[0])
    for _ in range(NWAIT):
        cenv.step(DUMMY)
    st0 = cenv.get_sim_state().copy()
    clean = model_input(cenv.regenerate_obs_from_state(st0))
    cenv.close()

    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    penv.env.set_xml_processor(sp.make_xml_processor(cfg, world, str(texp)))
    penv.seed(args.seed); penv.reset()
    got = model_input(penv.regenerate_obs_from_state(st0))
    penv.close()

    m = np.abs(got.astype(np.int16) - clean.astype(np.int16)).max(-1) > args.thresh
    print(f"[render] 足迹 {int(m.sum())}px", flush=True)
    cv2.imwrite(str(OUT / "drift_warp_check_render.png"), got[:, :, ::-1])
    assert m.sum() > 100, "足迹太小,纹理注入可能没生效"

    ys, xs = np.where(m)
    px = got[ys, xs].astype(np.float32)
    # 观测象限:按最近的象限颜色判(排除白块像素,它们对 4 色都不近)
    dist = np.stack([np.linalg.norm(px - QUAD_COLORS[k].astype(np.float32), axis=1)
                     for k in range(4)])
    obs_q = np.argmin(dist, 0)
    conf = dist.min(0) < 140.0                # 只用颜色够纯的像素判定
    print(f"[render] 颜色可判定像素 {int(conf.sum())}/{int(m.sum())}", flush=True)

    # 预测象限:图像像素 → 纹理坐标(drift_texture.py 用的单应的逆)
    src = quad[:, ::-1].astype(np.float32)                       # (x=col,y=row)
    dst = np.array([[0, 0], [TEX - 1, 0], [TEX - 1, TEX - 1], [0, TEX - 1]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)                    # 图像 → 纹理
    pts = np.stack([xs, ys], 1).astype(np.float32)[None]
    uv_xy = cv2.perspectiveTransform(pts, H)[0]                  # (u=x, v=y) in texture
    uv = np.stack([uv_xy[:, 1], uv_xy[:, 0]], 1)                 # → (row,col)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < TEX) & (uv[:, 1] >= 0) & (uv[:, 1] < TEX)
    use = conf & inside
    print(f"[match] 参与比对的像素 {int(use.sum())}\n", flush=True)

    print(f"{'变换':>16s} {'象限一致率':>10s}")
    best = None
    for name, fn in DIHEDRAL.items():
        tq = fn(tex)                                  # 变换后的纹理
        # 变换后纹理里,预测 uv 处的颜色应当是什么象限
        uu = np.clip(uv[use].astype(int), 0, TEX - 1)
        pred_col = tq[uu[:, 0], uu[:, 1]].astype(np.float32)
        dd = np.stack([np.linalg.norm(pred_col - QUAD_COLORS[k].astype(np.float32), axis=1)
                       for k in range(4)])
        pred_q = np.argmin(dd, 0)
        acc = float((pred_q == obs_q[use]).mean())
        flag = ""
        if best is None or acc > best[1]:
            best = (name, acc)
        print(f"{name:>16s} {acc:9.1%}{flag}")
    print(f"\n[结论] 最佳变换 = **{best[0]}**  一致率 {best[1]:.1%}")
    if best[0] == "identity" and best[1] > 0.85:
        print("  ✅ 反 warp 管线朝向正确 ⇒ 评测 B 的低相关是真实的物理链损耗(低通+光照),不是 bug。")
    elif best[1] > 0.85:
        print(f"  ❌ **管线有朝向 bug**:需要在 drift_texture.py 里对纹理施加 `{best[0]}`。")
        print("     ⚠️ 同时意味着 §2.3 第 3 条的老负结果(优化图案闭环零效果)可能部分是这个 bug 的假象,")
        print("        因为 patch_texture_from_train.py 用的是同一套反 warp。")
    else:
        print("  ⚠️ 所有变换一致率都不高 ⇒ 可能不是简单二面体变换(UV 缩放/偏移,或纹理被平铺)。")
        print("     下一步:看 out/drift_warp_check_render.png 肉眼判断象限落位。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
