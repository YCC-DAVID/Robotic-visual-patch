#!/usr/bin/env python3
"""DRIFT 复现 · 位置对照格选取(对应 HANDOFF 4.17 的设计)。

为什么需要这个
------------
0.26 m 打穿了,但那个尺寸盖住 ~19 个 token 格 ⇒ 位置不是可分离变量,
**不能反推 influence 选点有效**。0.13 m 实测 ASR 54%(未饱和、有动态范围),
且 ≈ 格间距 0.09m 的 1.4 倍 ⇒ 隔开 2 格以上基本不重叠,**位置可分辨**。

所以在 0.13 m 上做位置对照:
  实验格 = influence 合法 argmax(已在跑)
  对照格 = **距离远 + influence 低 + 足迹面积匹配 ±12%** 的合法格
足迹必须匹配,否则又会变成"面积差异"而不是"位置差异"
(这正是 FastWAM 踩过的坑,见 HANDOFF 2.1;OpenVLA 那边用 ovla_footprint_ctrl.py 同法处理)。

足迹面积用与 drift_pool_prep.project_quad 完全相同的投影算(共用已校准的相机 T 与朝向 code),
不需要跑渲染:透视下四边形面积可直接由四个角点用鞋带公式算。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_pick_ctrl.py --patch-m 0.13
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"

RES, RESIZE = 256, 224
Z_TABLE = 0.9


def project_quad(cw, cell_xy, patch_m):
    """与 drift_pool_prep.project_quad 逐字同法:patch 四个世界角 → 224 图像 (row,col)。"""
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
    return np.array(q, np.float64)


def quad_area(q):
    """鞋带公式:凸四边形在图像平面的面积(px²)。"""
    x, y = q[:, 1], q[:, 0]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-m", type=float, default=0.13)
    ap.add_argument("--tol", type=float, default=0.12, help="足迹面积匹配容差(±比例)")
    ap.add_argument("--min-dist", type=float, default=0.18,
                    help="与实验格的最小世界距离(m);0.13m 贴纸隔 0.18m 即不重叠")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    cw = np.load(OUT / "grad" / "pi05_cell_world.npz", allow_pickle=True)
    d = np.load(OUT / "percell_scores_plate.npz", allow_pickle=True)
    infl, legal, aw = d["influence"], d["legal"].astype(bool), d["anchor_world"]
    legi = np.where(legal)[0]
    am = int(legi[int(np.argmax(infl[legi]))])
    tgt = (float(aw[am][0]), float(aw[am][1]))
    A_tgt = quad_area(project_quad(cw, tgt, args.patch_m))
    imax = float(infl[legi].max())
    print(f"[实验格] idx {am}  world=({tgt[0]:+.4f},{tgt[1]:+.4f})  "
          f"influence={float(infl[am]):.4f}(合法最大)  足迹面积={A_tgt:.0f}px²", flush=True)
    print(f"[筛选] 面积匹配 ±{args.tol:.0%}  世界距离 ≥{args.min_dist:.2f}m  "
          f"共 {len(legi)} 个合法格\n", flush=True)

    rows = []
    for i in legi:
        i = int(i)
        if i == am:
            continue
        xy = (float(aw[i][0]), float(aw[i][1]))
        dist = float(np.hypot(xy[0] - tgt[0], xy[1] - tgt[1]))
        A = quad_area(project_quad(cw, xy, args.patch_m))
        ratio = A / A_tgt
        if dist < args.min_dist or abs(ratio - 1.0) > args.tol:
            continue
        rows.append((i, xy, dist, A, ratio, float(infl[i]), float(infl[i]) / imax))
    rows.sort(key=lambda r: r[5])          # influence 升序 ⇒ 最"无关"的位置排最前
    if not rows:
        print("[空] 没有满足条件的格 —— 放宽 --tol 或 --min-dist", flush=True)
        return 1
    print(f"{'idx':>4s} {'world (x,y)':>18s} {'距离m':>7s} {'足迹px²':>9s} {'面积比':>7s} "
          f"{'influence':>10s} {'占最大':>7s}")
    for i, xy, dist, A, ratio, inf, rel in rows[:args.top]:
        print(f"{i:4d} ({xy[0]:+.4f},{xy[1]:+.4f}) {dist:7.3f} {A:9.0f} {ratio:6.2f}× "
              f"{inf:10.4f} {rel:6.1%}")
    best = rows[0]
    print(f"\n[建议对照格] idx {best[0]}  world=({best[1][0]:+.4f},{best[1][1]:+.4f})")
    print(f"  influence 只有最优的 {best[6]:.1%};足迹 {best[4]:.2f}× 匹配;距离 {best[2]:.3f}m 不重叠")
    print(f"\n  跑法(与实验格完全同一套流程):")
    print(f"    drift_pool_prep.py --patch-m {args.patch_m} --episodes 40 "
          f"--cell {best[1][0]:.4f},{best[1][1]:.4f} --out drift_pool_ctrl")
    print(f"    drift_patch_train.py --pool drift_pool_ctrl --iters 500 --tag ctrl")
    print(f"    drift_eval_digital.py --patch drift_patch_ctrl --episodes 50 --out drift_evalA_ctrl")
    print(f"\n  判读:对照格守住(≈100%)而实验格 46% ⇒ influence 选点在 DRIFT loss 下成立;")
    print(f"        两格都掉到 ~46% ⇒ 位置不重要,loss 自己就够(会推翻 2.3 第 1 条在 flow matching 上的适用性)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
