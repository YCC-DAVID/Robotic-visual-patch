#!/usr/bin/env python3
"""DRIFT 复现 · 逐格**渲染实测**足迹扫描(修正 drift_pick_ctrl.py 的几何面积口径)。

为什么必须改用渲染实测
------------------
`drift_pick_ctrl.py` 用鞋带公式算投影四边形的**几何面积**,结果实验格与对照格都是 1121 px²、
看起来完美匹配。但实际渲染出来是 **1182 vs 848 px,差 28%** ——
因为对照格 (+0.0453,+0.3556) 投影到 column≈7,**贴纸被 224 图像的左边界裁掉了一半**。
几何面积不知道图像边界,也不知道被 fixture 静态遮挡。

⚠️ 这正是本项目在 FastWAM 上踩过的面积混淆坑(见 HANDOFF 2.1):
足迹没对齐就做位置对比,量到的是面积差异而不是位置差异。

所以本脚本对每个合法格**真渲一次**,量无遮挡足迹像素数(用同一参考 sim state,
机械臂在初始位姿),再按"与实验格足迹匹配 + 距离远 + influence 低"挑对照格。
77 个合法格约 2 分钟。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_footprint_scan.py --patch-m 0.13
"""
import argparse
import math
import os
import pathlib
import sys

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

import numpy as np  # noqa: E402
import yaml  # noqa: E402

RES, RESIZE, NWAIT = 256, 224, 10
DUMMY = [0.0] * 6 + [-1.0]
TASK = "put_the_bowl_on_the_plate"


def model_input(obs):
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-m", type=float, default=0.13)
    ap.add_argument("--tol", type=float, default=0.12, help="足迹**实测**匹配容差(±比例)")
    ap.add_argument("--min-dist", type=float, default=0.18)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--thresh", type=int, default=12)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="drift_footprint_scan")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    import torch

    cfg0 = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    plane = sp.Plane.from_cfg(cfg0["plane"])
    lift = cfg0["patch"]["thickness"] / 2.0 + cfg0["patch"]["normal_offset"]
    tex_probe = str(PROBE / "config" / "probe_texture.png")

    d = np.load(OUT / "percell_scores_plate.npz", allow_pickle=True)
    infl, legal, aw = d["influence"], d["legal"].astype(bool), d["anchor_world"]
    legi = np.where(legal)[0]
    am = int(legi[int(np.argmax(infl[legi]))])
    tgt = (float(aw[am][0]), float(aw[am][1]))
    imax = float(infl[legi].max())

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid = next(i for i in range(suite.n_tasks)
               if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                    "libero_goal" / f"{TASK}.pruned_init")))

    # 参考 sim state:与 pool 同一 init + 同样 10 步 warm-up
    cenv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    cenv.seed(args.seed); cenv.reset()
    cenv.set_init_state(inits[0])
    for _ in range(NWAIT):
        cenv.step(DUMMY)
    st0 = cenv.get_sim_state().copy()
    clean = model_input(cenv.regenerate_obs_from_state(st0))
    cenv.close()

    def footprint(cell_xy):
        c = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
        c["patch"]["size_wh"] = [args.patch_m, args.patch_m]
        e = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        e.env.set_xml_processor(sp.make_xml_processor(
            c, plane.to_world(cell_xy[0], cell_xy[1], lift), tex_probe))
        e.seed(args.seed); e.reset()
        img = model_input(e.regenerate_obs_from_state(st0))
        e.close()
        m = np.abs(img.astype(np.int16) - clean.astype(np.int16)).max(-1) > args.thresh
        ys, xs = np.where(m)
        cen = (float(ys.mean()), float(xs.mean())) if m.any() else (float("nan"),) * 2
        return int(m.sum()), cen

    A_tgt, cen_tgt = footprint(tgt)
    print(f"[实验格] idx {am}  world=({tgt[0]:+.4f},{tgt[1]:+.4f})  influence={float(infl[am]):.1f}"
          f"(合法最大)  **实测足迹={A_tgt}px**  质心=({cen_tgt[0]:.0f},{cen_tgt[1]:.0f})", flush=True)
    print(f"[扫描] {len(legi)} 个合法格,每格真渲一次 patch_m={args.patch_m}…\n", flush=True)

    rows = []
    for n, i in enumerate(legi):
        i = int(i)
        xy = (float(aw[i][0]), float(aw[i][1]))
        A, cen = footprint(xy)
        dist = float(np.hypot(xy[0] - tgt[0], xy[1] - tgt[1]))
        rows.append((i, xy, dist, A, A / max(A_tgt, 1), float(infl[i]), float(infl[i]) / imax, cen))
        if n % 20 == 0:
            print(f"  {n}/{len(legi)}", flush=True)

    np.savez_compressed(OUT / f"{args.out}.npz",
                        idx=np.array([r[0] for r in rows]),
                        world=np.array([r[1] for r in rows]),
                        dist=np.array([r[2] for r in rows]),
                        footprint=np.array([r[3] for r in rows]),
                        ratio=np.array([r[4] for r in rows]),
                        influence=np.array([r[5] for r in rows]),
                        centroid=np.array([r[7] for r in rows]),
                        target_idx=am, target_world=np.array(tgt), target_footprint=A_tgt,
                        patch_m=args.patch_m)

    cand = [r for r in rows if r[0] != am and r[2] >= args.min_dist
            and abs(r[4] - 1.0) <= args.tol]
    cand.sort(key=lambda r: r[5])
    print(f"\n[筛选] 实测足迹匹配 ±{args.tol:.0%}  距离 ≥{args.min_dist:.2f}m  ⇒ {len(cand)} 个候选")
    if not cand:
        print("[空] 放宽 --tol / --min-dist;下面列出全部格按足迹接近程度排序供参考:")
        cand = sorted([r for r in rows if r[0] != am and r[2] >= args.min_dist],
                      key=lambda r: abs(r[4] - 1.0))
    print(f"\n{'idx':>4s} {'world (x,y)':>18s} {'距离m':>7s} {'实测足迹':>9s} {'比':>6s} "
          f"{'influence':>10s} {'占最大':>7s} {'质心(r,c)':>12s}")
    for i, xy, dist, A, ratio, inf, rel, cen in cand[:args.top]:
        print(f"{i:4d} ({xy[0]:+.4f},{xy[1]:+.4f}) {dist:7.3f} {A:9d} {ratio:5.2f}× "
              f"{inf:10.1f} {rel:6.1%}  ({cen[0]:3.0f},{cen[1]:3.0f})")
    b = cand[0]
    print(f"\n[建议对照格] idx {b[0]}  world=({b[1][0]:+.4f},{b[1][1]:+.4f})")
    print(f"  实测足迹 {b[3]}px = 实验格的 {b[4]:.2f}×;influence 只有最优的 {b[6]:.1%};"
          f"距离 {b[2]:.3f}m")
    print(f"\n  drift_pool_prep.py --patch-m {args.patch_m} --episodes 40 "
          f"--cell {b[1][0]:.4f},{b[1][1]:.4f} --out drift_pool_ctrl")
    print(f"  drift_patch_train.py --pool drift_pool_ctrl --iters 500 --tag ctrl")
    print(f"  drift_eval_digital.py --patch drift_patch_ctrl --episodes 50 --out drift_evalA_ctrl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
