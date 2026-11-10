#!/usr/bin/env python3
"""DRIFT 复现 · 从实测数据拟合"渲染代理变换",供 EOT 训练使用(对应 HANDOFF 4.24)。

为什么需要
--------
评测 B 实测(§4.21/§4.23):
  · 满分辨率图案 → 渲染相关 +0.502、TV 保留 25% → 物理 ASR 0
  · band32 图案   → 渲染相关 +0.952、TV 保留 75% → 物理 ASR 仍 ~0(47/50)
⇒ **保真度不是瓶颈,脆弱性才是**:12% 逐像素误差 + 18% 压暗就把 92% ASR 打成 0。

所以要在训练里把渲染变换套上(EOT),让解自带余量。本脚本从**已有的渲染配对数据**
拟合这个变换,不用重新跑模型:

    render(P) ≈ a ⊙ blur_σ(P) + b        (逐通道 affine + 各向同性高斯模糊)

做法:对若干候选 σ 做高斯模糊,再对每个通道最小二乘拟合 (a,b),取残差最小的 σ。
拟合用的配对来自真渲一次的足迹像素(与 drift_eval_physical.py 的缺口诊断同法)。

产物 out/drift_render_fit.npz:sigma / a / b / 残差 / 相关,给 drift_patch_train.py --eot 用。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_fit_render.py \
        --patch drift_patch_band32
"""
import argparse
import os
import pathlib
import sys

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

import cv2  # noqa: E402
import yaml  # noqa: E402

RES, RESIZE, NWAIT = 256, 224, 10
DUMMY = [0.0] * 6 + [-1.0]
TASK = "put_the_bowl_on_the_plate"
SIGMAS = (0.0, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0)


def model_input(obs):
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="drift_patch_band32",
                    help="用哪个图案的渲染配对来拟合(建议用带限版,保真度高、拟合更可信)")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--thresh", type=int, default=12)
    ap.add_argument("--frames", type=int, default=6, help="用多少个不同 sim state 的渲染做拟合")
    ap.add_argument("--out", default="drift_render_fit")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    import torch

    d = np.load(OUT / f"{args.patch}.npz", allow_pickle=True)
    P = d["P"].transpose(1, 2, 0)                       # [224,224,3] in [0,1]
    patch_m = float(d["patch_m"])
    cell_xy = (float(d["cell_xy"][0]), float(d["cell_xy"][1]))
    ptag = args.patch.replace("drift_patch_", "")
    tex = PROBE / "config" / f"drift_texture_{ptag}.png"
    assert tex.is_file(), f"缺纹理 {tex} —— 先跑 drift_texture.py --patch {args.patch}"

    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    cfg["patch"]["size_wh"] = [patch_m, patch_m]
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

    # 取若干不同 sim state(不同 init + 不同 warm-up 步数)⇒ 覆盖不同机械臂位姿/阴影
    # ⚠️ 每帧**重建**两个 env 并用同一 seed:fixture 不在 qpos 里(见 HANDOFF §5 与 attack_rollout
    #    的 "每次 reset 前重 seed" 注释),复用 env 反复 reset 会让两边 fixture 错位,
    #    mask 就会把全图差异算进来(实测足迹被虚增到 ~8000px,相关掉到 +0.28)。
    A_list, B_list = [], []
    for f in range(args.frames):
        cenv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        cenv.seed(args.seed); cenv.reset()
        cenv.set_init_state(inits[f % len(inits)])
        for _ in range(NWAIT + f * 4):
            cenv.step(DUMMY)
        st = cenv.get_sim_state().copy()
        c_img = model_input(cenv.regenerate_obs_from_state(st))
        cenv.close()

        penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        penv.env.set_xml_processor(sp.make_xml_processor(cfg, world, str(tex)))
        penv.seed(args.seed); penv.reset()
        p_img = model_input(penv.regenerate_obs_from_state(st))
        penv.close()

        m = np.abs(p_img.astype(np.int16) - c_img.astype(np.int16)).max(-1) > args.thresh
        print(f"  [frame {f}] 足迹 {int(m.sum())}px", flush=True)
        if m.sum() < 200 or m.sum() > 6000:
            print("    ↑ 足迹异常(应约 3700~4300px),跳过 —— 可能 fixture 错位", flush=True)
            continue
        A_list.append((P, m))
        B_list.append(p_img.astype(np.float32) / 255.0)
    assert A_list, "没有可用的渲染配对"

    print(f"\n[拟合] render(P) ≈ a ⊙ blur_σ(P) + b,逐通道最小二乘,扫 σ\n", flush=True)
    print(f"{'σ':>6s} {'残差RMSE':>10s} {'相关':>8s}   a(RGB)                b(RGB)")
    best = None
    for s in SIGMAS:
        num = np.zeros(3); den = np.zeros(3)
        xs = [[] for _ in range(3)]; ys = [[] for _ in range(3)]
        for (Pi, m), got in zip(A_list, B_list):
            Pb = Pi if s == 0 else cv2.GaussianBlur(Pi, (0, 0), s)
            for c in range(3):
                xs[c].append(Pb[..., c][m]); ys[c].append(got[..., c][m])
        a = np.zeros(3); b = np.zeros(3); res = []; cor = []
        for c in range(3):
            x = np.concatenate(xs[c]); y = np.concatenate(ys[c])
            X = np.stack([x, np.ones_like(x)], 1)
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            a[c], b[c] = sol
            pred = X @ sol
            res.append(np.sqrt(((pred - y) ** 2).mean()))
            cor.append(np.corrcoef(x, y)[0, 1])
        rmse = float(np.mean(res)); cc = float(np.mean(cor))
        print(f"{s:6.1f} {rmse:10.4f} {cc:+8.3f}   "
              f"[{a[0]:.3f} {a[1]:.3f} {a[2]:.3f}]   [{b[0]:.3f} {b[1]:.3f} {b[2]:.3f}]")
        if best is None or rmse < best[1]:
            best = (s, rmse, cc, a.copy(), b.copy())
    s, rmse, cc, a, b = best
    print(f"\n[最优] σ={s:.1f}  RMSE={rmse:.4f}  相关={cc:+.3f}")
    print(f"  a(RGB) = [{a[0]:.3f} {a[1]:.3f} {a[2]:.3f}]   ← 对比度增益(<1 = 被压缩)")
    print(f"  b(RGB) = [{b[0]:.3f} {b[1]:.3f} {b[2]:.3f}]   ← 亮度偏置")
    print(f"\n  解读:σ={s:.1f} 是渲染链的等效模糊;a<1 说明对比度被压缩,")
    print(f"        这两项合起来就是 §4.23 里 12% 逐像素误差 + 18% 压暗的来源。")
    np.savez(OUT / f"{args.out}.npz", sigma=s, a=a, b=b, rmse=rmse, corr=cc,
             patch=args.patch, sigmas=np.array(SIGMAS), frames=len(A_list))
    print(f"\n[written] {OUT/(args.out+'.npz')}  → 给 drift_patch_train.py --eot 用", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
