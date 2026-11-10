#!/usr/bin/env python3
"""S2.1 反事实扫描 + 聚合 → 影响力热图(py3.11,in-process,每帧共享 ε)。

对每帧 t:生成 ε_t(clean 与全部 M 锚点共享)⇒ Δa 里 ε 抵消,只剩 patch 效应。
分通道(计划 D,不对 7 维 L2):平移 a[0:3] / 旋转 a[3:6] 测地 / 夹爪 sign。
只在 executed prefix(前 5 步)上算。聚合 influence[i,c] = mean_t d[i,t,c]。
每帧另测 ε 地板(换 ε 前向 N 次的散布),标出未过地板的格子(计划 D5,排序不可信)。

用法:
    # 先 s2_scan_dump.py(py3.8)
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/s2_scan.py
"""
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"
PATCHED_TF = ROOT / "third_party" / "transformers_patched"
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"

for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np
# 注:绘图放到单独的 s2_heat.py(py3.8 有 matplotlib;py3.11 openpi-server 没有)。
# 本脚本只做 GPU 前向 + 存 s2_influence.npz / .txt。

AH, AD, EX = 10, 32, 5
NFLOOR = 10
SEED = 777


def rot_geodesic(a3, b3):
    from scipy.spatial.transform import Rotation as R
    return float(np.linalg.norm((R.from_rotvec(a3).inv() * R.from_rotvec(b3)).as_rotvec()))


def dchan(a, b):
    dt = float(np.linalg.norm(a[:, 0:3] - b[:, 0:3]))
    dr = float(np.sum([rot_geodesic(a[i, 3:6], b[i, 3:6]) for i in range(a.shape[0])]))
    dg = int(np.sum(np.sign(a[:, 6]) != np.sign(b[:, 6])))
    return dt, dr, dg


def main():
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    d = np.load(OUT / "s2_scan_obs.npz", allow_pickle=True)
    prompt = str(d["prompt"]); T = int(d["T"]); M = int(d["M"])
    cimg, cwri, cstate = d["clean_img224"], d["clean_wrist224"], d["clean_state8"]
    pimg, pwri = d["patched_img224"], d["patched_wrist224"]
    aw, legal, keep, vpx = d["anchor_world"], d["anchor_legal"], d["anchor_keepout"], d["visible_px"]
    print(f"[data] M={M} anchors  T={T} frames  prompt={prompt!r}", flush=True)

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    print("[policy] loaded", flush=True)
    rng = np.random.RandomState(SEED)

    def infer(img, wri, state, eps):
        el = {"observation/image": img, "observation/wrist_image": wri,
              "observation/state": state, "prompt": prompt}
        return np.asarray(policy.infer(el, noise=eps)["actions"])[:EX]

    eps_t = [rng.normal(size=(AH, AD)).astype(np.float32) for _ in range(T)]

    # 每帧 clean 动作 + ε 地板
    a_clean = []
    floor_t, floor_r = np.zeros(T), np.zeros(T)
    for t in range(T):
        a_clean.append(infer(cimg[t], cwri[t], cstate[t], eps_t[t]))
        ft, fr = [], []
        base_samples = [infer(cimg[t], cwri[t], cstate[t], rng.normal(size=(AH, AD)).astype(np.float32))
                        for _ in range(NFLOOR)]
        for i in range(NFLOOR):
            for j in range(i + 1, NFLOOR):
                x, y, _ = dchan(base_samples[i], base_samples[j]); ft.append(x); fr.append(y)
        floor_t[t] = np.percentile(ft, 95); floor_r[t] = np.percentile(fr, 95)
        print(f"  [floor] frame {t:2d}: 平移p95={floor_t[t]:.3f} 旋转p95={floor_r[t]:.3f}", flush=True)

    # 扫描:每锚点每帧 Δa
    D_t = np.zeros((M, T)); D_r = np.zeros((M, T)); D_g = np.zeros((M, T), int)
    for i in range(M):
        for t in range(T):
            ap = infer(pimg[i, t], pwri[i, t], cstate[t], eps_t[t])
            dt, dr, dg = dchan(ap, a_clean[t])
            D_t[i, t] = dt; D_r[i, t] = dr; D_g[i, t] = dg
        if i % 6 == 0:
            print(f"  [scan] anchor {i:2d}/{M}  meanΔ平移={D_t[i].mean():.3f}", flush=True)

    # 聚合(mean over frames)+ 地板(mean over frames of p95)
    I_t, I_r = D_t.mean(1), D_r.mean(1)
    I_g = D_g.sum(1)
    F_t, F_r = floor_t.mean(), floor_r.mean()
    reliable = (I_t > F_t) | (I_r > F_r) | (I_g > 0)

    np.savez_compressed(OUT / "s2_influence.npz",
                        anchor_world=aw, anchor_legal=legal, anchor_keepout=keep, visible_px=vpx,
                        D_trans=D_t, D_rot=D_r, D_grip=D_g,
                        I_trans=I_t, I_rot=I_r, I_grip=I_g, reliable=reliable,
                        floor_trans_t=floor_t, floor_rot_t=floor_r, ts=d["ts"])

    # ---- 报告
    lines = []
    def out(s=""):
        print(s, flush=True); lines.append(s)
    out("=" * 96)
    out(f"S2.1 影响力扫描  M={M} 锚点 × T={T} 帧  分通道(平移 m×0.05 / 旋转 rad / 夹爪翻转)")
    out(f"  ε 地板(帧均 p95):平移={F_t:.3f}  旋转={F_r:.3f}")
    out("=" * 96)
    order = np.argsort(-I_t)
    out("  排名 anchor world(x,y)     legal keepout        I平移(mm) I旋转  夹爪  过地板")
    for rank, i in enumerate(order):
        out(f"  {rank+1:3d}  #{int(d['anchor_idx'][i]):2d} ({aw[i][0]:5.2f},{aw[i][1]:5.2f}) "
            f"{str(bool(legal[i])):5s} {str(keep[i])[:12]:12s} "
            f"{I_t[i]*0.05*1000:7.1f}  {I_r[i]:.3f}  {I_g[i]:3d}   {'✅' if reliable[i] else '·'}")
    nrel = int(reliable.sum())
    out(f"\n  {nrel}/{M} 个格子过 ε 地板(排序可信);其余按 D5 排序不可信。")
    out("  (热图另由 s2_heat.py 从 s2_influence.npz 出图)")

    (OUT / "s2_influence.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT/'s2_influence.txt'} + s2_influence.npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
