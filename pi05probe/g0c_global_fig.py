#!/usr/bin/env python3
"""「初始帧梯度能否指导全局 influence」对比图(用户点名的形态)。

面板(同 61 加密锚点,世界坐标平面):
  ① FD influence 全局 = 16 帧均值(琥珀,昂贵基准:61 锚点 × 16 帧前向扫描)
  ② S_grad_pooled @ 帧 0(初始观测,青色 —— 用户想要的「开局一眼」代理)
  ③ S_grad_pooled @ 帧 8(放置阶段,青色,对照)
每个梯度帧用**该帧自己的贴纸像素掩码**(patched−clean 渲染差分)聚合。
文本里附帧 4(抓取)的相关作参考。

用法: /home/user1/miniconda3/envs/openpi-libero/bin/python pi05probe/g0c_global_fig.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402

REPO = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = REPO / "pi05probe" / "out"
GOUT = OUT / "grad"
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list(
    "amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
TEAL = LinearSegmentedColormap.from_list(
    "teal", ["#eef7f6", "#c9e9e3", "#93d4c8", "#4db3a4", "#1f7a6e", "#0b4a43"])
OBJ = {"bowl": (-0.098, -0.009), "plate": (0.062, -0.009)}
EX = 5


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def pooled(zo, gmag, fr):
    clean = zo["clean_img224"][fr].astype(np.int16)
    M = int(zo["M"])
    S = np.zeros(M)
    for i in range(M):
        m = (np.abs(zo["patched_img224"][i, fr].astype(np.int16) - clean) > 2).any(-1)
        S[i] = float(gmag[m].sum()) if m.any() else 0.0
    return S


def main():
    za = np.load(OUT / "s2f_actions.npz", allow_pickle=True)
    zo = np.load(OUT / "s2f_scan_obs.npz", allow_pickle=True)
    ze = np.load(GOUT / "g0_grad_early.npz", allow_pickle=True)   # 帧 0/4
    z8 = np.load(GOUT / "g0_gradcheck.npz", allow_pickle=True)    # 帧 8
    aw = za["anchor_world"][:, :2]; idx = za["anchor_idx"]; M = aw.shape[0]

    Ac, Ap = za["A_clean"], za["A_patched"]
    d = Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]
    fd_global = (np.linalg.norm(d.reshape(M, d.shape[1], -1), axis=2) * 50.0).mean(1)

    frames = [int(f) for f in ze["frames"]]
    S = {fr: pooled(zo, ze["gmag"][k], fr) for k, fr in enumerate(frames)}
    S[8] = pooled(zo, z8["gmag_eps"].mean(0), 8)

    r = {fr: spear(fd_global, S[fr]) for fr in S}
    r08 = spear(S[0], S[8])
    print("[corr] 全局FD ↔ grad_pooled: " +
          "  ".join(f"帧{fr}={r[fr]:+.2f}" for fr in sorted(r)) +
          f"   grad帧0↔grad帧8={r08:+.2f}")

    panels = [
        ("FD influence · GLOBAL (mean of 16 frames)\n61 anchors × 16 forward scans", fd_global, RAMP),
        (f"S_grad_pooled · frame 0 (initial obs)\none backward · rank corr vs global FD {r[0]:+.2f}", S[0], TEAL),
        (f"S_grad_pooled · frame 8 (placement)\none backward · rank corr vs global FD {r[8]:+.2f}", S[8], TEAL),
    ]
    fig = plt.figure(figsize=(13.6, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, wspace=.15, left=.045, right=.985, top=.75, bottom=.09)
    for pi, (title, v, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, pi])
        ax.scatter(aw[:, 1], aw[:, 0], s=86, c=v, cmap=cmap,
                   norm=Normalize(0, float(v.max())), edgecolor="white",
                   linewidth=.5, zorder=3)
        for nm, (oxx, oyy) in OBJ.items():
            ax.plot(oyy, oxx, "o", ms=9, mfc="none", mec="#2a78d6", mew=2.0, zorder=6)
            ax.annotate(nm, (oyy, oxx), textcoords="offset points", xytext=(7, 5),
                        fontsize=8, color="#2a78d6")
        am = int(v.argmax())
        ax.plot(aw[am, 1], aw[am, 0], "o", ms=16, mfc="none", mec=INK, mew=1.9, zorder=5)
        ax.invert_xaxis(); ax.set_aspect("equal")
        ax.set_title(f"{title}\npeak #{int(idx[am])} ({aw[am,0]:+.2f},{aw[am,1]:+.2f})",
                     fontsize=9.3, color=INK2, pad=6)
        ax.tick_params(colors=INK2, labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("π0.5 · can a single-backward gradient map predict GLOBAL FD influence?",
                 fontsize=13, color=INK, y=.965)
    extra = "  ".join(f"frame {fr}: {r[fr]:+.2f}" for fr in sorted(r))
    fig.text(.045, .885,
             f"black ring = each panel's argmax · rank corr vs global FD — {extra} · "
             f"grad f0↔f8 {r08:+.2f}\n"
             "gradient: translation channels, fixed ε, per-anchor sum over that frame's "
             "rendered sticker pixels; no delta_max correction",
             fontsize=9, color=INK2, va="top")
    f = GOUT / "fig_g0_grad_vs_global_influence.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE)
    print(f"[written] {f}")
    np.savez_compressed(GOUT / "g0_global_scores.npz",
                        anchor_idx=idx, anchor_world=za["anchor_world"],
                        fd_global=fd_global,
                        S_pooled_frames=np.array(sorted(S)),
                        S_pooled=np.stack([S[fr] for fr in sorted(S)]),
                        spear_vs_global=np.array([r[fr] for fr in sorted(S)]))


if __name__ == "__main__":
    main()
