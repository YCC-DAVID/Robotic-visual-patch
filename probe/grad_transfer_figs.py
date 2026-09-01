#!/usr/bin/env python3
"""梯度代理迁移性两张图:
  A) 跨模型:同一 plate 任务,π0.5(61 锚点)vs FastWAM(78 锚点),
     上排 FD 全局 influence / 下排初始帧一次 backward 的梯度。
  B) 跨任务:FastWAM 三目标(plate/rack/cabinet),上排 FD / 下排梯度。
两张都在世界坐标平面(与之前的 influence 图同款),标秩相关。

数据:pi05probe/out/grad/g0_global_scores.npz(π0.5 帧0/4/8,取帧0)
     probe/out/fw_grad_f0.npz(FastWAM 三任务帧0)

用法: /home/user1/miniconda3/envs/openpi-libero/bin/python probe/grad_transfer_figs.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402

REPO = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = REPO / "probe" / "out"
GOUT = REPO / "pi05probe" / "out" / "grad"
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list(
    "amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
TEAL = LinearSegmentedColormap.from_list(
    "teal", ["#eef7f6", "#c9e9e3", "#93d4c8", "#4db3a4", "#1f7a6e", "#0b4a43"])
PLATE = (0.062, -0.009)
TGT = {"put_the_bowl_on_the_plate": ("plate", (0.062, -0.009)),
       "put_the_wine_bottle_on_the_rack": ("rack", (-0.267, -0.251)),
       "put_the_bowl_on_top_of_the_cabinet": ("cabinet", (0.040, -0.234))}


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def panel(ax, aw, v, cmap, title, tgt=None, xlim=(.36, -.36), ylim=(-.34, .32)):
    ax.scatter(aw[:, 1], aw[:, 0], s=60, c=v, cmap=cmap, norm=Normalize(0, float(v.max())),
               edgecolor="white", linewidth=.5, zorder=3)
    if tgt is not None:
        ax.plot(tgt[1], tgt[0], "*", ms=14, mfc="#f2b736", mec=INK, mew=.8, zorder=6)
    am = int(v.argmax())
    ax.plot(aw[am, 1], aw[am, 0], "o", ms=14, mfc="none", mec=INK, mew=1.8, zorder=5)
    ax.invert_xaxis(); ax.set_aspect("equal"); ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=9.5, color=INK2, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_alpha(.25)


def fig_cross_model():
    zp = np.load(GOUT / "g0_global_scores.npz", allow_pickle=True)
    zf = np.load(OUT / "fw_grad_f0.npz", allow_pickle=True)
    awp = zp["anchor_world"][:, :2]
    fdp, gp = zp["fd_global"], zp["S_pooled"][0]           # π0.5 帧0
    rp = float(zp["spear_vs_global"][0])
    awf = zf["anchor_world"][:, :2]
    stem = "put_the_bowl_on_the_plate"
    fdf, gf = zf[f"{stem}__fd_total"], zf[f"{stem}__S_pooled"]
    rf = float(zf[f"{stem}__spear_total"])

    fig = plt.figure(figsize=(9.4, 8.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, wspace=.10, hspace=.20, left=.05, right=.97,
                          top=.82, bottom=.05)
    cols = [("π0.5  (PaliGemma + flow-matching, 61 anchors)", awp, fdp, gp, rp),
            ("FastWAM  (video DiT + action expert, 78 anchors)", awf, fdf, gf, rf)]
    for ci, (mtitle, aw, fd, g, r) in enumerate(cols):
        panel(fig.add_subplot(gs[0, ci]), aw, fd, RAMP,
              f"{mtitle}\nFD influence (global, 100s of forwards)", PLATE)
        panel(fig.add_subplot(gs[1, ci]), aw, g, TEAL,
              f"gradient @ initial frame (ONE backward)\nrank corr vs global FD  {r:+.2f}", PLATE)
    fig.suptitle("Gradient proxy · CROSS-MODEL  (same task: bowl → plate)",
                 fontsize=13, color=INK, y=.955)
    fig.text(.05, .89,
             "same recipe both models (fixed ε · translation channel-sums · sticker-mask pooling) · "
             "black ring = argmax · gold star = plate\n"
             f"a single backward reproduces the expensive influence ranking on BOTH architectures "
             f"— π0.5 {rp:+.2f}, FastWAM {rf:+.2f}",
             fontsize=9, color=INK2, va="top")
    f = OUT / "fig_grad_cross_model.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}  π0.5 {rp:+.2f}  FastWAM {rf:+.2f}")


def fig_cross_task():
    zf = np.load(OUT / "fw_grad_f0.npz", allow_pickle=True)
    aw = zf["anchor_world"][:, :2]
    stems = [str(s) for s in zf["tasks"]]
    labels = {"put_the_bowl_on_the_plate": "bowl → plate",
              "put_the_wine_bottle_on_the_rack": "bottle → rack",
              "put_the_bowl_on_top_of_the_cabinet": "bowl → cabinet"}

    fig = plt.figure(figsize=(13.2, 8.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, wspace=.10, hspace=.20, left=.04, right=.985,
                          top=.82, bottom=.05)
    corrs = []
    for ci, stem in enumerate(stems):
        fd, g = zf[f"{stem}__fd_total"], zf[f"{stem}__S_pooled"]
        r = float(zf[f"{stem}__spear_total"]); corrs.append(r)
        tn, txy = TGT[stem]
        panel(fig.add_subplot(gs[0, ci]), aw, fd, RAMP,
              f"{labels[stem]}\nFD influence (global, 780 forwards)", txy)
        panel(fig.add_subplot(gs[1, ci]), aw, g, TEAL,
              f"gradient @ initial frame (ONE backward)\nrank corr vs global FD  {r:+.2f}", txy)
    fig.suptitle("Gradient proxy · CROSS-TASK  (FastWAM, three destinations)",
                 fontsize=13, color=INK, y=.955)
    fig.text(.04, .89,
             "same 78 anchors · same probe patch · only the destination changes · "
             "black ring = argmax · gold star = destination\n"
             f"one backward tracks the influence ranking as the goal moves — "
             f"plate {corrs[0]:+.2f} · rack {corrs[1]:+.2f} · cabinet {corrs[2]:+.2f}",
             fontsize=9, color=INK2, va="top")
    f = OUT / "fig_grad_cross_task.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}  " + " ".join(f"{s.split('_')[-1]}={c:+.2f}"
          for s, c in zip(stems, corrs)))


if __name__ == "__main__":
    fig_cross_model()
    fig_cross_task()
