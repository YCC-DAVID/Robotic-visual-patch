#!/usr/bin/env python3
"""FastWAM · 梯度代理跨任务验证图(2×3):
上排 = 三任务的 FD influence(10 帧总,琥珀,每个 = 780 次前向的 ground truth);
下排 = 各任务初始帧一次 backward 的梯度分数(青),标秩相关。
数据:fw_grad_f0.npz(nnmc62 跑的 fw_grad.py)。

用法: /home/user1/miniconda3/envs/openpi-libero/bin/python probe/fw_grad_fig.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/probe/out")
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list(
    "amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
TEAL = LinearSegmentedColormap.from_list(
    "teal", ["#eef7f6", "#c9e9e3", "#93d4c8", "#4db3a4", "#1f7a6e", "#0b4a43"])

z = np.load(OUT / "fw_grad_f0.npz", allow_pickle=True)
aw = z["anchor_world"][:, :2]
idx = z["anchor_idx"]
TASKS = [
    ("put_the_bowl_on_the_plate",          "bowl → plate",   ("plate",   (0.062, -0.009))),
    ("put_the_wine_bottle_on_the_rack",    "bottle → rack",  ("rack",    (-0.267, -0.251))),
    ("put_the_bowl_on_top_of_the_cabinet", "bowl → cabinet", ("cabinet", (0.040, -0.234))),
]

fig = plt.figure(figsize=(13.2, 8.8), facecolor=SURFACE)
gs = fig.add_gridspec(2, 3, wspace=.13, hspace=.22, left=.04, right=.985,
                      top=.83, bottom=.05)
for ci, (stem, title, (tn, txy)) in enumerate(TASKS):
    fd = z[f"{stem}__fd_total"]
    S = z[f"{stem}__S_pooled"]
    r = float(z[f"{stem}__spear_total"])
    for ri, (v, cmap, lab) in enumerate([
            (fd, RAMP, "FD influence (10-frame total, 780 forwards)"),
            (S, TEAL, "gradient @ initial frame (ONE backward)")]):
        ax = fig.add_subplot(gs[ri, ci])
        ax.scatter(aw[:, 1], aw[:, 0], s=64, c=v, cmap=cmap,
                   norm=Normalize(0, float(v.max())), edgecolor="white",
                   linewidth=.5, zorder=3)
        ax.plot(txy[1], txy[0], "*", ms=14, mfc="#f2b736", mec=INK, mew=.8, zorder=6)
        am = int(v.argmax())
        ax.plot(aw[am, 1], aw[am, 0], "o", ms=14, mfc="none", mec=INK, mew=1.8, zorder=5)
        ax.invert_xaxis(); ax.set_aspect("equal")
        ax.set_xlim(.36, -.36); ax.set_ylim(-.34, .32)
        head = f"{title}\n{lab}\npeak #{int(idx[am])}"
        if ri == 1:
            head += f" · rank corr {r:+.2f}"
        ax.set_title(head, fontsize=9, color=INK2, pad=5)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_alpha(.25)
fig.suptitle("FastWAM · does the single-backward gradient proxy hold on a second model, "
             "across destinations?", fontsize=12.5, color=INK, y=.965)
fig.text(.04, .915,
         "same recipe as π0.5 (fixed ε, translation channel-sums, sticker-mask pooling) · "
         "faithfulness red line: reimplementation ≡ official infer_action bit-for-bit\n"
         "rank corr vs 10-frame FD ground truth: plate +0.90 · rack +0.83 · cabinet +0.68  "
         "(π0.5 plate was +0.94) — gradient proxy transfers across model AND destination · "
         "gold star = destination",
         fontsize=9, color=INK2, va="top")
f = OUT / "fig_fw_grad_vs_influence.png"
fig.savefig(f, dpi=135, facecolor=SURFACE)
print(f"[written] {f}")
for stem, title, _ in TASKS:
    print(f"  {title:15s} corr_total={float(z[f'{stem}__spear_total']):+.2f} "
          f"corr_f0={float(z[f'{stem}__spear_f0']):+.2f}")
