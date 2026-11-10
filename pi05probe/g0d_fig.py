#!/usr/bin/env python3
"""token 通路分解的平面图(2×3):全局 FD 基准 + 五个 KV 通路变体的帧 0 梯度分数。
数据来自 g0_desttoken.npz(g0d_desttoken.py)。

用法: /home/user1/miniconda3/envs/openpi-libero/bin/python pi05probe/g0d_fig.py
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

z = np.load(GOUT / "g0_desttoken.npz", allow_pickle=True)
za = np.load(OUT / "s2f_actions.npz", allow_pickle=True)
aw = za["anchor_world"][:, :2]; idx = za["anchor_idx"]
variants = [str(v) for v in z["variants"]]
S, rs, gsum = z["S"], z["spear_vs_global"], z["gsum"]
fd = z["fd_global"]

LBL = {"full": "full (all routes)", "dest": "dest: “plate” token only",
       "src": "src: “bowl” token only", "lang": "all 8 language tokens",
       "img": "all 768 image tokens"}
panels = [("FD influence · GLOBAL (16-frame mean)\nexpensive baseline", fd, RAMP)]
for k, name in enumerate(variants):
    share = gsum[k] / gsum[variants.index("full")] * 100
    panels.append((f"grad @ frame 0 via {LBL[name]}\nrank corr vs global FD "
                   f"{rs[k]:+.2f} · carries {share:.1f}% of |g|", S[k], TEAL))

fig = plt.figure(figsize=(13.2, 8.6), facecolor=SURFACE)
gs = fig.add_gridspec(2, 3, wspace=.13, hspace=.30, left=.04, right=.985,
                      top=.795, bottom=.05)
for pi, (title, v, cmap) in enumerate(panels):
    ax = fig.add_subplot(gs[pi // 3, pi % 3])
    ax.scatter(aw[:, 1], aw[:, 0], s=64, c=v, cmap=cmap,
               norm=Normalize(0, float(np.max(v))), edgecolor="white",
               linewidth=.5, zorder=3)
    for nm, (oxx, oyy) in OBJ.items():
        ax.plot(oyy, oxx, "o", ms=8, mfc="none", mec="#2a78d6", mew=1.8, zorder=6)
    am = int(np.argmax(v))
    ax.plot(aw[am, 1], aw[am, 0], "o", ms=14, mfc="none", mec=INK, mew=1.8, zorder=5)
    ax.invert_xaxis(); ax.set_aspect("equal")
    ax.set_title(f"{title}\npeak #{int(idx[am])}", fontsize=9, color=INK2, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_alpha(.25)
fig.suptitle("π0.5 · gradient routed through single prefix-token KV paths "
             "(frame 0) — every route predicts global influence",
             fontsize=12.5, color=INK, y=.965)
fig.text(.04, .905, "surgery: all KV positions except the kept ones are detached "
         "(forward bit-identical, verified) · black ring = argmax · blue rings = bowl/plate\n"
         "the placement ranking survives even through a single token's route — the spatial "
         "signal is redundantly mixed into every prefix token, destination is not special",
         fontsize=9, color=INK2, va="top")
f = GOUT / "fig_g0_token_routes.png"
fig.savefig(f, dpi=135, facecolor=SURFACE)
print(f"[written] {f}")
