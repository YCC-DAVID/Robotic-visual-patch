#!/usr/bin/env python3
"""G0 梯度显著性可视化(2×3):
上排:干净帧 / gmag(ε₀) 叠加 / 3ε 平均 gmag;
下排:三个平移通道 s_0 s_1 s_2 各自的 |g|(RGB 合成)。
朝向:LiberoInputs 无旋转翻转(libero_policy.py:20-26 只转 dtype/布局),
梯度图与 s2f_scan_obs.npz 存的 224×224 帧逐像素对齐,直接叠。

用法: /home/user1/miniconda3/envs/openpi-libero/bin/python pi05probe/g0_viz.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = REPO / "pi05probe" / "out"
GOUT = OUT / "grad"
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"

z = np.load(GOUT / "g0_gradcheck.npz", allow_pickle=True)
d = np.load(OUT / "s2f_scan_obs.npz", allow_pickle=True)
FR = int(z["frame"])
img = d["clean_img224"][FR]                        # [224,224,3] uint8
g_ch = z["g_ch"]                                   # [3ch,3rgb,224,224]
gmag_eps = z["gmag_eps"]                           # [3,224,224]

gmag = gmag_eps[0]
gmean = gmag_eps.mean(0)
chan = np.linalg.norm(g_ch, axis=1)                # [3ch,224,224] 每个 s_c 的 RGB 合成

fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.6), facecolor=SURFACE)
panels = [
    (img, None, f"clean frame {FR}/16 (model-input pixels)"),
    (img, gmag, "‖∂s/∂x‖ per pixel, ε₀  (saliency)"),
    (img, gmean, "mean over 3 ε  (pairwise rank corr .98–.99)"),
    (None, chan[0], "s₀ = Σ a[k,0]  (x-translation)"),
    (None, chan[1], "s₁ = Σ a[k,1]  (y-translation)"),
    (None, chan[2], "s₂ = Σ a[k,2]  (z-translation)"),
]
for ax, (bg, hm, title) in zip(axes.flat, panels):
    if bg is not None:
        ax.imshow(bg)
    if hm is not None:
        vmax = np.percentile(hm, 99.8)
        ax.imshow(hm, cmap="inferno", alpha=.75 if bg is not None else 1.0,
                  vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10, color=INK2, pad=5)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("π0.5 · G0 gradient saliency, per-pixel (224×224 — vs FastWAM attention's 7×7 ceiling)",
             fontsize=12.5, color=INK, y=.97)
fig.text(.02, .925, "display normalized per panel (p99.8); bottom row = per-channel |∂s_c/∂x| "
         "combined over RGB — this is the resolution FD-influence can be proxied at",
         fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, .91))
f = GOUT / "viz_g0_saliency.png"
fig.savefig(f, dpi=135, facecolor=SURFACE)
print(f"[written] {f}")
