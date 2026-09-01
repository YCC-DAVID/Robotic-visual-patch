#!/usr/bin/env python3
"""全轨迹 influence 热图,两种坐标系各一版。纯后处理。

画的量
-----
    Imag_avg[i] = mean_over_3_textures  Σ_{t=0..15} ‖ Σ_{k=0..4}(a_patch[i,t,k,0:3] − a_clean[t,k,0:3]) ‖₂ × 50
即"贴在第 i 个位置,整条 16 帧轨迹上被推动的平移命令总量(mm)"。只有平移,不含旋转/夹爪。
⚠️ 是**命令量**不是末端实际位移(实现率约 0.24)。

左图 · 相机像素坐标
    每个锚点画出它**真实的贴纸投影足迹**(clean/patched 图像差分掩码,frame 0),按 influence 上色。
    远处的锚点足迹天然更小 —— 这是真的透视,不是画法。
    斜纹 = 非法位置(贴纸与物体包围盒重叠,实际不能贴)。

右图 · 桌面世界坐标(俯视)
    6×6 网格,**已按左图的朝向摆放**(实测 col ≈ −299·y,row ≈ +222·x ⇒ 横轴= y 反向,纵轴= x)。
    这样两张图左右上下一致,可以直接对着看。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/make_influence_heatmap.py
"""
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
THR = 10                        # 差分掩码阈值,与 report_b4 一致
BINS = [0, 50, 100, 150, 200, 250, 300]          # 6 个颜色级,mm
INK1, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list(
    "amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
OBJ = {"bowl": (-0.098, -0.009), "plate": (0.062, -0.009), "wine bottle": (-0.204, -0.062)}


def main():
    tx = np.load(OUT / "texture_axis.npz", allow_pickle=True)
    ob = np.load(OUT / "s2_scan_obs.npz", allow_pickle=True)
    I, leg, aw, aidx = tx["Imag_avg"], tx["anchor_legal"].astype(bool), tx["anchor_world"], tx["anchor_idx"]
    M = len(I)

    clean = ob["clean_img224"][0].astype(np.int16)
    patch = ob["patched_img224"][:, 0].astype(np.int16)
    fp = (np.abs(patch - clean[None]).max(-1) > THR)                  # (M,224,224) 逐锚点足迹
    vis = fp.reshape(M, -1).sum(1) > 0

    norm = BoundaryNorm(BINS, RAMP.N)
    gmax, lmax = int(I.argmax()), int(np.where(leg)[0][np.argmax(I[leg])])

    fig = plt.figure(figsize=(13.6, 6.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.18, .045], wspace=.15,
                          left=.045, right=.935, top=.80, bottom=.10)
    axL, axR, axC = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2]))

    # ---------------- 左:像素坐标,真实足迹 ----------------
    g = np.dstack([clean.astype(float).mean(-1) / 255.0] * 3) * 0.72 + 0.24
    canvas = np.concatenate([g, np.ones((224, 224, 1))], axis=2)
    yy, xx = np.mgrid[0:224, 0:224]
    stripe = ((xx + yy) % 13) < 2                                     # 非法位置的斜纹
    for i in np.argsort(I):                                           # 弱的先画,强的压在上面
        if not vis[i]:
            continue
        canvas[fp[i]] = np.array(RAMP(norm(I[i])))
        if not leg[i]:
            canvas[fp[i] & stripe] = np.array([1.0, 1.0, 1.0, 1.0])
    axL.imshow(canvas, extent=[0, 224, 224, 0], interpolation="nearest")
    gx, gy = np.meshgrid(np.arange(224) + .5, np.arange(224) + .5)    # contour 不能用 extent 对齐
    for i in range(M):                                                # 逐锚点描边,否则相邻足迹连成一片
        if vis[i]:
            axL.contour(gx, gy, fp[i].astype(float), levels=[.5], colors=["#5a5854"], linewidths=.7)
    for i, lab, xy in [(lmax, "strongest legal", (.05, .12)), (gmax, "strongest overall\n(illegal)", (.62, .06))]:
        r, cc = np.nonzero(fp[i])
        axL.plot(cc.mean(), r.mean(), "o", ms=15, mfc="none", mec="#111111", mew=2.4, zorder=5)
        axL.annotate(lab, (cc.mean(), r.mean()), xy, textcoords="axes fraction", fontsize=9.5,
                     color=INK1, weight="bold", ha="left", va="center",
                     arrowprops=dict(arrowstyle="-", color=INK1, lw=1.2, shrinkB=9))
    axL.set_xticks([]); axL.set_yticks([])
    axL.set_title("where the probe actually lands in the camera image\n"
                  f"{int(vis.sum())} of {M} anchors are visible from this view",
                  fontsize=11, color=INK2, pad=8)

    # ---------------- 右:世界坐标(朝向与左图一致) ----------------
    xs, ys = np.unique(aw[:, 0]), np.unique(aw[:, 1])
    dx, dy = np.diff(xs).mean(), np.diff(ys).mean()
    H = np.full((len(xs), len(ys)), np.nan)
    L = np.zeros_like(H, dtype=bool)
    for i in range(M):
        r = int(np.argmin(abs(xs - aw[i, 0]))); c = len(ys) - 1 - int(np.argmin(abs(ys - aw[i, 1])))
        H[r, c], L[r, c] = I[i], leg[i]
    ext = [ys[-1] + dy / 2, ys[0] - dy / 2, xs[-1] + dx / 2, xs[0] - dx / 2]   # ← 格边界,不是格中心
    axR.imshow(H, cmap=RAMP, norm=norm, extent=ext, interpolation="nearest", aspect="equal")
    for r in range(len(xs)):
        for c in range(len(ys)):
            yc, xc = ys[len(ys) - 1 - c], xs[r]
            if not L[r, c]:
                axR.add_patch(plt.Rectangle((yc + dy / 2, xc - dx / 2), -dy, dx, fill=False,
                                            hatch="///", edgecolor="#ffffff", lw=0.0))
            axR.text(yc, xc, f"{H[r, c]:.0f}", ha="center", va="center", fontsize=8.5,
                     color="#ffffff" if H[r, c] > 150 else INK1)
    for nm, (ox, oy) in OBJ.items():
        axR.plot(oy, ox, "o", ms=9, mfc="none", mec="#2a78d6", mew=2.2, zorder=6)
        axR.annotate(nm, (oy, ox), xytext=(13, 0), textcoords="offset points", zorder=6,
                     ha="left", va="center", fontsize=9, color="#2a78d6", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.16", fc=SURFACE, ec="none", alpha=.88))
    axR.set_xlim(ext[0], ext[1])
    axR.set_ylim(ext[2], ext[3] - 0.035)              # 顶部留一点,别把酒瓶标记切掉
    axR.set_xlabel("world y (m)   —   axes oriented to match the camera image", color=INK2, fontsize=9.5)
    axR.set_ylabel("world x (m)", color=INK2, fontsize=9.5)
    axR.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    axR.set_title("the same numbers on the table, in millimetres\n"
                  "hatched = illegal (the patch would overlap an object)",
                  fontsize=11, color=INK2, pad=8)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=axC, ticks=BINS)
    cb.set_label("influence  (mm, summed over the whole trajectory)", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=INK2, labelsize=8.5)
    cb.outline.set_visible(False)

    fig.suptitle("Full-trajectory influence of a random probe patch, per position",
                 fontsize=14.5, color=INK1, y=.975)
    fig.text(.5, .925, "translation channel only  ·  averaged over 3 independent random textures  ·  "
                       "commanded amount, not how far the arm actually moves (~0.24x)",
             ha="center", fontsize=9.5, color=INK2)
    fig.text(.5, .885, f"strongest position overall = {I[gmax]:.0f} mm but illegal   ·   "
                       f"strongest legal = {I[lmax]:.0f} mm = {100*I[lmax]/I[gmax]:.0f}% of it   ·   "
                       f"legal mean {I[leg].mean():.0f} mm vs illegal mean {I[~leg].mean():.0f} mm",
             ha="center", fontsize=10, color=INK1)
    f = OUT / "fig_influence_heatmap.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}")
    print(f"  全局最大 #{int(aidx[gmax])} world({aw[gmax,0]:.2f},{aw[gmax,1]:.2f}) "
          f"{I[gmax]:.0f} mm  legal={bool(leg[gmax])}")
    print(f"  合法最大 #{int(aidx[lmax])} world({aw[lmax,0]:.2f},{aw[lmax,1]:.2f}) "
          f"{I[lmax]:.0f} mm  = 全局的 {100*I[lmax]/I[gmax]:.0f}%")
    print(f"  合法 {int(leg.sum())}/{M};合法均值 {I[leg].mean():.0f} mm,非法均值 {I[~leg].mean():.0f} mm")


if __name__ == "__main__":
    main()
