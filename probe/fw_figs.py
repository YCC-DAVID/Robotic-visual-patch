#!/usr/bin/env python3
"""FastWAM 的两张热图,和 π0.5 那套(make_influence_heatmap / make_task_attn_figs)对齐。

图 1 · influence 热图(相机像素足迹 + 桌面世界俯视)
    与 pi05probe/make_influence_heatmap.py 同款:左图每个锚点画真实贴纸投影足迹按 influence 上色,
    右图桌面世界坐标(朝向与相机一致)。这里 78 个点全合法,不画斜纹。

图 2 · attention 热图(文本→视频,叠在 agentview 上)
    与 pi05probe/make_task_attn_figs.py 同款:把 bowl/plate 名词 token 的图像图叠在 agentview 上,
    命中环 = attention 峰值格是否落在 influence 首选锚点的投影足迹上。
    ⚠️ FastWAM 是视频查询×文本键,取名词那一列、在 98 个 video token 上重新归一;base=左 7×7。

用法(有 matplotlib 的 env,如 openpi-libero):
    /home/user1/miniconda3/envs/openpi-libero/bin/python probe/fw_figs.py
"""
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402

REPO = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = REPO / "probe" / "out"
THR = 10
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list(
    "amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
OBJ = {"bowl": (-0.098, -0.009), "plate": (0.062, -0.009)}
BOWL, PLATE = np.array([-0.098, -0.009]), np.array([0.062, -0.009])
EX = 10


def influence(z):
    Ac, Ap = z["A_clean"], z["A_patched"]
    v = (Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2)
    return np.linalg.norm(v, axis=2).sum(1) * 50.0


def fig_influence():
    z = np.load(OUT / "fw_scan.npz", allow_pickle=True)
    I = influence(z)
    aw, idx = z["anchor_world"], z["anchor_idx"]
    vpx = z["visible_px"].mean(1)
    M = len(I)
    bins = [0, 50, 100, 150, 200, 250]
    norm = BoundaryNorm(bins, RAMP.N)
    gmax = int(I.argmax())

    fig = plt.figure(figsize=(12.6, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.18, .045], wspace=.16,
                          left=.04, right=.93, top=.82, bottom=.10)
    axL, axR, axC = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    # 左:世界俯视点图(FastWAM 没存逐锚点足迹掩码,用点大小=面积、颜色=influence)
    sc = axL.scatter(aw[:, 1], aw[:, 0], s=(vpx / vpx.max()) * 300 + 30, c=I, cmap=RAMP, norm=norm,
                     edgecolor="white", linewidth=.6, zorder=3)
    for nm, (ox, oy) in OBJ.items():
        axL.plot(oy, ox, "o", ms=10, mfc="none", mec="#2a78d6", mew=2.2, zorder=6)
        axL.annotate(nm, (oy, ox), xytext=(8, 0), textcoords="offset points", color="#2a78d6",
                     weight="bold", fontsize=9, va="center")
    axL.plot(aw[gmax, 1], aw[gmax, 0], "o", ms=17, mfc="none", mec=INK, mew=2.2, zorder=5)
    axL.annotate(f"strongest #{int(idx[gmax])}\n{I[gmax]:.0f} mm", (aw[gmax, 1], aw[gmax, 0]),
                 xytext=(10, 14), textcoords="offset points", fontsize=9, weight="bold", color=INK)
    axL.invert_xaxis()
    axL.set_xlabel("world y (m)  — oriented to match the camera", color=INK2, fontsize=9.5)
    axL.set_ylabel("world x (m)", color=INK2, fontsize=9.5)
    axL.set_title("FastWAM influence per legal position\nmarker size = patch visible area",
                  fontsize=11, color=INK2, pad=8)
    axL.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # 右:桌面世界热图(格状,数值标注),与 make_influence_heatmap 右图同款
    xs, ys = np.unique(aw[:, 0]), np.unique(aw[:, 1])
    H = np.full((len(xs), len(ys)), np.nan)
    for i in range(M):
        r = int(np.argmin(abs(xs - aw[i, 0]))); c = len(ys) - 1 - int(np.argmin(abs(ys - aw[i, 1])))
        H[r, c] = I[i]
    dx = np.diff(xs).mean() if len(xs) > 1 else 0.02
    dy = np.diff(ys).mean() if len(ys) > 1 else 0.02
    ext = [ys[-1] + dy / 2, ys[0] - dy / 2, xs[-1] + dx / 2, xs[0] - dx / 2]
    axR.imshow(H, cmap=RAMP, norm=norm, extent=ext, interpolation="nearest", aspect="equal")
    for r in range(len(xs)):
        for c in range(len(ys)):
            if not np.isnan(H[r, c]):
                yc, xc = ys[len(ys) - 1 - c], xs[r]
                axR.text(yc, xc, f"{H[r, c]:.0f}", ha="center", va="center", fontsize=7.5,
                         color="#ffffff" if H[r, c] > 130 else INK)
    for nm, (ox, oy) in OBJ.items():
        axR.plot(oy, ox, "o", ms=9, mfc="none", mec="#2a78d6", mew=2.2, zorder=6)
    axR.set_xlim(ext[0], ext[1]); axR.set_ylim(ext[2], ext[3])
    axR.set_xlabel("world y (m)", color=INK2, fontsize=9.5)
    axR.set_title("same numbers on the table (mm)\nlegal positions only (78)", fontsize=11, color=INK2, pad=8)
    axR.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=axC, ticks=bins)
    cb.set_label("influence (mm, translation, whole trajectory)", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("FastWAM · full-trajectory influence of a random probe patch, per position",
                 fontsize=13.5, color=INK, y=.965)
    f = OUT / "fig_fw_influence_heatmap.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}")


def fig_attention():
    za = np.load(OUT / "fw_attn.npz", allow_pickle=True)
    zs = np.load(OUT / "fw_scan.npz", allow_pickle=True)
    zc = np.load(OUT / "fw_anchor_cells.npz", allow_pickle=True)
    attn = za["attn"]                       # [T,L,Sq,Sk]
    toks = list(za["tokens"]); noun_idx = [int(v) for v in za["noun_idx"]]
    imgs = za["agentview"]                   # [T,256,256,3] 正立
    gh, gw, bc = int(za["grid_h"]), int(za["grid_w"]), int(za["base_cols"])
    T, L, Sq, Sk = attn.shape
    I = influence(zs); idx = zs["anchor_idx"]
    assert np.array_equal(zc["anchor_idx"], idx), "fw_anchor_cells 与 fw_scan 锚点顺序不一致"
    gmax = int(I.argmax())
    gcell = tuple(int(v) for v in zc["cell"][gmax])          # 模型帧 (row,col)
    gcen = zc["centroid"][gmax]                              # 模型帧 224px 质心

    # 只用接近段前几帧平均(与 pi05 make_task_attn_figs 的 FRAMES 一致精神)
    FR = min(T, 6)

    # 名词图 = 该词那一列(全部 video 查询对它的注意)→ 左 7×7 base → base 上重归一
    def noun_map(layer, nouns):
        col = attn[:FR, layer][:, :, nouns].sum(-1).mean(0)   # [Sq]
        base = col.reshape(gh, gw)[:, :bc]                    # 模型帧 base 7×7
        return base / (base.sum() + 1e-9)

    def entropy(b):
        p = b.flatten() + 1e-12; p = p / p.sum(); return float(-(p * np.log(p)).sum())
    ent = [entropy(noun_map(l, noun_idx)) for l in range(L)]
    best = int(np.argmin(ent))
    layers = [max(1, best // 3), best, min(L - 1, best + (L - best) // 2)]
    names = [f"early L{layers[0]}", f"sharpest L{best}", f"late L{layers[2]}"]

    # ⚠️ 对齐:模型输入 = agentview 的 rot180;存的底图是正立(只翻了行)。
    #    两者行相同、列互为镜像 ⇒ base 7×7 要左右翻列才能叠回正立图。
    gx = 256.0 - float(gcen[1]) * 256.0 / 224.0              # 质心 → 正立像素
    gy = float(gcen[0]) * 256.0 / 224.0

    fig = plt.figure(figsize=(12.6, 9.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, wspace=.06, hspace=.22, left=.03, right=.98,
                          top=.855, bottom=.05)
    base_img = imgs[0]
    hits = []
    for ri, (nname, nlist) in enumerate([("bowl", [noun_idx[0]]), ("plate", [noun_idx[1]])]):
        for ci, layer in enumerate(layers):
            ax = fig.add_subplot(gs[ri, ci])
            ax.imshow(base_img, extent=[0, 256, 256, 0])
            b = noun_map(layer, nlist)
            disp = b[:, ::-1]                                # 模型帧 → 正立帧(翻列)
            up = np.kron(disp, np.ones((256 // gh + 1, 256 // bc + 1)))[:256, :256]
            ax.imshow(up, extent=[0, 256, 256, 0], cmap="hot", alpha=.5,
                      interpolation="nearest")
            pr, pc = np.unravel_index(int(b.argmax()), b.shape)   # 峰值格(模型帧)
            hy = (pr + .5) * 256 / gh; hx = ((bc - 1 - pc) + .5) * 256 / bc
            hit = (pr, pc) == gcell
            cheb = max(abs(pr - gcell[0]), abs(pc - gcell[1]))
            hits.append(((nname, layer), (pr, pc), hit))
            ax.plot(hx, hy, "s", ms=21, mfc="none", mec="#33e0c0", mew=2.6, zorder=5)
            ax.plot(gx, gy, "X", ms=13, mfc="#e040fb", mec="white", mew=1.3, zorder=6)
            tag = "HIT" if hit else ("adjacent (1 away)" if cheb == 1 else f"{cheb} cells away")
            ax.set_title(f"{names[ci]} · '{nname}'\npeak cell ({pr},{pc}) vs influence-max "
                         f"cell {gcell} → {tag}", fontsize=9.5, color=INK2)
            ax.set_xticks([]); ax.set_yticks([])
    fig.text(.03, .895, "teal square = attention peak base-cell   ·   magenta X = strongest "
             f"influence anchor #{int(idx[gmax])} ({I[gmax]:.0f} mm)   ·   grid mirrored back "
             "to the upright view (model input is rot180)", fontsize=9.5, color=INK2)
    fig.suptitle("FastWAM · text(bowl / plate) → video cross-attention on agentview "
                 "(head-summed, base view, re-normalized over 49 base tokens)",
                 fontsize=13, color=INK, y=.955)
    f = OUT / "fig_fw_attention.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE); plt.close(fig)
    nh = sum(1 for _, _, h in hits if h)
    print(f"[written] {f}  sharpest L{best} entropy {ent[best]:.2f}  "
          f"peak==influence-max-cell 命中 {nh}/{len(hits)} 面板")


if __name__ == "__main__":
    fig_influence()
    if (OUT / "fw_attn.npz").is_file():
        fig_attention()
    else:
        print("[skip] fw_attn.npz 还没有,先跑 fw_attn.py")
