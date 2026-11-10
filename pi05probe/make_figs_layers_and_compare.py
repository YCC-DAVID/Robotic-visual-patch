#!/usr/bin/env python3
"""三张汇报图(纯后处理,零 GPU):

  fig_attn_layers_frames.png     跨帧 × 跨层的 attention,叠在模型真实输入图上
  fig_attn_object_vs_dest.png    定量:每层的注意力有多少落在【操作对象=碗】vs【目的地=盘子】
  fig_legal_attn_vs_influence.png 刨掉非法位置后,跨帧 attention 叠加 vs 全轨迹 influence(左右对比)

关键实现约定
-----------
1. **碗会被抓起来移动**(z 0.898 → 1.000,xy 位移 0.172 m),所以物体位置**必须逐帧**
   从 qpos 取(碗 = flatten[10:13],盘子 = flatten[31:34]),不能用静态掩码。
2. 世界 → 图像格的投影:**只翻列、不翻行**。这是用 36 个锚点的 diff 掩码质心实测定出来的
   (四种翻转组合误差 8.98 / 12.77 / **0.53** / 8.10 格)。`reproject.py` 的行是翻反的。
3. attention 一律:head 求和(存盘时已做)→ 在 512 个图像 token 上重归一化 → 取名词图。
   重归一化是规格 A4 的要求(不做的话质量会塌到起始符上)。
4. 左右对比图用**名次**(1..17)而不是原始数值 —— attention 分数和 influence(mm)量纲不同,
   直接并排看颜色会误导。名次是两者唯一可公平并排的量。

配色(取自 dataviz 规范参考配色的固定前两槽,已算过分离度)
    碗/操作对象 = #2a78d6(blue, slot 1)   盘子/目的地 = #eb6834(orange, slot 2)
    正常视觉 ΔE 33.6 / 绿色盲 31.7 / 红色盲 24.7,全部达标。
    两条序列同时用**不同标记形状**(圆 vs 方),身份不依赖颜色单独承载。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/make_figs_layers_and_compare.py
"""
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NOUN_ROWS = [3, 6]          # ['▁bowl', '▁plate']
BOWL_QPOS = slice(10, 13)
PLATE_QPOS = slice(31, 34)
WIN = 1                      # 物体窗口半径(格) → 3×3

C_OBJ = "#2a78d6"            # 操作对象(碗)
C_DEST = "#eb6834"           # 目的地(盘子)
INK1, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"

LAYERS_GRID = [0, 3, 7, 10, 17]
FRAMES_GRID = [0, 5, 9, 12, 15]

# 单色顺序阶(一个色相,浅→深),用 alpha 承载量级 —— 不用彩虹图
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e8f1fb", "#2a78d6", "#10305a"])


def world_to_cell(Pw, K, E):
    """世界 3D → 模型输入图的 16×16 格坐标 (row, col)。只翻列、不翻行(见文件头 2)。"""
    Pc = np.linalg.inv(E) @ np.array([Pw[0], Pw[1], Pw[2], 1.0])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = Pc[2]
    u_raw, v_raw = cx + fx * Pc[0] / z, cy + fy * Pc[1] / z
    u224 = ((255.0 - u_raw) + 0.5) * 224.0 / 256.0 - 0.5
    v224 = (v_raw + 0.5) * 224.0 / 256.0 - 0.5
    return v224 / 14.0 - 0.5, u224 / 14.0 - 0.5


def win_mass(sal, r, c, win=WIN):
    """sal (16,16) 在 (r,c) 周围 (2*win+1)² 窗口内的注意力质量占全图的比例。"""
    ri, ci = int(round(r)), int(round(c))
    r0, r1 = max(0, ri - win), min(16, ri + win + 1)
    c0, c1 = max(0, ci - win), min(16, ci + win + 1)
    tot = sal.sum()
    if tot <= 0 or r1 <= r0 or c1 <= c0:
        return np.nan
    return float(sal[r0:r1, c0:c1].sum() / tot)


def renorm_noun(A_layer):
    """A_layer (T,Z,V,16,16) 原始 head-求和 → 名词图 (T,16,16),主视角,已重归一化。"""
    X = A_layer / np.clip(A_layer.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)
    return X[:, NOUN_ROWS, 0].sum(1)


def token_map(A_layer, tok_row):
    X = A_layer / np.clip(A_layer.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)
    return X[:, tok_row, 0]


def overlay(ax, img, sal, vmax=None):
    """把 sal 以单色阶 + alpha 叠在 img 上。"""
    ax.imshow(img)
    s = np.kron(sal, np.ones((14, 14)))
    vmax = vmax if vmax is not None else s.max()
    a = np.clip(s / max(vmax, 1e-12), 0, 1)
    ax.imshow(s, cmap=CMAP_SEQ, alpha=0.30 + 0.55 * a, vmin=0, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    az = np.load(OUT / "attn_traj_put_the_bowl_on_the_plate.npz", allow_pickle=True)
    tz = np.load(OUT / "traj_put_the_bowl_on_the_plate.npz", allow_pickle=False)
    sf = np.load(OUT / "shared_frame.npz", allow_pickle=True)
    b4 = np.load(OUT / "b4_attn_vs_influence.npz", allow_pickle=True)
    tx = np.load(OUT / "texture_axis.npz", allow_pickle=True)
    K = sf["put_the_bowl_on_the_plate__K_agentview"]
    E = sf["put_the_bowl_on_the_plate__E_agentview"]

    A = az["bowl_plate_orig__attn"].astype(np.float64)          # (T,L,Z,V,16,16)
    T, L = A.shape[0], A.shape[1]
    ts = tz["ts"]
    imgs = np.stack([tz[f"f{k:03d}__img224"] for k in range(T)])
    flats = np.stack([tz[f"f{k:03d}__flatten"] for k in range(T)])
    bowl_w = flats[:, BOWL_QPOS]
    plate_w = flats[:, PLATE_QPOS]
    bowl_rc = np.array([world_to_cell(bowl_w[t], K, E) for t in range(T)])
    plate_rc = np.array([world_to_cell(plate_w[t], K, E) for t in range(T)])
    print(f"[data] T={T} L={L}  碗 xy 位移={np.linalg.norm(bowl_w[-1,:2]-bowl_w[0,:2]):.3f} m  "
          f"碗格 {bowl_rc[0].round(1)} → {bowl_rc[-1].round(1)}")

    # ============================================ 图 1:跨帧 × 跨层
    nr, nc = len(LAYERS_GRID), len(FRAMES_GRID)
    fig, axes = plt.subplots(nr, nc, figsize=(2.35 * nc + 1.2, 2.35 * nr + 1.0),
                             facecolor=SURFACE)
    for i, l in enumerate(LAYERS_GRID):
        sal = renorm_noun(A[:, l])
        vmax = np.percentile(sal[FRAMES_GRID], 99.5)
        for j, t in enumerate(FRAMES_GRID):
            ax = axes[i, j]
            overlay(ax, imgs[t], sal[t], vmax)
            ax.plot((bowl_rc[t, 1] + .5) * 14, (bowl_rc[t, 0] + .5) * 14, "o",
                    mec=C_OBJ, mfc="none", ms=15, mew=2.6)
            ax.plot((plate_rc[t, 1] + .5) * 14, (plate_rc[t, 0] + .5) * 14, "s",
                    mec=C_DEST, mfc="none", ms=15, mew=2.6)
            if i == 0:
                ax.set_title(f"env step {ts[t]}", fontsize=10, color=INK1, pad=6)
            if j == 0:
                ax.set_ylabel(f"layer {l}", fontsize=11, color=INK1)
    # 图例:形状 + 颜色双编码
    h = [plt.Line2D([], [], marker="o", mec=C_OBJ, mfc="none", ms=11, mew=2.4, ls="none",
                    label="bowl  (the object to move)"),
         plt.Line2D([], [], marker="s", mec=C_DEST, mfc="none", ms=11, mew=2.4, ls="none",
                    label="plate  (the destination)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=11,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Attention on the noun tokens, across layers (rows) and across the trajectory "
                 "(columns)\n"
                 "Object markers are re-projected per frame - the bowl is lifted after env step 55 "
                 "and sits on the plate by step 85. Layers 8-12 lock onto the PLATE.",
                 fontsize=12.5, color=INK1)
    fig.tight_layout(rect=[0, 0.035, 1, 0.93])
    fig.savefig(OUT / "fig_attn_layers_frames.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)
    print("[written] fig_attn_layers_frames.png")

    # ============================================ 图 2:定量 对象 vs 目的地
    # ⚠️ 帧 14-15 碗已经放到盘子上,两个 3×3 窗口重叠(切比雪夫距离 1.5 / 0.5 ≤ 2)⇒ 剔除,
    #    否则"碗上的注意力"和"盘子上的注意力"测的是同一块地方。
    cheb = np.maximum(np.abs(bowl_rc[:, 0] - plate_rc[:, 0]), np.abs(bowl_rc[:, 1] - plate_rc[:, 1]))
    keep = np.where(cheb > 2 * WIN)[0]
    dropped = [int(t) for t in range(T) if t not in keep]
    lift = np.array([flats[t][BOWL_QPOS][2] for t in range(T)])
    pre = np.array([t for t in keep if lift[t] < lift[0] + 0.005])     # 碗还在桌上
    mid = np.array([t for t in keep if lift[t] >= lift[0] + 0.005])    # 碗已离地
    print(f"[相位] 剔除窗口重叠帧 {dropped};抓取前 {pre.tolist()};搬运中 {mid.tolist()}")

    def masses(rng):
        o, d = [], []
        for l in range(L):
            sal = renorm_noun(A[:, l])
            o.append(np.mean([win_mass(sal[t], *bowl_rc[t]) for t in rng]))
            d.append(np.mean([win_mass(sal[t], *plate_rc[t]) for t in rng]))
        return np.array(o), np.array(d)

    base = 9 / 256.0     # 3×3 窗口占全图的面积比 = 均匀分布下的期望
    panels = [(pre, "Before the grasp  (bowl still on the table)"),
              (mid, "While carrying  (bowl lifted off the table)")]
    fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.6), facecolor=SURFACE, sharey=True)
    wins = []
    for a, (rng, ttl) in zip(ax, panels):
        yo, yd = masses(rng)
        wins.append(int((yd > yo).sum()))
        a.plot(range(L), yo, marker="o", ms=6, lw=2, color=C_OBJ,
               label="on the bowl  (the object to move)")
        a.plot(range(L), yd, marker="s", ms=6, lw=2, color=C_DEST,
               label="on the plate  (the destination)")
        a.axhline(base, color=INK3, ls="--", lw=1.4, label="uniform attention (a 3x3 of 16x16)")
        a.set_xlabel("transformer layer", color=INK2)
        a.set_title(f"{ttl}\ndestination higher in {int((yd > yo).sum())} of {L} layers",
                    fontsize=11, color=INK1)
        a.set_xticks(range(0, L, 2))
        a.tick_params(colors=INK2)
        a.grid(alpha=0.25)
        a.legend(fontsize=9, labelcolor=INK2, frameon=False)
        for sp_ in a.spines.values():
            sp_.set_color(INK3)
        a.set_facecolor(SURFACE)
    ax[0].set_ylabel("share of attention mass in a 3x3 window", color=INK2)
    yo0, yd0 = masses(pre)
    pk = int(np.argmax(yd0))
    fig.suptitle("Object or destination?   The DESTINATION dominates the middle layers - "
                 f"the plate peaks at layer {pk} with {yd0[pk]/base:.0f}x uniform attention,\n"
                 "while the bowl never exceeds ~5x. The bowl only leads in the first two and "
                 "the last few layers.", fontsize=12, color=INK1)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "fig_attn_object_vs_dest.png", dpi=115, facecolor=SURFACE)
    plt.close(fig)
    print("[written] fig_attn_object_vs_dest.png")
    print(f"  均匀水平={base:.4f}  盘子更高的层数:抓取前 {wins[0]}/{L},搬运中 {wins[1]}/{L}")
    print("  层 | 抓取前 碗/盘子 | 搬运中 碗/盘子")
    yo1, yd1 = masses(mid)
    for l in range(L):
        print(f"  {l:3d} | {yo0[l]:.4f} / {yd0[l]:.4f} | {yo1[l]:.4f} / {yd1[l]:.4f}")

    # ============================================ 图 3:合法位置 左右对比
    cov = b4["cov_base"]
    Iavg = tx["Imag_avg"]
    leg = b4["anchor_legal"].astype(bool)
    aw, aidx = b4["anchor_world"], b4["anchor_idx"]
    vis = (cov.sum((2, 3)) > 0).any(1)
    sel = leg & vis
    w = np.where(sel)[0]

    # 跨帧 attention 叠加:对中间层(3..8,零下溢且信息最强)平均后,按 patch 覆盖读分
    S = np.zeros(len(aidx))
    for l in range(3, 9):
        sal = renorm_noun(A[:, l])
        num = np.einsum("mtij,tij->mt", cov, sal)
        den = cov.sum((2, 3))
        S += np.divide(num, den, out=np.zeros_like(num), where=den > 0).sum(1)
    S /= 6.0

    def rank_grid(vals):
        """只对合法位置排名(1=最高),非法位置为 nan。"""
        g = np.full(len(aidx), np.nan)
        order = w[np.argsort(-vals[w])]
        for r, i in enumerate(order):
            g[i] = r + 1
        return g.reshape(6, 6)

    ra, ri = rank_grid(S), rank_grid(Iavg)
    ext = [aw[:, 0].min(), aw[:, 0].max(), aw[:, 1].min(), aw[:, 1].max()]
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.4), facecolor=SURFACE)
    for a, (g, vals, ttl) in zip(ax, [
            (ra, S, "Ranked by ATTENTION  (before placing anything)"),
            (ri, Iavg, "Ranked by measured INFLUENCE  (ground truth)")]):
        im = a.imshow(g, origin="lower", extent=ext, aspect="auto",
                      cmap=CMAP_SEQ.reversed(), vmin=1, vmax=len(w))
        for i in range(len(aidx)):
            x, y = aw[i][0], aw[i][1]
            if not sel[i]:
                a.plot(x, y, "x", color=INK3, ms=9, mew=2)
                continue
            rk = int(g.ravel()[i]) if not np.isnan(g.ravel()[i]) else None
            a.annotate(f"{rk}", (x, y), color=INK1, fontsize=9, ha="center", va="center",
                       bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75))
        top = w[np.argmax(vals[w])]
        a.plot(aw[top][0], aw[top][1], "*", color=C_DEST, ms=26, mec=INK1, mew=1.2, zorder=6)
        a.plot(-0.098, -0.009, "o", mec=C_OBJ, mfc="none", ms=15, mew=2.6, zorder=5)
        a.plot(0.062, -0.009, "s", mec=C_DEST, mfc="none", ms=15, mew=2.6, zorder=5)
        a.set_title(ttl, fontsize=11.5, color=INK1)
        a.set_xlabel("world x (m)", color=INK2); a.set_ylabel("world y (m)", color=INK2)
        a.tick_params(colors=INK2)
        cb = fig.colorbar(im, ax=a, shrink=0.85)
        cb.set_label("rank among the 17 placeable positions (1 = best)", color=INK2)
        cb.ax.tick_params(colors=INK2)
    same = int(aidx[w[np.argmax(S[w])]]) == int(aidx[w[np.argmax(Iavg[w])]])
    got = Iavg[w[np.argmax(S[w])]] / Iavg[w[np.argmax(Iavg[w])]]
    h2 = [plt.Line2D([], [], marker="*", color=C_DEST, mec=INK1, ms=17, ls="none",
                     label="each panel's own top choice"),
          plt.Line2D([], [], marker="o", mec=C_OBJ, mfc="none", ms=11, mew=2.2, ls="none",
                     label="bowl"),
          plt.Line2D([], [], marker="s", mec=C_DEST, mfc="none", ms=11, mew=2.2, ls="none",
                     label="plate"),
          plt.Line2D([], [], marker="x", color=INK3, ms=9, mew=2, ls="none",
                     label="excluded - patch would cover an object")]
    fig.legend(handles=h2, loc="lower center", ncol=4, frameon=False, fontsize=10,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Same 17 placeable positions, ranked two ways.   "
                 f"Top choices {'AGREE' if same else 'DISAGREE'} - "
                 f"following attention yields {100*got:.0f}% of the best available influence.\n"
                 "Shown as rank, not raw value: attention score and influence (mm) are different "
                 "units and cannot share a scale.", fontsize=12, color=INK1)
    fig.tight_layout(rect=[0, 0.06, 1, 0.90])
    fig.savefig(OUT / "fig_legal_attn_vs_influence.png", dpi=115, facecolor=SURFACE)
    plt.close(fig)
    print("[written] fig_legal_attn_vs_influence.png")
    print(f"  attention 首选 = #{int(aidx[w[np.argmax(S[w])]])}  "
          f"influence 首选 = #{int(aidx[w[np.argmax(Iavg[w])]])}  "
          f"跟 attention 走拿到 {100*got:.0f}%")


if __name__ == "__main__":
    main()
