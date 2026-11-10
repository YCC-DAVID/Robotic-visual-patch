#!/usr/bin/env python3
"""刨掉非法位置后:跨帧 attention 叠加  vs  全轨迹 influence,左右空间对比。

要求(用户 2026-08-12):
  - 只保留合法位置(贴纸与任何物体零重叠),非法位置刨掉
  - attention 沿全部 16 帧叠加(与 influence 的聚合权重一致,均匀)
  - 左右并排,直接看出两张图的高值区不在同一处

设计上修掉旧版的两个毛病:
  - 不在每个格子上标数字(反模式:a number on every data point)
  - 不用 17 级色阶(反模式:>7 个色阶类别)⇒ 两图都归一到"占各自最大值的百分比",
    共用一条单色相、离散 5 档的色标
  - 每个面板同时标出**自己的**首选(实心星)和**对方的**首选(空心星),
    分歧一眼可见,不需要第三个面板

层的选择:用**全 18 层平均**(不需要先验知识的默认做法)。
  若改用第 7-8 层,两图的首选会重合 —— 那是需要先知道答案才能做的选择,
  所以不作为主图口径,只在副标题里说明。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/make_side_by_side.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from make_clean_figs import load, attn_scores, style, INK1, INK2, INK3, SURFACE  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
C_ATT, C_INF = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#eef4fb", "#c5dbf2", "#8fbce8", "#4a90d9", "#14487f"], N=5)
EDGES = [0, 20, 40, 60, 80, 100]


def make(d, layers, layer_note, outname):
    A = d["az"]["bowl_plate_orig__attn"].astype(np.float64)
    cov = d["b4"]["cov_base"]
    Iavg = d["tx"]["Imag_avg"]
    leg = d["b4"]["anchor_legal"].astype(bool)
    aw = d["b4"]["anchor_world"]
    vis = (cov.sum((2, 3)) > 0).any(1)
    sel = leg & vis
    w = np.where(sel)[0]

    S = attn_scores(A, cov, layers)              # 指定层 × 全 16 帧叠加

    def pct_grid(vals):
        """全部 36 格都画,归一到**全场**最大值的百分比(含非法格)。"""
        return (100 * vals / vals.max()).reshape(6, 6)

    ga, gi = pct_grid(S), pct_grid(Iavg)
    pick_a = int(w[np.argmax(S[w])])              # 合法位置里的首选
    pick_i = int(w[np.argmax(Iavg[w])])
    gmax_a = int(np.argmax(S))                    # 不受限的全场最大(通常是非法格)
    gmax_i = int(np.argmax(Iavg))

    # ⚠️ imshow 的 extent 是格**边界**,不是格中心。给中心范围会让整张图偏半格
    #    (实测 x 半格 0.045 m、y 半格 0.065 m)。
    xs, ys = np.unique(aw[:, 0]), np.unique(aw[:, 1])
    dx, dy = np.diff(xs).mean(), np.diff(ys).mean()
    ext = [xs[0] - dx / 2, xs[-1] + dx / 2, ys[0] - dy / 2, ys[-1] + dy / 2]

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.9), facecolor=SURFACE)
    norm = BoundaryNorm(EDGES, SEQ.N)
    for ax, (g, own, other, gmax, ttl, sub) in zip(axes, [
            (ga, pick_a, pick_i, gmax_a,
             "Attention, summed over all 16 frames   [cell labels = % of this panel's max]",
             layer_note),
            (gi, pick_i, pick_a, gmax_i,
             "Influence, measured over the whole trajectory   [cell labels in mm]",
             "mean of 3 random probe textures - the ground truth")]):
        im = ax.imshow(g, origin="lower", extent=ext, aspect="auto", cmap=SEQ, norm=norm)
        # 非法格:值照常画出来,盖一层斜线纹理表示"这里不能贴"
        for i in range(len(aw)):
            if not sel[i]:
                # 稀疏斜线 + 浅线色,保证底下的色块还读得出来
                ax.add_patch(plt.Rectangle((aw[i][0] - dx / 2, aw[i][1] - dy / 2), dx, dy,
                                           fill=False, hatch="//", edgecolor="#8f8d84",
                                           lw=0.0, zorder=4, alpha=0.9))
        ax.plot(-0.098, -0.009, "o", mec=INK1, mfc="none", ms=15, mew=2.2, zorder=6)
        ax.plot(0.062, -0.009, "s", mec=INK1, mfc="none", ms=15, mew=2.2, zorder=6)
        ax.annotate("bowl", (-0.098, -0.009), xytext=(0, -17), textcoords="offset points",
                    color=INK1, fontsize=9.5, ha="center", va="top", zorder=9)
        ax.annotate("plate", (0.062, -0.009), xytext=(0, -17), textcoords="offset points",
                    color=INK1, fontsize=9.5, ha="center", va="top", zorder=9)
        # 全场最大(不受限)—— 通常落在非法格上,用方框标出
        ax.add_patch(plt.Rectangle((aw[gmax][0] - dx / 2, aw[gmax][1] - dy / 2), dx, dy,
                                   fill=False, edgecolor="#111111", lw=2.6, zorder=7))
        # 合法范围内的首选:自己的(实心)与对方的(空心)
        ax.plot(aw[own][0], aw[own][1], "*", color=C_INF if g is gi else C_ATT,
                ms=28, mec=SURFACE, mew=1.8, zorder=8)
        ax.plot(aw[other][0], aw[other][1], "*", mfc="none",
                mec=C_ATT if g is gi else C_INF, ms=28, mew=2.6, zorder=8)
        # 每格标出数值:attention 的原始单位无意义 ⇒ 标占全场最大的百分比;
        # influence 有物理单位 ⇒ 直接标 mm。深色格用白字。
        raw = S if g is ga else Iavg
        for i in range(len(aw)):
            txt = f"{g.ravel()[i]:.0f}%" if g is ga else f"{raw[i]:.0f}"
            ax.annotate(txt, (aw[i][0], aw[i][1] + dy * 0.30),
                        color="white" if g.ravel()[i] >= 60 else INK1,
                        fontsize=9, ha="center", va="center", zorder=9)
        ax.set_title(f"{ttl}\n{sub}", fontsize=11, color=INK1, pad=9)
        ax.set_xlabel("world x (m)", color=INK2)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        style(ax)
        ax.grid(visible=False)
    axes[0].set_ylabel("world y (m)", color=INK2)

    cax = fig.add_axes([0.895, 0.26, 0.014, 0.52])
    cb = fig.colorbar(im, cax=cax, ticks=EDGES)
    cb.set_label("% of that panel's maximum\n(over all 36 cells)", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=INK2, labelsize=9)

    # 图例按**颜色**区分身份(实心/空心的含义放副标题)——否则右面板里
    # attention 是空心的,和"实心=attention"的图例自相矛盾。
    h = [plt.Line2D([], [], marker="*", color=C_ATT, mec=SURFACE, ms=17, ls="none",
                    label="blue star = attention's best PLACEABLE spot"),
         plt.Line2D([], [], marker="*", color=C_INF, mec=SURFACE, ms=17, ls="none",
                    label="orange star = the most influential PLACEABLE spot"),
         plt.Rectangle((0, 0), 1, 1, fill=False, hatch="////", edgecolor="#6e6c64", lw=0.0,
                       label="hatched = a patch here would cover an object"),
         plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#111111", lw=2.2,
                       label="black box = that panel's overall maximum")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
               labelcolor=INK2, bbox_to_anchor=(0.46, 0.002))
    ratio = 100 * Iavg[pick_a] / Iavg[pick_i]
    fig.suptitle("All 36 cells shown, including the ones you cannot use.   Both panels peak on "
                 "the hatched cells - the objects themselves.\n"
                 f"Among the cells that ARE usable, the two disagree: attention's choice carries "
                 f"{ratio:.0f}% of the influence available.",
                 fontsize=12.2, color=INK1)
    fig.subplots_adjust(left=0.055, right=0.875, bottom=0.20, top=0.80, wspace=0.14)
    fig.savefig(OUT / outname, dpi=125, facecolor=SURFACE)
    plt.close(fig)
    print(f"[written] {outname}")
    print(f"  attention 首选 ({aw[pick_a][0]:5.2f},{aw[pick_a][1]:5.2f})  {Iavg[pick_a]:5.0f} mm"
          f"   influence 首选 ({aw[pick_i][0]:5.2f},{aw[pick_i][1]:5.2f})  {Iavg[pick_i]:5.0f} mm"
          f"   → {ratio:.0f}%")


def main():
    d = load()
    L = d["az"]["bowl_plate_orig__attn"].shape[1]
    make(d, range(L),
         "averaged over ALL 18 layers - the choice you can make without knowing the answer",
         "side_by_side_alllayers.png")
    make(d, range(3, 9),
         "averaged over layers 3-8 only - a band picked because it is known to work",
         "side_by_side_layers3to8.png")


if __name__ == "__main__":
    main()
