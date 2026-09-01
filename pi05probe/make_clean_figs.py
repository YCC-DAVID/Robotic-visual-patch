#!/usr/bin/env python3
"""重做汇报图:6 张**单面板**图,每张只回答一个问题。纯后处理,零 GPU。

为什么重做(旧图违反的反模式)
--------------------------
  ❌ 每个数据点都标数字 —— 旧的左右对比图在 17 个格子上全标了名次
  ❌ 超过 ~7 个色阶类别 —— 用颜色编码 1..17 的名次,相邻类别根本分不开
  ❌ 用 25 格小图讲一个结论 —— 该用"强调一个、其余压灰",不是堆面板
  ❌ 两个色标并排 —— 读者要在两个 colorbar 之间来回换算

产出
----
  clean1_rank_slope.png       attention 名次 → influence 名次(斜线图,名次用位置而非颜色)
  clean2_where.png            那两个位置在真实画面里的哪儿(一张图,两个标记)
  clean3_object_vs_dest.png   注意力在操作对象 vs 目的地(逐层,两条序列)
  clean4_layer_dependence.png 读第几层决定成败(逐层拿到多少影响力)
  clean5_influence_map.png    影响力地图(单色相,离散 6 档)
  clean6_ablation.png         消融:换任何方法口径,结论都不变(点线图 + 随机基线)

配色(dataviz 规范参考配色固定槽,已算过分离度)
    slot1 blue  #2a78d6 = attention / 操作对象
    slot2 orange #eb6834 = influence 真值 / 目的地
    其余一律压成中性灰,靠"强调 + 压灰"而不是多色相
    正常视觉 ΔE 33.6 / 绿色盲 31.7 / 红色盲 24.7

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/make_clean_figs.py
"""
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NOUN_ROWS = [3, 6]
BOWL_QPOS, PLATE_QPOS = slice(10, 13), slice(31, 34)
MID_LAYERS = range(3, 9)          # 零下溢且信息最强;不挑单层,避免择优质疑

C_ATT = "#2a78d6"                 # attention
C_INF = "#eb6834"                 # influence(真值)
MUTED = "#b5b3aa"
INK1, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE = "#fcfcfb"
# 单色相顺序阶,离散 6 档(反模式:>7 个色阶类别)
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#eef4fb", "#c5dbf2", "#8fbce8", "#5599db", "#2a78d6", "#14487f"], N=6)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5df", lw=0.8, ls="-")       # 实线细网格,不用虚线
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, length=3)


def world_to_cell(Pw, K, E):
    """只翻列、不翻行 —— 实测定出(四种组合误差 8.98/12.77/0.53/8.10 格)。"""
    Pc = np.linalg.inv(E) @ np.array([Pw[0], Pw[1], Pw[2], 1.0])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = Pc[2]
    u, v = cx + fx * Pc[0] / z, cy + fy * Pc[1] / z
    return ((v + 0.5) * 224 / 256 - 0.5) / 14 - 0.5, (((255 - u) + 0.5) * 224 / 256 - 0.5) / 14 - 0.5


def win_mass(sal, r, c, win=1):
    ri, ci = int(round(r)), int(round(c))
    r0, r1 = max(0, ri - win), min(16, ri + win + 1)
    c0, c1 = max(0, ci - win), min(16, ci + win + 1)
    return float(sal[r0:r1, c0:c1].sum() / max(sal.sum(), 1e-12))


def load():
    d = {}
    d["az"] = np.load(OUT / "attn_traj_put_the_bowl_on_the_plate.npz", allow_pickle=True)
    d["tz"] = np.load(OUT / "traj_put_the_bowl_on_the_plate.npz", allow_pickle=False)
    d["sf"] = np.load(OUT / "shared_frame.npz", allow_pickle=True)
    d["b4"] = np.load(OUT / "b4_attn_vs_influence.npz", allow_pickle=True)
    d["tx"] = np.load(OUT / "texture_axis.npz", allow_pickle=True)
    d["ob"] = np.load(OUT / "s2_scan_obs.npz", allow_pickle=True)
    d["K"] = d["sf"]["put_the_bowl_on_the_plate__K_agentview"]
    d["E"] = d["sf"]["put_the_bowl_on_the_plate__E_agentview"]
    return d


def renorm_noun(A_layer):
    X = A_layer / np.clip(A_layer.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)
    return X[:, NOUN_ROWS, 0].sum(1)


def attn_scores(A, cov, layers, mode="renorm", pool="noun", view=0, frames=None):
    """每个候选位置的 attention 分数:patch 覆盖的格上加权平均,再沿帧求和。"""
    T = A.shape[0]
    fr = range(T) if frames is None else frames
    S = np.zeros(cov.shape[0])
    for l in layers:
        X = A[:, l]
        if mode == "renorm":
            X = X / np.clip(X.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)
        if pool == "max":
            sal = X[:, :, view].max(1)
        elif pool == "sum":
            sal = X[:, :, view].sum(1)
        else:
            sal = X[:, NOUN_ROWS, view].sum(1)
        num = np.einsum("mtij,tij->mt", cov, sal)
        den = cov.sum((2, 3))
        per = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        S += per[:, list(fr)].sum(1)
    return S / max(len(list(layers)), 1)


def main():
    d = load()
    A = d["az"]["bowl_plate_orig__attn"].astype(np.float64)
    T, L = A.shape[0], A.shape[1]
    cov = d["b4"]["cov_base"]
    Iavg = d["tx"]["Imag_avg"]
    leg = d["b4"]["anchor_legal"].astype(bool)
    aw, aidx = d["b4"]["anchor_world"], d["b4"]["anchor_idx"]
    vis = (cov.sum((2, 3)) > 0).any(1)
    sel = leg & vis
    w = np.where(sel)[0]
    n = len(w)
    best_i = w[np.argmax(Iavg[w])]

    S_mid = attn_scores(A, cov, MID_LAYERS)
    att_i = w[np.argmax(S_mid[w])]

    def lab(i):
        return f"({aw[i][0]:.2f}, {aw[i][1]:.2f})"

    # ============================================== 图 1:名次斜线图,三种读法并排
    # 只画"最有利"那一档会让差异看起来很小 —— 必须把"没有先验知识时会怎样"一起画。
    def spearman(a, b):
        def rk(x):
            x = np.asarray(x, np.float64)
            o = np.argsort(x, kind="stable")
            r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=np.float64)
            _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
            s = np.zeros(len(cnt)); np.add.at(s, inv, r)
            return (s / cnt)[inv]
        ra_, rb_ = rk(a), rk(b)
        ra_ = ra_ - ra_.mean(); rb_ = rb_ - rb_.mean()
        return float((ra_ * rb_).sum() / np.sqrt((ra_ * ra_).sum() * (rb_ * rb_).sum()))

    ri = {int(i): r + 1 for r, i in enumerate(w[np.argsort(-Iavg[w])])}
    CFGS = [(S_mid, "read layers 3-8", "hand-picked: the most favourable band"),
            (attn_scores(A, cov, range(L)), "average all 18 layers",
             "the assumption-free default"),
            (attn_scores(A, cov, [0]), "let a rule pick the layer",
             "sharpest / most concentrated / most stable\nall three choose layer 0")]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 6.2), facecolor=SURFACE, sharey=True)
    for ax, (S, ttl, sub) in zip(axes, CFGS):
        ra = {int(i): r + 1 for r, i in enumerate(w[np.argsort(-S[w])])}
        pick = int(w[np.argmax(S[w])])
        for i in w:
            i = int(i)
            hl = i in (int(best_i), pick)
            col = C_INF if i == int(best_i) else (C_ATT if i == pick else MUTED)
            ax.plot([0, 1], [ra[i], ri[i]], "-", color=col, lw=2.8 if hl else 1.1,
                    alpha=1.0 if hl else 0.45, zorder=3 if hl else 1, solid_capstyle="round")
            ax.plot([0, 1], [ra[i], ri[i]], "o", color=col, ms=8 if hl else 4.5,
                    mec=SURFACE, mew=1.5, zorder=3 if hl else 1)
        got_ = Iavg[pick] / Iavg[best_i]
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(n + 0.7, 0.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ATTENTION", "INFLUENCE"], fontsize=10.5)
        style(ax)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{ttl}\n{sub}\n"
                     f"rank correlation {spearman(S[w], Iavg[w]):+.2f}   "
                     f"top pick keeps {100*got_:.0f}%",
                     fontsize=11, color=INK1, pad=10)
    axes[0].set_ylabel("rank among the 17 placeable positions  (1 = best)", color=INK2)
    h1 = [plt.Line2D([], [], color=C_INF, lw=2.8, label="the truly most influential spot"),
          plt.Line2D([], [], color=C_ATT, lw=2.8, label="attention's own top pick"),
          plt.Line2D([], [], color=MUTED, lw=1.4, label="the other 15 positions")]
    fig.legend(handles=h1, loc="lower center", ncol=3, frameon=False, fontsize=10.5,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("The same comparison under three ways of reading attention.   "
                 "Only the hand-picked mid band comes close;\n"
                 "with no prior knowledge of which layers matter, the two rankings barely relate.",
                 fontsize=12.5, color=INK1)
    fig.tight_layout(rect=[0, 0.05, 1, 0.90])
    fig.savefig(OUT / "clean1_rank_slope.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print("[written] clean1_rank_slope.png")

    # ============================================== 图 2:那两个位置在画面哪儿
    frame = 4
    img = d["ob"]["clean_img224"][frame]
    fig, ax = plt.subplots(figsize=(6.6, 6.6), facecolor=SURFACE)
    ax.imshow(img)
    for i, col, name in [(int(best_i), C_INF, "most influential"),
                         (int(att_i), C_ATT, "attention's pick")]:
        r, c = world_to_cell((aw[i][0], aw[i][1], 0.9015), d["K"], d["E"])
        x, y = (c + .5) * 14, (r + .5) * 14
        ax.add_patch(plt.Circle((x, y), 26, fill=False, ec=SURFACE, lw=5, zorder=4))
        ax.add_patch(plt.Circle((x, y), 26, fill=False, ec=col, lw=3, zorder=5))
        ax.annotate(f"{name}\n{lab(i)}   {Iavg[i]:.0f} mm",
                    (x, y), xytext=(0, -42), textcoords="offset points",
                    color=col, fontsize=10.5, ha="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec="none", alpha=0.88))
    for pw, mk, nm in [(d["tz"]["f004__flatten"][BOWL_QPOS], "o", "bowl"),
                       (d["tz"]["f004__flatten"][PLATE_QPOS], "s", "plate")]:
        r, c = world_to_cell(pw, d["K"], d["E"])
        ax.plot((c + .5) * 14, (r + .5) * 14, mk, mec=INK1, mfc="none", ms=17, mew=2.2, zorder=6)
        ax.annotate(nm, ((c + .5) * 14, (r + .5) * 14), xytext=(15, -4),
                    textcoords="offset points", color=INK1, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("The same two spots, in the image the model actually receives\n"
                 "attention sends the patch to bare table in the near field",
                 fontsize=12.5, color=INK1, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "clean2_where.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("[written] clean2_where.png")

    # ============================================== 图 3:对象 vs 目的地
    flats = np.stack([d["tz"][f"f{k:03d}__flatten"] for k in range(T)])
    brc = np.array([world_to_cell(flats[t][BOWL_QPOS], d["K"], d["E"]) for t in range(T)])
    prc = np.array([world_to_cell(flats[t][PLATE_QPOS], d["K"], d["E"]) for t in range(T)])
    lift = flats[:, BOWL_QPOS][:, 2]
    pre = [t for t in range(T) if lift[t] < lift[0] + 0.005]     # 碗还在桌上,两窗口不重叠
    yo, yd = [], []
    for l in range(L):
        sal = renorm_noun(A[:, l])
        yo.append(np.mean([win_mass(sal[t], *brc[t]) for t in pre]))
        yd.append(np.mean([win_mass(sal[t], *prc[t]) for t in pre]))
    yo, yd = np.array(yo), np.array(yd)
    base = 9 / 256
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=SURFACE)
    ax.plot(range(L), yd / base, "-s", color=C_INF, lw=2.4, ms=6,
            label="on the plate  (the destination)")
    ax.plot(range(L), yo / base, "-o", color=C_ATT, lw=2.4, ms=6,
            label="on the bowl  (the object to be moved)")
    ax.axhline(1.0, color=INK3, lw=1.2)
    ax.annotate("uniform attention", (L - 0.4, 1.0), xytext=(0, 6),
                textcoords="offset points", color=INK2, fontsize=9, ha="right")
    pk = int(np.argmax(yd))
    ax.annotate(f"{yd[pk]/base:.0f}x", (pk, yd[pk] / base), xytext=(0, 9),
                textcoords="offset points", color=C_INF, fontsize=11,
                ha="center", fontweight="bold")
    pk2 = int(np.argmax(yo))
    ax.annotate(f"{yo[pk2]/base:.0f}x", (pk2, yo[pk2] / base), xytext=(0, 9),
                textcoords="offset points", color=C_ATT, fontsize=11,
                ha="center", fontweight="bold")
    ax.set_xlabel("transformer layer", color=INK2)
    # y 轴含义:物体处 3×3 窗口内的注意力占全图的比例,再除以均匀分布下的期望 9/256。
    # 1 = 和随便撒一样;21 = 密度是随机的 21 倍。
    ax.set_ylabel("attention density on the object\n"
                  "(share in a 3x3 window / share if spread evenly)", color=INK2)
    ax.set_xticks(range(0, L, 2))
    style(ax)
    ax.legend(fontsize=10, labelcolor=INK2, frameon=False, loc="upper left")
    ax.set_title("Mid-network attention tracks the DESTINATION, not the object\n"
                 "measured before the grasp, with object positions re-projected per frame",
                 fontsize=12.5, color=INK1, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "clean3_object_vs_dest.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("[written] clean3_object_vs_dest.png")

    # ============================================== 图 4:读第几层决定成败
    ys = []
    for l in range(L):
        S = attn_scores(A, cov, [l])
        p = w[np.argmax(S[w])]
        ys.append(100 * Iavg[p] / Iavg[best_i])
    ys = np.array(ys)
    hit = ys >= 99.5
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=SURFACE)
    ax.plot(range(L), ys, "-", color=MUTED, lw=2, zorder=1)
    ax.plot(np.where(~hit)[0], ys[~hit], "o", color=MUTED, ms=7, mec=SURFACE, mew=1.5, zorder=2)
    ax.plot(np.where(hit)[0], ys[hit], "o", color=C_INF, ms=11, mec=SURFACE, mew=1.8, zorder=3)
    rnd = 100 * Iavg[w].mean() / Iavg[best_i]
    ax.axhline(rnd, color=INK3, lw=1.2)
    ax.annotate(f"picking at random: {rnd:.0f}%", (L - 0.3, rnd), xytext=(0, 7),
                textcoords="offset points", color=INK2, fontsize=9.5, ha="right")
    hl = np.where(hit)[0]
    ax.annotate(f"layers {hl[0]} and {hl[-1]}" if len(hl) > 1 else f"layer {hl[0]}",
                (float(hl.mean()), 100.0), xytext=(0, -26), textcoords="offset points",
                color=C_INF, fontsize=11, ha="center", fontweight="bold")
    ax.set_xlabel("which layer the attention is read from", color=INK2)
    ax.set_ylabel("influence obtained  (% of the best placeable spot)", color=INK2)
    ax.set_xticks(range(0, L, 2))
    ax.set_ylim(-4, 108)
    style(ax)
    ax.set_title("Whether attention succeeds is decided entirely by which layer you read\n"
                 f"only {int(hit.sum())} of {L} layers find the best spot - and you cannot know "
                 "which without the answer", fontsize=12.5, color=INK1, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "clean4_layer_dependence.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("[written] clean4_layer_dependence.png")

    # ============================================== 图 5:影响力地图
    g = Iavg.reshape(6, 6)
    edges = np.array([0, 20, 40, 70, 110, 180, 320])
    fig, ax = plt.subplots(figsize=(7.4, 6.2), facecolor=SURFACE)
    ext = [aw[:, 0].min(), aw[:, 0].max(), aw[:, 1].min(), aw[:, 1].max()]
    im = ax.imshow(g, origin="lower", extent=ext, aspect="auto", cmap=SEQ,
                   norm=BoundaryNorm(edges, SEQ.N))
    for i in range(len(aidx)):
        if not sel[i]:
            ax.plot(aw[i][0], aw[i][1], marker=(4, 2, 45), color="white", ms=13, mew=2.4,
                    ls="none", zorder=4)
    ax.plot(-0.098, -0.009, "o", mec=INK1, mfc="none", ms=16, mew=2.4, zorder=5)
    ax.plot(0.062, -0.009, "s", mec=INK1, mfc="none", ms=16, mew=2.4, zorder=5)
    ax.annotate("bowl", (-0.098, -0.009), xytext=(14, -4), textcoords="offset points",
                color=INK1, fontsize=10)
    ax.annotate("plate", (0.062, -0.009), xytext=(14, -4), textcoords="offset points",
                color=INK1, fontsize=10)
    ax.plot(aw[best_i][0], aw[best_i][1], "*", color=C_INF, ms=26, mec=SURFACE, mew=1.6, zorder=6)
    ax.annotate("best placeable", (aw[best_i][0], aw[best_i][1]), xytext=(0, 20),
                textcoords="offset points", color=C_INF, fontsize=10.5, ha="center",
                fontweight="bold")
    ax.set_xlabel("world x (m)", color=INK2); ax.set_ylabel("world y (m)", color=INK2)
    style(ax)
    ax.grid(visible=False)
    cb = fig.colorbar(im, ax=ax, shrink=0.86, spacing="proportional")
    cb.set_label("influence  (mm of action deviation)", color=INK2)
    cb.ax.tick_params(colors=INK2)
    ax.set_title("Influence is concentrated on the objects themselves\n"
                 "white crosses = a patch there would cover an object, so it is not placeable",
                 fontsize=12.5, color=INK1, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "clean5_influence_map.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("[written] clean5_influence_map.png")

    # ============================================== 图 6:消融(全部来自已有数据)
    variants = []

    def add(name, S):
        p = w[np.argmax(S[w])]
        variants.append((name, 100 * Iavg[p] / Iavg[best_i], int(aidx[p])))

    for pool in ("noun", "max", "sum"):
        add(f"pooling = {pool}", attn_scores(A, cov, MID_LAYERS, pool=pool))
    for mode in ("renorm", "raw"):
        add(f"normalise = {mode}", attn_scores(A, cov, MID_LAYERS, mode=mode))
    add("layers 3-8 (mid band)", attn_scores(A, cov, MID_LAYERS))
    add("all 18 layers", attn_scores(A, cov, range(L)))
    add("layers 0-2 (early)", attn_scores(A, cov, range(0, 3)))
    add("layers 13-17 (late)", attn_scores(A, cov, range(13, L)))
    add("wrist view", attn_scores(A, cov, MID_LAYERS, view=1))
    add("last frame only (POAP)", attn_scores(A, cov, MID_LAYERS, frames=[T - 1]))
    add("first frame only", attn_scores(A, cov, MID_LAYERS, frames=[0]))
    add("best possible layer (7)", attn_scores(A, cov, [7]))

    # ---- 关键一组:用**不需要知道答案**的准则去挑层。若某个准则能挑中好层,
    #      那 attention+该准则就是一个成立的方法;实测三个准则全部选中第 0 层。
    def rn(X):
        return X / np.clip(X.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)

    ent, conc, stab = [], [], []
    for l in range(L):
        s = rn(A[:, l])[:, NOUN_ROWS, 0].sum(1)
        p = s / np.clip(s.sum((1, 2), keepdims=True), 1e-12, None)
        ent.append(float(np.mean(-(p * np.log(p + 1e-12)).sum((1, 2)))))
        conc.append(float(np.mean(s.max((1, 2)) / np.clip(s.mean((1, 2)), 1e-12, None))))
        f = s.reshape(T, -1)
        f = f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-12, None)
        stab.append(float(np.mean([f[t] @ f[t + 1] for t in range(T - 1)])))
    for nm, arr, how in [("auto-pick: lowest entropy", ent, "min"),
                         ("auto-pick: sharpest peak", conc, "max"),
                         ("auto-pick: most stable", stab, "max")]:
        l = int(np.argmin(arr) if how == "min" else np.argmax(arr))
        add(f"{nm}  (chose layer {l})", attn_scores(A, cov, [l]))

    variants.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9.6, 6.8), facecolor=SURFACE)
    ys_ = np.arange(len(variants))
    vals = np.array([v[1] for v in variants])
    ax.hlines(ys_, 0, vals, color="#e0dfd8", lw=2.2, zorder=1)
    ax.plot(vals, ys_, "o", color=C_ATT, ms=9, mec=SURFACE, mew=1.6, zorder=3)
    ax.axvline(rnd, color=INK3, lw=1.4, zorder=2)
    ax.annotate(f"picking at random\n{rnd:.0f}%", (rnd, len(variants) - 0.4),
                xytext=(8, 0), textcoords="offset points", color=INK2, fontsize=9.5,
                va="top")
    ax.axvline(100, color=C_INF, lw=1.4, zorder=2)
    ax.annotate("perfect\n100%", (100, len(variants) - 0.4), xytext=(-8, 0),
                textcoords="offset points", color=C_INF, fontsize=9.5, va="top", ha="right")
    ax.set_yticks(ys_)
    ax.set_yticklabels([v[0] for v in variants], fontsize=10)
    ax.set_xlabel("influence obtained by attention's top pick  (% of the best placeable spot)",
                  color=INK2)
    ax.set_xlim(0, 112)
    style(ax)
    ax.grid(axis="y", visible=False)
    nb = int((vals > rnd).sum())
    ax.set_title("Ablation: the layer you read decides everything\n"
                 f"{len(variants)-nb} of {len(variants)} variants land at or below random, and "
                 "all three automatic layer-picking rules chose layer 0",
                 fontsize=11.8, color=INK1, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "clean6_ablation.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("[written] clean6_ablation.png")
    print(f"\n  随机基线 = {rnd:.0f}%   完美 = 100%")
    print("  消融明细(按拿到的影响力排序):")
    for nm, v, pk_ in variants:
        print(f"    {nm:26s} {v:6.1f}%   选中 #{pk_}")


    # ====================================== 图 7:为什么合法位置只能读到 attention 的尾巴
    # 用户的洞察:attention 的峰值本来就落在物体上(或根本不在任何候选能覆盖的地方),
    # 所以限定在合法位置后,排序读的是**残余**的注意力,不是峰值。
    covm = (cov > 0.01).any(1)                                  # (M,16,16)
    sal_mid = np.mean([renorm_noun(A[:, l]) for l in MID_LAYERS], axis=0).mean(0)
    m_leg = np.zeros((16, 16), bool)
    m_ill = np.zeros((16, 16), bool)
    for i in w:
        m_leg |= covm[i]
    for i in np.where((~leg) & vis)[0]:
        m_ill |= covm[i]
    only_leg = m_leg & ~m_ill
    reach_ill = m_ill
    none = ~(m_leg | m_ill)
    tot = sal_mid.sum()
    shares = {"legal only": 100 * sal_mid[only_leg].sum() / tot,
              "illegal (covers an object)": 100 * sal_mid[reach_ill & ~only_leg].sum() / tot,
              "no candidate reaches it": 100 * sal_mid[none].sum() / tot}
    pk = np.unravel_index(int(sal_mid.argmax()), sal_mid.shape)

    fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.6), facecolor=SURFACE)
    ax[0].imshow(d["ob"]["clean_img224"][4])
    ax[0].imshow(np.kron(sal_mid, np.ones((14, 14))), cmap=SEQ, alpha=0.62)
    ax[0].contour(np.kron(only_leg.astype(float), np.ones((14, 14))), levels=[0.5],
                  colors=[C_ATT], linewidths=2.4)
    ax[0].plot((pk[1] + .5) * 14, (pk[0] + .5) * 14, "x", color=C_INF, ms=18, mew=4, zorder=5)
    ax[0].annotate("attention's peak", ((pk[1] + .5) * 14, (pk[0] + .5) * 14),
                   xytext=(12, -18), textcoords="offset points", color=C_INF,
                   fontsize=11, fontweight="bold")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[0].set_title("Blue outline = the only cells a legal patch can reach.\n"
                    "The peak is outside it.", fontsize=11.5, color=INK1)

    names = list(shares.keys())
    vals = [shares[k] for k in names]
    cols = [C_ATT, MUTED, "#dcdbd4"]
    ax[1].barh(range(3), vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax[1].annotate(f"{v:.1f}%", (v, i), xytext=(6, 0), textcoords="offset points",
                       color=INK1, fontsize=11, va="center", fontweight="bold")
    ax[1].set_yticks(range(3))
    ax[1].set_yticklabels(names, fontsize=10.5)
    ax[1].set_xlabel("share of the attention mass", color=INK2)
    ax[1].set_xlim(0, 70)
    style(ax[1])
    ax[1].grid(axis="y", visible=False)
    pl = 100 * np.nanmax([sal_mid[covm[i]].max() for i in w]) / sal_mid[pk]
    ax[1].set_title(f"A legal patch can touch at most {pl:.0f}% of the peak value\n"
                    "so the ranking among legal spots reads the tail, not the peak",
                    fontsize=11.5, color=INK1)
    fig.suptitle("Why attention cannot rank the placeable positions well: "
                 "its signal lives where you cannot put a patch",
                 fontsize=12.8, color=INK1)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUT / "clean7_unreachable.png", dpi=125, facecolor=SURFACE)
    plt.close(fig)
    print("[written] clean7_unreachable.png")
    print(f"  合法位置能触到的峰值 = 全局峰值的 {pl:.1f}%;"
          f"  质量分布 {', '.join(f'{k} {v:.1f}%' for k, v in shares.items())}")


if __name__ == "__main__":
    main()
