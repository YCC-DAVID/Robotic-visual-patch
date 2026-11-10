#!/usr/bin/env python3
"""按**腕部视角**重做核心分析图。纯后处理。

为什么要重做
----------
消融实测:主视角 value 置 0,任务仍 15/15 成功(只慢 19%);腕部 value ×0.1 → 0/15。
⇒ **承重的是腕部相机**,而之前所有 attention 分析读的是主视角(那个可以整个扔掉的)。

产出
----
    wrist1_side_by_side.png   腕部 attention vs influence,全 36 格,非法格加斜线
    wrist2_layer.png          逐层首选拿到多少(腕部 vs 主视角 对照)
    wrist3_ablation.png       消融:换口径/自动选层(腕部)
    wrist4_reachability.png   合法位置能触到腕部注意力的多少

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/make_wrist_figs.py
"""
import collections
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NOUN = [3, 6]
MID = range(3, 9)
C_ATT, C_INF, MUTED = "#2a78d6", "#eb6834", "#b5b3aa"
INK1, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#eef4fb", "#c5dbf2", "#8fbce8", "#4a90d9", "#14487f"], N=5)
EDGES = [0, 20, 40, 60, 80, 100]


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5df", lw=0.8, ls="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, length=3)


def load():
    az = np.load(OUT / "attn_traj_put_the_bowl_on_the_plate.npz", allow_pickle=True)
    ob = np.load(OUT / "s2_scan_obs.npz", allow_pickle=True)
    b4 = np.load(OUT / "b4_attn_vs_influence.npz", allow_pickle=True)
    tx = np.load(OUT / "texture_axis.npz", allow_pickle=True)
    A = az["bowl_plate_orig__attn"].astype(np.float64)
    T = A.shape[0]

    def cover(ck, pk):
        c = ob[ck].astype(np.int16); p = ob[pk].astype(np.int16)
        m = (np.abs(p - c[None]).max(-1) > 10)
        return m.reshape(m.shape[0], T, 16, 14, 16, 14).mean((3, 5))

    cov = {0: cover("clean_img224", "patched_img224"),
           1: cover("clean_wrist224", "patched_wrist224")}
    return A, cov, tx["Imag_avg"], b4["anchor_legal"].astype(bool), b4["anchor_world"], b4["anchor_idx"]


def main():
    A, cov, I, leg, aw, aidx = load()
    L = A.shape[1]
    vis = (cov[0].sum((2, 3)) > 0).any(1)
    w = np.where(leg & vis)[0]
    best = w[np.argmax(I[w])]

    def score(view, layers, mode="renorm", pool="noun"):
        S = np.zeros(cov[view].shape[0])
        for l in layers:
            X = A[:, l]
            if mode == "renorm":
                X = X / np.clip(X.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)
            sal = {"max": X[:, :, view].max(1), "sum": X[:, :, view].sum(1),
                   "noun": X[:, NOUN, view].sum(1)}[pool]
            num = np.einsum("mtij,tij->mt", cov[view], sal)
            den = cov[view].sum((2, 3))
            S += np.divide(num, den, out=np.zeros_like(num), where=den > 0).sum(1)
        return S

    xs, ys_ = np.unique(aw[:, 0]), np.unique(aw[:, 1])
    dx, dy = np.diff(xs).mean(), np.diff(ys_).mean()
    ext = [xs[0] - dx / 2, xs[-1] + dx / 2, ys_[0] - dy / 2, ys_[-1] + dy / 2]

    # ---------------- 图 1:左右对比(腕部 attention vs influence)
    S = score(1, MID)
    pa, pi_ = int(w[np.argmax(S[w])]), int(best)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.9), facecolor=SURFACE)
    norm = BoundaryNorm(EDGES, SEQ.N)
    for ax, (v, own, other, ttl, sub, unit) in zip(axes, [
            (S, pa, pi_, "WRIST-view attention, summed over 16 frames",
             "layers 3-8; the wrist is the view the policy actually needs", "%"),
            (I, pi_, pa, "Influence, measured over the whole trajectory",
             "mean of 3 random probe textures", "mm")]):
        g = (100 * v / v.max()).reshape(6, 6)
        im = ax.imshow(g, origin="lower", extent=ext, aspect="auto", cmap=SEQ, norm=norm)
        for i in range(len(aw)):
            if not (leg & vis)[i]:
                ax.add_patch(plt.Rectangle((aw[i][0] - dx / 2, aw[i][1] - dy / 2), dx, dy,
                                           fill=False, hatch="//", edgecolor="#8f8d84",
                                           lw=0.0, zorder=4, alpha=0.9))
            txt = f"{g.ravel()[i]:.0f}%" if unit == "%" else f"{v[i]:.0f}"
            ax.annotate(txt, (aw[i][0], aw[i][1] + dy * 0.30),
                        color="white" if g.ravel()[i] >= 60 else INK1,
                        fontsize=9, ha="center", va="center", zorder=9)
        gm = int(np.argmax(v))
        ax.add_patch(plt.Rectangle((aw[gm][0] - dx / 2, aw[gm][1] - dy / 2), dx, dy,
                                   fill=False, edgecolor="#111111", lw=2.6, zorder=7))
        ax.plot(-0.098, -0.009, "o", mec=INK1, mfc="none", ms=15, mew=2.2, zorder=6)
        ax.plot(0.062, -0.009, "s", mec=INK1, mfc="none", ms=15, mew=2.2, zorder=6)
        ax.plot(aw[own][0], aw[own][1], "*", color=C_INF if unit == "mm" else C_ATT,
                ms=28, mec=SURFACE, mew=1.8, zorder=8)
        ax.plot(aw[other][0], aw[other][1], "*", mfc="none",
                mec=C_ATT if unit == "mm" else C_INF, ms=28, mew=2.6, zorder=8)
        ax.set_title(f"{ttl}\n{sub}", fontsize=11, color=INK1, pad=9)
        ax.set_xlabel("world x (m)", color=INK2)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        style(ax); ax.grid(visible=False)
    axes[0].set_ylabel("world y (m)", color=INK2)
    cax = fig.add_axes([0.895, 0.26, 0.014, 0.52])
    cb = fig.colorbar(im, cax=cax, ticks=EDGES)
    cb.set_label("% of that panel's maximum", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=INK2, labelsize=9)
    same = pa == pi_
    fig.suptitle("Read on the WRIST view, attention's best placeable spot is the right one.\n"
                 f"{'The two panels now agree' if same else 'They still disagree'} - "
                 f"attention's pick carries {100*I[pa]/I[pi_]:.0f}% of the influence available.",
                 fontsize=12.2, color=INK1)
    h = [plt.Line2D([], [], marker="*", color=C_ATT, mec=SURFACE, ms=17, ls="none",
                    label="blue star = attention's best placeable spot"),
         plt.Line2D([], [], marker="*", color=C_INF, mec=SURFACE, ms=17, ls="none",
                    label="orange star = the most influential placeable spot"),
         plt.Rectangle((0, 0), 1, 1, fill=False, hatch="//", edgecolor="#8f8d84", lw=0.0,
                       label="hatched = would cover an object"),
         plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#111111", lw=2.2,
                       label="black box = that panel's overall maximum")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=9.5,
               labelcolor=INK2, bbox_to_anchor=(0.46, 0.002))
    fig.subplots_adjust(left=0.055, right=0.875, bottom=0.20, top=0.80, wspace=0.14)
    fig.savefig(OUT / "wrist1_side_by_side.png", dpi=125, facecolor=SURFACE)
    plt.close(fig)
    print("[written] wrist1_side_by_side.png")

    # ---------------- 图 2:逐层,腕部 vs 主视角
    yv = {v: np.array([100 * I[w[np.argmax(score(v, [l])[w])]] / I[best] for l in range(L)])
          for v in (0, 1)}
    fig, ax = plt.subplots(figsize=(9.2, 4.9), facecolor=SURFACE)
    ax.plot(range(L), yv[0], "-o", color=MUTED, lw=2, ms=6, label="read the BASE view")
    ax.plot(range(L), yv[1], "-o", color=C_ATT, lw=2.4, ms=7, label="read the WRIST view")
    rnd = 100 * I[w].mean() / I[best]
    ax.axhline(rnd, color=INK3, lw=1.2)
    ax.annotate(f"picking at random: {rnd:.0f}%", (L - 0.3, rnd), xytext=(0, 7),
                textcoords="offset points", color=INK2, fontsize=9.5, ha="right")
    ax.set_xlabel("which layer the attention is read from", color=INK2)
    ax.set_ylabel("influence obtained (% of the best placeable spot)", color=INK2)
    ax.set_xticks(range(0, L, 2)); ax.set_ylim(-4, 108)
    style(ax)
    ax.legend(fontsize=10, labelcolor=INK2, frameon=False, loc="lower right")
    ax.set_title("The wrist view finds the right spot in 6 layers; the base view in 2\n"
                 "but neither tells you which layer to read", fontsize=12.5, color=INK1, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "wrist2_layer.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("[written] wrist2_layer.png")

    # ---------------- 图 3:消融(腕部)
    def rn(X):
        return X / np.clip(X.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)

    variants = []

    def add(name, S_):
        p = w[np.argmax(S_[w])]
        variants.append((name, 100 * I[p] / I[best]))

    for pool in ("noun", "max", "sum"):
        add(f"pooling = {pool}", score(1, MID, pool=pool))
    for mode in ("renorm", "raw"):
        add(f"normalise = {mode}", score(1, MID, mode=mode))
    add("layers 3-8 (mid band)", score(1, MID))
    add("all 18 layers", score(1, range(L)))
    add("layers 0-2 (early)", score(1, range(0, 3)))
    add("layers 13-17 (late)", score(1, range(13, L)))
    add("base view instead", score(0, MID))
    ent, conc, stab = [], [], []
    T = A.shape[0]
    for l in range(L):
        s = rn(A[:, l])[:, NOUN, 1].sum(1)
        p = s / np.clip(s.sum((1, 2), keepdims=True), 1e-12, None)
        ent.append(float(np.mean(-(p * np.log(p + 1e-12)).sum((1, 2)))))
        conc.append(float(np.mean(s.max((1, 2)) / np.clip(s.mean((1, 2)), 1e-12, None))))
        f = s.reshape(T, -1); f = f / np.clip(np.linalg.norm(f, axis=1, keepdims=True), 1e-12, None)
        stab.append(float(np.mean([f[t] @ f[t + 1] for t in range(T - 1)])))
    for nm, arr, how in [("auto-pick: lowest entropy", ent, "min"),
                         ("auto-pick: sharpest peak", conc, "max"),
                         ("auto-pick: most stable", stab, "max")]:
        l = int(np.argmin(arr) if how == "min" else np.argmax(arr))
        add(f"{nm}  (chose layer {l})", score(1, [l]))
    variants.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9.6, 6.2), facecolor=SURFACE)
    yy = np.arange(len(variants)); vals = np.array([v for _, v in variants])
    ax.hlines(yy, 0, vals, color="#e0dfd8", lw=2.2)
    ax.plot(vals, yy, "o", color=C_ATT, ms=9, mec=SURFACE, mew=1.6)
    ax.axvline(rnd, color=INK3, lw=1.4)
    ax.annotate(f"random\n{rnd:.0f}%", (rnd, len(variants) - 0.4), xytext=(8, 0),
                textcoords="offset points", color=INK2, fontsize=9.5, va="top")
    ax.axvline(100, color=C_INF, lw=1.4)
    ax.annotate("perfect\n100%", (100, len(variants) - 0.4), xytext=(-8, 0),
                textcoords="offset points", color=C_INF, fontsize=9.5, va="top", ha="right")
    ax.set_yticks(yy); ax.set_yticklabels([n for n, _ in variants], fontsize=10)
    ax.set_xlabel("influence obtained by attention's top pick (%)", color=INK2)
    ax.set_xlim(0, 112)
    style(ax); ax.grid(axis="y", visible=False)
    ax.set_title("Ablation on the WRIST view: several settings now reach 100%,\n"
                 "but every automatic layer-picking rule still fails",
                 fontsize=12, color=INK1, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "wrist3_ablation.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("[written] wrist3_ablation.png")
    for n, v in variants:
        print(f"    {n:34s} {v:6.1f}%")

    # ---------------- 图 4:可达性(腕部 vs 主视角)
    fig, ax = plt.subplots(figsize=(9.0, 4.6), facecolor=SURFACE)
    labs, base_v, wrist_v = [], [], []
    for view, store in ((0, base_v), (1, wrist_v)):
        sal = np.mean([rn(A[:, l])[:, NOUN, view].sum(1) for l in MID], axis=0).mean(0)
        occ = (cov[view] > 0.01).any(1)
        pk = sal.max()
        # ⚠️ 有些锚点在某个视角下完全不可见(覆盖为空),要跳过,否则 max 会在空数组上报错
        def reach(idxs):
            v = [sal[occ[i]].max() for i in idxs if occ[i].any()]
            return 100 * max(v) / pk if v else 0.0
        store.append(reach(w))
        store.append(reach(np.where((~leg) & vis)[0]))
    labs = ["best LEGAL spot can touch", "best ILLEGAL spot can touch"]
    x = np.arange(2); bw = 0.36
    ax.bar(x - bw / 2, base_v, bw, color=MUTED, label="base view")
    ax.bar(x + bw / 2, wrist_v, bw, color=C_ATT, label="wrist view")
    for xi, (a, b) in enumerate(zip(base_v, wrist_v)):
        ax.annotate(f"{a:.0f}%", (xi - bw / 2, a), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=INK1, fontsize=10)
        ax.annotate(f"{b:.0f}%", (xi + bw / 2, b), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=INK1, fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=10.5)
    ax.set_ylabel("% of the attention peak value", color=INK2)
    ax.set_ylim(0, 115)
    style(ax); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=10, labelcolor=INK2, frameon=False)
    ax.set_title("The strongest attention still sits where you cannot place a patch\n"
                 "on the wrist view the gap is starker: 100% vs 27%",
                 fontsize=12.5, color=INK1, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "wrist4_reachability.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("[written] wrist4_reachability.png")


if __name__ == "__main__":
    main()
