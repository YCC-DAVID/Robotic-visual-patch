#!/usr/bin/env python3
"""补齐总结文档需要的 3 张图(纯后处理)。标签一律英文,避免 matplotlib 缺 CJK 字形。

    fig_texture_consistency.png  换随机纹理后 influence 是否一致
    fig_legal_positions.png      合法位置的 influence 排名 + attention 逐层选哪个
    fig_patched_attention.png    贴上 patch 后 attention 搬家了吗

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/make_summary_figs.py
"""
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NOUN_ROWS = [3, 6]


def spearman(a, b):
    def rank(x):
        x = np.asarray(x, np.float64)
        o = np.argsort(x, kind="stable")
        r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=np.float64)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt)); np.add.at(s, inv, r)
        return (s / cnt)[inv]
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


# ---------------------------------------------------------------- 图 1:纹理一致性
def fig_texture():
    tx = np.load(OUT / "texture_axis.npz", allow_pickle=True)
    tags = [str(t) for t in tx["tags"]]
    I = [tx[f"Imag_{t}"] for t in tags]
    leg = tx["anchor_legal"].astype(bool)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for j, (a, b) in enumerate([(0, 1), (0, 2)]):
        ax[0].scatter(I[a][leg], I[b][leg], s=40, marker="o", label=f"legal, tex1 vs tex{b+1}"
                      if j == 0 else None, c="tab:blue" if j == 0 else "tab:cyan",
                      edgecolors="k", linewidths=0.4)
        ax[0].scatter(I[a][~leg], I[b][~leg], s=40, marker="^",
                      label="occludes object" if j == 0 else None,
                      c="tab:red" if j == 0 else "tab:orange", edgecolors="k", linewidths=0.4)
    lim = max(I[0].max(), I[1].max(), I[2].max()) * 1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=1)
    ax[0].set_xlabel("influence with random texture #1 (mm)")
    ax[0].set_ylabel("influence with texture #2 / #3 (mm)")
    ax[0].set_title("Same positions, different random probe textures\n"
                    f"rank correlation = {spearman(I[0], I[1]):.4f} / {spearman(I[0], I[2]):.4f}",
                    fontsize=10)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    stack = np.stack(I)
    cv = stack.std(0) / np.maximum(stack.mean(0), 1e-9)
    o = np.argsort(-stack.mean(0))
    ax[1].bar(range(len(o)), 100 * cv[o],
              color=["tab:blue" if leg[i] else "tab:red" for i in o])
    ax[1].set_xlabel("positions, sorted by influence (highest left)")
    ax[1].set_ylabel("spread across 3 textures (% of mean)")
    ax[1].set_title("Per-position variability across textures\n"
                    "blue = legal, red = occludes object", fontsize=10)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(OUT / "fig_texture_consistency.png", dpi=120)
    plt.close(fig)
    print("[written] fig_texture_consistency.png")


# ------------------------------------------------- 图 2:合法位置 + attention 逐层选点
def fig_legal():
    b4 = np.load(OUT / "b4_attn_vs_influence.npz", allow_pickle=True)
    tx = np.load(OUT / "texture_axis.npz", allow_pickle=True)
    ap = np.load(OUT / "attn_patched_grid.npz", allow_pickle=True)
    cov = b4["cov_base"]; Iavg = tx["Imag_avg"]
    leg = b4["anchor_legal"].astype(bool); aw = b4["anchor_world"]
    Ac = ap["attn_clean"].astype(np.float64)
    vis = (cov.sum((2, 3)) > 0).any(1)
    w = np.where(leg & vis)[0]
    best = w[np.argmax(Iavg[w])]
    L = Ac.shape[1]

    def score(sal):
        num = np.einsum("mtij,tij->mt", cov, sal)
        den = cov.sum((2, 3))
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0).sum(1)

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    o = w[np.argsort(-Iavg[w])]
    lbl = [f"({aw[i][0]:.2f}, {aw[i][1]:.2f})" for i in o]
    ax[0].barh(range(len(o))[::-1], Iavg[o], color="tab:green")
    ax[0].set_yticks(range(len(o))[::-1]); ax[0].set_yticklabels(lbl, fontsize=7)
    ax[0].set_xlabel("influence (mm, action space)")
    ax[0].set_title("The 17 legal (placeable) positions, ranked by measured influence\n"
                    "top bar = the position an attacker should choose", fontsize=10)
    ax[0].grid(alpha=0.3, axis="x")

    for kind, mk, cl in [("max", "o", "tab:blue"), ("sum", "s", "tab:orange"),
                         ("noun", "^", "tab:green")]:
        ys = []
        for l in range(L):
            X = Ac[:, l]
            X = X / np.clip(X.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)
            sal = (X[:, :, 0].max(1) if kind == "max" else
                   X[:, :, 0].sum(1) if kind == "sum" else X[:, NOUN_ROWS, 0].sum(1))
            p = w[np.argmax(score(sal)[w])]
            ys.append(100 * Iavg[p] / Iavg[best])
        ax[1].plot(range(L), ys, marker=mk, ms=5, color=cl, label=f"text-token pooling = {kind}")
    ax[1].axhline(100, color="k", ls="--", lw=1.2, label="the true best legal position")
    ax[1].set_xlabel("transformer layer the attention is read from")
    ax[1].set_ylabel("influence obtained (% of best legal position)")
    ax[1].set_title("Which legal position does attention pick?\n"
                    "only layers 7-8 land on the true best", fontsize=10)
    ax[1].set_xticks(range(0, L, 2)); ax[1].set_ylim(-3, 112)
    ax[1].legend(fontsize=7, loc="upper right"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_legal_positions.png", dpi=120)
    plt.close(fig)
    print("[written] fig_legal_positions.png")


# ---------------------------------------------------------- 图 3:clean vs patched attention
def fig_patched():
    b4 = np.load(OUT / "b4_attn_vs_influence.npz", allow_pickle=True)
    tx = np.load(OUT / "texture_axis.npz", allow_pickle=True)
    ap = np.load(OUT / "attn_patched_grid.npz", allow_pickle=True)
    cov = b4["cov_base"]; Iavg = tx["Imag_avg"]
    leg = b4["anchor_legal"].astype(bool)
    Ac = ap["attn_clean"].astype(np.float64)
    Ap = ap["attn_patched"].astype(np.float64)
    M, L = Ap.shape[0], Ap.shape[2]
    vis = (cov.sum((2, 3)) > 0).any(1)
    w = np.where(leg & vis)[0]
    sel = leg & vis
    RELIABLE = 10          # layers >= 10 有 float16 下溢,虚线画出但不用于结论

    def rn(X):
        return X / np.clip(X.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)

    gains, s_ci, s_pi, s_cp = [], [], [], []
    for l in range(L):
        Sc_sal = rn(Ac[:, l])[:, NOUN_ROWS, 0].sum(1)
        num = np.einsum("mtij,tij->mt", cov, Sc_sal); den = cov.sum((2, 3))
        Sc = np.divide(num, den, out=np.zeros_like(num), where=den > 0).sum(1)
        Sp = np.zeros(M)
        for i in range(M):
            sal = rn(Ap[i, :, l])[:, NOUN_ROWS, 0].sum(1)
            n2 = np.einsum("tij,tij->t", cov[i], sal); d2 = cov[i].sum((1, 2))
            Sp[i] = np.divide(n2, d2, out=np.zeros_like(n2), where=d2 > 0).sum()
        g = np.divide(Sp, Sc, out=np.zeros_like(Sp), where=Sc > 0)
        gains.append(np.median(g[sel]))
        s_ci.append(spearman(Sc[w], Iavg[w]))
        s_pi.append(spearman(Sp[w], Iavg[w]))
        s_cp.append(spearman(Sc[w], Sp[w]))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(L)
    ax[0].plot(x[:RELIABLE], np.array(gains)[:RELIABLE], marker="o", color="tab:red", lw=2,
               label="reliable (float32-equivalent)")
    ax[0].plot(x[RELIABLE - 1:], np.array(gains)[RELIABLE - 1:], marker="o", color="tab:red",
               lw=1.2, ls=":", alpha=0.55, label="float16 underflow - do not use")
    ax[0].axhline(1.0, color="k", ls="--", lw=1.2, label="no change")
    ax[0].set_xlabel("transformer layer"); ax[0].set_ylabel("attention on the patch:  after / before")
    ax[0].set_title("Does the patch pull attention onto itself?\n"
                    "median over the 17 legal positions", fontsize=10)
    ax[0].set_xticks(range(0, L, 2)); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)

    for y, lab, cl in [(s_ci, "attention BEFORE placing  vs  influence", "tab:blue"),
                       (s_pi, "attention AFTER placing  vs  influence", "tab:green"),
                       (s_cp, "attention before  vs  attention after", "tab:purple")]:
        y = np.array(y)
        ax[1].plot(x[:RELIABLE], y[:RELIABLE], marker="o", ms=4, color=cl, lw=2, label=lab)
        ax[1].plot(x[RELIABLE - 1:], y[RELIABLE - 1:], color=cl, lw=1.2, ls=":", alpha=0.55)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("transformer layer"); ax[1].set_ylabel("rank correlation")
    ax[1].set_title("Attention measured after placing the patch predicts influence better\n"
                    "- but the attacker cannot obtain it", fontsize=10)
    ax[1].set_xticks(range(0, L, 2)); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_patched_attention.png", dpi=120)
    plt.close(fig)
    print("[written] fig_patched_attention.png")


if __name__ == "__main__":
    fig_texture(); fig_legal(); fig_patched()
