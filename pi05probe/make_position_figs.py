#!/usr/bin/env python3
"""用**原始图像**论证核心结论的配图,并回答"中间层的注意力到底落在哪"。

产出
----
    fig_positions_rendered.png   4 个条件的真实渲染图(模型看到的那张)+ 影响力标注
    fig_attention_where.png      第 3/7/8 层注意力叠在原图上,标出碗/盘子/sink/候选位置
    fig_gain_vs_influence.png    贴纸吸走的注意力增益 vs 影响力(是否普遍升高)

世界坐标 → 模型输入图像格 的正向投影
--------------------------------
`reproject.py` 做的是反方向(格 → 世界)且已自检通过(桌面反投影 = z=0.900)。
本脚本实现正方向,并用**同一批锚点的 diff 掩码质心**做自检:
投影出来的锚点中心必须落在该锚点 diff 掩码的质心附近(误差 < 1.5 格)。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/make_position_figs.py
"""
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NOUN_ROWS = [3, 6]
BOWL = (-0.098, -0.009, 0.93)
PLATE = (0.062, -0.009, 0.91)
SINK_CELLS = [(7, 9), (7, 8), (6, 9)]
# 三个关键位置(见 RESULTS.md 第 11 节)
KEY = [((-0.06, 0.22), "influence best", "tab:green"),
       ((0.21, 0.22), "attention pick #1", "tab:red"),
       ((0.12, 0.35), "attention pick #2", "tab:orange")]


def world_to_cell(Pw, K, E):
    """世界 3D → 模型输入图(224)的 16×16 网格坐标 (row, col),浮点。

    ⚠️ 行/列的翻转不对称 —— 这是实测定出来的,不是推导的:
       **只翻列,不翻行。**
       用 36 个锚点的 diff 掩码质心当真值,四种翻转组合的平均误差分别是
           翻行+翻列 8.98 格 / 只翻行 12.77 / **只翻列 0.53** / 都不翻 8.10
       原因:深度缓冲的行序与喂给模型的 RGB 的行序相反(MuJoCo/OpenGL 原点在左下)。
       ⇒ `reproject.py` 的格→世界映射**行是翻反的**,它报的世界坐标全部作废。
    """
    Pw = np.asarray(Pw, float)
    Pc = np.linalg.inv(E) @ np.array([Pw[0], Pw[1], Pw[2], 1.0])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = Pc[2]
    u_raw = cx + fx * Pc[0] / z
    v_raw = cy + fy * Pc[1] / z
    u256 = 255.0 - u_raw        # 列:翻
    v256 = v_raw                # 行:不翻
    u224 = (u256 + 0.5) * 224.0 / 256.0 - 0.5
    v224 = (v256 + 0.5) * 224.0 / 256.0 - 0.5
    return v224 / 14.0 - 0.5, u224 / 14.0 - 0.5      # (row, col) in cell units


def main():
    ob = np.load(OUT / "s2_scan_obs.npz", allow_pickle=True)
    b4 = np.load(OUT / "b4_attn_vs_influence.npz", allow_pickle=True)
    tx = np.load(OUT / "texture_axis.npz", allow_pickle=True)
    ap = np.load(OUT / "attn_patched_grid.npz", allow_pickle=True)
    sf = np.load(OUT / "shared_frame.npz", allow_pickle=True)
    K = sf["put_the_bowl_on_the_plate__K_agentview"]
    E = sf["put_the_bowl_on_the_plate__E_agentview"]

    cov = b4["cov_base"]; Iavg = tx["Imag_avg"]
    aw = b4["anchor_world"]; leg = b4["anchor_legal"].astype(bool)
    cimg = ob["clean_img224"]; pimg = ob["patched_img224"]
    Ac = ap["attn_clean"].astype(np.float64)
    Ap = ap["attn_patched"].astype(np.float64)
    M, T = cov.shape[0], cov.shape[1]
    vis = (cov.sum((2, 3)) > 0).any(1)
    w = np.where(leg & vis)[0]
    best = w[np.argmax(Iavg[w])]

    # ---------------- 自检:投影 vs diff 掩码质心 ----------------
    print("[自检] 世界→格 投影 与 diff 掩码质心 的偏差(格):")
    errs = []
    for i in range(M):
        m = cov[i, 0]
        if m.sum() <= 0:
            continue
        rr, cc = np.nonzero(m)
        wt = m[rr, cc]
        cen = (np.average(rr, weights=wt), np.average(cc, weights=wt))
        pr, pc = world_to_cell((aw[i][0], aw[i][1], 0.9015), K, E)
        errs.append(np.hypot(cen[0] - pr, cen[1] - pc))
    errs = np.array(errs)
    print(f"  n={len(errs)}  mean={errs.mean():.2f}  median={np.median(errs):.2f}  max={errs.max():.2f}")
    ok = np.median(errs) < 1.5
    print("  " + ("✅ 投影正确,可信下面的标注" if ok else
                  "❌ 投影偏差过大,下面的标注不可信"))

    def sal_avg(A, l):
        """(T,L,Z,V,16,16) → 名词图,在图像 token 上重归一化,沿 t 平均。"""
        X = A[:, l]
        X = X / np.clip(X.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)
        return X[:, NOUN_ROWS, 0].sum(1).mean(0)

    # ================= 图 1:真实渲染图 =================
    frame = 8            # 中段(env 步 50),手臂在场景里,能看出遮挡关系
    conds = [(None, "clean (no patch)", "k")] + KEY
    fig, ax = plt.subplots(2, 4, figsize=(16, 8.4))
    sal7 = sal_avg(Ac, 7)
    for j, (pos, name, cl) in enumerate(conds):
        if pos is None:
            img = cimg[frame]
            inf = None
        else:
            i = int(np.argmin(np.hypot(aw[:, 0] - pos[0], aw[:, 1] - pos[1])))
            img = pimg[i, frame]
            inf = Iavg[i]
        ax[0, j].imshow(img)
        ttl = name if inf is None else f"{name}\nworld ({pos[0]:.2f}, {pos[1]:.2f})   influence {inf:.0f} mm"
        ax[0, j].set_title(ttl, fontsize=10, color=cl)
        ax[0, j].axis("off")
        if pos is not None:
            r, c = world_to_cell((pos[0], pos[1], 0.9015), K, E)
            ax[0, j].add_patch(plt.Circle(((c + 0.5) * 14, (r + 0.5) * 14), 22, fill=False,
                                          ec=cl, lw=2.5))
        # 第二行:叠第 7 层注意力
        ax[1, j].imshow(img)
        ax[1, j].imshow(np.kron(sal7, np.ones((14, 14))), cmap="inferno", alpha=0.55)
        ax[1, j].set_title("+ layer-7 attention (noun map, clean image)", fontsize=9)
        ax[1, j].axis("off")
    fig.suptitle("What the model actually sees, at env step 50.   Bottom row: layer-7 attention "
                 "computed on the CLEAN image, overlaid.\n"
                 "Attention's own top pick (red) sits where influence is only 25% of the best "
                 "legal position (green).", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig_positions_rendered.png", dpi=110); plt.close(fig)
    print("[written] fig_positions_rendered.png")

    # ================= 图 2:注意力落在哪 =================
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
    br, bc = world_to_cell(BOWL, K, E)
    pr, pc = world_to_cell(PLATE, K, E)
    for j, l in enumerate([3, 7, 8, 10]):
        s = sal_avg(Ac, l)
        ax[j].imshow(cimg[0])
        ax[j].imshow(np.kron(s, np.ones((14, 14))), cmap="inferno", alpha=0.6)
        pk = np.unravel_index(int(s.argmax()), s.shape)
        ax[j].plot((bc + .5) * 14, (br + .5) * 14, "o", mec="cyan", mfc="none", ms=16, mew=2.5)
        ax[j].plot((pc + .5) * 14, (pr + .5) * 14, "s", mec="lime", mfc="none", ms=16, mew=2.5)
        ax[j].plot((pk[1] + .5) * 14, (pk[0] + .5) * 14, "x", color="white", ms=15, mew=3)
        for (sr, sc_) in SINK_CELLS:
            ax[j].add_patch(plt.Rectangle((sc_ * 14, sr * 14), 14, 14, fill=False,
                                          ec="deepskyblue", lw=1.4, ls="--"))
        for pos, nm, cl in KEY:
            r, c = world_to_cell((pos[0], pos[1], 0.9015), K, E)
            ax[j].plot((c + .5) * 14, (r + .5) * 14, "+", color=cl, ms=14, mew=3)
        ax[j].set_title(f"layer {l}   attention peak at cell {pk}", fontsize=10)
        ax[j].axis("off")
    fig.suptitle("Where does the attention actually peak?   cyan circle = bowl,  green square = plate,  "
                 "white X = attention peak,  dashed blue = cells bright for EVERY instruction,  "
                 "+ = the three candidate placements\n"
                 "The peak IS on the task objects - next to the bowl at layers 3 and 7, on the plate at "
                 "layers 8 and 10.  But every placeable candidate (+) is far away from them.",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig_attention_where.png", dpi=110); plt.close(fig)
    print("[written] fig_attention_where.png")

    # ================= 图 3:增益 vs 影响力 =================
    def rn(X):
        return X / np.clip(X.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for j, l in enumerate([4, 7, 8]):
        Sc_sal = rn(Ac[:, l])[:, NOUN_ROWS, 0].sum(1)
        num = np.einsum("mtij,tij->mt", cov, Sc_sal); den = cov.sum((2, 3))
        Sc = np.divide(num, den, out=np.zeros_like(num), where=den > 0).sum(1)
        Sp = np.zeros(M)
        for i in range(M):
            s = rn(Ap[i, :, l])[:, NOUN_ROWS, 0].sum(1)
            n2 = np.einsum("tij,tij->t", cov[i], s); d2 = cov[i].sum((1, 2))
            Sp[i] = np.divide(n2, d2, out=np.zeros_like(n2), where=d2 > 0).sum()
        g = np.divide(Sp, Sc, out=np.zeros_like(Sp), where=Sc > 0)
        ax[j].scatter(Iavg[w], g[w], s=55, c="tab:purple", edgecolors="k", linewidths=0.5)
        ax[j].axhline(1.0, color="k", ls="--", lw=1.2)
        ax[j].scatter([Iavg[best]], [g[best]], s=140, marker="*", c="tab:green",
                      edgecolors="k", label="influence-best position")
        ax[j].set_yscale("log")
        ax[j].set_xlabel("influence (mm)")
        ax[j].set_ylabel("attention gain  after / before  (log)")
        ax[j].set_title(f"layer {l}:  gain spans {g[w].min():.1f}x - {g[w].max():.1f}x,\n"
                        f"but is uncorrelated with influence", fontsize=10)
        ax[j].legend(fontsize=7); ax[j].grid(alpha=0.3)
    fig.suptitle("Placing the patch does inflate attention on it - but by wildly different amounts, "
                 "and the amount has nothing to do with the patch's actual influence.\n"
                 "The best position (green star) is among the LEAST inflated.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT / "fig_gain_vs_influence.png", dpi=110); plt.close(fig)
    print("[written] fig_gain_vs_influence.png")


if __name__ == "__main__":
    main()
