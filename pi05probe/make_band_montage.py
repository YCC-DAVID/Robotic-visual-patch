#!/usr/bin/env python3
"""3 × 6 拼版:行 = 3 个层带,列 = 原句 + 5 个改写。

行的分带按之前总结的层结构(用户 2026-08-12 指定):
    layers 0-2    前段:被跨指令不变的热格(sink)主导
    layers 3-14   中段:落在操作对象(碗)和目的地(盘子)上
    layers 15-17  后段:又回到 sink

列 = 同一任务的 6 种说法(只改动词/句法/框架词,**不改名词**):
    put / place / set / move the bowl on the plate
    on the plate, put the bowl        (换句法)
    please put the bowl on the plate  (加框架词)

口径
----
- attention:head 求和(存盘已做)→ 在 512 个图像 token 上重归一化(规格 A4)
  → 名词图(bowl + plate 两个 token 求和)
- 只用**抓取前**的帧(碗还在桌上不动),所以碗/盘子的标记位置是确定的;
  跨这些帧平均。若用全 16 帧,碗被抓起来移动,标记就没有单一位置了。
- 颜色**按行归一化**:同一层带内 6 列共用一个刻度,这样"换说法有没有让图变化"
  可以横向比较;跨行不可比(不同层的注意力量级差很多)。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/make_band_montage.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from make_clean_figs import INK1, INK2, INK3, SURFACE  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NOUN_ROWS = [3, 6]
BOWL_QPOS, PLATE_QPOS = slice(10, 13), slice(31, 34)
C_OBJ, C_DEST = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7fafd", "#9cc4ea", "#2a78d6", "#0e2f56"])

BANDS = [(range(0, 3), "layers 0-2", "early\nsink-dominated"),
         (range(3, 15), "layers 3-14", "middle\non object +\ndestination"),
         (range(15, 18), "layers 15-17", "late\nback to sink")]

# 列:原句 + 5 个改写。名称取自 attn_traj 的 variant key
COLS = [("bowl_plate_orig", "put the bowl\non the plate", "original"),
        ("bowl_plate_L1_place", "place the bowl\non the plate", "verb"),
        ("bowl_plate_L1_set", "set the bowl\non the plate", "verb"),
        ("bowl_plate_L1_move", "move the bowl\non the plate", "verb"),
        ("bowl_plate_L2_front", "on the plate,\nput the bowl", "syntax"),
        ("bowl_plate_L3_please", "please put the bowl\non the plate", "framing")]


def world_to_cell(Pw, K, E):
    """只翻列、不翻行(用 36 个锚点的 diff 掩码质心实测定出,误差 0.53 格)。"""
    Pc = np.linalg.inv(E) @ np.array([Pw[0], Pw[1], Pw[2], 1.0])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    z = Pc[2]
    u, v = cx + fx * Pc[0] / z, cy + fy * Pc[1] / z
    return ((v + 0.5) * 224 / 256 - 0.5) / 14 - 0.5, (((255 - u) + 0.5) * 224 / 256 - 0.5) / 14 - 0.5


def main():
    az = np.load(OUT / "attn_traj_put_the_bowl_on_the_plate.npz", allow_pickle=True)
    tz = np.load(OUT / "traj_put_the_bowl_on_the_plate.npz", allow_pickle=False)
    sf = np.load(OUT / "shared_frame.npz", allow_pickle=True)
    K = sf["put_the_bowl_on_the_plate__K_agentview"]
    E = sf["put_the_bowl_on_the_plate__E_agentview"]
    T = int(tz["n_frames"])

    missing = [c for c, _, _ in COLS if f"{c}__attn" not in az.files]
    assert not missing, f"attn_traj 里缺这些 variant:{missing}"

    # 抓取前的帧:碗的高度还没变
    z0 = tz["f000__flatten"][BOWL_QPOS][2]
    pre = [t for t in range(T) if tz[f"f{t:03d}__flatten"][BOWL_QPOS][2] < z0 + 0.005]
    bg = tz[f"f{pre[len(pre)//2]:03d}__img224"]
    brc = world_to_cell(tz["f000__flatten"][BOWL_QPOS], K, E)
    prc = world_to_cell(tz["f000__flatten"][PLATE_QPOS], K, E)
    print(f"[cfg] 抓取前帧 {pre}  背景用第 {pre[len(pre)//2]} 帧")
    print(f"[cfg] 碗格 ({brc[0]:.1f},{brc[1]:.1f})  盘子格 ({prc[0]:.1f},{prc[1]:.1f})")

    def band_map(variant, layers):
        A = az[f"{variant}__attn"].astype(np.float64)
        out = []
        for l in layers:
            X = A[pre, l]                                     # (npre, Z, V, 16, 16)
            X = X / np.clip(X.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)
            out.append(X[:, NOUN_ROWS, 0].sum(1).mean(0))      # 名词图,主视角,帧平均
        return np.mean(out, axis=0)

    nr, nc = len(BANDS), len(COLS)
    fig, axes = plt.subplots(nr, nc, figsize=(2.28 * nc + 2.2, 2.28 * nr + 1.9),
                            facecolor=SURFACE)
    for r, (layers, rlab, rnote) in enumerate(BANDS):
        maps = [band_map(c, layers) for c, _, _ in COLS]
        vmax = max(m.max() for m in maps)                      # 行内共用刻度
        for c, (m, (_, text, kind)) in enumerate(zip(maps, COLS)):
            ax = axes[r, c]
            ax.imshow(bg)
            ax.imshow(np.kron(m, np.ones((14, 14))), cmap=SEQ, alpha=0.60,
                      vmin=0, vmax=vmax)
            ax.plot((brc[1] + .5) * 14, (brc[0] + .5) * 14, "o", mec=C_OBJ, mfc="none",
                    ms=15, mew=2.4)
            ax.plot((prc[1] + .5) * 14, (prc[0] + .5) * 14, "s", mec=C_DEST, mfc="none",
                    ms=15, mew=2.4)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#dcdbd4")
            if r == 0:
                ax.set_title(f"{text}\n({kind})", fontsize=9.5, color=INK1, pad=8)
            if c == 0:
                # 层带标签:粗体带名 + 换行的说明,一次画完,避免和列标题打架
                ax.set_ylabel(f"{rlab}\n{rnote}", fontsize=9.5, color=INK1,
                              labelpad=8, linespacing=1.5)

    h = [plt.Line2D([], [], marker="o", mec=C_OBJ, mfc="none", ms=11, mew=2.2, ls="none",
                    label="bowl (the object to move)"),
         plt.Line2D([], [], marker="s", mec=C_DEST, mfc="none", ms=11, mew=2.2, ls="none",
                    label="plate (the destination)")]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=10,
               labelcolor=INK2, bbox_to_anchor=(0.53, 0.005))
    fig.suptitle("Same task, six ways of saying it (nouns never changed).   "
                 "Rows are layer bands; colour scale is shared within a row.\n"
                 "Rephrasing barely moves the maps - across the trajectory the wording "
                 "changes attention less than advancing one time step does.",
                 fontsize=12, color=INK1)
    fig.tight_layout(rect=[0.015, 0.035, 1, 0.90])
    fig.savefig(OUT / "band_montage_rephrase.png", dpi=125, facecolor=SURFACE)
    plt.close(fig)
    print("[written] band_montage_rephrase.png")

    # 顺手把定量数字打出来,方便和图对照
    print("\n每层带内,改写图 vs 原句图 的秩相关(名词图,主视角,抓取前帧平均):")

    def sp(a, b):
        def rk(x):
            x = np.asarray(x, float).ravel()
            o = np.argsort(x, kind="stable")
            r = np.empty(len(x)); r[o] = np.arange(len(x))
            _, inv, cc = np.unique(x, return_inverse=True, return_counts=True)
            s = np.zeros(len(cc)); np.add.at(s, inv, r)
            return (s / cc)[inv]
        ra, rb = rk(a), rk(b)
        ra = ra - ra.mean(); rb = rb - rb.mean()
        return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))

    for layers, rlab, _ in BANDS:
        base = band_map("bowl_plate_orig", layers)
        vals = [sp(base, band_map(c, layers)) for c, _, _ in COLS[1:]]
        print(f"  {rlab:12s} " + "  ".join(f"{v:.3f}" for v in vals) +
              f"   均值 {np.mean(vals):.3f}")


if __name__ == "__main__":
    main()
