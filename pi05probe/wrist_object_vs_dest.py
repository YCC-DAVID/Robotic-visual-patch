#!/usr/bin/env python3
"""腕部视角:注意力落在操作对象(碗)还是目的地(盘子)。纯后处理。

为什么要重做
----------
之前这条是在**主视角**上算的,而消融实测主视角可以整个扔掉(置 0 仍 15/15 成功),
承重的是腕部。所以那个结论(目的地主导)必须在腕部上重验。

两个必须先解决的技术点
--------------------
1. 腕部相机**跟着夹爪动**,每帧位姿不同 ⇒ 用 `dump_wrist_cams.py` 存的逐帧外参。
2. 世界→图像格的翻转约定**不能照搬主视角**(主视角实测是"只翻列不翻行")。
   这里用腕部自己的 diff 掩码质心当真值,把四种组合都试一遍,选误差最小的。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/wrist_object_vs_dest.py
"""
import itertools
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NOUN = [3, 6]
BOWL_Q, PLATE_Q = slice(10, 13), slice(31, 34)
C_OBJ, C_DEST = "#2a78d6", "#eb6834"
INK1, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"


def project(Pw, K, E, flip_row, flip_col):
    Pc = np.linalg.inv(E) @ np.array([Pw[0], Pw[1], Pw[2], 1.0])
    z = Pc[2]
    if z <= 1e-6:
        return np.nan, np.nan          # 在相机后面
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = cx + fx * Pc[0] / z, cy + fy * Pc[1] / z
    r256 = (255.0 - v) if flip_row else v
    c256 = (255.0 - u) if flip_col else u
    return (((r256 + 0.5) * 224 / 256 - 0.5) / 14 - 0.5,
            ((c256 + 0.5) * 224 / 256 - 0.5) / 14 - 0.5)


def win_mass(sal, r, c, win=1):
    if not np.isfinite(r) or not np.isfinite(c):
        return np.nan
    ri, ci = int(round(r)), int(round(c))
    if not (-win <= ri < 16 + win and -win <= ci < 16 + win):
        return np.nan                   # 投影到画面外
    r0, r1 = max(0, ri - win), min(16, ri + win + 1)
    c0, c1 = max(0, ci - win), min(16, ci + win + 1)
    if r1 <= r0 or c1 <= c0:
        return np.nan
    return float(sal[r0:r1, c0:c1].sum() / max(sal.sum(), 1e-12))


def main():
    az = np.load(OUT / "attn_traj_put_the_bowl_on_the_plate.npz", allow_pickle=True)
    tz = np.load(OUT / "traj_put_the_bowl_on_the_plate.npz", allow_pickle=False)
    ob = np.load(OUT / "s2_scan_obs.npz", allow_pickle=True)
    cams = np.load(OUT / "wrist_cams.npz", allow_pickle=False)
    b4 = np.load(OUT / "b4_attn_vs_influence.npz", allow_pickle=True)
    A = az["bowl_plate_orig__attn"].astype(np.float64)
    T, L = A.shape[0], A.shape[1]
    Kw, Ew = cams["K_wrist"], cams["E_wrist"]
    aw = b4["anchor_world"]

    # ---------- 定翻转约定:用腕部 diff 掩码质心当真值 ----------
    c = ob["clean_wrist224"].astype(np.int16)
    p = ob["patched_wrist224"].astype(np.int16)
    m = (np.abs(p - c[None]).max(-1) > 10)                 # (M,T,224,224)
    g = m.reshape(m.shape[0], T, 16, 14, 16, 14).mean((3, 5))
    print("[定约定] 用腕部 diff 掩码质心当真值,四种翻转组合的平均误差(格):")
    best_cfg, best_err = None, np.inf
    for fr, fc in itertools.product([True, False], repeat=2):
        errs = []
        for i in range(g.shape[0]):
            for t in range(T):
                cell = g[i, t]
                if cell.sum() <= 0:
                    continue
                rr, cc = np.nonzero(cell)
                wt = cell[rr, cc]
                gr, gc = np.average(rr, weights=wt), np.average(cc, weights=wt)
                pr, pc = project((aw[i][0], aw[i][1], 0.9015), Kw, Ew[t], fr, fc)
                if np.isfinite(pr):
                    errs.append(np.hypot(gr - pr, gc - pc))
        e = np.median(errs) if errs else np.inf
        print(f"    翻行={str(fr):5s} 翻列={str(fc):5s} → 中位误差 {e:6.2f}  (n={len(errs)})")
        if e < best_err:
            best_cfg, best_err = (fr, fc), e
    fr, fc = best_cfg
    print(f"[定约定] 选中 翻行={fr} 翻列={fc},中位误差 {best_err:.2f} 格 "
          f"{'✅ 可信' if best_err < 2.0 else '❌ 误差过大,下面结果不可信'}")
    assert best_err < 2.0, "腕部投影没有一种约定能对上,不要用下面的数"

    # ---------- 逐帧投影碗/盘子 ----------
    flats = np.stack([tz[f"f{k:03d}__flatten"] for k in range(T)])
    lift = flats[:, BOWL_Q][:, 2]
    pre = [t for t in range(T) if lift[t] < lift[0] + 0.005]     # 碗还在桌上
    brc = np.array([project(flats[t][BOWL_Q], Kw, Ew[t], fr, fc) for t in range(T)])
    prc = np.array([project(flats[t][PLATE_Q], Kw, Ew[t], fr, fc) for t in range(T)])
    inb = lambda rc: np.isfinite(rc[0]) and -1 <= rc[0] < 17 and -1 <= rc[1] < 17
    print(f"\n[相位] 抓取前帧 {pre}")
    print(f"  碗在腕部画面内的帧数  {sum(inb(brc[t]) for t in pre)}/{len(pre)}")
    print(f"  盘子在腕部画面内的帧数 {sum(inb(prc[t]) for t in pre)}/{len(pre)}")

    def rn(X):
        return X / np.clip(X.sum(axis=(-3, -2, -1), keepdims=True), 1e-12, None)

    base = 9 / 256
    yo, yd = [], []
    for l in range(L):
        sal = rn(A[:, l])[:, NOUN, 1].sum(1)                     # 腕部 = view 1
        yo.append(np.nanmean([win_mass(sal[t], *brc[t]) for t in pre]))
        yd.append(np.nanmean([win_mass(sal[t], *prc[t]) for t in pre]))
    yo, yd = np.array(yo) / base, np.array(yd) / base

    print("\n腕部视角:注意力密度(×均匀),抓取前帧平均")
    print("  层 |   碗   | 盘子  | 谁高")
    for l in range(L):
        print(f"  {l:3d} | {yo[l]:6.1f} | {yd[l]:5.1f} | {'碗' if yo[l] > yd[l] else '盘子'}")
    nb = int(np.nansum(yo > yd))
    print(f"\n  碗更高的层数 = {nb}/{L}   (主视角上是 10/18,中段由盘子主导)")

    fig, ax = plt.subplots(figsize=(8.8, 4.8), facecolor=SURFACE)
    ax.plot(range(L), yo, "-o", color=C_OBJ, lw=2.4, ms=6, label="on the bowl (the object)")
    ax.plot(range(L), yd, "-s", color=C_DEST, lw=2.4, ms=6, label="on the plate (the destination)")
    ax.axhline(1.0, color=INK3, lw=1.2)
    ax.annotate("uniform attention", (L - 0.4, 1.0), xytext=(0, 6), textcoords="offset points",
                color=INK2, fontsize=9, ha="right")
    ax.set_xlabel("transformer layer", color=INK2)
    ax.set_ylabel("attention density on the object\n(3x3 window / uniform)", color=INK2)
    ax.set_xticks(range(0, L, 2))
    ax.set_facecolor(SURFACE)
    ax.grid(True, color="#e6e5df", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
    ax.tick_params(colors=INK2)
    ax.legend(fontsize=10, labelcolor=INK2, frameon=False)
    ax.set_title("WRIST view: object or destination?\n"
                 "objects re-projected per frame with the moving wrist camera",
                 fontsize=12.5, color=INK1, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "wrist5_object_vs_dest.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print("[written] wrist5_object_vs_dest.png")


if __name__ == "__main__":
    main()
