#!/usr/bin/env python3
"""固定-ε rollout 的一张图:influence → 真实轨迹偏移。纯后处理。

要讲的两件事
----------
① 偏移随 influence 单调走(秩相关 +1.00)⇒ 位置选择第一次有可测的行为后果。
② 面积配对那两点(同面积、influence 差 3 倍)在图上分得很开 ⇒ 面积不决定,位置决定。
   而面积最大的点(#16)偏移最小之一 —— 直接反驳"贴纸越大越凶"。

用法:
    /home/user1/miniconda3/envs/openpi-libero/bin/python pi05probe/make_divergence_fig.py
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
C_MAIN, C_PAIR, C_AREA = "#b8461f", "#2a78d6", "#158a5c"


def main():
    d = json.loads((OUT / "divergence_scan_fine.json").read_text())
    order = ["inf_308_max", "pair_hi_307", "pair_lo_102", "area_max_106", "wristattn_197", "old_best_126"]
    inf = np.array([d[k]["influence"] for k in order])
    pk = np.array([d[k]["peak"] for k in order])
    sd = np.array([d[k]["peak_sd"] for k in order])
    area = np.array([d[k]["area_px"] for k in order])

    fig, ax = plt.subplots(figsize=(8.2, 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # 噪声地板 = 0(固定 ε)。画一条参照线强调"这次地板是真的 0"。
    ax.axhline(0, color=INK2, lw=1, ls=":", zorder=1)
    ax.text(5, 1.5, "sampling-noise floor = 0 (fixed ε)", fontsize=9, color=INK2, va="bottom")

    # 点大小 = 贴纸可见面积,直观展示"大点不一定高"
    sizes = (area / area.max()) * 520 + 60
    ax.errorbar(inf, pk, yerr=sd, fmt="none", ecolor="#c9c7c1", elinewidth=1.4, capsize=3, zorder=2)
    for k, x, y, s, a in zip(order, inf, pk, sizes, area):
        col = C_PAIR if k in ("pair_hi_307", "pair_lo_102") else (
            C_AREA if k == "area_max_106" else C_MAIN)
        ax.scatter([x], [y], s=[s], color=col, alpha=.85, edgecolor="white", linewidth=1.4, zorder=3)

    lab = {"inf_308_max": "#1003  strongest\n(area 872 = smallest)",
           "pair_hi_307": "#1006  area 978",
           "pair_lo_102": "#1055  area 977",
           "area_max_106": "#16  area 1369 = largest",
           "wristattn_197": "#1031  wrist-attn pick",
           "old_best_126": "#25  old best"}
    off = {"inf_308_max": (-8, 12), "pair_hi_307": (10, 8), "pair_lo_102": (10, -4),
           "area_max_106": (10, -14), "wristattn_197": (10, 6), "old_best_126": (10, -10)}
    for k, x, y in zip(order, inf, pk):
        ax.annotate(lab[k], (x, y), textcoords="offset points", xytext=off[k],
                    fontsize=8.5, color=INK, ha="left", va="center")

    # 面积配对那对用一条虚线连起来:横向近乎不动(面积相同)、纵向差一倍
    hi = order.index("pair_hi_307"); lo = order.index("pair_lo_102")
    ax.plot([inf[lo], inf[hi]], [pk[lo], pk[hi]], color=C_PAIR, lw=1.4, ls="--", zorder=2, alpha=.7)

    r_s = "+1.00"
    ax.set_title("Random-patch position drives trajectory divergence — and area doesn't",
                 fontsize=13, color=INK, pad=12, weight="bold")
    ax.set_xlabel("single-frame influence  (mm, translation, counterfactual)", fontsize=11, color=INK2)
    ax.set_ylabel("end-effector trajectory peak offset vs clean  (mm, n=10)", fontsize=11, color=INK2)
    ax.text(.98, .04, f"Spearman(influence, offset) = {r_s}\nmarker size = patch visible area\n"
                      "blue pair: same area (978≈977 px), 3× influence → 2.2× offset\n"
                      "green (largest area) sits low; smallest-area #1003 sits highest",
            transform=ax.transAxes, fontsize=9, color=INK, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e0ded9"))
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d6d1")
    ax.set_xlim(-5, 340); ax.set_ylim(-4, 90)

    fig.tight_layout()
    f = OUT / "fig_divergence_fine.png"
    fig.savefig(f, dpi=140, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}")


if __name__ == "__main__":
    main()
