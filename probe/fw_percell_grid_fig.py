#!/usr/bin/env python3
"""FastWAM per-patch 3任务 × 3量 总图(纯 7×7 token 网格俯视,从 fw_percell_scores_{task}.npz)。
行=plate/cabinet/rack,列=influence GT / 像素梯度 / destination attention。标对 influence 的 Spearman。"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/probe/out")
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list("amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
TEAL = LinearSegmentedColormap.from_list("teal", ["#eef7f6", "#c9e9e3", "#93d4c8", "#4db3a4", "#1f7a6e", "#0b4a43"])
PURP = LinearSegmentedColormap.from_list("purp", ["#f2eef7", "#d9c9e8", "#b79bd4", "#8f66b8", "#673f94", "#3f1d63"])
NS = 7
DEST_XY = {"plate": (0.062, -0.009), "rack": (-0.267, -0.251), "cabinet": (0.040, -0.234)}
TASKS = ["plate", "cabinet", "rack"]
COLS = [("influence GT", RAMP, "influence"), ("pixel gradient", TEAL, "gradient"),
        ("destination attention", PURP, "attn")]


def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    fig = plt.figure(figsize=(10.6, 10.6), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, wspace=.08, hspace=.18, left=.06, right=.99, top=.81, bottom=.03)
    summ = []
    for ri, task in enumerate(TASKS):
        d = np.load(OUT / f"fw_percell_scores_{task}.npz", allow_pickle=True)
        aw, rc, leg, fd = d["anchor_world"], d["rc"], d["legal"].astype(bool), d["influence"]
        dc = rc[int(np.argmin(((aw - np.array(DEST_XY[task]))**2).sum(1)))]
        summ.append((task, spear(fd[leg], d["gradient"][leg]), spear(fd[leg], d["attn"][leg])))
        for ci, (lab, cmap, key) in enumerate(COLS):
            v = d[key]
            ax = fig.add_subplot(gs[ri, ci])
            g = np.full((NS, NS), np.nan)
            for i, (r, c) in enumerate(rc):
                g[int(r), int(c)] = v[i]
            cmap.set_bad(SURFACE)
            ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=0, vmax=float(v.max()),
                      origin="upper", interpolation="nearest", zorder=2)
            ax.scatter(rc[~leg, 1], rc[~leg, 0], marker="x", s=38, c="#2b2a27", linewidths=1.0, zorder=4)
            ax.plot(dc[1], dc[0], "*", ms=14, mfc="#f2b736", mec=INK, mew=.8, zorder=6)
            legi = np.where(leg)[0]; am = int(legi[int(np.argmax(v[legi]))])
            ax.plot(rc[am, 1], rc[am, 0], "o", ms=15, mfc="none", mec=INK, mew=1.8, zorder=5)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_alpha(.25)
            if ri == 0:
                ax.set_title(lab, fontsize=9.5, color=INK2, pad=4)
            if ci == 0:
                ax.set_ylabel(f"→ {task}", fontsize=10, color=INK)
            if ci > 0:
                rcorr = spear(fd, v); rleg = spear(fd[leg], v[leg])
                ax.set_title(f"{lab if ri==0 else ''}\nall {rcorr:+.2f} · legal {rleg:+.2f}",
                             fontsize=8.4, color=INK2, pad=4)

    fig.suptitle("FastWAM · per-patch on native 7×7 token grid (13cm patch, 35 cells): "
                 "influence GT vs gradient vs attention", fontsize=12, color=INK, y=.955)
    leg_txt = " | ".join(f"{t}: grad {g:+.2f} · attn {a:+.2f}" for t, g, a in summ)
    fig.text(.06, .90, "rank corr vs influence on LEGAL cells — " + leg_txt +
             "\ngradient beats attention on legal cells in all 3 tasks. pure top-down 7×7 grid · "
             "✗ = illegal cell · black ring = argmax over LEGAL cells · gold star = destination.",
             fontsize=8.6, color=INK2, va="top")
    f = OUT / "fig_fw_percell_3task.png"
    fig.savefig(f, dpi=133, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}")
    for t, g, a in summ:
        print(f"  {t:8s} legal: grad {g:+.3f}  attn {a:+.3f}")


if __name__ == "__main__":
    main()
