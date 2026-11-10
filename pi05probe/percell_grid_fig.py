#!/usr/bin/env python3
"""π0.5 · per-patch 3任务 × 4量 总图(从 percell_scores_{task}.npz 出,纯后处理)。
行=plate/cabinet/rack,列=influence GT / 像素梯度 / attention 现有法 / attention rollout。
每格标对 influence 的 Spearman(全 176 格 & 仅合法格两个数)。"""
import pathlib
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NS = 16
_CW = np.load(OUT / "grad" / "pi05_cell_world.npz", allow_pickle=True)
CELLS, WORLDS = _CW["cells"], _CW["worlds"]     # (176,2) (row,col) 与 (x,y),与 scores 同序
KEEP = yaml.safe_load((pathlib.Path(__file__).parent / "config" / "scene.yaml").read_text())["keepout"]
LABELS = {"wooden_cabinet_1": "cabinet", "flat_stove_1": "stove", "wine_rack_1": "rack",
          "akita_black_bowl_1": "bowl", "cream_cheese_1": "cheese", "wine_bottle_1": "bottle",
          "plate_1": "plate"}


def draw_table(ax, label=False):
    """俯视:横轴=world y、纵轴=world x(与散点一致)。画 keepout 物体足迹=非法区域。"""
    for name, b in KEEP.items():
        x0, x1 = b["x"]; y0, y1 = b["y"]
        ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, facecolor="#9a9791",
                     alpha=0.18, edgecolor="#7a7873", linewidth=.5, zorder=1))
        if label:
            ax.text((y0 + y1) / 2, (x0 + x1) / 2, LABELS.get(name, name), fontsize=5.5,
                    color="#55534e", ha="center", va="center", zorder=2)
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list("amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
TEAL = LinearSegmentedColormap.from_list("teal", ["#eef7f6", "#c9e9e3", "#93d4c8", "#4db3a4", "#1f7a6e", "#0b4a43"])
PURP = LinearSegmentedColormap.from_list("purp", ["#f2eef7", "#d9c9e8", "#b79bd4", "#8f66b8", "#673f94", "#3f1d63"])
BLUE = LinearSegmentedColormap.from_list("blue", ["#eef2fb", "#c7d5f0", "#93b0e0", "#5b82c9", "#2f56a3", "#152f66"])
DEST = {"plate": (0.062, -0.009), "rack": (-0.267, -0.251), "cabinet": (0.040, -0.234)}
TASKS = ["plate", "cabinet", "rack"]
COLS = [("influence GT", RAMP, "influence"), ("pixel gradient", TEAL, "gradient"),
        ("attn · mid L4-12", PURP, "attn_mid"), ("attn · rollout", BLUE, "attn_roll")]


def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    fig = plt.figure(figsize=(13.4, 10.4), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 4, wspace=.08, hspace=.20, left=.055, right=.99, top=.83, bottom=.03)
    summ = []
    for ri, task in enumerate(TASKS):
        d = np.load(OUT / f"percell_scores_{task}.npz", allow_pickle=True)
        aw = d["anchor_world"]; leg = d["legal"].astype(bool); fd = d["influence"]
        txy = DEST[task]
        for ci, (lab, cmap, key) in enumerate(COLS):
            v = d[key]
            ax = fig.add_subplot(gs[ri, ci])
            g = np.full((NS, NS), np.nan)
            for i, (r, c) in enumerate(CELLS):
                g[int(r), int(c)] = v[i]
            cmap.set_bad(SURFACE)
            ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=0, vmax=float(v.max()),
                      origin="upper", interpolation="nearest", zorder=2)
            ill = ~leg
            ax.scatter(CELLS[ill, 1], CELLS[ill, 0], marker="x", s=10, c="#2b2a27",
                       linewidths=.5, zorder=4)                          # 非法格 ✗
            dc = CELLS[int(np.argmin(((WORLDS - np.array(txy))**2).sum(1)))]
            ax.plot(dc[1], dc[0], "*", ms=12, mfc="#f2b736", mec=INK, mew=.7, zorder=6)  # 目的地格
            legi = np.where(leg)[0]
            am = int(legi[int(np.argmax(v[legi]))])       # 选点只在合法格
            ax.plot(CELLS[am, 1], CELLS[am, 0], "o", ms=12, mfc="none", mec=INK, mew=1.6, zorder=5)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_alpha(.22)
            if key != "influence":
                rc_all, rc_leg = spear(fd, v), spear(fd[leg], v[leg])
                ax.set_title(f"{lab}\nall {rc_all:+.2f} · legal {rc_leg:+.2f}",
                             fontsize=8.4, color=INK2, pad=3)
                if ci == 1:
                    summ.append((task, rc_leg, spear(fd[leg], d["attn_mid"][leg]),
                                 spear(fd[leg], d["attn_roll"][leg])))
            elif ci == 0:
                ax.set_title(f"{lab}", fontsize=8.4, color=INK2, pad=3)
            if ci == 0:
                ax.set_ylabel(f"bowl/bottle → {task}", fontsize=10, color=INK)

    fig.suptitle("π0.5 · per-patch on native 16×16 token grid (6cm patch, 176 cells): "
                 "influence GT vs gradient vs attention", fontsize=13, color=INK, y=.975)
    fig.text(.055, .935,
             "panel titles show rank corr vs influence (all 176 cells · legal-only). "
             "gradient beats both attention baselines on legal cells in all 3 tasks.\n"
             "pure top-down 16×16 token grid (perspective dropped) · ✗ = illegal cell (patch off-table / "
             "overlaps object) · black ring = argmax over LEGAL cells only · gold star = destination.",
             fontsize=8.8, color=INK2, va="top")
    f = OUT / "fig_percell_pi05_3task.png"
    fig.savefig(f, dpi=133, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}")
    for t, g, am, ar in summ:
        print(f"  {t:8s} legal: grad {g:+.3f}  attn_mid {am:+.3f}  attn_roll {ar:+.3f}")


if __name__ == "__main__":
    main()
