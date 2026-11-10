#!/usr/bin/env python3
"""Per-cell 收尾分析 + 出图:influence(GT)vs 像素梯度 vs attention(现有法 mid + rollout)。

四个量都落在同一套 176 个 token 格上,逐格可比(这正是"per-patch = token 网格级"的目的):
  * influence[i]  : FD 反事实,标量口径与 g0h 3×3 图一致(EX=5 先求和→取模→按帧求和 ×50)
  * grad[i]       : g0_grad_f0 的 |∇| 在该格 6cm patch 脚印内求和(与 g0h pooled_grad 一致)
  * attn_mid[i]   : 现有法 a16(dest 词×base256、head 求和、中层 L4-12 平均)按脚印池化
  * attn_roll[i]  : attention rollout a16 按脚印池化
报 Spearman(各量 ↔ influence),全 176 格 + 仅合法格两种口径;出四联图。

用法(py3.8 有 matplotlib):
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/percell_analyze.py --task plate
"""
import argparse
import pathlib
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
GOUT = OUT / "grad"
EX = 5
KEEP = yaml.safe_load((pathlib.Path(__file__).parent / "config" / "scene.yaml").read_text())["keepout"]
LABELS = {"wooden_cabinet_1": "cabinet", "flat_stove_1": "stove", "wine_rack_1": "rack",
          "akita_black_bowl_1": "bowl", "cream_cheese_1": "cheese", "wine_bottle_1": "bottle",
          "plate_1": "plate"}


def draw_table(ax, label=True):
    ax.add_patch(Rectangle((-0.6, -0.5), 1.2, 1.0, fill=False, edgecolor="#c9c7c2", linewidth=.8, zorder=0))
    for name, b in KEEP.items():
        x0, x1 = b["x"]; y0, y1 = b["y"]
        ax.add_patch(Rectangle((y0, x0), y1 - y0, x1 - x0, facecolor="#9a9791",
                     alpha=0.18, edgecolor="#7a7873", linewidth=.5, zorder=1))
        if label:
            ax.text((y0 + y1) / 2, (x0 + x1) / 2, LABELS.get(name, name), fontsize=6,
                    color="#55534e", ha="center", va="center", zorder=2)
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list("amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
TEAL = LinearSegmentedColormap.from_list("teal", ["#eef7f6", "#c9e9e3", "#93d4c8", "#4db3a4", "#1f7a6e", "#0b4a43"])
PURP = LinearSegmentedColormap.from_list("purp", ["#f2eef7", "#d9c9e8", "#b79bd4", "#8f66b8", "#673f94", "#3f1d63"])
BLUE = LinearSegmentedColormap.from_list("blue", ["#eef2fb", "#c7d5f0", "#93b0e0", "#5b82c9", "#2f56a3", "#152f66"])
# destination 世界坐标(与 g0h 一致)
DEST = {"plate": (0.062, -0.009), "rack": (-0.267, -0.251), "cabinet": (0.040, -0.234)}
STEM = {"plate": "put_the_bowl_on_the_plate", "rack": "put_the_wine_bottle_on_the_rack",
        "cabinet": "put_the_bowl_on_top_of_the_cabinet"}


def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def influence(za):
    v = (za["A_patched"][:, :, :EX, 0:3] - za["A_clean"][None, :, :EX, 0:3]).sum(2)   # (M,T,3)
    return np.linalg.norm(v, axis=2).sum(1) * 50.0                                     # (M,)


def cover16(patched0, clean0):
    """patch 脚印(224×224 bool)降采到 16×16 覆盖计数。"""
    m = (np.abs(patched0.astype(np.int16) - clean0.astype(np.int16)) > 2).any(-1)
    return m, m.reshape(16, 14, 16, 14).sum((1, 3)).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="plate")
    args = ap.parse_args()
    za = np.load(OUT / f"percell_actions_{args.task}.npz", allow_pickle=True)
    zo = np.load(OUT / f"percell_obs_{args.task}.npz", allow_pickle=True)
    zg = np.load(GOUT / f"g0_grad_f0_{args.task}.npz", allow_pickle=True)
    zt = np.load(GOUT / f"percell_attn_{args.task}.npz", allow_pickle=True)
    M = int(za["M"])
    aw = za["anchor_world"][:, :2]
    legal = zo["anchor_legal"]; keep = zo["anchor_keepout"]
    a16_mid, a16_roll = zt["a16_mid"], zt["a16_roll"]
    gmag = zg["gmag"]
    clean0 = zo["clean_img224"][0]

    fd = influence(za)
    grad = np.zeros(M); att_mid = np.zeros(M); att_roll = np.zeros(M)
    for i in range(M):
        m, cov = cover16(zo["patched_img224"][i, 0], clean0)
        grad[i] = float(gmag[m].sum()) if m.any() else 0.0
        att_mid[i] = float((cov * a16_mid).sum())
        att_roll[i] = float((cov * a16_roll).sum())

    # ε 地板:floor 样本对(不同 ε 的 clean)按同一标量口径的散布 p95
    Af = za["A_floor"]  # (T,NF,AH,7)
    T, NF = Af.shape[0], Af.shape[1]
    fl = []
    for a in range(NF):
        for b in range(a + 1, NF):
            v = (Af[:, a, :EX, 0:3] - Af[:, b, :EX, 0:3]).sum(1)   # (T,3) per-frame coherent
            fl.append(np.linalg.norm(v, axis=1).sum() * 50.0)
    floor = float(np.percentile(fl, 95))
    above = fd > floor

    leg = legal.astype(bool)
    def report(mask, tag):
        if mask.sum() < 3:
            print(f"[{tag}] 样本<3,跳过"); return
        print(f"[{tag}] N={int(mask.sum())}  "
              f"grad↔FD {spear(fd[mask],grad[mask]):+.3f}  "
              f"attn_mid↔FD {spear(fd[mask],att_mid[mask]):+.3f}  "
              f"attn_roll↔FD {spear(fd[mask],att_roll[mask]):+.3f}", flush=True)

    print(f"=== task={args.task}  M={M}  legal(6cm)={int(leg.sum())}  ε地板={floor:.1f}mm  "
          f"过地板={int(above.sum())}/{M} ===", flush=True)
    report(np.ones(M, bool), "全 176 格")
    report(leg, "仅合法格")
    report(above, "仅过地板格")
    # 攻击选点:各量 argmax 落在哪
    for nm, v in [("influence", fd), ("gradient", grad), ("attn_mid", att_mid), ("attn_roll", att_roll)]:
        i = int(v.argmax())
        print(f"  argmax[{nm:9s}] cell world=({aw[i,0]:+.2f},{aw[i,1]:+.2f}) legal={bool(leg[i])} "
              f"keepout={str(keep[i])[:16]}", flush=True)

    np.savez_compressed(OUT / f"percell_scores_{args.task}.npz",
                        anchor_world=aw, legal=leg, keepout=keep, above_floor=above, floor=floor,
                        influence=fd, gradient=grad, attn_mid=att_mid, attn_roll=att_roll)

    # ---- 四联图(散点在世界坐标,与 g0h 同风格)
    txy = DEST[args.task]
    panels = [("FD influence (GT)", RAMP, fd, None),
              ("pixel gradient (|∇|, pooled)", TEAL, grad, spear(fd, grad)),
              ("attention · mid-band L4-12", PURP, att_mid, spear(fd, att_mid)),
              ("attention · rollout", BLUE, att_roll, spear(fd, att_roll))]
    fig = plt.figure(figsize=(13.2, 3.7), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 4, wspace=.10, left=.03, right=.99, top=.80, bottom=.04)
    legi = np.where(leg)[0]
    cells_rc = zo["anchor_cell"]                                # (M,2) (row,col)
    dc = cells_rc[int(np.argmin(((aw - np.array(txy))**2).sum(1)))]   # 目的地格
    for ci, (lab, cmap, v, rc) in enumerate(panels):
        ax = fig.add_subplot(gs[0, ci])
        g = np.full((16, 16), np.nan)
        for i, (r, c) in enumerate(cells_rc):
            g[int(r), int(c)] = v[i]
        cmap.set_bad(SURFACE)
        ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=0, vmax=float(v.max()),
                  origin="upper", interpolation="nearest", zorder=2)
        ax.scatter(cells_rc[~leg, 1], cells_rc[~leg, 0], marker="x", s=12, c="#2b2a27",
                   linewidths=.6, zorder=4)                    # 非法格 ✗
        ax.plot(dc[1], dc[0], "*", ms=13, mfc="#f2b736", mec=INK, mew=.8, zorder=6)
        am = int(legi[int(np.argmax(v[legi]))])                # 选点只在合法格
        ax.plot(cells_rc[am, 1], cells_rc[am, 0], "o", ms=13, mfc="none", mec=INK, mew=1.7, zorder=5)
        ax.set_aspect("equal")
        t = lab if rc is None else f"{lab}\ncorr vs FD  {rc:+.2f}"
        ax.set_title(t, fontsize=9, color=INK2, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_alpha(.25)
    fig.suptitle(f"π0.5 · {args.task} · per-patch top-down (6cm, 176 token-grid cells): "
                 f"influence GT vs gradient vs attention (2 baselines)",
                 fontsize=12, color=INK, y=.97)
    fig.text(.03, .875, "pure top-down 16×16 token grid (perspective dropped) · ✗ = illegal cell · "
             "black ring = argmax over LEGAL cells only · gold star = destination",
             fontsize=8.4, color=INK2, va="top")
    f = OUT / f"fig_percell_{args.task}.png"
    fig.savefig(f, dpi=133, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}", flush=True)


if __name__ == "__main__":
    main()
