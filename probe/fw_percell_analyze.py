#!/usr/bin/env python3
"""FastWAM per-cell 分析(任务参数化):influence(GT)vs 像素梯度 vs destination attention,
全在原生 7×7 token 网格上。存 fw_percell_scores_{task}.npz 供三任务合图。

  * influence[i]: fw_percell_scan_{task} 的 Δaction(EX=exec_prefix 先求和→取模→按帧求和 ×50)
  * grad[i]     : fw_grad_f0 的 gmag(左相机 224)每格 32×32 块求和
  * attn[i]     : fw_attn_{task} 的 base 7×7 × destination 名词列(noun_idx),层平均、帧0
用法: /home/user1/miniconda3/envs/openpi-libero/bin/python probe/fw_percell_analyze.py --task plate
"""
import argparse
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
NS, CELL, GW = 7, 32, 14
STEM = {"plate": "put_the_bowl_on_the_plate", "rack": "put_the_wine_bottle_on_the_rack",
        "cabinet": "put_the_bowl_on_top_of_the_cabinet"}
DEST_XY = {"plate": (0.062, -0.009), "rack": (-0.267, -0.251), "cabinet": (0.040, -0.234)}


def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def compute(task):
    stem = STEM[task]
    za = np.load(OUT / f"fw_percell_scan_{task}.npz", allow_pickle=True)
    EX = int(za["exec_prefix"]); M = int(za["M"])
    aw = za["anchor_world"][:, :2]; legal = za["anchor_legal"].astype(bool)
    idx = za["anchor_idx"]
    rc = np.array([(int(k) // NS, int(k) % NS) for k in idx])   # index=r*7+c
    v = (za["A_patched"][:, :, :EX, 0:3] - za["A_clean"][None, :, :EX, 0:3]).sum(2)
    fd = np.linalg.norm(v, axis=2).sum(1) * 50.0

    zg = np.load(OUT / "fw_grad_f0.npz", allow_pickle=True)
    gmag = zg[f"{stem}__gmag"][:, :224]
    grad = np.array([gmag[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL].sum() for r, c in rc])

    zt = np.load(OUT / f"fw_attn_{task}.npz", allow_pickle=True)
    attn = zt["attn"]; noun = list(zt["noun_idx"])
    a_base = attn[0][:, :, noun].sum(-1).mean(0)                # (layers,98,nd)→sum noun→mean layers→(98,)
    a_grid = a_base.reshape(NS, GW)[:, :NS]
    att = np.array([a_grid[r, c] for r, c in rc])
    return dict(aw=aw, rc=rc, legal=legal, fd=fd, grad=grad, att=att, EX=EX, M=M)


def panel(ax, rc, q, legal, dc, cmap):
    g = np.full((NS, NS), np.nan)
    for i, (r, c) in enumerate(rc):
        g[int(r), int(c)] = q[i]
    cmap.set_bad(SURFACE)
    ax.imshow(np.ma.masked_invalid(g), cmap=cmap, vmin=0, vmax=float(q.max()),
              origin="upper", interpolation="nearest", zorder=2)
    ill = ~legal
    ax.scatter(rc[ill, 1], rc[ill, 0], marker="x", s=40, c="#2b2a27", linewidths=1.0, zorder=4)
    ax.plot(dc[1], dc[0], "*", ms=15, mfc="#f2b736", mec=INK, mew=.8, zorder=6)
    legi = np.where(legal)[0]
    am = int(legi[int(np.argmax(q[legi]))])
    ax.plot(rc[am, 1], rc[am, 0], "o", ms=16, mfc="none", mec=INK, mew=1.8, zorder=5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_alpha(.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="plate")
    args = ap.parse_args()
    d = compute(args.task)
    aw, rc, legal, fd, grad, att = d["aw"], d["rc"], d["legal"], d["fd"], d["grad"], d["att"]
    dc = rc[int(np.argmin(((aw - np.array(DEST_XY[args.task]))**2).sum(1)))]

    print(f"=== FastWAM · {args.task} · per-cell  M={d['M']}  legal(13cm)={int(legal.sum())}  EX={d['EX']} ===")
    for tag, mask in [("全格", np.ones(d["M"], bool)), ("仅合法格", legal)]:
        if mask.sum() < 3:
            print(f"[{tag}] N={int(mask.sum())}<3 跳过"); continue
        print(f"[{tag}] N={int(mask.sum())}  grad↔FD {spear(fd[mask],grad[mask]):+.3f}  "
              f"attn↔FD {spear(fd[mask],att[mask]):+.3f}")
    np.savez_compressed(OUT / f"fw_percell_scores_{args.task}.npz",
                        anchor_world=aw, rc=rc, legal=legal, influence=fd, gradient=grad, attn=att)

    panels = [("FD influence (GT)", RAMP, fd, None),
              ("pixel gradient (|∇|)", TEAL, grad, spear(fd, grad)),
              ("destination attention", PURP, att, spear(fd, att))]
    fig = plt.figure(figsize=(10.2, 3.9), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, wspace=.10, left=.03, right=.99, top=.72, bottom=.04)
    for ci, (lab, cmap, q, rcorr) in enumerate(panels):
        ax = fig.add_subplot(gs[0, ci])
        panel(ax, rc, q, legal, dc, cmap)
        t = lab if rcorr is None else f"{lab}\ncorr vs FD  {rcorr:+.2f}"
        ax.set_title(t, fontsize=9, color=INK2, pad=4)
    fig.suptitle(f"FastWAM · {args.task} · per-patch on native 7×7 token grid (13cm patch): "
                 "influence GT vs gradient vs attention", fontsize=11.5, color=INK, y=.965)
    fig.text(.03, .88, "pure top-down 7×7 token grid (perspective dropped) · ✗ = illegal cell · "
             "black ring = argmax over LEGAL cells only · gold star = destination",
             fontsize=8.2, color=INK2, va="top")
    f = OUT / f"fig_fw_percell_{args.task}.png"
    fig.savefig(f, dpi=133, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}")


if __name__ == "__main__":
    main()
