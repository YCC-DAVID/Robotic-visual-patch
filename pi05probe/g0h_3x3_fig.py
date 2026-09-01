#!/usr/bin/env python3
"""π0.5 · 跨任务 3×3:行 = FD influence / 梯度(帧0一次backward)/ destination-token attention,
列 = plate/rack/cabinet。梯度、attention 都标对全局 FD 的秩相关。

输入:s2f_actions_{task}.npz / grad/g0_grad_f0_{task}.npz / grad/g0_attn_task.npz
用法: /home/user1/miniconda3/envs/openpi-libero/bin/python pi05probe/g0h_3x3_fig.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
GOUT = OUT / "grad"
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list("amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
TEAL = LinearSegmentedColormap.from_list("teal", ["#eef7f6", "#c9e9e3", "#93d4c8", "#4db3a4", "#1f7a6e", "#0b4a43"])
PURP = LinearSegmentedColormap.from_list("purp", ["#f2eef7", "#d9c9e8", "#b79bd4", "#8f66b8", "#673f94", "#3f1d63"])
EX = 5
TASKS = [
    ("plate",   "s2f_actions.npz",         (0.062, -0.009)),
    ("rack",    "s2f_actions_rack.npz",    (-0.267, -0.251)),
    ("cabinet", "s2f_actions_cabinet.npz", (0.040, -0.234)),
]


def spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def influence(za):
    v = (za["A_patched"][:, :, :EX, 0:3] - za["A_clean"][None, :, :EX, 0:3]).sum(2)
    return np.linalg.norm(v, axis=2).sum(1) * 50.0


def pooled_grad(zo, gmag):
    clean = zo["clean_img224"][0].astype(np.int16); M = int(zo["M"]); S = np.zeros(M)
    for i in range(M):
        m = (np.abs(zo["patched_img224"][i, 0].astype(np.int16) - clean) > 2).any(-1)
        S[i] = float(gmag[m].sum()) if m.any() else 0.0
    return S


def main():
    za_att = np.load(GOUT / "g0_attn_task.npz", allow_pickle=True)
    rows = []
    for short, af, txy in TASKS:
        za = np.load(OUT / af, allow_pickle=True)
        obf = "s2f_scan_obs.npz" if short == "plate" else f"s2f_scan_obs_{short}.npz"
        zo = np.load(OUT / obf, allow_pickle=True)
        zg = np.load(GOUT / f"g0_grad_f0_{short}.npz", allow_pickle=True)
        aw, idx = za["anchor_world"][:, :2], za["anchor_idx"]
        fd = influence(za)
        S_grad = pooled_grad(zo, zg["gmag"])
        S_att = za_att[f"{short}__S_attn"]
        rows.append((short, aw, idx, txy, fd, S_grad, S_att,
                     spear(fd, S_grad), spear(fd, S_att)))
        print(f"[{short:8s}] grad↔FD {spear(fd,S_grad):+.2f}   attn↔FD {spear(fd,S_att):+.2f}", flush=True)

    ROWS = [("FD influence (global)", RAMP, None),
            ("gradient @ frame 0 (ONE backward)", TEAL, "grad"),
            ("destination-token attention (frame 0)", PURP, "attn")]
    fig = plt.figure(figsize=(13.0, 13.4), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, wspace=.10, hspace=.24, left=.04, right=.985, top=.80, bottom=.03)
    for ri, (rlab, cmap, kind) in enumerate(ROWS):
        for ci, (short, aw, idx, txy, fd, S_grad, S_att, rg, ra) in enumerate(rows):
            ax = fig.add_subplot(gs[ri, ci])
            v = {None: fd, "grad": S_grad, "attn": S_att}[kind]
            ax.scatter(aw[:, 1], aw[:, 0], s=58, c=v, cmap=cmap, norm=Normalize(0, float(v.max())),
                       edgecolor="white", linewidth=.4, zorder=3)
            ax.plot(txy[1], txy[0], "*", ms=13, mfc="#f2b736", mec=INK, mew=.8, zorder=6)
            am = int(v.argmax())
            ax.plot(aw[am, 1], aw[am, 0], "o", ms=13, mfc="none", mec=INK, mew=1.7, zorder=5)
            ax.invert_xaxis(); ax.set_aspect("equal"); ax.set_xlim(.34, -.34); ax.set_ylim(-.34, .32)
            t = f"bowl/bottle → {short}\n{rlab}" if ri == 0 else rlab
            if kind == "grad":
                t += f"  ·  corr {rg:+.2f}"
            elif kind == "attn":
                t += f"  ·  corr {ra:+.2f}"
            ax.set_title(t, fontsize=8.8, color=INK2, pad=4)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_alpha(.25)
    cg = [r[7] for r in rows]; ca = [r[8] for r in rows]
    fig.suptitle("π0.5 · cross-task: FD influence vs gradient proxy vs attention  "
                 "(three destinations, 61 anchors)", fontsize=13, color=INK, y=.965)
    fig.text(.04, .915,
             "black ring = argmax · gold star = destination · gradient & attention both at initial frame\n"
             f"rank corr vs global FD — gradient: plate {cg[0]:+.2f} · rack {cg[1]:+.2f} · cabinet {cg[2]:+.2f}   |   "
             f"attention: plate {ca[0]:+.2f} · rack {ca[1]:+.2f} · cabinet {ca[2]:+.2f}\n"
             "on this near-object pool all three rank similarly (+0.7–0.9); π0.5's destination-token attention "
             "(16×16 SigLIP) is peaked and tracks influence here — unlike FastWAM's diffuse 7×7 attention.\n"
             "gradient leads on plate/cabinet; attention edges it on rack. the pool is a small L near the "
             "objects, so all three share the same “closer to objects = higher” trend",
             fontsize=9, color=INK2, va="top")
    f = OUT / "fig_pi05_3x3_infl_grad_attn.png"
    fig.savefig(f, dpi=133, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}")


if __name__ == "__main__":
    main()
