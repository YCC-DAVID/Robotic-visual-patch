#!/usr/bin/env python3
"""π0.5 · 跨任务梯度代理验证图(与 FastWAM fig_grad_cross_task 同款)。
三任务 plate/rack/cabinet:上排 FD 全局 influence、下排初始帧一次 backward 的梯度。

输入(每任务):
    s2f_actions_{task}.npz        FD 原始动作(s2_scan_actions.py)
    s2f_scan_obs_{task}.npz       clean+patched 观测(贴纸掩码用)
    grad/g0_grad_f0_{task}.npz    帧0梯度(g0e_grad_task.py)
plate 复用 s2f_actions.npz / s2f_scan_obs.npz。

用法: /home/user1/miniconda3/envs/openpi-libero/bin/python pi05probe/g0f_crosstask_fig.py
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
RAMP = LinearSegmentedColormap.from_list(
    "amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
TEAL = LinearSegmentedColormap.from_list(
    "teal", ["#eef7f6", "#c9e9e3", "#93d4c8", "#4db3a4", "#1f7a6e", "#0b4a43"])
EX = 5
# 目标世界坐标(x,y)。plate/rack/cabinet
TASKS = [
    ("plate",   "put_the_bowl_on_the_plate",        "s2f_actions.npz",         "s2f_scan_obs.npz",         (0.062, -0.009)),
    ("rack",    "put_the_wine_bottle_on_the_rack",  "s2f_actions_rack.npz",    "s2f_scan_obs_rack.npz",    (-0.267, -0.251)),
    ("cabinet", "put_the_bowl_on_top_of_the_cabinet", "s2f_actions_cabinet.npz", "s2f_scan_obs_cabinet.npz", (0.040, -0.234)),
]


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def influence(za):
    Ac, Ap = za["A_clean"], za["A_patched"]
    v = (Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2)
    return np.linalg.norm(v, axis=2).sum(1) * 50.0


def pooled(zo, gmag):
    clean = zo["clean_img224"][0].astype(np.int16)
    M = int(zo["M"])
    S = np.zeros(M)
    for i in range(M):
        m = (np.abs(zo["patched_img224"][i, 0].astype(np.int16) - clean) > 2).any(-1)
        S[i] = float(gmag[m].sum()) if m.any() else 0.0
    return S


def main():
    data = []
    for short, stem, af, of, txy in TASKS:
        za = np.load(OUT / af, allow_pickle=True)
        zo = np.load(OUT / of, allow_pickle=True)
        zg = np.load(GOUT / f"g0_grad_f0_{short}.npz", allow_pickle=True)
        aw = za["anchor_world"][:, :2]
        fd = influence(za)
        S = pooled(zo, zg["gmag"])
        r = spear(fd, S)
        Tn = int(za["T"])
        data.append((short, aw, za["anchor_idx"], fd, S, r, txy, Tn))
        print(f"[{short:8s}] T={Tn}帧  grad_f0 ↔ FD全局 = {r:+.2f}", flush=True)

    fig = plt.figure(figsize=(13.2, 8.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, wspace=.10, hspace=.22, left=.04, right=.985,
                          top=.77, bottom=.04)
    for ci, (short, aw, idx, fd, S, r, txy, Tn) in enumerate(data):
        for ri, (v, cmap, lab) in enumerate([
                (fd, RAMP, f"FD influence (global, {Tn} frames)"),
                (S, TEAL, "gradient @ initial frame (ONE backward)")]):
            ax = fig.add_subplot(gs[ri, ci])
            ax.scatter(aw[:, 1], aw[:, 0], s=64, c=v, cmap=cmap,
                       norm=Normalize(0, float(v.max())), edgecolor="white",
                       linewidth=.5, zorder=3)
            ax.plot(txy[1], txy[0], "*", ms=14, mfc="#f2b736", mec=INK, mew=.8, zorder=6)
            am = int(v.argmax())
            ax.plot(aw[am, 1], aw[am, 0], "o", ms=14, mfc="none", mec=INK, mew=1.8, zorder=5)
            ax.invert_xaxis(); ax.set_aspect("equal")
            ax.set_xlim(.34, -.34); ax.set_ylim(-.34, .32)
            title = f"bowl/bottle → {short}\n{lab}\npeak #{int(idx[am])}"
            if ri == 1:
                title += f" · rank corr {r:+.2f}"
            ax.set_title(title, fontsize=9, color=INK2, pad=5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_alpha(.25)
    corrs = [d[5] for d in data]
    fig.suptitle("π0.5 · gradient proxy · CROSS-TASK  (three destinations)",
                 fontsize=13, color=INK, y=.955)
    fig.text(.04, .89,
             "same 61 fine anchors · same probe patch · own successful rollout per task · "
             "black ring = argmax · gold star = destination\n"
             f"one backward tracks global influence as the goal moves — "
             f"plate {corrs[0]:+.2f} · rack {corrs[1]:+.2f} · cabinet {corrs[2]:+.2f}",
             fontsize=9, color=INK2, va="top")
    f = OUT / "fig_pi05_grad_cross_task.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}")


if __name__ == "__main__":
    main()
