#!/usr/bin/env python3
"""FastWAM · 跨任务(换 destination)influence 对比图(1×3)。

同一批 78 个合法锚点、同一张随机探针纹理,只换任务目标:
    bowl→plate / bottle→rack / bowl→cabinet
共用色标 ⇒ 一眼比较「influence 最强点跟不跟着目标走、在不在近机器人侧」。

已知口径注意(写进图注):
  - rack 任务 clean rollout 在 10 个 replan 帧窗口内没完成(success=False),
    只覆盖 approach+grasp 两个阶段 ⇒ 量级偏低是覆盖问题,不可与另两任务直接比大小。
  - 锚点池是围绕 bowl/plate 加密采的,距 rack 最近 48.8 cm ⇒
    「近 rack」假设在该池上不可检验,只能看方向趋势。

用法: /home/user1/miniconda3/envs/openpi-libero/bin/python probe/fw_crosstask_fig.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402

REPO = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = REPO / "probe" / "out"
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list(
    "amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
EX = 10
ROBOT_X, CAM_X = -0.66, 0.659          # 机器人基座 / agentview 相机的世界 x

# (面板标题, npz, (被操作物名, xy), (目标名, xy), 覆盖备注)
TASKS = [
    ("bowl → plate",   "fw_scan.npz",
     ("bowl",   (-0.098, -0.009)), ("plate",   (0.062, -0.009)),
     "full cycle (grasp+release)"),
    ("bottle → rack",  "fw_scan_rack.npz",
     ("bottle", (-0.194, -0.037)), ("rack",    (-0.267, -0.251)),
     "approach+grasp only (unfinished in 10-frame window)"),
    ("bowl → cabinet", "fw_scan_cabinet.npz",
     ("bowl",   (-0.100,  0.010)), ("cabinet", (0.040, -0.234)),
     "full cycle (grasp+release)"),
]


def influence(z):
    Ac, Ap = z["A_clean"], z["A_patched"]
    v = (Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2)
    return np.linalg.norm(v, axis=2).sum(1) * 50.0


def main():
    data = []
    for title, f, src, tgt, note in TASKS:
        z = np.load(OUT / f, allow_pickle=True)
        data.append((title, influence(z), z["anchor_world"][:, :2],
                     z["anchor_idx"], src, tgt, note))
    vmax = max(float(d[1].max()) for d in data)
    norm = Normalize(0.0, vmax)

    fig = plt.figure(figsize=(13.6, 5.9), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, wspace=.14, left=.045, right=.905,
                          top=.73, bottom=.10)
    from matplotlib.patches import Circle
    for pi, (title, I, aw, idx, (sn, sxy), (tn, txy), note) in enumerate(data):
        ax = fig.add_subplot(gs[0, pi])
        # 以 destination 为圆心的同心圆(10–50 cm 淡环作标尺):
        for r in (0.1, 0.2, 0.3, 0.4, 0.5):
            ax.add_patch(Circle((txy[1], txy[0]), r, fill=False, ls="--", lw=.7,
                                ec="#d4d1cb", zorder=1))
            ax.annotate(f"{int(r*100)}", (txy[1], txy[0] + r), ha="center", va="bottom",
                        fontsize=6.5, color="#b3afa8", zorder=1, annotation_clip=True)
        # 两个关键环:穿过 I-max 的环(橙实线)vs 池内离目标最近的环(蓝细线)
        d_tgt_all = np.linalg.norm(aw - np.array(txy), axis=1)
        g0 = int(I.argmax())
        r_imax, r_near = float(d_tgt_all[g0]), float(d_tgt_all.min())
        ax.add_patch(Circle((txy[1], txy[0]), r_near, fill=False, ls="-", lw=1.0,
                            ec="#2a78d6", alpha=.85, zorder=2))
        ax.add_patch(Circle((txy[1], txy[0]), r_imax, fill=False, ls="-", lw=1.5,
                            ec="#eb6834", zorder=2))
        ax.text(.02, .022, f"orange ring through I-max: {r_imax*100:.0f} cm",
                transform=ax.transAxes, fontsize=7.5, color="#b8461f", va="bottom", zorder=4)
        ax.text(.02, .082, f"blue ring = nearest legal: {r_near*100:.0f} cm",
                transform=ax.transAxes, fontsize=7.5, color="#2a78d6", va="bottom", zorder=4)
        ax.scatter(aw[:, 1], aw[:, 0], s=74, c=I, cmap=RAMP, norm=norm,
                   edgecolor="white", linewidth=.5, zorder=3)
        # 被操作物(蓝圈)与目标(金星)
        ax.plot(sxy[1], sxy[0], "o", ms=10, mfc="none", mec="#2a78d6",
                mew=2.0, zorder=6)
        ax.annotate(sn, (sxy[1], sxy[0]), textcoords="offset points",
                    xytext=(7, 6), fontsize=8.5, color="#2a78d6")
        ax.plot(txy[1], txy[0], "*", ms=17, mfc="#f2b736", mec=INK,
                mew=.9, zorder=6)
        ax.annotate(tn, (txy[1], txy[0]), textcoords="offset points",
                    xytext=(7, 6), fontsize=8.5, color=INK)
        # 该任务 influence 最强锚点
        g = int(I.argmax())
        ax.plot(aw[g, 1], aw[g, 0], "o", ms=16, mfc="none", mec=INK,
                mew=1.9, zorder=5)
        d_tgt = float(np.hypot(*(aw[g] - np.array(txy)))) * 100
        ax.set_xlim(.34, -.34); ax.set_ylim(-.36, .30)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nI-max #{int(idx[g])} = {I[g]:.0f} mm · "
                     f"{d_tgt:.0f} cm from {tn}\n{note}",
                     fontsize=9.5, color=INK2, pad=6)
        ax.tick_params(colors=INK2, labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        # 机器人在下(x=-0.66),相机在上(x=+0.66):所有面板同一几何
        ax.annotate("camera side (x=+0.66) ↑", (.5, .985), xycoords="axes fraction",
                    ha="center", va="top", fontsize=8, color="#9a4a12")
        ax.set_xlabel("robot side (x=−0.66) ↓", fontsize=8, color=INK2)

    cax = fig.add_axes([.925, .16, .016, .58])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=cax)
    cb.set_label("total influence over 10 frames (mm, translation)",
                 color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("FastWAM · same 78 positions, same probe patch, three destinations",
                 fontsize=13.5, color=INK, y=.965)
    fig.text(.045, .935,
             "black ring = each task's influence-max · gold star = destination · blue ring = grasped object · "
             "dashed rings = distance to destination (10–50 cm)\n"
             "hot zone shifts with the destination (rank corr with −dist-to-target: plate +0.88, "
             "cabinet +0.66 rotated to cabinet's −y side; rack untestable, pool ≥49 cm away)\n"
             "side hypotheses (robot-near / camera-near) are NOT testable here: 75/78 legal anchors "
             "sit on the +x half — any I-max lands there by pool composition alone",
             fontsize=9.5, color=INK2, va="top")
    f = OUT / "fig_fw_crosstask_influence.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE)
    plt.close(fig)
    print(f"[written] {f}  shared vmax={vmax:.1f} mm")
    for title, I, aw, idx, _, (tn, txy), _ in data:
        g = int(I.argmax())
        print(f"  {title:14s} I-max #{int(idx[g])} ({aw[g,0]:+.2f},{aw[g,1]:+.2f}) "
              f"{I[g]:.0f} mm  d({tn})={np.hypot(*(aw[g]-np.array(txy)))*100:.0f} cm")


if __name__ == "__main__":
    main()
