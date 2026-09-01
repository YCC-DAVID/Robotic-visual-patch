#!/usr/bin/env python3
"""FastWAM · 分阶段 influence 热图(2×3 小多组)。

把一条 clean rollout 的 T=10 个 replan 帧按夹爪开合 + z 走向切成 6 个阶段,
每格画该阶段 78 个合法锚点的 influence 空间分布,六格共用同一色标 ⇒
一眼看出「哪个阶段对动作最敏感」。阶段值用**每帧均值**(按帧数归一,可比)。

阶段(从 A_clean 的夹爪通道 dim6 两个翻转点 + z 走向读出,非猜):
    approach 0-2 | grasp 3 | lift+move 4-5 | descend-place 6-7 | release 8 | retreat 9

用法: /home/user1/miniconda3/envs/openpi-libero/bin/python probe/fw_phase_figs.py
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
OBJ = {"bowl": (-0.098, -0.009), "plate": (0.062, -0.009)}
EX = 10

# (标题, 帧列表)  —— 与轨迹阶段一一对应
PHASES = [
    ("① approach / descend to bowl", [0, 1, 2]),
    ("② grasp (close gripper)",      [3]),
    ("③ lift + move to plate",       [4, 5]),
    ("④ descend to place",           [6, 7]),
    ("⑤ release (open gripper)",     [8]),
    ("⑥ retreat",                    [9]),
]


def fig_attn_phase():
    """分阶段 attention 热图,与 fig_influence_phase 对照:
    influence 随阶段锐化+迁移;attention 平且几乎不动 ⇒ attention 不追踪动作敏感区。"""
    za = np.load(OUT / "fw_attn.npz", allow_pickle=True)
    zs = np.load(OUT / "fw_scan.npz", allow_pickle=True)
    zc = np.load(OUT / "fw_anchor_cells.npz", allow_pickle=True)
    attn = za["attn"]                                   # [T,L,Sq,Sk] head-summed
    noun_idx = [int(v) for v in za["noun_idx"]]
    imgs = za["agentview"]                              # [T,256,256,3] 正立
    gh, gw, bc = int(za["grid_h"]), int(za["grid_w"]), int(za["base_cols"])
    T, L, Sq, Sk = attn.shape
    MIDL = list(range(8, 22))                           # 中层带:跨注意力语义最干净
    # influence 最强锚点作固定参照
    Ac, Ap = zs["A_clean"], zs["A_patched"]
    Iv = np.linalg.norm((Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2), axis=2).sum(1) * 50.0
    idx = zs["anchor_idx"]; gmax = int(Iv.argmax())
    assert np.array_equal(zc["anchor_idx"], idx), "fw_anchor_cells 与 fw_scan 锚点顺序不一致"
    gcen = zc["centroid"][gmax]
    gx = 256.0 - float(gcen[1]) * 256.0 / 224.0         # #1007 质心 → 正立像素
    gy = float(gcen[0]) * 256.0 / 224.0

    # 每阶段代表帧(取阶段中间那一帧作底图)
    reps = [1, 3, 5, 7, 8, 9]

    def noun_map(frames):                               # 模型帧 base 7×7,重归一
        col = attn[frames][:, MIDL][:, :, :, noun_idx].sum(-1).mean(0).mean(0)  # [Sq]
        b = col.reshape(gh, gw)[:, :bc]
        return b / (b.sum() + 1e-9)

    def entropy(b):
        p = b.flatten() + 1e-12; p = p / p.sum(); return float(-(p * np.log(p)).sum())

    pmaps = [noun_map(fr) for _, fr in PHASES]
    vmax = max(float(b.max()) for b in pmaps)           # 共用色标 ⇒ 平就显平
    unif = 1.0 / (gh * bc)

    fig = plt.figure(figsize=(13.2, 8.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, wspace=.06, hspace=.24, left=.02, right=.90,
                          top=.855, bottom=.03)
    for pi, ((title, fr), b) in enumerate(zip(PHASES, pmaps)):
        ax = fig.add_subplot(gs[pi // 3, pi % 3])
        ax.imshow(imgs[reps[pi]], extent=[0, 256, 256, 0])
        disp = b[:, ::-1]                               # 模型帧 → 正立(翻列)
        up = np.kron(disp, np.ones((256 // gh + 1, 256 // bc + 1)))[:256, :256]
        im = ax.imshow(up, extent=[0, 256, 256, 0], cmap="hot", alpha=.5,
                       vmin=0, vmax=vmax, interpolation="nearest")
        pr, pc = np.unravel_index(int(b.argmax()), b.shape)
        hy = (pr + .5) * 256 / gh; hx = ((bc - 1 - pc) + .5) * 256 / bc
        ax.plot(hx, hy, "s", ms=19, mfc="none", mec="#33e0c0", mew=2.4, zorder=5)
        ax.plot(gx, gy, "X", ms=13, mfc="#e040fb", mec="white", mew=1.3, zorder=6)
        ax.set_title(f"{title}  (frames {fr[0]}"
                     f"{'-' + str(fr[-1]) if len(fr) > 1 else ''})\n"
                     f"entropy {entropy(b):.2f} / max {np.log(gh * bc):.2f} · "
                     f"peak {b.max() / unif:.1f}× uniform", fontsize=9.5, color=INK2, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
    cax = fig.add_axes([.915, .12, .016, .66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("bowl+plate attention (re-normalized over 49 base cells)", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=INK2, labelsize=8); cb.outline.set_visible(False)
    fig.suptitle("FastWAM · per-phase text(bowl/plate)→video attention  "
                 "(mid layers L8-21, shared scale, on each phase's own frame)",
                 fontsize=13, color=INK, y=.955)
    fig.text(.02, .905, "teal square = attention peak cell · magenta X = influence-max #"
             f"{int(idx[gmax])} · attention stays diffuse (≈uniform) and barely moves — "
             "unlike influence, which sharpens & migrates", fontsize=9.5, color=INK2)
    f = OUT / "fig_fw_phase_attention.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE); plt.close(fig)
    print(f"[written] {f}  vmax={vmax:.3f}")
    for (title, fr), b in zip(PHASES, pmaps):
        pr, pc = np.unravel_index(int(b.argmax()), b.shape)
        print(f"  {title:32s} entropy {entropy(b):.2f}  peak ({pr},{pc}) {b.max()/unif:.1f}x uniform")


def fig_attn_phase_plane():
    """分阶段 attention 的**纯平面**版(与 fig_fw_phase_influence 同款散点布局):
    每个锚点按它落在的 7×7 token 格子的 attention 值上色 ⇒ 与 influence 图逐点可比。
    注意:同格锚点同色(32px 格子是架构分辨率天花板),色块状是**如实呈现**,不是 bug。"""
    za = np.load(OUT / "fw_attn.npz", allow_pickle=True)
    zs = np.load(OUT / "fw_scan.npz", allow_pickle=True)
    zc = np.load(OUT / "fw_anchor_cells.npz", allow_pickle=True)
    attn = za["attn"]
    noun_idx = [int(v) for v in za["noun_idx"]]
    gh, gw, bc = int(za["grid_h"]), int(za["grid_w"]), int(za["base_cols"])
    MIDL = list(range(8, 22))
    aw, idx = zs["anchor_world"], zs["anchor_idx"]
    assert np.array_equal(zc["anchor_idx"], idx)
    cell = zc["cell"].astype(int)                       # [M,2] 模型帧 (row,col)

    Ac, Ap = zs["A_clean"], zs["A_patched"]
    Iv = np.linalg.norm((Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2), axis=2).sum(1) * 50.0
    gmax = int(Iv.argmax())                             # influence 全局最强 #1007

    def noun_map(frames):
        col = attn[frames][:, MIDL][:, :, :, noun_idx].sum(-1).mean(0).mean(0)
        b = col.reshape(gh, gw)[:, :bc]
        return b / (b.sum() + 1e-9)

    pmaps = [noun_map(fr) for _, fr in PHASES]
    avals = [b[cell[:, 0], cell[:, 1]] for b in pmaps]  # 每锚点的格子 attention
    vmax = max(float(v.max()) for v in avals)
    norm = Normalize(0.0, vmax)
    unif = 1.0 / (gh * bc)
    TEAL = LinearSegmentedColormap.from_list(
        "teal", ["#eef7f6", "#c9e9e3", "#93d4c8", "#4db3a4", "#1f7a6e", "#0b4a43"])

    fig = plt.figure(figsize=(13.2, 8.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, wspace=.13, hspace=.28, left=.035, right=.905,
                          top=.825, bottom=.055)
    for pi, ((title, fr), b, av) in enumerate(zip(PHASES, pmaps, avals)):
        ax = fig.add_subplot(gs[pi // 3, pi % 3])
        ax.scatter(aw[:, 1], aw[:, 0], s=78, c=av, cmap=TEAL, norm=norm,
                   edgecolor="white", linewidth=.5, zorder=3)
        for nm, (ox, oy) in OBJ.items():
            ax.plot(oy, ox, "o", ms=9, mfc="none", mec="#2a78d6", mew=2.0, zorder=6)
        am = int(av.argmax())                           # 锚点池内 attention 最强
        ax.plot(aw[am, 1], aw[am, 0], "o", ms=15, mfc="none", mec=INK, mew=1.8, zorder=5)
        ax.plot(aw[gmax, 1], aw[gmax, 0], "X", ms=12, mfc="#e040fb",
                mec="white", mew=1.2, zorder=6)
        ax.invert_xaxis(); ax.set_aspect("equal")
        ax.set_title(f"{title}\nmap peak {b.max()/unif:.1f}× uniform · "
                     f"pool max only {av.max()/unif:.1f}×",
                     fontsize=9.5, color=INK2, pad=6)
        ax.tick_params(colors=INK2, labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    cax = fig.add_axes([.925, .12, .016, .66])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=TEAL), cax=cax)
    cb.set_label("bowl+plate attention of the anchor's token cell "
                 "(re-normalized over 49 base cells)", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("FastWAM · per-phase attention on the SAME world-plane layout as the "
                 "influence figure  (mid layers L8-21, shared scale)",
                 fontsize=13, color=INK, y=.955)
    fig.text(.035, .893, "black ring = pool's strongest attention anchor · magenta X = "
             f"influence-max #{int(idx[gmax])} · same-cell anchors share one color "
             "(32px token cell = architectural ceiling)\nattention's hot cell sits OFF the "
             "anchor pool in all 6 phases (①-④ robot/bowl area, ⑤-⑥ image corner — suspected "
             "sink); on the pool it is near-uniform (≤1.5×) and static — unlike influence",
             fontsize=9.5, color=INK2, va="top")
    f = OUT / "fig_fw_phase_attention_plane.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE)
    plt.close(fig)
    print(f"[written] {f}  vmax={vmax:.4f}")
    for (title, fr), b, av in zip(PHASES, pmaps, avals):
        am = int(av.argmax())
        print(f"  {title:32s} pool-max #{int(idx[am])} ({aw[am,0]:+.2f},{aw[am,1]:+.2f}) "
              f"{av[am]/unif:.1f}x uniform  spread {av.max()/max(av.min(),1e-9):.2f}x")


def main():
    z = np.load(OUT / "fw_scan.npz", allow_pickle=True)
    Ac, Ap = z["A_clean"], z["A_patched"]
    aw, idx = z["anchor_world"], z["anchor_idx"]
    M = Ap.shape[0]
    # 逐帧 influence [M,T](执行段前缀求和,平移 L2,mm)
    v = (Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2)
    perT = np.linalg.norm(v, axis=2) * 50.0

    # 每阶段:锚点 × 该阶段各帧的均值 → [M]
    pmaps = [perT[:, fr].mean(1) for _, fr in PHASES]
    vmax = max(float(p.max()) for p in pmaps)
    norm = Normalize(0.0, vmax)

    fig = plt.figure(figsize=(13.2, 8.0), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, wspace=.13, hspace=.28, left=.035, right=.905,
                          top=.855, bottom=.055)
    for pi, ((title, fr), pm) in enumerate(zip(PHASES, pmaps)):
        ax = fig.add_subplot(gs[pi // 3, pi % 3])
        # 点位固定,颜色=该阶段 influence;白描边分隔重叠点
        ax.scatter(aw[:, 1], aw[:, 0], s=78, c=pm, cmap=RAMP, norm=norm,
                   edgecolor="white", linewidth=.5, zorder=3)
        for nm, (ox, oy) in OBJ.items():
            ax.plot(oy, ox, "o", ms=9, mfc="none", mec="#2a78d6", mew=2.0, zorder=6)
        # 该阶段自己的最强锚点
        am = int(pm.argmax())
        ax.plot(aw[am, 1], aw[am, 0], "o", ms=15, mfc="none", mec=INK, mew=1.8, zorder=5)
        ax.invert_xaxis()
        ax.set_aspect("equal")
        ax.set_title(f"{title}   (frames {fr[0]}"
                     f"{'-' + str(fr[-1]) if len(fr) > 1 else ''})\n"
                     f"phase mean {pm.mean():.1f} mm · peak #{int(idx[am])} {pm[am]:.1f} mm",
                     fontsize=9.5, color=INK2, pad=6)
        ax.tick_params(colors=INK2, labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    cax = fig.add_axes([.925, .12, .016, .66])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=cax)
    cb.set_label("influence per frame (mm, translation)", color=INK2, fontsize=9.5)
    cb.ax.tick_params(colors=INK2, labelsize=8)
    cb.outline.set_visible(False)
    fig.suptitle("FastWAM · per-phase influence of a random probe patch  "
                 "(same 78 legal positions, shared color scale)",
                 fontsize=13.5, color=INK, y=.955)
    fig.text(.035, .905, "black ring = each phase's own strongest position · "
             "blue rings = bowl / plate · placement (④⑤) is ~3× more sensitive "
             "than approach/grasp/transport", fontsize=9.5, color=INK2)
    f = OUT / "fig_fw_phase_influence.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE)
    plt.close(fig)
    print(f"[written] {f}  vmax={vmax:.1f} mm")
    for (title, fr), pm in zip(PHASES, pmaps):
        print(f"  {title:32s} frames {str(fr):9s} mean {pm.mean():5.1f}  max {pm.max():5.1f}")


if __name__ == "__main__":
    main()
    if (OUT / "fw_attn.npz").is_file():
        fig_attn_phase()
        fig_attn_phase_plane()
    else:
        print("[skip] fw_attn.npz 不在,跳过分阶段 attention 图")
