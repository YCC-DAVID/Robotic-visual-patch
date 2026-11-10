#!/usr/bin/env python3
"""π0.5 · FD influence vs 梯度显著性,同款世界坐标平面图并排(用户点名的对比)。

口径(都在第 8 帧,梯度只算了这一帧):
  - FD influence:‖a_patched[i,8,:EX,0:3] − a_clean[8,:EX,0:3]‖_F × 50 mm
    (与 s2_scan.py dchan 的平移通道一致;a 是 7 维输出动作,ε 与 clean 共享)
  - S_grad:把 G0 的逐像素 gmag(3ε 平均,ε 不敏感 0.98-0.99)在"该锚点贴纸实际覆盖
    的像素"上聚合。掩码 = |patched_img − clean_img| > 2 任意通道(逐字来自渲染差分,
    不用拟合投影)。pooled = 掩码内求和(主口径),raw = 掩码内最大。
  已知口径差(如实标注,不修):梯度对归一化模型输出求导(旋转/夹爪通道不含),
  FD 是反归一化后的 7 维动作;G2 的 delta_max 振幅修正未加;梯度 ε=固定配方,
  FD ε=s2f 存的 eps_shared(梯度图对 ε 不敏感,影响可忽略)。

用法: /home/user1/miniconda3/envs/openpi-libero/bin/python pi05probe/g0_vs_influence_fig.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, LinearSegmentedColormap  # noqa: E402

REPO = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = REPO / "pi05probe" / "out"
GOUT = OUT / "grad"
INK, INK2, SURFACE = "#111111", "#6b6a66", "#fcfcfb"
RAMP = LinearSegmentedColormap.from_list(
    "amber", ["#fdf1e9", "#f9d0b4", "#f4a97c", "#eb6834", "#b8461f", "#78290f"])
TEAL = LinearSegmentedColormap.from_list(
    "teal", ["#eef7f6", "#c9e9e3", "#93d4c8", "#4db3a4", "#1f7a6e", "#0b4a43"])
OBJ = {"bowl": (-0.098, -0.009), "plate": (0.062, -0.009)}
EX, FR = 5, 8


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main():
    za = np.load(OUT / "s2f_actions.npz", allow_pickle=True)
    zo = np.load(OUT / "s2f_scan_obs.npz", allow_pickle=True)
    zg = np.load(GOUT / "g0_gradcheck.npz", allow_pickle=True)
    assert int(zg["frame"]) == FR
    aw = za["anchor_world"][:, :2]
    idx = za["anchor_idx"]
    M = aw.shape[0]

    # FD influence(帧 8 与 16 帧均值两个口径)
    Ac, Ap = za["A_clean"], za["A_patched"]
    d = Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]              # [M,T,EX,3]
    fd_t = np.linalg.norm(d.reshape(M, d.shape[1], -1), axis=2) * 50.0   # [M,T] mm
    fd8, fd_mean = fd_t[:, FR], fd_t.mean(1)

    # 贴纸像素掩码(帧 8)→ 梯度聚合
    clean = zo["clean_img224"][FR].astype(np.int16)
    gmag = zg["gmag_eps"].mean(0)                               # [224,224]
    Sp = np.zeros(M); Sr = np.zeros(M); npx = np.zeros(M, int)
    for i in range(M):
        pat = zo["patched_img224"][i, FR].astype(np.int16)
        m = (np.abs(pat - clean) > 2).any(-1)
        npx[i] = int(m.sum())
        if npx[i]:
            Sp[i] = float(gmag[m].sum())
            Sr[i] = float(gmag[m].max())
    ok = npx > 0
    print(f"[mask] 可见贴纸像素:min={npx[ok].min()} med={int(np.median(npx[ok]))} "
          f"max={npx.max()}  不可见锚点 {int((~ok).sum())}/{M}")

    r8p, r8r = spear(fd8[ok], Sp[ok]), spear(fd8[ok], Sr[ok])
    rmp = spear(fd_mean[ok], Sp[ok])
    print(f"[corr] 帧8: FD↔grad_pooled={r8p:+.2f}  FD↔grad_raw={r8r:+.2f}  "
          f"16帧均值FD↔grad_pooled={rmp:+.2f}")
    # 参照:面积(可见像素数)单独能解释多少
    print(f"[ref ] 帧8 FD↔面积={spear(fd8[ok], npx[ok].astype(float)):+.2f}  "
          f"grad_pooled↔面积={spear(Sp[ok], npx[ok].astype(float)):+.2f}")

    panels = [
        ("FD influence · frame 8", fd8, RAMP, "mm"),
        ("gradient score S_grad_pooled · frame 8", Sp, TEAL, "Σ|∂a/∂x| over sticker px"),
        ("gradient score S_grad_raw (max) · frame 8", Sr, TEAL, "max|∂a/∂x| in sticker"),
    ]
    fig = plt.figure(figsize=(13.6, 5.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, wspace=.15, left=.045, right=.985, top=.76, bottom=.09)
    for pi, (title, v, cmap, unit) in enumerate(panels):
        ax = fig.add_subplot(gs[0, pi])
        vv = np.where(ok, v, np.nan)
        ax.scatter(aw[ok, 1], aw[ok, 0], s=86, c=v[ok], cmap=cmap,
                   norm=Normalize(0, np.nanmax(vv)), edgecolor="white",
                   linewidth=.5, zorder=3)
        for nm, (oxx, oyy) in OBJ.items():
            ax.plot(oyy, oxx, "o", ms=9, mfc="none", mec="#2a78d6", mew=2.0, zorder=6)
            ax.annotate(nm, (oyy, oxx), textcoords="offset points", xytext=(7, 5),
                        fontsize=8, color="#2a78d6")
        am = int(np.nanargmax(vv))
        ax.plot(aw[am, 1], aw[am, 0], "o", ms=16, mfc="none", mec=INK, mew=1.9, zorder=5)
        ax.invert_xaxis(); ax.set_aspect("equal")
        ax.set_title(f"{title}\npeak #{int(idx[am])} ({aw[am,0]:+.2f},{aw[am,1]:+.2f}) · {unit}",
                     fontsize=9.5, color=INK2, pad=6)
        ax.tick_params(colors=INK2, labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("π0.5 · FD influence vs gradient saliency on the SAME 61 fine anchors "
                 "(world plane, frame 8 = placement)", fontsize=13, color=INK, y=.965)
    fig.text(.045, .89,
             f"black ring = each panel's argmax · rank corr (frame 8): FD↔pooled {r8p:+.2f}, "
             f"FD↔raw {r8r:+.2f}; vs 16-frame-mean FD {rmp:+.2f}\n"
             "caveats: gradient is 1 frame / normalized-output translation only / no delta_max "
             "amplitude correction — full G2 aggregation pending user sign-off",
             fontsize=9, color=INK2, va="top")
    f = GOUT / "fig_g0_grad_vs_influence.png"
    fig.savefig(f, dpi=135, facecolor=SURFACE)
    print(f"[written] {f}")
    np.savez_compressed(GOUT / "g0_anchor_scores.npz",
                        anchor_idx=idx, anchor_world=za["anchor_world"],
                        fd8=fd8, fd_mean=fd_mean, S_pooled=Sp, S_raw=Sr,
                        mask_px=npx, spear_fd8_pooled=r8p, spear_fd8_raw=r8r,
                        spear_fdmean_pooled=rmp)


if __name__ == "__main__":
    main()
