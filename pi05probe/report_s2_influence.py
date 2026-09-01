#!/usr/bin/env python3
"""S2 影响力报告:**全轨迹层 + 逐帧层**两层(纯后处理,零 GPU)。

输入 out/s2_actions.npz(原始 7 维动作),输出数字 + 热图。

两层的分工
----------
全轨迹层 = 最终的 influence,用来给候选位置排序(决定 patch 贴哪)。
逐帧层   = 挂在它下面,用来看**最大影响发生在轨迹的哪个相位**。

全轨迹聚合有两种写法,两者都报,比值即"方向一致性"
--------------------------------------------------
每帧先把 executed prefix(前 EX 步)的偏差**求和**成一个向量 v[i,t]
(动作是 delta 位置指令 ⇒ 求和 = 该 chunk 造成的净位移偏差):

    幅度版 I_mag[i] = Σ_t ‖v[i,t]‖      方向盲:先推 +x 再推 −x 与一直推 +x 同分
    系统版 I_sys[i] = ‖Σ_t v[i,t]‖      只剩跨帧同向的部分 = 末端净漂的一阶估计
    coherence[i]    = I_sys / I_mag ∈ [1/T, 1]
                      ≈1    每帧同向,累积按 T 长
                      ≈1/√T 每帧乱跑,累积按 √T 长(与 ε 抖动同构)

⚠️ I_sys 是**一阶估计**,不是真实末端漂移:真跑起来轨迹一分叉,后面的观测就变了。
   真实后果只能由 S4 跑完整 rollout 裁决(规格 D6)。

ε 地板按**同样两种形式**算(否则不同量纲)
------------------------------------------
w[t,j] = Σ_s (A_floor[t,j,s] − A_clean[t,s])   只有 ε 变了的偏差,与 patch 偏差同基准
逐帧地板   = p95_j ‖w[t,j]‖
全轨迹地板 = bootstrap:每帧独立抽一个 j,Σ_t w[t,j_t],取 p95
             ⇒ 因为 i.i.d. 零均值,它的系统版天然按 √T 长,这才是公平对照。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/report_s2_influence.py
"""
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
EX = 5              # executed prefix(replan_steps=5,见 FINDINGS Q4)
MM = 0.05 * 1000    # 平移 action 单位 → mm
NU = NV = 6
NBOOT = 20000
SEED = 20260811


def spearman(a, b):
    def rank(x):
        x = np.asarray(x, np.float64)
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x)); r[order] = np.arange(len(x), dtype=np.float64)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt)); np.add.at(s, inv, r)
        return (s / cnt)[inv]
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def rot_tangent(ref, x):
    """log(R_ref^{-1} R_x) —— 切空间里的旋转偏差,可加(小量下近似合成)。ref,x: (...,3)"""
    sh = ref.shape[:-1]
    a = R.from_rotvec(ref.reshape(-1, 3)); b = R.from_rotvec(x.reshape(-1, 3))
    return (a.inv() * b).as_rotvec().reshape(sh + (3,))


def main():
    d = np.load(OUT / "s2_actions.npz", allow_pickle=True)
    Ac, Ap, Af = d["A_clean"], d["A_patched"], d["A_floor"]
    T, M, NF = int(d["T"]), int(d["M"]), int(d["nfloor"])
    aw, leg, keep = d["anchor_world"], d["anchor_legal"].astype(bool), d["anchor_keepout"]
    aidx, ts, st = d["anchor_idx"], d["ts"], d["clean_state8"]
    lines = []

    def out(s=""):
        print(s, flush=True); lines.append(s)

    out("=" * 104)
    out(f"S2 影响力报告   M={M} 锚点 × T={T} 帧   executed prefix={EX} 步   "
        f"合法锚点={int(leg.sum())}/{M}")
    out(f"  指令={str(d['prompt'])!r}   env 步={ts.tolist()}")
    out("=" * 104)

    # ---------- 每帧的偏差向量(executed prefix 求和)----------
    # patch:v[i,t]   ε:w[t,j]
    v_tr = (Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2)                    # (M,T,3)
    w_tr = (Af[:, :, :EX, 0:3] - Ac[:, None, :EX, 0:3]).sum(2)                    # (T,NF,3)
    v_ro = rot_tangent(np.broadcast_to(Ac[None, :, :EX, 3:6], Ap[:, :, :EX, 3:6].shape),
                       Ap[:, :, :EX, 3:6]).sum(2)                                 # (M,T,3)
    w_ro = rot_tangent(np.broadcast_to(Ac[:, None, :EX, 3:6], Af[:, :, :EX, 3:6].shape),
                       Af[:, :, :EX, 3:6]).sum(2)                                 # (T,NF,3)

    # ---------- 夹爪:翻转 + 余量 ----------
    gc = Ac[:, :EX, 6]                                                            # (T,EX)
    flip = (np.sign(Ap[:, :, :EX, 6]) != np.sign(gc[None]))                       # (M,T,EX)
    flip_f = flip.sum(2)                                                          # (M,T)
    fl_eps = (np.sign(Af[:, :, :EX, 6]) != np.sign(gc[:, None])).sum(2)           # (T,NF)

    # ---------- 全轨迹层 ----------
    Imag_t = np.linalg.norm(v_tr, axis=2).sum(1) * MM          # Σ_t‖v‖   (mm)
    Isys_t = np.linalg.norm(v_tr.sum(1), axis=1) * MM          # ‖Σ_t v‖  (mm)
    Imag_r = np.linalg.norm(v_ro, axis=2).sum(1)               # rad
    Isys_r = np.linalg.norm(v_ro.sum(1), axis=1)               # rad
    coh_t = np.where(Imag_t > 0, Isys_t / np.maximum(Imag_t, 1e-12), 0.0)
    coh_r = np.where(Imag_r > 0, Isys_r / np.maximum(Imag_r, 1e-12), 0.0)
    Igrip = flip_f.sum(1)

    # ---------- ε 地板(同形式)----------
    rng = np.random.RandomState(SEED)
    pick = rng.randint(0, NF, size=(NBOOT, T))
    fi = np.arange(T)[None, :]
    bt = w_tr[fi, pick]                                        # (NBOOT,T,3)
    br = w_ro[fi, pick]
    Fmag_t = np.percentile(np.linalg.norm(bt, axis=2).sum(1), 95) * MM
    Fsys_t = np.percentile(np.linalg.norm(bt.sum(1), axis=1), 95) * MM
    Fmag_r = np.percentile(np.linalg.norm(br, axis=2).sum(1), 95)
    Fsys_r = np.percentile(np.linalg.norm(br.sum(1), axis=1), 95)
    Fcoh_t = float(np.median(np.linalg.norm(bt.sum(1), axis=1)
                             / np.maximum(np.linalg.norm(bt, axis=2).sum(1), 1e-12)))
    Fgrip = np.percentile(fl_eps.sum(0), 95)
    frame_floor_t = np.percentile(np.linalg.norm(w_tr, axis=2), 95, axis=1) * MM   # (T,)
    frame_floor_r = np.percentile(np.linalg.norm(w_ro, axis=2), 95, axis=1)

    out()
    out("-" * 104)
    out("① 全轨迹层(最终 influence):幅度版 vs 系统版,以及方向一致性")
    out("-" * 104)
    out(f"  ε 地板(bootstrap p95,同形式):")
    out(f"    平移  幅度版 Σ‖w‖={Fmag_t:7.2f} mm    系统版 ‖Σw‖={Fsys_t:7.2f} mm    "
        f"(比值 {Fsys_t/Fmag_t:.3f},理论 i.i.d. ≈1/√T={1/np.sqrt(T):.3f})")
    out(f"    旋转  幅度版 Σ‖w‖={Fmag_r:7.4f} rad   系统版 ‖Σw‖={Fsys_r:7.4f} rad   "
        f"(比值 {Fsys_r/Fmag_r:.3f})")
    out(f"    夹爪  全轨迹翻转步数 p95 = {Fgrip:.1f}")
    out(f"  ε 抖动自身的 coherence 中位数 = {Fcoh_t:.3f}  ← patch 的 coherence 要显著高于它才叫'系统性'")
    out()
    hdr = ("  rank anc  world(x,y)   legal keepout      | 平移幅度mm 平移系统mm  coh | "
           "旋转幅度  旋转系统  coh | 夹爪翻转 | 过地板")
    out(hdr)
    order = np.argsort(-Isys_t)
    for r_, i in enumerate(order):
        ok = []
        if Isys_t[i] > Fsys_t: ok.append("平移系统")
        if Imag_t[i] > Fmag_t: ok.append("平移幅度")
        if Isys_r[i] > Fsys_r: ok.append("旋转系统")
        if Igrip[i] > Fgrip: ok.append("夹爪")
        out(f"  {r_+1:4d} #{int(aidx[i]):2d}  ({aw[i][0]:5.2f},{aw[i][1]:5.2f}) "
            f"{str(bool(leg[i])):5s} {str(keep[i])[:12]:12s} | "
            f"{Imag_t[i]:9.2f} {Isys_t[i]:10.2f} {coh_t[i]:5.2f} | "
            f"{Imag_r[i]:8.4f} {Isys_r[i]:9.4f} {coh_r[i]:5.2f} | "
            f"{int(Igrip[i]):8d} | {','.join(ok) if ok else '·'}")

    rel = (Isys_t > Fsys_t) | (Imag_t > Fmag_t) | (Isys_r > Fsys_r) | (Igrip > Fgrip)
    out()
    out(f"  过地板:{int(rel.sum())}/{M} 总计;其中合法 {int((rel & leg).sum())}/{int(leg.sum())}")
    out(f"  排序是否因聚合方式而变:Spearman(幅度版, 系统版) = {spearman(Imag_t, Isys_t):.4f}")
    out(f"    top-5 幅度版 = {[int(aidx[j]) for j in np.argsort(-Imag_t)[:5]]}")
    out(f"    top-5 系统版 = {[int(aidx[j]) for j in np.argsort(-Isys_t)[:5]]}")
    if leg.any():
        bl = np.where(leg)[0][np.argmax(Isys_t[leg])]
        out(f"  最强【合法】锚点 #{int(aidx[bl])} ({aw[bl][0]:.2f},{aw[bl][1]:.2f}): "
            f"系统版 {Isys_t[bl]:.2f} mm vs 地板 {Fsys_t:.2f} mm "
            f"⇒ {'过' if Isys_t[bl] > Fsys_t else '未过'}  (coherence {coh_t[bl]:.2f})")

    # ---------- 逐帧层 ----------
    out()
    out("-" * 104)
    out("② 逐帧层:影响最大发生在哪个相位")
    out("-" * 104)
    disp = np.r_[np.linalg.norm(np.diff(st[:, 0:3], axis=0), axis=1) * 1000, np.nan]
    grip_q = st[:, 6]
    pf = np.linalg.norm(v_tr, axis=2) * MM          # (M,T)
    out("  帧 env步 夹爪qpos 区间位移mm | 逐帧地板mm | 全锚最大mm (锚点) | 过地板锚点数(合法) | "
        "clean夹爪余量")
    for t in range(T):
        bi = int(np.argmax(pf[:, t]))
        n = int((pf[:, t] > frame_floor_t[t]).sum())
        nl = int((pf[leg, t] > frame_floor_t[t]).sum())
        out(f"  {t:2d} {ts[t]:5d} {grip_q[t]:8.4f} {disp[t]:11.1f} | {frame_floor_t[t]:10.2f} | "
            f"{pf[:, t].max():9.2f} (#{int(aidx[bi]):2d}) | {n:14d} ({nl}) | "
            f"{np.abs(gc[t]).min():.4f}")
    out()
    out(f"  逐帧过地板的 (锚点,帧) 组合:{int((pf > frame_floor_t[None]).sum())}/{M*T}"
        f"   其中合法锚点贡献 {int((pf[leg] > frame_floor_t[None]).sum())}")

    # 夹爪翻转的可信度:clean 值离 0 太近时,翻转是数值巧合而非真攻击
    marg = np.abs(gc)
    fmask = flip.any(0)
    out(f"  夹爪 clean 值的绝对值:min={marg.min():.4f} median={np.median(marg):.4f} "
        f"max={marg.max():.4f}")
    if fmask.any():
        tt, ss = np.where(fmask)
        out(f"  发生过翻转的 (帧,步) 的 clean 余量:"
            f"{sorted(set(np.round(marg[tt, ss], 4).tolist()))[:10]}")
        out("    ⚠️ 余量接近 0 的翻转是数值巧合(clean 本就在开合边界),不能算攻击成功。")
    else:
        out("  无任何夹爪翻转。")

    # ---------- 规格 D 的 std 归一化 ----------
    out()
    out("-" * 104)
    out("③ 规格 D 的归一化尺度:std_clean(a) —— 该维在整条轨迹上的正常波动")
    out("-" * 104)
    sd = Ac[:, :EX, :].reshape(-1, 7).std(0)
    names = ["dx", "dy", "dz", "rx", "ry", "rz", "grip"]
    out("  " + "  ".join(f"{n}={s:.4f}" for n, s in zip(names, sd)))
    out(f"  平移三维合成 std = {np.linalg.norm(sd[0:3]):.4f} action = "
        f"{np.linalg.norm(sd[0:3])*MM:.2f} mm/步")
    out("  注:这是'动作本身在轨迹上变化多大',与 ε 地板('同一帧换骰子变多大')是两把不同的尺子。")

    np.savez_compressed(OUT / "s2_influence2.npz",
                        anchor_world=aw, anchor_legal=leg, anchor_keepout=keep, anchor_idx=aidx,
                        v_trans=v_tr, v_rot=v_ro, flip_frame=flip_f,
                        Imag_trans=Imag_t, Isys_trans=Isys_t, coh_trans=coh_t,
                        Imag_rot=Imag_r, Isys_rot=Isys_r, coh_rot=coh_r, Igrip=Igrip,
                        Fmag_trans=Fmag_t, Fsys_trans=Fsys_t, Fmag_rot=Fmag_r, Fsys_rot=Fsys_r,
                        Fgrip=Fgrip, Fcoh_trans=Fcoh_t, reliable=rel,
                        frame_floor_trans=frame_floor_t, frame_floor_rot=frame_floor_r,
                        per_frame_trans=pf, std_clean=sd, ts=ts)

    # ---------- 热图 ----------
    ext = [aw[:, 0].min(), aw[:, 0].max(), aw[:, 1].min(), aw[:, 1].max()]
    panels = [(Imag_t, f"trans_magnitude (mm)  floor={Fmag_t:.1f}", Fmag_t),
              (Isys_t, f"trans_systematic (mm)  floor={Fsys_t:.1f}", Fsys_t),
              (coh_t, f"trans_coherence  eps_median={Fcoh_t:.2f}", Fcoh_t),
              (Isys_r, f"rot_systematic (rad)  floor={Fsys_r:.3f}", Fsys_r),
              (Igrip.astype(float), f"gripper_flips  floor={Fgrip:.1f}", Fgrip)]
    for arr, title, fl in panels:
        fig, ax = plt.subplots(figsize=(5.6, 4.8))
        im = ax.imshow(arr.reshape(NV, NU), origin="lower", extent=ext, aspect="auto",
                       cmap="inferno")
        ax.scatter([-0.098], [-0.009], c="cyan", marker="o", s=90, label="bowl",
                   edgecolors="k", zorder=5)
        ax.scatter([0.062], [-0.009], c="lime", marker="s", s=90, label="plate",
                   edgecolors="k", zorder=5)
        for i in range(M):
            if str(keep[i]):
                ax.plot(aw[i, 0], aw[i, 1], "x", color="white", ms=6, mew=1.3, zorder=4)
            if arr[i] <= fl:
                ax.plot(aw[i, 0], aw[i, 1], ".", color="gray", ms=3, zorder=4)
        ax.set_title(f"S2 {title}\n[x=occludes obj, .=below floor]", fontsize=9)
        ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
        ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        tag = title.split()[0]
        fig.tight_layout(); fig.savefig(OUT / f"s2b_heat_{tag}.png", dpi=120); plt.close(fig)
        out(f"[written] s2b_heat_{tag}.png")

    # 逐帧剖面图:前 6 名锚点 + 地板
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i in np.argsort(-Isys_t)[:6]:
        ax.plot(ts, pf[i], marker="o", ms=3.5,
                label=f"#{int(aidx[i])} {'legal' if leg[i] else 'OCCL'} coh={coh_t[i]:.2f}")
    ax.plot(ts, frame_floor_t, "k--", lw=2, label="per-frame eps floor p95")
    ax.set_xlabel("env step"); ax.set_ylabel("per-frame |Δ translation| (mm)")
    ax.set_title("S2 per-frame influence profile (top-6 by systematic aggregation)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "s2b_frame_profile.png", dpi=120); plt.close(fig)
    out("[written] s2b_frame_profile.png")

    (OUT / "s2_influence2.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT/'s2_influence2.txt'} + s2_influence2.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
