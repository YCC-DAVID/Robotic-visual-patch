#!/usr/bin/env python3
"""加密合法区扫描的报告:把新的 61 个近距离合法锚点并进原来的候选集,重做核心命题检验。

为什么要加密
----------
原 6×6 网格(x 间距 9 cm / y 间距 13 cm)最近的合法点距 bowl/plate 有 15 cm,
而 attention 的空间衰减在 10 cm 之外就比 influence 快得多:
    10–20 cm  influence 52.6%  attention 21–24%
    20–30 cm  influence 14.5%  attention  3.8–4.6%
也就是**原来测过的合法点全在 attention 已经塌掉的那一侧**。加密扫描把候选挪进它还没塌的区间,
才算对 attention 公平。

判据与口径全部沿用原管线,一个字都没改:
  influence = Σ_t ‖Σ_{k<5}(a_patch − a_clean)[0:3]‖ × 50 mm,3 张纹理平均(report_texture_axis 的 Imag)
  attention = patch cell 覆盖占比加权平均、沿 t 求和(report_b4 的 score),18 层 × 3 种 token 聚合 × 2 视角

用法(纯后处理):
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/report_fine_legal.py
"""
import pathlib

import numpy as np

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
EX, MM, THR, CELL = 5, 0.05 * 1000, 10, 14
NOUNS = ("bowl", "plate", "cabinet", "stove", "wine", "bottle", "rack")
BOWL, PLATE = np.array([-0.098, -0.009]), np.array([0.062, -0.009])
OLD = [("t1", "s2_actions.npz"), ("t2", "s2_actions_t2.npz"), ("t3", "s2_actions_t3.npz")]
NEW = [("t1", "s2f_actions.npz"), ("t2", "s2f_actions_t2.npz"), ("t3", "s2f_actions_t3.npz")]
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def influence(fn):
    z = np.load(OUT / fn, allow_pickle=True)
    Ac, Ap = z["A_clean"], z["A_patched"]
    v = (Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2)          # (M,T,3)
    return np.linalg.norm(v, axis=2).sum(1) * MM, Ac


def coverage(obs_npz, view_is_wrist):
    ck = "clean_wrist224" if view_is_wrist else "clean_img224"
    pk = "patched_wrist224" if view_is_wrist else "patched_img224"
    c, p = obs_npz[ck].astype(np.int16), obs_npz[pk].astype(np.int16)
    m = (np.abs(p - c[None]).max(-1) > THR)
    M, T = m.shape[0], m.shape[1]
    return m.reshape(M, T, 16, CELL, 16, CELL).mean((3, 5))


def score(sal, g):
    num = np.einsum("mtij,tlij->mtl", g, sal)
    den = g.sum((2, 3))[:, :, None]
    per_t = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return per_t.sum(1), (den[:, :, 0] > 0).any(1)


def main():
    # ---------- influence ----------
    Io, Aco = zip(*[influence(f) for _, f in OLD])
    In, Acn = zip(*[influence(f) for _, f in NEW])
    dclean = max(float(np.abs(a - Aco[0]).max()) for a in list(Aco) + list(Acn))
    out("=" * 104)
    out("加密合法区扫描 · 核心命题重测")
    out("=" * 104)
    # 红线:两批 run 的 clean 侧必须可比。ε 逐位相同(已验),clean_img224/state8 逐位相同,
    # 唯一差异来自 EGL 渲染的非确定性:clean_wrist224 有 4/2,408,448 个像素差 1/255,
    # 传到 A_clean 上是 1.99e-3 action = 0.10 mm —— 相对 50–311 mm 的 influence 是 0.03–0.2%。
    # 每批**各自**用自己的 A_clean 算 Δa(批内仍然精确),这里只检查跨批可比性。
    out(f"  [红线] 新旧两批 run 的 A_clean 最大差 = {dclean:.3e} action = {dclean*MM:.3f} mm  "
        f"(ε 逐位相同;差异来自 EGL 渲染 4/2.4M 个像素的 1/255 抖动)")
    assert dclean * MM < 1.0, f"clean 动作跨批差 {dclean*MM:.2f} mm,已不可忽略,合并前必须查清"
    Iold, Inew = np.mean(Io, 0), np.mean(In, 0)

    obo = np.load(OUT / "s2_scan_obs.npz", allow_pickle=True)
    obn = np.load(OUT / "s2f_scan_obs.npz", allow_pickle=True)
    lego = obo["anchor_legal"].astype(bool)
    awo, awn = obo["anchor_world"], obn["anchor_world"]
    ido, idn = obo["anchor_idx"], obn["anchor_idx"]

    # 合并候选池:旧的合法锚点 + 新的(全部合法)
    aw = np.concatenate([awo[lego], awn])
    idx = np.concatenate([ido[lego], idn])
    I = np.concatenate([Iold[lego], Inew])
    isnew = np.concatenate([np.zeros(int(lego.sum()), bool), np.ones(len(Inew), bool)])
    d = np.minimum(np.linalg.norm(aw[:, :2] - BOWL, axis=1), np.linalg.norm(aw[:, :2] - PLATE, axis=1))

    out(f"  合并候选池 = 旧网格合法 {int(lego.sum())} + 加密 {len(Inew)} = {len(I)} 个,全部合法")
    out(f"  距 bowl/plate:旧 min={d[~isnew].min()*100:.1f}cm  新 min={d[isnew].min()*100:.1f}cm")
    out()
    out("-" * 104)
    out("① 加密后合法区的 influence 天花板抬高了多少")
    out("-" * 104)
    bo = int(np.argmax(I[~isnew])); bn = int(np.argmax(I))
    out(f"  旧候选集最强 : #{idx[~isnew][bo]}  ({aw[~isnew][bo,0]:+.2f},{aw[~isnew][bo,1]:+.2f})  "
        f"d={d[~isnew][bo]*100:4.1f}cm  {I[~isnew][bo]:6.1f} mm")
    out(f"  合并后最强   : #{idx[bn]}  ({aw[bn,0]:+.2f},{aw[bn,1]:+.2f})  "
        f"d={d[bn]*100:4.1f}cm  {I[bn]:6.1f} mm   "
        f"⇒ 天花板 ×{I[bn]/I[~isnew][bo]:.2f}  ({'来自加密点' if isnew[bn] else '仍是旧点'})")
    out()
    out("  合并候选池 influence 前 12:")
    out("    rank  锚点   world(x,y)      距物体  influence   来源")
    for r, i in enumerate(np.argsort(-I)[:12]):
        out(f"    {r+1:4d}  #{idx[i]:<5d} ({aw[i,0]:+.2f},{aw[i,1]:+.2f})  {d[i]*100:5.1f}cm  "
            f"{I[i]:7.1f} mm   {'加密' if isnew[i] else '旧网格'}")
    out()
    out("  按距物体分箱(合并池内,归一到池内最大值):")
    out("    距离        n  | influence 均值 | influence 最大")
    for lo, hi in [(.12, .15), (.15, .18), (.18, .21), (.21, .26), (.26, .35), (.35, 1)]:
        m = (d >= lo) & (d < hi)
        if m.sum():
            out(f"    {lo*100:3.0f}-{hi*100:3.0f} cm  {int(m.sum()):3d} | "
                f"{100*I[m].mean()/I.max():13.1f}% | {100*I[m].max()/I.max():13.1f}%")

    # ---------- attention ----------
    az = np.load(OUT / "attn_traj_put_the_bowl_on_the_plate.npz", allow_pickle=True)
    A = az["bowl_plate_orig__attn"].astype(np.float64)            # (T,L,Z,V,16,16)
    pieces = [str(x) for x in az["bowl_plate_orig__pieces"]]
    views = [str(v) for v in az["views"]]
    nouns = [i for i, p in enumerate(pieces) if any(n in p.lower() for n in NOUNS)]
    L = A.shape[1]

    out()
    out("-" * 104)
    out("② 核心命题:合法候选池里,attention 选的点是不是 influence 最大的点")
    out("-" * 104)
    out(f"  influence 首选 = #{idx[bn]} ({aw[bn,0]:+.2f},{aw[bn,1]:+.2f}) {I[bn]:.1f} mm")
    rnd_hit, rnd_got = None, None
    for vi, vname in enumerate(views):
        wr = "wrist" in vname.lower()
        g = np.concatenate([coverage(obo, wr)[lego], coverage(obn, wr)], axis=0)
        sals = {"max": A[:, :, :, vi].max(2), "sum": A[:, :, :, vi].sum(2),
                "noun": A[:, :, :, vi][:, :, nouns].sum(2)}
        S, valid = {}, None
        for k, s in sals.items():
            S[k], valid = score(s, g)
        cand = valid
        best = int(np.where(cand)[0][np.argmax(I[cand])])
        rnd_hit, rnd_got = 100 / cand.sum(), 100 * (I[cand] / I[best]).mean()
        got, hit, newpick = [], 0, 0
        rows = []
        for l in range(L):
            cells = []
            for k in ("max", "sum", "noun"):
                s = S[k][:, l].copy(); s[~cand] = -np.inf
                i = int(s.argmax()); got.append(I[i] / I[best]); hit += (i == best)
                newpick += isnew[i]
                cells.append(f"#{idx[i]:<5d}{100*I[i]/I[best]:4.0f}%{'*' if isnew[i] else ' '}")
            rows.append(f"    {l:3d} | " + " | ".join(cells))
        out(f"\n  视角 {vname}   候选 {int(cand.sum())} 个(该视角可见)  "
            f"influence 首选 #{idx[best]} {I[best]:.1f} mm")
        out("     层 | max            | sum            | noun            (* = 加密新点)")
        for r in rows:
            out(r)
        out(f"    命中 {hit}/{3*L} = {100*hit/(3*L):.1f}%(随机 {rnd_hit:.1f}%)   "
            f"拿到均值 {100*np.mean(got):.1f}%(随机 {rnd_got:.1f}%)   "
            f"选中加密新点 {newpick}/{3*L}")

    # ---------- 免费基线 ----------
    out()
    out("-" * 104)
    out("③ 两条不用跑模型的基线,在同一个合并池上")
    out("-" * 104)
    px = np.concatenate([obo["visible_px"].mean(1)[lego], obn["visible_px"].mean(1)])
    for nm, sc in [("离最近任务物体最近", -d), ("贴纸可见面积最大", px)]:
        i = int(np.argmax(sc))
        out(f"  {nm:20s} → #{idx[i]:<5d} ({aw[i,0]:+.2f},{aw[i,1]:+.2f}) d={d[i]*100:4.1f}cm  "
            f"拿到 {100*I[i]/I[bn]:5.1f}%  {'✅命中' if i == bn else '未命中'}")

    (OUT / "report_fine_legal.txt").write_text("\n".join(LINES) + "\n")
    print(f"\n[written] {OUT/'report_fine_legal.txt'}")


if __name__ == "__main__":
    main()
