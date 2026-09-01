#!/usr/bin/env python3
"""FastWAM 扫描的报告:把三问在第二个模型上重做一遍,与 π0.5 逐条对照。纯后处理。

三问(与 report_fine_legal.py 同口径)
----------------------------------
① 面积 vs 位置:面积↔influence 的相关,以及**控制距离后的偏相关**。
   π0.5 上:合并 78 点池 面积↔influence 只有 +0.08,控制距离后偏相关 −0.48(面积是距离的影子)。
   这条能不能复制到 FastWAM,是"面积不决定、位置决定"跨模型成立与否的关键。
② influence 排名与距物体距离的分箱。
③ `--attn`:attention vs influence 命中检验(需先有 fw_attn.npz + fw_anchor_cells.npz)。
   π0.5 的问法是「attention 的首选位置 = influence 的首选位置吗」。FastWAM 侧口径:
   名词列 → 左 7×7 base 重归一;逐层看峰值格是否 = influence 最强锚点(#1007)的落格,
   再在 78 个锚点上算 attention(锚点所在格) ↔ influence 的秩相关。
   ⚠️ 分辨率注意:base 只有 49 格(≈32 px/格),78 个锚点会共享格 ⇒ 命中检验是格级粗判。

通道拆分(ENV.md §4.4):平移 3 维合成 L2、旋转 SO(3) 测地、夹爪翻转;**绝不 7 维整体 L2**。
只累加 executed prefix(FastWAM 的 replan_steps=10)那几步,与 influence 定义一致。

用法:
    /home/user1/miniconda3/envs/wamattack/bin/python probe/fw_report.py
"""
from __future__ import annotations

import pathlib

import numpy as np

REPO = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = REPO / "probe" / "out"
BOWL, PLATE = np.array([-0.098, -0.009]), np.array([0.062, -0.009])
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(s)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def partial(x, y, z):
    """控制 z 后 x 与 y 的偏秩相关。"""
    rx, ry, rz = [np.argsort(np.argsort(np.asarray(v, float))).astype(float) for v in (x, y, z)]
    rxy = np.corrcoef(rx, ry)[0, 1]; rxz = np.corrcoef(rx, rz)[0, 1]; ryz = np.corrcoef(ry, rz)[0, 1]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def rot_geodesic(a3, b3):
    from scipy.spatial.transform import Rotation as R
    return float(np.linalg.norm((R.from_rotvec(a3).inv() * R.from_rotvec(b3)).as_rotvec()))


def influence(z):
    """平移影响力(mm),口径同 π0.5 的 Imag:Σ_t‖Σ_{k<EX}(Δa平移)‖ × 50。"""
    EX = int(z["exec_prefix"])
    Ac, Ap = z["A_clean"], z["A_patched"]                 # [T,32,7], [M,T,32,7]
    v = (Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2)   # [M,T,3]
    return np.linalg.norm(v, axis=2).sum(1) * 0.05 * 1000


def main():
    z = np.load(OUT / "fw_scan.npz", allow_pickle=True)
    I = influence(z)
    aw, idx, leg = z["anchor_world"], z["anchor_idx"], z["anchor_legal"].astype(bool)
    area = z["visible_px"].mean(1)
    d = np.minimum(np.linalg.norm(aw[:, :2] - BOWL, axis=1), np.linalg.norm(aw[:, :2] - PLATE, axis=1))
    M = len(I)

    out("=" * 100)
    out(f"FastWAM 扫描报告  task={str(z['task'])}  候选池 {M} 个合法点  "
        f"clean success={bool(z['success'])}  T={int(z['T'])}")
    out("=" * 100)
    out(f"  最强合法点:#{int(idx[I.argmax()])} "
        f"({aw[I.argmax(),0]:+.2f},{aw[I.argmax(),1]:+.2f}) "
        f"d={d[I.argmax()]*100:.1f}cm  {I.max():.1f} mm  面积 {area[I.argmax()]:.0f} px")
    out()
    out("-" * 100)
    out("① 面积 vs 位置(核心:跨模型复制 π0.5 的「面积不决定、位置决定」)")
    out("-" * 100)
    out(f"  面积 ↔ influence          秩相关 {spearman(area, I):+.3f}")
    out(f"  距离(−d) ↔ influence      秩相关 {spearman(-d, I):+.3f}")
    out(f"  面积 ↔ 距离(−d)           秩相关 {spearman(area, -d):+.3f}")
    out(f"  偏相关 面积↔influence | 距离 = {partial(area, I, -d):+.3f}"
        f"   (π0.5 上是 −0.48;转负⇒面积只是距离的影子)")
    out(f"  偏相关 距离↔influence | 面积 = {partial(-d, I, area):+.3f}   (π0.5 上是 +0.85)")
    out()
    out("  按距物体分箱(归一到池内最大):")
    out("    距离        n  | influence 均值 | influence 最大")
    for lo, hi in [(.12, .15), (.15, .18), (.18, .21), (.21, .26), (.26, .35), (.35, 1)]:
        m = (d >= lo) & (d < hi)
        if m.sum():
            out(f"    {lo*100:3.0f}-{hi*100:3.0f} cm  {int(m.sum()):3d} | "
                f"{100*I[m].mean()/I.max():13.1f}% | {100*I[m].max()/I.max():13.1f}%")
    out()
    out("  合并池 influence 前 12:")
    out("    rank  锚点   world(x,y)      距物体  面积px  influence")
    for r, i in enumerate(np.argsort(-I)[:12]):
        out(f"    {r+1:4d}  #{int(idx[i]):<5d} ({aw[i,0]:+.2f},{aw[i,1]:+.2f})  {d[i]*100:5.1f}cm  "
            f"{area[i]:6.0f}  {I[i]:7.1f} mm")

    out()
    out("-" * 100)
    out("② 面积配对(与 π0.5 rollout 用的同一对):面积几乎相同、influence 差几倍")
    out("-" * 100)
    for a_id, b_id in [(1006, 1055), (1011, 1046)]:
        ia = np.where(idx == a_id)[0]; ib = np.where(idx == b_id)[0]
        if len(ia) and len(ib):
            ia, ib = int(ia[0]), int(ib[0])
            out(f"  #{a_id} 面积 {area[ia]:.0f}px influence {I[ia]:.1f}mm   vs   "
                f"#{b_id} 面积 {area[ib]:.0f}px influence {I[ib]:.1f}mm   "
                f"⇒ influence 比 {I[ia]/max(I[ib],1e-9):.2f}×")

    (OUT / "fw_report.txt").write_text("\n".join(LINES) + "\n")
    print(f"\n[written] {OUT/'fw_report.txt'}")
    return 0


def attn_main():
    """③ attention vs influence:FastWAM 上「attention 的首选 = influence 的首选吗」。"""
    za = np.load(OUT / "fw_attn.npz", allow_pickle=True)
    zc = np.load(OUT / "fw_anchor_cells.npz", allow_pickle=True)
    z = np.load(OUT / "fw_scan.npz", allow_pickle=True)
    I = influence(z)
    idx, aw = z["anchor_idx"], z["anchor_world"]
    assert np.array_equal(zc["anchor_idx"], idx), "fw_anchor_cells 与 fw_scan 锚点顺序不一致"
    cell = zc["cell"]                                        # [M,2] 模型帧 (row,col)
    attn = za["attn"]                                        # [T,L,Sq,Sk]
    gh, gw, bc = int(za["grid_h"]), int(za["grid_w"]), int(za["base_cols"])
    noun_idx = [int(v) for v in za["noun_idx"]]
    toks = list(za["tokens"])
    T, L, Sq, Sk = attn.shape
    FR = min(T, 6)
    M = len(I)
    order = np.argsort(-I)
    gmax = int(order[0]); gcell = tuple(int(v) for v in cell[gmax])
    top3cells = {tuple(int(v) for v in cell[i]) for i in order[:3]}

    def bmap(layer, nouns):
        col = attn[:FR, layer][:, :, nouns].sum(-1).mean(0)  # [Sq]
        b = col.reshape(gh, gw)[:, :bc]                      # 模型帧 base 7×7
        return b / (b.sum() + 1e-9)

    out("=" * 100)
    out(f"FastWAM attention vs influence  ({M} 个合法锚点,base 7×7 格级判定,前 {FR} 帧平均)")
    out("=" * 100)
    out(f"  influence 最强:#{int(idx[gmax])} ({aw[gmax,0]:+.2f},{aw[gmax,1]:+.2f}) "
        f"{I[gmax]:.1f} mm  落在 base 格 {gcell}")
    out(f"  influence 前 3 的落格集合:{sorted(top3cells)}")
    out(f"  名词 token:{[toks[i] for i in noun_idx]} @ {noun_idx}")
    out()
    out("  逐层(bowl+plate 合并列,head 求和):")
    out("    layer | 峰值格   | =最强格? | ∈前3格? | 最强格的注意排名/49 | spearman(attn@格, influence)")
    sps, hit1, hit3, peaks = [], [], [], []
    for l in range(L):
        b = bmap(l, noun_idx)
        pr, pc = np.unravel_index(int(b.argmax()), b.shape)
        rank = int((b.flatten() > b[gcell]).sum()) + 1       # 最强锚点落格在 49 格里的注意名次
        av = b[cell[:, 0], cell[:, 1]]                       # 每个锚点所在格的注意
        sp = spearman(av, I)
        h1 = (pr, pc) == gcell; h3 = (pr, pc) in top3cells
        sps.append(sp); hit1.append(h1); hit3.append(h3); peaks.append((int(pr), int(pc)))
        out(f"    L{l:<4d} | ({pr},{pc})    | {'HIT' if h1 else '  .'}     | "
            f"{'HIT' if h3 else '  .'}    | {rank:19d} | {sp:+.3f}")
    sps = np.array(sps)
    cheb = [max(abs(p[0] - gcell[0]), abs(p[1] - gcell[1])) for p in peaks]
    occ = {tuple(int(v) for v in c) for c in cell}           # 有合法锚点的格
    out()
    out("  汇总:")
    out(f"    峰值格 = influence 最强锚点落格:{int(np.sum(hit1))}/{L} 层")
    out(f"    峰值格 ∈ influence 前 3 落格:  {int(np.sum(hit3))}/{L} 层")
    out(f"    峰值格与最强格棋盘距离 ≤1(邻格命中):{sum(1 for c in cheb if c <= 1)}/{L} 层")
    out(f"    峰值格内没有任何合法锚点的层:{sum(1 for p in peaks if p not in occ)}/{L}"
        "(物体/机械臂所在格不许放贴纸 ⇒ 注意峰值若压在物体上,精确命中在结构上不可能,"
        "邻格命中才是可达上限)")
    out(f"    spearman(attn@格, influence) 逐层中位 {np.median(sps):+.3f},"
        f"最好 L{int(np.argmax(sps))} = {sps.max():+.3f},最差 L{int(np.argmin(sps))} = {sps.min():+.3f}")
    out()
    out("  π0.5 §11 同款收益口径:每层让 attention 在 78 个合法锚点里选(所在格注意最大的格,")
    out("  格内取均值——同格锚点注意并列),拿到的 influence 占全池最大值的比例:")
    ylds = []
    for l in range(L):
        b = bmap(l, noun_idx)
        av = b[cell[:, 0], cell[:, 1]]
        cc = tuple(int(v) for v in cell[int(np.argmax(av))])
        inh = [i for i in range(M) if tuple(cell[i]) == cc]
        ylds.append(float(np.mean(I[inh]) / I.max()))
    ylds = np.array(ylds)
    out(f"    attention(层未知,30 层平均):{100*ylds.mean():.0f}%  中位 {100*np.median(ylds):.0f}%  "
        f"最好 L{int(np.argmax(ylds))} = {100*ylds.max():.0f}%  最差 {100*ylds.min():.0f}%")
    out(f"    随机均匀挑一个合法锚点基线:  {100*I.mean()/I.max():.0f}%")
    out(f"    (π0.5 上是 32% vs 随机 35% —— attention 不优于随机)")
    bl = int(np.argmax(sps))
    b = bmap(bl, noun_idx)
    out()
    out(f"  最相关层 L{bl} 的注意前 5 格 vs 各格里的 influence 最大锚点:")
    flat = np.argsort(-b.flatten())[:5]
    for k, f in enumerate(flat):
        r, c = int(f // bc), int(f % bc)
        inhab = [i for i in range(M) if tuple(cell[i]) == (r, c)]
        if inhab:
            j = inhab[int(np.argmax(I[inhab]))]
            out(f"    {k+1}. 格 ({r},{c}) 注意 {b[r,c]*100:.1f}%  内含 {len(inhab)} 锚点,"
                f"最强 #{int(idx[j])} {I[j]:.1f} mm(全池第 {int(np.where(order==j)[0][0])+1} 名)")
        else:
            out(f"    {k+1}. 格 ({r},{c}) 注意 {b[r,c]*100:.1f}%  (无锚点——物体/机械臂本体区)")
    out()
    out("  逐名词(sharpest 判定用熵最低层):")
    for ni, nm in zip(noun_idx, [toks[i] for i in noun_idx]):
        ent = []
        for l in range(L):
            p = bmap(l, [ni]).flatten() + 1e-12; p /= p.sum()
            ent.append(float(-(p * np.log(p)).sum()))
        sl = int(np.argmin(ent))
        bb = bmap(sl, [ni])
        pr, pc = np.unravel_index(int(bb.argmax()), bb.shape)
        out(f"    '{nm}':最尖锐 L{sl}(熵 {ent[sl]:.2f}),峰值格 ({pr},{pc})"
            f"{'  = influence 最强格' if (pr,pc)==gcell else ''}")

    (OUT / "fw_attn_report.txt").write_text("\n".join(LINES) + "\n")
    print(f"\n[written] {OUT/'fw_attn_report.txt'}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", action="store_true", help="③ attention vs influence 命中检验")
    raise SystemExit(attn_main() if ap.parse_args().attn else main())
