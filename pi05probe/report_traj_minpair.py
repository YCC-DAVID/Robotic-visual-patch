#!/usr/bin/env python3
"""最小对(minimal pair)的**全轨迹** attention 比较,对照 B3 相邻帧地板。

为什么必须单独做这一项
--------------------
`attn_traj_*.txt` 的"换任务"那一列比的是 stove / bottle_rack / bowl_cabinet 三条指令的
**全名词求和图**。但这三条指令提到的物体本来就不同 ⇒ 图不同可能只是因为
**文本里的名词不同**,而不是"模型编码了任务语义"。这是个弱得多的结论。

最小对能把两者分开:
    bowl_plate_orig    = "put the bowl on the plate"
    bowl_cabinet_orig  = "put the bowl on top of the cabinet"
名词集只差目的地,**共享 `bowl` token**。所以:
  - 比共享的 `bowl` 图  → 操作对象的定位是否稳定
  - 比各自的目的地图    → 目的地是否被编码(plate 的图 vs cabinet 的图 落在哪)

判据一律对照 **B3 地板**(同一指令、相邻两帧、同一张图)。
纯后处理,零额外前向。

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/report_traj_minpair.py
"""

import pathlib

import numpy as np

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NPZ = OUT / "attn_traj_put_the_bowl_on_the_plate.npz"
TXT = OUT / "report_traj_minpair.txt"
TOPK = 8


def spearman(a, b):
    """秩相关。a,b 为 1-D 等长。用平均秩处理并列。"""
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # 并列取平均秩
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(np.asarray(a, np.float64)), rank(np.asarray(b, np.float64))
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def topk_iou(a, b, k=TOPK):
    ia = set(np.argsort(np.asarray(a).ravel())[-k:].tolist())
    ib = set(np.argsort(np.asarray(b).ravel())[-k:].tolist())
    return len(ia & ib) / len(ia | ib)


def find_token(pieces, want):
    hits = [i for i, p in enumerate(pieces) if want in str(p).lower()]
    return hits


def main():
    d = np.load(NPZ, allow_pickle=True)
    views = [str(v) for v in d["views"]]
    base = views.index("base") if "base" in views else 0
    ks, ts = d["ks"], d["ts"]
    T = len(ks)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    out("=" * 100)
    out(f"最小对全轨迹比较   T={T} 帧   view='{views[base]}'(主视角)   renorm={str(d['renorm'])}")
    out(f"  帧对应的 env 步: {ts.tolist()}")

    A = d["bowl_plate_orig__attn"]        # (T, L, Z, V, 16, 16)
    B = d["bowl_cabinet_orig__attn"]
    pa = [str(x) for x in d["bowl_plate_orig__pieces"]]
    pb = [str(x) for x in d["bowl_cabinet_orig__pieces"]]
    out(f"\n  bowl_plate_orig   tokens: {pa}")
    out(f"  bowl_cabinet_orig tokens: {pb}")

    ia, ib = find_token(pa, "bowl"), find_token(pb, "bowl")
    out(f"\n  'bowl' token 位置: plate 图 {ia} / cabinet 图 {ib}")
    if not ia or not ib:
        out("  ❌ 找不到共享的 bowl token,退出")
        TXT.write_text("\n".join(lines)); return
    ia, ib = ia[0], ib[0]

    # 目的地 token
    da, db = find_token(pa, "plate"), find_token(pb, "cabinet")
    out(f"  目的地 token 位置: 'plate' {da} / 'cabinet' {db}")

    L = A.shape[1]
    out("\n" + "=" * 100)
    out("① 共享 `bowl` token 的图:换目的地 vs B3 相邻帧地板")
    out("=" * 100)
    out("  地板 = 同一指令(bowl_plate)相邻两帧、同一张 bowl 图")
    out("")
    out("  layer | 地板 Sp mean | 最小对 Sp mean | 地板 IoU | 最小对 IoU | 结论")
    for l in range(L):
        fa = [A[t, l, ia, base].ravel() for t in range(T)]
        fb = [B[t, l, ib, base].ravel() for t in range(T)]
        floor_sp = np.mean([spearman(fa[t], fa[t + 1]) for t in range(T - 1)])
        floor_io = np.mean([topk_iou(fa[t], fa[t + 1]) for t in range(T - 1)])
        pair_sp = np.mean([spearman(fa[t], fb[t]) for t in range(T)])
        pair_io = np.mean([topk_iou(fa[t], fb[t]) for t in range(T)])
        verdict = "目的地未编码" if pair_sp > floor_sp else "有差异"
        out(f"  {l:5d} | {floor_sp:12.4f} | {pair_sp:14.4f} | {floor_io:8.3f} | {pair_io:10.3f} | {verdict}")

    out("\n  读法:若【最小对 Sp > 地板 Sp】,说明换掉目的地对 `bowl` 图的影响")
    out("        **比单纯过一个时间步还小** ⇒ 目的地没有被编码进这张图。")

    out("\n" + "=" * 100)
    out("② 目的地 token 自己的图:plate 的图 vs cabinet 的图,落在同一处吗")
    out("=" * 100)
    if not da or not db:
        out("  ⚠️ 找不到目的地 token,跳过")
    else:
        out("  layer | argmax(plate) | argmax(cabinet) | 是否同格 | Sp | IoU")
        for l in range(L):
            sps, ios, same = [], [], 0
            am_a = am_b = None
            for t in range(T):
                ga = A[t, l, da[0], base]
                gb = B[t, l, db[0], base]
                sps.append(spearman(ga.ravel(), gb.ravel()))
                ios.append(topk_iou(ga, gb))
                pa_ = np.unravel_index(int(ga.argmax()), ga.shape)
                pb_ = np.unravel_index(int(gb.argmax()), gb.shape)
                same += int(pa_ == pb_)
                if t == 0:
                    am_a, am_b = pa_, pb_
            out(f"  {l:5d} | {str(am_a):13s} | {str(am_b):15s} | {same:2d}/{T} | "
                f"{np.mean(sps):.4f} | {np.mean(ios):.3f}")
        out("\n  读法:两个目的地词的图若高度重合(Sp 接近 1、argmax 同格),")
        out("        说明模型没把 `plate` 和 `cabinet` 指向场景里不同的位置。")

    TXT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {TXT}")


if __name__ == "__main__":
    main()
