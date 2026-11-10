#!/usr/bin/env python3
"""纹理轴(计划 S3):同一批锚点换不同随机探针纹理,influence 排序是否一致。纯后处理。

为什么这是 influence 作为"排序函数"的**正确门槛**
------------------------------------------------
因为 clean 与 patched 共享同一个 ε,`Δa` 是精确量、零测量噪声 ⇒ 两个位置之间的
**排序本身是精确的**,不需要过 ε 地板。ε 地板回答的是另一个问题:
"随机 patch 自己能不能攻击成功"(那是 S2.2 优化阶段的事)。

对"用 influence 给候选位置排序"这个用途,真正的门槛是:
**换一张统计性质相同的随机纹理,排序还在不在。**
在 → 排序是位置的性质,可用来和 attention 比;
不在 → 排序是这一张噪声图的性质,B4 的相关系数没有意义。

只改 seed、不改 block 大小/饱和度 ⇒ 受控对照。

用法(需先跑完各纹理的 s2_scan_actions.py):
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/report_texture_axis.py
"""
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation as R

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
EX = 5
MM = 0.05 * 1000
TOPK = 5
RUNS = [("t1_seed20260810", "s2_actions.npz"),
        ("t2_seed20260811", "s2_actions_t2.npz"),
        ("t3_seed20260812", "s2_actions_t3.npz")]


def spearman(a, b):
    def rank(x):
        x = np.asarray(x, np.float64)
        o = np.argsort(x, kind="stable")
        r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=np.float64)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt)); np.add.at(s, inv, r)
        return (s / cnt)[inv]
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def topk_iou(a, b, k=TOPK):
    ia = set(np.argsort(np.asarray(a))[-k:].tolist())
    ib = set(np.argsort(np.asarray(b))[-k:].tolist())
    return len(ia & ib) / len(ia | ib)


def rot_tangent(ref, x):
    sh = ref.shape[:-1]
    a = R.from_rotvec(ref.reshape(-1, 3)); b = R.from_rotvec(x.reshape(-1, 3))
    return (a.inv() * b).as_rotvec().reshape(sh + (3,))


def load(fn):
    d = np.load(OUT / fn, allow_pickle=True)
    Ac, Ap = d["A_clean"], d["A_patched"]
    v_tr = (Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2)
    v_ro = rot_tangent(np.broadcast_to(Ac[None, :, :EX, 3:6], Ap[:, :, :EX, 3:6].shape),
                       Ap[:, :, :EX, 3:6]).sum(2)
    return dict(
        Imag=np.linalg.norm(v_tr, axis=2).sum(1) * MM,
        Isys=np.linalg.norm(v_tr.sum(1), axis=1) * MM,
        Imag_r=np.linalg.norm(v_ro, axis=2).sum(1),
        per_frame=np.linalg.norm(v_tr, axis=2) * MM,
        idx=d["anchor_idx"], world=d["anchor_world"], legal=d["anchor_legal"].astype(bool))


def main():
    lines = []

    def out(s=""):
        print(s, flush=True); lines.append(s)

    have = [(tag, fn) for tag, fn in RUNS if (OUT / fn).is_file()]
    missing = [fn for _, fn in RUNS if not (OUT / fn).is_file()]
    D = {tag: load(fn) for tag, fn in have}
    tags = [t for t, _ in have]

    out("=" * 100)
    out(f"纹理轴:S={len(tags)} 张随机探针纹理(只换 seed,block=12px 不变)")
    out(f"  已有:{tags}")
    if missing:
        out(f"  ⚠️ 缺失(还没跑完):{missing} —— 下面只用已有的算")
    out("=" * 100)

    if len(tags) < 2:
        out("  至少要 2 张纹理才能算一致性,退出。")
        (OUT / "texture_axis.txt").write_text("\n".join(lines) + "\n"); return 0

    ref = D[tags[0]]
    for t in tags[1:]:
        assert np.array_equal(D[t]["idx"], ref["idx"]), f"{t} 的锚点集与 {tags[0]} 不一致"

    out()
    out("① 每张纹理各自的强度尺度(检查三张凶度是否可比)")
    out("  纹理             | 幅度版 max | 幅度版 mean | 系统版 max | top5 锚点")
    for t in tags:
        x = D[t]
        out(f"  {t:16s} | {x['Imag'].max():10.1f} | {x['Imag'].mean():11.1f} | "
            f"{x['Isys'].max():10.1f} | {[int(x['idx'][i]) for i in np.argsort(-x['Imag'])[:5]]}")

    out()
    out("② 纹理间排序一致性 —— **这是 influence 作为排序函数的门槛**")
    out("  通道/度量          | " + " | ".join(f"{a}×{b}" for i, a in enumerate(tags)
                                                for b in tags[i + 1:]))
    pairs = [(a, b) for i, a in enumerate(tags) for b in tags[i + 1:]]
    for label, key, fn in [("平移幅度版 Spearman", "Imag", spearman),
                           ("平移系统版 Spearman", "Isys", spearman),
                           ("旋转幅度版 Spearman", "Imag_r", spearman),
                           ("平移幅度版 top5 IoU", "Imag", topk_iou)]:
        vals = [fn(D[a][key], D[b][key]) for a, b in pairs]
        out(f"  {label:20s} | " + " | ".join(f"{v:11.4f}" for v in vals))

    out()
    out("  读法:Spearman 接近 1 ⇒ 排序是**位置**的性质,不是这一张噪声图的性质")
    out("        ⇒ B4 的 attention-influence 相关系数才有意义。")
    out("        top5 IoU 单独看:只贴一个 patch 时,头部换不换才是操作上要紧的。")

    out()
    out("③ 逐锚点:三张纹理的 influence(幅度版 mm),看是否量级一致")
    out("  anc  world(x,y)   legal | " + " | ".join(f"{t[:2]:>9s}" for t in tags) + " | 变异系数")
    for i in np.argsort(-ref["Imag"]):
        vs = np.array([D[t]["Imag"][i] for t in tags])
        cv = vs.std() / vs.mean() if vs.mean() > 1e-9 else np.nan
        out(f"  #{int(ref['idx'][i]):2d}  ({ref['world'][i][0]:5.2f},{ref['world'][i][1]:5.2f}) "
            f"{str(bool(ref['legal'][i])):5s} | " + " | ".join(f"{v:9.1f}" for v in vs) +
            f" | {cv:9.3f}")

    out()
    out("④ 跨纹理平均后的 influence(S>1 一致才可平均,见计划 S3)")
    avg = np.mean([D[t]["Imag"] for t in tags], axis=0)
    out("  平均后 top8:" + str([int(ref["idx"][i]) for i in np.argsort(-avg)[:8]]))
    np.savez_compressed(OUT / "texture_axis.npz", tags=np.array(tags),
                        anchor_idx=ref["idx"], anchor_world=ref["world"],
                        anchor_legal=ref["legal"], Imag_avg=avg,
                        **{f"Imag_{t}": D[t]["Imag"] for t in tags},
                        **{f"Isys_{t}": D[t]["Isys"] for t in tags})

    (OUT / "texture_axis.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT/'texture_axis.txt'} + texture_axis.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
