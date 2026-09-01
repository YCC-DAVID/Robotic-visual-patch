#!/usr/bin/env python3
"""B4 的 sink 控制:把跨指令不变的 register/sink 格屏蔽后,attention-influence 相关还剩多少。

为什么必须做
----------
`sink_cells.txt` 已查明 layer 7 的跨指令不变高值格是 (7,9)/(7,8);而 layer 7 的
token-max saliency 在 16 帧里有 7 帧 argmax 就落在 (7,9)。
B4 的锚点分数 = patch 覆盖的那些 cell 上 saliency 的覆盖占比加权平均 ⇒
如果高值全来自 sink 格,那 Spearman=0.88 实际测的是
**"这个 patch 有没有压到 sink 格"**,不是"attention 指向了这个位置",相关就是伪的。

sink 格怎么定(不硬编码)
----------------------
用 attn_traj 里的 9 条指令 variant:每条 variant 的 saliency 各自 min-max 归一化,
再对 variant 取 **min** —— min 高 ⇒ 对**每一条指令**都亮 ⇒ 与指令内容无关的结构。
取 min 最高的 N 格为 sink,屏蔽(置 0)后重算。

判读
----
Spearman 基本不变 ⇒ 相关不是 sink 驱动的,0.88 站得住。
Spearman 明显塌 ⇒ 相关主要来自 sink 位置,B4 结论要撤回。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/report_b4_sink_control.py
"""
import pathlib

import numpy as np

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NSINK = (0, 1, 2, 3, 5, 8)      # 屏蔽格数的扫描(0 = 不屏蔽,作对照)
LAYERS = (3, 4, 5, 7, 10)


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


def topk_iou(a, b, k=5):
    ia = set(np.argsort(np.asarray(a))[-k:].tolist())
    ib = set(np.argsort(np.asarray(b))[-k:].tolist())
    return len(ia & ib) / len(ia | ib)


def main():
    az = np.load(OUT / "attn_traj_put_the_bowl_on_the_plate.npz", allow_pickle=True)
    b4 = np.load(OUT / "b4_attn_vs_influence.npz", allow_pickle=True)
    tx = np.load(OUT / "texture_axis.npz", allow_pickle=True)
    views = [str(v) for v in az["views"]]
    base = [i for i, v in enumerate(views) if "wrist" not in v.lower()][0]
    cov = b4["cov_base"]                                  # (M,T,16,16)
    idx = b4["anchor_idx"]
    Iavg = tx["Imag_avg"]
    M, T = cov.shape[0], cov.shape[1]
    lines = []

    def out(s=""):
        print(s, flush=True); lines.append(s)

    variants = [k[:-6] for k in az.files if k.endswith("__attn")]
    out("=" * 100)
    out(f"B4 sink 控制   M={M} 锚点 × T={T} 帧   variant 数={len(variants)}(用于定 sink 格)")
    out("=" * 100)

    # ---------- 定 sink 格:跨 variant 取 min ----------
    out("\n① sink 格识别(每 variant 的 saliency 各自 min-max 后对 variant 取 min)")
    sink_rank = {}
    for l in LAYERS:
        mins = []
        for v in variants:
            A = az[f"{v}__attn"]                          # (T,L,Z,V,16,16)
            sal = A[:, l, :, base].max(1).mean(0)         # token-max,再沿 t 平均 → (16,16)
            rng = sal.max() - sal.min()
            mins.append((sal - sal.min()) / rng if rng > 0 else sal * 0)
        mn = np.min(np.stack(mins), axis=0)               # (16,16)
        order = np.argsort(-mn.ravel())
        sink_rank[l] = [np.unravel_index(int(o), (16, 16)) for o in order]
        top = [(f"({r},{c})", float(mn[r, c])) for r, c in sink_rank[l][:6]]
        out(f"  layer {l:2d}: " + "  ".join(f"{s}:{v:.2f}" for s, v in top))

    # ---------- 屏蔽 sink 后重算 ----------
    out("\n② 屏蔽 N 个 sink 格后的 Spearman(attention, influence 跨纹理平均)")
    out("  layer | " + " | ".join(f"屏蔽{n}格" for n in NSINK))
    A0 = az["bowl_plate_orig__attn"]
    for l in LAYERS:
        row = []
        for n in NSINK:
            mask = np.ones((16, 16))
            for r, c in sink_rank[l][:n]:
                mask[r, c] = 0.0
            sal = A0[:, l, :, base].max(1) * mask[None]   # (T,16,16)
            g = cov * mask[None, None]
            num = np.einsum("mtij,tij->mt", g, sal)
            den = g.sum((2, 3))
            per_t = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
            S = per_t.sum(1)
            valid = (den > 0).any(1)
            row.append((spearman(S[valid], Iavg[valid]), int(valid.sum()),
                        topk_iou(S[valid], Iavg[valid])))
        out(f"  {l:5d} | " + " | ".join(f"{s:.4f}({nv})" for s, nv, _ in row))
    out("    括号内 = 参与相关的锚点数(屏蔽后可能有锚点完全没有可用 cell)")

    # ---------- 哪些锚点压在 sink 格上 ----------
    out("\n③ 各锚点对 sink 格 (7,9)/(7,8)/(6,9) 的覆盖 vs 其 influence 排名")
    sinks = [(7, 9), (7, 8), (6, 9)]
    scov = np.stack([cov[:, :, r, c] for r, c in sinks]).sum(0).mean(1)   # (M,)
    ordi = np.argsort(-Iavg)
    out("  influence排名 anc  world  | 对 sink 3 格的平均覆盖 | influence(mm)")
    for r_, i in enumerate(ordi[:12]):
        w = b4["anchor_world"][i]
        out(f"  {r_+1:12d}  #{int(idx[i]):2d}  ({w[0]:5.2f},{w[1]:5.2f}) | "
            f"{scov[i]:22.4f} | {Iavg[i]:12.1f}")
    out(f"\n  Spearman(对 sink 格的覆盖, influence) = {spearman(scov, Iavg):+.4f}")
    out("    若这个数很高 ⇒ 高 influence 的锚点恰好就压在 sink 格上,混淆严重。")

    (OUT / "b4_sink_control.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT/'b4_sink_control.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
