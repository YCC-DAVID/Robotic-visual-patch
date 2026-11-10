#!/usr/bin/env python3
"""核心命题检验:**限定在能贴的位置里**,attention 估出来的点是不是 influence 最大的点。

命题(用户 2026-08-11 明确的版本)
--------------------------------
"根据 attention 贴图"这个方法:用 instruction + image 算 attention → 估出一个位置。
在**能贴的情况下**(合法位置,不压住物体),这个位置**不一定**是 influence 最大的位置。

所以候选集 = 17 个合法锚点(patch 与任何物体 AABB 零重叠),不含压物体的格子。
两个排序函数在同一候选集上各选一个 argmax,比是否同一点、以及跟着 attention 走
能拿到 influence-max 的百分之几。

三张图(本脚本全部产出)
----------------------
  ① clean attention   —— 攻击者贴之前唯一拿得到的
  ② patched attention —— 贴之后真实的 attention(patch 可能把注意力吸到自己身上)
  ③ influence         —— 实测动作影响(跨 3 张随机纹理平均,排序已验证与纹理无关)

规格
----
- 归一化:报 raw 与 renorm 两版。renorm = 每 token 行在 512 个图像 token 上求和后除
  (规格 A4)。⚠️ `attn_traj_*.npz` 存的是**原始**值,其 `renorm` 字段只记录了
  当时脚本报告用的 CLI 参数,不代表数组已归一化 —— 别被那个字段骗了。
- 层:全 18 层扫,不挑层(规格 A3)。同时报"允许挑最有利的层"和"层未知"两种情形,
  因为这两者给出**相反**的结论,而真实攻击设定属于后者。
- token 聚合:max(忠实 POAP)/ sum(稳健)/ noun(固定名词集,对归一化口径不敏感)。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/report_legal_argmax.py
"""
import collections
import pathlib

import numpy as np

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NOUN_ROWS = [3, 6]          # ['▁bowl', '▁plate'](pieces 里的下标)
KINDS = ("max", "sum", "noun")
MODES = ("raw", "renorm")


def spearman(a, b):
    def rank(x):
        x = np.asarray(x, np.float64)
        o = np.argsort(x, kind="stable")
        r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=np.float64)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        s = np.zeros(len(cnt)); np.add.at(s, inv, r)
        return (s / cnt)[inv]
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def main():
    b4 = np.load(OUT / "b4_attn_vs_influence.npz", allow_pickle=True)
    tx = np.load(OUT / "texture_axis.npz", allow_pickle=True)
    ap = np.load(OUT / "attn_patched_grid.npz", allow_pickle=True)

    cov = b4["cov_base"]                              # (M,T,16,16) patch 的 cell 覆盖占比
    Iavg = tx["Imag_avg"]                             # 跨 3 纹理平均的 influence(幅度版, mm)
    leg = b4["anchor_legal"].astype(bool)
    idx, aw = b4["anchor_idx"], b4["anchor_world"]
    Ac = ap["attn_clean"].astype(np.float64)          # (T,L,Z,V,16,16) 原始 head-求和
    Ap = ap["attn_patched"].astype(np.float64)        # (M,T,L,Z,V,16,16)
    M, T, L = Ap.shape[0], Ap.shape[1], Ap.shape[2]
    pieces = [str(x) for x in ap["pieces"]]
    lines = []

    def out(s=""):
        print(s, flush=True); lines.append(s)

    vis = (cov.sum((2, 3)) > 0).any(1)
    sel = leg & vis
    w = np.where(sel)[0]
    best = w[np.argmax(Iavg[w])]

    out("=" * 104)
    out("核心命题:能贴的位置里,attention 估的点 ≠ influence 最大的点?")
    out("=" * 104)
    out(f"  候选集 = 合法(零重叠)且主视角可见的锚点:{len(w)} 个 → "
        f"{[int(idx[i]) for i in w]}")
    out(f"  influence argmax(跨 3 纹理平均)= #{int(idx[best])} "
        f"({aw[best][0]:.2f},{aw[best][1]:.2f})  {Iavg[best]:.1f} mm")
    out(f"  候选集内 influence 前 5:" +
        "  ".join(f"#{int(idx[i])}={Iavg[i]:.0f}mm"
                  for i in w[np.argsort(-Iavg[w])][:5]))
    out(f"  tokens={pieces}   名词行={NOUN_ROWS} → {[pieces[i] for i in NOUN_ROWS]}")

    def renorm(X):
        """X:(...,L,Z,V,16,16) 的单层切片 (…,Z,V,16,16);在 512 个图像 token 上逐行归一化。"""
        return X / np.clip(X.sum(axis=(-4, -3, -2, -1) if False else (-3, -2, -1),
                                 keepdims=True), 1e-12, None)

    def sal_from(X, kind):
        """X:(T,Z,V,16,16) → (T,16,16),主视角(V=0)。"""
        if kind == "max":
            return X[:, :, 0].max(1)
        if kind == "sum":
            return X[:, :, 0].sum(1)
        return X[:, NOUN_ROWS, 0].sum(1)

    def score(sal):
        """sal (T,16,16) → (M,)。patch 覆盖占比加权平均,沿 t 求和(与 influence 同权重)。"""
        num = np.einsum("mtij,tij->mt", cov, sal)
        den = cov.sum((2, 3))
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0).sum(1)

    def score_patched(kind, mode, l):
        """patched attention:每个锚点用**它自己那次前向**的 saliency 读它自己的 cell。"""
        s = np.zeros(M)
        for i in range(M):
            X = Ap[i, :, l]
            if mode == "renorm":
                X = renorm(X)
            sal = sal_from(X, kind)
            num = np.einsum("tij,tij->t", cov[i], sal)
            den = cov[i].sum((1, 2))
            s[i] = np.divide(num, den, out=np.zeros_like(num), where=den > 0).sum()
        return s

    # ---------------- ① clean attention:逐层 ----------------
    out()
    out("-" * 104)
    out("① clean attention(攻击者唯一拿得到的),逐层在候选集里选点  [归一化=renorm,合规 A4]")
    out("-" * 104)
    out("  layer | token=max 选点  占max | token=sum 选点  占max | token=noun 选点 占max | Sp(noun)")
    hits = collections.Counter()
    ratios = []
    detail = []
    for l in range(L):
        cells = []
        for kind in KINDS:
            S = score(sal_from(renorm(Ac[:, l]), kind))
            p = w[np.argmax(S[w])]
            cells.append((int(idx[p]), 100 * Iavg[p] / Iavg[best]))
        Sn = score(sal_from(renorm(Ac[:, l]), "noun"))
        mark = " ★" if any(c[0] == int(idx[best]) for c in cells) else ""
        out(f"  {l:5d} | " + " | ".join(f"#{a:2d} {b:9.0f}%" for a, b in cells) +
            f" | {spearman(Sn[w], Iavg[w]):+.4f}{mark}")
    for mode in MODES:
        for kind in KINDS:
            for l in range(L):
                X = Ac[:, l]
                S = score(sal_from(renorm(X) if mode == "renorm" else X, kind))
                p = w[np.argmax(S[w])]
                hits[int(idx[p])] += 1
                ratios.append(Iavg[p] / Iavg[best])
                detail.append((mode, kind, l, int(idx[p])))
    r = np.array(ratios)
    out()
    out("  ★ = 该层至少一种 token 聚合选中了 influence-argmax")
    out(f"\n  全部 {len(r)} 种配置(2 归一化 × 3 token聚合 × {L} 层)的选点分布:")
    for a, c in hits.most_common():
        i = int(np.where(idx == a)[0][0])
        out(f"    #{a:2d} ({aw[i][0]:5.2f},{aw[i][1]:5.2f})  被选 {c:3d} 次  "
            f"influence={Iavg[i]:6.1f}mm  占 legal-max {100*Iavg[i]/Iavg[best]:3.0f}%")
    out(f"\n  选中 influence-argmax 的比例 = {int((r == 1.0).sum())}/{len(r)} = "
        f"{100*(r == 1.0).mean():.0f}%")
    out(f"  拿到的 influence 占 legal-max:mean={100*r.mean():.0f}%  "
        f"median={100*np.median(r):.0f}%  min={100*r.min():.0f}%  max={100*r.max():.0f}%")
    ok_layers = sorted({d[2] for d in detail if d[3] == int(idx[best])})
    out(f"  能选对的层 = {ok_layers}  ← 只有这些层;其余层都选错")

    # ---------------- ② clean vs patched attention ----------------
    out()
    out("-" * 104)
    out("② patched attention:贴上去之后 attention 搬家了吗(clean 能不能预测 patched)")
    out("-" * 104)
    out("  layer | patch 上 attention 增益 median (patched/clean) | Sp(clean分数, patched分数) | "
        "patched 选点 占max")
    for l in range(L):
        Sc = score(sal_from(renorm(Ac[:, l]), "noun"))
        Sp_ = score_patched("noun", "renorm", l)
        gain = np.divide(Sp_, Sc, out=np.zeros_like(Sp_), where=Sc > 0)
        p = w[np.argmax(Sp_[w])]
        out(f"  {l:5d} | {np.median(gain[sel]):45.3f} | "
            f"{spearman(Sc[w], Sp_[w]):26.4f} | #{int(idx[p]):2d} "
            f"{100*Iavg[p]/Iavg[best]:6.0f}%")
    out("\n  增益 >1 ⇒ patch 把注意力吸到了自己身上 ⇒ clean attention 低估了它,")
    out("  ⇒ 用 clean attention 排序与贴上后的真实 attention 不是一回事。")

    # ---------------- ③ 三张图两两 ----------------
    out()
    out("-" * 104)
    out("③ 三张图在候选集(17 合法点)上的两两 Spearman   [renorm + noun]")
    out("-" * 104)
    out("  layer | clean↔influence | patched↔influence | clean↔patched")
    for l in range(L):
        Sc = score(sal_from(renorm(Ac[:, l]), "noun"))
        Sp_ = score_patched("noun", "renorm", l)
        out(f"  {l:5d} | {spearman(Sc[w], Iavg[w]):+15.4f} | "
            f"{spearman(Sp_[w], Iavg[w]):+17.4f} | {spearman(Sc[w], Sp_[w]):+13.4f}")

    np.savez_compressed(OUT / "legal_argmax.npz",
                        cand=w, best=best, anchor_idx=idx, anchor_world=aw,
                        anchor_legal=leg, Iavg=Iavg, ratios=r,
                        detail=np.array(detail, dtype=object))
    (OUT / "legal_argmax.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT/'legal_argmax.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
