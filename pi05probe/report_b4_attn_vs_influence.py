#!/usr/bin/env python3
"""B4:attention 排序 vs influence 排序,在**同一批 36 个锚点位置**上比较。纯后处理,零 GPU。

这是本阶段的目标问题:attention 能不能当"贴 patch 之前的候选位置排序函数"。
attention 与 influence 是两个**平行**的排序函数,谁更好最终由 S4 交叉代价裁决;
这里只回答"两者相关多高"。

怎么把 attention(图像 patch 网格)对到 influence(世界位置)
--------------------------------------------------------
不做几何投影。`s2_scan_obs.npz` 里同时有 clean 与 patched 的 224 输入图 ⇒
    mask[i,t] = |patched − clean| > THR      (224×224)
就是该锚点的 patch 在**模型实际视野里**的像素覆盖,零几何假设,且自动包含遮挡效应
(被机械臂挡住的部分本来就不出现在 diff 里)。
再把 mask 按 14×14 聚合到 16×16 cell,得到每个 cell 的覆盖占比 → 对 saliency 加权。

⚠️ 覆盖为 0 的锚点(patch 在该视角完全不可见)**没有 attention 分数**,必须剔除后再算相关,
   否则等于拿 0 分去和别人比排序。

saliency 定义(规格 A2 / B1)
---------------------------
head 已在存盘时求和。本脚本报三种 token 聚合:
    max  = 逐 token 取 max(忠实 POAP,主报)
    sum  = 全 token 求和(稳健版,附报;max 对单个弥散 token 极敏感)
    noun = 名词子词求和
逐层全扫(规格 A3:不许随手挑一层;取对 attention 最有利的那层当 baseline)。

聚合权重:与 influence 的幅度版一致(沿 t 均匀求和,规格 A1 要求两边一致)。
  注:influence 的**系统版**含方向,attention 没有方向对应物 ⇒ 天然只能对幅度版;
      两版的 Spearman 都报,读者自行判断。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/report_b4_attn_vs_influence.py
"""
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
THR = 10          # uint8 逐通道最大绝对差阈值(抗重采样软边)
CELL = 14         # 224 / 16
NOUNS = ("bowl", "plate")
TOPK = 5


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


def main():
    ob = np.load(OUT / "s2_scan_obs.npz", allow_pickle=True)
    inf = np.load(OUT / "s2_influence2.npz", allow_pickle=True)
    az = np.load(OUT / "attn_traj_put_the_bowl_on_the_plate.npz", allow_pickle=True)

    M, T = int(ob["M"]), int(ob["T"])
    aidx, aw = inf["anchor_idx"], inf["anchor_world"]
    leg = inf["anchor_legal"].astype(bool)
    Imag, Isys = inf["Imag_trans"], inf["Isys_trans"]
    views = [str(v) for v in az["views"]]
    A = az["bowl_plate_orig__attn"]                      # (T,L,Z,V,16,16)
    pieces = [str(x) for x in az["bowl_plate_orig__pieces"]]
    L, Z, V = A.shape[1], A.shape[2], A.shape[3]
    assert A.shape[0] == T, f"attention 帧数 {A.shape[0]} != 扫描帧数 {T}"
    assert list(az["ts"]) == list(ob["ts"]), "两边的 env 步不一致,帧没对上"

    lines = []

    def out(s=""):
        print(s, flush=True); lines.append(s)

    out("=" * 100)
    out(f"B4  attention 排序 vs influence 排序   M={M} 锚点 × T={T} 帧 × L={L} 层 × V={V} 视角")
    out(f"  attention renorm={str(az['renorm'])}   tokens={pieces}")
    out(f"  帧对齐 ✅ env 步 {list(ob['ts'])}")
    out("=" * 100)

    # ---------- patch 在模型视野里的 cell 覆盖 ----------
    # ⚠️ views 的实际取值是 'base_0_rgb' / 'left_wrist_0_rgb',不是 'base'/'wrist' ——
    #    用 substring 判定,别用 ==(曾因此把两个视角都喂成 wrist 图,而且不报错)
    is_wrist = {v: ("wrist" in v.lower()) for v in views}
    BASE = [v for v in views if not is_wrist[v]]
    assert len(BASE) == 1, f"主视角识别失败:views={views}"
    BASE = BASE[0]

    cov = {}          # view -> (M,T,16,16) 覆盖占比
    for vi, vname in enumerate(views):
        ck = "clean_wrist224" if is_wrist[vname] else "clean_img224"
        pk = "patched_wrist224" if is_wrist[vname] else "patched_img224"
        c, p = ob[ck].astype(np.int16), ob[pk].astype(np.int16)
        m = (np.abs(p - c[None]).max(-1) > THR)                       # (M,T,224,224)
        g = m.reshape(M, T, 16, CELL, 16, CELL).mean((3, 5))          # (M,T,16,16)
        cov[vname] = g
        px = m.sum((2, 3))
        out(f"\n  视角 '{vname}':patch 像素数 mean={px.mean():.0f} max={px.max()} "
            f"覆盖为 0 的锚点={int((px.sum(1) == 0).sum())}")
        if not is_wrist[vname]:
            vp = ob["visible_px"].mean(1)
            out(f"    与 seg 的 visible_px 相关(应≈1,验证 diff 掩码正确)= "
                f"{np.corrcoef(px.mean(1), vp)[0, 1]:.4f}")

    # ---------- saliency:三种 token 聚合 × 每层 ----------
    noun_rows = [i for i, p_ in enumerate(pieces) if any(n in p_.lower() for n in NOUNS)]
    out(f"\n  名词 token 行 = {noun_rows} → {[pieces[i] for i in noun_rows]}")

    def saliency(kind, vi):
        x = A[:, :, :, vi]                                            # (T,L,Z,16,16)
        if kind == "max":
            return x.max(2)
        if kind == "sum":
            return x.sum(2)
        return x[:, :, noun_rows].sum(2)

    # ---------- 每锚点的 attention 分数 ----------
    def score(sal, g):
        """sal (T,L,16,16), g (M,T,16,16) → (M,L)。cell 覆盖占比加权平均,再沿 t 求和。"""
        num = np.einsum("mtij,tlij->mtl", g, sal)
        den = g.sum((2, 3))[:, :, None]                               # (M,T,1)
        per_t = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        return per_t.sum(1), (den[:, :, 0] > 0)                       # (M,L), 有效帧掩码

    out()
    out("-" * 100)
    out("① 逐层:Spearman(attention 分数, influence) —— 规格 A3 全层扫,不挑层")
    out("-" * 100)
    results = {}
    for vname in views:
        for kind in ("max", "sum", "noun"):
            sal = saliency(kind, views.index(vname))
            S, okf = score(sal, cov[vname])
            valid = okf.any(1)                                        # 至少一帧可见
            n = int(valid.sum())
            rows = []
            for l in range(L):
                rows.append((l,
                             spearman(S[valid, l], Imag[valid]),
                             spearman(S[valid, l], Isys[valid]),
                             topk_iou(S[valid, l], Imag[valid])))
            results[(vname, kind)] = (S, valid, rows)
            best = max(rows, key=lambda r: abs(r[1]))
            out(f"\n  view={vname:5s} token聚合={kind:4s}  有效锚点={n}/{M}")
            out("   layer |  Sp(幅度版) | Sp(系统版) | top5 IoU")
            for l, s1, s2, io in rows:
                mark = "  ← |Sp| 最大" if l == best[0] else ""
                out(f"   {l:5d} | {s1:11.4f} | {s2:10.4f} | {io:8.3f}{mark}")

    # ---------- 汇总:每 (view, kind) 的最优层 ----------
    out()
    out("-" * 100)
    out("② 汇总:各提取方式对 attention **最有利**的那一层(规格 A3)")
    out("-" * 100)
    out("  view  kind |  最优层 | Sp(幅度版) | Sp(系统版) | top5 IoU | 有效锚点")
    for (vname, kind), (S, valid, rows) in results.items():
        l, s1, s2, io = max(rows, key=lambda r: abs(r[1]))
        out(f"  {vname:5s} {kind:4s} | {l:6d} | {s1:11.4f} | {s2:10.4f} | {io:8.3f} | "
            f"{int(valid.sum())}/{M}")

    # ---------- 主结果表:base + max,最优层 ----------
    S, valid, rows = results[(BASE, "max")]
    lbest = max(rows, key=lambda r: abs(r[1]))[0]
    out()
    out("-" * 100)
    out(f"③ 主结果(view=base, token=max, layer={lbest}):逐锚点对照")
    out("-" * 100)
    out("  rank(influence)  anc  world(x,y)   legal | influence幅度mm | attention分数 | rank(attn)")
    ordi = np.argsort(-Imag)
    ranka = {int(i): r for r, i in enumerate(np.argsort(-np.where(valid, S[:, lbest], -np.inf)))}
    for r_, i in enumerate(ordi):
        tag = "" if valid[i] else "  (不可见,未参与相关)"
        out(f"  {r_+1:14d}  #{int(aidx[i]):2d}  ({aw[i][0]:5.2f},{aw[i][1]:5.2f}) "
            f"{str(bool(leg[i])):5s} | {Imag[i]:14.2f} | {S[i, lbest]:13.4f} | "
            f"{ranka[int(i)]+1:9d}{tag}")

    np.savez_compressed(OUT / "b4_attn_vs_influence.npz",
                        anchor_idx=aidx, anchor_world=aw, anchor_legal=leg,
                        Imag=Imag, Isys=Isys,
                        **{f"S_{v}_{k}": results[(v, k)][0] for v, k in results},
                        **{f"valid_{v}_{k}": results[(v, k)][1] for v, k in results},
                        cov_base=cov[BASE], base_view=BASE, best_layer_base_max=lbest)

    # ---------- 图:逐层 Spearman 曲线 + 散点 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for (vname, kind), (_, _, rows) in results.items():
        if vname != BASE:
            continue
        ax[0].plot([r[0] for r in rows], [r[1] for r in rows], marker="o", ms=3.5,
                   label=f"token={kind}")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("Spearman(attention, influence)")
    ax[0].set_title("B4 per-layer rank correlation (view=base)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    sc = ax[1].scatter(S[valid, lbest], Imag[valid],
                       c=np.where(leg[valid], 0.0, 1.0), cmap="coolwarm", s=45,
                       edgecolors="k", linewidths=0.5)
    for i in np.where(valid)[0]:
        ax[1].annotate(str(int(aidx[i])), (S[i, lbest], Imag[i]), fontsize=6,
                       xytext=(3, 2), textcoords="offset points")
    ax[1].set_xlabel(f"attention score (layer {lbest}, token=max)")
    ax[1].set_ylabel("influence magnitude (mm)")
    ax[1].set_title(f"blue=legal  red=occludes obj   Sp={spearman(S[valid, lbest], Imag[valid]):.3f}")
    ax[1].grid(alpha=0.3)
    fig.colorbar(sc, ax=ax[1], ticks=[0, 1], shrink=0.8).set_ticklabels(["legal", "occl"])
    fig.tight_layout(); fig.savefig(OUT / "b4_attn_vs_influence.png", dpi=120); plt.close(fig)
    out("\n[written] b4_attn_vs_influence.png")

    (OUT / "b4_attn_vs_influence.txt").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT/'b4_attn_vs_influence.txt'} + b4_attn_vs_influence.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
