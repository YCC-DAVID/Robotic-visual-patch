#!/usr/bin/env python3
"""PART B 第 3 步:B1/B2 的分析。**纯后处理,零额外前向。**

输入 `out/attn_b1b2.npz`。

为什么不能直接用"全 token max"图做跨指令比较(§C)
------------------------------------------------
各指令的真实 token 数不同(实测 Z = 6/8/9/10),而 **token 越多、max 的期望值越高**
⇒ "全 token max" 图跨指令**有偏**。所以 B1/B2 一律改用:
  · **名词专属图**(对该名词的子词 token 求和)
  · **逐 token 图**
本脚本主要报名词专属图,同时把"全 token max/sum"作为对照一起报,并标明它有偏。

§A3 的顺序(承重):head 求和 → **逐行归一化** → token 归约。
本脚本额外报一份"不做重归一化"的对照,用来量化 §A-4 的 sink 到底有多致命。

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/report_b1b2.py
"""

import itertools
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = ROOT / "pi05probe" / "out"
N_SIDE, N_LAYER, N_HEAD = 16, 18, 8
WIN = 3

# 每条指令里哪些 token 是名词(按 pieces 精确匹配;POAP 的"名词求和"就是对这些行求和)
NOUNS = {
    "B1_stove":        {"stove": ["▁stove"]},
    "B1_bottle_rack":  {"bottle": ["▁wine", "▁bottle"], "rack": ["▁rack"]},
    "B1_bowl_plate":   {"bowl": ["▁bowl"], "plate": ["▁plate"]},
    "B1_bowl_cabinet": {"bowl": ["▁bowl"], "cabinet": ["▁top", "▁of", "▁cabinet"]},
    "B2_L1_place":     {"bowl": ["▁bowl"], "plate": ["▁plate"]},
    "B2_L1_set":       {"bowl": ["▁bowl"], "plate": ["▁plate"]},
    "B2_L1_move":      {"bowl": ["▁bowl"], "plate": ["▁plate"]},
    "B2_L2_frontPP":   {"bowl": ["▁bowl"], "plate": ["▁plate"]},
    "B2_L3_please":    {"bowl": ["▁bowl"], "plate": ["▁plate"]},
}
B1_NAMES = ["B1_stove", "B1_bottle_rack", "B1_bowl_plate", "B1_bowl_cabinet"]
B2_NAMES = ["B2_L1_place", "B2_L1_set", "B2_L1_move", "B2_L2_frontPP", "B2_L3_please"]

_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def rank(x):
    """平均秩(处理并列),给 Spearman 用。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float(ra @ rb / den) if den > 0 else float("nan")


def topk_iou(a, b, k):
    ia = set(np.argsort(np.asarray(a).ravel())[::-1][:k].tolist())
    ib = set(np.argsort(np.asarray(b).ravel())[::-1][:k].tolist())
    return len(ia & ib) / len(ia | ib)


def best_window(S, s=WIN):
    best, pos = -np.inf, None
    for i in range(N_SIDE - s + 1):
        for j in range(N_SIDE - s + 1):
            v = S[i:i + s, j:j + s].sum()
            if v > best:
                best, pos = v, (i + s // 2, j + s // 2)
    return pos


def rows(hs, layer, view, renorm=True):
    """hs:(L,Z,V,16,16) → (Z,256),按 §A3 第 2 步逐行归一化。"""
    A = hs[layer, :, view].reshape(hs.shape[1], -1).astype(np.float64)
    if renorm:
        A = A / np.clip(A.sum(axis=1, keepdims=True), 1e-12, None)
    return A


def main():
    z = np.load(OUT / "attn_b1b2.npz", allow_pickle=False)
    names = [str(x) for x in z["variant_names"]]
    views = [str(v) for v in z["views"]]
    VI = views.index("base_0_rgb")
    D = {n: dict(head_sum=z[f"{n}__head_sum"], rollout=z[f"{n}__rollout"],
                 pieces=[str(p) for p in z[f"{n}__pieces"]],
                 prompt=str(z[f"{n}__prompt"])) for n in names}

    say("=" * 100)
    say("PART B · B1/B2 分析(纯后处理)")
    say("=" * 100)
    say(f"  variants: {len(names)}   views: {views}   主视角 = base_0_rgb (index {VI})")
    say(f"  场景:四条指令面对**逐位相同**的一帧(S0.5 检查 B 已过);attention 噪声地板 = 0")
    say("")
    for n in B1_NAMES + B2_NAMES:
        say(f"  {n:20s} Z={len(D[n]['pieces']):2d}  {D[n]['prompt']!r}")
        say(f"  {'':20s} {D[n]['pieces']}")

    def noun_map(n, noun, layer, renorm=True, src="head_sum"):
        """名词专属图:对该名词的子词 token 行求和(§C / POAP 优化阶段的做法)。"""
        pieces = D[n]["pieces"]
        idx = [i for i, p in enumerate(pieces) if p in NOUNS[n][noun]]
        assert idx, f"{n} 里找不到名词 {noun} 的 token:{pieces}"
        if src == "rollout":
            A = D[n]["rollout"][:, VI].reshape(len(pieces), -1).astype(np.float64)
        else:
            A = rows(D[n]["head_sum"], layer, VI, renorm=renorm)
        return A[idx].sum(axis=0).reshape(N_SIDE, N_SIDE), idx

    # ================================================================ 1 最小对
    say("")
    say("=" * 100)
    say("1 · 最小对(§C 的核心):put the bowl on the PLATE  vs  on top of the CABINET")
    say("=" * 100)
    say("  判读:`bowl` 的图应该几乎不动(同一操作对象);目的地名词的图应该搬家。")
    say("        3、4 的 attention 几乎一样 ⇒ attention 只锁定操作对象,没编码目的地。")
    say("")
    say("  layer | bowl图 Spearman | bowl图 top8 IoU | bowl窗口(plate/cab) | 目的地窗口 plate→cab | 窗口位移(格)")
    for l in range(N_LAYER):
        bp, _ = noun_map("B1_bowl_plate", "bowl", l)
        bc, _ = noun_map("B1_bowl_cabinet", "bowl", l)
        dp, _ = noun_map("B1_bowl_plate", "plate", l)
        dc, _ = noun_map("B1_bowl_cabinet", "cabinet", l)
        wbp, wbc, wdp, wdc = best_window(bp), best_window(bc), best_window(dp), best_window(dc)
        shift = np.hypot(wdp[0] - wdc[0], wdp[1] - wdc[1])
        say(f"  {l:5d} | {spearman(bp, bc):15.4f} | {topk_iou(bp, bc, 8):15.3f} | "
            f"{str(wbp):>9s}/{str(wbc):<9s} | {str(wdp):>9s}→{str(wdc):<9s} | {shift:11.2f}")

    # ================================================================ 2 B1 正交性
    say("")
    say("=" * 100)
    say("2 · B1 正交性:stove / bottle+rack / bowl+plate 的物体集互不相交")
    say("=" * 100)
    say("  用每条指令的【全部名词求和】图两两比。互不相交 ⇒ 相关应该低。")
    say("")

    def all_noun_map(n, l, renorm=True):
        acc = None
        for noun in NOUNS[n]:
            m, _ = noun_map(n, noun, l, renorm=renorm)
            acc = m if acc is None else acc + m
        return acc

    pairs = list(itertools.combinations(B1_NAMES, 2))
    say("  layer | " + " | ".join(f"{a.replace('B1_','')}~{b.replace('B1_','')}"[:19].rjust(19)
                                  for a, b in pairs))
    for l in range(N_LAYER):
        say(f"  {l:5d} | " + " | ".join(
            f"{spearman(all_noun_map(a, l), all_noun_map(b, l)):19.4f}" for a, b in pairs))

    # ================================================================ 3 B2 稳定性
    say("")
    say("=" * 100)
    say("3 · B2 稳定性:同一任务、五种改写 —— `bowl` 与 `plate` 的图应该稳定")
    say("=" * 100)
    ref = "B1_bowl_plate"
    for noun in ("bowl", "plate"):
        say(f"  --- 名词 `{noun}`,与 {ref} 比 Spearman")
        say("  layer | " + " | ".join(f"{n.replace('B2_',''):>13s}" for n in B2_NAMES))
        for l in range(N_LAYER):
            r, _ = noun_map(ref, noun, l)
            say(f"  {l:5d} | " + " | ".join(
                f"{spearman(r, noun_map(n, noun, l)[0]):13.4f}" for n in B2_NAMES))
        say("")

    # ================================================================ 4 判读
    say("=" * 100)
    say("4 · 判读(B1 × B2):换任务的差异是否**明显大于**换措辞的差异?")
    say("=" * 100)
    say("  理想:换任务变、换措辞不变。任一反向都是对'用 attention 定位'的实质质疑。")
    say("")
    say("  用同一把尺子:`bowl` 名词图的 Spearman")
    say("    · 换措辞组 = B1_bowl_plate vs 五个 B2 改写")
    say("    · 换任务组 = B1_bowl_plate vs B1_bowl_cabinet(最小对,只换目的地)")
    say("")
    say("  layer | 换措辞 min | 换措辞 mean | 换任务(最小对) | 判读")
    verdicts = []
    for l in range(N_LAYER):
        rb, _ = noun_map(ref, "bowl", l)
        s_para = [spearman(rb, noun_map(n, "bowl", l)[0]) for n in B2_NAMES]
        s_task = spearman(rb, noun_map("B1_bowl_cabinet", "bowl", l)[0])
        gap = min(s_para) - s_task
        v = "✅ 换措辞更稳" if gap > 0.05 else ("~ 分不开" if gap > -0.05 else "⚠️ 反向")
        verdicts.append((l, min(s_para), float(np.mean(s_para)), s_task, gap, v))
        say(f"  {l:5d} | {min(s_para):10.4f} | {np.mean(s_para):11.4f} | "
            f"{s_task:14.4f} | {v} (gap={gap:+.4f})")

    say("")
    good = [v for v in verdicts if v[5].startswith("✅")]
    say(f"  ⇒ 18 层里有 {len(good)} 层满足'换措辞比换任务更稳'")
    if good:
        best = max(good, key=lambda v: v[4])
        say(f"  ⇒ §A3 要求'取对 attention 最有利的那层':gap 最大的是 **layer {best[0]}**"
            f"(换措辞 min={best[1]:.4f} vs 换任务={best[3]:.4f},gap={best[4]:+.4f})")

    # ================================================================ 5 sink 对照
    say("")
    say("=" * 100)
    say("5 · §A-4 对照:**不做逐行重归一化**会怎样(证明 §A3 第 2 步是承重的)")
    say("=" * 100)
    say("  统计量用'全 token max'(POAP 的定位量)。若不重归一化,sink token `\\n`")
    say("  (质量 0.381,是其余 token 的 11–18 倍)会主导整张图 ⇒ 跨指令几乎无差别。")
    say("")
    say("  layer | 重归一化后 4 条 B1 两两 Spearman 均值 | 不重归一化 | 差")
    for l in range(N_LAYER):
        def tokmax(n, renorm):
            return rows(D[n]["head_sum"], l, VI, renorm=renorm).max(axis=0)
        with_r = np.mean([spearman(tokmax(a, True), tokmax(b, True)) for a, b in pairs])
        wo_r = np.mean([spearman(tokmax(a, False), tokmax(b, False)) for a, b in pairs])
        say(f"  {l:5d} | {with_r:37.4f} | {wo_r:10.4f} | {with_r - wo_r:+.4f}")
    say("")
    say("  (相关越接近 1 = 跨指令越没差别 = 定位越'指令盲')")

    # ================================================================ 6 rollout
    say("")
    say("=" * 100)
    say("6 · 第二种提取方式:attention rollout(§A3 要求不能只报一种)")
    say("=" * 100)
    say("  rollout 是全层累乘的结果,没有'层'这一维。")
    rb = noun_map(ref, "bowl", 0, src="rollout")[0]
    rc = noun_map("B1_bowl_cabinet", "bowl", 0, src="rollout")[0]
    dp = noun_map(ref, "plate", 0, src="rollout")[0]
    dc = noun_map("B1_bowl_cabinet", "cabinet", 0, src="rollout")[0]
    say(f"  最小对 `bowl` 图:Spearman={spearman(rb, rc):.4f}  top8 IoU={topk_iou(rb, rc, 8):.3f}")
    say(f"  目的地窗口:plate {best_window(dp)} → cabinet {best_window(dc)}")
    s_para = [spearman(rb, noun_map(n, "bowl", 0, src="rollout")[0]) for n in B2_NAMES]
    say(f"  换措辞 `bowl` 图 Spearman:min={min(s_para):.4f} mean={np.mean(s_para):.4f}")
    say(f"  ⇒ gap(换措辞 min − 换任务)= {min(s_para) - spearman(rb, rc):+.4f}")

    # ================================================================ 缺口
    say("")
    say("=" * 100)
    say("⚠️ 还缺的东西(不补齐则以上结论不能定稿)")
    say("=" * 100)
    say("  1. **B3 噪声地板**:同一条指令、**相邻两帧**之间的 attention 差异。")
    say("     指令带来的差异必须**超过**这个地板才算数。现在只有单帧 ⇒ 地板未知,")
    say("     上面所有 Spearman 都缺一个'多少算大'的参照。**这是最重要的缺口。**")
    say("  2. **S0.5 检查 A**(交叉评估):若模型忽略文本,B1 的前提就不成立。")
    say("  3. **B2 各改写句的 clean 成功率**:LIBERO 指令模板化,模型可能对脱离模板的说法很脆。")
    say("     成功率塌了的句子不能进对比 —— 否则比的是'正常 vs 懵了',不是改写鲁棒性。")
    say("     `B2_L2_frontPP`(倒装)最可疑,必须实测。")
    say("  4. attention→世界坐标反投影(§A-5):要把网格位移换成**米**才好解读;")
    say("     depth/K/[R|t] 已在 shared_frame.npz 里,是纯后处理。")

    (OUT / "report_b1b2.txt").write_text("\n".join(_lines) + "\n")
    say("")
    say(f"[written] {OUT/'report_b1b2.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
