#!/usr/bin/env python3
"""收尾项 C:给 B1/B2 的 Spearman 配上 **B3 噪声地板**,把"0.95 算高吗"变成有参照的判读。

report_b1b2.txt 的所有 Spearman 都是**单帧 t000**、且**没有地板** —— 三份 report 自己
都在末尾喊这是最大缺口。本脚本用 attn_traj npz(已含 9 个 variant 沿 16 帧的 attention),
纯后处理产出:
  · 地板 = 同一指令、相邻两帧、同一张图的 Spearman(该图自然漂移多少)
  · 换措辞 = bowl_plate vs 5 个改写,共享 `bowl` token 图(名词不变,隔离改写效应)
  · 换任务(最小对) = bowl_plate vs bowl_cabinet,共享 `bowl` token 图
  · B1 正交性 = stove / bottle_rack / bowl_plate 的**全名词求和图**两两比
所有比较都沿同 16 帧逐帧算再平均(§A1:与地板同口径),对照地板判读。

判据(计划 B1×B2):理想 = 换任务变(< 地板)、换措辞不变(≥ 地板)。

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/report_floor.py
"""
import pathlib
import numpy as np

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NPZ = OUT / "attn_traj_put_the_bowl_on_the_plate.npz"
TXT = OUT / "report_floor.txt"

# 名词词表(全名词求和图用)。只认内容名词,不含功能词。
NOUNS = ("bowl", "plate", "cabinet", "stove", "wine", "bottle", "rack")
REPHRASE = ["bowl_plate_L1_place", "bowl_plate_L1_set", "bowl_plate_L1_move",
            "bowl_plate_L2_front", "bowl_plate_L3_please"]


def spearman(a, b):
    def rank(x):
        r = np.empty(len(x), np.float64)
        r[np.argsort(x, kind="stable")] = np.arange(len(x))
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    ra, rb = rank(np.asarray(a, np.float64)), rank(np.asarray(b, np.float64))
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def noun_rows(pieces):
    return [i for i, p in enumerate(pieces) if any(n in str(p).lower() for n in NOUNS)]


def tok_rows(pieces, want):
    return [i for i, p in enumerate(pieces) if want in str(p).lower()]


def main():
    d = np.load(NPZ, allow_pickle=True)
    views = [str(v) for v in d["views"]]
    base = views.index("base") if "base" in views else 0
    T = len(d["ks"])
    A = {}
    P = {}
    for k in d.files:
        if k.endswith("__attn"):
            name = k[:-len("__attn")]
            A[name] = d[k]
            P[name] = [str(x) for x in d[f"{name}__pieces"]]
    L = A["bowl_plate_orig"].shape[1]

    def bowl_map(name):
        i = tok_rows(P[name], "bowl")[0]
        return [A[name][t, :, i, base] for t in range(T)]  # per-frame (L,16,16)

    def noun_map(name):
        rows = noun_rows(P[name])
        return [A[name][t, :, rows, base].sum(axis=0) for t in range(T)]  # (L,16,16)

    lines = []
    def out(s=""):
        print(s, flush=True); lines.append(s)

    out("=" * 100)
    out(f"收尾项 C · B1/B2 对照 B3 噪声地板   T={T} 帧   view={views[base]}   renorm={str(d['renorm'])}")
    out(f"  帧对应 env 步: {d['ts'].tolist()}")
    out("  地板 = bowl_plate 的 `bowl` 图,相邻两帧 Spearman(该图自然漂移)")

    # 预备:各 variant 的 bowl 图(逐帧)
    bp = bowl_map("bowl_plate_orig")
    floor = [np.mean([spearman(bp[t][l].ravel(), bp[t + 1][l].ravel()) for t in range(T - 1)])
             for l in range(L)]

    bc = bowl_map("bowl_cabinet_orig")
    task_sp = [np.mean([spearman(bp[t][l].ravel(), bc[t][l].ravel()) for t in range(T)]) for l in range(L)]

    rp_maps = {r: bowl_map(r) for r in REPHRASE}
    rp_sp = {}
    for r in REPHRASE:
        rm = rp_maps[r]
        rp_sp[r] = [np.mean([spearman(bp[t][l].ravel(), rm[t][l].ravel()) for t in range(T)]) for l in range(L)]

    out("\n" + "=" * 100)
    out("1 · B1×B2 判读(共享 `bowl` 图,对照地板)")
    out("=" * 100)
    out("  地板↑=图越稳。换措辞应 ≥ 地板(改写不动图);换任务应 < 地板(换目的地动图)。")
    out("")
    out("  layer |  地板  | 换措辞 min | 换措辞 mean | 换任务(最小对) | 判读")
    ideal = 0
    for l in range(L):
        rmin = min(rp_sp[r][l] for r in REPHRASE)
        rmean = np.mean([rp_sp[r][l] for r in REPHRASE])
        # 换措辞不变(≥地板)且 换任务变(<地板) = 理想
        stable_rephrase = rmean >= floor[l]
        moved_task = task_sp[l] < floor[l]
        if stable_rephrase and moved_task:
            verd = "✅ 理想(措辞稳/任务动)"; ideal += 1
        elif not stable_rephrase and not moved_task:
            verd = "⚠️ 全反(措辞动/任务稳)"
        elif not moved_task:
            verd = "· 任务也没动"
        else:
            verd = "· 措辞也动了"
        out(f"  {l:5d} | {floor[l]:.4f} | {rmin:10.4f} | {rmean:11.4f} | {task_sp[l]:14.4f} | {verd}")
    out(f"\n  ⇒ 18 层里 {ideal} 层达到理想(换措辞≥地板 且 换任务<地板)")

    out("\n" + "=" * 100)
    out("2 · B1 正交性(全名词求和图,对照地板):物体集互不相交 ⇒ 应低于地板")
    out("=" * 100)
    tasks = ["stove_orig", "bottle_rack_orig", "bowl_plate_orig"]
    short = {"stove_orig": "stove", "bottle_rack_orig": "bottle_rack", "bowl_plate_orig": "bowl_plate"}
    nmap = {tk: noun_map(tk) for tk in tasks}
    out("\n  layer |  地板  | stove~bottle | stove~bowl_pl | bottle~bowl_pl | 都<地板?")
    for l in range(L):
        def pair(x, y):
            return np.mean([spearman(nmap[x][t][l].ravel(), nmap[y][t][l].ravel()) for t in range(T)])
        s_b = pair("stove_orig", "bottle_rack_orig")
        s_p = pair("stove_orig", "bowl_plate_orig")
        b_p = pair("bottle_rack_orig", "bowl_plate_orig")
        alllow = "✅" if max(s_b, s_p, b_p) < floor[l] else ""
        out(f"  {l:5d} | {floor[l]:.4f} | {s_b:12.4f} | {s_p:13.4f} | {b_p:14.4f} | {alllow}")

    out("\n  读法:全名词图混入了'名词本来就不同'这个 confound(见 report_traj_minpair 的说明),")
    out("        所以正交性是弱证据;最小对(第 1 节)才是干净的'任务语义'判据。")

    TXT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {TXT}")


if __name__ == "__main__":
    main()
