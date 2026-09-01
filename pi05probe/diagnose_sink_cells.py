#!/usr/bin/env python3
"""诊断图像侧的 register / spatial-sink patch,以及 POAP 的 3×3 定位有多少落在它们上面。

起因(人工看图发现):多张热图的左下角、左边缘、左上角有孤立的白/黄方块,
**跨全部指令、跨多个层出现在同一格**。L00 的左下白块尤其顽固。
这是 ViT 里熟知的 register / high-norm artifact patch(图像侧的 attention sink),
和文本侧的 `\\n` sink 是两回事:
    · 文本侧 sink = 某个 **query 行**(`\\n`)吃掉大量质量
    · 图像侧 sink = 某些 **key 列**(固定的几个 patch)被所有 query 注意
后者更致命,因为 §A2 第 5 步的 3×3 窗口是在**图像格**上滑的 ⇒ 会直接选中它们。

本脚本回答三个问题:
  1. 哪些格子是跨指令不变的高值格?(定量的伪影 mask)
  2. 它们是 key 侧还是 query 侧的?(对全部 text token 取 **min**,连最小值都高 ⇒ key 侧)
  3. POAP 的 3×3 窗口有多少比例落在这些格子 / 落在图像边界上?

⚠️ 只用 t000 的数据就能做,不依赖轨迹。

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/diagnose_sink_cells.py
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = ROOT / "pi05probe" / "out"
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from instructions import META, roles  # noqa: E402
from make_attn_maps import best_window  # noqa: E402

N_SIDE, N_IMG, N_LAYER, N_HEAD = 16, 256, 18, 8
WIN = 3
_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def mm(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def main():
    z = np.load(OUT / "attn_b1b2.npz", allow_pickle=False)
    names = [str(n) for n in z["variant_names"] if str(n) != "REPEAT_bowl_plate"]
    views = [str(v) for v in z["views"]]
    VI = views.index("base_0_rgb")

    say("=" * 100)
    say("图像侧 register / sink patch 诊断(t000,主视角 base_0_rgb)")
    say("=" * 100)
    say(f"  variants = {len(names)}   grid = {N_SIDE}×{N_SIDE}   窗口 s={WIN}")
    say("  ⚠️ 3×3 窗口中心只能取 1..14 ⇒ 中心 = 1 或 14 就说明**窗口贴着图像边界**")
    say("")

    def rows(name, layer):
        """(Z,256),按 img512 口径逐行归一化(与出图脚本一致)。"""
        hs = z[f"{name}__head_sum"]                  # (L,Z,V,16,16)
        pv = [hs[layer, :, j].reshape(hs.shape[1], -1).astype(np.float64)
              for j in range(len(views))]
        den = sum(p.sum(1, keepdims=True) for p in pv)
        return pv[VI] / np.clip(den, 1e-12, None)

    # ============================================================ 1 伪影 mask
    say("=" * 100)
    say("1 · 跨指令不变的高值格 —— 定量伪影 mask")
    say("=" * 100)
    say("  做法:每个 variant 的 allmax 图各自 min-max 到 [0,1],然后对 variant 取 **min**。")
    say("        min 高 ⇒ 这一格对**每一条指令**都亮 ⇒ 与指令无关的结构。")
    say("")
    say("  layer | 跨指令 min 最高的 6 格 (row,col):min值            | 这些格里贴边的占比")
    artifact = {}
    for l in range(N_LAYER):
        maps = [mm(rows(n, l).max(axis=0)) for n in names]      # 每个 variant 的 allmax
        mn = np.min(np.stack(maps), axis=0).reshape(N_SIDE, N_SIDE)
        flat = mn.ravel()
        top = np.argsort(flat)[::-1][:6]
        cells = [(int(t // N_SIDE), int(t % N_SIDE), float(flat[t])) for t in top]
        artifact[l] = mn
        onb = sum(1 for r, c, _ in cells if r in (0, 1, N_SIDE - 2, N_SIDE - 1)
                  or c in (0, 1, N_SIDE - 2, N_SIDE - 1))
        say(f"  {l:5d} | " + "  ".join(f"({r:2d},{c:2d}):{v:.2f}" for r, c, v in cells)
            + f" | {onb}/6")

    # ============================================================ 2 key 侧还是 query 侧
    say("")
    say("=" * 100)
    say("2 · 是 key 侧(固定 patch 被所有人注意)还是 query 侧(某个 text token 的偏好)?")
    say("=" * 100)
    say("  做法:对**全部 text token 行**取 min。若连最小值都在同几格上高,")
    say("        说明不是某个 token 的偏好,而是那几个 image patch 本身在吸引全部 query")
    say("        ⇒ key 侧的 register patch。")
    say("")
    say("  layer | 对 token 取 min 后最高的 4 格 | 该格 min/该图均值 | 结论")
    for l in range(N_LAYER):
        A = rows("bowl_plate_orig", l)                  # (Z,256)
        mn_tok = A.min(axis=0)                          # 对 token 取 min
        top = np.argsort(mn_tok)[::-1][:4]
        ratio = float(mn_tok[top[0]] / max(mn_tok.mean(), 1e-12))
        cells = [(int(t // N_SIDE), int(t % N_SIDE)) for t in top]
        verdict = "key 侧 register" if ratio > 5 else ("偏 key 侧" if ratio > 2 else "不明显")
        say(f"  {l:5d} | " + " ".join(f"({r:2d},{c:2d})" for r, c in cells)
            + f" | {ratio:17.1f}× | {verdict}")

    # ============================================================ 3 窗口落在哪
    say("")
    say("=" * 100)
    say("3 · ⚠️ 关键:POAP 的 3×3 窗口有多少落在伪影 / 边界上")
    say("=" * 100)
    say("  伪影格定义:该层跨指令 min 图的 top-8 格(第 1 节的做法)。")
    say("  '命中伪影' = 窗口的 3×3 范围与任一伪影格相交。")
    say("")
    for kind in ("allmax", "allsum", "noun"):
        say(f"  ---- map = {kind}")
        say("  layer | 窗口中心(23 条指令)                       | 贴边 | 命中伪影 | 唯一位置数")
        tot_b = tot_a = tot_n = 0
        for l in range(N_LAYER):
            mn = artifact[l]
            art = set(map(tuple, np.argwhere(
                mn >= np.sort(mn.ravel())[::-1][7])))     # top-8 格
            wins, nb, na = [], 0, 0
            for n in names:
                A = rows(n, l)
                if kind == "allmax":
                    S = A.max(axis=0)
                elif kind == "allsum":
                    S = A.sum(axis=0)
                else:
                    S = A[roles(n, [str(p) for p in z[f"{n}__pieces"]])["noun"]].sum(axis=0)
                w, _ = best_window(S.reshape(N_SIDE, N_SIDE))
                wins.append(w)
                if w[0] in (1, N_SIDE - 2) or w[1] in (1, N_SIDE - 2):
                    nb += 1
                span = {(w[0] + di, w[1] + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)}
                if span & art:
                    na += 1
            uniq = len(set(wins))
            tot_b += nb; tot_a += na; tot_n += len(names)
            shown = " ".join(f"({a},{b})" for a, b in sorted(set(wins)))[:44]
            say(f"  {l:5d} | {shown:44s} | {nb:2d}/{len(names)} | "
                f"{na:2d}/{len(names)}   | {uniq:2d}")
        say(f"  ==== {kind} 合计:贴边 {tot_b}/{tot_n} = {100*tot_b/tot_n:.0f}%   "
            f"命中伪影 {tot_a}/{tot_n} = {100*tot_a/tot_n:.0f}%")
        say("")

    say("=" * 100)
    say("读法")
    say("=" * 100)
    say("  · '唯一位置数' 接近 1 ⇒ 该层 23 条指令选出**同一个**窗口 ⇒ 定位与指令无关。")
    say("  · '贴边' 和 '命中伪影' 比例高 ⇒ POAP 的定位实际在挑 register patch,不是物体。")
    say("  · 这两项都是**只用 t000 就能确立**的结论,不依赖 B3 地板。")
    say("  · 注意 `noun` 图应当比 `allmax` 好(不含 `\\n` sink 那一行);")
    say("    若 `noun` 也大量贴边,说明伪影是 key 侧的,换 token 子集救不了。")

    (OUT / "sink_cells.txt").write_text("\n".join(_lines) + "\n")
    np.savez_compressed(OUT / "sink_cells.npz",
                        **{f"artifact_min_L{l:02d}": artifact[l] for l in range(N_LAYER)})
    say("")
    say(f"[written] {OUT/'sink_cells.txt'}")
    say(f"[written] {OUT/'sink_cells.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
