#!/usr/bin/env python3
"""把 attn_maps 里的单图拼成对比 montage:**行 = 层,列 = 指令/改写**。

855 张单图没法横向比;要回答"换指令到底动没动",必须把同一层的不同指令并排放。

输出 out/attn_maps/montage/
    montage_{图类型}_{view}_tasks.png      列 = 四条 B1 指令(换任务)
    montage_{图类型}_{view}_rephrase.png   列 = bowl_plate + 五个改写(换措辞)

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/make_attn_montage.py
"""

import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from instructions import GROUPS, META, disp_tok, roles  # noqa: E402  单一来源
from make_attn_maps import best_window, draw_window  # noqa: E402

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = ROOT / "pi05probe" / "out"
MAPS = OUT / "attn_maps"
MON = MAPS / "montage"
N_SIDE = 16
CELL = 144                       # 每格边长
HDR, LBL = 26, 46                # 顶部列标题高、左侧层标签宽
FOOT = 76                        # 底部:放完整指令文本
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17, "rollout"]

# ⚠️ 标签一律用 ASCII:PIL 的默认位图字体没有 CJK 字形,中文会渲染成方块 □。
# 列的构成从 instructions.py 推导:
#   tasks           = 四条原生指令(换任务)
#   rephrase_<组>   = 该组 orig + 它的全部改写(换措辞),每组一张
COLS = {"tasks": [(n, META[n]["group"]) for n in
                  [v for v in META if META[v]["tag"] == "orig" and v != "REPEAT_bowl_plate"]]}
for _gdir, _members in GROUPS.items():
    _g = _gdir.replace("group_", "")
    COLS[f"rephrase_{_g}"] = [
        (n, f"{META[n]['tag']} {'/'.join(META[n]['verbs'])}") for n in
        sorted(_members, key=lambda m: ("orig", "L1", "L2", "L3").index(META[m]["tag"]))]


def wrap(s, width):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def hot_lut():
    t = np.linspace(0, 1, 256)
    return np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], 1)


LUT = hot_lut()


def cell_img(S, base_gray):
    """叠加:alpha 随 saliency 强度变化,低显著区几乎保留原图 ⇒ 能看清物体在哪。
    (之前用固定 alpha=0.55 + 45% 灰底,整张过暗、认不出物体。)"""
    lo, hi = float(S.min()), float(S.max())
    Sn = (S - lo) / (hi - lo) if hi > lo else np.zeros_like(S)
    r = CELL // N_SIDE
    Su = np.kron(Sn, np.ones((r, r)))
    heat = LUT[(Su * 255).astype(np.uint8)] * 255.0
    a = (0.12 + 0.80 * Su)[:, :, None]
    return np.clip((1 - a) * base_gray + a * heat, 0, 255).astype(np.uint8)


def main():
    sal = np.load(MAPS / "saliency.npz", allow_pickle=False)
    fr = np.load(OUT / "shared_frame.npz", allow_pickle=False)
    az = np.load(OUT / "attn_b1b2.npz", allow_pickle=False)
    t0 = str(fr["tasks"][0])
    imgs = {"base": fr[f"{t0}__img224"], "wrist": fr[f"{t0}__wrist224"]}
    MON.mkdir(parents=True, exist_ok=True)

    made = 0
    for kind in ("noun", "verb", "allmax", "allsum", "func"):
        for view in ("base",):
            g = np.asarray(Image.fromarray(imgs[view]).resize((CELL, CELL), Image.NEAREST),
                           dtype=np.float64) @ np.array([0.299, 0.587, 0.114])
            base_gray = np.repeat(g[:, :, None], 3, axis=2)
            for setname, cols in COLS.items():
                if not all(f"{n}|{view}|00|{kind}" in sal for n, _ in cols):
                    continue
                W = LBL + CELL * len(cols)
                H = HDR + CELL * len(LAYERS) + FOOT
                im = Image.new("RGB", (W, H), (14, 14, 14))
                d = ImageDraw.Draw(im)
                d.text((3, 3), f"map={kind}   view={view}   t=000   "
                               f"rows=layer  cols=instruction", fill=(255, 220, 140))
                for ci, (_, lab) in enumerate(cols):
                    d.text((LBL + ci * CELL + 3, 15), lab[:24], fill=(200, 225, 255))
                for ri, L in enumerate(LAYERS):
                    ltag = "rollout" if L == "rollout" else f"{L:02d}"
                    d.text((3, HDR + ri * CELL + CELL // 2 - 4),
                           "roll" if L == "rollout" else f"L{ltag}", fill=(180, 180, 180))
                    for ci, (n, _) in enumerate(cols):
                        k = f"{n}|{view}|{ltag}|{kind}"
                        if k not in sal:
                            continue
                        # 青色框 = **实际选出的 3×3 窗口**(§A2 第 5 步)。
                        # 必须画:很多层它落在跨指令不变的边缘/角落伪影上,不在任何物体上。
                        c = draw_window(Image.fromarray(cell_img(sal[k], base_gray)),
                                        best_window(sal[k])[0], CELL)
                        im.paste(c, (LBL + ci * CELL, HDR + ri * CELL))
                # ---- 底部:每列的**完整指令文本** + 这张图实际用了哪些 token
                y0 = HDR + CELL * len(LAYERS) + 3
                d.line([(0, y0 - 2), (W, y0 - 2)], fill=(70, 70, 70))
                for ci, (n, _) in enumerate(cols):
                    x = LBL + ci * CELL + 3
                    prompt = str(az[f"{n}__prompt"])
                    pieces = [str(p) for p in az[f"{n}__pieces"]]
                    for li, line in enumerate(wrap(prompt, 23)[:3]):
                        d.text((x, y0 + li * 12), line, fill=(255, 255, 255))
                    tag = {"noun": "nouns", "verb": "verb"}.get(kind)
                    if tag:
                        idx = roles(n, pieces)["noun" if kind == "noun" else "verb"]
                        used = [disp_tok(pieces[i]) for i in idx]
                        for li, line in enumerate(wrap(f"{tag}: {' '.join(used)}", 23)[:2]):
                            d.text((x, y0 + 38 + li * 12), line, fill=(150, 210, 150))
                    else:
                        d.text((x, y0 + 38), f"Z={len(pieces)}", fill=(150, 150, 150))
                if kind in ("allmax", "allsum"):
                    d.text((3, y0 + 62), "WARNING: all-token maps are BIASED across "
                                         "instructions (Z differs: 6/8/9/10) - use `noun`",
                           fill=(255, 140, 140))
                fn = MON / f"montage_{kind}_{view}_{setname}.png"
                im.save(fn)
                made += 1
                print(f"[written] {fn}", flush=True)
    print(f"\n共 {made} 张 montage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
