#!/usr/bin/env python3
"""全路径 attention 出图:**时间轴**热力图 + GIF。

输入 out/attn_traj_<task>.npz + out/traj_<task>.npz
输出 out/attn_maps/traj_<task>/
    timegrid_{variant}_{kind}_L{layer}.png   行/列铺开全部时间步,每格叠在**该帧自己的图**上
    anim_{variant}_{kind}_L{layer}.gif       同上的动画版,看 attention 跟不跟着机械臂走
    bylayer_t{t}_{kind}_{variant}.png        某一帧:行=层(和单帧那套一致)

⚠️ 每格必须叠在**该时间步自己的观测图**上 —— 机械臂在动,叠到 t0 那张会完全错位。

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/make_traj_maps.py
"""

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = ROOT / "pi05probe" / "out"
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from instructions import META, disp_tok, roles  # noqa: E402

N_SIDE = 16
CELL = 128
HDR, LBL, FOOT = 26, 40, 44
LAYERS_MAIN = [4, 8, 12]          # 单图/GIF 主用的层(中层,单帧实验里定位最清楚的一段)
LAYERS_ALL = [0, 2, 4, 6, 8, 10, 12, 14, 16, 17]


def hot_lut():
    t = np.linspace(0, 1, 256)
    return np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)], 1)


LUT = hot_lut()


def cell(S, img224, size=CELL):
    """热图叠在**这一帧**的观测上;alpha 随强度变化,低显著区保留原图。"""
    base = np.asarray(Image.fromarray(img224).resize((size, size), Image.NEAREST), np.float64)
    g = np.repeat((base @ np.array([0.299, 0.587, 0.114]))[:, :, None], 3, 2)
    lo, hi = float(S.min()), float(S.max())
    Sn = (S - lo) / (hi - lo) if hi > lo else np.zeros_like(S)
    Su = np.kron(Sn, np.ones((size // N_SIDE, size // N_SIDE)))
    heat = LUT[(Su * 255).astype(np.uint8)] * 255.0
    a = (0.12 + 0.80 * Su)[:, :, None]
    return np.clip((1 - a) * g + a * heat, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="put_the_bowl_on_the_plate")
    ap.add_argument("--kinds", default="noun,allmax")
    args = ap.parse_args()

    az = np.load(OUT / f"attn_traj_{args.task}.npz", allow_pickle=False)
    tz = np.load(OUT / f"traj_{args.task}.npz", allow_pickle=False)
    names = [str(x) for x in az["variant_names"]]
    views = [str(v) for v in az["views"]]
    renorm = str(az["renorm"])
    ks = az["ks"].tolist()
    ts = az["ts"].tolist()
    nF = len(ks)
    imgs = [tz[f"f{k:03d}__img224"] for k in ks]

    dst = OUT / "attn_maps" / f"traj_{args.task}"
    dst.mkdir(parents=True, exist_ok=True)

    def saliency(name, t, layer, kind, vi=0):
        A = az[f"{name}__attn"][t, layer]                    # (Z, V, 16, 16)
        pieces = [str(p) for p in az[f"{name}__pieces"]]
        pv = [A[:, j].reshape(A.shape[0], -1).astype(np.float64) for j in range(len(views))]
        if renorm == "none":
            R = pv[vi]
        else:
            den = pv[0].sum(1, keepdims=True) if renorm == "base" else \
                sum(p.sum(1, keepdims=True) for p in pv)
            R = pv[vi] / np.clip(den, 1e-12, None)
        r = roles(name, pieces)
        if kind == "allmax":
            S = R.max(axis=0)
        elif kind == "allsum":
            S = R.sum(axis=0)
        else:
            idx = r[kind]
            S = R[idx].sum(axis=0) if idx else np.zeros(R.shape[1])
        return S.reshape(N_SIDE, N_SIDE), pieces, r

    made = 0
    ncol = 8
    nrow = (nF + ncol - 1) // ncol
    for name in names:
        prompt = META[name]["prompt"]
        for kind in args.kinds.split(","):
            for layer in LAYERS_MAIN:
                # ---------- timegrid:全部时间步铺开
                W, H = LBL + CELL * ncol, HDR + CELL * nrow + FOOT
                im = Image.new("RGB", (W, H), (14, 14, 14))
                d = ImageDraw.Draw(im)
                d.text((3, 3), f"{prompt}   |   map={kind}  layer={layer}  "
                               f"view=base  renorm={renorm}  ({nF} frames along rollout)",
                       fill=(255, 220, 140))
                for i in range(nF):
                    r, c = divmod(i, ncol)
                    S, pieces, rl = saliency(name, i, layer, kind)
                    im.paste(Image.fromarray(cell(S, imgs[i])),
                             (LBL + c * CELL, HDR + r * CELL))
                    d.text((LBL + c * CELL + 3, HDR + r * CELL + 2),
                           f"k={ks[i]} t={ts[i]}", fill=(255, 255, 255))
                y = HDR + CELL * nrow + 3
                used = ([disp_tok(pieces[j]) for j in rl[kind]]
                        if kind in ("noun", "verb", "func") else
                        [disp_tok(p) for p in pieces])
                d.text((3, y), f"tokens ({kind}): {' '.join(used)}", fill=(150, 210, 150))
                d.text((3, y + 14), "each cell is overlaid on ITS OWN frame "
                                    "(the arm moves; overlaying on t0 would be misaligned)",
                       fill=(160, 200, 255))
                if kind in ("allmax", "allsum"):
                    d.text((3, y + 28), "WARNING: all-token maps are BIASED across "
                                        "instructions (Z differs) - use `noun`",
                           fill=(255, 140, 140))
                im.save(dst / f"timegrid_{name}_{kind}_L{layer:02d}.png")
                made += 1

                # ---------- GIF
                fr = []
                for i in range(nF):
                    S, _, _ = saliency(name, i, layer, kind)
                    big = Image.fromarray(cell(S, imgs[i], size=256))
                    canvas = Image.new("RGB", (256, 256 + 26), (14, 14, 14))
                    canvas.paste(big, (0, 0))
                    dd = ImageDraw.Draw(canvas)
                    dd.text((3, 258), f"{prompt[:38]}", fill=(255, 255, 255))
                    dd.text((3, 269), f"{kind} L{layer}  k={ks[i]} t={ts[i]}",
                            fill=(255, 220, 140))
                    fr.append(canvas)
                fr[0].save(dst / f"anim_{name}_{kind}_L{layer:02d}.gif", save_all=True,
                           append_images=fr[1:], duration=250, loop=0)
                made += 1
        print(f"  [{name}] 完成", flush=True)

    print(f"\n共 {made} 个文件 → {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
