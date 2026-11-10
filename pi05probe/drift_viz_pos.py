#!/usr/bin/env python3
"""可视化位置实验(§4.25):每组 = 整图(贴 P)/ 足迹放大(贴 P)/ 足迹放大(贴同 seed 随机图案)。
图像就是评测 A 里模型看到的:clean·(1−mask) + P·mask,取观测池里可见足迹最大的一帧。

用法: ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_viz_pos.py
产物 out/viz_pos_paste.png
"""
import pathlib

import cv2
import numpy as np

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
SEED_RAND = 20260902                     # 与 drift_eval_digital.py 一致
ZOOM = 5
GROUPS = [  # (patch tag, 标签)
    ("drift13", "0.13m @ cell 101 (top influence)  ASR 54%"),
    ("ctrl161_13", "0.13m @ cell 161 (low influence)  ASR 0%"),
    ("drift06", "0.06m @ cell 101 (top influence)  ASR 0%"),
    ("ctrl161_06", "0.06m @ cell 161 (low influence)  ASR 0%"),
]
# 所有组共用的放大窗口大小(像素,224 图里),保证各组放大倍率一致、可直接比大小
WIN = 56


def label(img, text, scale=0.42):
    img = img.copy()
    for col, th in (((255, 255, 255), 3), ((0, 0, 0), 1)):
        cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)
    return img


def main():
    R = np.random.RandomState(SEED_RAND).randint(0, 256, (224, 224, 3), dtype=np.uint8)
    rows = []
    for tag, title in GROUPS:
        d = np.load(OUT / f"drift_patch_{tag}.npz", allow_pickle=True)
        pool = np.load(OUT / f"{str(d['pool'])}.npz", allow_pickle=True)
        px = pool["masks"].reshape(pool["masks"].shape[0], -1).sum(1)
        k = int(np.argmax(px))
        clean = pool["clean_img224"][k]
        m = pool["masks"][k].astype(bool)
        P = np.clip(d["P"].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
        paste = np.where(m[..., None], P, clean)
        rand = np.where(m[..., None], R, clean)

        ys, xs = np.where(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        r0 = int(np.clip(cy - WIN // 2, 0, 224 - WIN)); c0 = int(np.clip(cx - WIN // 2, 0, 224 - WIN))
        z = lambda im: cv2.resize(im[r0:r0 + WIN, c0:c0 + WIN], None, fx=ZOOM, fy=ZOOM,
                                  interpolation=cv2.INTER_NEAREST)
        full = cv2.resize(paste, (WIN * ZOOM, WIN * ZOOM), interpolation=cv2.INTER_NEAREST)
        box = full.copy()
        s = WIN * ZOOM / 224
        cv2.rectangle(box, (int(c0 * s), int(r0 * s)), (int((c0 + WIN) * s), int((r0 + WIN) * s)),
                      (255, 255, 0), 2)
        tiles = [label(box, title, 0.5), label(z(paste), f"zoom x{ZOOM}: optimized P  ({int(m.sum())}px)"),
                 label(z(rand), f"zoom x{ZOOM}: random (control)")]
        pad = lambda x: cv2.copyMakeBorder(x, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        rows.append(np.concatenate([pad(t) for t in tiles], 1))

    grid = np.concatenate(rows, 0)
    outp = OUT / "viz_pos_paste.png"
    cv2.imwrite(str(outp), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[written] {outp}  {grid.shape}")


if __name__ == "__main__":
    main()
