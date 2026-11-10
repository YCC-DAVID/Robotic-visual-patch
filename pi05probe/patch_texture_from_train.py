#!/usr/bin/env python3
"""对抗 patch —— Phase B→C 衔接:把训练好的归一化 patch P(224 图像空间的足迹像素)
反 warp 成方形纹理 PNG,供 scene_patch 注入做物理 rollout 验收。

P 在 base_0_rgb 归一化空间 [-1,1] ⇒ 先还原 [0,255];quad224 是足迹四角(row,col);
用单应把图像里的足迹四边形反 warp 成正方形纹理(cv2)。存 config/probe_texture_adv.png。
同时存一张合成预览(模型实际看到的对抗图),便于肉眼检查。

用法: ~/miniconda3/envs/openpi-libero/bin/python pi05probe/patch_texture_from_train.py
"""
import argparse
import pathlib
import numpy as np
import cv2

PI05 = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe")
OUT = PI05 / "out"
TEX = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="away")
    args = ap.parse_args()
    d = np.load(OUT / f"patch_trained_{args.loss}.npz", allow_pickle=True)
    P = d["P"]                                  # (3,224,224) [-1,1]
    quad = d["quad224"].astype(np.float32)      # (4,2) row,col;顺序 BL,BR,TR,TL(世界角)
    img = (((P.transpose(1, 2, 0) + 1) / 2) * 255).clip(0, 255).astype(np.uint8)  # (224,224,3) RGB

    src = quad[:, ::-1].copy()                  # cv2 要 (x=col,y=row)
    dst = np.array([[0, 0], [TEX - 1, 0], [TEX - 1, TEX - 1], [0, TEX - 1]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    tex = cv2.warpPerspective(img, H, (TEX, TEX))          # 足迹 → 方形纹理
    texp = PI05 / "config" / f"probe_texture_adv_{args.loss}.png"
    cv2.imwrite(str(texp), tex[:, :, ::-1])   # RGB→BGR
    print(f"[written] {texp}  ({TEX}×{TEX})", flush=True)

    # 合成预览:把优化 patch 贴回一张 clean 图看效果
    pp = np.load(OUT / "patch_prep.npz", allow_pickle=True)
    clean = pp["clean_img224"][0].astype(np.float32)       # (224,224,3) [0,255]
    m = pp["vis_mask"][0][..., None].astype(np.float32)
    comp = (clean * (1 - m) + img.astype(np.float32) * m).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / f"patch_adv_preview_{args.loss}.png"), comp[:, :, ::-1])
    print(f"[written] {OUT/('patch_adv_preview_'+args.loss+'.png')}  (模型看到的对抗图)", flush=True)
    print(f"[info] 训练:随机偏移 {float(d['rand_dev']):.3f} → 优化后 {float(d['final_dev']):.3f} "
          f"(×{float(d['final_dev'])/max(float(d['rand_dev']),1e-6):.2f})", flush=True)


if __name__ == "__main__":
    main()
