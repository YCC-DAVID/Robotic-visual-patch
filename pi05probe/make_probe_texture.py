#!/usr/bin/env python3
"""生成 S2 探针纹理:块状随机噪声(计划 S2 硬要求)。

为什么块状而非逐像素随机(计划原文):
  逐像素随机噪声在渲染降采样(256→224 再到 patch 网格)后被平均成灰块
  ⇒ 变成在测"遮挡"而非"扰动"。所以用**大色块**,降采样后仍保留结构。

规格:色块 12 px、每块 RGB 各通道满 0–255(高饱和)、固定 seed、全网格统一同一张。
不放文字/logo/可读符号/真实物体照片(计划明确禁止)。

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/make_probe_texture.py
"""
import argparse
import pathlib
import numpy as np
from PIL import Image

CFG = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/config")
SIZE = 256
BLOCK = 12          # 8–16 px 之间
SEED = 20260810     # 默认 seed(= 第一批扫描用的那张)


def main():
    # 纹理轴(计划 S3):同一位置换不同随机纹理,排序应当一致。
    # 只改 seed、不改 block/饱和度 ⇒ 统计性质相同,是受控对照。
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--out", default=None, help="默认 config/probe_texture.png(seed 非默认时加后缀)")
    args = ap.parse_args()

    out = pathlib.Path(args.out) if args.out else (
        CFG / ("probe_texture.png" if args.seed == SEED else f"probe_texture_s{args.seed}.png"))
    rng = np.random.RandomState(args.seed)
    nb = (SIZE + args.block - 1) // args.block
    blocks = rng.randint(0, 256, size=(nb, nb, 3), dtype=np.uint8)   # 每块随机满 0–255
    img = np.repeat(np.repeat(blocks, args.block, axis=0), args.block, axis=1)[:SIZE, :SIZE]
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, "RGB").save(out)
    uniq = len(np.unique(img.reshape(-1, 3), axis=0))
    print(f"[written] {out}  {img.shape}  block={args.block}px  seed={args.seed}  "
          f"唯一色={uniq}  min/max={img.min()}/{img.max()}")


if __name__ == "__main__":
    main()
