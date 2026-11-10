#!/usr/bin/env python3
"""DRIFT 复现 · 阶段 D:把 L₀ 优化好的图案反 warp 成方形纹理 PNG,供物理渲染(评测 B)。

P 在 [0,1] 的 224 图像空间(注意:与老的 patch_texture_from_train.py 不同,那个是 [-1,1]);
quad224 是足迹四角 (row,col),顺序 BL,BR,TR,TL。用单应把足迹四边形反 warp 成正方形纹理。

同时生成**同尺寸随机纹理**作物理对照 —— 0.26m 贴纸与 plate/bowl/cream_cheese 的 keep-out 盒
重叠,注入真几何会把物体顶起 1.5mm,必须有随机对照才能把攻击效果与几何干扰分开(见 HANDOFF 4.4)。
随机纹理用与 drift_eval_digital.py 相同的 SEED_RAND,保证数字/物理两条评测的随机对照同源。

用法: ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_texture.py --patch drift_patch_drift
"""
import argparse
import pathlib

import cv2
import numpy as np

PI05 = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe")
OUT = PI05 / "out"
TEX = 128
SEED_RAND = 20260902          # 必须与 drift_eval_digital.py 一致

# ⚠️ 红线(2026-09-03 由 drift_warp_check.py 实测确定):
# 单应反 warp 得到的方形图还需要**上下翻转**才与 MuJoCo 的纹理 UV 约定一致。
# 实测:identity 象限一致率 0.7%,**flipud 97.6%**,其余变换 <3%(rot90/270 约 45~54%)。
# 不加这一步 ⇒ 注入场景的纹理相对于优化出来的图案上下颠倒,
#   模型看到的不是被优化的那张图(渲染足迹与设计图案相关系数仅 +0.002)。
# ⚠️ `patch_texture_from_train.py` 用同一套反 warp 且**没有这一步** ⇒
#   HANDOFF §2.3 第 3 条"优化图案闭环不如随机"的老结论可能部分是该 bug 的假象
#   (随机图案上下翻转后统计等价,所以只有优化图案受害 —— 恰好produces "优化不如随机")。
# 改动此处后必须重跑 drift_warp_check.py 确认 identity 一致率 >85%。
FLIP_NAME = "flipud"
def FLIP(a):                                      # noqa: N802
    return a[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="drift_patch_drift")
    ap.add_argument("--tex", type=int, default=TEX)
    args = ap.parse_args()
    d = np.load(OUT / f"{args.patch}.npz", allow_pickle=True)
    P = d["P"]                                   # (3,224,224) in [0,1]
    quad = d["quad224"].astype(np.float32)       # (4,2) row,col
    img = (P.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

    src = quad[:, ::-1].copy()                   # cv2 要 (x=col, y=row)
    dst = np.array([[0, 0], [args.tex - 1, 0], [args.tex - 1, args.tex - 1],
                    [0, args.tex - 1]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    tex = cv2.warpPerspective(img, H, (args.tex, args.tex))
    tex = FLIP(tex)                              # ← 见 FLIP 定义处的红线说明
    tag = args.patch.replace("drift_patch_", "")
    tp = PI05 / "config" / f"drift_texture_{tag}.png"
    cv2.imwrite(str(tp), tex[:, :, ::-1])
    print(f"[written] {tp}  ({args.tex}×{args.tex},已施加 {FLIP_NAME})", flush=True)

    # 同尺寸随机纹理(物理干扰对照)—— 与数字评测的 rand 同 seed 同源
    R = np.random.RandomState(SEED_RAND).randint(0, 256, (224, 224, 3)).astype(np.uint8)
    rtex = FLIP(cv2.warpPerspective(R, H, (args.tex, args.tex)))
    rp = PI05 / "config" / "drift_texture_rand.png"
    cv2.imwrite(str(rp), rtex[:, :, ::-1])
    print(f"[written] {rp}  (随机对照,SEED_RAND={SEED_RAND})", flush=True)

    # 合成预览:模型在数字评测里实际看到的对抗图(便于肉眼核对反 warp 是否合理)
    pool = np.load(OUT / f"{str(d['pool'])}.npz", allow_pickle=True)
    px = pool["masks"].reshape(pool["masks"].shape[0], -1).sum(1)
    k = int(np.argmax(px))                       # 挑遮挡最少的一帧
    clean = pool["clean_img224"][k].astype(np.float32)
    m = pool["masks"][k][..., None].astype(np.float32)
    comp = (clean * (1 - m) + img.astype(np.float32) * m).clip(0, 255).astype(np.uint8)
    pv = OUT / f"drift_preview_{tag}.png"
    cv2.imwrite(str(pv), comp[:, :, ::-1])
    print(f"[written] {pv}  (数字评测里模型看到的图,帧 {k},足迹 {int(px[k])}px)", flush=True)
    print(f"[info] L₀ {float(d['L_init']):.3e} → {float(d['L_final']):.3e}  "
          f"(随机参照 {float(d['L_rand']):.3e})", flush=True)


if __name__ == "__main__":
    main()
