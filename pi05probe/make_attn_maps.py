#!/usr/bin/env python3
"""PART B 出图:把 attention 做成可浏览的热力图 + 可分析的干净数值。

输入 `out/attn_b1b2.npz`(raw `head_sum[L, Z, V, 16, 16]`,层/head/token 都没合并)。
输出 `out/attn_maps/`:

    README.txt                  轴与命名约定、归一化口径、朝向说明
    manifest.csv                每张图的完整元信息(含未归一化的 min/max,信息不丢)
    orientation_check.png       朝向对照:raw / [::-1] / [::-1,::-1]
    saliency.npz                **干净数值**,给 SSIM / 秩相关用(不要在叠加图上做 SSIM!)
    group_<组>/                 一条指令 + 它的改写在同一个文件夹
        t000_L00_allmax_B1_bowl_plate_base.png
        t000_Lrollout_noun_B1_bowl_plate_base.png
        ...

命名:t{时间步}_L{层}_{图类型}_{variant}_{view}.png
     图类型 = allmax | allsum | noun | verb | func    (§B 的零成本后处理项)
     L      = 00..17 | rollout

⚠️ 三件必须写清楚的事
--------------------
1. **SSIM 不要在 overlay 上做** —— 叠加图混了底图与 colormap。用 saliency.npz。
2. **时间步**:目前只有一帧(共享帧 = warmup 10 步之后),所以只有 t000。
   多时间步需要沿 rollout 采帧(模型在 py3.11、渲染在 py3.8,要跨进程),另做。
3. **朝向**:模型输入是 `obs[...][::-1, ::-1]`(`main.py:115`)。
   LIBERO 原始 buffer 是 bottom-up 的 R=V(A),所以喂给模型的是 V(H(R))=**H(A)**,
   即**左右镜像的正立图**。热图叠在模型输入上是自洽的(两者同一坐标系);
   但若要跟世界坐标或 LIBERO 自然朝向的图对照,必须把这个镜像解掉。
   看 orientation_check.png 确认。

归一化口径(§A3 第 2 步,逐行):分母 = **两路真实图像的 512 个 token**
(base 256 + left_wrist 256)。right_wrist 是补零且 `image_mask=False`,不进分母。
理由:POAP 说的是"在 image token 上重新归一化";只用 base 的 256 会丢掉
"这个 token 到底更关注 base 还是 wrist"的信息。口径写进文件名前缀与 README。

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/make_attn_maps.py
    ... --per-token          # 额外出逐 token 图(文件数会多很多)
    ... --renorm base        # 换成只在 base 的 256 内归一化
"""

import argparse
import csv
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OUT = ROOT / "pi05probe" / "out"
MAPS = OUT / "attn_maps"
N_SIDE, N_IMG, N_LAYER, N_HEAD = 16, 256, 18, 8
WIN = 3             # §A2 第 5 步的 s×s 窗口
DISP = 448          # 输出边长(224 的 2 倍,好让 caption 看得清)
CAPTION_H = 60      # 底部字幕:完整指令 + 层/图类型/口径 + 原始 min/max + token 列表

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from instructions import GROUPS, META, disp_tok, roles  # noqa: E402  单一来源


def hot_lut():
    """经典 'hot':亮度单调,适合叠加。"""
    t = np.linspace(0, 1, 256)
    return np.stack([np.clip(3 * t, 0, 1), np.clip(3 * t - 1, 0, 1), np.clip(3 * t - 2, 0, 1)],
                    axis=1)


LUT = hot_lut()


def up_nn(S, size):
    """最近邻上采样 —— 保留 patch 边界,不造出不存在的平滑。主用这个。"""
    r = size // N_SIDE
    return np.kron(S, np.ones((r, r), dtype=S.dtype))


def best_window(S, s=WIN):
    """§A2 第 5 步:在 saliency 上滑 s×s 窗口,取**窗口聚合分数最大**的位置(不是单点 argmax)。

    返回 (中心 (i,j), 窗口分数)。注意中心只能取 s//2 .. N_SIDE-1-s//2,
    即 s=3 时是 1..14 ⇒ **中心落在 1 或 14 就说明窗口贴着图像边界**。
    """
    best, pos = -np.inf, None
    for i in range(N_SIDE - s + 1):
        for j in range(N_SIDE - s + 1):
            v = float(S[i:i + s, j:j + s].sum())
            if v > best:
                best, pos = v, (i + s // 2, j + s // 2)
    return pos, best


def draw_window(im, center, size, s=WIN, color=(0, 255, 255)):
    """把**实际选出的 s×s 窗口**画在图上。

    为什么必须画:多个层的窗口 argmax 落在跨指令不变的边缘/角落伪影上
    (典型的 register / spatial sink patch),不落在任何物体上。
    不画出来,光看热图会误以为定位落在物体上。
    青色是刻意选的 —— hot 色标里没有青,不会撞色。
    """
    r = size // N_SIDE
    ci, cj = center
    y0, x0 = (ci - s // 2) * r, (cj - s // 2) * r
    d = ImageDraw.Draw(im)
    d.rectangle([x0, y0, x0 + s * r - 1, y0 + s * r - 1], outline=color, width=2)
    return im


def overlay(img224, S16):
    """热图叠在模型真实输入之上。

    alpha 随 saliency 强度变化(0.12→0.92),低显著区几乎保留原图 ⇒ 能看清物体在哪。
    (固定 alpha + 45% 灰底的旧版整张过暗,认不出物体位置。)
    """
    base = np.asarray(Image.fromarray(img224).resize((DISP, DISP), Image.NEAREST), dtype=np.float64)
    lo, hi = float(S16.min()), float(S16.max())
    Sn = (S16 - lo) / (hi - lo) if hi > lo else np.zeros_like(S16)
    Su = up_nn(Sn, DISP)
    heat = LUT[(Su * 255).astype(np.uint8)] * 255.0
    # 底图压成灰度,免得原图彩色和 colormap 抢眼
    gray = base @ np.array([0.299, 0.587, 0.114])
    base_g = np.repeat(gray[:, :, None], 3, axis=2)
    a = (0.12 + 0.80 * Su)[:, :, None]
    return np.clip((1 - a) * base_g + a * heat, 0, 255).astype(np.uint8), lo, hi


def with_caption(rgb, lines):
    """⚠️ 一律用 ASCII:PIL 默认位图字体没有 CJK 字形,中文会渲染成方块 □。"""
    im = Image.new("RGB", (DISP, DISP + CAPTION_H), (16, 16, 16))
    im.paste(Image.fromarray(rgb), (0, 0))
    d = ImageDraw.Draw(im)
    for i, s in enumerate(lines[:4]):
        d.text((5, DISP + 3 + i * 13), s[:104], fill=(235, 235, 235))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renorm", default="img512", choices=["img512", "base", "none"])
    ap.add_argument("--per-token", action="store_true")
    ap.add_argument("--views", default="base", choices=["base", "both"])
    args = ap.parse_args()

    z = np.load(OUT / "attn_b1b2.npz", allow_pickle=False)
    fr = np.load(OUT / "shared_frame.npz", allow_pickle=False)
    views_all = [str(v) for v in z["views"]]
    views = ["base_0_rgb"] if args.views == "base" else views_all
    names = [str(x) for x in z["variant_names"]]

    tasks = [str(t) for t in fr["tasks"]]
    img224 = {"base_0_rgb": fr[f"{tasks[0]}__img224"],
              "left_wrist_0_rgb": fr[f"{tasks[0]}__wrist224"]}
    rgb256 = fr[f"{tasks[0]}__rgb256"]

    MAPS.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 朝向对照
    panels = [("raw obs (bottom-up)", rgb256),
              ("[::-1]  = LIBERO 自己下游用的", rgb256[::-1]),
              ("[::-1,::-1] = openpi 喂模型的", rgb256[::-1, ::-1])]
    W = 256
    oc = Image.new("RGB", (W * 3, W + CAPTION_H), (16, 16, 16))
    d = ImageDraw.Draw(oc)
    for i, (lbl, a) in enumerate(panels):
        oc.paste(Image.fromarray(np.ascontiguousarray(a)), (i * W, 0))
        d.text((i * W + 4, W + 6), lbl[:40], fill=(235, 235, 235))
    d.text((4, W + 26), "看第三张:模型真实看到的朝向。热图都叠在这个坐标系上。",
           fill=(160, 200, 255))
    oc.save(MAPS / "orientation_check.png")
    print(f"[written] {MAPS/'orientation_check.png'}", flush=True)

    # ---------------------------------------------------------------- 归一化
    VIEW_LO = {"base_0_rgb": 0, "left_wrist_0_rgb": N_IMG}

    def token_rows(name, layer, view):
        """→ (Z, 256) 该层该视图下、按 args.renorm 口径归一化后的逐 token 图。"""
        hs = z[f"{name}__head_sum"]              # (L, Z, V, 16, 16)
        Z = hs.shape[1]
        if layer == "rollout":
            blk = z[f"{name}__rollout"]          # (Z, V, 16, 16)
            per_view = {v: blk[:, views_all.index(v)].reshape(Z, -1).astype(np.float64)
                        for v in views_all}
        else:
            per_view = {v: hs[layer, :, views_all.index(v)].reshape(Z, -1).astype(np.float64)
                        for v in views_all}
        if args.renorm == "none":
            return per_view[view]
        if args.renorm == "base":
            den = per_view["base_0_rgb"].sum(axis=1, keepdims=True)
        else:  # img512 = base + left_wrist(right_wrist 补零且 mask=False,不进分母)
            den = sum(per_view[v].sum(axis=1, keepdims=True) for v in views_all)
        return per_view[view] / np.clip(den, 1e-12, None)

    def kinds(name, A, pieces):
        """A:(Z,256) → 各类 saliency 图。token 角色由 instructions.roles() 按【词】匹配,
        匹配不上会直接报错(不静默漏掉),见 instructions.match_words 的注释。"""
        r = roles(name, pieces)
        ni, vi, fi = r["noun"], r["verb"], r["func"]
        out = {
            "allmax": A.max(axis=0),                       # POAP 的定位量
            "allsum": A.sum(axis=0),                       # §B1 的稳健变体
            "noun": A[ni].sum(axis=0) if ni else None,     # 跨指令比较用这个(§C)
            "verb": A[vi].sum(axis=0) if vi else None,
            "func": A[fi].sum(axis=0) if fi else None,      # 功能词(含 BOS/\n,sink 在这儿)
        }
        return {k: v.reshape(N_SIDE, N_SIDE) for k, v in out.items() if v is not None}, ni, vi

    # ---------------------------------------------------------------- 出图
    rows_csv = []
    sal = {}
    layers = list(range(N_LAYER)) + ["rollout"]
    n = 0
    for group, members in GROUPS.items():
        gdir = MAPS / group
        gdir.mkdir(exist_ok=True)
        for name in members:
            pieces = [str(p) for p in z[f"{name}__pieces"]]
            prompt = str(z[f"{name}__prompt"])
            for view in views:
                vtag = "base" if view == "base_0_rgb" else "wrist"
                for layer in layers:
                    A = token_rows(name, layer, view)
                    mp, ni, vi = kinds(name, A, pieces)
                    ltag = "rollout" if layer == "rollout" else f"{layer:02d}"
                    for kind, S in mp.items():
                        rgb, lo, hi = overlay(img224[view], S)
                        win, wscore = best_window(S)
                        im = draw_window(Image.fromarray(rgb), win, DISP)
                        edge = win[0] in (1, N_SIDE - 2) or win[1] in (1, N_SIDE - 2)
                        fn = f"t000_L{ltag}_{kind}_{name}_{vtag}.png"
                        used = {"noun": [pieces[i] for i in ni],
                                "verb": [pieces[i] for i in vi]}.get(kind)
                        with_caption(np.asarray(im), [
                            f'INSTRUCTION: "{prompt}"',
                            f"layer={ltag}  map={kind}  view={vtag}  t=000  renorm={args.renorm}"
                            + ("   [BIASED across instructions: Z differs]"
                               if kind in ("allmax", "allsum") else ""),
                            (f"tokens used ({kind}): {' '.join(map(disp_tok, used))}" if used
                             else f"all {len(pieces)} tokens: "
                                  f"{' '.join(map(disp_tok, pieces))}"),
                            f"3x3 window (cyan box) center=(row {win[0]}, col {win[1]})"
                            + ("  <-- TOUCHES IMAGE BORDER" if edge else "")
                            + f"   raw min={lo:.3e} max={hi:.3e}",
                        ]).save(gdir / fn)
                        rows_csv.append(dict(
                            group=group, file=f"{group}/{fn}", variant=name, prompt=prompt,
                            t=0, layer=ltag, map=kind, view=vtag, renorm=args.renorm,
                            Z=len(pieces), raw_min=lo, raw_max=hi,
                            win_row=win[0], win_col=win[1], win_score=wscore,
                            win_on_border=int(edge),
                            noun_tokens=";".join(pieces[i] for i in ni),
                            verb_tokens=";".join(pieces[i] for i in vi),
                        ))
                        sal[f"{name}|{vtag}|{ltag}|{kind}"] = S.astype(np.float32)
                        n += 1
                    if args.per_token:
                        for ti, piece in enumerate(pieces):
                            S = A[ti].reshape(N_SIDE, N_SIDE)
                            rgb, lo, hi = overlay(img224[view], S)
                            safe = piece.replace("▁", "_").replace("\n", "NL").replace("<", "").replace(">", "")
                            fn = f"t000_L{ltag}_tok{ti:02d}{safe}_{name}_{vtag}.png"
                            with_caption(rgb, [
                                f"{prompt}", f"token[{ti}]={piece!r}  layer={ltag}  view={vtag}",
                                f"raw min={lo:.3e} max={hi:.3e}",
                            ]).save(gdir / fn)
                            sal[f"{name}|{vtag}|{ltag}|tok{ti:02d}"] = S.astype(np.float32)
                            n += 1
            print(f"  [{group}] {name} 完成", flush=True)

    with open(MAPS / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    np.savez_compressed(MAPS / "saliency.npz", **sal)

    (MAPS / "README.txt").write_text(f"""PART B attention 热力图
{'=' * 78}

命名: t{{时间步}}_L{{层}}_{{图类型}}_{{variant}}_{{view}}.png
  层     = 00..17 逐层,或 rollout(全层累乘,第二种提取方式)
  图类型 = allmax  全部 token 逐元素 max —— POAP 的定位量
           allsum  全部 token 求和 —— §B1 要求的稳健变体
           noun    名词子词 token 求和 —— **跨指令比较必须用这个**
           verb    主动词 token
           func    功能词(含 <bos> 与 \\n;attention sink 在这里)
  view   = base(agentview,主视角) | wrist(robot0_eye_in_hand)

⚠️ 跨指令比较不能用 allmax/allsum
  各指令真实 token 数 Z 不同(实测 6/8/9/10),token 越多 max 的期望越高
  ⇒ allmax 图**跨指令有偏**。B1/B2 一律用 noun 或逐 token 图。

⚠️ SSIM 请用 saliency.npz,不要用 PNG
  PNG 是叠加图(灰度底图 + hot colormap,alpha 0.55,**每图各自 min-max 归一化**),
  在它上面做 SSIM 会同时吃进底图和 colormap。
  saliency.npz 的 key 是 "{{variant}}|{{view}}|{{layer}}|{{图类型}}",值是 (16,16) float32,
  **未做 min-max 显示归一化**,只做了下面那步逐行归一化。
  每张图的原始 min/max 也都记在 manifest.csv 里,信息没丢。

归一化口径(§A3 第 2 步,逐行,在 token 归约之前)
  本次用 renorm={args.renorm}
    img512 = 每个 text token 在【base 256 + left_wrist 256】上归一化
             (right_wrist 补零且 image_mask=False,不进分母)
    base   = 只在 base 的 256 内归一化
    none   = 不归一化(用于验证 sink 的影响)
  顺序承重:head 求和 → 逐行归一化 → token 归约。若先 max 再归一化,
  sink token 的绝对量级会压过其他 token。

⚠️ 朝向(见 orientation_check.png)
  模型输入是 obs[...][::-1, ::-1](examples/libero/main.py:115)。
  LIBERO 原始 buffer 是 bottom-up 的 R=V(A) ⇒ 喂给模型的是 V(H(R))=**H(A)**,
  即左右镜像的正立图。热图叠在模型输入上是自洽的(同一坐标系),
  但要跟世界坐标 / LIBERO 自然朝向对照时,必须把这个镜像解掉。

⚠️ 时间步
  目前只有 t000 = 共享帧(set_init_state + 10 步 dummy warmup 之后)。
  四条 B1 指令面对的这一帧已验证**逐位相同**(S0.5 检查 B),唯一变量是文本。
  多时间步要沿 rollout 采帧(渲染在 py3.8、模型在 py3.11,需跨进程),另做;
  那一步同时给出 B3 的噪声地板 —— **没有地板,现在所有相关系数都缺一个"多少算大"的参照**。

上采样: 最近邻(np.kron),保留 patch 边界。16×16 → 448×448,每个 patch 28×28 像素。
        没做双线性 —— 那会造出不存在的平滑,看起来像定位更准。

层数据来源: out/attn_b1b2.npz 的 head_sum[layer, token, view, 16, 16](head 已按 §A2 求和)
            per_head[...] 也存着,可查 head 间方差。
""")
    print(f"\n共 {n} 张图")
    print(f"[written] {MAPS/'manifest.csv'}")
    print(f"[written] {MAPS/'saliency.npz'}  ({(MAPS/'saliency.npz').stat().st_size/2**20:.1f} MiB)")
    print(f"[written] {MAPS/'README.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
