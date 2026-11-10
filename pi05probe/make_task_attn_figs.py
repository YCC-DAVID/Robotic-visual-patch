#!/usr/bin/env python3
"""两张最直观的图:直接把 attention 贴回相机画面上,不算任何相关系数。

图 1 `fig_minpair_maps_<view>.png` —— 最小对(同一个碗,换目的地)
    行1/行2  `bowl` token 的图,分别来自 "...on the plate" 和 "...on top of the cabinet"
             ⇒ 两行看起来一不一样,就是"换目的地会不会动操作对象的图"
    行3/行4  目的地词自己的图(`plate` vs `cabinet`)
             ⇒ 峰值落没落在真的盘子/柜子上,就是"attention 准不准"

图 2 `fig_crosstask_maps_<view>.png` —— 四条不同任务
    每行一条指令(全名词求和图),最后一列是 L04–L12 带平均
    ⇒ 四行的峰值是不是落在同一个格子里

画面上的标注
    橙色热区 = attention(每格按本格自己的最大值归一化,只看落点不看强弱)
    空心大圈 = 该图(16 帧平均后)的峰值格;**绿圈=峰值压在目标物上,红圈=落空**
    小实心点 = 逐帧峰值(16 个),散开的程度就是跨帧稳定性
    蓝色轮廓 = 操作对象   绿色轮廓 = 目的地   (来自 segmentation,不是投影,无需翻转约定)

用法:
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/make_task_attn_figs.py
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/make_task_attn_figs.py --view wrist
"""
import argparse
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
TASK = "put_the_bowl_on_the_plate"

LAYERS = [0, 3, 8, 12, 17]
BAND = list(range(4, 13))                      # L04–L12,与 reproject.py 的带一致
NOUNS = ("bowl", "plate", "cabinet", "stove", "wine", "bottle", "rack")

# 只用抓取前的帧:碗从第 10 帧(env 步 60)起被抬走,之后第 0 帧的物体掩码就失效了。
# 实测 traj 里 bowl 位移 = [0,0,0,0,0,0,0,0,0,0.5, 12.3, 55.6, ...] mm ⇒ 前 10 帧场景静止。
FRAMES = list(range(10))

C_HEAT = "#eb6834"
C_OBJ, C_DEST = "#2a78d6", "#1f8a70"
C_HIT, C_MISS = "#158a5c", "#c0392b"
INK1, INK2, INK3, SURFACE = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"

# geom_id 列表见 out/geom_ids.txt(dump_geom_ids.py 实测,seg 值 == geom_id 无偏移)
GEOM = {"bowl": range(84, 125), "plate": range(151, 162), "cabinet": range(162, 205),
        "stove": range(205, 228), "bottle": range(127, 151), "rack": range(228, 240)}

HEAT = LinearSegmentedColormap.from_list("heat", [(1, 1, 1, 0.0), (*matplotlib.colors.to_rgb(C_HEAT), 0.92)])


def obj_mask224(seg256, names):
    """seg(256²,原始朝向) → 模型输入朝向的 224² 布尔掩码。"""
    ids = [i for n in names for i in GEOM[n]]
    m = np.isin(seg256, ids)[::-1, ::-1]        # 与 model_input 的 [::-1,::-1] 一致
    idx = (np.arange(224) * 256 / 224).astype(int)
    return m[np.ix_(idx, idx)]


def cell_frac(mask224):
    """每个 16×16 格里目标物占的像素比例。"""
    return mask224.reshape(16, 14, 16, 14).mean((1, 3))


def draw(ax, bg, sal, per_frame_pk, contours, tgt_frac):
    ax.imshow(bg, extent=[0, 224, 224, 0], zorder=0)
    ax.imshow(sal / max(sal.max(), 1e-12), cmap=HEAT, vmin=0, vmax=1,
              extent=[0, 224, 224, 0], interpolation="nearest", zorder=1)
    # ⚠️ contour 不能用 extent 对齐 imshow(会上下翻转),必须显式给像素中心坐标
    gx, gy = np.meshgrid(np.arange(224) + 0.5, np.arange(224) + 0.5)
    for m, col in contours:
        ax.contour(gx, gy, m.astype(float), levels=[0.5], colors=[col], linewidths=1.6, zorder=2)
    for r_, c_ in per_frame_pk:
        ax.plot((c_ + .5) * 14, (r_ + .5) * 14, "o", ms=2.8, color="#111111",
                alpha=.5, mew=0, zorder=3)
    r, c = np.unravel_index(int(sal.argmax()), sal.shape)
    hit = tgt_frac is not None and tgt_frac[r, c] > 0.10
    ax.plot((c + .5) * 14, (r + .5) * 14, "o", ms=14, mfc="none",
            mec=(C_HIT if hit else C_MISS) if tgt_frac is not None else "#111111",
            mew=2.6, zorder=4)
    ax.set_xlim(0, 224); ax.set_ylim(224, 0); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(INK3); s.set_linewidth(0.6)
    return r, c, hit


def peaks_per_frame(A, layers, toks, v):
    out = []
    for t in FRAMES:
        s = A[t][layers][:, toks, v].mean((0, 1))
        out.append(np.unravel_index(int(s.argmax()), s.shape))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", default="base", choices=["base", "wrist"])
    args = ap.parse_args()
    v = 0 if args.view == "base" else 1
    vlabel = "AGENTVIEW (fixed overhead camera)" if v == 0 else "WRIST camera"

    az = np.load(OUT / f"attn_traj_{TASK}.npz", allow_pickle=True)
    ob = np.load(OUT / "s2_scan_obs.npz", allow_pickle=True)
    sf = np.load(OUT / "shared_frame.npz", allow_pickle=True)
    pieces = {n: [str(x) for x in az[f"{n}__pieces"]] for n in az["variant_names"]}
    attn = {n: az[f"{n}__attn"].astype(np.float64) for n in az["variant_names"]}

    key = "clean_img224" if v == 0 else "clean_wrist224"
    rgb = ob[key][0].astype(np.float64) / 255.0
    bg = np.dstack([rgb.mean(-1)] * 3) * 0.72 + 0.24          # 压成灰底,让热区跳出来
    seg = sf[f"{TASK}__seg"][:, :, 0].astype(int)
    has_seg = (v == 0)                                        # seg 只有主视角这一帧

    def tok(name, word):
        return [i for i, p in enumerate(pieces[name]) if p.lstrip("▁") == word]

    def nouns(name):
        return [i for i, p in enumerate(pieces[name]) if p.lstrip("▁") in NOUNS]

    def sal(name, toks, layers):
        return attn[name][FRAMES][:, layers][:, :, toks, v].mean((0, 1, 2))

    cols = [(f"layer {l}", [l]) for l in LAYERS] + [("layers 4-12\n(mean)", BAND)]

    def grid(rows, title, sub, fname, foot):
        n, m = len(rows), len(cols)
        fig, axes = plt.subplots(n, m, figsize=(1.62 * m + 2.6, 1.62 * n + 1.25), facecolor=SURFACE)
        pk_last, hits = [], 0
        for i, (lab, name, toks, objs, colr) in enumerate(rows):
            tf = cell_frac(obj_mask224(seg, objs)) if has_seg else None
            ct = [(obj_mask224(seg, [o]), c) for o, c in zip(objs, colr)] if has_seg else []
            for j, (clab, layers) in enumerate(cols):
                r, c, hit = draw(axes[i, j], bg, sal(name, toks, layers),
                                 peaks_per_frame(attn[name], layers, toks, v), ct, tf)
                hits += bool(hit)
                if i == 0:
                    axes[i, j].set_title(clab, fontsize=9.5, color=INK2, pad=5)
                if j == m - 1:
                    pk_last.append((r, c))
                    axes[i, j].text(1.05, .5, f"peak\ncell\n({r},{c})", transform=axes[i, j].transAxes,
                                    fontsize=8, color=INK2, va="center")
            axes[i, 0].set_ylabel(lab, fontsize=9.5, color=INK1, rotation=0,
                                  ha="right", va="center", labelpad=10)
        fig.suptitle(title, fontsize=13, color=INK1, y=.985)
        fig.text(.5, .928, sub, ha="center", fontsize=9.5, color=INK2)
        fig.text(.5, .028, foot, ha="center", fontsize=9.5, color=INK2)
        fig.subplots_adjust(left=.235, right=.945, top=.878, bottom=.062, wspace=.06, hspace=.06)
        fig.savefig(OUT / fname, dpi=135, facecolor=SURFACE)
        plt.close(fig)
        print(f"[written] {OUT/fname}   峰值压在目标物上的面板 {hits}/{len(rows)*m}")
        return pk_last

    legend = ("green ring = the peak sits on the outlined target   |   red ring = it misses"
              if has_seg else "no segmentation for this view — rings are neutral")
    sub = (f"{vlabel}   ·   orange = attention, each panel scaled to its own max\n"
           f"big ring = peak of the {len(FRAMES)}-frame mean   ·   small dots = the {len(FRAMES)} per-frame "
           "peaks   ·   frames restricted to the approach phase, before anything in the scene moves")

    # ================= 图 1:最小对 =================
    P, C = "bowl_plate_orig", "bowl_cabinet_orig"
    r1 = [("`bowl`\nfrom \"...on the plate\"", P, tok(P, "bowl"), ["bowl"], [C_OBJ]),
          ("`bowl`\nfrom \"...on top of the cabinet\"", C, tok(C, "bowl"), ["bowl"], [C_OBJ]),
          ("`plate`\nthe destination word", P, tok(P, "plate"), ["plate"], [C_DEST]),
          ("`cabinet`\nthe destination word", C, tok(C, "cabinet"), ["cabinet"], [C_DEST])]
    grid(r1, "Same bowl, different destination — where does attention actually land?", sub,
         f"fig_minpair_maps_{args.view}.png",
         "blue outline = the bowl (the manipulated object)      "
         "green outline = the destination (plate / cabinet)\n" + legend)

    # ================= 图 2:四条任务 =================
    T4 = [("turn on the stove", "stove_orig", ["stove"]),
          ("put the wine bottle\non the rack", "bottle_rack_orig", ["bottle", "rack"]),
          ("put the bowl\non the plate", "bowl_plate_orig", ["bowl", "plate"]),
          ("put the bowl\non top of the cabinet", "bowl_cabinet_orig", ["bowl", "cabinet"])]
    r2 = [(f'"{lab}"', name, nouns(name), objs, [C_OBJ] + [C_DEST] * (len(objs) - 1))
          for lab, name, objs in T4]
    band_pk = grid(r2, "Four different tasks, one identical picture — does the peak move?", sub,
                   f"fig_crosstask_maps_{args.view}.png",
                   "attention summed over the nouns of each instruction   ·   "
                   "blue outline = manipulated object, green outline = destination\n" + legend)
    print(f"[数] 四条指令 L04-L12 带平均峰值格 = {band_pk}  ⇒ "
          f"{len(set(band_pk))} 个不同的格")

    # 逐层"峰值压在本条指令的物体上"的比例 —— 图 2 的定量摘要,不再画成第三张图
    if has_seg:
        nF = len(FRAMES)
        tf = {name: cell_frac(obj_mask224(seg, objs)) for _, name, objs in T4}
        base = np.mean([(t > 0.10).mean() for t in tf.values()])
        lines = ["", "=" * 72,
                 f"逐层:注意力峰值是否压在该指令自己的物体上   前 {nF} 帧 × 4 条指令 = {4*nF} 次",
                 f"  随机基线 = 目标物平均占据 {base*100:.1f}% 的格 ⇒ 约 {base*4*nF:.1f}/{4*nF}",
                 "=" * 72, "  层 | 逐帧命中     | 命中率"]
        for l in range(attn[T4[0][1]].shape[1]):
            h = 0
            for _, name, objs in T4:
                A, tk = attn[name], nouns(name)
                for t in FRAMES:
                    r, c = np.unravel_index(int(A[t, l][tk, v].mean(0).argmax()), (16, 16))
                    h += tf[name][r, c] > 0.10
            bar = "█" * int(round(h / (4 * nF) * 30))
            lines.append(f"  {l:3d} | {h:2d}/{4*nF} {bar:<30s} | {h/(4*nF)*100:5.1f}%")
        txt = "\n".join(lines)
        print(txt)
        (OUT / f"task_attn_hitrate_{args.view}.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
