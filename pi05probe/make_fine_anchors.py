#!/usr/bin/env python3
"""在**合法区靠近任务物体的那一环**上生成加密锚点。纯几何,不开 LIBERO、不上 GPU。

为什么
------
原来的 6×6 网格(x 间距 9 cm / y 间距 13 cm)最近的合法点距 bowl/plate 有 **15 cm**,
而连续解算出来最近可以到 **11.3 cm**。10–20 cm 这一段正是 attention 还没塌的区间
(attention 还有 21–24% 的峰值,influence 还有 53%),但原网格一个点都没采到。

合法判据与 `scene_patch.make_anchors` **逐字一致**:贴纸方框与任一 keep-out 轴对齐包围盒
在 x 和 y 上同时重叠即非法;四角必须落在桌面边界内。这里只是换了撒点方式。

输出 out/fine_anchors.json,给 `s2_scan_dump.py --anchors` 用。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/make_fine_anchors.py --dmax 0.20
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

PROBE = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe")
sys.path.insert(0, str(PROBE))
OUT = PROBE / "out"

# 任务物体中心(世界),与 out/reproject.txt 里 mujoco body 位置一致
BOWL, PLATE = np.array([-0.098, -0.009]), np.array([0.062, -0.009])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.02, help="撒点间距(米)")
    ap.add_argument("--dmax", type=float, default=0.20, help="只保留距 bowl/plate 小于此值的合法点")
    ap.add_argument("--out", default=str(OUT / "fine_anchors.json"))
    args = ap.parse_args()

    import scene_patch as sp
    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    plane = sp.Plane.from_cfg(cfg["plane"])
    w, h = cfg["patch"]["size_wh"]
    ko = cfg["keepout"]
    ru, rv = cfg["grid"]["range_u"], cfg["grid"]["range_v"]

    us = np.arange(ru[0], ru[1] + 1e-9, args.step)
    vs = np.arange(rv[0], rv[1] + 1e-9, args.step)
    rows, n_all, n_legal = [], 0, 0
    for v in vs:
        for u in us:
            n_all += 1
            hits = [n for n, b in ko.items()
                    if sp._overlaps(u - w / 2, u + w / 2, b["x"][0], b["x"][1])       # noqa: SLF001
                    and sp._overlaps(v - h / 2, v + h / 2, b["y"][0], b["y"][1])]     # noqa: SLF001
            if hits or not plane.contains_uv(u, v, w, h):
                continue
            n_legal += 1
            d = min(np.hypot(u - BOWL[0], v - BOWL[1]), np.hypot(u - PLATE[0], v - PLATE[1]))
            if d <= args.dmax:
                rows.append(dict(u=float(u), v=float(v), d_obj=float(d)))

    rows.sort(key=lambda r: r["d_obj"])
    for i, r in enumerate(rows):
        r["index"] = 1000 + i                     # 1000+ 段,和原 0–35 号锚点不撞号
    d = np.array([r["d_obj"] for r in rows])
    print(f"[撒点] {args.step*100:.0f} cm 间距,共 {n_all} 点,合法 {n_legal},"
          f"距物体 ≤{args.dmax*100:.0f}cm 的合法点 = **{len(rows)}**")
    print(f"[距离] min={d.min()*100:.1f}cm  median={np.median(d)*100:.1f}cm  max={d.max()*100:.1f}cm")
    for lo, hi in [(0, .12), (.12, .15), (.15, .18), (.18, .21)]:
        print(f"   {lo*100:3.0f}-{hi*100:3.0f} cm : {int(((d >= lo) & (d < hi)).sum()):3d} 个")

    pathlib.Path(args.out).write_text(json.dumps(
        dict(step=args.step, dmax=args.dmax, patch_wh=[w, h], anchors=rows), indent=1))
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
