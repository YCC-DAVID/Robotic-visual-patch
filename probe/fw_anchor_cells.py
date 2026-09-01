#!/usr/bin/env python3
"""把 78 个锚点各自映射到模型输入的 7×7 base 格(纯渲染,不上模型)。

为什么要它
--------
attention 图是 base 视角 7×7 的粗格。要回答"attention 选的格 = influence 最大的锚点所在格吗",
必须知道每个锚点的贴纸投影落在哪个 base 格。用**实际渲染的贴纸 seg 质心**,不用拟合公式
(rot180 + 左右镜像的对齐很容易错,渲染最稳)。

坐标链(与 eval 的预处理逐字一致)
------------------------------
agentview 原始 256²(bottom-up)→ 模型输入 `[::-1,::-1]`(rot180)→ resize 256→224 →
每 32 px 一个 token ⇒ base 7×7。质心 (r,c) 就落在某个 base 格。

用法(wamattack env,一张卡):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free> \
      /home/user1/miniconda3/envs/wamattack/bin/python probe/fw_anchor_cells.py
    → probe/out/fw_anchor_cells.npz
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path("/home/user1/workspace/chence/WAMattack")
PI05 = REPO / "pi05probe"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / "probe" / "config" / "libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(REPO / "third_party" / "FastWAM" / "experiments" / "libero"))
sys.path.insert(0, str(PI05))

import numpy as np
import torch
import yaml

_torch_load_orig = torch.load
torch.load = lambda *a, **k: _torch_load_orig(*a, **{**k, "weights_only": False})

TASK = "put_the_bowl_on_the_plate"
SUITE = "libero_goal"
RES = 256


def main():
    import scene_patch as sp
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    cfg = yaml.safe_load((PI05 / "config" / "scene.yaml").read_text())
    spec = json.loads((PI05 / "out" / "fine_anchors.json").read_text())
    plane = sp.Plane.from_cfg(cfg["plane"]); w, h = cfg["patch"]["size_wh"]
    lift = cfg["patch"]["thickness"] / 2.0 + cfg["patch"]["normal_offset"]
    old = [a for a in sp.make_anchors(cfg) if a.legal]
    fine = [sp.Anchor(index=int(r["index"]), plane=plane.name, u=float(r["u"]), v=float(r["v"]),
                      world=plane.to_world(float(r["u"]), float(r["v"]), lift),
                      inside_plane=plane.contains_uv(float(r["u"]), float(r["v"]), w, h),
                      keepout_hits=()) for r in spec["anchors"]]
    anchors = old + fine
    M = len(anchors)

    suite_obj = benchmark.get_benchmark_dict()[SUITE]()
    tid = next(i for i in range(suite_obj.n_tasks)
               if Path(suite_obj.get_task(i).bddl_file).stem == TASK)
    task = suite_obj.get_task(tid)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    inits = np.array(torch.load(str(Path(get_libero_path("init_states")) / SUITE / f"{TASK}.pruned_init")))
    tex = str(PI05 / "config" / "probe_texture.png")

    cells = np.full((M, 2), -1, np.int32)          # (row_cell, col_cell) in model-input base 7×7
    cen = np.full((M, 2), np.nan)                   # 质心(model-input 224 像素)
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES,
                              camera_segmentations="element")
    for i, a in enumerate(anchors):
        penv.env.set_xml_processor(sp.make_xml_processor(cfg, a.world, tex))
        penv.seed(int(cfg["shared_seed"])); penv.reset()
        gid = sp.patch_geom_id(penv, cfg)
        obs = penv.set_init_state(inits[0])
        seg = obs["agentview_segmentation_element"][..., 0]     # (256,256) bottom-up
        # 转到模型输入帧:rot180 后 resize 256→224
        segm = seg[::-1, ::-1]
        ys, xs = np.nonzero(segm == gid)
        if len(ys) == 0:
            continue
        r256, c256 = ys.mean(), xs.mean()
        r224, c224 = r256 * 224 / RES, c256 * 224 / RES
        cen[i] = (r224, c224)
        cells[i] = (min(int(r224 // 32), 6), min(int(c224 // 32), 6))
        if i % 12 == 0:
            print(f"  [{i:2d}/{M}] #{a.index} cell=({cells[i,0]},{cells[i,1]}) npx={len(ys)}", flush=True)
    penv.close()

    outp = PI05.parent / "probe" / "out" / "fw_anchor_cells.npz"
    np.savez_compressed(outp, anchor_idx=np.array([a.index for a in anchors]),
                        anchor_world=np.array([a.world for a in anchors]),
                        cell=cells, centroid=cen)
    vis = int((cells[:, 0] >= 0).sum())
    print(f"[written] {outp}  {vis}/{M} 个锚点在 agentview 里可见", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
