#!/usr/bin/env python3
"""枚举 libero_goal 的 geom_id → 名字 → 物体归属,给 grounding check 用。

为什么必须重新枚举:今天实测 `ngeom=240`,而之前在 LIBERO `8f1084e` 上记的基线是 **239**
⇒ 当前 commit(`f78abd6`)多了一个 geom,**旧记录的绝对 geom_id 全部作废**。

分割图 `obs["agentview_segmentation_element"]` 的值**就等于 mujoco 的 `geom_id`,无偏移**
(FastWAM 阶段已验证)。⚠️ 但 `geom_id 0` 是 `floor`,和背景在数值上分不开,
所以不能用 `seg == 0` 当背景。

不需要重新渲染 —— geom 名字是编译后模型的属性,与具体哪一帧无关。

跑在 py3.8 的 openpi-libero 环境:
    ~/miniconda3/envs/openpi-libero/bin/python ~/workspace/chence/WAMattack/pi05probe/dump_geom_ids.py
"""

import json
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
OUT = ROOT / "pi05probe" / "out"

for p in reversed([OPENPI / "packages" / "openpi-client" / "src", OPENPI / "third_party" / "libero"]):
    sys.path.insert(0, str(p))
os.environ["LIBERO_CONFIG_PATH"] = str(ROOT / "pi05probe" / "libero_config")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYTHONNOUSERSITE"] = "1"

import numpy as np  # noqa: E402

# 物体归属:按 body / geom 名字的前缀归到我们关心的语义物体上。
# 这些前缀来自实测的 body 名列表(见 out/shared_frame.txt)。
OBJECTS = {
    "bowl":         ["akita_black_bowl_1"],
    "plate":        ["plate_1"],
    "cream_cheese": ["cream_cheese_1"],
    "bottle":       ["wine_bottle_1"],
    "rack":         ["wine_rack_1"],
    "cabinet":      ["wooden_cabinet_1"],
    "stove":        ["flat_stove_1"],
    "robot":        ["robot0_", "gripper0_", "mount0_"],
    "table":        ["table"],
    "floor":        ["floor"],
}


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = None
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        if pathlib.Path(t.bddl_file).stem == "put_the_bowl_on_the_plate":
            task = t
            break
    assert task is not None
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    m = env.env.sim.model

    ngeom = m.ngeom
    print(f"ngeom = {ngeom}   (⚠️ 8f1084e 上记的是 239;这里是当前 commit 的实测值)")
    rows = []
    for gid in range(ngeom):
        gname = m.geom_id2name(gid)
        bid = int(m.geom_bodyid[gid])
        bname = m.body_id2name(bid)
        key = f"{gname or ''}|{bname or ''}"
        obj = None
        for o, prefixes in OBJECTS.items():
            if any(p in key for p in prefixes):
                obj = o
                break
        rows.append(dict(geom_id=gid, geom_name=gname, body_id=bid, body_name=bname,
                         obj=obj, group=int(m.geom_group[gid])))

    # 汇总
    print("\n物体 → geom_id 列表:")
    obj2ids = {}
    for o in list(OBJECTS) + [None]:
        ids = [r["geom_id"] for r in rows if r["obj"] == o]
        if not ids:
            continue
        obj2ids[str(o)] = ids
        shown = str(ids) if len(ids) <= 14 else f"{ids[:14]} … 共 {len(ids)} 个"
        print(f"  {str(o):14s} {len(ids):3d} geoms  {shown}")

    unassigned = [r for r in rows if r["obj"] is None]
    if unassigned:
        print(f"\n⚠️ 未归类的 {len(unassigned)} 个 geom(前 20 个):")
        for r in unassigned[:20]:
            print(f"    id={r['geom_id']:3d}  geom={r['geom_name']!r}  body={r['body_name']!r}")

    outp = OUT / "geom_ids.json"
    outp.write_text(json.dumps(dict(ngeom=int(ngeom), rows=rows, obj2ids=obj2ids),
                               indent=1, ensure_ascii=False))
    print(f"\n[written] {outp}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
