#!/usr/bin/env python3
"""S2.1 扫描 dump(py3.8,纯渲染):网格 M 锚点 × 轨迹 T 帧的 clean + patched 观测。

反事实查询:被扰动的动作**不执行**,轨迹全程保持 clean(计划 S2.1)。
所以对每个锚点,把 patch 静态注入,回放同一条 clean 轨迹的 T 帧状态,渲染观测。
keepout 命中的格子照样跑,只**打标**(Δa 混入遮挡效应),不剔除。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/s2_scan_dump.py
    → out/s2_scan_obs.npz
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"
sys.path.insert(0, str(PROBE))

import numpy as np
import yaml
import s2_dump as base
import scene_patch as sp


def main():
    # --texture / --out:纹理轴用(同一批锚点、只换纹理,受控对照)
    ap = argparse.ArgumentParser()
    ap.add_argument("--texture", default=None, help="探针纹理 PNG;默认 config/probe_texture.png")
    ap.add_argument("--out", default=None, help="默认 out/s2_scan_obs.npz")
    ap.add_argument("--anchors", default=None,
                    help="用 make_fine_anchors.py 出的 json 覆盖网格锚点(加密扫描用);"
                         "合法性仍由 scene_patch 的同一套判据在那边算好")
    ap.add_argument("--task", default=None,
                    help="任务 stem(跨任务用);默认 bowl→plate。轨迹读 out/traj_{task}.npz,"
                         "prompt 取该任务的 language。LIBERO-Goal 十任务同一张桌 ⇒ 锚点判据不变")
    args = ap.parse_args()

    from libero.libero.envs import OffScreenRenderEnv
    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    tex = str(args.texture or cfg["patch"].get("texture")
              or (PROBE / "config" / "probe_texture.png"))
    assert pathlib.Path(tex).is_file(), f"纹理不存在:{tex}"
    print(f"[cfg] texture={tex}", flush=True)
    seed = int(cfg["shared_seed"])
    task_stem = args.task or base.TASK
    bddl = base.bddl_path(task_stem)
    if args.task:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()["libero_goal"]()
        prompt = next(str(suite.get_task(i).language) for i in range(suite.n_tasks)
                      if pathlib.Path(suite.get_task(i).bddl_file).stem == task_stem)
        print(f"[cfg] task={task_stem}  prompt={prompt!r}", flush=True)
    else:
        prompt = "put the bowl on the plate"

    tr = np.load(OUT / f"traj_{task_stem}.npz", allow_pickle=True)
    T = int(tr["n_frames"])
    flats = [tr[f"f{k:03d}__flatten"] for k in range(T)]
    ts = tr["ts"]

    if args.anchors:
        import json
        spec = json.loads(pathlib.Path(args.anchors).read_text())
        plane = sp.Plane.from_cfg(cfg["plane"])
        w, h = cfg["patch"]["size_wh"]
        lift = cfg["patch"]["thickness"] / 2.0 + cfg["patch"]["normal_offset"]
        assert spec["patch_wh"] == [w, h], "锚点文件的贴纸尺寸与 scene.yaml 不一致,合法性判据会对不上"
        anchors = [sp.Anchor(index=int(r["index"]), plane=plane.name, u=float(r["u"]), v=float(r["v"]),
                             world=plane.to_world(float(r["u"]), float(r["v"]), lift),
                             inside_plane=plane.contains_uv(float(r["u"]), float(r["v"]), w, h),
                             keepout_hits=()) for r in spec["anchors"]]
        assert all(a.legal for a in anchors), "锚点文件里混进了非法点"
        M = len(anchors)
        print(f"[cfg] 加密锚点 {M} 个(间距 {spec['step']*100:.0f} cm,距物体 ≤{spec['dmax']*100:.0f} cm,"
              f"全部合法)  T={T} 帧  seed={seed}", flush=True)
    else:
        anchors = sp.make_anchors(cfg)
        M = len(anchors)
        print(f"[cfg] 网格 {cfg['grid']['n_u']}×{cfg['grid']['n_v']}={M}  "
              f"合法={sum(a.legal for a in anchors)}  T={T} 帧  seed={seed}", flush=True)

    # ---- clean:所有 T 帧
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                             camera_segmentations="element")
    env.seed(seed); env.reset()
    c_img, c_wri, c_state = [], [], []
    for k in range(T):
        obs = env.regenerate_obs_from_state(flats[k])
        pi, pw = base.model_input(obs)
        c_img.append(pi); c_wri.append(pw); c_state.append(base.state8(obs))
    env.close()
    print("[clean] T 帧渲染完成", flush=True)

    # ---- patched:每个锚点注入一次,回放 T 帧
    P_img = np.zeros((M, T, 224, 224, 3), np.uint8)
    P_wri = np.zeros((M, T, 224, 224, 3), np.uint8)
    vpx = np.zeros((M, T), np.int32)
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                              camera_segmentations="element")
    for i, a in enumerate(anchors):
        penv.env.set_xml_processor(sp.make_xml_processor(cfg, a.world, tex))
        penv.seed(seed); penv.reset()
        gid = sp.patch_geom_id(penv, cfg)
        for k in range(T):
            obs = penv.regenerate_obs_from_state(flats[k])
            pi, pw = base.model_input(obs)
            P_img[i, k] = pi; P_wri[i, k] = pw
            vpx[i, k] = sp.visible_px(obs, "agentview", gid)
        if i % 6 == 0:
            print(f"  [{i:2d}/{M}] anchor#{a.index} world=({a.world[0]:.2f},{a.world[1]:.2f}) "
                  f"legal={a.legal} vpx[t0]={vpx[i,0]}", flush=True)
    penv.close()

    outp = pathlib.Path(args.out) if args.out else OUT / "s2_scan_obs.npz"
    np.savez_compressed(
        outp, task=task_stem, prompt=prompt, shared_seed=seed, T=T, M=M, texture=tex,
        ts=ts, ks=tr["ks"],
        clean_img224=np.array(c_img), clean_wrist224=np.array(c_wri), clean_state8=np.array(c_state),
        patched_img224=P_img, patched_wrist224=P_wri, visible_px=vpx,
        anchor_world=np.array([a.world for a in anchors]),
        anchor_uv=np.array([[a.u, a.v] for a in anchors]),
        anchor_idx=np.array([a.index for a in anchors]),
        anchor_legal=np.array([a.legal for a in anchors]),
        anchor_keepout=np.array([",".join(a.keepout_hits) for a in anchors]))
    print(f"[written] {outp}  ({outp.stat().st_size/2**20:.0f} MiB)  M={M} T={T}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
