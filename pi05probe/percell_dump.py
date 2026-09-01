#!/usr/bin/env python3
"""Per-cell dump(py3.8,纯渲染):把 16×16 token 网格反投影得到的 176 个桌面格
各贴一张 6cm patch,回放该任务的 clean 轨迹 T 帧,渲染 clean + patched 观测。

与 s2_scan_dump 的区别:
  * 锚点来自 out/grad/pi05_cell_world.npz(g0i 反投影出的格子中心世界坐标),不是网格采样;
  * patch 尺寸压到 6cm(≈token 格中心间距 5.7cm),让每格独立、不糊到邻格;
  * keepout 只**打标**不剔除(要填满整张 16×16 网格当 influence ground truth)。

输出格式对齐 s2_scan_obs.npz ⇒ 可直接喂 s2_scan_actions.py 跑 FD 前向。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/percell_dump.py \
        --task put_the_bowl_on_the_plate --out out/percell_obs_plate.npz
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

PATCH_M = 0.06          # 6cm patch(token 格中位物理尺寸 5–7cm)


def keepout_hits(cfg, x, y, w, h):
    hits = []
    for name, box in (cfg.get("keepout") or {}).items():
        if sp._overlaps(x - w / 2, x + w / 2, box["x"][0], box["x"][1]) and \
           sp._overlaps(y - h / 2, y + h / 2, box["y"][0], box["y"][1]):
            hits.append(name)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="put_the_bowl_on_the_plate")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cells", default=str(OUT / "grad" / "pi05_cell_world.npz"))
    args = ap.parse_args()

    from libero.libero.envs import OffScreenRenderEnv
    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    cfg["patch"]["size_wh"] = [PATCH_M, PATCH_M]          # 覆盖成 6cm
    w, h = cfg["patch"]["size_wh"]
    tex = str(cfg["patch"].get("texture") or (PROBE / "config" / "probe_texture.png"))
    assert pathlib.Path(tex).is_file(), f"纹理不存在:{tex}"
    seed = int(cfg["shared_seed"])
    lift = cfg["patch"]["thickness"] / 2.0 + cfg["patch"]["normal_offset"]

    plane = sp.Plane.from_cfg(cfg["plane"])

    # 任务 prompt
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    prompt = next(str(suite.get_task(i).language) for i in range(suite.n_tasks)
                  if pathlib.Path(suite.get_task(i).bddl_file).stem == args.task)
    bddl = base.bddl_path(args.task)
    print(f"[cfg] task={args.task} prompt={prompt!r} patch={w*100:.0f}cm seed={seed}", flush=True)

    # 轨迹帧
    tr = np.load(OUT / f"traj_{args.task}.npz", allow_pickle=True)
    T = int(tr["n_frames"])
    flats = [tr[f"f{k:03d}__flatten"] for k in range(T)]
    ts, ks = tr["ts"], tr["ks"]

    # 176 个桌面格
    cw = np.load(args.cells, allow_pickle=True)
    cells = cw["cells"]           # (M,2) (row,col)
    worlds = cw["worlds"]         # (M,2) world x,y
    M = len(cells)
    legal = np.zeros(M, bool)
    keep = []
    for i in range(M):
        x, y = float(worlds[i, 0]), float(worlds[i, 1])
        hits = keepout_hits(cfg, x, y, w, h)
        inside = plane.contains_uv(x, y, w, h)
        legal[i] = inside and not hits
        keep.append(",".join(hits))
    print(f"[cells] M={M}  legal(6cm)={int(legal.sum())}  T={T}", flush=True)

    # ---- clean:所有 T 帧
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                             camera_segmentations="element")
    env.seed(seed); env.reset()
    c_img = np.zeros((T, 224, 224, 3), np.uint8)
    c_wri = np.zeros((T, 224, 224, 3), np.uint8)
    c_state = np.zeros((T, 8), np.float64)
    for k in range(T):
        obs = env.regenerate_obs_from_state(flats[k])
        pi, pw = base.model_input(obs)
        c_img[k] = pi; c_wri[k] = pw; c_state[k] = base.state8(obs)
    env.close()
    print("[clean] T 帧渲染完成", flush=True)

    # ---- patched:每格注入一次,回放 T 帧
    P_img = np.zeros((M, T, 224, 224, 3), np.uint8)
    P_wri = np.zeros((M, T, 224, 224, 3), np.uint8)
    vpx = np.zeros((M, T), np.int32)
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                              camera_segmentations="element")
    for i in range(M):
        world = plane.to_world(float(worlds[i, 0]), float(worlds[i, 1]), lift)
        penv.env.set_xml_processor(sp.make_xml_processor(cfg, world, tex))
        penv.seed(seed); penv.reset()
        gid = sp.patch_geom_id(penv, cfg)
        for k in range(T):
            obs = penv.regenerate_obs_from_state(flats[k])
            pi, pw = base.model_input(obs)
            P_img[i, k] = pi; P_wri[i, k] = pw
            vpx[i, k] = sp.visible_px(obs, "agentview", gid)
        if i % 20 == 0:
            print(f"  [{i:3d}/{M}] cell=({cells[i,0]},{cells[i,1]}) "
                  f"world=({world[0]:.2f},{world[1]:.2f}) legal={legal[i]} vpx[t0]={vpx[i,0]}",
                  flush=True)
    penv.close()

    outp = pathlib.Path(args.out) if args.out else OUT / f"percell_obs_{args.task}.npz"
    np.savez_compressed(
        outp, task=args.task, prompt=prompt, shared_seed=seed, T=T, M=M, texture=tex,
        patch_m=PATCH_M, ts=ts, ks=ks,
        clean_img224=c_img, clean_wrist224=c_wri, clean_state8=c_state,
        patched_img224=P_img, patched_wrist224=P_wri, visible_px=vpx,
        anchor_world=worlds, anchor_uv=worlds, anchor_cell=cells,
        anchor_idx=np.arange(M), anchor_legal=legal,
        anchor_keepout=np.array(keep))
    print(f"[written] {outp}  ({outp.stat().st_size/2**20:.0f} MiB)  M={M} T={T}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
