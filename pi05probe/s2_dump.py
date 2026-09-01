#!/usr/bin/env python3
"""S2 观测 dump(py3.8,纯渲染,无模型)。

在共享帧上,渲染 clean 观测 + 每个合法锚点的 patched 观测,dump 成 npz。
py3.11 的 s2_signal.py / s2_scan.py 再 in-process 前向(共享 ε)算 Δa。

为什么要 dump 再前向(FINDINGS §PT-6):
  Δa 必须共享同一 ε,而 websocket 传不了 noise;渲染在 py3.8、模型在 py3.11。
  纯反事实查询(clean 侧不闭环)⇒ 把观测 dump 出来,在 py3.11 批量前向即可。

共享帧 = B1/B2 attention 用的那一帧(shared_frame.npz),所以 influence 与 attention
逐位对齐,后面可直接比。patch 用 Q7 的合法锚点网格(config/scene.yaml)。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/s2_dump.py
"""
import math
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"

sys.path.insert(0, str(PROBE))
for p in reversed([OPENPI / "packages" / "openpi-client" / "src", OPENPI / "third_party" / "libero"]):
    sys.path.insert(0, str(p))
os.environ["LIBERO_CONFIG_PATH"] = str(PROBE / "libero_config")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYTHONNOUSERSITE"] = "1"

import numpy as np
import yaml
import scene_patch as sp

CAM, RES, RESIZE = "agentview", 256, 224
TASK = "put_the_bowl_on_the_plate"


def quat2axisangle(quat):
    q = np.array(quat, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def model_input(obs):
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wri = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return (image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE)),
            image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, RESIZE, RESIZE)))


def state8(obs):
    return np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                           obs["robot0_gripper_qpos"]])


def bddl_path(stem):
    from libero.libero import benchmark, get_libero_path
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        if pathlib.Path(t.bddl_file).stem == stem:
            return pathlib.Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
    raise AssertionError(stem)


def main():
    from libero.libero.envs import OffScreenRenderEnv
    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    tex = str(cfg["patch"].get("texture") or (PROBE / "config" / "probe_texture.png"))
    seed = int(cfg["shared_seed"])
    bddl = bddl_path(TASK)

    sf = np.load(OUT / "shared_frame.npz", allow_pickle=False)
    shared_flat = sf[f"{TASK}__flatten"]
    prompt = "put the bowl on the plate"

    anchors = sp.make_anchors(cfg)
    legal = [a for a in anchors if a.legal]
    print(f"[cfg] 合法锚点 {len(legal)}/{len(anchors)}  seed={seed}  prompt={prompt!r}", flush=True)

    # ---- clean(无 patch)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES,
                             camera_segmentations="element")
    env.seed(seed); env.reset()
    obs = env.regenerate_obs_from_state(shared_flat)
    c_img, c_wri = model_input(obs)
    c_state = state8(obs)
    print(f"[clean] state8={np.round(c_state,3).tolist()}", flush=True)
    env.close()

    # ---- patched(逐锚点重注入)
    P_img, P_wri, P_world, P_uv, P_idx, P_vpx = [], [], [], [], [], []
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES,
                              camera_segmentations="element")
    for j, a in enumerate(legal):
        penv.env.set_xml_processor(sp.make_xml_processor(cfg, a.world, tex))
        penv.seed(seed); penv.reset()
        obs = penv.regenerate_obs_from_state(shared_flat)
        gid = sp.patch_geom_id(penv, cfg)
        vpx = sp.visible_px(obs, CAM, gid)
        pi, pw = model_input(obs)
        P_img.append(pi); P_wri.append(pw); P_world.append(a.world)
        P_uv.append([a.u, a.v]); P_idx.append(a.index); P_vpx.append(vpx)
        if j % 5 == 0:
            print(f"  [{j:2d}/{len(legal)}] anchor#{a.index} world=({a.world[0]:.2f},{a.world[1]:.2f}) vpx={vpx}", flush=True)
    penv.close()

    outp = OUT / "s2_obs.npz"
    np.savez_compressed(
        outp, task=TASK, prompt=prompt, shared_seed=seed,
        clean_img224=c_img, clean_wrist224=c_wri, clean_state8=c_state,
        patched_img224=np.array(P_img), patched_wrist224=np.array(P_wri),
        anchor_world=np.array(P_world), anchor_uv=np.array(P_uv),
        anchor_idx=np.array(P_idx), visible_px=np.array(P_vpx))
    print(f"[written] {outp}  ({outp.stat().st_size/2**20:.1f} MiB)  patched={len(P_img)}", flush=True)
    print(f"  visible_px: min={min(P_vpx)} max={max(P_vpx)} mean={np.mean(P_vpx):.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
