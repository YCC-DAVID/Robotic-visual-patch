#!/usr/bin/env python3
"""FastWAM 未来预测头对 patch 敏不敏感(Δfuture 探针)。

同一把 FD 探针,测量对象从 action 换成 infer_joint 的 pred["video"](预测未来帧)。
在 influence 热格 vs 远处控制格贴 patch,量:
  Δaction = ‖a_patched − a_clean‖   Δfuture = ‖video_patched − video_clean‖
各自除以「换 seed 重采」的地板 ⇒ 无量纲 SNR,可判"未来通路可不可攻"、并与 π0.5 action 头 SNR 对读。
固定 seed ⇒ 逐位确定(先验红线)。用几帧、几个位置即可,不用全网格。

用法(wamattack env,一张 ≥13GB 卡):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free> \
      /home/user1/miniconda3/envs/wamattack/bin/python probe/fw_future_probe.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

REPO = Path("/home/user1/workspace/chence/WAMattack")
FASTWAM = REPO / "third_party" / "FastWAM"
PI05 = REPO / "pi05probe"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / "probe" / "config" / "libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(REPO / "checkpoints"))
sys.path.insert(0, str(FASTWAM / "experiments" / "libero"))
sys.path.insert(0, str(PI05))

import numpy as np
import torch
import yaml

_torch_load_orig = torch.load
torch.load = lambda *a, **k: _torch_load_orig(*a, **{**k, "weights_only": False})

TASK, SUITE = "put_the_bowl_on_the_plate", "libero_goal"
NFLOOR = 4          # 换 seed 次数(未来头 decode 贵,少量即可)
NFRAMES = 3         # 探几帧
PATCH_M = 0.13


def main():
    import scene_patch as sp
    from run_instruction_sweep import build_cfg, encode_instructions, load_model
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_utils import get_libero_dummy_action, LIBERO_ENV_RESOLUTION
    from eval_libero_single import _obs_to_model_input, _denormalize_action

    cfg_scene = yaml.safe_load((PI05 / "config" / "scene.yaml").read_text())
    cfg_scene["patch"]["size_wh"] = [PATCH_M, PATCH_M]
    lift = cfg_scene["patch"]["thickness"] / 2 + cfg_scene["patch"]["normal_offset"]
    plane = sp.Plane.from_cfg(cfg_scene["plane"])

    # 探针位置:influence 热格(hot)+ 远处低-influence 合法格(control)
    sc = np.load(REPO / "probe" / "out" / "fw_percell_scores.npz", allow_pickle=True)
    fd, leg, aw = sc["influence"], sc["legal"].astype(bool), sc["anchor_world"]
    hot = aw[int(fd.argmax())]
    legal_idx = np.where(leg)[0]
    far = aw[legal_idx[int(np.argmin(fd[legal_idx]))]]     # 合法格里 influence 最低的
    probes = [("hot", hot), ("far_legal", far)]
    print(f"[probe] hot=({hot[0]:.2f},{hot[1]:.2f})  far_legal=({far[0]:.2f},{far[1]:.2f})", flush=True)

    class A:
        suite, task_id, init_index = SUITE, None, 0
    suite_obj = benchmark.get_benchmark_dict()[SUITE]()
    tid = next(i for i in range(suite_obj.n_tasks)
               if Path(suite_obj.get_task(i).bddl_file).stem == TASK)
    task = suite_obj.get_task(tid)
    sa = A(); sa.task_id = tid
    cfg = build_cfg(sa)
    base_seed = int(cfg.seed)
    prompt = DEFAULT_PROMPT.format(task=str(task.language))
    context, cmask = encode_instructions(cfg, [prompt], "cpu")
    model, processor, device, dtype = load_model(cfg)
    Hd, Wd = [int(v) for v in cfg.data.train.video_size]
    AH = int(cfg.data.train.num_frames) - 1
    NVF = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1
    ns = int(cfg.EVALUATION.num_inference_steps)
    print(f"[cfg] AH={AH} num_video_frames={NVF} steps={ns} base_seed={base_seed}", flush=True)

    def infer_joint(obs, seed):
        image, proprio, _ = _obs_to_model_input(obs, cfg=cfg, processor=processor,
                                                width=Wd, height=Hd, device=device, dtype=dtype)
        with torch.no_grad():
            pred = model.infer_joint(
                prompt=None, input_image=image, num_video_frames=NVF, action_horizon=AH,
                proprio=proprio, context=context.to(device=device, dtype=dtype),
                context_mask=cmask.to(device=device), num_inference_steps=ns,
                sigma_shift=cfg.EVALUATION.get("sigma_shift"), seed=seed,
                rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")), tiled=False,
                test_action_with_infer_action=False)
        act = _denormalize_action(pred["action"], processor)[0].astype(np.float64)  # [32,7]
        v = pred["video"]                                        # list[PIL.Image] 或 tensor
        if isinstance(v, list):
            vid = np.stack([np.asarray(im, dtype=np.float32) for im in v]).ravel()
        else:
            vid = np.asarray(v.detach().to("cpu", torch.float32)).ravel()
        return act, vid

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                             camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
    env.seed(base_seed); env.reset()
    inits = np.array(torch.load(str(Path(get_libero_path("init_states")) / SUITE / f"{TASK}.pruned_init")))
    obs = env.set_init_state(inits[0])
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action())

    # 红线:固定 seed 连推两次逐位相同(action + video)
    a0, v0 = infer_joint(obs, base_seed)
    a1, v1 = infer_joint(obs, base_seed)
    print(f"[红线] |Δaction|max={np.abs(a1-a0).max():.2e}  |Δvideo|max={np.abs(v1-v0).max():.2e}", flush=True)

    # 取几帧 clean 状态
    states, plan = [], []
    from collections import deque
    plan = deque()
    while len(states) < NFRAMES:
        if not plan:
            a, _ = infer_joint(obs, base_seed)
            states.append(env.get_sim_state().copy())
            plan.extend(a[:int(cfg.EVALUATION.replan_steps)].tolist())
        act = np.array(plan.popleft()); act[-1] = np.sign(-(act[-1]*2-1))
        obs, _, _, _ = env.step(act.tolist())
    env.close()

    tex = str(PI05 / "config" / "probe_texture.png")
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                              camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")

    # clean(每帧,固定 seed)+ 地板(每帧换 seed)
    Aclean, Vclean, Afloor, Vfloor = [], [], [], []
    cenv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                              camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
    cenv.seed(base_seed); cenv.reset()
    for k in range(NFRAMES):
        obs = cenv.regenerate_obs_from_state(states[k])
        a, v = infer_joint(obs, base_seed); Aclean.append(a); Vclean.append(v)
        af, vf = [], []
        for j in range(NFLOOR):
            aa, vv = infer_joint(obs, base_seed + 1000 + j); af.append(aa); vf.append(vv)
        Afloor.append(af); Vfloor.append(vf)
    cenv.close()
    Aclean, Vclean = np.array(Aclean), np.array(Vclean)

    # 地板 = 换 seed 两两差的 p95(与 patch 同口径:action 取平移前5步合计范数;video 取整体 L2)
    def afloor_stat(k):
        s = []
        for i in range(NFLOOR):
            for j in range(i+1, NFLOOR):
                d = (np.array(Afloor[k][i])[:5,0:3]-np.array(Afloor[k][j])[:5,0:3]).sum(0)
                s.append(np.linalg.norm(d)*50)
        return np.percentile(s, 95) if s else 0.0
    def vfloor_stat(k):
        s = [np.linalg.norm(np.array(Vfloor[k][i])-np.array(Vfloor[k][j]))
             for i in range(NFLOOR) for j in range(i+1, NFLOOR)]
        return np.percentile(s, 95) if s else 0.0

    print("\n=== FastWAM 未来头 / action 头 对 patch 的敏感度(除以各自 seed 地板)===", flush=True)
    for lab, world in probes:
        pw = plane.to_world(float(world[0]), float(world[1]), lift)
        penv.env.set_xml_processor(sp.make_xml_processor(cfg_scene, pw, tex))
        penv.seed(base_seed); penv.reset()
        da_snr, dv_snr = [], []
        for k in range(NFRAMES):
            obs = penv.regenerate_obs_from_state(states[k])
            ap, vp = infer_joint(obs, base_seed)
            da = np.linalg.norm((ap[:5,0:3]-Aclean[k][:5,0:3]).sum(0))*50
            dv = np.linalg.norm(vp - Vclean[k])
            fa, fv = afloor_stat(k), vfloor_stat(k)
            da_snr.append(da/(fa+1e-9)); dv_snr.append(dv/(fv+1e-9))
        print(f"  [{lab:9s}] Δaction/floor 均值 {np.mean(da_snr):.2f}  "
              f"Δfuture/floor 均值 {np.mean(dv_snr):.2f}  (逐帧 action {np.round(da_snr,2)} / future {np.round(dv_snr,2)})",
              flush=True)
    penv.close()
    print("\n对读:π0.5 action 头合法格 SNR max/floor=1.11、median/floor=0.13(plate)。", flush=True)


if __name__ == "__main__":
    main()
