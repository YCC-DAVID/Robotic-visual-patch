#!/usr/bin/env python3
"""FastWAM 版影响力扫描:同一批 78 个合法锚点,跨模型复制 π0.5 的核心测量。

跨模型复制的干净之处
------------------
FastWAM 与 π0.5 共用**同一个 LIBERO、同一个 Libero_Tabletop_Manipulation 桌面
(世界 z=0.900)、同一个任务 put_the_bowl_on_the_plate**。所以:
  - 直接 import pi05probe/scene_patch.py,合法性判据逐字相同;
  - 直接读 pi05probe/out/fine_anchors.json + config/scene.yaml,锚点世界坐标逐字相同。
唯一变量是模型。这样"面积不决定、位置决定"这个结论能不能复制到第二个模型,是干净的。

反事实扫描(与 pi05probe/s2_scan_dump.py 同一套)
--------------------------------------------
1. 跑一条 clean rollout,在每个 replan 边界记 sim 状态 s_t 和 clean 动作 A_clean[t]。
   ⚠️ 必须是成功 episode(assert done),否则夹爪通道没测到。
2. 每个锚点 i:开一个注入了 patch 的 env,逐帧 regenerate_obs_from_state(s_t) → 前向,
   得 A_patched[i,t]。被扰动的动作**不执行**,轨迹全程 clean。
3. Δa = A_patched − A_clean。flow-matching 的 ε 用固定 seed 钉死 ⇒ 逐位确定
   (ENV.md §4.7:噪声地板实测恰好为 0),Δa 是精确量。

省显存
------
指令固定 ⇒ 文本只需编码一次。照 run_instruction_sweep.py 的做法:先建 umT5 编码完 context,
del 掉,再用 load_text_encoder=false 建策略,把 context 传进 infer_action ⇒ 峰值 ~13 GB。

动作语义(ENV.md §4.4)
---------------------
7 维 = eef_delta(6) + gripper(1,绝对);delta_action_dim_mask=[T,T,T,T,T,T,F]。
**不做 7 维 L2**;通道拆分在 fw_report.py 里(平移 L2 / 旋转测地 / 夹爪翻转)。这里只存原始 7 维。

用法(wamattack env,py3.10,需一张 ≥13 GB 空闲的卡):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free> \
      /home/user1/miniconda3/envs/wamattack/bin/python probe/fw_scan.py
    → probe/out/fw_scan.npz
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

REPO = Path("/home/user1/workspace/chence/WAMattack")
FASTWAM = REPO / "third_party" / "FastWAM"
PI05 = REPO / "pi05probe"

os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / "probe" / "config" / "libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(REPO / "checkpoints"))
sys.path.insert(0, str(FASTWAM / "experiments" / "libero"))
sys.path.insert(0, str(PI05))            # 复用 scene_patch.py

import numpy as np
import torch
import yaml

# 与 run_instruction_sweep.py 同款 torch.load 补丁(.pruned_init / .pt 都是可信本地 pickle)
_torch_load_orig = torch.load
torch.load = lambda *a, **k: _torch_load_orig(*a, **{**k, "weights_only": False})

TASK = "put_the_bowl_on_the_plate"
SUITE = "libero_goal"
CKPT = REPO / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224.pt"
STATS = REPO / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224_dataset_stats.json"
DUMMY = None                              # 由 libero_utils.get_libero_dummy_action() 提供


def find_task(suite_obj):
    for i in range(suite_obj.n_tasks):
        if Path(suite_obj.get_task(i).bddl_file).stem == TASK:
            return i, suite_obj.get_task(i)
    raise SystemExit(f"{SUITE} 里没有 {TASK}")


def main():
    global TASK, SUITE
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-index", type=int, default=0)
    ap.add_argument("--settle", type=int, default=10, help="开局 dummy 步数(π0.5 侧用 10)")
    ap.add_argument("--replan", type=int, default=5, help="取 clean 状态的步距;与 π0.5 侧对齐用 5")
    ap.add_argument("--n-frames", type=int, default=10, help="最多取几帧(π0.5 侧接近段用 10)")
    ap.add_argument("--texture", default=str(PI05 / "config" / "probe_texture.png"))
    ap.add_argument("--task", default=TASK, help="libero_goal 任务 stem(共用同一桌面/物体/锚点)")
    ap.add_argument("--suite", default=SUITE)
    ap.add_argument("--out", default=None, help="默认 out/fw_scan_<task>.npz")
    ap.add_argument("--text-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--cells", default=None,
                    help="per-cell 模式:吃 fw_cell_world.npz 的全部桌面格(keepout 只打标不剔除)")
    ap.add_argument("--patch-m", type=float, default=None,
                    help="覆盖贴纸边长(米);per-cell 用 ~0.13 匹配 7×7 格尺寸")
    args = ap.parse_args()
    TASK, SUITE = args.task, args.suite
    if args.out is None:
        stem = "" if TASK == "put_the_bowl_on_the_plate" else "_" + TASK
        args.out = str(REPO / "probe" / "out" / f"fw_scan{stem}.npz")

    # ---- 复用 pi05 的场景配置与锚点 ----
    cfg_scene = yaml.safe_load((PI05 / "config" / "scene.yaml").read_text())
    spec = json.loads((PI05 / "out" / "fine_anchors.json").read_text())
    import scene_patch as sp
    if args.patch_m:
        cfg_scene["patch"]["size_wh"] = [args.patch_m, args.patch_m]
    plane = sp.Plane.from_cfg(cfg_scene["plane"])
    w, h = cfg_scene["patch"]["size_wh"]
    lift = cfg_scene["patch"]["thickness"] / 2.0 + cfg_scene["patch"]["normal_offset"]

    if args.cells:
        # per-cell 模式:7×7 反投影得到的全部桌面格,keepout 只打标(填满 token 网格当 GT)
        cw = np.load(args.cells, allow_pickle=True)
        worlds = cw["worlds"]; cells_rc = cw["cells"]
        def _keep(x, y):
            return tuple(n for n, b in (cfg_scene.get("keepout") or {}).items()
                         if sp._overlaps(x-w/2, x+w/2, b["x"][0], b["x"][1])
                         and sp._overlaps(y-h/2, y+h/2, b["y"][0], b["y"][1]))
        anchors = [sp.Anchor(index=int(r*7+c), plane=plane.name, u=float(worlds[i, 0]),
                             v=float(worlds[i, 1]), world=plane.to_world(float(worlds[i, 0]),
                             float(worlds[i, 1]), lift),
                             inside_plane=plane.contains_uv(float(worlds[i, 0]), float(worlds[i, 1]), w, h),
                             keepout_hits=_keep(float(worlds[i, 0]), float(worlds[i, 1])))
                   for i, (r, c) in enumerate(cells_rc)]
        M = len(anchors)
        print(f"[cfg] per-cell 池 = {M} 桌面格(patch={w*100:.0f}cm,合法 {sum(a.legal for a in anchors)},"
              f" keepout 只打标)", flush=True)
    else:
        assert spec["patch_wh"] == [w, h], "锚点文件贴纸尺寸与 scene.yaml 不一致"
        # 合并候选池 = 旧 6×6 网格里的合法点 + 加密 61 —— 与 report_fine_legal.py 完全同一个 78 点池。
        old = sp.make_anchors(cfg_scene)
        old_legal = [a for a in old if a.legal]
        fine = [sp.Anchor(index=int(r["index"]), plane=plane.name, u=float(r["u"]), v=float(r["v"]),
                          world=plane.to_world(float(r["u"]), float(r["v"]), lift),
                          inside_plane=plane.contains_uv(float(r["u"]), float(r["v"]), w, h),
                          keepout_hits=()) for r in spec["anchors"]]
        anchors = old_legal + fine
        assert all(a.legal for a in anchors), "候选池里混进非法点"
        M = len(anchors)
        print(f"[cfg] 候选池 = 旧合法 {len(old_legal)} + 加密 {len(fine)} = {M} 个(全部合法)", flush=True)

    # ---- FastWAM cfg / 模型 / processor(复用官方 eval 路径) ----
    from run_instruction_sweep import build_cfg, encode_instructions, load_model

    class A:  # build_cfg 只用这三个字段
        suite, task_id, init_index = SUITE, None, args.init_index
    sa = A()
    from libero.libero import benchmark
    suite_obj = benchmark.get_benchmark_dict()[SUITE]()
    tid, task = find_task(suite_obj)
    sa.task_id = tid
    cfg = build_cfg(sa)
    seed = int(cfg.seed)
    print(f"[cfg] task_id={tid} seed={seed} steps={cfg.EVALUATION.num_inference_steps} "
          f"replan(exec)={cfg.EVALUATION.replan_steps}", flush=True)

    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    from libero_utils import get_libero_dummy_action, LIBERO_ENV_RESOLUTION
    from eval_libero_single import _obs_to_model_input, _denormalize_action

    prompt = DEFAULT_PROMPT.format(task=str(task.language))
    context, cmask = encode_instructions(cfg, [prompt], args.text_device)   # 编码一次就 del 掉 umT5
    model, processor, device, dtype = load_model(cfg)
    # ⚠️ 与 run_instruction_sweep.py:251 逐字一致:h,w = video_size,再 width=w,height=h。
    # 两路相机各 224² 横向拼成 (H=224, W=448),别把 width/height 顺序弄反(否则 assert 报尺寸不符)。
    Hd, Wd = [int(v) for v in cfg.data.train.video_size]      # h=224, w=448
    AH = int(cfg.data.train.num_frames) - 1                   # 32
    ns = int(cfg.EVALUATION.num_inference_steps)

    def infer(obs):
        image, proprio, _ = _obs_to_model_input(obs, cfg=cfg, processor=processor,
                                                width=Wd, height=Hd, device=device, dtype=dtype)
        with torch.no_grad():
            pred = model.infer_action(
                prompt=None, input_image=image, action_horizon=AH, proprio=proprio,
                context=context.to(device=device, dtype=dtype), context_mask=cmask.to(device=device),
                num_inference_steps=ns, sigma_shift=cfg.EVALUATION.get("sigma_shift"),
                seed=seed, rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")), tiled=False)
        # 去归一化,但**不做**执行阶段的 gripper 翻转/二值化 —— 我们要的是模型原始输出的差,
        # 那些是执行侧的后处理(eval_libero_single.py:423-428),会掩盖 Δ。
        return _denormalize_action(pred["action"], processor)[0].astype(np.float64)   # [32,7]

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    # ---- 红线(ENV.md §4.7 / 计划红线 1):同一 obs 连推两次逐位相同 ----
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                             camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
    env.seed(seed); env.reset()
    inits = np.array(torch.load(str(Path(get_libero_path("init_states")) / SUITE / f"{TASK}.pruned_init")))
    obs = env.set_init_state(inits[args.init_index])
    for _ in range(args.settle):
        obs, _, _, _ = env.step(get_libero_dummy_action())
    a_rep0, a_rep1 = infer(obs), infer(obs)
    d_rep = float(np.abs(a_rep1 - a_rep0).max())
    print(f"[红线] 同一 obs 连推两次最大差 = {d_rep:.3e}", flush=True)
    assert d_rep == 0.0, f"ε 已按 seed 钉死但仍差 {d_rep:.3e},还有别的随机源,先查清"

    # ---- clean rollout:记 replan 边界的 sim 状态 + clean 动作,跑到成功 ----
    goal = env.env.parsed_problem["goal_state"]
    states, A_clean, ok, t = [], [], False, 0
    plan = collections.deque()
    max_steps = 400
    while t < max_steps + args.settle and len(states) < args.n_frames:
        if t < args.settle:
            obs, _, _, _ = env.step(get_libero_dummy_action()); t += 1; continue
        if not plan:
            a = infer(obs)                                   # [32,7],A_clean 的一帧
            states.append(env.get_sim_state().copy())
            A_clean.append(a)
            plan.extend(a[:int(cfg.EVALUATION.replan_steps)].tolist())
        # 执行侧的 gripper 变换必须**逐字**照 eval_libero_single.py:423-428,否则抓不起碗、
        # clean rollout 走不到成功。顺序:{0,1}→{-1,1}(×2−1)→ invert(×−1)→ binarize(sign)。
        # 前 6 维是 eef delta,原样喂。
        act = np.array(plan.popleft(), np.float64)
        act[-1] = np.sign(-(act[-1] * 2 - 1))                # = sign(1 − 2·g)
        obs, _, done, _ = env.step(act.tolist())
        if not ok and all(env.env._eval_predicate(s) for s in goal):    # noqa: SLF001
            ok = True
        t += 1
    env.close()
    T = len(states)
    A_clean = np.array(A_clean)                              # [T,32,7]
    print(f"[clean] 取到 {T} 帧(每 {args.replan} 执行步一帧)success={ok}", flush=True)

    # ---- patched:每个锚点注入一次,回放 T 帧 ----
    tex = str(args.texture)
    P = np.zeros((M, T, AH, 7), np.float64)
    vpx = np.zeros((M, T), np.int32)
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                              camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
    for i, a in enumerate(anchors):
        penv.env.set_xml_processor(sp.make_xml_processor(cfg_scene, a.world, tex))
        penv.seed(seed); penv.reset()
        gid = sp.patch_geom_id(penv, cfg_scene)
        for k in range(T):
            obs = penv.regenerate_obs_from_state(states[k])
            P[i, k] = infer(obs)
            vpx[i, k] = sp.visible_px(obs, "agentview", gid)
        if i % 6 == 0:
            print(f"  [{i:2d}/{M}] #{a.index} world=({a.world[0]:.2f},{a.world[1]:.2f}) "
                  f"vpx[t0]={vpx[i,0]}", flush=True)
    penv.close()

    outp = Path(args.out)
    np.savez_compressed(
        outp, task=TASK, suite=SUITE, seed=seed, T=T, M=M, action_horizon=AH,
        exec_prefix=int(cfg.EVALUATION.replan_steps), success=bool(ok), texture=tex,
        A_clean=A_clean, A_patched=P, visible_px=vpx,
        anchor_world=np.array([a.world for a in anchors]),
        anchor_uv=np.array([[a.u, a.v] for a in anchors]),
        anchor_idx=np.array([a.index for a in anchors]),
        anchor_legal=np.array([a.legal for a in anchors]),
        anchor_keepout=np.array([",".join(a.keepout_hits) for a in anchors]))
    print(f"[written] {outp}  ({outp.stat().st_size/2**20:.0f} MiB)  M={M} T={T}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
