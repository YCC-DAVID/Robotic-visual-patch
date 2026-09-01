#!/usr/bin/env python3
"""FastWAM 闭环 rollout 验收(π0.5 attack_rollout.py 的跨模型对应件,wamattack env,py3.10)。

在攻击格贴上**物理纹理**(优化过的 fw_texture_adv_{loss}.png 或随机 probe_texture.png),
跑完整 episode,与 clean 对照比任务成功率。模型在进程内直接跑(不用 websocket)。
成功判据:LIBERO 官方 _eval_predicate + latch,与 eval_libero_single 的 done 同口径。

同一批 init_states、同一 seed;贴纸整个 episode 静止(威胁模型=桌上一张静态贴纸)。

用法:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 \
      /home/user1/miniconda3/envs/wamattack/bin/python probe/fw_attack_rollout.py \
        --episodes 15 --textures away,curve
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
OUT = REPO / "probe" / "out"

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

TASK = "put_the_bowl_on_the_plate"
SUITE = "libero_goal"
MAX_STEPS = 400
CELL_XY = (0.2227855, -0.00254161)     # gradient/influence argmax 合法格
PATCH_M = 0.13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--textures", default="away,curve",
                    help="逗号分隔:clean 之外要测哪些优化纹理(对应 config/fw_texture_adv_<x>.png)")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--out", default="fw_attack_rollout")
    args = ap.parse_args()

    cfg_scene = yaml.safe_load((PI05 / "config" / "scene.yaml").read_text())
    cfg_scene["patch"]["size_wh"] = [PATCH_M, PATCH_M]
    import scene_patch as sp
    plane = sp.Plane.from_cfg(cfg_scene["plane"])
    lift = cfg_scene["patch"]["thickness"] / 2.0 + cfg_scene["patch"]["normal_offset"]
    cell_world = plane.to_world(CELL_XY[0], CELL_XY[1], lift)

    from run_instruction_sweep import build_cfg, encode_instructions, load_model
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_utils import get_libero_dummy_action, LIBERO_ENV_RESOLUTION
    from eval_libero_single import _obs_to_model_input, _denormalize_action

    suite_obj = benchmark.get_benchmark_dict()[SUITE]()
    tid = next(i for i in range(suite_obj.n_tasks)
               if Path(suite_obj.get_task(i).bddl_file).stem == TASK)
    task = suite_obj.get_task(tid)

    class A:
        suite, task_id, init_index = SUITE, tid, 0
    cfg = build_cfg(A())
    seed = int(cfg.seed)
    prompt = DEFAULT_PROMPT.format(task=str(task.language))
    context, cmask = encode_instructions(cfg, [prompt], "cpu")
    model, processor, device, dtype = load_model(cfg)
    Hd, Wd = [int(v) for v in cfg.data.train.video_size]
    AH = int(cfg.data.train.num_frames) - 1
    ns = int(cfg.EVALUATION.num_inference_steps)
    sshift = cfg.EVALUATION.get("sigma_shift")
    rdev = str(cfg.EVALUATION.get("rand_device", "cpu"))
    replan = int(cfg.EVALUATION.replan_steps)
    SETTLE = int(cfg.EVALUATION.get("num_steps_wait", 5))
    ctx = context.to(device=device, dtype=dtype)
    cmk = cmask.to(device=device)

    def infer(obs):
        x, proprio, _ = _obs_to_model_input(obs, cfg=cfg, processor=processor,
                                            width=Wd, height=Hd, device=device, dtype=dtype)
        with torch.no_grad():
            pred = model.infer_action(
                prompt=None, input_image=x, action_horizon=AH, proprio=proprio,
                context=ctx, context_mask=cmk, num_inference_steps=ns, sigma_shift=sshift,
                seed=seed, rand_device=rdev, tiled=False)
        return _denormalize_action(pred["action"], processor)[0]     # [32,7]

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    inits = np.array(torch.load(str(Path(get_libero_path("init_states")) / SUITE / f"{TASK}.pruned_init")))
    N = min(args.episodes, len(inits))
    print(f"[cfg] task={TASK} seed={seed} 每条件 {N} episode  攻击格 world=({CELL_XY[0]:.3f},{CELL_XY[1]:.3f})",
          flush=True)

    def run_episode(env, goal, init_state, epseed):
        env.seed(epseed); env.reset()
        obs = env.set_init_state(init_state)
        plan = collections.deque()
        ok, t_ok, t = False, None, 0
        while t < MAX_STEPS + SETTLE:
            if t < SETTLE:
                obs, _, _, _ = env.step(get_libero_dummy_action()); t += 1; continue
            if not plan:
                a = infer(obs)
                plan.extend(a[:replan].tolist())
            act = np.array(plan.popleft(), np.float64)
            act[-1] = np.sign(-(act[-1] * 2 - 1))
            obs, _, done, _ = env.step(act.tolist())
            if not ok and all(env.env._eval_predicate(s) for s in goal):  # noqa: SLF001
                ok, t_ok = True, t
            if done:
                ok = True
                if t_ok is None:
                    t_ok = t
                break
            t += 1
        return ok, t_ok

    conds = [("clean", None)]
    for name in args.textures.split(","):
        name = name.strip()
        if not name or name == "clean":
            continue
        tp = PI05 / "config" / f"fw_texture_adv_{name}.png"
        assert tp.is_file(), f"纹理不存在:{tp}"
        conds.append((name, str(tp)))

    results = {}
    for name, tex in conds:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                                 camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
        if tex is not None:
            env.env.set_xml_processor(sp.make_xml_processor(cfg_scene, cell_world, tex))
        env.seed(seed); env.reset()
        goal = env.env.parsed_problem["goal_state"]
        succ, steps = [], []
        for ep in range(N):
            ok, t_ok = run_episode(env, goal, inits[ep], args.seed + ep)
            succ.append(ok)
            if ok:
                steps.append(t_ok)
            print(f"  [{name:10s}] ep {ep+1:2d}/{N} success={ok}{'' if not ok else f' t={t_ok}'}",
                  flush=True)
        env.close()
        rate = float(np.mean(succ))
        results[name] = dict(rate=rate, n=N, successes=int(np.sum(succ)),
                             mean_steps=(float(np.mean(steps)) if steps else None),
                             per_ep=[bool(x) for x in succ])
        print(f"  ==> {name:10s} 成功率 {int(np.sum(succ))}/{N} = {rate:.1%}"
              f"{'' if not steps else f'  平均成功步数 {np.mean(steps):.0f}'}\n", flush=True)

    base = results.get("clean", {}).get("rate")
    lines = ["=" * 80, f"FastWAM 闭环 rollout 成功率  task={TASK}  每条件 {N} episode",
             f"  攻击格 world=({CELL_XY[0]:.3f},{CELL_XY[1]:.3f})  patch={PATCH_M*100:.0f}cm  seed={seed}",
             "=" * 80, "  条件         成功率        平均成功步数"]
    for name, _ in conds:
        r = results[name]
        ms = " -- " if r["mean_steps"] is None else f"{r['mean_steps']:5.0f}"
        lines.append(f"  {name:11s}  {r['successes']:2d}/{r['n']} = {r['rate']:5.1%}   {ms}")
    if base is not None:
        lines.append("\n  相对 clean 的成功率下降(攻击效果):")
        for name, _ in conds[1:]:
            d = base - results[name]["rate"]
            lines.append(f"    {name:11s} {d:+.1%}  ({'有效' if d > 0 else '无效果/反向'})")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    (OUT / f"{args.out}.txt").write_text(txt + "\n")
    (OUT / f"{args.out}.json").write_text(json.dumps(results, indent=2))
    print(f"\n[written] {OUT/(args.out+'.txt')} + .json", flush=True)


if __name__ == "__main__":
    main()
