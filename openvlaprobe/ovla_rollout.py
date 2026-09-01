#!/usr/bin/env python3
"""离散 OpenVLA(原版 7B,自回归 token action head)在 LIBERO-goal 上的闭环 rollout。

阳性对照的意义
-------------
UADA/最大偏移这些攻击是在**离散 token action head**上设计的(翻转 argmax token)。要证明
"它们在 flow-matching 头(π0.5/FastWAM)上失效"是**头架构差异**而非实验问题,就得把原版离散
OpenVLA 当阳性对照:同一 LIBERO-goal put_the_bowl_on_the_plate、同一批 init_states、同一成功谓词。
本脚本先验 **clean 基线成功率**(应 ~90%+),攻击复现是下一步。

对齐 π0.5/FastWAM harness
------------------------
- 同 task / 同 init_states(前 N)/ 同 seed / LIBERO 官方 _eval_predicate + latch。
- OpenVLA 特有(官方 experiments/robot/libero 口径):
    图像 agentview[::-1,::-1](180°)→ processor resize 224;单第三人称图,无 wrist。
    prompt "In: What action should the robot take to {task}?\nOut:"。
    动作 predict_action(unnorm_key='libero_goal') → gripper: sign(2g-1) 再 ×-1(净 -sign(2g-1))。
    单动作/步,每步重推(无 chunk)。settle=10。

用法(user58 openvla env,transformers 4.40.1;HF_HOME 指向已缓存权重):
    PYTHONNOUSERSITE=1 HF_HOME=/shared/user58/.cache/huggingface \
    LIBERO_CONFIG_PATH=/home/user1/workspace/chence/WAMattack/pi05probe/libero_config \
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=6 \
      /shared/user58/miniconda3/envs/openvla/bin/python openvlaprobe/ovla_rollout.py --episodes 15
"""
import argparse
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
LIBERO = ROOT / "third_party" / "openpi" / "third_party" / "libero"
PROBE = ROOT / "openvlaprobe"
OUT = PROBE / "out"
sys.path.insert(0, str(LIBERO))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

RES = 256
MAX_STEPS = 300
NWAIT = 10
TASK = "put_the_bowl_on_the_plate"
GOAL_CKPT = "/shared/user58/.cache/huggingface/hub/models--openvla--openvla-7b-finetuned-libero-goal/snapshots"


def quat2axisangle(quat):
    q = np.array(quat, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--out", default="ovla_clean")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    snap_root = pathlib.Path(GOAL_CKPT)
    snap = str(snap_root / os.listdir(snap_root)[0])
    print(f"[model] loading {snap}", flush=True)
    proc = AutoProcessor.from_pretrained(snap, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        snap, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True).to("cuda:0").eval()
    print(f"[model] loaded dtype={next(vla.parameters()).dtype}", flush=True)

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid = next(i for i in range(suite.n_tasks)
               if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    task_desc = str(task.language)
    prompt = f"In: What action should the robot take to {task_desc.lower()}?\nOut:"
    print(f"[task] {task_desc!r}  prompt={prompt!r}", flush=True)

    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                    "libero_goal" / f"{TASK}.pruned_init")))
    N = min(args.episodes, len(inits))
    print(f"[cfg] init_states={len(inits)} 用前 {N}  seed={args.seed}", flush=True)

    def predict(obs):
        img = obs["agentview_image"][::-1, ::-1]                 # 180° 对齐训练
        pil = Image.fromarray(np.ascontiguousarray(img)).convert("RGB")
        inp = proc(prompt, pil).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            a = np.asarray(vla.predict_action(**inp, unnorm_key="libero_goal", do_sample=False),
                           dtype=np.float64)
        a[-1] = np.sign(2.0 * a[-1] - 1.0) * -1.0                # normalize+binarize+invert gripper
        return a

    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    env.seed(args.seed); env.reset()
    goal_state = env.env.parsed_problem["goal_state"]

    succ, steps = [], []
    for ep in range(N):
        env.seed(args.seed + ep); env.reset()
        obs = env.set_init_state(inits[ep])
        ok, t_ok, t = False, None, 0
        while t < MAX_STEPS + NWAIT:
            if t < NWAIT:
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1]); t += 1; continue
            a = predict(obs)
            obs, _, done, _ = env.step(a.tolist())
            if not ok and all(env.env._eval_predicate(s) for s in goal_state):  # noqa: SLF001
                ok, t_ok = True, t
            if done:
                ok = True; t_ok = t_ok or t; break
            t += 1
        succ.append(ok)
        if ok:
            steps.append(t_ok)
        print(f"  [clean] ep {ep+1:2d}/{N} success={ok}{'' if not ok else f' t={t_ok}'}", flush=True)
    env.close()

    rate = float(np.mean(succ))
    res = dict(rate=rate, n=N, successes=int(np.sum(succ)),
               mean_steps=(float(np.mean(steps)) if steps else None),
               per_ep=[bool(x) for x in succ])
    txt = (f"OpenVLA(离散 7B) LIBERO-goal clean rollout  task={TASK}\n"
           f"  成功率 {int(np.sum(succ))}/{N} = {rate:.1%}"
           f"{'' if not steps else f'  平均成功步数 {np.mean(steps):.0f}'}  seed={args.seed}")
    print("\n" + txt, flush=True)
    (OUT / f"{args.out}.txt").write_text(txt + "\n")
    (OUT / f"{args.out}.json").write_text(json.dumps(res, indent=2))
    print(f"[written] {OUT/(args.out+'.txt')}", flush=True)


if __name__ == "__main__":
    main()
