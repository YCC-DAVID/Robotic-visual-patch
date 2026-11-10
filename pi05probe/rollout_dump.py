#!/usr/bin/env python3
"""沿一条 clean rollout 采帧,给 PART B 的时间轴(和 S2 用同一批帧)。

跑在 py3.8 的 openpi-libero 环境;policy 在 py3.11,走 websocket
(先起:run_demo.py --torch --server-only)。

为什么要沿轨迹采
--------------
单帧(t0)时机械臂还没动、场景是静的,看不出 attention 在跟什么。
而且 **B3 噪声地板**必须有相邻帧才能算 —— 没有它,"换措辞 Spearman 0.95 算高吗"
这个问题没有参照。

采样点 = **每个 replan 边界**,即 `examples/libero/main.py` 里
`action_plan` 空掉、需要重新查询模型的那些步(replan_steps=5)。
这跟 S2 要的采样点是同一套(计划:T = clean rollout 的全部 replan 边界),
所以这批帧后面 Δa 扫描可以直接复用。

初始状态用 shared_frame.npz 里那个共享锚点(S0.5 检查 B 已验证四条指令逐位相同),
**每次 reset 前重新 seed**(Q10 的坑:三个 fixture 不在 qpos 里)。

⚠️ 必须是**成功**的 episode(计划的硬要求):夹爪通道的关键事件只在最后松手那一刻发生,
   轨迹没走到成功,夹爪那一路等于没测。失败会 assert 并提示换 seed。

用法:
    # 1) 先起 server(py3.11)
    #    run_demo.py --torch --server-only
    # 2) 再跑本脚本(py3.8)
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/rollout_dump.py
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/rollout_dump.py --task put_the_bowl_on_the_plate
"""

import argparse
import collections
import math
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

RES, RESIZE, REPLAN = 256, 224, 5
DUMMY = [0.0] * 6 + [-1.0]
MAX_STEPS = 300                      # libero_goal(main.py:65)


def quat2axisangle(quat):
    q = np.array(quat, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def model_input(obs):
    """main.py:113-122。⚠️ [::-1, ::-1] 是 180° 旋转 buffer,相对真实场景是左右镜像。"""
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wri = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return (image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE)),
            image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, RESIZE, RESIZE)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="put_the_bowl_on_the_plate")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy

    # ---- 共享锚点:直接从 shared_frame.npz 读,保持单一来源
    fr = np.load(OUT / "shared_frame.npz", allow_pickle=False)
    shared_state = fr["shared_state"]
    shared_seed = int(fr["shared_seed"])
    n_wait = int(fr["num_steps_wait"])
    print(f"[cfg] shared_seed={shared_seed}  num_steps_wait={n_wait}  "
          f"state_source={str(fr['state_source'])}", flush=True)

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid, task = None, None
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        if pathlib.Path(t.bddl_file).stem == args.task:
            tid, task = i, t
            break
    assert task is not None, f"找不到 task {args.task}"
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    prompt = str(task.language)
    print(f"[cfg] task[{tid}] {args.task}   language={prompt!r}", flush=True)

    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    env.seed(shared_seed)            # ⚠️ 必须在【每一次】 reset() 之前
    env.reset()
    obs = env.set_init_state(shared_state)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("[net] 已连上 server(首次 infer 要等 torch.compile,可能几分钟)", flush=True)

    frames, plan = [], collections.deque()
    t, done, k = 0, False, 0
    while t < MAX_STEPS + n_wait:
        if t < n_wait:
            obs, _, done, _ = env.step(DUMMY)
            t += 1
            continue
        img, wri = model_input(obs)
        if not plan:
            # ---- replan 边界:这一帧就是采样点
            state8 = np.concatenate([obs["robot0_eef_pos"],
                                     quat2axisangle(obs["robot0_eef_quat"]),
                                     obs["robot0_gripper_qpos"]])
            frames.append(dict(k=k, t=t, img224=img, wrist224=wri, state8=state8,
                               rgb256=obs["agentview_image"].copy(),
                               flatten=env.env.sim.get_state().flatten().copy()))
            chunk = client.infer({"observation/image": img, "observation/wrist_image": wri,
                                  "observation/state": state8, "prompt": prompt})["actions"]
            assert len(chunk) >= REPLAN, f"chunk 只有 {len(chunk)} 步 < replan {REPLAN}"
            plan.extend(chunk[:REPLAN])
            if k % 5 == 0:
                print(f"  [k={k:3d} t={t:3d}] 已采 {len(frames)} 帧", flush=True)
            k += 1
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            break
        t += 1

    print(f"\n[结果] success={done}  env 步数 t={t}  采样帧数={len(frames)}", flush=True)
    assert done, ("❌ 这条 rollout 没成功。计划的硬要求是必须用成功的 episode —— "
                  "夹爪通道的关键事件只在最后松手那一刻发生,没走到成功等于夹爪那一路没测。"
                  "换 seed 或换共享状态重跑。")

    outp = pathlib.Path(args.out) if args.out else OUT / f"traj_{args.task}.npz"
    save = {f"f{r['k']:03d}__{key}": v for r in frames for key, v in r.items() if key != "k"}
    np.savez_compressed(
        outp, n_frames=len(frames), task=np.array(args.task), prompt=np.array(prompt),
        task_id=tid, success=bool(done), env_steps=t, replan=REPLAN, num_steps_wait=n_wait,
        shared_seed=shared_seed, ks=np.array([r["k"] for r in frames]),
        ts=np.array([r["t"] for r in frames]), **save)
    print(f"[written] {outp}  ({outp.stat().st_size / 2**20:.1f} MiB)", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
