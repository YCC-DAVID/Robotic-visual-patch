#!/usr/bin/env python3
"""去掉主视角后多出来的那十几步,花在哪个阶段?

假设
----
主视角(固定俯瞰相机)是唯一能看到**目的地**的视角 —— 腕部在抓取段盘子直接出画面。
若把主视角置 0,策略就得靠移动手腕去找盘子 ⇒ **多出来的步数应集中在抓取之后的搬运/放置段**,
而不是接近段。若多出的步数均匀分布在两段,那就不是"找盘子",而是整体变迟钝。

做法
----
同一批 init_state,分别对着两个 server 跑 clean rollout,逐步记录末端位置与夹爪开合。
用夹爪闭合的时刻切分阶段:
    接近段 = 起点 → 夹爪首次闭合
    搬运段 = 夹爪闭合 → 任务成功

用法(py3.8,两个 server 都要在):
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/phase_timing.py --episodes 8
"""
import argparse
import collections
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"

for p in reversed([OPENPI / "packages" / "openpi-client" / "src", OPENPI / "third_party" / "libero"]):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(PROBE))
os.environ["LIBERO_CONFIG_PATH"] = str(PROBE / "libero_config")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYTHONNOUSERSITE"] = "1"

import numpy as np  # noqa: E402

RES, RESIZE, REPLAN, NWAIT = 256, 224, 5, 10
MAX_STEPS = 300
DUMMY = [0.0] * 6 + [-1.0]
TASK = "put_the_bowl_on_the_plate"
GRIP_CLOSED = 0.02          # qpos[0] 低于此判为闭合(clean 张开约 0.039,闭合约 0.002)


def quat2axisangle(q):
    q = np.array(q, dtype=np.float64)
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


def run(env, client, prompt, goal, init, seed):
    env.seed(seed); env.reset()
    obs = env.set_init_state(init)
    plan = collections.deque()
    eef, grip = [], []
    ok, t_ok, t = False, None, 0
    while t < MAX_STEPS + NWAIT:
        if t < NWAIT:
            obs, _, _, _ = env.step(DUMMY); t += 1; continue
        if not plan:
            img, wri = model_input(obs)
            st = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                 obs["robot0_gripper_qpos"]])
            plan.extend(client.infer({"observation/image": img, "observation/wrist_image": wri,
                                      "observation/state": st,
                                      "prompt": prompt})["actions"][:REPLAN])
        obs, _, _, _ = env.step(plan.popleft().tolist())
        eef.append(obs["robot0_eef_pos"].copy())
        grip.append(float(obs["robot0_gripper_qpos"][0]))
        if not ok and all(env.env._eval_predicate(s) for s in goal):     # noqa: SLF001
            ok, t_ok = True, t
        t += 1
        if ok:
            break
    return np.array(eef), np.array(grip), ok, t_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--ports", default="8128:base_normal,8127:base_zeroed")
    args = ap.parse_args()

    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    prompt = str(task.language)
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                    "libero_goal" / f"{TASK}.pruned_init")))
    N = min(args.episodes, len(inits))

    res = {}
    for spec in args.ports.split(","):
        port, name = spec.split(":")
        client = websocket_client_policy.WebsocketClientPolicy(args.host, int(port))
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        env.seed(args.seed); env.reset()
        goal = env.env.parsed_problem["goal_state"]
        rows = []
        for ep in range(N):
            eef, grip, ok, t_ok = run(env, client, prompt, goal, inits[ep], args.seed + ep)
            closed = np.where(grip < GRIP_CLOSED)[0]
            t_grasp = int(closed[0]) + NWAIT if len(closed) else None
            rows.append(dict(ok=bool(ok), t_success=t_ok, t_grasp=t_grasp, n=len(eef),
                             path=float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum() * 1000)))
            print(f"  [{name:12s}] ep {ep+1}/{N}  success={ok}  抓取@{t_grasp}  成功@{t_ok}  "
                  f"总行程={rows[-1]['path']:.0f}mm", flush=True)
        env.close()
        res[name] = rows

    lines = ["=" * 96,
             f"去掉主视角后多出的步数花在哪个阶段   task={TASK}   每条件 {N} 个 episode",
             "  接近段 = 起点 → 夹爪首次闭合;搬运段 = 夹爪闭合 → 任务成功",
             "=" * 96,
             "  条件           成功率 | 接近段步数  搬运段步数  总步数 | 末端总行程 mm"]
    for name, rows in res.items():
        ok = [r for r in rows if r["ok"] and r["t_grasp"]]
        app = np.mean([r["t_grasp"] - NWAIT for r in ok])
        car = np.mean([r["t_success"] - r["t_grasp"] for r in ok])
        tot = np.mean([r["t_success"] for r in ok])
        pl = np.mean([r["path"] for r in ok])
        lines.append(f"  {name:14s} {len(ok)}/{len(rows)}  | {app:10.1f} {car:11.1f} "
                     f"{tot:8.1f} | {pl:13.0f}")
    if len(res) == 2:
        a, b = list(res)
        oa = [r for r in res[a] if r["ok"] and r["t_grasp"]]
        obb = [r for r in res[b] if r["ok"] and r["t_grasp"]]
        d_app = np.mean([r["t_grasp"] - NWAIT for r in obb]) - np.mean([r["t_grasp"] - NWAIT for r in oa])
        d_car = (np.mean([r["t_success"] - r["t_grasp"] for r in obb])
                 - np.mean([r["t_success"] - r["t_grasp"] for r in oa]))
        lines += ["", f"  {b} 相对 {a} 的增量:接近段 {d_app:+.1f} 步,搬运段 {d_car:+.1f} 步",
                  "  判读:若增量几乎全在搬运段 ⇒ 多出的步数是在找盘子(目的地);",
                  "        若两段都增 ⇒ 不是找目的地,是整体变迟钝。"]
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    (OUT / "phase_timing.txt").write_text(txt + "\n")
    (OUT / "phase_timing.json").write_text(json.dumps(res, indent=2))
    print(f"\n[written] {OUT/'phase_timing.txt'}")


if __name__ == "__main__":
    main()
