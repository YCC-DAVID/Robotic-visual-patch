#!/usr/bin/env python3
"""S0.5 检查 A(交叉评估 4×4)+ 收尾项 B(B2 改写句 clean 成功率)。

这是 PART B(①换任务 / ②换措辞)能不能定稿的两个 go/no-go:
  A · 交叉评估:给指令 A,用**目标 B 的成功判据**评。若模型忽略文本、只按场景
     affordance 行动,交叉成功率就不会 ≈0 ⇒ B1 的前提(指令在控制行为)不成立。
  B · 改写句成功率:LIBERO 指令模板化,`倒装`之类脱离模板的说法可能把模型跑懵。
     成功率塌了的句子不能进 B2 对比 —— 否则比的是"正常 vs 懵",不是改写鲁棒性。

为什么一个 acting env 就能评 4 个目标
------------------------------------
libero_goal 十条 task 的 bddl **逐字节相同,只差 :goal/:language/:obj_of_interest 三行**
(计划已核 + Q10 实测:nbody/nsite/物体名跨 task 全同)。⇒ 任何一个 env 的
`object_states_dict` 都含全部物体/区域(bowl/plate/wine_bottle/rack/stove/cabinet)。
所以在指令 A 的 rollout 上,直接用 acting_env 评另外三条 task 的 goal_state 即可,
不必开 4 个并行 step 的 env。

四条 goal(从各自 bddl :goal 解析):
  stove       : (Turnon flat_stove_1)                            # 一元,读 stove 按钮
  bottle_rack : (On wine_bottle_1 wine_rack_1_top_region)        # fixture 区域
  bowl_plate  : (On akita_black_bowl_1 plate_1)                  # 物体-物体
  bowl_cabinet: (On akita_black_bowl_1 wooden_cabinet_1_top_side)# fixture 区域

跑法(两进程,和 demo 同架构):
    # 1) 先起 torch server(py3.11)
    #    python pi05probe/run_demo.py --torch --server-only
    # 2) 再跑本脚本(py3.8)
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/crosseval.py --episodes 10
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
MAX_STEPS = 300
NWAIT = 10

# 四条 B1 指令(文件名 stem)。stove 无抓取;1/2/3 物体集互不相交;3、4 最小对(只换目的地)。
B1_TASKS = [
    "turn_on_the_stove",
    "put_the_wine_bottle_on_the_rack",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_top_of_the_cabinet",
]

# B2 五个改写句(全在 bowl_plate 场景;只动动词/句法,绝不换名词 —— 计划硬约束)。
B2_REPHRASE = {
    "L1_place":   "place the bowl on the plate",
    "L1_set":     "set the bowl on the plate",
    "L1_move":    "move the bowl on the plate",
    "L2_frontPP": "on the plate, put the bowl",
    "L3_please":  "please put the bowl on the plate",
}


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


def eval_goal(env, goal_state):
    """在 env 当前 sim 状态上评估一条 goal_state(谓词合取),与 _check_success 同口径。"""
    return bool(all(env.env._eval_predicate(s) for s in goal_state))


def run_episode(env, client, prompt, goal_states, init_state, seed):
    """跑一条 rollout。返回 dict:{每个 goal_key: 是否曾达成(latch)}。"""
    env.seed(seed)                       # ⚠️ 每次 reset 前重 seed(Q10:fixture 不在 qpos 里)
    env.reset()
    obs = env.set_init_state(init_state)

    reached = {k: False for k in goal_states}
    plan = collections.deque()
    t = 0
    while t < MAX_STEPS + NWAIT:
        if t < NWAIT:
            obs, _, _, _ = env.step(DUMMY)
            t += 1
            continue
        if not plan:
            img, wri = model_input(obs)
            state8 = np.concatenate([obs["robot0_eef_pos"],
                                     quat2axisangle(obs["robot0_eef_quat"]),
                                     obs["robot0_gripper_qpos"]])
            chunk = client.infer({"observation/image": img, "observation/wrist_image": wri,
                                  "observation/state": state8, "prompt": prompt})["actions"]
            plan.extend(chunk[:REPLAN])
        obs, _, _, _ = env.step(plan.popleft().tolist())
        # 每步评全部 4 条 goal,latch(LIBERO 成功一旦达成即锁定)。
        # 不按自身目标提前停:交叉目标可能在自身目标之后才(偶然)达成,要如实统计整条轨迹。
        for k, gs in goal_states.items():
            if not reached[k] and eval_goal(env, gs):
                reached[k] = True
        t += 1
    return reached


def load_task(suite, stem):
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        if pathlib.Path(t.bddl_file).stem == stem:
            return i, t
    raise AssertionError(f"找不到 task {stem}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10, help="每条指令的 episode 数")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-b2", action="store_true", help="只做交叉评估,跳过 B2 改写成功率")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    init_dir = pathlib.Path(get_libero_path("init_states")) / "libero_goal"

    # 建 4 个 env,取各自 goal_state 与 prompt/init_states;这 4 个 env 复用作 acting。
    import torch
    envs, goals, prompts, inits, tids = {}, {}, {}, {}, {}
    for stem in B1_TASKS:
        tid, task = load_task(suite, stem)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        env.seed(args.seed)
        env.reset()
        envs[stem] = env
        goals[stem] = env.env.parsed_problem["goal_state"]
        prompts[stem] = str(task.language)
        tids[stem] = tid
        # ⚠️ py3.8 的 openpi-libero 是 torch 1.11,不接受 weights_only 关键字(那是 ≥2.6 才有)
        inits[stem] = np.array(torch.load(str(init_dir / f"{stem}.pruned_init")))
        print(f"[task {tid}] {stem}: prompt={prompts[stem]!r}  goal={goals[stem]}  "
              f"init_states={inits[stem].shape}", flush=True)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("[net] 连上 server(首次 infer 要等 torch.compile,可能几分钟)", flush=True)

    N = args.episodes
    goal_keys = list(B1_TASKS)

    # ---------- 项 A:交叉评估 4×4 ----------
    print("\n" + "=" * 90 + "\n项 A · 交叉评估:行=下达的指令,列=用哪条 goal 评\n" + "=" * 90, flush=True)
    matrix = {a: {b: 0 for b in goal_keys} for a in B1_TASKS}
    for a in B1_TASKS:
        env = envs[a]
        for ep in range(N):
            init = inits[a][ep % len(inits[a])]
            reached = run_episode(env, client, prompts[a], goals, init, args.seed + ep)
            for b in goal_keys:
                matrix[a][b] += int(reached[b])
            diag = reached[a]
            off = {b: reached[b] for b in goal_keys if b != a and reached[b]}
            print(f"  [{a[:22]:22s} ep{ep:02d}] self={int(diag)}  "
                  f"cross_hit={list(off.keys()) if off else '-'}", flush=True)

    # ---------- 项 B:B2 改写句成功率 ----------
    b2 = {}
    if not args.skip_b2:
        print("\n" + "=" * 90 + "\n项 B · B2 改写句 clean 成功率(全在 bowl_plate 场景,评 bowl_plate goal)\n"
              + "=" * 90, flush=True)
        bp = "put_the_bowl_on_the_plate"
        env = envs[bp]
        gkey = {bp: goals[bp]}
        # 基线:原句
        variants = {"B1_orig": prompts[bp], **B2_REPHRASE}
        for name, prompt in variants.items():
            hits = 0
            for ep in range(N):
                init = inits[bp][ep % len(inits[bp])]
                reached = run_episode(env, client, prompt, gkey, init, args.seed + ep)
                hits += int(reached[bp])
            b2[name] = hits
            print(f"  {name:12s} {prompt!r:42s}  成功率 {hits}/{N}", flush=True)

    # ---------- 汇总 ----------
    print("\n" + "=" * 90 + f"\n交叉评估矩阵(每格 = 成功次数 / {N})\n" + "=" * 90, flush=True)
    short = {"turn_on_the_stove": "stove", "put_the_wine_bottle_on_the_rack": "bottle_rack",
             "put_the_bowl_on_the_plate": "bowl_plate", "put_the_bowl_on_top_of_the_cabinet": "bowl_cabinet"}
    hdr = "指令\\评判        " + "".join(f"{short[b]:>13s}" for b in goal_keys)
    print(hdr, flush=True)
    for a in B1_TASKS:
        row = f"{short[a]:16s}" + "".join(f"{matrix[a][b]:>12d} " for b in goal_keys)
        print(row, flush=True)
    print("\n读法:对角线(自身)应高,非对角(交叉)应 ≈0。"
          "\n  交叉 ≈0 ⇒ 指令确实在控制行为 ⇒ B1 前提成立,①② 可定稿。"
          "\n  交叉不低 ⇒ 模型在广撒网/忽略指令 ⇒ 停下报告。", flush=True)

    res = {"episodes": N, "seed": args.seed, "short": short,
           "matrix": {a: matrix[a] for a in B1_TASKS}, "b2_success": b2,
           "tids": tids, "prompts": prompts}
    outp = OUT / "crosseval.json"
    outp.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n[written] {outp}", flush=True)
    for e in envs.values():
        e.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
