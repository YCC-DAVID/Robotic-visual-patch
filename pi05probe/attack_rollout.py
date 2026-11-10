#!/usr/bin/env python3
"""在指定位置贴上探针贴纸,跑**完整 episode**,比任务成功率。

为什么必须做这个
--------------
之前所有结论都建立在"模型这一帧的输出偏了多少"上。但小幅偏差可能被控制器自己修正回来 ⇒
**单帧动作偏差始终只是代理指标**。要证明"attention 选的位置 vs 影响力选的位置,差别有实际后果",
只有跑完整 rollout 比成功率这一条路。

对比的三个位置(来自 RESULTS.md 第 11 节)
--------------------------------------
    (-0.06, 0.22)   影响力最大的合法位置          126.4 mm  (100%)
    ( 0.21, 0.22)   attention 最常选的位置(36/108) 31.4 mm  ( 25%)
    ( 0.12, 0.35)   attention 次常选的位置(32/108) 18.9 mm  ( 15%)
外加 clean 对照(不贴)。

受控设计
-------
四个条件用**同一批 init_states**(LIBERO 官方的,取前 N 个)、同一个 seed。
贴纸整个 episode 静止不动(这就是威胁模型:一张贴在桌上的静态贴纸)。
成功判据用 LIBERO 自己的谓词,且 latch(一旦达成即锁定),与官方 `_check_success` 同口径。

⚠️ 预期:随机纹理很可能三个位置都撼不动模型(幅度压不过采样抖动)。那也是有价值的结果 ——
   它给出一个**下界**,并且验证整条 attacked-rollout 管线通不通;之后换优化过的贴纸
   只是替换 --texture,其余不变。

用法:
    # 1) 先起 server(py3.11)
    #    python pi05probe/run_demo.py --torch --server-only
    # 2) 再跑本脚本(py3.8)
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/attack_rollout.py --episodes 15
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
import yaml  # noqa: E402

RES, RESIZE, REPLAN, NWAIT = 256, 224, 5, 10
MAX_STEPS = 300
DUMMY = [0.0] * 6 + [-1.0]
TASK = "put_the_bowl_on_the_plate"

# 名字 → 世界 (x, y);None = clean 对照
CONDITIONS = [
    ("clean", None, "不贴,对照"),
    ("influence_best", (-0.06, 0.22), "影响力最大的合法位置 126.4mm"),
    ("attention_top1", (0.21, 0.22), "attention 最常选 36/108,影响力 31.4mm"),
    ("attention_top2", (0.12, 0.35), "attention 次常选 32/108,影响力 18.9mm"),
    ("trained_cell", (0.045, 0.117), "per-cell influence/gradient 合法首选;对抗训练用格"),
]


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


def run_episode(env, client, prompt, goal_state, init_state, seed):
    """跑一条完整 rollout。返回 (success, 成功时的步数或 None, 末端轨迹)。"""
    env.seed(seed)                 # ⚠️ 每次 reset 前重 seed(fixture 不在 qpos 里)
    env.reset()
    obs = env.set_init_state(init_state)
    plan = collections.deque()
    ok, t_ok, traj = False, None, []
    t = 0
    while t < MAX_STEPS + NWAIT:
        if t < NWAIT:
            obs, _, _, _ = env.step(DUMMY)
            t += 1
            continue
        if not plan:
            img, wri = model_input(obs)
            st = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                 obs["robot0_gripper_qpos"]])
            chunk = client.infer({"observation/image": img, "observation/wrist_image": wri,
                                  "observation/state": st, "prompt": prompt})["actions"]
            plan.extend(chunk[:REPLAN])
        obs, _, _, _ = env.step(plan.popleft().tolist())
        traj.append(obs["robot0_eef_pos"].copy())
        if not ok and all(env.env._eval_predicate(s) for s in goal_state):  # noqa: SLF001
            ok, t_ok = True, t
        t += 1
    return ok, t_ok, np.array(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--texture", default=None)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--only", default=None,
                    help="逗号分隔的条件名,只跑这些(如 clean)。默认全跑")
    ap.add_argument("--out", default="attack_rollout")
    args = ap.parse_args()
    conds = CONDITIONS if not args.only else [
        c for c in CONDITIONS if c[0] in args.only.split(",")]
    assert conds, f"--only {args.only} 没匹配到任何条件;可选:{[c[0] for c in CONDITIONS]}"

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    import scene_patch as sp

    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    tex = str(args.texture or (PROBE / "config" / "probe_texture.png"))
    assert pathlib.Path(tex).is_file(), f"纹理不存在:{tex}"

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid = task = None
    for i in range(suite.n_tasks):
        t_ = suite.get_task(i)
        if pathlib.Path(t_.bddl_file).stem == TASK:
            tid, task = i, t_
            break
    assert task is not None
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    prompt = str(task.language)

    # ⚠️ LIBERO 的 init_states 是 `.pruned_init`,得用 torch.load,不是 np.load。
    #    py3.8 这个 env 是 torch 1.11,不接受 weights_only 关键字(那是 ≥2.6 才有)。
    import torch
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                    "libero_goal" / f"{TASK}.pruned_init")))
    N = min(args.episodes, len(inits))
    print(f"[cfg] task={TASK}  prompt={prompt!r}  可用 init_states={len(inits)}  用前 {N} 个",
          flush=True)
    print(f"[cfg] texture={tex}  seed={args.seed}", flush=True)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("[net] 连上 server(首次 infer 要等 torch.compile,可能几分钟)", flush=True)

    results = {}
    for name, pos, desc in conds:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        if pos is not None:
            world = (pos[0], pos[1], cfg["plane"]["origin"][2] +
                     cfg["patch"]["thickness"] / 2 + cfg["patch"]["normal_offset"])
            env.env.set_xml_processor(sp.make_xml_processor(cfg, world, tex))
        env.seed(args.seed); env.reset()
        goal_state = env.env.parsed_problem["goal_state"]

        succ, steps = [], []
        for ep in range(N):
            ok, t_ok, _ = run_episode(env, client, prompt, goal_state, inits[ep], args.seed + ep)
            succ.append(ok)
            if ok:
                steps.append(t_ok)
            print(f"  [{name:15s}] ep {ep+1:2d}/{N}  success={ok}"
                  f"{'' if not ok else f'  t={t_ok}'}", flush=True)
        env.close()
        rate = float(np.mean(succ))
        results[name] = dict(rate=rate, n=N, successes=int(np.sum(succ)),
                             mean_steps=(float(np.mean(steps)) if steps else None),
                             pos=pos, desc=desc, per_ep=[bool(x) for x in succ])
        print(f"  ==> {name:15s} 成功率 {np.sum(succ)}/{N} = {rate:.1%}"
              f"{'' if not steps else f'  平均成功步数 {np.mean(steps):.0f}'}\n", flush=True)

    lines = ["=" * 92,
             f"完整 rollout 成功率对比   task={TASK}   每条件 {N} 个 episode(同一批 init_states)",
             f"  纹理={pathlib.Path(tex).name}   seed={args.seed}",
             "=" * 92,
             "  条件              位置(x,y)        成功率      平均成功步数  说明"]
    base = results.get("clean", {}).get("rate")
    for name, pos, desc in conds:
        r = results[name]
        ps = "  --  " if pos is None else f"({pos[0]:5.2f},{pos[1]:5.2f})"
        ms = "  -- " if r["mean_steps"] is None else f"{r['mean_steps']:5.0f}"
        lines.append(f"  {name:16s}  {ps:15s}  {r['successes']:2d}/{r['n']} = {r['rate']:5.1%}"
                     f"   {ms}       {desc}")
    lines.append("")
    if base is not None:
        lines.append("  相对 clean 的成功率下降(攻击效果):")
    for name, pos, desc in [c for c in conds[1:] if base is not None]:
        d = base - results[name]["rate"]
        lines.append(f"    {name:16s} {d:+.1%}"
                     f"   ({'有效' if d > 0 else '无效果/反向'})")
    lines.append("")
    lines.append("  ⚠️ 用的是**随机**探针纹理,不是优化过的对抗纹理。三个位置都无下降属于预期,")
    lines.append("     那说明随机纹理不足以攻击成功,不说明位置排序错 —— 优化纹理是下一步。")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    (OUT / f"{args.out}.txt").write_text(txt + "\n")
    (OUT / f"{args.out}.json").write_text(json.dumps(results, indent=2))
    print(f"\n[written] {OUT/(args.out+'.txt')} + .json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
