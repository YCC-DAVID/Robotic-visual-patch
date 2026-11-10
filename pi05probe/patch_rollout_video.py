#!/usr/bin/env python3
"""真实 patch rollout:录像 + 记录末端轨迹,看误差叠加后是怎么被纠回来的。

和之前那个反事实扫描的区别
------------------------
扫描是"每帧都从同一个 clean 状态出发,只问模型会不会想歪",轨迹永远不发散。
这里是**真贴上去跑**:第一帧的偏差会带着状态走,后面越差越多 —— 直到控制器/策略
把它拉回来(或者拉不回来)。

⚠️ 一个必须处理的混淆
--------------------
websocket 的 `infer` 传不了噪声(in-process 才能传 `noise=`),所以每次查询都会重采。
于是 clean 与 patched 的轨迹差里**混着采样噪声**。
⇒ 必须同时跑 **clean 两遍**(同一初始状态、不同噪声),拿 clean-vs-clean 的偏移当参照。
   patched 的偏移要明显超过它才算贴纸的作用。

产出(每个条件一份)
    out/rollout_video/<cond>.mp4          agentview 录像
    out/rollout_video/traj.npz            每步的末端位置 + 夹爪 + 成功步号
    out/rollout_video/divergence.png      偏移随时间的曲线(含 clean-vs-clean 参照)

用法(py3.8,需要先起 server):
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/patch_rollout_video.py --episode 0
"""
import argparse
import collections
import math
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out" / "rollout_video"

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

# (标签, 位置 or None, 说明)。clean 跑两遍 ⇒ 拿到"只有采样噪声"的参照。
# influence 与 attention 各自选的位置都录,直接对着看。两者都是**合法**位置。
CONDS = [
    ("clean_a", None, "clean, run A"),
    ("clean_b", None, "clean, run B (same state, different sampling noise)"),
    ("influence_pick", (-0.06, 0.22), "influence's pick  (-0.06, 0.22)  125 mm"),
    ("attention_pick", (0.21, 0.22), "attention's pick  (0.21, 0.22)   31 mm"),
]
PATCHED = ("influence_pick", "attention_pick")
COLORS = {"clean_b": "#8a8880", "influence_pick": "#eb6834", "attention_pick": "#2a78d6"}


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


def run(env, client, prompt, goal, init, seed):
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init)
    plan = collections.deque()
    frames, eef, grip = [], [], []
    ok, t_ok, t = False, None, 0
    while t < MAX_STEPS + NWAIT:
        if t < NWAIT:
            obs, _, _, _ = env.step(DUMMY)
            t += 1
            continue
        if not plan:
            img, wri = model_input(obs)
            st = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                 obs["robot0_gripper_qpos"]])
            plan.extend(client.infer({"observation/image": img, "observation/wrist_image": wri,
                                      "observation/state": st,
                                      "prompt": prompt})["actions"][:REPLAN])
        obs, _, _, _ = env.step(plan.popleft().tolist())
        frames.append(np.ascontiguousarray(obs["agentview_image"][::-1]))   # 竖直翻正,便于观看
        eef.append(obs["robot0_eef_pos"].copy())
        grip.append(float(obs["robot0_gripper_qpos"][0]))
        if not ok and all(env.env._eval_predicate(s) for s in goal):        # noqa: SLF001
            ok, t_ok = True, t
        t += 1
        if ok and t > t_ok + 12:        # 成功后多录一点点就停,省时间
            break
    return dict(frames=np.array(frames), eef=np.array(eef), grip=np.array(grip),
                success=ok, t_success=t_ok, n=len(eef))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0, help="用第几个官方 init_state")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--texture", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import imageio
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    import scene_patch as sp

    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    tex = str(args.texture or (PROBE / "config" / "probe_texture.png"))
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    prompt = str(task.language)
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                    "libero_goal" / f"{TASK}.pruned_init")))
    init = inits[args.episode]
    print(f"[cfg] task={TASK}  episode={args.episode}  prompt={prompt!r}", flush=True)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    res = {}
    for name, pos, desc in CONDS:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        if pos is not None:
            world = (pos[0], pos[1], cfg["plane"]["origin"][2] +
                     cfg["patch"]["thickness"] / 2 + cfg["patch"]["normal_offset"])
            env.env.set_xml_processor(sp.make_xml_processor(cfg, world, tex))
        env.seed(args.seed); env.reset()
        goal = env.env.parsed_problem["goal_state"]
        r = run(env, client, prompt, goal, init, args.seed)
        env.close()
        res[name] = r
        mp4 = OUT / f"{name}.mp4"
        imageio.mimwrite(mp4, r["frames"], fps=20, quality=8)
        print(f"  [{name:11s}] success={r['success']}  t_success={r['t_success']}  "
              f"帧数={r['n']}  → {mp4.name}", flush=True)

    # ---- 偏移曲线:以 clean_a 为基准
    n = min(r["n"] for r in res.values())
    base = res["clean_a"]["eef"][:n]
    curves = {k: np.linalg.norm(res[k]["eef"][:n] - base, axis=1) * 1000
              for k in (("clean_b",) + PATCHED)}
    np.savez_compressed(OUT / "traj.npz",
                        **{f"{k}__eef": res[k]["eef"] for k in res},
                        **{f"{k}__grip": res[k]["grip"] for k in res},
                        **{f"{k}__success": res[k]["success"] for k in res},
                        **{f"{k}__t_success": (res[k]["t_success"] or -1) for k in res},
                        episode=args.episode, n_common=n)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK1, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
    fig, ax = plt.subplots(figsize=(9.6, 4.9), facecolor=SURFACE)
    ax.plot(np.arange(n) + NWAIT, curves["clean_b"], color=COLORS["clean_b"], lw=2.2,
            label="clean vs clean  -  sampling noise only")
    for k in PATCHED:
        d = next(c[2] for c in CONDS if c[0] == k)
        ax.plot(np.arange(n) + NWAIT, curves[k], color=COLORS[k], lw=2.4, label=f"patch at {d}")
    for k in ("clean_a",) + PATCHED:
        if res[k]["t_success"]:
            ax.axvline(res[k]["t_success"], color=COLORS.get(k, INK3), lw=1.1, alpha=0.55)
    ax.set_xlabel("environment step", color=INK2)
    ax.set_ylabel("end-effector distance from the clean run  (mm)", color=INK2)
    ax.grid(True, color="#e6e5df", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.legend(fontsize=10, labelcolor=INK2, frameon=False)
    ax.set_title("Does the deviation accumulate, or get corrected back?\n"
                 f"one episode (init state #{args.episode}); vertical lines mark task success",
                 fontsize=12.5, color=INK1, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "divergence.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    print("\n偏移(mm) 摘要:")
    for k, v in curves.items():
        print(f"  {k:11s} 峰值 {v.max():7.1f}  末值 {v[-1]:7.1f}  均值 {v.mean():7.1f}")
    print(f"[written] {OUT}/  (mp4 ×{len(res)}, traj.npz, divergence.png)")


if __name__ == "__main__":
    main()
