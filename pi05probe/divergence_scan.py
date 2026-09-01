#!/usr/bin/env python3
"""影响力 → 真实轨迹偏移:influence 到底有没有行为效力。

要回答的问题
----------
加密扫描把合法区的 influence 天花板从 126.4 mm 抬到 **308.4 mm(×2.44)**,
而且拿到最高分的 #1003 投影面积是加密集里**最小**的(872 px)。
所以现在能第一次同时问两件事:

  ① 偏移随 influence 单调增长吗 ⇒ influence 有没有行为效力
  ② 面积相同、influence 差 3 倍的一对位置,行为上分得开吗
     ⇒ 「面积不是决定性因素、位置才是」在行为层成不成立

⚠️ 上一轮(preset=legacy)什么也没测出来,原因**不是效应太小,是参照太脏**:
   websocket 的 infer 传不了噪声 ⇒ 每次重采 ε ⇒ 两条 clean 之间就有 28.3 mm 峰值偏移,
   四个条件 27.6–38.7 mm 全埋在里面。那是地板效应。
   本轮必须配 `serve_policy_fixed_noise.py`(ε 钉死)⇒ **噪声地板 = 0**,
   `clean` 与 `clean_repeat` 必须逐位相同,任何非零偏移都归因到贴纸。

设计
----
条件**全部合法**(贴纸与任何物体零重叠)—— 非法位置会遮住物体,不是可部署的攻击。
每个条件跑同一批官方 `.pruned_init`,同一个 env seed。

偏移定义:‖eef_cond(t) − eef_clean(t)‖,取两条轨迹的公共长度。
同时记**命令动作**的偏移(接 RESULTS.md §13 的 0.24 实现率口径:
influence 量的是命令量,末端实际只走 0.24 倍)。

用法:
    # 1) 先起固定 ε 的 server(py3.11,见 autorun_rollout.py 会自动做)
    # 2) ~/miniconda3/envs/openpi-libero/bin/python pi05probe/divergence_scan.py --episodes 10
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

# (标签, 位置, influence mm, 锚点号, 平均可见面积 px, 距 bowl/plate cm, 角色)
# influence / 面积 均来自 78 个合法点的合并候选池(旧 17 + 加密 61),
# 数值出处 report_fine_legal.py;面积 = s2*_scan_obs.npz 的 visible_px 沿帧平均。
FINE = [
    ("clean",          None,           0.0,   -1,    0, 0.0, "参照"),
    ("clean_repeat",   None,           0.0,   -1,    0, 0.0, "红线:必须与 clean 逐位相同"),
    ("inf_308_max",    (0.05, 0.12),  308.4, 1003,  872, 13.0, "全池最强,面积最小"),
    ("pair_hi_307",    (0.09, 0.12),  307.0, 1006,  978, 13.2, "面积配对·高"),
    ("pair_lo_102",    (0.09, 0.18),  101.8, 1055,  977, 19.1, "面积配对·低(同面积,1/3 influence)"),
    ("area_max_106",   (0.21, -0.04), 105.7,   16, 1369, 15.1, "面积最大的免费基线"),
    ("wristattn_197",  (0.05, 0.16),  196.9, 1031,  874, 16.9, "腕部 attention 首选"),
    ("old_best_126",   (-0.06, 0.22), 126.4,   25,  676, 23.2, "旧网格最佳,接续 §15.1"),
]

# 上一轮(RESULTS.md §15.1)的条件表,保留以便复现那张表。⚠️ 它是在**不定 ε** 的 server 上跑的。
LEGACY = [
    ("clean_a",        None,            0.0, -1, 0, 0.0, "参照"),
    ("clean_b",        None,            0.0, -1, 0, 0.0, "只有采样噪声的参照"),
    ("inf_125_rank1",  (-0.06, 0.22), 125.0, 25, 0, 0.0, "合法区第 1"),
    ("inf_106_rank2",  (0.21, -0.04), 106.0, 16, 0, 0.0, "合法区第 2"),
    ("attn_31",        (0.21, 0.22),   31.0, -1, 0, 0.0, "attention 最常选"),
    ("attn_19",        (0.12, 0.35),   19.0, -1, 0, 0.0, "attention 次常选"),
]
PRESETS = {"fine": FINE, "legacy": LEGACY}


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
    """跑一条 rollout。除末端位置外**同时记下每一步的命令动作** ——
    influence 量的是命令量,末端实际只走其中约 0.24(RESULTS.md §13),
    两个都记下来才能把 influence 和轨迹偏移放在同一个口径上比。"""
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init)
    plan = collections.deque()
    eef, act, ok, t_ok, t = [], [], False, None, 0
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
        a = plan.popleft()
        act.append(np.asarray(a, np.float64).copy())
        obs, _, _, _ = env.step(a.tolist())
        eef.append(obs["robot0_eef_pos"].copy())
        if not ok and all(env.env._eval_predicate(s) for s in goal):     # noqa: SLF001
            ok, t_ok = True, t
        t += 1
        if ok:
            break
    return np.array(eef), np.array(act), ok, t_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--texture", default=None)
    ap.add_argument("--preset", default="fine", choices=list(PRESETS))
    ap.add_argument("--conds", default=None, help="json 覆盖条件表(格式同 PRESETS 的一项)")
    ap.add_argument("--out", default=None, help="默认 out/divergence_scan_<preset>")
    ap.add_argument("--allow-noisy-clean", action="store_true",
                    help="跳过 clean/clean_repeat 逐位相同的红线(只在 legacy 复现时用)")
    args = ap.parse_args()
    CONDS = PRESETS[args.preset]
    if args.conds:
        CONDS = [tuple(r) for r in json.loads(pathlib.Path(args.conds).read_text())]
    stem = args.out or f"divergence_scan_{args.preset}"

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
    N = min(args.episodes, len(inits))
    print(f"[cfg] task={TASK}  每条件 {N} 个 episode  texture={pathlib.Path(tex).name}", flush=True)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    meta = client.get_server_metadata() or {}
    fixed_eps = "fixed_noise_seed" in meta
    print(f"[net] server metadata={meta}  固定 ε={'是' if fixed_eps else '否'}", flush=True)
    if not fixed_eps and not args.allow_noisy_clean:
        raise SystemExit("这个 server 没有钉死 ε,clean 之间会有 ~28 mm 采样噪声,"
                         "测不出东西。请用 serve_policy_fixed_noise.py,"
                         "或显式加 --allow-noisy-clean 复现 legacy。")

    ref_name = CONDS[0][0]                      # 第一条永远是 clean 参照
    traj = {}
    for name, pos, inf, anc, area, dist, desc in CONDS:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        if pos is not None:
            world = (pos[0], pos[1], cfg["plane"]["origin"][2] +
                     cfg["patch"]["thickness"] / 2 + cfg["patch"]["normal_offset"])
            env.env.set_xml_processor(sp.make_xml_processor(cfg, world, tex))
        env.seed(args.seed); env.reset()
        goal = env.env.parsed_problem["goal_state"]
        runs = []
        for ep in range(N):
            e, a, ok, t_ok = run(env, client, prompt, goal, inits[ep], args.seed + ep)
            runs.append(dict(eef=e, act=a, ok=ok, t=t_ok))
            print(f"  [{name:16s}] ep {ep+1}/{N}  success={ok}  steps={len(e)}", flush=True)
        env.close()
        traj[name] = runs

    # ---- 偏移:每个 episode 内和参照 clean 比,取公共长度 ----
    def diverge(key, ep, field):
        a, b = traj[ref_name][ep][field], traj[key][ep][field]
        n = min(len(a), len(b))
        # eef 是米 → mm;命令动作前 3 维也是位移量,用与 influence 相同的 ×50 口径
        s = 1000.0 if field == "eef" else 0.05 * 1000
        return np.linalg.norm(b[:n, :3] - a[:n, :3], axis=1) * s

    lines = ["=" * 104,
             f"影响力 → 真实轨迹偏移   task={TASK}   每条件 {N} 个 episode   preset={args.preset}",
             f"  server 固定 ε = {fixed_eps}   ⇒ 采样噪声地板 {'= 0' if fixed_eps else '未消除'}",
             f"  偏移 = ‖eef_cond(t) − eef_{ref_name}(t)‖,取公共长度",
             "=" * 104,
             "  条件               influence  面积px  距离  成功率 |  末端偏移 mm(峰/均/末)  |  命令偏移 mm(峰/均)"]
    summary = {}
    for name, pos, inf, anc, area, dist, desc in CONDS:
        if name == ref_name:
            continue
        pk = [diverge(name, ep, "eef").max() for ep in range(N)]
        mn = [diverge(name, ep, "eef").mean() for ep in range(N)]
        fin = [diverge(name, ep, "eef")[-1] for ep in range(N)]
        apk = [diverge(name, ep, "act").max() for ep in range(N)]
        amn = [diverge(name, ep, "act").mean() for ep in range(N)]
        succ = float(np.mean([r["ok"] for r in traj[name]]))
        summary[name] = dict(influence=inf, anchor=anc, area_px=area, dist_cm=dist, desc=desc,
                             success=succ, peak=float(np.mean(pk)), mean=float(np.mean(mn)),
                             final=float(np.mean(fin)), act_peak=float(np.mean(apk)),
                             act_mean=float(np.mean(amn)), peak_all=[float(x) for x in pk],
                             peak_sd=float(np.std(pk)))
        lines.append(f"  {name:18s} {inf:7.1f}mm {area:6d} {dist:5.1f}cm {succ:5.0%} |"
                     f" {np.mean(pk):8.1f} {np.mean(mn):7.1f} {np.mean(fin):7.1f}"
                     f"  ±{np.std(pk):5.1f} | {np.mean(apk):8.1f} {np.mean(amn):7.1f}")

    # ---- 红线:ε 钉死后,什么都不改的重复必须逐位相同 ----
    rep = next((c[0] for c in CONDS if c[0] == "clean_repeat"), None)
    if rep is not None:
        r = summary[rep]["peak"]
        lines.append("")
        lines.append(f"  [红线] clean 重复跑的偏移峰值 = {r:.6f} mm")
        if fixed_eps and not args.allow_noisy_clean:
            assert r == 0.0, (f"ε 已钉死但两条 clean 差 {r:.4f} mm —— 还有别的随机源(env seed?"
                              "渲染?),查清前不要用本轮结果")
            lines.append("         ⇒ 噪声地板 = 0,下面任何非零偏移都归因到贴纸。")

    # ---- 命题①:偏移随 influence 走吗 ----
    pat = [c for c in CONDS if c[1] is not None]
    infs = [summary[c[0]]["influence"] for c in pat]
    pks = [summary[c[0]]["peak"] for c in pat]
    lines.append("")
    if len(set(infs)) > 2:
        rk = lambda x: np.argsort(np.argsort(x)).astype(float)      # noqa: E731
        c_s = float(np.corrcoef(rk(infs), rk(pks))[0, 1])
        c_p = float(np.corrcoef(infs, pks)[0, 1])
        lines.append(f"  ① influence ↔ 末端偏移峰值:秩相关 {c_s:+.3f},皮尔逊 {c_p:+.3f}"
                     f"   ({len(pat)} 个贴纸条件)")

    # ---- 命题②:面积配对 —— 面积一样、influence 差 3 倍,行为分得开吗 ----
    if "pair_hi_307" in summary and "pair_lo_102" in summary:
        hi, lo = summary["pair_hi_307"], summary["pair_lo_102"]
        lines.append("")
        lines.append("  ② 面积配对(这是「面积不决定行为」的直接证据):")
        lines.append(f"     pair_hi_307  面积 {hi['area_px']} px  influence {hi['influence']:.1f} mm"
                     f"  → 偏移峰值 {hi['peak']:.1f} ± {hi['peak_sd']:.1f} mm")
        lines.append(f"     pair_lo_102  面积 {lo['area_px']} px  influence {lo['influence']:.1f} mm"
                     f"  → 偏移峰值 {lo['peak']:.1f} ± {lo['peak_sd']:.1f} mm")
        lines.append(f"     面积几乎相同(978 vs 977 px),influence 差 {hi['influence']/lo['influence']:.2f}×,"
                     f"偏移差 {hi['peak']/max(lo['peak'], 1e-9):.2f}×")
    if "inf_308_max" in summary and "area_max_106" in summary:
        a, b = summary["inf_308_max"], summary["area_max_106"]
        lines.append(f"     最强位置面积 {a['area_px']} px(最小)偏移 {a['peak']:.1f} mm  vs  "
                     f"面积最大 {b['area_px']} px 偏移 {b['peak']:.1f} mm")

    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    (OUT / f"{stem}.txt").write_text(txt + "\n")
    (OUT / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        OUT / f"{stem}.npz",
        conds=np.array([c[0] for c in CONDS]),
        **{f"{k}__ep{i}__eef": traj[k][i]["eef"] for k in traj for i in range(N)},
        **{f"{k}__ep{i}__act": traj[k][i]["act"] for k in traj for i in range(N)})
    print(f"\n[written] {OUT/(stem+'.txt')} + .json + .npz", flush=True)


if __name__ == "__main__":
    main()
