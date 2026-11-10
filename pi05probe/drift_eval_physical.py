#!/usr/bin/env python3
"""DRIFT 复现 · 阶段 E:评测 B —— **物理纹理**闭环 rollout(对应 HANDOFF 4.6 的评测 B)。

与评测 A 的唯一区别就是图案怎么进画面:
  评测 A(drift_eval_digital.py):把 P **直接替换**进 224 模型输入的足迹像素。
      ↑ 这也是**论文自己的协议**(§4.1 "the overlay operator A replaces the pixels in R with δ";
        §5.2.2 "the optimized patch is pasted onto the wrist image")⇒ A 与论文可比。
  评测 B(本脚本):P 反 warp 成方形纹理 → `scene_patch` 注入成**真几何** →
      场景光照/阴影 → 透视重采样 → 256→224 抗锯齿降采样 → 才进模型。

**A − B 的差值 = 物理可实现性缺口**,这是本项目一直悬着未量化的量,也是论文没做的部分
(论文只在威胁模型层面主张 "physically realizable",没有渲染验证)。
⚠️ 本项目 §2.3 第 3 条有前科:上一轮优化图案在数字合成上 ‖Δa‖ ×3~6,过了这条链后闭环零效果。

除成功率外还量**像素级缺口**(这是判断"为什么掉"的关键):
  · 渲染出来的足迹像素 vs 设计图案 P 的 mean|Δ| 与相关系数
  · **总变差(TV)保留率** —— sign-PGD 图案是逐像素饱和的高频花纹,
    而透视+降采样本质是低通滤波;TV 掉得多 ⇒ 高频被滤掉,这就是缺口的机制。

⚠️ 0.26m 贴纸与 plate/bowl/cream_cheese 的 keep-out 盒重叠,注入真几何会把物体顶起 1.5mm
⇒ **必须有同尺寸随机纹理对照**,否则分不清是攻击还是几何干扰(见 HANDOFF 4.4)。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_eval_physical.py \
        --patch drift_patch_drift --episodes 50 --conds clean,rand,drift
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
CONDS = ("clean", "rand", "drift")


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


def tv(x):
    """总变差(逐通道均值):高频内容的量度。x float [H,W,3] in [0,1]。"""
    return float(np.abs(np.diff(x, axis=0)).mean() + np.abs(np.diff(x, axis=1)).mean())


def run_episode(env, client, prompt, goal_state, init_state, seed):
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    plan = collections.deque()
    ok, t_ok = False, None
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
        if not ok and all(env.env._eval_predicate(s) for s in goal_state):  # noqa: SLF001
            ok, t_ok = True, t
        t += 1
    return ok, t_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="drift_patch_drift")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--conds", default="clean,rand,drift")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--thresh", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    conds = [c for c in args.conds.split(",") if c]
    assert all(c in CONDS for c in conds), f"--conds 只能取 {CONDS}"
    tag = args.out or f"drift_evalB_{args.patch.replace('drift_patch_', '')}"

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    import scene_patch as sp
    import torch

    d = np.load(OUT / f"{args.patch}.npz", allow_pickle=True)
    P = d["P"]                                          # [3,224,224] in [0,1]
    patch_m = float(d["patch_m"])
    cell_xy = (float(d["cell_xy"][0]), float(d["cell_xy"][1]))
    ptag = args.patch.replace("drift_patch_", "")
    tex_drift = PROBE / "config" / f"drift_texture_{ptag}.png"
    tex_rand = PROBE / "config" / "drift_texture_rand.png"
    for f in (tex_drift, tex_rand):
        assert f.is_file(), f"缺纹理 {f} —— 先跑 drift_texture.py --patch {args.patch}"
    print(f"[patch] {args.patch}  patch_m={patch_m:.2f}m  cell=({cell_xy[0]:+.4f},{cell_xy[1]:+.4f})",
          flush=True)
    print(f"[tex] drift={tex_drift.name}  rand={tex_rand.name}", flush=True)

    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    cfg["patch"]["size_wh"] = [patch_m, patch_m]
    plane = sp.Plane.from_cfg(cfg["plane"])
    lift = cfg["patch"]["thickness"] / 2.0 + cfg["patch"]["normal_offset"]
    cell_world = plane.to_world(cell_xy[0], cell_xy[1], lift)

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid = next(i for i in range(suite.n_tasks)
               if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    prompt = str(task.language)
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                    "libero_goal" / f"{TASK}.pruned_init")))
    N = min(args.episodes, len(inits))
    print(f"[cfg] task={TASK}  用前 {N} 个 init_states  conds={conds}  seed={args.seed}", flush=True)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("[net] 连上 server", flush=True)

    # ---------- 像素级可实现性缺口诊断 ----------
    print("\n[缺口诊断] 渲染出来的足迹像素 vs 设计图案 P", flush=True)
    cenv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    cenv.seed(args.seed); cenv.reset()
    cenv.set_init_state(inits[0])
    for _ in range(NWAIT):
        cenv.step(DUMMY)
    st0 = cenv.get_sim_state().copy()
    c_img, _ = model_input(cenv.regenerate_obs_from_state(st0))
    cenv.close()
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    penv.env.set_xml_processor(sp.make_xml_processor(cfg, cell_world, str(tex_drift)))
    penv.seed(args.seed); penv.reset()
    p_img, _ = model_input(penv.regenerate_obs_from_state(st0))
    penv.close()
    m = np.abs(p_img.astype(np.int16) - c_img.astype(np.int16)).max(-1) > args.thresh
    Pd = P.transpose(1, 2, 0)                                     # [224,224,3] in [0,1]
    got = p_img.astype(np.float32) / 255.0
    if m.sum() > 0:
        a, b = Pd[m], got[m]
        mad = float(np.abs(a - b).mean())
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        ys, xs = np.where(m)
        r0, r1, c0, c1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tv_want, tv_got = tv(Pd[r0:r1, c0:c1]), tv(got[r0:r1, c0:c1])
        gap = dict(footprint_px=int(m.sum()), mean_abs_diff=mad, corr=corr,
                   tv_intended=tv_want, tv_rendered=tv_got,
                   tv_retained=float(tv_got / max(tv_want, 1e-9)),
                   mean_intended=float(a.mean()), mean_rendered=float(b.mean()))
        print(f"  足迹 {int(m.sum())}px  mean|Δ|={mad:.4f}(像素值 0~1 尺度)  相关系数={corr:+.3f}",
              flush=True)
        print(f"  亮度均值:设计 {a.mean():.3f} → 渲染 {b.mean():.3f}", flush=True)
        print(f"  **总变差 TV:设计 {tv_want:.4f} → 渲染 {tv_got:.4f}  保留 {tv_got/max(tv_want,1e-9):.1%}**",
              flush=True)
        print(f"  (TV 保留率低 ⇒ sign-PGD 的高频花纹被透视+降采样低通滤掉,这就是缺口的机制)",
              flush=True)
    else:
        gap = {}
        print("  [警告] 渲染差分为空 —— 纹理注入可能没生效", flush=True)

    # ---------- 闭环 ----------
    results = {}
    for name in conds:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        if name != "clean":
            tex = tex_drift if name == "drift" else tex_rand
            env.env.set_xml_processor(sp.make_xml_processor(cfg, cell_world, str(tex)))
        env.seed(args.seed); env.reset()
        goal_state = env.env.parsed_problem["goal_state"]
        succ, steps = [], []
        for ep in range(N):
            ok, t_ok = run_episode(env, client, prompt, goal_state, inits[ep], args.seed + ep)
            succ.append(ok)
            if ok:
                steps.append(t_ok)
            print(f"  [{name:6s}] ep {ep+1:2d}/{N}  success={ok}"
                  f"{'' if not ok else f'  t={t_ok}'}", flush=True)
        env.close()
        rate = float(np.mean(succ))
        results[name] = dict(rate=rate, n=N, successes=int(np.sum(succ)),
                             mean_steps=(float(np.mean(steps)) if steps else None),
                             per_ep=[bool(x) for x in succ])
        print(f"  ==> {name:6s} 成功率 {np.sum(succ)}/{N} = {rate:.1%}\n", flush=True)

    # ---------- 与评测 A 对比 ----------
    af = OUT / f"drift_evalA_{ptag}.json"
    if ptag == "drift":
        af = OUT / "drift_evalA_drift.json"
    A = json.load(open(af))["results"] if af.is_file() else {}

    lines = ["=" * 96,
             f"评测 B · 物理纹理闭环   task={TASK}   每条件 {N} 集(同一批 init_states,seed={args.seed})",
             f"  贴纸 = {args.patch}   patch_m={patch_m:.2f}m   格 world=({cell_xy[0]:+.4f},{cell_xy[1]:+.4f})",
             f"  纹理经过:反 warp → scene_patch 真几何注入 → 光照/阴影 → 透视重采样 → 256→224 降采样",
             "=" * 96,
             "  条件     评测B(物理)      评测A(数字)     A−B 缺口"]
    for name in conds:
        r = results[name]
        av = A.get(name, {}).get("rate")
        astr = f"{av:6.1%}" if av is not None else "   --  "
        gstr = f"{(av - r['rate'])*100:+6.1f} 点" if av is not None else "   --"
        lines.append(f"  {name:7s}  {r['successes']:2d}/{r['n']} = {r['rate']:6.1%}   {astr}"
                     f"      {gstr}")
    if gap:
        lines.append("")
        lines.append("  像素级可实现性缺口(设计图案 P vs 实际渲染出来的足迹像素):")
        lines.append(f"    足迹 {gap['footprint_px']}px   mean|Δ| = {gap['mean_abs_diff']:.4f}"
                     f"   相关系数 = {gap['corr']:+.3f}")
        lines.append(f"    亮度均值:设计 {gap['mean_intended']:.3f} → 渲染 {gap['mean_rendered']:.3f}")
        lines.append(f"    总变差 TV:设计 {gap['tv_intended']:.4f} → 渲染 {gap['tv_rendered']:.4f}"
                     f"   **保留 {gap['tv_retained']:.1%}**")
        lines.append("    TV 保留率低 ⇒ sign-PGD 的高频花纹被透视+降采样低通滤掉 = 缺口的机制。")
    lines.append("")
    lines.append("  判读:")
    lines.append("    B ≈ A ⇒ 物理链几乎无损,攻击真正可实现(超出论文的结论,论文只做了数字粘贴)。")
    lines.append("    B ≫ A(成功率回升)⇒ 存在可实现性缺口;结合 TV 保留率判断是否为高频被滤。")
    lines.append("    rand 也掉 ⇒ 是几何干扰(1.5mm 抬升顶起物体),不是攻击 —— 见 HANDOFF 4.4。")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    (OUT / f"{tag}.txt").write_text(txt + "\n")
    (OUT / f"{tag}.json").write_text(json.dumps(
        dict(results=results, evalA=A, gap=gap, patch=args.patch, patch_m=patch_m,
             cell_xy=list(cell_xy), episodes=N, seed=args.seed), indent=2))
    print(f"\n[written] {OUT/(tag+'.txt')} + .json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
