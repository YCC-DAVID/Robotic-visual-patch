#!/usr/bin/env python3
"""DRIFT 复现 · 阶段 C(py3.8,openpi-libero):评测 A —— **数字合成**闭环 rollout。

对应 HANDOFF.md 第 4.6 节的"评测 A"。它负责**归因**:把"L₀ loss 有不有效"与
"位置/物理可实现链有没有把效果吃掉"分开。

做法(与 drift_patch_train.py 的合成口径**逐字一致**,这是本脚本存在的意义):
  训练时  comp = clean224 * (1 − mask_k) + P * mask_k        (P 在 224 图像空间,[0,1])
  评测时  完全同一式子,只是 clean224 与 mask_k 来自**闭环当帧**而不是观测池。
⇒ 不做反 warp、不进 MuJoCo 贴图渲染。打不穿就不是"物理链损耗"的问题,直接归到位置/loss。

逐帧遮挡 mask 怎么来:贴纸与相机都静止 ⇒ 图像足迹四边形固定,只有机械臂/物体遮挡在变。
所以每个 inference 步用两个**只用来渲染**的旁路 env(一个干净、一个带探针贴纸几何),
按当前 sim state `regenerate_obs_from_state` 各渲一次求差 —— 与 drift_pool_prep.py 同法。
主 env 不受影响(旁路 env 是独立实例)。

三个条件(HANDOFF 4.6 条件清单里属于评测 A 的部分):
  clean  : 不合成,基线
  rand   : 同格同尺寸的**随机**图案 —— 分离"大贴纸本身"的贡献(4.4 ②),不能省
  drift  : DRIFT-L₀ 优化出来的图案 —— loss 有效性

用法:
    # 0) 先起 server(py3.11):python pi05probe/run_demo.py --torch --server-only --port 8123
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_eval_digital.py \
        --patch drift_patch_drift --episodes 50 --conds clean,rand,drift
"""
import argparse
import collections
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path(os.environ.get("WAMATTACK_PROJECT_ROOT", "/home/user1/workspace/chence/WAMattack"))
OPENPI = ROOT / "third_party" / "openpi"
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"

for p in reversed([OPENPI / "packages" / "openpi-client" / "src", OPENPI / "third_party" / "libero"]):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(PROBE))
os.environ["LIBERO_CONFIG_PATH"] = os.environ.get("WAMATTACK_LIBERO_CONFIG_PATH", str(PROBE / "libero_config"))
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYTHONNOUSERSITE"] = "1"

import numpy as np  # noqa: E402
import yaml  # noqa: E402

RES, RESIZE, REPLAN, NWAIT = 256, 224, 5, 10
MAX_STEPS = 300
DUMMY = [0.0] * 6 + [-1.0]
TASK = "put_the_bowl_on_the_plate"
SEED_RAND = 20260902          # 随机对照图案的固定 seed(可复现)
CONDS = ("clean", "rand", "drift")


def quat2axisangle(quat):
    q = np.array(quat, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def model_input(obs):
    """与 attack_rollout.py / drift_pool_prep.py 逐字一致:180° 翻转 + resize_with_pad 到 224。"""
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wri = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return (image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE)),
            image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, RESIZE, RESIZE)))


class MaskRenderer:
    """两个旁路 env(clean / patched),按 sim state 渲染求差 ⇒ 当帧遮挡感知足迹 mask。"""

    def __init__(self, bddl, cfg_patched, cell_world, tex_probe, seed, thresh):
        from libero.libero.envs import OffScreenRenderEnv
        import scene_patch as sp
        self.thresh = thresh
        self.cenv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        self.cenv.seed(seed); self.cenv.reset()
        self.penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        self.penv.env.set_xml_processor(sp.make_xml_processor(cfg_patched, cell_world, tex_probe))
        self.penv.seed(seed); self.penv.reset()

    def mask_for(self, sim_state):
        c_img, _ = model_input(self.cenv.regenerate_obs_from_state(sim_state))
        p_img, _ = model_input(self.penv.regenerate_obs_from_state(sim_state))
        d = np.abs(p_img.astype(np.int16) - c_img.astype(np.int16)).max(-1)
        return d > self.thresh

    def close(self):
        self.cenv.close(); self.penv.close()


class WristRenderer:
    """--wrist patched:旁路 env 贴**真实图案纹理**,按 sim state 渲 wrist 图。
    动机:评测 B 里贴纸在场景中,wrist 相机也拍得到它;而训练池与评测 A 的 wrist 都是干净的。
    本消融只改 wrist 这一路,base 仍按评测 A 数字合成 ⇒ 单独量"wrist 看到贴纸"的影响。"""

    def __init__(self, bddl, cfg_patched, cell_world, tex, seed):
        from libero.libero.envs import OffScreenRenderEnv
        import scene_patch as sp
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        self.env.env.set_xml_processor(sp.make_xml_processor(cfg_patched, cell_world, str(tex)))
        self.env.seed(seed); self.env.reset()

    def wrist_for(self, sim_state):
        _, wri = model_input(self.env.regenerate_obs_from_state(sim_state))
        return wri

    def close(self):
        self.env.close()


def run_episode(env, client, prompt, goal_state, init_state, seed, patch_u8, mr, wr=None):
    """跑一条完整 rollout。patch_u8=None ⇒ clean(不合成、也不渲 mask)。
    wr 非 None ⇒ wrist 图换成带真实贴纸纹理的旁路渲染(--wrist patched)。

    返回 (success, t_ok, 逐帧 mask 像素数, mask 落在 munion 外的像素数)。
    """
    env.seed(seed)                 # ⚠️ 每次 reset 前重 seed(fixture 不在 qpos 里)
    env.reset()
    obs = env.set_init_state(init_state)
    plan = collections.deque()
    ok, t_ok, mpx = False, None, []
    t = 0
    while t < MAX_STEPS + NWAIT:
        if t < NWAIT:
            obs, _, _, _ = env.step(DUMMY)
            t += 1
            continue
        if not plan:
            img, wri = model_input(obs)
            if patch_u8 is not None:
                sim_state = env.get_sim_state().copy()
                m = mr.mask_for(sim_state)
                mpx.append(int(m.sum()))
                img = np.where(m[..., None], patch_u8, img).astype(np.uint8)
                if wr is not None:
                    wri = wr.wrist_for(sim_state)
            st = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                 obs["robot0_gripper_qpos"]])
            chunk = client.infer({"observation/image": img, "observation/wrist_image": wri,
                                  "observation/state": st, "prompt": prompt})["actions"]
            plan.extend(chunk[:REPLAN])
        obs, _, _, _ = env.step(plan.popleft().tolist())
        if not ok and all(env.env._eval_predicate(s) for s in goal_state):  # noqa: SLF001
            ok, t_ok = True, t
        t += 1
    return ok, t_ok, mpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="drift_patch_drift", help="drift_patch_train.py 的产物名")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--conds", default="clean,rand,drift")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--thresh", type=int, default=12, help="mask 阈值,必须与 pool_prep 一致")
    ap.add_argument("--wrist", default="clean", choices=["clean", "patched"],
                    help="clean=原评测 A;patched=wrist 换成带真实贴纸纹理的渲染(评测 B 的 wrist 条件)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    conds = [c for c in args.conds.split(",") if c]
    assert all(c in CONDS for c in conds), f"--conds 只能取 {CONDS}"
    tag = args.out or (f"drift_evalA_{args.patch.replace('drift_patch_', '')}"
                       + ("_wristP" if args.wrist == "patched" else ""))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    import scene_patch as sp
    import torch

    d = np.load(OUT / f"{args.patch}.npz", allow_pickle=True)
    P = d["P"]                                     # [3,224,224] in [0,1]
    munion = d["munion"]
    patch_m = float(d["patch_m"])
    cell_xy = (float(d["cell_xy"][0]), float(d["cell_xy"][1]))
    P_u8 = np.clip(np.transpose(P, (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    rs = np.random.RandomState(SEED_RAND)
    R_u8 = rs.randint(0, 256, (RESIZE, RESIZE, 3), dtype=np.uint8)
    print(f"[patch] {args.patch}  patch_m={patch_m:.2f}m  cell=({cell_xy[0]:+.4f},{cell_xy[1]:+.4f})  "
          f"munion={int(munion.sum())}px  L₀ {float(d['L_init']):.3e}→{float(d['L_final']):.3e}",
          flush=True)

    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    cfg["patch"]["size_wh"] = [patch_m, patch_m]
    plane = sp.Plane.from_cfg(cfg["plane"])
    lift = cfg["patch"]["thickness"] / 2.0 + cfg["patch"]["normal_offset"]
    cell_world = plane.to_world(cell_xy[0], cell_xy[1], lift)
    tex_probe = str(PROBE / "config" / "probe_texture.png")

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid = next(i for i in range(suite.n_tasks)
               if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    prompt = str(task.language)

    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                    "libero_goal" / f"{TASK}.pruned_init")))
    N = min(args.episodes, len(inits))
    print(f"[cfg] task={TASK}  init_states={len(inits)} 用前 {N} 个  conds={conds}  seed={args.seed}",
          flush=True)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("[net] 连上 server(首次 infer 要等 torch.compile,可能几分钟)", flush=True)

    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    env.seed(args.seed); env.reset()
    goal_state = env.env.parsed_problem["goal_state"]
    mr = None
    if any(c != "clean" for c in conds):
        mr = MaskRenderer(bddl, cfg, cell_world, tex_probe, args.seed, args.thresh)
        print("[mask] 旁路渲染 env 就绪(clean + patched,逐帧求差)", flush=True)

    pats = {"clean": None, "rand": R_u8, "drift": P_u8}
    ptag = args.patch.replace("drift_patch_", "")
    wtex = {"rand": PROBE / "config" / "drift_texture_rand.png",
            "drift": PROBE / "config" / f"drift_texture_{ptag}.png"}
    if args.wrist == "patched":
        for c in conds:
            if c != "clean":
                assert wtex[c].is_file(), f"缺纹理 {wtex[c]} —— 先跑 drift_texture.py"
        print(f"[wrist] patched:wrist 用真实纹理渲染  {[wtex[c].name for c in conds if c != 'clean']}",
              flush=True)
    results = {}
    for name in conds:
        wr = None
        if args.wrist == "patched" and name != "clean":
            wr = WristRenderer(bddl, cfg, cell_world, wtex[name], args.seed)
        succ, steps, allpx = [], [], []
        for ep in range(N):
            ok, t_ok, mpx = run_episode(env, client, prompt, goal_state, inits[ep],
                                        args.seed + ep, pats[name], mr, wr)
            succ.append(ok); allpx += mpx
            if ok:
                steps.append(t_ok)
            print(f"  [{name:6s}] ep {ep+1:2d}/{N}  success={ok}"
                  f"{'' if not ok else f'  t={t_ok}'}"
                  f"{'' if not mpx else f'  mask 中位 {int(np.median(mpx))}px'}", flush=True)
        rate = float(np.mean(succ))
        results[name] = dict(rate=rate, n=N, successes=int(np.sum(succ)),
                             mean_steps=(float(np.mean(steps)) if steps else None),
                             mask_px_median=(int(np.median(allpx)) if allpx else None),
                             mask_px_min=(int(np.min(allpx)) if allpx else None),
                             mask_px_max=(int(np.max(allpx)) if allpx else None),
                             per_ep=[bool(x) for x in succ])
        print(f"  ==> {name:6s} 成功率 {np.sum(succ)}/{N} = {rate:.1%}\n", flush=True)
        if wr is not None:
            wr.close()
    env.close()
    if mr is not None:
        mr.close()

    lines = ["=" * 96,
             f"评测 A · 数字合成闭环   task={TASK}   每条件 {N} 集(同一批 init_states,seed={args.seed})",
             f"  wrist = {args.wrist}"
             + ("(带真实贴纸纹理渲染,= 评测 B 的 wrist 条件)" if args.wrist == "patched" else "(干净)"),
             f"  贴纸 = {args.patch}   物理尺寸 {patch_m:.2f} m ≡ 足迹 union {int(munion.sum())} px "
             f"({int(munion.sum())/RESIZE**2*100:.1f}% of 224²,DRIFT 的 64×64=4096px 对标)",
             f"  格 world=({cell_xy[0]:+.4f},{cell_xy[1]:+.4f})   L₀ 灰初始 {float(d['L_init']):.3e} "
             f"→ 优化后 {float(d['L_final']):.3e}   随机参照 {float(d['L_rand']):.3e}",
             "=" * 96,
             "  条件     成功率        平均成功步数   逐帧可见 mask px(中位/min/max)"]
    for name in conds:
        r = results[name]
        ms = "  -- " if r["mean_steps"] is None else f"{r['mean_steps']:5.0f}"
        mp = "  --" if r["mask_px_median"] is None else \
            f"{r['mask_px_median']}/{r['mask_px_min']}/{r['mask_px_max']}"
        lines.append(f"  {name:7s}  {r['successes']:2d}/{r['n']} = {r['rate']:5.1%}   {ms}"
                     f"          {mp}")
    base = results.get("clean", {}).get("rate")
    if base is not None:
        lines.append("")
        lines.append("  相对 clean 的下降:")
        for name in conds:
            if name == "clean":
                continue
            dd = base - results[name]["rate"]
            lines.append(f"    {name:7s} {dd:+.1%}  ({dd*N:+.0f} 集)")
        if "rand" in results and "drift" in results:
            gap = results["rand"]["rate"] - results["drift"]["rate"]
            lines.append("")
            lines.append(f"  DRIFT 相对同尺寸随机图案的净增益:{gap:+.1%}({gap*N:+.0f} 集)")
            lines.append("  判读(HANDOFF 4.6):drift 明显低于 rand 且低于 clean ⇒ L₀ 在这个位置有效,")
            lines.append("       再做评测 B 量物理可实现性缺口;drift ≈ rand ⇒ 是位置问题不是 loss,")
            lines.append("       与本项目 2.3 节第 2 条(可见面积 × 暴露时长)一致,可直接收结论。")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    (OUT / f"{tag}.txt").write_text(txt + "\n")
    (OUT / f"{tag}.json").write_text(json.dumps(
        dict(results=results, patch=args.patch, patch_m=patch_m, cell_xy=list(cell_xy), wrist=args.wrist,
             munion_px=int(munion.sum()), episodes=N, seed=args.seed, thresh=args.thresh,
             L_init=float(d["L_init"]), L_final=float(d["L_final"]), L_rand=float(d["L_rand"])),
        indent=2))
    print(f"\n[written] {OUT/(tag+'.txt')} + .json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
