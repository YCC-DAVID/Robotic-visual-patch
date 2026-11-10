#!/usr/bin/env python3
"""DRIFT 复现 · 阶段 A(py3.8,openpi-libero):准备通用观测池 + 逐帧遮挡 mask + 尺寸标定。

对应 HANDOFF.md 第 4 节。做三件事:

1. `--calibrate`:在目标格上依次渲染多个物理尺寸,量各自在 224 base 图上的**足迹像素数**,
   用来挑出 ≈ DRIFT 的 4096 px(64×64)有效攻击面。纯渲染,不用模型,几十秒。
2. 观测池:连 websocket server 跑 N 条**干净** episode(不同 init_state),每 REPLAN 执行步取一帧,
   记 sim 状态 + clean img224/wrist224/state8。论文用 696 帧 ⇒ 默认 10 集 × ~70 帧。
3. 逐帧遮挡 mask:贴纸与相机都静止 ⇒ 图像足迹四边形固定,只有机械臂遮挡在变。
   对每个记下的 sim 状态,用 clean env 与 patched env **同法 regenerate 渲染**求差,得逐帧 mask。
   (两边都用 regenerate 才能只隔离出贴纸;这是 fw_patch_train 已验证的做法。)

产物 out/drift_pool.npz:clean_img224/clean_wrist224/clean_state8/masks/munion/quad224/cell/patch_m。

用法:
    # 0) 先起 server(py3.11):python pi05probe/run_demo.py --torch --server-only --port 8123
    # 1) 尺寸标定(不连 server)
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_pool_prep.py --calibrate
    # 2) 建池(用标定选出的 --patch-m)
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_pool_prep.py \
        --patch-m 0.25 --episodes 10 --port 8123
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
Z_TABLE = 0.9
CAL_SIZES = (0.13, 0.18, 0.22, 0.26, 0.30)
DRIFT_TARGET_PX = 64 * 64      # π0.5 在论文里的 64×64 = 4096 px


def quat2axisangle(quat):
    q = np.array(quat, dtype=np.float64)
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def model_input(obs):
    """与 attack_rollout.py 逐字一致:180° 翻转 + resize_with_pad 到 224。"""
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wri = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return (image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE)),
            image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, RESIZE, RESIZE)))


def load_cell(args):
    """确定目标格:默认从 per-cell influence 的合法 argmax 取(不照抄硬编码数字)。"""
    if args.cell:
        return tuple(float(v) for v in args.cell.split(",")), "手动指定"
    f = OUT / "percell_scores_plate.npz"
    assert f.is_file(), f"缺 {f};用 --cell x,y 手动指定"
    d = np.load(f, allow_pickle=True)
    infl, legal, aw = d["influence"], d["legal"].astype(bool), d["anchor_world"]
    legi = np.where(legal)[0]
    am = int(legi[int(np.argmax(infl[legi]))])
    return (float(aw[am][0]), float(aw[am][1])), f"percell influence 合法 argmax(idx {am})"


def project_quad(cell_xy, patch_m):
    """patch 4 个世界角 → 224 base 图像坐标(row,col),顺序 BL,BR,TR,TL。
    复用 π0.5 已校准的相机 T 与朝向 code(pi05_cell_world.npz)。"""
    cw = np.load(OUT / "grad" / "pi05_cell_world.npz", allow_pickle=True)
    T, code = cw["T"], tuple(int(v) for v in cw["code"])
    cx, cy = cell_xy
    hh = patch_m / 2.0
    q = []
    for x, y in [(cx - hh, cy - hh), (cx + hh, cy - hh), (cx + hh, cy + hh), (cx - hh, cy + hh)]:
        ph = T @ np.array([x, y, Z_TABLE, 1.0])
        r, c = ph[1] / ph[2], ph[0] / ph[2]
        fr, fc, sw = code
        if fr:
            r = (RES - 1) - r
        if fc:
            c = (RES - 1) - c
        if sw:
            r, c = c, r
        q.append([r * RESIZE / RES, c * RESIZE / RES])
    return np.array(q, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="只做尺寸标定,不建池")
    ap.add_argument("--patch-m", type=float, default=0.25)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--replan", type=int, default=REPLAN)
    ap.add_argument("--cell", default=None, help="世界 x,y;默认自动取 influence 合法 argmax")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--thresh", type=int, default=12, help="mask 阈值(uint8 RGB 差)")
    ap.add_argument("--out", default="drift_pool")
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    import torch

    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    plane = sp.Plane.from_cfg(cfg["plane"])
    cell_xy, how = load_cell(args)
    print(f"[cell] 目标格 world=({cell_xy[0]:+.4f},{cell_xy[1]:+.4f})  来源:{how}", flush=True)

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid = next(i for i in range(suite.n_tasks)
               if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    prompt = str(task.language)
    tex_probe = str(PROBE / "config" / "probe_texture.png")

    def lift_for(pm):
        c = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
        c["patch"]["size_wh"] = [pm, pm]
        return c, c["patch"]["thickness"] / 2.0 + c["patch"]["normal_offset"]

    # ---------------- 尺寸标定 ----------------
    if args.calibrate:
        cenv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        cenv.seed(args.seed); cenv.reset()
        inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                        "libero_goal" / f"{TASK}.pruned_init")))
        obs = cenv.set_init_state(inits[0])
        for _ in range(NWAIT):
            obs, _, _, _ = cenv.step(DUMMY)
        st0 = cenv.get_sim_state().copy()
        clean_img, _ = model_input(cenv.regenerate_obs_from_state(st0))
        cenv.close()
        print(f"\n[标定] 目标 ≈ DRIFT 的 {DRIFT_TARGET_PX} px(64×64,占 224² 的 8.2%)")
        print(f"{'patch_m':>9s} {'足迹px':>8s} {'占224²':>8s} {'vs 4096':>9s}")
        rows = []
        for pm in CAL_SIZES:
            c, lift = lift_for(pm)
            penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
            penv.env.set_xml_processor(
                sp.make_xml_processor(c, plane.to_world(cell_xy[0], cell_xy[1], lift), tex_probe))
            penv.seed(args.seed); penv.reset()
            pimg, _ = model_input(penv.regenerate_obs_from_state(st0))
            penv.close()
            px = int((np.abs(pimg.astype(np.int16) - clean_img.astype(np.int16)).max(-1) > args.thresh).sum())
            rows.append((pm, px))
            print(f"{pm:9.2f} {px:8d} {px/RESIZE**2*100:7.2f}% {px/DRIFT_TARGET_PX:8.2f}×")
        best = min(rows, key=lambda r: abs(r[1] - DRIFT_TARGET_PX))
        print(f"\n[建议] --patch-m {best[0]:.2f}(足迹 {best[1]} px,最接近 4096)")
        print("[注意] 报告里要写明:物理尺寸 X m ≡ 有效像素面积 Y px ≈ DRIFT 的 4096 px")
        return 0

    # ---------------- 建观测池 ----------------
    cfg_p, lift = lift_for(args.patch_m)
    cell_world = plane.to_world(cell_xy[0], cell_xy[1], lift)
    quad = project_quad(cell_xy, args.patch_m)

    from openpi_client import websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"[net] 连上 server {args.host}:{args.port}(首次 infer 可能等 torch.compile)", flush=True)

    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                    "libero_goal" / f"{TASK}.pruned_init")))
    N = min(args.episodes, len(inits))
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    env.seed(args.seed); env.reset()
    goal = env.env.parsed_problem["goal_state"]

    states, imgs, wris, sts, ep_id = [], [], [], [], []
    for ep in range(N):
        env.seed(args.seed + ep); env.reset()
        obs = env.set_init_state(inits[ep])
        plan = collections.deque()
        ok, t = False, 0
        while t < MAX_STEPS + NWAIT:
            if t < NWAIT:
                obs, _, _, _ = env.step(DUMMY); t += 1; continue
            img, wri = model_input(obs)
            st = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                 obs["robot0_gripper_qpos"]])
            if (t - NWAIT) % args.replan == 0:
                states.append(env.get_sim_state().copy())
                imgs.append(img); wris.append(wri); sts.append(st.astype(np.float32)); ep_id.append(ep)
            if not plan:
                chunk = client.infer({"observation/image": img, "observation/wrist_image": wri,
                                      "observation/state": st, "prompt": prompt})["actions"]
                plan.extend(chunk[:args.replan])
            obs, _, _, _ = env.step(plan.popleft().tolist())
            if not ok and all(env.env._eval_predicate(s) for s in goal):  # noqa: SLF001
                ok = True
                break
            t += 1
        print(f"  [ep {ep+1:2d}/{N}] success={ok}  池累计 {len(states)} 帧", flush=True)
    env.close()
    F = len(states)
    print(f"[pool] 共 {F} 帧(论文 696;{N} 集 × 每 {args.replan} 步一帧)", flush=True)

    # ---------------- 逐帧遮挡 mask(clean/patched 同法 regenerate) ----------------
    cenv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    cenv.seed(args.seed); cenv.reset()
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    penv.env.set_xml_processor(sp.make_xml_processor(cfg_p, cell_world, tex_probe))
    penv.seed(args.seed); penv.reset()
    clean224, masks = [], []
    for k in range(F):
        c_img, _ = model_input(cenv.regenerate_obs_from_state(states[k]))
        p_img, _ = model_input(penv.regenerate_obs_from_state(states[k]))
        clean224.append(c_img)
        masks.append(np.abs(p_img.astype(np.int16) - c_img.astype(np.int16)).max(-1) > args.thresh)
        if k % 100 == 0:
            print(f"  [mask] {k}/{F} px={int(masks[-1].sum())}", flush=True)
    cenv.close(); penv.close()
    masks = np.array(masks, bool)
    munion = masks.any(0)
    px = masks.reshape(F, -1).sum(1)
    print(f"[mask] 足迹 union={int(munion.sum())} px  逐帧 min/中位/max = "
          f"{px.min()}/{int(np.median(px))}/{px.max()}  全遮挡帧数={int((px==0).sum())}/{F}", flush=True)
    ys, xs = np.where(munion)
    print(f"[align] mask 质心=({ys.mean():.0f},{xs.mean():.0f})  "
          f"quad 质心=({quad[:,0].mean():.0f},{quad[:,1].mean():.0f})", flush=True)

    outp = OUT / f"{args.out}.npz"
    np.savez_compressed(
        outp, clean_img224=np.array(clean224, np.uint8),
        clean_wrist224=np.array(wris, np.uint8), clean_state8=np.array(sts, np.float32),
        masks=masks, munion=munion, quad224=quad, cell_xy=np.array(cell_xy),
        cell_world=cell_world, patch_m=args.patch_m, prompt=prompt,
        ep_id=np.array(ep_id), episodes=N, replan=args.replan, seed=args.seed)
    print(f"[written] {outp}  ({outp.stat().st_size/2**20:.0f} MiB)  F={F}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
