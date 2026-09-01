#!/usr/bin/env python3
"""FastWAM 对抗 patch 训练(π0.5 patch_train.py 的跨模型对应件,wamattack env,py3.10)。

和 π0.5 侧同一套威胁模型/目标,只换模型:
  - 攻击格:FastWAM 7×7 上 gradient(=influence)argmax 的**合法**格
    idx31 rc[6,3] world≈(0.223,-0.003)(fw_percell_scores_plate.npz);patch 13cm(与 per-cell 一致)。
  - 可微前向复用 fw_grad.infer_action_grad(fastwam.py 逐字复刻、去 @no_grad,不改模型)。
  - 目标(方向性,适配 flow-matching;UADA 的最大幅度 dev 只作对照):
      away  = 动作净平移推离 destination(plate 中心);
      curve = 横向弯折(POAP 曲率代理);
      dev   = 最大化 |Δa|(离散头路子,对照)。
  - EOT 跨 clean rollout 抓的 F 帧、固定共享 ε(红线=0)。
  - patch 与相机静止 ⇒ 图像足迹逐帧不变,只有机械臂遮挡在变:**逐帧渲染遮挡感知 mask**,
    共享一张 P,munion 做梯度掩码(比 π0.5 的常数足迹更忠实,已注明)。

产物:out/fw_patch_trained_{loss}.npz、config/fw_texture_adv_{loss}.png、out/fw_patch_preview_{loss}.png。
之后 fw_attack_rollout.py 注入物理纹理做闭环验收。

用法:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1 \
      /home/user1/miniconda3/envs/wamattack/bin/python probe/fw_patch_train.py --loss away --steps 200 --lr 0.02
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

REPO = Path("/home/user1/workspace/chence/WAMattack")
FASTWAM = REPO / "third_party" / "FastWAM"
PI05 = REPO / "pi05probe"
OUT = REPO / "probe" / "out"

os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / "probe" / "config" / "libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(REPO / "checkpoints"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(FASTWAM / "experiments" / "libero"))
sys.path.insert(0, str(PI05))
sys.path.insert(0, str(REPO / "probe"))

import numpy as np
import torch
import yaml
import cv2

_torch_load_orig = torch.load
torch.load = lambda *a, **k: _torch_load_orig(*a, **{**k, "weights_only": False})

from fw_grad import infer_action_grad  # noqa: E402

TASK = "put_the_bowl_on_the_plate"
SUITE = "libero_goal"
EX = 10
TEX_SQUARE = 128
DEST = {"put_the_bowl_on_the_plate": (0.062, -0.009)}


def project_quad(cell_xy, patch_m):
    """patch 4 世界角 → 模型输入 224 agentview 坐标(row,col),顺序 BL,BR,TR,TL。
    用 fw_cell_world.npz 里已校准的相机 T + code(对着模型 7×7 网格 77/78 精确)。"""
    from fw_project import world_to_px256, orient
    cw = np.load(OUT / "fw_cell_world.npz", allow_pickle=True)
    T, code = cw["T"], tuple(int(v) for v in cw["code"])
    cx, cy = cell_xy
    hh = patch_m / 2.0
    corners = [(cx - hh, cy - hh), (cx + hh, cy - hh), (cx + hh, cy + hh), (cx - hh, cy + hh)]
    q = []
    for xy in corners:
        r, c = world_to_px256(T, xy)
        r, c = orient(r, c, code)
        q.append([r * 224.0 / 256.0, c * 224.0 / 256.0])
    return np.array(q, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="away", choices=["away", "curve", "dev"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--cell", default="0.2227855,-0.00254161", help="攻击格 world x,y")
    ap.add_argument("--patch-m", type=float, default=0.13)
    ap.add_argument("--thresh", type=float, default=0.05, help="足迹 mask 的渲染差阈值")
    args = ap.parse_args()
    cell_xy = tuple(float(v) for v in args.cell.split(","))
    dest_xy = DEST[TASK]

    # ---- 场景配置 + patch 尺寸覆盖 ----
    cfg_scene = yaml.safe_load((PI05 / "config" / "scene.yaml").read_text())
    cfg_scene["patch"]["size_wh"] = [args.patch_m, args.patch_m]
    import scene_patch as sp
    plane = sp.Plane.from_cfg(cfg_scene["plane"])
    lift = cfg_scene["patch"]["thickness"] / 2.0 + cfg_scene["patch"]["normal_offset"]
    cell_world = plane.to_world(cell_xy[0], cell_xy[1], lift)

    # ---- FastWAM 模型 / processor / context(复用官方 eval 路径,与 fw_scan 一致)----
    from run_instruction_sweep import build_cfg, encode_instructions, load_model
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_utils import get_libero_dummy_action, LIBERO_ENV_RESOLUTION
    from eval_libero_single import _obs_to_model_input, _denormalize_action

    suite_obj = benchmark.get_benchmark_dict()[SUITE]()
    tid = next(i for i in range(suite_obj.n_tasks)
               if Path(suite_obj.get_task(i).bddl_file).stem == TASK)
    task = suite_obj.get_task(tid)

    class A:
        suite, task_id, init_index = SUITE, tid, 0
    cfg = build_cfg(A())
    seed = int(cfg.seed)
    prompt = DEFAULT_PROMPT.format(task=str(task.language))
    context, cmask = encode_instructions(cfg, [prompt], "cpu")
    model, processor, device, dtype = load_model(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    Hd, Wd = [int(v) for v in cfg.data.train.video_size]     # 224, 448
    AH = int(cfg.data.train.num_frames) - 1                  # 32
    ns = int(cfg.EVALUATION.num_inference_steps)
    sshift = cfg.EVALUATION.get("sigma_shift")
    rdev = str(cfg.EVALUATION.get("rand_device", "cpu"))
    ctx = context.to(device=device, dtype=dtype)
    cmk = cmask.to(device=device)
    print(f"[cfg] task={TASK} seed={seed} AH={AH} EX={EX} steps(infer)={ns} loss={args.loss}", flush=True)
    print(f"[cfg] 攻击格 world=({cell_xy[0]:.4f},{cell_xy[1]:.4f}) patch={args.patch_m*100:.0f}cm "
          f"dest={dest_xy}", flush=True)

    # ---- 去归一化每通道仿射(scale·x + z0) ----
    z0 = _denormalize_action(torch.zeros(1, AH, model.action_expert.action_dim), processor)[0][0]  # [7]
    z1 = _denormalize_action(torch.ones(1, AH, model.action_expert.action_dim), processor)[0][0]
    scale = torch.tensor((z1 - z0)[0:3], dtype=torch.float32, device=device)
    zoff = torch.tensor(z0[0:3], dtype=torch.float32, device=device)

    def denorm_trans(a_norm):
        """a_norm [AH,7] → 执行前缀净平移(denorm 世界 delta 累加) [3]。"""
        return (a_norm[:EX, 0:3].float() * scale + zoff).sum(0)

    def model_input(obs, dev=device):
        return _obs_to_model_input(obs, cfg=cfg, processor=processor,
                                   width=Wd, height=Hd, device=dev, dtype=dtype)

    def infer_norm(obs):
        x, proprio, _ = model_input(obs)
        with torch.no_grad():
            a = infer_action_grad(model, x, ctx, cmk, proprio, AH, ns, sshift, seed, rdev)
        return a[0].detach(), x.detach(), proprio.detach()

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    # ---- clean rollout:抓 F 帧(x_clean 224×448、proprio、eef world、sim state)----
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                             camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
    env.seed(seed); env.reset()
    inits = np.array(torch.load(str(Path(get_libero_path("init_states")) / SUITE / f"{TASK}.pruned_init")))
    obs = env.set_init_state(inits[0])
    SETTLE = int(cfg.EVALUATION.get("num_steps_wait", 5))
    replan = int(cfg.EVALUATION.replan_steps)
    goal = env.env.parsed_problem["goal_state"]

    eefs, states, a_cleans = [], [], []
    plan = collections.deque()
    ok, t = False, 0
    while t < 400 + SETTLE and len(states) < args.n_frames:
        if t < SETTLE:
            obs, _, _, _ = env.step(get_libero_dummy_action()); t += 1; continue
        if not plan:
            a_norm, _, _ = infer_norm(obs)
            eefs.append(np.asarray(obs["robot0_eef_pos"], np.float32).copy())
            states.append(env.get_sim_state().copy())
            a_cleans.append(a_norm)
            plan.extend(_denormalize_action(a_norm[None], processor)[0][:replan].tolist())
        act = np.array(plan.popleft(), np.float64)
        act[-1] = np.sign(-(act[-1] * 2 - 1))    # 执行侧 gripper 变换(eval_libero_single:423-428)
        obs, _, done, _ = env.step(act.tolist())
        if not ok and all(env.env._eval_predicate(s) for s in goal):  # noqa: SLF001
            ok = True
        t += 1
    env.close()
    F = len(states)
    print(f"[clean] 抓到 {F} 帧(每 {replan} 执行步一帧)clean-episode success={ok}", flush=True)

    # ---- base(clean)与 mask 都用同法 regenerate_obs_from_state 渲染,只隔离出 patch ----
    cenv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                              camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
    cenv.seed(seed); cenv.reset()
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                              camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
    penv.env.set_xml_processor(sp.make_xml_processor(
        cfg_scene, cell_world, str(PI05 / "config" / "probe_texture.png")))
    penv.seed(seed); penv.reset()
    xs, props, masks = [], [], []
    for k in range(F):
        co = cenv.regenerate_obs_from_state(states[k])
        xc, proprio, _ = model_input(co)
        po = penv.regenerate_obs_from_state(states[k])
        xp, _, _ = model_input(po, dev="cpu")
        xs.append(xc.detach()); props.append(proprio.detach())
        diff = (xp[0] - xc[0].float().cpu()).abs().amax(0).numpy()   # [224,448]
        m = diff > args.thresh
        m[:, 224:] = False                                          # 只左半 agentview
        masks.append(m)
    cenv.close(); penv.close()
    munion_np = np.clip(np.sum(masks, 0), 0, 1).astype(bool)
    print(f"[mask] 逐帧足迹像素 {[int(m.sum()) for m in masks]}  union={int(munion_np.sum())}", flush=True)

    # ---- 投影 quad + mask 对齐校验 ----
    quad = project_quad(cell_xy, args.patch_m)
    ys, xs_ = np.where(munion_np)
    cen_mask = (ys.mean(), xs_.mean()); cen_quad = quad.mean(0)
    print(f"[align] mask 质心(r,c)=({cen_mask[0]:.0f},{cen_mask[1]:.0f})  "
          f"quad 质心=({cen_quad[0]:.0f},{cen_quad[1]:.0f})  "
          f"偏差={np.hypot(cen_mask[0]-cen_quad[0], cen_mask[1]-cen_quad[1]):.1f}px", flush=True)

    # ---- 张量化 ----
    base = [x.to(device=device, dtype=dtype) for x in xs]                 # [1,3,224,448]
    M = [torch.from_numpy(m.astype(np.float32)).to(device)[None, None] for m in masks]
    munion = torch.from_numpy(munion_np.astype(np.float32)).to(device)[None, None]
    eps = None  # infer_action_grad 内部用 seed 钉 ε

    # clean denorm 平移 + 方向
    dest3 = torch.tensor([dest_xy[0], dest_xy[1], 0.9], dtype=torch.float32, device=device)
    dir_dest, cdir, trans_clean = [], [], []
    for k in range(F):
        tc = denorm_trans(a_cleans[k])
        trans_clean.append(tc.detach())
        cdir.append(tc / (tc.norm() + 1e-9))
        eefk = torch.tensor(eefs[k], dtype=torch.float32, device=device)
        vv = dest3 - eefk
        dir_dest.append(vv / (vv.norm() + 1e-9))

    def score(a_norm, k):
        trans = denorm_trans(a_norm)
        if args.loss == "away":
            return -(trans * dir_dest[k]).sum()
        if args.loss == "curve":
            perp = trans - (trans * cdir[k]).sum() * cdir[k]
            return torch.linalg.vector_norm(perp)
        return torch.linalg.vector_norm(trans - trans_clean[k])   # dev

    # ---- 红线:同帧同 seed 两次前向一致 ----
    with torch.no_grad():
        r1 = infer_action_grad(model, base[0], ctx, cmk, props[0], AH, ns, sshift, seed, rdev)
        r2 = infer_action_grad(model, base[0], ctx, cmk, props[0], AH, ns, sshift, seed, rdev)
    print(f"[红线] 固定 seed 重复前向 |Δ|max={float((r1-r2).abs().max()):.2e}", flush=True)

    def eval_total(P):
        tot = 0.0
        with torch.no_grad():
            for k in range(F):
                comp = base[k] * (1 - M[k]) + P.clamp(-1, 1).to(dtype) * M[k]
                a = infer_action_grad(model, comp, ctx, cmk, props[k], AH, ns, sshift, seed, rdev)[0]
                tot += float(score(a, k))
        return tot

    P = torch.empty(1, 3, 224, 448, device=device, dtype=torch.float32).uniform_(-1, 1).requires_grad_(True)
    rand_score = eval_total(P)
    print(f"[init] loss={args.loss} 随机 patch 目标值(EOT合计)={rand_score:.3f}", flush=True)

    opt = torch.optim.Adam([P], lr=args.lr)
    hist = []
    for it in range(args.steps):
        opt.zero_grad()
        tot = 0.0
        for k in range(F):                       # 逐帧 backward,一次只留一帧的图
            comp = base[k] * (1 - M[k]) + P.clamp(-1, 1).to(dtype) * M[k]
            a = infer_action_grad(model, comp, ctx, cmk, props[k], AH, ns, sshift, seed, rdev)[0]
            sk = score(a, k)
            (-sk).backward()
            tot += float(sk)
        with torch.no_grad():
            P.grad *= munion
        opt.step()
        with torch.no_grad():
            P.clamp_(-1, 1)
        hist.append(tot)
        if it % 20 == 0 or it == args.steps - 1:
            print(f"  [step {it:3d}] {args.loss} 目标值={tot:.3f} (随机 {rand_score:.3f}) "
                  f"显存峰值 {torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)

    final = hist[-1]
    print(f"[done] loss={args.loss} 最终目标值={final:.3f} vs 随机 {rand_score:.3f}", flush=True)

    Pn = P.detach().cpu().numpy()[0]                            # [3,224,448] [-1,1]
    outp = OUT / f"fw_patch_trained_{args.loss}.npz"
    np.savez_compressed(outp, P=Pn, munion=munion_np, quad224=quad, cell_world=cell_world,
                        cell_xy=np.array(cell_xy), patch_m=args.patch_m, loss=args.loss,
                        hist=np.array(hist), rand_score=rand_score, final_score=final,
                        dest_xy=np.array(dest_xy), clean_success=bool(ok))
    print(f"[written] {outp}", flush=True)

    # ---- unwarp:左半 agentview 足迹四边形 → 方形纹理 PNG ----
    img = (((Pn[:, :, :224].transpose(1, 2, 0) + 1) / 2) * 255).clip(0, 255).astype(np.uint8)  # RGB 224²
    src = quad[:, ::-1].copy()                                  # cv2 要 (x=col,y=row)
    dst = np.array([[0, 0], [TEX_SQUARE - 1, 0], [TEX_SQUARE - 1, TEX_SQUARE - 1],
                    [0, TEX_SQUARE - 1]], np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    tex = cv2.warpPerspective(img, H, (TEX_SQUARE, TEX_SQUARE))
    texp = PI05 / "config" / f"fw_texture_adv_{args.loss}.png"
    cv2.imwrite(str(texp), tex[:, :, ::-1])
    print(f"[written] {texp} ({TEX_SQUARE}²)", flush=True)

    # 合成预览(模型看到的对抗图,左半 agentview)
    clean_img = (((base[0].float().cpu().numpy()[0, :, :, :224].transpose(1, 2, 0) + 1) / 2) * 255)
    mm = munion_np[:, :224][..., None].astype(np.float32)
    comp = (clean_img * (1 - mm) + img.astype(np.float32) * mm).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / f"fw_patch_preview_{args.loss}.png"), comp[:, :, ::-1])
    print(f"[written] {OUT/('fw_patch_preview_'+args.loss+'.png')}", flush=True)


if __name__ == "__main__":
    main()
