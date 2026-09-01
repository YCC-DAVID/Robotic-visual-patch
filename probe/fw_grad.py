#!/usr/bin/env python3
"""FastWAM 帧 0 梯度显著性 × 三任务:验证「梯度代理」跨模型是否成立(用户点名)。

已有的 ground truth(不再跑任何 rollout/扫描):
    fw_scan.npz / fw_scan_rack.npz / fw_scan_cabinet.npz —— 78 锚点 × 10 帧 FD influence。
本脚本只算:每个任务在**初始帧(settle 后第一帧)**上做一次(3 通道)backward,
把逐像素梯度按每个锚点贴纸的渲染像素掩码聚合成 S_pooled[78],与 FD influence 秩相关。

梯度路径审计(已读码确认,不改模型文件):
  - 阻断仅在装饰器:infer_action(fastwam.py:905)、
    _encode_input_image_latents_tensor(:253)、_predict_action_noise_with_cache(:694),
    以及结尾 .detach()(:1047)。三个函数体在本文件逐字复刻,去掉装饰器。
  - VAE encode(wan_video_vae.py:1218,tiled=False)、pre_dit/post_dit、
    mot.prefill_video_cache / forward_action_with_video_cache、scheduler.step(非原地,
    scheduler_continuous.py:83-88)全部无 no_grad/detach。
  - _build_mot_attention_mask 带 no_grad 但只产布尔掩码,不在像素梯度路径上,原样调用。
  - ε:torch.Generator(seed) 钉死(红线=0 已在 fw_scan 验证过;本脚本再验)。

口径(与 crosstask.py 的 influence 完全对齐):
  s_c = scale_c · Σ_{k<EX} a_raw[k,c],c=0,1,2(平移),EX=10;
  scale_c 从 _denormalize_action 的仿射里数值探出(去归一化对梯度只是每通道常数)。
  gmag = √(Σ_c (scale_c g_c)²) 对 RGB 合成;S_pooled = 贴纸掩码内求和。

用法(nnmc62,wamattack env,≥15 GB 空闲卡):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free> \
      /home/user1/miniconda3/envs/wamattack/bin/python probe/fw_grad.py
    → probe/out/fw_grad_f0.npz
"""
from __future__ import annotations

import json
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
sys.path.insert(0, str(FASTWAM / "experiments" / "libero"))
sys.path.insert(0, str(PI05))

import numpy as np
import torch
import yaml

_torch_load_orig = torch.load
torch.load = lambda *a, **k: _torch_load_orig(*a, **{**k, "weights_only": False})

SUITE = "libero_goal"
TASKS = [
    ("put_the_bowl_on_the_plate",        "fw_scan.npz"),
    ("put_the_wine_bottle_on_the_rack",  "fw_scan_rack.npz"),
    ("put_the_bowl_on_top_of_the_cabinet", "fw_scan_cabinet.npz"),
]
EX = 10
SETTLE = 10
TEX = str(PI05 / "config" / "probe_texture.png")


def infer_action_grad(model, input_image, context, context_mask, proprio,
                      action_horizon, num_inference_steps, sigma_shift, seed, rand_device):
    """fastwam.py:906-1048 逐字复刻,去掉 @torch.no_grad 与结尾 .detach。
    内联被装饰的 _encode_input_image_latents_tensor(:254)与
    _predict_action_noise_with_cache(:695)的函数体。"""
    model.eval()
    if input_image.ndim == 3:
        input_image = input_image.unsqueeze(0)
    generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
    latents_action = torch.randn(
        (1, action_horizon, model.action_expert.action_dim),
        generator=generator, device=rand_device, dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)

    input_image = input_image.to(device=model.device, dtype=model.torch_dtype)
    # --- _encode_input_image_latents_tensor(:254-266)内联 ---
    image = input_image.to(device=model.device)[0].unsqueeze(1)
    z = model.vae.encode([image], device=model.device, tiled=False)
    if isinstance(z, list):
        z = z[0].unsqueeze(0)
    first_frame_latents = z
    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    context = context.to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
    context_mask = context_mask.to(device=model.device, dtype=torch.bool, non_blocking=True)
    if proprio is not None:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
        context, context_mask = model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio)

    timestep_video = torch.zeros((first_frame_latents.shape[0],),
                                 dtype=first_frame_latents.dtype, device=model.device)
    video_pre = model.video_expert.pre_dit(
        x=first_frame_latents, timestep=timestep_video, context=context,
        context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse_flag)
    video_seq_len = int(video_pre["tokens"].shape[1])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len, action_seq_len=latents_action.shape[1],
        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        device=video_pre["tokens"].device)
    video_kv_cache = model.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"], video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={"context": video_pre["context"],
                               "mask": video_pre["context_mask"]},
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len])

    ts_a, deltas_a = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps, device=model.device,
        dtype=latents_action.dtype, shift_override=sigma_shift)
    for step_t, step_delta in zip(ts_a, deltas_a):
        timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=model.device)
        # --- _predict_action_noise_with_cache(:695-723)内联 ---
        action_pre = model.action_expert.pre_dit(
            action_tokens=latents_action, timestep=timestep_action,
            context=context, context_mask=context_mask)
        action_tokens = model.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"], action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={"context": action_pre["context"],
                                    "mask": action_pre["context_mask"]},
            video_kv_cache=video_kv_cache, attention_mask=attention_mask,
            video_seq_len=video_seq_len)
        pred_action = model.action_expert.post_dit(action_tokens, action_pre)
        latents_action = model.infer_action_scheduler.step(pred_action, step_delta, latents_action)
    return latents_action                                    # [1,AH,dim],不 detach


def main():
    from run_instruction_sweep import build_cfg, encode_instructions, load_model
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_utils import get_libero_dummy_action, LIBERO_ENV_RESOLUTION
    from eval_libero_single import _obs_to_model_input, _denormalize_action
    import scene_patch as sp

    # ---- 78 锚点(与 fw_scan.py 逐字同一池) ----
    cfg_scene = yaml.safe_load((PI05 / "config" / "scene.yaml").read_text())
    spec = json.loads((PI05 / "out" / "fine_anchors.json").read_text())
    plane = sp.Plane.from_cfg(cfg_scene["plane"])
    w, h = cfg_scene["patch"]["size_wh"]
    lift = cfg_scene["patch"]["thickness"] / 2.0 + cfg_scene["patch"]["normal_offset"]
    old_legal = [a for a in sp.make_anchors(cfg_scene) if a.legal]
    fine = [sp.Anchor(index=int(r["index"]), plane=plane.name, u=float(r["u"]), v=float(r["v"]),
                      world=plane.to_world(float(r["u"]), float(r["v"]), lift),
                      inside_plane=plane.contains_uv(float(r["u"]), float(r["v"]), w, h),
                      keepout_hits=()) for r in spec["anchors"]]
    anchors = old_legal + fine
    M = len(anchors)
    print(f"[cfg] 候选池 {M} 锚点(旧合法 {len(old_legal)} + 加密 {len(fine)})", flush=True)

    suite_obj = benchmark.get_benchmark_dict()[SUITE]()
    stems = {Path(suite_obj.get_task(i).bddl_file).stem: i for i in range(suite_obj.n_tasks)}

    # ---- 一次编码三个任务的 context(umT5 在 cpu,编完即 del) ----
    class A:
        suite, task_id, init_index = SUITE, stems[TASKS[0][0]], 0
    cfg0 = build_cfg(A())
    prompts = [DEFAULT_PROMPT.format(task=str(suite_obj.get_task(stems[s]).language))
               for s, _ in TASKS]
    ctx_all, cmask_all = encode_instructions(cfg0, prompts, "cpu")
    print(f"[text] 三个 context 编码完成 {tuple(ctx_all.shape)}", flush=True)

    model, processor, device, dtype = load_model(cfg0)
    for p in model.parameters():
        p.requires_grad_(False)
    Hd, Wd = [int(v) for v in cfg0.data.train.video_size]
    AH = int(cfg0.data.train.num_frames) - 1
    ns = int(cfg0.EVALUATION.num_inference_steps)
    sshift = cfg0.EVALUATION.get("sigma_shift")
    rdev = str(cfg0.EVALUATION.get("rand_device", "cpu"))

    # ---- 去归一化的每通道仿射 scale(对梯度只是常数) ----
    z0 = _denormalize_action(torch.zeros(1, AH, model.action_expert.action_dim), processor)[0]
    z1 = _denormalize_action(torch.ones(1, AH, model.action_expert.action_dim), processor)[0]
    scale = (z1 - z0)[0]                                     # [7]
    print(f"[norm] denorm scale = {np.round(scale, 4)}", flush=True)

    def rank(a):
        return np.argsort(np.argsort(a)).astype(float)

    def spear(a, b):
        return float(np.corrcoef(rank(a), rank(b))[0, 1])

    results = {}
    for ti, (stem, inf_file) in enumerate(TASKS):
        tid = stems[stem]
        class T:
            suite, task_id, init_index = SUITE, tid, 0
        cfg = build_cfg(T())
        seed = int(cfg.seed)
        task = suite_obj.get_task(tid)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ctx, cmask = ctx_all[ti], cmask_all[ti]
        print(f"\n===== [{ti}] {stem}  seed={seed} =====", flush=True)

        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                                 camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
        env.seed(seed); env.reset()
        inits = np.array(torch.load(str(Path(get_libero_path("init_states")) / SUITE / f"{stem}.pruned_init")))
        obs = env.set_init_state(inits[0])
        for _ in range(SETTLE):
            obs, _, _, _ = env.step(get_libero_dummy_action())
        st0 = env.get_sim_state().copy()
        x_clean, proprio, _ = _obs_to_model_input(obs, cfg=cfg, processor=processor,
                                                  width=Wd, height=Hd, device=device, dtype=dtype)
        env.close()

        # ---- 红线:复刻版自身重复 + 对官方 infer_action 逐位一致 ----
        with torch.no_grad():
            a1 = infer_action_grad(model, x_clean.clone(), ctx.clone(), cmask.clone(),
                                   proprio.clone(), AH, ns, sshift, seed, rdev)
            a2 = infer_action_grad(model, x_clean.clone(), ctx.clone(), cmask.clone(),
                                   proprio.clone(), AH, ns, sshift, seed, rdev)
            off = model.infer_action(prompt=None, input_image=x_clean.clone(), action_horizon=AH,
                                     proprio=proprio.clone(),
                                     context=ctx.clone().to(device=device, dtype=dtype),
                                     context_mask=cmask.clone().to(device=device),
                                     num_inference_steps=ns, sigma_shift=sshift,
                                     seed=seed, rand_device=rdev, tiled=False)["action"]
        rep = float((a1 - a2).abs().max())
        vs = float((a1[0].float().cpu() - off).abs().max())
        print(f"[红线] 复刻重复 max|Δ|={rep:.3e}  复刻vs官方 max|Δ|={vs:.3e}", flush=True)
        assert rep == 0.0 and vs == 0.0, "复刻不忠实,停"

        # ---- 梯度:三平移通道分开 backward ----
        leaf = x_clean.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            a = infer_action_grad(model, leaf, ctx.clone(), cmask.clone(),
                                  proprio.clone(), AH, ns, sshift, seed, rdev)
        gs = []
        for c in range(3):
            a[0, :EX, c].sum().backward(retain_graph=(c < 2))
            gs.append(leaf.grad.detach()[0].float().clone() * float(scale[c]))
            leaf.grad = None
        g = torch.stack(gs)                                  # [3ch,3rgb,224,448] f32
        assert not torch.isnan(g).any() and not torch.isinf(g).any()
        gmag = g.norm(dim=(0, 1)).cpu().numpy()              # [224,448]
        nz = float((g != 0).float().mean())
        print(f"[grad] 非零 {nz*100:.1f}%  |g|max={float(g.abs().max()):.2e}  "
              f"显存峰值 {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
        del a, leaf
        torch.cuda.empty_cache()

        # ---- 78 锚点贴纸掩码(帧 0 渲染差分)→ S_pooled ----
        xc = x_clean[0].float().cpu().numpy()                # [3,224,448]
        penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                                  camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
        S = np.zeros(M); npx = np.zeros(M, int)
        masks = np.zeros((M, 224, 224), bool)
        for i, an in enumerate(anchors):
            penv.env.set_xml_processor(sp.make_xml_processor(cfg_scene, an.world, TEX))
            penv.seed(seed); penv.reset()
            po = penv.regenerate_obs_from_state(st0)
            xp, _, _ = _obs_to_model_input(po, cfg=cfg, processor=processor,
                                           width=Wd, height=Hd, device="cpu", dtype=torch.float32)
            m = (np.abs(xp[0].numpy() - xc).max(0) > 0.05)[:, :224]   # 左半 = agentview
            masks[i] = m; npx[i] = int(m.sum())
            S[i] = float(gmag[:, :224][m].sum()) if m.any() else 0.0
            if i % 20 == 0:
                print(f"  [mask] {i:2d}/{M} #{an.index} px={npx[i]}", flush=True)
        penv.close()

        out = dict(S_pooled=S, mask_px=npx, gmag=gmag, task=stem, seed=seed)
        # 若本机有 FD influence 文件,顺手算相关(否则回本地算)
        fpath = OUT / inf_file
        if fpath.is_file():
            zi = np.load(fpath, allow_pickle=True)
            Ac, Ap = zi["A_clean"], zi["A_patched"]
            v = (Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]).sum(2)
            I_tot = np.linalg.norm(v, axis=2).sum(1) * 50.0
            perT = np.linalg.norm(v, axis=2) * 50.0
            assert np.array_equal(zi["anchor_idx"], np.array([an.index for an in anchors]))
            r_tot, r_f0 = spear(I_tot, S), spear(perT[:, 0], S)
            print(f"[corr] grad_f0 ↔ FD总(10帧)= {r_tot:+.2f}   ↔ FD帧0 = {r_f0:+.2f}", flush=True)
            out.update(fd_total=I_tot, fd_f0=perT[:, 0], spear_total=r_tot, spear_f0=r_f0)
        results[stem] = out

    save = {}
    for s, o in results.items():
        for k, v in o.items():
            if k != "task":
                save[f"{s}__{k}"] = v
    np.savez_compressed(OUT / "fw_grad_f0.npz",
                        anchor_idx=np.array([an.index for an in anchors]),
                        anchor_world=np.array([an.world for an in anchors]),
                        tasks=np.array([s for s, _ in TASKS]), **save)
    print(f"\n[written] {OUT/'fw_grad_f0.npz'}", flush=True)


if __name__ == "__main__":
    main()
