#!/usr/bin/env python3
"""FastWAM 版 attention 提取:文本 token → 视频 token 的 cross-attention。

跨模型对应关系
------------
π0.5:文本 token 是**查询**、图像 patch 是**键**,在图像上归一。
FastWAM:视频 token 是**查询**、文本 token 是**键**(`wan_video_dit.py:265` 的 CrossAttention),
方向反过来。取某个词指向的图像图 = attention 矩阵**该词那一列**(键固定、扫所有视频查询),
再**在视频 token 上重新归一**(因为一列不是归一化的)—— 与扩散模型 cross-attn 可视化的通行做法一致。

token 几何(勘察确认)
-------------------
video 输入 224×448 =(agentview‖wrist),VAE ×16 → 14×28,DiT patch [1,2,2] → 网格 f1×h7×w14。
⇒ 98 个 video token,顺序 (f h w),index = h*14 + w。
   **base(agentview)= 左 7 列(w 0..6),wrist = 右 7 列(w 7..13)**,每 token ≈32×32 px。

只在 prefill_video_cache 里算一次(与去噪步无关),所以每次 infer_action 视频 cross-attn 触发一遍。
钩子只挂**视频专家**的 blocks[i].cross_attn(动作专家的 cross-attn 是 action→text,不要)。

红线:head 求和前先确认钩子真的触发(assert 命中层数 == 层数)。

用法(wamattack env,一张 ≥13 GB 空闲卡):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<free> \
      /home/user1/miniconda3/envs/wamattack/bin/python probe/fw_attn.py
    → probe/out/fw_attn.npz
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

TASK = "put_the_bowl_on_the_plate"
SUITE = "libero_goal"
NOUNS = ("bowl", "plate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-index", type=int, default=0)
    ap.add_argument("--settle", type=int, default=10)
    ap.add_argument("--n-frames", type=int, default=10)
    ap.add_argument("--out", default=str(REPO / "probe" / "out" / "fw_attn.npz"))
    ap.add_argument("--text-device", default="cpu")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--nouns", default=",".join(NOUNS), help="逗号分隔;destination attention 用 dest 名词")
    args = ap.parse_args()
    task_stem = args.task
    nouns = tuple(n for n in args.nouns.split(",") if n)

    from run_instruction_sweep import build_cfg, encode_instructions, load_model

    class A:
        suite, task_id, init_index = SUITE, None, args.init_index
    sa = A()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero_utils import get_libero_dummy_action, LIBERO_ENV_RESOLUTION
    from eval_libero_single import _obs_to_model_input, _denormalize_action
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    suite_obj = benchmark.get_benchmark_dict()[SUITE]()
    tid = next(i for i in range(suite_obj.n_tasks)
               if Path(suite_obj.get_task(i).bddl_file).stem == task_stem)
    task = suite_obj.get_task(tid)
    sa.task_id = tid
    cfg = build_cfg(sa)
    seed = int(cfg.seed)
    prompt = DEFAULT_PROMPT.format(task=str(task.language))
    context, cmask = encode_instructions(cfg, [prompt], args.text_device)
    model, processor, device, dtype = load_model(cfg)
    Hd, Wd = [int(v) for v in cfg.data.train.video_size]     # 224, 448
    AH = int(cfg.data.train.num_frames) - 1
    ns = int(cfg.EVALUATION.num_inference_steps)

    # ---- 找名词 token 位置(用同一个 tokenizer 重切一遍 prompt) ----
    from fastwam.models.wan22.helpers.loader import _resolve_configs
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
    _, _, _, tok_c = _resolve_configs(model_id=cfg.model.model_id,
                                      tokenizer_model_id=cfg.model.tokenizer_model_id,
                                      redirect_common_files=bool(cfg.model.redirect_common_files))
    tok_c.download_if_necessary()
    tokenizer = HuggingfaceTokenizer(name=tok_c.path, seq_len=int(cfg.model.tokenizer_max_len),
                                     clean="whitespace")
    ids, m = tokenizer([prompt], return_mask=True, add_special_tokens=True)
    real_len = int(m[0].gt(0).sum())
    toks = [tokenizer.tokenizer.decode([int(t)]).strip().lower() for t in ids[0, :real_len]]
    noun_idx = [i for i, t in enumerate(toks) if any(n in t for n in nouns)]
    print(f"[tok] 真实 token 数={real_len}  名词位置={noun_idx} "
          f"({[toks[i] for i in noun_idx]})", flush=True)

    # ---- 钩子:挂视频专家每层 cross_attn,重算 softmax,head 求和 ----
    vblocks = model.mot.mixtures["video"].blocks
    Ln = len(vblocks)
    cap = {}

    def mk_hook(li):
        def hook(module, inp, out):
            x, ctx = inp[0], inp[1]                       # x=norm3(video), ctx=text(+proprio)
            q = module.norm_q(module.q(x)).float()
            k = module.norm_k(module.k(ctx)).float()
            H, Dh = module.num_heads, module.attn_head_dim
            B, Sq, _ = q.shape
            Sk = k.shape[1]
            q = q.view(B, Sq, H, Dh).permute(0, 2, 1, 3)   # [B,H,Sq,Dh]
            k = k.view(B, Sk, H, Dh).permute(0, 2, 1, 3)
            sc = torch.matmul(q, k.transpose(-1, -2)) / (Dh ** 0.5)   # [B,H,Sq,Sk]
            att = torch.softmax(sc, dim=-1)
            cap[li] = att[0].sum(0).detach().cpu().numpy()  # head 求和 → [Sq,Sk]
        return hook

    handles = [vblocks[i].cross_attn.register_forward_hook(mk_hook(i)) for i in range(Ln)]
    print(f"[hook] 挂了 {Ln} 层视频 cross_attn", flush=True)

    def infer_capture(obs):
        cap.clear()
        image, proprio, _ = _obs_to_model_input(obs, cfg=cfg, processor=processor,
                                                width=Wd, height=Hd, device=device, dtype=dtype)
        with torch.no_grad():
            model.infer_action(prompt=None, input_image=image, action_horizon=AH, proprio=proprio,
                               context=context.to(device=device, dtype=dtype),
                               context_mask=cmask.to(device=device), num_inference_steps=ns,
                               sigma_shift=cfg.EVALUATION.get("sigma_shift"), seed=seed,
                               rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")), tiled=False)
        assert len(cap) == Ln, f"钩子只命中 {len(cap)}/{Ln} 层,cross_attn 没全触发"
        return np.stack([cap[i] for i in range(Ln)])       # [L,Sq,Sk]

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=LIBERO_ENV_RESOLUTION,
                             camera_widths=LIBERO_ENV_RESOLUTION, camera_segmentations="element")
    env.seed(seed); env.reset()
    inits = np.array(torch.load(str(Path(get_libero_path("init_states")) / SUITE / f"{TASK}.pruned_init")))
    obs = env.set_init_state(inits[args.init_index])
    for _ in range(args.settle):
        obs, _, _, _ = env.step(get_libero_dummy_action())

    # 红线:同一帧连抽两次,attention 逐位相同(确定性)
    a0 = infer_capture(obs); a1 = infer_capture(obs)
    dr = float(np.abs(a1 - a0).max())
    print(f"[红线] 同一帧 attention 连抽两次最大差 = {dr:.3e}", flush=True)
    assert dr == 0.0, f"attention 不确定({dr:.3e}),先查清"
    Sq, Sk = a0.shape[1], a0.shape[2]
    print(f"[shape] L={Ln} Sq(video)={Sq} Sk(context)={Sk}", flush=True)

    # ---- clean rollout,逐帧抽 attention + 存 agentview 图 ----
    goal = env.env.parsed_problem["goal_state"]
    ATT, IMGS, ok, t = [], [], False, 0
    plan = collections.deque()
    while t < 400 + args.settle and len(ATT) < args.n_frames:
        if t < args.settle:
            obs, _, _, _ = env.step(get_libero_dummy_action()); t += 1; continue
        if not plan:
            ATT.append(infer_capture(obs))
            IMGS.append(obs["agentview_image"][::-1].copy())   # 存正立 agentview,给热图叠底
            a = _denormalize_action(model.infer_action(
                prompt=None, input_image=_obs_to_model_input(obs, cfg=cfg, processor=processor,
                    width=Wd, height=Hd, device=device, dtype=dtype)[0],
                action_horizon=AH, proprio=_obs_to_model_input(obs, cfg=cfg, processor=processor,
                    width=Wd, height=Hd, device=device, dtype=dtype)[1],
                context=context.to(device=device, dtype=dtype), context_mask=cmask.to(device=device),
                num_inference_steps=ns, seed=seed,
                rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")), tiled=False)["action"],
                processor)[0]
            plan.extend(a[:int(cfg.EVALUATION.replan_steps)].tolist())
        act = np.array(plan.popleft(), np.float64)
        act[-1] = np.sign(-(act[-1] * 2 - 1))
        obs, _, done, _ = env.step(act.tolist())
        if not ok and all(env.env._eval_predicate(s) for s in goal):     # noqa: SLF001
            ok = True
        t += 1
    env.close()
    for h in handles:
        h.remove()
    ATT = np.stack(ATT)                                     # [T,L,Sq,Sk]
    print(f"[clean] 取到 {len(ATT)} 帧 attention success={ok}", flush=True)

    np.savez_compressed(
        Path(args.out), task=task_stem, seed=seed, prompt=prompt, n_layers=Ln,
        Sq=Sq, Sk=Sk, real_len=real_len, tokens=np.array(toks, dtype=object),
        noun_idx=np.array(noun_idx), attn=ATT.astype(np.float32),
        agentview=np.stack(IMGS), success=bool(ok),
        grid_h=7, grid_w=14, base_cols=7)
    print(f"[written] {args.out}  attn shape {ATT.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
