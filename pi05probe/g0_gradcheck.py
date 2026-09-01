#!/usr/bin/env python3
"""G0 · 梯度校验(只验证,不做 FD 扫描 / 不做代理评估 / 不打补丁)。

三步,全部通过才允许写 G1/G2 代码;任何一步失败就停下报告:
  G0.1 梯度能否到达像素叶子:image.requires_grad → 重实现的 sample_actions 前向
       (绕开 @torch.no_grad 装饰器,不改模型内部)→ backward → x.grad 非 None 非全零。
  G0.2 数值校验:对 s_c = Σ_{k<EX} a[k,c](c=0,1,2 平移三通道,EX=5)——
       解析梯度(backward) vs 中心差分(h ∈ {1e-2,1e-3,1e-4},固定 ε),
       选解析 |g| 最大的 5 个像素 + 3 个随机像素,相对误差 ~10% 算过。
       另演示:用户 spec 的 L = Σ‖a−a_clean‖²(a_clean=detach 同一前向)在 x 处
       解析梯度恒等于 0(∇L = 2Jᵀ(a−a_clean) = 0)—— 这是 spec 问题的证据,不是 bug。
  G0.3 健康检查:NaN/inf、量级、backward 用时、显存峰值;
       3 条不同 ε 的梯度图两两秩相关 = 梯度自身的 ε 噪声地板。

绕开方式(不改模型文件):逐字复刻 pi0_pytorch.sample_actions(:376-420)的函数体
到本文件的 sample_actions_grad(),在 torch.enable_grad() 下调用未加装饰的
embed_prefix / denoise_step 等方法。_apply_checkpoint 在 eval 下是直通(:151)。

用法:
    CUDA_VISIBLE_DEVICES=0 /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/g0_gradcheck.py
"""
import os
import pathlib
import sys
import time

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
OUT = ROOT / "pi05probe" / "out"
GOUT = OUT / "grad"
PATCHED_TF = ROOT / "third_party" / "transformers_patched"
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"

for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

AH, AD, EX = 10, 32, 5     # 与 s2_scan.py:35 相同
FR = 8                     # 用第 8 帧(放置阶段,画面里 bowl 已在 plate 上方)
SEED_EPS = 20260817        # serve_policy_fixed_noise.py 的固定 ε 配方
NPIX_TOP, NPIX_RND = 5, 3
HS = (1e-2, 1e-3, 1e-4)    # 中心差分步长([-1,1] 像素尺度)


def sample_actions_grad(model, device, observation, noise, num_steps=10):
    """pi0_pytorch.PI0Pytorch.sample_actions(:376-420)逐字复刻,仅去掉 @torch.no_grad。"""
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    bsize = observation.state.shape[0]
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation, train=False)

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    dt = -1.0 / num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    t = torch.tensor(1.0, dtype=torch.float32, device=device)
    while t >= -dt / 2:
        expanded_time = t.expand(bsize)
        v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
        x_t = x_t + dt * v_t
        t = t + dt          # 原文 time += dt;等价、且对 autograd 无影响(t 不在图里)
    return x_t


def main():
    import torch
    import jax
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.models import model as _model

    GOUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def out(s=""):
        print(s, flush=True)
        lines.append(s)

    d = np.load(OUT / "s2f_scan_obs.npz", allow_pickle=True)
    prompt = str(d["prompt"])
    cimg, cwri, cstate = d["clean_img224"][FR], d["clean_wrist224"][FR], d["clean_state8"][FR]
    out(f"[data] frame {FR}/{d['clean_img224'].shape[0]}  prompt={prompt!r}")

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    model = policy._model
    device = policy._pytorch_device
    for p in model.parameters():
        p.requires_grad_(False)         # 只要输入梯度,省参数梯度显存
    n_bf16 = sum(1 for p in model.parameters() if p.dtype == torch.bfloat16)
    n_f32 = sum(1 for p in model.parameters() if p.dtype == torch.float32)
    out(f"[policy] loaded  device={device}  参数 dtype: bf16×{n_bf16} f32×{n_f32}")

    def build_obs():
        """复刻 policy.infer(:68-90) 的输入路径,返回 Observation(image 已是 [1,3,224,224] f32 [-1,1])。"""
        el = {"observation/image": cimg, "observation/wrist_image": cwri,
              "observation/state": cstate, "prompt": prompt}
        inputs = jax.tree.map(lambda x: x, el)
        inputs = policy._input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    obs = build_obs()
    base_img = obs.images["base_0_rgb"]
    assert base_img.shape == (1, 3, 224, 224) and base_img.dtype == torch.float32, base_img.shape
    assert float(base_img.min()) >= -1.001 and float(base_img.max()) <= 1.001
    eps = torch.from_numpy(
        np.random.RandomState(SEED_EPS).standard_normal((AH, AD)).astype(np.float32)
    ).to(device)[None]

    # ---------- 前置红线:同 ε 重复前向逐位相同(中心差分的前提) ----------
    with torch.no_grad():
        a1 = sample_actions_grad(model, device, build_obs(), eps.clone())
        a2 = sample_actions_grad(model, device, build_obs(), eps.clone())
    rep = float((a1 - a2).abs().max())
    out(f"[红线] 同 ε 重复前向 max|Δa| = {rep:.3e}  {'✅' if rep == 0.0 else '❌ 必须为 0'}")
    if rep != 0.0:
        (GOUT / "FINDINGS_grad.md").write_text("\n".join(lines) + "\n")
        return 1

    # ---------- G0.1 梯度到达像素叶子 ----------
    leaf = base_img.clone().detach().requires_grad_(True)
    obs.images["base_0_rgb"] = leaf
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    with torch.enable_grad():
        acts = sample_actions_grad(model, device, obs, eps.clone())
    t_fwd = time.monotonic() - t0
    out(f"\n== G0.1 梯度到达性 ==")
    out(f"  前向(带图)用时 {t_fwd:.1f}s  acts.requires_grad={acts.requires_grad}  "
        f"grad_fn={type(acts.grad_fn).__name__ if acts.grad_fn else None}")
    if not acts.requires_grad:
        out("  ❌ 输出不在计算图上,梯度路径被阻断 —— 停")
        (GOUT / "FINDINGS_grad.md").write_text("\n".join(lines) + "\n")
        return 1

    # 用户 spec 的 L(a_clean = 同一前向 detach)→ 解析梯度应恒为 0
    a_clean = acts.detach()
    L = ((acts - a_clean)[0, :EX, :] ** 2).sum()
    L.backward(retain_graph=True)
    userL_gmax = float(leaf.grad.abs().max())
    leaf.grad = None
    out(f"  [spec 演示] 用户定义 L=Σ‖a−a_clean‖²(固定 ε ⇒ a≡a_clean):"
        f"L={float(L):.3e}, max|∇L|={userL_gmax:.3e}(预期恰好 0)")

    # 三个平移通道分开 backward(G1 口径预演:s_c = Σ_{k<EX} a[k,c])
    g_ch, t_bwd = [], []
    for c in range(3):
        s_c = acts[0, :EX, c].sum()
        t0 = time.monotonic()
        s_c.backward(retain_graph=True)
        t_bwd.append(time.monotonic() - t0)
        g_ch.append(leaf.grad.detach().clone())
        leaf.grad = None
    g = torch.stack([x[0] for x in g_ch])           # [3ch, 3rgb, 224, 224]
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    del acts, a_clean, L                            # 释放保留的计算图,给差分前向腾显存
    torch.cuda.empty_cache()
    nz = float((g != 0).float().mean())
    out(f"  ∂s_c/∂x:非零元素占比 {nz*100:.1f}%  "
        f"{'✅ 非 None 非全零' if nz > 0 else '❌ 全零 —— 停'}")
    out(f"  backward 用时 {[f'{x:.1f}s' for x in t_bwd]}  显存峰值 {peak_gb:.1f} GB")
    if nz == 0:
        (GOUT / "FINDINGS_grad.md").write_text("\n".join(lines) + "\n")
        return 1

    # ---------- G0.2 中心差分数值校验 ----------
    out(f"\n== G0.2 数值校验(中心差分,固定 ε,EX={EX}) ==")
    G = g.norm(dim=0)                               # [3rgb,224,224] 跨 s_c 通道合成,选点用
    flat = G.flatten()
    top = torch.topk(flat, NPIX_TOP).indices.cpu().numpy()
    rng = np.random.RandomState(0)
    rnd = rng.choice(flat.numel(), NPIX_RND, replace=False)
    pix = [np.unravel_index(int(i), G.shape) for i in np.concatenate([top, rnd])]

    def fwd_sc(img_tensor):
        o = build_obs()
        o.images["base_0_rgb"] = img_tensor
        with torch.no_grad():
            a = sample_actions_grad(model, device, o, eps.clone())
        return a[0, :EX, :3].sum(0)                 # [3] = (s_0, s_1, s_2)

    x0 = base_img.clone().detach()
    hdr = "  像素(rgb,y,x)      通道   解析梯度      " + "  ".join(f"relerr@h={h:g}" for h in HS)
    out(hdr)
    relerr_all = {h: [] for h in HS}
    for (rc, py, px) in pix:
        nums = {}
        for h in HS:
            xp = x0.clone(); xp[0, rc, py, px] += h
            xm = x0.clone(); xm[0, rc, py, px] -= h
            nums[h] = (fwd_sc(xp) - fwd_sc(xm)) / (2 * h)   # [3]
        for c in range(3):
            ana = float(g[c, rc, py, px])
            if abs(ana) < 1e-9:
                continue                             # 解析≈0 的通道不计入(相对误差无定义)
            errs = []
            for h in HS:
                num = float(nums[h][c])
                e = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
                errs.append(e)
                relerr_all[h].append(e)
            out(f"  ({rc},{py:3d},{px:3d})   s_{c}   {ana:+.3e}   "
                + "   ".join(f"{e*100:9.1f}%" for e in errs))
    med = {h: (np.median(relerr_all[h]) if relerr_all[h] else np.nan) for h in HS}
    out("  中位相对误差: " + "  ".join(f"h={h:g}: {med[h]*100:.1f}%" for h in HS))
    best_h = min(med, key=lambda h: med[h])
    g02_pass = med[best_h] <= 0.15
    out(f"  G0.2 {'✅ 通过' if g02_pass else '❌ 未过(~10% 线)—— 停'}"
        f"(最优 h={best_h:g},中位 {med[best_h]*100:.1f}%)")

    # ---------- G0.3 健康检查 + 3ε 稳定性 ----------
    out(f"\n== G0.3 健康检查 ==")
    n_nan = int(torch.isnan(g).sum()); n_inf = int(torch.isinf(g).sum())
    out(f"  NaN={n_nan}  inf={n_inf}  {'✅' if n_nan + n_inf == 0 else '❌'}")
    out(f"  量级:|g| max={float(g.abs().max()):.3e}  mean={float(g.abs().mean()):.3e}  "
        f"p99={float(torch.quantile(g.abs().flatten().float(), 0.99)):.3e}")

    def gmag_for_eps(seed):
        e = torch.from_numpy(
            np.random.RandomState(seed).standard_normal((AH, AD)).astype(np.float32)
        ).to(device)[None]
        o = build_obs()
        lf = o.images["base_0_rgb"].clone().detach().requires_grad_(True)
        o.images["base_0_rgb"] = lf
        with torch.enable_grad():
            a = sample_actions_grad(model, device, o, e)
        gs = []
        for c in range(3):
            a[0, :EX, c].sum().backward(retain_graph=(c < 2))
            gs.append(lf.grad.detach().clone()[0])
            lf.grad = None
        return torch.stack(gs).norm(dim=(0, 1)).cpu().numpy()   # [224,224] 像素显著性

    maps = [torch.stack(g_ch)[:, 0].norm(dim=(0, 1)).cpu().numpy()]  # ε₀ 复用已算的
    for k in (1, 2):
        maps.append(gmag_for_eps(SEED_EPS + k))
        torch.cuda.empty_cache()

    def spear(a, b):
        ra = np.argsort(np.argsort(a.flatten())).astype(float)
        rb = np.argsort(np.argsort(b.flatten())).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    r01, r02, r12 = spear(maps[0], maps[1]), spear(maps[0], maps[2]), spear(maps[1], maps[2])
    out(f"  3 条 ε 的梯度显著性图两两秩相关:{r01:+.3f} / {r02:+.3f} / {r12:+.3f}"
        f"(= 梯度自身的 ε 噪声地板;后续 S_grad↔influence 相关高于此才有意义)")

    np.savez_compressed(GOUT / "g0_gradcheck.npz",
                        frame=FR, eps_seed=SEED_EPS,
                        g_ch=g.cpu().numpy(),               # [3ch,3rgb,224,224] 解析梯度
                        gmag_eps=np.stack(maps),            # [3,224,224] 三条 ε 的显著性
                        pix=np.array(pix), hs=np.array(HS),
                        med_relerr=np.array([med[h] for h in HS]))
    out(f"\n[written] {GOUT/'g0_gradcheck.npz'}")

    verdict = ("✅ G0 三项全过,可进入 G1(等用户确认 L 定义)" if g02_pass and nz > 0 and n_nan + n_inf == 0
               else "❌ G0 未全过 —— 停,见上")
    out(f"\n== 结论 == {verdict}")
    md = ["# FINDINGS_grad.md — G0 梯度校验(π0.5 pi05_libero PyTorch)", "",
          f"日期 2026-08-20 · 帧 {FR}/16(放置阶段)· 固定 ε seed={SEED_EPS} · EX={EX}", "",
          "```", *lines, "```", ""]
    (GOUT / "FINDINGS_grad.md").write_text("\n".join(md))
    print(f"[written] {GOUT/'FINDINGS_grad.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
