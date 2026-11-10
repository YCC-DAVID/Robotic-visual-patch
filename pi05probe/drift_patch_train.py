#!/usr/bin/env python3
"""DRIFT 复现 · 阶段 B(py3.11,openpi-server):用 L₀(第一步速度场分歧)优化桌面贴纸图案。

对应 HANDOFF.md 第 4 节。目标函数(arXiv:2608.03207):

    L₀(δ) = E_{o∼D} || v_θ(A^{τ(0)}, o⁺) − v_θ(A^{τ(0)}, o) ||²₂     ← 最大化

其中 v_θ 是 flow matching 的**速度场预测**,A^{τ(0)} 是第一个去噪步(t=1.0)上的带噪动作。
π0.5 的去噪循环从 t=1.0、x_t=noise 起(见 g0_gradcheck.sample_actions_grad),
所以 v_θ(A^{τ(0)}, o) 就是**第一次 denoise_step 的返回值** —— 只跑 1 步而不是 10 步。

论文的核心论证:逐步梯度互相抵消(g₀ 与 g₉ 余弦 ≈ −0.35,梯度存活比 1.00→0.45),
所以只攻第一步比攻多步窗口又强又省。本脚本默认单步(--steps-set 0),可用 --steps-set 0,1,2 做消融。

与论文的差异(有意,见 HANDOFF 4.3):图案贴在 **base 相机的桌面 influence 最高格**,
而非论文的夹爪/腕部相机。位置归因由 4.6 的数字合成评测负责。

优化按论文:PGD、灰色初始化(0.5)、[0,1] 像素空间、每步 clip、lr 0.01、500 步。
观测池来自 drift_pool_prep.py(逐帧遮挡 mask,共享一张图案,union 掩梯度)。

用法:
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/drift_patch_train.py --iters 500 --batch 8
"""
import argparse
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(os.environ.get("WAMATTACK_PROJECT_ROOT", "/home/user1/workspace/chence/WAMattack"))
OPENPI = ROOT / "third_party" / "openpi"
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"
PATCHED_TF = ROOT / "third_party" / "transformers_patched"
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"

for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(PROBE))
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["OPENPI_DATA_HOME"] = os.environ.get("WAMATTACK_OPENPI_DATA_HOME", "/home/user1/.cache/openpi")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

AH, AD = 10, 32
NUM_DENOISE = 10
SEED_NOISE = 20260901
BASE_KEY = "base_0_rgb"


def prefix_and_cache(model, device, observation):
    """sample_actions_grad(:50-73) 的前半段:预处理 + embed_prefix + paligemma 前向拿 KV cache。
    图像梯度就是从这里流出去的。返回 denoise_step 需要的全部条件量。"""
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(  # noqa: SLF001
        observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
        past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True)
    return state, prefix_pad_masks, past_key_values


def velocity_at_steps(model, device, observation, noise, step_set=(0,)):
    """返回指定去噪步上的速度场 v_θ,形状 [len(step_set), B, AH, AD]。

    π0.5 的循环:dt=-1/K,t 从 1.0 起,每步 v_t = denoise_step(..., x_t, t);x_t += dt·v_t。
    step_set={0} ⇒ 只需 t=1.0、x_t=noise 的那一次 denoise_step(DRIFT 的默认与最优设定)。
    step_set 含更大的 k 时,前面的步必须按原式推进 x_t(用于消融 L_S)。
    """
    import torch
    state, prefix_pad_masks, past_key_values = prefix_and_cache(model, device, observation)
    bsize = observation.state.shape[0]
    dt = torch.tensor(-1.0 / NUM_DENOISE, dtype=torch.float32, device=device)
    x_t = noise
    t = torch.tensor(1.0, dtype=torch.float32, device=device)
    kmax = max(step_set)
    outs = {}
    for k in range(kmax + 1):
        v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, t.expand(bsize))
        if k in step_set:
            outs[k] = v_t
        if k < kmax:
            x_t = x_t + dt * v_t
            t = t + dt
    return torch.stack([outs[k] for k in sorted(step_set)])


def integrate_actions(model, device, observation, noise):
    """完整 10 步积分(诊断用:看第一步的速度场分歧传导了多少到最终动作)。"""
    import torch
    state, prefix_pad_masks, past_key_values = prefix_and_cache(model, device, observation)
    bsize = observation.state.shape[0]
    dt = torch.tensor(-1.0 / NUM_DENOISE, dtype=torch.float32, device=device)
    x_t, t = noise, torch.tensor(1.0, dtype=torch.float32, device=device)
    while t >= -dt / 2:
        v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, t.expand(bsize))
        x_t = x_t + dt * v_t
        t = t + dt
    return x_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="drift_pool", help="drift_pool_prep.py 的产物名")
    ap.add_argument("--iters", type=int, default=500, help="PGD 步数(论文 500)")
    ap.add_argument("--lr", type=float, default=0.01, help="PGD 步长 α(论文 0.01)")
    ap.add_argument("--batch", type=int, default=8, help="每步从观测池采样几帧")
    ap.add_argument("--steps-set", default="0", help="攻击哪些去噪步(论文最优=只 0);消融例 0,1,2")
    ap.add_argument("--pgd", default="sign", choices=["sign", "norm"],
                    help="sign=标准 sign 梯度 PGD(先用这个);norm=按梯度范数归一化步长")
    ap.add_argument("--init", default="gray", choices=["gray", "rand"], help="论文用灰色 0.5")
    ap.add_argument("--band", type=int, default=0,
                    help="优化N×N图案并双线性上采样到224；0=满分辨率。偏向低频，非严格傅里叶截止")
    ap.add_argument("--eot", action="store_true",
                    help="旧EOT：颜色affine＋Gaussian blur代理，参数来自--eot-fit；不含完整几何链")
    ap.add_argument("--eot-fit", default="drift_render_fit", help="拟合产物名")
    ap.add_argument("--eot-jitter", type=float, default=0.15,
                    help="对 (a,b,σ) 的相对抖动幅度 ⇒ 求期望而非单点,增加余量")
    ap.add_argument("--render-calibration", default="",
                    help="E17随机图案校准JSON：可微128纹理往返＋对齐＋颜色代理，与--eot互斥")
    ap.add_argument("--render-stage", choices=["resample", "resample_align", "full"], default="full",
                    help="校准代理的消融阶段；full包含重采样、子像素对齐和颜色")
    ap.add_argument("--render-shift-jitter", type=float, default=0.25,
                    help="对校准位移加±N像素均匀抖动；预设实验范围，不是实测置信区间")
    ap.add_argument("--render-seed", type=int, default=20261005)
    ap.add_argument("--tag", default="drift")
    ap.add_argument("--diag-frames", type=int, default=8, help="诊断:算最终动作偏差用几帧")
    args = ap.parse_args()
    if args.eot and args.render_calibration:
        ap.error("--eot与--render-calibration互斥")
    if args.render_shift_jitter < 0 or args.eot_jitter < 0:
        ap.error("jitter范围必须非负")
    if args.render_calibration and (OUT / f"drift_patch_{args.tag}.npz").exists():
        ap.error("校准代理实验要求新的--tag，避免覆盖已有图案")
    step_set = tuple(int(v) for v in args.steps_set.split(","))

    import torch
    import jax
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.models import model as _model

    d = np.load(OUT / f"{args.pool}.npz", allow_pickle=True)
    IMG, WRI, ST = d["clean_img224"], d["clean_wrist224"], d["clean_state8"]
    MASKS, munion, quad = d["masks"], d["munion"], d["quad224"]
    prompt = str(d["prompt"])
    F = IMG.shape[0]
    calibrated_proxy = None
    render_rng = np.random.RandomState(args.render_seed)
    if args.render_calibration:
        from render_training_proxy import RenderTrainingProxy
        # Validate projected geometry before loading policy weights. A 26cm
        # base calibration cannot silently be reused for 13cm or wrist.
        calibrated_proxy = RenderTrainingProxy(args.render_calibration, quad, "cpu", args.render_stage)
        print(f"[render proxy] {args.render_calibration} stage={args.render_stage}; "
              "候选固定base代理，尚非真实MuJoCo渲染", flush=True)
    px = MASKS.reshape(F, -1).sum(1)
    vis = np.where(px > 0)[0]          # 完全被遮挡的帧对 loss 无贡献,直接排除
    print(f"[pool] F={F} 帧  足迹 union={int(munion.sum())}px  "
          f"逐帧 min/中位/max={px.min()}/{int(np.median(px))}/{px.max()}  "
          f"可用帧(未全遮挡)={len(vis)}", flush=True)
    print(f"[cfg] patch_m={float(d['patch_m']):.2f}m  cell=({d['cell_xy'][0]:+.4f},{d['cell_xy'][1]:+.4f})  "
          f"attack_steps={step_set}  PGD={args.pgd} lr={args.lr} iters={args.iters} batch={args.batch}",
          flush=True)

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    model = policy._model                                    # noqa: SLF001
    device = policy._pytorch_device                          # noqa: SLF001
    if calibrated_proxy is not None:
        calibrated_proxy = RenderTrainingProxy(args.render_calibration, quad, device, args.render_stage)
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[policy] loaded device={device}", flush=True)

    def build_obs(k):
        el = {"observation/image": IMG[k], "observation/wrist_image": WRI[k],
              "observation/state": ST[k], "prompt": prompt}
        inputs = policy._input_transform(jax.tree.map(lambda x: x, el))   # noqa: SLF001
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    # 固定 A^{τ(0)}:干净项与贴纸项必须喂完全相同的噪声,差值才纯粹由图案引起
    noise = torch.from_numpy(
        np.random.RandomState(SEED_NOISE).standard_normal((AH, AD)).astype(np.float32)
    ).to(device)[None]

    # ---- 红线:同帧同噪声两次前向逐位一致 ----
    with torch.no_grad():
        v1 = velocity_at_steps(model, device, build_obs(int(vis[0])), noise.clone(), step_set)
        v2 = velocity_at_steps(model, device, build_obs(int(vis[0])), noise.clone(), step_set)
    rep = float((v1 - v2).abs().max())
    print(f"[红线] 同帧同噪声重复前向 max|Δv| = {rep:.3e} {'OK' if rep == 0.0 else '必须为 0,停'}",
          flush=True)
    assert rep == 0.0, "速度场前向不确定,先查随机源"

    # ---- 干净项 v_θ(A^{τ(0)}, o) 与图案无关 ⇒ 每帧预计算一次并缓存 ----
    print(f"[cache] 预计算 {len(vis)} 帧的干净速度场…", flush=True)
    t0 = time.monotonic()
    v_clean = {}
    base_clean = {}
    mask_t = {}
    for i, k in enumerate(vis):
        k = int(k)
        o = build_obs(k)
        with torch.no_grad():
            v_clean[k] = velocity_at_steps(model, device, o, noise.clone(), step_set).detach()
        # 缓存放 CPU:628 帧 × (600KB 图 + 200KB mask) ≈ 0.5GB,留给前向的显存更宽裕
        base_clean[k] = o.images[BASE_KEY].detach().clone().cpu()    # [1,3,224,224] in [-1,1]
        mask_t[k] = torch.from_numpy(MASKS[k].astype(np.float32))[None, None]
        if i % 100 == 0:
            print(f"  {i}/{len(vis)}  {(time.monotonic()-t0)/60:.1f}min", flush=True)
    print(f"[cache] 完成 {(time.monotonic()-t0)/60:.1f}min", flush=True)

    munion_t = torch.from_numpy(munion.astype(np.float32)).to(device)[None, None]

    # ---- 图案在 [0,1] 像素空间;喂模型前 → [-1,1] ----
    # --band N:优化变量 Q 是 N×N,P = 双线性上采样(Q) → 224；偏向低频，非严格带限。
    B = args.band if args.band > 0 else 224
    if args.init == "gray":
        Q = torch.full((1, 3, B, B), 0.5, device=device)
    else:
        Q = torch.rand(1, 3, B, B, device=device)
    Q = Q.requires_grad_(True)

    def patch_from(q):
        """低分辨率变量 → 224 图像空间图案。B=224 时是直通。"""
        if B == 224:
            return q
        return torch.nn.functional.interpolate(q, size=(224, 224), mode="bilinear",
                                               align_corners=False)

    # 梯度掩膜:满分辨率时按 munion 逐像素掩;带限时按"该低频格里是否含足迹像素"掩
    # (完全落在足迹外的格子本来梯度就是 0,这里只是显式化)。
    if B == 224:
        gmask = munion_t
    else:
        gmask = (torch.nn.functional.adaptive_max_pool2d(munion_t, (B, B)) > 0).float()
    print(f"[param] 带限 B={B}({'满分辨率' if B == 224 else f'{B}×{B} → 上采样 224'})  "
          f"可更新自由度={int(gmask.sum())*3}", flush=True)

    # ---- EOT:渲染代理变换 render(P) ≈ a⊙blur_σ(P)+b,可微(高斯用 separable conv2d) ----
    eot_fit = None
    if args.eot:
        f = np.load(OUT / f"{args.eot_fit}.npz", allow_pickle=True)
        eot_fit = (float(f["sigma"]), f["a"].astype(np.float32), f["b"].astype(np.float32))
        print(f"[eot] 代理变换 σ={eot_fit[0]:.2f}  a={np.round(eot_fit[1],3)}  "
              f"b={np.round(eot_fit[2],3)}  (拟合 RMSE={float(f['rmse']):.4f} 相关={float(f['corr']):+.3f})"
              f"  抖动 ±{args.eot_jitter:.0%}", flush=True)

    def gauss_blur(x, sigma):
        if sigma <= 1e-3:
            return x
        r = max(1, int(3 * sigma))
        t = torch.arange(-r, r + 1, dtype=torch.float32, device=x.device)
        k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
        k = k / k.sum()
        x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1).expand(3, 1, 1, -1),
                                       padding=(0, r), groups=3)
        return torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1).expand(3, 1, -1, 1),
                                          padding=(r, 0), groups=3)

    def render_proxy(Pc, jitter):
        """把图案过一遍渲染代理;jitter=True 时对 (a,b,σ) 抖动做期望。"""
        if calibrated_proxy is not None:
            return calibrated_proxy(Pc, rng=render_rng if jitter else None,
                                    color_jitter=args.eot_jitter if jitter else 0.,
                                    shift_jitter_px=args.render_shift_jitter if jitter else 0.)
        if eot_fit is None:
            return Pc
        s, a, b = eot_fit
        if jitter and args.eot_jitter > 0:
            j = args.eot_jitter
            s = s * float(np.random.uniform(1 - j, 1 + j))
            a = a * np.random.uniform(1 - j, 1 + j, 3).astype(np.float32)
            b = b + np.random.uniform(-j * 0.1, j * 0.1, 3).astype(np.float32)
        at = torch.from_numpy(np.asarray(a)).to(Pc.device).view(1, 3, 1, 1)
        bt = torch.from_numpy(np.asarray(b)).to(Pc.device).view(1, 3, 1, 1)
        return (at * gauss_blur(Pc, s) + bt).clamp(0, 1)

    def loss_for(k, Pc, jitter=False):
        bc, mk = base_clean[k].to(device), mask_t[k].to(device)
        Pc = render_proxy(Pc, jitter)
        comp = bc * (1 - mk) + (Pc * 2 - 1) * mk
        o = build_obs(k)
        o.images[BASE_KEY] = comp
        v = velocity_at_steps(model, device, o, noise.clone(), step_set)
        return ((v - v_clean[k]) ** 2).sum()          # ||Δv||²₂,按论文对 step_set 求和

    def eval_L0(Pc, idx):
        tot = 0.0
        with torch.no_grad():
            for k in idx:
                tot += float(loss_for(int(k), Pc.clamp(0, 1)))
        return tot / max(len(idx), 1)

    rng = np.random.RandomState(0)
    diag_idx = vis[rng.choice(len(vis), min(args.diag_frames, len(vis)), replace=False)]
    L_init = eval_L0(patch_from(Q), diag_idx)
    with torch.no_grad():
        L_rand = eval_L0(torch.rand(1, 3, 224, 224, device=device), diag_idx)
    print(f"[init] L₀(灰初始)={L_init:.5e}   L₀(满分辨率随机图案参照)={L_rand:.5e}", flush=True)

    hist = []
    t0 = time.monotonic()
    for it in range(args.iters):
        batch = vis[rng.choice(len(vis), min(args.batch, len(vis)), replace=False)]
        if Q.grad is not None:
            Q.grad = None
        tot = 0.0
        for k in batch:
            L = loss_for(int(k), patch_from(Q).clamp(0, 1), jitter=True)
            (-L).backward()                            # 最大化 L₀
            tot += float(L)
        tot /= len(batch)
        with torch.no_grad():
            g = Q.grad * gmask                         # 只更新落在足迹上的自由度
            if args.pgd == "sign":
                Q -= args.lr * g.sign()                # g 是 -L 的梯度 ⇒ 减号即上升 L
            else:
                Q -= args.lr * g / (g.norm() + 1e-12) * g.numel() ** 0.5
            Q.clamp_(0, 1)
        hist.append(tot)
        if it % 25 == 0 or it == args.iters - 1:
            print(f"  [it {it:3d}] L₀(batch)={tot:.5e}  ×{tot/max(L_init,1e-30):.1f} vs 灰初始  "
                  f"{(time.monotonic()-t0)/60:.1f}min  显存 {torch.cuda.max_memory_allocated()/2**30:.1f}GB",
                  flush=True)

    P = patch_from(Q).detach()                     # 带限时:低分辨率变量 → 224 图案
    L_final = eval_L0(P, diag_idx)
    print(f"[done] L₀: 灰初始 {L_init:.5e} → 最终 {L_final:.5e}  "
          f"(×{L_final/max(L_init,1e-30):.1f});随机图案参照 {L_rand:.5e}", flush=True)

    # ---- 诊断:第一步速度场分歧传导到最终动作多少(复核论文的梯度冲突论证) ----
    dev = []
    with torch.no_grad():
        for k in diag_idx:
            k = int(k)
            a_c = integrate_actions(model, device, build_obs(k), noise.clone())
            o = build_obs(k)
            bc, mk = base_clean[k].to(device), mask_t[k].to(device)
            # E17 run: report final action difference under the same nominal
            # proxy as L0. Historical runs retain their digital diagnostic.
            diagnostic_P = render_proxy(P.clamp(0, 1), False) if calibrated_proxy is not None else P.clamp(0, 1)
            o.images[BASE_KEY] = bc * (1 - mk) + (diagnostic_P * 2 - 1) * mk
            a_p = integrate_actions(model, device, o, noise.clone())
            dev.append(float((a_p - a_c).norm()))
    print(f"[诊断] 最终动作偏差 ‖Δa‖:中位 {np.median(dev):.4f}  max {max(dev):.4f}  "
          f"(在 {len(dev)} 帧上;看第一步分歧传导了多少)", flush=True)

    Pn = P.detach().cpu().numpy()[0]
    outp = OUT / f"drift_patch_{args.tag}.npz"
    np.savez_compressed(outp, P=Pn, munion=munion, masks_px=px, quad224=quad,
                        cell_xy=d["cell_xy"], cell_world=d["cell_world"], patch_m=d["patch_m"],
                        step_set=np.array(step_set), pgd=args.pgd, lr=args.lr, iters=args.iters,
                        batch=args.batch, init=args.init, hist=np.array(hist),
                        L_init=L_init, L_final=L_final, L_rand=L_rand,
                        act_dev=np.array(dev), pool=args.pool, seed_noise=SEED_NOISE,
                        band=B, Q=Q.detach().cpu().numpy()[0], eot=bool(args.eot),
                        eot_jitter=args.eot_jitter,
                        render_calibration=args.render_calibration,
                        render_config_json="" if calibrated_proxy is None else json.dumps(calibrated_proxy.config, sort_keys=True),
                        render_stage=args.render_stage if calibrated_proxy is not None else "",
                        render_shift_jitter=args.render_shift_jitter,
                        render_seed=args.render_seed,
                        act_dev_input="nominal_render_proxy" if calibrated_proxy is not None else "digital")
    print(f"[written] {outp}", flush=True)
    print("[next] 反 warp 成方形纹理 → 数字合成评测(评测 A)→ 物理纹理评测(评测 B),见 HANDOFF 4.6",
          flush=True)


if __name__ == "__main__":
    main()
