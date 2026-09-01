#!/usr/bin/env python3
"""G0 续跑:补 OOM 前没跑完的 G0.3(3ε 稳定性)+ 方向性差分诊断(钉死 G0.2 失败原因)。

第一次运行(g0_gradcheck.py)已完成并记录:红线 0 ✅ / G0.1 ✅ / user-L≡0 演示 ✅ /
G0.2 逐像素中心差分全 ~100% ❌ / NaN=0、量级。死因:算第二条 ε 的图时别的进程
吃掉 GPU 0 的 30 GB → OOM。本脚本只补:
  a) 方向性差分:u = g_0/‖g_0‖(整图方向),数值 (s_c(x+hu)−s_c(x−hu))/2h vs
     解析 ⟨g_c,u⟩,h ∈ {0.01,0.03,0.1,0.3}。单像素扰动的效应(~1e-4)低于 bf16
     前向量化地板,整图方向把效应放大 ~‖g‖/max|g| 倍;若大 h 吻合、小 h 崩,
     即证明"梯度本身对、逐像素差分死于 bf16 地板"。
  b) bf16 输出噪声地板实测:5 个随机单位方向 r,|s_c(x+h·r)−s_c(x)| @ h=0.01
     (⟨g,r⟩ 期望 ~‖g‖/√N,可忽略)⇒ 输出对"无信息扰动"的响应量级。
  c) 3 条 ε 的梯度显著性图两两秩相关(每条算完立刻释放图,防显存爬升)。

用法:
    CUDA_VISIBLE_DEVICES=0 /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/g0b_stability.py
"""
import pathlib
import sys
import time

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
sys.path.insert(0, str(ROOT / "pi05probe"))

from g0_gradcheck import (  # noqa: E402  (导入时已挂好 openpi 的 sys.path 与环境变量)
    AH, AD, EX, FR, SEED_EPS, OUT, GOUT, TORCH_CKPT, sample_actions_grad,
)
import numpy as np  # noqa: E402

HS_DIR = (0.01, 0.03, 0.1, 0.3)
NRAND = 5


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

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    model = policy._model
    device = policy._pytorch_device
    for p in model.parameters():
        p.requires_grad_(False)
    out(f"[policy] loaded  frame {FR}  prompt={prompt!r}")

    def build_obs():
        el = {"observation/image": cimg, "observation/wrist_image": cwri,
              "observation/state": cstate, "prompt": prompt}
        inputs = jax.tree.map(lambda x: x, el)
        inputs = policy._input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def eps_for(seed):
        return torch.from_numpy(
            np.random.RandomState(seed).standard_normal((AH, AD)).astype(np.float32)
        ).to(device)[None]

    def grads_for_eps(seed):
        """返回 g_ch [3ch,3rgb,224,224];算完立刻释放图。"""
        o = build_obs()
        lf = o.images["base_0_rgb"].clone().detach().requires_grad_(True)
        o.images["base_0_rgb"] = lf
        with torch.enable_grad():
            a = sample_actions_grad(model, device, o, eps_for(seed))
        gs = []
        for c in range(3):
            a[0, :EX, c].sum().backward(retain_graph=(c < 2))
            gs.append(lf.grad.detach().clone()[0])
            lf.grad = None
        del a, o, lf
        torch.cuda.empty_cache()
        return torch.stack(gs)                      # [3,3,224,224]

    base_img = build_obs().images["base_0_rgb"]
    eps0 = eps_for(SEED_EPS)

    def fwd_sc(img_tensor):
        o = build_obs()
        o.images["base_0_rgb"] = img_tensor
        with torch.no_grad():
            a = sample_actions_grad(model, device, o, eps0.clone())
        return a[0, :EX, :3].sum(0)                 # [3]

    # ---------- a) 方向性差分 ----------
    t0 = time.monotonic()
    g = grads_for_eps(SEED_EPS)                     # ε₀ 解析梯度(与第一次运行同配方)
    out(f"[grad] ε₀ 三通道 backward 完成({time.monotonic()-t0:.0f}s)")
    u = (g[0] / g[0].norm())[None]                  # 方向 = g_0 归一化,[1,3,224,224]
    ana_dir = [float((g[c] * u[0]).sum()) for c in range(3)]   # ⟨g_c,u⟩
    x0 = base_img.clone().detach()
    out("\n== 方向性差分(整图方向 u = g_0/‖g_0‖;单像素效应会被 bf16 地板吃掉,整图不会) ==")
    out(f"  解析方向导数 ⟨g_c,u⟩ = {ana_dir[0]:+.4e} / {ana_dir[1]:+.4e} / {ana_dir[2]:+.4e}")
    out("  h        s_0 数值/相对误差      s_1 数值/相对误差      s_2 数值/相对误差")
    dir_table = []
    for h in HS_DIR:
        num = ((fwd_sc(x0 + h * u) - fwd_sc(x0 - h * u)) / (2 * h)).cpu().numpy()
        errs = [abs(num[c] - ana_dir[c]) / max(abs(num[c]), abs(ana_dir[c]), 1e-12)
                for c in range(3)]
        dir_table.append([h, *num.tolist(), *errs])
        out(f"  {h:<7g} " + "  ".join(f"{num[c]:+.3e}/{errs[c]*100:6.1f}%" for c in range(3)))

    # ---------- b) bf16 输出噪声地板 ----------
    rng = np.random.RandomState(123)
    s0 = fwd_sc(x0).cpu().numpy()
    noise = []
    for _ in range(NRAND):
        r = torch.from_numpy(rng.standard_normal(x0.shape).astype(np.float32)).to(device)
        r = r / r.norm()
        noise.append(np.abs((fwd_sc(x0 + 0.01 * r).cpu().numpy() - s0)))
    noise = np.array(noise)                         # [NRAND,3]
    out(f"\n== bf16 输出噪声地板(‖扰动‖=0.01 的随机方向,|Δs_c|) ==")
    out(f"  s_c 本身量级: {np.abs(s0).round(3)}")
    out(f"  |Δs| 中位: {np.median(noise, 0)}  最大: {noise.max(0)}")
    out(f"  对比:单像素 h=0.01 的预期信号 |g|·2h ≤ {float(g.abs().max())*0.02:.1e}"
        f" —— 低于上面地板即解释 G0.2 逐像素全挂")

    # ---------- c) 3ε 稳定性 ----------
    maps = [g.norm(dim=(0, 1)).cpu().numpy()]
    for k in (1, 2):
        maps.append(grads_for_eps(SEED_EPS + k).norm(dim=(0, 1)).cpu().numpy())
        out(f"[grad] ε{k} 完成")

    def spear(a, b):
        ra = np.argsort(np.argsort(a.flatten())).astype(float)
        rb = np.argsort(np.argsort(b.flatten())).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    rs = [spear(maps[0], maps[1]), spear(maps[0], maps[2]), spear(maps[1], maps[2])]
    out(f"\n== 3ε 稳定性 == 两两秩相关 {rs[0]:+.3f} / {rs[1]:+.3f} / {rs[2]:+.3f}"
        f"(= 梯度显著性图自身的 ε 噪声地板)")
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    out(f"[mem] 本次显存峰值 {peak_gb:.1f} GB")

    np.savez_compressed(GOUT / "g0_gradcheck.npz",
                        frame=FR, eps_seed=SEED_EPS,
                        g_ch=g.cpu().numpy(),
                        gmag_eps=np.stack(maps),
                        dir_table=np.array(dir_table),
                        ana_dir=np.array(ana_dir),
                        noise_floor=noise, s0=s0,
                        spearman_eps=np.array(rs))
    (GOUT / "g0b_output.txt").write_text("\n".join(lines) + "\n")
    print(f"[written] {GOUT/'g0_gradcheck.npz'} + g0b_output.txt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
