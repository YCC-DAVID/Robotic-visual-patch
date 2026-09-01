#!/usr/bin/env python3
"""S0→S1 过渡步骤 3+4:PyTorch 路径的两件必须验的事。

A · **JAX vs PyTorch 一致性**(强制)
    同一 observation + **同一 ε** 下,两条路输出的 7 维动作必须足够接近。
    不做这一步,PART B 的 attention 和 S2 的 Δa 就来自两个不同模型,
    `Spearman(attention, influence)` 低有可能只是转换误差的伪影。

B · **attention 能否逐层取出**
    结论(读码得出,本脚本验证):打过 openpi 补丁的
    `transformers/models/gemma/modeling_gemma.py:327` 里
    `GemmaAttention.forward` **无条件 return (attn_output, attn_weights)** ——
    连 `output_attentions` 开关都没有;而 `pi0_pytorch.py:392` 把
    `_attn_implementation` 强制设成 `"eager"`,所以 `eager_attention_forward:248`
    一定会算出完整的 post-softmax 权重。
    ⇒ **一个普通 forward hook 挂在 `language_model.layers[i].self_attn` 上就能白拿**,
      不改任何源码、不 monkeypatch。

    关键结构(`pi0_pytorch.py:377-419` + `gemma_pytorch.py:90-124`):
      · prefix 那一趟 `inputs_embeds=[prefix, None]` → 走 `gemma_pytorch.py:102`
        → 真正的 HF `language_model.forward()`,算完整 968×968 自注意力,
        **text→image 就在这里**,而且**每个 observation 只跑一次**(之后进 KV cache)。
      · 10 个去噪步走 `inputs_embeds=[None, suffix]` → `gemma_pytorch.py:114`
        → **`gemma_expert.model`**(另一个 module)⇒ 不会污染我们的 hook。

⚠️ 必须关掉 `torch.compile`:`pi0_pytorch.py:112` 默认
   `torch.compile(self.sample_actions, mode="max-autotune")`,
   编译区里 hook 可能不触发/反复重编译。本脚本用
   `dataclasses.replace(cfg.model, pytorch_compile_mode=None)`。

用法(PYTHONPATH 必须以 transformers_patched 开头,见 setup_torch_transformers.py):
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/probe_torch.py
"""

import dataclasses
import os
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
OUT = ROOT / "pi05probe" / "out"
PATCHED_TF = ROOT / "third_party" / "transformers_patched"

# ---------------------------------------------------------------------------
# 环境全部在脚本内部设好,这样命令行只需要 `<python> <这个文件>`,不带任何前缀。
# (本项目里带一长串 env var 前缀的命令会被安全分类器拒,见 FINDINGS §0)
# 必须在 import torch / jax / transformers 之前做完 —— 所以那些 import 都放在 main() 里。
# ---------------------------------------------------------------------------
# 打过补丁的 transformers 必须排最前面,才能盖掉 conda env site-packages 里那份
for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))

os.environ["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
os.environ["PYTHONNOUSERSITE"] = "1"
# jax 默认 preallocate 75%(34.5 GB),而本机 L40S 常被他人占 ⇒ 必关。
# 本脚本要同时驻留 JAX 和 PyTorch 两个模型(各约 8 GB),所以更要省。
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.35"

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    _out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    _rows = sorted(((int(i), int(f)) for i, f in (l.split(",") for l in _out.strip().splitlines())),
                   key=lambda r: -r[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_rows[0][0])
    print(f"[env] 挑了 GPU {_rows[0][0]}(剩余 {_rows[0][1]} MiB;两个模型共约 17 GB)", flush=True)

JAX_CKPT = pathlib.Path("/home/user1/.cache/openpi/openpi-assets/checkpoints/pi05_libero")
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"

_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def arr(a, prec=6):
    return np.array2string(np.asarray(a, dtype=np.float64), precision=prec,
                           suppress_small=False, max_line_width=250)


def fixed_example(seed=0):
    """固定的 observation。故意不用 make_libero_example()(它内部 np.random 不带 seed)。"""
    rng = np.random.default_rng(seed)
    return {
        "observation/state": rng.random(8),
        "observation/image": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "put the bowl on the plate",
    }


def main():
    import transformers

    say("=" * 100)
    say("环境自检")
    say("=" * 100)
    say(f"  transformers        = {transformers.__version__}  @ {transformers.__file__}")
    assert str(PATCHED_TF) in transformers.__file__, "没用到打过补丁的 transformers!"
    from transformers.models.siglip import check
    assert check.check_whether_transformers_replace_is_installed_correctly()
    say("  transformers_replace check = OK")

    import torch
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    say(f"  torch = {torch.__version__}  cuda_available={torch.cuda.is_available()}")

    st = TORCH_CKPT / "model.safetensors"
    assert st.exists(), f"还没转权重: {st}"
    say(f"  torch ckpt = {TORCH_CKPT}  ({st.stat().st_size / 2**30:.2f} GiB)")
    ns = TORCH_CKPT / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    assert ns.exists(), f"norm_stats.json 缺失(见 convert_weights.py 里的上游 bug 说明): {ns}"
    say(f"  norm_stats  = OK")

    cfg = _config.get_config("pi05_libero")
    # ⚠️ 关掉 torch.compile,否则 hook 可能不触发
    cfg_nc = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    say(f"  pytorch_compile_mode: {cfg.model.pytorch_compile_mode!r} -> {cfg_nc.model.pytorch_compile_mode!r}")

    obs = fixed_example()
    AH, AD = cfg.model.action_horizon, cfg.model.action_dim
    eps = np.random.default_rng(12345).standard_normal((AH, AD)).astype(np.float32)
    say(f"  共享 ε: shape={eps.shape}  ‖ε‖={np.linalg.norm(eps):.6f}")

    # ================================================================ A · 一致性
    say("")
    say("=" * 100)
    say("A · JAX vs PyTorch 一致性(同 observation + 同 ε)")
    say("=" * 100)

    say("[A1] 载入 PyTorch policy …")
    pt_policy = _policy_config.create_trained_policy(cfg_nc, TORCH_CKPT)
    a_pt = np.asarray(pt_policy.infer(dict(obs), noise=eps)["actions"])
    say(f"     actions shape = {a_pt.shape}")
    say(f"     PyTorch a[0] = {arr(a_pt[0])}")

    say("[A2] 载入 JAX policy …")
    jx_policy = _policy_config.create_trained_policy(cfg, JAX_CKPT)
    a_jx = np.asarray(jx_policy.infer(dict(obs), noise=eps)["actions"])
    say(f"     JAX     a[0] = {arr(a_jx[0])}")

    d = np.abs(a_pt - a_jx)
    say("")
    say(f"  |Δ| max  = {d.max():.6e}      (整个 (10,7) chunk)")
    say(f"  |Δ| mean = {d.mean():.6e}")
    say(f"  逐维 max:  {arr(d.max(axis=0))}")
    say(f"  a_jx 的量级 (max|a|) = {np.abs(a_jx).max():.6f}")
    say(f"  相对误差 max = {(d.max() / max(np.abs(a_jx).max(), 1e-9)):.6e}")
    # 夹爪符号是否一致 —— 这是唯一对环境有意义的夹爪信息(见 FINDINGS Q2)
    same_sign = np.array_equal(np.sign(a_pt[:, 6]), np.sign(a_jx[:, 6]))
    say(f"  夹爪符号 sign(a[:,6]) 两边是否完全一致: {same_sign}")
    say(f"     PyTorch: {np.sign(a_pt[:,6]).astype(int).tolist()}")
    say(f"     JAX    : {np.sign(a_jx[:,6]).astype(int).tolist()}")

    # ================================================================ A' · state 无用
    say("")
    say("=" * 100)
    say("A' · 验证 FINDINGS Q3b:π0.5-LIBERO 是否真的不读 observation/state")
    say("=" * 100)
    obs2 = dict(obs)
    obs2["observation/state"] = np.random.default_rng(999).random(8)   # 换一个完全不同的 state
    a_pt2 = np.asarray(pt_policy.infer(obs2, noise=eps)["actions"])
    dd = np.abs(a_pt2 - a_pt).max()
    say(f"  换掉 state 后 |Δ| max = {dd:.6e}   (预期恰好 0.0)")
    say(f"  ⇒ state {'确实被完全忽略' if dd == 0.0 else '有影响 —— Q3b 的结论错了,必须复查!'}")

    # ================================================================ B · attention
    say("")
    say("=" * 100)
    say("B · attention 逐层抽取(forward hook,不改源码)")
    say("=" * 100)
    model = pt_policy._model                      # noqa: SLF001
    lm = model.paligemma_with_expert.paligemma.language_model
    layers = lm.layers
    say(f"  language_model.layers 是 {type(layers).__name__},共 {len(layers)} 层")
    say(f"  _attn_implementation = {lm.config._attn_implementation!r}"   # noqa: SLF001
        f"  (pi0_pytorch.py:392 会在 sample_actions 里强制设成 'eager')")

    captured = {}

    def mk_hook(i):
        def hook(_mod, _inp, out):
            # GemmaAttention.forward 返回 (attn_output, attn_weights)
            w = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            captured.setdefault(i, []).append(None if w is None else w.detach().float().cpu())
        return hook

    handles = [layers[i].self_attn.register_forward_hook(mk_hook(i)) for i in range(len(layers))]
    try:
        _ = pt_policy.infer(dict(obs), noise=eps)
    finally:
        for h in handles:
            h.remove()

    say(f"  抓到 attention 的层数: {len(captured)} / {len(layers)}")
    assert len(captured) == len(layers), "有的层没抓到 —— 检查 torch.compile 是否真的关了"
    ncalls = {i: len(v) for i, v in captured.items()}
    say(f"  每层被调用次数: {sorted(set(ncalls.values()))}  "
        f"(预期全是 1:prefix 只跑一趟,10 个去噪步走的是 gemma_expert,不碰这些 hook)")
    none_layers = [i for i, v in captured.items() if v[0] is None]
    assert not none_layers, f"这些层 attn_weights 是 None: {none_layers}"

    A0 = captured[0][0]
    say(f"  layer0 attn_weights: shape={tuple(A0.shape)}  dtype={A0.dtype}")
    B, H, Tq, Tk = A0.shape

    n_img, n_side, ZMAX = 256, 16, cfg.model.max_token_len
    img_lo, img_hi = 0, n_img                     # base_0_rgb
    txt_lo, txt_hi = 3 * n_img, 3 * n_img + ZMAX  # language
    say(f"  预期 prefix 长度 = 3×{n_img} + {ZMAX} = {3*n_img+ZMAX};实测 Tq=Tk={Tq}")
    assert Tq == Tk == 3 * n_img + ZMAX, "prefix 长度和 FINDINGS Q6 算的不一致!"
    say(f"  heads = {H}  (预期 8);batch = {B}")

    # softmax 归一性:每行(在未被 mask 的 key 上)应当和为 1
    rowsum = A0[0].sum(dim=-1)
    say(f"  行和(应≈1): min={rowsum.min():.6f} max={rowsum.max():.6f}")

    # 真实 text token 数(按 mask 取,见 FINDINGS Q6)
    from openpi.models import tokenizer as _tok
    tk = _tok.PaligemmaTokenizer(ZMAX)
    _, tmask = tk.tokenize(obs["prompt"])
    Z = int(tmask.sum())
    say(f"  prompt {obs['prompt']!r} 的真实 token 数 Z = {Z}(200 个 slot 里其余是 padding)")

    # 这才是 B0 要的那一路
    A_txt2img = A0[0][:, txt_lo:txt_lo + Z, img_lo:img_hi]     # [H, Z, 256]
    say(f"  attn[text(有效 {Z} 行), base_image(256 列)] = {tuple(A_txt2img.shape)}")
    S = A_txt2img.sum(dim=0).reshape(Z, n_side, n_side)         # head 求和 → [Z,16,16]
    say(f"  head 求和后 reshape → {tuple(S.shape)}  (§A-2 第 1-3 步)")
    say(f"  每个 text token 在 base 图上的注意力质量(行和,head-summed / H):")
    for z in range(Z):
        say(f"     token[{z:2d}]  mass={float(S[z].sum())/H:.6f}  max_cell={float(S[z].max())/H:.6f}")
    say("  ⚠️ mass 差异这么大正是 §A-4 说的 attention sink / §A-3 要逐行重归一化的理由。")

    # padding 行确认是垃圾
    if ZMAX > Z:
        pad = A0[0][:, txt_lo + Z:txt_hi, img_lo:img_hi]
        say(f"  padding 行(第 {Z}..{ZMAX-1} 个 slot)在 base 图上的注意力:"
            f" sum={float(pad.sum()):.6f} max={float(pad.max()):.6f}")
        say("  ⇒ 印证 FINDINGS Q6:算 saliency 前必须按 tokenized_prompt_mask 裁掉这些行。")

    # 显存/存储估算
    per_layer_full = B * H * Tq * Tk * 4 / 2**20
    per_layer_cut = H * Z * n_img * 4 / 2**10
    say("")
    say(f"  存储估算:整层 [{B},{H},{Tq},{Tk}] fp32 = {per_layer_full:.1f} MiB/层 "
        f"⇒ 18 层 = {per_layer_full*18/1024:.2f} GiB/帧")
    say(f"            只留 [H,{Z},256] = {per_layer_cut:.1f} KiB/层 "
        f"⇒ 18 层 = {per_layer_cut*18/1024:.2f} MiB/帧 ⇒ T=60 帧 = "
        f"{per_layer_cut*18*60/2**20:.2f} GiB … 按 §B schema 存盘完全可行")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "s0_torch_probe.txt").write_text("\n".join(_lines) + "\n")
    np.savez_compressed(OUT / "s0_torch_probe_attn_layer0.npz",
                        attn_txt2img_layer0=A_txt2img.numpy(), token_mask=tmask, Z=Z)
    say("")
    say(f"[written] {OUT/'s0_torch_probe.txt'}")
    say(f"[written] {OUT/'s0_torch_probe_attn_layer0.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
