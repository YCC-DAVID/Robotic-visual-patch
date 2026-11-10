#!/usr/bin/env python3
"""destination token 通路的梯度(用户点名):像素梯度只允许从指定 prefix 位置的
KV 流回,其余位置的 KV detach ⇒ 分解"梯度信号走的是哪些 token 的路"。

原理:denoise(action expert)只通过 prefix 的 KV cache 看图像/语言
(modeling_gemma.py:309-310 纯读 cat,不回写);把 cache 里除保留位置外的
K/V 换成 `t*m + t.detach()*(1-m)`(前向数值逐位不变,反向只剩保留位置的路)。
像素梯度只能经 cache 流回 ⇒ 这是干净的通路分解。

变体:full(全通路,应复现 g0c 帧0)/ dest(只 "plate")/ src(只 "bowl")/
lang(全部语言位)/ img(全部图像位)。帧 0(初始观测),口径同 G0:
s_c = Σ_{k<EX} a[k,c] 三通道分开 backward,固定 ε,pooled 聚合,对全局 FD 秩相关。

用法:
    CUDA_VISIBLE_DEVICES=1 /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/g0d_desttoken.py
"""
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
sys.path.insert(0, str(ROOT / "pi05probe"))

from g0_gradcheck import (  # noqa: E402
    AH, AD, EX, SEED_EPS, OUT, GOUT, TORCH_CKPT,
)
import numpy as np  # noqa: E402

FR = 0


def prefix_forward(model, device, observation):
    """sample_actions(:384-400)的 prefix 段,返回 (state, prefix_pad_masks, cache)。"""
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

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
    return state, prefix_pad_masks, past_key_values


def euler(model, device, state, prefix_pad_masks, past_key_values, noise, num_steps=10):
    """sample_actions(:402-420)的 Euler 段,逐字。"""
    import torch

    bsize = state.shape[0]
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    t = torch.tensor(1.0, dtype=torch.float32, device=device)
    while t >= -dt / 2:
        v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, t.expand(bsize))
        x_t = x_t + dt * v_t
        t = t + dt
    return x_t


def mask_cache_grad(cache, keep, seq, device, dtype):
    """cache 中除 keep 位置外的 K/V 全 detach(前向数值不变)。"""
    import torch

    m = torch.zeros(seq, device=device, dtype=dtype)
    m[keep] = 1.0
    m = m[None, None, :, None]
    for li in range(len(cache.key_cache)):
        k, v = cache.key_cache[li], cache.value_cache[li]
        cache.key_cache[li] = k * m + k.detach() * (1.0 - m)
        cache.value_cache[li] = v * m + v.detach() * (1.0 - m)


def main():
    import torch
    import jax
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.models import model as _model
    from openpi.models.tokenizer import PaligemmaTokenizer

    d = np.load(OUT / "s2f_scan_obs.npz", allow_pickle=True)
    za = np.load(OUT / "s2f_actions.npz", allow_pickle=True)
    prompt = str(d["prompt"])

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    model = policy._model
    device = policy._pytorch_device
    for p in model.parameters():
        p.requires_grad_(False)

    el = {"observation/image": d["clean_img224"][FR],
          "observation/wrist_image": d["clean_wrist224"][FR],
          "observation/state": d["clean_state8"][FR], "prompt": prompt}
    inputs_np = policy._input_transform(jax.tree.map(lambda x: x, el))
    tok = np.asarray(inputs_np["tokenized_prompt"])
    tmask = np.asarray(inputs_np["tokenized_prompt_mask"])
    sp = PaligemmaTokenizer()._tokenizer
    pieces = [sp.id_to_piece(int(t)) if tmask[j] else "<pad>" for j, t in enumerate(tok)]
    dest_j = [j for j, p in enumerate(pieces) if p == "▁plate"]
    src_j = [j for j, p in enumerate(pieces) if p == "▁bowl"]
    lang_j = [j for j, p in enumerate(pieces) if tmask[j]]
    print(f"[tok] 语言段前 20 pieces: {pieces[:20]}", flush=True)
    print(f"[tok] dest('▁plate') 位置 {dest_j}  src('▁bowl') 位置 {src_j}  "
          f"语言 token 总数 {len(lang_j)}", flush=True)
    assert dest_j and src_j

    def build_obs():
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs_np)
        return _model.Observation.from_dict(inputs)

    eps = torch.from_numpy(
        np.random.RandomState(SEED_EPS).standard_normal((AH, AD)).astype(np.float32)
    ).to(device)[None]

    # 贴纸掩码(帧 0)与全局 FD
    clean = d["clean_img224"][FR].astype(np.int16)
    M = int(d["M"])
    smasks = [(np.abs(d["patched_img224"][i, FR].astype(np.int16) - clean) > 2).any(-1)
              for i in range(M)]
    Ac, Ap = za["A_clean"], za["A_patched"]
    dd = Ap[:, :, :EX, 0:3] - Ac[None, :, :EX, 0:3]
    fd_global = (np.linalg.norm(dd.reshape(M, dd.shape[1], -1), axis=2) * 50.0).mean(1)

    def rank(a):
        return np.argsort(np.argsort(a)).astype(float)

    def spear(a, b):
        return float(np.corrcoef(rank(a), rank(b))[0, 1])

    a_ref = None
    results, gmags = {}, {}
    seq = None
    for name in ["full", "dest", "src", "lang", "img"]:
        o = build_obs()
        lf = o.images["base_0_rgb"].clone().detach().requires_grad_(True)
        o.images["base_0_rgb"] = lf
        with torch.enable_grad():
            state, ppm, cache = prefix_forward(model, device, o)
            seq = cache.key_cache[0].shape[2]
            n_img = seq - len(tok)
            keep = {"full": None,
                    "dest": [n_img + j for j in dest_j],
                    "src": [n_img + j for j in src_j],
                    "lang": [n_img + j for j in lang_j],
                    "img": list(range(n_img))}[name]
            if keep is not None:
                mask_cache_grad(cache, keep, seq, device, cache.key_cache[0].dtype)
            a = euler(model, device, state, ppm, cache, eps.clone())
        if a_ref is None:
            a_ref = a.detach().clone()
        else:  # 手术不许改前向数值
            assert float((a - a_ref).abs().max()) == 0.0, f"{name}: 前向被手术改变!"
        gs = []
        for c in range(3):
            a[0, :EX, c].sum().backward(retain_graph=(c < 2))
            gs.append(lf.grad.detach().clone()[0])
            lf.grad = None
        gmag = torch.stack(gs).norm(dim=(0, 1)).cpu().numpy()
        S = np.array([gmag[m].sum() for m in smasks])
        r = spear(fd_global, S)
        share = float(gmag.sum())
        results[name] = (S, r, share)
        gmags[name] = gmag
        print(f"[{name:4s}] keep={'all' if keep is None else len(keep)} 位  "
              f"Σ|g|={share:.3e}  秩相关 vs 全局FD = {r:+.2f}", flush=True)
        del a, o, lf, cache, state, ppm
        torch.cuda.empty_cache()

    full_sum = results["full"][2]
    print("\n== 汇总(帧 0,pooled,对 16 帧全局 FD 的秩相关;Σ|g| 占比=该通路承载的梯度量)==")
    for name in results:
        S, r, share = results[name]
        print(f"  {name:4s}: corr={r:+.2f}  Σ|g|/full={share/full_sum*100:5.1f}%")
    print(f"  dest 与 full 的梯度图秩相关 = {spear(gmags['dest'].ravel(), gmags['full'].ravel()):+.2f}")
    print(f"  dest 与 src  的梯度图秩相关 = {spear(gmags['dest'].ravel(), gmags['src'].ravel()):+.2f}")

    np.savez_compressed(GOUT / "g0_desttoken.npz",
                        frame=FR, variants=list(results),
                        gmag=np.stack([gmags[n] for n in results]),
                        S=np.stack([results[n][0] for n in results]),
                        spear_vs_global=np.array([results[n][1] for n in results]),
                        gsum=np.array([results[n][2] for n in results]),
                        dest_pos=np.array(dest_j), src_pos=np.array(src_j),
                        fd_global=fd_global, anchor_idx=za["anchor_idx"])
    print(f"[written] {GOUT/'g0_desttoken.npz'}", flush=True)


if __name__ == "__main__":
    main()
