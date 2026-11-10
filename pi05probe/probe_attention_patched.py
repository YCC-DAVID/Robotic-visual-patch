#!/usr/bin/env python3
"""贴上 patch **之后**的 attention:clean-attention 能不能预测 patched-attention。

要回答的问题(这是攻击设定的要害)
--------------------------------
攻击者在贴之前只有 clean 图,所以 POAP 那类方法只能用 **clean attention** 选位置。
但贴上去之后:
  - 高饱和随机纹理本身可能**把 attention 吸到自己身上** ⇒ patched attention 的峰值在 patch 上,
    与 clean attention 的峰值完全无关;
  - 那么"clean attention 的峰值"既不是 patched 图里 attention 的峰值,也不是影响力的峰值。
三张图(clean attention / patched attention / influence)是**三个不同的排序函数**,
本脚本产出第二张,让三者可以两两比较。

做法
----
沿用 S2 的反事实设计:轨迹全程 clean,对每个(锚点, 帧)单独查询一次,patched 动作不执行。
输入 out/s2_scan_obs.npz(36 锚点 × 16 帧的 patched/clean 观测,已在盘上)。

存**原始 head-求和块**(与 attn_traj_*.npz 同格式),不预先归一化、不预先聚合 ⇒
下游可以套用完全相同的 renorm / token 聚合代码,口径不会漂。

    attn_patched[M, T, L, Z, V, 16, 16]     float16 省空间(数值范围 O(1),够用)
    attn_clean  [   T, L, Z, V, 16, 16]

前向次数 = 36×16 + 16 = 592。约 40–60 分钟。

用法:
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/probe_attention_patched.py
"""
import dataclasses
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
OUT = ROOT / "pi05probe" / "out"
PATCHED_TF = ROOT / "third_party" / "transformers_patched"
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"

for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
os.environ["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"

import numpy as np  # noqa: E402

N_IMG, N_SIDE, N_LAYER, N_HEAD = 256, 16, 18, 8
VIEWS = ["base_0_rgb", "left_wrist_0_rgb"]
TXT_LO = 3 * N_IMG
VLO = {"base_0_rgb": 0, "left_wrist_0_rgb": N_IMG}


def main():
    import transformers
    assert str(PATCHED_TF) in transformers.__file__, "没用上 patched transformers"
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi.models import tokenizer as _tok

    d = np.load(OUT / "s2_scan_obs.npz", allow_pickle=True)
    prompt = str(d["prompt"])
    T, M = int(d["T"]), int(d["M"])
    cimg, cwri, cstate = d["clean_img224"], d["clean_wrist224"], d["clean_state8"]
    pimg, pwri = d["patched_img224"], d["patched_wrist224"]
    print(f"[data] M={M} T={T} prompt={prompt!r}  前向次数={M*T+T}", flush=True)

    cfg = _config.get_config("pi05_libero")
    # ⚠️ 必须关 torch.compile,否则 forward hook 可能不触发(见 probe_attention_traj.py)
    cfg_nc = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model,
                                                                pytorch_compile_mode=None))
    policy = _policy_config.create_trained_policy(cfg_nc, TORCH_CKPT)
    layers = policy._model.paligemma_with_expert.paligemma.language_model.layers  # noqa: SLF001
    assert len(layers) == N_LAYER, f"层数 {len(layers)} != {N_LAYER}"

    tk = _tok.PaligemmaTokenizer(cfg.model.max_token_len)
    ids, msk = tk.tokenize(prompt)
    pieces = [tk._tokenizer.id_to_piece(int(i)) for i in ids[:int(msk.sum())]]  # noqa: SLF001
    Z = len(pieces)
    print(f"[tok] Z={Z} tokens: {pieces}", flush=True)

    def grab(img, wri, st):
        cap = {}

        def mk(i):
            def hook(_m, _i, out):
                cap[i] = out[1].detach().float().cpu()
            return hook

        hs = [layers[i].self_attn.register_forward_hook(mk(i)) for i in range(N_LAYER)]
        try:
            policy.infer({"observation/image": img, "observation/wrist_image": wri,
                          "observation/state": st, "prompt": prompt})
        finally:
            for h in hs:
                h.remove()
        assert len(cap) == N_LAYER, f"hook 只抓到 {len(cap)}/{N_LAYER} 层"
        full = np.stack([cap[l][0].numpy() for l in range(N_LAYER)])   # (L,H,S,S)
        blk = np.stack([full[:, :, TXT_LO:TXT_LO + Z, VLO[v]:VLO[v] + N_IMG]
                        .reshape(N_LAYER, N_HEAD, Z, N_SIDE, N_SIDE)
                        for v in VIEWS], axis=3).sum(axis=1)           # head 求和 → (L,Z,V,h,w)
        return blk.astype(np.float16)

    Aclean = np.zeros((T, N_LAYER, Z, 2, N_SIDE, N_SIDE), np.float16)
    for t in range(T):
        Aclean[t] = grab(cimg[t], cwri[t], cstate[t])
        print(f"  [clean] frame {t:2d}/{T}", flush=True)

    Apat = np.zeros((M, T, N_LAYER, Z, 2, N_SIDE, N_SIDE), np.float16)
    for i in range(M):
        for t in range(T):
            Apat[i, t] = grab(pimg[i, t], pwri[i, t], cstate[t])
        print(f"  [patched] anchor {i:2d}/{M}", flush=True)

    outp = OUT / "attn_patched_grid.npz"
    np.savez_compressed(
        outp, attn_clean=Aclean, attn_patched=Apat,
        pieces=np.array(pieces), views=np.array(VIEWS), prompt=prompt,
        T=T, M=M, ts=d["ts"], ks=d["ks"], visible_px=d["visible_px"],
        anchor_world=d["anchor_world"], anchor_idx=d["anchor_idx"],
        anchor_legal=d["anchor_legal"], anchor_keepout=d["anchor_keepout"],
        renorm=np.array("raw_head_sum"))
    print(f"\n[written] {outp}  ({outp.stat().st_size/2**20:.1f} MiB)", flush=True)
    print("  下一步:report_attn_shift.py(clean vs patched attention,纯后处理)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
