#!/usr/bin/env python3
"""π0.5 · 逐任务 destination-token attention,池化到 61 锚点(给 3×3 图第三行)。

对每个任务的初始帧(s2f_scan_obs_{task}.npz 的 clean 帧0)跑一次 infer,hook 抓
language_model 每层 self_attn 的权重 (H,968,968);取 destination 名词那一行 × base
256 图像列 → 16×16,head 求和、中层带(L4-12)平均、在 256 上重归一化。
锚点池化:每个锚点的贴纸 224px 掩码降采到 16×16 作覆盖权重,S_attn[i]=Σ 覆盖·attn。
口径与 make_attn_maps 的 "noun"+base 归一一致;与 FastWAM 的落格池化同思路。

用法:
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/g0g_attn_task.py
"""
import dataclasses
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"
PATCHED_TF = ROOT / "third_party" / "transformers_patched"
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"
for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))
import os
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"

import numpy as np  # noqa: E402

N_SIDE, N_IMG, N_LAYER, N_HEAD = 16, 256, 18, 8
TXT_LO = 3 * N_IMG          # 768
MIDL = list(range(4, 13))   # 中层带(π0.5 可用带 ~L4-12)
TASKS = [
    ("plate",   "put_the_bowl_on_the_plate",          "s2f_scan_obs.npz",         "plate"),
    ("rack",    "put_the_wine_bottle_on_the_rack",     "s2f_scan_obs_rack.npz",    "rack"),
    ("cabinet", "put_the_bowl_on_top_of_the_cabinet",  "s2f_scan_obs_cabinet.npz", "cabinet"),
]


def main():
    import torch
    from openpi.models import tokenizer as _tok
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    cfg = _config.get_config("pi05_libero")
    cfg_nc = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = _policy_config.create_trained_policy(cfg_nc, TORCH_CKPT)
    model = policy._model
    layers = model.paligemma_with_expert.paligemma.language_model.layers
    tk = _tok.PaligemmaTokenizer(cfg.model.max_token_len)
    print(f"[policy] loaded  layers={len(layers)}  mid band L{MIDL[0]}-{MIDL[-1]}", flush=True)

    out = {}
    for short, prompt, obsf, destword in TASKS:
        d = np.load(OUT / obsf, allow_pickle=True)
        img, wri, st = d["clean_img224"][0], d["clean_wrist224"][0], d["clean_state8"][0]
        ids, mask = tk.tokenize(prompt)
        Z = int(mask.sum())
        pieces = [tk._tokenizer.id_to_piece(int(i)) for i in ids[:Z]]
        dest_z = [j for j, p in enumerate(pieces) if destword in p.lower()]
        assert dest_z, f"{short}: 找不到 destination 词 {destword!r} in {pieces}"

        cap = {}
        def mk(i):
            def hook(_m, _i, o):
                cap[i] = o[1].detach().float().cpu()      # (B,H,968,968)
            return hook
        hs = [layers[i].self_attn.register_forward_hook(mk(i)) for i in range(N_LAYER)]
        try:
            policy.infer({"observation/image": img, "observation/wrist_image": wri,
                          "observation/state": st, "prompt": prompt})
        finally:
            for h in hs:
                h.remove()
        assert len(cap) == N_LAYER, f"{short}: 只抓到 {len(cap)} 层"
        full = np.stack([cap[l][0].numpy() for l in range(N_LAYER)])   # (L,H,968,968)

        # destination 行 × base 256 列 → head 求和 → dest词平均 → 逐层 (L,256)
        blk = full[:, :, TXT_LO:TXT_LO + Z, 0:N_IMG][:, :, dest_z]     # (L,H,nd,256)
        a_layers = blk.sum(1).mean(1)                                  # (L,256) 每层
        a16 = a_layers[MIDL].mean(0).reshape(N_SIDE, N_SIDE)           # 中层平均(主口径)
        a16 = a16 / (a16.sum() + 1e-12)

        # 锚点覆盖矩阵 C (M,256):贴纸掩码 224→16 降采,存下来供逐层扫描
        clean = d["clean_img224"][0].astype(np.int16)
        M = int(d["M"])
        C = np.zeros((M, N_IMG))
        for i in range(M):
            m = (np.abs(d["patched_img224"][i, 0].astype(np.int16) - clean) > 2).any(-1)  # 224×224
            cov = m.reshape(N_SIDE, 224 // N_SIDE, N_SIDE, 224 // N_SIDE).sum((1, 3)).astype(float)
            C[i] = cov.flatten()
        S = C @ a16.flatten()
        out[short] = dict(S_attn=S, a16=a16, a_layers=a_layers.astype(np.float32),
                          cov=C.astype(np.float32), dest=destword, pieces=pieces,
                          anchor_idx=d["anchor_idx"], anchor_world=d["anchor_world"])
        print(f"[{short:8s}] dest={destword!r}@{dest_z}  attn16 max={a16.max():.3f} "
              f"(uniform {1/256:.4f})  touched cells={int((C.sum(0)>0).sum())}/256  "
              f"rank(C)={np.linalg.matrix_rank(C, tol=1e-6)}", flush=True)

    save = {}
    for s, o in out.items():
        for k, v in o.items():
            if k != "dest":
                save[f"{s}__{k}"] = v
    np.savez_compressed(OUT / "grad" / "g0_attn_task.npz", tasks=[s for s, *_ in TASKS], **save)
    print(f"[written] {OUT/'grad'/'g0_attn_task.npz'}", flush=True)


if __name__ == "__main__":
    main()
