#!/usr/bin/env python3
"""π0.5 帧0 attention → 16×16,两套 baseline:
  (1) 现有法 mid:destination 词行 × base256,head 求和、中层带 L4-12 平均(与 g0g/旧图同口径);
  (2) rollout:逐层 head 平均 attention (0.5A+0.5I) 逐层复合,再取 destination 行→base256→16×16。
     rollout 不挑层、无可调旋钮,是更站得住的 baseline(层敏感度问题见那条对话 USER128-130)。
额外存 all-layer 平均版 + 逐层 (18,256),方便后处理换口径。

用法:
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/percell_attn.py --task put_the_bowl_on_the_plate \
        --obs out/s2f_scan_obs.npz --dest plate --out out/grad/percell_attn_plate.npz
"""
import argparse
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
TXT_LO = 3 * N_IMG          # 768:文本 token 起点
MIDL = list(range(4, 13))   # 中层带 L4-12(现有法)


def norm16(v):
    v = v.reshape(N_SIDE, N_SIDE).astype(np.float64)
    return v / (v.sum() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="put_the_bowl_on_the_plate")
    ap.add_argument("--obs", default=str(OUT / "s2f_scan_obs.npz"))
    ap.add_argument("--dest", default="plate", help="destination 名词(token 里匹配的子串)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

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

    d = np.load(args.obs, allow_pickle=True)
    assert "prompt" in d.files, f"{args.obs} 缺 prompt 键"
    prompt = str(d["prompt"])
    img, wri, st = d["clean_img224"][0], d["clean_wrist224"][0], d["clean_state8"][0]
    ids, mask = tk.tokenize(prompt)
    Z = int(mask.sum())
    pieces = [tk._tokenizer.id_to_piece(int(i)) for i in ids[:Z]]
    dest_z = [j for j, p in enumerate(pieces) if args.dest in p.lower()]
    assert dest_z, f"找不到 destination 词 {args.dest!r} in {pieces}"
    print(f"[cfg] task={args.task} prompt={prompt!r} dest={args.dest!r}@{dest_z} layers={len(layers)}",
          flush=True)

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
    assert len(cap) == N_LAYER, f"只抓到 {len(cap)} 层"
    full = np.stack([cap[l][0].numpy() for l in range(N_LAYER)])   # (L,H,968,968)
    Ntok = full.shape[-1]
    print(f"[attn] full {full.shape}  Ntok={Ntok}", flush=True)

    # ---- (1) 现有法:dest 行 × base256 → head 求和 → dest 平均 → 逐层 (L,256)
    blk = full[:, :, TXT_LO:TXT_LO + Z, 0:N_IMG][:, :, dest_z]     # (L,H,nd,256)
    a_layers = blk.sum(1).mean(1)                                  # (L,256)
    a16_mid = norm16(a_layers[MIDL].mean(0))
    a16_all = norm16(a_layers.mean(0))

    # ---- (2) rollout:逐层 head 平均 (0.5A+0.5I) 复合
    R = np.eye(Ntok)
    for l in range(N_LAYER):
        A = full[l].mean(0)                          # (968,968) head 平均
        A = A / (A.sum(1, keepdims=True) + 1e-12)    # 行归一
        Ahat = 0.5 * A + 0.5 * np.eye(Ntok)
        Ahat = Ahat / (Ahat.sum(1, keepdims=True) + 1e-12)
        R = Ahat @ R
    roll = R[TXT_LO:TXT_LO + Z, 0:N_IMG][dest_z].sum(0)           # dest 行 → base256
    a16_roll = norm16(roll)

    outp = pathlib.Path(args.out) if args.out else OUT / "grad" / f"percell_attn_{args.task}.npz"
    np.savez_compressed(outp, task=args.task, dest=args.dest, pieces=pieces, dest_z=dest_z,
                        a16_mid=a16_mid, a16_all=a16_all, a16_roll=a16_roll,
                        a_layers=a_layers.astype(np.float32))
    print(f"[written] {outp}", flush=True)
    print(f"  a16_mid  max={a16_mid.max():.3f} argmax(r,c)={np.unravel_index(a16_mid.argmax(), a16_mid.shape)}",
          flush=True)
    print(f"  a16_roll max={a16_roll.max():.3f} argmax(r,c)={np.unravel_index(a16_roll.argmax(), a16_roll.shape)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
