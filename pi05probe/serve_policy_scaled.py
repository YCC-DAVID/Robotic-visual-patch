#!/usr/bin/env python3
"""起一个 policy server,但把**某一路相机的 image token 的 value 向量整体缩放**。

目的
----
判定"腕部相机对动作到底有多重要"。比把图像涂黑更干净:输入分布完全不变,
只削弱该路 token 对残差流的贡献。缩放 0.1 后跑完整 rollout:
  - 成功率不掉  ⇒ 该路相机对动作贡献很小
  - 成功率塌掉  ⇒ 该路相机是承重的

token 布局(实测确认)
-------------------
KV 序列长 968 = 3 × 256 图像 token + 200 文本 token:
    [  0, 256) base_0_rgb        主视角
    [256, 512) left_wrist_0_rgb  腕部
    [512, 768) 第三路(LIBERO 只有两路,这段是补位)
    [768, 968) 文本
`probe_attention_patched.py` 就是按这个布局切的,而 base 那一段切出来的
patch 位置与分割掩码的相关是 0.9984,布局已验证。

做法:在 paligemma 每一层的 `self_attn.v_proj` 上挂 forward hook,把输出在
指定 token 区间上乘以 scale。这样 prefix 自注意力和被缓存给 action expert 的
value 都会被同样削弱。

⚠️ 必须关 torch.compile,否则 hook 可能不触发(与 probe_attention_traj.py 同一坑)。

用法(py3.11):
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/serve_policy_scaled.py --scale 0.1 --view wrist --port 8124
    # scale 1.0 = 对照(应与未改动的 server 行为一致)
"""
import argparse
import dataclasses
import logging
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
PATCHED_TF = ROOT / "third_party" / "transformers_patched"
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"

for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))

N_IMG = 256
SPANS = {"base": (0, N_IMG), "wrist": (N_IMG, 2 * N_IMG)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True, help="value 向量的缩放倍数")
    ap.add_argument("--view", default="wrist", choices=list(SPANS))
    ap.add_argument("--port", type=int, default=8124)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    import torch
    import transformers
    assert str(PATCHED_TF) in transformers.__file__, "没用上 patched transformers"
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi.serving import websocket_policy_server

    cfg = _config.get_config("pi05_libero")
    cfg_nc = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model,
                                                                pytorch_compile_mode=None))
    policy = _policy_config.create_trained_policy(cfg_nc, TORCH_CKPT)
    layers = policy._model.paligemma_with_expert.paligemma.language_model.layers  # noqa: SLF001
    lo, hi = SPANS[args.view]
    n_fired = [0]

    def mk_hook(scale, lo, hi, counter):
        def hook(_m, _inp, out):
            # out: (B, S, D)。只在序列足够长(prefix 前向)时动手,且只动指定区间。
            if out.dim() != 3 or out.shape[1] <= hi:
                return out
            out = out.clone()
            out[:, lo:hi, :] = out[:, lo:hi, :] * scale
            counter[0] += 1
            return out
        return hook

    for lyr in layers:
        lyr.self_attn.v_proj.register_forward_hook(mk_hook(args.scale, lo, hi, n_fired))

    print(f"[scaled-server] view={args.view}  token 区间=[{lo},{hi})  scale={args.scale}", flush=True)
    print(f"[scaled-server] 挂了 {len(layers)} 个 v_proj hook;torch.compile 已关", flush=True)

    # 自检:跑一次假前向,确认 hook 真的触发了
    import numpy as np
    dummy = {"observation/image": np.zeros((224, 224, 3), np.uint8),
             "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
             "observation/state": np.zeros(8, np.float32),
             "prompt": "put the bowl on the plate"}
    policy.infer(dummy)
    assert n_fired[0] > 0, "hook 一次都没触发 —— 缩放没有生效,不要用这个 server 的结果"
    print(f"[scaled-server] 自检通过:hook 触发 {n_fired[0]} 次 / 一次 infer", flush=True)
    print(f"[scaled-server] 监听 {args.host}:{args.port}", flush=True)

    logging.basicConfig(level=logging.WARNING)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port,
        metadata={"scaled_view": args.view, "scale": args.scale},
    ).serve_forever()


if __name__ == "__main__":
    main()
