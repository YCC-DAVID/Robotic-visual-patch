#!/usr/bin/env python3
"""起一个 policy server,但把 flow matching 的**初始噪声 ε 钉死成一条固定向量**。

为什么需要
--------
上一轮 rollout 什么也没测出来,原因**不是效应太小,是参照太脏**:
`websocket_client_policy` 的 `infer` 传不了噪声 ⇒ 每次前向重采一次 ε ⇒
两条什么都没改的 clean 轨迹之间就有 **28.3 mm** 的末端偏移峰值,
而四个贴纸条件是 27.6–38.7 mm —— 全埋在采样噪声里(RESULTS.md §15.1)。
那是**地板效应**,不是"influence 不预测轨迹"。

但 `Policy.infer(obs, *, noise=...)` 本来就支持外部传 ε
(`third_party/openpi/src/openpi/policies/policy.py:68,83-88`),
`sample_actions` 也只在 `noise is None` 时才自己采
(`models_pytorch/pi0_pytorch.py:377-382`)。
所以只要在 server 这一侧把 ε 钉死,clean 与 patched 就**共享同一个 ε** ——
与 S2.1 反事实扫描完全同一口径,**噪声地板归零**,
任何非零的轨迹偏移都能归因到贴纸。

ε 的形状从 `policy._model.config` 读(pi05_libero 是 action_horizon=10、action_dim=32),
不硬编码;dtype 对齐 `sample_noise` 的 float32(`pi0_pytorch.py:173-180`)。

⚠️ 这让策略变成确定性的。这不是缺陷:威胁模型里贴纸是静态的,
   我们要测的是"位置"这一个自变量,把采样方差消掉正是受控实验该做的事。

用法(py3.11):
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/serve_policy_fixed_noise.py --port 8123
    # 一般由 autorun_rollout.py 自动拉起,不用手敲
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--noise-seed", type=int, default=20260817, help="生成固定 ε 的 seed")
    ap.add_argument("--no-compile", action="store_true",
                    help="关掉 torch.compile(排查数值问题时用;正常跑不用关)")
    args = ap.parse_args()

    import numpy as np
    import transformers
    assert str(PATCHED_TF) in transformers.__file__, "没用上 patched transformers"
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi.serving import websocket_policy_server

    cfg = _config.get_config("pi05_libero")
    if args.no_compile:
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model,
                                                                 pytorch_compile_mode=None))
    policy = _policy_config.create_trained_policy(cfg, TORCH_CKPT)

    mcfg = policy._model.config                                        # noqa: SLF001
    H, D = int(mcfg.action_horizon), int(mcfg.action_dim)
    # 固定 ε:一条 (H, D) 的标准正态,server 生命周期内永不改变。
    # 用 numpy 的 RandomState 生成,与 torch 的 RNG 状态解耦 —— 谁也别想影响它。
    eps = np.random.RandomState(args.noise_seed).standard_normal((H, D)).astype(np.float32)
    print(f"[fixed-noise] ε shape={eps.shape} dtype={eps.dtype} seed={args.noise_seed} "
          f"‖ε‖={np.linalg.norm(eps):.6f}", flush=True)

    _orig_infer = policy.infer

    def infer_fixed(obs, *, noise=None):
        # 调用方(websocket server)不会传 noise,所以这里总是用固定 ε。
        # 保留 noise 参数是为了万一有人显式指定,不被静默覆盖。
        return _orig_infer(obs, noise=eps if noise is None else noise)

    policy.infer = infer_fixed          # 实例属性遮蔽类方法,websocket server 走这条

    # ---- 自检(计划红线 1):什么都不改的重复前向必须逐位相同 ----
    dummy = {"observation/image": np.zeros((224, 224, 3), np.uint8),
             "observation/wrist_image": np.zeros((224, 224, 3), np.uint8),
             "observation/state": np.zeros(8, np.float32),
             "prompt": "put the bowl on the plate"}
    a0 = policy.infer(dummy)["actions"]          # 第 1 次会触发 torch.compile,不参与比较
    a1 = policy.infer(dummy)["actions"]
    a2 = policy.infer(dummy)["actions"]
    d = float(np.abs(np.asarray(a2) - np.asarray(a1)).max())
    print(f"[fixed-noise] 自检:重复前向最大差 = {d:.3e}  (a0 与 a1 差 "
          f"{float(np.abs(np.asarray(a1) - np.asarray(a0)).max()):.3e},首次含编译)", flush=True)
    assert d == 0.0, (f"ε 已钉死但重复前向仍差 {d:.3e} —— 还有别的随机源,"
                      "不要用这个 server 的结果")
    print(f"[fixed-noise] ✅ 噪声地板 = 0,动作 shape={np.asarray(a1).shape}", flush=True)
    print(f"[fixed-noise] 监听 {args.host}:{args.port}", flush=True)

    logging.basicConfig(level=logging.WARNING)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port,
        metadata={"fixed_noise_seed": args.noise_seed, "action_horizon": H, "action_dim": D},
    ).serve_forever()


if __name__ == "__main__":
    main()
