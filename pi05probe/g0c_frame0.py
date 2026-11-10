#!/usr/bin/env python3
"""初始帧梯度:用第 0 帧(初始观测)的输入算逐像素梯度,存图,供
「初始帧梯度能否指导全局 influence」的对比(用户点名)。

口径与 G0 完全一致:通道和标量 s_c = Σ_{k<EX} a[k,c](c=0,1,2 平移),
固定 ε(seed 20260817 配方),复刻的 sample_actions_grad,不改模型。
另把帧 4(抓取前后)也一起算了,一次加载模型三个帧价钱一样。

用法:
    CUDA_VISIBLE_DEVICES=1 /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/g0c_frame0.py
"""
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
sys.path.insert(0, str(ROOT / "pi05probe"))

from g0_gradcheck import (  # noqa: E402
    AH, AD, EX, SEED_EPS, OUT, GOUT, TORCH_CKPT, sample_actions_grad,
)
import numpy as np  # noqa: E402

FRAMES = [0, 4]


def main():
    import torch
    import jax
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.models import model as _model

    d = np.load(OUT / "s2f_scan_obs.npz", allow_pickle=True)
    prompt = str(d["prompt"])

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    model = policy._model
    device = policy._pytorch_device
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[policy] loaded  frames={FRAMES}", flush=True)

    eps = torch.from_numpy(
        np.random.RandomState(SEED_EPS).standard_normal((AH, AD)).astype(np.float32)
    ).to(device)[None]

    g_all, gmag_all = [], []
    for fr in FRAMES:
        el = {"observation/image": d["clean_img224"][fr],
              "observation/wrist_image": d["clean_wrist224"][fr],
              "observation/state": d["clean_state8"][fr], "prompt": prompt}
        inputs = jax.tree.map(lambda x: x, el)
        inputs = policy._input_transform(inputs)
        inputs = jax.tree.map(
            lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        obs = _model.Observation.from_dict(inputs)
        lf = obs.images["base_0_rgb"].clone().detach().requires_grad_(True)
        obs.images["base_0_rgb"] = lf
        with torch.enable_grad():
            a = sample_actions_grad(model, device, obs, eps.clone())
        gs = []
        for c in range(3):
            a[0, :EX, c].sum().backward(retain_graph=(c < 2))
            gs.append(lf.grad.detach().clone()[0])
            lf.grad = None
        g = torch.stack(gs)                                 # [3ch,3rgb,224,224]
        gm = g.norm(dim=(0, 1))
        assert not torch.isnan(g).any() and not torch.isinf(g).any()
        g_all.append(g.cpu().numpy()); gmag_all.append(gm.cpu().numpy())
        print(f"[grad] frame {fr}: 非零 {float((g != 0).float().mean())*100:.1f}%  "
              f"|g|max={float(g.abs().max()):.2e}", flush=True)
        del a, obs, lf, g
        torch.cuda.empty_cache()

    np.savez_compressed(GOUT / "g0_grad_early.npz",
                        frames=np.array(FRAMES), eps_seed=SEED_EPS,
                        g_ch=np.stack(g_all), gmag=np.stack(gmag_all))
    print(f"[written] {GOUT/'g0_grad_early.npz'}", flush=True)


if __name__ == "__main__":
    main()
