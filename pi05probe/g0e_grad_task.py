#!/usr/bin/env python3
"""跨任务帧 0 梯度(g0c 的参数化版):对指定 obs 文件的第 0 帧算三通道梯度图。

用法:
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/g0e_grad_task.py --obs out/s2f_scan_obs_rack.npz --out out/grad/g0_grad_f0_rack.npz
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
sys.path.insert(0, str(ROOT / "pi05probe"))

from g0_gradcheck import (  # noqa: E402
    AH, AD, EX, SEED_EPS, TORCH_CKPT, sample_actions_grad,
)
import numpy as np  # noqa: E402

FR = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import jax
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.models import model as _model

    d = np.load(args.obs, allow_pickle=True)
    prompt = str(d["prompt"])
    print(f"[data] task={d['task']}  prompt={prompt!r}  frame {FR}", flush=True)

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    model = policy._model
    device = policy._pytorch_device
    for p in model.parameters():
        p.requires_grad_(False)

    el = {"observation/image": d["clean_img224"][FR],
          "observation/wrist_image": d["clean_wrist224"][FR],
          "observation/state": d["clean_state8"][FR], "prompt": prompt}
    inputs = policy._input_transform(jax.tree.map(lambda x: x, el))
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
    obs = _model.Observation.from_dict(inputs)
    lf = obs.images["base_0_rgb"].clone().detach().requires_grad_(True)
    obs.images["base_0_rgb"] = lf
    eps = torch.from_numpy(
        np.random.RandomState(SEED_EPS).standard_normal((AH, AD)).astype(np.float32)
    ).to(device)[None]

    with torch.enable_grad():
        a = sample_actions_grad(model, device, obs, eps)
    gs = []
    for c in range(3):
        a[0, :EX, c].sum().backward(retain_graph=(c < 2))
        gs.append(lf.grad.detach().clone()[0])
        lf.grad = None
    g = torch.stack(gs)
    gmag = g.norm(dim=(0, 1))
    assert not torch.isnan(g).any() and not torch.isinf(g).any()
    print(f"[grad] 非零 {float((g != 0).float().mean())*100:.1f}%  "
          f"|g|max={float(g.abs().max()):.2e}", flush=True)

    outp = pathlib.Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, frame=FR, eps_seed=SEED_EPS, task=str(d["task"]),
                        g_ch=g.cpu().numpy(), gmag=gmag.cpu().numpy())
    print(f"[written] {outp}", flush=True)


if __name__ == "__main__":
    main()
