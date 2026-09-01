#!/usr/bin/env python3
"""对抗 patch 训练 —— Phase B(py3.11,openpi-server,可微前向):
在归一化 base_0_rgb 图像空间优化目标格的固定足迹像素(patch 与相机静止 ⇒ 足迹每帧相同),
EOT 跨 8 帧、固定共享 ε(Δa 里 ε 抵消)、最大化执行前缀 EX=5 的动作偏移。
可微前向复用 g0_gradcheck.sample_actions_grad(逐字复刻 sample_actions、去 @no_grad,不改模型)。

存 out/patch_trained.npz:优化后的归一化 patch P(3,224,224)、足迹 mask、quad224、
逐步 loss、达成的动作偏移(vs clean、vs 随机初始)。
之后 patch_texture_from_train.py(py3.8)反 warp 成方形 PNG,attack_rollout.py 做物理验收。

用法:
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/patch_train.py --steps 200 --lr 0.02
"""
import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
PI05 = ROOT / "pi05probe"
OUT = PI05 / "out"
PATCHED_TF = ROOT / "third_party" / "transformers_patched"
TORCH_CKPT = ROOT / "checkpoints" / "pi05_libero_pytorch"
for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(PI05))
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
from g0_gradcheck import sample_actions_grad

AH, AD, EX = 10, 32, 5
SEED = 20260824


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--loss", default="away", choices=["away", "curve", "dev"],
                    help="away=动作推离 destination(方向性,适配 flow-matching);"
                         "curve=横向弯折(POAP 曲率代理);dev=最大化|Δa|(UADA 路子,只适合离散头,做对照)")
    ap.add_argument("--dest", default="0.062,-0.009", help="destination world x,y(plate)")
    args = ap.parse_args()
    dest_xy = np.array([float(v) for v in args.dest.split(",")])

    import torch
    import jax
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.models import model as _model

    d = np.load(OUT / "patch_prep.npz", allow_pickle=True)
    prompt = "put the bowl on the plate"
    C = d["clean_img224"]; W = d["clean_wrist224"]; S = d["clean_state8"]; VIS = d["vis_mask"]
    Fn = C.shape[0]
    print(f"[data] frames={Fn} patch_px={int(VIS[0].sum())}", flush=True)

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    model = policy._model
    device = policy._pytorch_device
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[policy] loaded device={device}", flush=True)

    def build_obs(k):
        el = {"observation/image": C[k], "observation/wrist_image": W[k],
              "observation/state": S[k], "prompt": prompt}
        inputs = policy._input_transform(jax.tree.map(lambda x: x, el))
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    # 每帧 clean obs(缓存 base 干净张量)+ 固定 ε 的 clean 动作
    obss = [build_obs(k) for k in range(Fn)]
    base_key = "base_0_rgb"
    base_clean = [o.images[base_key].detach().clone() for o in obss]   # (1,3,224,224) [-1,1]
    M = [torch.from_numpy(VIS[k].astype(np.float32)).to(device)[None, None] for k in range(Fn)]
    eps = torch.from_numpy(np.random.RandomState(SEED).normal(size=(1, AH, AD)).astype(np.float32)).to(device)

    with torch.no_grad():
        a_clean = []
        for k in range(Fn):
            obss[k].images[base_key] = base_clean[k]
            a_clean.append(sample_actions_grad(model, device, obss[k], eps.clone())[0, :EX].detach())
        # 红线:同帧同 ε 两次一致
        r1 = sample_actions_grad(model, device, obss[0], eps.clone())
        obss[0].images[base_key] = base_clean[0]
        r2 = sample_actions_grad(model, device, obss[0], eps.clone())
        print(f"[红线] 固定 ε 重复前向 |Δ|max={float((r1-r2).abs().max()):.2e}", flush=True)

    # 方向向量(world 系;假设 LIBERO OSC 动作平移 delta 与 eef 位置同系)
    zt = float(d["z"])
    dest3 = torch.tensor([dest_xy[0], dest_xy[1], zt], dtype=torch.float32, device=device)
    dir_dest, cdir = [], []
    for k in range(Fn):
        eefk = torch.tensor(np.asarray(S[k][:3], np.float32), device=device)
        vv = dest3 - eefk; dir_dest.append(vv / (vv.norm() + 1e-9))
        cv = a_clean[k][:, 0:3].sum(0); cdir.append(cv / (cv.norm() + 1e-9))

    def score(a, k):
        """每帧要**最大化**的方向性目标标量(可反传)。"""
        trans = a[:, 0:3].sum(0)                        # 执行前缀净平移 (3,)
        if args.loss == "away":                         # 推离 destination
            return -(trans * dir_dest[k]).sum()
        if args.loss == "curve":                        # 横向弯折(POAP 曲率代理)
            perp = trans - (trans * cdir[k]).sum() * cdir[k]
            return torch.linalg.vector_norm(perp)
        return torch.linalg.vector_norm(a - a_clean[k])  # dev(UADA,对照)

    def eval_score(P):
        tot = 0.0
        with torch.no_grad():
            for k in range(Fn):
                comp = base_clean[k] * (1 - M[k]) + P.clamp(-1, 1) * M[k]
                obss[k].images[base_key] = comp
                a = sample_actions_grad(model, device, obss[k], eps.clone())[0, :EX]
                tot += float(score(a, k))
        return tot

    munion = torch.clamp(sum(M), 0, 1)
    P = torch.empty(1, 3, 224, 224, device=device).uniform_(-1, 1).requires_grad_(True)
    rand_dev = eval_score(P)
    print(f"[init] loss={args.loss} 随机 patch 目标值(EOT合计)={rand_dev:.3f}", flush=True)

    opt = torch.optim.Adam([P], lr=args.lr)
    curve = []
    for it in range(args.steps):
        opt.zero_grad()
        tot = 0.0
        for k in range(Fn):                    # 逐帧 backward 累积梯度,一次只留一帧的图
            comp = base_clean[k] * (1 - M[k]) + P.clamp(-1, 1) * M[k]
            obss[k].images[base_key] = comp
            a = sample_actions_grad(model, device, obss[k], eps.clone())[0, :EX]
            sk = score(a, k)
            (-sk).backward()                   # 最大化 score
            tot += float(sk)
        with torch.no_grad():
            P.grad *= munion                   # 只更新足迹像素
        opt.step()
        with torch.no_grad():
            P.clamp_(-1, 1)
        curve.append(tot)
        if it % 20 == 0 or it == args.steps - 1:
            print(f"  [step {it:3d}] {args.loss} 目标值={tot:.3f}  (随机 {rand_dev:.3f})", flush=True)

    final = curve[-1]
    print(f"[done] loss={args.loss} 最终目标值={final:.3f}  vs 随机 {rand_dev:.3f}", flush=True)
    outp = OUT / f"patch_trained_{args.loss}.npz"
    np.savez_compressed(outp,
                        P=P.detach().cpu().numpy()[0], vis_union=torch.clamp(sum(M),0,1).cpu().numpy()[0,0],
                        quad224=d["quad224"], cell_world=d["cell_world"], z=d["z"], patch_m=d["patch_m"],
                        loss=args.loss, curve=np.array(curve), rand_dev=rand_dev, final_dev=final)
    print(f"[written] {outp}", flush=True)


if __name__ == "__main__":
    main()
