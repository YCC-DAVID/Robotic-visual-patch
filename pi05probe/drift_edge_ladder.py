#!/usr/bin/env python3
"""DRIFT 复现 · 开环阶梯:拆"直接贴 P"与"渲染"之间到底哪个差异杀死攻击。

背景(HANDOFF §4.25):wrist 消融已排除 wrist ⇒ 缺口只在 base 图足迹像素。
用户假设:优化器可能用了**硬边界**这个 shortcut —— 数字合成是硬切 mask(像 PS 贴上去),
渲染出来的边缘是柔和、融入场景的。本脚本在观测池帧上逐级施加单一变换,看 L₀ / ‖Δa‖ 在哪一级塌。

变体(都与训练同一合成式 comp = clean·(1−α) + (P'·2−1)·α,只改 α 或 P'):
  边界类  hard / feather1/2/3(向内线性过渡)/ ring2(只贴边缘 2px)/
          interior_hard(只贴内部,边界 2px 置干净,硬边仍在但移到内部)/ interior_feather
  渲染类  affine(拟合的 a,b)/ blur1.3 / shift1px / proxy(affine+blur)
  参照    rand(同足迹随机图案)

用法(openpi-server, py3.11):
    CUDA_VISIBLE_DEVICES=<free> ~/miniconda3/envs/openpi-server/bin/python \
        pi05probe/drift_edge_ladder.py --patch drift_patch_drift --pool drift_pool_big
"""
import argparse
import json
import sys

import drift_patch_train as T          # 复用 sys.path/env 设置与 velocity_at_steps / integrate_actions
import numpy as np

OUT = T.OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="drift_patch_drift")
    ap.add_argument("--pool", default="drift_pool_big")
    ap.add_argument("--fit", default="drift_render_fit")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--no-act", action="store_true", help="跳过 10 步积分的 ‖Δa‖(快一倍)")
    args = ap.parse_args()

    import torch
    import jax
    from scipy.ndimage import distance_transform_edt
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.models import model as _model

    d = np.load(OUT / f"{args.pool}.npz", allow_pickle=True)
    IMG, WRI, ST, MASKS = d["clean_img224"], d["clean_wrist224"], d["clean_state8"], d["masks"]
    prompt = str(d["prompt"])
    px = MASKS.reshape(MASKS.shape[0], -1).sum(1)
    vis = np.where(px > 0)[0]
    rng = np.random.RandomState(0)
    idx = vis[rng.choice(len(vis), min(args.frames, len(vis)), replace=False)]

    pz = np.load(OUT / f"{args.patch}.npz", allow_pickle=True)
    P0 = torch.from_numpy(pz["P"].astype(np.float32))[None]          # [1,3,224,224] in [0,1]
    fit = np.load(OUT / f"{args.fit}.npz", allow_pickle=True)
    sig, fa, fb = float(fit["sigma"]), fit["a"].astype(np.float32), fit["b"].astype(np.float32)
    print(f"[cfg] patch={args.patch} pool={args.pool} frames={len(idx)}  fit σ={sig} a={fa.round(3)} b={fb.round(3)}",
          flush=True)

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), T.TORCH_CKPT)
    model, device = policy._model, policy._pytorch_device               # noqa: SLF001
    for p in model.parameters():
        p.requires_grad_(False)

    def build_obs(k):
        el = {"observation/image": IMG[k], "observation/wrist_image": WRI[k],
              "observation/state": ST[k], "prompt": prompt}
        inputs = policy._input_transform(jax.tree.map(lambda x: x, el))   # noqa: SLF001
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    noise = torch.from_numpy(np.random.RandomState(T.SEED_NOISE).standard_normal(
        (T.AH, T.AD)).astype(np.float32)).to(device)[None]

    def gblur(x, s):
        r = max(1, int(3 * s))
        t = torch.arange(-r, r + 1, dtype=torch.float32)
        k = torch.exp(-(t ** 2) / (2 * s ** 2)); k = k / k.sum()
        x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1).expand(3, 1, 1, -1), padding=(0, r), groups=3)
        return torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1).expand(3, 1, -1, 1), padding=(r, 0), groups=3)

    at, bt = torch.from_numpy(fa).view(1, 3, 1, 1), torch.from_numpy(fb).view(1, 3, 1, 1)
    P_aff = (at * P0 + bt).clamp(0, 1)
    P_blur = gblur(P0, sig).clamp(0, 1)
    P_prox = (at * gblur(P0, sig) + bt).clamp(0, 1)
    P_shift = torch.roll(P0, shifts=(1, 1), dims=(2, 3))
    P_rand = torch.from_numpy(np.random.RandomState(20260902).rand(1, 3, 224, 224).astype(np.float32))

    # α 构造:D = 足迹内每个像素到最近非足迹像素的距离(px)
    def alphas(m):
        D = distance_transform_edt(m)
        hard = m.astype(np.float32)
        out = {"hard": hard}
        for w in (1, 2, 3):
            out[f"feather{w}"] = np.clip(D / (w + 1), 0, 1).astype(np.float32) * hard
        out["ring2"] = ((D > 0) & (D <= 2)).astype(np.float32)
        out["interior_hard"] = (D > 2).astype(np.float32)
        out["interior_feather"] = np.clip((D - 2) / 3, 0, 1).astype(np.float32)
        return out

    variants = [  # (名字, α 键, 图案)
        ("hard", "hard", P0), ("feather1", "feather1", P0), ("feather2", "feather2", P0),
        ("feather3", "feather3", P0), ("ring2", "ring2", P0), ("interior_hard", "interior_hard", P0),
        ("interior_feather", "interior_feather", P0),
        ("affine", "hard", P_aff), ("blur1.3", "hard", P_blur), ("shift1px", "hard", P_shift),
        ("proxy", "hard", P_prox), ("proxy+feather2", "feather2", P_prox),
        ("rand", "hard", P_rand),
    ]
    L = {n: [] for n, _, _ in variants}
    A = {n: [] for n, _, _ in variants}
    apx = {n: [] for n, _, _ in variants}

    with torch.no_grad():
        for i, k in enumerate(idx):
            k = int(k)
            o = build_obs(k)
            bc = o.images[T.BASE_KEY].detach().clone()
            vc = T.velocity_at_steps(model, device, o, noise.clone())
            ac = None if args.no_act else T.integrate_actions(model, device, build_obs(k), noise.clone())
            al = alphas(MASKS[k].astype(bool))
            for name, akey, P in variants:
                a = torch.from_numpy(al[akey])[None, None].to(device)
                comp = bc * (1 - a) + (P.to(device) * 2 - 1) * a
                o2 = build_obs(k); o2.images[T.BASE_KEY] = comp
                v = T.velocity_at_steps(model, device, o2, noise.clone())
                L[name].append(float(((v - vc) ** 2).sum()))
                apx[name].append(float(al[akey].sum()))
                if ac is not None:
                    o3 = build_obs(k); o3.images[T.BASE_KEY] = comp
                    A[name].append(float((T.integrate_actions(model, device, o3, noise.clone()) - ac).norm()))
            if i % 8 == 0:
                print(f"  {i+1}/{len(idx)}  hard L₀={L['hard'][-1]:.3f}  feather2={L['feather2'][-1]:.3f}  "
                      f"ring2={L['ring2'][-1]:.3f}", flush=True)

    base = np.mean(L["hard"])
    lines = ["=" * 92,
             f"开环阶梯 · {args.patch} · {len(idx)} 帧(观测池 {args.pool})· 同一固定噪声",
             "=" * 92,
             f"  {'变体':18s} {'L₀ 均值':>9s} {'L₀ 中位':>9s} {'÷hard':>7s} {'‖Δa‖中位':>9s} {'α 和(px)':>9s}"]
    for name, _, _ in variants:
        l = np.array(L[name])
        am = f"{np.median(A[name]):9.3f}" if A[name] else "      --"
        lines.append(f"  {name:18s} {l.mean():9.3f} {np.median(l):9.3f} {l.mean()/base:7.1%} {am} "
                     f"{np.median(apx[name]):9.0f}")
    lines += ["",
              "  判读:",
              "    feather 就塌(÷hard ≪)而 affine/blur/shift 不塌 ⇒ 攻击依赖硬边界(shortcut)",
              "    ring2 ≈ hard 而 interior_* ≪ ⇒ 攻击能量集中在边缘一圈",
              "    interior_hard ≈ hard 但 interior_feather ≪ ⇒ 依赖的是'硬跳变'本身而非具体边缘像素",
              "    affine/blur/shift 某一项塌 ⇒ 对应的渲染因素是主因",
              "    参照:rand 的 L₀ ≈ 0.28(训练脚本实测),塌到此量级 = 攻击失效"]
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    tag = f"drift_ladder_{args.patch.replace('drift_patch_', '')}"
    (OUT / f"{tag}.txt").write_text(txt + "\n")
    (OUT / f"{tag}.json").write_text(json.dumps(dict(L=L, A=A, alpha_px=apx, frames=[int(k) for k in idx]),
                                                indent=1))
    print(f"[written] {OUT/(tag+'.txt')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
