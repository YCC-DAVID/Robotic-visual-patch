#!/usr/bin/env python3
"""DRIFT 复现 · 诊断:优化图案把**动作**推向哪里 —— 相干偏移 还是 不相干抖动?

为什么必须做这个
--------------
DRIFT 的 L₀ 是**无目标**的:只最大化 ‖v_θ(A^τ⁽⁰⁾,o⁺) − v_θ(A^τ⁽⁰⁾,o)‖²,不指定方向。
⇒ L₀ 很大有两种截然不同的可能,而 L₀ 数值本身分不出来:
  (a) **相干偏移**:每帧都把动作往同一个错误方向推 ⇒ 闭环 replan 修不回来,任务会崩;
  (b) **不相干抖动**:每帧方向随机 ⇒ 闭环自己平均掉,成功率不动。
本项目 §2.3 第 3 条已经踩过一次(开环 Δa ×3~6,闭环零效果),所以这个量必须单独测。

量三件事(对 DRIFT 图案 / 同尺寸随机图案 同口径):
  1. Δa = a(贴纸) − a(干净),完整 10 步积分后的最终动作;按 LIBERO 语义分解
     (dim 0-2 平移、3-5 旋转、6 夹爪;7-31 是 padding,单独报以确认确实没信号)。
  2. **跨帧方向一致性**:把每帧的 Δa 单位化后求平均向量的模 ‖mean(û)‖ ∈ [0,1]。
     →1 = 每帧同向(相干);→0 = 各走各的(抖动)。另报两两余弦的中位数。
  3. 相干偏移的**绝对量级**:mean(Δa) 各维,与该维动作本身的量级比。

用法:
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/drift_delta_dir.py --patch drift_patch_drift --frames 64
"""
import argparse
import os
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
sys.path.insert(0, str(PROBE))
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402

BASE_KEY = "base_0_rgb"
SEED_RAND = 20260902          # 必须与 drift_eval_digital.py 的随机对照同 seed
GROUPS = [("平移 xyz", slice(0, 3)), ("旋转 rpy", slice(3, 6)),
          ("夹爪", slice(6, 7)), ("padding 7-31", slice(7, 32))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="drift_patch_drift")
    ap.add_argument("--frames", type=int, default=64, help="用池里多少帧做统计")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tag = args.out or f"drift_deltadir_{args.patch.replace('drift_patch_', '')}"

    import torch
    import jax
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.models import model as _model
    import drift_patch_train as dpt          # 复用 integrate_actions,口径逐字一致

    dp = np.load(OUT / f"{args.patch}.npz", allow_pickle=True)
    P = torch.from_numpy(dp["P"]).float()[None]              # [1,3,224,224] in [0,1]
    pool = str(dp["pool"])
    d = np.load(OUT / f"{pool}.npz", allow_pickle=True)
    IMG, WRI, ST, MASKS = d["clean_img224"], d["clean_wrist224"], d["clean_state8"], d["masks"]
    prompt = str(d["prompt"])
    F = IMG.shape[0]
    px = MASKS.reshape(F, -1).sum(1)
    vis = np.where(px > 0)[0]
    rng = np.random.RandomState(0)
    idx = vis[rng.choice(len(vis), min(args.frames, len(vis)), replace=False)]
    print(f"[cfg] patch={args.patch}  pool={pool}  F={F}  用 {len(idx)} 帧", flush=True)

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    model = policy._model                                    # noqa: SLF001
    device = policy._pytorch_device                          # noqa: SLF001
    for p_ in model.parameters():
        p_.requires_grad_(False)
    P = P.to(device)
    R = torch.from_numpy(
        np.random.RandomState(SEED_RAND).randint(0, 256, (224, 224, 3)).astype(np.float32) / 255.0
    ).permute(2, 0, 1)[None].to(device)
    print(f"[policy] loaded device={device}", flush=True)

    def build_obs(k):
        el = {"observation/image": IMG[k], "observation/wrist_image": WRI[k],
              "observation/state": ST[k], "prompt": prompt}
        inputs = policy._input_transform(jax.tree.map(lambda x: x, el))    # noqa: SLF001
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    noise = torch.from_numpy(
        np.random.RandomState(dpt.SEED_NOISE).standard_normal((dpt.AH, dpt.AD)).astype(np.float32)
    ).to(device)[None]

    D = {"drift": [], "rand": []}
    A0 = []
    with torch.no_grad():
        for i, k in enumerate(idx):
            k = int(k)
            o = build_obs(k)
            base = o.images[BASE_KEY].detach().clone()
            mk = torch.from_numpy(MASKS[k].astype(np.float32)).to(device)[None, None]
            a_c = dpt.integrate_actions(model, device, build_obs(k), noise.clone())
            A0.append(a_c[0].float().cpu().numpy())
            for name, Pc in (("drift", P), ("rand", R)):
                o2 = build_obs(k)
                o2.images[BASE_KEY] = base * (1 - mk) + (Pc * 2 - 1) * mk
                a_p = dpt.integrate_actions(model, device, o2, noise.clone())
                D[name].append((a_p - a_c)[0].float().cpu().numpy())
            if i % 16 == 0:
                print(f"  {i}/{len(idx)}", flush=True)
    A0 = np.array(A0)                                        # [N, AH, AD]
    for n in D:
        D[n] = np.array(D[n])

    lines = ["=" * 92,
             f"DRIFT 诊断 · 动作偏差的方向性   patch={args.patch}  pool={pool}  {len(idx)} 帧",
             "  L₀ 是无目标的 ⇒ 必须分清:相干偏移(闭环修不回)vs 不相干抖动(闭环平均掉)",
             "=" * 92]
    for name in ("drift", "rand"):
        dd = D[name]                                          # [N, AH, AD]
        flat = dd.reshape(len(idx), -1)
        nrm = np.linalg.norm(flat, axis=1)
        u = flat / np.maximum(nrm[:, None], 1e-12)
        coh = float(np.linalg.norm(u.mean(0)))                # ‖mean(û)‖ ∈[0,1]
        C = u @ u.T
        iu = np.triu_indices(len(idx), 1)
        cos_med = float(np.median(C[iu]))
        lines.append("")
        lines.append(f"  【{name}】 ‖Δa‖ 中位 {np.median(nrm):.4f}  max {nrm.max():.4f}")
        lines.append(f"    跨帧方向一致性 ‖mean(û)‖ = {coh:.3f}   两两余弦中位 = {cos_med:+.3f}")
        lines.append(f"      (→1 相干同向,闭环修不回;→0 各走各的,闭环会平均掉)")
        lines.append(f"    分维度 |Δa| 中位 / 该维动作本身 |a| 中位:")
        for gname, sl in GROUPS:
            dm = float(np.median(np.abs(dd[:, :, sl])))
            am = float(np.median(np.abs(A0[:, :, sl])))
            rel = dm / am if am > 1e-9 else float("nan")
            mean_signed = dd[:, :, sl].mean()
            lines.append(f"      {gname:14s} |Δa|={dm:.4f}  |a|={am:.4f}  相对 {rel:6.1%}"
                         f"   有符号均值 {mean_signed:+.4f}")
    gap = np.median(np.linalg.norm(D["drift"].reshape(len(idx), -1), axis=1)) / \
        max(np.median(np.linalg.norm(D["rand"].reshape(len(idx), -1), axis=1)), 1e-12)
    lines.append("")
    lines.append(f"  DRIFT 相对随机图案的动作偏差倍数:×{gap:.2f}")
    lines.append("  判读:倍数大但一致性低 ⇒ 只是更吵的噪声,闭环大概率无效(呼应 §2.3 第 3 条);")
    lines.append("        倍数大且一致性高 ⇒ 真·相干误导,闭环应当见效。")
    txt = "\n".join(lines)
    print("\n" + txt, flush=True)
    (OUT / f"{tag}.txt").write_text(txt + "\n")
    np.savez_compressed(OUT / f"{tag}.npz", d_drift=D["drift"], d_rand=D["rand"],
                        a_clean=A0, idx=idx, patch=args.patch, pool=pool)
    print(f"\n[written] {OUT/(tag+'.txt')} + .npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
