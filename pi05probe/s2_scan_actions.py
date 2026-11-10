#!/usr/bin/env python3
"""S2.1 反事实扫描 —— **只 dump 原始 7 维动作**,不做任何通道运算。

为什么重写(s2_scan.py 的错误)
------------------------------
`s2_scan.py` 在 GPU 侧就把 Δa 压成了标量范数 `‖a_patched − a_clean‖`。这违反计划的硬性禁止
(「❌ 提前把 Δa 合并成标量存盘」/「存完整 7 维,绝不提前合并成标量」),后果是**方向信息丢失**:
无法区分「每帧都朝同一方向推」和「每帧方向乱跑」。而这两者的轨迹后果差一个 √T 量级 ——
    ε 抖动:每帧独立重采、零均值 ⇒ 控制器平均掉,累积按 √T
    patch 偏差:同一 patch 同一位置 ⇒ 若同向,累积按 T
拿单帧幅度去比 i.i.d. 噪声的单帧 p95,会系统性低估 patch。

本脚本只存原始动作,聚合/归一化/通道拆分全部交给 report_s2_influence.py。
好处:以后换任何聚合方式(全轨迹幅度版 / 系统版 / coherence / 逐帧剖面 / std_clean 归一化)
都是零成本后处理,不用再上 GPU。

存什么
------
    A_clean   (T, AH, 7)            每帧 clean 动作,用共享 ε_t
    A_patched (M, T, AH, 7)         每锚点每帧 patched 动作,**用同一个 ε_t**(ε 在 Δa 里抵消)
    A_floor   (T, NFLOOR, AH, 7)    每帧换 ε 的 clean 动作 ⇒ ε 地板可同样算幅度版/系统版
存全部 AH=10 步(不只 executed prefix 的 5 步),截断也留给后处理。

用法:
    CUDA_VISIBLE_DEVICES=<free> /home/user1/miniconda3/envs/openpi-server/bin/python \
        pi05probe/s2_scan_actions.py
输入 out/s2_scan_obs.npz(已在盘上,由 s2_scan_dump.py 出,不用再开 LIBERO)
输出 out/s2_actions.npz
"""
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
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np  # noqa: E402

AH, AD = 10, 32          # action horizon / noise dim(见 FINDINGS Q4)
NFLOOR = 10              # 每帧换 ε 的次数,用于 ε 地板
SEED = 777               # 与 s2_scan.py 同 seed,便于对照


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default=str(OUT / "s2_scan_obs.npz"))
    ap.add_argument("--out", default=str(OUT / "s2_actions.npz"))
    ap.add_argument("--prompt", default=None,
                    help="覆盖指令,用于跨任务对照。⚠️ 帧来自 bowl_plate 的轨迹,"
                         "对其他指令是离策略的 —— 与反事实设计一致,但要在结论里写明")
    args = ap.parse_args()

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    d = np.load(args.obs, allow_pickle=True)
    prompt = args.prompt if args.prompt else str(d["prompt"])
    if args.prompt:
        print(f"[cfg] ⚠️ 指令被覆盖为 {prompt!r}(原 {str(d['prompt'])!r});"
              f"帧仍来自 bowl_plate 的轨迹,对本指令是离策略的", flush=True)
    T, M = int(d["T"]), int(d["M"])
    cimg, cwri, cstate = d["clean_img224"], d["clean_wrist224"], d["clean_state8"]
    pimg, pwri = d["patched_img224"], d["patched_wrist224"]
    print(f"[data] M={M} anchors  T={T} frames  prompt={prompt!r}", flush=True)

    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    print("[policy] loaded", flush=True)

    def infer(img, wri, state, eps):
        el = {"observation/image": img, "observation/wrist_image": wri,
              "observation/state": state, "prompt": prompt}
        return np.asarray(policy.infer(el, noise=eps)["actions"], dtype=np.float64)

    rng = np.random.RandomState(SEED)
    eps_shared = np.stack([rng.normal(size=(AH, AD)).astype(np.float32) for _ in range(T)])
    eps_floor = np.stack([np.stack([rng.normal(size=(AH, AD)).astype(np.float32)
                                    for _ in range(NFLOOR)]) for _ in range(T)])

    # ---- 红线 1:先确认固定 ε 下逐位相同(任何非零都说明后面的 Δa 不可信)
    r1 = infer(cimg[0], cwri[0], cstate[0], eps_shared[0])
    r2 = infer(cimg[0], cwri[0], cstate[0], eps_shared[0])
    dmax = float(np.abs(r1 - r2).max())
    print(f"[红线1] 固定 ε 重复前向 |Δ|max = {dmax:.3e}  {'✅' if dmax == 0.0 else '❌ 非确定,停'}",
          flush=True)
    assert dmax == 0.0, "固定 ε 下前向不确定,Δa 会混入非确定性噪声,必须先查清"
    assert r1.shape == (AH, 7), f"动作形状意外:{r1.shape},预期 ({AH}, 7)"

    A_clean = np.zeros((T, AH, 7))
    A_floor = np.zeros((T, NFLOOR, AH, 7))
    for t in range(T):
        A_clean[t] = infer(cimg[t], cwri[t], cstate[t], eps_shared[t])
        for j in range(NFLOOR):
            A_floor[t, j] = infer(cimg[t], cwri[t], cstate[t], eps_floor[t, j])
        print(f"  [clean+floor] frame {t:2d}/{T}", flush=True)

    A_patched = np.zeros((M, T, AH, 7))
    for i in range(M):
        for t in range(T):
            A_patched[i, t] = infer(pimg[i, t], pwri[i, t], cstate[t], eps_shared[t])
        if i % 6 == 0:
            # 只作进度提示;真正的判读全在后处理
            dd = np.linalg.norm(A_patched[i, :, :5, 0:3] - A_clean[None, :, :5, 0:3])
            print(f"  [scan] anchor {i:2d}/{M}  ‖Δ平移‖(全帧合计)={dd:.4f}", flush=True)

    outp = pathlib.Path(args.out)
    np.savez_compressed(
        outp,
        A_clean=A_clean, A_patched=A_patched, A_floor=A_floor,
        eps_shared=eps_shared, eps_floor=eps_floor,
        seed=SEED, AH=AH, AD=AD, nfloor=NFLOOR, prompt=prompt,
        T=T, M=M, ts=d["ts"], ks=d["ks"],
        clean_state8=cstate, visible_px=d["visible_px"],
        anchor_world=d["anchor_world"], anchor_uv=d["anchor_uv"],
        anchor_idx=d["anchor_idx"], anchor_legal=d["anchor_legal"],
        anchor_keepout=d["anchor_keepout"])
    print(f"\n[written] {outp}  ({outp.stat().st_size / 2**20:.2f} MiB)", flush=True)
    print("  下一步:report_s2_influence.py(纯后处理,全轨迹 + 逐帧两层)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
