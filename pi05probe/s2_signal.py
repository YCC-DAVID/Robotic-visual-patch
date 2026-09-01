#!/usr/bin/env python3
"""S2.0 信号检查(py3.11,in-process,共享 ε)。

三件事(计划 S2.0):
  ① ε 噪声地板:同一 clean 观测、**换 ε** 前向 N 次 → 动作的 ε 诱导散布(分通道)。
  ② 确定性 + batch 约定:同一观测、**同一 ε** 前向两次 → Δ≈0(共享-ε 路径确定)。
     红线 3(batch 形状改结果):我们**全程 batch=1**(policy.infer 固定 batch=1),
     从构造上避免 —— 计划许可的兜底("全部逐个 batch=1")。
  ③ patch 信号:几个锚点 clean vs patched、**共享 ε** → 分通道 ‖Δa‖ 是否显著高于 ε 地板。

分通道(计划 D,绝不对 7 维直接 L2):
  平移 a[0:3] / 旋转 a[3:6](axis-angle)/ 夹爪 a[6](只看 sign 是否翻)。
  在 executed prefix(前 5 步,replan=5)上聚合。

必须 in-process:websocket 传不了 noise(§PT-6)。
用法:
    # 先 s2_dump.py 出 s2_obs.npz(py3.8)
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/s2_signal.py
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

# transformers_patched 必须最前;再 openpi src / openpi-client
for p in reversed([PATCHED_TF, OPENPI / "src", OPENPI / "packages" / "openpi-client" / "src"]):
    sys.path.insert(0, str(p))
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["OPENPI_DATA_HOME"] = "/home/user1/.cache/openpi"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

import numpy as np

AH, AD = 10, 32       # action_horizon, 内部 action_dim
EX = 5                # executed prefix(replan_steps)
NFLOOR = 20
SEED = 12345


def rot_geodesic(a3, b3):
    """两个 axis-angle 向量间的 SO(3) 测地距离(弧度)。"""
    from scipy.spatial.transform import Rotation as R
    ra, rb = R.from_rotvec(a3), R.from_rotvec(b3)
    rel = ra.inv() * rb
    return float(np.linalg.norm(rel.as_rotvec()))


def channels(a_ex):
    """a_ex: (EX,7)。返回可比的三通道量。"""
    return a_ex


def delta_channels(a, b):
    """a,b: (EX,7)。分通道差:平移 Frob(m 需 ×0.05)、旋转逐步测地和、夹爪翻转步数。"""
    dt = np.linalg.norm((a[:, 0:3] - b[:, 0:3]))                    # 平移(action 单位,×0.05=m)
    dr = float(np.sum([rot_geodesic(a[i, 3:6], b[i, 3:6]) for i in range(a.shape[0])]))  # 旋转测地和
    dg = int(np.sum(np.sign(a[:, 6]) != np.sign(b[:, 6])))         # 夹爪翻转步数
    return dt, dr, dg


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="s2_obs.npz")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import jax
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    d = np.load(OUT / args.obs, allow_pickle=True)
    prompt = str(d["prompt"])
    clean = {"observation/image": d["clean_img224"], "observation/wrist_image": d["clean_wrist224"],
             "observation/state": d["clean_state8"], "prompt": prompt}
    P_img, P_wri = d["patched_img224"], d["patched_wrist224"]
    vpx, aw = d["visible_px"], d["anchor_world"]
    M = len(P_img)
    print(f"[data] patched 锚点 {M} 个  clean prompt={prompt!r}", flush=True)

    assert TORCH_CKPT.joinpath("model.safetensors").exists(), "先转权重"
    policy = _policy_config.create_trained_policy(_config.get_config("pi05_libero"), TORCH_CKPT)
    print("[policy] in-process 加载完成(PyTorch)", flush=True)

    rng = np.random.RandomState(SEED)

    def infer(elem, eps):
        return np.asarray(policy.infer(elem, noise=eps)["actions"])[:EX]   # (EX,7)

    lines = []
    def out(s=""):
        print(s, flush=True); lines.append(s)

    # ---------- ② 确定性(先做:验证共享-ε 路径确定) ----------
    eps0 = rng.normal(size=(AH, AD)).astype(np.float32)
    a1 = infer(clean, eps0)
    a2 = infer(clean, eps0)
    det = float(np.abs(a1 - a2).max())
    out("=" * 88)
    out("② 确定性 / batch 约定")
    out("=" * 88)
    out(f"  同一 clean 观测、同一 ε 前向两次:|Δ|max = {det:.3e}  "
        + ("✅ 确定" if det < 1e-4 else "⚠️ 不确定,共享-ε 假设有问题"))
    out("  红线3:全程 batch=1(policy.infer 固定 batch=1)⇒ batch 形状不引入伪差异。")

    # ---------- ① ε 噪声地板 ----------
    out("\n" + "=" * 88)
    out(f"① ε 噪声地板:clean 观测换 {NFLOOR} 个 ε 前向,executed prefix(前{EX}步)的动作散布")
    out("=" * 88)
    acts = [infer(clean, rng.normal(size=(AH, AD)).astype(np.float32)) for _ in range(NFLOOR)]
    # 两两 Δ 分通道
    fl_t, fl_r, fl_g = [], [], []
    for i in range(NFLOOR):
        for j in range(i + 1, NFLOOR):
            t, r, g = delta_channels(acts[i], acts[j])
            fl_t.append(t); fl_r.append(r); fl_g.append(g)
    fl_t, fl_r, fl_g = np.array(fl_t), np.array(fl_r), np.array(fl_g)
    out(f"  平移 ‖Δ‖(action单位; ×0.05=m): mean={fl_t.mean():.4f} p95={np.percentile(fl_t,95):.4f}  "
        f"(≈{fl_t.mean()*0.05*1000:.1f} mm)")
    out(f"  旋转 测地和(rad):            mean={fl_r.mean():.4f} p95={np.percentile(fl_r,95):.4f}")
    out(f"  夹爪 翻转步数(/{EX}):         mean={fl_g.mean():.3f} max={fl_g.max()}")
    floor_t95 = np.percentile(fl_t, 95); floor_r95 = np.percentile(fl_r, 95)

    # ---------- ③ patch 信号 ----------
    out("\n" + "=" * 88)
    out("③ patch 信号:clean vs patched,**共享 ε**,分通道 Δa 是否 > ε 地板 p95")
    out("=" * 88)
    # 锚点少(诊断集)则全测;多则挑 可见像素最大/中位/最小 + 前两个
    if M <= 8:
        pick = list(range(M))
    else:
        order = np.argsort(vpx)
        pick = sorted(set([int(order[-1]), int(order[len(order)//2]), int(order[0]), 0, min(1, M-1)]))
    out(f"  ε 地板 p95:平移={floor_t95:.4f}  旋转={floor_r95:.4f}")
    out("")
    out("  anchor  world(x,y)      vpx |  平移‖Δ‖ (mm) |  旋转Δ(rad) | 夹爪翻转 | 显著?")
    sig_any = False
    for idx in pick:
        elem = {"observation/image": P_img[idx], "observation/wrist_image": P_wri[idx],
                "observation/state": d["clean_state8"], "prompt": prompt}
        ap = infer(elem, eps0)                       # 共享 eps0
        ac = infer(clean, eps0)                       # 同 ε 的 clean 参考
        t, r, g = delta_channels(ap, ac)
        sig = (t > floor_t95) or (r > floor_r95) or (g > 0)
        sig_any = sig_any or sig
        out(f"  #{d['anchor_idx'][idx]:<3d} ({aw[idx][0]:5.2f},{aw[idx][1]:5.2f}) {vpx[idx]:4d} | "
            f"{t:8.4f} ({t*0.05*1000:5.1f}) | {r:9.4f} | {g:6d}/{EX} | {'✅' if sig else '·'}")

    out("\n" + "=" * 88)
    out(f"S2.0 结论:确定性 {'✅' if det<1e-4 else '❌'};patch 信号 "
        + ("✅ 至少一个锚点显著高于 ε 地板 ⇒ influence 这条路走得通,可进 S2.1。"
           if sig_any else
           "❌ 没有锚点超过 ε 地板 ⇒ 纹理不够凶或位置不对,先加强再继续(计划 S2.0)。"))
    outtxt = OUT / f"s2_signal{args.tag}.txt"
    outtxt.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {outtxt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
