#!/usr/bin/env python3
"""S2.0 诊断:把 patch 放到**任务相关**位置(bowl/plate 上及旁边),隔离"位置"这一个变量。

S2.0 信号检查发现:front-right 网格(x 0.05–0.30)上所有锚点的 Δa 都远低于 ε 地板。
但 bowl(x≈-0.10)/plate(x≈0.06)才是驱动动作的物体,网格根本没覆盖它们。
本脚本在同一共享帧、同样大小的 patch、只改**位置**:若 on-bowl/on-plate 的 Δa 暴涨、
far-ref 仍≈0 ⇒ 是位置问题(而且正是 influence 该有的空间结构),网格要伸向任务物体。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/s2_diag_dump.py
    /home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/s2_signal.py --obs s2_obs_diag.npz --tag _diag
"""
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"
sys.path.insert(0, str(PROBE))

import argparse
import numpy as np
import yaml
import s2_dump as base   # 复用 model_input/state8/bddl_path,并已设好 env vars & sys.path
import scene_patch as sp

# (label, world_x, world_y) —— z 用桌面 top。bowl=(-0.098,-0.009) plate=(0.062,-0.009)
POSITIONS = [
    ("on_bowl",     -0.098, -0.009),   # 直接盖在 bowl 上(诊断,允许重叠)
    ("on_plate",     0.062, -0.009),   # 盖在 plate 上
    ("between",     -0.020, -0.005),   # bowl 与 plate 之间(动作路径上)
    ("beside_bowl", -0.098,  0.160),   # bowl 同 x、y 抬出 keepout(靠近但不遮挡)
    ("far_ref",      0.240,  0.020),   # 任务无关参照(现网格里的点)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=None,
                    help="用 traj 的第几个 replan 帧的状态;缺省=共享帧 t000(arm 静止)")
    args = ap.parse_args()

    from libero.libero.envs import OffScreenRenderEnv
    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())
    tex = str(cfg["patch"].get("texture") or (PROBE / "config" / "probe_texture.png"))
    seed = int(cfg["shared_seed"])
    bddl = base.bddl_path(base.TASK)
    z_top = cfg["plane"]["origin"][2] + cfg["patch"]["thickness"] / 2 + cfg["patch"]["normal_offset"]

    if args.frame is None:
        sf = np.load(OUT / "shared_frame.npz", allow_pickle=False)
        shared_flat = sf[f"{base.TASK}__flatten"]
        frame_tag = "t000"
    else:
        tr = np.load(OUT / f"traj_{base.TASK}.npz", allow_pickle=True)
        shared_flat = tr[f"f{args.frame:03d}__flatten"]
        frame_tag = f"f{args.frame:03d}(t={int(tr['ts'][args.frame])})"
    prompt = "put the bowl on the plate"
    print(f"[cfg] 帧={frame_tag}", flush=True)

    # clean
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                             camera_segmentations="element")
    env.seed(seed); env.reset()
    obs = env.regenerate_obs_from_state(shared_flat)
    c_img, c_wri = base.model_input(obs); c_state = base.state8(obs)
    env.close()

    P_img, P_wri, P_world, P_idx, P_vpx, labels = [], [], [], [], [], []
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                              camera_segmentations="element")
    for j, (lab, x, y) in enumerate(POSITIONS):
        world = np.array([x, y, z_top])
        penv.env.set_xml_processor(sp.make_xml_processor(cfg, world, tex))
        penv.seed(seed); penv.reset()
        obs = penv.regenerate_obs_from_state(shared_flat)
        gid = sp.patch_geom_id(penv, cfg)
        vpx = sp.visible_px(obs, "agentview", gid)
        pi, pw = base.model_input(obs)
        P_img.append(pi); P_wri.append(pw); P_world.append(world)
        P_idx.append(j); P_vpx.append(vpx); labels.append(lab)
        print(f"  {lab:12s} world=({x:.3f},{y:.3f}) vpx={vpx}", flush=True)
    penv.close()

    fsuf = "t000" if args.frame is None else f"f{args.frame:03d}"
    outp = OUT / f"s2_obs_diag_{fsuf}.npz"
    np.savez_compressed(
        outp, task=base.TASK, prompt=prompt, shared_seed=seed,
        clean_img224=c_img, clean_wrist224=c_wri, clean_state8=c_state,
        patched_img224=np.array(P_img), patched_wrist224=np.array(P_wri),
        anchor_world=np.array(P_world), anchor_uv=np.array(P_world)[:, :2],
        anchor_idx=np.array(P_idx), visible_px=np.array(P_vpx), labels=np.array(labels))
    print(f"[written] {outp}  patched={len(P_img)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
