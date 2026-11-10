#!/usr/bin/env python3
"""对抗 patch 训练 —— Phase A(py3.8,纯渲染):在目标合法格渲染 plate 轨迹各帧的
clean 模型输入 + patch 可见性 mask + patch 四角。

关键简化:patch 与相机都静止 ⇒ patch 在 224 图像里的投影足迹每帧相同,只是被机械臂
遮挡的部分不同。所以训练时在**固定的 224 足迹像素**上优化,每帧用可见性 mask 合成,
自动跨帧一致,无需可微渲染 / homography。最后把优化好的足迹反 warp 成方形 PNG 做物理验收。

输出 out/patch_prep.npz:clean_img224/wrist224/state8(逐帧)、vis_mask(逐帧 224 bool)、
patch 四角(224 空间,给反 warp)、目标格 world。
用法: ~/miniconda3/envs/openpi-libero/bin/python pi05probe/patch_prep.py
"""
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
PI05 = ROOT / "pi05probe"
OUT = PI05 / "out"
sys.path.insert(0, str(PI05))
sys.path.insert(0, str(ROOT / "third_party" / "openpi" / "third_party" / "libero"))

import numpy as np
import yaml
import s2_dump as base
import scene_patch as sp

TASK = "put_the_bowl_on_the_plate"
CELL_WORLD = (0.045, 0.117)          # influence & gradient 合法首选格
PATCH_M = 0.10                       # 训练用 10cm(足迹够大)
FRAMES = [0, 2, 4, 6, 8, 10, 12, 14]  # 跨轨迹取 8 帧做 EOT


def cam_T():
    from robosuite.utils import transform_utils as TU
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import torch
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid = next(i for i in range(suite.n_tasks)
               if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    t = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / t.problem_folder / t.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(10000); env.reset()
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) / "libero_goal" / f"{TASK}.pruned_init")))
    env.set_init_state(inits[0])
    sim = env.env.sim
    cid = sim.model.camera_name2id("agentview")
    fovy = sim.model.cam_fovy[cid]
    f = 0.5 * 256 / np.tan(fovy * np.pi / 360)
    K = np.array([[f, 0, 128], [0, f, 128], [0, 0, 1]]); Kx = np.eye(4); Kx[:3, :3] = K
    R = TU.make_pose(sim.data.cam_xpos[cid], sim.data.cam_xmat[cid].reshape(3, 3))
    R = R @ np.array([[1., 0, 0, 0], [0, -1., 0, 0], [0, 0, -1., 0], [0, 0, 0, 1.]])
    env.close()
    return Kx @ TU.pose_inv(R)


def corners224(T, cx, cy, z, half):
    """patch 四角 world → 224 模型输入(朝向 code=(0,1,0):撤 rot,只翻列)。"""
    pts = [(cx-half, cy-half), (cx+half, cy-half), (cx+half, cy+half), (cx-half, cy+half)]
    out = []
    for x, y in pts:
        ph = T @ np.array([x, y, z, 1.0]); r256, c256 = ph[1]/ph[2], ph[0]/ph[2]
        c256 = 255 - c256                    # code=(0,1,0)
        out.append((r256 * 224 / 256, c256 * 224 / 256))    # (row,col) 224
    return np.array(out)


def main():
    from libero.libero.envs import OffScreenRenderEnv
    cfg = yaml.safe_load((PI05 / "config" / "scene.yaml").read_text())
    cfg["patch"]["size_wh"] = [PATCH_M, PATCH_M]
    tex = str(PI05 / "config" / "probe_texture.png")
    seed = int(cfg["shared_seed"])
    lift = cfg["patch"]["thickness"] / 2 + cfg["patch"]["normal_offset"]
    z = cfg["plane"]["origin"][2] + lift
    world = np.array([CELL_WORLD[0], CELL_WORLD[1], z])
    bddl = base.bddl_path(TASK)

    tr = np.load(OUT / f"traj_{TASK}.npz", allow_pickle=True)
    flats = [tr[f"f{k:03d}__flatten"] for k in FRAMES]
    T = cam_T()
    quad = corners224(T, CELL_WORLD[0], CELL_WORLD[1], z, PATCH_M / 2)
    print(f"[cfg] cell world={CELL_WORLD} z={z:.4f} patch={PATCH_M*100:.0f}cm frames={FRAMES}", flush=True)
    print(f"[quad] 224 corners row,col=\n{np.round(quad,1)}", flush=True)

    # clean
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                             camera_segmentations="element")
    env.seed(seed); env.reset()
    c_img, c_wri, c_st = [], [], []
    for fl in flats:
        obs = env.regenerate_obs_from_state(fl)
        pi, pw = base.model_input(obs)
        c_img.append(pi); c_wri.append(pw); c_st.append(base.state8(obs))
    env.close()

    # patched → vis mask(224,patch geom 可见像素)
    penv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                              camera_segmentations="element")
    penv.env.set_xml_processor(sp.make_xml_processor(cfg, world, tex))
    penv.seed(seed); penv.reset()
    gid = sp.patch_geom_id(penv, cfg)
    vis = []
    for fl in flats:
        obs = penv.regenerate_obs_from_state(fl)
        seg = obs["agentview_segmentation_element"][..., 0][::-1, ::-1]   # 256 模型输入朝向
        from openpi_client import image_tools
        m256 = np.repeat(((seg == gid).astype(np.uint8) * 255)[..., None], 3, axis=2)
        m224 = np.asarray(image_tools.resize_with_pad(m256, 224, 224))[..., 0] > 127
        vis.append(m224)
        print(f"  frame vpx224={int(m224.sum())}", flush=True)
    penv.close()

    np.savez_compressed(OUT / "patch_prep.npz", task=TASK, cell_world=np.array(CELL_WORLD),
                        z=z, patch_m=PATCH_M, frames=np.array(FRAMES), quad224=quad,
                        clean_img224=np.array(c_img), clean_wrist224=np.array(c_wri),
                        clean_state8=np.array(c_st), vis_mask=np.array(vis))
    print(f"[written] {OUT/'patch_prep.npz'}  frames={len(FRAMES)}", flush=True)


if __name__ == "__main__":
    main()
