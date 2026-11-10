#!/usr/bin/env python3
"""回放那条轨迹的 16 帧,逐帧存腕部相机的内外参。

为什么需要
--------
腕部相机跟着夹爪走,每一帧位姿都不同。要把碗/盘子的世界位置投影到腕部图像里
(判断腕部注意力落在操作对象还是目的地),必须有**逐帧**的外参。
`shared_frame.npz` 里只有单帧的 `K_wrist`/`E_wrist`,不够用。

相机矩阵的约定与 `dump_shared_frame.py` 完全一致(直接复用它的两个函数),
否则和已验证过的主视角投影链路对不上。

用法(py3.8):
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/dump_wrist_cams.py
"""
import os
import pathlib
import sys

ROOT = pathlib.Path("/home/user1/workspace/chence/WAMattack")
OPENPI = ROOT / "third_party" / "openpi"
PROBE = ROOT / "pi05probe"
OUT = PROBE / "out"

for p in reversed([OPENPI / "packages" / "openpi-client" / "src", OPENPI / "third_party" / "libero"]):
    sys.path.insert(0, str(p))
sys.path.insert(0, str(PROBE))
os.environ["LIBERO_CONFIG_PATH"] = str(PROBE / "libero_config")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["PYTHONNOUSERSITE"] = "1"

import numpy as np  # noqa: E402
from dump_shared_frame import cam_intrinsics, cam_extrinsics  # noqa: E402  单一来源

RES = 256
TASK = "put_the_bowl_on_the_plate"


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    tz = np.load(OUT / f"traj_{TASK}.npz", allow_pickle=False)
    T = int(tz["n_frames"])
    seed = int(tz["shared_seed"])

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
    env.seed(seed)
    env.reset()

    Kw = None
    Ew = np.zeros((T, 4, 4))
    Ka = None
    Ea = np.zeros((T, 4, 4))
    for k in range(T):
        env.regenerate_obs_from_state(tz[f"f{k:03d}__flatten"])
        if Kw is None:
            Kw, _ = cam_intrinsics(env, "robot0_eye_in_hand", RES)
            Ka, _ = cam_intrinsics(env, "agentview", RES)
        Ew[k] = cam_extrinsics(env, "robot0_eye_in_hand")
        Ea[k] = cam_extrinsics(env, "agentview")
    env.close()

    # 自检 1:主视角外参应逐帧不变(它是固定相机)
    da = np.abs(Ea - Ea[0]).max()
    # 自检 2:腕部外参必须逐帧变化(它跟着夹爪)
    dw = np.abs(Ew - Ew[0]).max()
    print(f"[自检] 主视角外参逐帧最大变化 = {da:.2e}  ({'✅ 固定' if da < 1e-9 else '❌ 竟然在动'})")
    print(f"[自检] 腕部外参逐帧最大变化 = {dw:.3f}  ({'✅ 在动' if dw > 1e-3 else '❌ 竟然不动'})")
    print(f"[自检] 主视角 K 与 shared_frame 的差 = "
          f"{np.abs(Ka - np.load(OUT/'shared_frame.npz')[f'{TASK}__K_agentview']).max():.2e}")

    out = OUT / "wrist_cams.npz"
    np.savez_compressed(out, K_wrist=Kw, E_wrist=Ew, K_agentview=Ka, E_agentview=Ea,
                        ts=tz["ts"], n_frames=T)
    print(f"[written] {out}  K_wrist={Kw.shape}  E_wrist={Ew.shape}")


if __name__ == "__main__":
    main()
