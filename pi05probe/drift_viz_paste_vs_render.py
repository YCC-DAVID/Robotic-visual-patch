#!/usr/bin/env python3
"""可视化:同一帧 · 干净 / 直接贴 P(评测 A)/ MuJoCo 渲染(评测 B)/ |差|,附足迹放大。

用法(openpi-libero, py3.8):
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/drift_viz_paste_vs_render.py \
        --patches drift,band32,eotfull --init 0 --warm 30
产物 out/viz_paste_vs_render.png
"""
import argparse
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

RES, RESIZE, NWAIT = 256, 224, 10
DUMMY = [0.0] * 6 + [-1.0]
TASK = "put_the_bowl_on_the_plate"
ZOOM = 4


def model_input(obs):
    from openpi_client import image_tools
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default="drift,band32,eotfull")
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--warm", type=int, default=30, help="init 后走几步 dummy")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--thresh", type=int, default=12)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    import torch

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tid = next(i for i in range(suite.n_tasks)
               if pathlib.Path(suite.get_task(i).bddl_file).stem == TASK)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path("init_states")) /
                                    "libero_goal" / f"{TASK}.pruned_init")))
    cfg = yaml.safe_load((PROBE / "config" / "scene.yaml").read_text())

    # 每次渲染都重建 env + 同 seed(fixture 不在 qpos 里,见 drift_fit_render.py 的注释)
    def render(tex, st, world):
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        if tex is not None:
            env.env.set_xml_processor(sp.make_xml_processor(cfg, world, str(tex)))
        env.seed(args.seed); env.reset()
        img = model_input(env.regenerate_obs_from_state(st))
        env.close()
        return img

    rows = []
    for ptag in args.patches.split(","):
        d = np.load(OUT / f"drift_patch_{ptag}.npz", allow_pickle=True)
        patch_m = float(d["patch_m"])
        cfg["patch"]["size_wh"] = [patch_m, patch_m]
        plane = sp.Plane.from_cfg(cfg["plane"])
        lift = cfg["patch"]["thickness"] / 2.0 + cfg["patch"]["normal_offset"]
        world = plane.to_world(float(d["cell_xy"][0]), float(d["cell_xy"][1]), lift)

        cenv = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=RES, camera_widths=RES)
        cenv.seed(args.seed); cenv.reset()
        cenv.set_init_state(inits[args.init])
        for _ in range(NWAIT + args.warm):
            cenv.step(DUMMY)
        st = cenv.get_sim_state().copy()
        clean = model_input(cenv.regenerate_obs_from_state(st))
        cenv.close()

        probe = render(PROBE / "config" / "probe_texture.png", st, world)
        m = np.abs(probe.astype(np.int16) - clean.astype(np.int16)).max(-1) > args.thresh
        P_u8 = np.clip(d["P"].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        paste = np.where(m[..., None], P_u8, clean).astype(np.uint8)
        mj = render(PROBE / "config" / f"drift_texture_{ptag}.png", st, world)
        diff = np.clip(np.abs(paste.astype(np.int16) - mj.astype(np.int16)) * 2, 0, 255).astype(np.uint8)
        mad = float(np.abs(paste.astype(np.float32) - mj.astype(np.float32))[m].mean() / 255)
        print(f"[{ptag}] footprint {int(m.sum())}px  mean|paste-mujoco| in footprint = {mad:.3f}", flush=True)

        ys, xs = np.where(m)
        r0, r1 = max(ys.min() - 4, 0), min(ys.max() + 5, RESIZE)
        c0, c1 = max(xs.min() - 4, 0), min(xs.max() + 5, RESIZE)
        full = [label(clean, f"{ptag}: clean"), label(paste, "PASTE (eval A)"),
                label(mj, "MUJOCO RENDER (eval B)"), label(diff, f"|diff|x2 mean={mad:.3f}")]
        crops = []
        for im, t in zip((clean, paste, mj, diff), ("clean", "PASTE", "MUJOCO", "|diff|x2")):
            c = cv2.resize(im[r0:r1, c0:c1], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
            crops.append(label(c, f"zoom x{ZOOM}: {t}"))
        pad = lambda x: cv2.copyMakeBorder(x, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        top = np.concatenate([pad(f) for f in full], 1)
        bot = np.concatenate([pad(c) for c in crops], 1)
        w = max(top.shape[1], bot.shape[1])
        top = cv2.copyMakeBorder(top, 0, 0, 0, w - top.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        bot = cv2.copyMakeBorder(bot, 0, 0, 0, w - bot.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        rows.append(np.concatenate([top, bot], 0))

    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 10, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            for r in rows]
    grid = np.concatenate(rows, 0)
    outp = OUT / "viz_paste_vs_render.png"
    cv2.imwrite(str(outp), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[written] {outp}  {grid.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
