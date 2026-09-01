#!/usr/bin/env python3
"""S2.1 影响力热图(纯后处理,从 s2_influence.npz 出图)。py3.8 有 matplotlib。

用法:
    ~/miniconda3/envs/openpi-libero/bin/python pi05probe/s2_heat.py
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path("/home/user1/workspace/chence/WAMattack/pi05probe/out")
NU, NV = 6, 6


def main():
    d = np.load(OUT / "s2_influence.npz", allow_pickle=True)
    aw = d["anchor_world"]; keep = d["anchor_keepout"]; reliable = d["reliable"]
    I_t, I_r, I_g = d["I_trans"], d["I_rot"], d["I_grip"]
    ext = [aw[:, 0].min(), aw[:, 0].max(), aw[:, 1].min(), aw[:, 1].max()]

    for arr, name, unit in [(I_t * 0.05 * 1000, "translation", "mm"),
                            (I_r, "rotation", "rad"), (I_g.astype(float), "gripper_flips", "count")]:
        grid = arr.reshape(NV, NU)
        fig, ax = plt.subplots(figsize=(5.4, 4.7))
        im = ax.imshow(grid, origin="lower", extent=ext, aspect="auto", cmap="inferno")
        ax.scatter([-0.098], [-0.009], c="cyan", marker="o", s=90, label="bowl", edgecolors="k", zorder=5)
        ax.scatter([0.062], [-0.009], c="lime", marker="s", s=90, label="plate", edgecolors="k", zorder=5)
        for i in range(len(aw)):
            if str(keep[i]):
                ax.plot(aw[i, 0], aw[i, 1], "x", color="white", ms=6, mew=1.3, zorder=4)
            if not reliable[i]:
                ax.plot(aw[i, 0], aw[i, 1], ".", color="gray", ms=3, zorder=4)
        ax.set_title(f"S2.1 influence: {name} ({unit})  [x=occludes obj, .=below floor]")
        ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
        ax.legend(loc="upper right", fontsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(); fig.savefig(OUT / f"s2_heat_{name}.png", dpi=120); plt.close(fig)
        print(f"[written] s2_heat_{name}.png", flush=True)


if __name__ == "__main__":
    main()
