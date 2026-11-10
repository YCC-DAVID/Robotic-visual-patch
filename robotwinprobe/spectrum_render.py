"""Render every spectrum test texture through real SAPIEN at fixed initial states (sim env, no policy)."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

from spectrum_common import DEFAULT, E_ROOT, SEEDS, SIZES, textures, dump
from e_method_common import CAMERAS, sha


def render(root, seed, size, tex):
    import bootstrap as B
    B.simulator_paths()
    import sapien
    from scene_smoke import make_task
    from render_geometry import Patch, camera_objects
    from eot_conditions import Lighting, LIGHTS
    out = root/'renders'/f's{seed}_size{int(round(size*100)):02d}.npz'
    if out.exists(): return
    p = json.loads((E_ROOT/'protocol.json').read_text())
    assets = root/'assets'/f's{seed}'; assets.mkdir(parents=True, exist_ok=True)
    random.seed(seed); task = None; started = time.time()
    try:
        task, _, _ = make_task(p['task'], seed, assets)
        prompt = 'Place the '+('bowl' if task.actor_name == '002_bowl' else 'cup')+' on the plate.'
        task.set_instruction(prompt)
        Lighting(task).apply(LIGHTS['nominal'])
        clean = task.get_obs(); state = clean['joint_action']['vector'].astype(np.float32)
        center = np.array(p['patch']['center_m'])
        data = dict(seed=seed, query=0, prompt=prompt, state14=state,
                    active_start=7 if task.container.get_pose().p[0] > 0 else 0,
                    center_m=center, size_m=size, yaw_deg=0., light='nominal')
        for camera in CAMERAS: data[camera+'_clean'] = clean['observation'][camera]['rgb'].copy()
        patch = Patch(task, assets, center=center, size=size)
        try:
            for index, (label, texture) in enumerate(tex.items()):
                patch.texture(texture)
                patch.actor.set_pose(sapien.Pose(center, [1, 0, 0, 0]))
                rendered = task.get_obs(); cameras = camera_objects(task)
                assert np.array_equal(rendered['joint_action']['vector'], clean['joint_action']['vector'])
                for camera in CAMERAS:
                    data[camera+'_'+label] = rendered['observation'][camera]['rgb'].copy()
                    if index == 0:
                        data[camera+'_mask'] = cameras[camera].get_picture('Segmentation')[..., 1] == patch.actor.per_scene_id
                        data[camera+'_intrinsic'] = cameras[camera].get_intrinsic_matrix()
                        data[camera+'_extrinsic'] = cameras[camera].get_extrinsic_matrix()
        finally: task.scene.remove_actor(patch.actor)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix('.tmp.npz'); np.savez_compressed(tmp, **data); tmp.replace(out)
        print('[render]', out.name, f'{time.time()-started:.0f}s', flush=True)
    finally:
        if task is not None: task.close_env()


def main(args):
    root = args.root.resolve(); tex, meta = textures()
    root.mkdir(parents=True, exist_ok=True); (root/'render_done.json').unlink(missing_ok=True)
    np.savez_compressed(root/'textures.npz', **tex)
    dump(root/'textures.json', dict(order=list(tex), meta=meta, sha256=sha(root/'textures.npz')))
    for seed in args.seeds:
        for size in args.sizes: render(root, seed, size, tex)
    dump(root/'render_done.json', dict(seeds=args.seeds, sizes=args.sizes, complete=True))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=DEFAULT)
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--sizes', type=float, nargs='+', default=list(SIZES))
    main(ap.parse_args())
