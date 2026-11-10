"""Validate actual SAPIEN device, assets and visual-only patch before training."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import bootstrap as B
from main_method_common import TASK,CAMERAS,CENTER,SIZE,dump,sha


def main(args):
    B.simulator_paths()
    from main_method_runtime import install_path_normalization
    install_path_normalization(args.out)
    from main_method_rollout import pin_sapien_device,actors,digest
    from scene_smoke import make_task
    from render_geometry import Patch,camera_objects,patterns
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    binding=pin_sapien_device(args.gpu_uuid,out)
    seed=25000  # training seed; test episodes are not inspected here
    random.seed(seed);task,config,embodiment=make_task(TASK,seed,out)
    try:
        clean=task.get_obs();state=clean['joint_action']['vector'].astype(np.float32).copy()
        original=actors(task)
        for c in CAMERAS:
            image=clean['observation'][c]['rgb']
            assert image.dtype==np.uint8 and image.shape==(240,320,3),(c,image.dtype,image.shape)
        patch=Patch(task,out,center=CENTER,size=SIZE);patch.texture(patterns()['random'])
        obs=task.get_obs()
        assert np.array_equal(state,obs['joint_action']['vector'].astype(np.float32))
        assert [a for a in actors(task) if a['name']!='wam_visual_patch']==original
        cameras=camera_objects(task)
        visible={c:int((cameras[c].get_picture('Segmentation')[...,1]==patch.actor.per_scene_id).sum()) for c in CAMERAS}
        assert any(visible.values()),'patch invisible in all policy cameras at training scene'
        report=dict(passed=True,task=TASK,training_seed=seed,embodiment=embodiment,
            size_m=SIZE,center_m=CENTER,binding=binding,visible_pixels=visible,
            state_sha256=digest(state),native_rgb_shapes={c:list(clean['observation'][c]['rgb'].shape) for c in CAMERAS},
            source_sha256={n:sha(B.PROBE/n) for n in ['main_method_scene_check.py','main_method_rollout.py','scene_smoke.py','render_geometry.py']})
        dump(out/'passed.json',report);print(json.dumps(report),flush=True)
    finally:task.close_env()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--gpu-uuid',required=True)
    main(ap.parse_args())
