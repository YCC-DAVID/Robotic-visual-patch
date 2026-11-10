"""Actual RoboTwin scene and perspective checker; no policy or synthetic overlay."""
import argparse
import json
import os
import numpy as np
import bootstrap as B
B.simulator_paths()


def make_task(task_name, seed, save_dir):
    os.chdir(B.ROBOTWIN)
    from eval_policy_xpolicylab import load_task_args, class_decorator
    args, embodiment = load_task_args({'task_name': task_name, 'task_config': 'demo_clean',
                                      'policy_name': 'wam_pi05', 'ckpt_name': 'motus_pi05'})
    args.update(eval_mode=True, save_data=False, collect_data=False, eval_video_log=False,
                render_freq=0, save_path=str(save_dir))
    task = class_decorator(task_name)
    task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    return task, args, embodiment


def add_patch(task, dest, center=(0., -.24, .741), size=.13):
    import sapien
    from PIL import Image
    colors = np.array([[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255]], np.uint8)
    yy, xx = np.indices((128,128))
    texture = colors[(xx//16+2*(yy//16))%len(colors)]
    texture[:8,:8] = 255; texture[-8:,-8:] = 0
    Image.fromarray(texture).save(dest/'checker.png')
    h = size/2
    obj = '\n'.join([f'v {-h} {-h} 0', f'v {h} {-h} 0', f'v {h} {h} 0', f'v {-h} {h} 0',
                     'vt 0 0', 'vt 1 0', 'vt 1 1', 'vt 0 1', 'vn 0 0 1',
                     'f 1/1/1 2/2/1 3/3/1', 'f 1/1/1 3/3/1 4/4/1'])
    (dest/'patch.obj').write_text(obj+'\n')
    material = sapien.render.RenderMaterial(base_color=[1,1,1,1], roughness=1, metallic=0)
    material.set_base_color_texture(sapien.render.RenderTexture2D(str(dest/'checker.png')))
    builder = task.scene.create_actor_builder()
    builder.add_visual_from_file(str(dest/'patch.obj'), material=material)
    builder.set_initial_pose(sapien.Pose(center))
    return builder.build_static(name='wam_visual_patch')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='place_container_plate')
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--center', type=float, nargs=3, default=[0., -.24, .741])
    ap.add_argument('--size', type=float, default=.13)
    ap.add_argument('--out', default=str(B.OUT/'scene_smoke'))
    args = ap.parse_args()
    from PIL import Image
    dest = B.Path(args.out).resolve(); dest.mkdir(parents=True, exist_ok=True)
    assert not (dest/'summary.json').exists(), 'preserve completed captures'
    task, config, embodiment = make_task(args.task, args.seed, dest)
    try:
        clean = task.get_obs(); repeat = task.get_obs()
        patch = add_patch(task, dest, center=args.center, size=args.size)
        patched = task.get_obs()
        assert np.array_equal(clean['joint_action']['vector'], patched['joint_action']['vector'])
        results = {}; arrays = {'state14': clean['joint_action']['vector']}
        for name in ['head_camera', 'left_camera', 'right_camera']:
            a = clean['observation'][name]['rgb']; b = patched['observation'][name]['rgb']
            c = repeat['observation'][name]['rgb']
            for condition, image in [('clean',a), ('repeat',c), ('checker',b)]:
                Image.fromarray(image).save(dest/(name+'_'+condition+'.png'))
                arrays[name+'_'+condition] = image
            results[name] = {'shape': list(a.shape), 'clean_repeat_MAE_u8': float(abs(a.astype(float)-c).mean()),
                             'checker_difference_MAE_u8': float(abs(a.astype(float)-b).mean())}
        np.savez_compressed(dest/'inputs.npz', **arrays)
        (dest/'summary.json').write_text(json.dumps({'task': args.task, 'seed': args.seed, 'embodiment': embodiment,
            'render': 'official ray tracing32spp,path_depth8,OIDN', 'camera_results': results,
            'patch': {'size_m': args.size, 'center_m': args.center, 'visual_only': True, 'explicit_UV_rectangle': True},
            'joint_state_unchanged': True, 'policy_evaluated': False}, indent=2)+'\n')
        print('[scene capture complete]', dest, flush=True)
    finally:
        task.close_env()


if __name__ == '__main__': main()
