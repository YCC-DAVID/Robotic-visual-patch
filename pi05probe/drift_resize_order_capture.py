"""Capture native 256 observations at paired initial states; no policy needed."""
import argparse
import pathlib

import drift_eval_digital as D  # LIBERO environment/path setup
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=16)
    ap.add_argument('--patch', default='drift_patch_drift')
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--out', default='drift_resize_order_native')
    args = ap.parse_args()
    import torch
    import yaml
    import scene_patch as sp
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import masks_from_pair, resize_rgb

    pz = np.load(D.OUT / (args.patch + '.npz'), allow_pickle=True)
    cfg = yaml.safe_load((D.PROBE / 'config' / 'scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(pz['patch_m'])] * 2
    world = pz['cell_world']
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    tid = next(i for i in range(suite.n_tasks) if pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path('init_states')) /
                                    'libero_goal' / (D.TASK + '.pruned_init'))))
    clean = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    probe = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    probe.env.set_xml_processor(sp.make_xml_processor(cfg, world, str(D.PROBE / 'config' / 'probe_texture.png')))
    images, wrists, states, m256, m224, ids = [], [], [], [], [], []
    try:
        for ep in range(min(args.episodes, len(inits))):
            clean.seed(args.seed + ep); clean.reset()
            probe.seed(args.seed + ep); probe.reset()
            obs = clean.set_init_state(inits[ep])
            for _ in range(D.NWAIT):
                obs, _, _, _ = clean.step(D.DUMMY)
            state = clean.get_sim_state().copy()
            cobs = clean.regenerate_obs_from_state(state)
            pobs = probe.regenerate_obs_from_state(state)
            c = np.ascontiguousarray(cobs['agentview_image'][::-1, ::-1])
            p = np.ascontiguousarray(pobs['agentview_image'][::-1, ::-1])
            w = np.ascontiguousarray(cobs['robot0_eye_in_hand_image'][::-1, ::-1])
            # Check fixture/camera state outside the patch bounding box.
            a, b = masks_from_pair(c, p)
            st = np.concatenate([cobs['robot0_eef_pos'], D.quat2axisangle(cobs['robot0_eef_quat']),
                                 cobs['robot0_gripper_qpos']])
            images.append(c); wrists.append(resize_rgb(w, 224)); states.append(st)
            m256.append(a); m224.append(b); ids.append(ep)
            print('[capture] ep={} native_px={} resized_px={}'.format(ep, int(a.sum()), int(b.sum())), flush=True)
    finally:
        clean.close(); probe.close()
    dest = D.OUT / (args.out + '.npz')
    np.savez_compressed(dest, clean_img256=np.array(images), clean_wrist224=np.array(wrists),
                        clean_state8=np.array(states, np.float32), masks256=np.array(m256), masks224=np.array(m224),
                        ep_id=np.array(ids), prompt=str(task.language), seed=args.seed,
                        sampling='one observation after 10 settling steps per initial state',
                        patch_m=pz['patch_m'], cell_world=world)
    print('[written]', dest, flush=True)


if __name__ == '__main__':
    main()
