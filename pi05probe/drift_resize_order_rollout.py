"""Small paired closed-loop pilot; native paste order, unchanged colors.

Requires drift_resize_order_probe.py --serve PORT. Sampling noise is indexed
by episode and replan index, so every condition has the same noise sequence.
Stops at first task success (the original evaluator's ever-success criterion).
"""
import argparse
import collections
import json
import pathlib
import time

import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--port', type=int, default=8135)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--out', default='drift_resize_order_rollout')
    ap.add_argument('--conds', default='clean,rand_post224,rand_pre256_native,drift_post224,drift_pre256_native,band32_post224,band32_pre256_native')
    args = ap.parse_args()
    import torch
    import yaml
    import scene_patch as sp
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    from drift_resize_order import compose_variants, masks_from_pair, resize_rgb

    root_patch = np.load(D.OUT / 'drift_patch_drift.npz', allow_pickle=True)
    cfg = yaml.safe_load((D.PROBE / 'config' / 'scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(root_patch['patch_m'])] * 2
    patterns = {}
    for tag in ['drift', 'band32']:
        pz = np.load(D.OUT / ('drift_patch_' + tag + '.npz'), allow_pickle=True)
        assert float(pz['patch_m']) == float(root_patch['patch_m'])
        assert np.array_equal(pz['cell_world'], root_patch['cell_world'])
        patterns[tag] = (pz['P'].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    patterns['rand'] = np.random.RandomState(D.SEED_RAND).randint(0, 256, (224, 224, 3), dtype=np.uint8)
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    tid = next(i for i in range(suite.n_tasks) if pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    task = suite.get_task(tid)
    bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    inits = np.array(torch.load(str(pathlib.Path(get_libero_path('init_states')) / 'libero_goal' /
                                    (D.TASK + '.pruned_init'))))
    conditions = args.conds.split(',')
    valid = {'clean'} | {tag + '_' + mode for tag in patterns for mode in ['post224', 'pre256_native', 'pre256_matched']}
    assert set(conditions) <= valid
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    probe = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    probe.env.set_xml_processor(sp.make_xml_processor(cfg, root_patch['cell_world'],
                                                     str(D.PROBE / 'config' / 'probe_texture.png')))
    results = {name: [] for name in conditions}
    start = time.monotonic()
    metadata = {'episodes': args.episodes, 'seed': args.seed, 'noise_seed_rule': '20260901 + episode*1000 + replan_index',
                'patch_m': float(root_patch['patch_m']), 'wrist': 'clean', 'max_steps': D.MAX_STEPS,
                'task': D.TASK, 'pilot': True, 'note': 'All conditions use identical native rendering and noise sequence; no color transform.'}

    def save():
        summary = {name: {'successes': sum(r['success'] for r in rs), 'n': len(rs),
                          'robot_success_rate': sum(r['success'] for r in rs) / len(rs) if rs else None}
                   for name, rs in results.items()}
        (D.OUT / (args.out + '.json')).write_text(json.dumps({'metadata': metadata, 'summary': summary,
                                                            'per_episode': results}, indent=2))
        lines = ['Native-256 composition-order closed-loop pilot (robot success rate)',
                 'Paired initial states and noise; no color/illumination transform; patch size 0.26m.',
                 'Small pilot, not a 50-episode success-rate estimate.']
        for name, row in summary.items():
            if row['n']:
                lines.append('{}: {}/{} = {:.1%}'.format(name, row['successes'], row['n'], row['robot_success_rate']))
        (D.OUT / (args.out + '.txt')).write_text('\n'.join(lines) + '\n')

    try:
        # Interleave conditions within each episode; checkpoint after every run.
        for ep in range(min(args.episodes, len(inits))):
            for name in conditions:
                env.seed(args.seed + ep); env.reset()
                probe.seed(args.seed + ep); probe.reset()
                obs = env.set_init_state(inits[ep])
                goal = env.env.parsed_problem['goal_state']
                for _ in range(D.NWAIT):
                    obs, _, _, _ = env.step(D.DUMMY)
                queue = collections.deque()
                ok, inference_index, t_ok = False, 0, None
                mask_pixels, infer_ms = [], []
                for t in range(D.MAX_STEPS):
                    if not queue:
                        clean256 = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
                        wrist = resize_rgb(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]), 224)
                        image = resize_rgb(clean256, 224)
                        if name != 'clean':
                            tag, mode = name.split('_', 1)
                            state = env.get_sim_state().copy()
                            cobs = env.regenerate_obs_from_state(state)
                            pobs = probe.regenerate_obs_from_state(state)
                            paired_clean = np.ascontiguousarray(cobs['agentview_image'][::-1, ::-1])
                            paired_probe = np.ascontiguousarray(pobs['agentview_image'][::-1, ::-1])
                            m256, m224 = masks_from_pair(paired_clean, paired_probe)
                            mask_pixels.append(int(m224.sum()))
                            image = compose_variants(clean256, m256, m224, patterns[tag])[mode]
                        st = np.concatenate([obs['robot0_eef_pos'], D.quat2axisangle(obs['robot0_eef_quat']),
                                             obs['robot0_gripper_qpos']])
                        noise = np.random.RandomState(20260901 + ep * 1000 + inference_index).standard_normal((10, 32)).astype(np.float32)
                        out = client.infer({'observation/image': image, 'observation/wrist_image': wrist,
                                            'observation/state': st, 'prompt': str(task.language), 'noise': noise})
                        infer_ms.append(out['policy_timing']['infer_ms'])
                        queue.extend(out['actions'][:D.REPLAN])
                        inference_index += 1
                    obs, _, _, _ = env.step(queue.popleft().tolist())
                    if all(env.env._eval_predicate(s) for s in goal):
                        ok, t_ok = True, t + D.NWAIT
                        break
                results[name].append({'episode': ep, 'success': ok, 'success_step': t_ok,
                                      'inferences': inference_index, 'infer_ms_median': float(np.median(infer_ms)),
                                      'mask_px_median': float(np.median(mask_pixels)) if mask_pixels else None})
                save()
                print('[rollout] ep={} {:24s} success={} step={} calls={} infer_ms={:.1f} elapsed={:.1f}s'.format(
                    ep, name, ok, t_ok, inference_index, np.median(infer_ms), time.monotonic() - start), flush=True)
    finally:
        env.close(); probe.close()
        client._ws.close()
    print((D.OUT / (args.out + '.txt')).read_text(), flush=True)


if __name__ == '__main__':
    main()
