"""Capture full clean-policy query trajectories at the frozen13cm geometry.

Requires an externally preflighted GPU, explicit UUID, and existing policy server.
This script does not allocate GPUs or start/stop a policy server.
"""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import bootstrap as B
from low_asr_retry_common import DEFAULT, SOURCE, TASK, CAMERAS, WIRE, CENTER, SIZE, DATA_SEEDS, prepare
from main_method_common import dump, sha


def capture(root, port, gpu_uuid, seeds):
    B.simulator_paths()
    from main_method_runtime import install_path_normalization
    install_path_normalization(root/'capture_runtime')
    from main_method_rollout import pin_sapien_device, actors, digest, pci, image_delta
    binding = pin_sapien_device(gpu_uuid, root/'capture_runtime')
    from scene_smoke import make_task
    from render_geometry import Patch, camera_objects, patterns
    from stage_joint_common import PhaseTracker
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    prepare(root)
    client = WebsocketClientPolicy('127.0.0.1', port)
    colors = {k: np.full((128, 128, 3), v, np.uint8) for k, v in
              [('black', 0), ('gray64', 64), ('gray', 128), ('gray192', 192), ('white', 255)]}
    colors.update({k: patterns()[k] for k in ['random', 'checker']})
    try:
        for seed in seeds:
            assert seed in DATA_SEEDS
            summary_path = root/'collection'/f'seed{seed}.json'
            if summary_path.exists():
                saved = json.loads(summary_path.read_text())
                assert saved['complete'] and saved['protocol_sha256'] == sha(root/'protocol.json')
                assert all(sha(r['path']) == r['sha256'] for r in saved['records'])
                continue
            assets = root/'assets'/f'seed{seed}'; assets.mkdir(parents=True, exist_ok=True)
            task = None; random.seed(seed); rows = []; actions = []; states = []; noises = []
            try:
                task, config, embodiment = make_task(TASK, seed, assets)
                assert pci(task.scene.render_system.device.pci_string) == pci(binding['nvidia_smi_pci'])
                prompt = 'Place the '+('bowl' if task.actor_name == '002_bowl' else 'cup')+' on the plate.'
                task.set_instruction(prompt)
                tracker = PhaseTracker(task.container.get_pose().p)
                while not task.eval_success and task.take_action_cnt < task.step_lim:
                    query = len(rows); obs = task.get_obs()
                    state = obs['joint_action']['vector'].astype(np.float32).copy()
                    before = actors(task)
                    data = dict(seed=seed, query=query, policy_action_step=task.take_action_cnt,
                                stage=tracker.phase, prompt=prompt, state14=state,
                                center_m=CENTER, size_m=SIZE, yaw_deg=0., light='nominal')
                    clean = {c: obs['observation'][c]['rgb'].copy() for c in CAMERAS}
                    repeat = task.get_obs()
                    diagnostics = dict(clean_repeat={c: image_delta(clean[c], repeat['observation'][c]['rgb']) for c in CAMERAS})
                    for camera in CAMERAS:
                        data[camera+'_clean'] = clean[camera]
                    patch = Patch(task, assets, center=CENTER, size=SIZE)
                    try:
                        for index, (name, texture) in enumerate(colors.items()):
                            patch.texture(texture); patched = task.get_obs(); cameras = camera_objects(task)
                            assert np.array_equal(patched['joint_action']['vector'].astype(np.float32), state)
                            assert [a for a in actors(task) if a['name'] != 'wam_visual_patch'] == before
                            for camera in CAMERAS:
                                data[camera+'_'+name] = patched['observation'][camera]['rgb'].copy()
                                if index == 0:
                                    data[camera+'_mask'] = cameras[camera].get_picture('Segmentation')[..., 1] == patch.actor.per_scene_id
                                    data[camera+'_intrinsic'] = cameras[camera].get_intrinsic_matrix()
                                    data[camera+'_extrinsic'] = cameras[camera].get_extrinsic_matrix()
                    finally:
                        task.scene.remove_actor(patch.actor)
                    restored = task.get_obs()
                    assert np.array_equal(restored['joint_action']['vector'].astype(np.float32), state)
                    assert actors(task) == before
                    diagnostics['restored_clean'] = {c: image_delta(clean[c], restored['observation'][c]['rgb']) for c in CAMERAS}
                    path = root/'bank'/f's{seed}_q{query:03d}.npz'; path.parent.mkdir(exist_ok=True)
                    temporary = path.with_suffix('.tmp.npz'); np.savez_compressed(temporary, **data); temporary.replace(path)
                    row = dict(seed=seed, query=query, step=task.take_action_cnt, stage=tracker.phase,
                               path=str(path), sha256=sha(path), diagnostics=diagnostics,
                               visible_pixels={c: int(data[c+'_mask'].sum()) for c in CAMERAS})
                    rows.append(row)
                    # Feed the restored clean native image, never a calibration render.
                    raw = {w: restored['observation'][c]['rgb'].copy() for c, w in zip(CAMERAS, WIRE)}
                    noise_seed = 20261006+seed*1000+query
                    noise = np.random.RandomState(noise_seed).standard_normal((32, 32)).astype(np.float32)
                    chunk = np.asarray(client.infer(dict(state=state, prompt=prompt, images=raw, noise=noise))['actions'])
                    assert chunk.shape == (32, 14) and np.isfinite(chunk).all()
                    states.append(state); actions.append(chunk); noises.append(noise_seed)
                    for action in chunk:
                        task.take_action(action, action_type='qpos')
                        current = np.asarray(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState(), np.float32)
                        tracker.update(current, task.container.get_pose().p, task.plate.get_pose().p, task.take_action_cnt)
                        if task.eval_success or task.take_action_cnt >= task.step_lim:
                            break
                    dump(root/'capture_progress.json', dict(seed=seed, queries=len(rows), stage=tracker.phase,
                                                          steps=task.take_action_cnt, complete=False))
                    print('[capture]', seed, query, row['stage'], row['visible_pixels'], flush=True)
                assert rows and (task.eval_success or task.take_action_cnt == task.step_lim)
                replay = root/'collection'/f'seed{seed}.npz'; replay.parent.mkdir(exist_ok=True)
                np.savez_compressed(replay, states14=states, actions=actions, noise_seeds=noises)
                dump(summary_path, dict(complete=True, seed=seed, records=rows, prompt=prompt,
                                        clean_success=bool(task.eval_success), steps=task.take_action_cnt,
                                        embodiment=embodiment, no_outcome_filter=True,
                                        replay_sha256=sha(replay), protocol_sha256=sha(root/'protocol.json')))
            finally:
                if task is not None:
                    task.close_env()
        dump(root/'capture_progress.json', dict(complete=all((root/'collection'/f'seed{s}.json').is_file() for s in DATA_SEEDS)))
    finally:
        client._ws.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=DEFAULT)
    ap.add_argument('--port', type=int, required=True); ap.add_argument('--gpu-uuid', required=True)
    ap.add_argument('--seeds', default=','.join(map(str, DATA_SEEDS)))
    args = ap.parse_args()
    capture(args.root.resolve(), args.port, args.gpu_uuid, [int(s) for s in args.seeds.split(',')])
