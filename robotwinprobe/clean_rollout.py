"""Small clean-only integration check; not an official leaderboard evaluation."""
import argparse
import json
import random
import time
import numpy as np
import bootstrap as B
B.simulator_paths()
from scene_smoke import make_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='place_container_plate')
    ap.add_argument('--seeds', default='10000,10001,10002,10003,10004')
    ap.add_argument('--port', type=int, default=8180)
    ap.add_argument('--replan', type=int, default=32)
    ap.add_argument('--out', default=str(B.OUT/'clean_pilot'))
    args = ap.parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from PIL import Image
    dest = B.Path(args.out).resolve(); dest.mkdir(parents=True, exist_ok=True)
    assert not (dest/'results.json').exists(), 'preserve prior runs'
    assert 1 <= args.replan <= 32
    client = WebsocketClientPolicy('127.0.0.1', args.port)
    results = []; start = time.monotonic()
    protocol = {'task': args.task, 'seeds': [int(x) for x in args.seeds.split(',')],
        'condition': 'clean', 'task_config': 'demo_clean', 'expert_prefilter': False,
        'replan': args.replan, 'model_horizon': 32, 'native_images': True,
        'noise_seed_rule': '20261006+seed*1000+query', 'purpose': 'integration pilot, not benchmark ASR',
        'prompt_rule': 'place the actual bowl/cup on the plate; no sampled language templates',
        'model_checkpoint': 'motus-robotics/pi0.5_robotwin2@effc6e1a99bd4d0edf9447e8ba313def67649cfe'}
    (dest/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    try:
        for seed in protocol['seeds']:
            random.seed(seed)
            task, config, embodiment = make_task(args.task, seed, dest)
            try:
                prompt = 'Place the '+('bowl' if task.actor_name == '002_bowl' else 'cup')+' on the plate.'
                task.set_instruction(prompt)
                states=[]; chunks=[]; noises=[]; images={n:[] for n in ['head_camera','left_camera','right_camera']}
                initial_actors = [{'name': actor.get_name(), 'p': actor.get_pose().p.tolist(),
                                   'q': actor.get_pose().q.tolist()} for actor in task.scene.get_all_actors()]
                queries = 0
                while task.take_action_cnt < task.step_lim and not task.eval_success:
                    obs = task.get_obs()
                    state = obs['joint_action']['vector'].astype(np.float32)
                    noise_seed = 20261006+seed*1000+queries
                    noise = np.random.RandomState(noise_seed).normal(size=(32,32)).astype(np.float32)
                    request = {'state': state, 'prompt': prompt, 'noise': noise, 'images': {}}
                    for source, target in [('head_camera','cam_high'), ('left_camera','cam_left_wrist'), ('right_camera','cam_right_wrist')]:
                        rgb = obs['observation'][source]['rgb']
                        assert rgb.dtype == np.uint8 and rgb.shape[-1] == 3
                        request['images'][target] = rgb
                        images[source].append(rgb.copy())
                    actions = np.asarray(client.infer(request)['actions'])
                    assert actions.shape == (32,14) and np.isfinite(actions).all()
                    states.append(state); chunks.append(actions); noises.append(noise_seed)
                    queries += 1
                    for action in actions[:args.replan]:
                        task.take_action(action, action_type='qpos')
                        if task.eval_success or task.take_action_cnt >= task.step_lim: break
                record = 'seed'+str(seed)+'.npz'
                final = task.get_obs()
                for camera in images:
                    Image.fromarray(images[camera][0]).save(dest/f'seed{seed}_{camera}_initial.png')
                    Image.fromarray(final['observation'][camera]['rgb']).save(dest/f'seed{seed}_{camera}_final.png')
                np.savez_compressed(dest/record, states14=np.array(states), actions=np.array(chunks),
                    noise_seeds=np.array(noises), final_state14=final['joint_action']['vector'],
                    **{k:np.array(v) for k,v in images.items()},
                    **{'final_'+k: final['observation'][k]['rgb'] for k in images})
                results.append({'seed': seed, 'success': bool(task.eval_success), 'final_predicate': bool(task.check_success()),
                    'steps': task.take_action_cnt, 'step_limit': task.step_lim, 'queries': queries,
                    'prompt': prompt, 'actor_name': task.actor_name, 'container_id': int(task.container_id),
                    'initial_actors': initial_actors, 'record': record})
                (dest/'results.json').write_text(json.dumps({'complete': len(results)==len(protocol['seeds']),
                    'task_successes': sum(r['success'] for r in results), 'episodes': results,
                    'elapsed_s': time.monotonic()-start}, indent=2)+'\n')
                print('[episode]', seed, results[-1]['success'], 'steps', task.take_action_cnt, flush=True)
            finally:
                task.close_env()
    finally:
        client._ws.close()


if __name__ == '__main__': main()
