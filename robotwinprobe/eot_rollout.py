"""Real-render robustness episodes with frozen texture and matched whole-scene light."""
import argparse, hashlib, json, random, time
import numpy as np
import bootstrap as B
B.simulator_paths()
from scene_smoke import make_task
from render_geometry import Patch, CAMERAS, WIRE
from eot_conditions import Lighting, evaluation_conditions


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--texture');ap.add_argument('--condition',required=True,choices=list(evaluation_conditions()))
    ap.add_argument('--port',type=int,default=8186)
    ap.add_argument('--seeds',default='10300,10301,10302,10303,10304')
    args=ap.parse_args()
    from PIL import Image
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    dest=B.Path(args.out).resolve();dest.mkdir(parents=True,exist_ok=True)
    condition=evaluation_conditions()[args.condition]
    texture=np.array(Image.open(args.texture).convert('RGB')) if args.texture else None
    protocol=vars(args)|{'transform':condition,'size_m':.20,'replan':32,
        'noise_rule':'20261006 + seed * 1000 + query',
        'texture_sha256':hashlib.sha256(B.Path(args.texture).read_bytes()).hexdigest() if args.texture else None}
    previous=[]
    if (dest/'protocol.json').exists():
        assert json.loads((dest/'protocol.json').read_text())==protocol
        if (dest/'results.json').exists():
            saved=json.loads((dest/'results.json').read_text())
            if saved['complete']:return
            previous=saved['episodes']
    else:(dest/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    seeds=list(map(int,args.seeds.split(',')));episodes=previous
    assert [r['seed'] for r in episodes]==seeds[:len(episodes)]
    client=WebsocketClientPolicy('127.0.0.1',args.port);started=time.monotonic()
    try:
        for seed in seeds[len(episodes):]:
            random.seed(seed);task,_,_=make_task('place_container_plate',seed,dest)
            try:
                Lighting(task).apply(condition['light_gain'])
                prompt='Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
                task.set_instruction(prompt);initial=task.get_obs()
                states=[];actions_all=[];noise_seeds=[];rgb={n:[] for n in WIRE}
                if texture is not None:
                    patch=Patch(task,dest,center=condition['center'],size=.20);patch.texture(texture)
                while task.take_action_cnt<task.step_lim and not task.eval_success:
                    obs=task.get_obs();state=obs['joint_action']['vector'].astype(np.float32)
                    images={w:obs['observation'][n]['rgb'] for n,w in zip(CAMERAS,WIRE)}
                    ns=20261006+seed*1000+len(states)
                    noise=np.random.RandomState(ns).standard_normal((32,32)).astype(np.float32)
                    actions=np.asarray(client.infer({'state':state,'prompt':prompt,'images':images,'noise':noise})['actions'])
                    assert actions.shape==(32,14) and np.isfinite(actions).all()
                    states.append(state);actions_all.append(actions);noise_seeds.append(ns)
                    for n in WIRE:rgb[n].append(images[n].copy())
                    for action in actions:
                        task.take_action(action,action_type='qpos')
                        if task.eval_success or task.take_action_cnt>=task.step_lim:break
                np.savez_compressed(dest/f'seed{seed}.npz',states14=states,actions=actions_all,noise_seeds=noise_seeds,
                    **{n:np.array(v) for n,v in rgb.items()},
                    **{'initial_clean_'+n:initial['observation'][n]['rgb'] for n in CAMERAS})
                episodes.append({'seed':seed,'success':bool(task.eval_success),'steps':task.take_action_cnt,
                                 'queries':len(states),'prompt':prompt})
                (dest/'results.json').write_text(json.dumps({'complete':len(episodes)==len(seeds),
                    'episodes':episodes,'task_successes':sum(e['success'] for e in episodes),
                    'elapsed_s_this_invocation':time.monotonic()-started},indent=2)+'\n')
                print('[episode]',episodes[-1],flush=True)
            finally:task.close_env()
    finally:client._ws.close()


if __name__=='__main__':main()
