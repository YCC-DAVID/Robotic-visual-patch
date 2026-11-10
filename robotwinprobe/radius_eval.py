"""Frozen-texture real rollouts on paired continuous test positions."""
import argparse, hashlib, json, random, time
from pathlib import Path
import numpy as np
import bootstrap as B
B.simulator_paths()
from radius_plan import dump
from render_geometry import CAMERAS, WIRE, Patch
from scene_smoke import make_task


def digest(arr):return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--method',required=True,choices=['clean','random','baseline','r01','r03','r05'])
    ap.add_argument('--port',type=int,required=True);ap.add_argument('--count',type=int,required=True)
    ap.add_argument('--max-episodes',type=int,default=0,help='smoke only; never marks full evaluation complete')
    args=ap.parse_args();root=Path(args.root).resolve();dest=root/'eval'/args.method;dest.mkdir(parents=True,exist_ok=True)
    from PIL import Image
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    texture=None;texture_hash=None
    if args.method!='clean':
        path=Path(json.loads((root/'textures.json').read_text())[args.method])
        texture=np.array(Image.open(path).convert('RGB'));texture_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(root/'plan.npz') as p:seeds=p['test_seeds'][:args.count].copy();positions=p['test_xy'][:args.count].copy()
    assert args.count in [50,100]
    protocol={'method':args.method,'texture_sha256':texture_hash,
        'plan_sha256':hashlib.sha256((root/'plan.npz').read_bytes()).hexdigest(),
        'noise_rule':'20261006+seed*1000+query','patch_size_m':.20,'fixed_within_episode':True,
        'light':'nominal','condition':'true SAPIEN RT'}
    if (dest/'protocol.json').exists():assert json.loads((dest/'protocol.json').read_text())==protocol
    else:dump(dest/'protocol.json',protocol)
    assets=dest/'assets';assets.mkdir(exist_ok=True)
    client=WebsocketClientPolicy('127.0.0.1',args.port);completed=0;started=time.monotonic()
    try:
        for si,seed0 in enumerate(seeds):
            seed=int(seed0)
            offsets=np.zeros((1,2)) if args.method=='clean' else np.concatenate([np.zeros((1,2)),positions[si]])
            for pi,xy in enumerate(offsets):
                stem=dest/f'seed{seed}_p{pi:02d}';result=stem.with_suffix('.json')
                if result.exists():
                    saved=json.loads(result.read_text())
                    assert saved['center']==[float(xy[0]),float(xy[1]),.741] and saved['texture_sha256']==texture_hash
                    assert stem.with_suffix('.npz').exists()
                    continue
                random.seed(seed);task,_,_=make_task('place_container_plate',seed,assets)
                try:
                    prompt='Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
                    task.set_instruction(prompt);initial=task.get_obs()
                    initial_rgb={n:initial['observation'][n]['rgb'].copy() for n in CAMERAS}
                    initial_state=initial['joint_action']['vector'].astype(np.float32)
                    center=[float(xy[0]),float(xy[1]),.741]
                    if texture is not None:
                        patch=Patch(task,assets,center=center,size=.20);patch.texture(texture)
                    states=[];chunks=[];noise_seeds=[];rgb={n:[] for n in WIRE}
                    while task.take_action_cnt<task.step_lim and not task.eval_success:
                        obs=task.get_obs();state=obs['joint_action']['vector'].astype(np.float32)
                        images={w:obs['observation'][n]['rgb'] for n,w in zip(CAMERAS,WIRE)}
                        ns=20261006+seed*1000+len(states)
                        noise=np.random.RandomState(ns).standard_normal((32,32)).astype(np.float32)
                        actions=np.asarray(client.infer({'state':state,'prompt':prompt,'images':images,'noise':noise})['actions'])
                        assert actions.shape==(32,14) and np.isfinite(actions).all()
                        states.append(state);chunks.append(actions);noise_seeds.append(ns)
                        for n in WIRE:rgb[n].append(images[n].copy())
                        for action in actions:
                            task.take_action(action,action_type='qpos')
                            if task.eval_success or task.take_action_cnt>=task.step_lim:break
                    out=stem.with_suffix('.npz');tmp=stem.with_suffix('.tmp.npz')
                    np.savez_compressed(tmp,states14=states,actions=chunks,noise_seeds=noise_seeds,
                        **{n:np.array(v) for n,v in rgb.items()},**{'initial_clean_'+n:v for n,v in initial_rgb.items()})
                    tmp.replace(out)
                    row={'seed':seed,'position_index':pi,'center':center,'radius_cm':float(np.linalg.norm(xy)*100),
                        'success':bool(task.eval_success),'steps':task.take_action_cnt,'queries':len(states),
                        'prompt':prompt,'texture_sha256':texture_hash,
                        'initial_rgb_sha256':{n:digest(v) for n,v in initial_rgb.items()},
                        'initial_state_sha256':digest(initial_state),'first_noise_seed':noise_seeds[0]}
                    dump(result,row);completed+=1
                    dump(dest/'progress.json',{'complete':False,'new_episodes':completed,'last':row,
                        'count':args.count,'elapsed_s':time.monotonic()-started})
                    print('[episode]',args.method,seed,pi,row['success'],flush=True)
                finally:task.close_env()
                if args.max_episodes and completed>=args.max_episodes:return
        dump(dest/'progress.json',{'complete':True,'count':args.count,
            'episodes':len(seeds)*(1 if args.method=='clean' else 11),'elapsed_s_this_run':time.monotonic()-started})
    finally:client._ws.close()


if __name__=='__main__':main()
