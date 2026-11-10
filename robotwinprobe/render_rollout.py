"""Paired closed loops with frozen world texture, forward perspective, or true RT."""
import argparse,hashlib,json,random,time
import numpy as np
import bootstrap as B
B.simulator_paths()
from scene_smoke import make_task
from render_geometry import Patch,CAMERAS,WIRE
from render_proxy import grid_for


def digital_images(clean, geometry, texture, center, size):
    import torch
    import torch.nn.functional as F
    from openpi_client.image_tools import resize_with_pad
    q=torch.from_numpy(texture.copy()).permute(2,0,1)[None].float()/255
    result={}
    for name,wire in zip(CAMERAS,WIRE):
        geo=geometry[name]
        g=grid_for(geo['intrinsic'],geo['extrinsic'],center,size,224,224,.5)
        t=F.grid_sample(q,torch.from_numpy(g)[None],mode='bilinear',padding_mode='border',align_corners=False)
        m=torch.from_numpy(geo['mask'])[None,None].float()
        m=F.pad(F.interpolate(m,size=(168,224),mode='nearest'),(0,0,28,28))
        c=torch.from_numpy(resize_with_pad(clean['observation'][name]['rgb'],224,224).copy()).permute(2,0,1)[None].float()/255
        result[wire]=((c*(1-m)+t*m)[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--texture',required=True)
    ap.add_argument('--condition',choices=['digital','render'],required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--port',type=int,default=8180)
    ap.add_argument('--seeds',default=','.join(map(str,range(10200,10210))))
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    from PIL import Image
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import torch
    torch.set_num_threads(4)
    texture=np.array(Image.open(args.texture).convert('RGB'))
    dest=B.Path(args.out).resolve();dest.mkdir(parents=True,exist_ok=True)
    protocol={k:v for k,v in vars(args).items() if k!='resume'}|{'center':[0,0,.741],'size_m':.20,
        'replan':32,'same_noise_rule_as_clean':True,'shared_texture_all_views':True,
        'texture_sha256':hashlib.sha256(B.Path(args.texture).read_bytes()).hexdigest()}
    episodes=[]
    if (dest/'results.json').exists():
        assert args.resume, 'preserve existing results; explicit --resume required'
        assert json.loads((dest/'protocol.json').read_text())==protocol, 'resume protocol differs'
        previous=json.loads((dest/'results.json').read_text())
        if previous['complete']:return
        episodes=previous['episodes']
    else:(dest/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    client=WebsocketClientPolicy('127.0.0.1',args.port)
    started=time.monotonic();seeds=list(map(int,args.seeds.split(',')))
    assert [e['seed'] for e in episodes]==seeds[:len(episodes)]
    try:
        for seed in seeds:
            if seed in {e['seed'] for e in episodes}:continue
            random.seed(seed);task,_,_=make_task('place_container_plate',seed,dest)
            try:
                prompt='Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
                task.set_instruction(prompt);states=[];actions_all=[];noise_seeds=[];rgb={n:[] for n in WIRE}
                initial=task.get_obs()
                if args.condition=='render':
                    patch=Patch(task,dest);patch.texture(texture)
                while task.take_action_cnt<task.step_lim and not task.eval_success:
                    obs=task.get_obs()
                    if args.condition=='digital':
                        patch=Patch(task,dest);patch.texture(texture);task.get_obs()
                        geometry=patch.geometry(task);task.scene.remove_actor(patch.actor)
                        images=digital_images(obs,geometry,texture,patch.center,patch.size)
                    else:images={w:obs['observation'][n]['rgb'] for n,w in zip(CAMERAS,WIRE)}
                    state=obs['joint_action']['vector'].astype(np.float32)
                    ns=20261006+seed*1000+len(states)
                    noise=np.random.RandomState(ns).standard_normal((32,32)).astype(np.float32)
                    actions=np.asarray(client.infer({'state':state,'prompt':prompt,'images':images,'noise':noise})['actions'])
                    assert actions.shape==(32,14) and np.isfinite(actions).all()
                    states.append(state);actions_all.append(actions);noise_seeds.append(ns)
                    for n in WIRE:rgb[n].append(images[n].copy())
                    for a in actions:
                        task.take_action(a,action_type='qpos')
                        if task.eval_success or task.take_action_cnt>=task.step_lim:break
                np.savez_compressed(dest/f'seed{seed}.npz',states14=states,actions=actions_all,noise_seeds=noise_seeds,
                    **{n:np.array(v) for n,v in rgb.items()},**{'initial_clean_'+n:initial['observation'][n]['rgb'] for n in CAMERAS})
                episodes.append({'seed':seed,'success':bool(task.eval_success),'steps':task.take_action_cnt,
                                 'prompt':prompt,'queries':len(states)})
                (dest/'results.json').write_text(json.dumps({'complete':len(episodes)==len(seeds),'episodes':episodes,
                    'task_successes':sum(e['success'] for e in episodes),'elapsed_s':time.monotonic()-started},indent=2)+'\n')
                print('[episode]',episodes[-1],flush=True)
            finally:task.close_env()
    finally:client._ws.close()


if __name__=='__main__':main()
