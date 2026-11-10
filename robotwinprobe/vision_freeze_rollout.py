"""Actual-render paired episodes, with optional same-state diagnostic captures."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import traceback
import numpy as np
import bootstrap as B
import checkpoint_study_common as S
B.simulator_paths()


def capture(task,root,seed,query,state,prompt,clean,manifest,patch_cfg):
    from PIL import Image
    from render_geometry import Patch
    dest=root/'diagnostic_frames'/f'seed{seed}_q{query:03d}';dest.mkdir(parents=True,exist_ok=True)
    meta=dict(seed=seed,query=query,prompt=prompt,reference_model=f'F_s{S.prepare_protocol(root)["model_steps"]}',conditions=[])
    np.savez_compressed(dest/'clean.npz',state14=state,**clean)
    patch=Patch(task,dest,center=patch_cfg['center_m'],size=patch_cfg['size_m'])
    try:
        for row in manifest:
            rgb=np.asarray(Image.open(row['path']).convert('RGB'))
            patch.texture(rgb);obs=task.get_obs()
            np.savez_compressed(dest/(row['id']+'.npz'),state14=state,
                **{w:obs['observation'][n]['rgb'] for n,w in zip(S.CAMERAS,S.WIRE)})
            meta['conditions'].append(dict(id=row['id'],texture_sha256=row['sha256']))
    finally:task.scene.remove_actor(patch.actor)
    obs=task.get_obs()
    assert np.array_equal(state,obs['joint_action']['vector'].astype(np.float32)),'Capture changed physics'
    np.savez_compressed(dest/'clean_repeat.npz',state14=state,
        **{w:obs['observation'][n]['rgb'] for n,w in zip(S.CAMERAS,S.WIRE)})
    S.dump(dest/'manifest.json',dict(complete=True,**meta))


def main(args):
    from PIL import Image
    from scene_smoke import make_task
    from render_geometry import Patch
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    p=S.prepare_protocol(args.root);pc=p['patch']
    dest=args.root/('pilot' if args.pilot else 'closedloop')/args.model/args.condition
    dest.mkdir(parents=True,exist_ok=True)
    textures=json.loads((args.root/'texture_manifest.json').read_text()) if (args.root/'texture_manifest.json').exists() else []
    texture=None;digest=None
    if args.condition!='clean':
        row=next(r for r in textures if r['id']==args.condition)
        assert S.sha(row['path'])==row['sha256'];digest=row['sha256']
        texture=np.asarray(Image.open(row['path']).convert('RGB'))
    seeds=list(range(23000,23010)) if args.pilot else p['evaluation']['seeds']
    if args.seeds:seeds=list(map(int,args.seeds.split(',')))
    protocol=dict(model=args.model,condition=args.condition,texture_sha256=digest,seeds=seeds,
        checkpoint_sha256=S.sha(args.root/'models'/args.model/'model.safetensors'),
        size_m=pc['size_m'],center_m=pc['center_m'],replan=32,
        source_sha256=S.sha(Path(__file__)),noise_rule='20261006+seed*1000+query',actual_render=True,
        smoke_max_actions=args.max_actions)
    if args.max_actions is not None:assert args.model=='smoke' and args.pilot
    if (dest/'protocol.json').exists():assert json.loads((dest/'protocol.json').read_text())==protocol
    else:S.dump(dest/'protocol.json',protocol)
    episodes=[]
    client=None
    for seed in seeds:
        result=dest/f'seed{seed}.json'
        if result.exists():
            row=json.loads(result.read_text());episodes.append(row)
            if row.get('error'):raise RuntimeError(f'Unresolved episode error: {result}')
            continue
        random.seed(seed);task=None;start=time.monotonic()
        try:
            task,_,_=make_task(p['task'],seed,dest)
            if args.max_actions is not None:task.step_lim=min(task.step_lim,args.max_actions)
            prompt='Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
            task.set_instruction(prompt);initial=task.get_obs()
            h=hashlib.sha256(initial['joint_action']['vector'].astype(np.float32).tobytes())
            for n in S.CAMERAS:h.update(initial['observation'][n]['rgb'].tobytes())
            initial_hash=h.hexdigest()
            clean_file=args.root/('pilot' if args.pilot else 'closedloop')/args.model/'clean'/f'seed{seed}.json'
            if args.condition!='clean' and clean_file.exists():
                assert json.loads(clean_file.read_text())['initial_clean_sha256']==initial_hash,'Paired initial scene differs'
            if texture is not None:
                patch=Patch(task,dest,center=pc['center_m'],size=pc['size_m']);patch.texture(texture)
            # Scene/CUDA initialization can take minutes and stall Python's
            # heartbeat thread. Connect only once the scene is ready, and use
            # a fresh connection for each independent episode.
            client=WebsocketClientPolicy('127.0.0.1',args.port)
            metadata=client.get_server_metadata()
            assert metadata['checkpoint_sha256']==protocol['checkpoint_sha256'],'Wrong checkpoint server'
            states=[];actions_all=[];noise_seeds=[];rgb={n:[] for n in S.WIRE}
            while task.take_action_cnt<task.step_lim and not task.eval_success:
                obs=task.get_obs();state=obs['joint_action']['vector'].astype(np.float32)
                images={w:obs['observation'][n]['rgb'] for n,w in zip(S.CAMERAS,S.WIRE)}
                query=len(states);ns=20261006+seed*1000+query
                noise=np.random.RandomState(ns).standard_normal((32,32)).astype(np.float32)
                actions=np.asarray(client.infer(dict(state=state,prompt=prompt,images=images,noise=noise))['actions'])
                assert actions.shape==(32,14) and np.isfinite(actions).all()
                if args.capture and seed in seeds[:12] and query in (0,3,6,9):
                    assert args.condition=='clean' and args.model==f'F_s{S.prepare_protocol(args.root)["model_steps"]}' and len(textures)>=7
                    capture(task,args.root,seed,query,state,prompt,images,textures,pc)
                states.append(state.copy());actions_all.append(actions);noise_seeds.append(ns)
                for n in S.WIRE:rgb[n].append(images[n].copy())
                for a in actions:
                    task.take_action(a,action_type='qpos')
                    if task.eval_success or task.take_action_cnt>=task.step_lim:break
            np.savez_compressed(dest/f'seed{seed}.npz',states14=states,actions=actions_all,noise_seeds=noise_seeds,
                **{n:np.array(v) for n,v in rgb.items()})
            row=dict(seed=seed,success=bool(task.eval_success),steps=task.take_action_cnt,queries=len(states),
                prompt=prompt,initial_clean_sha256=initial_hash,elapsed_s=time.monotonic()-start)
            S.dump(result,row);episodes.append(row)
            S.dump(dest/'results.json',dict(complete=len(episodes)==len(seeds),episodes=episodes,
                target=len(seeds),task_successes=sum(e['success'] for e in episodes)))
            print('[episode]',args.model,args.condition,row,flush=True)
        except Exception as exc:
            S.dump(result,dict(seed=seed,error=repr(exc),traceback=traceback.format_exc()))
            raise
        finally:
            if client is not None:
                client._ws.close()
                client=None
            if task is not None:task.close_env()
    S.dump(dest/'results.json',dict(complete=True,episodes=episodes,target=len(seeds),
        task_successes=sum(e['success'] for e in episodes)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=S.DEFAULT)
    ap.add_argument('--model',required=True);ap.add_argument('--condition',required=True)
    ap.add_argument('--port',type=int,required=True);ap.add_argument('--pilot',action='store_true')
    ap.add_argument('--capture',action='store_true');ap.add_argument('--seeds')
    ap.add_argument('--max-actions',type=int)
    main(ap.parse_args())
