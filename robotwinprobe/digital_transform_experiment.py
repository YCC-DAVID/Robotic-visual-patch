"""Capture, measure and evaluate sensitivity of a frozen digital patch.

Simulator stages run through run_sim.sh; probe stage uses openpi-server Python.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import bootstrap as B
from digital_transform_common import (ROOT, TRANSFER, METHODS, CONDITIONS, dump, sha,
    textures, texture_paths, condition, center, geometry_at, composite, clean_images, prepare)


def capture(dest):
    B.simulator_paths()
    import torch
    from scene_smoke import make_task
    from render_geometry import CAMERAS, WIRE
    from render_rollout import digital_images
    torch.set_num_threads(4)
    if (dest/'capture_audit.json').exists():
        assert (dest/'inputs.npz').exists()
        return
    pats = textures(); rows = []; all_images = []; all_masks = []; checker_images = []
    from render_geometry import patterns
    for seed in [10110, 10111]:
        with np.load(TRANSFER/f'clean/seed{seed}.npz') as z:
            ref = {k:z[k].copy() for k in z.files}
        assets = dest/f'capture_assets_{seed}'; assets.mkdir(exist_ok=True)
        random.seed(seed); task, _, _ = make_task('place_container_plate', seed, assets)
        try:
            prompt = 'Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
            task.set_instruction(prompt)
            for query, actions in enumerate(ref['actions']):
                obs = task.get_obs(); state = obs['joint_action']['vector'].astype(np.float32)
                assert np.array_equal(state, ref['states14'][query]), (seed, query, 'state')
                for n in CAMERAS:
                    assert np.array_equal(obs['observation'][n]['rgb'], ref[n][query]), (seed,query,n)
                by_condition = []; masks_condition = []; geos = {}
                for c in CONDITIONS:
                    key = (tuple(c['xy_cm']), c['yaw_deg'])
                    if key not in geos:
                        geos[key] = geometry_at(task, assets, c, pats['full'])
                    geo = geos[key]
                    images = {'clean':clean_images(obs, c['gain'])}
                    for method in ['full', 'random']:
                        images[method], masks = composite(obs, geo, pats[method], c, True)
                        if c['name']=='nominal':
                            old = digital_images(obs, geo, pats[method], center(c), .20)
                            assert all(np.array_equal(images[method][w], old[w]) for w in WIRE)
                    by_condition.append([[images[m][w] for w in WIRE] for m in METHODS])
                    masks_condition.append([masks[w] for w in WIRE])
                    if seed==10110 and query==0:
                        checker = composite(obs, geo, patterns()['checker'], c)
                        checker_images.append([checker[w] for w in WIRE])
                restored = task.get_obs()
                for n in CAMERAS:
                    assert np.array_equal(restored['observation'][n]['rgb'], obs['observation'][n]['rgb'])
                rows.append(dict(seed=seed, query=query, state14=state.tolist(), prompt=prompt))
                all_images.append(by_condition); all_masks.append(masks_condition)
                print('[capture]', seed, query, flush=True)
                for action in actions:
                    task.take_action(action, action_type='qpos')
                    if task.eval_success or task.take_action_cnt>=task.step_lim: break
        finally:
            task.close_env()
    assert len(rows)==9
    np.savez_compressed(dest/'inputs.npz', images=np.asarray(all_images), masks=np.asarray(all_masks),
                        checker=np.asarray(checker_images), methods=METHODS,
                        conditions=[c['name'] for c in CONDITIONS], wires=WIRE)
    dump(dest/'frames.json', rows)
    dump(dest/'capture_audit.json', dict(complete=True, frames=len(rows),
        original_clean_RGB_state_exact=True, nominal_digital_compositor_exact=True,
        patch_removed_restores_clean_exact=True, masks_fresh_at_transformed_world_pose=True,
        input_sha256=sha(dest/'inputs.npz')))


def probe(dest):
    B.model_paths()
    import torch
    import render_model as M
    if (dest/'probe.json').exists(): return
    rows = json.loads((dest/'frames.json').read_text())
    with np.load(dest/'inputs.npz') as z:
        images = z['images']; wires = z['wires'].tolist()
    policy = M.load(); model = policy._model; device = policy._pytorch_device
    seeds = [20261021, 20261022, 20261023]
    noises = np.stack([np.random.RandomState(s).standard_normal((32,32)).astype(np.float32) for s in seeds])
    velocity = np.zeros((len(rows),len(CONDITIONS),len(METHODS),3,32,32),np.float32)
    clean_cache = {}
    with torch.inference_mode():
        for i, f in enumerate(rows):
            for ci, c in enumerate(CONDITIONS):
                for mi, method in enumerate(METHODS):
                    cachekey = (i, c['gain'])
                    if method=='clean' and cachekey in clean_cache:
                        velocity[i,ci,mi] = clean_cache[cachekey]; continue
                    raw = dict(zip(wires,images[i,ci,mi]))
                    obs = M.observation(policy,np.asarray(f['state14'],np.float32),f['prompt'],raw)
                    state, pad, kv = M.prefix(model,obs)
                    for ni, noise in enumerate(noises):
                        v = model.denoise_step(state,pad,kv,torch.tensor(noise,device=device)[None],
                                              torch.ones(1,device=device))
                        velocity[i,ci,mi,ni] = v.float().cpu().numpy()[0]
                    if method=='clean': clean_cache[cachekey] = velocity[i,ci,mi].copy()
            print('[probe]',f['seed'],f['query'],flush=True)
    np.savez_compressed(dest/'velocities.npz',velocities=velocity,noises=noises,
                        conditions=[c['name'] for c in CONDITIONS],methods=METHODS)
    metrics = []
    for ci,c in enumerate(CONDITIONS):
        for mi, method in enumerate(METHODS[1:],1):
            values = np.square(velocity[:,ci,mi]-velocity[:,ci,0]).sum(axis=(-2,-1)).mean(1)
            nominal = np.square(velocity[:,0,mi]-velocity[:,0,0]).sum(axis=(-2,-1)).mean(1)
            metrics.append(dict(condition=c['name'],method=method,L0_mean=float(values.mean()),
                L0_median=float(np.median(values)),relative_to_nominal=float(values.mean()/nominal.mean()),
                per_state=values.tolist()))
    dump(dest/'probe.json',dict(complete=True,states=9,independent_trajectories=2,noises=seeds,
         L0_definition='sum squared t=1 velocity difference over all32x32 entries; mean over noises then states',
         clean_is_same_brightness=True,rows=metrics))


def evaluate(dest, method, cname, port, seeds):
    B.simulator_paths()
    import torch
    from scene_smoke import make_task
    from render_geometry import CAMERAS, WIRE
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    torch.set_num_threads(4)
    c = condition(cname); out = dest/'eval'/cname/method; out.mkdir(parents=True,exist_ok=True)
    assets = out/'assets'; assets.mkdir(exist_ok=True)
    protocol = dict(condition=c, method=method, seeds=seeds,
                    texture_sha256=None if method=='clean' else sha(texture_paths()[method]))
    if (out/'protocol.json').exists(): assert json.loads((out/'protocol.json').read_text())==protocol
    else: dump(out/'protocol.json', protocol)
    pats = textures(); client = WebsocketClientPolicy('127.0.0.1',port)
    try:
        for seed in seeds:
            stem = out/f'seed{seed}'
            if stem.with_suffix('.json').exists():
                assert stem.with_suffix('.npz').exists(); continue
            random.seed(seed); task, _, _ = make_task('place_container_plate',seed,assets)
            try:
                prompt = 'Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
                task.set_instruction(prompt); initial = task.get_obs()
                with np.load(TRANSFER/f'clean/seed{seed}.npz') as ref:
                    assert np.array_equal(initial['joint_action']['vector'].astype(np.float32),ref['states14'][0])
                    for n in CAMERAS: assert np.array_equal(initial['observation'][n]['rgb'],ref[n][0])
                states=[]; actions_all=[]; noises=[]; rgbs={w:[] for w in WIRE}; pixels=[]
                while task.take_action_cnt<task.step_lim and not task.eval_success:
                    obs=task.get_obs(); state=obs['joint_action']['vector'].astype(np.float32)
                    if method=='clean':
                        images=clean_images(obs,c['gain']); pixels.append([0,0,0])
                    else:
                        geo=geometry_at(task,assets,c,pats[method])
                        images,masks=composite(obs,geo,pats[method],c,True)
                        pixels.append([int(masks[w].sum()) for w in WIRE])
                    ns=20261006+seed*1000+len(states)
                    noise=np.random.RandomState(ns).standard_normal((32,32)).astype(np.float32)
                    actions=np.asarray(client.infer(dict(state=state,prompt=prompt,images=images,noise=noise))['actions'])
                    assert actions.shape==(32,14) and np.isfinite(actions).all()
                    states.append(state); actions_all.append(actions); noises.append(ns)
                    for w in WIRE: rgbs[w].append(images[w].copy())
                    for action in actions:
                        task.take_action(action,action_type='qpos')
                        if task.eval_success or task.take_action_cnt>=task.step_lim:break
                tmp=stem.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,states14=states,actions=actions_all,noise_seeds=noises,mask_pixels=pixels,
                    **{w:np.array(v) for w,v in rgbs.items()},
                    **{'initial_clean_'+n:initial['observation'][n]['rgb'] for n in CAMERAS})
                tmp.replace(stem.with_suffix('.npz'))
                row=dict(seed=seed,condition=cname,method=method,success=bool(task.eval_success),
                    steps=task.take_action_cnt,queries=len(states),prompt=prompt,
                    texture_sha256=protocol['texture_sha256'],initial_clean_and_state_equal=True,
                    noise_rule_verified=True,mask_pixels_by_query=pixels)
                dump(stem.with_suffix('.json'),row)
                print('[episode]',cname,method,seed,row['success'],flush=True)
            finally:task.close_env()
        dump(out/'complete.json',dict(complete=True,episodes=len(seeds)))
    finally:client._ws.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True)
    ap.add_argument('--stage',choices=['capture','probe','eval'],required=True)
    ap.add_argument('--method',choices=METHODS); ap.add_argument('--condition')
    ap.add_argument('--port',type=int,default=8265)
    ap.add_argument('--seeds',default=','.join(map(str,range(10200,10210))))
    args=ap.parse_args(); dest=Path(args.root).resolve(); prepare(dest)
    if args.stage=='capture':capture(dest)
    elif args.stage=='probe':probe(dest)
    else:evaluate(dest,args.method,args.condition,args.port,list(map(int,args.seeds.split(','))))


if __name__=='__main__':main()
