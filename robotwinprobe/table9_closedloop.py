"""Frozen nine-site 10 cm patch pilot: matched true-render ASR and retention."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import bootstrap as B
from radius_plan import disk, dump
from table9_plan import DEFAULT, sha
from eot_conditions import LIGHTS


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists():
        p = json.loads((root/'protocol.json').read_text())
        for path, h in p['texture_sha256'].items():
            assert sha(path) == h, path
        assert sha(DEFAULT/'protocol.json') == p['training_protocol_sha256']
        assert sha(root/'jobs.json') == p['jobs_sha256']
        return p
    training = json.loads((DEFAULT/'protocol.json').read_text())
    seeds = list(range(10500, 10510))
    rng = np.random.RandomState(202610130)
    offsets = .03*disk(rng, (len(seeds), 3))
    yaws = rng.uniform(-15, 15, (len(seeds), 3))
    # Every seed includes each light; each remains paired with its own clean.
    light_order = [rng.permutation(list(LIGHTS)).tolist() for _ in seeds]
    random_path = root/'random.png'
    Image.fromarray(np.random.RandomState(20261007).randint(0,256,(128,128,3),dtype=np.uint8)).save(random_path)
    paths = {s['id']:str(DEFAULT/'sites'/s['id']/'train/texture_0500.png') for s in training['sites']}
    jobs = []
    for si, seed in enumerate(seeds):
        for light in LIGHTS:
            jobs.append(dict(id=f'clean_{seed}_{light}', method='clean', seed=seed, light=light,
                             condition='clean', center=None, yaw_deg=0, texture=None))
        for site in training['sites']:
            conditions = [dict(condition='nominal', sample=0, light='nominal',
                               center=site['center_m'], yaw_deg=0)]
            for pi in range(3):
                center = (np.array(site['center_m'])+np.r_[offsets[si,pi],0]).tolist()
                conditions.append(dict(condition='position', sample=pi, light='nominal',
                                       center=center, yaw_deg=0))
                conditions.append(dict(condition='joint', sample=pi, light=light_order[si][pi],
                                       center=center, yaw_deg=float(yaws[si,pi])))
            for c in conditions:
                for method in ['trained','random']:
                    jobs.append(dict(id=f"{site['id']}_{seed}_{c['condition']}_{c['sample']}_{method}",
                        method=method, site=site['id'], seed=seed, texture=paths[site['id']] if method=='trained' else str(random_path), **c))
    dump(root/'jobs.json', jobs)
    p = dict(model='pi0.5_robotwin2', source=str(DEFAULT), training_protocol_sha256=sha(DEFAULT/'protocol.json'),
        texture_sha256={path:sha(path) for path in [*paths.values(),str(random_path)]},
        patch_size_m=.10, single_patch_per_scene=True, task='place_container_plate', task_config='demo_clean',
        seeds=seeds, sites=training['sites'], cohort='10 new initial states; pilot, no expert preselection or outcome-based expansion',
        conditions=dict(nominal='own training center, yaw 0, nominal light',
            position='three new area-uniform 3 cm disk offsets per initial state, yaw 0, nominal light',
            joint='same offsets, new yaw in [-15,+15], all three training light settings per initial state'),
        shared_draws='all sites and both textures share initial states, offsets, yaws, lighting and action noise',
        noise_rule='20261013 + seed*1000 + query', replan=32, renderer='actual SAPIEN RT',
        clean_pairing='one clean per initial state and scene light; state, initial clean RGB, prompt and noise checked',
        positions='world placement, yaw, lighting and texture fixed within each episode',
        endpoint='ASR conditional on same-light clean success; report task SR and paired origin retention separately',
        independent_units='initial states; three placements within each state are not independent scenes',
        later_ablation='choose candidates from this pilot; test any pruned texture on a separate initial-state pool',
        jobs_sha256=sha(root/'jobs.json'), episodes=len(jobs))
    dump(root/'protocol.json', p)
    return p


def rollout(root, port, worker, workers, limit):
    B.simulator_paths()
    import sapien
    import torch
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from scene_smoke import make_task
    from render_geometry import Patch, CAMERAS, WIRE, camera_objects
    from eot_conditions import Lighting
    torch.set_num_threads(4)
    p = prepare(root)
    jobs = json.loads((root/'jobs.json').read_text())
    jobs = [j for i,j in enumerate(jobs) if i % workers == worker]
    if limit:
        # Exercise paired clean, trained texture and Random rather than only clean jobs.
        ids = {'clean_10500_nominal', 'site00_10500_nominal_0_trained',
               'site00_10500_nominal_0_random'}
        jobs = [j for j in jobs if j['id'] in ids]
    client = WebsocketClientPolicy('127.0.0.1', port)
    assets = root/'assets'/f'worker{worker}'
    assets.mkdir(parents=True, exist_ok=True)
    done = 0
    try:
        for job in jobs:
            output = root/'episodes'/(job['id']+'.json')
            artifact = output.with_suffix('.npz')
            if output.exists():
                saved = json.loads(output.read_text())
                assert saved['job'] == job and saved['protocol_sha256'] == sha(root/'protocol.json')
                assert artifact.exists()
                continue
            random.seed(job['seed'])
            task, _, _ = make_task('place_container_plate', job['seed'], assets)
            try:
                lighting = Lighting(task)
                lighting.apply(LIGHTS[job['light']])
                prompt = 'Place the '+('bowl' if task.actor_name == '002_bowl' else 'cup')+' on the plate.'
                task.set_instruction(prompt)
                initial = task.get_obs()
                state0 = initial['joint_action']['vector'].astype(np.float32)
                initial_rgb = {n:initial['observation'][n]['rgb'].copy() for n in CAMERAS}
                patch = None
                if job['method'] != 'clean':
                    patch = Patch(task, assets, center=job['center'], size=.10)
                    patch.texture(np.asarray(Image.open(job['texture']).convert('RGB')))
                    yaw = np.deg2rad(job['yaw_deg'])/2
                    patch.actor.set_pose(sapien.Pose(job['center'], [np.cos(yaw),0,0,np.sin(yaw)]))
                states, chunks, noise_seeds, visibility = [], [], [], []
                actual_first = {}
                while task.take_action_cnt < task.step_lim and not task.eval_success:
                    obs = task.get_obs()
                    state = obs['joint_action']['vector'].astype(np.float32)
                    raw = {w:obs['observation'][n]['rgb'] for n,w in zip(CAMERAS,WIRE)}
                    if not states:
                        actual_first = {n:obs['observation'][n]['rgb'].copy() for n in CAMERAS}
                        assert np.array_equal(state, state0)
                    if patch is not None:
                        cams = camera_objects(task)
                        visibility.append([int((cams[n].get_picture('Segmentation')[...,1] == patch.actor.per_scene_id).sum())
                                           for n in CAMERAS])
                    ns = 20261013+job['seed']*1000+len(states)
                    noise = np.random.RandomState(ns).standard_normal((32,32)).astype(np.float32)
                    actions = np.asarray(client.infer(dict(state=state,prompt=prompt,images=raw,noise=noise))['actions'])
                    assert actions.shape == (32,14) and np.isfinite(actions).all()
                    states.append(state);chunks.append(actions);noise_seeds.append(ns)
                    for action in actions:
                        task.take_action(action,action_type='qpos')
                        if task.eval_success or task.take_action_cnt >= task.step_lim:
                            break
                assert noise_seeds, 'unexpected initially complete task'
                artifact.parent.mkdir(parents=True, exist_ok=True)
                tmp = artifact.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, states14=states, actions=chunks, noise_seeds=noise_seeds,
                    visible_pixels=visibility, **{'initial_clean_'+n:v for n,v in initial_rgb.items()},
                    **{'actual_first_'+n:v for n,v in actual_first.items()})
                tmp.replace(artifact)
                row = dict(job=job, protocol_sha256=sha(root/'protocol.json'),
                    success=bool(task.eval_success), steps=task.take_action_cnt, queries=len(states), prompt=prompt,
                    initial_rgb_sha256={n:digest(v) for n,v in initial_rgb.items()},
                    initial_state_sha256=digest(state0), first_noise_seed=noise_seeds[0],
                    visible_pixels_mean=np.array(visibility).mean(0).tolist() if visibility else None)
                dump(output,row)
                done += 1
                dump(root/f'worker{worker}_status.json',dict(stage='running',new_episodes=done,last_job=job['id'],
                    success=row['success'],assigned_jobs=len(jobs)))
                print('[episode]',job['id'],row['success'],row['steps'],flush=True)
            finally:
                task.close_env()
            if limit and done >= limit:
                return
    finally:
        client._ws.close()
    dump(root/f'worker{worker}_status.json',dict(stage='smoke_complete' if limit else 'complete',assigned_jobs=len(jobs)))


def report(root):
    p = prepare(root)
    jobs = json.loads((root/'jobs.json').read_text())
    results = {}
    for job in jobs:
        path = root/'episodes'/(job['id']+'.json')
        if path.exists():
            d = json.loads(path.read_text())
            assert d['job'] == job and d['protocol_sha256'] == sha(root/'protocol.json')
            results[job['id']] = d
    rows = []
    for site in p['sites']:
        for condition in ['nominal','position','joint']:
            matched = {}
            for method in ['trained','random']:
                attacked = []
                for job in jobs:
                    if job.get('site') != site['id'] or job['condition'] != condition or job['method'] != method:
                        continue
                    clean = results.get(f"clean_{job['seed']}_{job['light']}")
                    r = results.get(job['id'])
                    if clean is None or r is None:
                        continue
                    for k in ['initial_rgb_sha256','initial_state_sha256','first_noise_seed','prompt']:
                        assert r[k] == clean[k], (job['id'],k)
                    attacked.append((job, r, clean))
                matched[method] = attacked
                eligible = [(j,r) for j,r,c in attacked if c['success']]
                failures = sum(not r['success'] for j,r in eligible)
                # Retention is restricted to clean success at both original and tested lights.
                retained = retention_n = 0
                if condition != 'nominal':
                    for j,r in eligible:
                        origin = results.get(f"{site['id']}_{j['seed']}_nominal_0_{method}")
                        origin_clean = results.get(f"clean_{j['seed']}_nominal")
                        if origin and origin_clean and origin_clean['success'] and not origin['success']:
                            retention_n += 1
                            retained += not r['success']
                rows.append(dict(site=site['id'],condition=condition,method=method,
                    paired_episodes=len(attacked),clean_success_pairs=len(eligible),attack_successes=failures,
                    ASR=failures/len(eligible) if eligible else None,
                    task_SR=sum(r['success'] for j,r,c in attacked)/len(attacked) if attacked else None,
                    origin_attack_retained=retained,origin_attack_pairs=retention_n,
                    origin_retention=retained/retention_n if retention_n else None,
                    independent_initial_states=len({j['seed'] for j,r,c in attacked})))
    dump(root/'summary.json',dict(complete=len(results)==len(jobs),completed_episodes=len(results),
        expected_episodes=len(jobs),rows=rows,protocol_sha256=sha(root/'protocol.json'),
        caution='10 initial states, one training seed per site; partial ASR denominators are not final; joint retention uses intersection of clean-success subsets'))
    with (root/'asr.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('[report]',len(results),'/',len(jobs),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['prepare','rollout','report'])
    ap.add_argument('--root',required=True)
    ap.add_argument('--port',type=int,default=8397)
    ap.add_argument('--worker',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--limit',type=int,default=0,help='smoke only')
    args=ap.parse_args();root=Path(args.root).resolve()
    if args.stage=='prepare':prepare(root)
    elif args.stage=='rollout':rollout(root,args.port,args.worker,args.workers,args.limit)
    else:report(root)
