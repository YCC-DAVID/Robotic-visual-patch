"""Frozen-patch delayed-onset experiment with clean semantic milestones.

Prepare/report need only NumPy and Pillow. Rollout uses the simulator wrapper.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import bootstrap as B
from radius_plan import dump

STAGES = ('initial', 'approach', 'lifted', 'above_plate')
DEFAULT_ROOT = B.OUT/'stage_injection_20261016_v2'
DEFAULT_TEXTURE = B.OUT/'table9_joint_eot_10cm_20261012/sites/site07/train/texture_0500.png'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prepare(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    assert seeds and len(set(seeds)) == len(seeds)
    assert args.duration_queries >= 0 and args.size > 0
    source = Path(args.texture).resolve()
    texture_hash = sha(source)
    config = dict(task='place_container_plate', task_config='demo_clean', seeds=seeds,
        texture_source=str(source), texture_sha256=texture_hash,
        center_m=args.center, size_m=args.size, duration_queries=args.duration_queries,
        stages=list(STAGES), replan=32, noise_base=20261016,
        approach_distance_m=.12, lift_height_m=.04, above_plate_xy_m=.08,
        lift_consecutive_actions=3,
        milestone_rule='first clean event; insert at next natural policy query before task termination',
        prefix_rule='replay clean action chunks; assert exact states, raw RGB and object poses through insertion',
        patch_rule='visual-only true SAPIEN RT; same world placement, no retraining',
        endpoint='clean-success conditional task failure after actual insertion; Random paired at same query',
        exposure_caution='delayed onset shortens exposure; duration_queries=1 provides a fixed-query-window follow-up',
        stage_caution='geometric events approximate task phases; no claim that proximity proves grasp',
        independent_units='initial seeds; report common reachable cohort as well as stage-specific cohort')
    path = root/'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        assert old['config'] == config, 'protocol differs; use a new output root'
        load_protocol(root)
        return old
    assert not (root/'episodes').exists(), 'preserve orphaned episodes'
    textures = root/'textures'
    textures.mkdir(exist_ok=True)
    (textures/'trained.png').write_bytes(source.read_bytes())
    Image.fromarray(np.random.RandomState(20261007).randint(0,256,(128,128,3),dtype=np.uint8)).save(textures/'random.png')
    names = ['stage_injection.py', 'stage_injection_pipeline.py', 'render_geometry.py',
             'scene_smoke.py', 'policy_server.py', 'run_sim.sh', 'bootstrap.py',
             'radius_pipeline.py', 'radius_plan.py']
    snapshot = root/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for name in names:
        data = (B.PROBE/name).read_bytes()
        (snapshot/name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    p = dict(config=config, source_sha256=hashes,
             textures_sha256={m:sha(textures/(m+'.png')) for m in ['trained','random']})
    dump(path,p)
    return p


def load_protocol(root):
    p = json.loads((root/'protocol.json').read_text())
    for name, h in p['source_sha256'].items():
        assert sha(B.PROBE/name) == h, f'source changed: {name}; use new output root'
        assert sha(root/'source_snapshot'/name) == h
    for method, h in p['textures_sha256'].items():
        assert sha(root/'textures'/(method+'.png')) == h
    return p


class Milestones:
    def __init__(self, initial_container, config):
        self.z0 = float(initial_container[2])
        self.config = config
        self.events = {'initial':0}
        self.lift_streak = 0

    def update(self, step, container, plate, tcp):
        c = self.config
        if 'approach' not in self.events and np.linalg.norm(tcp-container) <= c['approach_distance_m']:
            self.events['approach'] = step
        high = container[2]-self.z0 >= c['lift_height_m']
        self.lift_streak = self.lift_streak+1 if high else 0
        if 'lifted' not in self.events and self.lift_streak >= c['lift_consecutive_actions']:
            self.events['lifted'] = step
        if ('above_plate' not in self.events and 'lifted' in self.events and high
                and np.linalg.norm(container[:2]-plate[:2]) <= c['above_plate_xy_m']):
            self.events['above_plate'] = step

    def schedule(self, query_steps):
        return {stage:next((q for q,step in enumerate(query_steps) if step >= self.events[stage]),None)
                if stage in self.events else None for stage in STAGES}


def positions(task, arm):
    tcp = task.robot.get_left_tcp_pose() if arm == 'left' else task.robot.get_right_tcp_pose()
    return np.asarray([task.container.get_pose().p,task.plate.get_pose().p,np.asarray(tcp)[:3]],dtype=np.float64)


def episode(root, client, seed, method, stage=None):
    from scene_smoke import make_task
    from render_geometry import Patch, CAMERAS, WIRE, camera_objects
    p = load_protocol(root)
    c = p['config']
    label = f'seed{seed}_{method}'+('' if stage is None else '_'+stage)
    output = root/'episodes'/(label+'.json')
    protocol_hash = sha(root/'protocol.json')
    if output.exists():
        row = json.loads(output.read_text())
        assert row['protocol_sha256'] == protocol_hash
        assert row['seed'] == seed and row['method'] == method and row['stage'] == stage
        if row['status'] == 'complete':
            assert sha(output.with_suffix('.npz')) == row['artifact_sha256']
        return row
    clean = None
    reference = None
    trigger = None
    if method != 'clean':
        clean_path = root/'episodes'/f'seed{seed}_clean.json'
        clean = json.loads(clean_path.read_text())
        assert clean['protocol_sha256'] == protocol_hash
        trigger = clean['schedule'][stage]
        if trigger is None:
            row = dict(seed=seed,method=method,stage=stage,status='no_injection_opportunity',
                protocol_sha256=protocol_hash,success=None,trigger_query=None,
                reason='event absent or no policy query remained after event')
            dump(output,row)
            return row
        clean_artifact = clean_path.with_suffix('.npz')
        assert sha(clean_artifact) == clean['artifact_sha256']
        with np.load(clean_artifact) as data:
            reference = {k:data[k] for k in data.files}
    assets = root/'assets'/label
    assets.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    task, _, _ = make_task(c['task'],seed,assets)
    try:
        arm = 'left' if task.container.get_pose().p[0] < 0 else 'right'
        tracker = Milestones(task.container.get_pose().p,c)
        prompt = 'Place the '+('bowl' if task.actor_name == '002_bowl' else 'cup')+' on the plate.'
        task.set_instruction(prompt)
        patch = None
        states, chunks, noise_seeds, query_steps, rgb_hashes, poses = [], [], [], [], [], []
        active_flags, visibility, frames = [], [], {}
        injection = None
        removed = False
        while task.take_action_cnt < task.step_lim and not task.eval_success:
            q = len(states)
            clean_obs = task.get_obs()
            state = clean_obs['joint_action']['vector'].astype(np.float32)
            raw_rgb = [clean_obs['observation'][n]['rgb'] for n in CAMERAS]
            pose = positions(task,arm)
            if method == 'clean':
                tracker.update(task.take_action_cnt,*pose)
            if reference is not None and q <= trigger:
                assert q < len(reference['states14']), 'terminated clean prefix mismatch'
                assert np.array_equal(state,reference['states14'][q]), (label,q,'state')
                assert np.array_equal(pose,reference['object_tcp_positions'][q]), (label,q,'poses')
                assert [digest(v) for v in raw_rgb] == reference['rgb_sha256'][q].tolist(), (label,q,'RGB')
                assert task.take_action_cnt == reference['query_steps'][q]
                assert prompt == clean['prompt']
            if q == trigger:
                patch = Patch(task,assets,center=c['center_m'],size=c['size_m'])
                patch.texture(np.asarray(Image.open(root/'textures'/(method+'.png')).convert('RGB')))
                obs = task.get_obs()
                assert np.array_equal(obs['joint_action']['vector'],clean_obs['joint_action']['vector'])
                assert np.array_equal(positions(task,arm),pose), 'visual insertion changed physics'
                injection = dict(query=q,step=task.take_action_cnt,
                    clean_event_step=clean['events'][stage],
                    delay_actions=task.take_action_cnt-clean['events'][stage])
                for n,v in zip(CAMERAS,raw_rgb):
                    frames['before_'+n] = v.copy()
                    frames['after_'+n] = obs['observation'][n]['rgb'].copy()
            elif (patch is not None and c['duration_queries'] and
                    q >= trigger+c['duration_queries'] and not removed):
                task.scene.remove_actor(patch.actor)
                removed = True
                obs = task.get_obs()
            else:
                obs = clean_obs
            active = patch is not None and not removed
            images = {w:obs['observation'][n]['rgb'] for n,w in zip(CAMERAS,WIRE)}
            ns = c['noise_base']+seed*1000+q
            if reference is not None and q < trigger:
                assert ns == int(reference['noise_seeds'][q])
                actions = reference['actions'][q].copy()
            else:
                noise = np.random.RandomState(ns).standard_normal((32,32)).astype(np.float32)
                actions = np.asarray(client.infer(dict(state=state,prompt=prompt,images=images,noise=noise))['actions'])
            assert actions.shape == (32,14) and np.isfinite(actions).all()
            states.append(state); chunks.append(actions); noise_seeds.append(ns)
            query_steps.append(task.take_action_cnt); poses.append(pose)
            rgb_hashes.append([digest(obs['observation'][n]['rgb']) for n in CAMERAS])
            active_flags.append(active)
            cams = camera_objects(task)
            visibility.append([int((cams[n].get_picture('Segmentation')[...,1] == patch.actor.per_scene_id).sum())
                               for n in CAMERAS] if active else [0,0,0])
            for action in actions:
                task.take_action(action,action_type='qpos')
                if method == 'clean':
                    tracker.update(task.take_action_cnt,*positions(task,arm))
                if task.eval_success or task.take_action_cnt >= task.step_lim:
                    break
        assert states
        if method != 'clean':
            assert injection is not None, 'terminated before scheduled insertion; prefix mismatch'
        final = task.get_obs()
        for n in CAMERAS:
            frames['final_'+n] = final['observation'][n]['rgb'].copy()
        artifact = output.with_suffix('.npz')
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temp = artifact.with_suffix('.tmp.npz')
        np.savez_compressed(temp,states14=states,actions=chunks,noise_seeds=noise_seeds,
            query_steps=query_steps,rgb_sha256=rgb_hashes,object_tcp_positions=poses,
            patch_active=active_flags,visible_pixels=visibility,**frames)
        temp.replace(artifact)
        row = dict(seed=seed,method=method,stage=stage,status='complete',
            protocol_sha256=protocol_hash,artifact_sha256=sha(artifact),
            success=bool(task.eval_success),steps=task.take_action_cnt,queries=len(states),
            prompt=prompt,events=tracker.events if method == 'clean' else None,
            schedule=tracker.schedule(query_steps) if method == 'clean' else None,
            injection=injection,trigger_query=trigger,
            active_queries=sum(active_flags),clean_prefix_verified=method != 'clean')
        dump(output,row)
        print('[episode]',label,row['success'],row['steps'],injection,flush=True)
        return row
    finally:
        task.close_env()


def rollout(args):
    B.simulator_paths()
    import torch
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    torch.set_num_threads(4)
    root = Path(args.root).resolve()
    p = load_protocol(root)
    assert 0 <= args.worker < args.workers
    client = WebsocketClientPolicy('127.0.0.1',args.port)
    try:
        seeds = p['config']['seeds'][args.worker::args.workers]
        if args.limit_seeds:
            seeds = seeds[:args.limit_seeds]
        for seed in seeds:
            episode(root,client,seed,'clean')
            for stage in STAGES:
                for method in ['trained','random']:
                    episode(root,client,seed,method,stage)
    finally:
        client._ws.close()


def report(root):
    p = load_protocol(root)
    records = {}
    for path in (root/'episodes').glob('*.json'):
        row = json.loads(path.read_text())
        assert row['protocol_sha256'] == sha(root/'protocol.json')
        records[(row['seed'],row['method'],row['stage'])] = row
    seeds = p['config']['seeds']
    common = [s for s in seeds if (s,'clean',None) in records
              and records[s,'clean',None]['success']
              and all(records[s,'clean',None]['schedule'][stage] is not None for stage in STAGES)]
    rows = []
    for stage in STAGES:
        pairs = []
        for seed in seeds:
            clean = records.get((seed,'clean',None))
            trained, random_row = [records.get((seed,m,stage)) for m in ['trained','random']]
            if clean and clean['success'] and trained and random_row:
                if trained['status'] == random_row['status'] == 'complete':
                    assert trained['injection'] == random_row['injection']
                    pairs.append((seed,trained,random_row))
        for cohort, values in [('stage_reachable',pairs),('all_stages_reachable',[r for r in pairs if r[0] in common])]:
            n = len(values)
            tf = sum(not t['success'] for _,t,r in values)
            rf = sum(not r['success'] for _,t,r in values)
            rows.append(dict(stage=stage,cohort=cohort,n=n,trained_failures=tf,random_failures=rf,
                trained_ASR=tf/n if n else None,random_ASR=rf/n if n else None,
                excess_ASR=(tf-rf)/n if n else None,seeds=[s for s,t,r in values],
                trained_only_failure=sum(not t['success'] and r['success'] for _,t,r in values),
                random_only_failure=sum(t['success'] and not r['success'] for _,t,r in values)))
    expected = len(seeds)*(1+2*len(STAGES))
    summary = dict(complete=len(records)==expected,recorded=len(records),expected=expected,
        clean_completed=sum((s,'clean',None) in records for s in seeds),
        clean_successes=sum(bool(records.get((s,'clean',None),{}).get('success')) for s in seeds),
        no_injection_opportunities=sum(r['status']=='no_injection_opportunity' for r in records.values()),
        common_reachable_seeds=common,rows=rows,
        caution='pilot; same-query stages may coincide; delayed onset changes exposure duration; proximity is geometric')
    dump(root/'summary.json',summary)
    with (root/'asr.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('[report]',summary['recorded'],'/',expected,flush=True)
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','rollout','report'])
    ap.add_argument('--root',default=str(DEFAULT_ROOT))
    ap.add_argument('--texture',default=str(DEFAULT_TEXTURE))
    ap.add_argument('--center',type=float,nargs=3,default=[-.04,.24,.741])
    ap.add_argument('--size',type=float,default=.10)
    ap.add_argument('--seeds',default=','.join(map(str,range(10800,10810))))
    ap.add_argument('--duration-queries',type=int,default=0)
    ap.add_argument('--port',type=int,default=8416)
    ap.add_argument('--worker',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--limit-seeds',type=int,default=0)
    return ap


if __name__ == '__main__':
    args = parser().parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'rollout':
        rollout(args)
    else:
        report(Path(args.root).resolve())
