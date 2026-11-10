"""Native-policy early-state collection and sustained real-render evaluation."""
import argparse
import json
from pathlib import Path
import random

import numpy as np

from e_method_common import DEFAULT,CAMERAS,WIRE,cohort,dump,load,request,sha


def collect(root,seed):
    import bootstrap as B
    B.simulator_paths()
    import sapien
    from scene_smoke import make_task
    from render_geometry import Patch,camera_objects,patterns
    from eot_conditions import Lighting,LIGHTS
    from stage_joint_common import PhaseTracker
    p=load(root);output=root/'collection'/f'seed{seed}.json'
    if output.exists():
        data=json.loads(output.read_text());assert data['protocol_sha256']==sha(root/'protocol.json')
        for row in data['records']:assert sha(row['path'])==row['sha256']
        return
    assets=root/'assets/collection'/f'seed{seed}';assets.mkdir(parents=True,exist_ok=True)
    random.seed(seed);task=None;records=[];traces=[];captured=0
    try:
        task,_,_=make_task(p['task'],seed,assets)
        prompt='Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
        task.set_instruction(prompt)
        tracker=PhaseTracker(task.container.get_pose().p)
        active_start=7 if task.container.get_pose().p[0]>0 else 0
        lighting=Lighting(task)
        query=0
        while not task.eval_success and task.take_action_cnt<task.step_lim:
            obs=task.get_obs();state=obs['joint_action']['vector'].astype(np.float32)
            if tracker.phase=='approach' and captured<3:
                for draw in range(2):
                    rng=np.random.default_rng(202610270+seed*100+query)
                    center=np.array(p['patch']['center_m']);yaw=0.;light='nominal'
                    if draw:
                        radius=.03*np.sqrt(rng.random());angle=rng.uniform(0,2*np.pi)
                        center[:2]+=radius*np.array([np.cos(angle),np.sin(angle)])
                        yaw=float(rng.uniform(-15,15));light=str(rng.choice(list(LIGHTS)))
                    lighting.apply(LIGHTS[light]);clean=task.get_obs()
                    clean_pose=np.array([task.container.get_pose().p,task.plate.get_pose().p])
                    ident=f's{seed}_q{query:03d}_d{draw}'
                    data=dict(seed=seed,query=query,policy_action_step=task.take_action_cnt,stage='approach',
                        prompt=prompt,state14=state,active_start=active_start,center_m=center,size_m=float(p['patch']['size_m']),
                        yaw_deg=yaw,light=light)
                    for camera in CAMERAS:data[camera+'_clean']=clean['observation'][camera]['rgb'].copy()
                    patch=Patch(task,assets,center=center,size=float(p['patch']['size_m']))
                    patterns_to_render={**{k:np.full((128,128,3),v,np.uint8) for k,v in
                        [('black',0),('gray64',64),('gray',128),('gray192',192),('white',255)]},
                        **{k:patterns()[k] for k in ('random','checker')}}
                    try:
                        for index,(label,texture) in enumerate(patterns_to_render.items()):
                            patch.texture(texture);angle=np.deg2rad(yaw)/2
                            patch.actor.set_pose(sapien.Pose(center,[np.cos(angle),0,0,np.sin(angle)]))
                            rendered=task.get_obs();cameras=camera_objects(task)
                            assert np.array_equal(rendered['joint_action']['vector'],clean['joint_action']['vector'])
                            assert np.array_equal(clean_pose,np.array([task.container.get_pose().p,task.plate.get_pose().p]))
                            for camera in CAMERAS:
                                data[camera+'_'+label]=rendered['observation'][camera]['rgb'].copy()
                                if index==0:
                                    data[camera+'_mask']=cameras[camera].get_picture('Segmentation')[...,1]==patch.actor.per_scene_id
                                    data[camera+'_intrinsic']=cameras[camera].get_intrinsic_matrix()
                                    data[camera+'_extrinsic']=cameras[camera].get_extrinsic_matrix()
                    finally:task.scene.remove_actor(patch.actor)
                    restored=task.get_obs()
                    for camera in CAMERAS:
                        assert np.array_equal(restored['observation'][camera]['rgb'],clean['observation'][camera]['rgb'])
                    path=root/'bank'/f'{ident}.npz';path.parent.mkdir(exist_ok=True)
                    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**data);temp.replace(path)
                    records.append(dict(seed=seed,query=query,draw=draw,path=str(path),sha256=sha(path)))
                    print('[capture]',ident,flush=True)
                lighting.apply(LIGHTS['nominal']);captured+=1
                dump(root/'collection_progress.json',dict(seed=seed,queries_captured=captured,records=len(records)))
            obs=task.get_obs();raw={wire:obs['observation'][camera]['rgb'].copy() for camera,wire in zip(CAMERAS,WIRE)}
            noise_seed=p['evaluation']['noise_base']+seed*1000+query
            action=np.asarray(request(dict(command='infer',state=state,prompt=prompt,images=raw,seed=noise_seed),root)['actions'])
            assert action.shape==(32,14) and np.isfinite(action).all()
            traces.append(dict(query=query,step=task.take_action_cnt,phase=tracker.phase,noise_seed=noise_seed))
            for item in action:
                task.take_action(item,action_type='qpos')
                current=np.asarray(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState(),np.float32)
                tracker.update(current,task.container.get_pose().p,task.plate.get_pose().p,task.take_action_cnt)
                if task.eval_success or task.take_action_cnt>=task.step_lim:break
            query+=1
        assert records,'No initial approach state captured'
        dump(output,dict(seed=seed,complete=True,records=records,traces=traces,
            clean_success=bool(task.eval_success),steps=task.take_action_cnt,
            protocol_sha256=sha(root/'protocol.json'),source='own clean policy; no outcome filtering'))
    finally:
        if task is not None:task.close_env()


def evaluate(root,split,seed,conditions):
    import stage_joint_rollout as R
    R.request=lambda payload,unused_root=None:request(payload,root)
    target=cohort(root,split)
    for condition in conditions:
        R.run_episode(target,dict(seed=seed,suite='native_full',stage='approach',condition=condition))


def main(args):
    root=args.root.resolve();p=load(root)
    if args.mode=='collect':
        for seed in p['method_study']['training_seeds']:collect(root,seed)
        rows=[json.loads((root/'collection'/f'seed{s}.json').read_text()) for s in p['method_study']['training_seeds']]
        dump(root/'collection_manifest.json',dict(complete=True,seeds=p['method_study']['training_seeds'],
            records=[r for row in rows for r in row['records']],
            clean_successes=sum(r['clean_success'] for r in rows),no_outcome_filter=True,
            protocol_sha256=sha(root/'protocol.json')))
    else:evaluate(root,args.split,args.seed,args.conditions.split(','))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--mode',choices=['collect','evaluate'],required=True)
    parser.add_argument('--split',choices=['development','test']);parser.add_argument('--seed',type=int)
    parser.add_argument('--conditions',default='clean,random')
    main(parser.parse_args())
