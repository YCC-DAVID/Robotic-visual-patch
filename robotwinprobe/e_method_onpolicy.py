"""Fourth E round: matched continuation with clean or frozen-patch rollout states."""
import argparse
import json
from pathlib import Path
import random

import numpy as np

from e_method_common import DEFAULT,CAMERAS,WIRE,dump,load,request,sha

ONPOLICY_METHODS=('velocity_replay_continue','velocity_onpolicy_mix')
CAPTURE_QUERIES=(0,1,2,4,8,16)


def prepare_onpolicy(runner):
    root=runner.root;path=root/'onpolicy_round/protocol.json'
    config=dict(round=4,methods=list(ONPOLICY_METHODS),loss_kind='velocity14',
        base_protocol_sha256=sha(root/'protocol.json'),clean_collection_sha256=sha(root/'collection_manifest.json'),
        parent_texture_sha256=sha(root/'patches/velocity14/texture_0500.png'),
        parent_state_sha256=sha(root/'training/velocity14/state.npz'),
        parent='velocity14 fixed500; same float32 parent texture for both continuations',
        training_seeds=runner.settings['training_seeds'],capture_queries=list(CAPTURE_QUERIES),
        capture_stage='approach only, using unchanged PhaseTracker; no success/failure filtering',
        collection_exposure='frozen parent texture persists at nominal site07/10cm for every executed action',
        calibration='same nominal plus recorded 3cm/15deg/light EOT; policy acts only after nominal lighting and parent patch restoration',
        budget='500 additional updates x8, same lr .01, same first-step14 L0, fixed final500 checkpoint',
        control='eight original clean-state draws per update',
        treatment='four identical original clean-state draws plus four frozen-parent attacked-state draws per update',
        matching='identical noises and nominal/EOT choices at each slot; first four state draws exactly shared; uniform source seed then query within either pool',
        replication='each additional training seed retrains its own gray-initialized velocity14 parent500 then continuation500; frozen pilot attacked-state bank shared as fixed training data',
        hypothesis='clean-only early training may miss states visited under persistent patch; not established by initial-only transfer diagnostic',
        diagnostic='selected training queries also infer clean RGB at the SAME attacked state/noise; paired image response is not a counterfactual clean trajectory or test ASR',
        development='same32 initial states/noises and existing GPU assignments; existing baseline data retained',
        independent_test='sealed; never used to choose training data, method, checkpoint, or denominator')
    if path.exists():assert json.loads(path.read_text())==config
    else:dump(path,config)
    return config


def grouped_records(rows):
    groups={}
    for row in rows:groups.setdefault(row['seed'],{}).setdefault(row['query'],{})[row['draw']]=row['path']
    assert groups and all(set(draws)=={0,1} for queries in groups.values() for draws in queries.values())
    return groups


def continuation_draw_plan(root,seed,method):
    assert method in ONPOLICY_METHODS
    old=grouped_records(json.loads((root/'collection_manifest.json').read_text())['records'])
    new=grouped_records(json.loads((root/'onpolicy_round/collection_manifest.json').read_text())['records'])
    rng_old=np.random.default_rng(seed);rng_new=np.random.default_rng(seed+1_000_000)
    rng_noise=np.random.default_rng(seed+2_000_000);rng_eot=np.random.default_rng(seed+3_000_000)
    def sample(pool,rng,draw):
        source=int(rng.choice(sorted(pool)));query=int(rng.choice(sorted(pool[source])))
        return pool[source][query][draw]
    result=[]
    for _ in range(load(root)['method_study']['steps']):
        batch=[]
        for slot in range(8):
            draw=int(rng_eot.random()>=.75)
            old_record=sample(old,rng_old,draw)
            new_record=sample(new,rng_new,draw)
            record=new_record if method=='velocity_onpolicy_mix' and slot>=4 else old_record
            batch.append(dict(record=record,seed=int(rng_noise.integers(1,2**30))))
        result.append(batch)
    return result


def parent_state_path(root,name):
    suffix=name.split('_replica',1)[1] if '_replica' in name else None
    return root/'training'/('velocity14_parent_replica'+suffix if suffix else 'velocity14')/'state.npz'


def parent_texture(root,name):
    config=json.loads((root/'onpolicy_round/protocol.json').read_text())
    path=parent_state_path(root,name)
    if '_replica' not in name:assert sha(path)==config['parent_state_sha256']
    else:
        saved=json.loads((path.parent/'complete.json').read_text())
        expected=load(root)['method_study']['training_random_seeds'][int(name.rsplit('_replica',1)[1])]
        assert saved['kind']=='velocity14' and saved['training_seed']==expected
    with np.load(path) as z:
        assert int(z['iteration'])==500
        return z['texture'].copy()


def collect_seed(root,seed):
    import bootstrap as B
    B.simulator_paths()
    import sapien
    from PIL import Image
    from scene_smoke import make_task
    from render_geometry import Patch,camera_objects,patterns
    from eot_conditions import Lighting,LIGHTS
    from stage_joint_common import PhaseTracker
    from stage_joint_rollout import pose,digest
    p=load(root);protocol=root/'onpolicy_round/protocol.json';config=json.loads(protocol.read_text())
    assert seed in config['training_seeds']
    output=root/'onpolicy_round/collection'/f'seed{seed}.json'
    if output.exists():
        row=json.loads(output.read_text());assert row['collection_protocol_sha256']==sha(protocol)
        for record in row['records']:assert sha(Path(record['path']))==record['sha256']
        return row
    texture_path=root/'patches/velocity14/texture_0500.png'
    assert sha(texture_path)==config['parent_texture_sha256']
    texture=np.asarray(Image.open(texture_path).convert('RGB'))
    assets=root/'assets/onpolicy_collection'/f'seed{seed}';assets.mkdir(parents=True,exist_ok=True)
    random.seed(seed);task=None;records=[];traces=[];query=0
    try:
        task,_,_=make_task(p['task'],seed,assets)
        prompt='Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
        task.set_instruction(prompt);tracker=PhaseTracker(task.container.get_pose().p)
        active_start=7 if task.container.get_pose().p[0]>0 else 0
        lighting=Lighting(task);patch=Patch(task,assets,center=p['patch']['center_m'],size=p['patch']['size_m'])
        patch.texture(texture)
        while not task.eval_success and task.take_action_cnt<task.step_lim:
            before=task.get_obs();joint_reference=before['joint_action']['vector'].copy()
            state=joint_reference.astype(np.float32)
            parent_rgb={c:before['observation'][c]['rgb'].copy() for c in CAMERAS}
            objects=pose(task).copy();clean_nominal=None
            selected=tracker.phase=='approach' and query in config['capture_queries']
            if selected:
                task.scene.remove_actor(patch.actor);patch=None
                for draw in range(2):
                    rng=np.random.default_rng(202610270+seed*100+query)
                    center=np.array(p['patch']['center_m']);yaw=0.;light='nominal'
                    if draw:
                        radius=.03*np.sqrt(rng.random());angle=rng.uniform(0,2*np.pi)
                        center[:2]+=radius*np.array([np.cos(angle),np.sin(angle)])
                        yaw=float(rng.uniform(-15,15));light=str(rng.choice(list(LIGHTS)))
                    lighting.apply(LIGHTS[light]);clean=task.get_obs()
                    if draw==0:clean_nominal={c:clean['observation'][c]['rgb'].copy() for c in CAMERAS}
                    data=dict(seed=seed,query=query,policy_action_step=task.take_action_cnt,stage=tracker.phase,
                        prompt=prompt,state14=state,active_start=active_start,center_m=center,size_m=p['patch']['size_m'],
                        yaw_deg=yaw,light=light,source='frozen_velocity14_attacked_policy_state')
                    for camera in CAMERAS:data[camera+'_clean']=clean['observation'][camera]['rgb'].copy()
                    calibration=Patch(task,assets,center=center,size=p['patch']['size_m'])
                    textures={**{k:np.full((128,128,3),v,np.uint8) for k,v in
                        [('black',0),('gray64',64),('gray',128),('gray192',192),('white',255)]},
                        **{k:patterns()[k] for k in ('random','checker')},'source_attack':texture}
                    try:
                        for index,(label,rgb) in enumerate(textures.items()):
                            calibration.texture(rgb);angle=np.deg2rad(yaw)/2
                            calibration.actor.set_pose(sapien.Pose(center,[np.cos(angle),0,0,np.sin(angle)]))
                            rendered=task.get_obs();cameras=camera_objects(task)
                            assert np.array_equal(rendered['joint_action']['vector'],joint_reference)
                            assert np.array_equal(pose(task),objects),'Calibration changed physics'
                            for camera in CAMERAS:
                                data[camera+'_'+label]=rendered['observation'][camera]['rgb'].copy()
                                if index==0:
                                    data[camera+'_mask']=cameras[camera].get_picture('Segmentation')[...,1]==calibration.actor.per_scene_id
                                    data[camera+'_intrinsic']=cameras[camera].get_intrinsic_matrix()
                                    data[camera+'_extrinsic']=cameras[camera].get_extrinsic_matrix()
                    finally:task.scene.remove_actor(calibration.actor)
                    restored=task.get_obs()
                    for camera in CAMERAS:assert np.array_equal(restored['observation'][camera]['rgb'],clean['observation'][camera]['rgb'])
                    path=root/'onpolicy_round/bank'/f's{seed}_q{query:03d}_d{draw}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**data);temp.replace(path)
                    records.append(dict(seed=seed,query=query,draw=draw,path=str(path),sha256=sha(path)))
                lighting.apply(LIGHTS['nominal'])
                patch=Patch(task,assets,center=p['patch']['center_m'],size=p['patch']['size_m']);patch.texture(texture)
                restored=task.get_obs()
                for camera in CAMERAS:assert np.array_equal(restored['observation'][camera]['rgb'],parent_rgb[camera]),'Parent patch RGB not restored'
                assert np.array_equal(restored['joint_action']['vector'],joint_reference) and np.array_equal(pose(task),objects)
            obs=task.get_obs();ns=p['evaluation']['noise_base']+seed*1000+query
            raw={w:obs['observation'][c]['rgb'].copy() for c,w in zip(CAMERAS,WIRE)}
            action=np.asarray(request(dict(command='infer',state=state,prompt=prompt,images=raw,seed=ns),root)['actions'])
            assert action.shape==(32,14) and np.isfinite(action).all()
            trace=dict(query=query,step=task.take_action_cnt,phase=tracker.phase,noise_seed=ns,captured=selected,
                state14=state.tolist(),parent_rgb_sha256=[digest(obs['observation'][c]['rgb']) for c in CAMERAS])
            if clean_nominal is not None:
                clean_raw={w:clean_nominal[c] for c,w in zip(CAMERAS,WIRE)}
                clean_action=np.asarray(request(dict(command='infer',state=state,prompt=prompt,images=clean_raw,seed=ns),root)['actions'])
                delta=action-clean_action
                trace.update(same_state_physical_action14_mse=float(np.square(delta).mean()),
                    same_state_active7_mse=float(np.square(delta[:,active_start:active_start+7]).mean()),
                    same_state_active_gripper_mse=float(np.square(delta[:,active_start+6]).mean()))
            traces.append(trace)
            for item in action:
                task.take_action(item,action_type='qpos')
                current=np.asarray(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState(),np.float32)
                tracker.update(current,task.container.get_pose().p,task.plate.get_pose().p,task.take_action_cnt)
                if task.eval_success or task.take_action_cnt>=task.step_lim:break
            query+=1
            dump(root/'onpolicy_round/collection_progress.json',dict(seed=seed,queries=query,records=len(records)))
        assert records
        row=dict(seed=seed,complete=True,records=records,traces=traces,training_rollout_success=bool(task.eval_success),
            steps=task.take_action_cnt,collection_protocol_sha256=sha(protocol),no_outcome_filter=True)
        dump(output,row);return row
    finally:
        if task is not None:task.close_env()


def collect(root):
    config=json.loads((root/'onpolicy_round/protocol.json').read_text())
    rows=[collect_seed(root,seed) for seed in config['training_seeds']]
    manifest=dict(complete=True,seeds=config['training_seeds'],records=[r for row in rows for r in row['records']],
        collection_protocol_sha256=sha(root/'onpolicy_round/protocol.json'),no_outcome_filter=True,
        scope='training rollout collection; success counts are not development/test ASR')
    dump(root/'onpolicy_round/collection_manifest.json',manifest)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    args=parser.parse_args();collect(args.root.resolve())
