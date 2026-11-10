"""Actual-render native, event-onset persistent, and one-query pulse experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np

from stage_joint_common import (DEFAULT,STAGES,CONDITIONS,CAMERAS,WIRE,PhaseTracker,
    load,request,dump,sha,jobs,job_id)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def output_path(root,job,pilot=False):
    return root/('pilot' if pilot else 'episodes')/(job_id(job)+'.json')


def native_job(seed):
    return dict(seed=seed,suite='native_full',stage='approach',condition='clean')


def noise_seed(base,seed,stage,suite,query):
    # Initial pulses and native runs share the entire noise sequence. Delayed and
    # pulse branches at the same event share their first and subsequent noises.
    return base+seed*1000+STAGES.index(stage)*100+query


def pose(task):
    return np.asarray([task.container.get_pose().p,task.plate.get_pose().p],dtype=np.float64)


def robot_state(task):
    return np.asarray(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState(),dtype=np.float32)


def check_prefix(current_state,current_pose,ref,index):
    assert np.array_equal(current_state,ref['step_states'][index]),('prefix state',index)
    assert np.array_equal(current_pose,ref['step_object_plate'][index]),('prefix objects',index)


def read_reference(root,seed,pilot=False):
    path=output_path(root,native_job(seed),pilot)
    row=json.loads(path.read_text());assert row['status']=='complete'
    assert sha(path.with_suffix('.npz'))==row['artifact_sha256']
    with np.load(path.with_suffix('.npz')) as z:
        arrays={k:z[k].copy() for k in z.files}
    return row,arrays


def run_episode(root,job,pilot=False):
    import bootstrap as B
    B.simulator_paths()
    from scene_smoke import make_task
    from render_geometry import Patch,camera_objects
    from PIL import Image
    p=load(root);path=output_path(root,job,pilot);ph=sha(root/'protocol.json')
    if path.exists():
        r=json.loads(path.read_text())
        assert r['job']==job and r['protocol_sha256']==ph
        if r.get('error'):
            raise RuntimeError(f'Preserved unresolved episode error: {path}')
        if r['status']=='complete':
            assert sha(path.with_suffix('.npz'))==r['artifact_sha256']
        return r
    suite=job['suite'];condition=job['condition'];stage=job['stage'];seed=job['seed']
    is_native_clean=suite=='native_full' and condition=='clean'
    native,ref=(None,None) if is_native_clean else read_reference(root,seed,pilot)
    trigger=0
    if suite!='native_full':
        trigger=native['events'].get(stage)
        if trigger is None or trigger>=native['steps']:
            r=dict(job=job,status='no_opportunity',success=None,protocol_sha256=ph,
                reason='No clean event before native task termination',trigger_step=trigger)
            dump(path,r);return r
    assets=root/'assets'/('pilot' if pilot else 'formal')/job_id(job)
    assets.mkdir(parents=True,exist_ok=True)
    task=None;began=time.monotonic();random.seed(seed)
    try:
        task,_,_=make_task(p['task'],seed,assets)
        prompt='Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
        task.set_instruction(prompt)
        initial=task.get_obs();state0=initial['joint_action']['vector'].astype(np.float32)
        initial_hash=digest(np.concatenate([state0.view(np.uint8).ravel()]+[initial['observation'][n]['rgb'].ravel() for n in CAMERAS]))
        if native is not None:
            assert native['initial_clean_sha256']==initial_hash
            assert native['prompt']==prompt
        tracker=PhaseTracker(task.container.get_pose().p)
        step_states=[robot_state(task)];step_poses=[pose(task)]
        prefix_actions=[];prefix_steps=0;prefix_queries=0
        if suite!='native_full':
            check_prefix(step_states[0],step_poses[0],ref,0)
            query_at_step={int(s):i for i,s in enumerate(ref['query_steps'])}
            for action in ref['executed_actions'][:trigger]:
                task.take_action(action,action_type='qpos')
                state=robot_state(task);obj=pose(task)
                check_prefix(state,obj,ref,task.take_action_cnt)
                tracker.update(state,*obj,task.take_action_cnt)
                step_states.append(state);step_poses.append(obj);prefix_actions.append(action.copy())
                if task.take_action_cnt in query_at_step:
                    qi=query_at_step[task.take_action_cnt];prefix_obs=task.get_obs()
                    assert [digest(prefix_obs['observation'][n]['rgb']) for n in CAMERAS]==ref['rgb_sha256'][qi].tolist()
                    prefix_queries+=1
                assert not task.eval_success,'Terminated before declared native event'
            prefix_steps=task.take_action_cnt
            assert prefix_steps==trigger
            before=task.get_obs()
            ei=int(np.flatnonzero(ref['event_steps']==trigger)[0])
            assert [digest(before['observation'][n]['rgb']) for n in CAMERAS]==ref['event_rgb_sha256'][ei].tolist()
        else:
            before=task.get_obs()
        texture=None
        if condition!='clean':
            tex_path=root/'patches'/condition/('texture.png' if condition=='random' else 'texture_0500.png')
            texture=np.asarray(Image.open(tex_path).convert('RGB'))
        patch=None;ever_inserted=False
        states=[];chunks=[];query_steps=[];noise_seeds=[];hashes=[];poses=[];phases=[]
        active=[];visibility=[];head=[];executed=list(prefix_actions)
        event_steps=[0];event_hashes=[[digest(initial['observation'][n]['rgb']) for n in CAMERAS]]
        onset_frames={};injection=None;affected_actions=0
        while task.take_action_cnt<task.step_lim and not task.eval_success:
            q=len(states);obs=task.get_obs()
            state=obs['joint_action']['vector'].astype(np.float32)
            should_patch=condition!='clean' and (suite!='pulse' or q==0)
            if should_patch and patch is None:
                baseline_state=obs['joint_action']['vector'].copy();baseline_pose=pose(task)
                onset_frames.update({'before_'+n:obs['observation'][n]['rgb'].copy() for n in CAMERAS})
                patch=Patch(task,assets,center=p['patch']['center_m'],size=p['patch']['size_m'])
                patch.texture(texture)
                obs=task.get_obs()
                assert np.array_equal(obs['joint_action']['vector'],baseline_state)
                assert np.array_equal(pose(task),baseline_pose),'Visual actor changed physics'
                onset_frames.update({'after_'+n:obs['observation'][n]['rgb'].copy() for n in CAMERAS})
                injection=dict(step=task.take_action_cnt,query=q,stage=tracker.phase)
                ever_inserted=True
            elif not should_patch and patch is not None:
                task.scene.remove_actor(patch.actor);patch=None;obs=task.get_obs()
            raw={w:obs['observation'][n]['rgb'] for n,w in zip(CAMERAS,WIRE)}
            ns=noise_seed(p['evaluation']['noise_base'],seed,stage,suite,q)
            chunk=np.asarray(request(dict(command='infer',state=state,prompt=prompt,images=raw,seed=ns),root)['actions'])
            assert chunk.shape==(32,14) and np.isfinite(chunk).all()
            states.append(state);chunks.append(chunk);query_steps.append(task.take_action_cnt);noise_seeds.append(ns)
            hashes.append([digest(obs['observation'][n]['rgb']) for n in CAMERAS]);poses.append(pose(task))
            phases.append(tracker.phase);active.append(patch is not None);head.append(raw[WIRE[0]].copy())
            cams=camera_objects(task)
            visibility.append([int((cams[n].get_picture('Segmentation')[...,1]==patch.actor.per_scene_id).sum()) for n in CAMERAS]
                              if patch is not None else [0,0,0])
            for action in chunk:
                prior_events=set(tracker.events)
                task.take_action(action,action_type='qpos')
                state=robot_state(task);obj=pose(task);tracker.update(state,*obj,task.take_action_cnt)
                executed.append(action.copy());step_states.append(state);step_poses.append(obj)
                if patch is not None:
                    affected_actions+=1
                if set(tracker.events)!=prior_events:
                    event_obs=task.get_obs()
                    event_steps.append(task.take_action_cnt)
                    event_hashes.append([digest(event_obs['observation'][n]['rgb']) for n in CAMERAS])
                if task.eval_success or task.take_action_cnt>=task.step_lim:
                    break
        assert states
        if condition!='clean':
            assert ever_inserted and sum(active)>0
            if suite=='pulse':
                assert sum(active)==1,'Pulse must affect exactly one policy query'
            assert injection['step']==trigger
        final=task.get_obs()
        artifact=path.with_suffix('.npz');artifact.parent.mkdir(parents=True,exist_ok=True)
        tmp=artifact.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,states14=states,actions=chunks,query_steps=query_steps,noise_seeds=noise_seeds,
            rgb_sha256=hashes,object_plate=poses,query_phases=phases,patch_active=active,visible_pixels=visibility,
            head=head,executed_actions=executed,step_states=step_states,step_object_plate=step_poses,
            event_steps=event_steps,event_rgb_sha256=event_hashes,**onset_frames,
            **{'final_'+n:final['observation'][n]['rgb'] for n in CAMERAS})
        tmp.replace(artifact)
        r=dict(job=job,status='complete',protocol_sha256=ph,artifact_sha256=sha(artifact),
            success=bool(task.eval_success),steps=task.take_action_cnt,queries=len(states),prompt=prompt,
            initial_clean_sha256=initial_hash,events=tracker.events,injection=injection,trigger_step=trigger,
            prefix_verified_actions=prefix_steps,prefix_verified_queries=prefix_queries,
            active_queries=sum(active),affected_actions=affected_actions,
            visible_queries=sum(any(v) for v in visibility),elapsed_s=time.monotonic()-began)
        dump(path,r);print('[episode]',job_id(job),r['success'],r['steps'],flush=True)
        return r
    except Exception as e:
        dump(path,dict(job=job,status='runtime_error',protocol_sha256=ph,error=repr(e),traceback=traceback.format_exc()))
        raise
    finally:
        if task is not None:
            task.close_env()


def run(root,seed,pilot=False):
    if pilot:
        selected=[dict(seed=seed,suite='native_full',stage='approach',condition=c) for c in ('clean','random')]
        for suite,stage in [('delayed','grasp'),('pulse','grasp')]:
            selected.extend(dict(seed=seed,suite=suite,stage=stage,condition=c) for c in ('clean','random'))
    else:
        selected=[j for j in jobs(root) if j['seed']==seed]
    for job in selected:
        run_episode(root,job,pilot)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--pilot',action='store_true')
    args=ap.parse_args();run(args.root.resolve(),args.seed,args.pilot)
