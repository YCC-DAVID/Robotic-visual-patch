"""Event-aligned complete expert replay and physical EOT calibration bank."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import bootstrap as B
from crossarch_common import DEFAULT,EXPERT,CAMERAS,STAGES,dump,prepare

def stages(states,objects,plate):
    active=13 if objects[0,0]>0 else 6
    def first(indices,name):
        if not len(indices):raise ValueError('Missing trajectory event: '+name)
        return int(indices[0])
    close=first(np.flatnonzero(states[:,active]<.85),'gripper closing')
    lift=first(np.flatnonzero((np.arange(len(states))>close)&(objects[:,2]>objects[0,2]+.03)),'lift 3cm')
    near=first(np.flatnonzero((np.arange(len(states))>lift)&(np.linalg.norm(objects[:,:2]-plate[:2],axis=1)<.06)),'arrival at plate')
    bounds=[0,close,lift,near,len(states)]
    assert all(b>a for a,b in zip(bounds,bounds[1:])),bounds
    return dict(zip(STAGES,zip(bounds,bounds[1:])))

def trace(root,seed):
    B.simulator_paths()
    from scene_smoke import make_task
    from PIL import Image,ImageDraw
    p=prepare(root);dest=root/'trajectories'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    with np.load(EXPERT/f'seed{seed}/demonstration.npz') as z:actions=z['state14'].copy();prompt=str(z['prompt'])
    task=None;random.seed(seed)
    try:
        task,_,_=make_task(p['task'],seed,dest)
        ss=[];objects=[];rgb={n:[] for n in CAMERAS};plate=task.plate.get_pose().p.copy()
        for action in actions:
            obs=task.get_obs();ss.append(obs['joint_action']['vector'].astype(np.float32));objects.append(task.container.get_pose().p.copy())
            for n in CAMERAS:rgb[n].append(obs['observation'][n]['rgb'].copy())
            task.take_action(action,action_type='qpos')
            if task.eval_success or task.take_action_cnt>=task.step_lim:break
        obs=task.get_obs();ss.append(obs['joint_action']['vector'].astype(np.float32));objects.append(task.container.get_pose().p.copy())
        for n in CAMERAS:rgb[n].append(obs['observation'][n]['rgb'].copy())
        assert task.eval_success,('Expert replay did not succeed',seed)
        spans=stages(np.array(ss),np.array(objects),plate)
        is_train=seed in p['data']['train_seeds'];count=4 if is_train else 2
        selected=[]
        for stage,(lo,hi) in spans.items():
            indices=np.unique(np.linspace(lo,hi-1,count).round().astype(int))
            selected.extend(dict(stage=stage,index=int(i)) for i in indices)
        indices=[r['index'] for r in selected]
        np.savez_compressed(dest/'frames.npz',state14=np.array(ss)[indices],query=indices,prompt=prompt,
            **{n:np.array(rgb[n])[indices] for n in CAMERAS})
        np.savez_compressed(dest/'trace.npz',state14=ss,object_position=objects,plate_position=plate,
                            actions=actions[:len(ss)-1],prompt=prompt)
        canvas=Image.new('RGB',(320*count,270*4),'white');draw=ImageDraw.Draw(canvas)
        for si,stage in enumerate(STAGES):
            for col,r in enumerate(x for x in selected if x['stage']==stage):
                canvas.paste(Image.fromarray(rgb['head_camera'][r['index']]),(col*320,si*270))
                draw.text((col*320+5,si*270+243),f'{stage} | frame {r["index"]}',fill='black')
        canvas.save(dest/'stages.png')
        dump(dest/'complete.json',dict(seed=seed,success=True,split='train' if is_train else 'validation',
            spans=spans,selected=selected,total_frames=len(ss),prompt=prompt))
        print('[trace]',seed,spans,flush=True)
    finally:
        if task is not None:task.close_env()

def records(root,seed):
    p=prepare(root);meta=json.loads((root/'trajectories'/f'seed{seed}/complete.json').read_text())
    rng=np.random.default_rng(20261025+seed);rs=[]
    train=meta['split']=='train'
    for row in meta['selected']:
        for draw in range(4 if train else 2):
            center=np.array(p['patch']['center_m']);nominal=not train and draw==0
            radius=.03*np.sqrt(rng.random());angle=rng.uniform(0,2*np.pi)
            if not nominal:center[:2]+=radius*np.array([np.cos(angle),np.sin(angle)])
            yaw=0. if nominal else float(rng.uniform(-15,15))
            light='nominal' if nominal else str(rng.choice(p['patch']['eot']['lights']))
            ident=f's{seed}_f{row["index"]:04d}_d{draw}'
            rs.append(dict(id=ident,seed=seed,query=row['index'],stage=row['stage'],split=meta['split'],
                center_m=center.tolist(),yaw_deg=yaw,light=light,prompt=meta['prompt'],
                path=str(root/'bank'/meta['split']/(ident+'.npz'))))
    return rs

def capture(root,seed,post=False):
    B.simulator_paths()
    import sapien
    from PIL import Image
    from scene_smoke import make_task
    from render_geometry import Patch,camera_objects,patterns
    from eot_conditions import Lighting,LIGHTS
    p=prepare(root);dest=root/'trajectories'/f'seed{seed}'
    rs=records(root,seed)
    if post:rs=[r for r in rs if r['split']=='validation']
    output=lambda r:root/'post'/(r['id']+'.npz') if post else Path(r['path'])
    rs=[r for r in rs if not output(r).exists()]
    if not rs:return
    if post:
        textures={k:np.asarray(Image.open(root/'patches'/k/'texture_0500.png').convert('RGB')) for k in ('pi','wam','joint')}
        textures['random']=patterns()['random']
    else:
        textures={k:np.full((128,128,3),v,np.uint8) for k,v in [('black',0),('gray64',64),('gray',128),('gray192',192),('white',255)]}
        if rs[0]['split']=='validation':textures.update({k:patterns()[k] for k in ('random','checker')})
    with np.load(dest/'trace.npz') as z:actions=z['actions'].copy();states=z['state14'].copy()
    with np.load(dest/'frames.npz') as z:refs={int(q):{n:z[n][i].copy() for n in CAMERAS} for i,q in enumerate(z['query'])}
    task=None;random.seed(seed)
    try:
        task,_,_=make_task(p['task'],seed,dest);lighting=Lighting(task)
        for query in range(max(r['query'] for r in rs)+1):
            group=[r for r in rs if r['query']==query]
            if group:
                lighting.apply(LIGHTS['nominal']);obs=task.get_obs();state=obs['joint_action']['vector'].astype(np.float32)
                assert np.array_equal(state,states[query]),('Replay state mismatch',seed,query)
                for n in CAMERAS:assert np.array_equal(obs['observation'][n]['rgb'],refs[query][n]),('Replay pixels mismatch',seed,query,n)
                for r in group:
                    lighting.apply(LIGHTS[r['light']]);clean=task.get_obs()
                    data=dict(state14=state,seed=seed,query=query,prompt=r['prompt'],stage=r['stage'],
                              center_m=r['center_m'],size_m=float(p['patch']['size_m']),yaw_deg=r['yaw_deg'],light=r['light'])
                    for n in CAMERAS:data[n+'_clean']=clean['observation'][n]['rgb'].copy()
                    patch=Patch(task,dest,center=r['center_m'],size=float(p['patch']['size_m']))
                    try:
                        for ti,(label,texture) in enumerate(textures.items()):
                            patch.texture(texture);a=np.deg2rad(r['yaw_deg'])/2
                            patch.actor.set_pose(sapien.Pose(r['center_m'],[np.cos(a),0,0,np.sin(a)]))
                            obs=task.get_obs();cams=camera_objects(task)
                            assert np.array_equal(state,obs['joint_action']['vector'].astype(np.float32))
                            for n in CAMERAS:
                                data[n+'_'+label]=obs['observation'][n]['rgb'].copy()
                                if ti==0:
                                    data[n+'_mask']=cams[n].get_picture('Segmentation')[...,1]==patch.actor.per_scene_id
                                    data[n+'_intrinsic']=cams[n].get_intrinsic_matrix();data[n+'_extrinsic']=cams[n].get_extrinsic_matrix()
                    finally:task.scene.remove_actor(patch.actor)
                    restored=task.get_obs()
                    for n in CAMERAS:assert np.array_equal(clean['observation'][n]['rgb'],restored['observation'][n]['rgb'])
                    target=output(r);target.parent.mkdir(parents=True,exist_ok=True)
                    tmp=target.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data);tmp.replace(target)
                    print('[capture]',r['id'],'post' if post else 'bank',flush=True)
                lighting.apply(LIGHTS['nominal'])
            if query<len(actions):task.take_action(actions[query],action_type='qpos')
        dump(dest/('post_complete.json' if post else 'capture_complete.json'),dict(complete=True,seed=seed))
    finally:
        if task is not None:task.close_env()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--mode',choices=['trace','capture','post'],required=True)
    a=ap.parse_args()
    if a.mode=='trace':trace(a.root,a.seed)
    else:capture(a.root,a.seed,post=a.mode=='post')
