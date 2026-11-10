"""Real per-draw joint pose/light calibration, with exactly one patch actor."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import bootstrap as B
from radius_plan import REFERENCE, dump
from table9_plan import records, sha
from eot_conditions import LIGHTS, Lighting


def capture(root, site, split):
    B.simulator_paths()
    import sapien
    from PIL import Image
    from scene_smoke import make_task
    from render_geometry import CAMERAS, Patch, camera_objects, patterns
    from table9_geometry_verify import projected_area, THRESHOLDS
    rs=records(root,site,split)
    pending=[r for r in rs if not Path(r['path']).exists()]
    texture_patterns={name:np.full((128,128,3),v,np.uint8) for name,v in [('black',0),('gray64',64),('gray',128),('gray192',192),('white',255)]}
    if split=='validation':
        texture_patterns.update({k:patterns()[k] for k in ['random','checker']})
        texture_patterns['baseline']=np.asarray(Image.open(REFERENCE/'train_proxy_full/texture_0500.png').convert('RGB'))
    if split=='post':
        paths={f'site{i:02d}':root/'sites'/f'site{i:02d}'/'train/texture_0500.png' for i in range(9)}
        hashes={k:sha(p) for k,p in paths.items()}
        source=root/'sites'/site/'post_texture_hashes.json'
        if source.exists():assert json.loads(source.read_text())==hashes
        else:dump(source,hashes)
        texture_patterns={'random':patterns()['random'],**{k:np.asarray(Image.open(p).convert('RGB')) for k,p in paths.items()}}
    geometry_label=next(iter(texture_patterns))
    assets=root/'sites'/site/'assets'/split;assets.mkdir(parents=True,exist_ok=True)
    complete=len(rs)-len(pending);started=time.monotonic()
    for seed in sorted({r['seed'] for r in pending}):
        these=[r for r in pending if r['seed']==seed]
        random.seed(seed);task,_,_=make_task('place_container_plate',seed,assets)
        try:
            lighting=Lighting(task)
            with np.load(REFERENCE/'clean'/f'seed{seed}.npz') as ref:
                last=max(r['query'] for r in these)
                for query,actions in enumerate(ref['actions']):
                    if query>last:break
                    group=[r for r in these if r['query']==query]
                    if group:
                        lighting.apply(LIGHTS['nominal']);nominal=task.get_obs()
                        state=nominal['joint_action']['vector'].astype(np.float32)
                        assert np.array_equal(state,ref['states14'][query])
                        for n in CAMERAS:assert np.array_equal(nominal['observation'][n]['rgb'],ref[n][query])
                        cams=camera_objects(task)
                        for light in LIGHTS:
                            selected=[r for r in group if r['light']==light]
                            if not selected:continue
                            lighting.apply(LIGHTS[light]);clean=task.get_obs()
                            clean_path=Path(selected[0]['clean_path']);clean_path.parent.mkdir(parents=True,exist_ok=True)
                            clean_data=dict(state14=state,frame=selected[0]['frame'],light=light,
                                **{n:clean['observation'][n]['rgb'].copy() for n in CAMERAS})
                            if clean_path.exists():
                                with np.load(clean_path) as old:
                                    for key,value in clean_data.items():assert np.array_equal(old[key],value)
                            else:
                                tmp=clean_path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**clean_data);tmp.replace(clean_path)
                            data={r['id']:dict(center_m=np.array(r['center']),size_m=np.array(.10),yaw_deg=np.array(r['yaw_deg']),
                                frame=r['frame'],seed=seed,query=query,state14=state,light=light,sample_id=r['id']) for r in selected}
                            patch=Patch(task,assets,center=selected[0]['center'],size=.10)
                            try:
                                for label,rgb in texture_patterns.items():
                                    patch.texture(rgb)
                                    for r in selected:
                                        patch.center=np.array(r['center']);a=np.deg2rad(r['yaw_deg'])/2
                                        patch.actor.set_pose(sapien.Pose(patch.center,[np.cos(a),0,0,np.sin(a)]))
                                        obs=task.get_obs();d=data[r['id']]
                                        assert np.array_equal(state,obs['joint_action']['vector'].astype(np.float32))
                                        for n in CAMERAS:
                                            d[n+'_'+label]=obs['observation'][n]['rgb'].copy()
                                            if label==geometry_label:
                                                d[n+'_mask']=cams[n].get_picture('Segmentation')[...,1]==patch.actor.per_scene_id
                                                d[n+'_intrinsic']=cams[n].get_intrinsic_matrix();d[n+'_extrinsic']=cams[n].get_extrinsic_matrix()
                                            elif label=='white':
                                                assert np.array_equal(d[n+'_mask'],cams[n].get_picture('Segmentation')[...,1]==patch.actor.per_scene_id)
                                        if label=='black' and split=='train':
                                            area=projected_area(d['head_camera_intrinsic'],d['head_camera_extrinsic'],np.r_[r['center'],r['yaw_deg']])
                                            exposure=float(d['head_camera_mask'].sum()/area)
                                            assert exposure>=THRESHOLDS['min_train_head_exposure'], ('EOT draw occluded; inspect geometry',r,exposure)
                            finally:task.scene.remove_actor(patch.actor)
                            restored=task.get_obs()
                            for n in CAMERAS:assert np.array_equal(clean['observation'][n]['rgb'],restored['observation'][n]['rgb'])
                            for r in selected:
                                output=Path(r['path']);output.parent.mkdir(parents=True,exist_ok=True)
                                tmp=output.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data[r['id']]);tmp.replace(output)
                            complete+=len(selected)
                            dump(root/'sites'/site/f'{split}_capture_progress.json',dict(completed=complete,total=len(rs),
                                seed=seed,query=query,light=light,elapsed_s=time.monotonic()-started,single_patch_per_scene=True))
                            print('[capture]',site,split,complete,len(rs),flush=True)
                        lighting.apply(LIGHTS['nominal'])
                    for action in actions:
                        task.take_action(action,action_type='qpos')
                        if task.eval_success or task.take_action_cnt>=task.step_lim:break
        finally:task.close_env()
    assert all(Path(r['path']).exists() and Path(r['clean_path']).exists() for r in rs)
    dump(root/'sites'/site/f'{split}_capture_complete.json',dict(complete=True,samples=len(rs),single_patch_per_scene=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--site',required=True)
    parser.add_argument('--split',choices=['train','validation','post'],required=True);args=parser.parse_args()
    capture(Path(args.root).resolve(),args.site,args.split)
