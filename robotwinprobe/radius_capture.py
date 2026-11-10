"""Exact per-draw real-render calibration, grouped by clean state for replay efficiency."""
import argparse, random, time
from pathlib import Path
import numpy as np
import bootstrap as B
B.simulator_paths()
from radius_plan import records, REFERENCE, dump
from render_geometry import CAMERAS, Patch, camera_objects, patterns
from scene_smoke import make_task


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--radius',type=int,choices=[1,3,5],required=True)
    ap.add_argument('--split',choices=['train','validation'],required=True)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args();root=Path(args.root).resolve()
    import sapien
    from PIL import Image
    rs=records(root,args.radius,args.split)
    if args.limit:rs=rs[:args.limit]
    pending=[r for r in rs if not Path(r['path']).exists()]
    if not pending:return
    gray={label:np.full((128,128,3),value,np.uint8) for label,value in
          [('black',0),('gray64',64),('gray',128),('gray192',192),('white',255)]}
    textures=dict(gray)
    if args.split=='validation':
        textures.update({k:patterns()[k] for k in ['random','checker']})
        textures['baseline']=np.array(Image.open(REFERENCE/'train_proxy_full/texture_0500.png').convert('RGB'))
    started=time.monotonic();done=len(rs)-len(pending)
    work=root/'capture'/f'r{args.radius:02d}'/('assets_'+args.split);work.mkdir(parents=True,exist_ok=True)
    for seed in sorted({r['seed'] for r in pending}):
        these=[r for r in pending if r['seed']==seed]
        random.seed(seed);task,_,_=make_task('place_container_plate',seed,work)
        try:
            with np.load(REFERENCE/'clean'/f'seed{seed}.npz') as ref:
                last=max(r['query'] for r in these)
                for query,actions in enumerate(ref['actions']):
                    if query>last:break
                    group=[r for r in these if r['query']==query]
                    if group:
                        clean=task.get_obs();state=clean['joint_action']['vector'].astype(np.float32)
                        assert np.array_equal(state,ref['states14'][query]),(seed,query,'state')
                        for n in CAMERAS:
                            assert np.array_equal(clean['observation'][n]['rgb'],ref[n][query]),(seed,query,n,'RGB')
                        cams=camera_objects(task)
                        data={r['id']:{'center_m':np.array(r['center']),'size_m':np.array(.20),
                            'sample_id':np.array(r['id']),'frame':np.array(r['frame']),
                            'seed':np.array(seed),'query':np.array(query),'state14':state.copy()} for r in group}
                        patch=Patch(task,work)
                        for label,rgb in textures.items():
                            # One material binding per gray level; world pose varies continuously.
                            patch.texture(rgb)
                            for r in group:
                                patch.center=np.array(r['center']);patch.actor.set_pose(sapien.Pose(patch.center))
                                obs=task.get_obs()
                                assert np.array_equal(state,obs['joint_action']['vector'].astype(np.float32))
                                d=data[r['id']]
                                for n in CAMERAS:
                                    d[n+'_'+label]=obs['observation'][n]['rgb'].copy()
                                    if label=='black':
                                        d[n+'_mask']=(cams[n].get_picture('Segmentation')[...,1]==patch.actor.per_scene_id)
                                        d[n+'_intrinsic']=cams[n].get_intrinsic_matrix()
                                        d[n+'_extrinsic']=cams[n].get_extrinsic_matrix()
                                    if label=='white':
                                        assert np.array_equal(d[n+'_mask'],cams[n].get_picture('Segmentation')[...,1]==patch.actor.per_scene_id)
                        task.scene.remove_actor(patch.actor);restored=task.get_obs()
                        for n in CAMERAS:assert np.array_equal(clean['observation'][n]['rgb'],restored['observation'][n]['rgb'])
                        for r in group:
                            output=Path(r['path']);output.parent.mkdir(parents=True,exist_ok=True)
                            tmp=output.with_suffix('.tmp.npz');np.savez_compressed(tmp,**data[r['id']]);tmp.replace(output)
                        done+=len(group)
                        progress={'radius_cm':args.radius,'split':args.split,'seed':seed,'query':query,
                            'completed_samples':done,'total_samples':len(rs),'elapsed_s':time.monotonic()-started}
                        dump(root/'capture'/f'r{args.radius:02d}'/f'{args.split}_progress.json',progress)
                        print('[capture]',progress,flush=True)
                    for action in actions:
                        task.take_action(action,action_type='qpos')
                        if task.eval_success or task.take_action_cnt>=task.step_lim:break
        finally:task.close_env()
    assert all(Path(r['path']).exists() for r in rs)
    dump(root/'capture'/f'r{args.radius:02d}'/f'{args.split}_complete.json',
         {'complete':True,'samples':len(rs),'limited_smoke':bool(args.limit)})


if __name__=='__main__':main()
