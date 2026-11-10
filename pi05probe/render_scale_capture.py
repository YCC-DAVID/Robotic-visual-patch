"""Geometry-specific calibration on the unchanged E22 clean state pool.

Only ramps, uniform colors and an independent random texture calibrate the
proxy. Segmentation and fixture checks are repeated for every new geometry.
"""
import argparse
import hashlib
import json
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-render-scale-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cm', type=float, required=True)
    ap.add_argument('--position', choices=['101', '161'], default='101')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import cv2, mujoco, yaml
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb
    import scene_patch as sp
    dest = D.OUT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest/'calibration.npz').exists()
    source = D.OUT/'render_proxy_eot/data/pool.npz'
    archive = np.load(source, allow_pickle=False)
    arrays = {k: archive[k] for k in archive.files if k not in ['masks', 'munion', 'quad224', 'cell_world', 'cell_xy', 'patch_m']}
    arrays.update(clean_base=arrays['clean_img224'], clean_wrist=arrays['clean_wrist224'],
                  state8=arrays['clean_state8'], steps=arrays['env_step'])
    geometry = np.load(D.OUT/('drift_patch_drift.npz' if args.position=='101' else 'drift_patch_ctrl161_13.npz'), allow_pickle=False)
    arrays.update(cell_world=geometry['cell_world'], cell_xy=geometry['cell_xy'], patch_m=args.cm/100)
    cfg = yaml.safe_load((D.PROBE/'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [args.cm/100]*2
    random = np.random.RandomState(20261007).randint(0,256,(128,128,3)).astype(np.uint8)
    arrays['random__texture'] = random
    texpath=dest/'random_texture.png'; Image.fromarray(random).save(texpath)
    x = np.rint(np.linspace(0,255,128)).astype(np.uint8)
    ramp=np.empty((128,128,3),np.uint8)
    ramp[...,0]=x[None,:]; ramp[...,1]=x[:,None]; ramp[...,2]=255
    suite=benchmark.get_benchmark_dict()['libero_goal']()
    task=next(suite.get_task(i) for i in range(suite.n_tasks) if D.pathlib.Path(suite.get_task(i).bddl_file).stem==D.TASK)
    bddl=D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    fixtures=json.loads((source.parent/'manifest.json').read_text())['fixtures']
    fixtures={r['env_seed']:r for r in fixtures}
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg, arrays['cell_world'],str(texpath)))
    buffers={k:[] for k in ['black256','white256','unlit_white256','ramp256','gray128_256','random256','mask']}
    previous=None
    checks={'fixture_positions_exact':True,'fixture_quaternions_exact':True,'far_background_exact':True,'random_repeat_exact':True}
    try:
        for k,state in enumerate(arrays['sim_states']):
            seed=int(arrays['env_seed'][k])
            if seed!=previous:
                env.seed(seed);env.reset();previous=seed
                model=env.env.sim.model
                fixture=fixtures[seed]
                n=len(fixture['body_pos'])
                assert np.array_equal(model.body_pos[:n],np.array(fixture['body_pos']))
                assert np.array_equal(model.body_quat[:n],np.array(fixture['body_quat']))
                del model
            env.regenerate_obs_from_state(state)
            sim=env.env.sim;ctx=sim._render_context_offscreen;model=sim.model._model
            ctx.gl_ctx.make_current()
            gid=sim.model.geom_name2id(cfg['patch']['name'])
            cid=sim.model.camera_name2id('agentview')
            tid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TEXTURE,cfg['patch']['name']+'_tex')
            adr=int(model.tex_adr[tid])
            assert int(ctx.con.offSamples)==4
            ctx.render(256,256,camera_id=cid,segmentation=True)
            seg=np.ascontiguousarray(ctx.read_pixels(256,256,segmentation=True)[::-1,::-1])
            visible=(seg[...,0]==mujoco.mjtObj.mjOBJ_GEOM)&(seg[...,1]==gid)
            mask=resize_rgb(np.repeat((visible.astype(np.uint8)*255)[...,None],3,-1),224)[...,0]>=128
            buffers['mask'].append(mask)
            def draw(texture,unlit=False):
                model.tex_data[adr:adr+128*128*3]=texture.reshape(-1)
                mujoco.mjr_uploadTexture(model,ctx.con,tid)
                ctx.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED;ctx.cam.fixedcamid=cid
                mujoco.mjv_updateScene(model,sim.data._data,ctx.vopt,ctx.pert,ctx.cam,mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
                flags=ctx.scn.flags.copy()
                if unlit:
                    ctx.scn.nlight=0
                    for flag in ['mjRND_SHADOW','mjRND_REFLECTION','mjRND_SKYBOX','mjRND_HAZE','mjRND_FOG']:
                        if hasattr(mujoco.mjtRndFlag,flag):ctx.scn.flags[getattr(mujoco.mjtRndFlag,flag)]=0
                    for g in ctx.scn.geoms[:ctx.scn.ngeom]:
                        g.emission=1. if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid==gid else 0.
                mujoco.mjr_render(mujoco.MjrRect(0,0,256,256),ctx.scn,ctx.con)
                image=np.ascontiguousarray(ctx.read_pixels(256,256)[::-1,::-1]);ctx.scn.flags[:]=flags
                return image
            for name,texture,unlit in [('random256',random,False),('black256',np.zeros_like(random),False),
                 ('white256',np.full_like(random,255),False),('unlit_white256',np.full_like(random,255),True),
                 ('ramp256',ramp,True),('gray128_256',np.full_like(random,128),False)]:
                image=draw(texture,unlit);buffers[name].append(image)
                if k==0 and name=='random256':assert np.array_equal(image,draw(texture,unlit))
            far=~cv2.dilate(mask.astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool)
            black224=resize_rgb(buffers['black256'][-1],224)
            assert np.array_equal(black224[far],arrays['clean_base'][k][far]),('background',k)
            del draw,ctx,sim,model
            if k%30==0:print('[capture]',args.cm,args.position,k,len(arrays['sim_states']),flush=True)
    finally:env.close()
    arrays.update({k:np.array(v) for k,v in buffers.items()})
    a,b=[],[]
    train=np.flatnonzero(arrays['train_mask'])[:8]
    counts=[]
    for k in train:
        valid=cv2.erode((arrays['unlit_white256'][k].min(-1)==255).astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
        yy,xx=np.where(valid);counts.append(len(yy))
        if len(yy)<8:continue
        uv=arrays['ramp256'][k][valid,:2].astype(float)
        uv=np.stack([np.interp(uv[:,j],x,np.arange(128)) for j in range(2)],-1)
        a.append(np.stack([xx,yy],-1).astype(float));b.append(uv)
    assert a and sum(len(t) for t in a)>=32,counts
    a,b=np.concatenate(a),np.concatenate(b)
    H,_=cv2.findHomography(a,b,method=0)
    ph=np.concatenate([a,np.ones((len(a),1))],-1)@H.T
    residual=ph[:,:2]/ph[:,2:]-b
    arrays['image256_to_texture_pixel_H']=H
    np.savez_compressed(dest/'calibration.npz',**arrays)
    metadata={'complete':True,'cm':args.cm,'position':args.position,'cell_world':arrays['cell_world'].tolist(),
        'frames':len(arrays['sim_states']),'checks':checks,'source_pool_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'geometry_fit_frames':train.tolist(),'geometry_fit_points':counts,
        'homography_RMSE_texture_pixels':float(np.sqrt((residual**2).mean())),
        'visible_pixels224_train_median':float(np.median(arrays['mask'][arrays['train_mask']].sum((1,2)))),
        'visible_pixels224_validation_median':float(np.median(arrays['mask'][arrays['validation_mask']].sum((1,2)))),
        'random_texture_seed':20261007,'notes':'Original E22 clean state pool, per-geometry segmentation/photometry/UV calibration. Geometry is never fitted to learned textures.'}
    (dest/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('[done]',json.dumps(metadata),flush=True)


if __name__=='__main__':main()
