"""E26: per-state, per-camera analytic geometry and independent photometric anchors."""
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/wam-dual-numba')
import json, hashlib
import drift_eval_digital as D
import numpy as np


def homography(sim,gid,camera,size=256):
    # PNG UV convention independently established by E25: corners 3,2,1,0.
    half=np.asarray(sim.model.geom_size[gid]); rot=np.asarray(sim.data.geom_xmat[gid]).reshape(3,3)
    pos=np.asarray(sim.data.geom_xpos[gid]); cid=sim.model.camera_name2id(camera)
    origin=np.array([-half[0],half[1],half[2]])@rot.T+pos
    U=np.array([2*half[0]/128,0,0])@rot.T
    V=np.array([0,-2*half[1]/128,0])@rot.T
    R=np.asarray(sim.data.cam_xmat[cid]).reshape(3,3)
    C=np.stack([U,V,origin+.5*U+.5*V-np.asarray(sim.data.cam_xpos[cid])],axis=1)
    C=R.T@C
    f=.5*size/np.tan(np.deg2rad(float(sim.model.cam_fovy[cid]))/2); c=(size-1)/2
    P=np.array([[-f,0,-c],[0,-f,-c],[0,0,-1.]])@C
    assert abs(np.linalg.det(P))>1e-12
    return np.linalg.inv(P)


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--calibration',default=str(D.OUT/'render_proxy_scale/cal_13_101/calibration.npz'))
    ap.add_argument('--out',default=str(D.OUT/'dual_view/calibration'))
    args=ap.parse_args()
    import cv2,mujoco,yaml
    from PIL import Image
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb
    import scene_patch as sp
    dest=D.pathlib.Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'base.npz').exists()
    source=D.pathlib.Path(args.calibration)
    old=np.load(source,allow_pickle=False)
    keys=['sim_states','ep_id','env_seed','env_step','train_mask','validation_mask','clean_base','clean_wrist','state8','prompt','cell_world','cell_xy','patch_m']
    common={k:old[k] for k in keys}
    cfg=yaml.safe_load((D.PROBE/'config/scene.yaml').read_text());cfg['patch']['size_wh']=[float(old['patch_m'])]*2
    random=old['random__texture'];x=np.rint(np.linspace(0,255,128)).astype(np.uint8)
    ramp=np.empty_like(random);ramp[...,0]=x[None,:];ramp[...,1]=x[:,None];ramp[...,2]=255
    low=np.random.RandomState(20261009).randint(0,256,(32,32,3)).astype(np.uint8)
    band=np.array(Image.fromarray(low).resize((128,128),Image.BILINEAR))
    textures={'black256':np.zeros_like(random),'white256':np.full_like(random,255),'random256':random,
              'check256':band,'ramp256':ramp,'unlit_white256':np.full_like(random,255)}
    Image.fromarray(random).save(dest/'random.png');Image.fromarray(band).save(dest/'check.png')
    fixtures=json.loads((D.OUT/'render_proxy_eot/data/manifest.json').read_text())['fixtures'];fixtures={r['env_seed']:r for r in fixtures}
    suite=benchmark.get_benchmark_dict()['libero_goal']();task=next(suite.get_task(i) for i in range(suite.n_tasks) if D.pathlib.Path(suite.get_task(i).bddl_file).stem==D.TASK)
    bddl=D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg,common['cell_world'],str(dest/'random.png')))
    cams={'base':'agentview','wrist':'robot0_eye_in_hand'}
    buffers={v:{k:[] for k in list(textures)+['mask','image256_to_texture_pixel_H']} for v in cams}
    audits=[];previous=None
    try:
        for k,state in enumerate(common['sim_states']):
            seed=int(common['env_seed'][k])
            if seed!=previous:
                env.seed(seed);env.reset();previous=seed
                fixture=fixtures[seed];n=len(fixture['body_pos'])
                assert np.array_equal(env.env.sim.model.body_pos[:n],fixture['body_pos'])
                assert np.array_equal(env.env.sim.model.body_quat[:n],fixture['body_quat'])
            env.regenerate_obs_from_state(state);sim=env.env.sim;ctx=sim._render_context_offscreen;model=sim.model._model
            ctx.gl_ctx.make_current();gid=sim.model.geom_name2id(cfg['patch']['name'])
            tid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TEXTURE,cfg['patch']['name']+'_tex');adr=int(model.tex_adr[tid])
            for view,camera in cams.items():
                cid=sim.model.camera_name2id(camera);buf=buffers[view]
                H=homography(sim,gid,camera);buf['image256_to_texture_pixel_H'].append(H)
                ctx.render(256,256,camera_id=cid,segmentation=True)
                seg=np.ascontiguousarray(ctx.read_pixels(256,256,segmentation=True)[::-1,::-1])
                visible=(seg[...,0]==mujoco.mjtObj.mjOBJ_GEOM)&(seg[...,1]==gid)
                mask=resize_rgb(np.repeat((visible.astype(np.uint8)*255)[...,None],3,-1),224)[...,0]>=128;buf['mask'].append(mask)
                for name,tex in textures.items():
                    model.tex_data[adr:adr+128*128*3]=tex.ravel();mujoco.mjr_uploadTexture(model,ctx.con,tid)
                    ctx.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED;ctx.cam.fixedcamid=cid
                    mujoco.mjv_updateScene(model,sim.data._data,ctx.vopt,ctx.pert,ctx.cam,mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
                    flags=ctx.scn.flags.copy()
                    if name in ['ramp256','unlit_white256']:
                        ctx.scn.nlight=0
                        for flag in ['mjRND_SHADOW','mjRND_REFLECTION','mjRND_SKYBOX','mjRND_HAZE','mjRND_FOG']:
                            if hasattr(mujoco.mjtRndFlag,flag):ctx.scn.flags[getattr(mujoco.mjtRndFlag,flag)]=0
                        for g in ctx.scn.geoms[:ctx.scn.ngeom]:g.emission=1. if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid==gid else 0.
                    mujoco.mjr_render(mujoco.MjrRect(0,0,256,256),ctx.scn,ctx.con)
                    image=np.ascontiguousarray(ctx.read_pixels(256,256)[::-1,::-1]);ctx.scn.flags[:]=flags;buf[name].append(image)
                    if view=='base' and name=='random256':assert np.array_equal(image,old['random256'][k]),('base replay',k)
                valid=cv2.erode((visible & (buf['unlit_white256'][-1].min(-1)==255)).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
                yy,xx=np.where(valid);error=None
                if len(yy):
                    p=np.c_[xx,yy,np.ones(len(xx))]@H.T;uv=p[:,:2]/p[:,2:]
                    measured=buf['ramp256'][-1][valid,:2].astype(float);measured=np.stack([np.interp(measured[:,j],x,np.arange(128)) for j in range(2)],-1)
                    error=float(np.sqrt(((uv-measured)**2).mean()))
                audits.append({'frame':k,'view':view,'visible224':int(mask.sum()),'uv_interior_points':len(yy),'UV_RMSE_texels':error})
            del ctx,sim,model
            if k%20==0:print('[capture]',k,len(common['sim_states']),audits[-1],flush=True)
    finally:env.close()
    for view in cams:
        np.savez_compressed(dest/(view+'.npz'),**common,**{k:np.array(v) for k,v in buffers[view].items()},random__texture=random,check__texture=band,camera=cams[view])
    report={'complete':True,'frames':len(common['sim_states']),'views':cams,'records':audits,'base_random_all187_exact':True,'fixtures_exact':True,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'geometry':'Analytic camera/box float64, PNG UV corners3,2,1,0. Each wrist frame has its own H. No learned textures used.'}
    (dest/'capture.json').write_text(json.dumps(report,indent=2)+'\n');print('[done]',flush=True)

if __name__=='__main__':main()
