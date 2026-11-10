"""E21: constant-white texture vs digital white in the same E15 mask."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/wam-white-patch-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='white_patch_probe');args=ap.parse_args()
    import cv2,mujoco,yaml
    from PIL import Image
    from OpenGL import GL
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb
    import scene_patch as sp
    dest=D.OUT/args.out;dest.mkdir(exist_ok=True)
    old=np.load(D.OUT/'kv_render_probe/inputs.npz',allow_pickle=False)
    trajectory=D.OUT/('traj_'+D.TASK+'.npz');traj=np.load(trajectory,allow_pickle=False)
    states=[traj['f{:03d}__flatten'.format(traj['ts'].tolist().index(s))] for s in old['steps']]
    patch=np.load(D.OUT/'drift_patch_drift.npz',allow_pickle=True)
    cfg=yaml.safe_load((D.PROBE/'config/scene.yaml').read_text())
    cfg['patch']['size_wh']=[float(patch['patch_m'])]*2
    H=cv2.getPerspectiveTransform(patch['quad224'].astype(np.float32)[:,::-1],
                                 np.array([[0,0],[127,0],[127,127],[0,127]],np.float32))
    white=np.full((224,224,3),255,np.uint8)
    texture=cv2.warpPerspective(white,H,(128,128))[::-1].copy()
    assert texture.min()==texture.max()==255,'white export is not constant'
    random=np.random.RandomState(20260902).randint(0,256,(224,224,3)).astype(np.uint8)
    random_texture=cv2.warpPerspective(random,H,(128,128))[::-1].copy()
    texpath=dest/'white_texture.png';Image.fromarray(texture).save(texpath)
    suite=benchmark.get_benchmark_dict()['libero_goal']()
    task=next(suite.get_task(i) for i in range(suite.n_tasks)
              if D.pathlib.Path(suite.get_task(i).bddl_file).stem==D.TASK)
    bddl=D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg,patch['cell_world'],str(texpath)))
    arrays={key:old[key] for key in ['steps','state8','prompt','clean_base','clean_wrist']}
    arrays['mask']=old['drift__mask']
    arrays['inner_mask']=np.array([cv2.erode(m.astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
                                  for m in arrays['mask']])
    assert all(m.any() for m in arrays['inner_mask'])
    collected={key:[] for key in ['render_white','render_white_unlit','render_white_raw',
        'render_white_interior','render_white_unlit_interior','digital_white','digital_matched_rgb',
        'digital_white_interior','seg_mask256','coverage224','matched_rgb']}
    frames=[]
    try:
        env.seed(int(traj['shared_seed']));env.reset()
        sim=env.env.sim;ctx=sim._render_context_offscreen;model=sim.model._model
        ctx.gl_ctx.make_current()
        tid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TEXTURE,cfg['patch']['name']+'_tex')
        gid=sim.model.geom_name2id(cfg['patch']['name'])
        adr=int(model.tex_adr[tid]);length=128*128*3
        GL.glActiveTexture(GL.GL_TEXTURE0);GL.glBindTexture(GL.GL_TEXTURE_2D,int(ctx.con.texture[tid]))
        runtime={'mujoco_version':mujoco.__version__,'offSamples':int(ctx.con.offSamples),
                 'texture_min_filter':int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER)),
                 'texture_mag_filter':int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER)),
                 'framebuffer_srgb_enabled':bool(GL.glIsEnabled(GL.GL_FRAMEBUFFER_SRGB))}
        assert runtime['offSamples']==4 and runtime['texture_min_filter']==int(GL.GL_LINEAR_MIPMAP_LINEAR)
        def upload(tex):
            model.tex_data[adr:adr+length]=tex.reshape(-1);mujoco.mjr_uploadTexture(model,ctx.con,tid)
        def draw(unlit=False):
            ctx.gl_ctx.make_current();ctx.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
            ctx.cam.fixedcamid=sim.model.camera_name2id('agentview')
            mujoco.mjv_updateScene(model,sim.data._data,ctx.vopt,ctx.pert,ctx.cam,mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
            flags=ctx.scn.flags.copy()
            if unlit:
                ctx.scn.nlight=0
                for flag in ['mjRND_SHADOW','mjRND_REFLECTION','mjRND_SKYBOX','mjRND_HAZE','mjRND_FOG']:
                    if hasattr(mujoco.mjtRndFlag,flag):ctx.scn.flags[getattr(mujoco.mjtRndFlag,flag)]=0
                for geom in ctx.scn.geoms[:ctx.scn.ngeom]:
                    geom.emission=1. if geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and geom.objid==gid else 0.
            mujoco.mjr_render(mujoco.MjrRect(0,0,256,256),ctx.scn,ctx.con)
            raw=ctx.read_pixels(256,256);ctx.scn.flags[:]=flags
            return np.ascontiguousarray(raw[::-1,::-1])
        for k,state in enumerate(states):
            env.regenerate_obs_from_state(state)
            upload(random_texture)
            assert np.array_equal(resize_rgb(draw(),224),old['drift__random_render_base'][k]),(k,'E15 anchor differs')
            upload(texture);lit_raw=draw();unlit_raw=draw(True)
            if k==0:
                assert np.array_equal(lit_raw,draw()),'white repeat differs'
                arrays['lit_native256_first']=lit_raw;arrays['unlit_native256_first']=unlit_raw
            lit,unlit=resize_rgb(lit_raw,224),resize_rgb(unlit_raw,224)
            # Actual geometry visibility is diagnostic only. All principal
            # model comparisons retain the very same original E15 mask.
            ctx.render(256,256,camera_id=sim.model.camera_name2id('agentview'),segmentation=True)
            seg=np.ascontiguousarray(ctx.read_pixels(256,256,segmentation=True)[::-1,::-1])
            visible=(seg[...,0]==mujoco.mjtObj.mjOBJ_GEOM)&(seg[...,1]==gid)
            assert visible.any()
            coverage=resize_rgb(np.repeat((visible.astype(np.uint8)*255)[...,None],3,axis=-1),224)[...,0]
            m=arrays['mask'][k];inner=arrays['inner_mask'][k];clean=arrays['clean_base'][k]
            rgb=np.rint(lit[inner].astype(float).mean(0)).astype(np.uint8)
            images={'render_white':np.where(m[...,None],lit,clean),
                    'render_white_unlit':np.where(m[...,None],unlit,clean),
                    'render_white_raw':lit,
                    'render_white_interior':np.where(inner[...,None],lit,clean),
                    'render_white_unlit_interior':np.where(inner[...,None],unlit,clean),
                    'digital_white':np.where(m[...,None],255,clean),
                    'digital_white_interior':np.where(inner[...,None],255,clean),
                    'digital_matched_rgb':np.where(m[...,None],rgb,clean)}
            for key,image in images.items():collected[key].append(image.astype(np.uint8))
            collected['seg_mask256'].append(visible);collected['coverage224'].append(coverage)
            collected['matched_rgb'].append(rgb)
            geometry_mask=coverage>=128
            error=np.abs(lit.astype(float)-255);unlit_error=np.abs(unlit.astype(float)-255)
            frame={'step':int(old['steps'][k]),'mask_px':int(m.sum()),'interior_px':int(inner.sum()),
                   'segmentation_IoU':float((geometry_mask&m).sum()/(geometry_mask|m).sum()),
                   'lit_interior_rgb_mean':lit[inner].mean(0).tolist(),
                   'lit_interior_rgb_min':lit[inner].min(0).tolist(),'lit_interior_rgb_max':lit[inner].max(0).tolist(),
                   'lit_MAE_all_mask_u8':float(error[m].mean()),'lit_MAE_interior_u8':float(error[inner].mean()),
                   'unlit_MAE_all_mask_u8':float(unlit_error[m].mean()),
                   'unlit_MAE_interior_u8':float(unlit_error[inner].mean()),
                   'unlit_interior_max_error_u8':int(unlit_error[inner].max()),
                   'raw_outside_mask_MAE_u8':float(np.abs(lit.astype(float)-clean)[~m].mean())}
            assert frame['unlit_interior_max_error_u8']<=1,(k,frame)
            frames.append(frame)
            print('[capture]',frame['step'],'lit interior RGB',frame['lit_interior_rgb_mean'],
                  'unlit interior max error',frame['unlit_interior_max_error_u8'],flush=True)
        del draw,upload,ctx,sim,model
    finally:env.close()
    arrays.update({key:np.array(value) for key,value in collected.items()})
    conditions=['digital_white','render_white','render_white_unlit','digital_matched_rgb','render_white_raw',
                'digital_white_interior','render_white_interior','render_white_unlit_interior']
    arrays['conditions']=np.array(conditions)
    np.savez_compressed(dest/'inputs.npz',**arrays)
    meta={'complete':True,'frames':len(states),'conditions':conditions,'runtime':runtime,'frame_metrics':frames,
        'checks':{'constant_exported_texture255':True,'E15_random_render_anchor_equal':True,
                  'repeat_white_equal':True,'unlit_interior_within1_u8_of255':True,'same_mask_and_clean_background':True},
        'source_inputs_sha256':hashlib.sha256((D.OUT/'kv_render_probe/inputs.npz').read_bytes()).hexdigest(),
        'trajectory_sha256':hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        'notes':'White P224 exports to exactly constant-white texture128; 26cm same geometry, seed and 16 states. '
                'Principal comparisons compose in the same E15 mask; interior controls share eroded4 mask. '
                'Raw render is separate. Segmentation/coverage are geometry diagnostics, not a replacement model mask. '
                'Matched RGB is per-state lit interior mean rounded to uint8, not a full photometric calibration. '
                'Unlit keeps white texture/material, zero lights, emission=1 for patch, no shadows/reflections. '
                'No model, training or rollout in capture.'}
    (dest/'capture.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    print('[done]',dest,flush=True)


if __name__=='__main__':main()
