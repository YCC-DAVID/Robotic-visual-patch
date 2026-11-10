"""Independent analytic camera/plane projection, high-resolution real checks."""
import json
import os
import xml.etree.ElementTree as ET
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-camera-checker-numba')
import drift_eval_digital as D
import numpy as np
from perspective_geometry import box_top_projection, transform_xy


def main():
    import cv2, mujoco, yaml
    from PIL import Image
    from OpenGL import GL
    import scene_patch as sp
    from drift_resize_order import resize_rgb
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    dest = D.OUT/'camera_checker'
    dest.mkdir(exist_ok=True)
    assert not (dest/'inputs.npz').exists()
    old = np.load(D.OUT/'color_checker_probe/inputs.npz', allow_pickle=False)
    cal = np.load(D.OUT/'render_proxy_eot/calibration_e15/calibration.npz', allow_pickle=False)
    cfg = yaml.safe_load((D.PROBE/'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(old['patch_m'])]*2
    sizes = [2,4,8,16]
    yy,xx = np.mgrid[:128,:128]
    patterns = {'cell'+str(s):old['palette'][(yy//s+2*(xx//s))%5] for s in sizes}
    selected = [0,7,15]
    arrays = {'steps':old['steps'][selected], 'patterns':np.array(list(patterns)),
              'sizes':np.array(sizes), 'patch_m':old['patch_m'], 'mask224_legacy':old['mask'][selected]}
    for name,q in patterns.items():
        arrays[name+'__Q']=q
        Image.fromarray(q).save(dest/(name+'_Q.png'))
    suite=benchmark.get_benchmark_dict()['libero_goal']()
    task=next(suite.get_task(i) for i in range(suite.n_tasks) if D.pathlib.Path(suite.get_task(i).bddl_file).stem==D.TASK)
    bddl=D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    results=[]; geometry_report=None
    for aa in [0,4]:
        inject=sp.make_xml_processor(cfg,old['cell_world'],str(dest/'cell8_Q.png'))
        def xml_processor(xml):
            root=ET.fromstring(inject(xml));visual=root.find('visual')
            if visual is None:visual=ET.SubElement(root,'visual')
            quality=visual.find('quality')
            if quality is None:quality=ET.SubElement(visual,'quality')
            quality.set('offsamples',str(aa))
            global_visual=visual.find('global')
            if global_visual is None:global_visual=ET.SubElement(visual,'global')
            global_visual.set('offwidth','1024');global_visual.set('offheight','1024')
            return ET.tostring(root,encoding='unicode')
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
        env.env.set_xml_processor(xml_processor)
        captures={}
        try:
            env.seed(int(old['environment_seed']));env.reset()
            sim=env.env.sim;model=sim.model._model;ctx=sim._render_context_offscreen
            ctx.gl_ctx.make_current();assert int(ctx.con.offSamples)==aa
            assert ctx.con.offWidth>=1024 and ctx.con.offHeight>=1024
            fixture=np.r_[model.body_pos.ravel(),model.body_quat.ravel()]
            assert np.array_equal(fixture,old['fixture_pose_aa'+str(aa)])
            cid=sim.model.camera_name2id('agentview');gid=sim.model.geom_name2id(cfg['patch']['name'])
            tid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TEXTURE,cfg['patch']['name']+'_tex')
            adr=int(model.tex_adr[tid]);glid=int(ctx.con.texture[tid])
            GL.glBindTexture(GL.GL_TEXTURE_2D,glid)
            default_min=int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER))
            default_mag=int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER))
            def upload(q):
                model.tex_data[adr:adr+q.size]=q.ravel();mujoco.mjr_uploadTexture(model,ctx.con,tid)
            def draw(size,unlit=False):
                ctx.gl_ctx.make_current();GL.glBindTexture(GL.GL_TEXTURE_2D,glid)
                GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,GL.GL_NEAREST if aa==0 else default_min)
                GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,GL.GL_NEAREST if aa==0 else default_mag)
                ctx.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED;ctx.cam.fixedcamid=cid
                mujoco.mjv_updateScene(model,sim.data._data,ctx.vopt,ctx.pert,ctx.cam,mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
                flags=ctx.scn.flags.copy()
                if unlit:
                    ctx.scn.nlight=0
                    for f in ['mjRND_SHADOW','mjRND_REFLECTION','mjRND_SKYBOX','mjRND_HAZE','mjRND_FOG']:
                        if hasattr(mujoco.mjtRndFlag,f):ctx.scn.flags[getattr(mujoco.mjtRndFlag,f)]=0
                    for g in ctx.scn.geoms[:ctx.scn.ngeom]:
                        g.emission=float(g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid==gid)
                mujoco.mjr_render(mujoco.MjrRect(0,0,size,size),ctx.scn,ctx.con)
                image=np.ascontiguousarray(ctx.read_pixels(size,size)[::-1,::-1]);ctx.scn.flags[:]=flags
                return image
            for j,k in enumerate(selected):
                env.regenerate_obs_from_state(old['sim_states'][k])
                if aa==0:
                    H1024,geometry=box_top_projection(sim,cfg['patch']['name'],1024,(3,2,1,0))
                    H224,_=box_top_projection(sim,cfg['patch']['name'],224,(3,2,1,0))
                    H256,_=box_top_projection(sim,cfg['patch']['name'],256,(3,2,1,0))
                    arrays['Q_to_image1024_H']=H1024;arrays['Q_to_image224_H']=H224
                    valid=cv2.erode(old['inner_mask'][k].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
                    y,x=np.where(valid);xy256=(np.c_[x,y]+.5)*256/224-.5
                    uv_a=transform_xy(np.linalg.inv(H256),xy256)
                    uv_b=transform_xy(cal['image256_to_texture_pixel_H'],xy256)
                    results.append({'step':int(old['steps'][k]),'analytic_vs_ramp_uv_RMSE_texels':float(np.sqrt(((uv_a-uv_b)**2).mean()))})
                    upload(np.full((128,128,3),255,np.uint8));white=draw(1024,True)
                    visible=white.min(-1)==255
                    captures.setdefault('visible1024',[]).append(visible)
                    geometry_report=geometry
                for name,q in patterns.items():
                    upload(q)
                    high=draw(1024,aa==0)
                    captures.setdefault(name+('__minimal1024' if aa==0 else '__lit1024'),[]).append(high)
                    if aa==0:
                        predicted=cv2.warpPerspective(q,H1024,(1024,1024),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REPLICATE)
                        captures.setdefault(name+'__projected1024',[]).append(predicted)
                    else:
                        native=draw(256,False)
                        captures.setdefault(name+'__lit256',[]).append(native)
                        captures.setdefault(name+'__model224',[]).append(resize_rgb(native,224))
                    assert np.array_equal(high,draw(1024,aa==0))
                print('[camera checker]',aa,int(old['steps'][k]),flush=True)
            arrays.update({key:np.array(value) for key,value in captures.items()})
            del upload,draw,ctx,model,sim
        finally:env.close()
    measurements=[]
    for name,s in zip(patterns,sizes):
        for j,step in enumerate(arrays['steps']):
            inner=cv2.erode(arrays['visible1024'][j].astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
            a=arrays[name+'__projected1024'][j];b=arrays[name+'__minimal1024'][j]
            measurements.append({'pattern':name,'step':int(step),'interior_MAE_u8':float(np.abs(a.astype(float)-b)[inner].mean()),
                                 'different_pixel_fraction':float((a!=b).any(-1)[inner].mean())})
        H=arrays['Q_to_image224_H']
        far=transform_xy(H,[[0,64-s/2],[0,64+s/2]])
        near=transform_xy(H,[[127,64-s/2],[127,64+s/2]])
        geometry_report.setdefault('cell_widths_model224',[]).append({'texels':s,'far_width_px':float(np.linalg.norm(far[1]-far[0])),
            'near_width_px':float(np.linalg.norm(near[1]-near[0]))})
    np.savez_compressed(dest/'inputs.npz',**arrays)
    report={'complete':True,'geometry':geometry_report,'geometry_checks':results,'pixel_checks':measurements,
            'new_real_checker_images':36,'steps':arrays['steps'].tolist(),
            'notes':'Camera fovy/extrinsics and actual box top corners determine homography. UV orientation fixed [3,2,1,0], '
                    'verified earlier by ramp. No fitted homography used for forward reference. '
                    'Native1024 verifies geometry at resolvable cell widths. Model input remains production256->224. '
                    'Visible support for high-resolution geometry check comes from unlit white, AA0. '
                    'Three correlated states, no model/closed-loop result.'}
    (dest/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print('[done]',json.dumps(report),flush=True)


if __name__=='__main__':main()
