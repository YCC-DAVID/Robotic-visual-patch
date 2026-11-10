"""E18 paired renderer interventions and same-pattern digital degradations.

Fixed E15 states, 26cm patch, full/band32/same random. Renderer interventions
are composited through the SAME hard E15 mask into clean E15 images, so scene
lighting/AA outside the patch cannot masquerade as an attack. Keep raw default
render as an anchor and a masked default to quantify this compositing control.
"""
import argparse
import hashlib
import json
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-render-ablation-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='render_ablation')
    ap.add_argument('--frames', type=int, default=16)
    args = ap.parse_args()
    import cv2
    import mujoco
    import yaml
    from OpenGL import GL
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb
    import scene_patch as sp
    import xml.etree.ElementTree as ET

    dest = D.OUT/args.out
    dest.mkdir(exist_ok=True)
    old = np.load(D.OUT/'kv_render_probe/inputs.npz', allow_pickle=False)
    traj_path = D.OUT/('traj_'+D.TASK+'.npz')
    traj = np.load(traj_path, allow_pickle=False)
    steps = old['steps'][:args.frames]
    indices = [traj['ts'].tolist().index(s) for s in steps]
    states = [traj['f{:03d}__flatten'.format(i)] for i in indices]
    n = len(states)
    p = np.load(D.OUT/'drift_patch_drift.npz', allow_pickle=True)
    cfg = yaml.safe_load((D.PROBE/'config/scene.yaml').read_text())
    cfg['patch']['size_wh'] = [float(p['patch_m'])]*2
    H = cv2.getPerspectiveTransform(p['quad224'].astype(np.float32)[:,::-1],
                                   np.array([[0,0],[127,0],[127,127],[0,127]],np.float32))
    random = np.random.RandomState(20260902).randint(0,256,(224,224,3)).astype(np.uint8)
    patterns = {'random':random}
    for name, tag in [('full','drift'),('band32','band32')]:
        data = np.load(D.OUT/('drift_patch_'+tag+'.npz'), allow_pickle=True)
        assert np.array_equal(p['quad224'],data['quad224'])
        assert np.array_equal(old['drift__mask'],old[tag+'__mask'])
        patterns[name] = np.clip(data['P'].transpose(1,2,0)*255,0,255).astype(np.uint8)
    textures = {k:cv2.warpPerspective(v,H,(128,128))[::-1].copy() for k,v in patterns.items()}
    texpath = dest/'initial_texture.png'
    Image.fromarray(textures['random']).save(texpath)
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem==D.TASK)
    bddl = D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    arrays = {'steps':steps, 'state8':old['state8'][:n], 'prompt':old['prompt'],
              'clean_base':old['clean_base'][:n], 'clean_wrist':old['clean_wrist'][:n],
              'mask':old['drift__mask'][:n], 'patterns':np.array(list(patterns))}
    checks, runtime = {}, {}
    variants = [
        ('render_roi','default',False,256,False,4),
        ('render_nomipmap','linear',False,256,False,4),
        ('render_nearest','nearest',False,256,False,4),
        ('render_unlit','default',True,256,False,4),
        ('render_unlit_nomipmap','linear',True,256,False,4),
        ('render_unlit_nearest','nearest',True,256,False,4),
        ('render_resize_nearest','default',False,256,True,4),
        ('render_native224','default',False,224,False,4),
        ('render_noaa','default',False,256,False,0),
        ('render_unlit_nearest_noaa','nearest',True,256,False,0),
        ('render_unlit_nearest_noaa_native224','nearest',True,224,False,0)]
    for aa in [4,0]:
        inject = sp.make_xml_processor(cfg,p['cell_world'],str(texpath))
        def xml_processor(xml):
            root = ET.fromstring(inject(xml))
            visual = root.find('visual')
            if visual is None: visual = ET.SubElement(root,'visual')
            quality = visual.find('quality')
            if quality is None: quality = ET.SubElement(visual,'quality')
            # Default is checked against E15 exact pixels below.
            quality.set('offsamples',str(aa))
            return ET.tostring(root,encoding='unicode')
        env = OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
        env.env.set_xml_processor(xml_processor)
        try:
            env.seed(int(traj['shared_seed'])); env.reset()
            sim = env.env.sim
            ctx = sim._render_context_offscreen
            ctx.gl_ctx.make_current()
            model = sim.model._model
            tid = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TEXTURE,cfg['patch']['name']+'_tex')
            gid = sim.model.geom_name2id(cfg['patch']['name'])
            adr = int(model.tex_adr[tid]); length = 128*128*3
            assert int(model.tex_nchannel[tid])==3 and int(model.tex_width[tid])==128
            glid = int(ctx.con.texture[tid])
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D,glid)
            default_min = int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER))
            default_mag = int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER))
            runtime[str(aa)] = {'mujoco_version':mujoco.__version__, 'offSamples':int(ctx.con.offSamples),
                'texture_min_filter':default_min,'texture_mag_filter':default_mag,
                'texture_internal_format':int(GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D,0,GL.GL_TEXTURE_INTERNAL_FORMAT)),
                'framebuffer_srgb_enabled':bool(GL.glIsEnabled(GL.GL_FRAMEBUFFER_SRGB)),
                'gl_vendor':GL.glGetString(GL.GL_VENDOR).decode(),
                'gl_renderer':GL.glGetString(GL.GL_RENDERER).decode()}
            assert int(ctx.con.offSamples)==aa
            assert default_min==int(GL.GL_LINEAR_MIPMAP_LINEAR) and default_mag==int(GL.GL_LINEAR), runtime[str(aa)]
            def filtering(mode):
                GL.glActiveTexture(GL.GL_TEXTURE0)
                GL.glBindTexture(GL.GL_TEXTURE_2D,glid)
                mini = {'default':default_min,'linear':GL.GL_LINEAR,'nearest':GL.GL_NEAREST}[mode]
                mag = GL.GL_NEAREST if mode=='nearest' else default_mag
                GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER,mini)
                GL.glTexParameteri(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MAG_FILTER,mag)
                assert int(GL.glGetTexParameteriv(GL.GL_TEXTURE_2D,GL.GL_TEXTURE_MIN_FILTER))==mini
            def draw(size, mode='default',unlit=False,white=False):
                ctx.gl_ctx.make_current(); filtering(mode)
                ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                ctx.cam.fixedcamid = sim.model.camera_name2id('agentview')
                mujoco.mjv_updateScene(model,sim.data._data,ctx.vopt,ctx.pert,ctx.cam,
                                      mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
                flags = ctx.scn.flags.copy()
                if unlit:
                    ctx.scn.nlight = 0
                    for flag in ['mjRND_SHADOW','mjRND_REFLECTION','mjRND_SKYBOX','mjRND_HAZE','mjRND_FOG']:
                        if hasattr(mujoco.mjtRndFlag,flag): ctx.scn.flags[getattr(mujoco.mjtRndFlag,flag)]=0
                    for geom in ctx.scn.geoms[:ctx.scn.ngeom]:
                        is_patch = geom.objtype==mujoco.mjtObj.mjOBJ_GEOM and geom.objid==gid
                        geom.emission = 1. if is_patch else 0.
                        if is_patch and white: geom.matid=-1
                mujoco.mjr_render(mujoco.MjrRect(0,0,size,size),ctx.scn,ctx.con)
                raw = ctx.read_pixels(size,size)
                ctx.scn.flags[:] = flags
                return np.ascontiguousarray(raw[::-1,::-1])
            for name, texture in textures.items():
                model.tex_data[adr:adr+length] = texture.reshape(-1)
                mujoco.mjr_uploadTexture(model,ctx.con,tid)
                outputs = {v[0]:[] for v in variants if v[-1]==aa}
                if aa==4: outputs['render']=[]
                for k,state in enumerate(states):
                    env.regenerate_obs_from_state(state)
                    for cond,mode,unlit,size,nearest,vaa in variants:
                        if vaa!=aa: continue
                        raw = draw(size,mode,unlit)
                        im = np.asarray(Image.fromarray(raw).resize((224,224),Image.NEAREST)) if nearest else resize_rgb(raw,224)
                        if cond=='render_roi':
                            outputs['render'].append(im)
                            sourcekey = 'drift__random_render_base' if name=='random' else ('drift' if name=='full' else 'band32')+'__optimized_render_base'
                            assert np.array_equal(im,old[sourcekey][k]), (name,k,'default render differs from E15')
                            if k==0:
                                assert np.array_equal(raw,draw(size,mode,unlit)), 'render repeat differs'
                                arrays[name+'__native256_first']=raw
                        outputs[cond].append(np.where(arrays['mask'][k,...,None],im,arrays['clean_base'][k]))
                    if name=='random' and k==0:
                        white = resize_rgb(draw(256,'default',True,True),224)
                        inner = cv2.erode(arrays['mask'][k].astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
                        checks['unlit_white_min_u8_aa'+str(aa)]=int(white[inner].min())
                        assert white[inner].min()==255, 'unlit pass is not unit albedo'
                        model.tex_data[adr:adr+length] = 128
                        mujoco.mjr_uploadTexture(model,ctx.con,tid)
                        gray = resize_rgb(draw(256,'default',True),224)
                        checks['unlit_gray128_range_aa'+str(aa)]=[int(gray[inner].min()),int(gray[inner].max())]
                        assert np.abs(gray[inner].astype(int)-128).max()<=1
                        model.tex_data[adr:adr+length] = texture.reshape(-1)
                        mujoco.mjr_uploadTexture(model,ctx.con,tid)
                for cond,ims in outputs.items(): arrays[name+'__'+cond]=np.array(ims)
                print('[capture]',name,'AA',aa,'frames',n,flush=True)
            # Release external references before closing the renderer. Keeping
            # the previous context alive while constructing the next can free
            # an EGL context after the new one is current.
            del draw, filtering, ctx, sim, model
        finally:
            env.close()
    for name,pattern in patterns.items():
        roundtrip = cv2.warpPerspective(textures[name][::-1].copy(),np.linalg.inv(H),(224,224))
        shifted = cv2.warpAffine(roundtrip,np.float32([[1,0,.5],[0,1,-1]]),(224,224))
        low32 = cv2.resize(cv2.resize(pattern,(32,32),interpolation=cv2.INTER_AREA),(224,224),interpolation=cv2.INTER_LINEAR)
        digital = {'digital':pattern,'digital_resize256':resize_rgb(resize_rgb(pattern,256),224),
                   'digital_export128':roundtrip,'digital_export128_align':shifted,
                   'digital_lowpass32':low32}
        for cond,image in digital.items():
            arrays[name+'__'+cond]=np.where(arrays['mask'][...,None],image,arrays['clean_base']).astype(np.uint8)
        sourcekey = 'drift__random_digital' if name=='random' else ('drift' if name=='full' else 'band32')+'__digital'
        assert np.array_equal(arrays[name+'__digital'],old[sourcekey][:n])
    conditions = list(digital)+['render']+[v[0] for v in variants]
    arrays['conditions']=np.array(conditions)
    np.savez_compressed(dest/'inputs.npz',**arrays)
    checks.update(default_render_equals_E15=True,digital_equals_E15=True,repeat_render_equal=True)
    meta = {'complete':True,'frames':n,'steps':steps.tolist(),'patterns':list(patterns),
            'conditions':conditions,'runtime':runtime,'checks':checks,
            'source_input_sha256':hashlib.sha256((D.OUT/'kv_render_probe/inputs.npz').read_bytes()).hexdigest(),
            'trajectory_sha256':hashlib.sha256(traj_path.read_bytes()).hexdigest(),
            'notes':'All renderer interventions except raw render use the same E15 hard mask and clean background. '
                    'Unlit uses zero scene lights and patch emission=1, white pass checks unit response. '
                    'No tonemapping operation is introduced. Fixed MuJoCo3.2.3 classic OpenGL. '
                    'AA changes offscreen samples for the scene but only patch ROI enters model. '
                    'Band32 is separately optimized; digital_lowpass32 is the same-pattern frequency ablation.'}
    (dest/'capture.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    print('[done]',dest,flush=True)


if __name__=='__main__': main()
