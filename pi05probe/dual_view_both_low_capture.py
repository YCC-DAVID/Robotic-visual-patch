"""Frozen shared textures in BOTH real cameras on the32 validation states."""
import os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/wam-dual-validation-numba')
import json
import drift_eval_digital as D
import numpy as np


def main():
    import mujoco,yaml
    from PIL import Image
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb
    import scene_patch as sp
    root=D.OUT/'dual_view';extension=D.OUT/'dual_view_both_low';dest=extension/'validation';dest.mkdir(exist_ok=True);assert not (dest/'inputs.npz').exists()
    cal={v:np.load(root/'calibration'/(v+'.npz'),allow_pickle=False) for v in ['base','wrist']};b=cal['base'];ids=np.flatnonzero(b['validation_mask'])
    textures={'random':np.array(Image.open(extension/'manifest/random.png'))}
    for mode in ['both_low']:
        run=extension/'train';q=np.load(run/'checkpoint_0500.npz')['Q'];tex=np.array(Image.open(run/'texture_0500.png'))
        assert np.array_equal(tex,np.rint(q.transpose(1,2,0)*255).clip(0,255).astype(np.uint8));textures[mode]=tex
    arrays={k:b[k][ids] for k in ['clean_base','clean_wrist','state8','ep_id','env_step']}
    arrays.update(frames=ids,prompt=b['prompt'],patterns=np.array(list(textures)))
    cfg=yaml.safe_load((D.PROBE/'config/scene.yaml').read_text());cfg['patch']['size_wh']=[.13,.13]
    suite=benchmark.get_benchmark_dict()['libero_goal']();task=next(suite.get_task(i) for i in range(suite.n_tasks) if D.pathlib.Path(suite.get_task(i).bddl_file).stem==D.TASK)
    bddl=D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg,b['cell_world'],str(extension/'train/texture_0500.png')))
    fixtures=json.loads((D.OUT/'render_proxy_eot/data/manifest.json').read_text())['fixtures'];fixtures={r['env_seed']:r for r in fixtures}
    cameras={'base':'agentview','wrist':'robot0_eye_in_hand'};buffers={p+'__'+v:[] for p in textures for v in cameras};previous=None
    try:
        for k in ids:
            seed=int(b['env_seed'][k])
            if seed!=previous:
                env.seed(seed);env.reset();previous=seed;f=fixtures[seed];n=len(f['body_pos'])
                assert np.array_equal(env.env.sim.model.body_pos[:n],f['body_pos']);assert np.array_equal(env.env.sim.model.body_quat[:n],f['body_quat'])
            env.regenerate_obs_from_state(b['sim_states'][k]);sim=env.env.sim;ctx=sim._render_context_offscreen;model=sim.model._model;ctx.gl_ctx.make_current()
            tid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TEXTURE,cfg['patch']['name']+'_tex');adr=int(model.tex_adr[tid])
            for name,texture in [('anchor',b['random__texture'])]+list(textures.items()):
                model.tex_data[adr:adr+128*128*3]=texture.ravel();mujoco.mjr_uploadTexture(model,ctx.con,tid)
                for view,camera in cameras.items():
                    ctx.render(256,256,camera_id=sim.model.camera_name2id(camera));image=np.ascontiguousarray(ctx.read_pixels(256,256)[::-1,::-1])
                    if name=='anchor':assert np.array_equal(image,cal[view]['random256'][k]),(view,k)
                    else:buffers[name+'__'+view].append(resize_rgb(image,224))
            del sim,ctx,model
            print('[capture]',int(k),flush=True)
    finally:env.close()
    arrays.update({k:np.array(v) for k,v in buffers.items()});arrays.update({name+'__texture':tex for name,tex in textures.items()})
    np.savez_compressed(dest/'inputs.npz',**arrays)
    (dest/'capture.json').write_text(json.dumps({'complete':True,'both_camera_native_anchors_exact':True,'fixture_positions_quaternions_exact':True,'frames':ids.tolist(),'patterns':list(textures)},indent=2)+'\n')

if __name__=='__main__':main()
