"""Add independently measured native coverage to the existing calibration.

Unlit white-minus-black isolates coverage without using lighting gains as a
mask. Existing white and random frames must replay exactly before accepting.
"""
import json
import hashlib
import drift_eval_digital as D
import numpy as np


def main():
    import mujoco, yaml
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    dest = D.OUT/'proxy_factorial'; dest.mkdir(parents=True, exist_ok=True)
    source = D.OUT/'render_proxy_scale/cal_13_101/calibration.npz'
    assert not (dest/'calibration.npz').exists()
    with np.load(source) as p: data = {k:p[k] for k in p.files}
    cfg=yaml.safe_load((D.PROBE/'config/scene.yaml').read_text())
    cfg['patch']['size_wh']=[.13,.13]
    suite=benchmark.get_benchmark_dict()['libero_goal']()
    task=next(suite.get_task(i) for i in range(suite.n_tasks) if D.pathlib.Path(suite.get_task(i).bddl_file).stem==D.TASK)
    bddl=D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg,data['cell_world'],str(source.parent/'random_texture.png')))
    black=[]; previous=None
    try:
        for k,state in enumerate(data['sim_states']):
            seed=int(data['env_seed'][k])
            if seed!=previous: env.seed(seed);env.reset();previous=seed
            env.regenerate_obs_from_state(state)
            sim=env.env.sim;ctx=sim._render_context_offscreen;model=sim.model._model
            ctx.gl_ctx.make_current();gid=sim.model.geom_name2id(cfg['patch']['name'])
            cid=sim.model.camera_name2id('agentview')
            tid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TEXTURE,cfg['patch']['name']+'_tex')
            adr=int(model.tex_adr[tid])
            def draw(texture,unlit):
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
                img=np.ascontiguousarray(ctx.read_pixels(256,256)[::-1,::-1]);ctx.scn.flags[:]=flags
                return img
            assert np.array_equal(draw(data['random__texture'],False),data['random256'][k]), ('random',k)
            assert np.array_equal(draw(np.full((128,128,3),255,np.uint8),True),data['unlit_white256'][k]), ('white',k)
            black.append(draw(np.zeros((128,128,3),np.uint8),True))
            del draw,ctx,model,sim
            if k%25==0:print('[capture]',k,flush=True)
    finally:env.close()
    data['unlit_black256']=np.array(black)
    np.savez_compressed(dest/'calibration.npz',**data)
    (dest/'calibration_check.json').write_text(json.dumps({'complete':True,'frames':len(black),
        'random_native_exact':True,'unlit_white_native_exact':True,
        'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2)+'\n')
    print('[done]',flush=True)


if __name__=='__main__':main()
