"""Separate stored-clean replay, patch background and manual-render discrepancies."""
import argparse
import json
import numpy as np
import drift_eval_digital as D


def error(a,b,mask=None):
    d=np.abs(a.astype(np.int16)-b.astype(np.int16))
    if mask is not None: d=d[mask]
    return dict(changed_pixels=int(np.any(d>0,axis=-1).sum()),max_abs=int(d.max()),
                mean_abs=float(d.mean()),rmse=float(np.sqrt((d.astype(float)**2).mean())))


def render_frame(env,state,patch_name=None):
    import mujoco,cv2
    from drift_resize_order import resize_rgb
    obs=env.regenerate_obs_from_state(state)
    sim=env.env.sim;ctx=sim._render_context_offscreen;ctx.gl_ctx.make_current()
    out={}
    for name,camera in [('base','agentview'),('wrist','robot0_eye_in_hand')]:
        native=np.ascontiguousarray(obs[camera+'_image'][::-1,::-1])
        out[name]=dict(native=resize_rgb(native,224))
        if patch_name is not None:
            gid=sim.model.geom_name2id(patch_name)
            cid=sim.model.camera_name2id(camera)
            ctx.render(256,256,camera_id=cid,segmentation=True)
            seg=np.ascontiguousarray(ctx.read_pixels(256,256,segmentation=True)[::-1,::-1])
            vis=(seg[...,0]==mujoco.mjtObj.mjOBJ_GEOM)&(seg[...,1]==gid)
            mask=resize_rgb(np.repeat((vis.astype(np.uint8)*255)[...,None],3,-1),224)[...,0]>=128
            ctx.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED;ctx.cam.fixedcamid=cid
            mujoco.mjv_updateScene(sim.model._model,sim.data._data,ctx.vopt,ctx.pert,ctx.cam,mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
            mujoco.mjr_render(mujoco.MjrRect(0,0,256,256),ctx.scn,ctx.con)
            manual=resize_rgb(np.ascontiguousarray(ctx.read_pixels(256,256)[::-1,::-1]),224)
            out[name].update(manual=manual,far=~cv2.dilate(mask.astype(np.uint8),np.ones((17,17),np.uint8)).astype(bool))
    return out,sim.model.body_pos.copy(),sim.model.body_quat.copy()


def capture(bddl,cfg,states,seeds,ks,world,texture):
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    if world is not None: env.env.set_xml_processor(sp.make_xml_processor(cfg,world,str(texture)))
    result={};previous=None
    try:
        for k in ks:
            if seeds[k]!=previous:
                env.seed(int(seeds[k]));env.reset();previous=seeds[k]
            result[int(k)]=render_frame(env,states[k],cfg['patch']['name'] if world is not None else None)
    finally: env.close()
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--task',default='put_the_bowl_on_the_plate')
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    import yaml
    from PIL import Image
    from libero.libero import benchmark,get_libero_path
    import visible_pos_pipeline as V
    dest=D.pathlib.Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    with np.load(V.pool_path(args.task),allow_pickle=False) as p:
        states,seeds,episodes=p['sim_states'],p['env_seed'],p['ep_id']
        stored={v:p['clean_'+v] for v in ['base','wrist']}
    ks=np.array([np.flatnonzero(episodes==e)[0] for e in [0,1,9]])
    suite=benchmark.get_benchmark_dict()['libero_goal']()
    task=next(suite.get_task(i) for i in range(suite.n_tasks) if D.pathlib.Path(suite.get_task(i).bddl_file).stem==args.task)
    bddl=D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    cfg=yaml.safe_load((D.PROBE/'config/scene.yaml').read_text());cfg['patch']['size_wh']=[.13]*2
    xy=json.loads((V.DEST/'selection.json').read_text())[args.task]['world']
    world=[*xy,float(np.load(D.OUT/'drift_patch_drift.npz')['cell_world'][2])]
    texture=dest/'black.png';Image.fromarray(np.zeros((128,128,3),np.uint8)).save(texture)
    clean=capture(bddl,cfg,states,seeds,ks,None,None)
    patched=capture(bddl,cfg,states,seeds,ks,world,texture)
    rows=[]
    for k in ks:
        c,cp,cq=clean[int(k)];p,pp,pq=patched[int(k)];n=len(cp)
        row=dict(frame=int(k),episode=int(episodes[k]),fixtures_exact=bool(np.array_equal(cp,pp[:n]) and np.array_equal(cq,pq[:n])),views={})
        for v in ['base','wrist']:
            row['views'][v]=dict(clean_vs_stored=error(c[v]['native'],stored[v][k]),
                manual_vs_native=error(p[v]['manual'],p[v]['native']),
                far_patch_vs_clean=error(p[v]['manual'],c[v]['native'],p[v]['far']),
                far_patch_vs_stored=error(p[v]['manual'],stored[v][k],p[v]['far']))
            Image.fromarray(np.concatenate([stored[v][k],c[v]['native'],p[v]['native'],p[v]['manual']],axis=1)).save(dest/f'{k}_{v}.png')
        rows.append(row);print(json.dumps(row),flush=True)
    (dest/'results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__': main()
