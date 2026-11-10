"""Frozen final-texture diagnostic only: paired heldout states and new noise."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from crossarch_common import ROOT,dump,sha
from fastwam_libero_trial import DEFAULT,CALIBRATION


def capture(root):
    sys.path.insert(0,str(ROOT/'pi05probe'))
    import drift_eval_digital as D
    import scene_patch as sp
    import mujoco,yaml
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    with np.load(CALIBRATION) as z:d={k:z[k].copy() for k in z.files}
    indices=np.flatnonzero(d['validation_mask'])
    assert len(indices)==32 and set(d['ep_id'][indices])=={10,11}
    dest=root/'l0_validation';dest.mkdir(exist_ok=True)
    textures={name:np.asarray(Image.open(root/path).convert('RGB')) for name,path in
              [('random','random.png'),('optimized','patch/texture_0500.png')]}
    assert np.array_equal(textures['random'],d['random__texture'])
    cfg=yaml.safe_load((ROOT/'pi05probe/config/scene.yaml').read_text());cfg['patch']['size_wh']=[.10,.10]
    suite=benchmark.get_benchmark_dict()['libero_goal']()
    task=next(suite.get_task(i) for i in range(suite.n_tasks) if Path(suite.get_task(i).bddl_file).stem==D.TASK)
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    clean=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    patched=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    patched.env.set_xml_processor(sp.make_xml_processor(cfg,d['cell_world'],str(root/'random.png')))
    output={'indices':indices,'ep_id':d['ep_id'][indices],'state8':d['state8'][indices],
            'prompt':d['prompt'],'mask224':d['mask'][indices]}
    images={f'{name}_{camera}':[] for name in ['clean','random','optimized'] for camera in ['base','wrist']}
    previous=None
    def render(env,camera):
        sim=env.env.sim;ctx=sim._render_context_offscreen;ctx.gl_ctx.make_current()
        ctx.render(256,256,camera_id=sim.model.camera_name2id('agentview' if camera=='base' else 'robot0_eye_in_hand'))
        return np.ascontiguousarray(ctx.read_pixels(256,256)[::-1,::-1])
    try:
        for k in indices:
            seed=int(d['env_seed'][k])
            if previous!=seed:
                for env in [clean,patched]:env.seed(seed);env.reset()
                n=min(len(clean.env.sim.model.body_pos),len(patched.env.sim.model.body_pos))
                assert np.array_equal(clean.env.sim.model.body_pos[:n],patched.env.sim.model.body_pos[:n])
                assert np.array_equal(clean.env.sim.model.body_quat[:n],patched.env.sim.model.body_quat[:n])
                previous=seed
            oc=clean.regenerate_obs_from_state(d['sim_states'][k])
            patched.regenerate_obs_from_state(d['sim_states'][k])
            state=np.concatenate([oc['robot0_eef_pos'],D.quat2axisangle(oc['robot0_eef_quat']),oc['robot0_gripper_qpos']]).astype(np.float32)
            assert np.array_equal(state,d['state8'][k]),('state',int(k))
            base=np.ascontiguousarray(oc['agentview_image'][::-1,::-1])
            if not np.array_equal(base,d['clean_img256'][k]):
                np.savez_compressed(dest/'clean_replay_diagnosis.npz',manual=base,
                    observation=np.ascontiguousarray(oc['agentview_image'][::-1,::-1]),
                    expected=d['clean_img256'][k],body_pos=clean.env.sim.model.body_pos,
                    body_quat=clean.env.sim.model.body_quat,index=k)
            assert np.array_equal(base,d['clean_img256'][k]),('clean pixels',int(k))
            images['clean_base'].append(base);images['clean_wrist'].append(np.ascontiguousarray(oc['robot0_eye_in_hand_image'][::-1,::-1]))
            sim=patched.env.sim;ctx=sim._render_context_offscreen;ctx.gl_ctx.make_current();model=sim.model._model
            tid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_TEXTURE,cfg['patch']['name']+'_tex');adr=int(model.tex_adr[tid])
            for name,q in textures.items():
                model.tex_data[adr:adr+q.size]=q.reshape(-1);mujoco.mjr_uploadTexture(model,ctx.con,tid)
                actual=patched.regenerate_obs_from_state(d['sim_states'][k])
                base=np.ascontiguousarray(actual['agentview_image'][::-1,::-1])
                if name=='random' and not np.array_equal(base,d['random256'][k]):
                    np.savez_compressed(dest/'random_replay_diagnosis.npz',observation=base,expected=d['random256'][k],index=k)
                    raise AssertionError(('Random anchor',int(k)))
                images[name+'_base'].append(base);images[name+'_wrist'].append(np.ascontiguousarray(actual['robot0_eye_in_hand_image'][::-1,::-1]))
            del model,ctx,sim
            print('[L0 capture]',int(k),flush=True)
        output.update({k:np.asarray(v) for k,v in images.items()})
        output.update({name+'_texture':q for name,q in textures.items()})
        np.savez_compressed(dest/'inputs.npz',**output)
        dump(dest/'capture.json',dict(complete=True,states=32,validation_episodes=[10,11],
            clean_native_pixels_exact=True,random_native_pixels_exact=True,fixtures_exact=True,
            calibrated_states_sha256=sha(CALIBRATION),texture_sha256={n:sha(root/p) for n,p in
            [('random','random.png'),('optimized','patch/texture_0500.png')]}))
    finally:clean.close();patched.close()


def measure(root):
    import torch
    from fastwam_adapter import Backend
    sys.path.insert(0,str(ROOT/'pi05probe'))
    from render_world_proxy import WorldTextureProxy
    torch.set_num_threads(4);backend=Backend(root)
    dest=root/'l0_validation'
    with np.load(dest/'inputs.npz') as z:d={k:z[k].copy() for k in z.files}
    proxy=WorldTextureProxy(CALIBRATION,device='cuda',fit=root/'proxy_fit/fit.json')
    protocol=json.loads((root/'protocol.json').read_text());wrist_proxy=None
    if protocol.get('dual_camera_proxy') or protocol.get('experiment')=='fastwam_libero_physical_10cm_both':
        from dual_view_proxy import CameraProxy
        cfg=protocol.get('dual_camera_proxy',dict(wrist_calibration=str(root/'camera_proxy/calibration/wrist.npz'),
            wrist_fit=str(root/'camera_proxy/fit/wrist.json')))
        wrist_proxy=CameraProxy(cfg['wrist_calibration'],device='cuda',fit=cfg['wrist_fit'])
    rows=[];seeds=[20261201,20261202,20261203]
    def tensor(rgb):return torch.tensor(rgb.copy(),device='cuda').permute(2,0,1)[None].float()/255
    with torch.no_grad():
        for j,k in enumerate(d['indices']):
            clean_views=[tensor(d['clean_base'][j]),tensor(d['clean_wrist'][j])]
            views={}
            for name in ['random','optimized']:
                q=tensor(d[name+'_texture'])
                views[name+'_render_both']=[tensor(d[name+'_base'][j]),tensor(d[name+'_wrist'][j])]
                views[name+'_render_base']=[tensor(d[name+'_base'][j]),clean_views[1]]
                views[name+'_proxy_base']=[proxy(q,int(k),native=True),clean_views[1]]
                if wrist_proxy is not None:
                    views[name+'_proxy_both']=[proxy(q,int(k),native=True),wrist_proxy(q,int(k),native=True)]
            for seed in seeds:
                prepared=backend.prepare(d['state8'][j],str(d['prompt']),clean_views)
                vc,_=backend.response(prepared,seed,full=False)
                for condition,current in views.items():
                    v,_=backend.response(backend.prepare(d['state8'][j],str(d['prompt']),current),seed,full=False)
                    dv=v.float()-vc.float()
                    rows.append(dict(index=int(k),ep_id=int(d['ep_id'][j]),noise_seed=seed,condition=condition,
                        L0_MSE=float(dv.square().mean()),L0_SSE=float(dv.square().sum())))
            dump(dest/'measurements.json',rows)
            print('[L0 measure]',j+1,32,flush=True)
    summary={}
    for condition in views:
        rr=[r for r in rows if r['condition']==condition]
        summary[condition]=dict(n=len(rr),**{metric:{'mean':float(np.mean([r[metric] for r in rr])),
            'median':float(np.median([r[metric] for r in rr]))} for metric in ['L0_MSE','L0_SSE']})
    dump(dest/'summary.json',dict(complete=True,states=32,noise_seeds=seeds,conditions=summary,
        scope='Fixed final500 texture and exact Random, 32 heldout states, 3 new common noises; MSE over32x7 velocity at first flow step. Both-view actual render is primary. Base-only paths isolate training camera scope.',
        model_weights_changed=False,training_restarted=False,closed_loop_repeated=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['capture','measure']);ap.add_argument('--root',type=Path,default=DEFAULT)
    args=ap.parse_args();globals()[args.command](args.root)
