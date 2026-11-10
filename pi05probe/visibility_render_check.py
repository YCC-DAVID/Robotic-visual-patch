"""Diagnose native/direct render equality without changing scan acceptance rules."""
import argparse
import json
import numpy as np
import drift_eval_digital as D


def frame(env, state, index, seed, out):
    import mujoco
    from PIL import Image
    from robosuite import macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
    obs = env.regenerate_obs_from_state(state)
    sim = env.env.sim
    ctx = sim._render_context_offscreen
    ctx.gl_ctx.make_current()
    report = dict(frame=int(index), seed=int(seed), renderer_current_model=ctx.model is sim.model,
                  renderer_current_data=ctx.data is sim.data, cameras=[])
    for camera in ['agentview', 'robot0_eye_in_hand']:
        images = [obs[camera+'_image'].copy()]
        for _ in range(3):
            ctx.render(256, 256, camera_id=sim.model.camera_name2id(camera))
            images.append(ctx.read_pixels(256, 256)[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]].copy())
        values = []
        for image in images[1:]:
            diff = np.abs(image.astype(np.int16)-images[0].astype(np.int16))
            values.append(dict(changed_pixels=int(np.any(diff, axis=-1).sum()), max_abs=int(diff.max()),
                               mean_abs=float(diff.mean()), native_min=int(images[0].min()), direct_min=int(image.min())))
        repeat = np.abs(images[2].astype(np.int16)-images[1].astype(np.int16))
        report['cameras'].append(dict(camera=camera, native_vs_direct=values,
                                      repeat_changed_pixels=int(np.any(repeat,axis=-1).sum()), repeat_max_abs=int(repeat.max())))
        if any(v['changed_pixels'] for v in values):
            Image.fromarray(np.concatenate(images, axis=1)[::-1]).save(out/f'{index}_{camera}.png')
    # Force a new native observation after the direct renders, then compare again.
    fresh = env.regenerate_obs_from_state(state)
    report['native_repeat'] = {c: int(np.count_nonzero(fresh[c+'_image'] != obs[c+'_image']))
                              for c in ['agentview', 'robot0_eye_in_hand']}
    print(json.dumps(report), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='turn_on_the_stove')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    import yaml
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    import visible_pos_pipeline as V
    out = D.pathlib.Path(args.out); out.mkdir(parents=True,exist_ok=True)
    cfg = yaml.safe_load((D.PROBE/'config/scene.yaml').read_text());cfg['patch']['size_wh']=[.13]*2
    home=np.load(D.OUT/'drift_patch_drift.npz')['cell_world']
    tex=out/'random.png'
    Image.fromarray(np.random.RandomState(20261007).randint(0,256,(128,128,3)).astype(np.uint8)).save(tex)
    suite=benchmark.get_benchmark_dict()['libero_goal']()
    task=next(suite.get_task(i) for i in range(suite.n_tasks) if D.pathlib.Path(suite.get_task(i).bddl_file).stem==args.task)
    bddl=D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    with np.load(V.pool_path(args.task),allow_pickle=False) as p:
        states,seeds,episodes=p['sim_states'],p['env_seed'],p['ep_id']
    ks=[]
    for episode in [0,1,2]:
        frames=np.flatnonzero(episodes==episode)
        ks.extend(frames[np.linspace(0,len(frames)-1,3,dtype=int)])
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg,home,str(tex)))
    previous=None;reports=[]
    try:
        for k in ks:
            if seeds[k]!=previous:
                env.seed(int(seeds[k]));env.reset();previous=seeds[k]
            reports.append(frame(env,states[k],k,seeds[k],out))
    finally:
        env.close()
    (out/'results.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    main()
