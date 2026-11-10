"""Debug: fresh env with patch AT a candidate; does the patch survive env.reset() across seeds?"""
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-visscan-numba')
import drift_eval_digital as D
import numpy as np


def main():
    import mujoco, yaml
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    task_name = 'put_the_bowl_on_the_plate'
    cfg = yaml.safe_load((D.PROBE/'config/scene.yaml').read_text()); cfg['patch']['size_wh'] = [.13]*2
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks) if D.pathlib.Path(suite.get_task(i).bddl_file).stem == task_name)
    bddl = D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    p = np.load(D.OUT/'multitask_patch/tasks'/task_name/'base.npz')
    states, seeds, ep = p['sim_states'], p['env_seed'], p['ep_id']
    import sys
    frames = np.flatnonzero(ep < 10); ks = frames[np.linspace(0, len(frames)-1, 6, dtype=int)]
    if 'seq' in sys.argv: ks = frames[:40]
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg, [0.09, 0.15, 0.9015], '/tmp/visscan_smoke/random.png'))
    previous = None
    for k in ks:
        if seeds[k] != previous:
            env.seed(int(seeds[k])); env.reset(); previous = seeds[k]
        env.regenerate_obs_from_state(states[k]); sim = env.env.sim
        gid = sim.model.geom_name2id(cfg['patch']['name']); bid = sim.model.body_name2id(cfg['patch']['name']+'_body')
        ctx = sim._render_context_offscreen; ctx.gl_ctx.make_current()
        ctx.render(256, 256, camera_id=sim.model.camera_name2id('agentview'), segmentation=True)
        seg = ctx.read_pixels(256, 256, segmentation=True)
        n = int(((seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 1] == gid)).sum())
        if 'seq' in sys.argv and k in (5, 20):
            from PIL import Image
            ctx.render(256, 256, camera_id=sim.model.camera_name2id('agentview'))
            rgb = ctx.read_pixels(256, 256)[::-1]
            Image.fromarray(np.concatenate([rgb, env.regenerate_obs_from_state(states[k])['agentview_image'][::-1]], 1)).save(f'/shared/user65/workspace/chence/WAMattack/pi05probe/out/visible_pos/debug_k{k}.png')
        ids = np.unique(seg[..., 1][seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM])
        if False: print('ctx.sim is sim', ctx.sim is sim, 'ctx.model is model', ctx.model is sim.model._model if hasattr(ctx, 'model') else None,
              'ctx.data is data', getattr(ctx, 'data', None) is sim.data._data, 'ngeom in scn', ctx.scn.ngeom, 'max seg id', ids.max() if len(ids) else None,
              'geomgroup', list(ctx.vopt.geomgroup), flush=True)
        print(k, int(seeds[k]), 'ngeom', sim.model.ngeom, 'gid', gid, 'body_pos', sim.model.body_pos[bid], 'xpos', sim.data.xpos[bid],
              'geom_xpos', sim.data.geom_xpos[gid], 'px', n, 'nq', sim.model.nq, 'state', len(states[k]), flush=True)
    env.close()


if __name__ == '__main__':
    main()
