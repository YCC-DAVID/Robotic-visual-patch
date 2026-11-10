"""Independent random/stripe transfer check of center versus four-grid models."""
import json
import os
from pathlib import Path
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-render-sampling-numba')
import drift_eval_digital as D
import numpy as np


def main():
    import mujoco, yaml
    from PIL import Image
    from OpenGL import GL
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import scene_patch as sp
    dest = D.OUT/'render_proxy_eot/sampling_check'; dest.mkdir(parents=True, exist_ok=True)
    patch = np.load(D.OUT/'drift_patch_drift.npz', allow_pickle=False)
    traj = np.load(D.OUT/('traj_'+D.TASK+'.npz'), allow_pickle=False)
    cfg = yaml.safe_load((D.PROBE/'config/scene.yaml').read_text()); cfg['patch']['size_wh']=[.26, .26]
    q = {'random_unseen': np.random.RandomState(20261007).randint(0, 256, (128, 128, 3)).astype(np.uint8)}
    for period in [2, 4, 8, 16]:
        yy, xx = np.mgrid[:128, :128]
        stripes = np.empty((128, 128, 3), np.uint8)
        stripes[..., 0] = np.rint((.5+.45*np.cos(2*np.pi*(xx+.37)/period))*255).astype(np.uint8)
        stripes[..., 1] = np.rint((.5+.45*np.cos(2*np.pi*(yy+.37)/period))*255).astype(np.uint8)
        stripes[..., 2] = np.rint((.5+.45*np.cos(2*np.pi*(xx+yy+.37)/period))*255).astype(np.uint8)
        q['stripe_period'+str(period)] = stripes
    initial = dest/'texture.png'; Image.fromarray(q['random_unseen']).save(initial)
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if Path(suite.get_task(i).bddl_file).stem==D.TASK)
    bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg, patch['cell_world'], str(initial)))
    arrays = {}
    try:
        env.seed(int(traj['shared_seed'])); env.reset()
        env.regenerate_obs_from_state(traj['f000__flatten'])
        sim, ctx = env.env.sim, env.env.sim._render_context_offscreen
        model = sim.model._model; ctx.gl_ctx.make_current()
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, cfg['patch']['name']+'_tex')
        adr = int(model.tex_adr[tid])
        for name, texture in q.items():
            model.tex_data[adr:adr+128*128*3] = texture.reshape(-1)
            mujoco.mjr_uploadTexture(model,ctx.con,tid)
            ctx.render(256,256,camera_id=sim.model.camera_name2id('agentview'))
            image = np.ascontiguousarray(ctx.read_pixels(256,256)[::-1,::-1])
            arrays[name+'__texture'], arrays[name+'__render256'] = texture, image
        previous = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        try:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER,int(ctx.con.offFBO))
            arrays['gl_sample_positions'] = np.array([GL.glGetMultisamplefv(GL.GL_SAMPLE_POSITION,i) for i in range(4)])
            arrays['gl_samples'] = np.array(int(GL.glGetIntegerv(GL.GL_SAMPLES)))
        finally:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER,previous)
        del ctx, sim, model
    finally:
        env.close()
    arrays['conditions'] = np.array(list(q))
    np.savez_compressed(dest/'inputs.npz', **arrays)
    print('[captured]',list(q),'sample positions',arrays['gl_sample_positions'].tolist(),flush=True)


if __name__ == '__main__':
    main()
