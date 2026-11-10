"""E-MT step 1: per-task clean pi0.5 pool + dual-view 13cm/101 calibration.

Same recipe as E22 pool (render_proxy_pool_capture) followed by E26 dual-view
capture (dual_view_capture), parameterized by LIBERO-goal task. Clean rollouts
use inits 0-11 (0-9 train, 10-11 validation); test inits 40-49 never enter.
"""
import argparse
import collections
import hashlib
import json
import os
import sys
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-multitask-numba')
import drift_eval_digital as D
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cell-world', default=None, help='x,y table position (z fixed .9015); default E26 site 101')
    ap.add_argument('--pool-from', default=None, help='npz holding a finished stage-A pool (skip clean rollouts)')
    ap.add_argument('--pool-only', action='store_true', help='run stage A only and save pool.npz')
    argv = sys.argv[1:]
    # argparse treats a negative comma-separated coordinate as a new option.
    if '--cell-world' in argv:
        i = argv.index('--cell-world')
        if i+1 < len(argv) and ',' in argv[i+1]:
            argv[i:i+2] = ['--cell-world='+argv[i+1]]
    args = ap.parse_args(argv)
    import cv2, mujoco, torch, yaml
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    from drift_resize_order import resize_rgb
    from dual_view_capture import homography
    import scene_patch as sp
    dest = D.pathlib.Path(args.out); dest.mkdir(parents=True, exist_ok=True)
    assert not (dest/('pool.npz' if args.pool_only else 'base.npz')).exists(), 'refuse to overwrite'
    geometry = np.load(D.OUT/'drift_patch_drift.npz', allow_pickle=False)
    cell_world, cell_xy, patch_m = geometry['cell_world'], geometry['cell_xy'], .13
    if args.cell_world:
        x, y = (float(v) for v in args.cell_world.split(','))
        cell_world, cell_xy = np.array([x, y, float(geometry['cell_world'][2])]), np.array([x, y])
    cfg = yaml.safe_load((D.PROBE/'config/scene.yaml').read_text()); cfg['patch']['size_wh'] = [patch_m]*2
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == args.task)
    bddl = D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    init_path = D.pathlib.Path(get_libero_path('init_states'))/'libero_goal'/(args.task+'.pruned_init')
    inits = np.array(torch.load(str(init_path)))
    indices = list(range(12))
    # ---------------- stage A: clean rollouts (identical rules to E22 pool) --------------
    POOL_KEYS = ['clean_base', 'clean_wrist', 'state8', 'sim_states', 'ep_id', 'env_seed', 'env_step', 'noise_seeds']
    if args.pool_from:
        with np.load(args.pool_from, allow_pickle=False) as z:
            A = {k: z[k] for k in POOL_KEYS}
            outcomes = json.loads(str(z['outcomes'])) if 'outcomes' in z.files else None
        sidecar = D.pathlib.Path(args.pool_from).with_name('capture.json')
        if outcomes is None and sidecar.is_file():
            outcomes = json.loads(sidecar.read_text()).get('episodes')
        fixtures = None
    else:
        A, outcomes, fixtures = stage_a(args, D, task, bddl, inits, indices, POOL_KEYS, resize_rgb)
        if args.pool_only:
            np.savez_compressed(dest/'pool.npz', **A, outcomes=np.array(json.dumps(outcomes)), task=np.array(args.task))
            print('[pool done]', args.task, len(A['ep_id']), [o['success'] for o in outcomes], flush=True); return
    calibrate(args, D, task, dest, cfg, bddl, A, outcomes, fixtures, cell_world, cell_xy, patch_m, init_path,
              resize_rgb, homography, sp)


def stage_a(args, D, task, bddl, inits, indices, POOL_KEYS, resize_rgb):
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    A = {k: [] for k in POOL_KEYS}
    outcomes, fixtures = [], {}
    try:
        for ep in indices:
            seed = 10000+ep
            env.seed(seed); env.reset()
            fixtures[seed] = (env.env.sim.model.body_pos.copy(), env.env.sim.model.body_quat.copy())
            obs = env.set_init_state(inits[ep])
            for _ in range(D.NWAIT):
                obs, _, _, _ = env.step(D.DUMMY)
            queue = collections.deque(); ok, inference, success_step = False, 0, None
            goal = env.env.parsed_problem['goal_state']
            for t in range(D.MAX_STEPS):
                if not queue:
                    state = env.get_sim_state().copy()
                    obs = env.regenerate_obs_from_state(state)
                    native = np.ascontiguousarray(obs['agentview_image'][::-1, ::-1])
                    st = np.concatenate([obs['robot0_eef_pos'], D.quat2axisangle(obs['robot0_eef_quat']),
                                         obs['robot0_gripper_qpos']]).astype(np.float32)
                    img = resize_rgb(native, 224)
                    wrist = resize_rgb(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1, ::-1]), 224)
                    noise_seed = 20261006+ep*1000+inference
                    noise = np.random.RandomState(noise_seed).standard_normal((10, 32)).astype(np.float32)
                    for key, value in zip(A, [img, wrist, st, state, ep, seed, t+D.NWAIT, noise_seed]):
                        A[key].append(value)
                    out = client.infer({'observation/image': img, 'observation/wrist_image': wrist,
                                        'observation/state': st, 'prompt': str(task.language), 'noise': noise})
                    queue.extend(out['actions'][:D.REPLAN]); inference += 1
                obs, _, _, _ = env.step(queue.popleft().tolist())
                if all(env.env._eval_predicate(s) for s in goal):
                    ok, success_step = True, t+D.NWAIT; break
            outcomes.append({'episode': ep, 'success': ok, 'success_step': success_step, 'frames': inference})
            print('[pool]', args.task, ep, ok, inference, flush=True)
    finally:
        env.close(); client._ws.close()
    return {k: np.array(v) for k, v in A.items()}, outcomes, fixtures


def calibrate(args, D, task, dest, cfg, bddl, A, outcomes, fixtures, cell_world, cell_xy, patch_m, init_path,
              resize_rgb, homography, sp):
    import cv2, mujoco
    from PIL import Image
    from libero.libero.envs import OffScreenRenderEnv
    # ---------------- stage B: dual-view calibration on the same states ------------------
    random = np.random.RandomState(20261007).randint(0, 256, (128, 128, 3)).astype(np.uint8)
    x = np.rint(np.linspace(0, 255, 128)).astype(np.uint8)
    ramp = np.empty_like(random); ramp[..., 0] = x[None, :]; ramp[..., 1] = x[:, None]; ramp[..., 2] = 255
    low = np.random.RandomState(20261009).randint(0, 256, (32, 32, 3)).astype(np.uint8)
    band = np.array(Image.fromarray(low).resize((128, 128), Image.BILINEAR))
    textures = {'black256': np.zeros_like(random), 'white256': np.full_like(random, 255), 'random256': random,
                'check256': band, 'ramp256': ramp, 'unlit_white256': np.full_like(random, 255)}
    texpath = dest/'random.png'; Image.fromarray(random).save(texpath)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.env.set_xml_processor(sp.make_xml_processor(cfg, cell_world, str(texpath)))
    cams = {'base': 'agentview', 'wrist': 'robot0_eye_in_hand'}
    buffers = {v: {k: [] for k in list(textures)+['mask', 'image256_to_texture_pixel_H']} for v in cams}
    audits, previous = [], None
    try:
        for k, state in enumerate(A['sim_states']):
            seed = int(A['env_seed'][k])
            if seed != previous:
                env.seed(seed); env.reset(); previous = seed
                if fixtures is not None:
                    pos, quat = fixtures[seed]; n = len(pos)
                    assert np.array_equal(env.env.sim.model.body_pos[:n], pos), ('fixture', seed)
                    assert np.array_equal(env.env.sim.model.body_quat[:n], quat), ('fixture', seed)
            env.regenerate_obs_from_state(state); sim = env.env.sim; ctx = sim._render_context_offscreen; model = sim.model._model
            ctx.gl_ctx.make_current(); gid = sim.model.geom_name2id(cfg['patch']['name'])
            tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, cfg['patch']['name']+'_tex'); adr = int(model.tex_adr[tid])
            for view, camera in cams.items():
                cid = sim.model.camera_name2id(camera); buf = buffers[view]
                H = homography(sim, gid, camera); buf['image256_to_texture_pixel_H'].append(H)
                ctx.render(256, 256, camera_id=cid, segmentation=True)
                seg = np.ascontiguousarray(ctx.read_pixels(256, 256, segmentation=True)[::-1, ::-1])
                visible = (seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 1] == gid)
                mask = resize_rgb(np.repeat((visible.astype(np.uint8)*255)[..., None], 3, -1), 224)[..., 0] >= 128
                buf['mask'].append(mask)
                for name, tex in textures.items():
                    model.tex_data[adr:adr+128*128*3] = tex.ravel(); mujoco.mjr_uploadTexture(model, ctx.con, tid)
                    ctx.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED; ctx.cam.fixedcamid = cid
                    mujoco.mjv_updateScene(model, sim.data._data, ctx.vopt, ctx.pert, ctx.cam, mujoco.mjtCatBit.mjCAT_ALL, ctx.scn)
                    flags = ctx.scn.flags.copy()
                    if name in ['ramp256', 'unlit_white256']:
                        ctx.scn.nlight = 0
                        for flag in ['mjRND_SHADOW', 'mjRND_REFLECTION', 'mjRND_SKYBOX', 'mjRND_HAZE', 'mjRND_FOG']:
                            if hasattr(mujoco.mjtRndFlag, flag): ctx.scn.flags[getattr(mujoco.mjtRndFlag, flag)] = 0
                        for g in ctx.scn.geoms[:ctx.scn.ngeom]:
                            g.emission = 1. if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == gid else 0.
                    mujoco.mjr_render(mujoco.MjrRect(0, 0, 256, 256), ctx.scn, ctx.con)
                    image = np.ascontiguousarray(ctx.read_pixels(256, 256)[::-1, ::-1]); ctx.scn.flags[:] = flags
                    buf[name].append(image)
                if view == 'base':
                    # Patch is visual-only: far from its footprint the patched render equals the clean input.
                    far = ~cv2.dilate(mask.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
                    assert np.array_equal(resize_rgb(buf['black256'][-1], 224)[far], A['clean_base'][k][far]), ('background', k)
                valid = cv2.erode((visible & (buf['unlit_white256'][-1].min(-1) == 255)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                yy, xx = np.where(valid); error = None
                if len(yy):
                    p = np.c_[xx, yy, np.ones(len(xx))]@H.T; uv = p[:, :2]/p[:, 2:]
                    measured = buf['ramp256'][-1][valid, :2].astype(float)
                    measured = np.stack([np.interp(measured[:, j], x, np.arange(128)) for j in range(2)], -1)
                    error = float(np.sqrt(((uv-measured)**2).mean()))
                audits.append({'frame': k, 'view': view, 'visible224': int(mask.sum()), 'uv_interior_points': len(yy), 'UV_RMSE_texels': error})
            del ctx, sim, model
            if k % 25 == 0: print('[capture]', args.task, k, len(A['sim_states']), audits[-1], flush=True)
    finally:
        env.close()
    common = dict(A, prompt=np.array(str(task.language)), task=np.array(args.task), cell_world=cell_world,
                  cell_xy=cell_xy, patch_m=np.array(patch_m), train_mask=A['ep_id'] < 10, validation_mask=A['ep_id'] >= 10)
    for view in cams:
        np.savez_compressed(dest/(view+'.npz'), **common, **{k: np.array(v) for k, v in buffers[view].items()},
                            random__texture=random, check__texture=band, camera=cams[view])
    uv = [r['UV_RMSE_texels'] for r in audits if r['UV_RMSE_texels'] is not None]
    meta = {'complete': True, 'task': args.task, 'prompt': str(task.language), 'episodes': outcomes,
            'frames': int(len(A['ep_id'])), 'train_frames': int((A['ep_id'] < 10).sum()),
            'validation_frames': int((A['ep_id'] >= 10).sum()),
            'init_states_sha256': hashlib.sha256(init_path.read_bytes()).hexdigest(),
            'visible224_median': {v: float(np.median([r['visible224'] for r in audits if r['view'] == v])) for v in cams},
            'UV_RMSE_texels_max': max(uv) if uv else None, 'records': audits,
            'rules': 'env seed 10000+init; noise 20261006+init*1000+replan; replan 5; 300 steps; 13cm @ cell_world',
            'cell_world': [float(v) for v in cell_world], 'pool_from': args.pool_from,
            'checks': ('fixtures exact per seed; ' if fixtures is not None else 'pool reused, fixture check n/a; ')
                      + 'base far-background equals clean input every frame'}
    (dest/'capture.json').write_text(json.dumps(meta, indent=2)+'\n')
    print('[done]', args.task, meta['frames'], None if outcomes is None else [o['success'] for o in outcomes], flush=True)


if __name__ == '__main__':
    main()
