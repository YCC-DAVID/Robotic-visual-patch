"""Per-task patch visibility scan over table positions (real MuJoCo segmentation).

For every clean TRAINING frame (inits 0-9) of a task pool and every legal
candidate position of the 13cm visual-only patch, count visible patch pixels
in the base and wrist policy inputs (224, same mask rule as multitask_capture).
The patch body is moved in place (model.body_pos + mj_kinematics); a fresh-env
spot check at two candidates verifies identical masks. Test inits never enter.
"""
import argparse
import hashlib
import json
import os
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/wam-visscan-numba')
import drift_eval_digital as D
import numpy as np

STEP, MARGIN, PATCH = .03, .01, .13
SCHEMA = 'visibility-reset-safe-v2'
CAMS = ['agentview', 'robot0_eye_in_hand']


def episode_samples(frames, episodes, count=2):
    return np.unique(np.concatenate([
        ks[np.linspace(0, len(ks)-1, count, dtype=int)]
        for e in np.unique(episodes[frames])
        for ks in [frames[episodes[frames] == e]]
    ]))


def policy_mask(seg, gid):
    import mujoco
    from drift_resize_order import resize_rgb
    visible = (seg[..., 0] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 1] == gid)
    return resize_rgb(np.repeat((visible.astype(np.uint8)*255)[..., None], 3, -1), 224)[..., 0] >= 128


def render_positions(env, positions, z, patch_name, observation=None, with_masks=False):
    """All references to the renderer expire BEFORE the caller can reset env.

    Keeping an old ctx alive across reset defers its GL cleanup until the new
    context is current, corrupting the new renderer in robosuite 1.4.
    """
    import mujoco
    sim = env.env.sim
    model, data = sim.model._model, sim.data._data
    ctx = sim._render_context_offscreen
    assert ctx.model is sim.model and ctx.data is sim.data
    ctx.gl_ctx.make_current()
    gid = sim.model.geom_name2id(patch_name)
    bid = sim.model.body_name2id(patch_name+'_body')
    home = model.body_pos[bid].copy()
    if observation is not None:
        from robosuite import macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        for cam in CAMS:
            ctx.render(256, 256, camera_id=sim.model.camera_name2id(cam))
            rgb = ctx.read_pixels(256, 256)[::IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]]
            assert np.array_equal(rgb, observation[cam+'_image']), ('RGB/native observation mismatch', cam)
    counts = np.zeros((len(positions), len(CAMS)), np.int32)
    masks = []
    try:
        for j, (x, y) in enumerate(positions):
            model.body_pos[bid] = [x, y, z]
            mujoco.mj_kinematics(model, data)
            views = []
            for c, cam in enumerate(CAMS):
                ctx.render(256, 256, camera_id=sim.model.camera_name2id(cam), segmentation=True)
                seg = np.ascontiguousarray(ctx.read_pixels(256, 256, segmentation=True)[::-1, ::-1])
                mask = policy_mask(seg, gid)
                counts[j, c] = int(mask.sum())
                if with_masks: views.append(mask)
            if with_masks: masks.append(views)
    finally:
        model.body_pos[bid] = home
        mujoco.mj_kinematics(model, data)
    return counts, np.array(masks) if with_masks else None


def fixed_masks(env, state, patch_name):
    """Independent reference: native sim.render, patch baked into the XML."""
    env.regenerate_obs_from_state(state)
    sim = env.env.sim
    sim._render_context_offscreen.gl_ctx.make_current()
    gid = sim.model.geom_name2id(patch_name)
    return np.array([policy_mask(np.ascontiguousarray(sim.render(
        width=256, height=256, camera_name=cam, segmentation=True)[::-1, ::-1]), gid) for cam in CAMS])


def candidates(cfg):
    plane, keep = cfg['plane'], cfg['keepout']
    h = PATCH/2
    out = []
    for x in np.arange(-.45, .4501, STEP):
        for y in np.arange(-.57, .5701, STEP):
            inside = plane['bounds_u'][0] <= x-h and x+h <= plane['bounds_u'][1] and plane['bounds_v'][0] <= y-h and y+h <= plane['bounds_v'][1]
            hits = [n for n, b in keep.items() if x+h > b['x'][0]-MARGIN and x-h < b['x'][1]+MARGIN
                    and y+h > b['y'][0]-MARGIN and y-h < b['y'][1]+MARGIN]
            if inside and not hits: out.append((round(float(x), 4), round(float(y), 4)))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--pool', required=True, help='npz with sim_states/env_seed/ep_id (stage-A pool or base.npz)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: first N training frames, first 40 candidates')
    ap.add_argument('--smoke', action='store_true', help='first/middle/last frame in EVERY training episode, 40 spread candidates')
    args = ap.parse_args()
    import mujoco, yaml
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb
    import scene_patch as sp
    cfg = yaml.safe_load((D.PROBE/'config/scene.yaml').read_text()); cfg['patch']['size_wh'] = [PATCH]*2
    z = float(np.load(D.OUT/'drift_patch_drift.npz')['cell_world'][2])
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if D.pathlib.Path(suite.get_task(i).bddl_file).stem == args.task)
    bddl = D.pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    with np.load(args.pool, allow_pickle=False) as p:
        states, seeds, ep = p['sim_states'], p['env_seed'], p['ep_id']
    frames = np.flatnonzero(ep < 10)
    cand = candidates(cfg)
    if args.limit: frames, cand = frames[:args.limit], cand[:40]
    if args.smoke:
        frames = episode_samples(frames, ep, 3)
        cand = cand[np.linspace(0, len(cand)-1, min(40, len(cand)), dtype=int)]
    dest = D.pathlib.Path(args.out); dest.mkdir(parents=True, exist_ok=True)
    tex = dest/'random.png'
    Image.fromarray(np.random.RandomState(20261007).randint(0, 256, (128, 128, 3)).astype(np.uint8)).save(tex)
    print('[scan]', args.task, 'frames', len(frames), 'candidates', len(cand), flush=True)

    def masks(env, positions, ks):
        """visible px [len(ks), len(positions), 2] with the patch moved to each position."""
        out = np.zeros((len(ks), len(positions), 2), np.int32); previous = None
        for i, k in enumerate(ks):
            reset = seeds[k] != previous
            if reset:
                env.seed(int(seeds[k])); env.reset(); previous = seeds[k]
            obs = env.regenerate_obs_from_state(states[k])
            out[i] = render_positions(env, positions, z, cfg['patch']['name'], obs if reset else None)[0]
        return out

    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    home = np.load(D.OUT/'drift_patch_drift.npz')['cell_world']
    env.env.set_xml_processor(sp.make_xml_processor(cfg, home, str(tex)))
    try:
        vis = np.zeros((len(frames), len(cand), 2), np.int32)
        for s in range(0, len(frames), 20):
            vis[s:s+20] = masks(env, cand, frames[s:s+20])
            print('[scan]', args.task, min(s+20, len(frames)), '/', len(frames), flush=True)
    finally:
        env.close()
    # Independent fresh scenes: first/last sampled frame of EVERY training episode.
    # Compare full masks, and also ensure rerenders match the original scan areas.
    check = []
    rng = np.random.RandomState(0)
    js = sorted(set([int(np.argmax(vis[..., 0].mean(0))), int(np.argmax(vis[..., 1].mean(0))), int(rng.randint(len(cand)))]))
    ks = episode_samples(frames, ep)
    for j in js:
        fresh, moved, mask_diff, moved_masks = [], [], [], {}
        moving = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        moving.env.set_xml_processor(sp.make_xml_processor(cfg, home, str(tex)))
        previous = None
        try:
            for k in ks:
                if seeds[k] != previous:
                    moving.seed(int(seeds[k])); moving.reset(); previous = seeds[k]
                obs = moving.regenerate_obs_from_state(states[k])
                area, rendered = render_positions(moving, [cand[j]], z, cfg['patch']['name'], obs, True)
                moved_masks[int(k)] = rendered[0]
                assert np.array_equal(area[0], vis[np.searchsorted(frames, k), j]), ('scan/rerender', j, int(k))
        finally:
            moving.close()
        for seed in np.unique(seeds[ks]):
            # New environments for each seed eliminate reset lifetime coupling in the reference.
            reference = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
            reference.env.set_xml_processor(sp.make_xml_processor(cfg, [*cand[j], z], str(tex)))
            try:
                reference.seed(int(seed)); reference.reset()
                for k in ks[seeds[ks] == seed]:
                    fixed_mask = fixed_masks(reference, states[k], cfg['patch']['name'])
                    fresh.append(fixed_mask.sum(axis=(1, 2)).tolist())
                    moved.append(vis[np.searchsorted(frames, k), j].tolist())
                    mask_diff.append(int(np.count_nonzero(moved_masks[int(k)] != fixed_mask)))
            finally:
                reference.close()
        check.append({'candidate': j, 'world': cand[j].tolist(), 'frames': ks.tolist(),
                      'episodes': ep[ks].tolist(), 'fresh': fresh, 'moved': moved,
                      'max_abs_diff': int(np.abs(np.array(fresh)-moved).max()),
                      'mask_disagreement_pixels_max': max(mask_diff)})
        print('[check]', check[-1], flush=True)
        assert check[-1]['max_abs_diff'] == 0 and max(mask_diff) == 0, ('fresh-scene validation failed', check[-1])
    np.savez_compressed(dest/'visibility.npz', candidates=cand, z=z, frames=frames, ep_id=ep[frames], base_px=vis[..., 0], wrist_px=vis[..., 1])
    (dest/'visibility.json').write_text(json.dumps({'schema': SCHEMA, 'validated': True,
        'source_sha256': hashlib.sha256(D.pathlib.Path(__file__).read_bytes()).hexdigest(),
        'smoke': bool(args.smoke or args.limit), 'task': args.task, 'pool': args.pool, 'frames': int(len(frames)),
        'candidates': int(len(cand)), 'grid_step_m': STEP, 'keepout_margin_m': MARGIN, 'patch_m': PATCH,
        'spot_check': check}, indent=2)+'\n')
    print('[done]', args.task, flush=True)


if __name__ == '__main__':
    main()
