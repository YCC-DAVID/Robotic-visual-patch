"""Replay the existing clean trajectory to measure physical texture visibility.

No policy inference or patch optimization. Sin/cos phase pairs normalized by
uniform white/black pairs estimate local texture modulation through native
MuJoCo rendering and the production 256->224 resize, in two world directions.
"""
import argparse
import json
import pathlib

import drift_eval_digital as D  # LIBERO paths/environment, no model load
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='0.06,0.13')
    ap.add_argument('--periods-mm', default='2,4,8,16,32')
    ap.add_argument('--out', default='view_scale_probe')
    args = ap.parse_args()
    import yaml
    from PIL import Image, ImageDraw
    from scipy.ndimage import binary_erosion
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scene_patch as sp
    from dump_shared_frame import cam_intrinsics, cam_extrinsics
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from drift_resize_order import resize_rgb

    dest = D.OUT / args.out
    dest.mkdir(exist_ok=True)
    traj = np.load(D.OUT / ('traj_' + D.TASK + '.npz'), allow_pickle=False)
    states = [traj['f{:03d}__flatten'.format(k)] for k in range(int(traj['n_frames']))]
    ts = traj['ts'].astype(int)
    seed = int(traj['shared_seed'])
    scores = np.load(D.OUT / 'percell_scores_plate.npz', allow_pickle=True)
    legal = np.where(scores['legal'])[0]
    anchor = int(legal[np.argmax(scores['influence'][legal])])
    xy = scores['anchor_world'][anchor]
    cfg = yaml.safe_load((D.PROBE / 'config' / 'scene.yaml').read_text())
    plane = sp.Plane.from_cfg(cfg['plane'])
    lift = cfg['patch']['thickness'] / 2 + cfg['patch']['normal_offset']
    world = plane.to_world(xy[0], xy[1], lift)
    surface = world.copy(); surface[2] += cfg['patch']['thickness'] / 2
    suite = benchmark.get_benchmark_dict()['libero_goal']()
    task = next(suite.get_task(i) for i in range(suite.n_tasks)
                if pathlib.Path(suite.get_task(i).bddl_file).stem == D.TASK)
    bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    views = {'base': 'agentview', 'wrist': 'robot0_eye_in_hand'}
    periods = [float(v) for v in args.periods_mm.split(',')]
    sizes = [float(v) for v in args.sizes.split(',')]
    selected = [0, 4, 6, 8, 10, 13, 15]
    results = {'metadata': {'task': D.TASK, 'trajectory': 'traj_' + D.TASK, 'trajectory_success': bool(traj['success']),
                            'sampling_steps': ts.tolist(), 'anchor_index': anchor, 'world_center': world.tolist(),
                            'sizes_m': sizes, 'periods_mm': periods, 'texture_resolution': 1024,
                            'model_input_resolution': 224, 'robot_policy_evaluated': False,
                            'measure': 'sqrt(mean(cos_response^2+sin_response^2)), normalized by uniform bright-dark rendering',
                            'limitations': 'One clean trajectory; modulation retention is not action influence. Aliasing remains possible.'}, 'sizes': {}}

    def project(point, K, E):
        xyz = np.linalg.inv(E) @ np.r_[point, 1]
        return (K @ xyz[:3])[:2] / xyz[2], xyz[2]

    def render(texture, use_seg=False):
        texpath = dest / 'current_texture.png'
        Image.fromarray(texture).save(texpath)
        kw = {'bddl_file_name': str(bddl), 'camera_heights': 256, 'camera_widths': 256}
        if use_seg:
            kw['camera_segmentations'] = 'element'
        env = OffScreenRenderEnv(**kw)
        env.env.set_xml_processor(sp.make_xml_processor(cfg, world, str(texpath)))
        env.seed(seed); env.reset()
        frames = {v: [] for v in views}
        masks = {v: [] for v in views}
        geometry = {v: [] for v in views}
        gid = sp.patch_geom_id(env, cfg)
        try:
            for k, state in enumerate(states):
                obs = env.regenerate_obs_from_state(state)
                for view, cam in views.items():
                    im = np.ascontiguousarray(obs[cam + '_image'][::-1, ::-1])
                    frames[view].append(resize_rgb(im, 224))
                    if use_seg:
                        raw = obs[cam + '_segmentation_element'][..., 0] == gid
                        m256 = np.ascontiguousarray(raw[::-1, ::-1])
                        alpha = np.asarray(Image.fromarray(m256.astype(np.uint8)*255).resize((224,224), Image.BILINEAR))/255
                        masks[view].append(binary_erosion(alpha > .99, iterations=2))
                        K, fovy = cam_intrinsics(env, cam, 256)
                        E = cam_extrinsics(env, cam)
                        uv, depth = project(surface, K, E)
                        jac = np.column_stack([(project(surface + d, K, E)[0] - project(surface - d, K, E)[0]) / 2
                                               for d in [np.array([.001,0,0]), np.array([0,.001,0])]]) * (224/256)
                        # With oblique viewing, a coordinate's stripe normal has
                        # scale 1/|J^-T e|, rather than simply |J e|.
                        invt = np.linalg.inv(jac).T
                        normal_scales = 1 / np.linalg.norm(invt, axis=0)
                        geometry[view].append({'step': int(ts[k]), 'distance_m': float(np.linalg.norm(E[:3,3]-surface)),
                                               'depth_m': float(depth), 'fovy_deg': float(fovy),
                                               'center_uv224_unflipped': ((uv+.5)*224/256-.5).tolist(),
                                               'center_in_frame': bool(depth>0 and (uv>=0).all() and (uv<256).all()),
                                               'visible_px256': int(m256.sum()), 'visible_area224': float(alpha.sum()),
                                               'interior_px224': int(masks[view][-1].sum()),
                                               'stripe_pixels_per_mm': normal_scales.tolist(),
                                               'jacobian_px_per_mm': jac.tolist()})
        finally:
            env.close()
        return {v: np.array(x) for v,x in frames.items()}, masks, geometry

    for size in sizes:
        cfg['patch']['size_wh'] = [size, size]
        label = '{:.2f}'.format(size)
        print('[size]', label, 'anchor', anchor, 'world', world.tolist(), flush=True)
        low, _, _ = render(np.full((1024,1024,3), 25, np.uint8))
        high, masks, geometry = render(np.full((1024,1024,3), 229, np.uint8), use_seg=True)
        modulation = {v: {} for v in views}
        coords = ((np.arange(1024)+.5)/1024-.5)*size*1000
        montage = Image.new('RGB', (224*len(selected), 248*2), 'white')
        draw = ImageDraw.Draw(montage)
        for period in periods:
            response = {v: [] for v in views}
            for phase in [0, np.pi/2]:
                cx = np.broadcast_to(np.cos(2*np.pi*coords[None,:]/period+phase), (1024,1024))
                cy = np.broadcast_to(np.cos(2*np.pi*coords[:,None]/period+phase), (1024,1024))
                pattern = np.stack([cx,cy,np.zeros_like(cx)], -1)
                plus, _, _ = render(np.rint(127+102*pattern).clip(0,255).astype(np.uint8))
                minus, _, _ = render(np.rint(127-102*pattern).clip(0,255).astype(np.uint8))
                for view in views:
                    delta = plus[view].astype(float)-minus[view].astype(float)
                    norm = high[view].astype(float)-low[view].astype(float)
                    pair = []
                    for k in range(len(states)):
                        axes = []
                        for ch in [0,1]:
                            valid = masks[view][k] & (norm[k,:,:,ch]>40)
                            value = float(np.mean((delta[k,:,:,ch][valid]/norm[k,:,:,ch][valid])**2)) if valid.sum()>=16 else None
                            axes.append(value)
                        pair.append(axes)
                    response[view].append(pair)
                    if period == 4 and phase == 0:
                        for j,k in enumerate(selected):
                            y = 0 if view=='base' else 248
                            montage.paste(Image.fromarray(plus[view][k]), (j*224,y+24))
                            draw.text((j*224+4,y+4), '{} step {} / 4mm'.format(view,int(ts[k])), fill='black')
            for view in views:
                modulation[view][str(period)] = [[None if response[view][0][k][ch] is None else
                                                  float(np.sqrt(response[view][0][k][ch]+response[view][1][k][ch]))
                                                  for ch in [0,1]] for k in range(len(states))]
            print('[period]', label, period, 'mm done', flush=True)
        results['sizes'][label] = {'geometry': geometry, 'modulation': modulation}
        montage.save(dest / ('montage_' + label + '.png'))
        (dest / 'results.json').write_text(json.dumps(results, indent=2))
        for view in views:
            geometry[view] = list(geometry[view])

    fig, axes = plt.subplots(3, len(sizes), figsize=(7*len(sizes),11), squeeze=False)
    for j,size in enumerate(sizes):
        row = results['sizes']['{:.2f}'.format(size)]
        for view,color in [('base','tab:blue'),('wrist','tab:orange')]:
            gs = row['geometry'][view]
            axes[0,j].plot(ts,[g['visible_area224'] for g in gs],'-o',label=view,color=color)
            axes[1,j].plot(ts,[min(g['stripe_pixels_per_mm'])*4 for g in gs],'-o',label=view,color=color)
        for period,style in [(2,':'),(4,'-'),(8,'--')]:
            for view,color in [('base','tab:blue'),('wrist','tab:orange')]:
                vals = [np.nan if any(x is None for x in p) else np.mean(p) for p in row['modulation'][view][str(float(period))]]
                axes[2,j].plot(ts,vals,linestyle=style,label='{} {}mm'.format(view,period),color=color)
        axes[0,j].set_title('Physical patch {}cm at anchor {}'.format(int(size*100),anchor))
        axes[0,j].set_ylabel('Visible area at 224 (pixels)')
        axes[1,j].set_ylabel('4mm stripe period at center\n(smaller direction, pixels)')
        axes[2,j].set_ylabel('Normalized modulation retention\n(phase-paired, interior only)')
        axes[2,j].set_xlabel('Environment step (clean replay)')
        for i in range(3):
            axes[i,j].axvspan(45,55,color='gray',alpha=.12)
            axes[i,j].grid(alpha=.2); axes[i,j].legend(fontsize=8)
        axes[1,j].axhline(2,color='black',linestyle=':',alpha=.5)
    fig.suptitle('Two-view texture sampling during approach / grasp / transport\nGray band: gripper closing; modulation is not policy influence')
    fig.tight_layout(rect=[0,0,1,.95]); fig.savefig(dest/'scale_and_modulation.png',dpi=160)
    lines = ['Two-view physical texture replay; robot policy not evaluated.',
             'Anchor {} world {}; one existing successful trajectory, 16 decision states.'.format(anchor,world.tolist()),
             'Normalized modulation uses sin/cos phase pairs and uniform bright/dark reference.']
    for label,row in results['sizes'].items():
        lines += ['', 'Patch {}m'.format(label), 'step view visible224 distance_m center_visible 4mm_px_min retention_4mm_mean']
        for k,t in enumerate(ts):
            for view in views:
                g=row['geometry'][view][k]; r=row['modulation'][view]['4.0'][k]
                retention = '--' if any(x is None for x in r) else '{:.3f}'.format(np.mean(r))
                lines.append('{} {} {:.1f} {:.3f} {} {:.2f} {}'.format(t,view,g['visible_area224'],g['distance_m'],
                            g['center_in_frame'],min(g['stripe_pixels_per_mm'])*4,retention))
    (dest/'results.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
