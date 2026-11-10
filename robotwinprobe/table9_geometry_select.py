"""Choose exposed, similarly sized screen footprints without attack outcomes.

Dense screening uses clean depth only; an independent actual-actor segmentation
pass must verify every selected center before a training plan can be frozen.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from radius_plan import dump


CAMERAS = ['head_camera', 'left_camera', 'right_camera']
SIZE = .10
RADIUS = .03


def poses():
    offsets = [[0., 0.]] + [[RADIUS*np.cos(a), RADIUS*np.sin(a)] for a in np.arange(8)*np.pi/4]
    return [(xy, yaw) for xy in offsets for yaw in [0., -15., 15.]]


def footprint(frame, camera, centers, yaws):
    """Texture-uniform samples estimate exposure; projected corners give area."""
    k, e = frame[camera + '_intrinsic'], frame[camera + '_extrinsic']
    depth = frame[camera + '_depth']
    valid_depth = frame[camera + '_depth_valid']
    axis = (np.arange(16) + .5)/16 - .5
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    local = np.stack([xx.ravel(), yy.ravel()], -1) * SIZE
    angles = np.deg2rad(yaws)
    c, s = np.cos(angles), np.sin(angles)
    rot = np.stack([c, -s, s, c], -1).reshape(-1, 2, 2)
    world_xy = local[None] @ rot.transpose(0, 2, 1) + centers[:, None, :2]
    world = np.concatenate([world_xy, np.full((*world_xy.shape[:2], 1), .741)], -1)
    cv = world @ e[:, :3].T + e[:, 3]
    projected = cv @ k.T
    z = cv[..., 2]
    uv = projected[..., :2] / np.maximum(projected[..., 2:], 1e-8)
    inside = (z > 0) & (uv[..., 0] >= 0) & (uv[..., 0] <= depth.shape[1]-1) & (uv[..., 1] >= 0) & (uv[..., 1] <= depth.shape[0]-1)
    ix = np.rint(uv[..., 0]).clip(0, depth.shape[1]-1).astype(int)
    iy = np.rint(uv[..., 1]).clip(0, depth.shape[0]-1).astype(int)
    visible = inside & valid_depth[iy, ix] & (depth[iy, ix] >= z - .002)
    # Plane projection Jacobian is proportional to inverse camera-depth cubed.
    weights = np.where(z > 0, 1 / np.maximum(z, 1e-5)**3, 0)
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
    fraction = (visible * weights).sum(1)
    in_frame = (inside * weights).sum(1)
    corners = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]]) * SIZE
    cw = corners[None] @ rot.transpose(0, 2, 1) + centers[:, None, :2]
    cw = np.concatenate([cw, np.full((*cw.shape[:2], 1), .741)], -1)
    cc = cw @ e[:, :3].T + e[:, 3]
    cp = cc @ k.T
    cuv = cp[..., :2] / np.maximum(cp[..., 2:], 1e-8)
    x, y = cuv[..., 0], cuv[..., 1]
    area = abs((x*np.roll(y, -1, axis=1)-y*np.roll(x, -1, axis=1)).sum(1))/2
    area = np.where((cc[..., 2] > 0).all(1), area, 0)
    return np.stack([fraction, in_frame, area*.49, fraction*area*.49], -1).astype(np.float32)


def load_frames(root):
    frames = []
    for path in sorted((root / 'geometry/clean').glob('seed*.npz')):
        with np.load(path) as z:
            for i, query in enumerate(z['query']):
                frame = {key: z[key][i].copy() for key in z.files if key not in ['seed']}
                frame.update(seed=int(z['seed']), query=int(query))
                frames.append(frame)
    return frames


def select(root):
    frames = load_frames(root)
    assert {f['seed'] for f in frames} == set(list(range(10100,10108)) + [10110,10111])
    # All possible EOT-rotated corners stay on the 1.2m x .7m tabletop.
    xx, yy = np.meshgrid(np.arange(-.48,.4801,.04), np.arange(-.24,.2401,.04))
    candidates = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size,.741)], -1)
    pp = poses()
    centers = np.array([[c[0]+xy[0], c[1]+xy[1], .741] for c in candidates for xy,yaw in pp])
    yaws = np.array([yaw for c in candidates for xy,yaw in pp])
    cache = root / 'geometry/screening.npz'
    if cache.exists():
        with np.load(cache) as z:
            assert np.array_equal(z['candidates'], candidates)
            stats = z['head_metrics'].copy()
    else:
        stats = []
        for fi, frame in enumerate(frames):
            chunks = [footprint(frame, 'head_camera', centers[i:i+256], yaws[i:i+256]) for i in range(0,len(centers),256)]
            stats.append(np.concatenate(chunks).reshape(len(candidates),len(pp),4))
            if fi % 5 == 0:
                dump(root / 'geometry/screen_progress.json', dict(frames=fi+1,total=len(frames)))
                print('[screen]',fi+1,len(frames),flush=True)
        stats = np.stack(stats, axis=1)
        np.savez_compressed(cache, candidates=candidates, head_metrics=stats,
            frame_seeds=[f['seed'] for f in frames], frame_queries=[f['query'] for f in frames])
    train = np.array([f['seed'] < 10110 for f in frames])
    a = stats[:,train]
    exposed = a[...,0].reshape(len(candidates),-1)
    in_frame = a[...,1].reshape(len(candidates),-1)
    area = a[...,3].mean((1,2))
    minimum = exposed.min(1)
    average = exposed.mean(1)
    # Match the SAME unobstructed state subset across all sites. Original
    # all-frame strict-screen failure is retained separately, never overwritten.
    frame_ok = (a[...,0].min(2) >= .90) & (a[...,1].min(2) >= .99)
    allowed = (frame_ok.mean(1) >= .90) & (area >= 20)
    eligible = np.flatnonzero(allowed)
    best = None
    for lower in area[eligible]:
        band = eligible[(area[eligible] >= lower-1e-6) & (area[eligible] <= lower*1.25+1e-6)]
        if len(band) < 9:
            continue
        for first in range(len(band)):
            ids = [int(band[first])]
            common = frame_ok[ids[0]].copy()
            for _ in range(8):
                distance = np.linalg.norm(candidates[band,None,:2]-candidates[ids][None,:,:2],axis=-1).min(1)
                intersection = (frame_ok[band] & common).mean(1)
                usable = (distance >= .08-1e-6) & (intersection >= .85)
                if not usable.any():break
                score = np.where(usable,distance+.001*intersection,-1)
                pick = int(band[score.argmax()]);ids.append(pick);common &= frame_ok[pick]
            if len(ids) != 9:continue
            points = candidates[ids,:2]
            d = np.linalg.norm(points[:,None]-points[None],axis=-1)
            d += np.eye(9)*100
            if d.min() < .08-1e-6:
                continue
            span = np.ptp(points,axis=0)
            retained_seeds = {f['seed'] for f,keep in zip([f for f in frames if f['seed']<10110],common) if keep}
            if retained_seeds != set(range(10100,10108)):continue
            score = float(span[0]*span[1] + .0001*common.sum())
            if best is None or score > best[0]:
                best = (score,ids,common)
    thresholds = dict(minimum_head_exposure_common_train_frames_and_probes=.90,
        minimum_fraction_common_training_frames=.85,all_eight_train_seeds_required=True,minimum_head_in_frame=.99,
        maximum_ratio_mean_head_visible_pixels_between_sites=1.25,minimum_center_separation_m=.08,
        depth_visibility_tolerance_m=.002,screening_texture_samples=256,
        note='Geometry-only shared frame filter, identical across all nine sites; no attack outcomes. Depth screen requires actual segmentation verification.')
    output = dict(feasible=best is not None, candidates=len(candidates), eligible_candidates=len(eligible),
        train_frames=int(train.sum()), validation_frames=int((~train).sum()), eot_boundary_probes=len(pp),
        thresholds=thresholds, centers=[], single_patch_per_scene=True)
    if best is not None:
        common = best[2]
        common_frames = [f for f,keep in zip([f for f in frames if f['seed']<10110],common) if keep]
        output['training_common_frames'] = [dict(seed=f['seed'],query=f['query']) for f in common_frames]
        output['training_common_frame_count'] = len(common_frames)
        ids = sorted(best[1], key=lambda j:(round(candidates[j,1],5),round(candidates[j,0],5)))
        for i,j in enumerate(ids):
            output['centers'].append(dict(id=f'site{i:02d}',candidate_index=int(j),center_m=candidates[j].tolist(),
                mean_head_visible_pixels=float(a[j,common,:,3].mean()),minimum_head_exposure=float(a[j,common,:,0].min()),
                mean_head_exposure=float(a[j,common,:,0].mean()),all_original_frames_minimum_exposure=float(minimum[j]),
                validation_min_head_exposure=float(stats[j,~train,:,0].min())))
        common_area = [s['mean_head_visible_pixels'] for s in output['centers']]
        output['head_visible_pixel_ratio'] = float(max(common_area)/min(common_area))
        output['center_span_m'] = np.ptp(candidates[ids,:2],axis=0).tolist()
        # Report wrist visibility separately rather than assuming equal visibility there.
        for site in output['centers']:
            wrist = {}
            nominal = np.asarray([site['center_m']])
            for camera in CAMERAS[1:]:
                values = np.array([footprint(f,camera,nominal,np.zeros(1))[0] for f in frames])
                wrist[camera] = dict(mean_visible_pixels=float(values[train][common,3].mean()),
                    fraction_train_frames_visible=float((values[train][common,3]>1).mean()),
                    frame_pool='shared unobstructed training frames')
            site['wrist_nominal'] = wrist
    dump(root / 'geometry/selection.json', output)
    print(json.dumps(output,ensure_ascii=False,indent=2))
    plot(root, candidates, area, allowed, output)


def plot(root,candidates,area,allowed,selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Circle
    fig,ax=plt.subplots(figsize=(10,6))
    ax.add_patch(Rectangle((-.6,-.35),1.2,.7,fill=False,color='black'))
    ax.scatter(candidates[~allowed,0],candidates[~allowed,1],s=10,color='lightgray',label='Fails exposure screen')
    im=ax.scatter(candidates[allowed,0],candidates[allowed,1],c=area[allowed],s=18,cmap='viridis')
    for site in selection['centers']:
        x,y,_=site['center_m']
        ax.add_patch(Rectangle((x-.05,y-.05),.10,.10,fill=False,color='red',linewidth=1.5))
        ax.add_patch(Circle((x,y),.03,fill=False,color='red',linestyle='--',alpha=.7))
        ax.text(x,y+.065,site['id'],ha='center',fontsize=9)
    ax.set(xlim=(-.65,.65),ylim=(-.40,.40),xlabel='World X (m)',ylabel='World Y (m)',aspect='equal')
    ax.set_title('Proposed independent training centers (layout only; one patch per scene)')
    fig.colorbar(im,ax=ax,label='Estimated visible head pixels at model input')
    ax.legend(loc='upper right');fig.tight_layout()
    fig.savefig(root/'geometry/site_selection.png',dpi=160);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    select(Path(parser.parse_args().root).resolve())
