"""CPU audit of fixed 13cm LIBERO patch exposure; no model/test outcomes used.

Observed masks come from existing real-render calibration. Homographies rebuild
an unoccluded *top-face* projection, separating projection loss from occlusion.
Thin box sides and raster boundaries can add a few observed pixels outside this
ideal. Alternative positions use only existing training scans (inits0-9).
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
TASKS = ['put_the_bowl_on_the_plate', 'put_the_bowl_on_top_of_the_cabinet',
         'put_the_wine_bottle_on_the_rack', 'turn_on_the_stove',
         'put_the_bowl_on_the_stove', 'put_the_cream_cheese_in_the_bowl',
         'open_the_middle_drawer_of_the_cabinet']
STRONG = {'put_the_bowl_on_the_plate', 'put_the_cream_cheese_in_the_bowl'}
LABELS = dict(put_the_bowl_on_the_plate='bowl -> plate',
              put_the_bowl_on_top_of_the_cabinet='bowl -> cabinet top',
              put_the_wine_bottle_on_the_rack='wine bottle -> rack',
              turn_on_the_stove='turn on stove', put_the_bowl_on_the_stove='bowl -> stove',
              put_the_cream_cheese_in_the_bowl='cheese -> bowl',
              open_the_middle_drawer_of_the_cabinet='open middle drawer')
QUANTILES = [0, 10, 25, 50, 75, 90, 100]
PIXELS = np.stack(np.meshgrid(np.arange(256), np.arange(256)), -1)
PIXELS = np.concatenate([PIXELS, np.ones((256, 256, 1))], -1)
CORNERS = np.array([[-.5, -.5, 1], [127.5, -.5, 1],
                    [127.5, 127.5, 1], [-.5, 127.5, 1]])
BILINEAR = getattr(Image, 'Resampling', Image).BILINEAR


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(path).with_suffix(Path(path).suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, names)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def percentiles(values):
    values = np.asarray(values)
    if not len(values):
        return None
    return {str(q): float(v) for q, v in zip(QUANTILES, np.percentile(values, QUANTILES))}


def exposure(values, ep):
    values, ep = np.asarray(values), np.asarray(ep)
    result = dict(frames=len(values), mean_px=float(values.mean()), area_percentiles_px=percentiles(values),
                  zero_fraction=float((values == 0).mean()))
    for threshold in [1, 50, 100, 300]:
        episode_values = [float((values[ep == e] >= threshold).mean()) for e in np.unique(ep)]
        result[f'coverage_ge{threshold}'] = float((values >= threshold).mean())
        result[f'episode_macro_coverage_ge{threshold}'] = float(np.mean(episode_values))
        result[f'episode_min_coverage_ge{threshold}'] = float(min(episode_values))
    return result


def projection(H):
    """Top-face policy mask and projected full area; no scene depth/occluders."""
    uvh = PIXELS@H.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = uvh[..., :2]/uvh[..., 2:]
    ideal256 = ((uvh[..., 2] > 0) & np.isfinite(uv).all(-1)
                & (uv[..., 0] >= -.5) & (uv[..., 0] <= 127.5)
                & (uv[..., 1] >= -.5) & (uv[..., 1] <= 127.5))
    ideal = np.asarray(Image.fromarray(ideal256.astype(np.uint8)*255).resize((224, 224), BILINEAR)) >= 128
    hp = CORNERS@np.linalg.inv(H).T
    front = hp[:, 2] > 0
    full_area = None
    polygon = None
    if front.all():
        polygon = hp[:, :2]/hp[:, 2:]
        full_area = float(abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1))
                              - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))*.5*(224/256)**2)
    return ideal, dict(ideal_in_frame_px=int(ideal.sum()), full_top_face_projected_area_px=full_area,
                       corners_in_front=int(front.sum()), camera_plane_crossing=bool(front.any() and not front.all()))


def cause(observed, geometry):
    ideal = geometry['ideal_in_frame_px']
    full = geometry['full_top_face_projected_area_px']
    if observed >= 50:
        return 'visible_ge50'
    if ideal >= 50:
        return 'occluded_despite_sufficient_projection'
    if geometry['corners_in_front'] == 0:
        return 'behind_camera'
    if full is not None and full < 50:
        return 'projected_too_small'
    if ideal == 0:
        return 'out_of_frame' if not geometry['camera_plane_crossing'] else 'camera_plane_crossing_out_of_view'
    return 'edge_clipped_to_lt50' if full is None or full >= 50 else 'projected_too_small'


def geometric_audit(Hs, observed_masks=None, observed_counts=None, offset=None, keep=None):
    rows, saved = [], {}
    if keep is None:
        keep = set()
    for k, H in enumerate(Hs):
        if offset is not None:
            du, dv = 128*offset[0]/.13, -128*offset[1]/.13
            H = np.array([[1., 0, -du], [0, 1., -dv], [0, 0, 1.]])@H
        ideal, g = projection(H)
        observed = int(observed_masks[k].sum()) if observed_masks is not None else int(observed_counts[k])
        if observed_masks is not None:
            missing = int(np.count_nonzero(ideal & ~observed_masks[k]))
            outside = int(np.count_nonzero(observed_masks[k] & ~ideal))
        else:
            missing = max(0, int(ideal.sum())-observed)
            outside = None
        g.update(visible_px=observed, missing_top_face_px=missing,
                 observed_outside_top_face_px=outside,
                 top_face_hidden_fraction=missing/max(1, int(ideal.sum())),
                 low_visibility_cause=cause(observed, g))
        rows.append(g)
        if k in keep:
            saved[k] = ideal
    return rows, saved


def geometry_summary(rows, selected):
    chosen = [rows[int(k)] for k in np.flatnonzero(selected)]
    categories = sorted({r['low_visibility_cause'] for r in chosen})
    zero = [r for r in chosen if r['visible_px'] == 0]
    substantial = [r for r in chosen if r['ideal_in_frame_px'] >= 50]
    return dict(frames=len(chosen),
        ideal_area_percentiles_px=percentiles([r['ideal_in_frame_px'] for r in chosen]),
        sufficient_projection_fraction=float(np.mean([r['ideal_in_frame_px'] >= 50 for r in chosen])),
        severe_occlusion_fraction=float(np.mean([r['ideal_in_frame_px'] >= 50 and r['top_face_hidden_fraction'] >= .5 for r in chosen])),
        hidden_fraction_if_projection_ge50_percentiles=percentiles([r['top_face_hidden_fraction'] for r in substantial]),
        low_visibility_cause_counts={c: sum(r['low_visibility_cause'] == c for r in chosen) for c in categories},
        zero_pixel_frames=len(zero), zero_pixel_projection_loss_frames=sum(r['ideal_in_frame_px'] < 50 for r in zero),
        zero_pixel_occluded_frames=sum(r['ideal_in_frame_px'] >= 50 for r in zero),
        top_face_caveat='Thin visual box sides and raster edges are omitted from the ideal; large missing fractions identify occlusion, not subpixel edge differences.')


def candidate_table(task, scan):
    base, wrist, ep = scan['base_px'], scan['wrist_px'], scan['ep_id']
    if not np.all((ep >= 0) & (ep < 10)):
        raise ValueError('Candidate selection accepts only training episodes0-9: '+task)
    base_cov = (base >= 300).mean(0)
    base_min_ep = np.min([(base[ep == e] >= 300).mean(0) for e in np.unique(ep)], axis=0)
    wrist_macro = np.mean([(wrist[ep == e] >= 100).mean(0) for e in np.unique(ep)], axis=0)
    wrist_min = np.min([(wrist[ep == e] >= 100).mean(0) for e in np.unique(ep)], axis=0)
    combined = np.minimum(base, 2000)+np.minimum(wrist, 2000)
    rows = []
    for j, world in enumerate(scan['candidates']):
        rows.append(dict(task=task, candidate=j, world=world.tolist(),
            eligible_original=bool(base_cov[j] >= .9), eligible_robust=bool(base_cov[j] >= .95 and base_min_ep[j] >= .9),
            base_mean_px=float(base[:, j].mean()), base_p10_px=float(np.percentile(base[:, j], 10)),
            base_coverage_ge300=float(base_cov[j]), base_min_episode_coverage_ge300=float(base_min_ep[j]),
            wrist_mean_px=float(wrist[:, j].mean()), wrist_median_px=float(np.median(wrist[:, j])),
            wrist_zero_fraction=float((wrist[:, j] == 0).mean()), wrist_coverage_ge100=float((wrist[:, j] >= 100).mean()),
            wrist_episode_macro_coverage_ge100=float(wrist_macro[j]), wrist_min_episode_coverage_ge100=float(wrist_min[j]),
            wrist_p10_px=float(np.percentile(wrist[:, j], 10)),
            combined_capped_p10_px=float(np.percentile(combined[:, j], 10)),
            combined_capped_episode_macro_mean_px=float(np.mean([combined[ep == e, j].mean() for e in np.unique(ep)]))))
    eligible = [r for r in rows if r['eligible_robust']]
    if not eligible:
        raise ValueError('No candidates meet the predeclared robust base visibility gate: '+task)
    robust = sorted(eligible, key=lambda r: (r['wrist_min_episode_coverage_ge100'],
                    r['wrist_episode_macro_coverage_ge100'], r['wrist_p10_px'],
                    r['combined_capped_episode_macro_mean_px'], -r['candidate']), reverse=True)
    balanced = sorted(eligible, key=lambda r: (r['combined_capped_p10_px'],
                      r['combined_capped_episode_macro_mean_px'], r['wrist_episode_macro_coverage_ge100'],
                      -r['candidate']), reverse=True)
    return rows, robust, balanced



def pipeline_recommendations(source, selection):
    recommendations = {}
    for task in TASKS:
        with np.load(source/'scan'/task/'visibility.npz', allow_pickle=False) as z:
            scan = {k:z[k] for k in ['candidates','frames','ep_id','base_px','wrist_px']}
        table, robust, _ = candidate_table(task, scan)
        old = table[int(selection[task]['candidate'])]
        best = robust[0]
        def key(row):
            return (row['wrist_min_episode_coverage_ge100'], row['wrist_episode_macro_coverage_ge100'],
                    row['wrist_p10_px'], row['combined_capped_episode_macro_mean_px'])
        improved = not old['eligible_robust'] or key(best) > key(old)
        chosen = best if improved else old
        recommendations[task] = dict(world=chosen['world'],candidate=chosen['candidate'],
            old_candidate=old['candidate'],old_world=old['world'],retrain_requested=task not in STRONG,
            selection_rule='training inits0-9 only; base>=300 on>=95% frames and>=90% in every episode; maximize min-episode wrist>=100 coverage, then episode-macro wrist>=100 coverage, wrist p10, capped dual-camera mean; retain original on no improvement',
            improvement_feasible_in_existing_grid=bool(improved),position_changed=chosen['candidate']!=old['candidate'],
            selection_note='Improves the prespecified ordered exposure score; efficacy remains untested.' if improved else
                'No legal scanned candidate improves this ordered training exposure score; retain original position.',
            train_only_statistics=dict(original=old,selected=chosen,frames=len(scan['frames']),
                training_episodes=sorted(int(e)for e in np.unique(scan['ep_id'])),robust_eligible_candidates=len(robust)),
            scan_metadata_sha256=sha(source/'scan'/task/'visibility.json'))
    return recommendations


def overlay(image, observed, ideal):
    a = np.asarray(Image.fromarray(np.asarray(image)).resize((224, 224), BILINEAR)).astype(float)
    green = observed
    red = ideal & ~observed
    a[green] = .5*a[green]+.5*np.array([0, 255, 90])
    a[red] = .55*a[red]+.45*np.array([255, 40, 30])
    return Image.fromarray(np.rint(a).clip(0, 255).astype(np.uint8))


def representative_frames(areas, train):
    ks = np.flatnonzero(train)
    order = ks[np.argsort(areas[ks], kind='stable')]
    return list(dict.fromkeys([int(order[0]), int(order[len(order)//2]), int(order[-1])]))


def montage(task, data, geometry, ideals, frames, out):
    width, tileh = 348, 306
    canvas = Image.new('RGB', (width*len(frames), 70+2*tileh), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), LABELS[task]+' | existing selected position | TRAIN ONLY', fill='black')
    draw.text((10, 29), 'Green: observed patch; red: unoccluded top-face projection hidden by scene.', fill='black')
    draw.text((10, 47), 'Columns: least / median / most wrist area. No new rendering.', fill='black')
    for col, k in enumerate(frames):
        for v, view in enumerate(['base', 'wrist']):
            tile = overlay(data[view]['random256'][k], data[view]['mask'][k], ideals[view][k])
            tile = tile.resize((280, 280))
            x, y = width*col+4, 70+tileh*v
            canvas.paste(tile, (x, y))
            g = geometry[view][k]
            draw.text((x, y+281), f'{view} frame{k} ep{data[view]["ep_id"][k]}: {g["visible_px"]}/{g["ideal_in_frame_px"]}px', fill='black')
            draw.text((x, y+293), g['low_visibility_cause'], fill='black')
    path = out/(task+'.png')
    canvas.save(path)
    return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT/'out/visible_pos')
    ap.add_argument('--out', type=Path, default=ROOT/'out/main_results/low_asr_audit/visibility')
    args = ap.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    images = out/'representative_images'
    images.mkdir(exist_ok=True)
    selection = json.loads((source/'selection.json').read_text())
    pipeline_selection = pipeline_recommendations(source, selection)
    dump(out/'selection.json', pipeline_selection)
    report = dict(schema='libero-low-asr-visibility-cpu-audit-v1', updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source=str(source), audit_source_sha256=sha(__file__), patch_m=.13, uses_test_outcomes_for_selection=False,
        scope='All seven existing task calibrations; task outcome groups supplied by parent only for reporting. Every candidate score uses clean training inits0-9 only.',
        original_rule='base>=300 pixels on>=90% training frames; maximize unbounded frame-weighted mean wrist pixels',
        proposed_rules=dict(robust_dual='base>=300 on>=95% frames AND>=90% in EVERY training episode; lexicographically maximize minimum-episode wrist>=100 coverage, episode-macro wrist>=100 coverage, wrist10th percentile, episode-macro mean capped(base,2000)+capped(wrist,2000)',
                            balanced_capped='same strict base gate; maximize10th percentile capped(base,2000)+capped(wrist,2000), then episode-macro mean of that total, then wrist>=100 episode-macro coverage'),
        limitations=['Training clean-trajectory visibility does not establish attack efficacy or attacked closed-loop visibility.',
                     'Top-face ideal ignores the thin box side faces and subpixel boundaries; no exact occluder identity can be inferred without scene depth.',
                     'Validation masks are reported separately and never used to select a candidate.',
                     'Alternative candidate visibility counts were already rendered in the original training scan; alternative homographies are derived by translating the known13cm plane.'], tasks={})
    view_csv, frame_csv, candidates_csv, candidate_csv, joint_csv = [], [], [], [], []
    for task in TASKS:
        print('[audit]', task, flush=True)
        with np.load(source/'scan'/task/'visibility.npz', allow_pickle=False) as z:
            scan = {k: z[k] for k in ['candidates', 'frames', 'ep_id', 'base_px', 'wrist_px']}
        table, robust, balanced = candidate_table(task, scan)
        candidates_csv.extend(table)
        current = int(selection[task]['candidate'])
        chosen = dict(original=table[current], robust_dual=robust[0], balanced_capped=balanced[0])
        print('[candidate]', task, json.dumps({key:dict(candidate=r['candidate'], world=r['world'],wrist_cov100=r['wrist_coverage_ge100'],wrist_macro_cov100=r['wrist_episode_macro_coverage_ge100'],base_mean=r['base_mean_px'])for key,r in chosen.items()}), flush=True)
        for rule, row in chosen.items():
            candidate_csv.append(dict(rule=rule, **row))
        data, geometry, ideals = {}, {}, {}
        for view in ['base', 'wrist']:
            path = source/'tasks'/task/(view+'.npz')
            with np.load(path, allow_pickle=False) as z:
                data[view] = {k: z[k] for k in ['mask', 'image256_to_texture_pixel_H', 'ep_id', 'train_mask', 'validation_mask', 'random256', 'cell_world', 'patch_m']}
            d = data[view]
            assert float(d['patch_m']) == .13
            assert np.array_equal(d['train_mask'], d['ep_id'] < 10)
            assert np.array_equal(d['validation_mask'], d['ep_id'] >= 10)
            assert np.allclose(d['cell_world'][:2], selection[task]['world'], rtol=0, atol=1e-9)
        base, wrist = data['base']['mask'].sum((1, 2)), data['wrist']['mask'].sum((1, 2))
        train, valid, ep = data['base']['train_mask'], data['base']['validation_mask'], data['base']['ep_id']
        assert np.array_equal(ep, data['wrist']['ep_id'])
        frames = representative_frames(wrist, train)
        item = dict(group='higher_ASR_comparator' if task in STRONG else 'lower_ASR_requested',
                    current_position=selection[task]['world'], chosen_candidates=chosen,
                    robust_top3=robust[:3], balanced_top3=balanced[:3], eligible_robust_candidates=sum(r['eligible_robust'] for r in table),
                    cameras={}, joint={}, candidate_geometry={}, provenance={})
        for view in ['base', 'wrist']:
            d = data[view]
            areas = d['mask'].sum((1, 2))
            geometry[view], ideals[view] = geometric_audit(d['image256_to_texture_pixel_H'], observed_masks=d['mask'], keep=set(frames))
            item['cameras'][view] = {}
            for split, selected in [('train', train), ('validation', valid)]:
                metrics = exposure(areas[selected], ep[selected])
                geom = geometry_summary(geometry[view], selected)
                item['cameras'][view][split] = dict(exposure=metrics, geometry=geom)
                row = dict(task=task, group=item['group'], view=view, split=split, **{k:v for k,v in metrics.items() if k!='area_percentiles_px'},
                           **{f'area_p{q}_px':v for q,v in metrics['area_percentiles_px'].items()},
                           **{k:v for k,v in geom.items() if k not in ['frames','ideal_area_percentiles_px','hidden_fraction_if_projection_ge50_percentiles','low_visibility_cause_counts','top_face_caveat']},
                           cause_counts=geom['low_visibility_cause_counts'])
                view_csv.append(row)
            for k, g in enumerate(geometry[view]):
                frame_csv.append(dict(task=task, view=view, frame=k, episode=int(ep[k]), split='train' if train[k] else 'validation', **g))
            scan_counts = scan['base_px' if view=='base' else 'wrist_px'][:, current]
            item['cameras'][view]['scan_vs_calibration_area_max_abs_diff_px'] = int(np.abs(scan_counts-areas[scan['frames']]).max())
            item['provenance'][view] = dict(path=str(source/'tasks'/task/(view+'.npz')),sha256=sha(source/'tasks'/task/(view+'.npz')))
        for split, selected in [('train', train), ('validation', valid)]:
            joint = dict(frames=int(selected.sum()), both_zero_fraction=float(((base==0)&(wrist==0))[selected].mean()),
                         union_zero_fraction=float(((base==0)&(wrist==0))[selected].mean()))
            for threshold in [1,50,100,300]:
                b,w = base[selected]>=threshold, wrist[selected]>=threshold
                joint[f'either_ge{threshold}_fraction']=float((b|w).mean())
                joint[f'both_ge{threshold}_fraction']=float((b&w).mean())
            item['joint'][split]=joint
            joint_csv.append(dict(task=task,split=split,**joint))
        for rule in ['robust_dual','balanced_capped']:
            row=chosen[rule];j=row['candidate'];offset=np.array(row['world'])-np.array(selection[task]['world'])
            item['candidate_geometry'][rule]={}
            for view in ['base','wrist']:
                g,_=geometric_audit(data[view]['image256_to_texture_pixel_H'][scan['frames']],observed_counts=scan['base_px' if view=='base' else 'wrist_px'][:,j],offset=offset)
                item['candidate_geometry'][rule][view]=geometry_summary(g,np.ones(len(g),bool))
        item['representative_image']=montage(task,data,geometry,ideals,frames,images)
        item['provenance']['scan_metadata']=dict(path=str(source/'scan'/task/'visibility.json'),sha256=sha(source/'scan'/task/'visibility.json'))
        report['tasks'][task]=item
        dump(out/(task+'.json'),item)
        print('[cause]',task,json.dumps(item['cameras']['wrist']['train']['geometry']),flush=True)
        del data,geometry,ideals
    dump(out/'audit.json',report)
    write_csv(out/'camera_metrics.csv',view_csv)
    write_csv(out/'frame_geometry.csv',frame_csv)
    write_csv(out/'all_training_candidates.csv',candidates_csv)
    write_csv(out/'recommended_positions.csv',candidate_csv)
    write_csv(out/'joint_coverage.csv',joint_csv)
    print('[done]',out,flush=True)


if __name__=='__main__':
    main()
