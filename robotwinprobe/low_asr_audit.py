"""CPU audit of the frozen RoboTwin run; writes only a separate audit report."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import bootstrap as B
from main_method_common import CAMERAS, TASK, records, dump, sha


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0)
    return dict(n=len(values), minimum=float(values.min()), p10=float(np.quantile(values, .1)),
                median=float(np.median(values)), mean=float(values.mean()),
                p90=float(np.quantile(values, .9)), maximum=float(values.max()),
                zero_n=int((values == 0).sum()))


def rmse(a, b, mask=None):
    difference = np.asarray(a, float) - np.asarray(b, float)
    if mask is not None:
        difference = difference[mask]
    return float(np.sqrt(np.mean(difference ** 2))) if difference.size else None


def audit(root, evaluate_proxy=False):
    import cv2
    from render_proxy import grid_for
    bank = records()
    report = dict(schema='robotwin-low-asr-cpu-audit-v1', source_run=str(root),
                  source_protocol_sha256=sha(root/'protocol.json'), training=[], rollouts=[])
    if evaluate_proxy:
        import torch
        from crossarch_proxy import images
        from render_geometry import patterns
        torch.set_num_threads(4)
        textures = {n: torch.tensor(patterns()[n].transpose(2, 0, 1)[None].copy()).float()/255
                    for n in ['random', 'checker']}
    for row in bank:
        record = dict(seed=row['seed'], query=row['query'], path=row['path'], views={})
        predicted = {}
        if evaluate_proxy:
            with torch.no_grad():
                predicted = {n: [v[0].permute(1, 2, 0).numpy() for v in images(q, row['path'])]
                             for n, q in textures.items()}
        with np.load(row['path'], allow_pickle=False) as z:
            record.update(stage=str(z['stage']), step=int(z['policy_action_step']), prompt=str(z['prompt']))
            for ci, camera in enumerate(CAMERAS):
                mask = z[camera+'_mask'].astype(bool)
                grid = grid_for(z[camera+'_intrinsic'], z[camera+'_extrinsic'],
                                z['center_m'], float(z['size_m']))
                projected = np.abs(grid).max(-1) <= 1
                interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                gain = (z[camera+'_white'].astype(float)-z[camera+'_black'].astype(float))/255
                clean = z[camera+'_clean']/255.
                data = dict(native_pixels=int(mask.sum()),
                            policy_support_pixels=int((cv2.resize(mask.astype(np.uint8), (224, 168),
                                                       interpolation=cv2.INTER_NEAREST)>0).sum()),
                            projected_pixels=int(projected.sum()),
                            visible_fraction_of_projection=float((mask & projected).sum()/max(projected.sum(), 1)),
                            outside_blackwhite_rmse=rmse(gain, np.zeros_like(gain), ~mask),
                            outside_blackwhite_changed_pixels=int((np.any(gain != 0, axis=-1) & ~mask).sum()),
                            random_clean_rmse_inside=rmse(z[camera+'_random']/255., clean, mask),
                            random_clean_rmse_outside=rmse(z[camera+'_random']/255., clean, ~mask))
                if evaluate_proxy:
                    data['proxy'] = {n: dict(native_rmse=rmse(v[ci], z[camera+'_'+n]/255.),
                                             mask_rmse=rmse(v[ci], z[camera+'_'+n]/255., mask),
                                             interior_rmse=rmse(v[ci], z[camera+'_'+n]/255., interior),
                                             outside_rmse=rmse(v[ci], z[camera+'_'+n]/255., ~mask))
                                     for n, v in predicted.items()}
                record['views'][camera] = data
        report['training'].append(record)
    files = sorted((root/'eval'/TASK/'episodes').glob('*.json'))
    for path in files:
        row = json.loads(path.read_text())
        item = {k: row[k] for k in ['seed', 'condition', 'status', 'success', 'steps', 'queries', 'prompt']}
        assert sha(path.with_suffix('.npz')) == row['artifact_sha256']
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
            visible = z['visible_pixels']
            item['views'] = {c: dict(pixels=distribution(visible[:, i]),
                                    all_query_pixels=visible[:, i].tolist()) for i, c in enumerate(CAMERAS)}
            item['action_range'] = [float(z['actions'].min()), float(z['actions'].max())]
            item['state_range'] = [float(z['states14'].min()), float(z['states14'].max())]
        report['rollouts'].append(item)
    collections = [json.loads(p.read_text()) for p in
                   sorted((B.OUT/'frequency_bands_13cm_20261030'/'collection').glob('seed*.json'))]
    report['training_trajectory_queries'] = dict(Counter(x['phase'] for row in collections for x in row['traces']))
    report['training_stage_counts'] = dict(Counter(x['stage'] for x in report['training']))
    report['training_query_counts'] = dict(Counter(str(x['query']) for x in report['training']))
    report['training_step_range'] = distribution([x['step'] for x in report['training']])
    report['training_summary'] = {}
    for camera in CAMERAS:
        views = [x['views'][camera] for x in report['training']]
        s = {key: distribution([v[key] for v in views]) for key in
             ['native_pixels', 'policy_support_pixels', 'visible_fraction_of_projection',
              'outside_blackwhite_rmse', 'outside_blackwhite_changed_pixels', 'random_clean_rmse_outside']}
        if evaluate_proxy:
            s['proxy'] = {n: {k: distribution([v['proxy'][n][k] for v in views if v['proxy'][n][k] is not None])
                                for k in ['native_rmse', 'mask_rmse', 'interior_rmse', 'outside_rmse']}
                          for n in textures}
        report['training_summary'][camera] = s
    report['rollout_summary'] = {}
    for condition in ['random', 'method']:
        episodes = [x for x in report['rollouts'] if x['condition'] == condition]
        report['rollout_summary'][condition] = {c: distribution([v for x in episodes for v in x['views'][c]['all_query_pixels']])
                                                 for c in CAMERAS}
    report['interface_review'] = dict(
        train_evaluation_match=['same pi05_aloha config/checkpoint, delta-joint and adapt_to_pi settings',
                                'all three native 240x320 RGB inputs use official Aloha transforms',
                                'native forward resize/pad is exact; differentiable backward uses 168x224 + 28px pad',
                                'same bowl/cup prompt from actor type; matched three conditions',
                                'train first-t=1 full32x32 velocity SSE; native evaluation uses 10-flow32x14 qpos actions',
                                'same-noise within each clean/attack training pair and within each rollout query'],
        differences=['train noise fixed20260901; closed-loop noise=20261006+seed*1000+query, intentional held-out noise',
                     'source clean-bank trajectories used noise_base202610270, not current evaluation20261006',
                     'training49 states are approach only; native evaluation continues grasp/transport/place',
                     'full32x32 velocity loss includes18 padded coordinates; both benchmarks use full padded velocity',
                     'RoboTwin: native32 replan,14 real coordinates,three views; LIBERO: replan5,horizon10,7 real coordinates,two views',
                     'RoboTwin global gray-LUT response outside visible mask; physically measured indirect signal but random-texture prediction requires audit',
                     'LIBERO proxy calibrated mipmapping with0.02 interior gate; RoboTwin linear-footprint five-gray proxy with no fidelity gate'],
        interpretation='Known interface differences are recorded; none establish that occlusion alone caused low ASR. No test-state position search.')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=B.OUT/'main_method')
    ap.add_argument('--out', type=Path, default=B.OUT/'low_asr_retry_20261108/audit.json')
    ap.add_argument('--proxy', action='store_true', help='also run the existing image proxy on CPU for all49 states')
    args = ap.parse_args()
    report = audit(args.root.resolve(), args.proxy)
    dump(args.out, report)
    print(json.dumps({k: report[k] for k in ['training_stage_counts', 'training_query_counts',
                       'training_trajectory_queries', 'training_summary', 'rollout_summary']}, indent=2))
