"""Frozen new-cohort replication of the original digital patch yaw experiment."""
import json
from pathlib import Path

from digital_transform_common import ROOT, PROBE, condition, dump, sha, texture_paths

DEFAULT = PROBE / 'out/rotation_expanded_20261009'
ANGLES = ['nominal', 'yaw_minus10', 'yaw_plus10']
GROUPS = [('nominal', 'clean')] + [(c, m) for c in ANGLES for m in ['full', 'random']]


def prepare(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        model='pi0.5', task='place_container_plate', task_config='demo_clean',
        initial_seeds=list(range(10600, 10700)), extension_seeds=list(range(10700, 10800)),
        cohort_rule='Run all 100 initial clean seeds. If fewer than 50 succeed, add all 100 extension clean seeds. Stop at 200 regardless of outcomes.',
        minimum_clean_successes=50,
        evaluation_rule='Freeze the cohort after clean only; run every attack condition on every selected seed, including clean failures.',
        old_pilot_seeds=list(range(10200, 10210)), old_pilot_excluded_from_new_estimates=True,
        conditions=[condition(c) for c in ANGLES], groups=GROUPS,
        patch_m=.20, center=[0, 0, .741], texture_size=[128, 128], texture_step=500,
        texture_paths={m: str(p) for m, p in texture_paths().items()},
        texture_sha256={m: sha(p) for m, p in texture_paths().items()},
        training=False, texture_selection='Keep original trained Full and original Random unchanged.',
        projection='Original digital perspective compositor; world rectangle and UV content both rotate; fresh segmentation every query.',
        yaw='0/-10/+10 degrees about world Z at fixed center; constant within episode',
        replan=32, noise_rule='20261006 + seed * 1000 + query',
        pairing='Identical initial joint state, raw clean RGB, prompt and query-index noise across all seven groups.',
        ASR='Number of patch failures among clean-success initial seeds / clean-success initial seeds',
        uncertainty='95% Wilson interval for each ASR; seed-paired percentile bootstrap (20000 draws) for differences; no frame/noise pseudoreplication.',
        primary_comparisons=['Full(-10)-Full(0)', 'Full(+10)-Full(0)'],
        secondary_comparisons=['Full-Random at each angle'],
        reporting='No interim ASR-based stopping or tuning. Report group ASR only once all its selected seeds finish. Intervals are pointwise.',
    )
    # JSON round-trip also canonicalizes tuples before resume comparisons.
    protocol = json.loads(json.dumps(protocol))
    path = root / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == protocol, 'Frozen protocol differs'
    else:
        dump(path, protocol)
    return protocol


def episode_dir(root, cname, method):
    return Path(root) / 'eval' / cname / method


def records(root, cname, method):
    rows = {}
    for path in episode_dir(root, cname, method).glob('seed*.json'):
        row = json.loads(path.read_text())
        assert path.with_suffix('.npz').exists(), path
        assert row['condition'] == cname and row['method'] == method, path
        assert row['seed'] not in rows, path
        rows[row['seed']] = row
    return rows
