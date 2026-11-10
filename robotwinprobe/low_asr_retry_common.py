"""Independent retry protocol. Never mutates the frozen main-method run."""
import json
from pathlib import Path
import bootstrap as B
from main_method_common import TASK, CAMERAS, WIRE, CENTER, SIZE, SEEDS, sha, dump

DEFAULT = B.OUT/'low_asr_retry_20261108'
SOURCE = B.OUT/'main_method'
TRAIN_SEEDS = list(range(25000, 25020))
VALIDATION_SEEDS = list(range(25020, 25024))
DATA_SEEDS = TRAIN_SEEDS+VALIDATION_SEEDS
STAGES = ('approach', 'grasp', 'transport', 'place')


def prepare(root):
    from texture_eot import RANGES
    root = Path(root).resolve()
    assert root != SOURCE.resolve() and SOURCE.resolve() not in root.parents, 'preserve original main run'
    original = json.loads((SOURCE/'protocol.json').read_text())
    value = dict(schema='robotwin-full-trajectory-retry-v1', benchmark='RoboTwin', tasks=[TASK],
                 source_run=str(SOURCE), source_protocol_sha256=sha(SOURCE/'protocol.json'),
                 retry_reason='ASR12.5% on original10 exposed seeds; original result retained, retry is adaptive follow-up',
                 data=dict(seeds=DATA_SEEDS, train_seeds=TRAIN_SEEDS, validation_seeds=VALIDATION_SEEDS,
                           collection='every clean native-policy query through success or official400-step limit',
                           outcome_filter=False, phase='action-level PhaseTracker, no event-forced replanning',
                           noise_base=20261006, replan=32, horizon=32,
                           patterns=['black', 'gray64', 'gray', 'gray192', 'white', 'random', 'checker'],
                           calibration='all query states, fixed nominal physical geometry/light, no world-pose search'),
                 method=dict(patch_m=SIZE, center_m=CENTER, resolution=128, steps=500, batch=8, lr=.01,
                             seed_batch=20261005, noise_seed=20260901, eot_seed=2026102901,
                             initialization='gray .5', eot=RANGES,
                             sampling='two records per stage per update; uniform seed within stage then uniform query within seed',
                             objective='full32x32 first-t=1 velocity SSE vs same-noise clean',
                             proxy='unchanged crossarch_proxy five-gray+linear footprint+global indirect response',
                             proxy_fidelity_gate=False, checkpoint_selection='fixed final step500; no test ASR selection'),
                 evaluation=original['evaluation'],
                 test_reuse='original26000-26009 are exposed; report as retry, not a new independent confirmatory result',
                 geometry_selection='original fixed13cm center retained; neither test RGB nor ASR used to choose position',
                 source_sha256={n: sha(B.PROBE/n) for n in ['low_asr_retry_common.py', 'low_asr_retry_capture.py',
                      'low_asr_retry_train.py', 'low_asr_retry_pipeline.py', 'main_method_rollout.py', 'main_method_runtime.py',
                      'main_method_common.py', 'main_method_train.py',
                      'crossarch_proxy.py', 'crossarch_backend.py', 'render_model.py', 'texture_eot.py',
                      'policy_server.py', 'stage_joint_common.py', 'scene_smoke.py', 'render_geometry.py']})
    target = root/'protocol.json'
    if target.exists():
        assert json.loads(target.read_text()) == value, 'retry protocol changed; preserve existing run'
    else:
        dump(target, value)
    return value


def collected_records(root):
    root = Path(root)
    protocol = prepare(root)
    rows = []
    episodes = []
    for seed in DATA_SEEDS:
        item = json.loads((root/'collection'/f'seed{seed}.json').read_text())
        assert item['complete'] and item['seed'] == seed and item['protocol_sha256'] == sha(root/'protocol.json')
        assert [r['query'] for r in item['records']] == list(range(len(item['records'])))
        for row in item['records']:
            assert row['seed'] == seed and sha(row['path']) == row['sha256']
            assert row['stage'] in STAGES
        rows.extend(item['records']); episodes.append(item)
    for name, seeds in [('train', TRAIN_SEEDS), ('validation', VALIDATION_SEEDS)]:
        for stage in STAGES:
            assert any(r['seed'] in seeds and r['stage'] == stage for r in rows), ('missing stage cannot be silently substituted', name, stage)
    return protocol, rows, episodes


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--root', type=Path, default=DEFAULT)
    args = ap.parse_args(); value = prepare(args.root)
    print(json.dumps(dict(prepared=True, root=str(args.root.resolve()), training_seeds=TRAIN_SEEDS,
                         validation_seeds=VALIDATION_SEEDS, evaluation_seeds=SEEDS, method=value['method']), indent=2))
