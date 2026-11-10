"""Paired native-render evaluation of ours and DRIFT at one fixed wrist pose."""
import argparse
from pathlib import Path

from drift_paper_common import dump, read, sha, now

CONDITIONS = ('clean', 'drift_render', 'random_render', 'ours_render')
TASKS = (0, 1, 2)
INITS = tuple(range(10))
SUITE = 'libero_spatial'


def validate_pair(root, row):
    import numpy as np
    from drift_paper_validate import same_layout
    assert row['runtime_error'] is False
    assert sha(row['record']) == row['record_sha256']
    with np.load(row['record'], allow_pickle=False) as record:
        assert len(record['actions']) == row['queries'] == len(record['noise_seeds'])
        assert np.isfinite(record['actions']).all() and np.isfinite(record['sim_states']).all()
        if row['condition'] == 'clean':
            return
        clean = read(root / 'eval' / row['suite'] / ('task%02d' % row['task_id']) /
                     'clean' / ('init%02d.json' % row['init_index']))
        assert clean['task'] == row['task'] and clean['env_seed'] == row['env_seed']
        assert clean['initial_state_sha256'] == row['initial_state_sha256']
        assert sha(clean['record']) == clean['record_sha256']
        with np.load(clean['record'], allow_pickle=False) as ref:
            assert np.array_equal(ref['initial_state'], record['initial_state'])
            assert same_layout(ref['fixture_body_pos'], ref['fixture_body_quat'],
                               record['fixture_body_pos'], record['fixture_body_quat'], True)
            n = min(len(ref['noise_seeds']), len(record['noise_seeds']))
            assert n > 0 and np.array_equal(ref['noise_seeds'][:n], record['noise_seeds'][:n])


def summarize(root, verify=True):
    root = Path(root)
    groups = {c: [] for c in CONDITIONS}
    for tid in TASKS:
        for condition in CONDITIONS:
            for init_id in INITS:
                path = root / 'eval' / SUITE / ('task%02d' % tid) / condition / ('init%02d.json' % init_id)
                if not path.exists():
                    continue
                row = read(path)
                assert (row['suite'], row['task_id'], row['init_index'], row['condition']) == (SUITE, tid, init_id, condition)
                assert type(row['success']) is bool and row['runtime_error'] is False
                if verify:
                    validate_pair(root, row)
                groups[condition].append(row)
    clean = {(r['task_id'], r['init_index']): r for r in groups['clean']}
    metrics = {}
    for condition, rows in groups.items():
        pairs = [(r, clean[(r['task_id'], r['init_index'])]) for r in rows]
        nclean = sum(c['success'] for r, c in pairs)
        nsuccess = sum(r['success'] for r, c in pairs)
        metrics[condition] = dict(n=len(rows), successes=nsuccess,
            SR=nsuccess / len(rows) if rows else None,
            matched_clean_successes=nclean,
            relative_ASR=(nclean - nsuccess) / nclean if nclean else None,
            paired_ASR=sum(c['success'] and not r['success'] for r, c in pairs) / nclean if nclean else None)
    value = dict(complete=all(len(rows) == 30 for rows in groups.values()),
                 updated=now(), episodes=sum(len(rows) for rows in groups.values()), target_episodes=120,
                 suite=SUITE, task_ids=list(TASKS), initial_indices=list(INITS),
                 conditions=metrics, validation='passed' if verify else 'pending',
                 scope='Single universal patch per method, three tasks, one training seed',
                 runtime_errors_counted=False)
    dump(root / 'summary.json', value)
    return value


def worker(args):
    import numpy as np
    from PIL import Image
    from drift_paper_common import bootstrap, configure_libero
    bootstrap(sim=True)
    configure_libero(args.root)
    import drift_paper_sim as sim
    from libero.libero import benchmark
    from openpi_client import websocket_client_policy
    root = Path(args.root)
    protocol = read(root / 'protocol.json')
    assert sha(root / 'geometry/geometry.json') == protocol['geometry_sha256']
    geometry = read(root / 'geometry/geometry.json')
    paths = dict(drift=root / 'baseline/drift_pi05_64px.png',
                 random=root / 'random.png', ours=root / 'train/ours_pi05_64px.png')
    assert sha(paths['drift']) == protocol['drift_patch_png_sha256']
    assert sha(paths['ours']) == read(root / 'train/complete.json')['png_sha256']
    suite = benchmark.get_benchmark_dict()[SUITE]()
    client = websocket_client_policy.WebsocketClientPolicy('127.0.0.1', args.port)
    original_visibility = sim.visibility
    try:
        for tid in TASKS:
            task, inits = suite.get_task(tid), suite.get_task_init_states(tid)
            for condition in CONDITIONS:
                dest = root / 'eval' / SUITE / ('task%02d' % tid) / condition
                dest.mkdir(parents=True, exist_ok=True)
                texture = paths.get(condition.split('_')[0])
                processor = sim.plane_processor(geometry, texture) if texture is not None else None
                env = sim.make_env(task, processor)
                diagnostic = sim.make_env(task, processor) if texture is not None else None
                try:
                    # Segmentation diagnostics use a separate context. The frozen
                    # episode's RGB-repeat and all physics/noise checks stay intact.
                    if diagnostic is not None:
                        def isolated_visibility(source_env):
                            state = source_env.get_sim_state().copy()
                            assert np.array_equal(source_env.env.sim.model.body_pos, diagnostic.env.sim.model.body_pos)
                            assert np.array_equal(source_env.env.sim.model.body_quat, diagnostic.env.sim.model.body_quat)
                            diagnostic.env.sim._render_context_offscreen.gl_ctx.make_current()
                            try:
                                diagnostic.regenerate_obs_from_state(state)
                                value = original_visibility(diagnostic)
                                assert np.array_equal(state, source_env.get_sim_state())
                                return value
                            finally:
                                source_env.env.sim._render_context_offscreen.gl_ctx.make_current()
                        sim.visibility = isolated_visibility
                    for init_id in INITS:
                        output = dest / ('init%02d' % init_id)
                        if output.with_suffix('.json').exists():
                            validate_pair(root, read(output.with_suffix('.json')))
                            continue
                        if diagnostic is not None:
                            seed = 700000 + tid * 100 + init_id
                            diagnostic.env.sim._render_context_offscreen.gl_ctx.make_current()
                            sim.reset(diagnostic, inits[init_id], seed)
                        env.env.sim._render_context_offscreen.gl_ctx.make_current()
                        patch = np.asarray(Image.open(texture).convert('RGB')) if texture is not None else None
                        row, _ = sim.run_episode(env, task, inits[init_id], init_id, SUITE, tid,
                            client, condition, patch, output.with_suffix('.npz'), video=init_id == 0)
                        validate_pair(root, row)
                        dump(output.with_suffix('.json'), row)
                        value = summarize(root, verify=False)
                        dump(root / 'eval/progress.json', dict(stage='evaluating', updated=now(),
                             task=tid, condition=condition, init_index=init_id, success=row['success'],
                             episodes=value['episodes'], target_episodes=120))
                finally:
                    sim.visibility = original_visibility
                    env.env.sim._render_context_offscreen.gl_ctx.make_current()
                    env.close()
                    if diagnostic is not None:
                        diagnostic.env.sim._render_context_offscreen.gl_ctx.make_current()
                        diagnostic.close()
    finally:
        client._ws.close()
    result = summarize(root)
    assert result['complete']
    dump(root / 'eval/complete.json', dict(complete=True, updated=now(), summary_sha256=sha(root / 'summary.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--port', type=int, default=18871)
    p.add_argument('--summary-only', action='store_true')
    args = p.parse_args()
    if args.summary_only:
        print(summarize(args.root))
    else:
        worker(args)
