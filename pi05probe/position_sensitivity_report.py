"""Audit and report frozen-texture position sensitivity without checkpoint selection."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent
DEST = ROOT/'out/position_sensitivity'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def main():
    assert read(DEST/'progress.json')['stage'] in ['measurements_complete', 'complete']
    protocol = read(DEST/'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT/name) == digest == sha(DEST/'source_snapshot'/name)
    clean_path = Path(protocol['clean_reference'])
    assert sha(clean_path) == protocol['clean_reference_sha256']
    clean_data = read(clean_path)
    clean = {r['init_index']: r for r in clean_data['per_episode']['clean']}
    good = sorted(ep for ep, r in clean.items() if r['success'])
    assert len(good) == 9
    results = {}; count = 0; origin_exact = {}
    references = {'distance': ROOT/'out/direction_loss/rollout/episodes.json',
                  'random': ROOT/'out/dual_view/rollout_random/episodes.json'}
    source_hashes = None
    for position in protocol['positions']:
        name = position['name']; folder = DEST/name
        assert read(folder/'complete.json')['new_episodes'] == 20
        manifest = read(folder/'manifest.json')
        expected_center = np.array(protocol['source_geometry']['cell_world'])
        expected_center[:2] += position['offset_xy_m']
        assert np.array_equal(expected_center, manifest['geometry']['cell_world'])
        assert manifest['geometry']['patch_m'] == .13
        if source_hashes is None: source_hashes = manifest['texture_sha256']
        assert source_hashes == manifest['texture_sha256']
        for condition, path in manifest['textures'].items():
            assert sha(path) == source_hashes[condition]
        data = read(folder/'rollout/episodes.json'); meta = data['metadata']
        assert meta['geometry'] == manifest['geometry']
        assert meta['manifest_sha256'] == sha(folder/'manifest.json')
        assert meta['task'] == protocol['task'] == clean_data['metadata']['task']
        assert meta['init_states_sha256'] == clean_data['metadata']['init_states_sha256']
        assert meta['max_steps'] == 300 and meta['replan'] == 5
        assert data['per_episode']['clean'] == clean_data['per_episode']['clean']
        conditions = {}
        for condition in protocol['conditions']:
            rows = data['per_episode'][condition]
            assert sorted(r['init_index'] for r in rows) == list(range(40, 50))
            assert meta['textures'][condition]['sha256'] == source_hashes[condition]
            medians = []; fractions = []; first_vis = []; deltas = []; per_episode = []
            for row in rows:
                ep = row['init_index']; calls = row['inferences']
                with np.load(folder/'rollout'/row['replay_record']) as a, np.load(clean_path.parent/clean[ep]['replay_record']) as b:
                    assert np.array_equal(a['initial_state'], b['initial_state'])
                    assert hashlib.sha256(a['initial_state'].tobytes()).hexdigest() == row['initial_state_sha256']
                    n = min(len(a['fixture_body_pos']), len(b['fixture_body_pos']))
                    for key in ['fixture_body_pos', 'fixture_body_quat']:
                        assert np.array_equal(a[key][:n], b[key][:n])
                    assert np.array_equal(a['state8'][0], b['state8'][0])
                    assert a['actions'].shape == (calls, 10, 7) and np.isfinite(a['actions']).all()
                    assert calls == len(a['base224']) == len(a['wrist224']) == len(a['sim_states'])
                    assert np.array_equal(a['noise_seeds'], 20261006+ep*1000+np.arange(calls))
                    v = a['visible_pixels224']; assert v.shape == (calls, 2)
                    assert ((v >= 0) & (v <= 224*224)).all()
                    assert np.array_equal(np.median(v, axis=0), row['visible_pixels224_median'])
                    medians.append(np.median(v, axis=0)); fractions.append((v > 0).mean(axis=0)); first_vis.append(v[0])
                    if name == 'origin':
                        reference_path = references[condition]
                        old = next(r for r in read(reference_path)['per_episode'][condition] if r['init_index'] == ep)
                        with np.load(reference_path.parent/old['replay_record']) as previous:
                            for key in ['actions', 'noise_seeds', 'sim_states', 'state8', 'base224', 'wrist224']:
                                assert np.array_equal(a[key], previous[key]), (condition, ep, key, 'origin replication differs')
                        assert row['success'] == old['success'] and row['success_step'] == old['success_step']
                        origin_exact.setdefault(condition, []).append(ep)
                if ep in good and row['success']:
                    deltas.append(row['success_step']-clean[ep]['success_step'])
                per_episode.append({'init': ep, 'task_success': row['success'], 'eligible_for_ASR': ep in good,
                                    'success_step': row['success_step'], 'visible_median': row['visible_pixels224_median']})
                count += 1
            failed = [r['init_index'] for r in rows if r['init_index'] in good and not r['success']]
            result = {'task_successes': sum(r['success'] for r in rows), 'episodes': 10,
                      'attack_successes': len(failed), 'clean_success_denominator': len(good),
                      'ASR': len(failed)/len(good), 'failed_clean_success_indices': failed,
                      'median_episode_median_visible_pixels_base_wrist': np.median(medians, axis=0).tolist(),
                      'median_episode_visible_fraction_base_wrist': np.median(fractions, axis=0).tolist(),
                      'median_first_frame_visible_pixels_base_wrist': np.median(first_vis, axis=0).tolist(),
                      'completion_step_delta_median_joint_success': float(np.median(deltas)) if deltas else None,
                      'per_episode': per_episode}
            assert result['attack_successes'] == data['summary'][condition]['paired_ASR_failures']
            assert result['task_successes'] == data['summary'][condition]['successes']
            conditions[condition] = result
        results[name] = {'offset_xy_cm': (100*np.array(position['offset_xy_m'])).tolist(), 'conditions': conditions}
    assert count == 180
    base_fail = set(results['origin']['conditions']['distance']['failed_clean_success_indices'])
    for r in results.values():
        failed = set(r['conditions']['distance']['failed_clean_success_indices'])
        r['distance_paired_vs_origin'] = {'attack_retained_indices': sorted(base_fail & failed),
                                         'attack_lost_indices': sorted(base_fail-failed),
                                         'attack_gained_indices': sorted(failed-base_fail)}
    for i in range(3): assert read(DEST/('worker'+str(i))/'service_stopped.json')['stopped']
    recovery = DEST/'recovery'
    if recovery.exists():
        assert read(recovery/'complete.json')['complete']
        assert read(recovery/'service_stopped.json')['stopped']
        recovery_protocol = read(recovery/'protocol.json')
        assert sha(ROOT/'position_sensitivity_retry_rollout.py') == recovery_protocol['rollout_sha256']
        assert sha(ROOT/'position_sensitivity_recover.py') == recovery_protocol['recover_script_sha256']
        with np.load(recovery/'origin49/init49_random.npz') as a, np.load(DEST/'origin/rollout/init49_random.npz') as b:
            for key in ['actions', 'sim_states', 'base224', 'wrist224', 'noise_seeds']:
                assert np.array_equal(a[key], b[key])
    audit = {'complete': True, 'new_paired_rollouts_checked': count, 'unique_initial_states': 10,
             'texture_bytes_identical_all_positions': True, 'initial_physics_fixture_noise_paired': True,
             'origin_exact_replay_vs_previous': origin_exact, 'owned_services_stopped': True}
    if recovery.exists():
        audit['recovery'] = {'one_RGB_repeat_assertion_aborted_before_first_inference': True,
                             'fresh_process_same_seed_passed_unchanged_assertion': True,
                             'origin49_retry_matches_historical_replay_exactly': True,
                             'extra_random_repeat_checked': 1, 'completed_rollouts_including_duplicate': count+1}
    summary = {'complete': True, 'positions': results, 'clean_success_indices': good,
               'clean_task_successes': len(good), 'audit': audit, 'limitations': protocol['limits']}
    (DEST/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (DEST/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    figures(summary)
    lines = ['# E31 frozen proxy-trained texture position sensitivity', '',
             'Attack success = robot task failure among nine clean-success states. One unchanged13cm texture; no retraining.', '',
             '| Position | Offset XY cm | Trained attack successes /9 | Random attack successes /9 | Trained robot successes /10 |',
             '|---|---|---:|---:|---:|']
    for name, r in results.items():
        d = r['conditions']['distance']; rand = r['conditions']['random']
        lines.append(f"| {name} | {r['offset_xy_cm']} | {d['attack_successes']}/9 | {rand['attack_successes']}/9 | {d['task_successes']}/10 |")
    (DEST/'results.md').write_text('\n'.join(lines)+'\n')
    shutil.copy2(__file__, DEST/'source_snapshot'/Path(__file__).name)
    (DEST/'progress.json').write_text(json.dumps({'stage': 'complete', 'new_episodes': count, 'audited': True})+'\n')
    files = {str(p.relative_to(DEST)): {'sha256': sha(p), 'bytes': p.stat().st_size}
             for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'provenance.json' and p.suffix != '.log'}
    (DEST/'provenance.json').write_text(json.dumps({'complete': True, 'files': files}, indent=2)+'\n')
    print('\n'.join(lines), flush=True)
    print(json.dumps(audit), flush=True)


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    positions = summary['positions']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, axis in zip(axes, range(2)):
        selected = sorted([(r['offset_xy_cm'][axis], r) for r in positions.values()
                           if r['offset_xy_cm'][1-axis] == 0], key=lambda item: item[0])
        for condition, marker in [('distance', 'o'), ('random', 's')]:
            x = [a for a, _ in selected]; y = [r['conditions'][condition]['attack_successes'] for _, r in selected]
            ax.plot(x, y, marker=marker, label=condition)
            for xx, yy in zip(x, y):
                ax.annotate(str(yy)+'/9', (xx, yy), xytext=(0, 9 if condition == 'distance' else -16),
                            textcoords='offset points', ha='center')
        ax.set(xticks=[-3, -1, 0, 1, 3], yticks=range(10), ylim=(-.7, 9.7),
               xlabel='World '+['X', 'Y'][axis]+' offset (cm)', ylabel='Task failures among 9 clean successes')
        ax.legend(); ax.grid(alpha=.2)
    fig.suptitle('Frozen proxy-trained 13 cm patch: same texture, actual MuJoCo render')
    fig.savefig(DEST/'position_sensitivity.png', dpi=160); plt.close(fig)
    names = list(positions); ids = summary['clean_success_indices']
    matrix = np.array([[int(ep in positions[n]['conditions']['distance']['failed_clean_success_indices']) for ep in ids] for n in names])
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.imshow(matrix, cmap='Oranges', vmin=0, vmax=1, aspect='auto')
    for i in range(len(names)):
        for j in range(len(ids)): ax.text(j, i, str(matrix[i,j]), ha='center', va='center')
    ax.set_xticks(range(len(ids)), ids); ax.set_yticks(range(len(names)), names)
    ax.set(xlabel='Clean-success init index', title='Trained patch: 1 = task failure (attack success), 0 = task success')
    fig.savefig(DEST/'paired_failures.png', dpi=160); plt.close(fig)
    for key, view in [('base224', 'base'), ('wrist224', 'wrist')]:
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
        for ax, name in zip(axes.flat, names):
            with np.load(DEST/name/'rollout/init40_distance.npz') as d:
                ax.imshow(d[key][0], interpolation='nearest')
                count = d['visible_pixels224'][0, 0 if view == 'base' else 1]
            ax.set_title(f'{name}; visible {count} px'); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f'{view}: same init40 first observation; true render, frozen texture')
        fig.savefig(DEST/(view+'_positions.png'), dpi=140); plt.close(fig)


if __name__ == '__main__':
    main()
