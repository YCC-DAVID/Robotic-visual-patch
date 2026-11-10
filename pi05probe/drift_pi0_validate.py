"""Read raw DRIFT episodes and reject mismatched physics, layout or noise pairs."""
import argparse
from pathlib import Path
import numpy as np
from drift_pi0_common import CONDITIONS, SUITES, TEST_INITS, dump, read, sha
from drift_pi0_pipeline import summary


def same_layout(clean_pos, clean_quat, other_pos, other_quat, rendered):
    if not rendered:
        return np.array_equal(clean_pos, other_pos) and np.array_equal(clean_quat, other_quat)
    if len(other_pos) != len(clean_pos) + 1:
        return False
    # The new fixed visual body may change body indices, but has no joints.
    # Removing exactly that one extra body must leave every old transform exact.
    return any(np.array_equal(clean_pos, np.delete(other_pos, i, axis=0)) and
               np.array_equal(clean_quat, np.delete(other_quat, i, axis=0))
               for i in range(len(other_pos)))


def validate(root, allow_partial=False):
    count, pairs, missing = 0, 0, []
    for suite in SUITES:
        for tid in range(10):
            for init_id in TEST_INITS:
                base = root / 'eval' / suite / ('task%02d' % tid)
                clean_path = base / 'clean' / ('init%02d.json' % init_id)
                clean_row = read(clean_path) if clean_path.exists() else None
                clean = None
                if clean_row:
                    clean = np.load(clean_row['record'], allow_pickle=False)
                try:
                    for condition in CONDITIONS:
                        path = base / condition / ('init%02d.json' % init_id)
                        if not path.exists():
                            missing.append(str(path.relative_to(root)))
                            continue
                        row = read(path)
                        assert row['suite'] == suite and row['task_id'] == tid and row['init_index'] == init_id
                        assert row['condition'] == condition and type(row['success']) is bool and row['runtime_error'] is False
                        assert sha(row['record']) == row['record_sha256'], ('record hash changed', path)
                        with np.load(row['record'], allow_pickle=False) as record:
                            assert len(record['actions']) == row['queries'] == len(record['noise_seeds'])
                            assert np.isfinite(record['actions']).all() and np.isfinite(record['sim_states']).all()
                            if condition != 'clean':
                                assert clean is not None, ('missing clean reference', path)
                                assert row['task'] == clean_row['task'] and row['env_seed'] == clean_row['env_seed']
                                assert np.array_equal(clean['initial_state'], record['initial_state']), ('physics mismatch', path)
                                assert same_layout(clean['fixture_body_pos'], clean['fixture_body_quat'],
                                    record['fixture_body_pos'], record['fixture_body_quat'], condition.endswith('_render')), ('fixture layout mismatch', path)
                                n = min(len(record['noise_seeds']), len(clean['noise_seeds']))
                                assert np.array_equal(record['noise_seeds'][:n], clean['noise_seeds'][:n]), ('noise pairing mismatch', path)
                                if condition.endswith('_digital'):
                                    assert record['input_sha256'][0, 0] == clean['input_sha256'][0, 0], ('digital baseline base image mismatch', path)
                                pairs += 1
                        count += 1
                finally:
                    if clean is not None:
                        clean.close()
    result = dict(complete=not missing, validated_episodes=count, validated_condition_pairs=pairs,
                  missing_episodes=len(missing), fixture_layout_exact=True, initial_state_exact=True,
                  noise_prefix_exact=True, runtime_errors_counted=False)
    dump(root / 'validation.json', result)
    if not allow_partial:
        assert not missing and count == 2000, ('incomplete experiment', result)
    summary(root)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    print(validate(args.root, args.allow_partial))
