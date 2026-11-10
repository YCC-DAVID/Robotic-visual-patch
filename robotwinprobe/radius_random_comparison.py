"""Post-hoc paired comparisons to Random, including both directions of outcome changes.

Reads completed rollouts only. Bootstrap unit is the initial seed, keeping all
ten positions together. Does not replace the original preregistered report.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def analyze(root):
    seeds = json.loads((root / 'cohort.json').read_text())['seeds']
    methods = ['baseline', 'r01', 'r03', 'r05', 'random']
    textures = json.loads((root / 'textures.json').read_text())
    source_hashes = {}

    def read(path):
        data = path.read_bytes()
        source_hashes[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    clean = {s: read(root / 'eval/clean' / f'seed{s}_p00.json') for s in seeds}
    with np.load(root / 'plan.npz') as z:
        positions = {int(s): xy.copy() for s, xy in zip(z['test_seeds'], z['test_xy'])}
    records = {}
    for method in methods:
        records[method] = {}
        texture_hash = hashlib.sha256(Path(textures[method]).read_bytes()).hexdigest()
        for seed in seeds:
            for pi in range(11):
                path = root / 'eval' / method / f'seed{seed}_p{pi:02d}.json'
                row = read(path)
                assert path.with_suffix('.npz').exists()
                assert (row['seed'], row['position_index'], row['texture_sha256']) == (seed, pi, texture_hash)
                xy = [0., 0.] if pi == 0 else list(map(float, positions[seed][pi - 1]))
                assert row['center'] == [*xy, .741]
                for key in ['initial_rgb_sha256', 'initial_state_sha256', 'first_noise_seed', 'prompt']:
                    assert row[key] == clean[seed][key], (method, seed, pi, key)
                assert row['first_noise_seed'] == 20261006 + seed * 1000
                records[method][seed, pi] = row
    rng = np.random.RandomState(20261012)
    indices = rng.randint(0, len(seeds), (20000, len(seeds)))
    rows, comparisons = [], []
    for region, pis in [('origin', [0]), ('disk_0_5cm', list(range(1, 11)))]:
        den = np.array([len(pis) * int(clean[s]['success']) for s in seeds])
        boot_den = den[indices].sum(1)
        valid = boot_den > 0
        harms = {}
        for method in methods:
            data = records[method]
            arr = np.array([sum(not data[s, pi]['success'] for pi in pis) * int(clean[s]['success']) for s in seeds])
            harms[method] = arr
            helped = sum(data[s, pi]['success'] for s in seeds for pi in pis if not clean[s]['success'])
            successes = sum(data[s, pi]['success'] for s in seeds for pi in pis)
            assert successes == den.sum() - arr.sum() + helped
            rows.append(dict(region=region, method=method, clean_successes=int(den.sum()),
                attacks=int(arr.sum()), helped=helped, task_successes=successes,
                episodes=len(seeds) * len(pis), ASR=float(arr.sum() / den.sum()),
                SR=successes / (len(seeds) * len(pis))))
        for reference in ['random', 'baseline']:
            for method in ['baseline', 'r01', 'r03', 'r05']:
                if method == reference:
                    continue
                delta = harms[method] - harms[reference]
                boot = delta[indices].sum(1)[valid] / boot_den[valid]
                comparisons.append(dict(region=region, method=method, reference=reference,
                    paired_ASR_delta=float(delta.sum() / den.sum()),
                    seed_cluster_95=np.quantile(boot, [.025, .975]).tolist()))
    original = json.loads((root / 'summary.json').read_text())
    assert original['complete']
    for row in rows:
        match = next(r for r in original['rows'] if (r['region'], r['method']) == (row['region'], row['method']))
        assert (row['attacks'], row['clean_successes'], row['task_successes'], row['episodes']) == (
            match['attacks'], match['eligible'], match['task_successes'], match['episodes'])
    output = dict(complete=True, checked_patch_episodes=sum(map(len, records.values())),
        clean_episodes=len(seeds), clean_successes=sum(row['success'] for row in clean.values()),
        audit='Episode JSON state/RGB hashes, prompts, first noise, texture hashes and planned centers matched; NPZ existence checked. NPZ payloads were not independently rehashed.',
        bootstrap='20000 paired draws over all initial seeds; preserve all positions within seed; pointwise 95% intervals. Post-hoc Random comparison.',
        rows=rows, comparisons=comparisons)
    (root / 'paired_random_comparison.json').write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n')
    (root / 'paired_random_source_hashes.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
    print(json.dumps({k: output[k] for k in ['complete', 'checked_patch_episodes', 'clean_episodes', 'clean_successes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    analyze(Path(parser.parse_args().root).resolve())
