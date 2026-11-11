"""Export the eight already-selected native replay states without GPU work."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FRAME_IDS = np.array([167, 168, 169, 170, 183, 184, 185, 186], dtype=np.int64)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    base = ROOT / 'result/estimated_proxy_closed_loop_20261105/capture/observations.npz'
    wrist = ROOT / 'result/estimated_proxy_dual_eot_20261105/wrist_capture/observations.npz'
    manifest_path = ROOT / 'pi05probe/out/render_proxy_eot/data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    fixtures = {int(r['episode']): r for r in manifest['fixtures']}
    per_frame = ['sim_states', 'env_seed', 'ep_id', 'state8', 'clean_base',
                 'clean_wrist', 'train_mask', 'validation_mask']
    constants = ['task', 'prompt', 'cell_world', 'patch_m']
    arrays = {'frame_ids': FRAME_IDS}
    with np.load(base, allow_pickle=False) as b, np.load(wrist, allow_pickle=False) as w:
        assert len(b['ep_id']) == len(w['ep_id']) == manifest['frames'] == 187
        for key in per_frame + constants:
            assert np.array_equal(b[key], w[key]), ('base/wrist mismatch', key)
        for key in per_frame:
            arrays[key] = b[key][FRAME_IDS].copy()
        for key in constants:
            arrays[key] = b[key].copy()
        arrays['native_base256'] = b['images'][FRAME_IDS, 0].copy()
        arrays['native_wrist256'] = w['images'][FRAME_IDS, 0].copy()
    assert arrays['validation_mask'].all() and not arrays['train_mask'].any()
    assert arrays['ep_id'].tolist() == [10] * 4 + [11] * 4
    assert arrays['sim_states'].shape == (8, 79)
    assert np.isfinite(arrays['sim_states']).all()
    for ep, seed in zip(arrays['ep_id'], arrays['env_seed']):
        assert int(seed) == int(fixtures[int(ep)]['env_seed']) == 10000 + int(ep)
    arrays['fixture_body_pos'] = np.array([fixtures[int(ep)]['body_pos'] for ep in arrays['ep_id']], dtype=np.float64)
    arrays['fixture_body_quat'] = np.array([fixtures[int(ep)]['body_quat'] for ep in arrays['ep_id']], dtype=np.float64)
    assert arrays['fixture_body_pos'].shape == (8, 38, 3)
    assert arrays['fixture_body_quat'].shape == (8, 38, 4)
    out.mkdir(parents=True)
    payload = out / 'native_replay_states.npz'
    np.savez_compressed(payload, **arrays)
    with np.load(payload, allow_pickle=False) as restored:
        assert set(restored.files) == set(arrays)
        for key, value in arrays.items():
            assert restored[key].dtype == value.dtype
            assert np.array_equal(restored[key], value), key
    shutil.copy2(manifest_path, out / 'source_pool_manifest.json')
    for relative in ['AGENTS.md', 'skills/gpu-preflight/SKILL.md',
                     'skills/gpu-preflight/scripts/check_gpus.py']:
        dest = out / 'current_gpu_policy' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    report = dict(
        complete=True, cpu_only=True, original_frame_ids=FRAME_IDS.tolist(),
        rows=8, source_pool_rows=187,
        selected_by='Exact missing IDs from dl stage_a/native_blocker.json; not selected by outcome',
        source_files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha256(p))
                      for p in [base, wrist, manifest_path]],
        arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype),
                        sha256_raw_c_order=hashlib.sha256(v.tobytes(order='C')).hexdigest())
                for k, v in arrays.items()},
        payload_files=[dict(path=str(p.relative_to(out)), bytes=p.stat().st_size, sha256=sha256(p))
                       for p in sorted(out.rglob('*')) if p.is_file()],
        replay='Map by frame_ids, not row numbers. Seed+reset matching env_seed/fixture, then regenerate_obs_from_state(sim_states[row]); no extra settling.',
        image_convention='native_*256 are saved legacy policy-oriented capture images (already flipped relative to raw LIBERO observation). clean_* are production224. Do not flip these references again.',
        fixture='Original clean-model body_pos/body_quat per frame from source manifest; a newly added visual patch body is additional.',
        gpu_policy='nnmc62 is prohibited for all new tasks/controllers, even idle or existing-service reuse; read-only GPU audit only.',
        exact_roundtrip=True,
    )
    (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(out=str(out), rows=8, payload_bytes=payload.stat().st_size,
                          payload_sha256=sha256(payload), exact_roundtrip=True)))


if __name__ == '__main__':
    main()
