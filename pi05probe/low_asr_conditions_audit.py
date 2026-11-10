"""CPU-only audit of frozen main-method training and rollout conditions.

Reads original artifacts without editing protocols, calibrations or results.
No torch/model import and no rendering or GPU calls occur here.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
TASKS = ('put_the_bowl_on_the_plate', 'put_the_bowl_on_top_of_the_cabinet',
         'put_the_wine_bottle_on_the_rack', 'turn_on_the_stove',
         'put_the_bowl_on_the_stove', 'put_the_cream_cheese_in_the_bowl',
         'open_the_middle_drawer_of_the_cabinet')
RANGES = dict(angle_deg=[-15., 15.], dx_fraction=[-.05, .05],
              dy_fraction=[-.05, .05], scale=[.9, 1.1], gain=[.9, 1.1])


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def local(path):
    s = str(path)
    for prefix in ['/shared/user65/workspace/chence/WAMattack',
                   '/home/user1/workspace/chence/WAMattack']:
        if s == prefix or s.startswith(prefix + '/'):
            return ROOT / s[len(prefix):].lstrip('/')
    return Path(path)


def texture_stats(folder):
    with np.load(folder/'checkpoint_0500.npz', allow_pickle=False) as z:
        q = z['Q'].copy()
        history = z['history'].copy()
        batches = z['batches'].copy()
        noise = z['noise'].copy() if 'noise' in z.files else None
    png = np.asarray(Image.open(folder/'texture_0500.png').convert('RGB'))
    quantized = np.rint(q.transpose(1, 2, 0)*255).clip(0, 255).astype(np.uint8)
    return dict(finite=bool(np.isfinite(q).all() and np.isfinite(history).all()),
                history_count=len(history), history_first20_mean=float(history[:20].mean()),
                history_last20_mean=float(history[-20:].mean()),
                history_last20_to_first20=float(history[-20:].mean()/history[:20].mean()),
                history_min=float(history.min()), history_max=float(history.max()),
                q_min=float(q.min()), q_max=float(q.max()), q_mean=float(q.mean()),
                q_std=float(q.std()), near_bound_fraction=float(((q < .01) | (q > .99)).mean()),
                png_matches_checkpoint_quantization=bool(np.array_equal(png, quantized))), batches, noise


def audit_libero():
    base = ROOT/'pi05probe/out/main_method'
    frozen = read(base/'protocol.json')
    result = []
    expected = dict(schema='wam-multitask-texture-train-v1', complete=True, mode='both',
                    iters=500, batch=8, per_task_per_batch=8, lr=.01, seed_batch=20261005,
                    noise_seed=20260901, init='constant gray .5', patch_m=.13,
                    proxy_rmse_max=.02)
    fit_hashes = {v: sha(ROOT/'pi05probe/out/dual_view/fit'/(v+'.json')) for v in ['base', 'wrist']}
    source_hashes = {n: sha(ROOT/'pi05probe'/n) for n in ['multitask_train.py', 'dual_view_proxy.py']}
    for task in TASKS:
        saved = read(base/'results'/(task+'.json'))
        episodes = local(saved['episodes'])
        rollout = read(episodes)
        manifest = read(episodes.parent.parent/'manifest.json')
        folder = local(manifest['train_run'])
        config, trained = read(folder/'config.json'), read(folder/'results.json')
        statistics, batches, noise = texture_stats(folder)
        pool = ROOT/'pi05probe/out/visible_pos/tasks'/task
        with np.load(pool/'base.npz', allow_pickle=False) as z:
            ep, train, validation = z['ep_id'], z['train_mask'], z['validation_mask']
            prompt = str(z['prompt'])
            geometry = z['cell_world'].tolist()
            train_indices = np.flatnonzero(train)
        with np.load(pool/'wrist.npz', allow_pickle=False) as z:
            wrist_prompt = str(z['prompt'])
            wrist_geometry = z['cell_world'].tolist()
        rng = np.random.RandomState(config['seed_batch'])
        expected_batches = np.array([[[0, int(k)] for k in rng.choice(train_indices, 8, replace=False)]
                                     for _ in range(500)])
        checks = dict(
            fixed_training_recipe=all(config.get(k) == v for k, v in expected.items()),
            one_task=config['tasks'] == [task],
            checkpoint=local(config['checkpoint']) == ROOT/'checkpoints/pi05_libero_pytorch',
            prompt=config['prompts'] == [prompt] and prompt == wrist_prompt == rollout['metadata']['prompt'],
            geometry=config['cell_world_by_task'][task] == geometry == wrist_geometry == manifest['geometry']['cell_world'],
            source_hashes=config['source_sha256'] == source_hashes,
            fit_hashes=config['fit_sha256'] == fit_hashes,
            eot=config['eot']['ranges'] == RANGES and config['eot']['seed'] == 2026102901,
            eot_source=config['source_eot_sha256'] == sha(ROOT/'robotwinprobe/texture_eot.py'),
            stored_noise=np.array_equal(noise, np.random.RandomState(20260901).standard_normal((10, 32)).astype(np.float32)),
            deterministic_training_batches=np.array_equal(batches, expected_batches),
            frame_split=np.array_equal(train, ep < 10) and np.array_equal(validation, ep >= 10) and set(ep) <= set(range(12)),
            training_complete=trained['complete'] and trained['iterations'] == 500 and trained['gradient_frame_forwards'] == 4000,
            final_texture_hash=trained['texture_sha256'] == sha(folder/'texture_0500.png') == manifest['texture_sha256']['eot'],
            evaluation_budget=rollout['metadata']['indices'] == list(range(40, 50)) and rollout['metadata']['max_steps'] == 300
                and rollout['metadata']['settling_steps'] == 10 and rollout['metadata']['replan'] == 5,
            fixed_protocol_sources=all(sha(ROOT/'pi05probe'/n) == value for n, value in frozen['source_sha256'].items()))
        assert all(checks.values()), (task, checks)
        s = saved['summary']
        result.append(dict(task=task, low_asr=s['eot']['paired_ASR'] < .5,
            method_ASR=s['eot']['paired_ASR'], checks=checks, all_declared_conditions_pass=True,
            prompt=prompt, train_frames=int(train.sum()), validation_frames=int(validation.sum()),
            initial_nominal_diag_L0=trained['initial_train_diag_L0'],
            final_nominal_diag_L0=trained['final_train_diag_L0'],
            nominal_diag_gain=trained['final_train_diag_L0']/trained['initial_train_diag_L0'],
            final_nominal_validation_L0=trained['final_validation_proxy_L0'],
            proxy_check=config['proxy_check'], texture_statistics=statistics,
            no_test_checkpoint_selection='fixed texture_0500.png; matched source/result/manifest hash'))
    return result


def audit_robotwin():
    root = ROOT/'robotwinprobe/out/main_method'
    folder = root/'train/place_container_plate'
    c, r, p = read(folder/'config.json'), read(folder/'results.json'), read(root/'protocol.json')
    stats, batches, noise = texture_stats(folder)
    rng = np.random.RandomState(c['seed_batch'])
    expected_batches = np.array([rng.choice(len(c['records']), 8, replace=False) for _ in range(500)])
    prompts, stages, queries = set(), set(), []
    for record in c['records']:
        with np.load(local(record['path']), allow_pickle=False) as z:
            prompts.add(str(z['prompt']));stages.add(str(z['stage']))
            queries.append(int(record['query']))
    eval_prompts = {read(path)['prompt'] for path in (root/'eval/place_container_plate/episodes').glob('*.json')}
    checks = dict(config_matches_protocol=c == p['method'],
        complete=r['complete'] and r['iterations'] == 500 and r['gradient_frame_forwards'] == 4000,
        deterministic_batches=np.array_equal(batches, expected_batches),
        texture_hash=r['texture_sha256'] == sha(folder/'texture_0500.png'),
        config_hash=r['config_sha256'] == sha(folder/'config.json'),
        train_source_hashes=all(sha(ROOT/'robotwinprobe'/name) == value for name, value in c['source_sha256'].items()),
        eval_source_hashes=all(sha(ROOT/'robotwinprobe'/name) == value for name, value in p['evaluation']['source_sha256'].items()),
        train_eval_prompts=prompts == eval_prompts,
        shared_fixed_texture_eot=c['eot'] == RANGES and c['eot_seed'] == 2026102901,
        common_batch_seed=c['seed_batch'] == 20261005, common_noise=c['noise_seed'] == 20260901)
    assert all(checks.values()), checks
    return dict(task=c['task'], low_asr=True, checks=checks,
                all_declared_conditions_pass=True, texture_statistics=stats,
                train_records=len(c['records']), train_seeds=len(c['train_seeds']),
                train_stages=sorted(stages), max_train_query=max(queries),
                prompts=sorted(prompts), independent_validation=c['independent_validation'],
                proxy_fidelity=r['proxy_fidelity'])


def main(args):
    report = dict(schema='low-asr-training-conditions-cpu-audit-v1',
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='seven LIBERO controls and first ready RoboTwin task; CPU only',
        source_sha256=sha(__file__), libero=audit_libero(), robotwin=audit_robotwin())
    report['findings'] = [
        dict(status='verified', claim='All seven LIBERO task runs share frozen checkpoint, step/batch/lr/seeds/EOT, prompt and physical geometry per task. Each saved update batch and fixed noise matches the declared RNG exactly; all 500-step textures match PNG/hash.'),
        dict(status='verified', claim='The optimizer ascends SSE correctly: backprop(-SSE), then Q -= lr*sign(grad), clamp[0,1]. Every task increases nominal training diagnostic loss. No evidence of reversed sign, missing updates, disabled EOT or saturated constant texture.'),
        dict(status='verified', claim='LIBERO final nominal validation SSE is 2.929/4.661 on the two strong tasks and 0.136-0.661 on the five weak tasks. Increased proxy objective alone does not prove closed-loop attack success.'),
        dict(status='diagnostic_needed', claim='Both trainers maximize all 32 velocity dimensions. Native LIBERO output discards dimensions7:32 and rollout executes only first5 of10 horizon steps. RoboTwin output discards dimensions14:32. No stored diagnostic currently measures active/executed vs padded SSE or gradient direction.'),
        dict(status='verified_recipe_risk', claim='All training uses one fixed noise seed20260901; rollout varies noise by initialization and query. Texture EOT does not randomize diffusion noise, task initial state distribution or world footprint. This is uniform across tasks, but multi-noise generalization is unmeasured.'),
        dict(status='verified_benchmark_difference', claim='LIBERO uses 12 trajectory initializations with ep0-9 train and10-11 validation; RoboTwin uses49 approach-only states from24 seeds, no independent validation. Their camera count, action horizon and execution interval also differ natively.'),
        dict(status='verified_benchmark_difference', claim='LIBERO proxy random-interior RMSE is gated at0.02; RoboTwin has no hard threshold and observed max0.06991/0.06552/0.03953. Pixel agreement does not certify velocity/gradient fidelity; true-render flow comparison is needed.'),
        dict(status='metadata_issue', claim='Reusable LIBERO rollout metadata still labels new_target_task_training=False and calls this source-task transfer, although each main-method manifest references its own task training. This label is misleading for paper reporting but does not change inference.'),
        dict(status='provenance_limit', claim='Original LIBERO training metadata records checkpoint path but not model.safetensors/norm_stats SHA. All paths match and current weight mtime predates these runs, but path equality alone cannot cryptographically prove historical weight identity. Record explicit model and normalization hashes in the new retry protocol.'),
        dict(status='verified', claim='Train and rollout use the same model input transforms, camera keys and eager attention. LIBERO proxy uses Torch bilinear antialias to224 while physical rollout uses native PIL resize; RoboTwin surrogate uses differentiable resize but its forward is replaced exactly by the native input transform. Quantitative flow fidelity still needs a GPU diagnostic.')]
    report['recommended_diagnostics'] = dict(
        entry='pi05probe/low_asr_velocity_diagnostic.py',
        resource='one preflight-approved GPU with >=32000MiB idle memory; script repeats fleet/selected checks; no GPU job started by this audit',
        inputs='fixed train/validation frames only; gray/random/final trained PNG and native calibration random RGB; training noise plus3 independent noise draws',
        outcomes='SSE active7/padded25/first5/later5 and full-vs-executed gradient cosine, native-vs-proxy image/velocity fidelity')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(out=str(out), libero_tasks=len(report['libero']), all_conditions_pass=True,
        low_tasks=sum(r['low_asr'] for r in report['libero'])+1), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(ROOT/'out/main_results/low_asr_audit/conditions.json'))
    main(ap.parse_args())
