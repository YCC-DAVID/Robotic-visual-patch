"""One fresh official DRIFT patch: Spatial pool -> upstream JAX PGD -> digital -> render.

Every GPU stage is separately preflighted. This runner never sends stop signals.
Policy services finish naturally when their sole rollout client disconnects.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from drift_pi0_common import (ROOT, DRIFT, SUITES, TRAIN_INITS, TEST_INITS, CONDITIONS,
                                dump, read, sha, stage, now, MODEL_CONFIG,
                                PATCH_SIZE, PATCH_Y, PATCH_X, ACTION_HORIZON, ACTION_DIM)


def protocol(args):
    checkpoint = Path(args.checkpoint)
    files = sorted(p for p in checkpoint.rglob('*') if p.is_file())
    return dict(schema='drift-official-pi0-digital-to-native-render-v1',
        purpose='Train one DRIFT patch as published, verify digital rollout, then apply the same patch through native renderer',
        upstream_repository='https://github.com/TaeHoseong/DRIFT',
        upstream_commit=subprocess.check_output(['git', '-C', str(DRIFT), 'rev-parse', 'HEAD'], text=True).strip(),
        model_config=MODEL_CONFIG, backend='JAX', checkpoint=str(checkpoint),
        action_horizon=ACTION_HORIZON, action_dim=ACTION_DIM,
        checkpoint_sha256={str(p.relative_to(checkpoint)): sha(p) for p in files},
        collection=dict(suite='libero_spatial', tasks=10, episodes_per_task=3,
            initial_indices=list(TRAIN_INITS), no_success_filter=True,
            observations='native RGB256 rotated180, official PIL resize224, state8 and per-task prompt',
            frame_count='measure actual query count; do not force 696'),
        training=dict(implementation='unmodified third_party/DRIFT/attack_patch.py',
            patch_hw=[PATCH_SIZE, PATCH_SIZE], wrist_top_left_yx=[PATCH_Y, PATCH_X], wrist_only=True,
            batch=8, steps=500, alpha=.01, seed=0, num_denoise_steps=10,
            attacked_steps=[0], initialization='gray0.5', noise='fresh official JAX random draw every PGD step',
            projection='clip[0,1]', eot=False, render_aware_training=False, checkpoint_selection='fixed final iteration500'),
        evaluation=dict(suites=list(SUITES), task_count=40, initial_indices=list(TEST_INITS),
            conditions=list(CONDITIONS), episodes_per_condition=400, total_episodes=2000,
            warmup_steps=10, replan_steps=5, denoise_steps=10,
            max_steps=dict(libero_spatial=220, libero_goal=300, libero_object=280, libero_10=520),
            primary_metric='(clean_SR-condition_SR)/clean_SR; higher ASR is stronger; lower robot SR is stronger',
            additional_metric='paired failures among clean successes, reported separately',
            runtime_errors='fail stage, never count as attack success',
            noise='explicit matched JAX-policy input; same numpy.default_rng seed by suite/task/init/query in all conditions',
            seed_count=1, patch_count=1),
        render_extension=dict(attachment='one rigid zero-mass visual plane on the gripper camera parent',
            pose_selection='back-project prescribed ROI at visible gripper depth from training init only',
            geometry='freeze before evaluation; preserve inertias, contacts, initial physics; check UV and segmentation',
            camera_inputs='native renderer for both cameras; no digital overwrite in render conditions',
            limitation='DRIFT gives an image ROI, not a metric pose. This mounted plate is an explicit physical extension, not claimed original geometry.'),
        deviations_from_released_defaults=[
            'Explicit Spatial-only collection (upstream collect script defaults to all suites; paper specifies Spatial only)',
            'Training init10-12 and evaluation init0-9 are disjoint (released scripts overlap init0-2)',
            'Explicit matched action noise for paired conditions; upstream evaluation does not reset policy RNG per condition',
            'One training seed / patch requested; no claim of the paper three-seed mean',
            'Per-episode atomic provenance/resume and strict failure handling instead of counting exceptions as failures'],
        upstream_sha256={str(p.relative_to(DRIFT)): sha(p) for p in sorted(DRIFT.rglob('*.py'))},
        source_sha256={n: sha(ROOT / 'pi05probe' / n) for n in
            ['drift_pi0_common.py', 'drift_pi0_model.py', 'drift_pi0_sim.py', 'drift_pi0_pipeline.py']})


def summary(root):
    import numpy as np
    per_suite, total = {}, 0
    for suite in SUITES:
        groups = {c: [] for c in CONDITIONS}
        for c in CONDITIONS:
            for path in sorted((root / 'eval' / suite).glob('task*/' + c + '/init*.json')):
                row = read(path)
                assert row['runtime_error'] is False and row['condition'] == c and row['suite'] == suite
                assert row['record_sha256'] == sha(row['record'])
                groups[c].append(row)
        clean = {(r['task_id'], r['init_index']): r for r in groups['clean']}
        assert len(clean) == len(groups['clean']), 'duplicate clean episode'
        values = {}
        for c, rows in groups.items():
            keys = {(r['task_id'], r['init_index']) for r in rows}
            assert len(keys) == len(rows), 'duplicate condition episode'
            complete = keys == {(t, i) for t in range(10) for i in TEST_INITS}
            sr = sum(r['success'] for r in rows) / len(rows) if rows else None
            paired = [r for r in rows if clean.get((r['task_id'], r['init_index']), {}).get('success')]
            clean_sr = sum(r['success'] for r in clean.values()) / 100 if len(clean) == 100 else None
            values[c] = dict(n=len(rows), complete=complete, successes=sum(r['success'] for r in rows), SR=sr,
                relative_ASR=(clean_sr - sr) / clean_sr if complete and clean_sr else None,
                paired_n=len(paired), paired_failures=sum(not r['success'] for r in paired),
                paired_ASR=sum(not r['success'] for r in paired) / len(paired)
                    if complete and len(clean) == 100 and paired else None)
            total += len(rows)
        per_suite[suite] = values
    complete = all(v['complete'] for s in per_suite.values() for v in s.values())
    result = dict(complete=complete, updated=now(), episodes=total, target_episodes=2000,
                  suites=per_suite, seed_count=1, patch_count=1)
    dump(root / 'summary.json', result)
    return result


def main(args):
    import visible_pos_pipeline as V
    from main_method_job_runtime import install
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ['logs', 'resources', 'services', 'train']:
        (root / name).mkdir(exist_ok=True)
    V.DEST = root
    V.P.MODEL = args.model_python
    V.P.SIM = args.sim_python
    install(V)
    gpu = json.loads(args.gpu)
    assert gpu['host'] == socket.gethostname().split('.')[0]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='4',
        XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.70',
        OPENPI_DATA_HOME=str(Path(args.checkpoint).parents[2]))
    sim_env = dict(env, MUJOCO_EGL_DEVICE_ID=str(gpu['egl_index']), OMP_NUM_THREADS='2')
    command = [args.sim_python, '-u', str(ROOT / 'pi05probe/drift_pi0_sim.py')]
    model_command = [args.model_python, '-u', str(ROOT / 'pi05probe/drift_pi0_model.py')]
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (root / 'protocol.json').exists():
            stage(root, 'preparing_protocol')
            dump(root / 'protocol.json', protocol(args))
        frozen = read(root / 'protocol.json')
        for name, digest in frozen['source_sha256'].items():
            assert digest == sha(ROOT / 'pi05probe' / name), 'source changed after protocol freeze: ' + name
        from PIL import Image
        import numpy as np
        if not (root / 'random.png').exists():
            random = np.random.default_rng(0).integers(0, 256, size=(PATCH_SIZE, PATCH_SIZE, 3), dtype=np.uint8)
            Image.fromarray(random).save(root / 'random.png')

        @contextmanager
        def allocation(label):
            """Wait for an idle card on this node; never stop another process."""
            phase = label
            while True:
                report = V.inventory(label + '_' + str(time.time_ns()), 'waiting_inventory', min_free_mib=24000)
                node = next((n for n in report['nodes'] if n['host'] == gpu['host']), None)
                available = [] if node is None or node.get('status') != 'ok' else [
                    g for g in node['gpus'] if g.get('eligible') and g.get('idle') and not g.get('faulty')
                    and (not gpu.get('name') or g['name'] == gpu['name'])
                    and not (gpu['host'] == 'nnmc63' and g['index'] == 0)]
                available.sort(key=lambda g: (g['uuid'] != gpu['uuid'], g['index']))
                if not available:
                    stage(root, 'waiting_for_idle_gpu', pending_phase=phase, host=gpu['host'])
                    time.sleep(30)
                    continue
                chosen = available[0]
                check = subprocess.run([args.model_python, str(ROOT / 'pi05probe/egl_device_audit.py'),
                    '--uuid', chosen['uuid']], env=dict(os.environ, CUDA_VISIBLE_DEVICES=chosen['uuid']),
                    capture_output=True, text=True, timeout=30)
                if check.returncode:
                    stage(root, 'waiting_for_valid_egl_mapping', pending_phase=phase, error=check.stderr[-1500:])
                    time.sleep(30)
                    continue
                mapping = json.loads(check.stdout)
                gpu.update(index=chosen['index'], uuid=chosen['uuid'], name=chosen['name'], egl_index=mapping['egl_index'])
                env['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
                sim_env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], MUJOCO_EGL_DEVICE_ID=str(gpu['egl_index']))
                dump(root / 'resources' / (label + '_mapping.json'), mapping)
                acquired = False
                try:
                    with V.allocation(label, gpu, 24000):
                        acquired = True
                        stage(root, phase, assigned_gpu=dict(gpu))
                        yield
                    return
                except V.ResourceBlocked:
                    if acquired:
                        raise
                    # One sequential controller; no other worker's abort is cleared.
                    V._ABORT.clear()
                    stage(root, 'waiting_for_idle_gpu', pending_phase=phase, reason='allocation race; full checks repeated')
                    time.sleep(30)

        @contextmanager
        def service(label):
            log = root / 'logs' / ('server_' + label + '.log')
            with log.open('w') as stream:
                cmd = model_command + ['serve', '--root', str(root), '--checkpoint', args.checkpoint,
                                       '--port', str(args.port)]
                p = subprocess.Popen(cmd, env=env, cwd=ROOT / 'pi05probe', stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
                V.process_event(p, cmd, log, env)
                try:
                    started = time.monotonic()
                    while '[finite-service] listening' not in log.read_text():
                        if p.poll() is not None:
                            raise RuntimeError('JAX policy failed before ready: ' + str(log))
                        if time.monotonic() - started > 1200:
                            raise RuntimeError('JAX policy readiness timeout; process left untouched: ' + str(p.pid))
                        time.sleep(1)
                    yield
                finally:
                    # The sole client's close triggers normal service return.
                    code = p.wait()
                    V.event('jax_service_finished', pid=p.pid, returncode=code)
                    if code:
                        raise subprocess.CalledProcessError(code, cmd)

        if not (root / 'geometry/complete.json').exists():
            stage(root, 'geometry')
            with allocation('drift_pi0_geometry'):
                V.run(command + ['geometry', '--root', str(root)], 'geometry.log', sim_env, 1200)
        if args.geometry_only:
            stage(root, 'geometry_verified')
            return
        if not (root / 'collection/complete.json').exists():
            stage(root, 'collecting')
            with allocation('drift_pi0_collect'), service('collect'):
                V.run(command + ['collect', '--root', str(root), '--port', str(args.port)],
                      'collect.log', sim_env, 14400)
        if not (root / 'train/complete.json').exists():
            stage(root, 'training')
            with allocation('drift_pi0_train'):
                V.run(model_command + ['train', '--root', str(root), '--checkpoint', args.checkpoint],
                      'train.log', env, 14400)
        for label, conditions in [('digital', 'clean,random_digital,drift_digital'),
                                  ('render', 'random_render,drift_render')]:
            existing = summary(root)
            if all(existing['suites'][s][c]['complete'] for s in SUITES for c in conditions.split(',')):
                continue
            stage(root, 'evaluating_' + label)
            with allocation('drift_pi0_eval_' + label), service(label):
                V.run(command + ['evaluate', '--root', str(root), '--port', str(args.port),
                                 '--conditions', conditions], 'eval_' + label + '.log', sim_env, 172800)
            summary(root)
        result = summary(root)
        assert result['complete'], 'incomplete rollout coverage'
        stage(root, 'complete', episodes=result['episodes'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--gpu', required=True)
    ap.add_argument('--checkpoint', default='/shared/user65/.cache/openpi/openpi-assets/checkpoints/pi0_libero')
    ap.add_argument('--model-python', default='/shared/user65/miniconda3/envs/openpi-server/bin/python')
    ap.add_argument('--sim-python', default='/shared/user65/miniconda3/envs/openpi-libero/bin/python')
    ap.add_argument('--port', type=int, default=18681)
    ap.add_argument('--geometry-only', action='store_true')
    args = ap.parse_args()
    try:
        main(args)
    except BaseException as error:
        stage(args.root, 'failed', error=repr(error))
        raise
