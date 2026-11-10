"""Read-only six-point replay of #57; never trains or runs closed loops."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(os.environ.get('WAMATTACK_PROJECT_ROOT', '/shared/user65/workspace/chence/WAMattack'))
sys.path.insert(0, str(ROOT / 'robotwinprobe'))
import numpy as np
import pi_imagewam_joint_eot as J

ORIGINAL = ROOT / 'robotwinprobe/out/pi_imagewam_joint_eot_20261031'
DIAG = ORIGINAL / 'audit/gradient_replay'
STEPS = [1, 100, 200, 300, 400, 500]
KINDS = ['pi', 'imagewam']
PORTS = {'pi': 18962, 'imagewam': 18963}
CPU = '/shared/user65/miniconda3/bin/python'
PY = {'pi': '/shared/user65/miniconda3/envs/openpi-server/bin/python',
      'imagewam': '/shared/user64/workspace/yuhao/imagewam/ImageWAM/.venv/bin/python'}


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ash(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    tmp.replace(p)


def fixed(p, obj):
    if p.exists():
        assert read(p) == obj, ('immutable diagnostic artifact changed', str(p))
    else:
        dump(p, obj)


def preserved():
    names = ['protocol.json', 'sealed_texture.json', 'training/state.npz',
             'training_plan.json', 'eot_plan.json', 'contexts.pt',
             'completion_verification.json', 'gradient_ledgers/pi.jsonl',
             'gradient_ledgers/imagewam.jsonl', 'textures/joint_eot.png']
    return {n: sha(ORIGINAL / n) for n in names}


def source_hashes():
    manifest = read(ORIGINAL / 'source_snapshots/prepare_245056/manifest.json')
    names = ['libero_joint_eot.py', 'imagewam_libero_self_attack.py',
             'imagewam_libero_whitebox_adapter.py', 'imagewam_libero_adapter.py',
             'libero_pi_adapter.py', 'texture_eot.py', 'dual_view_proxy.py',
             'render_world_proxy.py', 'libero_joint_maxmin.py',
             'pi_imagewam_joint_eot.py', 'imagewam_libero_transfer.py']
    result = {}
    for name in names:
        folder = 'pi05probe' if name in ['dual_view_proxy.py', 'render_world_proxy.py'] else 'robotwinprobe'
        p = ROOT / folder / name
        result[name] = sha(p)
        assert result[name] == manifest[name], ('original source differs', name)
    return result


def cpu_prepare():
    DIAG.mkdir(parents=True, exist_ok=True)
    before = preserved()
    seal = read(ORIGINAL / 'sealed_texture.json')
    assert before['protocol.json'] == seal['protocol_sha256']
    assert before['training/state.npz'] == seal['state_sha256']
    assert before['textures/joint_eot.png'] == seal['png_sha256'] == 'b8ea653d0b4f88266058910fe724b13be306a7afbbd0d1f7f442a4f340827efb'
    for kind in KINDS:
        assert before[f'gradient_ledgers/{kind}.jsonl'] == seal['ledger_sha256'][kind]
    p = read(ORIGINAL / 'protocol.json')
    assert before['training_plan.json'] == p['training']['batch_plan_sha256']
    assert before['eot_plan.json'] == p['training']['eot_plan_sha256']
    assert before['contexts.pt'] == p['contexts_sha256']
    sources = source_hashes()
    batches = read(ORIGINAL / 'training_plan.json')
    draws = read(ORIGINAL / 'eot_plan.json')
    ledgers = {k: [json.loads(x) for x in (ORIGINAL / f'gradient_ledgers/{k}.jsonl').read_text().splitlines()] for k in KINDS}
    with np.load(ORIGINAL / 'training/state.npz') as z:
        history = json.loads(str(z['history']))
        signs = z['update_signs'].copy()
        final = z['texture'].copy()
    assert len(history) == len(signs) == 500
    assert signs.shape == (500, 1, 3, 128, 128)
    q = np.full((1, 3, 128, 128), .5, np.float32)
    points = []
    for idx, h in enumerate(history):
        step = idx + 1
        assert h['iteration'] == step and ash(q) == h['before_sha256']
        assert ash(signs[idx]) == h['sign_sha256'] and np.isin(signs[idx], [-1, 0, 1]).all()
        for k in KINDS:
            prov = h['provenance'][k]
            assert prov == ledgers[k][idx]
            assert prov['source_model'] == prov['server_model'] == k and prov['iteration'] == step
            assert prov['texture_sha256'] == ash(q)
            assert prov['indices'] == batches[idx]['indices'] and prov['seeds'] == batches[idx]['seeds']
            assert prov['draws_sha256'] == J.digest(draws[idx])
        if step in STEPS:
            f = DIAG / 'inputs' / f'step_{step:04d}.npz'
            f.parent.mkdir(parents=True, exist_ok=True)
            if not f.exists():
                np.savez_compressed(f, q=q.copy(), original_update_sign=signs[idx])
            with np.load(f) as saved:
                assert np.array_equal(saved['q'], q) and np.array_equal(saved['original_update_sign'], signs[idx])
            points.append(dict(step=step, input_npz=str(f), input_npz_sha256=sha(f),
                               before_sha256=ash(q), history=h, batch=batches[idx], draws=draws[idx]))
        q = np.clip(q + .01 * signs[idx].astype(np.float32), 0, 1).astype(np.float32)
        assert ash(q) == h['after_sha256']
    assert np.array_equal(q, final) and preserved() == before
    routes = J.cpu_checks()
    assert routes['passed']
    plan = dict(steps=STEPS, points=points, original_hashes=before, source_hashes=sources,
                ports=PORTS, policy_training=False, texture_training=False, closed_loops=0,
                reproduction_gate='exact raw loss, gradient norm, cosine and selected sign; stop and preserve any discrepancy',
                candidate_step=.01, candidate_clip=[0, 1], protocol='same original batch8/noise/draw; two candidate updates only')
    fixed(DIAG / 'replay_inputs.json', plan)
    out = dict(passed=True, original_500_before_after_sha_verified=True, source_hashes_match_original_snapshot=True,
               six_points=STEPS, routing_checks=routes, inputs_sha256=sha(DIAG / 'replay_inputs.json'),
               original_files_unchanged=True, runner_sha256=sha(Path(__file__)))
    fixed(DIAG / 'cpu_verification.json', out)
    return plan


def guard_work(work):
    work = work.resolve()
    assert work.is_relative_to((DIAG / 'work').resolve()) and work != ORIGINAL.resolve()
    assert not work.is_symlink()
    return work


def prepare_work(plan, attempt):
    work = guard_work(DIAG / 'work' / attempt)
    for name in ['audit', 'logs', 'gradient_ledgers', 'models/pi', 'models/imagewam', 'libero_config']:
        (work / name).mkdir(parents=True, exist_ok=True)
    for name in ['contexts.pt', 'training_plan.json', 'eot_plan.json', 'reference.json',
                 'audit/asset_hashes.json', 'libero_config/config.yaml']:
        dest = work / name
        assert not dest.exists(), ('attempt output already exists', str(dest))
        shutil.copy2(ORIGINAL / name, dest)
        assert sha(dest) == sha(ORIGINAL / name)
    p = read(ORIGINAL / 'protocol.json')
    p['diagnostic_only'] = dict(training=False, closed_loops=0, steps=STEPS, original_root=str(ORIGINAL))
    dump(work / 'protocol.json', p)
    for k in KINDS:
        cfg = read(ORIGINAL / f'models/{k}/protocol.json')
        cfg.update(parent_root=str(work), control_port=PORTS[k], isolation='fixed diagnostic replay; no training or closed loop')
        cfg['evaluation'].update(conditions=[], expected_new_episodes=0, texture_paths={}, texture_paths_by_init={})
        dump(work / f'models/{k}/protocol.json', cfg)
        assert cfg['training']['isolated_source_model'] == k
        assert cfg['parent_root'] != str(ORIGINAL) and cfg['control_port'] not in J.PORTS.values()
    assert not (work / 'models/fast').exists()
    dump(work / 'isolation_verification.json', dict(passed=True, root=str(work), ports=PORTS,
        writes_under_work_only=True, original_hashes=plan['original_hashes']))
    return work


def preflight(work, gpu, tag):
    skill = ROOT / 'skills/gpu-preflight/SKILL.md'
    assert skill.read_text()
    script = ROOT / 'skills/gpu-preflight/scripts/check_gpus.py'
    host = socket.gethostname().split('.')[0]
    assert not (host == 'nnmc63' and gpu == 0)
    for suffix, extra in [('cluster', []), ('target', ['--hosts', host])]:
        proc = subprocess.run([CPU, str(script), '--json', *extra], capture_output=True, text=True)
        data = json.loads(proc.stdout)
        dump(work / 'audit' / f'{tag}_{suffix}_preflight.json', data)
        node = next(n for n in data['nodes'] if n['host'] == host)
        assert node['status'] == 'ok', (host, node)
        card = next(g for g in node['gpus'] if g['index'] == gpu)
        assert card['eligible'] and card['name'] == 'NVIDIA L40S'
        assert not card['pids'] and card['used_mib'] <= 256 and card['utilization_pct'] == 0
    uuid = card['uuid']
    ecc = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
        '--query-gpu=ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total',
        '--format=csv,noheader,nounits'], text=True).strip()
    assert [s.strip() for s in ecc.split(',')] == ['0', '0'], ('bad or unknown ECC', ecc)
    record = dict(host=host, gpu=gpu, uuid=uuid, ecc=ecc, checked_at=time.time(), skill_sha256=sha(skill))
    dump(work / 'audit' / f'{tag}_ecc.json', record)
    return record


def rpc(kind, command, **payload):
    from multiprocessing.connection import Client
    with Client(('127.0.0.1', PORTS[kind]), authkey=J.AUTH) as c:
        c.send(dict(command=command, **payload))
        assert c.poll(1800), ('RPC timeout', kind, command)
        out = c.recv()
    if 'error' in out:
        raise RuntimeError(kind + ': ' + out['error'] + '\n' + out.get('traceback', ''))
    return out


def status(stage, **extra):
    completed = [s for s in STEPS if (DIAG / 'points' / f'step_{s:04d}.json').exists()]
    dump(DIAG / 'status.json', dict(stage=stage, host=socket.gethostname(), pid=os.getpid(),
        updated=time.time(), completed_replay_points=len(completed), completed_steps=completed,
        expected_replay_points=6, **extra))


def verify():
    plan = read(DIAG / 'replay_inputs.json')
    assert preserved() == plan['original_hashes'] and source_hashes() == plan['source_hashes']
    rows = []
    for point in plan['points']:
        step = point['step']
        f = DIAG / 'points' / f'step_{step:04d}.npz'
        row = read(f.with_suffix('.json'))
        assert row['arrays_sha256'] == sha(f) and row['passed']
        with np.load(f) as a:
            assert ash(a['q']) == point['before_sha256']
            for k in KINDS:
                g = a['gradient_' + k]
                delta = np.clip(a['q'] + .01 * np.sign(g), 0, 1).astype(np.float32) - a['q']
                assert np.array_equal(delta, a['delta_' + k])
                assert ash(g) == row['gradient_sha256'][k]
                assert np.isfinite(g).all() and np.linalg.norm(g) > 0
                assert float(np.mean(a['losses_before_' + k])) == point['history']['raw'][k]
                assert float(np.linalg.norm(g)) == point['history']['provenance'][k]['gradient_norm']
                for candidate in KINDS:
                    projection = float(np.sum(g.astype(np.float64) * a['delta_' + candidate].astype(np.float64)))
                    diff = float(np.mean(a[f'losses_after_{candidate}_on_{k}']) - np.mean(a['losses_before_' + k]))
                    assert projection == row['effects'][candidate][k]['first_order_projection']
                    assert diff == row['effects'][candidate][k]['finite_step_loss_difference']
            cosine = float(np.sum(a['gradient_pi'] * a['gradient_imagewam']) / (np.linalg.norm(a['gradient_pi']) * np.linalg.norm(a['gradient_imagewam'])))
            assert cosine == point['history']['cosine'] == row['cosine']
            agreement = float(np.mean(np.sign(a['gradient_pi']) == np.sign(a['gradient_imagewam'])))
            assert agreement == row['sign_agreement']
            with np.load(point['input_npz']) as original:
                chosen = point['history']['chosen']
                assert np.array_equal(np.sign(a['gradient_' + chosen]).astype(np.int8), original['original_update_sign'])
        audit_work = Path(row['work'])
        assert read(audit_work / 'models/pi/native_audit.json')['passed']
        assert read(audit_work / 'audit/native_and_gradient.json')['passed']
        assert read(audit_work / 'audit/imagewam_gray_match.json')['passed']
        rows.append(row)
    out = dict(passed=True, fixed_steps=STEPS, verified_points=len(rows), original_files_unchanged=True,
        source_paths_match_original_snapshot=True, exact_loss_norm_cosine_and_selected_sign_reproduction=True,
        actual_clipped_delta_and_same_draw_loss_differences_verified=True, closed_loop_episodes=0,
        texture_updates_committed=0, inputs_sha256=sha(DIAG / 'replay_inputs.json'),
        points={str(r['step']):sha(DIAG / 'points' / f"step_{r['step']:04d}.json") for r in rows},
        verified_at=time.time(), limitation='Six fixed points do not establish a causal account of all 500 updates or zero closed-loop ASR')
    dump(DIAG / 'completion_verification.json', out)
    return out


def run(gpus):
    DIAG.mkdir(parents=True, exist_ok=True)
    lock = (DIAG / 'diagnostic.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = cpu_prepare()
    assert len(gpus) == 2 and len(set(gpus)) == 2
    attempt = 'attempt_' + str(time.time_ns())
    work = prepare_work(plan, attempt)
    services, records = {}, {}
    execution = dict(host=socket.gethostname().split('.')[0], pipeline_pid=os.getpid(),
        command=sys.argv, work=str(work), diagnostic_lock=str(DIAG / 'diagnostic.lock'),
        gpus=gpus, backends=records, started_at=time.time(), runner_sha256=sha(Path(__file__)))
    def save_execution():
        execution['updated_at'] = time.time()
        dump(DIAG / 'execution.json', execution)
        dump(work / 'execution.json', execution)
    save_execution()
    try:
        J.S.R.check_assets(work)
        for k, gpu in zip(KINDS, gpus):
            status('start_' + k, work=str(work), gpu_job_started=True)
            # Fresh full-cluster and immediate selected-host/ECC before EACH model load.
            card = preflight(work, gpu, 'load_' + k)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=card['uuid'], PYTHONNOUSERSITE='1',
                PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', WAMATTACK_PROJECT_ROOT=str(ROOT),
                WAMATTACK_GPU_PREFLIGHT='/shared/user65/.codex/skills/gpu-preflight/scripts/check_gpus.py',
                WAMATTACK_LIBERO_CONFIG_PATH=str(work / 'libero_config'), HF_HUB_OFFLINE='1',
                JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
            with (work / 'logs' / (k + '_backend.log')).open('a') as log:
                proc = subprocess.Popen([PY[k], '-u', str(Path(__file__).resolve()), 'server',
                    '--work', str(work), '--kind', k], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            services[k] = proc
            records[k] = dict(card, pid=proc.pid, command=[PY[k], 'server', str(work), k])
            save_execution()
            ready = work / ('models/pi/ready.json' if k == 'pi' else 'ready.json')
            deadline = time.monotonic() + 600
            while not ready.exists():
                assert proc.poll() is None and time.monotonic() < deadline, ('backend load failed', k, proc.poll())
                time.sleep(2)
            assert read(ready)['pid'] == proc.pid
            status('native_audit_' + k, work=str(work), gpu_job_started=True)
            assert rpc(k, 'audit')['passed']
            if k == 'imagewam':
                actual = read(work / 'audit/native_and_gradient.json')['nominal_gray_loss']
                expected = read(ORIGINAL / 'audit/native_and_gradient.json')['nominal_gray_loss']
                assert abs(actual - expected) / expected < 1e-5
                dump(work / 'audit/imagewam_gray_match.json', dict(passed=True, actual=actual, reference=expected))
        for point in plan['points']:
            step = point['step']
            dest = DIAG / 'points' / f'step_{step:04d}.npz'
            if dest.with_suffix('.json').exists():
                continue
            status('replay_gradients', step=step, work=str(work), gpu_job_started=True)
            with np.load(point['input_npz']) as z:
                q = z['q'].copy()
                original_sign = z['original_update_sign'].copy()
            payload = dict(texture=q, indices=point['batch']['indices'], seeds=point['batch']['seeds'],
                           draws=point['draws'], iteration=step)
            out = {k:rpc(k, 'batch', **payload, gradient=True, source_model=k) for k in KINDS}
            arrays = dict(q=q)
            reproduction = {}
            for k in KINDS:
                arrays['gradient_' + k] = out[k]['gradient']
                arrays['losses_before_' + k] = np.asarray(out[k]['losses'])
                reproduction[k] = dict(raw_loss=float(np.mean(out[k]['losses'])),
                    gradient_norm=float(np.linalg.norm(out[k]['gradient'])), provenance=out[k]['gradient_provenance'])
            gg = {k:arrays['gradient_' + k] for k in KINDS}
            cosine = float(np.sum(gg['pi'] * gg['imagewam']) / (np.linalg.norm(gg['pi']) * np.linalg.norm(gg['imagewam'])))
            # Preserve the full evidence BEFORE applying any reproduction gate.
            evidence = work / 'evidence' / f'step_{step:04d}'
            evidence.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(evidence / 'raw_gradients.npz', **arrays)
            dump(evidence / 'reproduction.json', dict(actual=reproduction, expected=point['history'], cosine=cosine))
            for k in KINDS:
                assert out[k]['gradient_provenance'] == point['history']['provenance'][k], ('gradient provenance/norm mismatch', step, k)
                assert reproduction[k]['raw_loss'] == point['history']['raw'][k], ('raw loss mismatch', step, k)
            assert cosine == point['history']['cosine'], ('cosine mismatch', step, cosine, point['history']['cosine'])
            assert np.array_equal(np.sign(gg[point['history']['chosen']]).astype(np.int8), original_sign), ('selected sign mismatch', step)
            effects = {}
            for candidate in KINDS:
                q_after = np.clip(q + .01 * np.sign(gg[candidate]), 0, 1).astype(np.float32)
                delta = q_after - q
                arrays['delta_' + candidate] = delta
                effects[candidate] = {}
                for k in KINDS:
                    status('candidate_loss', step=step, candidate=candidate, source=k, work=str(work), gpu_job_started=True)
                    result = rpc(k, 'batch', **dict(payload, texture=q_after), gradient=False, source_model=k)
                    key = f'losses_after_{candidate}_on_{k}'
                    arrays[key] = np.asarray(result['losses'])
                    effects[candidate][k] = dict(
                        first_order_projection=float(np.sum(gg[k].astype(np.float64) * delta.astype(np.float64))),
                        finite_step_loss_difference=float(np.mean(arrays[key]) - np.mean(arrays['losses_before_' + k])),
                        loss_before=float(np.mean(arrays['losses_before_' + k])), loss_after=float(np.mean(arrays[key])))
            dest.parent.mkdir(parents=True, exist_ok=True)
            assert not dest.exists(), ('preserve existing partial evidence', str(dest))
            np.savez_compressed(dest, **arrays)
            dump(dest.with_suffix('.json'), dict(passed=True, step=step, work=str(work), cosine=cosine,
                sign_agreement=float(np.mean(np.sign(gg['pi']) == np.sign(gg['imagewam']))),
                gradient_sha256={k:ash(gg[k]) for k in KINDS}, arrays_sha256=sha(dest),
                input_npz_sha256=point['input_npz_sha256'], effects=effects,
                reproduction=reproduction, completed_at=time.time()))
            assert preserved() == plan['original_hashes']
            print('completed fixed replay point', step, json.dumps(effects), flush=True)
        status('verifying', work=str(work), gpu_job_started=True)
        verify()
        status('verified_awaiting_process_exit', work=str(work), gpu_job_started=True)
    except Exception as exc:
        err = dict(error=repr(exc), traceback=traceback.format_exc(), work=str(work), time=time.time())
        dump(work / 'error.json', err)
        dump(DIAG / 'errors' / (attempt + '.json'), err)
        status('runtime_error', work=str(work), gpu_job_started=bool(services), error=repr(exc))
        raise
    finally:
        for k, proc in services.items():
            if proc.poll() is None:
                try:
                    rpc(k, 'shutdown')
                except Exception:
                    proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            records[k].update(exit_code=proc.returncode, exited_at=time.time())
        execution.update(backends_exited=all(p.poll() is not None for p in services.values()), finished_at=time.time())
        save_execution()
        assert preserved() == plan['original_hashes']
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['cpu-checks', 'run', 'server', 'verify'])
    ap.add_argument('--work', type=Path)
    ap.add_argument('--kind', choices=KINDS)
    ap.add_argument('--gpus', type=int, nargs=2)
    args = ap.parse_args()
    if args.command == 'cpu-checks':
        cpu_prepare()
        print(json.dumps(read(DIAG / 'cpu_verification.json')))
    elif args.command == 'server':
        work = guard_work(args.work)
        plan = read(DIAG / 'replay_inputs.json')
        assert preserved() == plan['original_hashes'] and source_hashes() == plan['source_hashes']
        J.PORTS = PORTS
        J.server(work, args.kind)
    elif args.command == 'verify':
        print(json.dumps(verify()))
    else:
        run(args.gpus)
