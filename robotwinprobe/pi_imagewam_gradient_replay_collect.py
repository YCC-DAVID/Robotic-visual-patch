"""Collect only the six authorized #57 diagnostic points; retain every exact historical gate result."""
from pi_imagewam_gradient_replay import *
import pi_imagewam_gradient_replay as D


def status(stage, **extra):
    work = Path(extra['work']) if extra.get('work') else None
    rows = [read(p) for p in sorted((DIAG / 'collected').glob('step_*.json'))]
    dump(DIAG / 'status.json', dict(stage=stage, host=socket.gethostname().split('.')[0], pid=os.getpid(),
        updated=time.time(), completed_replay_points=sum(bool(r['passed']) for r in rows),
        completed_steps=[r['step'] for r in rows if r['passed']], expected_replay_points=6,
        collected_diagnostic_points=len(rows), collected_steps=[r['step'] for r in rows],
        historical_gate_tolerance=0, **extra))


def verify_collection(work):
    plan = read(DIAG / 'replay_inputs.json')
    assert preserved() == plan['original_hashes'] and source_hashes() == plan['source_hashes']
    rows = []
    for point in plan['points']:
        f = DIAG / 'collected' / f"step_{point['step']:04d}.npz"
        row = read(f.with_suffix('.json')); assert row['arrays_sha256'] == sha(f)
        assert row['collection_complete'] and row['input_npz_sha256'] == point['input_npz_sha256']
        with np.load(f) as a:
            assert ash(a['q']) == point['before_sha256']
            for k in KINDS:
                g = a['gradient_' + k]
                assert ash(g) == row['gradient_sha256'][k]
                assert np.array_equal(a['delta_' + k], np.clip(a['q'] + .01 * np.sign(g), 0, 1).astype(np.float32) - a['q'])
                for candidate in KINDS:
                    effect = row['effects'][candidate][k]
                    assert effect['first_order_projection'] == float(np.sum(g.astype(np.float64) * a['delta_' + candidate].astype(np.float64)))
                    assert effect['finite_step_loss_difference'] == float(np.mean(a[f'losses_after_{candidate}_on_{k}']) - np.mean(a['losses_before_' + k]))
        rows.append(row)
    for p in ['models/pi/native_audit.json','audit/native_and_gradient.json','audit/imagewam_gray_match.json']:
        assert all(read(Path(row['work']) / p)['passed'] for row in rows)
    out = dict(collection_integrity_verified=True, fixed_steps=STEPS, collected_points=len(rows),
        exact_historical_passed_points=sum(bool(r['passed']) for r in rows),
        exact_historical_gradient_gate_passed=all(r['passed'] for r in rows), historical_tolerance=0,
        original_files_unchanged=True, source_paths_match_original=True, closed_loop_episodes=0,
        texture_updates_committed=0, points={str(r['step']): sha(DIAG / 'collected' / f"step_{r['step']:04d}.json") for r in rows},
        limitation='Native recomputed gradients. Exact historical mismatches remain failures. Six points cannot establish all-500-step causality or explain zero ASR by themselves.')
    dump(work / 'collection_verification.json', out)
    return out


def run(gpus):
    DIAG.mkdir(parents=True, exist_ok=True)
    lock = (DIAG / 'diagnostic.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = cpu_prepare()
    assert len(gpus) == 2 and len(set(gpus)) == 2
    attempt = 'collection_' + str(time.time_ns())
    work = prepare_work(plan, attempt)
    services, records = {}, {}
    execution = dict(host=socket.gethostname().split('.')[0], pipeline_pid=os.getpid(),
        command=sys.argv, work=str(work), diagnostic_lock=str(DIAG / 'diagnostic.lock'),
        gpus=gpus, backends=records, started_at=time.time(), runner_sha256=sha(Path(__file__)))
    def save_execution():
        execution['updated_at'] = time.time()
        dump(DIAG / 'execution.json', execution)
        dump(work / 'execution.json', execution)
    if (DIAG / 'execution.json').exists():
        shutil.copy2(DIAG / 'execution.json', work / 'previous_execution.json')
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
            dest = DIAG / 'collected' / f'step_{step:04d}.npz'
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
            exact_checks = {
                'provenance': {k: out[k]['gradient_provenance'] == point['history']['provenance'][k] for k in KINDS},
                'raw_loss': {k: reproduction[k]['raw_loss'] == point['history']['raw'][k] for k in KINDS},
                'cosine': cosine == point['history']['cosine'],
                'selected_sign': bool(np.array_equal(np.sign(gg[point['history']['chosen']]).astype(np.int8), original_sign))}
            exact_passed = all(exact_checks['provenance'].values()) and all(exact_checks['raw_loss'].values()) and exact_checks['cosine'] and exact_checks['selected_sign']
            dump(evidence / 'exact_historical_checks.json', dict(passed=exact_passed, checks=exact_checks, tolerance=0))
            # Identity and native audits remain hard gates. Numeric discrepancies are evidence, never accepted as a historical match.
            for k in KINDS:
                expected = point['history']['provenance'][k]
                actual = out[k]['gradient_provenance']
                assert reproduction[k]['raw_loss'] == point['history']['raw'][k], ('new forward loss mismatch; preserve and localize first', step, k)
                if k == 'pi' and actual != expected:
                    # Exact historical checks above stay failed. Only independent recomputed
                    # diagnostic evidence may proceed after the native numerical source is localized.
                    localized = read(DIAG / 'pi_operator_localization_analysis.json')
                    assert localized['located_equal_input_unequal_output_operator']
                    assert localized['same_original_batch_noise_draw'] and localized['native_functions_unchanged']
                    assert localized['operator'] == 'aten._upsample_bilinear2d_aa_backward.default'
                    assert all(sample['inputs_equal'] and not sample['outputs_equal'] for sample in localized['samples'])
                    for sample in localized['samples']:
                        for path, digest in sample['trace_sha256'].items():
                            assert sha(Path(path)) == digest
                assert set(expected) == set(actual)
                assert {key: val for key, val in actual.items() if key != 'gradient_norm'} == {key: val for key, val in expected.items() if key != 'gradient_norm'}
                assert gg[k].shape == q.shape and np.isfinite(gg[k]).all() and np.linalg.norm(gg[k]) > 0
                assert reproduction[k]['gradient_norm'] == actual['gradient_norm']
                assert np.isfinite(arrays['losses_before_' + k]).all()
            # Both native backward discrepancies have been localized at step 1. All historical mismatches remain failed; these are recomputed diagnostic values.
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
            dump(dest.with_suffix('.json'), dict(passed=bool(exact_passed), collection_complete=True, exact_historical_gradient_gate_passed=bool(exact_passed), exact_checks=exact_checks, gradient_origin='native recomputation, not recovered historical tensors', step=step, work=str(work), cosine=cosine,
                sign_agreement=float(np.mean(np.sign(gg['pi']) == np.sign(gg['imagewam']))),
                gradient_sha256={k:ash(gg[k]) for k in KINDS}, arrays_sha256=sha(dest),
                input_npz_sha256=point['input_npz_sha256'], effects=effects,
                reproduction=reproduction, completed_at=time.time()))
            assert preserved() == plan['original_hashes']
            print('collected fixed diagnostic point', step, json.dumps(effects), flush=True)
        status('verifying', work=str(work), gpu_job_started=True)
        verify_collection(work)
        status('collection_finished_awaiting_independent_verification', work=str(work), gpu_job_started=True)
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
    ap.add_argument('command', choices=['cpu-checks', 'run', 'server'])
    ap.add_argument('--work', type=Path)
    ap.add_argument('--kind', choices=KINDS)
    ap.add_argument('--gpus', type=int, nargs=2)
    args = ap.parse_args()
    if args.command == 'cpu-checks':
        plan = cpu_prepare()
        ast_source = Path(__file__).read_text()
        import ast
        ast.parse(ast_source)
        tree = ast.parse(ast_source)
        calls = [n.func for n in ast.walk(tree) if isinstance(n, ast.Call)]
        assert any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'J' and n.attr == 'server' for n in calls)
        assert not any(isinstance(n, ast.Name) and n.id == 'verify' for n in calls)
        dump(DIAG / 'collection_cpu_verification.json', dict(passed=True, fixed_steps=STEPS,
            runner_sha256=sha(Path(__file__)), original_runner_sha256=sha(Path(D.__file__)),
            inputs_sha256=sha(DIAG / 'replay_inputs.json'), original_files_unchanged=preserved() == plan['original_hashes'],
            original_500_sha_replay=True, routes=read(DIAG / 'cpu_verification.json')['routing_checks'],
            exact_historical_gates_retained=True, exact_historical_tolerance=0, no_training_or_closed_loops=True))
        print(json.dumps(read(DIAG / 'collection_cpu_verification.json')))
    elif args.command == 'server':
        work = guard_work(args.work)
        plan = read(DIAG / 'replay_inputs.json')
        assert preserved() == plan['original_hashes'] and source_hashes() == plan['source_hashes']
        J.PORTS = PORTS
        J.server(work, args.kind)
    else:
        run(args.gpus)
