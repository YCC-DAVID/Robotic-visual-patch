"""Single-GPU serial collection of only the six authorized #57 fixed points."""
from pi_imagewam_gradient_replay_collect import *
import pi_imagewam_gradient_replay_collect as C


def load_cache(point, kind):
    f = DIAG / 'serial_staging' / f"step_{point['step']:04d}_{kind}.npz"
    meta = read(f.with_suffix('.json'))
    assert meta['input_sha256'] == point['input_npz_sha256'] and meta['arrays_sha256'] == sha(f)
    with np.load(f) as z:
        arrays = {k: z[k].copy() for k in z.files}
    assert ash(arrays['q']) == point['before_sha256']
    return arrays, meta


def validate_native(point, kind, arrays, provenance):
    expected = point['history']['provenance'][kind]
    assert set(expected) == set(provenance)
    assert {k:v for k,v in expected.items() if k != 'gradient_norm'} == {k:v for k,v in provenance.items() if k != 'gradient_norm'}
    assert float(np.mean(arrays['losses'])) == point['history']['raw'][kind], ('new forward loss mismatch', point['step'], kind)
    g = arrays['gradient']
    assert g.shape == arrays['q'].shape and np.isfinite(g).all() and np.linalg.norm(g) > 0
    assert float(np.linalg.norm(g)) == provenance['gradient_norm']
    if provenance != expected:
        loc = read(DIAG / ('pi_operator_localization_analysis.json' if kind == 'pi' else 'operator_localization_analysis.json'))
        if kind == 'pi':
            assert loc['located_equal_input_unequal_output_operator'] and loc['native_functions_unchanged']
            for sample in loc['samples']:
                assert sample['inputs_equal'] and not sample['outputs_equal']
                for path, digest in sample['trace_sha256'].items(): assert sha(Path(path)) == digest
        else:
            assert loc  # Existing native ImageWAM backward localization; exact gate remains failed.


def run_serial(gpu):
    lock = (DIAG / 'diagnostic.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = cpu_prepare()
    attempt = 'serial_collection_' + str(time.time_ns())
    work = prepare_work(plan, attempt)
    execution = dict(host=socket.gethostname().split('.')[0], pipeline_pid=os.getpid(), command=sys.argv,
        work=str(work), diagnostic_lock=str(DIAG / 'diagnostic.lock'), backends={}, started_at=time.time(),
        runner_sha256=sha(Path(__file__)), single_gpu_strict_serial=True, gpu=gpu)
    if (DIAG / 'execution.json').exists(): shutil.copy2(DIAG / 'execution.json', work / 'previous_execution.json')
    def save():
        execution['updated_at'] = time.time()
        dump(DIAG / 'execution.json', execution); dump(work / 'execution.json', execution)
    save()
    proc = None
    current = None
    def close_backend(kind):
        nonlocal proc
        if proc is not None:
            if proc.poll() is None:
                try: rpc(kind, 'shutdown')
                except Exception: proc.terminate()
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            execution['backends'][current].update(exit_code=proc.returncode, exited_at=time.time())
            save(); proc = None
    try:
        for phase, kind in [('pi_base', 'pi'), ('imagewam_base_and_cross', 'imagewam'), ('pi_imagewam_candidate', 'pi')]:
            pending = []
            for point in plan['points']:
                dest = DIAG / 'collected' / f"step_{point['step']:04d}.json"
                if dest.exists(): continue
                suffix = 'pi' if phase == 'pi_base' else 'imagewam' if kind == 'imagewam' else 'cross_pi'
                cache = DIAG / 'serial_staging' / f"step_{point['step']:04d}_{suffix}.json"
                if not cache.exists(): pending.append(point)
            if not pending: continue
            phase_work = prepare_work(plan, attempt + '_' + phase)
            J.S.R.check_assets(phase_work)
            current = phase
            status('serial_load_' + phase, work=str(work), gpu_job_started=False)
            assert proc is None
            card = preflight(phase_work, gpu, 'load_' + phase)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=card['uuid'], PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='4', WAMATTACK_PROJECT_ROOT=str(ROOT), WAMATTACK_GPU_PREFLIGHT='/shared/user65/.codex/skills/gpu-preflight/scripts/check_gpus.py',
                WAMATTACK_LIBERO_CONFIG_PATH=str(phase_work / 'libero_config'), HF_HUB_OFFLINE='1', JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
            command = [PY[kind], '-u', str(ROOT / 'robotwinprobe/pi_imagewam_gradient_replay_collect.py'), 'server', '--work', str(phase_work), '--kind', kind]
            with (phase_work / 'logs' / (kind + '_backend.log')).open('a') as log:
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            execution['backends'][phase] = dict(card, pid=proc.pid, command=command, kind=kind, work=str(phase_work)); save()
            ready = phase_work / ('models/pi/ready.json' if kind == 'pi' else 'ready.json')
            deadline = time.monotonic() + 600
            while not ready.exists():
                assert proc.poll() is None and time.monotonic() < deadline, ('backend load failed', phase, proc.poll())
                time.sleep(2)
            assert read(ready)['pid'] == proc.pid
            assert rpc(kind, 'audit')['passed']
            if kind == 'imagewam':
                actual = read(phase_work / 'audit/native_and_gradient.json')['nominal_gray_loss']
                expected = read(ORIGINAL / 'audit/native_and_gradient.json')['nominal_gray_loss']
                assert abs(actual - expected) / expected < 1e-5
                dump(phase_work / 'audit/imagewam_gray_match.json', dict(passed=True, actual=actual, reference=expected))
            for point in pending:
                step = point['step']
                status('serial_' + phase, step=step, work=str(work), gpu_job_started=True)
                with np.load(point['input_npz']) as z: q = z['q'].copy()
                payload = dict(texture=q, indices=point['batch']['indices'], seeds=point['batch']['seeds'], draws=point['draws'], iteration=step, source_model=kind)
                arrays = dict(q=q)
                if phase != 'pi_imagewam_candidate':
                    out = rpc(kind, 'batch', **payload, gradient=True)
                    arrays.update(gradient=out['gradient'], losses=np.asarray(out['losses']))
                    evidence = phase_work / 'evidence' / f'step_{step:04d}'
                    evidence.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(evidence / 'raw_gradient.npz', **arrays)
                    dump(evidence / 'reproduction.json', dict(provenance=out['gradient_provenance'], raw_loss=float(np.mean(arrays['losses'])), expected=point['history']))
                    validate_native(point, kind, arrays, out['gradient_provenance'])
                    arrays['delta'] = np.clip(q + .01 * np.sign(arrays['gradient']), 0, 1).astype(np.float32) - q
                    candidates = {kind: q + arrays['delta']}
                    if kind == 'imagewam':
                        pi_arrays, pi_meta = load_cache(point, 'pi')
                        validate_native(point, 'pi', pi_arrays, pi_meta['provenance'])
                        candidates['pi'] = q + pi_arrays['delta']
                    for candidate, texture in candidates.items():
                        result = rpc(kind, 'batch', **dict(payload, texture=texture), gradient=False)
                        arrays['losses_after_' + candidate] = np.asarray(result['losses'])
                    provenance = out['gradient_provenance']; suffix = kind
                else:
                    image_arrays, image_meta = load_cache(point, 'imagewam')
                    validate_native(point, 'imagewam', image_arrays, image_meta['provenance'])
                    before = rpc('pi', 'batch', **payload, gradient=False)
                    pi_arrays, pi_meta = load_cache(point, 'pi')
                    assert np.array_equal(np.asarray(before['losses']), pi_arrays['losses']), ('reloaded pi forward mismatch', step)
                    after = rpc('pi', 'batch', **dict(payload, texture=q + image_arrays['delta']), gradient=False)
                    arrays['losses_after_imagewam'] = np.asarray(after['losses'])
                    provenance = None; suffix = 'cross_pi'
                dest = DIAG / 'serial_staging' / f'step_{step:04d}_{suffix}.npz'
                dest.parent.mkdir(exist_ok=True)
                assert not dest.exists(), ('partial evidence retained', str(dest))
                assert all(np.isfinite(a).all() for a in arrays.values())
                np.savez_compressed(dest, **arrays)
                dump(dest.with_suffix('.json'), dict(input_sha256=point['input_npz_sha256'], arrays_sha256=sha(dest), work=str(phase_work), provenance=provenance, phase=phase))
            close_backend(kind)
        for point in plan['points']:
            step = point['step']; dest = DIAG / 'collected' / f'step_{step:04d}.npz'
            if dest.with_suffix('.json').exists(): continue
            pa, pm = load_cache(point, 'pi'); ia, im = load_cache(point, 'imagewam'); ca, cm = load_cache(point, 'cross_pi')
            for kind, a, meta in [('pi', pa, pm), ('imagewam', ia, im)]: validate_native(point, kind, a, meta['provenance'])
            with np.load(point['input_npz']) as z: original_sign = z['original_update_sign'].copy()
            arrays = dict(q=pa['q'], gradient_pi=pa['gradient'], gradient_imagewam=ia['gradient'],
                losses_before_pi=pa['losses'], losses_before_imagewam=ia['losses'], delta_pi=pa['delta'], delta_imagewam=ia['delta'],
                losses_after_pi_on_pi=pa['losses_after_pi'], losses_after_imagewam_on_pi=ca['losses_after_imagewam'],
                losses_after_pi_on_imagewam=ia['losses_after_pi'], losses_after_imagewam_on_imagewam=ia['losses_after_imagewam'])
            gg = {k: arrays['gradient_' + k] for k in KINDS}
            cosine = float(np.sum(gg['pi'] * gg['imagewam']) / (np.linalg.norm(gg['pi']) * np.linalg.norm(gg['imagewam'])))
            reproduction = {k: dict(raw_loss=float(np.mean(a['losses'])), gradient_norm=float(np.linalg.norm(a['gradient'])), provenance=m['provenance']) for k,a,m in [('pi',pa,pm),('imagewam',ia,im)]}
            checks = dict(provenance={k:reproduction[k]['provenance']==point['history']['provenance'][k] for k in KINDS}, raw_loss={k:reproduction[k]['raw_loss']==point['history']['raw'][k] for k in KINDS}, cosine=cosine==point['history']['cosine'], selected_sign=bool(np.array_equal(np.sign(gg[point['history']['chosen']]).astype(np.int8),original_sign)))
            exact = all(checks['provenance'].values()) and all(checks['raw_loss'].values()) and checks['cosine'] and checks['selected_sign']
            effects = {c:{k:dict(first_order_projection=float(np.sum(gg[k].astype(np.float64)*arrays['delta_'+c].astype(np.float64))),finite_step_loss_difference=float(np.mean(arrays[f'losses_after_{c}_on_{k}'])-np.mean(arrays['losses_before_'+k])),loss_before=float(np.mean(arrays['losses_before_'+k])),loss_after=float(np.mean(arrays[f'losses_after_{c}_on_{k}']))) for k in KINDS} for c in KINDS}
            point_work = work / f'point_{step:04d}'
            for src, name in [(Path(pm['work']), 'models/pi/native_audit.json'),(Path(im['work']), 'audit/native_and_gradient.json'),(Path(im['work']), 'audit/imagewam_gray_match.json')]:
                target = point_work/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src/name,target)
            dest.parent.mkdir(exist_ok=True); assert not dest.exists(); np.savez_compressed(dest,**arrays)
            dump(dest.with_suffix('.json'),dict(passed=bool(exact),collection_complete=True,exact_historical_gradient_gate_passed=bool(exact),exact_checks=checks,gradient_origin='native recomputation, not recovered historical tensors',step=step,work=str(point_work),cosine=cosine,sign_agreement=float(np.mean(np.sign(gg['pi'])==np.sign(gg['imagewam']))),gradient_sha256={k:ash(gg[k]) for k in KINDS},arrays_sha256=sha(dest),input_npz_sha256=point['input_npz_sha256'],effects=effects,reproduction=reproduction,serial_phase_work={k:m['work'] for k,m in [('pi',pm),('imagewam',im),('pi_cross',cm)]},completed_at=time.time()))
            status('serial_point_collected',step=step,work=str(work),gpu_job_started=False)
        C.verify_collection(work)
        status('collection_finished_awaiting_independent_verification',work=str(work),gpu_job_started=False)
    except Exception as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc(),work=str(work),time=time.time())
        dump(work/'error.json',error);dump(DIAG/'errors'/(attempt+'.json'),error)
        status('runtime_error',work=str(work),gpu_job_started=proc is not None,error=repr(exc));raise
    finally:
        if proc is not None: close_backend(execution['backends'][current]['kind'])
        execution.update(backends_exited=all('exit_code' in b for b in execution['backends'].values()),finished_at=time.time());save()
        assert preserved()==plan['original_hashes'];fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['cpu-checks','run']);ap.add_argument('--gpu',type=int);args=ap.parse_args()
    if args.command=='cpu-checks':
        plan=cpu_prepare();compile(Path(__file__).read_text(),str(Path(__file__)),'exec')
        dump(DIAG/'serial_cpu_verification.json',dict(passed=True,runner_sha256=sha(Path(__file__)),collector_sha256=sha(Path(C.__file__)),inputs_sha256=sha(DIAG/'replay_inputs.json'),fixed_steps=STEPS,original_500_sha_replay=True,original_files_unchanged=preserved()==plan['original_hashes'],routes=read(DIAG/'cpu_verification.json')['routing_checks'],historical_tolerance=0,phases=['pi_base','imagewam_base_and_cross','pi_imagewam_candidate'],one_backend_at_a_time=True,no_training_or_closed_loops=True))
        print(json.dumps(read(DIAG/'serial_cpu_verification.json')))
    else:run_serial(args.gpu)
