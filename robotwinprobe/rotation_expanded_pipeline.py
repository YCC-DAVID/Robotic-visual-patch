"""Resumable independent-cohort evaluation, owning one service and four simulators."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from digital_transform_common import ROOT, PROBE, TRANSFER, dump, sha
from rotation_expanded_plan import DEFAULT, GROUPS, prepare, records
from rotation_expanded_report import build_report, audit

PY = '/home/user1/miniconda3/envs/openpi-server/bin/python'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(DEFAULT))
    ap.add_argument('--model-gpu', default='2')
    ap.add_argument('--sim-gpus', default='4,5,6,7')
    ap.add_argument('--port', type=int, default=8295)
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    protocol = prepare(root)
    if args.prepare_only:
        print(json.dumps(build_report(root), ensure_ascii=False)); return
    lock = (root / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sim_gpus = args.sim_gpus.split(',')
    assert len(set(sim_gpus + [args.model_gpu])) == len(sim_gpus) + 1
    snapshot = root / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ['rotation_expanded_plan.py', 'rotation_expanded_eval.py', 'rotation_expanded_report.py',
                 'rotation_expanded_pipeline.py', 'digital_transform_common.py', 'digital_transform_pipeline.py',
                 'render_geometry.py', 'render_proxy.py', 'render_rollout.py', 'scene_smoke.py',
                 'policy_server.py', 'bootstrap.py', 'run_sim.sh']:
        src, target = PROBE / name, snapshot / name
        if target.exists():
            assert sha(src) == sha(target), f'Source changed on resume: {name}'
        else:
            target.write_bytes(src.read_bytes())
    dump(snapshot / 'manifest.json', {p.name: sha(p) for p in snapshot.iterdir() if p.name != 'manifest.json'})
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update(CUDA_VISIBLE_DEVICES=args.model_gpu, PYTHONNOUSERSITE='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', MPLCONFIGDIR='/tmp/rotation_expanded_mpl')
    children, server = [], None

    def status(stage, **extra):
        dump(root / 'status.json', dict(stage=stage, pid=os.getpid(),
            updated_utc=datetime.now(timezone.utc).isoformat(), model_gpu=args.model_gpu,
            sim_gpus=sim_gpus, port=args.port, **extra))

    def spawn(command, logname, child_env=None):
        with (root / logname).open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env if child_env is None else child_env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        children.append(process)
        return process

    def stage(name, seeds, groups):
        expected = {(c, m): records(root, c, m) for c, m in groups}
        jobs = [dict(seed=s, condition=c, method=m) for s in seeds for c, m in groups if s not in expected[c, m]]
        if not jobs:
            return
        workers = []
        job_dir = root / 'jobs'
        job_dir.mkdir(exist_ok=True)
        attempt = time.time_ns()
        for i, gpu in enumerate(sim_gpus):
            shard = jobs[i::len(sim_gpus)]
            if not shard:
                continue
            path = job_dir / f'{name}_{attempt}_worker{i}.json'
            dump(path, shard)
            workers.append(spawn(['bash', str(PROBE / 'run_sim.sh'), str(PROBE / 'rotation_expanded_eval.py'),
                '--root', str(root), '--jobs', str(path), '--port', str(args.port)],
                f'{name}_worker{i}.log', dict(env, CUDA_VISIBLE_DEVICES=gpu)))
        status(name, server_pid=server.pid, worker_pids=[p.pid for p in workers],
            scheduled_this_attempt=len(jobs), selected_initial_states=len(seeds))
        last_report = 0
        while any(p.poll() is None for p in workers):
            if server.poll() is not None:
                raise RuntimeError('Owned policy server exited')
            if any(p.poll() not in [None, 0] for p in workers):
                raise RuntimeError(f'{name}: simulator worker failed; see worker logs')
            if time.monotonic() - last_report > 20:
                result = build_report(root)
                status(name, server_pid=server.pid, worker_pids=[p.pid for p in workers if p.poll() is None],
                    scheduled_this_attempt=len(jobs), selected_initial_states=len(seeds),
                    completed_by_group={r['condition'] + '/' + r['method']: r['completed'] for r in result['rows']})
                last_report = time.monotonic()
            time.sleep(2)
        assert all(p.returncode == 0 for p in workers), name
        for cname, method in groups:
            assert set(seeds) <= records(root, cname, method).keys(), (name, cname, method)
        build_report(root)

    def interrupted(sig, frame):
        raise KeyboardInterrupt(f'signal {sig}')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        status('checking_compositor'); build_report(root)
        command = ('import sys; from pathlib import Path; '
                   f'sys.path.insert(0, {str(PROBE)!r}); '
                   'from digital_transform_pipeline import cpu_audit; '
                   f'cpu_audit(Path({str(root)!r}))')
        cpu = spawn([PY, '-c', command], 'cpu_audit.log')
        if cpu.wait():
            raise RuntimeError('Geometry / nominal RGB audit failed')
        status('waiting_for_available_gpus')
        while True:
            available = True
            for gpu in [args.model_gpu] + sim_gpus:
                raw = subprocess.check_output(['nvidia-smi', '-i', gpu,
                    '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
                used, util = map(int, raw.strip().split(','))
                if used > 500 or util > 10:
                    available = False
            if available:
                break
            time.sleep(20)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', args.port))
        meta = root / f'server_{os.getpid()}.json'
        server = spawn([PY, '-u', str(PROBE / 'policy_server.py'), '--port', str(args.port),
            '--metadata', str(meta)], 'server.log')
        dump(root / 'owned_server.json', dict(pid=server.pid, stopped=False))
        status('starting_owned_policy_server', server_pid=server.pid); build_report(root)
        deadline = time.monotonic() + 360
        while True:
            if server.poll() is not None:
                raise RuntimeError('Policy server exited before ready')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/healthz', timeout=1) as response:
                    ready = response.status == 200 and meta.exists()
                if ready:
                    break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Policy server startup')
            time.sleep(2)
        assert json.loads(meta.read_text()) == json.loads((TRANSFER / 'worker_full/server.json').read_text())
        stage('clean_initial_100', protocol['initial_seeds'], [('nominal', 'clean')])
        initial_clean = records(root, 'nominal', 'clean')
        initial_success = sum(initial_clean[s]['success'] for s in protocol['initial_seeds'])
        extend = initial_success < protocol['minimum_clean_successes']
        seeds = protocol['initial_seeds'] + (protocol['extension_seeds'] if extend else [])
        if extend:
            stage('clean_extension_100', protocol['extension_seeds'], [('nominal', 'clean')])
        cohort = dict(seeds=seeds, extended=extend, first_100_clean_successes=initial_success,
            rule=protocol['cohort_rule'], frozen_before_attack_evaluation=True)
        path = root / 'cohort.json'
        if path.exists():
            assert json.loads(path.read_text()) == cohort
        else:
            dump(path, cohort)
        stage('paired_rotation_evaluation', seeds, GROUPS[1:])
        status('auditing_paired_episodes'); audit(root)
        assert build_report(root)['complete']
        status('complete', complete=True, evaluated_initial_states=len(seeds), evaluated_episodes=len(seeds) * len(GROUPS))
        build_report(root)
    except BaseException as exc:
        status('failed', complete=False, error=repr(exc)); build_report(root)
        raise
    finally:
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        if server is not None:
            dump(root / 'owned_server.json', dict(pid=server.pid, stopped=True, returncode=server.returncode))


if __name__ == '__main__':
    main()
