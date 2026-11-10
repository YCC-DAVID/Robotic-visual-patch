"""Natural-exit subprocesses and single-client policy services for paper runs.

Install only in the main-method orchestrator. Policy inference stays in the
original service entry point and upstream websocket handler; this module only
bounds the service lifetime to its one rollout client's session.
"""
import argparse
import ast
import asyncio
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


SOURCE = Path(__file__).resolve()
FINITE_READY = '[finite-service] listening'
DEFAULT_FIRST_CLIENT_TIMEOUT = 600.0


def source_sha256():
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


def _runtime_event(V, name, **fields):
    V.event(name, runtime_source=str(SOURCE), runtime_sha256=source_sha256(), **fields)


def _job(V):
    return getattr(V._LOCAL, 'job', None)


def _wait_naturally(proc, V, timeout=None):
    """A deadline is diagnostic: it never authorizes sending a process signal."""
    try:
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _runtime_event(V, 'natural_exit_deadline_exceeded', job=_job(V), pid=proc.pid,
                           deadline_s=timeout, action='continue_waiting_for_natural_exit')
            return proc.wait()
    except BaseException as error:
        _runtime_event(V, 'natural_exit_wait_interrupted', job=_job(V), pid=proc.pid,
                       returncode=proc.poll(), error=repr(error), action='process_left_untouched')
        raise


def natural_run(V, cmd, log, env, timeout):
    """V.run-compatible implementation; waits for the child to exit itself."""
    logpath = Path(V.DEST)/'logs'/log
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with logpath.open('w') as stream:
        proc = subprocess.Popen(cmd, cwd=V.P.ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
        V.process_event(proc, cmd, logpath, env)
        _runtime_event(V, 'natural_exit_process_started', job=_job(V), pid=proc.pid,
                       command=[str(x) for x in cmd], log=str(logpath), deadline_s=timeout)
        code = _wait_naturally(proc, V, timeout)
        _runtime_event(V, 'process_finished', job=_job(V), pid=proc.pid, returncode=code,
                       exit_policy='natural')
        if code:
            raise subprocess.CalledProcessError(code, cmd)


@contextmanager
def _service(V, root, gpu, port, model, env, service):
    root = Path(root)
    folder = root/'services'
    folder.mkdir(parents=True, exist_ok=True)
    logs = root/'logs'
    logs.mkdir(parents=True, exist_ok=True)
    metadata = folder/('policy.json' if service == 'robotwin' else f'policy_{port}.json')
    first_timeout = float(env.get('WAMATTACK_FIRST_CLIENT_TIMEOUT', DEFAULT_FIRST_CLIENT_TIMEOUT))
    if first_timeout <= 0:
        raise ValueError('first client timeout must be positive')
    cmd = [str(model), '-u', str(SOURCE), '--service', service, '--port', str(port),
           '--first-client-timeout', str(first_timeout)]
    if service == 'robotwin':
        cmd += ['--metadata', str(metadata)]
    logpath = logs/f'policy_server_{port}.log'
    with logpath.open('w') as stream:
        proc = subprocess.Popen(cmd, cwd=V.P.ROOT.parent if service == 'robotwin' else V.P.ROOT,
                                env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        V.process_event(proc, cmd, logpath, env)
        owned = dict(pid=proc.pid, port=port, gpu=gpu, job=_job(V), service=service,
                     started=V.now(), runtime_source=str(SOURCE), runtime_sha256=source_sha256(),
                     first_client_timeout_s=first_timeout, exit_policy='one_client_session_then_natural_exit')
        V.write_json(folder/f'owned_service_{port}.json', owned)
        _runtime_event(V, 'finite_service_started', job=_job(V), **{k: v for k, v in owned.items()
                       if k not in ['job', 'runtime_source', 'runtime_sha256']}, log=str(logpath), command=cmd)
        failure = None
        try:
            started = time.monotonic()
            while FINITE_READY not in logpath.read_text():
                if proc.poll() is not None:
                    raise RuntimeError('finite policy service exited before listening: '+str(logpath))
                if time.monotonic()-started >= 900:
                    raise RuntimeError('finite policy service readiness deadline exceeded: '+str(logpath))
                time.sleep(.25)
            yield proc
        except BaseException as error:
            failure = error
            raise
        finally:
            code = _wait_naturally(proc, V)
            _runtime_event(V, 'process_finished', job=_job(V), pid=proc.pid, returncode=code,
                           service=service, exit_policy='natural')
            if code and failure is None:
                raise subprocess.CalledProcessError(code, cmd)


def robotwin_server(root, gpu, port, model, model_env, V):
    """Context helper: model is its Python executable, model_env is dict/callable."""
    env = model_env(gpu) if callable(model_env) else dict(model_env)
    return _service(V, root, gpu, port, model, env, 'robotwin')


def install(V):
    """Override V.run/server in this process; original science sources stay intact."""
    if getattr(V, '_natural_job_runtime_installed', False):
        return dict(runtime_source=str(SOURCE), runtime_sha256=source_sha256())

    def run(cmd, log, env, timeout):
        return natural_run(V, cmd, log, env, timeout)

    def server(gpu, port):
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu['uuid'], 'OMP_NUM_THREADS': '4'}
        return _service(V, V.DEST, gpu, port, V.P.MODEL, env, 'libero')

    original_check = V.check_allocation

    def check_allocation(job, gpu, min_free_mib):
        deadline = time.monotonic()+60
        while True:
            try:
                return original_check(job, gpu, min_free_mib)
            except V.ResourceBlocked as error:
                prefix = 'selected GPU is occupied, faulty or insufficient: '
                message = str(error)
                if not message.startswith(prefix):
                    raise
                try:
                    snapshot = ast.literal_eval(message[len(prefix):])
                except (ValueError, SyntaxError):
                    raise error
                # Missing UUID/index, UNKNOWN or a faulty device isn't residual utilization.
                if (not isinstance(snapshot, dict) or snapshot.get('uuid') != gpu['uuid']
                        or snapshot.get('index') != int(gpu['index']) or snapshot.get('faulty')
                        or 'idle' not in snapshot or 'eligible' not in snapshot):
                    raise
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise
                delay = min(10, remaining)
                _runtime_event(V, 'allocation_strict_recheck', job=job, gpu=gpu,
                               delay_s=delay, remaining_s=remaining, previous_error=message,
                               policy='repeat_original_full_cluster_EGL_selected_checks_without_relaxation')
                time.sleep(delay)
                if time.monotonic() >= deadline:
                    raise

    V.run = run
    V.server = server
    V.check_allocation = check_allocation
    V._natural_job_runtime_installed = True
    provenance = dict(runtime_source=str(SOURCE), runtime_sha256=source_sha256())
    _runtime_event(V, 'natural_exit_runtime_installed')
    return provenance


def patch_finite_websocket_service(first_client_timeout=DEFAULT_FIRST_CLIENT_TIMEOUT):
    """Patch lifetime only. Every request still uses upstream self._handler."""
    from openpi.serving import websocket_policy_server as original
    if first_client_timeout <= 0:
        raise ValueError('first client timeout must be positive')

    async def finite_run(self):
        connected = asyncio.Event()
        finished = asyncio.Event()
        accepted = False
        errors = []

        async def one_session(websocket):
            nonlocal accepted
            if accepted:
                await websocket.close(code=1013, reason='This finite service accepts one rollout session')
                return
            accepted = True
            connected.set()
            try:
                await self._handler(websocket)
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

        async with original._server.serve(one_session, self._host, self._port,
                compression=None, max_size=None, process_request=original._health_check):
            print('[serve] ready finite_session=True port={}'.format(self._port), flush=True)
            print(FINITE_READY+' port={} runtime_sha256={}'.format(self._port, source_sha256()), flush=True)
            try:
                await asyncio.wait_for(connected.wait(), timeout=first_client_timeout)
            except asyncio.TimeoutError:
                print('[finite-service] first client deadline exceeded; returning naturally', flush=True)
                raise
            await finished.wait()
        if errors:
            raise errors[0]
        print('[finite-service] session closed; returning naturally', flush=True)

    original.WebsocketPolicyServer.run = finite_run
    return original


def _configure_paths():
    project = SOURCE.parent.parent
    for path in [project/'third_party/transformers_patched', project/'third_party/openpi/src',
                 project/'third_party/openpi/packages/openpi-client/src']:
        sys.path.insert(0, str(path))
    return project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service', choices=['libero', 'robotwin', 'dummy'], required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--metadata')
    parser.add_argument('--first-client-timeout', type=float, default=DEFAULT_FIRST_CLIENT_TIMEOUT)
    args = parser.parse_args()
    project = _configure_paths()
    original = patch_finite_websocket_service(args.first_client_timeout)
    if args.service == 'dummy':
        import numpy as np
        class DummyPolicy:
            def infer(self, element):
                return {'actions': np.asarray(element['value'])+1}
        original.WebsocketPolicyServer(DummyPolicy(), host='127.0.0.1', port=args.port,
                                        metadata={'finite_dummy': True}).serve_forever()
        return
    target = project/('pi05probe/render_proxy_policy_server.py' if args.service == 'libero'
                      else 'robotwinprobe/policy_server.py')
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), '--port', str(args.port)]
    if args.service == 'robotwin':
        if not args.metadata:
            raise ValueError('--metadata is required for RoboTwin service')
        sys.argv += ['--metadata', args.metadata]
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
