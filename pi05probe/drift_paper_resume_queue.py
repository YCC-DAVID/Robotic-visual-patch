"""CPU queue to resume the frozen pi0.5 experiment after a natural failure.

Keep every render assertion. A repeated runtime failure is recorded, not retried
or converted into an attack success. No process is ever sent a signal.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from drift_paper_common import ROOT, dump, now, read, sha

PYTHON = '/shared/user65/miniconda3/envs/openpi-server/bin/python'


def supervise(args):
    root, attempt = Path(args.root), Path(args.attempt)
    gpu = json.loads(args.gpu)
    assert socket.gethostname().split('.')[0] == gpu['host']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1',
               WAMATTACK_PROJECT_ROOT=str(ROOT), WAMATTACK_ENVROOT='/shared/user65/miniconda3/envs')
    cmd = [PYTHON, '-u', str(ROOT / 'pi05probe/drift_paper_pipeline.py'),
           '--root', str(root), '--checkpoint', read(root / 'protocol.json')['checkpoint'],
           '--gpu', json.dumps(gpu)]
    with (attempt / 'controller.log').open('w') as stream:
        child = subprocess.Popen(cmd, cwd=ROOT / 'pi05probe', env=env,
                                 stdin=subprocess.DEVNULL, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        launch = dict(controller_pid=child.pid, supervisor_pid=os.getpid(), host=gpu['host'],
                      gpu=gpu, command=cmd, started=now())
        dump(attempt / 'launch.json', launch)
        final = [PYTHON, '-u', str(ROOT / 'pi05probe/drift_paper_finalize.py'),
                 '--root', str(root), '--controller-pid', str(child.pid)]
        with (attempt / 'finalization.log').open('w') as final_log:
            watcher = subprocess.Popen(final, cwd=ROOT / 'pi05probe', env=env,
                                       stdin=subprocess.DEVNULL, stdout=final_log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        launch['finalizer_pid'] = watcher.pid
        dump(attempt / 'launch.json', launch)
        code = child.wait()
    dump(attempt / 'exit.json', dict(controller_pid=child.pid, returncode=code,
                                   finished=now(), exit_policy='natural'))


REMOTE = r'''
import json,os,subprocess
from pathlib import Path
attempt=Path(payload['attempt'])
cmd=['/shared/user65/miniconda3/envs/openpi-server/bin/python','-u',
     payload['script'],'--supervise','--root',payload['root'],
     '--attempt',str(attempt),'--gpu',json.dumps(payload['gpu'])]
with (attempt/'supervisor.log').open('w') as stream:
    proc=subprocess.Popen(cmd,cwd=str(Path(payload['script']).parent),
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1'),
        stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(supervisor_pid=proc.pid,host=payload['gpu']['host'])))
'''


def queue(args):
    root = Path(args.root).resolve()
    recovery = root / 'recovery'
    recovery.mkdir(exist_ok=True)
    with (recovery / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read(root / 'protocol.json')
        for name, digest in protocol['source_sha256'].items():
            assert sha(ROOT / 'pi05probe' / name) == digest
        assert read(root / 'progress.json')['stage'] == 'failed', 'only resume a failed controller'
        with (root / 'pipeline.lock').open('a') as pipeline_lock:
            fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive = recovery / ('failed_' + str(time.time_ns()))
        archive.mkdir()
        shutil.copytree(root / 'logs', archive / 'logs')
        for name in ['controller.log', 'progress.json', 'finalization.json', 'validation.json', 'summary.json']:
            if (root / name).exists():
                shutil.copy2(root / name, archive / name)
        dump(archive / 'reason.json', dict(reason='diagnostic altered input assertion; retain assertion during one resume',
             protocol_sha256=sha(root / 'protocol.json'), queue_sha256=sha(__file__), no_science_changes=True))
        resources = recovery / 'resources'
        resources.mkdir(exist_ok=True)
        state_path = recovery / 'queue_progress.json'
        while True:
            result = subprocess.run([sys.executable, str(ROOT / 'skills/gpu-preflight/scripts/check_gpus.py'),
                                     '--json', '--min-free-mib', '24000'],
                                    capture_output=True, text=True, timeout=100)
            if result.returncode not in (0, 2):
                dump(state_path, dict(stage='waiting_for_preflight', pid=os.getpid(), updated=now(),
                                     error=result.stderr[-1000:]))
                time.sleep(30)
                continue
            report = json.loads(result.stdout)
            dump(resources / 'latest_fleet.json', report)
            candidates = [dict(g, host=n['host']) for n in report['nodes'] if n.get('status') == 'ok'
                for g in n.get('gpus', []) if g.get('idle') and g.get('eligible') and not g.get('faulty')
                and g['name'] == 'NVIDIA L40S' and not (n['host'] == 'nnmc63' and g['index'] == 0)]
            candidates.sort(key=lambda g: (g['host'] != 'nnmc62' or g['index'] != 3, g['host'], g['index']))
            if not candidates:
                dump(state_path, dict(stage='waiting_for_idle_gpu', pid=os.getpid(), updated=now(),
                                     archive=str(archive), no_stop_operations=True))
                time.sleep(30)
                continue
            launched = None
            for card in candidates:
                audit = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', card['host'],
                    'env', 'CUDA_VISIBLE_DEVICES=' + card['uuid'], PYTHON,
                    str(ROOT / 'pi05probe/egl_device_audit.py'), '--uuid', card['uuid']],
                    capture_output=True, text=True, timeout=40)
                if audit.returncode:
                    continue
                gpu = {key: card[key] for key in ('host', 'index', 'uuid', 'name')}
                gpu['egl_index'] = json.loads(audit.stdout)['egl_index']
                attempt = recovery / ('attempt_' + str(time.time_ns()))
                attempt.mkdir()
                payload = dict(root=str(root), attempt=str(attempt), script=str(Path(__file__).resolve()), gpu=gpu)
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', card['host'],
                    'python3', '-'], input='payload = ' + repr(payload) + '\n' + REMOTE,
                    capture_output=True, text=True, timeout=40)
                if result.returncode:
                    dump(attempt / 'launch_error.json', dict(error=result.stderr[-2000:]))
                    continue
                launched = dict(attempt=str(attempt), gpu=gpu, **json.loads(result.stdout))
                dump(state_path, dict(stage='controller_launched', pid=os.getpid(), updated=now(), **launched))
                break
            if launched is None:
                time.sleep(30)
                continue
            while not (attempt / 'exit.json').exists():
                time.sleep(30)
            exit_info = read(attempt / 'exit.json')
            if exit_info['returncode']:
                progress = read(root / 'progress.json')
                if 'ResourceBlocked' in progress.get('error', ''):
                    continue
                dump(state_path, dict(stage='failed', pid=os.getpid(), updated=now(), **launched,
                                     exit=exit_info, note='Runtime assertion retained. No automatic retry or success credit.'))
            else:
                dump(state_path, dict(stage='rollouts_complete_pending_finalizer', pid=os.getpid(),
                                     updated=now(), **launched, exit=exit_info))
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--attempt')
    parser.add_argument('--gpu')
    opts = parser.parse_args()
    if opts.supervise:
        supervise(opts)
    else:
        queue(opts)
