"""CPU-only global queue for priority native-render checks; never stop a job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from drift_pi0_common import ROOT, dump, now, read

REMOTE_ROOT = Path('/shared/user65/workspace/chence/WAMattack')
MODEL_PYTHON = '/shared/user65/miniconda3/envs/openpi-server/bin/python'


def command(job, gpu=None):
    cmd = [MODEL_PYTHON, '-u', str(REMOTE_ROOT / 'pi05probe/drift_priority_render.py'),
           '--model', job['model'], '--source-root', job['source_root'], '--root', job['root'],
           '--checkpoint', job['checkpoint'], '--suite', 'libero_spatial',
           '--tasks', '0,1,2', '--port', str(job['port'])]
    if gpu is not None:
        cmd += ['--gpu', json.dumps(gpu)]
    return cmd


def supervise(payload):
    """Wait for one child normally, recording its exit for the CPU queue."""
    job, gpu = payload['job'], payload['gpu']
    assert socket.gethostname().split('.')[0] == gpu['host']
    dest = Path(job['root'])
    stamp = payload['stamp']
    cmd = command(job, gpu)
    launch_path = dest / 'launches' / (stamp + '.json')
    result_path = dest / 'launches' / (stamp + '.exit.json')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONNOUSERSITE='1',
               WAMATTACK_PROJECT_ROOT=str(REMOTE_ROOT), WAMATTACK_ENVROOT='/shared/user65/miniconda3/envs')
    with (dest / 'launches' / (stamp + '.log')).open('w') as stream:
        p = subprocess.Popen(cmd, cwd=REMOTE_ROOT / 'pi05probe', env=env,
                             stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                             start_new_session=True)
        dump(launch_path, dict(pid=p.pid, supervisor_pid=os.getpid(), gpu=gpu,
                              command=cmd, started=now(), stage='controller_started'))
        code = p.wait()
    dump(result_path, dict(pid=p.pid, returncode=code, finished=now(), exit_policy='natural'))


REMOTE_LAUNCH = r'''
import json,os,subprocess
from pathlib import Path
root=Path(payload['job']['root']);(root/'launches').mkdir(exist_ok=True)
cmd=['/shared/user65/miniconda3/envs/openpi-server/bin/python','-u',
     '/shared/user65/workspace/chence/WAMattack/pi05probe/drift_priority_render_queue.py',
     '--supervise-json',json.dumps(payload)]
with (root/'launches'/(payload['stamp']+'.supervisor.log')).open('w') as stream:
    p=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1'),
        cwd='/shared/user65/workspace/chence/WAMattack/pi05probe',stdin=subprocess.DEVNULL,
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(supervisor_pid=p.pid,host=payload['gpu']['host'],started=True)))
'''


def jobs(root):
    return [dict(model=model, root=str(root / model),
                 source_root=str(REMOTE_ROOT / 'pi05probe/out' / source),
                 checkpoint='/shared/user65/.cache/openpi/openpi-assets/checkpoints/' + config,
                 port=port, hardware=hardware)
            for model, source, config, port, hardware in [
                ('pi0', 'drift_pi0_render_20261109/run', 'pi0_libero', 18781, 'NVIDIA H100 NVL'),
                ('pi05', 'drift_paper_render_20261109/run_v2', 'pi05_libero', 18771, 'NVIDIA L40S')]]


def source_render_started(job):
    source = Path(job['source_root'])
    if any((source / 'eval').glob('*/task*/*_render/init*.json')):
        return True
    value = read(source / 'progress.json')
    return any('eval_render' in name for name in value.get('active_jobs', {}))


def prepare(job):
    result = subprocess.run(command(job) + ['--prepare-only'], capture_output=True,
                            text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), timeout=180)
    if result.returncode:
        raise RuntimeError('CPU preparation failed: ' + result.stdout[-1500:] + result.stderr[-1500:])


def main(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'resources').mkdir(exist_ok=True)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        work = jobs(root)
        state_path = root / 'queue_progress.json'
        state = read(state_path) if state_path.exists() else dict(jobs={})
        state['pid'] = os.getpid()
        for job in work:
            prepare(job)
            state['jobs'].setdefault(job['model'], dict(stage='waiting_for_idle_gpu'))
        while True:
            pending = []
            for job in work:
                status = state['jobs'][job['model']]
                dest = Path(job['root'])
                summary_path = dest / 'priority_summary.json'
                summary = read(summary_path) if summary_path.exists() else {}
                if status['stage'] == 'complete':
                    continue
                if (summary.get('complete') and summary.get('validation') != 'not run'
                        and (not status.get('stamp') or (dest / 'complete.json').exists())):
                    status.update(stage='complete', summary=str(summary_path))
                    continue
                if status.get('stamp'):
                    exit_path = dest / 'launches' / (status['stamp'] + '.exit.json')
                    if not exit_path.exists():
                        continue
                    result = read(exit_path)
                    if result['returncode'] == 75:
                        status.pop('stamp')
                        status.update(stage='waiting_for_idle_gpu', last_allocation_race=result)
                    elif result['returncode']:
                        status.update(stage='failed', error=result)
                        continue
                    else:
                        # Success requires the scoped validator, not only exit 0.
                        if (summary_path.exists() and read(summary_path).get('complete')
                                and read(summary_path).get('validation') != 'not run'
                                and (dest / 'complete.json').exists()):
                            status.update(stage='complete', summary=str(summary_path))
                            continue
                        status.update(stage='failed', error='controller exited without complete scoped summary')
                        continue
                if status['stage'] == 'failed':
                    continue
                if source_render_started(job):
                    prepare(job)
                    if summary_path.exists() and read(summary_path).get('complete'):
                        status.update(stage='complete', summary=str(summary_path))
                    else:
                        status.update(stage='following_original_native_render', source_root=job['source_root'])
                    continue
                pending.append(job)
            if all(s['stage'] in ('complete', 'failed') for s in state['jobs'].values()):
                state.update(stage='complete' if all(s['stage'] == 'complete' for s in state['jobs'].values()) else 'failed', updated=now())
                dump(state_path, state)
                return
            if pending:
                preflight = subprocess.run([sys.executable, str(ROOT / 'skills/gpu-preflight/scripts/check_gpus.py'),
                    '--json', '--min-free-mib', '24000'], capture_output=True, text=True, timeout=100)
                report = json.loads(preflight.stdout)
                dump(root / 'resources/latest_fleet.json', report)
                available = [dict(g, host=node['host']) for node in report['nodes'] if node.get('status') == 'ok'
                    for g in node.get('gpus', []) if g.get('idle') and g.get('eligible') and not g.get('faulty')
                    and not (node['host'] == 'nnmc63' and g['index'] == 0)]
                for job in pending:
                    candidates = sorted([g for g in available if g['name'] == job['hardware']],
                                        key=lambda g: (g['host'], g['index']))
                    for gpu in candidates:
                        audit = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', gpu['host'],
                            'env', 'CUDA_VISIBLE_DEVICES=' + gpu['uuid'], MODEL_PYTHON,
                            str(REMOTE_ROOT / 'pi05probe/egl_device_audit.py'), '--uuid', gpu['uuid']],
                            capture_output=True, text=True, timeout=40)
                        if audit.returncode:
                            continue
                        mapping = json.loads(audit.stdout)
                        assigned = {key: gpu[key] for key in ('host', 'index', 'uuid', 'name')}
                        assigned['egl_index'] = mapping['egl_index']
                        stamp = str(time.time_ns())
                        payload = dict(job=job, gpu=assigned, stamp=stamp)
                        script = 'payload = ' + repr(payload) + '\n' + REMOTE_LAUNCH
                        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                            gpu['host'], 'python3', '-'], input=script, capture_output=True, text=True, timeout=40)
                        if result.returncode:
                            state['jobs'][job['model']].update(last_launch_error=result.stderr[-1500:])
                            continue
                        launch = json.loads(result.stdout)
                        state['jobs'][job['model']].update(stage='controller_launched', stamp=stamp,
                                                         gpu=assigned, launch=launch)
                        available = [g for g in available if g['uuid'] != gpu['uuid']]
                        break
            state.update(stage='active', updated=now(), no_stop_operations=True,
                         note='Controllers repeat full fleet, EGL UUID audit and immediate selected-node check before GPU launch.')
            dump(state_path, state)
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--supervise-json')
    opts = parser.parse_args()
    if opts.supervise_json:
        supervise(json.loads(opts.supervise_json))
    else:
        if not opts.root:
            parser.error('--root is required')
        try:
            main(opts)
        except BaseException as error:
            dump(Path(opts.root) / 'queue_error.json', dict(error=repr(error), updated=now()))
            raise
