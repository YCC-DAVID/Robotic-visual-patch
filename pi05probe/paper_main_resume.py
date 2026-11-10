"""Resume the confirmed LIBERO seven / RoboTwin one paper rollout batch.

This controller uses no GPU. Each child checks the fleet, the device mapping,
and its selected idle GPU before every integrated computation. It never sends
signals to children or services; jobs finish their own bounded work.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'out/main_results/paper_resume_20261108'
ENVROOT = Path(os.environ.get('WAMATTACK_ENVROOT', '/shared/user65/miniconda3/envs'))
TASKS = ['put_the_bowl_on_top_of_the_cabinet', 'put_the_wine_bottle_on_the_rack',
         'turn_on_the_stove', 'put_the_bowl_on_the_stove',
         'put_the_cream_cheese_in_the_bowl', 'open_the_middle_drawer_of_the_cabinet']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--libero-gpus', required=True)
    ap.add_argument('--robotwin-gpu', required=True)
    args = ap.parse_args()
    libero, robotwin = json.loads(args.libero_gpus), json.loads(args.robotwin_gpu)
    assigned = libero + [robotwin]
    host = socket.gethostname().split('.')[0]
    assert assigned and all(g['host'] == host for g in assigned)
    assert len({g['uuid'] for g in assigned}) == len(assigned)
    DEST.mkdir(parents=True, exist_ok=True)
    with (DEST / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        commands = {
            'libero': [str(ENVROOT / 'openpi-server/bin/python'), '-u',
                       str(ROOT / 'pi05probe/main_method_pipeline.py'),
                       '--gpus', json.dumps(libero), '--tasks', ','.join(TASKS),
                       '--port-base', '18461'],
            'robotwin': [str(ENVROOT / 'openpi-server/bin/python'), '-u',
                         str(ROOT / 'robotwinprobe/main_method_pipeline.py'),
                         '--gpus', json.dumps([robotwin]), '--tasks', 'place_container_plate',
                         '--port', '18452'],
        }
        state = dict(started=now(), host=host, pid=os.getpid(), stage='starting',
                     scope='LIBERO existing seven tasks + RoboTwin place_container_plate',
                     purpose='paper main experiment real closed-loop rollout',
                     assigned_gpus=assigned, commands=commands, jobs={},
                     controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     note='Assigned devices are checked again by each worker; no reservation or stop permission implied')
        dump(DEST / 'launch.json', state)
        children, logs = {}, {}
        try:
            for name, cmd in commands.items():
                logs[name] = (DEST / (name + '.log')).open('a')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
                child = subprocess.Popen(cmd, cwd=ROOT / 'pi05probe', env=env,
                                         stdout=logs[name], stderr=subprocess.STDOUT)
                children[name] = child
                state['jobs'][name] = dict(pid=child.pid, log=str(DEST / (name + '.log')),
                                           exit_code=None)
            state['stage'] = 'running'
            dump(DEST / 'progress.json', dict(state, updated=now()))
            while any(child.poll() is None for child in children.values()):
                for name, child in children.items():
                    state['jobs'][name]['exit_code'] = child.poll()
                dump(DEST / 'progress.json', dict(state, updated=now()))
                time.sleep(10)
            for name, child in children.items():
                state['jobs'][name]['exit_code'] = child.returncode
            state['stage'] = 'workers_finished' if all(c.returncode == 0 for c in children.values()) else 'incomplete'
            dump(DEST / 'progress.json', dict(state, updated=now()))
        finally:
            # Do not stop jobs on controller interruption. Their own work and
            # resource records remain available for status / approved handling.
            for log in logs.values():
                log.close()


if __name__ == '__main__':
    main()
