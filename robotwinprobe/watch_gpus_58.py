#!/usr/bin/env python3
"""Poll nnmc60-65 for two idle healthy L40S on one launchable node and start #58.

Exit codes: 0 launched (JSON on stdout), 3 deadline reached without cards,
4 refused because #58 already has execution state, 5 launch failed.
The runner itself re-checks eligibility/ECC per card before using it.
"""
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

HOME_ROOT = '/home/user1/workspace/chence/WAMattack'
NFS_ROOT = '/shared/user65/workspace/chence/WAMattack'
EXP_REL = 'robotwinprobe/out/libero_neighborhood_pilot_20261102'
EXP = Path(HOME_ROOT) / EXP_REL
CHECK = Path(HOME_ROOT) / 'skills/gpu-preflight/scripts/check_gpus.py'
LOG = EXP / 'logs' / 'gpu_watch.log'
# Only nodes where the CPU readiness check passed (paths resolve to the same files).
LAUNCH = {
    'nnmc65': dict(root=HOME_ROOT, py='/home/user1/miniconda3/bin/python'),
    'nnmc60': dict(root=NFS_ROOT, py='/shared/user65/miniconda3/bin/python'),
    'nnmc61': dict(root=NFS_ROOT, py='/shared/user65/miniconda3/bin/python'),
}
ORDER = ['nnmc65', 'nnmc60', 'nnmc61']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def log(msg):
    with LOG.open('a') as f:
        f.write(f'{now()} {msg}\n')


def remote(host, command, timeout=40):
    cmd = ['bash', '-lc', command] if host == 'nnmc65' else [
        'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, command]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def scan(hosts=None):
    cmd = [sys.executable, str(CHECK), '--json', '--max-used-mib', '256', '--max-util', '0']
    if hosts:
        cmd += ['--hosts', *hosts]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return json.loads(p.stdout)


def free_cards(data):
    out = {}
    for n in data['nodes']:
        if n['status'] != 'ok':
            continue
        cards = [g for g in n['gpus'] if g['eligible'] and 'L40S' in g['name']
                 and not (n['host'] == 'nnmc63' and g['index'] == 0)]
        if cards:
            out[n['host']] = cards
    return out


def ecc_ok(host, uuid):
    p = remote(host, f'nvidia-smi --id={uuid} --query-gpu=ecc.errors.uncorrected.volatile.total,'
                     'ecc.errors.uncorrected.aggregate.total --format=csv,noheader,nounits')
    return p.returncode == 0 and [x.strip() for x in p.stdout.strip().split(',')] == ['0', '0']


def main():
    deadline = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 7000)
    log('watch start')
    while time.time() < deadline:
        if (EXP / 'execution.json').exists():
            log('execution.json exists; refusing to launch')
            print(json.dumps(dict(result='already_has_execution')))
            return 4
        try:
            first = free_cards(scan())
        except Exception as exc:  # unknown scan never implies availability
            log(f'scan error {exc!r}')
            time.sleep(15)
            continue
        log('free ' + json.dumps({h: [g['index'] for g in c] for h, c in first.items()}))
        host = next((h for h in ORDER if len(first.get(h, [])) >= 2), None)
        if host is None:
            time.sleep(15)
            continue
        time.sleep(10)
        again = free_cards(scan([host])).get(host, [])
        keep = [g for g in first[host] if g['uuid'] in {x['uuid'] for x in again}]
        keep = [g for g in keep if ecc_ok(host, g['uuid'])][:2]
        if len(keep) < 2:
            log(f'{host} not stable/healthy on recheck')
            time.sleep(10)
            continue
        cfg = LAUNCH[host]
        exp = f"{cfg['root']}/{EXP_REL}"
        ts = time.strftime('%Y%m%d_%H%M%S')
        a, b = keep[0]['index'], keep[1]['index']
        uuids = ','.join(g['uuid'] for g in keep)
        out = f'{exp}/logs/pipeline_{ts}.out'
        command = (f"cd {cfg['root']} && CUDA_VISIBLE_DEVICES={uuids} nohup {cfg['py']} "
                   f"robotwinprobe/libero_neighborhood_trial.py pipeline --root {exp} --gpus {a} {b} "
                   f"> {out} 2>&1 < /dev/null & echo $!")
        p = remote(host, command)
        pid = p.stdout.strip()
        record = dict(host=host, pid=pid, command=command, gpus=[a, b], uuids=uuids.split(','),
                      preflight_at=now(), stdout_log=out, launcher_rc=p.returncode, launcher_err=p.stderr[-2000:])
        (EXP / 'logs' / f'launch_{ts}.json').write_text(json.dumps(record, indent=2, ensure_ascii=False))
        log('launch ' + json.dumps(record, ensure_ascii=False))
        time.sleep(30)
        alive = remote(host, f'ps -o pid=,args= -p {pid}') if pid.isdigit() else None
        record['alive_after_30s'] = bool(alive and alive.returncode == 0 and 'pipeline' in alive.stdout)
        print(json.dumps(record, ensure_ascii=False))
        return 0 if record['alive_after_30s'] else 5
    log('deadline reached, no cards')
    print(json.dumps(dict(result='no_cards_before_deadline')))
    return 3


if __name__ == '__main__':
    sys.exit(main())
