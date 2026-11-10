"""Wait for one idle GPU (gpu-preflight rules) and launch run_mask_crop.sh there; then exit.

Polls every 5 min; re-checks the chosen host immediately before launch; never touches busy/UNKNOWN cards;
excludes nnmc63 GPU0 (historical ECC).  Records every decision in OUT/launch.json.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/shared/user65/workspace/chence/WAMattack')
OUT = ROOT / 'robotwinprobe/out/libero13_mask_crop_20261104'
CHECK = '/shared/user65/.codex/skills/gpu-preflight/scripts/check_gpus.py'
NEED = 24000
EXCLUDE = {('nnmc63', 0)}
DEADLINE = time.time() + 48 * 3600


def check(hosts=None):
    cmd = [sys.executable, CHECK, '--json', '--min-free-mib', str(NEED)] + (['--hosts', *hosts] if hosts else [])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return json.loads(r.stdout)


def eligible(info):
    for node in info['nodes']:
        if node['status'] != 'ok':
            continue
        for g in node['gpus']:
            if g['eligible'] and (node['host'], g['index']) not in EXCLUDE:
                yield node['host'], g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / 'launch.json'
    rows = []
    while time.time() < DEADLINE:
        try:
            cands = list(eligible(check()))
            # prefer H100 node, then most free memory
            cands.sort(key=lambda hg: (hg[0] != 'nnmc71', -hg[1]['free_mib']))
            for host, g in cands:
                again = {gg['uuid']: gg for _, gg in eligible(check([host]))}
                if g['uuid'] not in again:
                    rows.append(dict(at=datetime.now(timezone.utc).isoformat(), host=host, gpu=g['index'], recheck='no longer idle'))
                    continue
                cmd = (f'cd {ROOT}/robotwinprobe && nohup bash run_mask_crop.sh {g["uuid"]} '
                       f'> {OUT}/run_{host}_gpu{g["index"]}.log 2>&1 < /dev/null & echo $!')
                pid = subprocess.run(['ssh', host, cmd], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=60).stdout.strip()
                rows.append(dict(at=datetime.now(timezone.utc).isoformat(), host=host, gpu=g['index'], uuid=g['uuid'],
                                 name=g.get('name'), launched_pid=pid, recheck='idle'))
                log.write_text(json.dumps(rows, indent=1))
                print('launched', host, g['index'], pid, flush=True)
                return
            rows.append(dict(at=datetime.now(timezone.utc).isoformat(), candidates=len(cands)))
            rows = rows[-50:]
            log.write_text(json.dumps(rows, indent=1))
        except Exception as e:  # keep waiting on transient ssh/parse errors
            rows.append(dict(at=datetime.now(timezone.utc).isoformat(), error=repr(e)))
            log.write_text(json.dumps(rows[-50:], indent=1))
        time.sleep(300)
    print('deadline reached without a free GPU', flush=True)


if __name__ == '__main__':
    main()
