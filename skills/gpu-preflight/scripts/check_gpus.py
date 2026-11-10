#!/usr/bin/env python3
"""Read-only GPU inventory; failed or incomplete queries never imply availability."""
import argparse
import concurrent.futures
import csv
import datetime
import io
import json
import socket
import subprocess
import sys


def query(host, args):
    def run(command):
        cmd = ['bash', '-lc', command] if host == socket.gethostname().split('.')[0] else [
            'ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host, command]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        if p.returncode:
            raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f'exit {p.returncode}')
        return list(csv.reader(io.StringIO(p.stdout), skipinitialspace=True))
    try:
        rows = run('nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits')
        processes = run('nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits')
        if not rows:
            raise ValueError('No GPUs returned')
        active, faulty = {}, set()
        for row in processes:
            if len(row) == 2 and row[0].startswith('GPU-') and row[1].strip() == '[N/A]':
                faulty.add(row[0].strip())  # per-GPU fault (e.g. requires reset): that card only is UNKNOWN
                continue
            if len(row) != 2 or not row[0].startswith('GPU-') or not row[1].strip().isdigit():
                raise ValueError(f'Unrecognized process data: {row}')
            active.setdefault(row[0].strip(), []).append(int(row[1]))
        gpus = []
        for row in rows:
            idx, uuid, name, total, used, util = [s.strip() for s in row]
            bad = uuid in faulty or not util.isdigit()
            gpu = dict(index=int(idx), uuid=uuid, name=name, total_mib=int(total),
                       used_mib=int(used), utilization_pct=None if bad else int(util), pids=active.get(uuid, []),
                       faulty=bad)
            gpu['free_mib'] = gpu['total_mib'] - gpu['used_mib']
            gpu['idle'] = (not bad and not gpu['pids'] and gpu['used_mib'] <= args.max_used_mib
                           and gpu['utilization_pct'] <= args.max_util)
            gpu['eligible'] = gpu['idle'] and gpu['free_mib'] >= args.min_free_mib
            gpus.append(gpu)
        return dict(host=host, status='ok', gpus=gpus)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return dict(host=host, status='unknown', error=str(exc), gpus=[])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    hosts = [f'nnmc{i}' for i in range(60, 66)] + ['nnmc71', 'nnmc76']  # GPU models and capacity are queried live
    p.add_argument('--hosts', nargs='+', choices=hosts, default=hosts)
    p.add_argument('--max-used-mib', type=int, default=256)
    p.add_argument('--max-util', type=int, default=0)
    p.add_argument('--min-free-mib', type=int, default=0)
    p.add_argument('--json', action='store_true')
    args = p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.hosts)) as pool:
        nodes = list(pool.map(lambda host: query(host, args), args.hosts))
    result = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), nodes=nodes)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print('Checked at:', result['checked_at'])
        for node in nodes:
            if node['status'] != 'ok':
                print(node['host'], 'UNKNOWN:', node['error'])
                continue
            for g in node['gpus']:
                state = 'FREE' if g['eligible'] else ('UNKNOWN(faulty)' if g['faulty'] else 'BUSY/INSUFFICIENT')
                print(f"{node['host']} GPU{g['index']} {state} {g['name']} used={g['used_mib']}/{g['total_mib']}MiB util={g['utilization_pct']}% pids={g['pids']} UUID={g['uuid']}")
    return 0 if all(n['status'] == 'ok' and not any(g['faulty'] for g in n['gpus']) for n in nodes) else 2

if __name__ == '__main__':
    sys.exit(main())
