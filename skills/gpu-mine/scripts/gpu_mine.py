#!/usr/bin/env python3
"""Who-am-I GPU ledger for the shared `user1` account.

  status      list this project's GPU jobs on every node: cards, memory, utilisation, liveness, IDLE/ACTIVE verdict
  candidates  given a card demand, propose which IDLE jobs to stop and write a plan (nothing is stopped)
  stop        execute a plan ONLY after the user explicitly approved that plan id; re-verifies every target first

Ownership = the process (or an ancestor below the login/agent session) has an owner marker in its cwd or command line.
"""
import argparse
import concurrent.futures
import datetime
import json
import os
import pathlib
import secrets
import socket
import subprocess
import sys
import time

HOSTS = [f'nnmc{i}' for i in range(60, 66)] + ['nnmc71', 'nnmc76']
MARKERS = ['/workspace/chence/']
PROBE = (pathlib.Path(__file__).resolve().parent / 'probe.py').read_text()
STATE = pathlib.Path(os.environ.get('GPU_MINE_STATE', pathlib.Path.home() / '.cache' / 'gpu-mine'))
PLAN_TTL_S = 1800


def remote(host, mode, opts, timeout):
    cmd = ['python3', '-', mode, json.dumps(opts)]
    if host != socket.gethostname().split('.')[0]:
        cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host,
               'python3 - ' + mode + " '" + json.dumps(opts).replace("'", "'\\''") + "'"]
    try:
        p = subprocess.run(cmd, input=PROBE, capture_output=True, text=True, timeout=timeout)
        res = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else \
            dict(status='error', error=p.stderr.strip()[-300:] or f'exit {p.returncode}')
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        res = dict(status='error', error=f'{type(exc).__name__}: {exc}')
    res['host'] = host
    return res


def classify(job, a):
    """IDLE only when every liveness signal is quiet; anything unmeasurable keeps the job out of the candidate list."""
    why_active, unknown = [], []
    for c in job['cards']:
        if c['util_avg'] is None:
            unknown.append(f"GPU{c['index']} util N/A")
        elif c['util_avg'] >= a.util_pct and c['shared']:
            unknown.append(f"GPU{c['index']} busy but shared with other processes")
        elif c['util_avg'] >= a.util_pct:
            why_active.append(f"GPU{c['index']} util {c['util_avg']}%")
    if job['cpu_pct'] >= a.cpu_pct:
        why_active.append(f"CPU {job['cpu_pct']}%")
    if job['last_write_age_s'] is not None and job['last_write_age_s'] < a.idle_min * 60:
        why_active.append(f"wrote {fmt_age(job['last_write_age_s'])} ago")
    if job['listen_ports']:
        if job['established'] is None:
            unknown.append('server connection count unknown')
        elif job['established'] > 0:
            why_active.append(f"server has {job['established']} live connections")
    if job['age_s'] < a.idle_min * 60:
        why_active.append(f"started {fmt_age(job['age_s'])} ago")
    if why_active:
        return 'ACTIVE', why_active
    if unknown:
        return 'UNKNOWN', unknown
    quiet = [f"util<{a.util_pct}% on its cards", f"CPU {job['cpu_pct']}%",
             'no open output file' if job['last_write_age_s'] is None else f"last write {fmt_age(job['last_write_age_s'])} ago"]
    if job['listen_ports']:
        quiet.append(f"server port {','.join(job['listen_ports'])} has 0 connections")
    return 'IDLE', quiet


def fmt_age(s):
    if s is None:
        return '?'
    d, s = divmod(int(s), 86400)
    h, s = divmod(s, 3600)
    return f'{d}d{h}h' if d else (f'{h}h{s // 60}m' if h else f'{s // 60}m')


def card_is_free(g, a):
    return not g['procs'] and g['util_avg'] is not None and (g['used_mib'] or 0) <= 256 and g['util_avg'] <= 0 \
        and g['total_mib'] - (g['used_mib'] or 0) >= a.min_free_mib


def collect(a):
    opts = dict(markers=a.marker or MARKERS, samples=a.samples, interval=a.interval)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.hosts)) as pool:
        nodes = list(pool.map(lambda h: remote(h, 'probe', opts, 90), a.hosts))
    for n in nodes:
        for j in n.get('jobs', []):
            j['verdict'], j['reasons'] = classify(j, a)
    snap = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), nodes=nodes)
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / 'last_status.json').write_text(json.dumps(snap, indent=1))
    return snap


def print_status(snap):
    print('checked_at', snap['checked_at'], '(UTC)')
    tot_cards = tot_mib = 0
    for n in snap['nodes']:
        if n['status'] != 'ok':
            print(f"\n[{n['host']}] UNKNOWN: {n.get('error')}")
            continue
        if not n['jobs']:
            continue
        print(f"\n[{n['host']}]")
        for j in sorted(n['jobs'], key=lambda j: min(c['index'] for c in j['cards'])):
            cards = ' '.join(f"GPU{c['index']}:{c['mib']}MiB/{c['util_avg']}%{'(shared)' if c['shared'] else ''}"
                             for c in j['cards'])
            tot_cards += len(j['cards'])
            tot_mib += sum(c['mib'] for c in j['cards'])
            print(f"  {j['verdict']:<7} pid={j['root']} age={fmt_age(j['age_s'])} {cards}")
            print(f"          cmd: {j['cmd'][:220]}")
            print(f"          cwd: {j['cwd']}")
            lw = 'none' if j['last_write_path'] is None else f"{fmt_age(j['last_write_age_s'])} ago {j['last_write_path']}"
            extra = f" ports={','.join(j['listen_ports'])} conns={j['established']}" if j['listen_ports'] else ''
            print(f"          cpu={j['cpu_pct']}% last_write={lw}{extra}" + (f" unit={j['unit']}" if j['unit'] else ''))
            print(f"          why: {'; '.join(j['reasons'])}")
    print(f"\nTOTAL: {tot_cards} card-slots, {tot_mib} MiB held by this project "
          "(a card shared by two own jobs is counted twice)")
    bad = [n['host'] for n in snap['nodes'] if n['status'] != 'ok']
    if bad:
        print('UNKNOWN nodes (own jobs there not visible):', ', '.join(bad))


def cmd_status(a):
    snap = collect(a)
    if a.json:
        print(json.dumps(snap, indent=1))
    else:
        print_status(snap)
    return 0 if all(n['status'] == 'ok' for n in snap['nodes']) else 2


def cmd_candidates(a):
    snap = collect(a)
    best = None
    for n in snap['nodes']:
        if n['status'] != 'ok':
            continue
        free = [g['index'] for g in n['gpus'] if card_is_free(g, a)]
        idle = [j for j in n['jobs'] if j['verdict'] == 'IDLE']
        idle_pids = {p for j in idle for p in j['gpu_pids']}
        # a card is reclaimable only if every process on it belongs to an IDLE own job
        reclaim = {}
        for g in n['gpus']:
            if g['procs'] and all(p['pid'] in idle_pids for p in g['procs']) \
                    and g['total_mib'] >= a.min_free_mib and g['util_avg'] is not None:
                owners = [j for j in idle if any(p['pid'] in j['gpu_pids'] for p in g['procs'])]
                reclaim[g['index']] = owners
        chosen, gained = [], []
        need = max(0, a.need - len(free))
        # greedy: jobs that free the most whole cards first, then the oldest
        for j in sorted(idle, key=lambda j: (-sum(1 for c in j['cards'] if c['index'] in reclaim), -j['age_s'])):
            if len(gained) >= need:
                break
            cards = [c['index'] for c in j['cards'] if c['index'] in reclaim]
            if not cards:
                continue
            chosen.append(j)
            gained += [c for c in cards if all(o in chosen for o in reclaim[c]) and c not in gained]
        ok = len(free) + len(gained) >= a.need
        score = (ok, -len(chosen), len(free) + len(gained))
        if best is None or score > best[0]:
            best = (score, n['host'], free, chosen, gained, ok)
    if best is None:
        print('No node could be inspected; nothing to propose.')
        return 2
    _, host, free, chosen, gained, ok = best
    print(f"need {a.need} card(s) with >= {a.min_free_mib} MiB on one node; best node: {host}")
    print(f"  already free: {free or 'none'}")
    if len(free) >= a.need:
        print('  enough free cards already - no stop needed. Re-run gpu-preflight before launching.')
        return 0
    if not chosen:
        print('  no IDLE own job can free a card there. Do NOT stop anything; report the shortage to the user.')
        return 1
    plan = dict(id=f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}", created=time.time(), host=host,
                need=a.need, free_before=free, cards_freed=sorted(gained), sufficient=ok,
                targets=[dict(host=host, root=j['root'], fingerprint=j['fingerprint'], unit=j['unit'], cmd=j['cmd'],
                              cwd=j['cwd'], cards=[c['index'] for c in j['cards']], reasons=j['reasons'],
                              age_s=j['age_s']) for j in chosen])
    (STATE / 'plans').mkdir(parents=True, exist_ok=True)
    (STATE / 'plans' / f"{plan['id']}.json").write_text(json.dumps(plan, indent=1))
    print(f"  PLAN {plan['id']} (valid {PLAN_TTL_S // 60} min) - stopping these would free GPU {sorted(gained)}"
          + ('' if ok else f'  [STILL SHORT of {a.need}]'))
    for t in plan['targets']:
        print(f"   - {host} pid={t['root']} GPU{t['cards']} age={fmt_age(t['age_s'])}" + (f" unit={t['unit']}" if t['unit'] else ''))
        print(f"     cmd: {t['cmd'][:220]}")
        print(f"     cwd: {t['cwd']}")
        print(f"     idle evidence: {'; '.join(t['reasons'])}")
    print('\nNOTHING HAS BEEN STOPPED. Show this plan to the user and wait for explicit approval of plan id '
          f"{plan['id']} before running: gpu_mine.py stop --plan {plan['id']} --approved '<user words>'")
    return 0 if ok else 1


def cmd_stop(a):
    path = STATE / 'plans' / f'{a.plan}.json'
    if not path.exists():
        print('unknown plan id', a.plan)
        return 2
    plan = json.loads(path.read_text())
    if plan.get('executed'):
        print('plan already executed at', plan['executed'])
        return 2
    if time.time() - plan['created'] > PLAN_TTL_S:
        print('plan expired; run candidates again and get fresh approval')
        return 2
    if not a.approved.strip():
        print('--approved must quote the user approval')
        return 2
    results = []
    for t in plan['targets']:
        r = remote(t['host'], 'stop', dict(root=t['root'], fingerprint=t['fingerprint'], unit=t['unit'],
                                           grace=a.grace, kill_after=a.kill_after), a.grace + 120)
        results.append(dict(target=t['root'], **r))
        print(f"{t['host']} pid={t['root']}: {r.get('status')} {r.get('action', '')} {r.get('detail', '')}"
              f"{' survivors=' + str(r['survivors']) if r.get('survivors') else ''}{r.get('error', '')}")
    plan['executed'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    plan['approved'], plan['results'] = a.approved, results
    path.write_text(json.dumps(plan, indent=1))
    with open(STATE / 'audit.jsonl', 'a') as f:
        f.write(json.dumps(dict(plan=plan['id'], at=plan['executed'], approved=a.approved, results=results)) + '\n')
    print('Done. Run gpu-preflight again before using the freed cards.')
    return 0 if all(r.get('status') == 'stopped' for r in results) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    for name in ('status', 'candidates'):
        s = sub.add_parser(name)
        s.add_argument('--hosts', nargs='+', choices=HOSTS, default=HOSTS)
        s.add_argument('--marker', action='append', help=f'owner marker substring (default {MARKERS})')
        s.add_argument('--samples', type=int, default=3)
        s.add_argument('--interval', type=float, default=2.0)
        s.add_argument('--idle-min', type=int, default=60, help='no output/start within this many minutes')
        s.add_argument('--util-pct', type=float, default=5)
        s.add_argument('--cpu-pct', type=float, default=10)
        s.add_argument('--min-free-mib', type=int, default=0)
        s.add_argument('--json', action='store_true')
        if name == 'candidates':
            s.add_argument('--need', type=int, required=True, help='cards needed on one node')
    s = sub.add_parser('stop')
    s.add_argument('--plan', required=True)
    s.add_argument('--approved', required=True, help="the user's approval, quoted verbatim")
    s.add_argument('--grace', type=int, default=60)
    s.add_argument('--kill-after', action='store_true', help='SIGKILL survivors after grace (needs its own approval)')
    a = p.parse_args()
    return dict(status=cmd_status, candidates=cmd_candidates, stop=cmd_stop)[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
