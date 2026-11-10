#!/usr/bin/env python3
"""Node-side helper for gpu_mine.py; it is piped over ssh stdin, so it must stay stdlib-only.

  probe.py probe '<json opts>'  -> JSON: GPUs + this account's jobs whose process tree matches the owner markers
  probe.py stop  '<json opts>'  -> JSON: re-verify fingerprints, then SIGTERM (or systemctl --user stop) one job
"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

AGENTS = ('codex', 'claude', 'node', 't3', 'cursor')
SESSION = ('sshd', 'systemd', 'init', 'tmux: server', 'tmux', 'screen', 'SCREEN', 'login') + AGENTS
SHELLS = ('bash', 'sh', 'zsh', 'dash', 'fish')
CLK = os.sysconf('SC_CLK_TCK')


def rd(path, binary=False):
    try:
        with open(path, 'rb' if binary else 'r') as f:
            return f.read()
    except OSError:
        return None


def boot_time():
    for line in (rd('/proc/stat') or '').splitlines():
        if line.startswith('btime'):
            return int(line.split()[1])
    return 0


def proc_table():
    uid, procs = os.getuid(), {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        pid = int(d)
        try:
            if os.stat(f'/proc/{pid}').st_uid != uid:
                continue
        except OSError:
            continue
        stat = rd(f'/proc/{pid}/stat')
        if not stat:
            continue
        rp = stat.rfind(')')
        comm, rest = stat[stat.find('(') + 1:rp], stat[rp + 2:].split()
        raw = rd(f'/proc/{pid}/cmdline', binary=True) or b''
        try:
            cwd = os.readlink(f'/proc/{pid}/cwd')
        except OSError:
            cwd = ''
        procs[pid] = dict(pid=pid, comm=comm, ppid=int(rest[1]), pgid=int(rest[2]), start=int(rest[19]),
                          cpu_ticks=int(rest[11]) + int(rest[12]),
                          cmd=raw.replace(b'\0', b' ').decode(errors='replace').strip(), cwd=cwd)
    return procs


def is_agent(p):
    return p['comm'].startswith(AGENTS)


def owned(pid, procs, markers, memo):
    if pid in memo:
        return memo[pid]
    memo[pid] = False                                           # cycle guard
    p = procs.get(pid)
    hit = bool(p) and not is_agent(p) and (any(m in p['cwd'] or m in p['cmd'] for m in markers)
                                           or owned(p['ppid'], procs, markers, memo))
    memo[pid] = hit
    return hit


def job_root(pid, procs, markers, memo):
    """Climb to the top-most owned ancestor that is not a login/agent session or its interactive shell."""
    cur = pid
    while True:
        par = procs.get(procs[cur]['ppid'])
        if not par or not owned(par['pid'], procs, markers, memo) or par['comm'].startswith(SESSION):
            return cur
        gp = procs.get(par['ppid'])
        if par['comm'] in SHELLS and gp and gp['comm'].startswith(SESSION) and gp['pid'] != 1 \
                and gp['cmd'] != '/lib/systemd/systemd --user':
            return cur                                          # shell owned by a session, not an orphaned job script
        cur = par['pid']


def descendants(root, procs):
    kids = {}
    for p in procs.values():
        kids.setdefault(p['ppid'], []).append(p['pid'])
    out, stack = [], [root]
    while stack:
        x = stack.pop()
        out.append(x)
        stack.extend(kids.get(x, []))
    return out


def fingerprint(p):
    return hashlib.sha1(f"{p['start']}|{p['cmd']}".encode()).hexdigest()[:16]


def systemd_unit(pid, members):
    cg = (rd(f'/proc/{pid}/cgroup') or '').strip().splitlines()
    path = next((l.split(':', 2)[2] for l in cg if l.startswith('0::')), '')
    leaf = path.rsplit('/', 1)[-1]
    if 'user@' not in path or not leaf.endswith('.service'):
        return None
    procs_in = rd(f'/sys/fs/cgroup{path}/cgroup.procs')
    pids = {int(x) for x in (procs_in or '').split()}
    return leaf if pids and pids <= set(members) else None    # only if the unit contains nothing but this job


def write_activity(pids):
    newest = (None, None)
    for pid in pids:
        try:
            fds = os.listdir(f'/proc/{pid}/fd')
        except OSError:
            continue
        for fd in fds:
            try:
                tgt = os.readlink(f'/proc/{pid}/fd/{fd}')
            except OSError:
                continue
            if not tgt.startswith('/') or tgt.startswith(('/dev/', '/proc/', '/sys/', '/tmp/')) \
                    or '/.cache/' in tgt or tgt.endswith(' (deleted)'):
                continue
            flags = next((l.split()[1] for l in (rd(f'/proc/{pid}/fdinfo/{fd}') or '').splitlines()
                          if l.startswith('flags:')), '0')
            if int(flags, 8) & 3 == 0:                          # read-only handle
                continue
            try:
                m = os.stat(tgt).st_mtime
            except OSError:
                continue
            if newest[0] is None or m > newest[0]:
                newest = (m, tgt)
    return newest


def sockets(pids):
    pids, listen, est = set(pids), set(), 0
    try:
        out = subprocess.run(['ss', '-ltnpH'], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if any(f'pid={p},' in line for p in pids):
                listen.add(line.split()[3].rsplit(':', 1)[1])
        if listen:
            out = subprocess.run(['ss', '-tnH', 'state', 'established'], capture_output=True, text=True, timeout=10).stdout
            est = sum(1 for l in out.splitlines() if len(l.split()) >= 4 and l.split()[2].rsplit(':', 1)[1] in listen)
    except (OSError, subprocess.SubprocessError, IndexError):
        return sorted(listen), None
    return sorted(listen), est


def gpu_query():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20, check=True)
    return [[c.strip() for c in l.split(',')] for l in q.stdout.strip().splitlines()]


def probe(opts):
    markers = opts['markers']
    samples, interval = max(1, opts.get('samples', 3)), opts.get('interval', 2.0)
    procs0 = proc_table()
    t0 = time.time()
    util = {}
    rows = []
    for i in range(samples):
        rows = gpu_query()
        for idx, uuid, name, tot, used, u in rows:
            util.setdefault(uuid, []).append(int(u) if u.isdigit() else None)
        if i + 1 < samples:
            time.sleep(interval)
    xml = ET.fromstring(subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True, timeout=30,
                                       check=True).stdout)
    procs = proc_table()
    dt = max(time.time() - t0, 1e-3)
    gpus, by_pid = {}, {}
    for idx, uuid, name, tot, used, _ in rows:
        us = util[uuid]
        gpus[uuid] = dict(index=int(idx), uuid=uuid, name=name, total_mib=int(tot),
                          used_mib=int(used) if used.isdigit() else None,
                          util_avg=None if None in us else round(sum(us) / len(us), 1), util_samples=us, procs=[])
    for g in xml.findall('gpu'):
        uuid = (g.findtext('uuid') or '').strip()
        for pi in g.findall('processes/process_info'):
            pid_txt = (pi.findtext('pid') or '').strip()
            mem = (pi.findtext('used_memory') or '').split()[0]
            if uuid in gpus and pid_txt.isdigit():
                rec = dict(pid=int(pid_txt), type=(pi.findtext('type') or '').strip(),
                           mib=int(mem) if mem.isdigit() else None)
                gpus[uuid]['procs'].append(rec)
                by_pid.setdefault(int(pid_txt), []).append(uuid)
    memo, jobs = {}, {}
    for pid in by_pid:
        if pid in procs and owned(pid, procs, markers, memo):
            jobs.setdefault(job_root(pid, procs, markers, memo), set()).add(pid)
    btime, now = boot_time(), time.time()
    out_jobs = []
    for root, gpids in jobs.items():
        members = descendants(root, procs)
        cpu = sum(procs[p]['cpu_ticks'] - procs0[p]['cpu_ticks'] for p in members if p in procs0)
        mtime, path = write_activity(members)
        ports, est = sockets(members)
        cards = []
        for uuid in sorted({u for p in gpids for u in by_pid[p]}, key=lambda u: gpus[u]['index']):
            g = gpus[uuid]
            mine = [x for x in g['procs'] if x['pid'] in members]
            cards.append(dict(index=g['index'], uuid=uuid, mib=sum(x['mib'] or 0 for x in mine),
                              types=sorted({x['type'] for x in mine}), util_avg=g['util_avg'],
                              shared=any(x['pid'] not in members for x in g['procs'])))
        r = procs[root]
        par = procs.get(r['ppid'])
        out_jobs.append(dict(
            root=root, comm=r['comm'], cmd=r['cmd'][:600], cwd=r['cwd'], fingerprint=fingerprint(r),
            pgid=r['pgid'], parent=dict(pid=r['ppid'], comm=par['comm'] if par else '?'),
            age_s=int(now - (btime + r['start'] / CLK)), members=len(members), gpu_pids=sorted(gpids),
            cpu_pct=round(100.0 * cpu / CLK / dt, 1), last_write_age_s=None if mtime is None else int(now - mtime),
            last_write_path=path, listen_ports=ports, established=est,
            unit=systemd_unit(root, members), cards=cards))
    return dict(status='ok', gpus=sorted(gpus.values(), key=lambda g: g['index']), jobs=out_jobs)


def stop(opts):
    root, grace = opts['root'], opts.get('grace', 60)
    procs = proc_table()
    p = procs.get(root)
    if not p or fingerprint(p) != opts['fingerprint']:
        return dict(status='mismatch', detail='root pid gone or reused; nothing signalled')
    members = descendants(root, procs)
    unit = systemd_unit(root, members)
    if opts.get('unit') and unit == opts['unit']:
        r = subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, text=True, timeout=grace + 30)
        action = f'systemctl --user stop {unit} -> rc={r.returncode} {r.stderr.strip()}'
    else:
        for pid in members:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        action = f'SIGTERM -> {len(members)} processes'
    deadline = time.time() + grace
    while time.time() < deadline:
        alive = [x for x in members if os.path.exists(f'/proc/{x}')]
        if not alive:
            break
        time.sleep(2)
    alive = [x for x in members if os.path.exists(f'/proc/{x}')]
    if alive and opts.get('kill_after'):
        for pid in alive:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(3)
        action += f'; SIGKILL -> {len(alive)}'
        alive = [x for x in members if os.path.exists(f'/proc/{x}')]
    return dict(status='stopped' if not alive else 'survivors', action=action, survivors=alive)


if __name__ == '__main__':
    mode, opts = sys.argv[1], json.loads(sys.argv[2])
    try:
        print(json.dumps(probe(opts) if mode == 'probe' else stop(opts)))
    except Exception as exc:                                    # reported as UNKNOWN by the controller
        print(json.dumps(dict(status='error', error=f'{type(exc).__name__}: {exc}')))
