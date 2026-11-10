"""Wait for cards (gpu-preflight), then launch #62 three-source joint pipeline.
Placement, first match wins: 3 idle L40S on one node -> pi|fast|imagewam; 2 idle L40S -> pi+fast | imagewam;
1 idle H100 on nnmc71 only if its render-determinism test passed -> all three on one card.
(B300 excluded: the π0.5 torch env is cu126 without sm_100+ kernels.)
The LIBERO 5-task scheduler is paused while waiting and restarted once all #62 backends are up (or #62 fails)."""
import json, subprocess, time, datetime, glob
from pathlib import Path

ROOT = Path('/shared/user65/workspace/chence/WAMattack')
OUT = ROOT / 'robotwinprobe/out/three_source_joint_eot_20261106'
DET = ROOT / 'robotwinprobe/out/egl_determinism_20261106'
CHECK = '/home/user1/.codex/skills/gpu-preflight/scripts/check_gpus.py'
L40S_HOSTS = ['nnmc61', 'nnmc62', 'nnmc63', 'nnmc64', 'nnmc65']   # user: only nnmc61-65, no H100 for now
EXCLUDED = {('nnmc63', 0)}                                             # historical ECC card
PY = '/shared/user65/miniconda3/bin/python'
L5_UNIT = 'wam-libero-crossarch5'
L5_CMD = ['systemd-run', '--user', '--unit=' + L5_UNIT, '--working-directory=' + str(ROOT / 'robotwinprobe'),
          '-p', 'StandardOutput=append:' + str(ROOT / 'robotwinprobe/out/libero_crossarch5_20261105/scheduler.log'),
          '-p', 'StandardError=append:' + str(ROOT / 'robotwinprobe/out/libero_crossarch5_20261105/scheduler.log'),
          '/shared/user65/miniconda3/envs/openpi-server/bin/python', '-u', str(ROOT / 'robotwinprobe/libero_crossarch5.py'), 'run']
LOG = OUT / 'logs/wait_launch.log'


def log(*a):
    with LOG.open('a') as f:
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), *a, file=f, flush=True)


def h100_render_ok():
    for f in glob.glob(str(DET / 'nnmc71_egl*.json')):
        r = json.loads(Path(f).read_text())
        cams = r.get('cams', {})
        if len(cams) == 2 and all(not any(v['px_diff_vs_obs']) and not any(v['px_diff_vs_first']) for v in cams.values()) \
                and r.get('segmentation_px_diff_vs_first') is not None and not any(r['segmentation_px_diff_vs_first']):
            return True
    return False


def idle(hosts):
    p = subprocess.run(['python3', CHECK, '--json', '--min-free-mib', '40000', '--hosts', *hosts], capture_output=True, text=True, timeout=300)
    inv = json.loads(p.stdout); out = {}
    for n in inv['nodes']:
        if n['status'] == 'ok':
            out[n['host']] = [g for g in n['gpus'] if g['eligible'] and (n['host'], g['index']) not in EXCLUDED]
    return inv['checked_at'], out


def plan(cards):
    """cards: {host: [idle gpus]} -> (host, [pi, fast, imagewam] physical indices, uuids) or None"""
    l40 = {h: [g for g in c if 'L40S' in g['name']] for h, c in cards.items() if h in L40S_HOSTS}
    for need in (3, 2):
        for h, c in sorted(l40.items(), key=lambda kv: -len(kv[1])):
            if len(c) >= need:
                gs = c[:3] if need == 3 else [c[0], c[0], c[1]]
                return h, [g['index'] for g in gs], [g['uuid'] for g in gs]
    return None


def main():
    (OUT / 'logs').mkdir(parents=True, exist_ok=True)
    subprocess.run(['systemctl', '--user', 'stop', L5_UNIT])
    log('paused', L5_UNIT)
    while True:
        try:
            at, cards = idle(L40S_HOSTS)
            pick = plan(cards)
            if pick:
                at2, again = idle([pick[0]])                        # re-check right before launch
                pick2 = plan(again)
                if pick2 and pick2[0] == pick[0]:
                    host, gpus, uuids = pick2
                    cmd = (f'cd {ROOT} && WAMATTACK_PROJECT_ROOT={ROOT} nohup setsid {PY} -u {ROOT}/robotwinprobe/three_source_joint_eot.py pipeline '
                           f'--gpus {gpus[0]} {gpus[1]} {gpus[2]} >> {OUT}/logs/pipeline.log 2>&1 < /dev/null & echo $!')
                    try:
                        p = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', host, cmd], capture_output=True, text=True, timeout=60); lp = p.stdout.strip()
                    except subprocess.TimeoutExpired:
                        lp = 'ssh timeout'   # the remote job may still have started: never relaunch blindly
                    rec = dict(host=host, gpus=dict(zip(['pi', 'fast', 'imagewam'], gpus)), uuids=uuids, checked_at=at2,
                               launcher_pid=lp, launched=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    (OUT / 'wait_launch.json').write_text(json.dumps(rec, indent=2))
                    log('launched', rec)
                    break
        except Exception as e:
            log('error', repr(e))
        time.sleep(120)
    began = time.time(); st = ''
    while time.time() - began < 3600:                          # wait until all backends hold their cards
        st = json.loads((OUT / 'status.json').read_text()).get('stage', '') if (OUT / 'status.json').exists() else ''
        if st.startswith(('native_and_gradient', 'train', 'evaluate', 'runtime_error', 'verifying', 'complete')):
            break
        time.sleep(30)
    log('pipeline stage', st)
    subprocess.run(['systemctl', '--user', 'reset-failed', L5_UNIT])
    subprocess.run(L5_CMD)
    log('resumed', L5_UNIT)


if __name__ == '__main__':
    main()
