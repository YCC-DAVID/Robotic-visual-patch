"""Run egl_render_determinism.py once per GPU model (H100 nnmc71, B300 nnmc76, L40S control) on the first card
that gpu-preflight reports idle; re-checks the node right before launch. Results in OUT/<host>_egl<k>.json."""
import json, subprocess, time, datetime
from pathlib import Path

ROOT = Path('/shared/user65/workspace/chence/WAMattack')
OUT = ROOT / 'robotwinprobe/out/egl_determinism_20261106'
CHECK = '/home/user1/.codex/skills/gpu-preflight/scripts/check_gpus.py'
SIM = '/shared/user65/miniconda3/envs/openpi-libero/bin/python'
WANT = {'H100': ['nnmc71'], 'B300': ['nnmc76'], 'L40S': ['nnmc60', 'nnmc61', 'nnmc62', 'nnmc64', 'nnmc65']}
EXCLUDED = {('nnmc63', 0)}


def log(*a):
    with (OUT / 'wait.log').open('a') as f:
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), *a, file=f, flush=True)


def idle(hosts):
    p = subprocess.run(['python3', CHECK, '--json', '--hosts', *hosts], capture_output=True, text=True, timeout=300)
    return [(n['host'], g) for n in json.loads(p.stdout)['nodes'] if n['status'] == 'ok'
            for g in n['gpus'] if g['eligible'] and (n['host'], g['index']) not in EXCLUDED]


def ssh(host, cmd, timeout=900):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', host, cmd], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def main():
    OUT.mkdir(parents=True, exist_ok=True); todo = dict(WANT)
    while todo:
        for model, hosts in list(todo.items()):
            try:
                cands = [c for c in idle(hosts) if model in c[1]['name']]
                if not cands:
                    continue
                host, g = cands[0]
                if not any(h == host and x['uuid'] == g['uuid'] for h, x in idle([host])):
                    continue
                m = ssh(host, f'CUDA_VISIBLE_DEVICES={g["uuid"]} {SIM} {ROOT}/robotwinprobe/egl_enum.py', 120)
                egl = [d['egl'] for d in json.loads(m.stdout)['devices'] if d['uuid'] == g['uuid']]
                assert len(egl) == 1, m.stdout
                e = egl[0]
                log('run', model, host, 'phys', g['index'], g['uuid'], 'egl', e)
                r = ssh(host, f'cd {ROOT}/robotwinprobe && WAMATTACK_PROJECT_ROOT={ROOT} CUDA_VISIBLE_DEVICES={e} MUJOCO_EGL_DEVICE_ID={e} '
                              f'LIBERO_CONFIG_PATH={ROOT}/robotwinprobe/out/libero_crossarch5_20261105/libero_config '
                              f'{SIM} -u egl_render_determinism.py {OUT}')
                (OUT / f'{host}_phys{g["index"]}.log').write_text(r.stdout + '\n--- stderr ---\n' + r.stderr[-6000:] + f'\nexit {r.returncode}\n')
                log('done', model, host, 'exit', r.returncode)
                if (OUT / f'{host}_egl{e}.json').exists():
                    del todo[model]
            except Exception as err:
                log('error', model, repr(err))
        if todo:
            time.sleep(120)
    log('all done')


if __name__ == '__main__':
    main()
