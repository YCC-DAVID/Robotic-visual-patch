"""Closed-loop perturbation diagnostics from stored E development episodes (CPU only, no new rollouts).

Per development seed whose clean episode succeeds, compare each patch condition with its paired clean episode:
- first-query action-chunk deviation (identical state and noise, so it is the pure image effect);
- joint-state deviation over time (accumulates vs. pulled back), and its direction persistence;
- failure stage: no grasp / grasp without lift / lifted then dropped / carried but not placed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EP = ROOT/'out/e_method_20261027/cohorts/development/episodes'
CONDITIONS = ('random', 'legacy', 'velocity14', 'action14', 'velocity_gripper', 'velocity_onpolicy_mix')
CHECK = (32, 64, 96, 128, 192, 256, 320)


def load(cond, seed):
    p = EP/f'native_full_approach_{cond}_s{seed}'
    if not p.with_suffix('.json').exists(): return None
    j = json.loads(p.with_suffix('.json').read_text())
    if j['status'] != 'complete': return None
    z = np.load(p.with_suffix('.npz'))
    return dict(success=bool(j['success']), steps=int(j['steps']), **{k: z[k] for k in
                ('actions', 'states14', 'step_states', 'step_object_plate', 'query_steps')})


def arm(clean):
    """Active arm = the one whose gripper closes in the clean episode."""
    g = clean['step_states'][:, [6, 13]].min(0)
    return 0 if g[0] < g[1] else 7


def events(ep, a):
    s, op = ep['step_states'], ep['step_object_plate']
    grip = s[:, a+6]; z = op[:, 0, 2]; z0 = z[0]
    closed = np.flatnonzero(grip < .5); lifted = np.flatnonzero(z > z0+.03)
    dist = np.linalg.norm(op[:, 0, :2]-op[:, 1, :2], axis=1)
    return dict(grasp=int(closed[0]) if closed.size else None, lift=int(lifted[0]) if lifted.size else None,
                dropped=bool(lifted.size and z[-1] < z0+.01 and dist[-1] > .06),
                final_dist=float(dist[-1]), near_plate=bool((dist < .06).any()))


def failure_mode(ev):
    if ev['grasp'] is None: return 'no_grasp'
    if ev['lift'] is None: return 'grasp_no_lift'
    if ev['dropped']: return 'dropped'
    if not ev['near_plate']: return 'carried_not_near_plate'
    return 'near_plate_not_placed'


def main(args):
    seeds = sorted({int(p.stem.rsplit('_s', 1)[1]) for p in EP.glob('native_full_approach_clean_s*.json')})
    out = {}
    for cond in CONDITIONS:
        rows = []
        for seed in seeds:
            c, x = load('clean', seed), load(cond, seed)
            if c is None or x is None or not c['success']: continue
            a = arm(c); idx = list(range(a, a+6))
            d0 = x['actions'][0]-c['actions'][0]                          # pure image effect, (32, 14)
            n = min(len(c['step_states']), len(x['step_states']))
            dq = x['step_states'][:n, idx]-c['step_states'][:n, idx]
            dev = np.linalg.norm(dq, axis=1)
            # direction persistence of the state deviation between consecutive 32-step query boundaries
            b = [t for t in range(32, n, 32)]
            cos = [float(dq[t1] @ dq[t0]/(np.linalg.norm(dq[t1])*np.linalg.norm(dq[t0])+1e-9)) for t0, t1 in zip(b, b[1:])
                   if np.linalg.norm(dq[t0]) > 1e-3 and np.linalg.norm(dq[t1]) > 1e-3]
            ev_c, ev_x = events(c, a), events(x, a)
            rows.append(dict(seed=seed, failed=not x['success'],
                             q0_arm_rms=float(np.sqrt((d0[:, idx]**2).mean())), q0_grip_mean=float(d0[:, a+6].mean()),
                             q0_chunk_dir_consistency=float(np.linalg.norm(d0[:, idx].mean(0))/(np.linalg.norm(d0[:, idx], axis=1).mean()+1e-9)),
                             dev={t: float(dev[t]) for t in CHECK if t < n}, peak=float(dev.max()), final=float(dev[n-1]),
                             dev_cos_consecutive=cos, mode=failure_mode(ev_x) if not x['success'] else 'success',
                             grasp_delay=(ev_x['grasp']-ev_c['grasp']) if ev_x['grasp'] is not None and ev_c['grasp'] is not None else None))
        out[cond] = rows
    (ROOT/'out/direction_closedloop.json').write_text(json.dumps(out, indent=1))
    for cond, rows in out.items():
        f = [r for r in rows if r['failed']]; s = [r for r in rows if not r['failed']]
        med = lambda rs, k: float(np.median([r[k] for r in rs])) if rs else float('nan')
        modes = {}
        for r in f: modes[r['mode']] = modes.get(r['mode'], 0)+1
        devs = {t: float(np.median([r['dev'][t] for r in rows if t in r['dev']])) for t in CHECK}
        cos = [c for r in rows for c in r['dev_cos_consecutive']]
        delays = [r['grasp_delay'] for r in rows if r['grasp_delay'] is not None]
        print(f'== {cond}: fail {len(f)}/{len(rows)}  modes {modes}')
        print(f'   q0 arm RMS fail/succ {med(f,"q0_arm_rms"):.3f}/{med(s,"q0_arm_rms"):.3f}  q0 grip mean {med(rows,"q0_grip_mean"):+.3f}'
              f'  chunk dir consistency {med(rows,"q0_chunk_dir_consistency"):.2f}')
        print('   median |dq| by step', {t: round(v, 3) for t, v in devs.items()},
              f' peak {med(rows,"peak"):.3f} final {med(rows,"final"):.3f}  final/peak fail {np.median([r["final"]/max(r["peak"],1e-9) for r in f]) if f else float("nan"):.2f} succ {np.median([r["final"]/max(r["peak"],1e-9) for r in s]):.2f}')
        print(f'   consecutive deviation cos median {np.median(cos):.2f} (n={len(cos)})  grasp delay median {np.median(delays) if delays else float("nan")} steps')


if __name__ == '__main__':
    main(argparse.ArgumentParser().parse_args())
