"""Recalibrate each physical EOT condition on exactly replayed clean trajectories."""
import argparse, json, random, time
import numpy as np
import bootstrap as B
B.simulator_paths()
from scene_smoke import make_task
from render_geometry import Patch, CAMERAS, patterns
from eot_conditions import Lighting, conditions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--seeds', default='10100,10101,10102,10103,10104,10105,10106,10107,10110,10111')
    ap.add_argument('--conditions', default='all')
    ap.add_argument('--max-queries', type=int, default=0)
    args = ap.parse_args()
    root = B.Path(args.root).resolve(); refroot = B.Path(args.reference).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cs = conditions()
    if args.conditions != 'all': cs = {k: cs[k] for k in args.conditions.split(',')}
    textures = {k: v for k, v in patterns().items() if k != 'uv'}
    textures.update(gray64=np.full((128,128,3),64,np.uint8), gray192=np.full((128,128,3),192,np.uint8))
    started = time.monotonic()
    for seed in map(int, args.seeds.split(',')):
        pending = {k:c for k,c in cs.items() if not (root/k/f'seed{seed}.npz').exists()}
        if not pending: continue
        work = root/f'assets_{seed}'; work.mkdir(exist_ok=True)
        random.seed(seed); task, _, _ = make_task('place_container_plate', seed, work)
        light = Lighting(task)
        rows = {k: [] for k in pending}
        try:
            with np.load(refroot/'clean'/f'seed{seed}.npz') as ref:
                prompt = 'Place the '+('bowl' if task.actor_name == '002_bowl' else 'cup')+' on the plate.'
                for query, actions in enumerate(ref['actions']):
                    if args.max_queries and query >= args.max_queries: break
                    light.apply([1,1,1]); nominal = task.get_obs()
                    state = nominal['joint_action']['vector'].astype(np.float32)
                    assert np.array_equal(state, ref['states14'][query]), (seed, query, 'state')
                    for n in CAMERAS:
                        assert np.array_equal(nominal['observation'][n]['rgb'], ref[n][query]), (seed, query, n, 'replay')
                    for key, c in pending.items():
                        light.apply(c['light_gain']); clean = task.get_obs()
                        row = {'state14': state.copy(), 'query': query, 'step': task.take_action_cnt}
                        for n in CAMERAS: row[n+'_clean'] = clean['observation'][n]['rgb'].copy()
                        patch = Patch(task, work, center=c['center'], size=c['size_m'])
                        for label, rgb in textures.items():
                            patch.texture(rgb); obs = task.get_obs()
                            assert np.array_equal(state, obs['joint_action']['vector'].astype(np.float32))
                            for n in CAMERAS: row[n+'_'+label] = obs['observation'][n]['rgb'].copy()
                        for n, geom in patch.geometry(task).items():
                            for k,v in geom.items(): row[n+'_'+k] = v
                        task.scene.remove_actor(patch.actor)
                        restored = task.get_obs()
                        for n in CAMERAS:
                            assert np.array_equal(row[n+'_clean'], restored['observation'][n]['rgb']), (seed, query, key, n, 'restore')
                        rows[key].append(row)
                    light.apply([1,1,1])
                    for action in actions:
                        task.take_action(action, action_type='qpos')
                        if task.eval_success or task.take_action_cnt >= task.step_lim: break
                    progress = {'stage':'capture','seed':seed,'query':query,'conditions':len(pending),
                                'elapsed_s':time.monotonic()-started}
                    (root/f'progress_{args.seeds.split(",")[0]}.json').write_text(json.dumps(progress, indent=2)+'\n')
                    print('[capture]',progress,flush=True)
            for key, rs in rows.items():
                d = root/key; d.mkdir(exist_ok=True)
                output = d/f'seed{seed}.npz'
                tmp = d/f'seed{seed}.tmp.npz'
                np.savez_compressed(tmp, **{k:np.array([r[k] for r in rs]) for k in rs[0]},
                    seed=np.array(seed), prompt=np.array(prompt), center_m=np.array(pending[key]['center']),
                    size_m=np.array(.20), light_gain=np.array(pending[key]['light_gain']))
                tmp.replace(output)
            print('[seed complete]',seed,flush=True)
        finally: task.close_env()


if __name__ == '__main__': main()
