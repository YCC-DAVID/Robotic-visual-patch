"""Replay saved clean actions and capture paired textures without advancing physics."""
import argparse
import json
import random
import numpy as np
import bootstrap as B
B.simulator_paths()
from scene_smoke import make_task
from render_geometry import Patch, CAMERAS, patterns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', default='10100,10101,10102,10103,10104,10105,10106,10107,10110,10111')
    ap.add_argument('--textures', help='JSON mapping names to texture PNG paths; omit for calibration')
    args = ap.parse_args()
    from PIL import Image
    src, dest = B.Path(args.clean).resolve(), B.Path(args.out).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    if args.textures:
        textures = {k: np.array(Image.open(v).convert('RGB')) for k,v in json.loads(B.Path(args.textures).read_text()).items()}
    else:
        textures = patterns()
        textures.update({'gray64': np.full((128,128,3),64,np.uint8), 'gray192': np.full((128,128,3),192,np.uint8)})
    summary = []
    for seed in map(int, args.seeds.split(',')):
        output = dest/f'seed{seed}.npz'
        if output.exists(): raise RuntimeError(f'preserve {output}')
        ref = np.load(src/f'seed{seed}.npz')
        work = dest/f'assets_{seed}'; work.mkdir(exist_ok=True)
        random.seed(seed); task, _, _ = make_task('place_container_plate', seed, dest)
        rows = []
        try:
            prompt = 'Place the '+('bowl' if task.actor_name == '002_bowl' else 'cup')+' on the plate.'
            for query, actions in enumerate(ref['actions']):
                clean = task.get_obs()
                row = {'state14': clean['joint_action']['vector'].astype(np.float32), 'query': query, 'step': task.take_action_cnt}
                assert np.array_equal(row['state14'], ref['states14'][query]), (seed, query, 'state replay')
                for name in CAMERAS:
                    row[name+'_clean'] = clean['observation'][name]['rgb'].copy()
                    assert np.array_equal(row[name+'_clean'], ref[name][query]), (seed,query,name,'image replay')
                patch = Patch(task, work)
                for label, rgb in textures.items():
                    patch.texture(rgb); obs = task.get_obs()
                    assert np.array_equal(row['state14'], obs['joint_action']['vector'].astype(np.float32))
                    for name in CAMERAS: row[name+'_'+label] = obs['observation'][name]['rgb'].copy()
                geometry = patch.geometry(task)
                for name in CAMERAS:
                    for key, value in geometry[name].items(): row[name+'_'+key] = value
                task.scene.remove_actor(patch.actor)
                restored = task.get_obs()
                for name in CAMERAS:
                    assert np.array_equal(restored['observation'][name]['rgb'], row[name+'_clean']), (seed,query,name,'remove replay')
                rows.append(row)
                for action in actions:
                    task.take_action(action, action_type='qpos')
                    if task.eval_success or task.take_action_cnt >= task.step_lim: break
            np.savez_compressed(output, **{k: np.array([r[k] for r in rows]) for k in rows[0]},
                                prompt=np.array(prompt), seed=np.array(seed), center_m=patch.center, size_m=patch.size)
            summary.append({'seed': seed, 'frames':len(rows), 'success':bool(task.eval_success),
                            'pixels_by_view':{n:[int(r[n+'_mask'].sum()) for r in rows] for n in CAMERAS}})
            (dest/('summary_'+args.seeds.split(',')[0]+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
            print('[capture]',seed,len(rows),flush=True)
        finally: task.close_env()


if __name__ == '__main__': main()
