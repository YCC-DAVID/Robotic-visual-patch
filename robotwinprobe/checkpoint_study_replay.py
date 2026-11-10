"""Verify recorded absolute expert qpos labels through the policy action interface."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import bootstrap as B
import checkpoint_study_common as S
B.simulator_paths()


def main(args):
    from scene_smoke import make_task
    rows=[]
    for seed in map(int,args.seeds.split(',')):
        random.seed(seed);dest=args.root/'expert_replay'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        with np.load(args.root/'expert'/f'seed{seed}'/'demonstration.npz') as z:actions=z['state14'].copy()
        task=None
        try:
            task,_,_=make_task('place_container_plate',seed,dest)
            initial=task.get_obs()['joint_action']['vector'].astype(np.float32)
            assert np.array_equal(initial,actions[0])
            for action in actions:
                task.take_action(action,action_type='qpos')
                if task.eval_success or task.take_action_cnt>=task.step_lim:break
            rows.append(dict(seed=seed,success=bool(task.eval_success),steps=task.take_action_cnt,recorded_frames=len(actions)))
            print('[replay]',rows[-1],flush=True)
        finally:
            if task is not None:task.close_env()
    S.dump(args.root/'expert_replay/audit.json',dict(passed=all(r['success'] for r in rows),rows=rows))
    assert all(r['success'] for r in rows),'Expert action cadence/encoding did not replay successfully'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=S.DEFAULT)
    ap.add_argument('--seeds',default='20000,20001,20002');main(ap.parse_args())
