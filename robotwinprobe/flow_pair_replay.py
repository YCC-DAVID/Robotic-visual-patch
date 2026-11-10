"""Replay attacked actions exactly and toggle visual patch at each query."""
import argparse,json,random
from pathlib import Path
import numpy as np
from PIL import Image
from e_method_common import load,request,dump,CAMERAS,WIRE,sha

def run(root,seed):
    import bootstrap as B
    B.simulator_paths()
    from scene_smoke import make_task
    from render_geometry import Patch
    from stage_joint_rollout import check_prefix,robot_state,pose,digest
    p=load(root);cfg=p['flow_diagnostic'];source=Path(cfg['source']);name=cfg['condition']
    folder=source/'cohorts/development/episodes';stem=f'native_full_approach_{name}_s{seed}'
    episode=folder/(stem+'.json');row=json.loads(episode.read_text());artifact=episode.with_suffix('.npz')
    assert sha(artifact)==row['artifact_sha256']
    with np.load(artifact) as z:ref={k:z[k].copy() for k in z.files}
    dest=root/'replay'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    random.seed(seed);task=None;paired=[]
    try:
        task,_,_=make_task(p['task'],seed,dest);task.set_instruction(row['prompt'])
        texture=np.asarray(Image.open(source/'patches'/name/'texture_0500.png').convert('RGB'))
        patch=Patch(task,dest,center=p['patch']['center_m'],size=p['patch']['size_m']);patch.texture(texture)
        current=0
        for qi,step in enumerate(ref['query_steps']):
            while current<int(step):
                task.take_action(ref['executed_actions'][current],action_type='qpos');current+=1
                check_prefix(robot_state(task),pose(task),ref,current)
            obs=task.get_obs();raw_state=obs['joint_action']['vector'].copy();state=raw_state.astype(np.float32)
            assert np.array_equal(state,ref['states14'][qi])
            rgb=[digest(obs['observation'][n]['rgb']) for n in CAMERAS]
            assert rgb==ref['rgb_sha256'][qi].tolist(),('patched RGB mismatch',seed,qi)
            patch.visible(False);clean=task.get_obs();patch.visible(True)
            assert np.array_equal(clean['joint_action']['vector'],raw_state), 'Visual toggle changed raw joint state'
            responses=[]
            for frame in (clean,obs):
                raw={w:frame['observation'][n]['rgb'] for n,w in zip(CAMERAS,WIRE)}
                responses.append(request(dict(command='flow_trace',state=state,prompt=row['prompt'],images=raw,seed=int(ref['noise_seeds'][qi])),root))
            c,a=responses;error=float(np.max(np.abs(a['actions']-ref['actions'][qi])))
            assert error<1e-4,('reproduced action mismatch',error)
            tmp=dest/f'q{qi:03d}.tmp.npz';out=dest/f'q{qi:03d}.npz'
            np.savez_compressed(tmp,clean_x=c['x'],patch_x=a['x'],clean_v=c['v'],patch_v=a['v'],
                clean_actions=c['actions'],patch_actions=a['actions'],state14=state,step=int(step),
                stage=ref['query_phases'][qi],noise=int(ref['noise_seeds'][qi]),
                active_start=7 if ref['object_plate'][0,0,0]>0 else 0,
                clean_head=clean['observation'][CAMERAS[0]]['rgb'],patch_head=obs['observation'][CAMERAS[0]]['rgb'])
            tmp.replace(out);paired.append(dict(query=qi,step=int(step),native_action_max_error=error))
            dump(dest/'progress.json',dict(seed=seed,queries=len(paired),total=len(ref['query_steps'])))
        dump(dest/'complete.json',dict(seed=seed,queries=paired,source_sha256=sha(artifact),task_success=row['success'],exact_replay=True))
    finally:
        if task is not None:task.close_env()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.root.resolve(),a.seed)
