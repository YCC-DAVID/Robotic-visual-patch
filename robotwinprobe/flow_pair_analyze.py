"""Flow integration cancellation and action-target direction; descriptive, not causal."""
import json
from pathlib import Path
import numpy as np
from e_method_common import dump

def coherence(vectors):
    den=np.linalg.norm(vectors,axis=-1).sum()
    return float(np.linalg.norm(vectors.sum(0))/den) if den>1e-12 else None

def cos(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a.ravel(),b.ravel())/den) if den>1e-12 else None

def analyze(root):
    rows=[];episodes=[]
    for dest in sorted((root/'replay').glob('seed*')):
        if not (dest/'complete.json').exists():continue
        meta=json.loads((dest/'complete.json').read_text());directions=[];query_rows=[]
        for path in sorted(dest.glob('q[0-9][0-9][0-9].npz')):
            with np.load(path) as z:
                dx=z['patch_x']-z['clean_x'];dv=z['patch_v']-z['clean_v'];da=z['patch_actions']-z['clean_actions'];side=int(z['active_start']);sl=slice(side,side+6)
                # Along two separate flow paths, these increments telescope exactly.
                error=float(np.max(np.abs(dx[-1]+.1*dv.sum(0))));assert error<2e-5,error
                steps=(-.1*dv[...,sl]).reshape(10,-1);arm=da[:,sl]
                direction=arm.mean(0);directions.append(direction)
                r=dict(seed=meta['seed'],task_success=meta['task_success'],query=int(path.stem[1:]),stage=str(z['stage']),step=int(z['step']),
                    flow_arm_rms_by_iteration=np.sqrt(np.mean(dx[...,sl]**2,axis=(1,2,3))).tolist(),
                    velocity_arm_rms_by_iteration=np.sqrt(np.mean(dv[...,sl]**2,axis=(1,2,3))).tolist(),
                    flow_increment_coherence=coherence(steps),flow_telescoping_error=error,
                    final_action_arm_rms=float(np.sqrt(np.mean(arm**2))),
                    gripper_mean_signed=float(da[:,side+6].mean()),gripper_rms=float(np.sqrt(np.mean(da[:,side+6]**2))),
                    within_chunk_direction_coherence=coherence(arm),
                    next_query_mean_direction_cos=None)
                query_rows.append(r)
        for i in range(1,len(query_rows)):query_rows[i]['previous_query_mean_direction_cos']=cos(directions[i-1],directions[i])
        rows.extend(query_rows);episodes.append(dict(seed=meta['seed'],task_success=meta['task_success'],query_count=len(query_rows),
            cross_query_mean_target_coherence=coherence(np.array(directions)),
            scope='joint-position target offsets across different states; not a physical displacement sum'))
    dump(root/'analysis.json',dict(episodes=episodes,queries=rows,
        interpretation={'flow':'0 means opposing integration increments cancel; 1 means aligned increments. dt=-0.1 included.',
                        'cross_query':'descriptive direction consistency in joint coordinates; targets are qpos, not motion increments; no claim of physical additive causation',
                        'counterfactual':'clean means no-patch render at the attacked state, not a separate clean-rollout state'}))
    return episodes
