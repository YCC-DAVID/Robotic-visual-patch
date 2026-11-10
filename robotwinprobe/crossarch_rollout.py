"""Unfiltered initial states; actual-render clean/Random/cross-model patches."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import traceback
import numpy as np
import bootstrap as B
from crossarch_common import DEFAULT,CAMERAS,WIRE,request,dump,prepare

def run(a):
    B.simulator_paths()
    from scene_smoke import make_task
    from render_geometry import Patch,patterns
    from PIL import Image
    p=prepare(a.root);group='pilot' if a.pilot else 'closedloop'
    dest=a.root/group/a.model/a.condition;dest.mkdir(parents=True,exist_ok=True)
    result=dest/f'seed{a.seed}.json'
    if result.exists():
        old=json.loads(result.read_text())
        if old.get('error'):raise RuntimeError('Unresolved recorded error '+str(result))
        return
    task=None;start=time.monotonic();random.seed(a.seed)
    try:
        task,_,_=make_task(p['task'],a.seed,dest)
        prompt='Place the '+('bowl' if task.actor_name=='002_bowl' else 'cup')+' on the plate.'
        task.set_instruction(prompt);initial=task.get_obs()
        h=hashlib.sha256(initial['joint_action']['vector'].astype(np.float32).tobytes())
        for n in CAMERAS:h.update(initial['observation'][n]['rgb'].tobytes())
        initial_hash=h.hexdigest()
        reference=a.root/group/a.model/'clean'/f'seed{a.seed}.json'
        if reference.exists():assert json.loads(reference.read_text())['initial_clean_sha256']==initial_hash
        if a.condition!='clean':
            texture=patterns()['random'] if a.condition=='random' else np.asarray(Image.open(a.root/'patches'/a.condition/'texture_0500.png').convert('RGB'))
            patch=Patch(task,dest,center=p['patch']['center_m'],size=.10);patch.texture(texture)
        states=[];actions=[];frames=[]
        while task.take_action_cnt<task.step_lim and not task.eval_success:
            obs=task.get_obs();state=obs['joint_action']['vector'].astype(np.float32)
            raw={w:obs['observation'][n]['rgb'] for n,w in zip(CAMERAS,WIRE)}
            query=len(states);ns=20261006+a.seed*1000+query
            chunk=request(a.model,dict(command='infer',state=state,prompt=prompt,images=raw,seed=ns))['actions']
            chunk=np.asarray(chunk)
            assert chunk.shape==(p['evaluation']['horizon'][a.model],14) and np.isfinite(chunk).all(),chunk.shape
            states.append(state);actions.append(chunk);frames.append(raw[WIRE[0]].copy())
            for action in chunk[:p['evaluation']['replan'][a.model]]:
                task.take_action(action,action_type='qpos')
                if task.eval_success or task.take_action_cnt>=task.step_lim:break
        np.savez_compressed(dest/f'seed{a.seed}.npz',states14=states,actions=actions,head=frames,prompt=prompt)
        row=dict(seed=a.seed,model=a.model,condition=a.condition,success=bool(task.eval_success),
                 steps=task.take_action_cnt,queries=len(states),initial_clean_sha256=initial_hash,elapsed_s=time.monotonic()-start)
        dump(result,row);print('[closedloop]',row,flush=True)
    except Exception as e:
        dump(result,dict(seed=a.seed,error=repr(e),traceback=traceback.format_exc()));raise
    finally:
        if task is not None:task.close_env()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    ap.add_argument('--model',choices=['pi','wam'],required=True);ap.add_argument('--condition',required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--pilot',action='store_true');run(ap.parse_args())
