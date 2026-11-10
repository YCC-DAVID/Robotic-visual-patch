"""Three independent textures, matched whole-trajectory draws and fixed scaling."""
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from crossarch_common import DEFAULT,STAGES,request,dump,prepare
from crossarch_capture import records

def train(root=DEFAULT):
    p=prepare(root)
    for m in ('pi','wam'):assert json.loads((root/f'audit/{m}_proxy.json').read_text())['passed']
    rs=[r for seed in p['data']['train_seeds'] for r in records(root,seed)]
    assert all(Path(r['path']).exists() for r in rs)
    stage_records={s:[r for r in rs if r['stage']==s] for s in STAGES}
    rng=np.random.default_rng(p['patch']['seed'])
    plan=[[dict(record=stage_records[s][int(rng.integers(len(stage_records[s])))]['path'],
                seed=int(rng.integers(1,2**30))) for s in STAGES for _ in range(2)] for _ in range(500)]
    path=root/'training_draws.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:dump(path,plan)
    scale_path=root/'random_scales.json'
    if not scale_path.exists():
        from render_geometry import patterns
        q=patterns()['random'].transpose(2,0,1)[None].astype(np.float32)/255
        scales={};values={}
        # Training-only, fixed 32 state/noise draws, eight per stage. No validation tuning.
        for model in ('pi','wam'):
            values[model]=[request(model,dict(command='gradient',texture=q,**r))['L0'] for batch in plan[:4] for r in batch]
            scales[model]=float(np.mean(values[model]));assert scales[model]>1e-10
        dump(scale_path,dict(scales=scales,values=values,draws='first four immutable training batches'))
    scales=json.loads(scale_path.read_text())['scales']
    q={k:np.full((1,3,128,128),.5,np.float32) for k in ('pi','wam','joint')};history=[];start=0
    checkpoint=root/'training_state.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            start=int(z['iteration']);q={k:z[k].copy() for k in q};history=json.loads(str(z['history']))
    began=time.monotonic()
    for it in range(start,500):
        row=dict(iteration=it+1,loss={})
        for objective in q:
            models=['pi','wam'] if objective=='joint' else [objective]
            gradient=np.zeros_like(q[objective]);values={m:[] for m in models}
            for draw in plan[it]:
                for m in models:
                    result=request(m,dict(command='gradient',texture=q[objective],**draw))
                    # Constant per-model normalization does not change a single-model sign update.
                    gradient+=result['gradient']/(8*len(models)*scales[m]);values[m].append(result['L0'])
            assert np.isfinite(gradient).all()
            q[objective]=np.clip(q[objective]+.01*np.sign(gradient),0,1)
            row['loss'][objective]={m:float(np.mean(v)) for m,v in values.items()}
        history.append(row)
        tmp=checkpoint.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,iteration=it+1,history=json.dumps(history),**q);tmp.replace(checkpoint)
        if it==0 or (it+1)%25==0:
            for k,texture in q.items():
                dest=root/'patches'/k;dest.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.rint(texture[0].transpose(1,2,0)*255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png')
        dump(root/'training_progress.json',dict(**row,complete=it+1==500,elapsed_s_this_run=time.monotonic()-began))
        print('[train]',row,flush=True)

if __name__=='__main__':train()
