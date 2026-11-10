"""Matched early/whole-stage training; global Random scale and exact log derivative."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from stage_joint_common import DEFAULT,TEXTURES,STAGES,load,training_plan,request,dump,sha


def objective_weight(kind,l0,scale):
    if kind=='joint_balanced':
        return float(np.log1p(l0/scale)),1/(scale+l0)
    return l0/scale,1/scale


def train(root):
    p=load(root)
    plan=json.loads((root/'training_draws.json').read_text())
    assert plan==training_plan(root),'training draws changed'
    for batch in plan['joint']+plan['early']:
        assert len(batch)==8
        assert all(Path(r['record']).exists() for r in batch),'calibration incomplete'
    from render_geometry import patterns
    random_q=patterns()['random'].transpose(2,0,1)[None].astype(np.float32)/255
    scale_path=root/'random_scale.json'
    if not scale_path.exists():
        draws=[r for batch in plan['joint'][:4] for r in batch]
        values=[request(dict(command='gradient',texture=random_q,**r),root)['L0'] for r in draws]
        scale=float(np.mean(values));assert np.isfinite(scale) and scale>1e-12
        dump(scale_path,dict(scale=scale,values=values,draws=draws,
            note='one fixed training-only global Random mean; no validation tuning'))
    scale=json.loads(scale_path.read_text())['scale']
    checkpoint=root/'training_state.npz'
    q={name:np.full((1,3,128,128),.5,np.float32) for name in TEXTURES}
    history=[];start=0
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            assert str(z['protocol_sha256'])==sha(root/'protocol.json')
            start=int(z['iteration']);q={name:z[name].copy() for name in TEXTURES}
            history=json.loads(str(z['history']))
    began=time.monotonic()
    for it in range(start,p['patch']['steps']):
        row=dict(iteration=it+1,variants={})
        for kind in TEXTURES:
            draws=plan['early' if kind=='early' else 'joint'][it]
            gradient=np.zeros_like(q[kind]);values=[];objectives=[];stage_values={s:[] for s in STAGES}
            for draw in draws:
                r=request(dict(command='gradient',texture=q[kind],**draw),root)
                objective,weight=objective_weight(kind,r['L0'],scale)
                gradient+=np.asarray(r['gradient'])*(weight/len(draws))
                values.append(r['L0']);objectives.append(objective);stage_values[draw['stage']].append(r['L0'])
            assert np.isfinite(gradient).all() and np.linalg.norm(gradient)>0
            q[kind]=np.clip(q[kind]+p['patch']['lr']*np.sign(gradient),0,1)
            row['variants'][kind]=dict(L0=float(np.mean(values)),objective=float(np.mean(objectives)),
                per_stage_L0={s:float(np.mean(v)) for s,v in stage_values.items() if v},gradient_norm=float(np.linalg.norm(gradient)))
        history.append(row)
        tmp=checkpoint.with_suffix('.tmp.npz')
        np.savez_compressed(tmp,iteration=it+1,history=json.dumps(history),protocol_sha256=sha(root/'protocol.json'),**q)
        tmp.replace(checkpoint)
        if it==0 or (it+1)%25==0 or it+1==p['patch']['steps']:
            for name in TEXTURES:
                dest=root/'patches'/name;dest.mkdir(parents=True,exist_ok=True)
                Image.fromarray(np.rint(q[name][0].transpose(1,2,0)*255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png')
        dump(root/'training_progress.json',dict(**row,complete=it+1==p['patch']['steps'],elapsed_s_this_run=time.monotonic()-began))
        print('[train]',row,flush=True)
    dump(root/'texture_manifest.json',dict(protocol_sha256=sha(root/'protocol.json'),
        textures={name:sha(root/'patches'/name/'texture_0500.png') for name in TEXTURES},scale=scale,
        matched_joint_draws=True,validation_checkpoint_selection=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    train(ap.parse_args().root.resolve())
