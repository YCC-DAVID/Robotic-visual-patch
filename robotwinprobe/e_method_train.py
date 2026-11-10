"""Matched early-state objective ablations with resumable final checkpoints."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from e_method_common import DEFAULT,ALL_OBJECTIVES,REFINE_OBJECTIVES,draw_plan,dump,load,request,sha
from e_method_groups import GROUP_METHODS,group_draw_plan,merge_gradients
from e_method_onpolicy import ONPOLICY_METHODS,continuation_draw_plan,parent_texture,parent_state_path
from e_method_coverage import COVERAGE_METHODS,coverage_draw_plan


def bandlimit(q, cutoff):
    f=np.fft.fftfreq(q.shape[-1])*q.shape[-1]
    mask=f[:,None]**2+f[None,:]**2 <= cutoff**2
    y=np.fft.ifft2(np.fft.fft2(q,axes=(-2,-1))*mask,axes=(-2,-1)).real
    # Global affine rescaling preserves spectral support; pointwise clipping does not.
    lo=min(0.,float(y.min()));hi=max(1.,float(y.max()))
    return ((y-lo)/(hi-lo)).astype(np.float32)


def train(root,name,kind,training_seed,cutoff=None):
    assert kind in ALL_OBJECTIVES
    protocol=load(root);settings=protocol['method_study']
    dest=root/'training'/name;dest.mkdir(parents=True,exist_ok=True)
    method_name=name.split('_replica',1)[0]
    grouped=method_name in GROUP_METHODS
    continuation=method_name in (*ONPOLICY_METHODS,*COVERAGE_METHODS)
    coverage=method_name in COVERAGE_METHODS
    draws=(coverage_draw_plan(root,training_seed) if coverage else
           continuation_draw_plan(root,training_seed,method_name) if continuation else
           group_draw_plan(root,training_seed) if grouped else draw_plan(root,training_seed))
    manifest=dict(kind=kind,training_seed=training_seed,protocol_sha256=sha(root/'protocol.json'),
        collection_sha256=sha(root/'collection_manifest.json'),steps=settings['steps'],batch=settings['batch'])
    if kind in REFINE_OBJECTIVES:
        manifest['objective_protocol_sha256']=sha(root/'refinement/protocol.json')
    if grouped:
        assert kind=='velocity14'
        manifest.update(group_method=method_name,objective_protocol_sha256=sha(root/'group_round/protocol.json'))
    if continuation:
        assert kind=='velocity14'
        manifest.update(continuation_method=method_name,objective_protocol_sha256=sha(root/('coverage_round' if coverage else 'onpolicy_round')/'protocol.json'),
            attacked_collection_sha256=sha(root/'onpolicy_round/collection_manifest.json'),
            initial_state_sha256=sha(parent_state_path(root,name)))
    if cutoff is not None:manifest['radial_cutoff_cycles']=cutoff
    config=dest/'config.json'
    if config.exists():assert json.loads(config.read_text())==manifest
    else:dump(config,manifest)
    draw_path=dest/'draws.json'
    if draw_path.exists():assert json.loads(draw_path.read_text())==draws
    else:dump(draw_path,draws)
    q=parent_texture(root,name) if continuation else np.full((1,3,128,128),.5,np.float32)
    history=[];start=0
    checkpoint=dest/'state.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            assert str(z['protocol_sha256'])==manifest['protocol_sha256']
            q=z['texture'].copy();start=int(z['iteration']);history=json.loads(str(z['history']))
    began=time.monotonic()
    for it in range(start,settings['steps']):
        gradient=np.zeros_like(q);responses=[];group_gradients={}
        for draw in draws[it]:
            r=request(dict(command='method_gradient',kind=kind,texture=q,**draw),root)
            gradient+=r['gradient']/settings['batch'];responses.append({k:v for k,v in r.items() if k!='gradient'})
            if grouped:
                group_gradients.setdefault(draw['group'],np.zeros_like(q))
                group_gradients[draw['group']]+=r['gradient']/2
        group_diagnostics=None
        if grouped:
            gradient,group_diagnostics=merge_gradients(group_gradients,project=method_name=='velocity_pcgrad',seed=training_seed+it)
        assert np.isfinite(gradient).all() and np.linalg.norm(gradient)>0
        candidate=q+settings['lr']*np.sign(gradient)
        q=bandlimit(candidate,cutoff) if cutoff is not None else np.clip(candidate,0,1)
        row=dict(iteration=it+1,total=settings['steps'],kind=kind,name=name,training_seed=training_seed,
            loss=float(np.mean([r['loss'] for r in responses])),gradient_norm=float(np.linalg.norm(gradient)),
            first14_mse=float(np.mean([r['first14_mse'] for r in responses])),
            active_arm6_mse=float(np.mean([r['active_arm6_mse'] for r in responses])),
            active_gripper_mse=float(np.mean([r['active_gripper_mse'] for r in responses])),
            action14_mse=float(np.mean([r['action14_mse'] for r in responses])) if kind.startswith('action') else None,
            complete=it+1==settings['steps'],elapsed_s_this_run=time.monotonic()-began,started_iteration=start)
        if group_diagnostics:row['group_diagnostics']=group_diagnostics
        history.append(row)
        temp=checkpoint.with_suffix('.tmp.npz')
        np.savez_compressed(temp,texture=q,iteration=it+1,history=json.dumps(history),protocol_sha256=manifest['protocol_sha256'])
        temp.replace(checkpoint)
        if it==0 or (it+1)%25==0:
            images=root/'patches'/name;images.mkdir(parents=True,exist_ok=True)
            Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(images/f'texture_{it+1:04d}.png')
        dump(dest/'progress.json',row);print('[train]',row,flush=True)
    dump(dest/'complete.json',dict(**manifest,texture_sha256=sha(root/'patches'/name/'texture_0500.png'),complete=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--name',required=True);parser.add_argument('--kind',choices=ALL_OBJECTIVES,required=True)
    parser.add_argument('--training-seed',type=int,required=True)
    parser.add_argument('--cutoff',type=float)
    args=parser.parse_args();train(args.root.resolve(),args.name,args.kind,args.training_seed,args.cutoff)
