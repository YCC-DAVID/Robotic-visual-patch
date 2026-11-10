"""Matched first-flow and end-to-end ten-flow-step ImageWAM patch training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from crossarch_common import STAGES,dump,sha
from crossarch_capture import records
from crossarch_stage_common import DEFAULT as SOURCE
from wam_orthogonal_control import call

OBJECTIVES=('first_flow','final_action')


def teacher(backend,record,seed,cache,full):
    import torch
    from crossarch_proxy import images
    key=(record,int(seed))
    entry=cache.setdefault(key,{})
    if 'v' not in entry or (full and 'action' not in entry):
        with np.load(record) as z:state=z['state14'].copy();prompt=str(z['prompt'])
        with torch.no_grad():
            v,a=backend.response(backend.prepare(state,prompt,images(None,record)),seed,full=full)
        entry['v']=v.detach().float().cpu()
        if full:entry['action']=a.detach().float().cpu()
    return entry


def gradient(backend,req,cache):
    import torch
    from crossarch_proxy import images
    began=time.monotonic();objective=req['objective'];assert objective in OBJECTIVES
    record=req['record'];seed=int(req['seed']);full=objective=='final_action'
    clean=teacher(backend,record,seed,cache,full)
    with np.load(record) as z:state=z['state14'].copy();prompt=str(z['prompt'])
    q=torch.tensor(req['texture'],device='cuda',requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    v,a=backend.response(backend.prepare(state,prompt,images(q,record)),seed,full=full)
    first=(v[...,:14].float()-clean['v'].to('cuda')[...,:14]).square().mean()
    loss=(a[...,:14].float()-clean['action'].to('cuda')[...,:14]).square().mean() if full else first
    g,=torch.autograd.grad(loss,q)
    assert torch.isfinite(loss) and torch.isfinite(g).all() and float(g.norm())>0
    assert all(p.grad is None for p in backend.model.parameters())
    result=dict(loss=float(loss.detach()),first_L0=float(first.detach()),gradient=g.cpu().numpy(),
        gradient_norm=float(g.norm()),objective=objective,
        peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
    result['compute_elapsed_s']=time.monotonic()-began
    return result


def diagnostic(backend,req,cache):
    import torch
    from crossarch_proxy import images
    record=req['record'];seed=int(req['seed'])
    clean=teacher(backend,record,seed,cache,True)
    with np.load(record) as z:state=z['state14'].copy();prompt=str(z['prompt'])
    with torch.no_grad():
        if req['path']=='proxy':
            q=torch.tensor(req['texture'],device='cuda')
            prepared=backend.prepare(state,prompt,images(q,record))
        else:
            from crossarch_common import CAMERAS,WIRE
            with np.load(req['capture']) as z:raw={w:z[n+'_'+req['texture_name']].copy() for n,w in zip(CAMERAS,WIRE)}
            prepared=backend.raw_prepared(state,prompt,raw)
        v,a=backend.response(prepared,seed,full=True)
        dv=v.float().cpu()-clean['v'];da=a.float().cpu()-clean['action']
        action=backend.M.denorm(backend.proc,a[0]);base=backend.M.denorm(backend.proc,clean['action'][0])
        joints=[0,1,2,3,4,5,7,8,9,10,11,12]
        return dict(first_L0=float(dv.square().mean()),final_action_MSE=float(da.square().mean()),
            joint_action_RMSE_rad=float(np.sqrt(np.mean((action[:,joints]-base[:,joints])**2))),
            delta_v=dv.numpy(),delta_action=da.numpy())


def plan(root):
    protocol=json.loads((root/'protocol.json').read_text());cfg=protocol['patch_trial']
    pools={stage:[r for seed in protocol['data']['train_seeds'] for r in records(SOURCE,seed) if r['stage']==stage]
           for stage in STAGES}
    assert all(len(rows)==384 for rows in pools.values())
    rng=np.random.default_rng(cfg['seed'])
    draws=[[dict(record=(row:=pools[stage][int(rng.integers(len(pools[stage])))])['path'],
                 stage=stage,seed=int(rng.integers(1,2**30))) for stage in STAGES for _ in range(2)]
           for iteration in range(cfg['steps'])]
    path=root/'training_plan.json'
    if path.exists():assert json.loads(path.read_text())==draws
    else:dump(path,draws)
    return draws


def preflight(root):
    draws=plan(root);results=[]
    q=np.full((1,3,128,128),.5,np.float32)
    for objective in OBJECTIVES:
        for draw in draws[0][::2]:
            result=call(root,'patch_gradient',objective=objective,texture=q,
                        record=draw['record'],seed=draw['seed'])
            assert np.isfinite(result['gradient']).all() and result['gradient_norm']>0
            results.append({k:v for k,v in result.items() if k!='gradient'}|dict(stage=draw['stage']))
    dump(root/'audit/patch_gradient_preflight.json',dict(passed=True,results=results,
        scope='Native weights frozen; first-step and complete ten-step image-to-texture gradients finite and nonzero.'))


def train(root):
    protocol=json.loads((root/'protocol.json').read_text());cfg=protocol['patch_trial'];draws=plan(root)
    snapshot=root/'source_snapshots'/f'patch_trainer_{os.getpid()}.py'
    shutil.copy2(Path(__file__),snapshot)
    dump(root/'audit'/f'patch_training_launch_{os.getpid()}.json',dict(pid=os.getpid(),time=time.time(),
        script=str(snapshot),script_sha256=sha(snapshot),protocol_sha256=sha(root/'protocol.json')))
    preflight(root)
    for objective in OBJECTIVES:
        dest=root/'patches/wam'/objective;dest.mkdir(parents=True,exist_ok=True)
        checkpoint=dest/'training_state.npz';q=np.full((1,3,128,128),.5,np.float32)
        history=[];start=0;compute_s=0.
        if checkpoint.exists():
            with np.load(checkpoint) as z:
                q=z['texture'].copy();start=int(z['iteration']);history=json.loads(str(z['history']))
                compute_s=float(z['compute_s'])
        began=time.monotonic()
        for iteration in range(start,cfg['steps']):
            gradient=np.zeros_like(q);values=[]
            for draw in draws[iteration]:
                result=call(root,'patch_gradient',objective=objective,texture=q,
                            record=draw['record'],seed=draw['seed'])
                gradient+=result['gradient']/cfg['batch'];values.append(result['loss'])
                compute_s+=result['compute_elapsed_s']
            norm=float(np.linalg.norm(gradient));assert np.isfinite(gradient).all() and norm>0
            q=np.clip(q+cfg['lr']*np.sign(gradient),0,1)
            history.append(float(np.mean(values)))
            temporary=checkpoint.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,texture=q,iteration=iteration+1,history=json.dumps(history),compute_s=compute_s)
            temporary.replace(checkpoint)
            if iteration==0 or (iteration+1)%25==0:
                Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(dest/f'texture_{iteration+1:04d}.png')
            dump(dest/'progress.json',dict(objective=objective,iteration=iteration+1,total=cfg['steps'],
                complete=iteration+1==cfg['steps'],loss=history[-1],gradient_norm=norm,
                compute_s=compute_s,elapsed_s_this_run=time.monotonic()-began,
                loss_name='first_velocity_MSE' if objective=='first_flow' else 'final_normalized_action_MSE'))
            print('[flow patch]',objective,iteration+1,history[-1],flush=True)
        # A restart may find the final checkpoint saved immediately before its PNG/progress.
        assert len(history)==cfg['steps']
        final=dest/f'texture_{cfg["steps"]:04d}.png'
        rgb=np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)
        if final.exists():assert np.array_equal(np.asarray(Image.open(final).convert('RGB')),rgb)
        else:Image.fromarray(rgb).save(final)
        progress=dest/'progress.json'
        if not progress.exists() or json.loads(progress.read_text())['iteration']!=cfg['steps']:
            dump(progress,dict(objective=objective,iteration=cfg['steps'],total=cfg['steps'],complete=True,
                loss=history[-1],gradient_norm=None,compute_s=compute_s,elapsed_s_this_run=time.monotonic()-began,
                recovered_final_metadata=True,
                loss_name='first_velocity_MSE' if objective=='first_flow' else 'final_normalized_action_MSE'))
    dump(root/'patch_training_complete.json',dict(complete=True,objectives=list(OBJECTIVES)))


def patch_rollout(root,objective,seed):
    from types import SimpleNamespace
    import crossarch_stage_rollout as runner
    def request(model,payload):
        assert model=='wam' and payload['command']=='infer'
        return call(root,'infer_image',**{k:v for k,v in payload.items() if k!='command'})
    def condition(name):
        assert name in OBJECTIVES
        return dict(texture=name,stage=None,mode='full')
    runner.request=request;runner.condition=condition
    runner.run(SimpleNamespace(root=root,model='wam',condition=objective,seed=seed,smoke=False))


def capture(root,seed):
    import wam_flux_diagnosis as renderer
    from render_geometry import patterns
    def textures():
        bank={'random':patterns()['random']}
        bank.update({name:np.asarray(Image.open(root/'patches/wam'/name/'texture_0500.png').convert('RGB')) for name in OBJECTIVES})
        return bank
    renderer.texture_bank=textures
    renderer.capture(root,seed)


def validate(root):
    protocol=json.loads((root/'protocol.json').read_text())
    textures={'random':None}
    from render_geometry import patterns
    textures['random']=patterns()['random']
    textures.update({name:np.asarray(Image.open(root/'patches/wam'/name/'texture_0500.png').convert('RGB')) for name in OBJECTIVES})
    dest=root/'patch_validation';dest.mkdir(exist_ok=True)
    for record in protocol['diagnosis']['records']:
        for name,rgb in textures.items():
            q=rgb.transpose(2,0,1)[None].astype(np.float32)/255
            for path in ['proxy','render']:
                for seed in protocol['diagnosis']['action_noise_seeds']:
                    output=dest/(record['id']+'_'+name+'_'+path+'_'+str(seed)+'.json')
                    if output.exists():continue
                    result=call(root,'patch_diagnostic',record=record['path'],seed=seed,path=path,
                        texture=q,capture=str(root/'captures'/(record['id']+'.npz')),texture_name=name)
                    np.savez_compressed(output.with_suffix('.npz'),delta_v=result.pop('delta_v'),delta_action=result.pop('delta_action'))
                    dump(output,dict(record=record['id'],stage=record['stage'],texture=name,path=path,noise_seed=seed,**result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','patch_rollout','capture','validate'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--objective',choices=OBJECTIVES)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.mode=='train':train(args.root)
    elif args.mode=='patch_rollout':patch_rollout(args.root,args.objective,args.seed)
    elif args.mode=='capture':capture(args.root,args.seed)
    else:validate(args.root)
