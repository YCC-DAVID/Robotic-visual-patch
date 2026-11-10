"""Joint pose/light rendering proxy validation and independent texture training."""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import bootstrap as B
from radius_plan import REFERENCE, dump
from table9_plan import records, sha, prepare


class Bank:
    def __init__(self, root, site, policy):
        import torch
        from render_proxy import Pool
        self.root, self.site, self.policy = root, site, policy
        self.device = policy._pytorch_device
        self.frames=[]
        for path in sorted((REFERENCE/'calibration').glob('seed*.npz')):
            with np.load(path) as z:
                for i,q in enumerate(z['query']):
                    self.frames.append(dict(seed=int(z['seed']),query=int(q),state=z['state14'][i].copy(),prompt=str(z['prompt'])))
        self.clean_cache={};self.obs_cache={}
        self.active=Pool.__new__(Pool)
        self.active.device=self.device;self.active.global_response=True
        self.active.offset=.5;self.active.blur_sigma=0.

    def clean(self,r):
        import torch
        from render_geometry import CAMERAS,WIRE
        import render_model as M
        key=(r['frame'],r['light'])
        if key not in self.clean_cache:
            with np.load(r['clean_path']) as z:
                assert int(z['frame'])==r['frame'] and str(z['light'])==r['light']
                f=self.frames[r['frame']]
                assert np.array_equal(z['state14'],f['state'])
                raw={w:z[n].copy() for n,w in zip(CAMERAS,WIRE)}
            self.clean_cache[key]=raw
            self.obs_cache[key]=M.observation(self.policy,f['state'],f['prompt'],raw)
        return self.obs_cache[key],self.clean_cache[key]

    def images(self,q,r):
        import torch
        from render_proxy import grid_for
        from digital_transform_common import rotated_grid
        from render_geometry import CAMERAS,WIRE
        _,raw=self.clean(r);views={}
        with np.load(r['path']) as z:
            assert int(z['frame'])==r['frame'] and str(z['light'])==r['light']
            assert np.array_equal(z['center_m'],r['center']) and float(z['yaw_deg'])==r['yaw_deg']
            for n,w in zip(CAMERAS,WIRE):
                view={'clean':torch.tensor(raw[w],device=self.device).permute(2,0,1)[None].float()/255}
                for key in ['black','gray64','gray','gray192','white']:
                    view[key]=torch.tensor(z[n+'_'+key],device=self.device).permute(2,0,1)[None].float()/255
                view['mask']=torch.tensor(z[n+'_mask'],device=self.device)[None,None].float()
                g=grid_for(z[n+'_intrinsic'],z[n+'_extrinsic'],z['center_m'],float(z['size_m']))
                view['grid']=torch.tensor(rotated_grid(g,r['yaw_deg']),device=self.device)[None]
                views[n]=view
        self.active.frames=[dict(views=views)]
        return self.active.images(q,0,'proxy')


def noises(root,device,probe=False):
    import torch
    with np.load(root/'plan.npz') as z:seeds=z['probe_noise_seeds' if probe else 'noise_seeds'].copy()
    return [torch.tensor(np.random.RandomState(int(s)).standard_normal((1,32,32)).astype(np.float32),device=device) for s in seeds]


def validate(root,site):
    B.model_paths()
    import torch
    from PIL import Image
    from openpi_client.image_tools import resize_with_pad
    from render_geometry import CAMERAS,WIRE,MODEL,patterns
    import render_model as M
    prepare(root)
    policy=M.load();model=policy._model;bank=Bank(root,site,policy)
    rs=records(root,site,'validation');noise=noises(root,bank.device,probe=True)[0]
    pats={k:patterns()[k] for k in ['random','checker']}
    pats['baseline']=np.asarray(Image.open(REFERENCE/'train_proxy_full/texture_0500.png').convert('RGB'))
    rows=[]
    with torch.no_grad():
        for r in rs:
            obs,_=bank.clean(r);vc=M.velocity(model,obs,noise);f=bank.frames[r['frame']]
            with np.load(r['path']) as z:
                for pattern,rgb in pats.items():
                    q=torch.tensor(rgb,device=bank.device).permute(2,0,1)[None].float()/255
                    images=bank.images(q,r)
                    raw={w:z[n+'_'+pattern] for n,w in zip(CAMERAS,WIRE)}
                    ro=M.observation(policy,f['state'],f['prompt'],raw)
                    vr=M.velocity(model,ro,noise);vp=M.velocity(model,dataclasses.replace(obs,images=images),noise)
                    row=dict(sample=r['id'],frame=r['frame'],pattern=pattern,light=r['light'],yaw_deg=r['yaw_deg'],
                        render_L0=float((vr-vc).square().sum()),proxy_L0=float((vp-vc).square().sum()),
                        proxy_render_velocity_SSE=float((vp-vr).square().sum()),pixels={})
                    for n,w,k in zip(CAMERAS,WIRE,MODEL):
                        target=torch.tensor(resize_with_pad(raw[w],224,224),device=bank.device).permute(2,0,1)[None].float()/255
                        delta=(images[k]+1)/2-target
                        mask=torch.tensor(z[n+'_mask'],device=bank.device)[None,None].float()
                        mask=torch.nn.functional.pad(torch.nn.functional.interpolate(mask,(168,224),mode='nearest'),(0,0,28,28)).bool().expand_as(delta)
                        row['pixels'][n]=dict(rmse=float(delta.square().mean().sqrt()),
                            mask_rmse=float(delta[mask].square().mean().sqrt()) if mask.any() else None)
                    rows.append(row)
            print('[validate]',site,r['id'],flush=True)
    r=rs[0];obs,_=bank.clean(r)
    with torch.no_grad():vc=M.velocity(model,obs,noise)
    q=torch.full((1,3,128,128),.5,device=bank.device,requires_grad=True)
    v=M.velocity(model,dataclasses.replace(obs,images=bank.images(q,r)),noise)
    (v-vc).square().sum().backward()
    gradient=dict(norm=float(q.grad.norm()),finite=bool(torch.isfinite(q.grad).all()))
    passed=gradient['finite'] and gradient['norm']>0 and all(
        p['rmse']<.06 and (p['mask_rmse'] is None or p['mask_rmse']<.15) for row in rows for p in row['pixels'].values())
    dump(root/'sites'/site/'proxy_validation.json',dict(passed=passed,rows=rows,gradient=gradient,
        gate='All-image RMSE < .06, visible mask RMSE < .15, finite nonzero gradient; no silent relaxation.',
        single_patch_per_scene=True))
    assert passed,'Rendering proxy validation failed'


def train(root,site):
    B.model_paths()
    import torch
    from PIL import Image
    import render_model as M
    protocol=prepare(root)
    assert json.loads((root/'sites'/site/'proxy_validation.json').read_text())['passed']
    assert json.loads((root/'sites'/site/'train_capture_complete.json').read_text())['complete']
    dest=root/'sites'/site/'train';dest.mkdir(parents=True,exist_ok=True)
    config=dict(site=site,protocol_sha256=sha(root/'protocol.json'),plan_sha256=sha(root/'plan.npz'),
        iters=500,batch=8,lr=.01,initialization='gray .5, independent parameter tensor',noise_bank=16,
        calibration_hashes={str(p.relative_to(root)):sha(p) for p in sorted((root/'sites'/site/'capture').glob('*/*.npz'))})
    path=dest/'config.json'
    if path.exists():assert json.loads(path.read_text())==config
    else:dump(path,config)
    torch.manual_seed(20261007);policy=M.load();model=policy._model;bank=Bank(root,site,policy)
    ns=noises(root,bank.device);probe_ns=noises(root,bank.device,probe=True)
    rs=records(root,site,'train');val=records(root,site,'validation')
    clean_velocity={}
    unique={}
    for r in rs+val:unique.setdefault((r['frame'],r['light']),r)
    with torch.no_grad():
        for key,r in unique.items():
            obs,_=bank.clean(r);state,pad,kv=M.prefix(model,obs)
            for ni,noise in enumerate(ns+probe_ns):
                clean_velocity[key+(ni,)]=model.denoise_step(state,pad,kv,noise,torch.ones(1,device=bank.device)).detach()
        del kv,state,pad
    q=torch.full((1,3,128,128),.5,device=bank.device,requires_grad=True)
    history=[];start=0
    checkpoints=sorted(dest.glob('checkpoint_*.npz'))
    if checkpoints:
        with np.load(checkpoints[-1]) as z:
            start=int(z['iteration']);history=z['history'].tolist()
            with torch.no_grad():q.copy_(torch.tensor(z['texture'],device=bank.device)[None])
    def loss(r,ni,noise):
        obs,_=bank.clean(r)
        v=M.velocity(model,dataclasses.replace(obs,images=bank.images(q,r)),noise)
        return (v-clean_velocity[r['frame'],r['light'],ni]).square().sum()
    began=time.monotonic()
    for it in range(start,500):
        q.grad=None;values=[]
        for r in rs[it*8:(it+1)*8]:
            value=loss(r,r['noise_id'],ns[r['noise_id']]);values.append(float(value.detach()));(value/8).backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        gradient_norm=float(q.grad.norm())
        with torch.no_grad():q.add_(.01*q.grad.sign()).clamp_(0,1)
        history.append(float(np.mean(values)))
        if it==0 or (it+1)%25==0:
            with torch.no_grad():
                # A disjoint action noise provides a consistent convergence diagnostic.
                validation=[float(loss(r,16,probe_ns[0])) for r in val]
                texture=q.detach().cpu().numpy()[0]
            Image.fromarray(np.rint(texture.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png')
            target=dest/f'checkpoint_{it+1:04d}.npz';tmp=target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,iteration=it+1,texture=texture,history=history,validation=validation);tmp.replace(target)
            row=dict(complete=it+1==500,iteration=it+1,site=site,train_batch_L0=history[-1],
                validation_nominal_L0=float(np.mean(validation[:len(val)//2])),validation_joint_L0=float(np.mean(validation[len(val)//2:])),
                gradient_norm=gradient_norm,elapsed_s_this_run=time.monotonic()-began)
            dump(dest/'progress.json',row);print('[train]',row,flush=True)
    assert (dest/'texture_0500.png').exists()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--site',required=True)
    parser.add_argument('--stage',choices=['validate','train'],required=True);args=parser.parse_args()
    (validate if args.stage=='validate' else train)(Path(args.root).resolve(),args.site)
