"""4000 continuous physical-position draws, same batches/noise/budget at every radius."""
import argparse, dataclasses, hashlib, json, time
from pathlib import Path
import numpy as np
import bootstrap as B
B.model_paths()
import render_model as M
from radius_plan import records, dump
from radius_proxy import CaptureProxy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--radius',type=int,choices=[0,1,3,5],required=True)
    ap.add_argument('--iters',type=int,default=500);ap.add_argument('--out')
    args=ap.parse_args();root=Path(args.root).resolve()
    dest=Path(args.out).resolve() if args.out else root/f'train_r{args.radius:02d}'
    dest.mkdir(parents=True,exist_ok=True)
    import torch
    from PIL import Image
    with np.load(root/'plan.npz') as p:
        batches=p['batches'].copy();val=p['validation_frames'].copy()
    train=records(root,args.radius,'train') if args.radius else None
    validation=records(root,args.radius,'validation') if args.radius else None
    if train:
        for r in train[:args.iters*8]+validation:assert Path(r['path']).exists(),r['path']
    config={'radius_cm':args.radius,'iters':args.iters,'batch':8,'lr':.01,
        'plan_sha256':hashlib.sha256((root/'plan.npz').read_bytes()).hexdigest(),
        'initialization':'gray0.5','loss':'squared first t=1 velocity difference, 32x32',
        'shared_texture':True,'views':'all three','light':'nominal fixed',
        'sampling':'one independent area-uniform disk draw per gradient frame; exact per-draw real calibration'}
    if (dest/'config.json').exists():assert json.loads((dest/'config.json').read_text())==config
    else:dump(dest/'config.json',config)
    torch.manual_seed(20261007);policy=M.load();model=policy._model;device=policy._pytorch_device
    proxy=CaptureProxy(device)
    noise_np=np.random.RandomState(20261007).standard_normal((32,32)).astype(np.float32)
    noise=torch.tensor(noise_np,device=device)[None]
    obs=[];clean=[]
    with torch.no_grad():
        for i,f in enumerate(proxy.reference.frames):
            o=M.observation(policy,f['state'],f['prompt'],f['raw']);obs.append(o)
            clean.append(M.velocity(model,o,noise).detach())
            if i%10==0:print('[cache]',i,flush=True)
    q=torch.full((1,3,128,128),.5,device=device,requires_grad=True)
    history=[];start=0
    checkpoints=sorted(dest.glob('checkpoint_*.npz'))
    if checkpoints:
        with np.load(checkpoints[-1]) as d:
            start=int(d['iteration']);history=d['history'].tolist()
            with torch.no_grad():q.copy_(torch.tensor(d['texture'],device=device)[None])
    def loss(frame,record):
        o=dataclasses.replace(obs[frame],images=proxy.images(q,frame,record))
        return (M.velocity(model,o,noise)-clean[frame]).square().sum()
    began=time.monotonic()
    for it in range(start,args.iters):
        q.grad=None;values=[]
        for j,f in enumerate(batches[it]):
            record=train[it*8+j] if train else None
            v=loss(int(f),record);values.append(float(v.detach()));(v/8).backward()
        assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.norm()>0
        with torch.no_grad():q.add_(.01*q.grad.sign()).clamp_(0,1)
        history.append(float(np.mean(values)))
        if it==0 or (it+1)%25==0 or it+1==args.iters:
            with torch.no_grad():
                nominal=[float(loss(int(f),None)) for f in val]
                transformed=[float(loss(int(f),r)) for f,r in zip(val,validation)] if validation else nominal
                texture=q.detach().cpu().numpy()[0]
            Image.fromarray(np.rint(texture.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png')
            target=dest/f'checkpoint_{it+1:04d}.npz';tmp=target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,iteration=it+1,texture=texture,history=history,
                batches=batches[:it+1],noise=noise_np,nominal_validation=nominal,transformed_validation=transformed)
            tmp.replace(target)
            progress={'complete':it+1==args.iters,'iteration':it+1,'radius_cm':args.radius,
                'train_loss':history[-1],'nominal_validation_mean':float(np.mean(nominal)),
                'transformed_validation_mean':float(np.mean(transformed)),'elapsed_s_this_run':time.monotonic()-began}
            dump(dest/'progress.json',progress);print('[train]',progress,flush=True)


if __name__=='__main__':main()
