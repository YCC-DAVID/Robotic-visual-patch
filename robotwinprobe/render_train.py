"""Train one world texture jointly through three RoboTwin camera views."""
import argparse, dataclasses, hashlib, json, time
import numpy as np
import bootstrap as B
B.model_paths()
from render_geometry import CAMERAS, MODEL
from render_proxy import Pool
import render_model as M


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pool',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--mode',choices=['digital','proxy'],required=True)
    ap.add_argument('--band',type=int,choices=[32,128],default=128)
    ap.add_argument('--iters',type=int,default=500);ap.add_argument('--batch',type=int,default=8)
    ap.add_argument('--global-response',action='store_true')
    args=ap.parse_args()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    dest=B.Path(args.out).resolve();dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'config.json').exists()
    paths=sorted(B.Path(args.pool).glob('seed*.npz'))
    seeds=[int(p.stem[4:]) for p in paths]
    assert seeds==list(range(10100,10108))+[10110,10111],seeds
    config=vars(args)|{'seeds':seeds,'train_seeds':list(range(10100,10108)),
        'validation_seeds':[10110,10111],'train_noise_seed':20261007,'batch_seed':20261007,
        'lr':.01,'initialization':'gray0.5','texture_size':128,'world_center':[0,0,.741],
        'size_m':.20,'loss':'maximize squared first t=1 velocity difference, all 32x32 dimensions',
        'joint_views':list(MODEL),'shared_world_texture':True,'eot':False,
        'budget_note':'nominal imaging comparison; no extra EOT transforms in first pass',
        'pool_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.manual_seed(20261007)
    policy=M.load();model=policy._model;device=policy._pytorch_device
    pool=Pool(paths,device,global_response=args.global_response)
    ids=[i for i,f in enumerate(pool.frames) if f['seed']<10110]
    val=[i for i,f in enumerate(pool.frames) if f['seed']>=10110]
    obs=[M.observation(policy,f['state'],f['prompt'],f['raw']) for f in pool.frames]
    noise_np=np.random.RandomState(20261007).standard_normal((32,32)).astype(np.float32)
    noise=torch.from_numpy(noise_np)[None].to(device)
    clean=[]
    with torch.no_grad():
        for i,o in enumerate(obs):
            clean.append(M.velocity(model,o,noise).detach())
            if i%10==0:print('[cache]',i,len(obs),flush=True)
    q=torch.full((1,3,args.band,args.band),.5,device=device,requires_grad=True)
    def texture():return q if args.band==128 else F.interpolate(q,(128,128),mode='bilinear',align_corners=False)
    def loss(i):
        o=dataclasses.replace(obs[i],images=pool.images(texture(),i,args.mode))
        return (M.velocity(model,o,noise)-clean[i]).square().sum()
    rng=np.random.RandomState(20261007);history=[];batches=[];started=time.monotonic()
    for it in range(args.iters):
        batch=rng.choice(ids,min(args.batch,len(ids)),replace=False);q.grad=None;values=[]
        for i in batch:
            v=loss(int(i));values.append(float(v.detach()));(v/len(batch)).backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        with torch.no_grad():q.add_(.01*q.grad.sign()).clamp_(0,1)
        history.append(float(np.mean(values)));batches.append(batch.tolist())
        if (it+1)%25==0 or it+1==args.iters:
            with torch.no_grad():
                vl=[float(loss(i)) for i in val]
                tex=texture().detach().cpu().numpy()[0]
            Image.fromarray(np.rint(tex.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png')
            np.savez_compressed(dest/f'checkpoint_{it+1:04d}.npz',latent=q.detach().cpu().numpy(),texture=tex,
                                history=history,batches=batches,validation=vl,noise=noise_np)
            progress={'complete':it+1==args.iters,'iteration':it+1,'train_loss':history[-1],
                      'validation_loss':float(np.mean(vl)),'elapsed_s':time.monotonic()-started}
            (dest/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            print('[train]',progress,flush=True)


if __name__=='__main__':main()
