"""Matched physical-position / whole-scene-light EOT ablations, one shared Q."""
import argparse, dataclasses, hashlib, json, time
import numpy as np
import bootstrap as B
B.model_paths()
from eot_conditions import allowed, conditions, POSITIONS, LIGHTS
from eot_bank import Bank
import render_model as M


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--mode',choices=['baseline','position','light','joint'],required=True)
    ap.add_argument('--iters',type=int,default=500);ap.add_argument('--batch',type=int,default=8)
    args=ap.parse_args()
    import torch
    from PIL import Image
    dest=B.Path(args.out).resolve();dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'config.json').exists(), 'preserve training run'
    root=B.Path(args.bank).resolve();keys=allowed(args.mode)
    config=vars(args)|{'conditions':{k:conditions()[k] for k in keys},'texture_size':128,
        'initialization':'gray0.5','shared_world_texture':True,'views':'all three joint',
        'lr':.01,'batch_seed':20261007,'transform_seed':20261008,'noise_seed':20261007,
        'loss':'maximize squared first t=1 velocity difference, all 32x32 dimensions',
        'clean_reference':'same state and actual lighting; no patch',
        'eot':'uniform finite measured distribution, one transform per gradient frame',
        'train_seeds':list(range(10100,10108)),'validation_seeds':[10110,10111],
        'pool_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for k in keys for p in sorted((root/k).glob('seed*.npz'))}}
    (dest/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    torch.manual_seed(20261007)
    policy=M.load();model=policy._model;device=policy._pytorch_device
    bank=Bank(root,keys,device);frames=bank.frames(keys[0])
    ids=[i for i,f in enumerate(frames) if f['seed']<10110]
    val=[i for i,f in enumerate(frames) if f['seed']>=10110]
    noise_np=np.random.RandomState(20261007).standard_normal((32,32)).astype(np.float32)
    noise=torch.tensor(noise_np,device=device)[None]
    cache={}
    for key in keys:
        light=conditions()[key]['light']
        for i,f in enumerate(bank.frames(key)):
            ck=(i,light)
            if ck in cache:
                assert all(np.array_equal(f['raw'][n], cache[ck][2][n]) for n in f['raw'])
                continue
            o=M.observation(policy,f['state'],f['prompt'],f['raw'])
            with torch.no_grad():v=M.velocity(model,o,noise).detach()
            cache[ck]=(o,v,f['raw'])
        print('[clean cache]',key,len(cache),flush=True)
    q=torch.full((1,3,128,128),.5,device=device,requires_grad=True)
    def loss(i,key):
        obs,clean,_=cache[(i,conditions()[key]['light'])]
        o=dataclasses.replace(obs,images=bank.images(q,key,i))
        return (M.velocity(model,o,noise)-clean).square().sum()
    rng=np.random.RandomState(20261007);trng=np.random.RandomState(20261008)
    history=[];batches=[];transforms=[];started=time.monotonic()
    for it in range(args.iters):
        batch=rng.choice(ids,min(args.batch,len(ids)),replace=False);q.grad=None;values=[];chosen=[]
        for i in batch:
            # Independent streams from state sampling, shared draws across ablations.
            p=list(POSITIONS)[trng.randint(len(POSITIONS))]
            l=list(LIGHTS)[trng.randint(len(LIGHTS))]
            if args.mode not in ('position','joint'):p='origin'
            if args.mode not in ('light','joint'):l='nominal'
            key=p+'__'+l;chosen.append(key)
            v=loss(int(i),key);values.append(float(v.detach()));(v/len(batch)).backward()
        assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.norm()>0
        with torch.no_grad():q.add_(.01*q.grad.sign()).clamp_(0,1)
        history.append(float(np.mean(values)));batches.append(batch.tolist());transforms.append(chosen)
        if (it+1)%25==0 or it+1==args.iters:
            with torch.no_grad():
                # Fixed validation transforms. Never used for checkpoint selection.
                vl={k:float(np.mean([float(loss(i,k)) for i in val])) for k in keys}
                tex=q.detach().cpu().numpy()[0]
            Image.fromarray(np.rint(tex.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png')
            np.savez_compressed(dest/f'checkpoint_{it+1:04d}.npz',texture=tex,history=history,
                                batches=batches,transforms=transforms,noise=noise_np)
            progress={'complete':it+1==args.iters,'iteration':it+1,'train_loss':history[-1],
                'validation_by_condition':vl,'validation_mean':float(np.mean(list(vl.values()))),
                'elapsed_s':time.monotonic()-started}
            (dest/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            print('[train]',progress,flush=True)


if __name__=='__main__':main()
