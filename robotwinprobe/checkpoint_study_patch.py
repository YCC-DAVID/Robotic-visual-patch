"""Matched patch seeds across frozen checkpoints; one physical three-view texture."""
import argparse
import dataclasses
import json
from pathlib import Path
import time
import numpy as np
import checkpoint_study_common as S
import checkpoint_study_model as M


def train(args):
    import torch
    from PIL import Image
    from table9_train import Bank
    from table9_plan import records
    p=S.prepare_protocol(args.root);pc=dict(p['patch']);reference=S.PATCH_REFERENCE;site=pc['site']
    if args.smoke:
        assert args.model=='smoke'
        pc['steps']=2
    assert json.loads((reference/'sites'/site/'proxy_validation.json').read_text())['passed']
    assert json.loads((reference/'sites'/site/'train_capture_complete.json').read_text())['complete']
    dest=args.root/'patches'/args.model/f'P{args.seed}';dest.mkdir(parents=True,exist_ok=True)
    checkpoint=args.root/'models'/args.model
    rs=records(reference,site,'train');val=records(reference,site,'validation')
    assert len(rs)==4000
    order=np.random.default_rng(args.seed).permutation(len(rs))[:pc['steps']*pc['batch']];rs=[rs[i] for i in order]
    if args.smoke:val=val[:2]
    config=dict(model=args.model,seed=args.seed,checkpoint_sha256=S.sha(checkpoint/'model.safetensors'),
        recipe=pc,source_sha256=S.sha(Path(__file__)),proxy_source_sha256=S.sha(S.ROOT/'robotwinprobe/render_proxy.py'),
        bank_plan_sha256=S.sha(reference/'plan.npz'),order=order.tolist(),
        noise_seeds=[args.seed*10+i for i in range(16)],probe_noise_seed=202610289)
    if (dest/'config.json').exists():assert json.loads((dest/'config.json').read_text())==config
    else:S.dump(dest/'config.json',config)
    if (dest/'status.json').exists() and json.loads((dest/'status.json').read_text()).get('complete'):return
    torch.manual_seed(args.seed);policy=M.load(checkpoint);model=policy._model;bank=Bank(reference,site,policy)
    def noise(seed):return torch.tensor(np.random.default_rng(seed).standard_normal((1,32,32)).astype(np.float32),device='cuda')
    ns=[noise(s) for s in config['noise_seeds']]+[noise(config['probe_noise_seed'])]
    clean={};unique={(r['frame'],r['light']):r for r in rs+val}
    S.dump(dest/'status.json',dict(stage='clean_cache',complete=False,total_frames=len(unique)))
    with torch.no_grad():
        for key,r in unique.items():
            obs,_=bank.clean(r);state,pad,kv=M.prefix(model,obs)
            for ni,n in enumerate(ns):
                clean[key+(ni,)]=model.denoise_step(state,pad,kv,n,torch.ones(1,device='cuda')).detach()
        del state,pad,kv
    q=torch.full((1,3,128,128),.5,device='cuda',requires_grad=True);start=0;history=[]
    saved=sorted(dest.glob('checkpoint_*.npz'))
    if saved:
        with np.load(saved[-1]) as z:
            start=int(z['iteration']);history=z['history'].tolist()
            with torch.no_grad():q.copy_(torch.tensor(z['texture'],device='cuda')[None])
    def loss(r,ni):
        obs,_=bank.clean(r)
        v=M.velocity(model,dataclasses.replace(obs,images=bank.images(q,r)),ns[ni])
        return (v-clean[r['frame'],r['light'],ni]).square().sum()
    began=time.monotonic()
    for it in range(start,pc['steps']):
        q.grad=None;values=[]
        for r in rs[it*8:(it+1)*8]:
            value=loss(r,r['noise_id']);values.append(float(value.detach()));(value/8).backward()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        gn=float(q.grad.norm())
        with torch.no_grad():q.add_(pc['lr']*q.grad.sign()).clamp_(0,1)
        history.append(float(np.mean(values)))
        if it==0 or (it+1)%25==0 or it+1==pc['steps']:
            with torch.no_grad():
                validation=[float(loss(r,16)) for r in val]
                texture=q.detach().cpu().numpy()[0]
            png=dest/f'texture_{it+1:04d}.png'
            Image.fromarray(np.rint(texture.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(png)
            target=dest/f'checkpoint_{it+1:04d}.npz';tmp=target.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,iteration=it+1,texture=texture,history=history,validation=validation);tmp.replace(target)
            status=dict(stage='trained' if it+1==pc['steps'] else 'training',complete=it+1==pc['steps'],
                iteration=it+1,train_L0=history[-1],validation_L0_mean=float(np.mean(validation)),
                validation_L0_median=float(np.median(validation)),gradient_norm=gn,elapsed_s=time.monotonic()-began,
                texture=str(png),texture_sha256=S.sha(png))
            S.dump(dest/'status.json',status);print('[patch]',args.model,args.seed,status,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=S.DEFAULT)
    ap.add_argument('--model',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--smoke',action='store_true')
    train(ap.parse_args())
