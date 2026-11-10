"""Gate EOT bank fidelity against measured random/checker renders and matched clean."""
import argparse, dataclasses, json
import numpy as np
import bootstrap as B
B.model_paths()
from render_geometry import CAMERAS, WIRE, MODEL, patterns
from eot_bank import Bank
import render_model as M


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bank',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--seeds',default='10110');ap.add_argument('--model',action='store_true')
    args=ap.parse_args()
    import torch
    from openpi_client.image_tools import resize_with_pad
    root=B.Path(args.bank).resolve();dest=B.Path(args.out).resolve();dest.mkdir(parents=True,exist_ok=True)
    keys=sorted(p.name for p in root.iterdir() if p.is_dir() and '__' in p.name)
    torch.set_num_threads(4)
    policy=M.load() if args.model else None
    device=policy._pytorch_device if policy else 'cpu'
    bank=Bank(root,keys,device,seeds=list(map(int,args.seeds.split(','))))
    rows=[]
    for key in keys:
        # One fixed held-out physical state, all transforms, two nonoptimized patterns.
        f=bank.frames(key)[0]
        with torch.no_grad():
            if policy:
                obs=M.observation(policy,f['state'],f['prompt'],f['raw'])
                noise=torch.tensor(np.random.RandomState(20261007).standard_normal((1,32,32)).astype(np.float32),device=device)
                vc=M.velocity(policy._model,obs,noise)
            for label in ['random','checker']:
                q=torch.tensor(patterns()[label],device=device).permute(2,0,1)[None].float()/255
                images=bank.images(q,key,0);row={'condition':key,'pattern':label,'pixels':{}}
                raw={}
                for n,w,m in zip(CAMERAS,WIRE,MODEL):
                    view=f['views'][n]
                    actual=(view[label][0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
                    raw[w]=actual
                    target=torch.tensor(resize_with_pad(actual,224,224),device=device).permute(2,0,1)[None].float()/255
                    delta=(images[m]+1)/2-target
                    mask=torch.nn.functional.pad(torch.nn.functional.interpolate(view['mask'].to(device),(168,224),mode='nearest'),(0,0,28,28)).bool().expand_as(delta)
                    row['pixels'][n]={'rmse':float(delta.square().mean().sqrt()),
                        'mask_rmse':float(delta[mask].square().mean().sqrt()) if mask.any() else None,
                        'mask_pixels':int(mask[0,0].sum())}
                if policy:
                    vr=M.velocity(policy._model,M.observation(policy,f['state'],f['prompt'],raw),noise)
                    vp=M.velocity(policy._model,dataclasses.replace(obs,images=images),noise)
                    row['velocity']={'render_clean':float((vr-vc).square().sum()),
                        'proxy_clean':float((vp-vc).square().sum()),'proxy_render':float((vp-vr).square().sum())}
                rows.append(row)
                print('[validation]',row,flush=True)
    # Predeclared conservative gate; no optimizing a patch against a broken bank.
    passed=all(p['rmse'] < .06 and (p['mask_rmse'] is None or p['mask_rmse'] < .15)
               for r in rows for p in r['pixels'].values())
    gradient={}
    if policy:
        key=keys[-1];f=bank.frames(key)[0]
        obs=M.observation(policy,f['state'],f['prompt'],f['raw'])
        with torch.no_grad():vc=M.velocity(policy._model,obs,noise)
        q=torch.full((1,3,128,128),.5,device=device,requires_grad=True)
        v=M.velocity(policy._model,dataclasses.replace(obs,images=bank.images(q,key,0)),noise)
        loss=(v-vc).square().sum();loss.backward()
        gradient={'loss':float(loss.detach()),'norm':float(q.grad.norm()),'finite':bool(torch.isfinite(q.grad).all())}
        passed=passed and gradient['finite'] and gradient['norm']>0
    report={'passed':passed,'rows':rows,'gradient':gradient,
        'gate':'all-image RMSE < .06, visible-mask RMSE < .15, finite nonzero model gradient',
        'limits':'pixel gate is not an ASR guarantee; per-condition paired velocity gaps are reported separately'}
    (dest/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    assert passed, 'proxy fidelity gate failed; inspect validation.json before training'


if __name__=='__main__':main()
