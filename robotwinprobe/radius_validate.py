"""Independent patterns at held-out continuous positions; pixel/model/gradient checks."""
import argparse, dataclasses
from pathlib import Path
import numpy as np
import bootstrap as B
B.model_paths()
from radius_plan import records, REFERENCE, dump
from radius_proxy import CaptureProxy
from render_geometry import CAMERAS,WIRE,MODEL,patterns
import render_model as M


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    ap.add_argument('--radius',type=int,choices=[1,3,5],required=True)
    ap.add_argument('--limit',type=int,default=0);args=ap.parse_args();root=Path(args.root).resolve()
    import torch
    from PIL import Image
    from openpi_client.image_tools import resize_with_pad
    policy=M.load();model=policy._model;device=policy._pytorch_device
    proxy=CaptureProxy(device);rs=records(root,args.radius,'validation')
    if args.limit:rs=rs[:args.limit]
    textures={k:patterns()[k] for k in ['random','checker']}
    textures['baseline']=np.array(Image.open(REFERENCE/'train_proxy_full/texture_0500.png').convert('RGB'))
    noise=torch.tensor(np.random.RandomState(20261007).standard_normal((1,32,32)).astype(np.float32),device=device)
    rows=[]
    for r in rs:
        f=proxy.reference.frames[r['frame']]
        with torch.no_grad():
            obs=M.observation(policy,f['state'],f['prompt'],f['raw']);vc=M.velocity(model,obs,noise)
            with np.load(r['path']) as d:
                for name,rgb in textures.items():
                    q=torch.tensor(rgb,device=device).permute(2,0,1)[None].float()/255
                    images=proxy.images(q,r['frame'],r);raw={w:d[n+'_'+name] for n,w in zip(CAMERAS,WIRE)}
                    ro=M.observation(policy,f['state'],f['prompt'],raw)
                    vr=M.velocity(model,ro,noise);vp=M.velocity(model,dataclasses.replace(obs,images=images),noise)
                    row={'sample':r['id'],'seed':r['seed'],'query':r['query'],'center':r['center'],
                        'pattern':name,'velocity':{'render_clean':float((vr-vc).square().sum()),
                        'proxy_clean':float((vp-vc).square().sum()),'proxy_render':float((vp-vr).square().sum())},'pixels':{}}
                    for n,w,k in zip(CAMERAS,WIRE,MODEL):
                        target=torch.tensor(resize_with_pad(raw[w],224,224),device=device).permute(2,0,1)[None].float()/255
                        delta=(images[k]+1)/2-target
                        mask=torch.tensor(d[n+'_mask'],device=device)[None,None].float()
                        mask=torch.nn.functional.pad(torch.nn.functional.interpolate(mask,(168,224),mode='nearest'),(0,0,28,28)).bool().expand_as(delta)
                        row['pixels'][n]={'rmse':float(delta.square().mean().sqrt()),
                            'mask_rmse':float(delta[mask].square().mean().sqrt()) if mask.any() else None}
                    rows.append(row)
        print('[validated]',r['id'],flush=True)
    q=torch.full((1,3,128,128),.5,device=device,requires_grad=True)
    v=M.velocity(model,dataclasses.replace(obs,images=proxy.images(q,r['frame'],r)),noise)
    loss=(v-vc).square().sum();loss.backward()
    gradient={'norm':float(q.grad.norm()),'finite':bool(torch.isfinite(q.grad).all())}
    passed=all(p['rmse']<.06 and (p['mask_rmse'] is None or p['mask_rmse']<.15)
               for row in rows for p in row['pixels'].values()) and gradient['finite'] and gradient['norm']>0
    out=root/'validation'/f'r{args.radius:02d}'/('smoke.json' if args.limit else 'results.json')
    dump(out,{'passed':passed,'rows':rows,'gradient':gradient,'samples':len(rs),
              'gate':'all-image RMSE < .06; visible mask RMSE < .15; finite nonzero gradient',
              'note':'model velocity gaps reported, not an ASR guarantee'})
    assert passed, 'validation failed: inspect results; do not silently relax gate'


if __name__=='__main__':main()
