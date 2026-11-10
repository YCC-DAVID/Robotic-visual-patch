"""Frozen 9x9 true-render cross-placement L0; random shares states and noises."""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import bootstrap as B
from radius_plan import dump
from table9_plan import records,sha,prepare


def probe(root,site):
    B.model_paths()
    import torch
    from PIL import Image
    import render_model as M
    from render_geometry import CAMERAS,WIRE
    from table9_train import Bank,noises
    prepare(root)
    output=root/'sites'/site/'cross_probe.json'
    rs=records(root,site,'post');cal=records(root,site,'validation')
    hashes=json.loads((root/'sites'/site/'post_texture_hashes.json').read_text())
    for sid,h in hashes.items():assert sha(root/'sites'/sid/'train/texture_0500.png')==h
    if output.exists():
        saved=json.loads(output.read_text());assert saved['texture_hashes']==hashes and saved['complete'];return
    policy=M.load();model=policy._model;bank=Bank(root,site,policy)
    ns=noises(root,bank.device,probe=True);cache={};rows=[]
    def velocity(obs):
        state,pad,kv=M.prefix(model,obs)
        return torch.cat([model.denoise_step(state,pad,kv,n,torch.ones(1,device=bank.device)) for n in ns])
    own=np.asarray(Image.open(root/'sites'/site/'train/texture_0500.png').convert('RGB'))
    q=torch.tensor(own.copy(),device=bank.device).permute(2,0,1)[None].float()/255
    with torch.no_grad():
        for r in rs:
            obs,_=bank.clean(r);key=(r['frame'],r['light'])
            if key not in cache:cache[key]=velocity(obs)
            vc=cache[key];f=bank.frames[r['frame']]
            vp=velocity(dataclasses.replace(obs,images=bank.images(q,cal[r['id']])))
            with np.load(r['path']) as z:
                assert int(z['frame'])==r['frame'] and float(z['yaw_deg'])==r['yaw_deg']
                for method in ['random']+list(hashes):
                    raw={w:z[n+'_'+method].copy() for n,w in zip(CAMERAS,WIRE)}
                    vr=velocity(M.observation(policy,f['state'],f['prompt'],raw))
                    l0=(vr-vc).square().flatten(1).sum(1).cpu().numpy()
                    row=dict(target_site=site,method=method,sample=r['id'],frame=r['frame'],seed=r['seed'],query=r['query'],
                        condition='nominal' if r['id']<len(rs)//2 else 'joint',yaw_deg=r['yaw_deg'],light=r['light'],
                        center=r['center'],L0_per_noise=l0.tolist(),L0_mean=float(l0.mean()))
                    if method==site:
                        row['proxy_L0_per_noise']=(vp-vc).square().flatten(1).sum(1).cpu().tolist()
                        row['proxy_render_SSE_per_noise']=(vp-vr).square().flatten(1).sum(1).cpu().tolist()
                    rows.append(row)
            dump(root/'sites'/site/'cross_probe_progress.json',dict(completed_samples=r['id']+1,total=len(rs),complete=r['id']+1==len(rs)))
            print('[cross_probe]',site,r['id']+1,len(rs),flush=True)
    dump(output,dict(complete=True,rows=rows,texture_hashes=hashes,
        capture_hashes={str(Path(r['path']).relative_to(root)):sha(r['path']) for r in rs},
        clean_pairing='identical state, real scene lighting and three disjoint action noises',
        single_patch_per_scene=True,endpoint='t=1 velocity SSE; not task ASR'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--site',required=True)
    args=parser.parse_args();probe(Path(args.root).resolve(),args.site)
