"""One fixed full-band 13cm SAPIEN proxy + texture-plane EOT patch, 500x8."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import bootstrap as B
from main_method_common import records,training_spec,dump,sha,CAMERAS


def fidelity(rows,device):
    import cv2,torch
    from crossarch_proxy import images
    from render_geometry import patterns
    q=torch.tensor(patterns()['random'].transpose(2,0,1)[None].copy(),device=device).float()/255
    report={c:[] for c in CAMERAS}
    with torch.no_grad():
        for k in np.linspace(0,len(rows)-1,8,dtype=int):
            vs=images(q,rows[k]['path'])
            with np.load(rows[k]['path'],allow_pickle=False) as z:
                for c,v in zip(CAMERAS,vs):
                    mask=cv2.erode(z[c+'_mask'].astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
                    if mask.sum()<20:continue
                    predicted=v[0].permute(1,2,0).cpu().numpy()
                    rmse=float(np.sqrt(((predicted[mask]-z[c+'_random'][mask]/255.)**2).mean()))
                    assert np.isfinite(rmse)
                    report[c].append(dict(record=int(k),pixels=int(mask.sum()),RMSE=rmse))
    return dict(per_frame=report,max_RMSE={c:max((x['RMSE'] for x in rs),default=None) for c,rs in report.items()},
                hard_threshold=None,note='Known SAPIEN proxy fidelity gap, not certified against LIBERO 0.02 threshold; true-render ASR is primary')


def train(out,iters=500):
    import torch
    import texture_eot as E
    from PIL import Image
    from crossarch_backend import Backend
    from crossarch_proxy import images
    rows=records();spec=training_spec(rows)
    if iters!=500:raise ValueError('main method pins500 updates')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'config.json').exists(),'refuse overwrite; archive interrupted run first'
    dump(out/'config.json',spec)
    backend=Backend('pi',out)
    device=backend.policy._pytorch_device
    report=fidelity(rows,device);dump(out/'proxy_fidelity.json',report)
    meta=[];clean=[]
    for row in rows:
        with np.load(row['path'],allow_pickle=False) as z:meta.append((z['state14'].copy(),str(z['prompt'])))
    with torch.no_grad():
        for i,(state,prompt) in enumerate(meta):
            obs=backend.prepare(state,prompt,images(None,rows[i]['path']))
            value,_=backend.response(obs,spec['noise_seed'])
            clean.append(value.detach())
        a,_=backend.response(backend.prepare(*meta[0],images(None,rows[0]['path'])),spec['noise_seed'])
        assert torch.equal(a,clean[0]),'clean forward must repeat exactly'
    q=torch.full((1,3,128,128),.5,device=device,requires_grad=True)
    plan=E.plan(500,8,spec['eot_seed']);rng=np.random.RandomState(spec['seed_batch'])
    history=[];batches=[];start=time.monotonic()
    for it in range(500):
        ids=rng.choice(len(rows),8,replace=False);q.grad=None;values=[]
        for slot,i in enumerate(ids):
            state,prompt=meta[int(i)]
            obs=backend.prepare(state,prompt,images(E.transform(q,plan[it][slot]),rows[int(i)]['path']))
            value,_=backend.response(obs,spec['noise_seed'])
            loss=(value.float()-clean[int(i)].float()).square().sum()
            (-loss/8).backward();values.append(float(loss.detach()))
        assert q.grad is not None and torch.isfinite(q.grad).all() and float(q.grad.norm())>0
        assert all(p.grad is None for p in backend.model.parameters()),'policy parameters must remain frozen'
        with torch.no_grad():q.sub_(.01*q.grad.sign()).clamp_(0,1)
        history.append(float(np.mean(values)));batches.append(ids.tolist())
        if it==0 or (it+1)%25==0:
            texture=q.detach().cpu().numpy()[0]
            Image.fromarray(np.rint(texture.transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(out/f'texture_{it+1:04d}.png')
            np.savez_compressed(out/f'checkpoint_{it+1:04d}.npz',Q=texture,history=history,batches=batches)
            dump(out/'progress.json',dict(complete=it+1==500,iterations=it+1,loss=history[-1],elapsed_training_s=time.monotonic()-start))
            print('[train]',it+1,history[-1],flush=True)
    dump(out/'results.json',dict(complete=True,iterations=500,gradient_frame_forwards=4000,
        texture_sha256=sha(out/'texture_0500.png'),config_sha256=sha(out/'config.json'),
        elapsed_training_s=time.monotonic()-start,proxy_fidelity=report))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    args=ap.parse_args();train(args.out)
