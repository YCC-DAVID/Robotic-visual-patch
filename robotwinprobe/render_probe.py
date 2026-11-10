"""Paired per-layer prefix KV, flow, and integrated-action diagnostics."""
import argparse,dataclasses,json,time
import numpy as np
import bootstrap as B
B.model_paths()
from render_geometry import CAMERAS,WIRE,MODEL,patterns
from render_proxy import Pool
import render_model as M


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pool',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--seeds',default='10110,10111');ap.add_argument('--capture')
    ap.add_argument('--textures');ap.add_argument('--smoke-gradient',action='store_true')
    ap.add_argument('--global-response',action='store_true')
    args=ap.parse_args()
    import torch
    from PIL import Image
    from openpi_client.image_tools import resize_with_pad
    dest=B.Path(args.out).resolve();dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'metrics.json').exists()
    policy=M.load();device=policy._pytorch_device;model=policy._model
    pool=Pool([B.Path(args.pool)/f'seed{s}.npz' for s in args.seeds.split(',')],device,global_response=args.global_response)
    textures={k:patterns()[k] for k in ['checker','random']}
    if args.textures:textures={k:np.array(Image.open(v).convert('RGB')) for k,v in json.loads(B.Path(args.textures).read_text()).items()}
    noise_np=np.array([np.random.RandomState(s).standard_normal((32,32)).astype(np.float32) for s in [20261007,20261008,20261009]])
    noises=[torch.tensor(n,device=device)[None] for n in noise_np]
    def forward(o):
        state,pad,cache=M.prefix(model,o)
        kv=[tuple(t[:,:,pad[0].bool()].float().clone() for t in layer) for layer in cache]
        velocities=[];actions=[]
        for noise in noises:
            a=noise.clone()
            for step in range(10):
                v=model.denoise_step(state,pad,cache,a,torch.full((1,),1-step/10,device=device))
                if step==0:velocities.append(v.float())
                a=a-.1*v
            actions.append(a.float())
        return torch.stack(velocities),torch.stack(actions),kv,pad
    def compare(a,b):
        v,act,kv,pad=a;vc,ac,kc,pc=b
        assert torch.equal(pad,pc)
        layers=[]
        for l,(x,y) in enumerate(zip(kv,kc)):
            row={'layer':l}
            for typ,tx,ty in zip(('K','V'),x,y):
                row[typ+'_relative_l2']=float((tx-ty).norm()/ty.norm().clamp_min(1e-12))
                row[typ+'_cosine']=float(torch.nn.functional.cosine_similarity(tx.flatten(),ty.flatten(),dim=0))
            layers.append(row)
        return {'L0':float((v-vc).square().flatten(1).sum(1).mean()),
                'L0_real14':float((v[...,:14]-vc[...,:14]).square().flatten(1).sum(1).mean()),
                'action_MSE_real14':float((act[...,:14]-ac[...,:14]).square().mean()),
                'mean_layer_K_relative_l2':float(np.mean([r['K_relative_l2'] for r in layers])),
                'mean_layer_V_relative_l2':float(np.mean([r['V_relative_l2'] for r in layers])), 'layers':layers}
    rows=[];allv=[];alla=[];started=time.monotonic()
    for i,f in enumerate(pool.frames):
        obs=M.observation(policy,f['state'],f['prompt'],f['raw'])
        if args.smoke_gradient and i==0:
            q=torch.full((1,3,128,128),.5,device=device,requires_grad=True)
            with torch.no_grad():vc=M.velocity(model,obs,noises[0])
            smoke={}
            for mode in ['digital','proxy']:
                q.grad=None
                o=dataclasses.replace(obs,images=pool.images(q,i,mode))
                loss=(M.velocity(model,o,noises[0])-vc).square().sum();loss.backward()
                assert torch.isfinite(q.grad).all() and float(q.grad.norm())>0
                smoke[mode]={'loss':float(loss.detach()),'gradient_norm':float(q.grad.norm()),'texel_coverage':float((q.grad.abs().sum(1)>0).float().mean())}
            (dest/'gradient_smoke.json').write_text(json.dumps(smoke,indent=2)+'\n')
        with torch.no_grad():
            clean=forward(obs)
            if i==0:
                repeat=forward(obs);assert torch.equal(clean[0],repeat[0]) and torch.equal(clean[1],repeat[1])
            for label,rgb in textures.items():
                q=torch.tensor(rgb,device=device).permute(2,0,1)[None].float()/255
                if args.capture:
                    with np.load(B.Path(args.capture)/f"seed{f['seed']}.npz") as d:
                        j=int(np.flatnonzero(d['query']==f['query'])[0])
                        raw={w:d[n+'_'+label][j] for n,w in zip(CAMERAS,WIRE)}
                else:
                    raw={w:(f['views'][n][label][0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8) for n,w in zip(CAMERAS,WIRE)}
                ro=M.observation(policy,f['state'],f['prompt'],raw);render=forward(ro)
                row={'seed':f['seed'],'query':f['query'],'texture':label,'render_vs_clean':compare(render,clean)}
                vv=[clean[0].cpu().numpy(),render[0].cpu().numpy()];aa=[clean[1].cpu().numpy(),render[1].cpu().numpy()]
                for mode in ['digital','proxy']:
                    images=pool.images(q,i,mode);o=dataclasses.replace(obs,images=images);r=forward(o)
                    row[mode+'_vs_clean']=compare(r,clean);row[mode+'_vs_render']=compare(r,render)
                    pixels={}
                    for n,key in zip(CAMERAS,MODEL):
                        mask=torch.nn.functional.pad(torch.nn.functional.interpolate(f['views'][n]['mask'],(168,224),mode='nearest'),(0,0,28,28)).bool().expand(-1,3,-1,-1)
                        delta=(images[key]-ro.images[key])/2
                        pixels[n]={'RMSE':float(delta.square().mean().sqrt()),
                            'mask_RMSE':float(delta[mask].square().mean().sqrt()) if mask.any() else None,
                            'outside_RMSE':float(delta[~mask].square().mean().sqrt())}
                    row[mode+'_pixels']=pixels;vv.append(r[0].cpu().numpy());aa.append(r[1].cpu().numpy())
                rows.append(row);allv.append(vv);alla.append(aa)
        print('[probe]',f['seed'],f['query'],flush=True)
    np.savez_compressed(dest/'outputs.npz',velocities=allv,actions=alla,noises=noise_np,conditions=['clean','render','digital','proxy'])
    summary={}
    for label in textures:
        rs=[r for r in rows if r['texture']==label]
        summary[label]={c:{m:float(np.mean([r[c][m] for r in rs])) for m in ['L0','L0_real14','action_MSE_real14','mean_layer_K_relative_l2','mean_layer_V_relative_l2']}
                        for c in ['render_vs_clean','digital_vs_clean','proxy_vs_clean','digital_vs_render','proxy_vs_render']}
    (dest/'metrics.json').write_text(json.dumps({'complete':True,'rows':rows,'summary':summary,
        'elapsed_s':time.monotonic()-started,'config':vars(args),'notes':'Actions are normalized model coordinates; per-layer KV distances use valid tokens only. Three common noises. No closed-loop conclusions.'},indent=2)+'\n')
    print('[complete]',summary,flush=True)


if __name__=='__main__':main()
