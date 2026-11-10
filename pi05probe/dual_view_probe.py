"""Shared-Q base/wrist/both diagnostics with independent KV and early velocity."""
import argparse,json,dataclasses,time
import drift_patch_train as T
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--modes',default='base,wrist,both');args=ap.parse_args()
    import torch,jax
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    from dual_view_proxy import CameraProxy
    torch.set_num_threads(4);dest=T.OUT/args.out;dest.mkdir(parents=True,exist_ok=True)
    root=T.OUT/'dual_view';base=np.load(root/'calibration/base.npz');wrist=np.load(root/'calibration/wrist.npz')
    if (dest/'inputs.npz').exists():d=np.load(dest/'inputs.npz',allow_pickle=False)
    else:
        ids=np.flatnonzero(base['validation_mask']);data={k:base[k][ids] for k in ['clean_base','clean_wrist','state8','ep_id','env_step']}
        data.update(frames=ids,prompt=base['prompt'],patterns=np.array(['random','check']))
        from PIL import Image
        for pattern,field in [('random','random256'),('check','check256')]:
            data[pattern+'__texture']=base[pattern+'__texture']
            for view,cal in [('base',base),('wrist',wrist)]:
                data[pattern+'__'+view]=np.array([np.array(Image.fromarray(cal[field][k]).resize((224,224),Image.Resampling.BILINEAR)) for k in ids])
        np.savez_compressed(dest/'inputs.npz',**data);d=data
    images={}
    for view in ['base','wrist']:
        proxy=CameraProxy(root/'calibration'/(view+'.npz'),fit=root/'fit'/(view+'.json'))
        with torch.no_grad():
            for pattern in d['patterns']:
                q=torch.from_numpy(d[pattern+'__texture'].transpose(2,0,1).astype(np.float32)/255)[None]
                images[pattern+'__'+view]=np.array([np.rint(proxy(q,int(k)).numpy()[0].transpose(1,2,0)*255).clip(0,255).astype(np.uint8) for k in d['frames']])
        del proxy
    np.savez_compressed(dest/'proxy_images.npz',**images)
    cfg=C.get_config('pi05_libero');cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    policy=PC.create_trained_policy(cfg,T.TORCH_CKPT);model,device=policy._model,policy._pytorch_device;model.eval()
    for p in model.parameters():p.requires_grad_(False)
    seeds=[20260901,20260902,20260903];noise_np=np.array([np.random.RandomState(s).standard_normal((10,32)).astype(np.float32) for s in seeds]);noises=[torch.from_numpy(n)[None].to(device) for n in noise_np]
    def forward(b,w,k):
        x=policy._input_transform({'observation/image':b,'observation/wrist_image':w,'observation/state':d['state8'][k],'prompt':str(d['prompt'])})
        x=jax.tree.map(lambda a:torch.from_numpy(np.array(a)).to(device)[None],x)
        state,mask,cache=T.prefix_and_cache(model,device,M.Observation.from_dict(x))
        v=torch.stack([model.denoise_step(state,mask,cache,e.clone(),torch.ones(1,device=device)) for e in noises]).float()
        kv=torch.cat([t[:,:,mask[0].bool()].float().flatten() for layer in cache for t in layer])
        return v,kv
    def metric(a,b):
        e=a-b
        return {'L0':float(e.square().flatten(1).sum(1).mean()),'L0_real7':float(e[...,:7].square().flatten(1).sum(1).mean()),'L0_exec7':float(e[:,:,:5,:7].square().flatten(1).sum(1).mean())}
    records=[];velocities=[];cleanvs=[];names=[];start=time.monotonic()
    with torch.no_grad():
        for k,frame in enumerate(d['frames']):
            cb,cw=d['clean_base'][k],d['clean_wrist'][k];vc,kc=forward(cb,cw,k)
            if k==0:
                vr,kr=forward(cb,cw,k);assert torch.equal(vc,vr) and torch.equal(kc,kr)
            framev=[];frame_names=[]
            for pattern in d['patterns']:
                for mode in args.modes.split(','):
                    pair=[]
                    for kind,source in [('render',d),('proxy',images)]:
                        b=source[pattern+'__base'][k] if mode in ['base','both'] else cb
                        w=source[pattern+'__wrist'][k] if mode in ['wrist','both'] else cw
                        v,kv=forward(b,w,k);pair.append((v,kv));name=str(pattern)+'/'+mode+'/'+kind
                        row={'frame':int(frame),'ep_id':int(d['ep_id'][k]),'env_step':int(d['env_step'][k]),'condition':name,'vs_clean':metric(v,vc),'KV_relative_clean':float((kv-kc).norm()/kc.norm())}
                        if kind=='proxy':row.update(vs_render=metric(v,pair[0][0]),KV_relative_render=float((kv-pair[0][1]).norm()/pair[0][1].norm()))
                        records.append(row);framev.append(v.cpu().numpy());frame_names.append(name)
                    del pair
            velocities.append(framev);cleanvs.append(vc.cpu().numpy());names=frame_names
            print('[probe]',int(frame),'elapsed',round(time.monotonic()-start),flush=True)
    summary={}
    for name in names:
        rs=[r for r in records if r['condition']==name];s={key:float(np.mean([r['vs_clean'][key] for r in rs])) for key in ['L0','L0_real7','L0_exec7']}
        if name.endswith('/proxy'):
            s.update(paired_L0_vs_render=float(np.mean([r['vs_render']['L0'] for r in rs])),KV_relative_render=float(np.mean([r['KV_relative_render'] for r in rs])))
        summary[name]=s
    for name in names:
        if name.endswith('/proxy'):
            s=summary[name];s['paired_energy_ratio']=s['paired_L0_vs_render']/max(summary[name.replace('/proxy','/render')]['L0'],1e-9)
    np.savez_compressed(dest/'velocities.npz',velocity=np.array(velocities),clean_velocity=np.array(cleanvs),noise=noise_np,conditions=np.array(names),frames=d['frames'])
    (dest/'metrics.json').write_text(json.dumps({'complete':True,'records':records,'seeds':seeds,'elapsed_s':time.monotonic()-start,'notes':'SameQ perview, actual render vs imaging proxy; base/wrist only are diagnostic input replacements, both is physical two-camera observation.'},indent=2)+'\n')
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('[done]',summary,flush=True)

if __name__=='__main__':main()
