"""E20: digital-to-render prefix KV interpolation at fixed noise/state/t=1."""
import argparse
import dataclasses
import hashlib
import json
import time
import drift_patch_train as T
import numpy as np

ALPHAS = [0.,.001,.0025,.005,.01,.02,.05,.1,.15,.2,.25,.3,.4,.5,.6,.7,.75,.8,.9,.95,.98,.99,1.]
PATTERNS = ['random','full','band32']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',default='kv_interpolation')
    ap.add_argument('--source',default='render_additive')
    ap.add_argument('--frames',type=int,default=16)
    args = ap.parse_args()
    import torch
    import jax
    from transformers.cache_utils import DynamicCache
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    torch.set_num_threads(4)
    dest = T.OUT/args.out; dest.mkdir(exist_ok=True)
    source = T.OUT/args.source
    pool = np.load(source/'inputs.npz',allow_pickle=False)
    previous = json.loads((source/'metrics.json').read_text())
    assert previous['complete']
    oldrows = {(r['pattern'],r['condition'],r['step']):r for r in previous['records']}
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg,T.TORCH_CKPT)
    model,device = policy._model,policy._pytorch_device
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    seeds = [20260901,20260902,20260903]
    noise_np = np.stack([np.random.RandomState(s).standard_normal((T.AH,T.AD)).astype(np.float32) for s in seeds])
    noises = [torch.from_numpy(a)[None].to(device) for a in noise_np]
    noise_copies = [e.clone() for e in noises]
    timestep = torch.ones(1,device=device)

    def prefix(image,k):
        x = policy._input_transform({'observation/image':image,'observation/wrist_image':pool['clean_wrist'][k],
            'observation/state':pool['state8'][k],'prompt':str(pool['prompt'])})
        x = jax.tree.map(lambda a:torch.from_numpy(np.array(a)).to(device)[None],x)
        return T.prefix_and_cache(model,device,M.Observation.from_dict(x))

    def head(pack,cache=None):
        cache = pack[2] if cache is None else cache
        return torch.stack([model.denoise_step(pack[0],pack[1],cache,e.clone(),timestep) for e in noises]).float()

    def velocity(a,b):
        d = a-b
        return {field:float(v.mean()) for field,v in [
            ('L0',d.square().flatten(1).sum(1)),
            ('L0_real7',d[...,:7].square().flatten(1).sum(1)),
            ('L0_exec7',d[:,:,:5,:7].square().flatten(1).sum(1))]} | {
            'per_noise_L0':d.square().flatten(1).sum(1).cpu().tolist()}

    def response(a,d,r,alpha):
        # Normalize each noise separately by its digital/render endpoint gap.
        change = (a-d).flatten(1); endpoint = (r-d).flatten(1)
        den2 = endpoint.square().sum(1).clamp_min(1e-20)
        distance = (change.square().sum(1)/den2).sqrt()
        projection = (change*endpoint).sum(1)/den2
        bend = ((change-alpha*endpoint).square().sum(1)/den2).sqrt()
        return {'fraction_of_endpoint_velocity_norm':distance.cpu().tolist(),
                'projection_fraction':projection.cpu().tolist(),
                'deviation_from_linear_velocity':bend.cpu().tolist()}

    def kv_stats(mixed,digital,rendered,valid,alpha):
        totals = {label:torch.zeros(6,device=device,dtype=torch.float64) for label in ['K','V']}
        layers = []
        for i,(m,d,r) in enumerate(zip(mixed,digital,rendered)):
            row = {'layer':i}
            for j,label in enumerate(['K','V']):
                dd=d[j][:,:,valid].float();rr=r[j][:,:,valid].float();mm=m[j][:,:,valid].float()
                change=mm-dd;endpoint=rr-dd
                values=torch.stack([change.square().sum(),endpoint.square().sum(),dd.square().sum(),
                    (change*endpoint).sum(),(change-alpha*endpoint).square().sum(),(mm!=dd).sum()]).double()
                totals[label] += values
                row[label]={'relative_to_digital':float((values[0]/values[2].clamp_min(1e-20)).sqrt()),
                            'fraction_of_endpoint_distance':float((values[0]/values[1].clamp_min(1e-20)).sqrt())}
            layers.append(row)
        def normalized(v,elements):
            return {'relative_to_digital':float((v[0]/v[2].clamp_min(1e-20)).sqrt()),
                    'fraction_of_endpoint_distance':float((v[0]/v[1].clamp_min(1e-20)).sqrt()),
                    'projection_fraction':float(v[3]/v[1].clamp_min(1e-20)),
                    'off_path_fraction':float((v[4]/v[1].clamp_min(1e-20)).sqrt()),
                    'changed_element_fraction':float(v[5]/elements)}
        elements = sum(d[0][:,:,valid].numel() for d in digital)
        return {'layers':layers,'global':{label:normalized(totals[label],elements) for label in ['K','V']}
                | {'KV':normalized(totals['K']+totals['V'],2*elements)}}

    records=[]; velocity_arrays=[]; clean_arrays=[]; start=time.monotonic()
    checks={'endpoints_equal_native':True,'old_endpoint_max_error':0.,'state_masks_equal':True}
    layout=None;n=min(args.frames,len(pool['steps']))
    with torch.no_grad():
        for k,step in enumerate(pool['steps'][:n]):
            clean=prefix(pool['clean_base'][k],k);vc=head(clean)
            if k==0:
                repeated=prefix(pool['clean_base'][k],k)
                assert all(torch.equal(a,b) for x,y in zip(clean[2],repeated[2]) for a,b in zip(x,y))
                assert torch.equal(vc,head(repeated));checks['repeat_clean_equal']=True
                del repeated
            clean_arrays.append(vc.cpu().numpy());state_arrays=[];random_velocities=None
            for pattern in PATTERNS:
                digital=prefix(pool[pattern+'__digital'][k],k)
                rendered=prefix(pool[pattern+'__render'][k],k)
                for pack in [digital,rendered]:
                    assert torch.equal(clean[0],pack[0]) and torch.equal(clean[1],pack[1])
                valid=clean[1][0].bool()
                vd,vr=head(digital),head(rendered)
                if layout is None:
                    layout={'layers':len(digital[2]),'cache_shape':list(digital[2][0][0].shape),
                            'valid_tokens':int(valid.sum()),
                            'K_dtypes':[str(layer[0].dtype) for layer in digital[2]],
                            'V_dtypes':[str(layer[1].dtype) for layer in digital[2]],
                            'interpolation':'FP32 arithmetic, cast to each original cache dtype; endpoints use native tensors'}
                    assert layout['layers']==18 and layout['valid_tokens']==520
                    print('[layout]',layout,flush=True)
                for cond,v in [('digital',vd),('render',vr)]:
                    before=oldrows[(pattern,cond,int(step))]
                    now=velocity(v,vc)
                    error=max(abs(now[f]-before['velocity'][f]) for f in ['L0','L0_real7','L0_exec7'])
                    checks['old_endpoint_max_error']=max(checks['old_endpoint_max_error'],error)
                    assert error<1e-4,(pattern,cond,int(step),error)
                # Capture endpoint tensors once; denoise_step(use_cache=False)
                # must neither append tokens nor mutate these caches.
                d_pairs=tuple(digital[2]);r_pairs=tuple(rendered[2])
                snapshots=tuple(tuple(t.clone() for t in layer) for layer in d_pairs+r_pairs) if k==0 else None
                path_velocities=[];last_v=None
                for ai,alpha in enumerate(ALPHAS):
                    if alpha==0.: pairs=d_pairs
                    elif alpha==1.: pairs=r_pairs
                    else:
                        pairs=tuple(tuple(torch.lerp(d[j].float(),r[j].float(),alpha).to(d[j].dtype)
                                          for j in [0,1]) for d,r in zip(d_pairs,r_pairs))
                    cache=DynamicCache.from_legacy_cache(pairs)
                    vm=head(clean,cache)
                    if alpha==0.: assert torch.equal(vm,vd)
                    if alpha==1.: assert torch.equal(vm,vr)
                    row={'pattern':pattern,'step':int(step),'alpha':alpha,
                        'kv':kv_stats(pairs,d_pairs,r_pairs,valid,alpha),
                        'velocity_vs_clean':velocity(vm,vc),'velocity_vs_digital':velocity(vm,vd),
                        'velocity_vs_render':velocity(vm,vr),'response':response(vm,vd,vr,alpha),
                        'velocity_vs_previous':velocity(vm,last_v if last_v is not None else vd)}
                    if random_velocities is not None: row['velocity_vs_random']=velocity(vm,random_velocities[ai])
                    records.append(row);path_velocities.append(vm);last_v=vm
                    del cache,pairs
                if snapshots is not None:
                    assert all(torch.equal(a,b) for layer,snapshot in zip(d_pairs+r_pairs,snapshots)
                               for a,b in zip(layer,snapshot))
                    checks['cache_not_mutated']=True
                if pattern=='random': random_velocities=path_velocities
                state_arrays.append(torch.stack(path_velocities).cpu().numpy())
                probes={str(a):round(records[-len(ALPHAS)+ALPHAS.index(a)]['velocity_vs_clean']['L0'],4)
                        for a in [0.,.01,.1,.5,1.]}
                print('[probe]',int(step),pattern,'L0 vs clean',probes,flush=True)
                del digital,rendered,vd,vr,d_pairs,r_pairs,snapshots,path_velocities,vm,last_v
            velocity_arrays.append(np.stack(state_arrays))
            for e,snapshot in zip(noises,noise_copies): assert torch.equal(e,snapshot)
            (dest/'metrics.json').write_text(json.dumps({'complete':False,'frames_completed':k+1,
                'alphas':ALPHAS,'patterns':PATTERNS,'records':records},allow_nan=False))
            del clean,vc,state_arrays,random_velocities
    checks['noise_unchanged']=True
    np.savez_compressed(dest/'velocities.npz',velocity=np.stack(velocity_arrays),clean_velocity=np.stack(clean_arrays),
        action_noise=noise_np,steps=pool['steps'][:n],alphas=np.array(ALPHAS),patterns=np.array(PATTERNS))
    meta={'complete':True,'frames':n,'patterns':PATTERNS,'alphas':ALPHAS,'records':records,'layout':layout,
        'noise_seeds':seeds,'checks':checks,'elapsed_s':time.monotonic()-start,
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'model_checkpoint':str(T.TORCH_CKPT),
        'source':str(source),'source_inputs_sha256':hashlib.sha256((source/'inputs.npz').read_bytes()).hexdigest(),
        'source_metrics_sha256':hashlib.sha256((source/'metrics.json').read_bytes()).hexdigest(),
        'notes':'All 18 layers of K and V jointly interpolate digital to raw render base; wrist/state/language clean. '
                'Each alpha/noise starts fresh from the same epsilon at t=1; no integration or rollout. '
                'Head receives complete caches including padding, distance metrics use valid tokens only. '
                'Interpolated caches are diagnostic interventions, not image-realizable observations. '
                'Global KV norm concatenates all 18 layers; normalized within this path, not semantic retention.'}
    (dest/'metrics.json').write_text(json.dumps(meta,allow_nan=False))
    print('[done]',len(records),'state/pattern/alpha records',round(time.monotonic()-start),'seconds',flush=True)


if __name__=='__main__':main()
