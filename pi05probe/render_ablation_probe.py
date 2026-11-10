"""E18: same-state full/band32/random image-to-KV-to-early-flow ablations."""
import argparse
import dataclasses
import hashlib
import json
import time
import drift_patch_train as T
import numpy as np


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='render_ablation')
    args=ap.parse_args()
    import torch
    import jax
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    torch.set_num_threads(4)
    dest=T.OUT/args.out
    pool=np.load(dest/'inputs.npz',allow_pickle=False)
    old=json.loads((T.OUT/'kv_render_probe/metrics.json').read_text())
    oldrows={(r['patch'],r['step']):r for r in old['records']}
    cfg=C.get_config('pi05_libero')
    cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    policy=PC.create_trained_policy(cfg,T.TORCH_CKPT)
    model,device=policy._model,policy._pytorch_device
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    seeds=[20260901,20260902,20260903]
    noises=[torch.from_numpy(np.random.RandomState(s).standard_normal((T.AH,T.AD)).astype(np.float32))[None].to(device) for s in seeds]
    features=[]
    handle=model.paligemma_with_expert.paligemma.model.multi_modal_projector.register_forward_hook(
        lambda module,inputs,output:features.append(output.detach().float()))
    def obs(image,k):
        x=policy._input_transform({'observation/image':image,'observation/wrist_image':pool['clean_wrist'][k],
            'observation/state':pool['state8'][k],'prompt':str(pool['prompt'])})
        x=jax.tree.map(lambda a:torch.from_numpy(np.array(a)).to(device)[None],x)
        return M.Observation.from_dict(x)
    def forward(image,k):
        features.clear()
        o=obs(image,k)
        pack=T.prefix_and_cache(model,device,o)
        feature=features[list(o.images).index(T.BASE_KEY)]
        vv=torch.stack([model.denoise_step(pack[0],pack[1],pack[2],e.clone(),torch.ones(1,device=device)) for e in noises])
        valid=pack[1][0].bool()
        cache=[tuple(t[:,:,valid].float() for t in layer) for layer in pack[2]]
        return {'features':feature,'kv':cache,'velocity':vv},valid
    def norm(x): return float(x.norm())
    def cosine(a,b):
        den=norm(a)*norm(b)
        return float((a*b).sum())/den if den>1e-12 else None
    def delta_stats(delta,reference):
        nr=norm(reference); nd=norm(delta)
        return {'norm_ratio':nd/nr if nr>1e-12 else None,'delta_cosine':cosine(delta,reference),
                'parallel_gain':float((delta*reference).sum())/(nr*nr) if nr>1e-12 else None,
                'residual_ratio':norm(delta-reference)/nr if nr>1e-12 else None}
    def velocity(a,b):
        d=a-b
        return {'L0':float(d.square().flatten(1).sum(1).mean()),
                'L0_real7':float(d[...,:7].square().flatten(1).sum(1).mean()),
                'L0_exec7':float(d[:,:, :5,:7].square().flatten(1).sum(1).mean()),
                'per_noise_L0':d.square().flatten(1).sum(1).cpu().tolist()}
    records=[]; start=time.monotonic();checks={'anchors_match_E15':True}
    conditions=pool['conditions'].tolist()
    try:
        with torch.no_grad():
            for k,step in enumerate(pool['steps']):
                clean,valid=forward(pool['clean_base'][k],k)
                if k==0:
                    repeat,repeat_valid=forward(pool['clean_base'][k],k)
                    assert torch.equal(clean['velocity'],repeat['velocity'])
                    assert all(torch.equal(a,b) for x,y in zip(clean['kv'],repeat['kv']) for a,b in zip(x,y))
                    checks['repeat_clean_equal']=True
                randoms={}
                for pattern in ['random','full','band32']:
                    packs={}
                    for cond in conditions:
                        target,v=forward(pool[pattern+'__'+cond][k],k)
                        assert torch.equal(valid,v)
                        packs[cond]=target
                        digital=packs['digital']
                        layers=[]
                        for i,(c,t,d) in enumerate(zip(clean['kv'],target['kv'],digital['kv'])):
                            item={'layer':i}
                            for j,label in enumerate(['K','V']):
                                delta=t[j]-c[j];ref=d[j]-c[j]
                                item[label]={'relative_clean':norm(delta)/max(norm(c[j]),1e-12),
                                             'relative_to_digital':norm(t[j]-d[j])/max(norm(d[j]),1e-12),
                                             **delta_stats(delta,ref)}
                            layers.append(item)
                        fdelta=target['features']-clean['features']
                        fref=digital['features']-clean['features']
                        row={'pattern':pattern,'condition':cond,'step':int(step),
                            'feature':{'relative_clean':norm(fdelta)/max(norm(clean['features']),1e-12),**delta_stats(fdelta,fref)},
                            'kv':layers,'velocity':velocity(target['velocity'],clean['velocity']),
                            'velocity_vs_digital':velocity(target['velocity'],digital['velocity'])}
                        if cond in ['digital','render']:
                            tag='band32' if pattern=='band32' else 'drift'
                            oldcond=('random_' if pattern=='random' else '')+('digital' if cond=='digital' else 'render_base')
                            previous=oldrows[(tag,int(step))]['conditions'][oldcond]
                            err=abs(row['velocity']['L0']-np.mean([r['L0'] for r in previous['velocity']]))
                            assert err<1e-4,(pattern,cond,int(step),err)
                        if pattern!='random':
                            r=randoms[cond];rd=randoms['digital']
                            row['velocity_vs_random']=velocity(target['velocity'],r['velocity'])
                            identity=[]
                            for i,(tk,rk,dk,rdk) in enumerate(zip(target['kv'],r['kv'],digital['kv'],rd['kv'])):
                                identity.append({'layer':i,**{label:delta_stats(tk[j]-rk[j],dk[j]-rdk[j]) for j,label in enumerate(['K','V'])}})
                            row['optimized_random_kv_alignment']=identity
                        records.append(row)
                    if pattern=='random': randoms=packs
                    print('[probe]',int(step),pattern,{r['condition']:round(r['velocity']['L0'],4) for r in records[-len(conditions):]},flush=True)
                (dest/'metrics.json').write_text(json.dumps({'complete':False,'records':records},allow_nan=False))
    finally:
        handle.remove()
    metadata={'complete':True,'frames':len(pool['steps']),'conditions':conditions,'records':records,'checks':checks,
        'noise_seeds':seeds,'elapsed_s':time.monotonic()-start,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
        'inputs_sha256':hashlib.sha256((dest/'inputs.npz').read_bytes()).hexdigest(),
        'notes':'Base-only changes, wrist/state/language clean. 18-layer valid prefix KV, post-RoPE K. '
                'L0 is first flow velocity at t=1, averaged over 3 common noises. '
                'exec7 is first 5 steps x real 7 dimensions of velocity, not final actions. '
                'No training, integration or rollout. Same 16 states, one trajectory, one random pattern.'}
    (dest/'metrics.json').write_text(json.dumps(metadata,allow_nan=False))
    print('[done]',len(records),round(time.monotonic()-start),'seconds',flush=True)


if __name__=='__main__':main()
