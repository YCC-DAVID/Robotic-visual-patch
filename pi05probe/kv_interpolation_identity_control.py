"""E20 numerical control: reconstruct unchanged native KV and compare head."""
import dataclasses
import json
import drift_patch_train as T
import numpy as np


def main():
    import torch,jax
    from transformers.cache_utils import DynamicCache
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    torch.set_num_threads(4)
    pool=np.load(T.OUT/'render_additive/inputs.npz',allow_pickle=False)
    cfg=C.get_config('pi05_libero')
    cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    policy=PC.create_trained_policy(cfg,T.TORCH_CKPT);model,device=policy._model,policy._pytorch_device
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    noises=[torch.from_numpy(np.random.RandomState(s).standard_normal((10,32)).astype(np.float32))[None].to(device)
            for s in [20260901,20260902,20260903]]
    records=[]
    with torch.no_grad():
        for step in [10,60,85]:
            k=pool['steps'].tolist().index(step)
            for pattern in ['full','band32','random']:
                x=policy._input_transform({'observation/image':pool[pattern+'__digital'][k],
                    'observation/wrist_image':pool['clean_wrist'][k],
                    'observation/state':pool['state8'][k],'prompt':str(pool['prompt'])})
                x=jax.tree.map(lambda a:torch.from_numpy(np.array(a)).to(device)[None],x)
                state,mask,original=T.prefix_and_cache(model,device,M.Observation.from_dict(x))
                def head(cache):
                    return torch.stack([model.denoise_step(state,mask,cache,e.clone(),torch.ones(1,device=device)) for e in noises])
                ref=head(original)
                row={'step':step,'pattern':pattern,'cache_controls':{}}
                for label,fn in [('clone',lambda t:t.clone()),
                                 ('float32_roundtrip',lambda t:t.float().to(t.dtype)),
                                 ('zero_lerp',lambda t:torch.lerp(t.float(),t.float(),0.).to(t.dtype))]:
                    pairs=tuple(tuple(fn(t) for t in layer) for layer in original)
                    assert all(torch.equal(a,b) for x,y in zip(original,pairs) for a,b in zip(x,y))
                    result=head(DynamicCache.from_legacy_cache(pairs))
                    difference=result-ref
                    row['cache_controls'][label]={'cache_equal':True,'velocity_equal':bool(torch.equal(result,ref)),
                        'max_abs_velocity_error':float(difference.abs().max()),'L0_difference':float(difference.square().flatten(1).sum(1).mean())}
                    assert torch.equal(result,ref),(step,pattern,label,row['cache_controls'][label])
                records.append(row)
                print('[control]',step,pattern,'all unchanged-KV reconstructions equal',flush=True)
    result={'complete':True,'records':records,'all_unchanged_cache_velocities_equal':True,
            'notes':'Three unchanged-KV memory/arithmetic reconstruction controls, 3 states × 3 patterns × 3 paired noises. '
                    'This checks reconstruction identity, not an FP32-model sensitivity experiment.'}
    (T.OUT/'kv_interpolation/identity_control.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
