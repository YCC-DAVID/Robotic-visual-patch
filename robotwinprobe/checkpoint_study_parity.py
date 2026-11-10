"""Three shared observations/noises: native JAX vs exported Torch response."""
import argparse
import os
from pathlib import Path
import numpy as np
import checkpoint_study_common as S


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    if args.backend=='jax':
        import checkpoint_study_ft as F
        from checkpoint_study_data import Data
        import jax
        import jax.numpy as jnp
        from openpi.models import model as M
        from openpi.shared import nnx_utils
        cfg=F.make_config(args.root,'parity',202610230,3 if args.smoke else 20000,args.smoke)
        data=Data(args.root,cfg,args.smoke)
        model=cfg.model.load(M.restore_params(args.checkpoint/'params',dtype=jnp.bfloat16));model.eval()
        infer=nnx_utils.module_jit(model.sample_actions)
        indices=np.array([0,len(data.states)//2,len(data.states)-1])
        for i,idx in enumerate(indices):
            obs,_=data.batch([idx]);obs=jax.tree.map(jnp.asarray,obs)
            noise=np.random.default_rng(202610280+i).standard_normal((1,32,32)).astype(np.float32)
            one=np.asarray(infer(jax.random.key(0),obs,noise=jnp.asarray(noise),num_steps=1))
            ten=np.asarray(infer(jax.random.key(0),obs,noise=jnp.asarray(noise),num_steps=10))
            np.savez_compressed(args.out/f'reference_{i}.npz',
                **{f'image_{k}':np.asarray(v) for k,v in obs.images.items()},
                state=np.asarray(obs.state),tokens=np.asarray(obs.tokenized_prompt),
                token_mask=np.asarray(obs.tokenized_prompt_mask),noise=noise,v=noise-one,action=ten)
            print('[jax reference]',idx,flush=True)
    else:
        import checkpoint_study_model as C
        import torch
        from openpi.models import model as M
        policy=C.load(args.checkpoint);rows=[]
        def guard(*_):raise AssertionError('Unused vocabulary head was called')
        policy._model.paligemma_with_expert.gemma_expert.lm_head.register_forward_pre_hook(guard)
        for file in sorted(args.out.glob('reference_*.npz')):
            d=dict(np.load(file));tensor=lambda v:torch.tensor(v,device='cuda')
            images={k[6:]:tensor(v).permute(0,3,1,2) for k,v in d.items() if k.startswith('image_')}
            obs=M.Observation(images=images,image_masks={k:torch.ones(1,dtype=torch.bool,device='cuda') for k in images},
                state=tensor(d['state']),tokenized_prompt=tensor(d['tokens']),tokenized_prompt_mask=tensor(d['token_mask']))
            with torch.no_grad():
                v,action,_=C.response(policy,obs,tensor(d['noise']))
                native=policy._model.sample_actions('cuda',obs,noise=tensor(d['noise']),num_steps=10).float().cpu().numpy()
            row=dict(reference=file.name)
            for name,value in [('v',v),('action',action)]:
                row[name+'_relative_l2']=float(np.linalg.norm(value-d[name])/max(np.linalg.norm(d[name]),1e-9))
            row['manual_native_max_abs']=float(abs(action-native).max())
            rows.append(row)
        assert len(rows)==3
        passed=all(r['v_relative_l2']<.05 and r['action_relative_l2']<.1 and r['manual_native_max_abs']<1e-4 for r in rows)
        S.dump(args.out/'parity.json',dict(passed=passed,rows=rows,
            gate='each of three: v relL2<.05, full action relL2<.1, manual/native maxabs<1e-4'))
        print('[parity]',passed,rows,flush=True)
        assert passed,'Export/integration fidelity gate failed'


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=['jax','torch'],required=True)
    ap.add_argument('--root',type=Path,default=S.DEFAULT);ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--smoke',action='store_true')
    main(ap.parse_args())
