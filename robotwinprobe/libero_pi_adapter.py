"""Native pi0.5 LIBERO inference and differentiable all-camera first-flow path."""
import dataclasses
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from crossarch_common import ROOT

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path.insert(0,str(ROOT/'pi05probe'))
import drift_patch_train as T


class Backend:
    def __init__(self,root):
        from openpi.training import config as C
        from openpi.policies import policy_config as PC
        cfg=C.get_config('pi05_libero')
        cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
        self.policy=PC.create_trained_policy(cfg,T.TORCH_CKPT)
        self.model=self.policy._model.eval().requires_grad_(False)
        self.device=self.policy._pytorch_device

    def prepare(self,state,prompt,views):
        import jax
        from openpi.models import model as M
        if len(views)!=2:raise ValueError('pi0.5 LIBERO requires base and wrist views')
        native=[]
        for view in views:
            rgb=np.rint(view.detach()[0].permute(1,2,0).float().cpu().numpy()*255).clip(0,255).astype(np.uint8)
            native.append(np.array(Image.fromarray(rgb).resize((224,224),Image.Resampling.BILINEAR)))
        transformed=self.policy._input_transform({'observation/image':native[0],
            'observation/wrist_image':native[1],'observation/state':np.asarray(state,dtype=np.float32),'prompt':str(prompt)})
        transformed=jax.tree.map(lambda a:torch.from_numpy(np.array(a)).to(self.device)[None],transformed)
        obs=M.Observation.from_dict(transformed);images=dict(obs.images)
        for key,view in zip(['base_0_rgb','left_wrist_0_rgb'],views):
            surrogate=F.interpolate(view,(224,224),mode='bilinear',align_corners=False,antialias=True)*2-1
            images[key]=surrogate+(obs.images[key]-surrogate).detach()
        return dataclasses.replace(obs,images=images)

    def noise(self,seed):
        return torch.from_numpy(np.random.RandomState(seed).standard_normal((1,10,32)).astype(np.float32)).to(self.device)

    def response(self,prepared,seed,full=False):
        state,mask,cache=T.prefix_and_cache(self.model,self.device,prepared)
        x=self.noise(seed);t=torch.tensor(1.,device=self.device,dtype=torch.float32)
        dt=torch.tensor(-.1,device=self.device,dtype=torch.float32);first=None
        for _ in range(10 if full else 1):
            v=self.model.denoise_step(state,mask,cache,x,t.expand(1))
            if first is None:first=v
            x=x+dt*v;t=t+dt
        return first,x

    def native_full(self,prepared,seed):
        return self.model.sample_actions(self.device,prepared,noise=self.noise(seed),num_steps=10).float().cpu()

    def infer(self,req):
        raw=req['images']
        native={name:np.array(Image.fromarray(raw[name]).resize((224,224),Image.Resampling.BILINEAR)) for name in ['base','wrist']}
        result=self.policy.infer({'observation/image':native['base'],'observation/wrist_image':native['wrist'],
            'observation/state':np.asarray(req['state'],dtype=np.float32),'prompt':req['prompt']},
            noise=self.noise(req['seed'])[0].cpu().numpy())
        return dict(actions=np.asarray(result['actions']))
