"""Own pi0.5 service; gradients through the complete ten-step action flow."""
import argparse
from collections import OrderedDict
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from crossarch_backend import Backend
from e_method_common import DEFAULT, PORT, AUTH, OBJECTIVES, ALL_OBJECTIVES, dump, load


def fixed_scales(root):
    scales=json.loads((root/'refinement/protocol.json').read_text())['fixed_training_random_scales']
    assert all(np.isfinite(scales[k]) and scales[k]>0 for k in ('arm6','gripper'))
    return scales


def objective(v,a,vc,ac,kind,active_start=0,scales=None):
    if kind=='velocity32':return (v.float()-vc.float()).square().mean()
    if kind=='velocity14':return (v[...,:14].float()-vc[...,:14].float()).square().mean()
    if kind=='action14':return (a[...,:14].float()-ac[...,:14].float()).square().mean()
    if kind=='action_active7':
        indices=slice(active_start,active_start+7)
        return (a[...,indices].float()-ac[...,indices].float()).square().mean()
    if kind in ('velocity_gripper','velocity_joint_balanced'):
        dv=v.float()-vc.float()
        gripper=dv[...,active_start+6].square().mean()
        if kind=='velocity_gripper':return gripper
        arm=dv[...,active_start:active_start+6].square().mean()
        assert scales and scales['arm6']>0 and scales['gripper']>0
        return .5*(arm/scales['arm6']+gripper/scales['gripper'])
    raise ValueError(kind)


class EBackend(Backend):
    def __init__(self,root):
        super().__init__('pi',root)
        self.method_cache=OrderedDict()

    def response(self,prepared,seed,full=False):
        x=self.noise(seed)
        state,pad,kv=self.M.prefix(self.model,prepared)
        first=None
        t=torch.tensor(1.,device=x.device,dtype=torch.float32)
        dt=torch.tensor(-.1,device=x.device,dtype=torch.float32)
        for _ in range(10 if full else 1):
            # Explicit timestep input avoids a late-bound loop-time closure.
            def step(xx,tt):return self.model.denoise_step(state,pad,kv,xx,tt)
            if full and torch.is_grad_enabled():
                v=checkpoint(step,x,t.expand(1),use_reentrant=False)
            else:v=step(x,t.expand(1))
            if first is None:first=v
            x=x+dt*v;t=t+dt
        return first,x

    def method_gradient(self,req):
        from crossarch_proxy import images
        started=time.monotonic();kind=req['kind'];assert kind in ALL_OBJECTIVES
        path=req['record'];seed=int(req['seed']);full=kind.startswith('action')
        with np.load(path) as z:
            state=z['state14'].copy();prompt=str(z['prompt'])
            active_start=int(z['active_start']) if 'active_start' in z else int(req.get('active_start',0))
        key=(path,seed,full)
        if key not in self.method_cache:
            with torch.no_grad():v,a=self.response(self.prepare(state,prompt,images(None,path)),seed,full)
            self.method_cache[key]=(v.float().cpu(),a.float().cpu())
            if len(self.method_cache)>4096:self.method_cache.popitem(last=False)
        vc,ac=(x.to('cuda') for x in self.method_cache[key])
        q=torch.tensor(np.asarray(req['texture']),device='cuda',requires_grad=True)
        v,a=self.response(self.prepare(state,prompt,images(q,path)),seed,full)
        scales=None
        if kind=='velocity_joint_balanced':
            scales=fixed_scales(self.root)
        loss=objective(v,a,vc,ac,kind,active_start,scales)
        loss.backward()
        assert torch.isfinite(loss) and q.grad is not None and torch.isfinite(q.grad).all()
        result=dict(loss=float(loss.detach()),gradient=q.grad.float().cpu().numpy(),
            first14_mse=float((v[...,:14].float()-vc[...,:14]).square().mean().detach()),
            first32_mse=float((v.float()-vc).square().mean().detach()),
            active_arm6_mse=float((v[...,active_start:active_start+6].float()-vc[...,active_start:active_start+6]).square().mean().detach()),
            active_gripper_mse=float((v[...,active_start+6].float()-vc[...,active_start+6]).square().mean().detach()),
            action14_mse=float((a[...,:14].float()-ac[...,:14]).square().mean().detach()) if full else None,
            gradient_norm=float(q.grad.norm()),seconds=time.monotonic()-started,
            peak_GiB=torch.cuda.max_memory_allocated()/2**30)
        return result

    def method_audit(self,req):
        path=req['record'];texture=np.full((1,3,128,128),.5,np.float32)
        results={}
        for kind in OBJECTIVES:
            r=self.method_gradient(dict(record=path,seed=42,kind=kind,texture=texture))
            assert r['gradient_norm']>0,(kind,r)
            results[kind]={k:v for k,v in r.items() if k!='gradient'}
        # The checkpointed differentiable path must match the native complete flow.
        from crossarch_proxy import images
        with np.load(path) as z:state=z['state14'].copy();prompt=str(z['prompt'])
        q=torch.tensor(texture,device='cuda',requires_grad=True)
        v,a=self.response(self.prepare(state,prompt,images(q,path)),42,True)
        with torch.no_grad():
            raw=self.prepare(state,prompt,images(q,path))
            native=self.model.sample_actions('cuda',raw,noise=self.noise(42),num_steps=10)
        error=float((a.detach()-native).abs().max());assert error<1e-4,error
        out=dict(passed=True,gradient_modes=results,checkpointed_native_action_max_error=error,
            policy_parameter_gradients=sum(x.grad is not None for x in self.model.parameters()))
        assert out['policy_parameter_gradients']==0
        dump(self.root/'audit/method_gradients.json',out)
        return out


def main(root):
    backend=EBackend(root);backend.audit()
    with Listener(('127.0.0.1',load(root)['backend_port']),authkey=AUTH) as listener:
        dump(root/'backend_ready.json',dict(ready=True,model='pi',port=PORT))
        while True:
            try:
                with listener.accept() as connection:
                    req=connection.recv();command=req.pop('command')
                    try:
                        result=dict(ready=True,model='pi') if command=='ping' else getattr(backend,command)(req)
                    except Exception as e:
                        result=dict(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
                        if isinstance(e,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
                    connection.send(result)
            except (EOFError,ConnectionResetError,BrokenPipeError):
                traceback.print_exc()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=DEFAULT)
    main(parser.parse_args().root.resolve())
