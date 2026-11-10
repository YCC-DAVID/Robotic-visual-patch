"""Directional open-gripper target through the complete native action flow."""
import argparse,time,traceback
from pathlib import Path
from multiprocessing.connection import Listener
import numpy as np
import torch
from e_method_backend import EBackend
from e_method_common import AUTH,load,dump
from crossarch_proxy import images


def open_loss(action,active_start):
    # Both grippers have positive output-transform slope. Normalized +1 maps
    # to q99, approximately physical 1 (fully open), rather than closed 0.
    return -(action[...,active_start+6].float()-1.).square().mean()


class OpenBackend(EBackend):
    def method_gradient(self,req):
        assert req['kind']=='action_gripper_open'
        path=req['record'];seed=int(req['seed'])
        with np.load(path) as z:
            state=z['state14'].copy();prompt=str(z['prompt'])
            active_start=int(z['active_start']) if 'active_start' in z else int(req.get('active_start',0))
        # Stage bank may omit active_start; determine the operating arm from
        # its recorded initial object position, supplied by the immutable plan.
        assert active_start in (0,7)
        q=torch.tensor(req['texture'],device='cuda',requires_grad=True)
        prepared=self.prepare(state,prompt,images(q,path))
        v,a=self.response(prepared,seed,True)
        loss=open_loss(a,active_start);loss.backward()
        assert torch.isfinite(loss) and q.grad is not None and torch.isfinite(q.grad).all()
        assert all(p.grad is None for p in self.model.parameters())
        return dict(loss=float(loss.detach()),gradient=q.grad.float().cpu().numpy(),
                    normalized_gripper_mean=float(a[...,active_start+6].float().mean().detach()))


def main(root):
    backend=OpenBackend(root);backend.audit()
    # Verify the real output transform: normalized +1 opens either gripper.
    values=[]
    for level in (-1.,1.):
        raw=np.full((32,32),level,np.float32)
        physical=backend.policy._output_transform(dict(actions=raw,state=np.zeros(14,np.float32)))['actions']
        values.append(np.asarray(physical)[0,[6,13]].tolist())
    assert all(b>a for a,b in zip(*values)),values
    assert np.allclose(values[1],[1,1],atol=.01),values
    dump(root/'audit/gripper_sign.json',dict(passed=True,normalized_levels=[-1,1],physical_grippers=values))
    with Listener(('127.0.0.1',load(root)['backend_port']),authkey=AUTH) as listener:
        dump(root/'backend_ready.json',dict(ready=True,model='pi'))
        while True:
            try:
                with listener.accept() as c:
                    req=c.recv();command=req.pop('command')
                    try:out=dict(ready=True) if command=='ping' else getattr(backend,command)(req)
                    except Exception as e:out=dict(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
                    c.send(out)
            except (EOFError,ConnectionResetError,BrokenPipeError):traceback.print_exc()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
