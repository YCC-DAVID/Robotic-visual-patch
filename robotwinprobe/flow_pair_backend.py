"""Same-state flow traces from true RGB; no training or surrogate images."""
import argparse,traceback
from pathlib import Path
from multiprocessing.connection import Listener
import numpy as np
import torch
from e_method_backend import EBackend
from e_method_common import AUTH,load,dump

class FlowBackend(EBackend):
    @torch.no_grad()
    def flow_trace(self,req):
        prepared=self.raw_prepared(req['state'],req['prompt'],req['images'])
        state,pad,kv=self.M.prefix(self.model,prepared)
        x=self.noise(int(req['seed']));xs=[x.cpu().numpy()];vs=[]
        t=torch.tensor(1.,device=x.device);dt=torch.tensor(-.1,device=x.device)
        for _ in range(10):
            v=self.model.denoise_step(state,pad,kv,x,t.expand(1));vs.append(v.float().cpu().numpy())
            x=x+dt*v;t=t+dt;xs.append(x.float().cpu().numpy())
        actions=self.policy._output_transform(dict(actions=x[0].cpu().numpy(),state=prepared.state[0].cpu().numpy()))['actions']
        if not hasattr(self,'trace_audited'):
            native=self.model.sample_actions('cuda',prepared,noise=self.noise(int(req['seed'])),num_steps=10)
            error=float((native-x).abs().max());assert error<1e-4,error
            public=self.infer(req)['actions']
            physical_error=float(np.max(np.abs(public-actions)));assert physical_error<1e-4,physical_error
            dump(self.root/'trace_audit.json',dict(passed=True,flow_native_max_error=error,physical_action_native_max_error=physical_error));self.trace_audited=True
        return dict(x=np.asarray(xs),v=np.asarray(vs),actions=np.asarray(actions))

def main(root):
    backend=FlowBackend(root);backend.audit()
    with Listener(('127.0.0.1',load(root)['backend_port']),authkey=AUTH) as listener:
        dump(root/'backend_ready.json',dict(ready=True))
        while True:
            try:
                with listener.accept() as c:
                    req=c.recv();command=req.pop('command')
                    try:r=getattr(backend,command)(req)
                    except Exception as e:r=dict(error=repr(e),traceback=traceback.format_exc());traceback.print_exc()
                    c.send(r)
            except (EOFError,ConnectionResetError,BrokenPipeError):traceback.print_exc()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
