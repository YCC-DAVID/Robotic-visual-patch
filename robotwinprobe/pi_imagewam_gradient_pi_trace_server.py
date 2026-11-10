"""Observe the unchanged pi native backward for the same fixed step-1 batch."""
import os,sys,hashlib,argparse
from pathlib import Path
ROOT=Path(os.environ.get('WAMATTACK_PROJECT_ROOT','/shared/user65/workspace/chence/WAMattack'))
sys.path.insert(0,str(ROOT/'robotwinprobe'))
import pi_imagewam_gradient_replay as D

def server(work):
 import torch
 from torch.utils._python_dispatch import TorchDispatchMode
 work=D.guard_work(work)
 original_grad=torch.autograd.grad
 counter=[0]
 def signature(value):
  if isinstance(value,torch.Tensor):
   with torch._C._DisableTorchDispatch():
    a=value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
  if isinstance(value,(list,tuple)):return [signature(v) for v in value]
  if isinstance(value,dict):return {str(k):signature(v) for k,v in value.items()}
  return value if value is None or isinstance(value,(bool,int,float,str)) else repr(value)
 class Observe(torch.utils._python_dispatch.TorchDispatchMode):
  def __init__(self):super().__init__();self.events=[]
  def __torch_dispatch__(self,func,types,args=(),kwargs=None):
   kwargs=kwargs or {};name=str(func)
   record='backward' in name and any(t in name for t in ['grid_sampler','upsample','attention'])
   inputs=signature((args,kwargs)) if record else None
   result=func(*args,**kwargs)
   if record:self.events.append(dict(operator=name,inputs=inputs,outputs=signature(result)))
   return result
 def traced_grad(*args,**kwargs):
  if not (work/'trace_enabled').exists():return original_grad(*args,**kwargs)
  index=counter[0];counter[0]+=1;observer=Observe()
  with observer:result=original_grad(*args,**kwargs)
  D.dump(work/'pi_backward_trace'/f'call_{index:03d}.json',dict(call=index,events=observer.events,final_gradients=signature(result),torch_version=torch.__version__,cuda_version=torch.version.cuda,deterministic_algorithms=torch.are_deterministic_algorithms_enabled()))
  return result
 torch.autograd.grad=traced_grad
 assert D.preserved()==D.read(D.DIAG/'replay_inputs.json')['original_hashes']
 D.J.PORTS=D.PORTS
 D.J.server(work,'pi')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('command',choices=['server']);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--kind',choices=['pi'],required=True);a=ap.parse_args();server(a.work)
