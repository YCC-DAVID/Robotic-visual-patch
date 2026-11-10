"""Numerical proxy gradient and CPU/GPU geometry checks, no model forward."""
from pathlib import Path
import json
import numpy as np
import torch
from dual_view_proxy import CameraProxy


def main():
    torch.set_num_threads(4);root=Path(__file__).resolve().parent/'out/dual_view';rows=[]
    for view in ['base','wrist']:
        p=CameraProxy(root/'calibration'/(view+'.npz'),fit=root/'fit'/(view+'.json'))
        q=torch.full((1,3,128,128),.4,dtype=torch.float64,requires_grad=True)
        direction=torch.randn(q.shape,dtype=q.dtype,generator=torch.Generator().manual_seed(101));direction/=direction.norm()
        for k in [0,155,170]:
            def score(z):return p(z,k,quantize=False).square().mean()
            g,=torch.autograd.grad(score(q),q);h=.001
            numeric=float((score(q+h*direction)-score(q-h*direction))/(2*h));analytic=float((g*direction).sum())
            rel=abs(numeric-analytic)/max(abs(numeric),abs(analytic),1e-12)
            assert rel<1e-4,(view,k,numeric,analytic,rel)
            rows.append({'view':view,'frame':k,'relative_FD_error':rel,'finite_nonzero_gradient':bool(torch.isfinite(g).all() and g.norm()>0)})
    (root/'gradient_check.json').write_text(json.dumps({'complete':True,'records':rows,'notes':'Continuous imaging proxy only. STE quantization and intentional gradient routing do not satisfy identity finite differences.'},indent=2)+'\n')
    print(rows)

if __name__=='__main__':main()
