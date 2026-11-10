"""Check control equivalence, smooth gradients and random-texture fidelity."""
import json
from pathlib import Path
import numpy as np
import torch
from proxy_factorial_proxy import WorldTextureProxy, VARIANTS
from render_world_proxy import WorldTextureProxy as Original


def main():
    torch.set_num_threads(4)
    dest=Path(__file__).resolve().parent/'out/proxy_factorial'
    source=dest/'calibration.npz'
    fit=dest.parent/'render_proxy_scale/fit_13_101/fit.json'
    data=np.load(source)
    q=torch.from_numpy(data['random__texture'].astype(np.float32).transpose(2,0,1)/255)[None]
    old=Original(source,'cpu',fit); full=WorldTextureProxy(source,'cpu',fit,'111')
    ids=np.flatnonzero(data['validation_mask'])
    equal=all(torch.equal(old(q,int(k)), full(q,int(k))) for k in ids)
    assert equal, 'full factorial control must match original forward exactly'
    torch.manual_seed(3106)
    latent=(torch.rand((1,3,128,128))*.6+.2).requires_grad_(True)
    direction=torch.randn_like(latent);direction/=direction.norm()
    checks={}
    for variant in VARIANTS:
        proxy=WorldTextureProxy(source,'cpu',fit,variant)
        def score(x): return proxy(x,155,quantize=False).square().mean()
        g,=torch.autograd.grad(score(latent),latent)
        h=.02
        numeric=float((score(latent+h*direction)-score(latent-h*direction))/(2*h))
        analytic=float((g*direction).sum())
        error=abs(numeric-analytic)/max(abs(numeric),abs(analytic),1e-9)
        # Float32 output reduction may lose precision; use a directional
        # linear functional in float64 for a decisive finite-difference check.
        probe=torch.randn_like(proxy(latent,155,quantize=False)).double()
        def linear(x):return (proxy(x,155,quantize=False).double()*probe).sum()
        gd,=torch.autograd.grad(linear(latent),latent)
        nd=float((linear(latent+h*direction)-linear(latent-h*direction))/(2*h))
        ad=float((gd*direction).sum())
        rel=abs(nd-ad)/max(abs(nd),abs(ad),1e-8)
        assert rel < .02, (variant,nd,ad,rel)
        pred=np.array([np.rint(proxy(q,int(k)).detach().numpy()[0].transpose(1,2,0)*255).clip(0,255).astype(np.uint8) for k in ids])
        # PIL production resize, not the differentiable resize being tested.
        from PIL import Image
        real=np.array([np.asarray(Image.fromarray(data['random256'][k]).resize((224,224),Image.Resampling.BILINEAR)) for k in ids])
        delta=(pred.astype(float)-real.astype(float))/255
        errors=[float(np.sqrt((d[m]**2).mean())) for d,m in zip(delta,data['mask'][ids])]
        checks[variant]={'gradient_relative_error':rel,'random_mask_RMSE':float(np.mean(errors)),
                         'per_frame_RMSE':errors,'gradient_finite':bool(torch.isfinite(gd).all())}
        print('[check]',variant,checks[variant]['random_mask_RMSE'],rel,flush=True)
    report={'complete':True,'full_forward_exact_original':equal,'variants':checks,
            'notes':'Fixed random Q,32 correlated frames from two episodes. Continuous derivative check excludes uint8 STE. Photometry includes black/gain scene response; measured coverage remains when bypassed.'}
    (dest/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
