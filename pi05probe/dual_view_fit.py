"""Independent random-only camera fitting, heldout random+smooth pixel checks."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from PIL import Image
from dual_view_proxy import CameraProxy,base_gradient_route,lowpass


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent/'out/dual_view')
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    torch.set_num_threads(4);root=args.root;dest=root/'fit';dest.mkdir(exist_ok=True)
    assert not (dest/'checks.json').exists()
    results={}
    for view in ['base','wrist']:
        source=root/'calibration'/(view+'.npz')
        # NPZ indexing inflates the whole compressed array each time. Keep the
        # fixed anchors in memory instead of decompressing them 800 times/view.
        with np.load(source,allow_pickle=False) as archive:
            d={key:archive[key] for key in ['train_mask','validation_mask','mask',
                'random__texture','check__texture','random256','check256','unlit_white256']}
        p=CameraProxy(source,device=args.device)
        train=np.flatnonzero(d['train_mask']);visible=d['mask'].sum((1,2))
        candidates=train[visible[train]>=100];ids=candidates[np.linspace(0,len(candidates)-1,8,dtype=int)]
        q=torch.from_numpy(d['random__texture'].transpose(2,0,1).astype(np.float32)/255)[None].to(args.device)
        params=torch.zeros(3,device=args.device,requires_grad=True);opt=torch.optim.Adam([params],lr=.02);history=[]
        for it in range(100):
            opt.zero_grad();loss=0
            for k in ids:
                target=torch.from_numpy(d['random256'][k].transpose(2,0,1).astype(np.float32)/255)[None].to(args.device)
                mask=torch.from_numpy((d['unlit_white256'][k].min(-1)==255).astype(np.float32))[None,None].to(args.device)
                diff=p(q,int(k),params,quantize=False,native=True)-target
                loss=loss+(diff.square()*mask).sum()/(mask.sum().clamp_min(1)*3*len(ids))
            loss.backward();opt.step()
            with torch.no_grad():params.clamp_(-1,1)
            history.append(float(loss))
        fit={'complete':True,'parameters':dict(zip(['dx_native_px','dy_native_px','lod_bias'],params.detach().tolist())),
             'fit_frames':ids.tolist(),'history':history,'calibration_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'texture':'random seed20261007 only','geometry_precision':'float64 each camera/state'}
        (dest/(view+'.json')).write_text(json.dumps(fit,indent=2)+'\n');p.config.update(fit['parameters'])
        rows=[]
        with torch.no_grad():
            for k in np.flatnonzero(d['validation_mask']):
                for pattern,field in [('random','random256'),('check','check256')]:
                    q=torch.from_numpy(d[pattern+'__texture'].transpose(2,0,1).astype(np.float32)/255)[None].to(args.device)
                    pred=p(q,int(k)).cpu().numpy()[0].transpose(1,2,0)
                    ref=np.array(Image.fromarray(d[field][k]).resize((224,224),Image.Resampling.BILINEAR)).astype(float)/255
                    mask=d['mask'][k];err=float(np.sqrt(((pred[mask]-ref[mask])**2).mean())) if mask.any() else None
                    rows.append({'frame':int(k),'pattern':pattern,'visible':int(mask.sum()),'mask_RMSE':err,'image_RMSE':float(np.sqrt(((pred-ref)**2).mean()))})
        results[view]={'parameters':fit['parameters'],'records':rows,'mask_RMSE_mean':float(np.mean([r['mask_RMSE'] for r in rows if r['mask_RMSE'] is not None]))}
        print(view,results[view]['mask_RMSE_mean'],fit['parameters'],flush=True)
    # Exact gradient-routing identity and adjoint checks, distinct from imaging FD.
    gen=torch.Generator().manual_seed(123);q=torch.rand((1,3,128,128),generator=gen,requires_grad=True);w=torch.randn(q.shape,generator=gen)
    routed=base_gradient_route(q);assert torch.equal(routed,q)
    a,=torch.autograd.grad((routed*w).sum(),q);b,=torch.autograd.grad((lowpass(q)*w).sum(),q)
    assert torch.equal(a,b)
    results['routing_check']={'identity_forward_exact':True,'backward_equals_lowpass_adjoint':True,'not_a_strict_bandlimit':True}
    (dest/'checks.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
