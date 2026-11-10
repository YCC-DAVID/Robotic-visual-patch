"""Fit only the independent random anchor and validate physical proxy pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from render_world_proxy import WorldTextureProxy


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--calibration',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(4)
    source=Path(args.calibration);dest=Path(args.out);dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'fit.json').exists()
    d=np.load(source,allow_pickle=False);proxy=WorldTextureProxy(source,samples=1)
    train=np.flatnonzero(d['train_mask'])[:8];heldout=np.flatnonzero(d['validation_mask'])
    q=torch.from_numpy(d['random__texture'].transpose(2,0,1).astype(np.float32)/255)[None]
    target=torch.from_numpy(d['random256'][train].transpose(0,3,1,2).astype(np.float32)/255)
    black,gain=proxy.black[train],proxy.gain[train]
    valid=torch.from_numpy((d['unlit_white256'][train].min(-1)==255).astype(np.float32))[:,None]
    assert valid.sum()>0
    parameters=torch.zeros(3,requires_grad=True);opt=torch.optim.Adam([parameters],lr=.02);history=[]
    for it in range(100):
        opt.zero_grad();texture=proxy.texture_native(q,*parameters,quantize=False)
        loss=(((black+gain*texture-target)**2)*valid).sum()/(valid.sum()*3)
        loss.backward();opt.step()
        with torch.no_grad():parameters.clamp_(-1,1)
        history.append(float(loss))
    config={'complete':True,'schema':'wam-world-render-proxy-v2','parameters':dict(zip(['dx_native_px','dy_native_px','lod_bias'],parameters.detach().tolist()))|{'samples':1},
        'fit_frames':train.tolist(),'heldout_frames':heldout.tolist(),'calibration_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'pattern':'independent world random Q128 seed20261007 only','history':history}
    proxy.config.update(config['parameters'])
    errors=[];grayerrors=[]
    with torch.no_grad():
        for k in heldout:
            pred=proxy(q,int(k)).numpy()[0].transpose(1,2,0)
            rendered=np.asarray(Image.fromarray(d['random256'][k]).resize((224,224),Image.Resampling.BILINEAR)).astype(float)/255
            mask=d['mask'][k]
            errors.append(float(np.sqrt(((pred[mask]-rendered[mask])**2).mean())))
            gray=proxy(torch.full_like(q,128/255),int(k),native=True).numpy()[0].transpose(1,2,0)
            valid=d['unlit_white256'][k].min(-1)==255
            grayerrors.append(float(np.sqrt(((gray[valid]-d['gray128_256'][k][valid]/255)**2).mean())))
    for name in ['H','black','gain','xy']:setattr(proxy,name,getattr(proxy,name).double())
    latent=torch.full((1,3,32,32),.4,dtype=torch.float64,requires_grad=True)
    direction=torch.randn(latent.shape,dtype=torch.float64,generator=torch.Generator().manual_seed(22));direction/=direction.norm()
    def score(z):
        q=torch.nn.functional.interpolate(z,(128,128),mode='bilinear',align_corners=False)
        return proxy(q,0,quantize=False).square().mean()
    gradient,=torch.autograd.grad(score(latent),latent);h=.001
    numeric=float((score(latent+h*direction)-score(latent-h*direction))/(2*h));analytic=float((gradient*direction).sum())
    relative=abs(numeric-analytic)/max(abs(numeric),abs(analytic),1e-10)
    assert relative<.02,(numeric,analytic,relative)
    check={'complete':True,'heldout_random_mask_RMSE':float(np.mean(errors)),'per_frame_random_mask_RMSE':errors,
        'heldout_gray128_affine_RMSE_native':float(np.mean(grayerrors)),
        'band32_continuous_gradient_check':{'numeric':numeric,'analytic':analytic,'relative_error':relative},
        'notes':'Low-resolution latent through continuous proxy finite difference. Quantization STE gradients remain approximations.'}
    (dest/'fit.json').write_text(json.dumps(config,indent=2)+'\n');(dest/'pixel_check.json').write_text(json.dumps(check,indent=2)+'\n')
    print('[done]',json.dumps(check),flush=True)


if __name__=='__main__':main()
