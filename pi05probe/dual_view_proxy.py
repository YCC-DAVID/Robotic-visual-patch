"""Per-frame camera proxy; geometry evaluated in float64 independent of model TF32."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from render_world_proxy import ste_u8


class CameraProxy:
    def __init__(self,calibration,device='cpu',fit=None):
        d=np.load(calibration,allow_pickle=False)
        self.H=torch.as_tensor(d['image256_to_texture_pixel_H'],dtype=torch.float64,device=device)
        self.black=torch.from_numpy(d['black256'].transpose(0,3,1,2).astype(np.float32)/255).to(device)
        self.gain=(torch.from_numpy(d['white256'].transpose(0,3,1,2).astype(np.float32)/255).to(device)-self.black).clamp_min(0)
        yy,xx=torch.meshgrid(torch.arange(256,device=device),torch.arange(256,device=device),indexing='ij')
        self.xy=torch.stack([xx,yy],-1).double();self.cache={}
        self.config={'dx_native_px':0.,'dy_native_px':0.,'lod_bias':0.}
        if fit:self.config.update(json.loads(Path(fit).read_text())['parameters'])

    def grid(self,k,dx=0.,dy=0.):
        dynamic=isinstance(dx,torch.Tensor) or isinstance(dy,torch.Tensor)
        key=(int(k),float(dx),float(dy)) if not dynamic else None
        if key in self.cache:return self.cache[key]
        xy=self.xy+torch.stack([torch.as_tensor(dx,device=self.xy.device,dtype=torch.float64),torch.as_tensor(dy,device=self.xy.device,dtype=torch.float64)])
        H=self.H[k];p=torch.cat([xy,torch.ones_like(xy[...,:1])],-1)@H.T
        z=p[...,2:];z=torch.where(z.abs()<1e-9,torch.where(z>=0,1e-9,-1e-9),z)
        uv=p[...,:2]/z
        du=(H[:2,0]-uv*H[2,0])/z;dv=(H[:2,1]-uv*H[2,1])/z
        lod=torch.maximum(du.square().sum(-1).sqrt(),dv.square().sum(-1).sqrt()).clamp_min(1e-6).log2().clamp_min(0)
        grid=(2*(uv+.5)/128-1)[None]
        if not dynamic:self.cache[key]=(grid,lod)
        return grid,lod

    def __call__(self,q,k,parameters=None,quantize=True,native=False):
        if parameters is None:parameters=[self.config[key] for key in ['dx_native_px','dy_native_px','lod_bias']]
        grid,lod=self.grid(k,*parameters[:2]);grid=grid.to(q.dtype);lod=(lod+parameters[2]).clamp(0,7).to(q.dtype)
        current=ste_u8(q) if quantize else q;out=torch.zeros((1,3,256,256),device=q.device,dtype=q.dtype)
        for level in range(8):
            weight=(1-(lod-level).abs()).clamp_min(0)[None,None]
            out=out+F.grid_sample(current,grid,mode='bilinear',padding_mode='border',align_corners=False)*weight
            if level<7:
                current=F.avg_pool2d(current,2)
                if quantize:current=ste_u8(current)
        image=(self.black[k:k+1]+self.gain[k:k+1]*out).clamp(0,1)
        if quantize:image=ste_u8(image)
        if native:return image
        image=F.interpolate(image,(224,224),mode='bilinear',align_corners=False,antialias=True)
        return ste_u8(image) if quantize else image


def lowpass(q):
    return F.interpolate(F.avg_pool2d(q,4),(128,128),mode='bilinear',align_corners=False)


def base_gradient_route(q):
    """Identity forward; backward is the adjoint of the declared32-grid lowpass.

    A surrogate optimization direction, not the true derivative of identity.
    The joint PGD update and final Q are NOT claimed to be bandlimited.
    """
    smooth=lowpass(q)
    return q.detach()+(smooth-smooth.detach())
