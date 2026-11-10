"""One native-resolution imaging proxy before either policy's preprocessing."""
import numpy as np
import torch
import torch.nn.functional as F
from crossarch_common import CAMERAS
from render_proxy import grid_for,linear,ste_u8

def images(q,record):
    with np.load(record) as z:
        if q is None:
            return [torch.tensor(z[n+'_clean'].copy(),device='cuda').permute(2,0,1)[None].float()/255 for n in CAMERAS]
        views=[];q=ste_u8(q.clamp(0,1))
        for n in CAMERAS:
            def tensor(key): return torch.tensor(z[n+'_'+key].copy(),device=q.device).permute(2,0,1)[None].float()/255
            g=grid_for(z[n+'_intrinsic'],z[n+'_extrinsic'],z['center_m'],float(z['size_m']))
            a=np.deg2rad(float(z['yaw_deg']));u,v=g[...,0],g[...,1]
            g=np.stack([np.cos(a)*u-np.sin(a)*v,np.sin(a)*u+np.cos(a)*v],-1)
            g=torch.tensor(g,device=q.device)[None]
            dx=torch.cat([g[:,:,1:]-g[:,:,:-1],g[:,:,-1:]-g[:,:,-2:-1]],2)
            dy=torch.cat([g[:,1:]-g[:,:-1],g[:,-1:]-g[:,-2:-1]],1)
            offsets=(torch.arange(4,device=q.device)+.5)/4-.5
            grids=torch.cat([g+a*dx+b*dy for a in offsets for b in offsets])
            colors=F.grid_sample(linear(q).expand(16,-1,-1,-1),grids,mode='bilinear',padding_mode='border',align_corners=False).mean(0,keepdim=True)
            knots=linear(q.new_tensor([0,64/255,128/255,192/255,1]))
            lut=[tensor(k) for k in ('black','gray64','gray','gray192','white')]
            def response(c):
                return lut[0]+sum((lut[j+1]-lut[j])*((c-knots[j])/(knots[j+1]-knots[j])).clamp(0,1) for j in range(4))
            mask=torch.tensor(z[n+'_mask'].copy(),device=q.device)[None,None].float()
            y=mask*response(colors)+(1-mask)*response(linear(q).mean((-2,-1),keepdim=True))
            views.append(ste_u8(y.clamp(0,1)))
    return views

def arrays(views):
    return [(x.detach()[0].permute(1,2,0).float().cpu().numpy()*255).round().clip(0,255).astype(np.uint8) for x in views]
