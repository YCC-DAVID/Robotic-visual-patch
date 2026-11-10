"""RoboTwin-specific forward perspective and candidate calibrated imaging proxy.

No MuJoCo mipmap assumptions: the uploaded SAPIEN texture uses one mip level.
The calibrated candidate models subpixel averaging in linear RGB and measured
per-pixel gray responses. Its fidelity must be measured before interpretation.
"""
import numpy as np
from render_geometry import CAMERAS, WIRE, MODEL


def linear(x):
    import torch
    return torch.where(x <= .04045, x/12.92, ((x+.055)/1.055).pow(2.4))


def ste_u8(x):
    return x + ((x*255).round()/255-x).detach()


def grid_for(k, e, center, size, height=240, width=320, offset=.5):
    yy, xx = np.indices((height, width), dtype=np.float32)
    if (height,width)==(224,224):
        xx = (xx+.5)/.7-.5; yy = (yy-28+.5)/.7-.5
    e4=np.eye(4); e4[:3]=e[:3]; pose=np.linalg.inv(e4)
    rays=np.stack([xx+offset,yy+offset,np.ones_like(xx)],-1)@np.linalg.inv(k).T
    direction=rays@pose[:3,:3].T; origin=pose[:3,3]
    distance=(center[2]-origin[2])/np.where(abs(direction[...,2])>1e-8,direction[...,2],1e-8)
    world=origin+distance[...,None]*direction
    return np.stack([2*(world[...,0]-center[0])/size,-2*(world[...,1]-center[1])/size],-1).astype(np.float32)


class Pool:
    def __init__(self, paths, device, offset=.5, global_response=False, blur_sigma=0.):
        import torch
        from openpi_client.image_tools import resize_with_pad
        self.frames=[]; self.device=device; self.offset=offset; self.global_response=global_response
        self.blur_sigma=blur_sigma
        for path in paths:
            with np.load(path) as d:
                for i in range(len(d['state14'])):
                    frame={'seed':int(d['seed']), 'query':int(d['query'][i]),'state':d['state14'][i],
                           'prompt':str(d['prompt']), 'views':{},'raw':{}}
                    for name,wire in zip(CAMERAS,WIRE):
                        raw=d[name+'_clean'][i]; frame['raw'][wire]=raw
                        view={}
                        for key in ('clean','black','gray64','gray','gray192','white','random','checker','uv'):
                            if name+'_'+key in d:
                                view[key]=torch.tensor(d[name+'_'+key][i].copy(),device=device).permute(2,0,1)[None].float()/255
                        view['clean224']=torch.tensor(resize_with_pad(raw,224,224).copy(),device=device).permute(2,0,1)[None].float()/255
                        view['mask']=torch.tensor(d[name+'_mask'][i].copy(),device=device)[None,None].float()
                        for shape,label in [((240,320),'grid'),((224,224),'grid224')]:
                            g=grid_for(d[name+'_intrinsic'][i],d[name+'_extrinsic'][i],d['center_m'],float(d['size_m']),*shape,offset)
                            view[label]=torch.tensor(g,device=device)[None]
                        frame['views'][name]=view
                    self.frames.append(frame)

    def image(self, q, i, name, mode='digital', samples=4):
        import torch
        import torch.nn.functional as F
        v=self.frames[i]['views'][name]; q=ste_u8(q.clamp(0,1))
        if mode=='digital':
            texture=F.grid_sample(q,v['grid224'],mode='bilinear',padding_mode='border',align_corners=False)
            mask=F.pad(F.interpolate(v['mask'],size=(168,224),mode='nearest'),(0,0,28,28))
            return ste_u8((1-mask)*v['clean224']+mask*texture)
        g=v['grid']
        # Pixel footprint from local ray derivatives; 4x4 midpoint quadrature.
        dx=torch.cat([g[:,:,1:]-g[:,:,:-1],g[:,:,-1:]-g[:,:,-2:-1]],2)
        dy=torch.cat([g[:,1:]-g[:,:-1],g[:,-1:]-g[:,-2:-1]],1)
        offsets=(torch.arange(samples,device=q.device)+.5)/samples-.5
        grids=torch.cat([g+a*dx+b*dy for a in offsets for b in offsets],0)
        colors=F.grid_sample(linear(q).expand(samples*samples,-1,-1,-1),grids,
                             mode='bilinear',padding_mode='border',align_corners=False).mean(0,keepdim=True)
        if self.blur_sigma>0:
            radius=int(np.ceil(3*self.blur_sigma))
            axis=torch.arange(-radius,radius+1,device=q.device,dtype=q.dtype)
            weights=torch.exp(-.5*(axis/self.blur_sigma).square());weights=weights/weights.sum()
            kernel=(weights[:,None]*weights[None,:])[None,None].expand(3,1,-1,-1)
            colors=F.conv2d(F.pad(colors,(radius,)*4,mode='replicate'),kernel,groups=3)
        knots=linear(torch.tensor([0,64/255,128/255,192/255,1.],device=q.device))
        lut=[v[k] for k in ('black','gray64','gray','gray192','white')]
        native=lut[0]+sum((lut[j+1]-lut[j])*((colors-knots[j])/(knots[j+1]-knots[j])).clamp(0,1) for j in range(4))
        outside=v['clean']
        if self.global_response:
            mean=linear(q).mean((-2,-1),keepdim=True)
            outside=lut[0]+sum((lut[j+1]-lut[j])*((mean-knots[j])/(knots[j+1]-knots[j])).clamp(0,1) for j in range(4))
        native=v['mask']*native+(1-v['mask'])*outside
        native=ste_u8(native.clamp(0,1))
        return ste_u8(F.pad(F.interpolate(native,size=(168,224),mode='bilinear',align_corners=False,antialias=True),(0,0,28,28)))

    def images(self,q,i,mode):
        return {key:self.image(q,i,name,mode)*2-1 for name,key in zip(CAMERAS,MODEL)}
