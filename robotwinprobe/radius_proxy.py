"""Use the existing imaging model with an exact real calibration for each draw."""
import numpy as np
from render_proxy import Pool, grid_for
from render_geometry import CAMERAS
from radius_plan import REFERENCE


class CaptureProxy:
    def __init__(self, device):
        self.reference=Pool(sorted((REFERENCE/'calibration').glob('seed*.npz')),'cpu',global_response=True)
        self.device=device
        self.active=Pool.__new__(Pool)
        self.active.device=device;self.active.global_response=True
        self.active.offset=.5;self.active.blur_sigma=0.

    def images(self,q,frame,record=None):
        import torch
        f=self.reference.frames[frame]
        if record is None:
            views={n:{k:v.to(self.device) for k,v in view.items()} for n,view in f['views'].items()}
        else:
            with np.load(record['path']) as d:
                assert int(d['frame'])==frame and int(d['seed'])==f['seed'] and int(d['query'])==f['query']
                assert np.array_equal(d['center_m'],record['center'])
                assert np.array_equal(d['state14'],f['state'])
                views={}
                for n in CAMERAS:
                    v={k:f['views'][n][k].to(self.device) for k in ['clean','clean224']}
                    for key in ['black','gray64','gray','gray192','white']:
                        v[key]=torch.tensor(d[n+'_'+key],device=self.device).permute(2,0,1)[None].float()/255
                    v['mask']=torch.tensor(d[n+'_mask'],device=self.device)[None,None].float()
                    g=grid_for(d[n+'_intrinsic'],d[n+'_extrinsic'],d['center_m'],float(d['size_m']))
                    v['grid']=torch.tensor(g,device=self.device)[None]
                    views[n]=v
        self.active.frames=[{'views':views}]
        return self.active.images(q,0,'proxy')
