"""Isolated model process: public inference, paired flow diagnostics, patch gradients."""
import argparse
import dataclasses
import json
from pathlib import Path
import traceback
import numpy as np
import torch
from crossarch_common import DEFAULT,EXPERT,WIRE,MODEL,PORTS,AUTH,dump,prepare

class Backend:
    def __init__(self,kind,root):
        self.kind=kind;self.root=root;self.clean_cache={}
        torch.set_num_threads(4)
        if kind=='pi':
            import render_model as M
            self.M=M;self.policy=M.load();self.model=self.policy._model
        else:
            import crossarch_wam as M
            self.M=M;self.model,self.proc,self.bank=M.load(root)

    def prepare(self,state,prompt,views):
        if len(views)!=len(WIRE):
            raise ValueError('RoboTwin requires head, left-wrist and right-wrist images, including patch training.')
        if self.kind=='wam':return self.M.compose(views),self.M.state(self.proc,state),self.bank[prompt]
        from crossarch_proxy import arrays
        raw=dict(zip(WIRE,arrays(views)))
        obs=self.M.observation(self.policy,state,prompt,raw)
        # Native input transform is the oracle forward; differentiable resize is the surrogate.
        import torch.nn.functional as F
        imgs={}
        for key,x in zip(MODEL,views):
            y=F.pad(F.interpolate(x,(168,224),mode='bilinear',align_corners=False,antialias=True),(0,0,28,28))*2-1
            imgs[key]=y+(obs.images[key]-y).detach()
        return dataclasses.replace(obs,images=imgs)

    def noise(self,seed):
        if self.kind=='wam':return self.M.noise(seed)
        return torch.tensor(np.random.RandomState(seed).standard_normal((1,32,32)).astype(np.float32),device='cuda')

    def response(self,prepared,seed,full=False):
        x=self.noise(seed)
        if self.kind=='pi':
            state,pad,kv=self.M.prefix(self.model,prepared)
            first=None
            t=torch.tensor(1.,device=x.device,dtype=torch.float32)
            dt=torch.tensor(-.1,device=x.device,dtype=torch.float32)
            for i in range(10 if full else 1):
                v=self.model.denoise_step(state,pad,kv,x,t.expand(1))
                if first is None:first=v
                x=x+dt*v;t=t+dt
        else:
            cache=self.M.prefix(self.model,*prepared)
            times,deltas=self.model.infer_action_scheduler.build_inference_schedule(10,device='cuda',dtype=x.dtype)
            first=None
            for t,dt in zip(times[:10 if full else 1],deltas):
                v=self.M.velocity(self.model,cache,x,t.expand(1))
                if first is None:first=v
                x=self.model.infer_action_scheduler.step(v,dt,x)
        return first,x

    def raw_prepared(self,state,prompt,raw):
        views=[torch.tensor(np.asarray(raw[k]).copy(),device='cuda').permute(2,0,1)[None].float()/255 for k in WIRE]
        return self.prepare(state,prompt,views)

    def infer(self,req):
        seed=int(req['seed']);raw=req['images'];state=req['state'];prompt=req['prompt']
        with torch.no_grad():
            if self.kind=='pi':
                result=self.policy.infer(dict(state=state,prompt=prompt,images={k:np.asarray(v).transpose(2,0,1).copy() for k,v in raw.items()}),noise=self.noise(seed)[0].cpu().numpy())
                return dict(actions=np.asarray(result['actions']))
            image,proprio,context=self.raw_prepared(state,prompt,raw)
            out=self.model.infer_action_flux2(prompt=None,input_image=image,action_horizon=16,proprio=proprio,
                context=context['context'],context_mask=context['mask'],num_inference_steps=10,seed=seed,rand_device='cpu')
            return dict(actions=self.M.denorm(self.proc,out['action']))

    def gradient(self,req):
        from crossarch_proxy import images
        path=req['record'];seed=int(req['seed'])
        with np.load(path) as z:state=z['state14'].copy();prompt=str(z['prompt'])
        key=(path,seed)
        if key not in self.clean_cache:
            with torch.no_grad():vc,_=self.response(self.prepare(state,prompt,images(None,path)),seed)
            self.clean_cache[key]=vc.float().cpu()
        vc=self.clean_cache[key].to('cuda')
        q=torch.tensor(req['texture'],device='cuda',requires_grad=True)
        v,_=self.response(self.prepare(state,prompt,images(q,path)),seed)
        loss=(v[...,:14].float()-vc[...,:14]).square().mean()
        loss.backward()
        assert q.grad is not None and torch.isfinite(q.grad).all() and torch.isfinite(loss)
        return dict(L0=float(loss.detach()),full_L0=float((v.float()-vc).square().mean().detach()),gradient=q.grad.cpu().numpy())

    def diagnostic(self,req):
        with torch.no_grad():
            v,a=self.response(self.raw_prepared(req['state'],req['prompt'],req['images']),int(req['seed']),full=True)
        return dict(v=v.float().cpu().numpy(),action=a.float().cpu().numpy())

    def validate(self,req):
        from crossarch_proxy import images,arrays
        from render_geometry import patterns
        from crossarch_common import CAMERAS
        rows=[]
        for path in req['records']:
            with np.load(path) as z:
                state=z['state14'].copy();prompt=str(z['prompt'])
                raw={w:z[n+'_clean'].copy() for n,w in zip(CAMERAS,WIRE)}
                with torch.no_grad():vc,_=self.response(self.raw_prepared(state,prompt,raw),42)
                for pattern in ('random','checker'):
                    q=torch.tensor(patterns()[pattern],device='cuda').permute(2,0,1)[None].float()/255
                    with torch.no_grad():
                        proxy=images(q,path);vp,_=self.response(self.prepare(state,prompt,proxy),42)
                        raw={w:z[n+'_'+pattern].copy() for n,w in zip(CAMERAS,WIRE)}
                        vr,_=self.response(self.raw_prepared(state,prompt,raw),42)
                    dvp=(vp[...,:14]-vc[...,:14]).float().flatten();dvr=(vr[...,:14]-vc[...,:14]).float().flatten()
                    lp=float(dvp.square().mean());lr=float(dvr.square().mean())
                    pixels=[]
                    for n,a in zip(CAMERAS,arrays(proxy)):
                        d=(a.astype(float)-z[n+'_'+pattern])/255;mask=z[n+'_mask'].astype(bool)
                        pixels.append(dict(camera=n,rmse=float(np.sqrt(np.mean(d*d))),
                            mask_rmse=float(np.sqrt(np.mean(d[mask]**2))) if mask.any() else None))
                    rows.append(dict(record=str(path),stage=str(z['stage']),pattern=pattern,proxy_L0=lp,render_L0=lr,
                        delta_cosine=float(torch.nn.functional.cosine_similarity(dvp[None],dvr[None])),
                        relative_L0_error=abs(lp-lr)/max(lr,1e-10),pixels=pixels))
        # Existing image-proxy tolerances, plus explicit flow fidelity checks.
        cosine=float(np.median([r['delta_cosine'] for r in rows]))
        error=float(np.median([r['relative_L0_error'] for r in rows]))
        pixel_pass=all(x['rmse']<.06 and (x['mask_rmse'] is None or x['mask_rmse']<.15) for r in rows for x in r['pixels'])
        passed=pixel_pass and cosine>=.8 and error<=.5
        report=dict(passed=passed,pixel_pass=pixel_pass,median_delta_cosine=cosine,median_relative_L0_error=error,
            gate='native RMSE < .06, mask RMSE < .15; median delta-v cosine >= .8 and relative L0 error <= .5',rows=rows)
        dump(self.root/f'audit/{self.kind}_proxy.json',report)
        return report

    def audit(self):
        p=prepare(self.root);seed=p['data']['train_seeds'][0]
        with np.load(EXPERT/f'seed{seed}/demonstration.npz') as z:
            state=z['state14'][0].copy();prompt=str(z['prompt'])
            from crossarch_common import CAMERAS
            raw={w:z[n][0].copy() for n,w in zip(CAMERAS,WIRE)}
        prepared=self.raw_prepared(state,prompt,raw)
        with torch.no_grad():v,a=self.response(prepared,42,True)
        if self.kind=='wam':
            captured=[];original=self.model.action_expert.post_dit
            def spy(*args,**kwargs):
                y=original(*args,**kwargs);captured.append(y.detach().float().cpu());return y
            self.model.action_expert.post_dit=spy
            try:official=self.infer(dict(state=state,prompt=prompt,images=raw,seed=42))['actions']
            finally:self.model.action_expert.post_dit=original
            manual=self.M.denorm(self.proc,a[0]);first_error=float((v.float().cpu()-captured[0]).abs().max())
            # Independently compare the image assembly to the official deployment method.
            from importlib.util import spec_from_file_location,module_from_spec
            from crossarch_common import WAM_SOURCE
            spec=spec_from_file_location('crossarch_native_deploy',WAM_SOURCE/'experiments/robotwin/imagewam_policy/deploy_policy.py')
            mod=module_from_spec(spec);spec.loader.exec_module(mod)
            from types import SimpleNamespace
            oracle=mod.WorldActionRobotWinPolicy._build_robotwin_image_tensor(
                SimpleNamespace(model=self.model,robotwin_camera_layout='compact_288x256'),
                {'observation':{n:{'rgb':raw[w]} for n,w in zip(CAMERAS,WIRE)}})
            pixel_error=float((prepared[0]-oracle).abs().max())
        else:
            # Compare against native sample_actions with identical input and noise.
            captured=[];original=self.model.denoise_step
            def spy(*args,**kwargs):
                y=original(*args,**kwargs);captured.append(y.detach().float().cpu());return y
            self.model.denoise_step=spy
            try:
                with torch.no_grad():native=self.model.sample_actions('cuda',prepared,noise=self.noise(42),num_steps=10)
            finally:self.model.denoise_step=original
            first_error=float((v.float().cpu()-captured[0]).abs().max());pixel_error=0.
            manual=a.detach().cpu().numpy();official=native.detach().cpu().numpy()
        action_error=float(np.max(np.abs(manual-official)))
        # Image derivative through the frozen model must reach input pixels.
        views=[torch.tensor(raw[w],device='cuda').permute(2,0,1)[None].float()/255 for w in WIRE]
        views[0].requires_grad_(True)
        vg,_=self.response(self.prepare(state,prompt,views),43)
        vg.float().square().mean().backward()
        g=views[0].grad
        passed=first_error<1e-4 and pixel_error==0 and action_error<1e-4 and bool(torch.isfinite(g).all()) and float(g.norm())>0
        row=dict(passed=passed,first_velocity_max_error=first_error,full_action_max_error=action_error,
                 native_preprocessing_max_error=pixel_error,image_gradient_norm=float(g.norm()),
                 policy_parameter_gradients=sum(p.grad is not None for p in self.model.parameters()),
                 gpu_peak_GiB=torch.cuda.max_memory_allocated()/2**30)
        dump(self.root/f'audit/{self.kind}.json',row);print('[audit]',self.kind,row,flush=True)
        assert passed and row['policy_parameter_gradients']==0,row

def main(args):
    backend=Backend(args.model,args.root);backend.audit()
    if args.audit_only:return
    from multiprocessing.connection import Listener
    with Listener(('127.0.0.1',PORTS[args.model]),authkey=AUTH) as listener:
        dump(args.root/f'{args.model}_ready.json',dict(ready=True,port=PORTS[args.model]))
        while True:
            with listener.accept() as conn:
                try:
                    req=conn.recv();cmd=req.pop('command')
                    if cmd=='ping':result={'ready':True,'model':args.model}
                    else:result=getattr(backend,cmd)(req)
                    conn.send(result)
                except Exception as e:
                    conn.send(dict(error=repr(e),traceback=traceback.format_exc()));traceback.print_exc()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['pi','wam'],required=True)
    ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--audit-only',action='store_true');main(ap.parse_args())
