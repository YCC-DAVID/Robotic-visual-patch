"""Complete denoising on matched clean states; actual two-camera rendered inputs.

Measures normalized model actions and output-transformed commands separately.
No task-failure inference is made from these open-loop diagnostic distances.
"""
from pathlib import Path
import dataclasses
import json
import time
import drift_patch_train as T
import numpy as np


def main():
    import torch
    import jax
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    torch.set_num_threads(4)
    dest=T.OUT/'dual_view_lowfreq/actions';dest.mkdir(exist_ok=True)
    assert not (dest/'summary.json').exists()
    new=np.load(dest.parent/'validation/inputs.npz')
    block=np.load(T.OUT/'dual_view_both_low/validation/inputs.npz')
    full=np.load(T.OUT/'dual_view/validation/inputs.npz')
    for other in [block,full]:
        for key in ['frames','clean_base','clean_wrist','state8']:
            assert np.array_equal(new[key],other[key]),key
    sources={str(name):(new,str(name)) for name in new['patterns']}
    sources.update(block_both=(block,'both_low'),full_both=(full,'both'))
    cfg=C.get_config('pi05_libero')
    cfg=dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    policy=PC.create_trained_policy(cfg,T.TORCH_CKPT)
    model,device=policy._model,policy._pytorch_device;model.eval()
    for p in model.parameters():p.requires_grad_(False)
    seeds=[20260901,20260902,20260903]
    noise_np=np.array([np.random.RandomState(s).standard_normal((10,32)).astype(np.float32) for s in seeds])
    noises=[torch.from_numpy(n)[None].to(device) for n in noise_np]
    verification={}

    def forward(b,w,k,verify=False):
        element={'observation/image':b,'observation/wrist_image':w,
                 'observation/state':new['state8'][k],'prompt':str(new['prompt'])}
        inputs=policy._input_transform(dict(element))
        inputs=jax.tree.map(lambda a:torch.from_numpy(np.array(a)).to(device)[None],inputs)
        obs=M.Observation.from_dict(inputs)
        state,mask,cache=T.prefix_and_cache(model,device,obs)
        velocity=[];raw=[];actions=[]
        for ni,noise in enumerate(noises):
            x=noise.clone();t=torch.tensor(1.,device=device);dt=torch.tensor(-.1,device=device);vs=[]
            for _ in range(10):
                v=model.denoise_step(state,mask,cache,x,t.expand(1));vs.append(v[0].cpu().numpy())
                x=x+dt*v;t=t+dt
            raw.append(x[0].cpu().numpy());velocity.append(vs)
            transformed=policy._output_transform({'state':inputs['state'][0].cpu().numpy().copy(),
                                                  'actions':raw[-1].copy()})['actions']
            actions.append(np.array(transformed,copy=True))
            if verify and ni==0:
                standard=policy.infer(dict(element),noise=noise_np[0].copy())['actions']
                err=float(np.max(np.abs(standard-actions[-1])))
                assert err<1e-6,(err,'manual denoise differs from standard policy')
                verification['standard_policy_max_abs_error']=max(verification.get('standard_policy_max_abs_error',0),err)
        return np.array(velocity),np.array(raw),np.array(actions)

    names=['clean']+list(sources);vel=[];raw=[];actions=[];start=time.monotonic()
    with torch.no_grad():
        for k,frame in enumerate(new['frames']):
            fv=[];fr=[];fa=[]
            for name in names:
                if name=='clean':b,w=new['clean_base'][k],new['clean_wrist'][k]
                else:
                    src,key=sources[name];b,w=src[key+'__base'][k],src[key+'__wrist'][k]
                v,r,a=forward(b,w,k,verify=(k==0 and name in ['clean','bilinear_both']))
                fv.append(v);fr.append(r);fa.append(a)
            vel.append(fv);raw.append(fr);actions.append(fa)
            print('[actions]',int(frame),'elapsed',round(time.monotonic()-start),flush=True)
    vel=np.array(vel);raw=np.array(raw);actions=np.array(actions)
    # [state, condition, noise, denoise_step, horizon, dimension]
    np.savez_compressed(dest/'actions.npz',velocity=vel,raw_actions=raw,actions=actions,
                        frames=new['frames'],ep_id=new['ep_id'],env_step=new['env_step'],
                        conditions=np.array(names),noise=noise_np)
    summary={}
    for j,name in enumerate(names[1:],1):
        dv=vel[:,j]-vel[:,0];dr=raw[:,j]-raw[:,0];da=actions[:,j]-actions[:,0]
        # Delta action equals dt times the sum of delta velocities, with common initial noise.
        reconstruction=float(np.max(np.abs(dr+.1*dv.sum(axis=2))))
        assert reconstruction<2e-5,reconstruction
        perframe_early=np.square(dv[:,:,0]).sum(axis=(-1,-2)).mean(axis=1)
        perframe_final=np.square(dr[:,:,:5,:7]).sum(axis=(-1,-2)).mean(axis=1)
        first=-.1*dv[:,:,0,:,:7];final=dr[:,:,:,:7]
        cosine=np.sum(first*final,axis=(-1,-2))/(np.linalg.norm(first,axis=(-1,-2))*np.linalg.norm(final,axis=(-1,-2))+1e-20)
        summary[name]={
            'early_L0':float(perframe_early.mean()),
            'final_raw_real7_squared':float(np.square(dr[...,:7]).sum(axis=(-1,-2)).mean()),
            'final_raw_exec5_real7_squared':float(perframe_final.mean()),
            'output_exec_translation_norm_mean':float(np.linalg.norm(da[:,:,:5,:3],axis=-1).mean()),
            'output_exec_rotation_norm_mean':float(np.linalg.norm(da[:,:,:5,3:6],axis=-1).mean()),
            'output_exec_gripper_absolute_mean':float(np.abs(da[:,:,:5,6]).mean()),
            'output_exec_gripper_sign_flip_fraction':float((np.sign(actions[:,j,:,:5,6])!=np.sign(actions[:,0,:,:5,6])).mean()),
            'first_Euler_delta_vs_final_delta_cosine_mean':float(cosine.mean()),
            'denoise_velocity_delta_squared_by_step':np.square(dv[...,:7]).sum(axis=(-1,-2)).mean(axis=(0,1)).tolist(),
            'perframe_early_L0':perframe_early.tolist(),
            'perframe_final_raw_exec5_real7_squared':perframe_final.tolist(),
            'integrated_delta_max_error':reconstruction}
    # Match independently calculated first-step probe on the four newly captured textures.
    early=json.loads((dest.parent/'validation/summary.json').read_text())
    for name in new['patterns']:
        error=abs(summary[str(name)]['early_L0']-early[str(name)+'/both/render']['L0'])
        assert error<1e-4,(name,error)
    (dest/'summary.json').write_text(json.dumps({'complete':True,'conditions':summary,'verification':verification,
        'frames':new['frames'].tolist(),'seeds':seeds,'elapsed_s':time.monotonic()-start,
        'notes':'Common clean states and noise. Both cameras real render. Full10-step denoise, output transform identical to policy. Open-loop effects, not episode success.32 correlated states from2trajectories.'},indent=2)+'\n')
    print('[done]',verification,flush=True)


if __name__=='__main__':main()
