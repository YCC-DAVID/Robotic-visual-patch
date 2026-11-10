"""One explicit Aloha interface for patch gradients, diagnostics and closed loops."""
import dataclasses
import os
import sys
import numpy as np
import checkpoint_study_common as S
S.training_paths()
sys.path.insert(0,str(S.ROOT/'realprobe/runtime_py'))
sys.path.insert(0,str(S.ROOT/'third_party/transformers_patched'))
os.environ.setdefault('JAX_PLATFORMS','cpu')


def load(checkpoint):
    import torch
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    torch.set_num_threads(4)
    cfg=dataclasses.replace(C.get_config('pi05_aloha'),model=S.model_config(),
        data=C.LeRobotAlohaDataConfig(assets=C.AssetsConfig(asset_id=S.ASSET),
            adapt_to_pi=True,use_delta_joint_actions=True))
    policy=PC.create_trained_policy(cfg,checkpoint,pytorch_device='cuda',sample_kwargs={'num_steps':10})
    policy._model.eval().requires_grad_(False)
    return policy


def observation(policy,state,prompt,images):
    import jax
    import torch
    from openpi.models import model as M
    x=policy._input_transform(dict(state=state,prompt=str(prompt),
        images={k:np.asarray(v).copy().transpose(2,0,1) for k,v in images.items()}))
    return M.Observation.from_dict(jax.tree.map(
        lambda a:torch.from_numpy(np.asarray(a).copy()).to(policy._pytorch_device)[None],x))


def prefix(model,obs):
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    images,masks,tokens,token_masks,state=model._preprocess_observation(obs,train=False)
    embeds,pad,att=model.embed_prefix(images,masks,tokens,token_masks)
    mask=model._prepare_attention_masks_4d(make_att_2d_masks(pad,att))
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation='eager'
    _,kv=model.paligemma_with_expert.forward(attention_mask=mask,
        position_ids=torch.cumsum(pad,dim=1)-1,past_key_values=None,
        inputs_embeds=[embeds,None],use_cache=True)
    return state,pad,kv


def velocity(model,obs,noise):
    import torch
    state,pad,kv=prefix(model,obs)
    return model.denoise_step(state,pad,kv,noise,torch.ones(len(noise),device=noise.device))


def response(policy,obs,noise):
    """First v and complete Euler integration share exactly one prefix and noise."""
    import torch
    state,pad,kv=prefix(policy._model,obs);x=noise.clone();v0=None
    time=torch.tensor(1.0,dtype=torch.float32,device=x.device);dt=-.1
    for step in range(10):
        v=policy._model.denoise_step(state,pad,kv,x,time.expand(len(x)))
        if step==0:v0=v
        x=x+dt*v;time=time+dt
    # Output transforms require the original normalized padded state for delta -> absolute.
    raw=policy._output_transform(dict(actions=x[0].float().cpu().numpy(),
        state=obs.state[0].float().cpu().numpy()))['actions']
    return v0.float().cpu().numpy(),x.float().cpu().numpy(),np.asarray(raw)
