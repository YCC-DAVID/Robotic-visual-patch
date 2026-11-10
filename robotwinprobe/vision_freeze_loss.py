"""Upstream flow objective with explicit, reproducible stochastic inputs."""
import numpy as np
import jax.numpy as jnp
from openpi.models import model as M, pi0


def draw_inputs(shape, seed):
    rng=np.random.default_rng(np.random.SeedSequence(seed))
    noise=rng.standard_normal(shape,dtype=np.float32)
    time=(rng.beta(1.5,1,size=shape[:-2])*.999+.001).astype(np.float32)
    return noise,time


def flow_loss(model, observation, actions, noise, time, *, train=False, preprocess_rng=None):
    observation=M.preprocess_observation(preprocess_rng,observation,train=train)
    t=time[...,None,None]
    xt=t*noise+(1-t)*actions
    target=noise-actions
    prefix,pmask,par=model.embed_prefix(observation)
    suffix,smask,sar,cond=model.embed_suffix(observation,xt,time)
    mask=jnp.concatenate([pmask,smask],axis=1)
    ar=jnp.concatenate([par,sar],axis=0)
    (_,ys),_=model.PaliGemma.llm([prefix,suffix],mask=pi0.make_attn_mask(mask,ar),
        positions=jnp.cumsum(mask,axis=1)-1,adarms_cond=[None,cond])
    v=model.action_out_proj(ys[:,-model.action_horizon:])
    return jnp.mean(jnp.square(v-target),axis=-1)
