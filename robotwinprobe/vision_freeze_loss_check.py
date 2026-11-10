"""Independent objective/gradient checks for explicit flow inputs (CPU only)."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
import vision_freeze_ft as V
from vision_freeze_loss import M,pi0,draw_inputs,flow_loss

class LLM(nnx.Module):
    def __call__(self,x,**kwargs):return (x[0],x[1]),None

class Toy(nnx.Module):
    def __init__(self):
        self.weight=nnx.Param(jnp.float32(.31))
        self.fixed=nnx.Param(jnp.float32(.17))
        self.action_horizon=4
        self.PaliGemma=nnx.Dict(llm=LLM())
    def embed_prefix(self,obs):
        return obs[:,None,:],jnp.ones((len(obs),1),bool),jnp.array([False])
    def embed_suffix(self,obs,xt,t):
        return xt,jnp.ones(xt.shape[:2],bool),jnp.array([True,False,False,False]),None
    def action_out_proj(self,x):return self.weight.value*x+self.fixed.value

def main():
    obs=jnp.ones((8,2));actions=jnp.arange(64,dtype=jnp.float32).reshape(8,4,2)/64
    noise,time=draw_inputs(actions.shape,[17,4,173]);noise=jnp.asarray(noise);time=jnp.asarray(time)
    t=time[:,None,None];xt=t*noise+(1-t)*actions;target=noise-actions
    m=Toy();graph,params=nnx.split(m)
    filt=nnx.All(nnx.Param,nnx.Not(V.nnx_utils.PathRegex('fixed')))
    tx=optax.adamw(.001,weight_decay=.0001)
    state=V.training_utils.TrainState(step=0,params=params,model_def=graph,tx=tx,
        opt_state=tx.init(params.filter(filt)),ema_decay=.99,ema_params=params)
    cfg=SimpleNamespace(batch_size=8,trainable_filter=filt)
    with patch.object(M,'preprocess_observation',lambda rng,observation,**kw:observation):
        actual=flow_loss(m,obs,actions,noise,time)
        expected=jnp.mean((m.weight.value*xt+m.fixed.value-target)**2,axis=-1)
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
        with patch.object(jax.random,'normal',return_value=noise),patch.object(jax.random,'beta',return_value=(time-.001)/.999),V.at.disable_typechecking():
            upstream=pi0.Pi0.compute_loss(m,jax.random.key(17),obs,actions,train=False)
        np.testing.assert_allclose(actual,upstream,rtol=1e-6,atol=1e-7)
        new,info=jax.jit(lambda st:V.functional_step(cfg,2,jax.random.key(17),st,(obs,actions,noise,time),explicit_flow=True))(state)
    g=jax.grad(lambda w:jnp.mean((w*xt+params['fixed'].value-target)**2))(params['weight'].value)
    grads=jax.tree.map(jnp.zeros_like,params.filter(filt));grads['weight'].value=g
    updates,_=tx.update(grads,state.opt_state,params.filter(filt))
    expected_params=optax.apply_updates(params.filter(filt),updates)
    np.testing.assert_allclose(new.params['weight'].value,expected_params['weight'].value,rtol=1e-6,atol=1e-7)
    assert np.array_equal(new.params['fixed'].value,state.params['fixed'].value)
    assert np.array_equal(new.ema_params['fixed'].value,state.ema_params['fixed'].value)
    assert abs(float(info['loss'])-float(actual.mean()))<1e-6
    n,t=draw_inputs((100000,1,1),[202610299])
    n2,t2=draw_inputs((100000,1,1),[202610299])
    assert np.array_equal(n,n2) and np.array_equal(t,t2)
    assert abs(float(n.mean()))<.02 and abs(float(n.var())-1)<.02
    assert abs(float(t.mean())-(.999*.6+.001))<.005
    assert np.all((t>=.001)&(t<=1))
    result=dict(passed=True,upstream_formula_matches=True,independent_gradient_and_adam_match=True,
        frozen_params_and_ema_preserved=True,host_draws_repeat_exact=True,
        normal_mean=float(n.mean()),normal_variance=float(n.var()),time_mean=float(t.mean()),
        source_sha256=V.S.sha(Path(V.__file__)),loss_source_sha256=V.S.sha(Path(__file__).with_name('vision_freeze_loss.py')))
    V.S.dump(V.S.ROOT/'robotwinprobe/out/vision_freeze_ablation_20261025/explicit_flow_loss_check.json',result)
    print(result)

if __name__=='__main__':main()
