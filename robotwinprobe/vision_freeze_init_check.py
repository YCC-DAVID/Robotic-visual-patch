"""CPU regression: freeze changes gradients, never initial vision precision."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import vision_freeze_ft as V


class Tiny(nnx.Module):
    def __init__(self):
        self.img=nnx.Param(jnp.array([[.1234567,.2345678],[.3456789,.4567891]],jnp.float32))
        self.llm=nnx.Param(jnp.array([.2468135,.1357924],jnp.float32))
        self.action=nnx.Param(jnp.array([.4123456,.5678912],jnp.float32))

    def compute_loss(self,key,obs,actions,train=True):
        return (obs @ self.img.value + self.llm.value + self.action.value - actions)**2


class Loader:
    def load(self,shapes):
        return jax.tree.map(lambda value,spec:np.asarray(value,dtype=spec.dtype),
                            nnx.state(Tiny()).to_pure_dict(),shapes)


def main(out):
    mesh=V.sharding.make_mesh(2)
    original=V.C.get_config('pi05_g1_eef6d')
    states=[];configs=[]
    for freeze_vision in (False,True):
        freeze=V.nnx_utils.PathRegex('llm|img' if freeze_vision else 'llm')
        cfg=SimpleNamespace(model=SimpleNamespace(create=lambda _:Tiny()),
            freeze_filter=freeze,trainable_filter=nnx.All(nnx.Param,nnx.Not(freeze)),
            optimizer=original.optimizer,lr_schedule=original.lr_schedule,
            ema_decay=.99,weight_loader=Loader(),batch_size=4)
        state,spec=V.init_train_state(cfg,jax.random.key(42),mesh,resume=False)
        schema,_=V.init_train_state(cfg,jax.random.key(42),mesh,resume=True)
        assert state.params['img'].value.dtype==jnp.float32
        assert schema.params['img'].value.dtype==jnp.float32
        assert np.array_equal(np.asarray(state.params['img'].value),np.asarray(Tiny().img.value))
        assert state.params['llm'].value.dtype==jnp.bfloat16
        for a,b in zip(jax.tree.leaves(state),jax.tree.leaves(schema),strict=True):
            assert np.shape(a)==b.shape and np.asarray(a).dtype==b.dtype
        states.append(state);configs.append(cfg)
    for a,b in zip(jax.tree.leaves(states[0].params),jax.tree.leaves(states[1].params),strict=True):
        assert np.array_equal(np.asarray(a),np.asarray(b))
    obs=jnp.array([[1.,2.],[2.,3.],[3.,4.],[4.,5.]],jnp.float32)
    actions=jnp.zeros((4,2),jnp.float32)
    for frozen,(state,cfg) in zip((False,True),zip(states,configs)):
        updated=state
        for _ in range(4):
            updated,_=jax.jit(lambda st:V.functional_step(cfg,2,jax.random.key(12),st,(obs,actions)))(updated)
        changed=not np.array_equal(np.asarray(updated.params['img'].value),np.asarray(state.params['img'].value))
        assert changed != frozen
        if frozen:
            assert np.array_equal(np.asarray(updated.ema_params['img'].value),np.asarray(state.ema_params['img'].value))
    V.S.dump(out,dict(passed=True,scope='CPU independent values: identical F/U initial params, full float32 vision, restore schema matches, vision updates only in U, frozen EMA unchanged',source_sha256=V.S.sha(Path(V.__file__))))
    print('paired initialization/restore schema/update regression passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
