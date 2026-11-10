"""Replay saved pre/post checkpoint losses with explicit identical noise/time."""
import json
import os
from pathlib import Path
import vision_freeze_ft as V
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from openpi.models import model as M, pi0
from openpi.training import sharding

root=V.S.ROOT/'robotwinprobe/out/vision_freeze_ablation_20261025'
run=root/'training/preflight_v2_U'
out=root/'explicit_time_probe'
V.S.dump(out/'status.json',dict(stage='loading',complete=False,pid=os.getpid()))
try:
    cfg=V.make_config(root,'preflight_v2_U',202610230,4,True,True,True)
    data=V.Data(root,cfg,True)
    mesh=sharding.make_mesh(2)
    rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    params=M.restore_params(Path(cfg.checkpoint_dir)/'4/params',restore_type=np.ndarray)
    model=cfg.model.load(params);model.eval();graph,state=nnx.split(model)
    ps=sharding.fsdp_sharding(state,mesh)
    state=jax.device_put(state,ps)
    del params,model

    def loss(p,batch,noise,time):
        model=nnx.merge(graph,p);model.eval()
        obs,actions=batch
        obs=M.preprocess_observation(None,obs,train=False)
        xt=time[:,None,None]*noise+(1-time[:,None,None])*actions
        prefix,pmask,par=model.embed_prefix(obs)
        suffix,smask,sar,cond=model.embed_suffix(obs,xt,time)
        mask=jnp.concatenate([pmask,smask],axis=1)
        ar=jnp.concatenate([par,sar],axis=0)
        (_,ys),_=model.PaliGemma.llm([prefix,suffix],mask=pi0.make_attn_mask(mask,ar),
            positions=jnp.cumsum(mask,axis=1)-1,adarms_cond=[None,cond])
        v=model.action_out_proj(ys[:,-model.action_horizon:])
        return jnp.mean(jnp.square(v-(noise-actions)),axis=-1)

    fn=jax.jit(loss,in_shardings=(ps,ds,ds,ds),out_shardings=rep)
    rows=[]
    for split,ids in [('train',data.train_ids),('validation',data.validation_ids)]:
        batch=jax.device_put(data.batch(ids[np.linspace(0,len(ids)-1,4,dtype=int)]),ds)
        for tag in ['original','restored']:
            with np.load(run/f'monitor_4_{split}_{tag}.npz') as z:
                noise=jax.device_put(z['noise'],ds);t=jax.device_put(z['time'],ds)
                expected=z['loss'].copy();times=z['time'].tolist()
            with sharding.set_mesh(mesh):
                actual=np.asarray(fn(state,batch,noise,t));repeat=np.asarray(fn(state,batch,noise,t))
            row=dict(split=split,time_source=tag,times=times,expected_mean=float(expected.mean()),
                replay_mean=float(actual.mean()),max_abs_diff=float(np.max(np.abs(actual-expected))),
                repeat_exact=bool(np.array_equal(actual,repeat)))
            rows.append(row);print(row,flush=True)
            np.save(out/f'{split}_{tag}_replay.npy',actual)
    V.S.dump(out/'results.json',dict(rows=rows,scope='same saved EMA and inputs; only recorded flow times differ'))
    V.S.dump(out/'status.json',dict(stage='complete',complete=True,pid=os.getpid()))
except Exception as exc:
    import traceback
    V.S.dump(out/'status.json',dict(stage='failed',complete=False,error=repr(exc),traceback=traceback.format_exc()))
    raise
