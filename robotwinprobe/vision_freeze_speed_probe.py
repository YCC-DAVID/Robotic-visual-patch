"""Isolated full-model throughput probe; never writes production checkpoints."""
import argparse
import functools
import os
from pathlib import Path
import statistics
import time
import numpy as np
import vision_freeze_ft as V
import jax


def main(args):
    args.out.mkdir(parents=True,exist_ok=True)
    V.S.dump(args.out/'status.json',dict(stage='initializing',pid=os.getpid(),complete=False))
    cfg=V.make_config(args.root,'speed_probe',202610230,4,False,False,True)
    data=V.Data(args.root,cfg)
    mesh=V.sharding.make_mesh(2)
    rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(V.sharding.DATA_AXIS))
    rng,key=jax.random.split(jax.random.key(cfg.seed))
    rng=V.replicate_key(rng,rep)
    state,spec=V.init_train_state(cfg,key,mesh,resume=False)
    jax.block_until_ready(state)
    audit=V.audit_replicas(state.params)
    assert audit['passed']
    V.S.dump(args.out/'initial_replicas.json',audit)
    fn=jax.jit(functools.partial(V.functional_step,cfg,args.micro_batch,explicit_flow=True),
        in_shardings=(rep,spec,ds),out_shardings=(spec,rep),donate_argnums=(1,))
    rows=[]
    for i in range(4):
        ids=np.random.default_rng(np.random.SeedSequence([cfg.seed,i])).choice(data.train_ids,16,replace=True)
        batch=data.batch(ids)
        batch=jax.device_put((*batch,*V.draw_inputs(batch[1].shape,[cfg.seed,i,173])),ds)
        jax.block_until_ready(batch)
        start=time.monotonic()
        with V.sharding.set_mesh(mesh):state,info=fn(rng,state,batch)
        info={k:float(v) for k,v in jax.device_get(info).items()}
        assert all(np.isfinite(v) for v in info.values())
        row=dict(step=i+1,seconds=time.monotonic()-start,**info)
        rows.append(row);print(row,flush=True)
        V.S.dump(args.out/'status.json',dict(stage='benchmark',complete=False,pid=os.getpid(),rows=rows))
    audit=V.audit_replicas(state.params)
    V.S.dump(args.out/'final_replicas.json',audit)
    assert audit['passed']
    result=dict(complete=True,micro_batch=args.micro_batch,batch=16,variant='U',rows=rows,
        median_step_s=statistics.median(r['seconds'] for r in rows[1:]),
        environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','NCCL_P2P_DISABLE','NCCL_IB_DISABLE','NCCL_SHM_DISABLE']},
        note='Timing only; microbatch changes augmentation grouping. Not approved for production based on speed alone.')
    V.S.dump(args.out/'result.json',result);V.S.dump(args.out/'status.json',dict(stage='complete',**result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--micro-batch',type=int,choices=[2,4],required=True)
    args=p.parse_args()
    try:main(args)
    except Exception as exc:
        import traceback
        V.S.dump(args.out/'status.json',dict(stage='failed',complete=False,error=repr(exc),traceback=traceback.format_exc()))
        raise
