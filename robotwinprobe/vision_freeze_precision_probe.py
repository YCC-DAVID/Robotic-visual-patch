"""Read-only, fresh-process FP32 monitor reproducibility for the v3 checkpoint."""
import argparse
import dataclasses
import hashlib
import os
import time
from pathlib import Path
import vision_freeze_ft as V
import jax
import numpy as np
from flax import nnx
from openpi.models import model as M
from openpi.training import sharding
from vision_freeze_loss import flow_loss


def main(args):
    root=V.S.ROOT/'robotwinprobe/out/vision_freeze_ablation_20261025'
    out=root/'precision_probe'/args.tag
    start=time.monotonic()
    V.S.dump(out/'status.json',dict(stage='loading',complete=False,pid=os.getpid()))
    cfg=V.make_config(root,'preflight_v3_U',202610230,4,True,True,True)
    data=V.Data(root,cfg,True)
    mesh=sharding.make_mesh(2)
    rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    params=M.restore_params(Path(cfg.checkpoint_dir)/'4/params',restore_type=np.ndarray)
    graph,state=nnx.split(dataclasses.replace(cfg.model,dtype=args.dtype).load(params))
    ps=sharding.fsdp_sharding(state,mesh)
    state=jax.device_put(state,ps);del params

    def forward(p,b):
        model=nnx.merge(graph,p);model.eval()
        return flow_loss(model,*b)

    fn=jax.jit(forward,in_shardings=(ps,ds),out_shardings=rep)
    rows=[]
    for label,ids in [('train',data.train_ids),('validation',data.validation_ids)]:
        batch=data.batch(ids[np.linspace(0,len(ids)-1,4,dtype=int)])
        with np.load(root/f'training/preflight_v3_U/monitor_4_{label}_original.npz') as z:
            batch=(*batch,z['noise'].copy(),z['time'].copy())
            original=z['loss'].copy()
        inputs=V.fingerprints(batch)
        batch=jax.device_put(batch,ds)
        V.S.dump(out/'status.json',dict(stage='evaluating_'+label,complete=False,pid=os.getpid()))
        with sharding.set_mesh(mesh),jax.default_matmul_precision('highest'):
            actual=np.asarray(fn(state,batch));repeat=np.asarray(fn(state,batch))
        np.save(out/f'{label}.npy',actual)
        row=dict(split=label,dtype=args.dtype,mean=float(actual.mean()),
            repeat_exact=bool(np.array_equal(actual,repeat)),
            sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
            relative_l2_to_original_bf16=float(np.linalg.norm(actual-original)/np.linalg.norm(original)),
            inputs=inputs)
        rows.append(row);print({k:v for k,v in row.items() if k!='inputs'},flush=True)
    V.S.dump(out/'results.json',dict(rows=rows,dtype=args.dtype,elapsed_s=time.monotonic()-start))
    V.S.dump(out/'status.json',dict(stage='complete',complete=True,pid=os.getpid()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--dtype',choices=['float32','bfloat16'],default='float32');args=p.parse_args()
    try:main(args)
    except Exception as exc:
        import traceback
        V.S.dump(V.S.ROOT/'robotwinprobe/out/vision_freeze_ablation_20261025/precision_probe'/args.tag/'status.json',
            dict(stage='failed',complete=False,error=repr(exc),traceback=traceback.format_exc()))
        raise
