"""Inspect ALL replicas; global snapshot deduplication cannot detect divergence."""
import argparse
import gc
import hashlib
import os
import time
from pathlib import Path
import vision_freeze_ft as V
import jax
import numpy as np
import torch
from openpi.training import sharding


def host(value):
    value.block_until_ready()
    t=torch.from_dlpack(value).cpu().contiguous()
    return t.view(torch.uint16).numpy().view(np.dtype('bfloat16')) if t.dtype==torch.bfloat16 else t.numpy()


def replica_rows(tree):
    rows=[]
    for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        groups={}
        for shard in value.addressable_shards:
            groups.setdefault(str(shard.index),[]).append(shard)
        for index,shards in groups.items():
            if len(shards)<2:continue
            digests=[]
            for shard in shards:
                a=host(shard.data)
                digests.append(dict(device=str(shard.device),sha256=hashlib.sha256(a.tobytes()).hexdigest()))
            rows.append(dict(path=jax.tree_util.keystr(path),index=index,shape=list(value.shape),
                             equal=len({x['sha256'] for x in digests})==1,replicas=digests))
    return rows


def main(args):
    root=V.S.ROOT/'robotwinprobe/out/vision_freeze_ablation_20261025'
    out=root/'replica_probe';out.mkdir(exist_ok=True)
    V.S.dump(out/'status.json',dict(stage='running',complete=False,pid=os.getpid(),mode=args.mode))
    torch.set_num_threads(4)
    cfg=V.make_config(root,'replica_probe',202610230,4,True,False,True)
    mesh=sharding.make_mesh(2);rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    _,key=jax.random.split(jax.random.key(cfg.seed))
    original_key=np.asarray(jax.random.key_data(key))
    if args.mode=='host_key':
        arrays=[jax.device_put(original_key.copy(),d) for d in rep.addressable_devices]
        data=jax.make_array_from_single_device_arrays(original_key.shape,rep,arrays)
        key=jax.random.wrap_key_data(data)
    else:key=jax.device_put(key,rep)
    keys=[dict(device=str(s.device),data=host(jax.random.key_data(s.data)).tolist()) for s in key.addressable_shards]
    print('keys',args.mode,keys,flush=True)
    start=time.monotonic()
    state,_=V.init_train_state(cfg,key,mesh,resume=False)
    jax.block_until_ready(state)
    rows=replica_rows(state.params)
    mismatches=[r for r in rows if not r['equal']]
    result=dict(mode=args.mode,keys=keys,expected_key=original_key.tolist(),replicated_leaves=len(rows),
        mismatch_count=len(mismatches),rows=rows,elapsed_s=time.monotonic()-start)
    V.S.dump(out/(args.mode+'.json'),result)
    print('replica audit',args.mode,'mismatches',len(mismatches),'/',len(rows),flush=True)
    for row in mismatches:print(row['path'],row['shape'],flush=True)
    V.S.dump(out/'status.json',dict(stage='complete',complete=True,pid=os.getpid(),mode=args.mode))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['default','host_key'],required=True);args=p.parse_args()
    try:main(args)
    except Exception as exc:
        import traceback
        V.S.dump(V.S.ROOT/'robotwinprobe/out/vision_freeze_ablation_20261025/replica_probe/status.json',
            dict(stage='failed',complete=False,error=repr(exc),traceback=traceback.format_exc()))
        raise
