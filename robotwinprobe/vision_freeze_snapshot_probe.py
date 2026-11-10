"""Isolated four-update diagnostic with before/after host-snapshot EMA probes."""
import argparse,functools,hashlib,json,os
from pathlib import Path
import vision_freeze_ft as V
import jax,numpy as np
from flax import nnx
from openpi.training import sharding
S=V.S
parent=S.ROOT/'robotwinprobe/out/vision_freeze_ablation_20261025'
root=parent/'snapshot_probe_20261026';root.mkdir(exist_ok=True)
for name in ['data_smoke','smoke_assets']:
 if not (root/name).exists():(root/name).symlink_to((parent/name).resolve(),target_is_directory=True)
if not (root/'protocol.json').exists():(root/'protocol.json').write_text((parent/'protocol.json').read_text())
original=V.F.host_snapshot
counter=0

def checked_snapshot(tree):
 global counter
 if not hasattr(tree,'ema_params'):return original(tree)
 cfg=V.make_config(root,'diagnostic_U',202610230,4,True,False,True)
 data=V.Data(root,cfg,True)
 mesh=sharding.make_mesh(2);ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS));rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
 ps=jax.tree.map(lambda x:x.sharding,tree.ema_params)
 def loss(params,batch,key):
  model=nnx.merge(tree.model_def,params);model.eval();return model.compute_loss(key,batch[0],batch[1],train=False)
 fn=jax.jit(loss,in_shardings=(ps,ds,rep),out_shardings=rep)
 batches={};input_hashes={}
 for label,ids in [('train',data.train_ids),('validation',data.validation_ids)]:
  batch=data.batch(ids[np.linspace(0,len(ids)-1,4,dtype=int)])
  input_hashes[label]={jax.tree_util.keystr(k):hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in jax.tree_util.tree_flatten_with_path(batch)[0]}
  batches[label]=jax.device_put(batch,ds)
 def evaluate(tag):
  values={}
  for label,batch in batches.items():
   with sharding.set_mesh(mesh):a=np.asarray(fn(tree.ema_params,batch,jax.random.key(202610299)))
   values[label]=float(a.mean());np.save(root/f'{tag}_{label}.npy',a)
  return values
 before=evaluate('before_snapshot')
 snapshot=original(tree)
 after=evaluate('after_snapshot')
 fingerprints={jax.tree_util.keystr(k):hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in jax.tree_util.tree_flatten_with_path(snapshot.ema_params)[0]}
 S.dump(root/'snapshot_audit.json',dict(before=before,after=after,ema_hashes=fingerprints,input_hashes=input_hashes,
    source_sha256=S.sha(Path(__file__)),step=int(tree.step),scope='isolated diagnostic only; formal pipeline remains stopped'))
 print('[snapshot audit]',before,after,flush=True)
 return snapshot
V.F.host_snapshot=checked_snapshot
args=argparse.Namespace(root=root,name='diagnostic_U',seed=202610230,steps=4,smoke=True,resume=False,unfreeze_vision=True,verify_only=False)
try:
 S.dump(root/'probe_status.json',dict(stage='running',complete=False,pid=os.getpid()))
 V.train(args)
 S.dump(root/'probe_status.json',dict(stage='complete',complete=True,pid=os.getpid()))
except Exception as e:
 import traceback
 S.dump(root/'probe_status.json',dict(stage='failed',complete=False,error=repr(e),traceback=traceback.format_exc()));raise
