"""Read-only checkpoint audit; compare serialized EMA and graph reconstruction."""
import argparse,json,hashlib,functools,os
from pathlib import Path
import vision_freeze_ft as V
import jax,numpy as np
from flax import nnx,traverse_util
from openpi.models import model as M
from openpi.training import checkpoints,sharding
S=V.S
ap=argparse.ArgumentParser()
ap.add_argument('--root',type=Path,default=S.ROOT/'robotwinprobe/out/vision_freeze_ablation_20261025')
ap.add_argument('--name',default='smoke_U')
ap.add_argument('--output',default='restore_probe_20261026')
args=ap.parse_args()
root=args.root.resolve()
out=root/args.output;out.mkdir(exist_ok=True)
S.dump(out/'status.json',dict(stage='loading',complete=False,pid=os.getpid()))
try:
 cfg=V.make_config(root,args.name,202610230,4,True,True,True)
 data=V.Data(root,cfg,smoke=True)
 mesh=sharding.make_mesh(2);ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS));rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
 _,key=jax.random.split(jax.random.key(cfg.seed))
 shape,ss=V.F.upstream.init_train_state(cfg,key,mesh,resume=True)
 manager,_=checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,keep_period=5000,overwrite=False,resume=True)
 state=checkpoints.restore_state(manager,shape,data);jax.block_until_ready(state)
 ck=Path(cfg.checkpoint_dir)/str(int(state.step))
 disk=M.restore_params(ck/'params',restore_type=np.ndarray)
 host=V.F.host_snapshot(state.ema_params).to_pure_dict()
 flat_disk=traverse_util.flatten_dict(disk);flat_host=traverse_util.flatten_dict(host)
 assert set(flat_disk)==set(flat_host)
 comparisons=[]
 for path,a in flat_disk.items():
  b=flat_host[path]
  same=a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
  comparisons.append(dict(path='/'.join(path),disk_dtype=str(a.dtype),restored_dtype=str(b.dtype),equal=bool(same)))
 S.dump(out/'ema_hash_audit.json',dict(all_equal=all(x['equal'] for x in comparisons),leaves=comparisons))
 print('EMA matches disk:',all(x['equal'] for x in comparisons),flush=True)
 snapshot_path=root/'snapshot_audit.json'
 snapshot_audit=json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
 if snapshot_audit:
  actual={jax.tree_util.keystr(k):hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
    for k,v in jax.tree_util.tree_flatten_with_path(V.F.host_snapshot(state.ema_params))[0]}
  snapshot_equal=actual==snapshot_audit['ema_hashes']
  S.dump(out/'presave_ema_audit.json',dict(all_equal=snapshot_equal,leaves=len(actual)))
  print('EMA matches original pre-save snapshot:',snapshot_equal,flush=True)
 del disk,host,flat_disk,flat_host
 rebuilt=cfg.model.load(state.ema_params.to_pure_dict());rebuilt.eval()
 rebuilt_graph,rebuilt_params=nnx.split(rebuilt)
 graphs={'restored':(state.model_def,state.ema_params),'rebuilt':(rebuilt_graph,rebuilt_params)}
 rows=[];inputs={}
 for name,(graph,params) in graphs.items():
  def loss(p,b,k):
   model=nnx.merge(graph,p);model.eval();return model.compute_loss(k,b[0],b[1],train=False)
  fn=jax.jit(loss,in_shardings=(ss.ema_params,ds,rep),out_shardings=rep)
  for split,indices in [('train',data.train_ids),('validation',data.validation_ids)]:
   ids=indices[np.linspace(0,len(indices)-1,4,dtype=int)];batch=data.batch(ids)
   inputs[split]={jax.tree_util.keystr(k):hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in jax.tree_util.tree_flatten_with_path(batch)[0]}
   batch=jax.device_put(batch,ds)
   with sharding.set_mesh(mesh):a=np.asarray(fn(params,batch,jax.random.key(202610299)));b=np.asarray(fn(params,batch,jax.random.key(202610299)))
   row=dict(graph=name,split=split,loss=float(a.mean()),repeat_exact=bool(np.array_equal(a,b)),sha256=hashlib.sha256(a.tobytes()).hexdigest());rows.append(row)
   np.save(out/f'{name}_{split}_loss.npy',a);print(row,flush=True)
 results=dict(rows=rows,inputs=inputs,all_ema_bytes_equal=all(x['equal'] for x in comparisons),graph_def_equal=state.model_def==rebuilt_graph)
 if snapshot_audit:
  results.update(presave_ema_equal=snapshot_equal,input_hashes_equal=inputs==snapshot_audit['input_hashes'],
    presave_losses=snapshot_audit['before'])
 S.dump(out/'results.json',results)
 manager.close();S.dump(out/'status.json',dict(stage='complete',complete=True,pid=os.getpid()))
except Exception as e:
 import traceback
 S.dump(out/'status.json',dict(stage='failed',complete=False,error=repr(e),traceback=traceback.format_exc()));raise
