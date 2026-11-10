"""Paired pi0.5 fine-tunes: identical recipe except whether vision weights update."""
import argparse
import dataclasses
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback
import hashlib
import jax.numpy as jnp
import optax

import checkpoint_study_common as S
S.training_paths()
import g1_simple_pick_ft as F
from checkpoint_study_data import Data
import flax.nnx as nnx
import jax
import numpy as np
from openpi.training import config as C, checkpoints, sharding
from openpi.training import optimizer as optimizer_lib, utils as training_utils
from openpi.shared import nnx_utils
from openpi.shared import array_typing as at
from orbax.checkpoint._src.serialization import type_handlers as orbax_types
from vision_freeze_loss import draw_inputs,flow_loss
from vision_freeze_device import replicate_key,audit_replicas,compare_fp32


def init_train_state(cfg, key, mesh, *, resume):
    """Keep identical vision precision before loading AND when restoring F/U.

    Upstream casts every frozen leaf to bf16. Casting vision back to fp32 after
    that loses precision and also gives Orbax a different restore schema.
    """
    rep=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    key=replicate_key(key,rep)
    tx=optimizer_lib.create_optimizer(cfg.optimizer,cfg.lr_schedule,weight_decay_mask=None)
    cast_filter=nnx.All(cfg.freeze_filter,nnx.Not(nnx_utils.PathRegex('.*img.*')))
    def initialize(rng,partial=None):
        _,model_rng=jax.random.split(rng)
        model=cfg.model.create(model_rng)
        graph,params=nnx.split(model)
        if partial is not None:params.replace_by_pure_dict(partial)
        params=nnx_utils.state_map(params,cast_filter,lambda p:p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(step=0,params=params,model_def=graph,tx=tx,
            opt_state=tx.init(params.filter(cfg.trainable_filter)),ema_decay=cfg.ema_decay,ema_params=params)
    shape=jax.eval_shape(initialize,key)
    spec=sharding.fsdp_sharding(shape,mesh,log=True)
    if resume:return shape,spec
    partial=F.upstream._load_weights_and_validate(cfg.weight_loader,shape.params.to_pure_dict())
    state=jax.jit(initialize,in_shardings=rep,out_shardings=spec,donate_argnums=(1,))(key,partial)
    return state,spec


def fingerprints(tree):
    return {jax.tree_util.keystr(k):dict(shape=list(np.asarray(v).shape),dtype=str(np.asarray(v).dtype),
            sha256=hashlib.sha256(np.asarray(v).tobytes()).hexdigest())
            for k,v in jax.tree_util.tree_flatten_with_path(tree)[0]}


def functional_step(cfg,micro_batch,rng,state,batch,*,explicit_flow=False):
    trainable,frozen=state.params.split(cfg.trainable_filter,...)
    key=jax.random.fold_in(rng,state.step)
    count=cfg.batch_size//micro_batch
    def loss_fn(params,obs,actions,key,*draws):
        model=nnx.merge(state.model_def,nnx.State.merge(params,frozen))
        model.train()
        if explicit_flow:
            return jnp.mean(flow_loss(model,obs,actions,*draws,train=True,
                preprocess_rng=jax.random.split(key,3)[0]))
        return jnp.mean(model.compute_loss(key,obs,actions,train=True))
    micro=jax.tree.map(lambda x:x.reshape((count,micro_batch)+x.shape[1:]),batch)
    def body(carry,inputs):
        i,obs,actions,*draws=inputs
        loss,grad=jax.value_and_grad(loss_fn)(trainable,obs,actions,jax.random.fold_in(key,i),*draws)
        return (jax.tree.map(lambda a,b:a+b,carry[0],grad),carry[1]+loss),None
    (grad,loss),_=jax.lax.scan(body,(jax.tree.map(jnp.zeros_like,trainable),jnp.float32(0)),
        (jnp.arange(count),*micro))
    grad=jax.tree.map(lambda g:g/count,grad)
    updates,opt=state.tx.update(grad,state.opt_state,trainable)
    updated=optax.apply_updates(trainable,updates)
    params=nnx.State.merge(frozen,updated)
    ema_train=jax.tree.map(lambda a,b:state.ema_decay*a+(1-state.ema_decay)*b,
        state.ema_params.filter(cfg.trainable_filter),updated)
    ema=nnx.State.merge(state.ema_params,ema_train)
    info=dict(loss=loss/count,grad_norm=optax.global_norm(grad),param_norm=optax.global_norm(trainable))
    return dataclasses.replace(state,step=state.step+1,params=params,opt_state=opt,ema_params=ema),info


def make_config(root, name, seed, steps, smoke=False, resume=False, unfreeze=False):
    original=C.get_config('pi05_g1_eef6d')
    model=S.model_config()
    data=S.data_factory(root)
    if smoke:
        data=dataclasses.replace(data,assets=C.AssetsConfig(assets_dir=str(root/'smoke_assets'),asset_id=S.ASSET))
    freeze=model.get_freeze_filter() if unfreeze else nnx.Any(model.get_freeze_filter(),nnx_utils.PathRegex('.*img.*'))
    return dataclasses.replace(original,name='pi05_robotwin_vision_ablation',exp_name=name,
        model=model,data=data,freeze_filter=freeze,seed=seed,
        batch_size=16,num_train_steps=steps,fsdp_devices=2,
        weight_loader=F.FastBaseWeightLoader(str(S.BASE)),
        checkpoint_base_dir=str(root/'checkpoints'),assets_base_dir=str(root/'assets'),
        save_interval=steps if smoke else 5000,keep_period=5000,log_interval=25,
        wandb_enabled=False,resume=resume,overwrite=False,eval_interval=0)


def train(args):
    root=args.root.resolve(); protocol=S.prepare_protocol(root)
    cfg=make_config(root,args.name,args.seed,args.steps,args.smoke,args.resume,args.unfreeze_vision)
    run=root/'training'/args.name;run.mkdir(parents=True,exist_ok=True)
    data_manifest=root/('data_smoke' if args.smoke else 'data_cache')/'manifest.json'
    norm=root/('smoke_assets' if args.smoke else 'assets')/S.ASSET/'norm_stats.json'
    recipe=dict(architecture=dataclasses.asdict(cfg.model),initialization=str(S.BASE),
        dataset_manifest_sha256=S.sha(data_manifest),normalization_sha256=S.sha(norm),
        batch=16,micro_batch=2,freeze_vision=not args.unfreeze_vision,vision_dtype="float32 in both variants",fsdp=2,language_lora_rank=16,full_action_expert=True,
        learning_rate=dataclasses.asdict(cfg.lr_schedule),optimizer=dataclasses.asdict(cfg.optimizer),
        ema=cfg.ema_decay,loss='upstream flow matching MSE; explicit host N(0,1), t~Beta(1.5,1)*.999+.001; paired seed/step streams',
        monitor_precision='float32 compute, highest matmul precision; elementwise rtol=1e-6 atol=1e-7; exact input/weight/replica bytes',
        replicated_rng='upload identical host key bytes per GPU; audit all replicated parameters before training/saving',
        image_augmentation='head crop95%/rotate+-5deg; all-view color jitter .3/.4/.5',
        action_horizon=32,raw_action_dim=14,discrete_state_input=True,
        source_hashes={str(p.relative_to(S.ROOT)):S.sha(p) for p in [Path(__file__),
            S.ROOT/'robotwinprobe/vision_freeze_loss.py',
            S.ROOT/'robotwinprobe/vision_freeze_device.py',
            S.ROOT/'robotwinprobe/checkpoint_study_data.py',S.ROOT/'robotwinprobe/checkpoint_study_common.py',
            S.ROOT/'realprobe/g1_simple_pick_ft.py',S.SERVER/'src/openpi/models/pi0.py',
            S.SERVER/'src/openpi/models/model.py',S.SERVER/'src/openpi/policies/aloha_policy.py']},
        versions={k:importlib.metadata.version(k) for k in ['jax','jaxlib','flax','optax','numpy']})
    manifest=dict(name=args.name,seed=args.seed,steps=args.steps,smoke=args.smoke,
        recipe=recipe,checkpoint_dir=str(cfg.checkpoint_dir),protocol_sha256=S.sha(root/'protocol.json'),
        python_executable=os.path.realpath(os.sys.executable),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if (run/'config.json').exists():
        previous=json.loads((run/'config.json').read_text())
        assert previous['recipe']==recipe and previous['seed']==args.seed and previous['smoke']==args.smoke
        assert args.resume,'Existing training requires --resume'
    else:S.dump(run/'config.json',manifest)
    start=time.monotonic()
    S.dump(run/'status.json',dict(stage='initializing',complete=False,pid=os.getpid(),seed=args.seed))
    orbax_types.register_type_handler(np.ndarray,F.ImmutableSnapshotHandler(),override=True)
    jax.config.update('jax_compilation_cache_dir',str(root/'jax_cache'))
    data=Data(root,cfg,smoke=args.smoke)
    mesh=sharding.make_mesh(2)
    ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    manager,resuming=checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,
        keep_period=cfg.keep_period,overwrite=False,resume=args.resume)
    train_rng,init_rng=jax.random.split(jax.random.key(args.seed))
    train_rng=replicate_key(train_rng,replicated)
    state,state_shard=init_train_state(cfg,init_rng,mesh,resume=resuming)
    if resuming:
        state=checkpoints.restore_state(manager,state,data)
        # ImmutableSnapshotHandler restores host NumPy arrays. Explicitly place
        # the complete state on the same FSDP layout before auditing replicas.
        state=jax.device_put(state,state_shard)
    jax.block_until_ready(state)
    replicas={k:audit_replicas(v) for k,v in [('params',state.params),('ema',state.ema_params)]}
    replica_tag=f'restored_{int(state.step)}' if resuming else 'initial'
    S.dump(run/f'replica_audit_{replica_tag}.json',replicas)
    assert all(v['passed'] for v in replicas.values()),'Initial/restored parameter replicas differ between GPUs'
    step_fn=jax.jit(functools.partial(functional_step,cfg,2,explicit_flow=True),
        in_shardings=(replicated,state_shard,ds),out_shardings=(state_shard,replicated),donate_argnums=(1,))
    # Monitoring is independent of bf16 training-kernel choices. Only the
    # diagnostic graph changes precision; training and stored weights do not.
    eval_graph=nnx.graphdef(nnx.eval_shape(dataclasses.replace(cfg.model,dtype='float32').create,
                                        jax.random.key(0)))
    def eval_loss(params,batch,key):
        model=nnx.merge(eval_graph,params);model.eval()
        obs,actions,noise,flow_time=batch
        return flow_loss(model,obs,actions,noise,flow_time),noise,flow_time
    eval_fn=jax.jit(eval_loss,in_shardings=(state_shard.ema_params,ds,replicated),out_shardings=replicated)
    train_monitor=data.train_ids[np.linspace(0,len(data.train_ids)-1,4,dtype=int)]
    val_monitor=data.validation_ids[np.linspace(0,len(data.validation_ids)-1,4,dtype=int)]
    monitor_batches={'train':data.batch(train_monitor),'validation':data.batch(val_monitor)}
    monitor_batches={label:(*batch,*draw_inputs(batch[1].shape,[202610299]))
                     for label,batch in monitor_batches.items()}
    input_fingerprints={label:fingerprints(batch) for label,batch in monitor_batches.items()}
    monitored={label:jax.device_put(batch,ds) for label,batch in monitor_batches.items()}
    counts=dict(total=sum(x.size for x in jax.tree.leaves(state.params)),
        trainable=sum(x.size for x in jax.tree.leaves(state.params.filter(cfg.trainable_filter))))
    S.dump(run/'parameter_counts.json',counts)

    def monitor(step):
        values={}
        with sharding.set_mesh(mesh),jax.default_matmul_precision('highest'):
            for label,batch in monitored.items():
                loss,noise,flow_time=map(np.asarray,eval_fn(state.ema_params,batch,jax.random.key(202610299)))
                values[label]=float(loss.mean())
                tag='restored' if args.verify_only else 'original'
                if not args.verify_only and (run/f'checkpoint_reference_{step}.json').exists():tag='resumed'
                np.savez(run/f'monitor_{step}_{label}_{tag}.npz',loss=loss,noise=noise,time=flow_time)
        assert all(np.isfinite(v) for v in values.values()),values
        with (run/'monitor.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(step=step,**values,smoke=args.smoke))+'\n')
        print('[monitor]',step,values,flush=True)
        return values

    reference=None
    if args.verify_only:
        assert args.resume and resuming
        reference=json.loads((run/f'checkpoint_reference_{int(state.step)}.json').read_text())
        assert input_fingerprints==reference['inputs'],'Checkpoint verification inputs changed'
    initial=monitor(int(state.step)); repeated=monitor(int(state.step))
    assert initial==repeated,'Initial forward is not repeatable'
    # Retain graph/IR evidence for the independent restore process.
    tag='restored' if args.verify_only else 'original'
    (run/f'model_graph_{tag}.txt').write_text(repr(eval_graph))
    # lower() reconstructs Observation with internal ArgInfo leaves rather than
    # arrays. Disable array checks only for this IR diagnostic, never training.
    with sharding.set_mesh(mesh),jax.default_matmul_precision('highest'),at.disable_typechecking():
        lowered=eval_fn.lower(state.ema_params,monitored['train'],jax.random.key(202610299))
    (run/f'monitor_ir_{tag}.txt').write_text(lowered.as_text())
    def vision_hash():
        host=F.host_snapshot(state.params.filter(nnx_utils.PathRegex('.*img.*')))
        return {jax.tree_util.keystr(k):hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
                for k,v in jax.tree_util.tree_flatten_with_path(host)[0]}
    before=vision_hash()
    if args.verify_only:
        expected=json.loads((run/'vision_update_audit.json').read_text())['after']
        assert before==expected,'Checkpoint restore changed vision weights'
        actual=fingerprints(F.host_snapshot(state.ema_params))
        assert actual==reference['ema'],'Checkpoint restore changed EMA weights/dtypes'
        comparisons={}
        for label in monitored:
            original_path=run/f'monitor_{int(state.step)}_{label}_original.npz'
            assert S.sha(original_path)==reference['output_file_hashes'][label],'Saved output baseline changed'
            with np.load(original_path) as a,np.load(run/f'monitor_{int(state.step)}_{label}_restored.npz') as b:
                assert np.array_equal(a['noise'],b['noise']) and np.array_equal(a['time'],b['time'])
                comparisons[label]=compare_fp32(b['loss'],a['loss'])
        S.dump(run/'restore_comparison.json',dict(step=int(state.step),expected=reference['monitor'],
            actual=initial,ema_equal=True,inputs_equal=True,replicas_equal=True,outputs=comparisons))
        assert all(v['passed'] for v in comparisons.values()),'Checkpoint loss changed beyond FP32 rounding tolerance'
        S.dump(run/'restore_audit.json',dict(passed=True,step=int(state.step),monitor=initial,
            ema_equal=True,inputs_equal=True,reference_sha256=S.sha(run/f'checkpoint_reference_{int(state.step)}.json')))
        S.dump(run/'status.json',dict(stage='verified',complete=True,step=int(state.step),seed=args.seed,
            checkpoint_dir=str(cfg.checkpoint_dir)))
        manager.close();return
    S.dump(run/'initial_audit.json',dict(monitor=initial,vision_hashes=before))
    durations=[]
    for iteration in range(int(state.step),args.steps):
        rng=np.random.default_rng(np.random.SeedSequence([args.seed,iteration]))
        ids=rng.choice(data.train_ids,size=cfg.batch_size,replace=True)
        batch=data.batch(ids)
        batch=jax.device_put((*batch,*draw_inputs(batch[1].shape,[args.seed,iteration,173])),ds)
        began=time.monotonic()
        with sharding.set_mesh(mesh):state,info=step_fn(train_rng,state,batch)
        values={k:float(v) for k,v in jax.device_get(info).items()}
        assert all(np.isfinite(v) for v in values.values()),values
        assert values['loss']<max(1e4,initial['train']*1e4) and values['grad_norm']<1e8,values
        durations.append(time.monotonic()-began)
        row=dict(step=iteration+1,step_s=durations[-1],elapsed_s=time.monotonic()-start,**values)
        with (run/'history.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
        if iteration==0 or (iteration+1)%25==0 or iteration+1==args.steps:
            avg=float(np.median(durations[-25:]))
            status=dict(stage='training',complete=False,target=args.steps,pid=os.getpid(),
                seed=args.seed,seconds_per_step=avg,remaining_estimate_s=avg*(args.steps-iteration-1),**row)
            S.dump(run/'status.json',status);print('[train]',status,flush=True)
        if (iteration+1)%500==0 or iteration+1==args.steps:
            current=monitor(iteration+1)
            if current['validation']>max(1e6,initial['validation']*1e4):
                raise FloatingPointError('Validation loss exploded')
        if (iteration+1)%cfg.save_interval==0 or iteration+1==args.steps:
            S.dump(run/'status.json',dict(stage='saving',complete=False,step=iteration+1,pid=os.getpid()))
            replicas={k:audit_replicas(v) for k,v in [('params',state.params),('ema',state.ema_params)]}
            S.dump(run/f'replica_audit_step{iteration+1}.json',replicas)
            assert all(v['passed'] for v in replicas.values()),'Parameter replicas diverged before save'
            snapshot=F.host_snapshot(state)
            reference_path=run/f'checkpoint_reference_{iteration+1}.json'
            assert not reference_path.exists(),'Do not overwrite an existing checkpoint reference'
            S.dump(reference_path,dict(step=iteration+1,monitor=current,inputs=input_fingerprints,
                ema=fingerprints(snapshot.ema_params),
                output_file_hashes={label:S.sha(run/f'monitor_{iteration+1}_{label}_original.npz') for label in monitored}))
            checkpoints.save_state(manager,snapshot,data,iteration+1)
            manager.wait_until_finished();del snapshot
            S.dump(run/'latest_checkpoint.json',dict(step=iteration+1,path=str(cfg.checkpoint_dir/str(iteration+1))))
    after=vision_hash()
    changed=before!=after
    if not resuming:
        assert changed==args.unfreeze_vision,'Vision update does not match freeze condition'
    S.dump(run/'vision_update_audit.json',dict(passed=True,changed=changed,expected=args.unfreeze_vision,
        before=before,after=after,resumed=resuming))
    manager.wait_until_finished();manager.close()
    S.dump(run/'status.json',dict(stage='trained',complete=True,step=args.steps,
        elapsed_s=time.monotonic()-start,seed=args.seed,checkpoint_dir=str(cfg.checkpoint_dir)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=S.DEFAULT)
    ap.add_argument('--name',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--steps',type=int,default=20000);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume',action='store_true');ap.add_argument('--unfreeze-vision',action='store_true');ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    try:train(args)
    except Exception as exc:
        S.dump(args.root/'training'/args.name/'status.json',dict(stage='failed',complete=False,
            error=repr(exc),traceback=traceback.format_exc()))
        raise
