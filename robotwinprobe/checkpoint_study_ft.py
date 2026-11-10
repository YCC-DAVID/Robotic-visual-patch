"""Matched pi0.5 base fine-tunes: only model seed varies between M0/M1/M2."""
import argparse
import dataclasses
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

import checkpoint_study_common as S
S.training_paths()
import g1_simple_pick_ft as F
from checkpoint_study_data import Data
import flax.nnx as nnx
import jax
import numpy as np
from openpi.training import config as C, checkpoints, sharding
from openpi.shared import nnx_utils
from orbax.checkpoint._src.serialization import type_handlers as orbax_types


def make_config(root, name, seed, steps, smoke=False, resume=False):
    original=C.get_config('pi05_g1_eef6d')
    model=S.model_config()
    data=S.data_factory(root)
    if smoke:
        data=dataclasses.replace(data,assets=C.AssetsConfig(assets_dir=str(root/'smoke_assets'),asset_id=S.ASSET))
    freeze=nnx.Any(model.get_freeze_filter(),nnx_utils.PathRegex('.*img.*'))
    return dataclasses.replace(original,name='pi05_robotwin_controlled',exp_name=name,
        model=model,data=data,freeze_filter=freeze,seed=seed,
        batch_size=16,num_train_steps=steps,fsdp_devices=1,
        weight_loader=F.FastBaseWeightLoader(str(S.BASE)),
        checkpoint_base_dir=str(root/'checkpoints'),assets_base_dir=str(root/'assets'),
        save_interval=steps if smoke else 5000,keep_period=5000,log_interval=25,
        wandb_enabled=False,resume=resume,overwrite=False,eval_interval=0)


def train(args):
    root=args.root.resolve(); protocol=S.prepare_protocol(root)
    cfg=make_config(root,args.name,args.seed,args.steps,args.smoke,args.resume)
    run=root/'training'/args.name;run.mkdir(parents=True,exist_ok=True)
    data_manifest=root/('data_smoke' if args.smoke else 'data_cache')/'manifest.json'
    norm=root/('smoke_assets' if args.smoke else 'assets')/S.ASSET/'norm_stats.json'
    recipe=dict(architecture=dataclasses.asdict(cfg.model),initialization=str(S.BASE),
        dataset_manifest_sha256=S.sha(data_manifest),normalization_sha256=S.sha(norm),
        batch=16,micro_batch=4,freeze_vision=True,language_lora_rank=16,full_action_expert=True,
        learning_rate=dataclasses.asdict(cfg.lr_schedule),optimizer=dataclasses.asdict(cfg.optimizer),
        ema=cfg.ema_decay,loss='upstream flow matching MSE, t~Beta(1.5,1)*.999+.001',
        image_augmentation='head crop95%/rotate+-5deg; all-view color jitter .3/.4/.5',
        action_horizon=32,raw_action_dim=14,discrete_state_input=True,
        source_hashes={str(p.relative_to(S.ROOT)):S.sha(p) for p in [Path(__file__),
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
    mesh=sharding.make_mesh(1)
    ds=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    manager,resuming=checkpoints.initialize_checkpoint_dir(cfg.checkpoint_dir,
        keep_period=cfg.keep_period,overwrite=False,resume=args.resume)
    train_rng,init_rng=jax.random.split(jax.random.key(args.seed))
    state,state_shard=F.upstream.init_train_state(cfg,init_rng,mesh,resume=resuming)
    if resuming:state=checkpoints.restore_state(manager,state,data)
    jax.block_until_ready(state)
    step_fn=jax.jit(functools.partial(F.accumulated_step,cfg,4),
        in_shardings=(replicated,state_shard,ds),out_shardings=(state_shard,replicated),donate_argnums=(1,))
    eval_fn=jax.jit(F.monitor_loss,in_shardings=(state_shard,ds,replicated),out_shardings=replicated)
    train_monitor=data.train_ids[np.linspace(0,len(data.train_ids)-1,4,dtype=int)]
    val_monitor=data.validation_ids[np.linspace(0,len(data.validation_ids)-1,4,dtype=int)]
    monitored={'train':jax.device_put(data.batch(train_monitor),ds),
               'validation':jax.device_put(data.batch(val_monitor),ds)}
    counts=dict(total=sum(x.size for x in jax.tree.leaves(state.params)),
        trainable=sum(x.size for x in jax.tree.leaves(state.params.filter(cfg.trainable_filter))))
    S.dump(run/'parameter_counts.json',counts)

    def monitor(step):
        with sharding.set_mesh(mesh):
            values={label:float(eval_fn(state,batch,jax.random.key(202610299))) for label,batch in monitored.items()}
        assert all(np.isfinite(v) for v in values.values()),values
        with (run/'monitor.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(step=step,**values,smoke=args.smoke))+'\n')
        print('[monitor]',step,values,flush=True)
        return values

    initial=monitor(int(state.step)); durations=[]
    for iteration in range(int(state.step),args.steps):
        rng=np.random.default_rng(np.random.SeedSequence([args.seed,iteration]))
        ids=rng.choice(data.train_ids,size=cfg.batch_size,replace=True)
        batch=jax.device_put(data.batch(ids),ds)
        began=time.monotonic()
        with sharding.set_mesh(mesh):state,info=step_fn(train_rng,state,batch)
        values={k:float(v) for k,v in jax.device_get(info).items()}
        assert all(np.isfinite(v) for v in values.values()),values
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
            snapshot=F.host_snapshot(state)
            checkpoints.save_state(manager,snapshot,data,iteration+1)
            manager.wait_until_finished();del snapshot
            S.dump(run/'latest_checkpoint.json',dict(step=iteration+1,path=str(cfg.checkpoint_dir/str(iteration+1))))
    manager.wait_until_finished();manager.close()
    S.dump(run/'status.json',dict(stage='trained',complete=True,step=args.steps,
        elapsed_s=time.monotonic()-start,seed=args.seed,checkpoint_dir=str(cfg.checkpoint_dir)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=S.DEFAULT)
    ap.add_argument('--name',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--steps',type=int,default=20000);ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    try:train(args)
    except Exception as exc:
        S.dump(args.root/'training'/args.name/'status.json',dict(stage='failed',complete=False,
            error=repr(exc),traceback=traceback.format_exc()))
        raise
