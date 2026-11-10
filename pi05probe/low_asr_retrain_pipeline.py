"""Independent, train-only placement audit -> fresh EOT -> paired rerun.

The original main-method artifacts are read-only. Corrective reruns on the
already-used init40-49 set remain explicitly labeled as such.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

import numpy as np
import main_method_pipeline as M
import visible_pos_pipeline as V
from main_method_job_runtime import install

P = M.P
ROOT = P.ROOT.parent
DEST = P.ROOT / 'out/low_asr_retrain_v1'
SELECTION = ROOT / 'out/main_results/low_asr_audit/visibility/selection.json'
TASKS = tuple(t for t in M.TASKS if t not in
              ('put_the_bowl_on_the_plate', 'put_the_cream_cheese_in_the_bowl'))
BATCH_SEED, EOT_SEED = 20261109, 2026110901


def plan(selection):
    checkpoint=ROOT/'checkpoints/pi05_libero_pytorch'
    return dict(schema='libero-low-asr-corrective-rerun-v1', tasks=list(TASKS),
        baseline_protocol_sha256=M.file_sha(M.DEST/'protocol.json'),
        baseline_summary_sha256=M.file_sha(M.DEST/'summary.json'),
        placement_selection_sha256=M.file_sha(SELECTION),
        checkpoint_sha256=M.file_sha(checkpoint/'model.safetensors'),
        checkpoint_config_sha256=M.file_sha(checkpoint/'config.json'),
        norm_stats_sha256={str(p.relative_to(checkpoint)):M.file_sha(p)
                          for p in checkpoint.glob('assets/**/norm_stats.json')},
        selection={t: selection[t] for t in TASKS},
        selection_data='training inits0-9 visibility only; no test-state position selection',
        training=dict(iters=500,batch=8,lr=.01,seed=BATCH_SEED,eot_seed=EOT_SEED,
                      noise_seed=20260901,patch_m=.13,resolution=128,
                      objective='unchanged full10x32 first-step velocity SSE',
                      initialization='constant gray .5',proxy_rmse_max=.02,
                      train_indices=list(range(10)),validation_indices=[10,11],
                      checkpoint='pi05_libero_pytorch',checkpoint_selection='fixed final step500'),
        evaluation=dict(indices=list(M.INDICES),conditions=list(M.CONDITIONS),
            interpretation='corrective rerun on already-used original test set; not a fresh independent holdout',
            metric='same task clean-success paired ASR; complete30 rows required'),
        source_sha256={n:M.file_sha(P.ROOT/n) for n in ['multitask_train.py',
            'dual_view_proxy.py','multitask_capture.py','cross_task_transfer_rollout.py',
            'render_proxy_policy_server.py','main_method_job_runtime.py']},
        eot_source_sha256=M.file_sha(ROOT/'robotwinprobe/texture_eot.py'))


def validate_calibration(task,folder,world):
    meta=M.read_json(folder/'capture.json')
    M.require(meta.get('complete') is True and meta['task']==task,'incomplete calibration')
    M.require(np.allclose(meta['cell_world'][:2],world,atol=1e-8,rtol=0),'wrong placement')
    for view in ['base','wrist']:
        with np.load(folder/(view+'.npz'),allow_pickle=False) as z:
            M.require(float(z['patch_m'])==.13,'wrong footprint')
            M.require(np.array_equal(z['cell_world'],meta['cell_world']),'view geometry mismatch')
            M.require(np.array_equal(z['train_mask'],z['ep_id']<10)
                      and np.array_equal(z['validation_mask'],z['ep_id']>=10),'wrong split')
            with np.load(M.SOURCE/'tasks'/task/'base.npz',allow_pickle=False) as source:
                for key in ['sim_states','state8','ep_id','env_seed','noise_seeds','clean_base','clean_wrist']:
                    M.require(np.array_equal(z[key],source[key]),'clean trajectory changed: '+key)
    return dict(folder=str(folder),cell_world=meta['cell_world'],frames=meta['frames'],
        train_frames=meta['train_frames'],validation_frames=meta['validation_frames'],
        init_states_sha256=meta['init_states_sha256'],capture_sha256=M.file_sha(folder/'capture.json'),
        calibration_sha256={v:M.file_sha(folder/(v+'.npz')) for v in ['base','wrist']})


def calibration(task,selection,gpu):
    folder=DEST/'tasks'/task
    source=M.SOURCE/'tasks'/task
    world=selection[task]['world']
    if not (folder/'capture.json').exists():
        if folder.exists() and any(folder.iterdir()):
            old=DEST/'recovery'/('calibration_'+task+'_'+V.now().replace(':',''))
            old.parent.mkdir(parents=True,exist_ok=True);folder.rename(old)
        folder.mkdir(parents=True,exist_ok=True)
        previous=M.read_json(source/'capture.json')
        if np.allclose(previous['cell_world'][:2],world,atol=1e-8,rtol=0):
            # Exact unchanged data are reused; fresh training never reuses Q.
            for name in ['base.npz','wrist.npz','capture.json']:
                (folder/name).symlink_to(source/name)
        else:
            V.stage('calibrating',task=task,world=world)
            with V.allocation('low_asr_calibrate_'+task,gpu,32000):
                V.run([P.SIM,'multitask_capture.py','--task',task,'--port','0',
                    '--out',str(folder),'--pool-from',str(source/'base.npz'),
                    '--cell-world='+','.join(map(str,world))],
                    'capture_'+task+'.log',V.sim_env(gpu),7200)
    return validate_calibration(task,folder,world)


def training(task,provenance,gpu):
    folder=DEST/'train'/task
    if not (folder/'results.json').exists():
        if folder.exists() and any(folder.iterdir()):
            old=DEST/'recovery'/('training_'+task+'_'+V.now().replace(':',''))
            old.parent.mkdir(parents=True,exist_ok=True);folder.rename(old)
        V.stage('training',task=task)
        with V.allocation('low_asr_train_'+task,gpu,32000):
            V.run([P.MODEL,'multitask_train.py','--tasks-root',str(DEST/'tasks'),
                '--tasks',task,'--out',str(folder),'--eot','--iters','500',
                '--batch','8','--lr','0.01','--seed',str(BATCH_SEED),
                '--eot-seed',str(EOT_SEED),'--proxy-rmse-max','0.02'],
                'train_'+task+'.log',dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='4'),14400)
    config,result=M.read_json(folder/'config.json'),M.read_json(folder/'results.json')
    M.require(config.get('complete') is True and config['seed_batch']==BATCH_SEED
              and config['eot']['seed']==EOT_SEED and config['iters']==500
              and config['batch']==8 and config['tasks']==[task],'training recipe differs')
    M.require(config['noise_seed']==20260901 and config['lr']==.01
              and config['mode']=='both' and config['patch_m']==.13
              and config['eot']['ranges']==M.EOT_RANGES
              and config['init']=='constant gray .5','training conditions differ')
    M.require(M.canonical_path(config['checkpoint'])==str(ROOT/'checkpoints/pi05_libero_pytorch'),
              'checkpoint differs')
    frozen=M.read_json(DEST/'protocol.json')
    for name in ['multitask_train.py','dual_view_proxy.py']:
        M.require(config['source_sha256'][name]==frozen['source_sha256'][name],
                  'training source changed: '+name)
    M.require(config['source_eot_sha256']==frozen['eot_source_sha256'],'EOT implementation changed')
    M.require(config['calibration_sha256'][task]==provenance['calibration_sha256'],'training data changed')
    M.require(result.get('complete') is True and result['iterations']==500
              and result['gradient_frame_forwards']==4000
              and result['texture_sha256']==M.file_sha(folder/'texture_0500.png'),'incomplete training')
    return folder


def evaluate(task,train,provenance,gpu,port):
    folder=DEST/'eval'/task;folder.mkdir(parents=True,exist_ok=True)
    textures=dict(clean=None,random=str(M.RANDOM),eot=str(train/'texture_0500.png'))
    manifest=dict(source_task=task,stage='low_asr_corrective_rerun',train_run=str(train),
        geometry=dict(patch_m=.13,cell_world=provenance['cell_world']),calibration=provenance,
        conditions=list(M.CONDITIONS),indices=list(M.INDICES),textures=textures,
        texture_sha256={k:None if v is None else M.file_sha(v) for k,v in textures.items()},
        protocol_sha256=M.file_sha(DEST/'protocol.json'))
    path=folder/'manifest.json';M.write_immutable(path,manifest)
    attempts=sorted(folder.glob('attempt_*'))
    for out in attempts:
        if (out/'episodes.json').exists():
            try:
                result=M.episode_summary(out/'episodes.json',task,path,True)
                V.write_json(DEST/'results'/(task+'.json'),result);return
            except (ValueError,KeyError,OSError,TypeError):
                pass
    number=max([int(x.name.split('_')[-1]) for x in attempts]+[-1])+1
    out=folder/('attempt_%03d'%number)
    V.stage('evaluating',task=task,attempt=number)
    with V.allocation('low_asr_eval_'+task,gpu,32000),V.server(gpu,port):
        V.run([P.SIM,'cross_task_transfer_rollout.py','--manifest',str(path),
            '--task',task,'--conditions',','.join(M.CONDITIONS),
            '--indices',','.join(map(str,M.INDICES)),'--port',str(port),
            '--out',str(out.relative_to(P.ROOT/'out'))],
            'eval_'+task+'_'+str(number)+'.log',V.sim_env(gpu),10800)
    result=M.episode_summary(out/'episodes.json',task,path,True)
    V.write_json(out/'complete.json',dict(complete=True,episodes_sha256=result['episodes_sha256']))
    V.write_json(DEST/'results'/(task+'.json'),result)


def main(args):
    global DEST,SELECTION
    DEST=Path(args.out).resolve();SELECTION=Path(args.selection).resolve()
    M.require(str(DEST).startswith(str(P.ROOT/'out')+'/'),'output outside pi05probe/out')
    for name in ['tasks','train','eval','logs','results','resources','services','recovery']:
        (DEST/name).mkdir(parents=True,exist_ok=True)
    V.DEST=DEST;install(V)
    selected=M.read_json(SELECTION)
    with (DEST/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        M.write_immutable(DEST/'protocol.json',plan(selected))
        requested=args.tasks.split(',') if args.tasks else list(TASKS)
        M.require(set(requested)<=set(TASKS) and len(set(requested))==len(requested),'invalid task subset')
        if args.prepare_only:
            print(json.dumps(dict(prepared=True,tasks=requested,out=str(DEST))));return
        gpu=json.loads(args.gpu)
        for task in requested:
            provenance=calibration(task,selected,gpu)
            train=training(task,provenance,gpu)
            evaluate(task,train,provenance,gpu,args.port)
        results={t:M.read_json(DEST/'results'/(t+'.json')) for t in TASKS
                 if (DEST/'results'/(t+'.json')).exists()}
        V.write_json(DEST/'summary.json',dict(complete=len(results)==len(TASKS),tasks=results,
            interpretation='corrective rerun on previously used test initializations'))
        V.stage('complete' if len(results)==len(TASKS) else 'requested_tasks_complete',tasks=requested)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu',default='{}');ap.add_argument('--tasks',default='')
    ap.add_argument('--selection',default=str(SELECTION));ap.add_argument('--out',default=str(DEST))
    ap.add_argument('--port',type=int,default=19461);ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    try:main(args)
    except BaseException as error:
        if not args.prepare_only:
            V.stage('blocked' if isinstance(error,V.ResourceBlocked) else 'failed',error=repr(error))
        raise
