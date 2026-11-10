"""One-model same-state Fourier exchange probe plus small LIBERO closed-loop diagnostic."""
import argparse,copy,fcntl,json,os,shutil,socket,subprocess,time
from pathlib import Path
import numpy as np
from crossarch_common import AUTH,dump,sha
from wam_orthogonal_control import call
ROOT=Path('/shared/user65/workspace/chence/WAMattack')
SOURCE=ROOT/'robotwinprobe/out/libero_pi_fastwam_crossarch_13cm_20261028'
DEFAULT=ROOT/'robotwinprobe/out/libero_frequency_restore_20261101'
MODES=['native_clean','native_attack','matched_clean','matched_attack','low_attack_high_clean','high_attack_low_clean']
PY={'pi':'/shared/user65/miniconda3/envs/openpi-server/bin/python','fast':'/shared/user65/miniconda3/envs/fastwam/bin/python'}
SIM='/shared/user65/miniconda3/envs/openpi-libero/bin/python'

def relocate(value):
    if isinstance(value,str):return value.replace('/home/user1/workspace/chence/WAMattack',str(ROOT))
    if isinstance(value,list):return [relocate(v) for v in value]
    if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
    return value

def prepare(root,kind):
    r=root/'models'/kind;r.mkdir(parents=True,exist_ok=True)
    p=relocate(json.loads((SOURCE/'models'/kind/'protocol.json').read_text()))
    p.update(experiment='libero_same_state_frequency_restore',model_kind=kind,control_port=18781 if kind=='pi' else 18782)
    p['frequency_restore']=dict(source_root=str(SOURCE),source_condition=kind,
        source_texture_sha256=sha(SOURCE/'textures'/(kind+'.png')),
        primary_cutoff=.125,probe_cutoffs=[.0625,.125,.25],frequency_units='radial cycles per 224-input pixel, per camera; not cycles per world patch',
        modes=MODES,probe_indices=list(range(40,50)),training=False,
        operation='L(attack)+H(clean) and L(clean)+H(attack) after native224 resize; same state/noise; global image-domain diagnostic',
        range='one positive affine map shared by four groups within a camera/pair/cutoff; original native clean/full separately rechecked',
        dtype='pi float32, FastWAM BF16; audit spectral drift after model input casting; no hidden clipping',
        scope='frequency exchange in policy input; edited regions may extend beyond physical patch; no claim of new material-realizable physical patch',
        closed_loop='primary cutoff only, init40-49, each policy original wait/limit/replan; no training or cutoff selection by outcomes')
    p['evaluation']['conditions']=MODES
    p['evaluation']['texture_paths']={'native_attack':str(SOURCE/'textures'/(kind+'.png'))}
    path=r/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:dump(path,p)
    (r/'logs').mkdir(exist_ok=True)
    config_dir=root/'libero_config';config_dir.mkdir(exist_ok=True)
    (config_dir/'config.yaml').write_text((ROOT/'pi05probe/libero_config/config.yaml').read_text().replace('/home/user1/workspace/chence/WAMattack',str(ROOT)))
    if kind=='fast':
        for name in ['prompts.npz','checkpoints/libero_uncond_2cam224.pt','checkpoints/libero_uncond_2cam224_dataset_stats.json']:
            dst=r/name;dst.parent.mkdir(exist_ok=True)
            if not dst.exists():os.link(SOURCE/'models/fast'/name,dst)
    snapshots=r/'source_snapshot';snapshots.mkdir(exist_ok=True)
    for name in ['libero_frequency_restore_run.py','libero_frequency_backend.py','libero_frequency_mix.py','libero_frequency_rollout.py','libero_pi_adapter.py','fastwam_adapter.py']:
        f=ROOT/'robotwinprobe'/name
        if f.exists():shutil.copy2(f,snapshots/name)
    dump(snapshots/'hashes.json',{f.name:sha(f) for f in snapshots.glob('*.py')})
    return r,p

def summarize(root):
    from stage_joint_report import interval,wilson
    outcomes={m:{i:json.loads((root/'episodes'/m/f'init{i}.json').read_text()) for i in range(40,50)} for m in MODES}
    rows=[]
    for mode in MODES:
        baseline='native_clean' if mode.startswith('native_') else 'matched_clean'
        eligible=[i for i in range(40,50) if outcomes[baseline][i]['success']]
        failures=[int(not outcomes[mode][i]['success']) for i in eligible]
        rows.append(dict(condition=mode,baseline=baseline,completed=10,n=len(eligible),failures=sum(failures),ASR=float(np.mean(failures)) if failures else None,CI95=wilson(failures)))
    baseline_set=[i for i in range(40,50) if outcomes['matched_clean'][i]['success']]
    for mode in ['low_attack_high_clean','high_attack_low_clean']:
        diff=[int(not outcomes[mode][i]['success'])-int(not outcomes['matched_attack'][i]['success']) for i in baseline_set]
        next(r for r in rows if r['condition']==mode).update(excess_over_matched_full=float(np.mean(diff)) if diff else None,paired_CI95=interval(diff))
    dump(root/'results.json',dict(complete=True,rows=rows,scope='small input-frequency intervention; not a physical-material attack proof'))

def main(a):
    root,p=prepare(a.root.resolve(),a.kind);logs=root/'logs'
    status=lambda stage,**kw:dump(root/'status.json',dict(stage=stage,host=socket.gethostname(),gpu=a.gpu,physical_gpu=a.physical_gpu,pid=os.getpid(),**kw))
    env=dict(os.environ,WAMATTACK_PROJECT_ROOT=str(ROOT),WAMATTACK_LIBERO_CONFIG_PATH=str(a.root.resolve()/'libero_config'),
        LIBERO_CONFIG_PATH=str(a.root.resolve()/'libero_config'),CUDA_VISIBLE_DEVICES=a.gpu,MUJOCO_EGL_DEVICE_ID=str(a.physical_gpu),
        MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENPI_DATA_HOME='/shared/user65/.cache/openpi')
    env.pop('PYTHONPATH',None);backend=None
    with (root/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status('loading_backend');(root/'ready.json').unlink(missing_ok=True)
            with (logs/'backend.log').open('a') as f:backend=subprocess.Popen([PY[a.kind],'-u',str(ROOT/'robotwinprobe/libero_frequency_backend.py'),'--root',str(root)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+900
            while not (root/'ready.json').exists():
                if backend.poll() is not None:raise RuntimeError('backend failed; see logs/backend.log')
                if time.monotonic()>deadline:raise TimeoutError('model loading')
                time.sleep(2)
            assert call(root,'ping')['ready']
            for index in p['frequency_restore']['probe_indices']:
                status('initial_same_state_probe',index=index,backend_pid=backend.pid)
                if not (root/'probe'/f'init{index}.json').exists():call(root,'probe',index=index)
            status('probe_complete',backend_pid=backend.pid)
            for mode in MODES:
                for index in range(40,50):
                    status('closed_loop',condition=mode,index=index,backend_pid=backend.pid)
                    # Legacy robosuite validates the physical EGL index against
                    # numeric CUDA_VISIBLE_DEVICES; the model keeps UUID binding.
                    sim_env=dict(env,CUDA_VISIBLE_DEVICES=str(a.physical_gpu))
                    with (logs/f'{mode}_init{index}.log').open('a') as f:subprocess.run([SIM,'-u',str(ROOT/'robotwinprobe/libero_frequency_rollout.py'),'--root',str(root),'--condition',mode,'--index',str(index)],cwd=ROOT,env=sim_env,stdout=f,stderr=subprocess.STDOUT,check=True)
                    if mode.startswith('native_'):
                        old_cond='clean' if mode=='native_clean' else a.kind
                        old=json.loads((SOURCE/'models'/a.kind/'episodes'/old_cond/f'init{index}.json').read_text())
                        new=json.loads((root/'episodes'/mode/f'init{index}.json').read_text())
                        assert old['success']==new['success'],('native outcome drift',mode,index)
            summarize(root);status('complete',backend_pid=backend.pid)
        except BaseException as e:status('failed',error=repr(e));raise
        finally:
            if backend is not None and backend.poll() is None:
                backend.terminate();backend.wait(timeout=30)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--kind',choices=['pi','fast'],required=True);p.add_argument('--gpu',required=True);p.add_argument('--physical-gpu',type=int,required=True);main(p.parse_args())
