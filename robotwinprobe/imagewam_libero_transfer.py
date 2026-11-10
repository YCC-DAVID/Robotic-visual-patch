"""Frozen LIBERO ImageWAM as a third, unseen target of existing EOT PNGs.

70 actual-render paired episodes. No texture or policy training and no target
gradients. The LIBERO release is pinned; clean usability gates attack evaluation.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Client,Listener

from crossarch_common import AUTH,dump,sha
import libero_joint_maxmin as G

ROOT=Path(os.environ.get('WAMATTACK_PROJECT_ROOT','/shared/user65/workspace/chence/WAMattack'))
DEFAULT=ROOT/'robotwinprobe/out/imagewam_libero_eot_transfer_20261030'
REPORT=ROOT/'result/53_imagewam_libero_eot_transfer.md'
WEIGHTS=ROOT/'checkpoints/imagewam_release/libero/flux2_klein_4b'
REVISION='5d1b7651b6a0e863fe0556ddc409213c38f6ee85'
EXPECTED_MODEL_SHA='53620f93f8772d2047f27f9ca9b4a1f56cdd7932de566282e0f4a80cfff21cb5'
AE=Path('/shared/user64/workspace/yuhao/imagewam/checkpoints/flux2/FLUX.2-dev/ae.safetensors')
MODEL_PY='/shared/user64/workspace/yuhao/imagewam/ImageWAM/.venv/bin/python'
SIM='/shared/user65/miniconda3/envs/openpi-libero/bin/python'
SOURCE=ROOT/'robotwinprobe/out/libero_pi_fastwam_crossarch_13cm_20261028'
SINGLE=ROOT/'robotwinprobe/out/libero13_eot_transfer_20261030'
JOINT=ROOT/'robotwinprobe/out/libero13_joint_eot_20261029'
PORT=18752
CONDITIONS=['clean','random','pi_control','fast_control','pi_eot','fast_eot','joint_eot']


def fixed(path,value):
    if path.exists():assert json.loads(path.read_text())==value,('immutable config changed',str(path))
    else:dump(path,value)


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    for name in ['logs','audit','textures']:(root/name).mkdir(exist_ok=True)
    paths=dict(random=SOURCE/'textures/random.png',pi_control=SOURCE/'textures/pi.png',
        fast_control=SOURCE/'textures/fast.png',pi_eot=SINGLE/'textures/pi_eot.png',
        fast_eot=SINGLE/'textures/fast_eot.png',joint_eot=JOINT/'textures/eot_nominal.png')
    protocol=dict(experiment='imagewam_libero_frozen_eot_transfer',target_only=True,
        task='put_the_bowl_on_the_plate',suite='libero_goal',
        model=dict(repo='yuyangalin/ImageWAM-FLUX.2-4B-LIBERO',revision=REVISION,
            checkpoint_sha256=EXPECTED_MODEL_SHA,weights=str(WEIGHTS),native_ae=str(AE),
            action_dim=7,proprio_dim=8,text_length=512,action_horizon=16,flow_steps=10),
        patch=dict(size_m=.13,resolution=128,position='same fixed table site101'),
        evaluation=dict(indices=list(range(40,50)),conditions=CONDITIONS,expected_episodes=70,
            wait=30,steps=400,replan=12,
            native_reference='official flux2 LIBERO script replan12/horizon16; release config10 denoising steps',
            seed_rule='env10000+init; action20261006+init*1000+query; paired within target',
            clean_minimum=6,ASR='paired failures / THIS ImageWAM model clean successes',
            images='same actual MuJoCo base+wrist renders; native PIL crop/resize224,224 then horizontal concat',
            model_kind='imagewam',gripper='official native denorm,2*x-1,invert,binarize'),
        textures={k:dict(source=str(p),sha256=sha(p)) for k,p in paths.items()},
        isolation='freeze all existing PNGs; no ImageWAM gradients, no new policy/texture optimization or selection',
        limits='nominal 13cm geometry; third-model transfer only, not whitebox robustness or physical placement coverage')
    fixed(root/'protocol.json',protocol)
    for name,path in paths.items():
        dest=root/'textures'/(name+'.png')
        if not dest.exists():shutil.copy2(path,dest)
        assert sha(dest)==protocol['textures'][name]['sha256']
    r=root/'models/imagewam';r.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/'models/fast/protocol.json').read_text())
    cfg.update(experiment=protocol['experiment'],model_kind='imagewam',control_port=PORT,
        checkpoint=str(WEIGHTS/'model.pt'),parent_root=str(root),
        calibration=str(ROOT/'pi05probe/out/dual_view/calibration/base.npz'),
        isolation=protocol['isolation'])
    cfg['training']=dict(enabled=False,target_only=True)
    cfg['evaluation'].update(conditions=CONDITIONS,steps=400,wait=30,replan=12,horizon=16,
        flow_steps=10,clean_minimum=6,baseline_reuse=False,expected_new_episodes=70,
        reference='fresh ImageWAM clean; never borrow another model denominator',
        texture_paths={c:str(root/'textures'/(c+'.png')) for c in CONDITIONS if c!='clean'},
        texture_paths_by_init={})
    fixed(r/'protocol.json',cfg)
    config=root/'libero_config';config.mkdir(exist_ok=True)
    (config/'config.yaml').write_text((SINGLE/'libero_config/config.yaml').read_text())
    snapshot=root/'source_snapshots'/str(os.getpid());snapshot.mkdir(parents=True,exist_ok=True)
    for name in ['imagewam_libero_adapter.py','imagewam_libero_transfer.py','fastwam_libero_trial.py']:
        shutil.copy2(ROOT/'robotwinprobe'/name,snapshot/name)
    fixed(snapshot/'manifest.json',{p.name:sha(p) for p in snapshot.glob('*.py')})
    return protocol


def report(root):
    s=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'prepared'}
    rows=G.outcomes(root,'imagewam');clean={i for i,v in rows.get('clean',{}).items() if v}
    lines=['# ImageWAM LIBERO：冻结EOT纹理向第三架构迁移','',
        f'状态：`{s["stage"]}`；节点：`{s.get("host","unknown")}`。','',
        '目标为官方ImageWAM-FLUX.2-4B-LIBERO发布权重，使用原生AE、7维动作/8维状态/512文本、16动作跨度/10步去噪/每12动作重规划。',
        '同一LIBERO put_the_bowl_on_the_plate任务、init40–49，13cm/site101桌面纹理，经原生MuJoCo两相机渲染。',
        '固定第52号两张单源EOT及第51号联训EOT PNG，另测两张无EOT专用纹理和Random；ImageWAM从未参与这些纹理优化。',
        '不重训patch/模型、不计算目标梯度；这里只测试第三模型零样本迁移，不等同自适应白盒鲁棒性。',
        'fresh clean至少6/10成功才进入攻击；每个条件ASR均使用ImageWAM自己的clean成功集，不能复用π/FastWAM分母。','',
        '| 条件 | 完成 | 配对失败/clean成功 |','|---|---:|---:|']
    for c in CONDITIONS:
        data=rows.get(c,{});pairs=clean & data.keys()
        score='—' if c=='clean' or not pairs else f'{sum(not data[i] for i in pairs)}/{len(pairs)}'
        lines.append(f'| {c} | {len(data)}/10 | {score} |')
    if s.get('error'):lines+=['',s['error']]
    lines+=['','单任务10初态、固定noise；仅nominal纹理曝光。零失败不能推断ImageWAM整体抗EOT，也不代表同任务白盒EOT训练失败。',
        f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(lines)+'\n')


def status(root,stage,**kwargs):
    dump(root/'status.json',dict(stage=stage,host=socket.gethostname(),pid=os.getpid(),updated=time.time(),**kwargs))
    report(root)


def check_assets(root):
    path=WEIGHTS/'model.pt'
    assert path.exists() and path.stat().st_size==9035379577,'official LIBERO checkpoint download not complete'
    manifest=json.loads((WEIGHTS/'release_manifest.json').read_text())
    assert manifest['sha']==REVISION
    audit=root/'audit/asset_hashes.json'
    stamp=dict(size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
    previous=json.loads(audit.read_text()) if audit.exists() else None
    if previous and previous['checkpoint_stat']==stamp and previous['model_sha256']==EXPECTED_MODEL_SHA:return
    actual=sha(path);assert actual==EXPECTED_MODEL_SHA,('release SHA256 mismatch',actual)
    dump(audit,dict(checkpoint_stat=stamp,model_sha256=actual,revision=REVISION,
        config_sha256=sha(WEIGHTS/'config.yaml'),stats_sha256=sha(WEIGHTS/'dataset_stats.json'),native_ae_sha256=sha(AE)))


def resource(root,gpu,own,tag,required=1024):
    skill=ROOT/'skills/gpu-preflight/SKILL.md';assert skill.read_text()
    script=os.environ.get('WAMATTACK_GPU_PREFLIGHT','/shared/user65/.codex/skills/gpu-preflight/scripts/check_gpus.py')
    scan=subprocess.run([sys.executable,script,'--json'],capture_output=True,text=True)
    data=json.loads(scan.stdout);data['skill_sha256']=sha(skill)
    dump(root/'audit'/('cluster_preflight_'+tag+'.json'),data)
    return G.preflight(root,gpu,own,tag,required)


def call(command,**kwargs):
    with Client(('127.0.0.1',PORT),authkey=AUTH) as connection:
        connection.send(dict(command=command,**kwargs));result=connection.recv()
    if 'error' in result:raise RuntimeError(result['error']+'\n'+result.get('traceback',''))
    return result


def server(root):
    from imagewam_libero_adapter import Backend
    backend=Backend(root)
    with Listener(('127.0.0.1',PORT),authkey=AUTH) as listener:
        dump(root/'ready.json',dict(pid=os.getpid(),host=socket.gethostname()))
        running=True
        while running:
            with listener.accept() as connection:
                try:
                    req=connection.recv();cmd=req['command']
                    if cmd=='shutdown':running=False;result=dict(stopped=True)
                    elif cmd=='audit':result=backend.audit(root)
                    elif cmd=='infer':result=backend.infer(req)
                    else:raise ValueError('target-only server forbids gradient/training commands: '+cmd)
                    connection.send(result)
                except Exception as error:
                    traceback.print_exc();connection.send(dict(error=repr(error),traceback=traceback.format_exc()))


def pipeline(root,gpu):
    root=root.absolute();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(root);backend=None
    try:
        status(root,'verify_release_assets');check_assets(root)
        env=dict(os.environ,WAMATTACK_PROJECT_ROOT=str(ROOT),WAMATTACK_LIBERO_CONFIG_PATH=str(root/'libero_config'),
            PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
        if not (root/'contexts.pt').exists():
            status(root,'encode_native_prompt')
            uuid=resource(root,gpu,[],'encode_prompt',10000)
            env.update(CUDA_VISIBLE_DEVICES=uuid,MUJOCO_EGL_DEVICE_ID=str(gpu))
            with (root/'logs/prompt.log').open('a') as log:
                subprocess.run([MODEL_PY,'-u',str(ROOT/'robotwinprobe/imagewam_libero_adapter.py'),
                    'encode-context','--root',str(root)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status(root,'start_imagewam')
        uuid=resource(root,gpu,[],'start_imagewam',16000)
        env.update(CUDA_VISIBLE_DEVICES=uuid,MUJOCO_EGL_DEVICE_ID=str(gpu))
        ready=root/'ready.json'
        if ready.exists():ready.unlink()
        with (root/'logs/model.log').open('a') as log:
            backend=subprocess.Popen([MODEL_PY,'-u',str(Path(__file__).absolute()),'server','--root',str(root)],
                env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        dump(root/'execution.json',dict(host=socket.gethostname(),pipeline_pid=os.getpid(),
            backend_pid=backend.pid,gpu=gpu,uuid=uuid))
        began=time.monotonic()
        while not ready.exists():
            if backend.poll() is not None or time.monotonic()-began>600:raise RuntimeError('ImageWAM initialization failed; see model.log')
            time.sleep(2)
        status(root,'native_audit');call('audit')
        for condition in CONDITIONS:
            for index in range(40,50):
                path=root/'models/imagewam/episodes'/condition/f'init{index}.json'
                if path.exists():assert 'success' in json.loads(path.read_text());continue
                status(root,f'evaluate_{condition}_{index}')
                resource(root,gpu,[backend.pid],f'render_{condition}_{index}')
                sim_env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu))
                with (root/'logs'/f'{condition}_{index}.log').open('a') as log:
                    subprocess.run([SIM,'-u',str(ROOT/'robotwinprobe/fastwam_libero_trial.py'),'rollout',
                        '--root',str(root/'models/imagewam'),'--condition',condition,'--index',str(index)],
                        env=sim_env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if condition=='clean':
                data=G.outcomes(root,'imagewam')['clean']
                assert len(data)==10 and sum(data.values())>=6,('ImageWAM clean below6/10; attack ASR not interpretable',data)
                dump(root/'audit/clean_gate.json',dict(passed=True,successes=sum(data.values()),episodes=10,outcomes=data))
        status(root,'complete')
    except Exception as error:
        status(root,'runtime_error',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        if backend and backend.poll() is None:
            try:call('shutdown')
            except Exception:backend.terminate()
            try:backend.wait(timeout=30)
            except subprocess.TimeoutExpired:backend.kill();backend.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','pipeline','server','report'])
    parser.add_argument('--root',type=Path,default=DEFAULT);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.root);status(args.root,'prepared')
    elif args.command=='pipeline':pipeline(args.root,args.gpu)
    elif args.command=='server':server(args.root)
    else:report(args.root)
