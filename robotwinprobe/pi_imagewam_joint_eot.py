"""#57: one frozen-policy pi/ImageWAM joint EOT texture; FastWAM held out."""
import argparse, concurrent.futures, fcntl, hashlib, json, os, shutil, socket
import subprocess, sys, time, traceback
from pathlib import Path
from multiprocessing.connection import Client
import numpy as np
from PIL import Image
from crossarch_common import AUTH, dump, sha
import libero_joint_maxmin as M
import libero_joint_eot as E
import imagewam_libero_self_attack as S
import imagewam_libero_source_transfer as T
from texture_eot import RANGES

ROOT=M.ROOT
DEFAULT=ROOT/'robotwinprobe/out/pi_imagewam_joint_eot_20261031'
REPORT=ROOT/'result/57_pi_imagewam_joint_eot.md'
PI_SOURCE=ROOT/'robotwinprobe/out/libero13_eot_transfer_20261030'
IMAGE_SOURCE=S.DEFAULT
SOURCES=['pi','imagewam']
KINDS=SOURCES+['fast']
CONDITIONS=['clean','random','joint_eot']
PORTS=dict(pi=18802,imagewam=18803,fast=18804)
PY=dict(M.PY,imagewam=S.R.MODEL_PY)
FILES=['pi_imagewam_joint_eot.py','libero_joint_eot.py','libero_joint_maxmin.py',
       'imagewam_libero_self_attack.py','imagewam_libero_whitebox_adapter.py',
       'imagewam_libero_adapter.py','imagewam_libero_transfer.py',
       'imagewam_libero_source_transfer.py','libero_pi_adapter.py','fastwam_adapter.py',
       'fastwam_libero_trial.py','texture_eot.py']

def read(path):return json.loads(path.read_text())
def fixed(path,value):
    if path.exists():assert read(path)==value,('immutable artifact changed',str(path))
    else:dump(path,value)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def array_sha(value):return hashlib.sha256(value.tobytes()).hexdigest()
def call(kind,command,**data):
    with Client(('127.0.0.1',PORTS[kind]),authkey=AUTH) as c:
        c.send(dict(command=command,**data));out=c.recv()
    if 'error' in out:raise RuntimeError(kind+': '+out['error']+'\n'+out.get('traceback',''))
    return out
def parallel(command,payloads):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={k:pool.submit(call,k,command,**v) for k,v in payloads.items()}
        return {k:f.result() for k,f in futures.items()}
def history(path):
    with np.load(path) as z:return json.loads(str(z['history']))
def baseline(kind):return S.PREVIOUS if kind=='imagewam' else M.SOURCE

def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    for n in ['logs','audit','textures','training','gradient_ledgers','source_snapshots']:(root/n).mkdir(exist_ok=True)
    prior_pi=read(PI_SOURCE/'completion_verification.json')
    assert prior_pi['completed_episodes']==100
    for k in ['pi','fast']:
        v=prior_pi['models'][k]
        assert v['training_steps']==v['source_only_gradient_records']==500 and v['history_ledger_match'] and v['native_audit']['passed']
    assert read(IMAGE_SOURCE/'completion_verification.json')['passed']
    states=dict(pi=PI_SOURCE/'training/pi/state.npz',imagewam=IMAGE_SOURCE/'training/state.npz')
    batches=read(M.SOURCE/'training_plan.json');draws=read(E.DEFAULT/'eot_plan.json')
    refs={}
    for k,path in states.items():
        hh=history(path);assert len(hh)==500
        for it,h in enumerate(hh):
            pr=h['gradient_provenance']
            assert pr['source_model']==pr['server_model']==k
            assert pr['indices']==batches[it]['indices'] and pr['seeds']==batches[it]['seeds']
            assert pr['draws_sha256']==digest(draws[it])
        refs[k]=[h['loss'] for h in hh]
        assert all(np.isfinite(x) and x>0 for x in refs[k])
    p=dict(experiment='pi_imagewam_joint_eot',sources=SOURCES,heldout_target='fast',
        task='put_the_bowl_on_the_plate',suite='libero_goal',patch=dict(size_m=.13,resolution=128,initialization=.5,footprint='fixed site101 square'),
        training=dict(steps=500,batch=8,lr=.01,policy_frozen=True,update='sign-PGD clip[0,1]',
            objective=dict(pi='native first10x32 velocity SSE from same-noise clean',imagewam='native first16x7 velocity MSE from same-noise clean'),
            fusion='r_k=current batch loss / fixed own single-source EOT loss at same iteration; choose smallest r_k; tie->pi; update chosen source gradient sign',
            references={k:dict(path=str(path),sha256=sha(path)) for k,path in states.items()},
            source_verification_sha256={str(x):sha(x/'completion_verification.json') for x in [PI_SOURCE,IMAGE_SOURCE]},
            reference_caveat='uses existing EOT specialist curves #52/#55; differs from #51 nominal specialist normalization; no new specialist fitting',
            batch_plan_sha256=sha(M.SOURCE/'training_plan.json'),eot_plan_sha256=sha(E.DEFAULT/'eot_plan.json'),
            eot_ranges=RANGES,shared_draw='same draw/state for both sources and both camera projections',
            source_states='existing common 155-state pi trajectory calibration bank',
            imagewam_gradient='native AE and checkpointed video cache, native PIL/quantization STE from audited #55',
            fast_feedback=False,selection='fixed final500, sealed before all evaluation'),
        evaluation=dict(models=KINDS,conditions=CONDITIONS,indices=list(range(40,50)),expected_episodes=90,
            baselines='fresh clean and random exactly match #46 pi/fast and #53 ImageWAM per initial state',
            imagewam_clean_gate=6,ASR='paired failure / each model own clean-success set',
            native='unchanged native model rollout protocols, physical MuJoCo dual camera rendering',stress=False),
        random_sha256=sha(IMAGE_SOURCE/'textures/random.png'),contexts_sha256=sha(IMAGE_SOURCE/'contexts.pt'),
        imagewam_model=read(IMAGE_SOURCE/'protocol.json')['model'],new_textures=1,new_policy_training=False)
    fixed(root/'protocol.json',p);fixed(root/'training_plan.json',batches);fixed(root/'eot_plan.json',draws)
    fixed(root/'specialist_reference.json',refs)
    fixed(root/'reference.json',read(E.CONTROL/'reference.json')) # pi's strict nominal audit
    for name in ['contexts.pt','textures/random.png','audit/asset_hashes.json','libero_config/config.yaml']:
        dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(IMAGE_SOURCE/name,dst)
        assert sha(dst)==sha(IMAGE_SOURCE/name)
    for k in KINDS:
        r=root/'models'/k;r.mkdir(parents=True,exist_ok=True)
        src=IMAGE_SOURCE if k=='imagewam' else PI_SOURCE
        cfg=M.rebase(read(src/'models'/k/'protocol.json'))
        cfg.update(experiment=p['experiment'],control_port=PORTS[k],parent_root=str(root),isolation='joint sources pi/imagewam; FastWAM inference only')
        cfg['training']=dict(p['training'],isolated_source_model=k) if k in SOURCES else dict(enabled=False,inference_only=True,target_gradients_forbidden=True)
        cfg['evaluation'].update(conditions=CONDITIONS,expected_new_episodes=30,baseline_reuse=False,
            texture_paths={c:str(root/'textures'/(c+'.png')) for c in ['random','joint_eot']},texture_paths_by_init={})
        fixed(r/'protocol.json',cfg)
        if k=='fast':
            for name in ['prompts.npz','checkpoints/libero_uncond_2cam224.pt','checkpoints/libero_uncond_2cam224_dataset_stats.json']:
                dst=r/name;dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():os.link(src/'models/fast'/name,dst)
                assert sha(dst)==sha(src/'models/fast'/name)
    snap=root/'source_snapshots'/('prepare_'+str(os.getpid()));snap.mkdir(exist_ok=True)
    for name in FILES:shutil.copy2(ROOT/'robotwinprobe'/name,snap/name)
    for name in ['dual_view_proxy.py','render_world_proxy.py']:shutil.copy2(ROOT/'pi05probe'/name,snap/name)
    fixed(snap/'manifest.json',{f.name:sha(f) for f in snap.glob('*.py')})
    return p

def report(root):
    st=read(root/'status.json') if (root/'status.json').exists() else {'stage':'prepared'}
    prog=read(root/'training/progress.json') if (root/'training/progress.json').exists() else {'iteration':0,'chosen':{}}
    lines=['# π＋ImageWAM EOT联训及向FastWAM迁移','',f"状态：`{st['stage']}`；训练 {prog['iteration']}/500；步数分配 {prog['chosen']}。",'',
        '只优化一张共享13cm/Q128纹理，π和官方ImageWAM LIBERO4B冻结；FastWAM不参与训练、归一化或纹理选择。',
        '灰色Q=.5，500×8，sign-PGD .01；沿用原batch/noise与四项EOT：旋转±15°、xy各±5%、缩放.9–1.1、统一RGB增益.9–1.1。同一变换供两个源及两相机共享。',
        '采用归一化max-min：用#52 π与#55 ImageWAM既有单源EOT曲线作固定标尺，每步选择相对达成率较低源的梯度符号更新。不是原始L0跨模型大小排序。#51用nominal专用曲线，本实验改为两源现有EOT曲线；未额外训练标尺。',
        '固定最终500步PNG，封存后才评估；三模型各clean/Random/joint_eot×10初态，共90条真实MuJoCo闭环。clean与Random须逐初态复现历史协议。','',
        '| 模型 | 角色 | 条件 | 有效回合 | 配对ASR | 失败初态 |','|---|---|---|---:|---:|---|']
    for k in KINDS:
        rows=M.outcomes(root,k);clean=rows.get('clean',{});eligible=[i for i,ok in clean.items() if ok]
        for c in CONDITIONS:
            d=rows.get(c,{});score='—';fails=[]
            if c!='clean' and len(d)==len(clean)==10:
                fails=[i for i in eligible if not d[i]];score=f'{len(fails)}/{len(eligible)}'
            lines.append(f"| {k} | {'训练源' if k in SOURCES else '留出目标'} | {c} | {len(d)}/10 | {score} | {fails if score!='—' else '—'} |")
    if st.get('error'):lines+=['',st['error']]
    if (root/'completion_verification.json').exists():
        v=read(root/'completion_verification.json')
        lines+=['','最终核验通过：90/90有效闭环、500次共享更新、每源500条梯度账本，history与账本/原始抽样/纹理SHA逐步匹配，并从灰色起点重放更新。FastWAM没有训练梯度请求。',
            '主要判断是共享纹理能否同时干扰两个训练源，以及能否向FastWAM迁移；自身能力与留出迁移分别报告，不用训练loss代替ASR。单次训练、同任务10初态用于判断本轮效果及与既有单源结果比较，不作稳定最优排名。',
            f"进程收尾：{v.get('remote_backends_exited',False)}；流水线锁释放：{v.get('pipeline_lock_released',False)}。"]
    lines+=['',f'原始目录：`{root}`。未创建新定时任务。']
    REPORT.write_text('\n'.join(lines)+'\n')
def status(root,stage,**kw):
    dump(root/'status.json',dict(stage=stage,host=socket.gethostname(),pid=os.getpid(),updated=time.time(),**kw));report(root)

def resource(root,gpu,own,tag,required=1024):
    skill=ROOT/'skills/gpu-preflight/SKILL.md';assert skill.read_text()
    for pid in own:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert 'pi_imagewam_joint_eot.py server' in cmd and str(root) in cmd,('foreign backend',pid)
    assert not(socket.gethostname().split('.')[0]=='nnmc63' and gpu==0)
    uuid=S.R.resource(root,gpu,own,tag,required)
    ecc=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total','--format=csv,noheader,nounits'],text=True).strip()
    assert [x.strip() for x in ecc.split(',')]==['0','0'],('bad/unknown ECC',ecc)
    dump(root/'audit'/('ecc_'+tag+'.json'),dict(uuid=uuid,ecc=ecc,time=time.time()));return uuid

def choose(raw,refs,it):
    ratios={k:raw[k]/refs[k][it] for k in SOURCES}
    assert all(np.isfinite(v) and v>=0 for v in ratios.values())
    return min(SOURCES,key=ratios.get),ratios
def train(root):
    dest=root/'training';state=dest/'state.npz';q=np.full((1,3,128,128),.5,np.float32);hh=[];signs=[]
    if state.exists():
        with np.load(state) as z:q=z['texture'].copy();hh=json.loads(str(z['history']));signs=list(z['update_signs'])
    assert len(signs)==len(hh)
    for k in SOURCES:
        lp=root/'gradient_ledgers'/(k+'.jsonl')
        ll=[json.loads(x) for x in lp.read_text().splitlines()] if lp.exists() else []
        assert len(ll)==len(hh),'Uncommitted ledger; preserve/reconcile before resume'
        assert all(h['provenance'][k]==l for h,l in zip(hh,ll))
    batches=read(root/'training_plan.json');draws=read(root/'eot_plan.json');refs=read(root/'specialist_reference.json')
    for it in range(len(hh),500):
        before=array_sha(q);base=dict(texture=q,indices=batches[it]['indices'],seeds=batches[it]['seeds'],draws=draws[it],gradient=True,iteration=it+1)
        out=parallel('batch',{k:dict(base,source_model=k) for k in SOURCES})
        raw={k:float(np.mean(out[k]['losses'])) for k in SOURCES};chosen,ratios=choose(raw,refs,it)
        prov={k:out[k]['gradient_provenance'] for k in SOURCES};gg={k:out[k]['gradient'] for k in SOURCES}
        for k in SOURCES:
            pr=prov[k];assert pr['server_model']==pr['source_model']==k and pr['iteration']==it+1
            assert pr['texture_sha256']==before and pr['indices']==base['indices'] and pr['seeds']==base['seeds'] and pr['draws_sha256']==digest(draws[it])
            assert np.isfinite(gg[k]).all() and np.linalg.norm(gg[k])>0
        step=np.sign(gg[chosen]).astype(np.int8);q=np.clip(q+.01*step.astype(np.float32),0,1).astype(np.float32)
        cosine=float(np.sum(gg['pi']*gg['imagewam'])/(np.linalg.norm(gg['pi'])*np.linalg.norm(gg['imagewam'])))
        hh.append(dict(iteration=it+1,raw=raw,ratios=ratios,chosen=chosen,cosine=cosine,provenance=prov,
            before_sha256=before,after_sha256=array_sha(q),sign_sha256=array_sha(step),updated=time.time()))
        signs.append(step);temp=state.with_suffix('.tmp.npz')
        np.savez_compressed(temp,texture=q,history=json.dumps(hh),update_signs=np.stack(signs));temp.replace(state)
        dump(dest/'progress.json',dict(iteration=it+1,total=500,chosen={k:sum(h['chosen']==k for h in hh) for k in SOURCES},last=hh[-1]))
        if it==0 or (it+1)%25==0:
            Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png');report(root)
        print('[pi/ImageWAM joint EOT]',it+1,raw,ratios,chosen,flush=True)
    final=root/'textures/joint_eot.png'
    rgb=np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)
    if not final.exists():Image.fromarray(rgb).save(final)
    assert np.array_equal(np.asarray(Image.open(final)),rgb)
    seal=dict(protocol_sha256=sha(root/'protocol.json'),steps=500,png_sha256=sha(final),state_sha256=sha(state),
        ledger_sha256={k:sha(root/'gradient_ledgers'/(k+'.jsonl')) for k in SOURCES},selection='fixed final500 before evaluation')
    fixed(root/'sealed_texture.json',seal)

def verify(root):
    p=read(root/'protocol.json');seal=read(root/'sealed_texture.json')
    assert seal['protocol_sha256']==sha(root/'protocol.json') and seal['state_sha256']==sha(root/'training/state.npz')
    assert seal['png_sha256']==sha(root/'textures/joint_eot.png')
    assert sha(root/'textures/random.png')==p['random_sha256'] and sha(root/'contexts.pt')==p['contexts_sha256']
    assert sha(root/'training_plan.json')==p['training']['batch_plan_sha256'] and sha(root/'eot_plan.json')==p['training']['eot_plan_sha256']
    for ref in p['training']['references'].values():assert sha(Path(ref['path']))==ref['sha256']
    for source,value in p['training']['source_verification_sha256'].items():assert sha(Path(source)/'completion_verification.json')==value
    assert read(root/'evaluation_start.json')['seal_sha256']==sha(root/'sealed_texture.json')
    assets=read(root/'audit/asset_hashes.json');assert assets==read(IMAGE_SOURCE/'audit/asset_hashes.json')
    assert assets['revision']==S.R.REVISION and assets['model_sha256']==S.R.EXPECTED_MODEL_SHA
    assert sha(S.R.WEIGHTS/'config.yaml')==assets['config_sha256'] and sha(S.R.WEIGHTS/'dataset_stats.json')==assets['stats_sha256']
    assert sha(S.R.AE)==assets['native_ae_sha256']
    with np.load(root/'training/state.npz') as z:hh=json.loads(str(z['history']));signs=z['update_signs'];final=z['texture']
    assert len(hh)==len(signs)==500
    ledgers={k:[json.loads(x) for x in (root/'gradient_ledgers'/(k+'.jsonl')).read_text().splitlines()] for k in SOURCES}
    for k in SOURCES:assert len(ledgers[k])==500 and sha(root/'gradient_ledgers'/(k+'.jsonl'))==seal['ledger_sha256'][k]
    q=np.full((1,3,128,128),.5,np.float32);batches=read(root/'training_plan.json');draws=read(root/'eot_plan.json');refs=read(root/'specialist_reference.json')
    for it,h in enumerate(hh):
        assert h['iteration']==it+1 and h['before_sha256']==array_sha(q)
        chosen,ratios=choose(h['raw'],refs,it);assert h['chosen']==chosen and h['ratios']==ratios
        for k in SOURCES:
            pr=ledgers[k][it];assert h['provenance'][k]==pr and pr['iteration']==it+1 and pr['source_model']==pr['server_model']==k
            assert pr['texture_sha256']==array_sha(q) and pr['indices']==batches[it]['indices'] and pr['seeds']==batches[it]['seeds'] and pr['draws_sha256']==digest(draws[it])
        step=signs[it];assert array_sha(step)==h['sign_sha256'] and np.isin(step,[-1,0,1]).all()
        q=np.clip(q+.01*step.astype(np.float32),0,1).astype(np.float32);assert array_sha(q)==h['after_sha256']
    assert np.array_equal(q,final) and np.array_equal(np.asarray(Image.open(root/'textures/joint_eot.png')),np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8))
    assert not (root/'gradient_ledgers/fast.jsonl').exists()
    assert read(root/'models/pi/native_audit.json')['passed'] and read(root/'audit/native_and_gradient.json')['passed'] and read(root/'models/fast/native_audit.json')['passed']
    fast_counts=read(root/'models/fast/request_counts.json');assert fast_counts['forbidden_gradient_requests']==0 and fast_counts['policy_frozen']
    for snap in (root/'source_snapshots').glob('*/manifest.json'):
        for name,value in read(snap).items():assert sha(snap.parent/name)==value
    results={};total=0
    for k in KINDS:
        cfg=read(root/'models'/k/'protocol.json');old_cfg=read(baseline(k)/'models'/k/'protocol.json')
        for field in ['steps','wait','replan','horizon','flow_steps']:assert cfg['evaluation'][field]==old_cfg['evaluation'][field]
        rows=M.outcomes(root,k);old=M.outcomes(baseline(k),k)
        assert rows['clean']==old['clean'] and rows['random']==old['random']
        eligible=[i for i,ok in rows['clean'].items() if ok];assert len(eligible)==(9 if k=='pi' else 10)
        results[k]=dict(clean_successes=len(eligible),conditions={})
        for c in CONDITIONS:
            assert len(rows[c])==10
            for i in range(40,50):
                f=root/'models'/k/'episodes'/c/f'init{i}.json';r=read(f);old_r=read(baseline(k)/'models'/k/'episodes/clean'/f'init{i}.json')
                assert 'error' not in r and r['condition']==c and r['init_index']==i and r['protocol_sha256']==sha(root/'models'/k/'protocol.json')
                assert r['initial_state_sha256']==old_r['initial_state_sha256']
                with np.load(f.with_suffix('.npz')) as z:
                    assert array_sha(z['initial_state'])==r['initial_state_sha256']
                    assert z['actions'].shape==(r['queries'],cfg['evaluation']['horizon'],7) and np.isfinite(z['actions']).all()
                    assert z['states'].shape==(r['queries'],8) and np.isfinite(z['states']).all()
                total+=1
            if c!='clean':
                fails=[i for i in eligible if not rows[c][i]]
                results[k]['conditions'][c]=dict(failures=len(fails),denominator=len(eligible),ASR=len(fails)/len(eligible),failed_indices=fails)
    assert total==90
    out=dict(passed=True,completed_episodes=total,training_steps=500,gradient_records={k:500 for k in SOURCES},
        target_gradient_requests=0,results=results,seal_verified=True,gray_start_updates_replayed=True,
        per_step_source_input_noise_draws_verified=True,native_audits_passed=True,baseline_match_per_init=True,verified_at=time.time())
    dump(root/'completion_verification.json',out);return out

def server(root,kind):
    if kind=='pi':E.PORTS=PORTS;E.server(root,kind)
    elif kind=='imagewam':S.PORT=PORTS[kind];S.server(root)
    else:T.PORTS=PORTS;T.server(root,kind)

def pipeline(root,gpus):
    root=root.absolute();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare(root);services={};envs={};records={};finished=False
    def execution():dump(root/'execution.json',dict(host=socket.gethostname(),pipeline_pid=os.getpid(),backends=records,gpus=gpus,updated=time.time()))
    def stop(k):
        proc=services[k]
        if proc.poll() is None:
            try:call(k,'shutdown')
            except Exception:proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        records[k]['exit_code']=proc.returncode;records[k]['exited_at']=time.time();execution()
    def start(k,gpu):
        own=[v.pid for name,v in services.items() if v.poll() is None and records[name]['gpu']==gpu]
        status(root,'start_'+k);uuid=resource(root,gpu,own,'start_'+k,20000)
        envs[k]=dict(os.environ,WAMATTACK_PROJECT_ROOT=str(ROOT),WAMATTACK_LIBERO_CONFIG_PATH=str(root/'libero_config'),
            CUDA_VISIBLE_DEVICES=uuid,MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        ready=root/'ready.json' if k=='imagewam' else root/'models'/k/'ready.json'
        if ready.exists():ready.unlink()
        with (root/'logs'/(k+'_backend.log')).open('a') as log:
            services[k]=subprocess.Popen([PY[k],'-u',str(Path(__file__).absolute()),'server','--root',str(root),'--kind',k],cwd=ROOT,env=envs[k],stdout=log,stderr=subprocess.STDOUT)
        records[k]=dict(pid=services[k].pid,gpu=gpu,uuid=uuid,started_at=time.time());execution()
        began=time.monotonic()
        while not ready.exists():
            if services[k].poll() is not None or time.monotonic()-began>600:raise RuntimeError(k+' backend initialization failed')
            time.sleep(2)
        assert read(ready)['pid']==services[k].pid
    def evaluate(k):
        gpu=records[k]['gpu'];call(k,'trim')
        for c in CONDITIONS:
            if c=='joint_eot':assert read(root/'audit'/('baseline_'+k+'.json'))['passed']
            for i in range(40,50):
                f=root/'models'/k/'episodes'/c/f'init{i}.json'
                if f.exists():assert 'success' in read(f);continue
                assert read(root/'sealed_texture.json')['png_sha256']==sha(root/'textures/joint_eot.png')
                status(root,f'evaluate_{k}_{c}_{i}');resource(root,gpu,[services[k].pid],f'render_{k}_{c}_{i}')
                env=dict(envs[k],CUDA_VISIBLE_DEVICES=str(gpu))
                with (root/'logs'/f'{k}_{c}_{i}.log').open('a') as log:
                    subprocess.run([M.SIM,'-u',str(ROOT/'robotwinprobe/fastwam_libero_trial.py'),'rollout','--root',str(root/'models'/k),'--condition',c,'--index',str(i)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if c in ['clean','random']:
                rows=M.outcomes(root,k);assert rows[c]==M.outcomes(baseline(k),k)[c],(k,c,'baseline changed')
                if c=='clean' and k=='imagewam':assert sum(rows[c].values())>=6
                if c=='random':dump(root/'audit'/('baseline_'+k+'.json'),dict(passed=True,clean=rows['clean'],random=rows['random']))
    try:
        if (root/'completion_verification.json').exists():verify(root);finished=True;return
        S.R.check_assets(root)
        start('pi',gpus[0]);start('imagewam',gpus[1]);status(root,'native_and_gradient_audits')
        parallel('audit',{k:{} for k in SOURCES})
        actual=read(root/'audit/native_and_gradient.json')['nominal_gray_loss'];expected=read(IMAGE_SOURCE/'audit/native_and_gradient.json')['nominal_gray_loss']
        assert abs(actual-expected)/expected<1e-5,('ImageWAM gray baseline mismatch',actual,expected)
        dump(root/'audit/imagewam_gray_match.json',dict(passed=True,actual=actual,reference=expected))
        if not (root/'sealed_texture.json').exists():
            for k in SOURCES:resource(root,records[k]['gpu'],[services[k].pid],'train_'+k)
            status(root,'train_pi_imagewam_joint_eot');train(root)
        else:
            seal=read(root/'sealed_texture.json');assert seal['state_sha256']==sha(root/'training/state.npz') and seal['png_sha256']==sha(root/'textures/joint_eot.png')
        parallel('trim',{k:{} for k in SOURCES})
        fixed(root/'evaluation_start.json',dict(seal_sha256=sha(root/'sealed_texture.json'),png_sha256=sha(root/'textures/joint_eot.png')))
        for k in SOURCES:evaluate(k);stop(k)
        # Held-out target is loaded only after training and source evaluations.
        start('fast',gpus[0]);status(root,'native_audit_fast');call('fast','audit');evaluate('fast');stop('fast')
        status(root,'verifying');verify(root);finished=True
    except Exception as err:
        dump(root/'audit'/('error_'+str(time.time_ns())+'.json'),dict(error=repr(err),traceback=traceback.format_exc()))
        status(root,'runtime_error',error=repr(err),traceback=traceback.format_exc());raise
    finally:
        for k in services:stop(k)
        if finished:
            v=read(root/'completion_verification.json');v.update(remote_backends_exited=all(x.poll() is not None for x in services.values()))
            fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
            with (root/'pipeline.lock').open('a') as probe:
                fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB);v['pipeline_lock_released']=True
            dump(root/'completion_verification.json',v);status(root,'complete')

def cpu_checks():
    refs=dict(pi=[2.],imagewam=[.01])
    assert choose(dict(pi=1.,imagewam=.01),refs,0)[0]=='pi'
    assert choose(dict(pi=3.,imagewam=.005),refs,0)[0]=='imagewam'
    assert choose(dict(pi=2.,imagewam=.01),refs,0)[0]=='pi'
    for k in SOURCES:
        try:T.validate_target_command('batch')
        except ValueError:pass
        else:raise AssertionError('target accepted gradient request')
        req=dict(gradient=True,source_model='fast')
        try:E.validate_source_request(req,k,k)
        except ValueError:pass
        else:raise AssertionError('cross-source request accepted')
    return dict(passed=True,normalization_and_tie_verified=True,fast_training_request_rejected=True,source_gradient_routing_verified=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','pipeline','server','report','verify','cpu-checks'])
    ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--kind',choices=KINDS);ap.add_argument('--gpus',type=int,nargs=2,default=[3,2]);a=ap.parse_args()
    if a.command=='prepare':prepare(a.root);dump(a.root/'audit/cpu_checks.json',cpu_checks());status(a.root,'prepared')
    elif a.command=='pipeline':pipeline(a.root,a.gpus)
    elif a.command=='server':server(a.root,a.kind)
    elif a.command=='verify':print(json.dumps(verify(a.root)))
    elif a.command=='cpu-checks':print(json.dumps(cpu_checks()))
    else:report(a.root)
