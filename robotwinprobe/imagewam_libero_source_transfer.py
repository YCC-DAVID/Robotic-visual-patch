"""Experiment #56: sealed ImageWAM-owned patch -> frozen pi and FastWAM.

Target servers are inference-only. No texture training or selection is allowed.
"""
import argparse,hashlib,json,os,shutil,socket,subprocess,time,traceback
from pathlib import Path
from multiprocessing.connection import Listener
import numpy as np
from crossarch_common import AUTH,dump,sha
import libero_joint_maxmin as M

ROOT=M.ROOT
DEFAULT=ROOT/'robotwinprobe/out/imagewam_libero_source_transfer_20261030'
SOURCE=ROOT/'robotwinprobe/out/imagewam_libero_self_eot_20261030'
REFERENCE=ROOT/'robotwinprobe/out/libero13_eot_transfer_20261030'
REPORT=ROOT/'result/56_imagewam_libero_source_transfer.md'
PORTS={'pi':18800,'fast':18801}
CONDITIONS=['clean','random','imagewam_eot']
TARGET_COMMANDS=frozenset(['shutdown','audit','infer','trim'])

def validate_target_command(command):
    if command not in TARGET_COMMANDS:
        raise ValueError('Inference-only target rejects training/gradient command: '+command)

def read(p):return json.loads(p.read_text())
def fixed(p,d):
    if p.exists():assert read(p)==d,('immutable artifact changed',str(p))
    else:dump(p,d)

def check_source(root):
    v=read(SOURCE/'completion_verification.json');seal=read(SOURCE/'sealed_texture.json')
    assert v['passed'] and v['completed_episodes']==30 and v['training_steps']==500
    assert v['conditions']['self_eot']['failures']==6 and v['clean_successes']==10
    assert sha(SOURCE/'protocol.json')==seal['protocol_sha256']
    assert sha(SOURCE/'textures/self_eot.png')==seal['png_sha256']
    assert sha(SOURCE/'training/state.npz')==seal['state_sha256']
    assert sha(SOURCE/'gradient_ledgers/imagewam.jsonl')==seal['ledger_sha256']
    if (root/'protocol.json').exists():
        p=read(root/'protocol.json')
        for c,info in p['textures'].items():
            assert sha(Path(info['source']))==info['sha256']==sha(root/'textures'/(c+'.png'))
        assert sha(SOURCE/'completion_verification.json')==p['source_verification_sha256']
    return v

def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    for n in ['audit','logs','textures','source_snapshots']:(root/n).mkdir(exist_ok=True)
    v=check_source(root);seal=read(SOURCE/'sealed_texture.json')
    paths={'random':SOURCE/'textures/random.png','imagewam_eot':SOURCE/'textures/self_eot.png'}
    p=dict(experiment='imagewam_source_frozen_texture_transfer',source_model='ImageWAM official LIBERO4B',
        source_experiment=str(SOURCE),source_seal_sha256=sha(SOURCE/'sealed_texture.json'),
        source_verification_sha256=sha(SOURCE/'completion_verification.json'),
        source_self_ASR=v['conditions']['self_eot'],task='put_the_bowl_on_the_plate',suite='libero_goal',
        patch=dict(size_m=.13,resolution=128,footprint='same fixed table site101 square'),
        training=dict(enabled=False,new_textures=0,target_gradients_forbidden=True,selection='same sealed #55 final500 PNG; never select by transfer results'),
        textures={k:dict(source=str(path),sha256=sha(path)) for k,path in paths.items()},
        evaluation=dict(targets=M.KINDS,indices=list(range(40,50)),conditions=CONDITIONS,expected_episodes=60,
            native='original per-target 10-flow protocols, image inputs from actual MuJoCo base/wrist',
            baselines='fresh clean and random; both exactly reproduce #46 per init before evaluating ImageWAM patch',
            ASR='paired target failure / its own clean-success initial states',stress=False),
        reference=str(REFERENCE),limits='same task10 initial states; source training uses common pi trajectory bank; transfer is zero target feedback for texture optimization, not unseen task/state-distribution transfer')
    fixed(root/'protocol.json',p)
    for c,path in paths.items():
        dest=root/'textures'/(c+'.png')
        if not dest.exists():shutil.copy2(path,dest)
        assert sha(dest)==p['textures'][c]['sha256']
    for kind in M.KINDS:
        model=root/'models'/kind;model.mkdir(parents=True,exist_ok=True)
        cfg=M.rebase(read(REFERENCE/'models'/kind/'protocol.json'))
        cfg.update(experiment=p['experiment'],control_port=PORTS[kind],parent_root=str(root),isolation='Frozen target inference only; no patch or policy optimization')
        cfg['training']=dict(enabled=False,inference_only=True,target_gradients_forbidden=True)
        cfg['evaluation'].update(conditions=CONDITIONS,expected_new_episodes=30,reference=p['evaluation']['baselines'],texture_paths={c:str(root/'textures'/(c+'.png')) for c in paths},texture_paths_by_init={})
        fixed(model/'protocol.json',cfg)
        if kind=='fast':
            for name in ['prompts.npz','checkpoints/libero_uncond_2cam224.pt','checkpoints/libero_uncond_2cam224_dataset_stats.json']:
                dst=model/name;dst.parent.mkdir(parents=True,exist_ok=True)
                if not dst.exists():os.link(REFERENCE/'models/fast'/name,dst)
                assert sha(dst)==sha(REFERENCE/'models/fast'/name)
    config=root/'libero_config';config.mkdir(exist_ok=True)
    if not (config/'config.yaml').exists():shutil.copy2(REFERENCE/'libero_config/config.yaml',config/'config.yaml')
    snap=root/'source_snapshots'/('prepare_'+str(os.getpid()));snap.mkdir(exist_ok=True)
    for name in ['imagewam_libero_source_transfer.py','libero_joint_maxmin.py','libero_pi_adapter.py','fastwam_adapter.py','fastwam_libero_trial.py']:
        shutil.copy2(ROOT/'robotwinprobe'/name,snap/name)
    fixed(snap/'manifest.json',{f.name:sha(f) for f in snap.glob('*.py')})
    return p

def report(root):
    st=read(root/'status.json') if (root/'status.json').exists() else {'stage':'prepared'}
    lines=['# ImageWAM训练纹理向π与FastWAM零样本迁移','',f"状态：`{st['stage']}`；节点：`{st.get('host','unknown')}`。",'',
        '第55号ImageWAM自身EOT纹理已固定最终500步并通过源自身6/10配对ASR验证。直接冻结同一PNG，迁移到π和FastWAM，目标模型不提供梯度或输出反馈来优化/选择纹理。',
        '不新增模型或纹理训练、不追加stress/网格。沿用同一LIBERO任务、13cm/site101、init40–49及各目标原生10步推理；fresh clean和Random须逐初态复现第46号。',
        '每目标clean/Random/ImageWAM纹理各10条，共60条真实MuJoCo闭环；分母为各目标自身clean成功集。','',
        '| 目标 | 条件 | 完成 | 配对ASR |','|---|---|---:|---:|']
    for kind in M.KINDS:
        rows=M.outcomes(root,kind);eligible={i for i,v in rows.get('clean',{}).items() if v}
        for c in CONDITIONS:
            d=rows.get(c,{});score='—'
            if c!='clean' and len(d)==len(rows.get('clean',{}))==10:score=f'{sum(not d[i] for i in eligible)}/{len(eligible)}'
            lines.append(f'| {kind} | {c} | {len(d)}/10 | {score} |')
    if st.get('error'):lines+=['',st['error']]
    lines+=['','训练使用共有的π轨迹状态库，因此这是跨架构、零目标反馈的纹理迁移，不声称未见任务或数据分布迁移。仅10初态，报告两个方向的实际成败，不按结果选纹理。',f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(lines)+'\n')

def server(root,kind):
    import torch
    torch.set_num_threads(4)
    if kind=='pi':from libero_pi_adapter import Backend
    else:from fastwam_adapter import Backend
    r=root/'models'/kind;b=Backend(r);b.model.eval().requires_grad_(False)
    with np.load(M.CALIBRATION) as z:
        data={n:z[n].copy() for n in ['state8','clean_base','clean_wrist','validation_mask']};prompt=str(z['prompt'])
    def audit():
        rows=[]
        with torch.no_grad():
            for i in np.flatnonzero(data['validation_mask'])[:2]:
                views=[torch.tensor(data[n][i].copy(),device='cuda').permute(2,0,1)[None].float()/255 for n in ['clean_base','clean_wrist']]
                prepared=b.prepare(data['state8'][i],prompt,views);_,a=b.response(prepared,20261201,full=True)
                if kind=='pi':native=b.native_full(prepared,20261201)
                else:
                    image,prop,(context,mask)=prepared
                    native=b.model.infer_action(prompt=None,input_image=image,proprio=prop,context=context,context_mask=mask,action_horizon=32,num_inference_steps=10,seed=20261201,rand_device='cpu')['action'][None].float().cpu()
                error=float((a.float().cpu()-native).abs().max());assert error==0,(kind,int(i),error)
                rows.append(dict(index=int(i),max_abs=error))
        assert all(not p.requires_grad and p.grad is None for p in b.model.parameters())
        out=dict(passed=True,rows=rows,inference_only=True,target_gradients_forbidden=True)
        dump(r/'native_audit.json',out);torch.cuda.empty_cache();return out
    counts={}
    with Listener(('127.0.0.1',PORTS[kind]),authkey=AUTH) as listener:
        dump(r/'ready.json',dict(host=socket.gethostname(),pid=os.getpid(),inference_only=True));running=True
        while running:
            with listener.accept() as c:
                try:
                    req=c.recv();cmd=req['command'];counts[cmd]=counts.get(cmd,0)+1
                    validate_target_command(cmd)
                    if cmd=='shutdown':running=False;out={'stopped':True}
                    elif cmd=='audit':out=audit()
                    elif cmd=='infer':
                        with torch.no_grad():out=b.infer(req)
                    elif cmd=='trim':torch.cuda.empty_cache();out={'trimmed':True}
                    c.send(out)
                except Exception as err:
                    traceback.print_exc();c.send(dict(error=repr(err),traceback=traceback.format_exc()))
        dump(r/'request_counts.json',dict(counts=counts,forbidden_gradient_requests=counts.get('batch',0),policy_frozen=all(not p.requires_grad and p.grad is None for p in b.model.parameters())))

def verify(root):
    check_source(root);p=read(root/'protocol.json');scores={};total=0
    for c,info in p['textures'].items():assert sha(root/'textures'/(c+'.png'))==info['sha256']
    for kind in M.KINDS:
        model=root/'models'/kind;cfg=read(model/'protocol.json');native=read(model/'native_audit.json')
        assert native['passed'] and native['inference_only'] and all(x['max_abs']==0 for x in native['rows'])
        assert cfg['training']['enabled'] is False
        rows=M.outcomes(root,kind);old=M.outcomes(M.SOURCE,kind)
        for c in CONDITIONS:assert len(rows[c])==10
        assert rows['clean']==old['clean'] and rows['random']==old['random']
        eligible=[i for i in rows['clean'] if rows['clean'][i]];assert len(eligible)==(9 if kind=='pi' else 10)
        for c in CONDITIONS:
            for i in range(40,50):
                path=model/'episodes'/c/f'init{i}.json';row=read(path);clean=read(model/'episodes/clean'/f'init{i}.json');historical=read(M.SOURCE/'models'/kind/'episodes/clean'/f'init{i}.json')
                assert 'error' not in row and row['init_index']==i and row['condition']==c and row['protocol_sha256']==sha(model/'protocol.json')
                assert row['initial_state_sha256']==clean['initial_state_sha256']==historical['initial_state_sha256']
                with np.load(path.with_suffix('.npz')) as z:
                    assert hashlib.sha256(z['initial_state'].tobytes()).hexdigest()==row['initial_state_sha256']
                    assert z['actions'].shape==(row['queries'],cfg['evaluation']['horizon'],7) and np.isfinite(z['actions']).all()
                    assert z['states'].shape==(row['queries'],8) and np.isfinite(z['states']).all()
                total+=1
        scores[kind]={}
        for c in ['random','imagewam_eot']:
            fails=[i for i in eligible if not rows[c][i]];scores[kind][c]=dict(failures=len(fails),denominator=len(eligible),ASR=len(fails)/len(eligible),failed_indices=fails)
    assert total==60
    out=dict(passed=True,completed_episodes=60,source_texture_sha256=p['textures']['imagewam_eot']['sha256'],source_self_ASR=p['source_self_ASR'],target_results=scores,new_training=False,targets_inference_only=True,clean_and_random_match_46_per_init=True,episode_protocol_initial_hashes_verified=True,sealed_source_verified=True,verified_at=time.time())
    dump(root/'completion_verification.json',out);return out

def configure():
    original_preflight=M.preflight;original_status=M.status
    M.DEFAULT=DEFAULT;M.REPORT=REPORT;M.PORT=PORTS;M.CONDITIONS=CONDITIONS
    M.prepare=prepare;M.report=report;M.train=check_source;M.heldout=check_source
    def preflight(root,gpu,own,tag,required=1024):
        skill=ROOT/'skills/gpu-preflight/SKILL.md';assert skill.read_text()
        if socket.gethostname().split('.')[0]=='nnmc63':assert gpu!=0
        for pid in own:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
            assert 'imagewam_libero_source_transfer.py server' in cmd and str(root) in cmd,('not own backend',pid)
        script=os.environ.get('WAMATTACK_GPU_PREFLIGHT','/shared/user65/.codex/skills/gpu-preflight/scripts/check_gpus.py')
        scan=subprocess.run([os.sys.executable,script,'--json'],capture_output=True,text=True)
        data=json.loads(scan.stdout);data['skill_sha256']=sha(skill);dump(root/'audit'/('cluster_preflight_'+tag+'.json'),data)
        uuid=original_preflight(root,gpu,own,tag,required)
        ecc=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total','--format=csv,noheader,nounits'],text=True).strip()
        assert [x.strip() for x in ecc.split(',')]==['0','0'],('ECC unknown/error',ecc)
        dump(root/'audit'/('ecc_'+tag+'.json'),dict(uuid=uuid,ecc=ecc,checked_at=time.time()));return uuid
    def status(root,stage,**kwargs):
        if stage=='train_maxmin':stage='verify_frozen_imagewam_source'
        elif stage=='heldout_loss':stage='verify_frozen_source_references'
        if stage=='evaluate_imagewam_eot_40':
            baseline={}
            for k in M.KINDS:
                new=M.outcomes(root,k);old=M.outcomes(M.SOURCE,k)
                assert new['clean']==old['clean'] and new['random']==old['random'],('baseline mismatch',k)
                baseline[k]=dict(passed=True,clean=new['clean'],random=new['random'])
            dump(root/'audit/baseline_match_46.json',baseline)
        if stage=='complete':verify(root)
        original_status(root,stage,**kwargs)
    M.preflight=preflight;M.status=status;M.__file__=str(Path(__file__).absolute())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','pipeline','server','report','verify']);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--kind',choices=['pi','fast']);p.add_argument('--gpus',nargs=2,type=int,default=[0,1]);a=p.parse_args();configure()
    if a.command=='prepare':prepare(a.root);M.status(a.root,'prepared')
    elif a.command=='pipeline':M.pipeline(a.root,a.gpus)
    elif a.command=='server':server(a.root,a.kind)
    elif a.command=='verify':print(json.dumps(verify(a.root)))
    else:report(a.root)
