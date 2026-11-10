"""Experiment #55: one ImageWAM-owned EOT patch and native self closed loops.

No transfer, no policy fitting, no grid, no timer. #53 artifacts stay immutable.
"""
import argparse,fcntl,hashlib,json,os,shutil,socket,subprocess,sys,time,traceback
from pathlib import Path
from multiprocessing.connection import Client,Listener
import numpy as np
from PIL import Image
from crossarch_common import AUTH,dump,sha
import imagewam_libero_transfer as R
from texture_eot import transform,IDENTITY,RANGES

ROOT=R.ROOT
PREVIOUS=R.DEFAULT
DEFAULT=ROOT/'robotwinprobe/out/imagewam_libero_self_eot_20261030'
REPORT=ROOT/'result/55_imagewam_libero_self_eot.md'
PORT=18790
CONDITIONS=['clean','random','self_eot']
SOURCE=R.SOURCE


def fixed(path,value):R.fixed(path,value)
def outcome(root,condition):
    return {i:json.loads(p.read_text())['success'] for i in range(40,50)
            if (p:=root/'models/imagewam/episodes'/condition/f'init{i}.json').exists() and 'success' in json.loads(p.read_text())}

def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    for name in ['audit','logs','textures','training','gradient_ledgers','source_snapshots']:(root/name).mkdir(exist_ok=True)
    prior=json.loads((PREVIOUS/'protocol.json').read_text())
    p=dict(experiment='imagewam_libero_source_only_self_eot',authorization='User explicitly requests ImageWAM train its own patch and test itself before any next step',
        task=prior['task'],suite=prior['suite'],model=prior['model'],patch=prior['patch'],
        training=dict(enabled=True,source_model='imagewam',policy_frozen=True,steps=500,batch=8,lr=.01,
            initialization=.5,update='sign-PGD clip[0,1]',objective='first native flow velocity 16x7 MSE from same-noise clean',
            source_states='155 saved LIBERO states from existing calibration bank; not an ImageWAM on-policy dataset',
            batch_noise_plan=str(SOURCE/'training_plan.json'),batch_noise_plan_sha256=sha(SOURCE/'training_plan.json'),
            eot_plan=str(R.JOINT/'eot_plan.json'),eot_plan_sha256=sha(R.JOINT/'eot_plan.json'),eot_ranges=RANGES,
            shared_views='one texture-plane draw per state before both camera projections',
            gradients='through native AE and FLUX2 video cache; quantization/native PIL resize use declared STE',
            memory='nonreentrant activation checkpoint per video block; exact native forward audit required',
            selection='fixed final500; no intermediate selection from closed-loop outcomes'),
        evaluation=dict(prior['evaluation'],conditions=CONDITIONS,expected_episodes=30,
            clean='fresh ImageWAM; exactly match #53 per init; minimum6/10; never another target denominator',
            ASR='this target clean success AND attacked failure / all this target clean successes',
            controls='same frozen random PNG as #53; no EOT evaluation stress'),
        random_png_sha256=sha(PREVIOUS/'textures/random.png'),
        scope='one ImageWAM-owned EOT texture; no transfer/no joint/no objective or seed grid/no new timer',
        limits='one task10 initial states, proxy-trained patch and native real-render nominal evaluation; failure of this attack is not robustness proof')
    fixed(root/'protocol.json',p)
    fixed(root/'training_plan.json',json.loads((SOURCE/'training_plan.json').read_text()))
    fixed(root/'eot_plan.json',json.loads((R.JOINT/'eot_plan.json').read_text()))
    for name in ['contexts.pt','textures/random.png','audit/asset_hashes.json']:
        dst=root/name
        if not dst.exists():shutil.copy2(PREVIOUS/name,dst)
        assert sha(dst)==sha(PREVIOUS/name)
    (root/'libero_config').mkdir(exist_ok=True)
    dst=root/'libero_config/config.yaml'
    if not dst.exists():shutil.copy2(PREVIOUS/'libero_config/config.yaml',dst)
    model=root/'models/imagewam';model.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((PREVIOUS/'models/imagewam/protocol.json').read_text())
    cfg.update(experiment=p['experiment'],control_port=PORT,parent_root=str(root),isolation='one own-source patch, frozen ImageWAM policy')
    cfg['training']=p['training']
    cfg['evaluation'].update(conditions=CONDITIONS,expected_new_episodes=30,reference=p['evaluation']['clean'],
        texture_paths={c:str(root/'textures'/(c+'.png')) for c in ['random','self_eot']},texture_paths_by_init={})
    fixed(model/'protocol.json',cfg)
    snapshot=root/'source_snapshots'/('prepare_'+str(os.getpid()));snapshot.mkdir(exist_ok=True)
    for name in ['imagewam_libero_self_attack.py','imagewam_libero_whitebox_adapter.py','imagewam_libero_adapter.py','imagewam_libero_transfer.py','texture_eot.py','fastwam_libero_trial.py']:
        shutil.copy2(ROOT/'robotwinprobe'/name,snapshot/name)
    fixed(snapshot/'manifest.json',{p.name:sha(p) for p in snapshot.glob('*.py')})
    return p


def report(root):
    st=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'prepared'}
    progress=json.loads((root/'training/progress.json').read_text()) if (root/'training/progress.json').exists() else {'iteration':0}
    clean=outcome(root,'clean');eligible={i for i in clean if clean[i]}
    text=['# ImageWAM LIBERO：自身EOT patch训练与自身闭环验证','',f"状态：`{st['stage']}`；实际训练checkpoint：{progress['iteration']}/500。",'',
        '用户要求先确认ImageWAM自己训练的patch能否干扰自身。只训练一张13cm/Q128纹理，官方LIBERO4B模型冻结；灰色起始、500×8、sign-PGD .01、原四项EOT及batch/noise抽样；首步16×7速度MSE。完整原生10步推理用于评估。',
        '训练使用原155个LIBERO标定状态（来自已有π轨迹，非ImageWAM自身采样数据），梯度只来自ImageWAM。纹理经两相机代理反传，评估在真实MuJoCo渲染中曝光，不向模型内部注入动作。',
        '固定最终500步PNG；fresh clean必须逐初态复现#53且至少6/10，Random为同一冻结对照。闭环是否失败为主要结果；损失增大不等于干扰成功。不追加迁移、网格或新的定时任务。','',
        '| 条件 | 完成 | 配对失败/自身clean成功 |','|---|---:|---:|']
    for c in CONDITIONS:
        rows=outcome(root,c);score='—'
        if c!='clean' and len(clean)==len(rows)==10:score=f'{sum(not rows[i] for i in eligible)}/{len(eligible)}'
        text.append(f'| {c} | {len(rows)}/10 | {score} |')
    if st.get('error'):text+=['',st['error']]
    text+=['','单任务10初态，nominal闭环；一次训练失败/零ASR不能证明对自适应攻击普遍鲁棒。',f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(text)+'\n')

def status(root,stage,**extra):
    dump(root/'status.json',dict(stage=stage,host=socket.gethostname(),pid=os.getpid(),updated=time.time(),**extra));report(root)

def resource(root,gpu,own,tag,required=1024):
    assert (ROOT/'skills/gpu-preflight/SKILL.md').read_text()
    for pid in own:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert 'imagewam_libero_self_attack.py server' in command and str(root) in command,('not own backend',pid,command)
    uuid=R.resource(root,gpu,own,tag,required)
    if socket.gethostname().split('.')[0]=='nnmc63':assert gpu!=0
    check=subprocess.run(['nvidia-smi','--id='+uuid,'--query-gpu=ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    values=[x.strip() for x in check.stdout.strip().split(',')]
    dump(root/'audit'/('ecc_'+tag+'.json'),dict(host=socket.gethostname(),uuid=uuid,values=values,checked_at=time.time()))
    assert len(values)==2 and all(x=='0' for x in values),('ECC unavailable/uncorrectable',values)
    return uuid

def call(command,**data):
    with Client(('127.0.0.1',PORT),authkey=AUTH) as c:
        c.send(dict(command=command,**data));reply=c.recv()
    if 'error' in reply:raise RuntimeError(reply['error']+'\n'+reply.get('traceback',''))
    return reply


def server(root):
    import torch
    from imagewam_libero_whitebox_adapter import Backend
    sys.path.insert(0,str(ROOT/'pi05probe'))
    from dual_view_proxy import CameraProxy
    torch.set_num_threads(4);b=Backend(root)
    with np.load(ROOT/'pi05probe/out/dual_view/calibration/base.npz') as z:
        data={k:z[k].copy() for k in ['state8','clean_base','clean_wrist','train_mask','validation_mask']}
    base=CameraProxy(ROOT/'pi05probe/out/dual_view/calibration/base.npz',device='cuda',fit=ROOT/'pi05probe/out/dual_view/fit/base.json')
    wrist=CameraProxy(ROOT/'pi05probe/out/dual_view/calibration/wrist.npz',device='cuda',fit=ROOT/'pi05probe/out/dual_view/fit/wrist.json')
    cache={}
    def views(k):return [torch.tensor(data[n][k].copy(),device='cuda').permute(2,0,1)[None].float()/255 for n in ['clean_base','clean_wrist']]
    def clean_response(k,seed):
        key=(int(k),int(seed))
        if key not in cache:
            with torch.no_grad():cache[key]=b.response(*b.prepare_views(data['state8'][k],views(k)),seed)[0].float().cpu()
        return cache[key].cuda()
    def batch(req):
        if req.get('gradient') and req.get('source_model')!='imagewam':raise ValueError('Only ImageWAM source gradients authorized')
        assert len(req['indices'])==len(req['seeds'])==len(req['draws'])
        losses=[];total=None
        for k,seed,draw in zip(req['indices'],req['seeds'],req['draws']):
            k=int(k);seed=int(seed);target=clean_response(k,seed)
            with torch.set_grad_enabled(req['gradient']):
                q=torch.tensor(req['texture'],device='cuda',dtype=torch.float32,requires_grad=req['gradient'])
                t=transform(q,draw);image,prop=b.prepare_views(data['state8'][k],[base(t,k,native=True),wrist(t,k,native=True)])
                first,_=b.response(image,prop,seed);loss=(first.float()-target).square().mean();assert torch.isfinite(loss)
                losses.append(float(loss.detach()))
                if req['gradient']:
                    g,=torch.autograd.grad(loss,q);assert torch.isfinite(g).all() and g.norm()>0
                    assert all(not p.requires_grad and p.grad is None for p in b.model.parameters())
                    a=g.cpu().numpy();total=a if total is None else total+a
        out={'losses':losses}
        if total is not None:
            out['gradient']=total/len(losses)
            record=dict(source_model='imagewam',server_model='imagewam',iteration=req['iteration'],
                texture_sha256=hashlib.sha256(np.asarray(req['texture']).tobytes()).hexdigest(),indices=req['indices'],seeds=req['seeds'],draws_sha256=hashlib.sha256(json.dumps(req['draws'],sort_keys=True).encode()).hexdigest(),gradient_norm=float(np.linalg.norm(out['gradient'])))
            with (root/'gradient_ledgers/imagewam.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n')
            out['gradient_provenance']=record
        return out
    def audit():
        rows=[]
        for k in np.flatnonzero(data['validation_mask'])[:2]:
            raw={n:data['clean_'+n][k].copy() for n in ['base','wrist']}
            with torch.no_grad():
                official,prop=b.prepare(data['state8'][k],raw)
                image,prop2=b.prepare_views(data['state8'][k],views(k))
                assert torch.equal(image,official) and torch.equal(prop,prop2),'native PIL/state input mismatch'
                native_first=[];original=b.model.action_expert.post_dit
                def capture(*args,**kwargs):
                    v=original(*args,**kwargs);native_first.append(v.detach().clone());return v
                b.model.action_expert.post_dit=capture
                try:expected=b.native(image,prop,20261201)
                finally:b.model.action_expert.post_dit=original
                repeated=b.native(image,prop,20261201)
            # Exercise the checkpointed differentiable forward, not just no_grad.
            active=image.detach().requires_grad_(True)
            first,full=b.response(active,prop,20261201,full=True)
            error=float((full[0].float().detach().cpu()-expected).abs().max())
            first_error=float((first.detach()-native_first[0]).abs().max())
            assert error==first_error==0 and torch.equal(expected,repeated),('strict native parity',int(k),error,first_error)
            del active,first,full
            rows.append(dict(index=int(k),native_full_max_abs=error,native_first_max_abs=first_error,repeated_equal=True))
            torch.cuda.empty_cache()
        batch0=json.loads((root/'training_plan.json').read_text())[0];k=batch0['indices'][0];seed=batch0['seeds'][0]
        q=torch.full((1,3,128,128),.5,device='cuda',requires_grad=True)
        target=clean_response(k,seed);draw=json.loads((root/'eot_plan.json').read_text())[0][0]
        t=transform(q,draw);image,prop=b.prepare_views(data['state8'][k],[base(t,k,native=True),wrist(t,k,native=True)])
        first,_,ref=b.response(image,prop,seed,return_intermediates=True);loss=(first.float()-target).square().mean()
        grads=torch.autograd.grad(loss,(q,image,ref))
        norms=[float(g.float().norm()) for g in grads]
        assert all(torch.isfinite(g).all() for g in grads) and all(x>0 for x in norms),('broken gradient route',norms)
        del q,t,image,prop,first,ref,loss,grads
        assert all(not p.requires_grad and p.grad is None for p in b.model.parameters())
        nominal=batch(dict(texture=np.full((1,3,128,128),.5,np.float32),indices=batch0['indices'],seeds=batch0['seeds'],draws=[IDENTITY]*8,gradient=False))
        gray=float(np.mean(nominal['losses']));assert np.isfinite(gray) and gray>0
        out=dict(passed=True,rows=rows,policy_frozen=True,own_source_gradients=True,
            gradient_route=['Q','EOT','both proxies','native PIL STE','native AE','native video-cache','native first velocity 16x7'],gradient_norms=dict(zip(['Q','image','ref_tokens'],norms)),nominal_gray_loss=gray,objective='same-noise native first-flow 16x7 MSE')
        dump(root/'audit/native_and_gradient.json',out);torch.cuda.empty_cache();return out
    with Listener(('127.0.0.1',PORT),authkey=AUTH) as listener:
        dump(root/'ready.json',dict(pid=os.getpid(),host=socket.gethostname()))
        running=True
        while running:
            with listener.accept() as c:
                try:
                    req=c.recv();cmd=req['command']
                    if cmd=='shutdown':running=False;out={'stopped':True}
                    elif cmd=='audit':out=audit()
                    elif cmd=='batch':out=batch(req)
                    elif cmd=='infer':out=b.infer(req)
                    elif cmd=='trim':torch.cuda.empty_cache();out={'trimmed':True}
                    else:raise ValueError(cmd)
                    c.send(out)
                except Exception as err:
                    traceback.print_exc();c.send(dict(error=repr(err),traceback=traceback.format_exc()))


def train(root):
    state=root/'training/state.npz';q=np.full((1,3,128,128),.5,np.float32);history=[]
    if state.exists():
        with np.load(state) as z:q=z['texture'].copy();history=json.loads(str(z['history']))
    batches=json.loads((root/'training_plan.json').read_text());draws=json.loads((root/'eot_plan.json').read_text())
    ledgerpath=root/'gradient_ledgers/imagewam.jsonl'
    if ledgerpath.exists():
        ledger=[json.loads(s) for s in ledgerpath.read_text().splitlines() if s.strip()]
        assert len(ledger)==len(history),'Interrupted ledger/checkpoint mismatch: preserve and reconcile before resume'
        assert all(h['gradient_provenance']==l for h,l in zip(history,ledger))
    for it in range(len(history),500):
        payload=dict(texture=q,indices=batches[it]['indices'],seeds=batches[it]['seeds'],draws=draws[it],gradient=True,source_model='imagewam',iteration=it+1)
        out=call('batch',**payload);pr=out['gradient_provenance']
        assert pr['source_model']==pr['server_model']=='imagewam' and pr['iteration']==it+1
        assert pr['texture_sha256']==hashlib.sha256(q.tobytes()).hexdigest()
        assert pr['indices']==payload['indices'] and pr['seeds']==payload['seeds']
        assert pr['draws_sha256']==hashlib.sha256(json.dumps(draws[it],sort_keys=True).encode()).hexdigest()
        g=out['gradient'];assert np.isfinite(g).all() and np.linalg.norm(g)>0
        q=np.clip(q+.01*np.sign(g),0,1).astype(np.float32)
        history.append(dict(iteration=it+1,source_model='imagewam',loss=float(np.mean(out['losses'])),gradient_provenance=pr,updated=time.time()))
        tmp=state.with_suffix('.tmp.npz');np.savez_compressed(tmp,texture=q,history=json.dumps(history));tmp.replace(state)
        dump(root/'training/progress.json',dict(iteration=it+1,total=500,complete=it==499,last=history[-1]))
        if it==0 or (it+1)%25==0:
            Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(root/'training'/f'texture_{it+1:04d}.png');report(root)
        print('[ImageWAM own-source EOT]',it+1,history[-1]['loss'],flush=True)
    png=root/'textures/self_eot.png'
    if not png.exists():shutil.copy2(root/'training/texture_0500.png',png)
    assert sha(png)==sha(root/'training/texture_0500.png')
    fixed(root/'sealed_texture.json',dict(protocol_sha256=sha(root/'protocol.json'),steps=500,png_sha256=sha(png),state_sha256=sha(state),ledger_sha256=sha(ledgerpath),selection='fixed final500 before evaluation'))


def verify(root):
    p=json.loads((root/'protocol.json').read_text());seal=json.loads((root/'sealed_texture.json').read_text())
    audit=json.loads((root/'audit/native_and_gradient.json').read_text());assert audit['passed']
    assert seal['protocol_sha256']==sha(root/'protocol.json') and seal['png_sha256']==sha(root/'textures/self_eot.png') and seal['state_sha256']==sha(root/'training/state.npz') and seal['ledger_sha256']==sha(root/'gradient_ledgers/imagewam.jsonl')
    with np.load(root/'training/state.npz') as z:history=json.loads(str(z['history']));q=z['texture']
    ledger=[json.loads(x) for x in (root/'gradient_ledgers/imagewam.jsonl').read_text().splitlines() if x.strip()]
    assert len(history)==len(ledger)==500
    batches=json.loads((root/'training_plan.json').read_text());draws=json.loads((root/'eot_plan.json').read_text())
    for it,(h,l) in enumerate(zip(history,ledger)):
        assert h['iteration']==l['iteration']==it+1 and h['gradient_provenance']==l and l['server_model']==l['source_model']=='imagewam'
        assert l['indices']==batches[it]['indices'] and l['seeds']==batches[it]['seeds'] and l['draws_sha256']==hashlib.sha256(json.dumps(draws[it],sort_keys=True).encode()).hexdigest()
    assert np.array_equal(np.asarray(Image.open(root/'textures/self_eot.png')),np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8))
    clean=outcome(root,'clean');assert len(clean)==10 and sum(clean.values())>=6 and clean==outcome(PREVIOUS,'clean')
    eligible=[i for i in clean if clean[i]];results={}
    for c in CONDITIONS:
        rows=outcome(root,c);assert len(rows)==10
        for i in range(40,50):
            f=root/'models/imagewam/episodes'/c/f'init{i}.json';row=json.loads(f.read_text());prior=json.loads((root/'models/imagewam/episodes/clean'/f'init{i}.json').read_text())
            assert row['protocol_sha256']==sha(root/'models/imagewam/protocol.json') and row['initial_state_sha256']==prior['initial_state_sha256']
            with np.load(f.with_suffix('.npz')) as z:
                assert hashlib.sha256(z['initial_state'].tobytes()).hexdigest()==row['initial_state_sha256']
                assert z['actions'].shape==(row['queries'],16,7) and np.isfinite(z['actions']).all()
                assert z['states'].shape==(row['queries'],8) and np.isfinite(z['states']).all()
        if c!='clean':
            fails=[i for i in eligible if not rows[i]];results[c]=dict(failures=len(fails),denominator=len(eligible),ASR=len(fails)/len(eligible),failed_indices=fails)
    result=dict(passed=True,completed_episodes=30,training_steps=500,source_only_gradient_records=500,native_and_gradient_audit=audit,clean_successes=len(eligible),clean_equal_53_per_init=True,conditions=results,sealed_manifest_verified=True,verified_at=time.time())
    dump(root/'completion_verification.json',result);return result


def pipeline(root,gpu):
    root=root.absolute();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'completion_verification.json').exists():assert verify(root)['passed'];return
    prepare(root);backend=None
    try:
        status(root,'verify_assets');R.check_assets(root)
        env=dict(os.environ,WAMATTACK_PROJECT_ROOT=str(ROOT),WAMATTACK_LIBERO_CONFIG_PATH=str(root/'libero_config'),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1')
        status(root,'start_imagewam_whitebox');uuid=resource(root,gpu,[],'start_backend',20000)
        env.update(CUDA_VISIBLE_DEVICES=uuid,MUJOCO_EGL_DEVICE_ID=str(gpu))
        ready=root/'ready.json'
        if ready.exists():ready.unlink()
        with (root/'logs/model.log').open('a') as log:
            backend=subprocess.Popen([R.MODEL_PY,'-u',str(Path(__file__).absolute()),'server','--root',str(root)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        dump(root/'execution.json',dict(host=socket.gethostname(),pipeline_pid=os.getpid(),backend_pid=backend.pid,gpu=gpu,uuid=uuid,started_at=time.time(),log=str(root/'logs/model.log')))
        began=time.monotonic()
        while not ready.exists():
            if backend.poll() is not None or time.monotonic()-began>600:raise RuntimeError('ImageWAM initialization failed; see model.log')
            time.sleep(2)
        status(root,'native_and_gradient_audit');call('audit')
        if not (root/'sealed_texture.json').exists():
            resource(root,gpu,[backend.pid],'train',1024);status(root,'train_own_source_eot');train(root)
        call('trim');status(root,'heldout_response')
        if not (root/'heldout.json').exists():
            with np.load(ROOT/'pi05probe/out/dual_view/calibration/base.npz') as z:idx=[int(x) for x in np.flatnonzero(z['validation_mask'])[:8]]
            scores={}
            for name in ['random','self_eot']:
                a=np.asarray(Image.open(root/'textures'/(name+'.png')).convert('RGB'),np.float32).transpose(2,0,1)[None]/255
                scores[name]=call('batch',texture=a,indices=idx,seeds=[20260901]*len(idx),draws=[IDENTITY]*len(idx),gradient=False)['losses']
            dump(root/'heldout.json',dict(indices=idx,losses=scores,diagnostic_only=True))
        call('trim')
        for c in CONDITIONS:
            for i in range(40,50):
                f=root/'models/imagewam/episodes'/c/f'init{i}.json'
                if f.exists():assert 'success' in json.loads(f.read_text());continue
                resource(root,gpu,[backend.pid],f'render_{c}_{i}')
                status(root,f'evaluate_{c}_{i}')
                simenv=dict(env,CUDA_VISIBLE_DEVICES=str(gpu))
                with (root/'logs'/f'{c}_{i}.log').open('a') as log:
                    subprocess.run([R.SIM,'-u',str(ROOT/'robotwinprobe/fastwam_libero_trial.py'),'rollout','--root',str(root/'models/imagewam'),'--condition',c,'--index',str(i)],env=simenv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if c=='clean':
                clean=outcome(root,'clean');assert len(clean)==10 and sum(clean.values())>=6 and clean==outcome(PREVIOUS,'clean'),('fresh clean gate',clean)
                dump(root/'audit/clean_gate.json',dict(passed=True,successes=sum(clean.values()),episodes=10,exactly_matches_53=True,outcomes=clean))
        verify(root);status(root,'complete')
    except Exception as err:
        stamp=str(time.time_ns());dump(root/'audit'/('error_'+stamp+'.json'),dict(error=repr(err),traceback=traceback.format_exc()))
        status(root,'runtime_error',error=repr(err),traceback=traceback.format_exc());raise
    finally:
        if backend and backend.poll() is None:
            try:call('shutdown')
            except Exception:backend.terminate()
            try:backend.wait(timeout=30)
            except subprocess.TimeoutExpired:backend.kill();backend.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','pipeline','server','report','verify']);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--gpu',type=int,default=0);a=p.parse_args()
    if a.command=='prepare':prepare(a.root);status(a.root,'prepared')
    elif a.command=='pipeline':pipeline(a.root,a.gpu)
    elif a.command=='server':server(a.root)
    elif a.command=='verify':print(json.dumps(verify(a.root)))
    else:report(a.root)
