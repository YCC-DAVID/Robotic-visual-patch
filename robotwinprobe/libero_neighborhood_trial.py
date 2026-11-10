"""Isolated, gradient-budget-matched texture-neighborhood pilot (#58).

Reuses audited native policy servers, never their old pipeline/train/report.
The inner minimization is one first-order step, not an exact worst-case solve.
No GPU is touched by prepare, report, verify-training, or resource-scan.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Client

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.environ['WAMATTACK_PROJECT_ROOT'] = str(ROOT)
import pi_imagewam_joint_eot as P
from crossarch_common import AUTH, dump, sha
from neighborhood_update import make_neighbor, ascent, describe_move
from texture_eot import IDENTITY

DEFAULT = ROOT/'robotwinprobe/out/libero_neighborhood_pilot_20261102'
SOURCE = ROOT/'robotwinprobe/out/pi_imagewam_joint_eot_20261031'
REPORT = ROOT/'result/58_libero_neighborhood_pilot.md'
PORTS = dict(pi=18832, imagewam=18833, fast=18834)
SOURCES = ('pi', 'imagewam')
KINDS = (*SOURCES, 'fast')
ARMS = ('vanilla', 'neighborhood')
CONDITIONS = ('clean', 'random', *ARMS)
SKILL = ROOT/'skills/gpu-preflight/SKILL.md'
PREFLIGHT = ROOT/'skills/gpu-preflight/scripts/check_gpus.py'


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def fixed(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Immutable protocol differs: {path}')
    else:
        dump(path, value)


def atomic_npz(path, **arrays):
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def prepare(root, steps=50, rho=2/255):
    if not 1 <= steps <= 500 or not 0 < rho <= 1:
        raise ValueError('steps must be 1..500 and rho in (0,1]')
    for name in ('audit', 'logs', 'textures', 'training', 'requests',
                 'gradient_ledgers', 'source_snapshots', 'libero_config'):
        (root/name).mkdir(parents=True, exist_ok=True)
    old = read(SOURCE/'protocol.json')
    assert read(SOURCE/'completion_verification.json')['passed']
    p = dict(experiment='libero_neighborhood_pilot', version=1,
        source=str(SOURCE), source_protocol_sha256=sha(SOURCE/'protocol.json'),
        task=old['task'], suite=old['suite'], patch=old['patch'],
        training=dict(sources=list(SOURCES), heldout='fast', policy_frozen=True,
            initialization=.5, batch=8, lr=.01, radius=rho, radius_units='RGB texture [0,1]',
            neighborhood_outer_steps=steps, vanilla_updates=2*steps,
            per_arm_gradient_calls_per_source=2*steps,
            per_arm_gradient_frames_per_source=16*steps,
            inner='one clip(Q-rho*sign(g_center),0,1) step per source',
            outer='clip(Q+.01*sign(g_neighbor_of_selected_source),0,1); from original center',
            selector='original weakest normalized CENTER loss; tie -> pi; fixed specialist references',
            budget='vanilla takes two sequential updates on each repeated batch/noise/EOT; robust takes center+neighbor gradients',
            caveat='matched gradient evaluations, not matched outer steps or identical optimization paths; one-step inner may fail to lower loss',
            objectives=old['training']['objective'], eot_ranges=old['training']['eot_ranges'],
            selection='fixed final pilot endpoint, sealed before closed-loop evaluation; no target feedback',
            plans={n:sha(SOURCE/n) for n in ['training_plan.json','eot_plan.json','specialist_reference.json']}),
        evaluation=dict(indices=list(range(40,50)), conditions=list(CONDITIONS),
            expected_episodes=120, native_protocols='unchanged per policy',
            primary='actual MuJoCo paired ASR, each target own fresh clean success set',
            reference='fresh clean/random must reproduce historical results per initial state',
            limits='single task/seed, previously used development initial states; pilot not confirmatory generalization'),
        model=old['imagewam_model'], random_sha256=sha(SOURCE/'textures/random.png'),
        contexts_sha256=sha(SOURCE/'contexts.pt'), asset_manifest_sha256=sha(SOURCE/'audit/asset_hashes.json'))
    fixed(root/'protocol.json', p)
    for name in ['training_plan.json','eot_plan.json','specialist_reference.json',
                 'contexts.pt','textures/random.png','audit/asset_hashes.json','libero_config/config.yaml']:
        dest = root/name
        if not dest.exists():
            shutil.copy2(SOURCE/name, dest)
        assert sha(dest) == sha(SOURCE/name)
    for kind in KINDS:
        model = root/'models'/kind
        model.mkdir(parents=True, exist_ok=True)
        cfg = P.M.rebase(read(SOURCE/'models'/kind/'protocol.json'))
        cfg.update(experiment=p['experiment'], control_port=PORTS[kind], parent_root=str(root))
        cfg['training'].update(enabled=kind in SOURCES, policy_frozen=True,
            target_gradients_forbidden=kind=='fast', isolated_source_model=kind if kind in SOURCES else None)
        cfg['evaluation'].update(conditions=list(CONDITIONS), expected_new_episodes=40,
            baseline_reuse=False, texture_paths={c:str(root/'textures'/(c+'.png')) for c in CONDITIONS if c!='clean'},
            texture_paths_by_init={})
        fixed(model/'protocol.json', cfg)
        if kind == 'fast':
            for name in ['prompts.npz','checkpoints/libero_uncond_2cam224.pt',
                         'checkpoints/libero_uncond_2cam224_dataset_stats.json']:
                dest = model/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                source = SOURCE/'models'/kind/name
                if not dest.exists():
                    dest.symlink_to(source)
                assert dest.resolve() == source.resolve()
    files = [Path(__file__), ROOT/'robotwinprobe/neighborhood_update.py',
             ROOT/'robotwinprobe/test_neighborhood_update.py']
    files += [ROOT/'robotwinprobe'/name for name in P.FILES]
    files += [ROOT/'pi05probe'/name for name in ['dual_view_proxy.py','render_world_proxy.py']]
    manifest = {}
    for path in files:
        dest = root/'source_snapshots'/path.name
        if dest.exists() and sha(dest) != sha(path):
            raise ValueError(f'Code changed after prepare; use a new experiment root: {path.name}')
        if not dest.exists():
            shutil.copy2(path, dest)
        manifest[path.name] = sha(dest)
    fixed(root/'source_snapshots/manifest.json', manifest)
    asset_paths = [root/'models'/k/'protocol.json' for k in KINDS]
    asset_paths += [root/'libero_config/config.yaml']
    proxy_root = ROOT/'pi05probe/out/dual_view'
    asset_paths += [proxy_root/name for name in [
        'calibration/base.npz','calibration/wrist.npz','fit/base.json','fit/wrist.json']]
    asset_paths += [ROOT/'pi05probe/config/scene.yaml', SKILL, PREFLIGHT]
    fixed(root/'prepared_assets.json', {
        (str(path.relative_to(root)) if path.is_relative_to(root) else str(path)):sha(path)
        for path in asset_paths})
    return p


def check_prepared(root):
    p = read(root/'protocol.json')
    assert p['source_protocol_sha256'] == sha(SOURCE/'protocol.json')
    for name, expected in p['training']['plans'].items():
        assert sha(root/name) == expected
    for name, expected in read(root/'source_snapshots/manifest.json').items():
        directory = 'pi05probe' if name in ['dual_view_proxy.py','render_world_proxy.py'] else 'robotwinprobe'
        assert sha(root/'source_snapshots'/name) == sha(ROOT/directory/name) == expected, name
    assert sha(root/'contexts.pt') == p['contexts_sha256']
    assert sha(root/'textures/random.png') == p['random_sha256']
    assert sha(root/'audit/asset_hashes.json') == p['asset_manifest_sha256']
    for name, expected in read(root/'prepared_assets.json').items():
        assert sha(root/name)==expected, ('prepared config changed',name)
    return p


def check_request_consistency(root):
    """Refuse orphaned RPCs before spending GPU time on a resumed trial.

    A server writes its ledger before replying. If the controller dies before
    persisting that reply, the gradient cannot be safely recomputed under the
    same id; keep the evidence and require explicit recovery instead.
    """
    ledgers={}
    for kind in SOURCES:
        path=root/'gradient_ledgers'/(kind+'.jsonl')
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
        if len({v['iteration'] for v in rows})!=len(rows):
            raise ValueError(f'Duplicate backend calls in {kind} ledger; preserve and reconcile')
        ledgers.update({(v['iteration'],kind):v for v in rows})
    replies={}
    for path in (root/'requests').glob('*.npz'):
        with np.load(path) as z:
            meta=json.loads(str(z['metadata']));prov=json.loads(str(z['provenance']))
            key=(meta['call_id'],meta['source'])
            if key in replies or array_sha(z['texture'])!=meta['texture_sha256']:
                raise ValueError('Duplicate or corrupted saved RPC response')
            replies[key]=prov
    if replies!=ledgers:
        raise ValueError('Backend ledger / saved responses differ; preserve orphaned RPCs and reconcile before resume')
    if (root/'gradient_ledgers/fast.jsonl').exists():
        raise ValueError('Held-out target has a training ledger')


def rpc(kind, command, **data):
    with Client(('127.0.0.1',PORTS[kind]), authkey=AUTH) as connection:
        connection.send(dict(command=command, **data))
        reply = connection.recv()
    if 'error' in reply:
        raise RuntimeError(kind+': '+reply['error']+'\n'+reply.get('traceback',''))
    return reply


def request(root, kind, q, pair, call_id, arm, step, phase):
    batches, draws = read(root/'training_plan.json'), read(root/'eot_plan.json')
    payload = dict(texture=q, indices=batches[pair]['indices'], seeds=batches[pair]['seeds'],
        draws=draws[pair], gradient=True, iteration=call_id, source_model=kind)
    meta = dict(call_id=call_id, source=kind, arm=arm, step=step, phase=phase,
        pair=pair, texture_sha256=array_sha(q), indices=payload['indices'], seeds=payload['seeds'],
        draws_sha256=digest(payload['draws']))
    path = root/'requests'/f'{call_id:06d}_{kind}.npz'
    if path.exists():
        with np.load(path) as z:
            assert json.loads(str(z['metadata'])) == meta and np.array_equal(q,z['texture'])
            return dict(losses=z['losses'].tolist(), gradient=z['gradient'].copy(),
                        gradient_provenance=json.loads(str(z['provenance'])))
    out = rpc(kind,'batch',**payload)
    provenance = out['gradient_provenance']
    assert provenance['iteration'] == call_id
    assert provenance['source_model'] == provenance['server_model'] == kind
    for key in ['texture_sha256','indices','seeds','draws_sha256']:
        assert provenance[key] == meta[key], key
    gradient = np.asarray(out['gradient'])
    assert gradient.shape == q.shape and np.isfinite(gradient).all() and np.linalg.norm(gradient)>0
    assert len(out['losses']) == 8 and np.isfinite(out['losses']).all()
    atomic_npz(path, texture=q, gradient=gradient, losses=out['losses'],
        metadata=json.dumps(meta), provenance=json.dumps(provenance))
    return out


def select(raw, refs, pair):
    ratios = {kind:raw[kind]/refs[kind][pair] for kind in SOURCES}
    assert all(np.isfinite(x) and x>=0 for x in ratios.values())
    return min(SOURCES,key=ratios.get), ratios


def train(root):
    p = check_prepared(root)
    check_request_consistency(root)
    cfg = p['training']; n = cfg['neighborhood_outer_steps']
    refs = read(root/'specialist_reference.json')
    for arm in ARMS:
        state = root/'training'/(arm+'.npz')
        q = np.full((1,3,128,128),.5,np.float32); history=[]; signs=[]
        if state.exists():
            with np.load(state) as z:
                q=z['texture'].copy(); history=json.loads(str(z['history'])); signs=list(z['signs'])
        total = 2*n if arm=='vanilla' else n
        for t in range(len(history),total):
            pair = t//2 if arm=='vanilla' else t
            base = 2*t if arm=='vanilla' else 4*n+4*t
            before = array_sha(q)
            central = {kind:request(root,kind,q,pair,base+j+1,arm,t+1,'center') for j,kind in enumerate(SOURCES)}
            raw = {kind:float(np.mean(value['losses'])) for kind,value in central.items()}
            chosen, ratios = select(raw,refs,pair)
            if arm=='neighborhood':
                neighbor = {kind:make_neighbor(q,central[kind]['gradient'],cfg['radius']) for kind in SOURCES}
                response = {kind:request(root,kind,neighbor[kind],pair,base+3+j,arm,t+1,'neighbor') for j,kind in enumerate(SOURCES)}
                weakened = {kind:float(np.mean(value['losses'])) for kind,value in response.items()}
                inner = {kind:dict(describe_move(q,neighbor[kind]),
                    loss_change=weakened[kind]-raw[kind], loss_lowered=weakened[kind]<=raw[kind]) for kind in SOURCES}
            else:
                response=central; weakened=None; inner=None
            gradient = response[chosen]['gradient']
            sign = np.sign(gradient).astype(np.int8)
            q = ascent(q,gradient,cfg['lr'])
            signs.append(sign)
            history.append(dict(step=t+1,pair=pair,raw=raw,ratios=ratios,chosen=chosen,
                neighbor_loss=weakened,inner=inner,before_sha256=before,after_sha256=array_sha(q),
                call_ids={k:[central[k]['gradient_provenance']['iteration']]+([] if arm=='vanilla' else [response[k]['gradient_provenance']['iteration']]) for k in SOURCES}))
            atomic_npz(state,texture=q,history=json.dumps(history),signs=np.stack(signs))
            dump(root/'training'/(arm+'_progress.json'),dict(step=t+1,total=total,last=history[-1]))
            print(arm,t+1,'/',total,'center',raw,'neighbor',weakened,'selected',chosen,flush=True)
            if t==0 or (t+1)%10==0:
                status(root,'training_'+arm)
        png=root/'textures'/(arm+'.png')
        rgb=np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)
        if not png.exists():
            Image.fromarray(rgb).save(png)
        assert np.array_equal(np.asarray(Image.open(png)),rgb)
    verify_training(root,seal=True)


def verify_training(root, seal=False):
    p=check_prepared(root);cfg=p['training'];n=cfg['neighborhood_outer_steps']
    records={}; expected_calls={k:set() for k in SOURCES}
    for path in sorted((root/'requests').glob('*.npz')):
        with np.load(path) as z:
            meta=json.loads(str(z['metadata']));prov=json.loads(str(z['provenance']))
            assert array_sha(z['texture']) == meta['texture_sha256'] == prov['texture_sha256']
            assert prov['iteration']==meta['call_id'] and prov['source_model']==prov['server_model']==meta['source']
            assert np.isfinite(z['gradient']).all() and np.linalg.norm(z['gradient'])>0
            for key in ['indices','seeds','draws_sha256']:
                assert prov[key]==meta[key]
            pair=meta['pair'];batch=read(root/'training_plan.json')[pair]
            assert meta['indices']==batch['indices'] and meta['seeds']==batch['seeds']
            assert meta['draws_sha256']==digest(read(root/'eot_plan.json')[pair])
            records[(meta['call_id'],meta['source'])]=(meta,prov,z['texture'].copy(),z['gradient'].copy(),float(z['losses'].mean()))
    summary={}
    refs=read(root/'specialist_reference.json')
    for arm in ARMS:
        with np.load(root/'training'/(arm+'.npz')) as z:
            end=z['texture'].copy();hist=json.loads(str(z['history']));signs=z['signs']
        total=2*n if arm=='vanilla' else n
        assert len(hist)==len(signs)==total
        q=np.full_like(end,.5)
        for t,h in enumerate(hist):
            pair=t//2 if arm=='vanilla' else t
            base=2*t if arm=='vanilla' else 4*n+4*t
            assert h['step']==t+1 and h['pair']==pair and h['before_sha256']==array_sha(q)
            actual_raw={}
            selected_records={}
            for j,kind in enumerate(SOURCES):
                ids=[base+j+1]+([] if arm=='vanilla' else [base+j+3])
                assert h['call_ids'][kind]==ids
                for phase,call_id in zip(['center','neighbor'],ids):
                    meta,prov,texture,gradient,loss=records[(call_id,kind)]
                    assert meta['arm']==arm and meta['step']==t+1 and meta['pair']==pair and meta['phase']==phase
                    expected_calls[kind].add(call_id)
                    if phase=='center':
                        assert np.array_equal(texture,q)
                        center_gradient=gradient;actual_raw[kind]=loss
                    else:
                        assert np.array_equal(texture,make_neighbor(q,center_gradient,cfg['radius']))
                        assert loss==h['neighbor_loss'][kind]
                    selected_records[kind]=gradient
                assert actual_raw[kind]==h['raw'][kind]
            chosen,ratios=select(actual_raw,refs,pair)
            assert h['chosen']==chosen and h['ratios']==ratios
            assert np.array_equal(signs[t],np.sign(selected_records[chosen]).astype(np.int8))
            q=ascent(q,selected_records[chosen],cfg['lr'])
            assert h['after_sha256']==array_sha(q)
        assert np.array_equal(q,end)
        assert np.array_equal(np.asarray(Image.open(root/'textures'/(arm+'.png'))),np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8))
        summary[arm]=dict(updates=total,gradient_calls_per_source=2*n)
    for kind in SOURCES:
        ledger=[json.loads(s) for s in (root/'gradient_ledgers'/(kind+'.jsonl')).read_text().splitlines() if s.strip()]
        assert len(ledger)==len(expected_calls[kind])==4*n
        assert {x['iteration'] for x in ledger}==expected_calls[kind]
        for item in ledger:
            assert records[(item['iteration'],kind)][1]==item
    assert len(records)==8*n
    result=dict(passed=True,arms=summary,actual_call_budget_matched=True,updates_and_inner_neighbors_replayed=True,
        heldout_training_requests=0,protocol_sha256=sha(root/'protocol.json'))
    dump(root/'audit/training_verification.json',result)
    if seal:
        fixed(root/'sealed_textures.json',dict(protocol_sha256=sha(root/'protocol.json'),
            states={a:sha(root/'training'/(a+'.npz')) for a in ARMS},
            pngs={a:sha(root/'textures'/(a+'.png')) for a in ARMS},
            ledgers={k:sha(root/'gradient_ledgers'/(k+'.jsonl')) for k in SOURCES},
            requests={path.name:sha(path) for path in sorted((root/'requests').glob('*.npz'))}))
    return result


def resource_scan(root, hosts=None, tag='cluster'):
    assert SKILL.read_text()
    command=[sys.executable,str(PREFLIGHT),'--json']
    if hosts:
        command+=['--hosts',*hosts]
    scan=subprocess.run(command,capture_output=True,text=True)
    data=json.loads(scan.stdout)
    if scan.returncode not in [0,2]:
        raise RuntimeError(scan.stderr)
    data['skill_sha256']=sha(SKILL)
    dump(root/'audit'/f'gpu_{tag}_{time.time_ns()}.json',data)
    return data


def resource(root,gpu,own,tag,required=1024):
    resource_scan(root,tag='cluster_'+tag)
    node=resource_scan(root,[socket.gethostname().split('.')[0]],tag)['nodes'][0]
    assert node['status']=='ok'
    card=next(c for c in node['gpus'] if c['index']==gpu)
    for pid in own:
        command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert Path(__file__).name+' server' in command and str(root) in command,('not own process',pid)
    reuse=bool(own) and set(card['pids']).issubset(set(own)) and card['free_mib']>=required
    assert card['eligible'] or reuse,('GPU occupied/unknown',card)
    assert not(node['host']=='nnmc63' and gpu==0),'historical ECC exclusion'
    ecc=subprocess.check_output(['nvidia-smi','--id='+card['uuid'],
        '--query-gpu=ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total',
        '--format=csv,noheader,nounits'],text=True).strip()
    assert [x.strip() for x in ecc.split(',')]==['0','0'],('bad/unknown ECC',ecc)
    dump(root/'audit'/f'allocation_{tag}.json',dict(card=card,own=own,reuse=reuse,ecc=ecc,checked_at=node.get('checked_at',time.time())))
    return card['uuid']


def status(root,stage,**extra):
    dump(root/'status.json',dict(stage=stage,host=socket.gethostname(),pid=os.getpid(),updated=time.time(),**extra))
    report(root)


def report(root):
    p=read(root/'protocol.json');st=read(root/'status.json') if (root/'status.json').exists() else {'stage':'prepared'}
    cfg=p['training'];lines=['# LIBERO：一阶纹理邻域稳健优化小试验','',
        f"状态：`{st['stage']}`；节点：`{st.get('host','unknown')}`。",'',
        f"π与ImageWAM为训练源，FastWAM完全留出。固定13cm/Q128、原成像代理及纹理EOT、首步flow目标。邻域半径{cfg['radius']:.8f}（2/255为默认），步长0.01。",'',
        f"稳健组{cfg['neighborhood_outer_steps']}次外层更新；普通组{cfg['vanilla_updates']}次更新。每组每源{cfg['per_arm_gradient_calls_per_source']}次batch梯度调用，每次8帧；匹配计算调用，不匹配更新次数。普通组每个batch/noise/EOT重复两次，第二次在更新后的纹理上求梯度。",'',
        '每源分别做一次内层下降，仍按原中心loss的固定参考达成率选择源；从原纹理用该源邻域梯度上升。内层仅一步，不宣称求得真正最坏点，实际loss变化逐步保存。', '',
        '| 组别 | 已完成更新 |','|---|---:|']
    for arm in ARMS:
        path=root/'training'/(arm+'_progress.json');v=read(path) if path.exists() else {'step':0,'total':cfg['vanilla_updates'] if arm=='vanilla' else cfg['neighborhood_outer_steps']}
        lines.append(f"| {arm} | {v['step']}/{v['total']} |")
    lines+=['','| 模型 | 条件 | 有效回合 | 配对ASR |','|---|---|---:|---:|']
    for kind in KINDS:
        rows=P.M.outcomes(root,kind);eligible={i for i,ok in rows.get('clean',{}).items() if ok}
        for c in CONDITIONS:
            values=rows.get(c,{});score='—'
            if c!='clean' and len(values)==len(rows.get('clean',{}))==10:
                score=f'{sum(not values[i] for i in eligible)}/{len(eligible)}'
            lines.append(f'| {kind} | {c} | {len(values)}/10 | {score} |')
    lines+=['','尚未完成GPU训练/评估时，CPU实现或解析函数测试不构成有效性证据。单任务、单训练种子、历史开发初态，仅为小试验。真实render指MuJoCo成像，不是真机。',
        f'','实验目录：`'+str(root)+'`。未创建资源轮询或定时任务。']
    if st.get('error'):
        lines+=['',str(st['error'])]
    REPORT.write_text('\n'.join(lines)+'\n')


def pipeline(root,gpus):
    p=check_prepared(root)
    check_request_consistency(root)
    if len(set(gpus))!=2:
        raise ValueError('two distinct idle GPUs required')
    lock=(root/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    services={};allocations={};envs={}
    def stop(kind):
        proc=services[kind]
        if proc.poll() is None:
            try:rpc(kind,'shutdown')
            except Exception:proc.terminate()
            try:proc.wait(timeout=30)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        allocations[kind]['exit_code']=proc.returncode
    def start(kind,gpu):
        own=[proc.pid for k,proc in services.items() if proc.poll() is None and allocations[k]['gpu']==gpu]
        uuid=resource(root,gpu,own,'start_'+kind,20000)
        env=dict(os.environ,WAMATTACK_PROJECT_ROOT=str(ROOT),
            WAMATTACK_LIBERO_CONFIG_PATH=str(root/'libero_config'),CUDA_VISIBLE_DEVICES=uuid,
            MUJOCO_EGL_DEVICE_ID=str(gpu),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='4',
            HF_HUB_OFFLINE='1',JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        envs[kind]=env
        ready=root/'ready.json' if kind=='imagewam' else root/'models'/kind/'ready.json'
        ready.unlink(missing_ok=True)
        log=(root/'logs'/(kind+'_backend.log')).open('a')
        try:
            services[kind]=subprocess.Popen([P.PY[kind],'-u',str(Path(__file__).resolve()),'server','--root',str(root),'--kind',kind],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        finally:log.close()
        allocations[kind]=dict(pid=services[kind].pid,gpu=gpu,uuid=uuid)
        dump(root/'execution.json',dict(host=socket.gethostname(),pid=os.getpid(),backends=allocations))
        began=time.monotonic()
        while not ready.exists():
            if services[kind].poll() is not None or time.monotonic()-began>600:
                raise RuntimeError(kind+' backend initialization failed')
            time.sleep(2)
        assert read(ready)['pid']==services[kind].pid
        rpc(kind,'audit')
    def evaluate(kind):
        rpc(kind,'trim')
        for condition in CONDITIONS:
            for i in p['evaluation']['indices']:
                path=root/'models'/kind/'episodes'/condition/f'init{i}.json'
                if path.exists():
                    assert 'success' in read(path) and read(path)['protocol_sha256']==sha(root/'models'/kind/'protocol.json')
                    continue
                if condition in ARMS:
                    assert sha(root/'textures'/(condition+'.png'))==read(root/'sealed_textures.json')['pngs'][condition]
                status(root,f'evaluate_{kind}_{condition}_{i}')
                resource(root,allocations[kind]['gpu'],[services[kind].pid],f'render_{kind}_{condition}_{i}')
                # This robosuite version requires the physical digit id to be a
                # literal substring of CUDA_VISIBLE_DEVICES. Only the renderer
                # uses that spelling, after checking the same physical card/UUID.
                # The policy backend retains its UUID mapping throughout.
                env=dict(envs[kind],CUDA_VISIBLE_DEVICES=str(allocations[kind]['gpu']))
                dump(root/'audit'/f'render_visibility_{kind}_{condition}_{i}.json',dict(
                    physical_index=allocations[kind]['gpu'],uuid=allocations[kind]['uuid'],
                    cuda_visible_devices=env['CUDA_VISIBLE_DEVICES'],egl_device=env['MUJOCO_EGL_DEVICE_ID'],
                    reason='robosuite EGL requires numeric physical index in CUDA_VISIBLE_DEVICES'))
                with (root/'logs'/f'{kind}_{condition}_{i}.log').open('a') as log:
                    subprocess.run([P.M.SIM,'-u',str(ROOT/'robotwinprobe/fastwam_libero_trial.py'),'rollout',
                        '--root',str(root/'models'/kind),'--condition',condition,'--index',str(i)],
                        cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if condition in ['clean','random']:
                rows=P.M.outcomes(root,kind)
                old=P.M.outcomes(P.baseline(kind),kind)
                assert rows[condition]==old[condition],(kind,condition,'baseline changed')
                if condition=='clean' and kind=='imagewam':
                    assert sum(rows[condition].values())>=6
    try:
        P.S.R.check_assets(root)
        start('pi',gpus[0]);start('imagewam',gpus[1])
        actual=read(root/'audit/native_and_gradient.json')['nominal_gray_loss']
        expected=read(P.IMAGE_SOURCE/'audit/native_and_gradient.json')['nominal_gray_loss']
        assert abs(actual-expected)/expected<1e-5
        for kind in SOURCES:
            resource(root,allocations[kind]['gpu'],[services[kind].pid],'train_'+kind)
        status(root,'training');train(root)
        fixed(root/'evaluation_start.json',dict(seal_sha256=sha(root/'sealed_textures.json')))
        for kind in SOURCES:
            evaluate(kind);stop(kind)
        start('fast',gpus[0]);evaluate('fast');stop('fast')
        verify(root);status(root,'complete')
    except Exception as error:
        status(root,'runtime_error',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        for kind in services:
            stop(kind)
        dump(root/'process_cleanup.json',dict(all_backends_exited=all(p.poll() is not None for p in services.values()),backends=allocations))
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def verify(root):
    verify_training(root)
    p=read(root/'protocol.json');seal=read(root/'sealed_textures.json')
    assert read(root/'evaluation_start.json')['seal_sha256']==sha(root/'sealed_textures.json')
    assert seal['protocol_sha256']==sha(root/'protocol.json')
    for group,directory,suffix in [('states','training','.npz'),('pngs','textures','.png'),('ledgers','gradient_ledgers','.jsonl')]:
        for key,value in seal[group].items():
            assert sha(root/directory/(key+suffix))==value
    for name,value in seal['requests'].items():
        assert sha(root/'requests'/name)==value
    scores={};total=0
    for kind in KINDS:
        rows=P.M.outcomes(root,kind);cfg=read(root/'models'/kind/'protocol.json')
        old=P.M.outcomes(P.baseline(kind),kind)
        assert rows['clean']==old['clean'] and rows['random']==old['random']
        assert all(len(rows[c])==10 for c in CONDITIONS)
        eligible=[i for i,ok in rows['clean'].items() if ok];scores[kind]={}
        assert read(root/'models'/kind/'native_audit.json')['passed'] if kind!='imagewam' else read(root/'audit/native_and_gradient.json')['passed']
        for c in CONDITIONS:
            for i in p['evaluation']['indices']:
                path=root/'models'/kind/'episodes'/c/f'init{i}.json';v=read(path)
                clean=read(root/'models'/kind/'episodes/clean'/f'init{i}.json')
                historical=read(P.baseline(kind)/'models'/kind/'episodes/clean'/f'init{i}.json')
                assert 'error' not in v and v['condition']==c and v['init_index']==i
                assert v['protocol_sha256']==sha(root/'models'/kind/'protocol.json')
                assert v['initial_state_sha256']==clean['initial_state_sha256']==historical['initial_state_sha256']
                with np.load(path.with_suffix('.npz')) as z:
                    assert array_sha(z['initial_state'])==v['initial_state_sha256']
                    assert z['actions'].shape==(v['queries'],cfg['evaluation']['horizon'],7) and np.isfinite(z['actions']).all()
                    assert z['states'].shape==(v['queries'],8) and np.isfinite(z['states']).all()
                total+=1
            if c!='clean':
                failures=[i for i in eligible if not rows[c][i]]
                scores[kind][c]=dict(failures=len(failures),denominator=len(eligible),failed_indices=failures)
    assert total==120
    assert not (root/'gradient_ledgers/fast.jsonl').exists()
    assert read(root/'models/fast/request_counts.json')['forbidden_gradient_requests']==0
    result=dict(passed=True,completed_episodes=total,results=scores,heldout_target_training_requests=0,
                gradient_budget_matched=True,verified_at=time.time())
    dump(root/'completion_verification.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','pipeline','server','resource-scan','report','verify-training','verify'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--rho',type=float,default=2/255)
    parser.add_argument('--kind',choices=KINDS)
    parser.add_argument('--gpus',nargs=2,type=int,required=False)
    args=parser.parse_args();root=args.root.resolve()
    if args.command=='prepare':
        prepare(root,args.steps,args.rho);status(root,'prepared')
    elif args.command=='server':
        if not args.kind:parser.error('--kind required')
        check_prepared(root);P.PORTS=PORTS;P.server(root,args.kind)
    elif args.command=='pipeline':
        if args.gpus is None:parser.error('--gpus required; no automatic GPU assignment')
        pipeline(root,args.gpus)
    elif args.command=='resource-scan':
        data=resource_scan(root)
        available=[(n['host'],g['index']) for n in data['nodes'] if n['status']=='ok'
            for g in n['gpus'] if g['eligible'] and not(n['host']=='nnmc63' and g['index']==0)]
        counts={host:sum(h==host for h,_ in available) for host,_ in available}
        status(root,'resource_checked' if any(v>=2 for v in counts.values()) else 'waiting_for_gpu',eligible=available)
        print(json.dumps(dict(checked_at=data['checked_at'],eligible=available)))
    elif args.command=='report':report(root)
    elif args.command=='verify-training':print(json.dumps(verify_training(root)))
    else:print(json.dumps(verify(root)))


if __name__=='__main__':
    main()
