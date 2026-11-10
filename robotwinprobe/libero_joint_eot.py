"""One controlled EOT addition to the completed #50 maxmin texture recipe.

Reuse the audited native backends, GPU gates, file lock and rollout pipeline.
Only texture-plane EOT changes the training distribution. Specialist reference
curves remain nominal historical constants, not EOT-trained upper bounds.
"""
import argparse
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
from multiprocessing.connection import Client, Listener

import numpy as np
from PIL import Image

import libero_joint_maxmin as M
from crossarch_common import AUTH, dump, sha
from texture_eot import IDENTITY, RANGES, plan, sample, transform

ROOT = M.ROOT
CONTROL = ROOT/'robotwinprobe/out/libero13_joint_maxmin_20261029'
DEFAULT = ROOT/'robotwinprobe/out/libero13_joint_eot_20261029'
REPORT = ROOT/'result/51_libero_joint_eot.md'
CONDITIONS = ['clean', 'control_nominal', 'eot_nominal', 'control_stress', 'eot_stress']
PORTS = dict(pi=18732, fast=18733)


def validate_source_request(req, kind, isolated_source):
    if isolated_source and req['gradient']:
        if isolated_source != kind or req.get('source_model') != kind:
            raise ValueError('Isolated source training forbids cross-model texture gradients')


def fixed_json(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, ('immutable protocol changed', str(path))
    else:
        dump(path, value)


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    for name in ['logs', 'audit', 'textures', 'training']:
        (root/name).mkdir(exist_ok=True)
    protocol = dict(experiment='libero13_joint_maxmin_eot', control=str(CONTROL),
        source=str(M.SOURCE), patch=dict(size_m=.13, resolution=128, footprint='fixed site101 square'),
        training=dict(steps=500, batch=8, lr=.01, initialization=.5,
            batch_noise_plan=str(M.SOURCE/'training_plan.json'),
            batch_noise_sha256=sha(M.SOURCE/'training_plan.json'),
            loss=dict(pi='first10x32 velocity SSE from same-noise clean',
                      fast='first32x7 velocity MSE from same-noise clean'),
            fusion='same #50: choose lowest current batch L0 / nominal historical specialist L0 at same step',
            reference_caveat='nominal historical curves kept fixed to isolate EOT; not EOT specialist bounds',
            eot=dict(ranges=RANGES, distribution='independent uniform', seed=2026102901,
                realizations='one per sampled state; identical for both models and both camera branches',
                location='transform one shared world texture BEFORE both CameraProxy projections',
                background='gray .5 inside fixed square; bilinear resample, zero padding after subtracting .5',
                scope='printed texture content and gain only; no patch boundary/pose/camera/light change')),
        evaluation=dict(indices=list(range(40,50)), conditions=CONDITIONS, expected_episodes=100,
            native='unchanged per-model 10 denoising steps and original native action protocol',
            clean='fresh per model; require exact per-init agreement with #46',
            nominal='fresh #50 fixed PNG and final EOT PNG, no texture change',
            stress='one new heldout transform per init shared by both textures/models, fixed before training',
            stress_seed=2026102902, physical='upload pretransformed PNG into real MuJoCo, fixed 13cm geometry',
            ASR='clean success AND attacked failure / all clean successes; report complete denominators only'),
        control_png_sha256=sha(CONTROL/'textures/maxmin.png'),
        references_sha256=sha(CONTROL/'reference.json'),
        limits='single task, 10 inits; proxy EOT training, real-render closed loop; no physical placement claim')
    fixed_json(root/'protocol.json', protocol)
    fixed_json(root/'eot_plan.json', plan())
    rng = np.random.default_rng(2026102902)
    fixed_json(root/'stress_draws.json', {str(i):sample(rng) for i in range(40,50)})
    fixed_json(root/'reference.json', json.loads((CONTROL/'reference.json').read_text()))
    for kind in M.KINDS:
        r = root/'models'/kind
        r.mkdir(parents=True, exist_ok=True)
        cfg = M.rebase(json.loads((CONTROL/'models'/kind/'protocol.json').read_text()))
        cfg.update(experiment=protocol['experiment'], control_port=PORTS[kind], parent_root=str(root),
            isolation='same #50 maxmin recipe, only add declared texture-plane EOT; no model training')
        cfg['training'].update(objective=protocol['training']['loss'][kind],
            eot=protocol['training']['eot'], seed=2026102901)
        cfg['evaluation'].update(conditions=CONDITIONS, baseline_reuse=False, expected_new_episodes=50,
            reference='fresh native clean and fresh #50 nominal control; no cross-model baseline reuse',
            texture_paths={c:str(root/'textures'/(c+'.png')) for c in ['control_nominal','eot_nominal']},
            texture_paths_by_init={c:{str(i):str(root/'textures'/c/f'init{i}.png') for i in range(40,50)}
                                   for c in ['control_stress','eot_stress']})
        fixed_json(r/'protocol.json', cfg)
        if kind == 'fast':
            for name in ['prompts.npz', 'checkpoints/libero_uncond_2cam224.pt',
                         'checkpoints/libero_uncond_2cam224_dataset_stats.json']:
                dest=r/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    os.link(CONTROL/'models/fast'/name, dest)
    config=root/'libero_config'
    config.mkdir(exist_ok=True)
    (config/'config.yaml').write_text((CONTROL/'libero_config/config.yaml').read_text())
    control=root/'textures/control_nominal.png'
    if not control.exists():
        shutil.copy2(CONTROL/'textures/maxmin.png', control)
    assert sha(control)==protocol['control_png_sha256']
    snapshots=root/'source_snapshots'/('prepare_'+str(os.getpid()))
    snapshots.mkdir(parents=True, exist_ok=True)
    for name in ['libero_joint_eot.py','texture_eot.py','libero_joint_maxmin.py',
                 'libero_pi_adapter.py','fastwam_adapter.py','fastwam_libero_trial.py']:
        shutil.copy2(ROOT/'robotwinprobe'/name, snapshots/name)
    for name in ['dual_view_proxy.py','render_world_proxy.py']:
        shutil.copy2(ROOT/'pi05probe'/name, snapshots/name)
    dump(snapshots/'manifest.json', {p.name:sha(p) for p in snapshots.glob('*.py')})
    return protocol


def report(root):
    s=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'prepared'}
    lines=['# LIBERO 13cm：最大最小联训加入纹理平面EOT', '',
        f'状态：`{s["stage"]}`；节点：`{s.get("host","unknown")}`。', '',
        '固定第50号融合规则、灰色初始化、500×8、步长0.01、状态/batch/动作noise、首步L0及冻结策略。只加入EOT。',
        '每帧独立均匀采样：旋转±15°、平移各轴±5%纹理宽高、缩放0.9–1.1、统一RGB增益0.9–1.1。',
        '同一次变换供两个模型及主摄/腕部共享，在成像投影前执行；13cm贴纸外轮廓、位置、相机与场景光照不变。',
        '达成率分母仍为第50号固定的历史nominal专用训练曲线，不称作EOT专用上限；本轮不另训专用纹理。',
        '训练4000帧/模型，无额外EOT重复前向；一帧一变换。评估预先固定10个新扰动、逐初态共享，新旧纹理均经真实MuJoCo渲染。', '']
    progress=root/'training/progress.json'
    if progress.exists():
        p=json.loads(progress.read_text())
        lines += [f'训练：{p["iteration"]}/500；步数分配：{p["chosen"]}。', '']
    lines += ['| 模型 | 条件 | 已完成 | 配对失败/clean成功 |', '|---|---|---:|---:|']
    for kind in M.KINDS:
        rows=M.outcomes(root,kind)
        clean={i for i,ok in rows.get('clean',{}).items() if ok}
        for condition in CONDITIONS:
            data=rows.get(condition,{})
            pairs=clean & data.keys()
            score='—' if condition=='clean' or not pairs else f'{sum(not data[i] for i in pairs)}/{len(pairs)}'
            lines.append(f'| {kind} | {condition} | {len(data)}/10 | {score} |')
    if s.get('error'):
        lines += ['', s['error']]
    lines += ['', '未完成条件的配对分母为已完成交集，不可当作最终ASR。完整结果以每模型clean成功集为分母。',
              '单任务10初态，扰动同时改变多种纹理统计；不分离各个EOT因子效果，不声称摆放/尺寸/照明迁移。',
              f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(lines)+'\n')


def server(root,kind):
    import torch
    torch.set_num_threads(4)
    torch.cuda.init()
    if kind=='pi':
        from libero_pi_adapter import Backend
    else:
        from fastwam_adapter import Backend
    sys.path.insert(0,str(ROOT/'pi05probe'))
    from dual_view_proxy import CameraProxy
    r=root/'models'/kind
    isolated_source=json.loads((r/'protocol.json').read_text())['training'].get('isolated_source_model')
    b=Backend(r)
    with np.load(M.CALIBRATION) as z:
        d={name:z[name].copy() for name in ['state8','clean_base','clean_wrist','train_mask','validation_mask']}
        prompt=str(z['prompt'])
    base=CameraProxy(M.CALIBRATION,device='cuda',fit=M.E26/'fit/base.json')
    wrist=CameraProxy(M.E26/'calibration/wrist.npz',device='cuda',fit=M.E26/'fit/wrist.json')
    cache={}
    def clean_views(k):
        return [torch.tensor(d[name][k].copy(),device='cuda').permute(2,0,1)[None].float()/255
                for name in ['clean_base','clean_wrist']]
    def response(k,seed,images):
        value,_=b.response(b.prepare(d['state8'][k],prompt,images),seed,full=False)
        return value.float() if kind=='pi' else value[...,:7].float()
    def batch(req):
        validate_source_request(req,kind,isolated_source)
        losses=[]
        total=None
        draws=req.get('draws',[IDENTITY]*len(req['indices']))
        assert len(draws)==len(req['indices'])==len(req['seeds'])
        for k,seed,draw in zip(req['indices'],req['seeds'],draws):
            k,seed=int(k),int(seed)
            if (k,seed) not in cache:
                with torch.no_grad():
                    cache[(k,seed)]=response(k,seed,clean_views(k)).cpu()
            with torch.set_grad_enabled(req['gradient']):
                q=torch.tensor(req['texture'],device='cuda',requires_grad=req['gradient'])
                transformed=transform(q,draw)
                delta=response(k,seed,[base(transformed,k),wrist(transformed,k)])-cache[(k,seed)].cuda()
                loss=delta.square().sum() if kind=='pi' else delta.square().mean()
                assert torch.isfinite(loss)
                losses.append(float(loss.detach()))
                if req['gradient']:
                    g,=torch.autograd.grad(loss,q)
                    assert torch.isfinite(g).all() and float(g.norm())>0
                    assert all(p.grad is None for p in b.model.parameters())
                    array=g.cpu().numpy()
                    total=array if total is None else total+array
        result=dict(losses=losses)
        if total is not None:
            result['gradient']=total/len(losses)
            if isolated_source:
                provenance=dict(source_model=kind,server_model=kind,iteration=req['iteration'],
                    texture_sha256=hashlib.sha256(np.asarray(req['texture']).tobytes()).hexdigest(),
                    indices=req['indices'],seeds=req['seeds'],
                    draws_sha256=hashlib.sha256(json.dumps(draws,sort_keys=True).encode()).hexdigest(),
                    gradient_norm=float(np.linalg.norm(result['gradient'])))
                ledger=root/'gradient_ledgers'/(kind+'.jsonl')
                ledger.parent.mkdir(parents=True,exist_ok=True)
                with ledger.open('a') as handle:
                    handle.write(json.dumps(provenance,sort_keys=True)+'\n')
                result['gradient_provenance']=provenance
        return result
    def audit():
        rows=[]
        with torch.no_grad():
            for k in np.flatnonzero(d['validation_mask'])[:2]:
                prepared=b.prepare(d['state8'][k],prompt,clean_views(k))
                _,a=b.response(prepared,20261201,full=True)
                if kind=='pi':
                    native=b.native_full(prepared,20261201)
                else:
                    image,proprio,(context,mask)=prepared
                    native=b.model.infer_action(prompt=None,input_image=image,proprio=proprio,context=context,context_mask=mask,
                        action_horizon=32,num_inference_steps=10,seed=20261201,rand_device='cpu')['action'][None].float().cpu()
                error=float((a.float().cpu()-native).abs().max())
                assert error==0,('native parity',kind,error)
                rows.append(dict(index=int(k),max_abs=error))
        first=json.loads((M.SOURCE/'training_plan.json').read_text())[0]
        nominal=batch(dict(texture=np.full((1,3,128,128),.5,np.float32),
            indices=first['indices'],seeds=first['seeds'],gradient=False))
        actual=float(np.mean(nominal['losses']))
        expected=json.loads((root/'reference.json').read_text())[kind][0]
        relative=abs(actual-expected)/expected
        assert relative<1e-5,('identity EOT loss differs from #50',kind,actual,expected)
        dump(r/'native_audit.json',dict(passed=True,rows=rows,
            nominal_gray_loss=dict(actual=actual,reference=expected,relative_error=relative)))
        return dict(passed=True)
    with Listener(('127.0.0.1',PORTS[kind]),authkey=AUTH) as listener:
        dump(r/'ready.json',dict(pid=os.getpid(),host=socket.gethostname(),ready=True))
        running=True
        while running:
            with listener.accept() as connection:
                try:
                    req=connection.recv()
                    cmd=req['command']
                    if cmd=='shutdown':
                        running=False; result=dict(stopped=True)
                    elif cmd=='audit':result=audit()
                    elif cmd=='batch':result=batch(req)
                    elif cmd=='validation_indices':result=dict(indices=[int(i) for i in np.flatnonzero(d['validation_mask'])])
                    elif cmd=='infer':result=b.infer(req)
                    elif cmd=='trim':torch.cuda.empty_cache(); result=dict(trimmed=True)
                    else:raise ValueError(cmd)
                    connection.send(result)
                except Exception as error:
                    traceback.print_exc()
                    connection.send(dict(error=repr(error),traceback=traceback.format_exc()))


def train(root):
    batches=json.loads((M.SOURCE/'training_plan.json').read_text())
    draws=json.loads((root/'eot_plan.json').read_text())
    references=json.loads((root/'reference.json').read_text())
    q=np.full((1,3,128,128),.5,np.float32)
    history=[]
    dest=root/'training'
    checkpoint=dest/'state.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as z:
            q=z['texture'].copy(); history=json.loads(str(z['history']))
    for it in range(len(history),500):
        batch=batches[it]
        result=M.parallel_rpc('batch',texture=q,indices=batch['indices'],seeds=batch['seeds'],
                              draws=draws[it],gradient=True)
        raw={k:float(np.mean(result[k]['losses'])) for k in M.KINDS}
        ratio={k:raw[k]/references[k][it] for k in M.KINDS}
        chosen='pi' if ratio['pi']<=ratio['fast'] else 'fast'
        gradients={k:result[k]['gradient'] for k in M.KINDS}
        cosine=float(np.sum(gradients['pi']*gradients['fast'])/
            (np.linalg.norm(gradients['pi'])*np.linalg.norm(gradients['fast'])))
        q=np.clip(q+.01*np.sign(gradients[chosen]),0,1).astype(np.float32)
        history.append(dict(iteration=it+1,raw=raw,ratio=ratio,chosen=chosen,cosine=cosine,
                            draws=draws[it],updated=time.time()))
        temp=checkpoint.with_suffix('.tmp.npz')
        np.savez_compressed(temp,texture=q,history=json.dumps(history))
        temp.replace(checkpoint)
        dump(dest/'progress.json',dict(iteration=it+1,total=500,complete=it==499,
            chosen={k:sum(h['chosen']==k for h in history) for k in M.KINDS},last=history[-1]))
        if it==0 or (it+1)%25==0:
            Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png')
            report(root)
        print('[maxmin EOT]',it+1,raw,ratio,chosen,flush=True)
    final=root/'textures/eot_nominal.png'
    if not final.exists():
        shutil.copy2(dest/'texture_0500.png',final)


def heldout(root):
    # Generate real-render test PNGs using exactly the differentiable transform,
    # with new predeclared draws; never select transforms/iterations by outcomes.
    import torch
    draws=json.loads((root/'stress_draws.json').read_text())
    for label in ['control','eot']:
        source=root/'textures'/(label+'_nominal.png')
        q=torch.from_numpy(M.load_png(source))
        for index,draw in draws.items():
            path=root/'textures'/(label+'_stress')/f'init{index}.png'
            path.parent.mkdir(parents=True,exist_ok=True)
            value=transform(q,draw).numpy()[0].transpose(1,2,0)
            rgb=np.rint(value*255).clip(0,255).astype(np.uint8)
            if path.exists():
                assert np.array_equal(np.asarray(Image.open(path)),rgb)
            else:Image.fromarray(rgb).save(path)
    fixed_json(root/'texture_controls.json',dict(draws=draws,
        hashes={str(p.relative_to(root)):sha(p) for p in sorted((root/'textures').rglob('*.png'))}))
    if (root/'heldout.json').exists():return
    indices=M.parallel_rpc('validation_indices')['pi']['indices'][:8]
    data=dict(indices=indices,noise_seed=20260901,scope='proxy loss only, never ASR')
    for label in ['control','eot']:
        q=M.load_png(root/'textures'/(label+'_nominal.png'))
        data[label]={}
        for condition,transforms in [('nominal',[IDENTITY]*8),
                                     ('stress',[draws[str(i)] for i in range(40,48)])]:
            result=M.parallel_rpc('batch',texture=q,indices=indices,seeds=[20260901]*8,
                                  draws=transforms,gradient=False)
            data[label][condition]={k:result[k]['losses'] for k in M.KINDS}
    dump(root/'heldout.json',data)


def configure():
    M.DEFAULT=DEFAULT; M.REPORT=REPORT; M.PORT=PORTS; M.CONDITIONS=CONDITIONS
    M.prepare=prepare; M.report=report; M.server=server; M.train=train; M.heldout=heldout
    original_preflight=M.preflight
    def checked_preflight(root,gpu,own,tag,required=1024):
        skill=ROOT/'skills/gpu-preflight/SKILL.md'
        skill_text=skill.read_text()
        assert skill_text and not (socket.gethostname()=='nnmc63' and gpu==0)
        script=os.environ.get('WAMATTACK_GPU_PREFLIGHT',
            '/shared/user65/.codex/skills/gpu-preflight/scripts/check_gpus.py')
        scan=subprocess.run([sys.executable,script,'--json'],capture_output=True,text=True)
        info=json.loads(scan.stdout)
        info['skill_sha256']=sha(skill)
        dump(root/'audit'/('cluster_preflight_'+tag+'.json'),info)
        # Reconfirm this host immediately after the full-cluster snapshot.
        return original_preflight(root,gpu,own,tag,required)
    M.preflight=checked_preflight
    original_status=M.status
    def guarded_status(root,stage,**kw):
        if stage=='evaluate_eot_nominal_40':
            comparisons={}
            for kind in M.KINDS:
                actual=M.outcomes(root,kind).get('control_nominal',{})
                reference=M.outcomes(CONTROL,kind)['maxmin']
                comparisons[kind]=dict(actual=actual,reference=reference,passed=actual==reference)
            dump(root/'audit/control_baseline.json',comparisons)
            assert all(v['passed'] for v in comparisons.values()),'fresh #50 nominal control changed'
        original_status(root,stage,**kw)
    M.status=guarded_status
    # Existing pipeline launches its own __file__ for services. Route those
    # subprocesses to this experiment's CLI, never to the historical server.
    M.__file__=str(Path(__file__).absolute())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','pipeline','server','report'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--kind',choices=M.KINDS)
    parser.add_argument('--gpus',type=int,nargs=2,default=[0,1])
    args=parser.parse_args()
    configure()
    if args.command=='prepare':prepare(args.root); M.status(args.root,'prepared')
    elif args.command=='pipeline':M.pipeline(args.root,args.gpus)
    elif args.command=='server':server(args.root,args.kind)
    else:report(args.root)
