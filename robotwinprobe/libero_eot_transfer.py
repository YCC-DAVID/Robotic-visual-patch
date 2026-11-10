"""Independent pi/FastWAM source-only EOT textures and frozen bidirectional transfer.

Each training request sends only Q_source to its own backend. Servers reject
cross-model gradient requests and append provenance ledgers. Both independent
final PNGs are sealed before any target-model transfer evaluation is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from multiprocessing.connection import Client

import numpy as np
from PIL import Image

import libero_joint_eot as E
from crossarch_common import AUTH, dump, sha

M=E.M
ROOT=E.ROOT
PREVIOUS=ROOT/'robotwinprobe/out/libero13_joint_eot_20261029'
DEFAULT=ROOT/'robotwinprobe/out/libero13_eot_transfer_20261030'
REPORT=ROOT/'result/52_libero_eot_bidirectional_transfer.md'
PORTS=dict(pi=18742,fast=18743)
CONDITIONS=['clean','pi_control','fast_control','pi_eot','fast_eot']


def prepare(root):
    root.mkdir(parents=True,exist_ok=True)
    for name in ['logs','audit','textures','training','gradient_ledgers']:
        (root/name).mkdir(exist_ok=True)
    previous=json.loads((PREVIOUS/'protocol.json').read_text())
    protocol=dict(experiment='libero13_source_only_eot_bidirectional_transfer',
        task='put_the_bowl_on_the_plate',models_frozen=True,
        source_recipe=str(PREVIOUS),non_eot_control=str(M.SOURCE),
        patch=dict(size_m=.13,resolution=128,footprint='fixed site101 square',initialization=.5),
        training=dict(steps_per_source=500,batch=8,lr=.01,independent_initialization=True,
            objective=previous['training']['loss'],
            batch_noise_sha256=sha(M.SOURCE/'training_plan.json'),
            eot_plan_sha256=sha(PREVIOUS/'eot_plan.json'),eot=previous['training']['eot'],
            routing='Q_pi -> pi gradient only; Q_fast -> fast gradient only; no fusion or target selection',
            target_isolation='servers reject mismatched source_model; per-source texture/gradient ledgers',
            checkpoint_selection='fixed final step500; no validation or transfer outcome selection'),
        evaluation=dict(indices=list(range(40,50)),conditions=CONDITIONS,expected_episodes=100,
            nominal='same native per-model protocol, 13cm world texture, both actual camera renders',
            source_success='pi_eot on pi; fast_eot on fast',
            transfer='pi_eot on fast; fast_eot on pi; zero target gradients used to optimize source texture',
            controls='fresh clean and both no-EOT specialist PNGs; require exact per-init reproduction of #46',
            ASR='paired failure / target clean successes, independent denominator per target model',
            texture_selection='sealed final step500 PNGs, same files for source and target'),
        control_png_sha256={k:sha(M.SOURCE/'textures'/(k+'.png')) for k in M.KINDS},
        limits='single task and10 inits; known common state pool; texture-plane EOT only; no joint training')
    E.fixed_json(root/'protocol.json',protocol)
    E.fixed_json(root/'eot_plan.json',json.loads((PREVIOUS/'eot_plan.json').read_text()))
    E.fixed_json(root/'reference.json',json.loads((PREVIOUS/'reference.json').read_text()))
    for kind in M.KINDS:
        r=root/'models'/kind
        r.mkdir(parents=True,exist_ok=True)
        cfg=M.rebase(json.loads((PREVIOUS/'models'/kind/'protocol.json').read_text()))
        cfg.update(experiment=protocol['experiment'],control_port=PORTS[kind],parent_root=str(root),
            isolation='two independent source-only textures; never optimize a texture with target gradients')
        cfg['training'].update(isolated_source_model=kind,
            source_texture=str(root/'textures'/(kind+'_eot.png')),
            fusion='none; this backend may only provide its own source texture gradient')
        cfg['evaluation'].update(conditions=CONDITIONS,baseline_reuse=False,expected_new_episodes=50,
            reference='fresh #46 clean and both specialist control outcomes, per init',
            texture_paths={c:str(root/'textures'/(c+'.png')) for c in CONDITIONS if c!='clean'},
            texture_paths_by_init={})
        E.fixed_json(r/'protocol.json',cfg)
        if kind=='fast':
            for name in ['prompts.npz','checkpoints/libero_uncond_2cam224.pt',
                         'checkpoints/libero_uncond_2cam224_dataset_stats.json']:
                dest=r/name
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():os.link(PREVIOUS/'models/fast'/name,dest)
        control=root/'textures'/(kind+'_control.png')
        if not control.exists():shutil.copy2(M.SOURCE/'textures'/(kind+'.png'),control)
        assert sha(control)==protocol['control_png_sha256'][kind]
    config=root/'libero_config'
    config.mkdir(exist_ok=True)
    (config/'config.yaml').write_text((PREVIOUS/'libero_config/config.yaml').read_text())
    snapshot=root/'source_snapshots'/('prepare_'+str(os.getpid()))
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in ['libero_eot_transfer.py','libero_joint_eot.py','libero_joint_maxmin.py',
                 'texture_eot.py','fastwam_adapter.py','libero_pi_adapter.py','fastwam_libero_trial.py']:
        shutil.copy2(ROOT/'robotwinprobe'/name,snapshot/name)
    for name in ['dual_view_proxy.py','render_world_proxy.py']:
        shutil.copy2(ROOT/'pi05probe'/name,snapshot/name)
    dump(snapshot/'manifest.json',{p.name:sha(p) for p in snapshot.glob('*.py')})
    return protocol


def report(root):
    status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'prepared'}
    lines=['# LIBERO 13cm：单模型EOT与双向零样本纹理迁移','',
        f'状态：`{status["stage"]}`；节点：`{status.get("host","unknown")}`。','',
        'π和FastWAM各自从灰色独立训练一张Q128纹理，各500×8、sign-PGD .01、首步L0；模型权重冻结。',
        '沿用第51号相同batch、动作noise和EOT抽样：旋转±15°，平移各轴±5%，缩放0.9–1.1，RGB统一增益0.9–1.1。',
        '每张纹理仅接收其源模型梯度；没有联训/梯度融合，另一模型的输出不参与更新或选纹理。服务端拒绝跨源梯度请求并记录逐步来源。',
        '两张最终500步PNG在迁移评估前封存；源模型与目标模型使用同一文件。真实MuJoCo两相机渲染，标准13cm外轮廓/位置。',
        '复测clean、两张无EOT专用纹理、两张单源EOT纹理；两目标各10初态，共100条闭环。','']
    for kind in M.KINDS:
        p=root/'training'/kind/'progress.json'
        if p.exists():
            v=json.loads(p.read_text());lines.append(f'{kind}单源EOT训练：{v["iteration"]}/500。')
    lines+=['','| 目标模型 | 纹理条件 | 完成 | 配对失败/clean成功 |','|---|---|---:|---:|']
    for kind in M.KINDS:
        rows=M.outcomes(root,kind)
        eligible={i for i,v in rows.get('clean',{}).items() if v}
        for c in CONDITIONS:
            data=rows.get(c,{})
            pairs=eligible & data.keys()
            score='—' if c=='clean' or not pairs else f'{sum(not data[i] for i in pairs)}/{len(pairs)}'
            lines.append(f'| {kind} | {c} | {len(data)}/10 | {score} |')
    if status.get('error'):lines+=['',status['error']]
    lines+=['','解释：pi_eot→pi与fast_eot→fast为源攻击阳性检查；pi_eot→fast与fast_eot→pi为本轮双向迁移。',
        '源攻击不成功时须明确报告，不能将其目标零结果单独归因迁移。未完成条件不报告最终ASR。',
        '单任务、10初态、固定训练/动作noise与EOT种子；不声称跨任务泛化，不把纹理内部变换称作物理摆放变化。',
        f'原始目录：`{root}`。']
    REPORT.write_text('\n'.join(lines)+'\n')


def source_rpc(payloads):
    """Different Q for each server; there is no shared-gradient training RPC."""
    connections={}
    result={}
    try:
        for kind,request in payloads.items():
            E.validate_source_request(request,kind,kind)
            connection=Client(('127.0.0.1',PORTS[kind]),authkey=AUTH)
            connections[kind]=connection
            connection.send(request)
            if M.SERIAL_RPC:
                result[kind]=connection.recv()
        if not M.SERIAL_RPC:
            result={kind:connection.recv() for kind,connection in connections.items()}
    finally:
        for connection in connections.values():connection.close()
    for kind,value in result.items():
        if 'error' in value:raise RuntimeError(kind+': '+value['error']+'\n'+value.get('traceback',''))
    return result


def train_sources(root):
    batches=json.loads((M.SOURCE/'training_plan.json').read_text())
    draws=json.loads((root/'eot_plan.json').read_text())
    textures={};histories={}
    for kind in M.KINDS:
        dest=root/'training'/kind
        dest.mkdir(exist_ok=True)
        state=dest/'state.npz'
        if state.exists():
            with np.load(state) as z:
                textures[kind]=z['texture'].copy();histories[kind]=json.loads(str(z['history']))
        else:
            textures[kind]=np.full((1,3,128,128),.5,np.float32);histories[kind]=[]
    for it in range(500):
        active=[k for k in M.KINDS if len(histories[k])<=it]
        if not active:continue
        batch=batches[it]
        requests={k:dict(command='batch',texture=textures[k],indices=batch['indices'],
            seeds=batch['seeds'],draws=draws[it],gradient=True,source_model=k,iteration=it+1) for k in active}
        results=source_rpc(requests)
        for kind in active:
            value=results[kind]
            provenance=value['gradient_provenance']
            input_sha=hashlib.sha256(textures[kind].tobytes()).hexdigest()
            assert provenance['source_model']==provenance['server_model']==kind
            assert provenance['iteration']==it+1 and provenance['texture_sha256']==input_sha
            assert provenance['indices']==batch['indices'] and provenance['seeds']==batch['seeds']
            expected_draw_sha=hashlib.sha256(json.dumps(draws[it],sort_keys=True).encode()).hexdigest()
            assert provenance['draws_sha256']==expected_draw_sha
            gradient=value['gradient'];assert np.isfinite(gradient).all() and np.linalg.norm(gradient)>0
            textures[kind]=np.clip(textures[kind]+.01*np.sign(gradient),0,1).astype(np.float32)
            histories[kind].append(dict(iteration=it+1,source_model=kind,
                loss=float(np.mean(value['losses'])),gradient_provenance=provenance,updated=time.time()))
            dest=root/'training'/kind
            temporary=dest/'state.tmp.npz'
            np.savez_compressed(temporary,texture=textures[kind],history=json.dumps(histories[kind]))
            temporary.replace(dest/'state.npz')
            dump(dest/'progress.json',dict(iteration=it+1,total=500,complete=it==499,
                source_model=kind,source_only=True,last=histories[kind][-1]))
            if it==0 or (it+1)%25==0:
                Image.fromarray(np.rint(textures[kind][0].transpose(1,2,0)*255).astype(np.uint8)).save(dest/f'texture_{it+1:04d}.png')
            print('[source-only EOT]',kind,it+1,histories[kind][-1]['loss'],flush=True)
        if it==0 or (it+1)%25==0:report(root)
    for kind in M.KINDS:
        assert len(histories[kind])==500
        final=root/'textures'/(kind+'_eot.png')
        source=root/'training'/kind/'texture_0500.png'
        if not final.exists():shutil.copy2(source,final)
        assert sha(final)==sha(source)


def seal_textures(root):
    """No model queries here: the target never chooses the source texture."""
    sealed=dict(protocol_sha256=sha(root/'protocol.json'),
        textures={p.name:sha(p) for p in sorted((root/'textures').glob('*.png'))},
        training_state_sha256={},source_gradient_checks={},
        selection='fixed step500 from independent source-only gradients, before any transfer evaluation')
    for kind in M.KINDS:
        state=root/'training'/kind/'state.npz'
        with np.load(state) as z:history=json.loads(str(z['history']))
        assert len(history)==500
        assert [v['iteration'] for v in history]==list(range(1,501))
        assert all(v['source_model']==kind and v['gradient_provenance']['source_model']==kind
                   and v['gradient_provenance']['server_model']==kind for v in history)
        sealed['training_state_sha256'][kind]=sha(state)
        sealed['source_gradient_checks'][kind]=dict(updates=500,source_model=kind,
            all_updates_source_only=True,ledger_sha256=sha(root/'gradient_ledgers'/(kind+'.jsonl')))
    E.fixed_json(root/'sealed_textures.json',sealed)


def configure():
    E.PORTS=PORTS
    E.configure()
    M.DEFAULT=DEFAULT;M.REPORT=REPORT;M.PORT=PORTS;M.CONDITIONS=CONDITIONS
    M.prepare=prepare;M.report=report;M.train=train_sources;M.heldout=seal_textures
    parent_status=M.status
    def guarded_status(root,stage,**kw):
        if stage=='evaluate_pi_eot_40':
            audit={}
            for kind in M.KINDS:
                new=M.outcomes(root,kind);old=M.outcomes(M.SOURCE,kind)
                audit[kind]={k:dict(passed=new[k+'_control']==old[k],
                    actual=new[k+'_control'],reference=old[k]) for k in M.KINDS}
            dump(root/'audit/no_eot_controls.json',audit)
            assert all(v['passed'] for a in audit.values() for v in a.values()),'no-EOT controls changed from #46'
        if stage=='train_maxmin':stage='train_independent_eot'
        if stage=='heldout_loss':stage='seal_source_textures'
        parent_status(root,stage,**kw)
    M.status=guarded_status
    M.__file__=str(Path(__file__).absolute())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','pipeline','server','report'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--kind',choices=M.KINDS)
    parser.add_argument('--gpus',type=int,nargs=2,default=[0,1])
    args=parser.parse_args()
    configure()
    if args.command=='prepare':prepare(args.root);M.status(args.root,'prepared')
    elif args.command=='pipeline':M.pipeline(args.root,args.gpus)
    elif args.command=='server':E.server(args.root,args.kind)
    else:report(args.root)
