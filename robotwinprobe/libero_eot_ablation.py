"""Paired single-factor and leave-one-out EOT source/transfer ablation.

Eight masks, two independently trained source textures per mask. Historical
full/no-EOT endpoints are verified #52 references. Every variant rechecks its
own native clean denominator before evaluating both frozen source textures.
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

import libero_eot_transfer as T
from crossarch_common import dump,sha
from eot_ablation_factors import FACTORS,VARIANTS,LABELS,masked_plan

E=T.E;M=T.M;ROOT=T.ROOT
DEFAULT=ROOT/'robotwinprobe/out/libero13_eot_ablation_20261030'
REFERENCE=T.DEFAULT
REPORT=ROOT/'result/54_libero_eot_factor_ablation.md'
CURRENT=None
MASTER=DEFAULT
CONDITIONS=['clean','pi_eot','fast_eot']


def prepare_master(root):
    root.mkdir(parents=True,exist_ok=True)
    for name in ['variants','lanes','audit']:(root/name).mkdir(exist_ok=True)
    reference=json.loads((REFERENCE/'completion_verification.json').read_text())
    assert reference['completed_episodes']==100
    full_plan=T.PREVIOUS/'eot_plan.json'
    p=dict(experiment='libero13_eot_only_one_leave_one_out',factors={k:list(v) for k,v in FACTORS.items()},variants=VARIANTS,
        groups=dict(single_factor='only_i minus none',conditional_contribution='full minus drop_i'),
        reference=str(REFERENCE),reference_verification_sha256=sha(REFERENCE/'completion_verification.json'),
        base_plan=str(full_plan),base_plan_sha256=sha(full_plan),
        training=dict(mode='independent source-only',sources=['pi','fast'],steps=500,batch=8,lr=.01,
            initialization=.5,patch_size_m=.13,resolution=128,
            source_batch_noise_plan_sha256=sha(M.SOURCE/'training_plan.json'),
            losses='unchanged native first-flow pi10x32 SSE / fast32x7 MSE',
            masking='keep exact same realization for every enabled parameter; disabled parameters at identity',
            new_textures=16,gradient_frames_per_texture=4000,selection='fixed final500; no ASR selection'),
        evaluation=dict(task='put_the_bowl_on_the_plate',indices=list(range(40,50)),
            targets=['pi','fast'],conditions_per_variant=CONDITIONS,
            source_and_transfer=True,episodes_per_variant=60,total_new_episodes=480,
            clean='fresh per variant, strict per-init match to #46',
            references='reuse #52 verified full-EOT and no-EOT per-init outcomes, no retraining endpoints',
            primary='each directed transfer ASR; report source ASR separately',
            no_new_imagewam_conditions=True),
        limits='one training/EOT seed and10 initial states; preliminary factor screening, not stable ranking or full interaction decomposition')
    E.fixed_json(root/'protocol.json',p)
    E.fixed_json(root/'reference_summary.json',reference)
    return p


def prepare_variant(root):
    variant=CURRENT;assert variant in VARIANTS
    master=root.parent.parent
    prepare_master(master)
    root.mkdir(parents=True,exist_ok=True)
    for name in ['logs','audit','textures','training','gradient_ledgers']:(root/name).mkdir(exist_ok=True)
    full=json.loads((T.PREVIOUS/'eot_plan.json').read_text())
    draws=masked_plan(full,variant)
    protocol=dict(experiment='libero13_source_only_eot_ablation',variant=variant,
        active_factors=VARIANTS[variant],factor_keys={k:list(v) for k,v in FACTORS.items()},source_recipe=str(REFERENCE),
        training=dict(steps_per_source=500,batch=8,lr=.01,initialization=.5,
            models_frozen=True,independent_sources=['pi','fast'],
            batch_noise_plan_sha256=sha(M.SOURCE/'training_plan.json'),
            full_draws_sha256=sha(T.PREVIOUS/'eot_plan.json'),
            masking='enabled draws exactly preserved; disabled entries use IDENTITY',
            no_target_gradients=True,no_gradient_fusion=True),
        evaluation=dict(conditions=CONDITIONS,indices=list(range(40,50)),expected_episodes=60,
            native_protocol='same as #52 for each model',fresh_clean_matches_46=True,
            ASR='attacked failure / this target clean successes'),
        reference_verification_sha256=sha(REFERENCE/'completion_verification.json'))
    E.fixed_json(root/'protocol.json',protocol)
    E.fixed_json(root/'eot_plan.json',draws)
    E.fixed_json(root/'reference.json',json.loads((REFERENCE/'reference.json').read_text()))
    index=list(VARIANTS).index(variant)
    ports=dict(pi=18760+index*2,fast=18761+index*2)
    for kind in M.KINDS:
        r=root/'models'/kind;r.mkdir(parents=True,exist_ok=True)
        cfg=M.rebase(json.loads((REFERENCE/'models'/kind/'protocol.json').read_text()))
        cfg.update(experiment=protocol['experiment'],ablation_variant=variant,
            control_port=ports[kind],parent_root=str(root))
        cfg['training'].update(isolated_source_model=kind,active_eot_factors=VARIANTS[variant],
            source_texture=str(root/'textures'/(kind+'_eot.png')),
            eot_masked_plan_sha256=sha(root/'eot_plan.json'))
        cfg['evaluation'].update(conditions=CONDITIONS,expected_new_episodes=30,
            reference='fresh variant clean must match #46; endpoint comparisons from verified #52',
            texture_paths={c:str(root/'textures'/(c+'.png')) for c in ['pi_eot','fast_eot']},
            texture_paths_by_init={})
        E.fixed_json(r/'protocol.json',cfg)
        if kind=='fast':
            for name in ['prompts.npz','checkpoints/libero_uncond_2cam224.pt',
                         'checkpoints/libero_uncond_2cam224_dataset_stats.json']:
                dest=r/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():os.link(REFERENCE/'models/fast'/name,dest)
    config=root/'libero_config';config.mkdir(exist_ok=True)
    (config/'config.yaml').write_text((REFERENCE/'libero_config/config.yaml').read_text())
    snapshot=root/'source_snapshots'/('prepare_'+str(os.getpid()));snapshot.mkdir(parents=True,exist_ok=True)
    for name in ['libero_eot_ablation.py','eot_ablation_factors.py','libero_eot_transfer.py',
                 'libero_joint_eot.py','libero_joint_maxmin.py','texture_eot.py',
                 'fastwam_libero_trial.py','fastwam_adapter.py','libero_pi_adapter.py']:
        shutil.copy2(ROOT/'robotwinprobe'/name,snapshot/name)
    dump(snapshot/'manifest.json',{p.name:sha(p) for p in snapshot.glob('*.py')})
    return protocol


def scores_from_rows(rows):
    eligible={i for i,v in rows.get('clean',{}).items() if v}
    out={}
    for source in ['pi','fast']:
        data=rows.get(source+'_eot',{})
        out[source]=None if len(data)!=10 or len(rows.get('clean',{}))!=10 else dict(
            failures=sum(not data[i] for i in eligible),denominator=len(eligible),
            ASR=sum(not data[i] for i in eligible)/len(eligible))
    return out


def aggregate(root):
    reference=json.loads((root/'reference_summary.json').read_text())
    data={}
    for endpoint,suffix in [('none','control'),('full','eot')]:
        data[endpoint]={target:{source:reference['models'][target]['conditions'][source+'_'+suffix]
            for source in ['pi','fast']} for target in ['pi','fast']}
    for variant in VARIANTS:
        r=root/'variants'/variant
        data[variant]={target:scores_from_rows(M.outcomes(r,target)) for target in ['pi','fast']}
    effects={}
    for factor in FACTORS:
        effects[factor]={}
        for target in ['pi','fast']:
            effects[factor][target]={}
            for source in ['pi','fast']:
                single=data['only_'+factor][target][source];drop=data['drop_'+factor][target][source]
                effects[factor][target][source]=dict(
                    single_gain=None if single is None else single['ASR']-data['none'][target][source]['ASR'],
                    conditional_contribution=None if drop is None else data['full'][target][source]['ASR']-drop['ASR'])
    return dict(scores=data,effects=effects,
        interpretation='single_gain is independent benefit; full-minus-drop is conditional contribution, not a proof of necessity; interactions may change ranks')


def report_master(root):
    if not (root/'reference_summary.json').exists():return
    data=aggregate(root)
    lines=['# LIBERO EOT：单项加入与去单项消融','',
        '四项：旋转±15°；二维平移各轴±5%纹理宽高；缩放0.9–1.1；RGB统一增益0.9–1.1。固定13cm贴纸外轮廓/位置，不改变相机或场景光照。',
        '两组各4个条件，π/FastWAM各自独立灰色训练500×8、sign-PGD .01，共16张新纹理；只使用各自源梯度。',
        '所有启用项复用第51/52号同一份逐帧抽样，只将禁用项替换为恒等值；状态、batch、动作noise、模型、loss和预算相同。',
        '完整EOT与无EOT端点复用已核验第52号逐初态结果；每个新条件重新核对clean并严格匹配第46号。新闭环共480条，源攻击与两个方向迁移均保留。',
        '主分析为跨架构迁移ASR，源攻击作为有效性检查。现有ImageWAM第53号单独继续，本轮不追加ImageWAM因子条件。','',
        '| 条件 | π源自身 | π→FastWAM | FastWAM源自身 | FastWAM→π |','|---|---:|---:|---:|---:|']
    def fmt(value):return '待完成' if value is None else f'{value["failures"]}/{value["denominator"]}'
    for key in ['none','full',*VARIANTS]:
        row=data['scores'][key]
        lines.append('| '+key+' | '+' | '.join(fmt(row[t][s]) for t,s in [('pi','pi'),('fast','pi'),('fast','fast'),('pi','fast')])+' |')
    lines+=['','## 进度','']
    for variant in VARIANTS:
        r=root/'variants'/variant
        status=json.loads((r/'status.json').read_text()) if (r/'status.json').exists() else {'stage':'prepared'}
        training=[]
        for k in ['pi','fast']:
            p=r/'training'/k/'progress.json'
            training.append(f'{k} '+(str(json.loads(p.read_text())['iteration']) if p.exists() else '0')+'/500')
        lines.append(f'- {variant}: {status["stage"]}；'+', '.join(training))
    lines+=['','## 分析口径','',
        '- 单项组：ASR(only_i) − ASR(none)，衡量独立收益。',
        '- 去项组：ASR(full) − ASR(drop_i)，衡量完整组合中的条件贡献；不能直接称严格必要性。',
        '- 两者分别对每个源/目标报告，不用两模型不同单位的L0排序，也不把源攻击增强与迁移增强混为一谈。',
        '- 高独立收益且高条件贡献：优先候选；单项弱但去掉损失大：可能依赖互补；单项强但去掉无损失：可能冗余。',
        '- 仅单训练/EOT种子、10初态；差1–2个失败不宣称稳定最优，不分解所有高阶交互。',
        f'原始目录：`{root}`。']
    temp=REPORT.with_name(REPORT.name+f'.{os.getpid()}.tmp')
    temp.write_text('\n'.join(lines)+'\n');temp.replace(REPORT)
    analysis_temp=root/f'analysis_latest.{os.getpid()}.tmp'
    analysis_temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    analysis_temp.replace(root/'analysis_latest.json')


def report_variant(root):
    status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'prepared'}
    dump(root/'summary.json',dict(variant=CURRENT,stage=status['stage'],
        scores={k:scores_from_rows(M.outcomes(root,k)) for k in ['pi','fast']}))
    report_master(root.parent.parent)


def configure(variant):
    global CURRENT
    CURRENT=variant
    index=list(VARIANTS).index(variant)
    ports=dict(pi=18760+index*2,fast=18761+index*2)
    E.PORTS=ports;T.PORTS=ports
    E.configure()
    M.PORT=ports;M.CONDITIONS=CONDITIONS;M.REPORT=REPORT
    T.report=report_variant
    M.prepare=prepare_variant;M.report=report_variant;M.train=T.train_sources;M.heldout=T.seal_textures
    original=M.status
    def named_status(root,stage,**kw):
        if stage=='train_maxmin':stage='train_source_only_ablation'
        if stage=='heldout_loss':stage='seal_source_textures'
        original(root,stage,**kw)
    M.status=named_status
    M.__file__=str(Path(__file__).absolute())


def lane(root,name,variants,gpus):
    prepare_master(root)
    directory=root/'lanes'/name;directory.mkdir(exist_ok=True)
    lock=(directory/'lane.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,WAMATTACK_PROJECT_ROOT=str(ROOT))
    for variant in variants:
        state=root/'variants'/variant/'status.json'
        if state.exists() and json.loads(state.read_text())['stage']=='complete':continue
        command=[sys.executable,'-u',str(Path(__file__).absolute()),'pipeline','--root',str(root),
            '--variant',variant,'--gpus',str(gpus[0]),str(gpus[1])]
        with (directory/'pipeline.log').open('a') as log:
            child=subprocess.Popen(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            dump(directory/'status.json',dict(stage='running',host=socket.gethostname(),pid=os.getpid(),
                child_pid=child.pid,variant=variant,gpus=gpus,variants=variants,updated=time.time()))
            code=child.wait()
        if code:
            dump(directory/'status.json',dict(stage='exited_error',host=socket.gethostname(),pid=os.getpid(),
                child_pid=child.pid,variant=variant,gpus=gpus,variants=variants,exit_code=code,updated=time.time()))
            raise RuntimeError('ablation variant failed: '+variant)
    dump(directory/'status.json',dict(stage='complete',host=socket.gethostname(),pid=os.getpid(),
        variants=variants,gpus=gpus,updated=time.time()))
    report_master(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','pipeline','server','report','lane'])
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--variant',choices=list(VARIANTS))
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS))
    parser.add_argument('--lane',default='lane_a')
    parser.add_argument('--kind',choices=['pi','fast'])
    parser.add_argument('--gpus',nargs=2,type=int,default=[0,1])
    args=parser.parse_args()
    if args.command=='lane':lane(args.root,args.lane,args.variants,args.gpus)
    elif args.command=='report':report_master(args.root)
    elif args.command=='prepare':
        prepare_master(args.root)
        for variant in VARIANTS:
            configure(variant);prepare_variant(args.root/'variants'/variant)
        report_master(args.root)
    else:
        if args.command=='server' and args.variant is None:args.variant=args.root.name
        assert args.variant in VARIANTS
        configure(args.variant)
        r=args.root if args.command=='server' else args.root/'variants'/args.variant
        if args.command=='server':E.server(r,args.kind)
        else:M.pipeline(r,args.gpus)
