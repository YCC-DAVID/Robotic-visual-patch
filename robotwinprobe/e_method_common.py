"""Independent E-method improvement study; frozen cohorts and real-render endpoints."""
import copy
import json
from multiprocessing.connection import Client
from pathlib import Path
import shutil

import numpy as np

from checkpoint_study_common import ROOT, CAMERAS, WIRE, dump, sha

DEFAULT = ROOT/'robotwinprobe/out/e_method_20261027'
SOURCE = ROOT/'robotwinprobe/out/stage_joint_20261026'
PORT = 18771
AUTH = b'robotwin-e-method-local-20261027'
PY = '/home/user1/miniconda3/envs/openpi-server/bin/python'
CPU = '/home/user1/miniconda3/bin/python'
OBJECTIVES = ('velocity14','velocity32','action14','action_active7')
REFINE_OBJECTIVES = ('velocity_gripper','velocity_joint_balanced')
ALL_OBJECTIVES = OBJECTIVES + REFINE_OBJECTIVES
BASELINES = ('clean','random','legacy','early_ref')
SCREEN = BASELINES + OBJECTIVES


def load(root):
    return json.loads((Path(root)/'protocol.json').read_text())


def prepare(root=DEFAULT):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():return load(root)
    p=copy.deepcopy(load(SOURCE))
    p.update(experiment='e_method_improvement',version=1,backend_port=PORT,
        approval='User requested improving E training for stable sustained-exposure ASR; engineering repair/resume authorized.',
        predecessor=str(SOURCE),predecessor_protocol_sha256=sha(SOURCE/'protocol.json'))
    # data.train_seeds remains the existing demonstration audit source. The actual
    # new training cohort is explicitly different and is never expert-filtered.
    p['method_study']=dict(training_seeds=list(range(25000,25024)),
        development_seeds=list(range(24000,24032)),test_seeds=list(range(26000,26100)),
        objectives=list(OBJECTIVES),training_random_seeds=[202610270,202610271,202610272],
        steps=500,batch=8,lr=.01,optimizer='projected sign ascent',
        data='native pi0.5 clean rollout query states in approach, first at most three per trajectory, retain successful and failed clean episodes',
        calibration='nominal plus one pre-recorded 3cm/15deg/three-light EOT draw per state',
        draws='75% nominal / 25% EOT, uniform source trajectory then state; identical draws and noises for all four objectives',
        loss='velocity14/velocity32: first t=1 MSE; action14: final normalized 10-flow MSE; action_active7: final normalized active-arm seven dimensions MSE',
        checkpoint='fixed final step500; no test-driven checkpoint selection',
        selection='largest development paired excess ASR, then ASR, then predeclared objective order',
        replication='only selected objective retrained from gray with two other fixed training seeds; evaluate on same development cohort',
        final='one fixed100-seed held-out cohort; clean/random/legacy/early_ref and all three replicas of selected objective',
        target=dict(ASR=.30,excess_ASR=.20,positive_paired_CI=True,min_test_clean_successes=50,
                    note='provisional practical target pending user preference; assess each of three training replicas'),
        method_scope='single pi0.5, original task/site07/10cm; no policy fine-tuning or position/size search',
        runtime_error='preserve error; engineering fixes and restart from validated artifacts; never count as attack success')
    p['evaluation']['noise_base']=202610270
    dump(root/'protocol.json',p)
    references={'random':SOURCE/'patches/random/texture.png',
        'early_ref':SOURCE/'patches/early/texture_0500.png',
        'legacy':ROOT/'robotwinprobe/out/table9_joint_eot_10cm_20261012/sites/site07/train/texture_0500.png'}
    for name,source in references.items():
        dest=root/'patches'/name;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest/('texture.png' if name=='random' else 'texture_0500.png'))
    dump(root/'reference_textures.json',{name:dict(source=str(path),sha256=sha(path)) for name,path in references.items()})
    dump(root/'progress.json',dict(stage='prepared'))
    return p


def request(payload,root):
    with Client(('127.0.0.1',load(root)['backend_port']),authkey=AUTH) as connection:
        connection.send(payload);response=connection.recv()
    if 'error' in response:raise RuntimeError(response['error']+'\n'+response.get('traceback',''))
    return response


def cohort(root,split):
    path=root/'cohorts'/split;path.mkdir(parents=True,exist_ok=True)
    protocol=path/'protocol.json'
    if not protocol.exists():shutil.copy2(root/'protocol.json',protocol)
    assert sha(protocol)==sha(root/'protocol.json')
    patches=path/'patches'
    if not patches.exists():patches.symlink_to(root/'patches',target_is_directory=True)
    return path


def draw_plan(root,seed):
    manifest=json.loads((root/'collection_manifest.json').read_text())
    grouped={}
    for row in manifest['records']:
        grouped.setdefault(row['seed'],{}).setdefault(row['query'],{})[row['draw']]=row['path']
    rng=np.random.default_rng(seed)
    result=[]
    for _ in range(load(root)['method_study']['steps']):
        batch=[]
        for _ in range(8):
            source=int(rng.choice(sorted(grouped)));query=int(rng.choice(sorted(grouped[source])))
            draw=int(rng.random()>=.75)
            batch.append(dict(record=grouped[source][query][draw],seed=int(rng.integers(1,2**30))))
        result.append(batch)
    return result


def results(root,split,conditions):
    from stage_joint_report import interval,wilson
    seeds=load(root)['method_study'][split+'_seeds']
    folder=root/'cohorts'/split/'episodes'
    rows={}
    errors=[]
    for path in folder.glob('*.json'):
        item=json.loads(path.read_text());rows[(item['job']['seed'],item['job']['condition'])]=item
        if item.get('error'):errors.append(dict(path=str(path),error=item['error']))
    out=[]
    for condition in conditions:
        triples=[(seed,rows.get((seed,'clean')),rows.get((seed,'random')),rows.get((seed,condition))) for seed in seeds]
        complete=[(s,c,r,a) for s,c,r,a in triples if all(x and x['status']=='complete' for x in (c,r,a))]
        eligible=[(s,c,r,a) for s,c,r,a in complete if c['success']]
        failed=[int(not a['success']) for s,c,r,a in eligible]
        rand=[int(not r['success']) for s,c,r,a in eligible]
        differences=[a-b for a,b in zip(failed,rand)]
        out.append(dict(condition=condition,completed_pairs=len(complete),n=len(eligible),failures=sum(failed),
            ASR=float(np.mean(failed)) if failed else None,ASR_CI95=wilson(failed),
            random_failures=sum(rand),random_ASR=float(np.mean(rand)) if rand else None,
            excess_ASR=float(np.mean(differences)) if failed else None,excess_CI95=interval(differences)))
    return dict(split=split,expected_seeds=len(seeds),
        clean_completed=sum(rows.get((s,'clean'),{}).get('status')=='complete' for s in seeds),
        clean_successes=sum(bool(rows.get((s,'clean'),{}).get('success')) for s in seeds),
        records=len(rows),errors=errors,rows=out)


def report(root):
    p=load(root);status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'prepared'}
    selection=json.loads((root/'selection.json').read_text()) if (root/'selection.json').exists() else None
    refinement_path=root/'refinement/protocol.json'
    refinement=json.loads(refinement_path.read_text()) if refinement_path.exists() else None
    refined_selection_path=root/'refinement/selection.json'
    refined_selection=json.loads(refined_selection_path.read_text()) if refined_selection_path.exists() else None
    group_path=root/'group_round/protocol.json'
    group_round=json.loads(group_path.read_text()) if group_path.exists() else None
    group_selection_path=root/'group_round/selection.json'
    group_selection=json.loads(group_selection_path.read_text()) if group_selection_path.exists() else None
    onpolicy_path=root/'onpolicy_round/protocol.json'
    onpolicy_round=json.loads(onpolicy_path.read_text()) if onpolicy_path.exists() else None
    onpolicy_selection_path=root/'onpolicy_round/selection.json'
    onpolicy_selection=json.loads(onpolicy_selection_path.read_text()) if onpolicy_selection_path.exists() else None
    coverage_path=root/'coverage_round/protocol.json'
    coverage_round=json.loads(coverage_path.read_text()) if coverage_path.exists() else None
    coverage_selection_path=root/'coverage_round/selection.json'
    coverage_selection=json.loads(coverage_selection_path.read_text()) if coverage_selection_path.exists() else None
    replication_path=root/'coverage_round/replication.json'
    coverage_replication=json.loads(replication_path.read_text()) if replication_path.exists() else None
    conditions=list(SCREEN)+list(REFINE_OBJECTIVES if refinement else ())+['replica1','replica2']
    if refined_selection:conditions+=refined_selection.get('replica_conditions',[])
    if group_round:conditions+=group_round['methods']
    if group_selection:conditions+=group_selection.get('replica_conditions',[])
    if onpolicy_round:conditions+=onpolicy_round['methods']
    if onpolicy_selection:conditions+=onpolicy_selection.get('replica_conditions',[])
    if coverage_round:conditions+=coverage_round['methods']
    if coverage_selection:conditions+=coverage_selection.get('replica_conditions',[])
    conditions=list(dict.fromkeys(conditions))
    splits={name:results(root,name,conditions) for name in ('development','test')}
    progress={}
    for path in (root/'training').glob('*/progress.json'):progress[path.parent.name]=json.loads(path.read_text())
    collected=list((root/'collection').glob('seed*.json'))
    out=dict(status=status,goal='Improve E sustained real-render closed-loop ASR, repeatable across training seeds',
        collection_seeds_completed=len(collected),collection_seeds_expected=24,training=progress,
        selection=selection,refinement=refinement,refined_selection=refined_selection,
        group_round=group_round,group_selection=group_selection,
        onpolicy_round=onpolicy_round,onpolicy_selection=onpolicy_selection,
        coverage_round=coverage_round,coverage_selection=coverage_selection,coverage_replication=coverage_replication,
        onpolicy_collection_seeds_completed=len(list((root/'onpolicy_round/collection').glob('seed*.json'))),
        results=splits,protocol_sha256=sha(root/'protocol.json'))
    dump(root/'summary.json',out)
    lines=['# RoboTwin：E 持续暴露训练方法改进','',
        '目标：在冻结π0.5、place_container_plate、site07和10cm条件下，建立可重复的持续暴露攻击基线。',
        f'当前状态：**{status["stage"]}**。策略自身接近阶段采集：{len(collected)}/24初态。',
        '', '计划：采集策略自身接近阶段状态 → 四种损失受控训练 → 32初态开发集比较 → 胜出目标另外两份训练seed复验 → 独立100初态最终测试与审计。',
        '四种目标共享数据与抽样：首步有效14维、首步完整32维、完整10步动作有效14维、完整10步操作臂7维。各500×8，从灰色初始化；75%原位标定、25%共同EOT。原有E和旧site07作为固定基线。',
        '暂定实用目标：每份训练seed在最终测试ASR≥30%、相对Random≥20个百分点、配对95%区间下界>0，且至少50个clean成功初态。该目标是验收标准，尚不代表已达到。', '']
    if progress:
        lines+=['训练进度：'+ '；'.join(f'{name} {r["iteration"]}/500' for name,r in progress.items()),'']
    for split,label in [('development','开发集'),('test','独立测试集')]:
        data=splits[split]
        lines += [f'{label}：clean {data["clean_successes"]}/{data["clean_completed"]}，预定{data["expected_seeds"]}初态。','',
            '| 纹理 | 配对已完成 | 攻击成功/有效初态 | ASR | Random | 配对增量 |','|---|---:|---:|---:|---:|---:|']
        for r in data['rows']:
            if r['condition']=='clean' or not r['completed_pairs']:continue
            values=[f'{r[k]:.1%}' if r[k] is not None else '待测' for k in ('ASR','random_ASR','excess_ASR')]
            lines.append(f'| {r["condition"]} | {r["completed_pairs"]} | {r["failures"]}/{r["n"]} | {values[0]} | {values[1]} | {values[2]} |')
        lines+=['']
    if selection:lines += ['选择与验收：`'+json.dumps(selection,ensure_ascii=False)+'`','']
    if refinement:
        lines += ['第二轮：保持首步L0、原24初态/49状态、相同batch/noise和500×8预算，从灰色重新训练。比较操作侧夹爪L0，以及操作臂六关节/夹爪各占一半、按训练Random固定尺度归一化的L0。',
            '动机：第一轮四目标均6/22，对照Random3/22；真实初始状态的有效动作扰动已有提升，夹爪扰动提升较弱。该动机是待检验假设，不是已确认的失败机制。',
            '开发集复用用于方法选择；独立测试集保持未用于训练和选择。第二轮协议：`'+str(refinement_path)+'`。','']
        if refined_selection:lines += ['第二轮选择：`'+json.dumps(refined_selection,ensure_ascii=False)+'`','']
    if group_round:
        lines += ['第三轮：保持首步14维L0、原数据与500×8预算，按训练集物体类型（碗/杯）及操作侧（左/右）分四组，每个batch各取2条。比较普通均值PGD与冲突梯度投影，二者共享全部抽样/noise，从相同灰色初始化。',
            '分组只使用训练元数据；梯度诊断是代理梯度分析，不作为真实物理梯度或闭环有效性的证明。开发集继续用于选择，独立测试保持未用于方法改进。',
            '第三轮协议：`'+str(group_path)+'`。','']
        diagnostic=group_round['diagnostic_support']
        lines += [f'固定首轮velocity14纹理的训练梯度诊断：{diagnostic["independent_training_seeds"]}初态、{diagnostic["states"]}状态；'
            f'先合并同初态的相关状态，再比较初态间梯度，其中{diagnostic["across_initial_states"]["negative_pair_fraction"]:.1%}的两两夹角余弦为负；'
            f'四组均值中{diagnostic["across_groups"]["negative_pair_fraction"]:.1%}的组对余弦为负。'
            f'同状态两份noise梯度余弦的中位数为{diagnostic["median_within_state_noise_cosine"]:.4f}。'
            '这些数值只支持检验状态间梯度牵制，不能证明其导致整条轨迹失败，也不能证明投影会提高ASR。','']
        if group_selection:lines += ['第三轮选择：`'+json.dumps(group_selection,ensure_ascii=False)+'`','']
    if onpolicy_round:
        count=out['onpolicy_collection_seeds_completed']
        lines += [f'第四轮：原24训练初态上，以冻结首轮velocity14纹理持续暴露，采集原approach阶段的固定query索引0、1、2、4、8、16；保留成功与失败，已完成{count}/24初态。临时标定不推进动作，执行前恢复原位光照与父纹理，并验证物理状态和RGB恢复。',
            '比较两版相同父纹理的500×8继续训练：原clean数据继续训练；每batch四条相同clean数据加四条受扰状态数据。两版首步14维L0、学习率、动作noise和EOT抽样相同，固定最终checkpoint。该对照同时控制继续训练预算。',
            '受扰训练轨迹只构建训练池，不计作开发/独立测试ASR。同状态的有/无纹理动作响应只能描述图像影响，不能证明整条轨迹因果机制。',
            '如开发集达标，两份新训练seed各自从灰色重训父纹理，再继续训练；受扰数据池保持固定。独立测试仍须方法冻结后执行。',
            '第四轮协议：`'+str(onpolicy_path)+'`。','']
        if onpolicy_selection:lines += ['第四轮选择：`'+json.dumps(onpolicy_selection,ensure_ascii=False)+'`','']
    if coverage_round:
        lines += ['第五轮：第四轮受扰训练池48状态中24个为初始query0，另外24个为执行第一个动作块后的query1。保持同一父纹理和500×8继续训练预算，把四个受扰槽位优先抽样各初态的非初始状态；前四个clean槽位、全部noise/EOT和受扰源初态均与第四轮混合版一致，不按结果筛选。',
            '完成该候选开发比较后，为开发集最佳方法补做两份独立训练seed：每份各自从灰色训练父纹理500步，再继续训练500步。即使首份未达暂定目标也做开发集稳定性诊断，不降低门槛、不提前使用独立测试。',
            '只有首份和两份新训练seed均达到原开发ASR/增量门槛，才冻结方法并进入独立测试；开发结果不作为稳定性验收结论。',
            '第五轮协议：`'+str(coverage_path)+'`。','']
        if coverage_selection:lines += ['第五轮选择：`'+json.dumps(coverage_selection,ensure_ascii=False)+'`','']
        if coverage_replication:lines += ['第五轮训练seed复验：`'+json.dumps(coverage_replication,ensure_ascii=False)+'`','']
    diagnostic_path=root/'transfer_diagnostic/summary.json'
    if diagnostic_path.exists():
        diagnostic=json.loads(diagnostic_path.read_text())
        lines += ['真实渲染输入机制诊断：使用开发集已保存的初始状态、三相机真实RGB、原动作noise及两份新noise，比较首步速度扰动与完整10步动作扰动。只作为初始状态关联诊断，不替代闭环ASR或证明整段轨迹因果机制。', '',
            '| 纹理 | 首步14维 MSE中位数 | 完整动作14维 MSE中位数 | 丢弃维度速度扰动能量比例中位数 |',
            '|---|---:|---:|---:|']
        for row in diagnostic['rows']:
            values=row['median_over_seeds']
            lines.append(f'| {row["condition"]} | {values["first14_mse"]:.6g} | {values["final14_mse"]:.6g} | {values["discarded_velocity_energy_fraction"]:.1%} |')
        lines+=['']
    lines += [f'数据与日志：`{root}`。开发集用于方法选择；测试集只在冻结方法后执行。运行错误独立记录，不计入ASR。未经独立测试通过，不称为稳定攻击。']
    (ROOT/'result/34_robotwin_e_method.md').write_text('\n'.join(lines)+'\n')
    return out
