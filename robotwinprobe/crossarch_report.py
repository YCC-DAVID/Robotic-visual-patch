"""Live status and paired, seed-based cross-architecture results; no imputed scores."""
import csv
import json
from pathlib import Path
import numpy as np
from crossarch_common import DEFAULT,ROOT,STAGES,prepare,dump

def report(root=DEFAULT):
    p=prepare(root);status=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {'stage':'preflight'}
    branch=root/'stage_specialists_20261026'
    branch_status=json.loads((branch/'status.json').read_text()) if (branch/'status.json').exists() else None
    text=['# RoboTwin π0.5 / ImageWAM：单模型与联合 patch','',
        '任务固定为 `place_container_plate / demo_clean`。直接冻结两份公开 RoboTwin 策略权重，分别训练 π0.5 patch、ImageWAM patch，以及一张共享的联合 patch。此实验不需要等待原第28号实验的策略微调。','',
        (f'当前转为分阶段方案：**{branch_status["stage"]}**。π0.5与ImageWAM各4张，测试全程与仅对应阶段；进度和结果见[分阶段报告](32_robotwin_stage_specialists.md)。下方保留原共享方案的停止记录。' if branch_status else f'当前阶段：**{status["stage"]}**。'+(' 异常：`'+status['error']+'`' if status.get('error') else '')),'',
        '| 固定项 | 设置 |','|---|---|',
        '| 纹理 | 10 cm、128×128、同一初始灰图、500次更新、每次8个样本 |',
        '| 采样 | 24条训练、8条独立验证专家轨迹；按接近、抓取、搬运、放置分段，每批每阶段2帧 |',
        '| render / EOT | 三相机共享同一世界纹理；3 cm位移圆盘、±15°旋转、三档光照；保留真实遮挡 |',
        '| 联合目标 | 两模型有效14维首步速度 MSE，分别除以训练前固定 Random 响应，再等权相加 |',
        '| 推理 | 各自原生预处理、归一化、10步 flow；π0.5动作段32步，ImageWAM 16步 |',
        '| 闭环 | 每模型同一100初态 × clean、Random、两张单模型patch和联合patch；共1000回合 |','',
        '## 要回答的问题','',
        '| patch优化来源 | 测试π0.5 | 测试ImageWAM |','|---|---|---|',
        '| 仅π0.5 | 白盒自身效果 | π0.5 → ImageWAM迁移 |',
        '| 仅ImageWAM | ImageWAM → π0.5迁移 | 白盒自身效果 |',
        '| 两模型联合 | 联合patch在π0.5上的效果 | 同一张联合patch在ImageWAM上的效果 |','',
        '单模型patch的优化和选择不能使用另一目标模型的攻击结果；联合patch明确使用两边梯度，属于双模型白盒优化，其效果不能当作未知模型迁移。两个方向的迁移率无需相同。',
        '', '联合优化最大化 J(Q) = 0.5 × L_pi(Q)/s_pi + 0.5 × L_wam(Q)/s_wam，其中s为优化前仅从训练数据估计并固定的Random响应。冻结策略权重，只更新同一张共享纹理；各模型使用原生预处理。',
        '', '联合有效需要分别报告两边的闭环ASR及相对Random的配对增量，并与两张单模型patch比较；平均损失增大不能保证两边都有效。当前500次共享纹理更新中，联合目标每次使用两个模型，模型梯度计算次数约为单模型的两倍，因此属于更新次数匹配，不是总计算量匹配。',
        '', '公开模型的训练数据、训练配方和执行动作段长度不同，本实验能测量这两份策略之间的迁移，不能把差异全部归因于架构。',
        '', '## 已验证接口','', '| 模型 | 首步速度误差 | 完整动作误差 | 梯度检查 |','|---|---:|---:|---|']
    for m in ('pi','wam'):
        path=root/f'audit/{m}.json'
        if path.exists():
            a=json.loads(path.read_text());text.append(f'| {m} | {a["first_velocity_max_error"]:.6g} | {a["full_action_max_error"]:.6g} | {"通过" if a["passed"] else "未通过"} |')
        else:text.append(f'| {m} | — | — | 待测 |')
    trace=list((root/'trajectories').glob('seed*/complete.json'))
    text+=['',f'完整轨迹阶段标记：{len(trace)}/32。训练标定：{len(list((root/"bank/train").glob("*.npz")))}/1536；验证标定：{len(list((root/"bank/validation").glob("*.npz")))}/128。']
    sample=root/'trajectories/seed20000/stages.png'
    if sample.exists():text+=['','首条轨迹的实际回放阶段（从上到下为接近、抓取、搬运、放置）：','',f'![四阶段回放]({sample})']
    train=root/'training_progress.json'
    if train.exists():text+=['',f'三张 patch 的共同训练进度：{json.loads(train.read_text())["iteration"]}/500。']
    else:text+=['','patch 优化尚未开始；先完成 clean 能力及代理检验。']
    text+=['','## Clean 预检与正式闭环','', '| 模型 | clean预检 | 正式clean SR | Random ASR | π patch ASR | WAM patch ASR | 联合patch ASR |','|---|---|---|---|---|---|']
    allrows={};summary=[]
    for m in ('pi','wam'):
        pilot=[json.loads(f.read_text()) for f in (root/'pilot'/m/'clean').glob('seed*.json')]
        pilot_ok=[r for r in pilot if 'success'in r]
        conditions={c:{r['seed']:r for f in (root/'closedloop'/m/c).glob('seed*.json') if 'success' in (r:=json.loads(f.read_text()))} for c in p['evaluation']['conditions']}
        allrows[m]=conditions;clean=conditions['clean']
        line=[m,f'{sum(r["success"] for r in pilot_ok)}/{len(pilot_ok)}' if pilot_ok else '待测',
              f'{sum(r["success"] for r in clean.values())}/{len(clean)}' if clean else '待测']
        for c in p['evaluation']['conditions'][1:]:
            paired=sorted(set(clean)&set(conditions[c]));eligible=[s for s in paired if clean[s]['success']]
            count=sum(not conditions[c][s]['success'] for s in eligible)
            line.append(f'{count}/{len(eligible)}（配对{len(paired)}/100）' if paired else '待测')
            if paired:summary.append(dict(model=m,condition=c,paired=len(paired),eligible=len(eligible),attack_failures=count,
                ASR=count/len(eligible) if eligible else None,SR=np.mean([r['success'] for r in conditions[c].values()]).item()))
        text.append('| '+' | '.join(line)+' |')
    # Paired bootstrap over initial scenes; target-specific denominator stays explicit.
    transfer=[];rng=np.random.default_rng(20261025)
    common=set(p['evaluation']['seeds'])
    for m in allrows:common&={s for s,r in allrows[m]['clean'].items() if r['success']}
    for m,conditions in allrows.items():
        clean=conditions['clean'];random=conditions['random']
        for c in ('pi','wam','joint'):
            paired=sorted(set(clean)&set(random)&set(conditions[c]))
            eligible=[s for s in paired if clean[s]['success']]
            if not eligible:continue
            a=np.array([not conditions[c][s]['success'] for s in eligible],float)
            r=np.array([not random[s]['success'] for s in eligible],float)
            draws=rng.integers(len(eligible),size=(2000,len(eligible)))
            shared=sorted(common&set(conditions[c]))
            transfer.append(dict(model=m,condition=c,paired_count=len(paired),eligible=len(eligible),
                ASR=float(a.mean()),ASR_CI95=np.quantile(a[draws].mean(1),[.025,.975]).tolist(),
                excess_over_Random=float((a-r).mean()),excess_CI95=np.quantile((a-r)[draws].mean(1),[.025,.975]).tolist(),
                common_clean_count=len(shared),common_clean_ASR=float(np.mean([not conditions[c][s]['success'] for s in shared])) if shared else None))
    dump(root/'closedloop_summary.json',dict(conditions=summary,paired_transfer=transfer))
    gate_path=root/'audit/clean_pilot.json'
    if gate_path.exists():
        gate=json.loads(gate_path.read_text())
        text+=['',f'Clean预检门槛：每模型至少{gate["gate_minimum"]}/{gate["total_per_model"]}；记录结果：'+('通过。' if gate['passed'] else '未通过。')]
        if not gate['passed']:
            review_path=root/'audit/clean_gate_review.json'
            if review_path.exists() and json.loads(review_path.read_text()).get('accepted'):
                text+=['','已完成诊断复核并恢复：旧10200–10209完整复跑8/10，全部状态、动作及保存的头部图像与历史逐值一致，且当前权重哈希匹配公开RoboTwin权重。用户要求继续训练；原5/10、8/10预检记录和6/10诊断门槛均保留，不改正式测试初态，真实render代理保真检验继续执行。复核记录：`audit/clean_gate_review.json`。']
            else:
                text+=['','当前流水线在此停止，尚无patch攻击结论。该门槛是实验脚本的诊断设置；π0.5的5/10本身不能证明接口错误。后续先核查失败回放和更大固定clean批次；若修订门槛或实验设计，应保留原记录、明确修订理由，不能筛选成功初态替换原分母。']
    text+=['','ASR = clean成功而受攻击失败 / clean成功；整体SR = 任务成功数 / 固定初态数。运行错误单列，不计入任务失败；正式结果完成前显示已完成的分母。最终同时给出两模型共同clean成功子集和相对Random的配对差值。',
        '', '## L0 与解释范围','',
        'L0在未参与优化的整段专家轨迹上测量，clean与各纹理固定状态、相机、动作噪声。分别汇总四阶段和阶段等权平均；保存有效14维的MSE/SSE以及各模型完整维度结果。完整动作差经过各自全部10步flow。跨模型原始L0量纲不同，不能直接按数值大小比较。',
        '', '全程均有patch、四阶段L0均升高，也不足以证明每个阶段都独立导致任务失败。要判断持续干扰的因果作用，还需后续在不同阶段才引入patch的闭环对照。',
        '', '复现入口：`robotwinprobe/crossarch_pipeline.py`。独立原始目录：`robotwinprobe/out/crossarch_joint_20261025/`。`protocol.json`固定数据/设置；`status.json`记录执行阶段；`logs/`保存各进程日志；`audit/`记录原生推理和真实render代理检查。']
    if transfer:
        text+=['','## 配对迁移统计','', '| 目标 | patch | ASR及95%区间 | 超过Random的ASR | 两模型共同clean成功ASR |','|---|---|---|---|---|']
        for r in transfer:
            lo,hi=r['ASR_CI95'];shared='—' if r['common_clean_ASR'] is None else f'{r["common_clean_ASR"]:.1%} / n={r["common_clean_count"]}'
            text.append(f'| {r["model"]} | {r["condition"]} | {r["ASR"]:.1%} [{lo:.1%}, {hi:.1%}] | {r["excess_over_Random"]:+.1%} | {shared} |')
    for m in ('pi','wam'):
        path=root/f'diagnostics/{m}_summary.json'
        if path.exists():
            text+=['',f'## {m}：留出轨迹的真实render阶段诊断','', '| patch | 阶段 | L0 MSE（14维） | 完整动作MSE（14维） |','|---|---|---:|---:|']
            for r in json.loads(path.read_text()):text.append(f'| {r["condition"]} | {r["stage"]} | {r["L0_mse14"]:.6g} | {r["action_mse14"]:.6g} |')
    diagnosis=root/'audit/proxy_diagnosis_20261026/summary.json'
    if diagnosis.exists():
        rows=json.loads(diagnosis.read_text())['rows']
        text+=['','## π0.5代理检验失败诊断','',
            '这是训练前的成像代理保真检查：固定真实状态和动作noise，比较代理图像与真实render引起的首步速度场变化。方向余弦不是任务成功率。独立诊断完整复现原检验的32组纹理对照，未修改阈值或正式代理，未重启流水线。',
            '', '| 阶段 | 方向余弦中位数 | L0相对误差中位数 |','|---|---:|---:|']
        for r in rows:
            if r['variant']=='baseline':
                text.append(f'| {r["stage"]} | {r["median_cosine"]:.4f} | {r["median_relative_L0_error"]:.1%} |')
        text+=['','抓取段起点样本的方向余弦为0.6313，整体0.7957接近0.8不能解释为各阶段都只差一点。扰动强度接近也不能保证方向一致。',
            '', '探索性对照：移除半像素偏移、保留贴纸外clean背景、使用重复纹理边界及其组合，均未提高整体方向余弦。尚未定位唯一代码错误；剩余成像误差、颜色响应、过滤、遮挡和间接光照近似仍需检查。',
            '', '检验子集实际为前两条验证轨迹、每阶段首帧的两个EOT条件，共8个独立回放状态、16个渲染条件、32组纹理对照；没有覆盖全部128条验证标定。',
            '', f'[详细诊断]({diagnosis}) · [解释与边界]({diagnosis.parent/"interpretation.json"})']
    camera_diagnosis=root/'audit/camera_diagnosis_20261026/summary.json'
    if camera_diagnosis.exists():
        camera=json.loads(camera_diagnosis.read_text())
        text+=['','## 抓取段起点的相机拆分诊断','',
            '固定前两条验证轨迹的抓取段起点、两个EOT条件、Random/Checker及三份共同noise。下表将指定相机的代理图像替换为真实render，其余相机仍使用代理；这是离线输入反事实，不是物理闭环。',
            '', '| 替换为真实render的相机 | 方向余弦中位数 | 剩余速度误差能量／原代理误差 |',
            '|---|---:|---:|']
        for r in camera['rows']:
            if r['family']=='repair_proxy' and r['actual_camera_mask'] in (0,1,2,4):
                name='无（全代理）' if not r['actual_cameras'] else '、'.join(r['actual_cameras'])
                text.append(f'| {name} | {r["cosine_to_real_delta_median"]:.4f} | {r["residual_error_fraction_median"]:.1%} |')
        text+=['','头相机在这四个渲染条件中可见约1055–1086个贴纸像素；左腕可见0或约2039–2113像素；右腕均不可见。头相机和可见贴纸时的左腕都有局部纹理误差，但单独修复任一路均未恢复方向一致性，不能将失败归结为某一路腕相机。',
            '', '原检验中抓取起点的实际patch首步速度响应RMS中位数约0.00249，接近起点约0.00540；抓取起点的代理误差RMS约0.00222，因而在较弱响应中占比较大。抓取接触可能对位置/时机敏感是机制假设，当前没有阶段限定闭环ASR，不能据此量化抓取失败概率。',
            '', f'[相机诊断数据]({camera_diagnosis}) · [seed21000相机画面]({camera_diagnosis.parent/"s21000_f0061_d0_camera_comparison.png"}) · [seed21002相机画面]({camera_diagnosis.parent/"s21002_f0056_d0_camera_comparison.png"})']
        repeat_path=camera_diagnosis.parent/'repeat_control.json'
        if repeat_path.exists():
            repeat=json.loads(repeat_path.read_text())
            if all(r['max_repeated_velocity_error']==0 and r['max_repeated_action_error']==0 for r in repeat['rows']):
                text+=['','重复推理对照：seed21000抓取起点、Checker、同noise42下，clean／真实render／代理输入各重复三次，首步速度和完整动作均逐值一致；该对照未发现运行随机波动。']
    if (root/'diagnostics/textures.png').exists():
        text+=['','## 纹理与频谱','',f'![纹理]({root/"diagnostics/textures.png"})','',f'![频谱]({root/"diagnostics/spectra.png"})',
            '', '频谱横轴为纹理空间频率（cycles/texel），纵轴为去各通道均值、加Hann窗后的径向平均功率。颜色或频谱相近不代表已确认共同攻击结构；需结合双向迁移与后续因果干预。公开模型的数据和训练配方不同，本轮也不能把全部迁移差异仅归因于架构。']
    regression=root/'clean_regression_10200_10209'
    if (regression/'comparison.json').exists():
        comparison=json.loads((regression/'comparison.json').read_text())
        text+=['','## 历史10初态 clean 复跑','',
            '独立复跑10200–10209，固定历史任务、提示词规则、32步动作段和动作noise规则；不并入正式跨架构攻击统计。',
            '',f'历史成功{comparison["historical_successes"]}/10；当前复跑完成{comparison["current_completed"]}/10，已完成部分成功{comparison["current_successes"]}。']
        if (regression/'weight_identity.json').exists():
            text+=['','当前本地π0.5权重的SHA256已与公开仓库 `motus-robotics/pi0.5_robotwin2` 固定版本 `effc6e1a99bd4d0edf9447e8ba313def67649cfe` 的LFS元数据匹配，确认为RoboTwin微调权重。历史记录指向同一来源和路径，但未保存当时权重文件的哈希。',
                '',f'[权重身份核验]({regression/"weight_identity.json"}) · [逐集比较]({regression/"comparison.json"})']
        completed=[r for r in comparison['rows'] if r['rerun_complete']]
        sequence_checked=[r for r in completed if 'full_action_sequence_equal' in r]
        if sequence_checked:
            matched=sum(all(r[k] for k in ('full_state_sequence_equal','full_action_sequence_equal','full_head_sequence_equal')) for r in sequence_checked)
            text+=['',f'整段记录核对：{matched}/{len(sequence_checked)}集的关节状态序列、动作序列及保存的头部相机帧与历史逐值一致。']
            if comparison['complete'] and matched==10 and all(r['success_matches'] and r['initial_state_and_rgb_match'] and r['prompt_matches'] for r in completed):
                text+=['','旧批次8/10已完整复现，未发现本轮接口相对历史的行为差异。新批次23000–23009的5/10不能据此解释为换错权重或模型退化；初态及其对应动作noise不同，需在更大固定批次上估计clean能力。']
        if completed:
            text+=['','| seed | 历史成功 | 当前成功 | 初始状态及三路RGB一致 | 提示词一致 | 首动作段最大绝对误差 |',
                '|---|---|---|---|---|---:|']
            for r in completed:
                text.append(f'| {r["seed"]} | {r["historical_success"]} | {r["current_success"]} | {r["initial_state_and_rgb_match"]} | {r["prompt_matches"]} | {r["first_action_max_abs_error"]:.6g} |')
    (ROOT/'result/31_robotwin_crossarch_joint.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':report()
