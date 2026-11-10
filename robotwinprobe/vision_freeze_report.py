"""Explicit stage status and paired clean-conditioned ASR; never invent missing scores."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import checkpoint_study_common as S

def read(path):return json.loads(path.read_text()) if path.exists() else {}
def main(root):
    status=read(root/'pipeline_status.json');train=read(root/'expert/train_manifest.json')
    model_steps=read(root/'protocol.json').get('model_steps',20000)
    restored={a:read(root/'training'/a/'restore_audit.json') for a in ['F','U']}
    verified={a:bool(v.get('passed') and v.get('step')==model_steps) for a,v in restored.items()}
    lines=['# RoboTwin：视觉编码器冻结／解冻配对实验','',
      '[结果总览](README.md) · [原多checkpoint实验](28_robotwin_checkpoint_study.md)','',
      '检验视觉编码器是否参与微调会不会改变patch的跨模型迁移。两份模型使用同一数据、base、模型seed和训练预算；本轮仅一对模型，不能外推为所有seed的普遍结论。', '',
      f'当前阶段：`{status.get("stage","preparing")}`。训练专家轨迹：{len(train.get("selected",[]))}/100；验证轨迹20条。', '']
    if status.get('error'):lines += ['运行错误：`'+status['error']+'`','']
    if model_steps==10000:
      restore_text='两组最终10k检查点均已通过独立恢复验证。' if all(verified.values()) else '最终10k检查点需通过独立恢复验证。'
      lines += ['当前用户指定预算：F/U各10,000步，训练采用不同GPU对并行执行。保持原配对学习率曲线；不据此预先认定已收敛。'+restore_text+'短性能测试仅用于诊断，不直接修改训练算法。','']
    restart=read(root/'fast_u_restart.json')
    if restart:
      lines += [f'用户要求优先缩短结果等待时间：U旧进度 {restart["discarded_step"]} 步已归档至 `{restart["archive"]}`，从同一base/seed重新训练10,000步，全程采用已验证SHM；F原进程保留。F前5000步禁用SHM、U全程SHM，通信数值路径不再完全配对，最终结论须披露该限制。','']
    if status.get('arms'):
      for arm,info in status['arms'].items():
        lines += [f'并行分支 {arm}：`{info.get("stage")}`；GPU `{info.get("gpus")}`；`{info.get("error", "")}`。','']
    speed=read(root/'speed_probe_20261027/suite_status.json')
    shm_validation=read(root/'shm_validation_20261027/validation_status.json')
    if shm_validation:
      lines += [f'共享内存通信完整验证：`{shm_validation.get("stage")}`，节点 `{shm_validation.get("execution_host", "nnmc65")}`，GPU `{shm_validation.get("gpus", "等待空闲卡")}`；独立F/U四步训练、保存恢复及配对检查。','']
      for arm in ['F','U']:
        transition=read(root/'training'/arm/'shm_transition.json')
        if transition:
          detail=('从step0使用已通过独立验证的SHM重训；'+('最终检查点已通过恢复验证' if verified[arm] else '最终检查点仍需恢复验证')) if transition.get('restart_from_base') else f'第 {transition["switch_step"]} 步完整保存后，恢复验证通过才继续使用SHM'
          lines += [f'{arm} 正式通信切换：`{transition.get("stage")}`；{detail}。生产已切换：`{transition.get("production_switched", False)}`。','']
    if speed:
      lines += [f'独立性能短测：`{speed.get("stage")}`，当前项目 `{speed.get("case", "已结束")}`。所有项目都测U组、总batch16；不是F当前每步耗时的直接对照。微批变化会改变增强随机数分组，不能仅凭耗时替换正式训练。','']
      for item in speed.get('results',[]):
        result=item.get('result')
        lines += [f'- `{item["case"]}`：'+(f'预热后中位 {result["median_step_s"]:.2f} 秒/步。' if result else f'未完成，退出码 {item["returncode"]}。')]
      lines += ['']
    if status.get('execution_host'):
      lines += ['执行节点：`'+status['execution_host']+'`；当前工作区保留同步状态与日志。检查训练进程必须在该节点进行，不能以本机PID判断远端进程是否存活。','']
    elif read(root/'execution_location.json').get('host'):
      location=read(root/'execution_location.json')
      lines += ['执行节点：`'+location['host']+'`，GPU `'+','.join(map(str,location['gpus']))+'`。以该节点的实际进程和当前状态文件为准。','']
    cache=read(root/'data_cache/manifest.json')
    if cache.get('complete'):
      lines += [f'正式数据缓存已完成：训练{cache["train_frames"]:,}帧，验证{cache["validation_frames"]:,}帧，共{cache["frames"]:,}帧；归一化统计仅来自训练集。','']
    data_audit=read(root/'data_audit/result.json')
    if data_audit.get('passed'):
      lines += [f'逐轨迹输入检查通过：{data_audit["episodes"]}条轨迹的起始／中间／末尾共{data_audit["checked_frames"]}帧，与上游state、action、三路图像和prompt变换一致；训练／验证轨迹不重叠，动作chunk未跨轨迹。','']
    if status.get('stage')=='waiting_for_gpu_memory':
      lines += ['流水线存活并等待GPU资源，此阶段没有启动模型训练。空闲显存（MiB）：`'+str(status.get('free_MiB',{}))+'`，要求每卡`'+str(status.get('required_each_MiB',34000))+' MiB`。排队及排障ETA未知。旧U预检稳定步约20秒，20k更新约111小时只是粗略计算，不包含排队、编译、保存、patch及闭环测试，也不能直接作为F组或总任务ETA。','']
    probe=read(root/'restore_probe_20261026/results.json')
    if probe:
      diag=read(root/'snapshot_probe_20261026/probe_status.json')
      lines += ['2026-10-26核查：解冻组四步预检训练及保存完成，但恢复固定loss检查失败。原训练监控train=0.186102、validation=0.202680；首次恢复为0.173916／0.201179，分别约6.55%／0.74%差异。不能当普通舍入误差直接放行。','',
        '只读诊断确认磁盘EMA与加载后权重逐字节一致，恢复与重建模型的输出逐位相同。隔离四步快照诊断状态：`'+diag.get('stage','未启动')+'`。正式模型及patch进度见下表，预检不计入正式训练。','']
    forensic=read(root/'snapshot_probe_20261026/restore_check/results.json')
    if forensic:
      lines += ['后续独立恢复检查：完整EMA与保存前指纹一致，输入字节也一致；但原train/validation loss 0.182992／0.203331，恢复后0.174149／0.201397。已排除本次快照的权重或输入字节变化，尚未定位输出差异原因，不能据此放行正式训练。','']
    if read(root/'initialization_v2_check.json').get('passed'):
      lines += ['第二版预检修复：视觉权重在初始化加载及恢复类型定义中均保持float32；原实现先转bf16再转回float32可能造成精度混淆。CPU回归已验证F/U初值相等、恢复类型匹配、仅U更新视觉权重。恢复loss改用每个checkpoint不可覆盖的保存前基准，同时核对全部EMA和输入指纹；新增noise/time与编译IR记录帮助定位差异。旧失败预检完整保留。','']
    schema=read(root/'actual_model_schema_audit.json')
    if schema.get('passed'):
      lines += [f'实际π0.5结构检查通过：两组全部参数形状及类型一致；可训练参数集合的差异恰好为{schema["vision_leaf_count"]}个视觉权重张量，语言及动作模块的冻结范围相同。这是CPU抽象结构检查，不代表GPU预检或训练已完成。','']
    if read(root/'explicit_flow_loss_check.json').get('passed'):
      lines += ['第三版预检：v2已完成4步更新和约20GB checkpoint保存，但恢复loss仍未通过。完整EMA/输入指纹一致；附加诊断发现相同seed的GPU Beta时间步输出不同（最大差0.345662），高斯噪声输出一致、编译IR一致。显式时间步重放仍有残差，不能断言唯一原因。v3将N(0,1)噪声和Beta(1.5,1)时间步在CPU按共同seed/step生成，作为真实前向输入传入；不改变flow matching MSE和采样分布，不放宽恢复检查。CPU小模型已对照上游公式、独立梯度和AdamW更新。','']
    v3=read(root/'training/preflight_v3_U/restore_comparison.json')
    if v3:
      lines += ['v3_U已完成4步及保存：噪声、时间步、完整EMA和输入均匹配，但bf16监控均值train 0.243753→0.245155、validation 0.270443→0.270243，严格恢复检查仍未通过。两次独立float32前向诊断状态：`'+read(root/'precision_probe/runner_status.json').get('stage','未启动')+'`。尚未证明剩余差异仅来自精度。','']
    if read(root/'precision_probe/comparison.json').get('all_identical'):
      lines += ['两次独立进程加载同一checkpoint，显式输入＋float32/最高矩阵乘精度的训练、验证loss向量均逐元素一致。v4据此只将校验前向改为float32，训练仍为bf16，损失和训练预算不变，恢复输出仍要求严格一致；完整更新→保存→恢复验证仍需通过。','']
    bad=read(root/'replica_probe/default.json');fixed=read(root/'replica_probe/host_key.json')
    if bad and fixed:
      lines += ['已定位多卡初始化错误：默认复制路径下，第二张GPU收到的初始化key为[0,0]，应为[3861832030,385742593]；32个复制权重张量中4个LoRA张量在两卡不同。全局CPU快照对重复索引只取第一份，旧EMA hash因此未覆盖第二份错误副本。将相同key字节从CPU分别上传每卡后，32/32副本一致。v4的显式noise/time与FP32仍未修正初始化key，故仍失败。','',
        'v5采用逐卡CPU上传初始化及训练key，并在训练前/保存前检查全部params和EMA副本。保存/恢复仍要求权重、输入、noise/time及副本字节一致；输出改为逐元素FP32舍入精度检查（rtol=1e-6、atol=1e-7），覆盖已观察到约1e-7的舍入差，旧约1%的错误无法通过。该数值检查已测试拒绝局部0.005错误及NaN。完整v5预检进度如下。','']

    if read(root/'functional_step_check.json').get('passed') and read(root/'representation_interface_check.json').get('passed'):
      lines += ['接口预检已通过：独立CPU小模型核对微批梯度与AdamW；已有smoke模型的一帧真实渲染输入核对三路视觉输出及各层KV，clean重复差为零。它们不代表新模型训练或攻击成功。','']
    lines += ['| 设置 | F：冻结视觉 | U：解冻视觉 |','|---|---|---|',
      '| 起点／模型seed | 官方π0.5 base／202610230 | 相同 |',
      '| 任务／数据 | place_container_plate/demo_clean，100训练＋20验证 | 相同 |',
      '| 视觉编码器 | 不更新，保持float32存储 | 更新，float32存储 |',
      '| 语言／action expert | 语言LoRA rank16／expert全量 | 相同 |',
      f'| 预算／执行 | batch16、micro2、两卡FSDP、{model_steps:,}更新 | 相同 |',
      '| LR／优化器／EMA | warmup500、2.5e-5→2.5e-6；AdamW；EMA .99 | 相同 |',
      '| patch | 10cm、Q128、灰初始化、3配对seed、500×8 | 相同 |',
      '| render／EOT | 共享代理bank、3cm位移、±15°旋转、三光照 | 相同 |','',
      '相同seed逐步决定相同抽帧和噪声。采用局部函数式微批梯度累积，冻结权重EMA保持不变。两组均保留float32视觉权重，避免把冻结引入的存储精度差异混入对照。四步预检检查初始权重／输出配对、视觉权重更新、有限且合理的loss／梯度、保存恢复；失败自动停止，不接用失败优化器状态。','',
      ]
    lines += ['v5 U已完成4步更新，保存前params与EMA的全部复制副本检查通过；恢复检查在CPU NumPy数组上调用GPU分片接口而退出，尚未比较恢复输出。v6增加恢复后按原FSDP布局显式放置完整state，再执行副本和输出验证；原v5失败记录保留，验证阈值不变。','']
    if read(root/'preflight_v6_paired_audit.json').get('passed'):
      lines += ['v6两组4步预检、保存恢复和配对检查均已通过。F/U各自保存前后训练、验证loss向量逐元素完全一致（max_abs=0）；权重、输入和副本检查通过。正式训练状态见下表。','']
    lines += ['| 模型 | 状态 | 步数 |','|---|---|---:|']
    for name in ['smoke_U','preflight_v2_U','preflight_v3_U','preflight_v4_U','preflight_v5_U','preflight_v5_F','preflight_v6_U','preflight_v6_F','F','U']:
      d=read(root/'training'/name/'status.json');saved=read(root/'training'/name/'latest_checkpoint.json')
      step=d.get('step',saved.get('step','—'))
      history=root/'training'/name/'history.jsonl'
      if d.get('stage')=='training' and history.exists():
        for row in reversed(history.read_text().splitlines()):
          try:step=json.loads(row)['step'];break
          except (ValueError,KeyError):continue
      lines.append(f'| {name} | {d.get("stage","未启动")} | {step} |')
    lines += ['','导出后先检查JAX/Torch一致性；每模型10个独立初态的clean预检至少成功5个，才进入patch训练。之后六张patch在两模型上全部交叉，另测clean和同一张Random：100个共同、未筛选测试初态，共1,600条正式闭环。','',
      '主指标是目标模型clean成功条件下的ASR，并同时报告相对Random增量、两模型共同clean成功初态上的ASR。SR以所有100初态为分母。每个条件使用同一初态和动作noise规则；运行错误不算任务失败。在同一真实渲染状态上，另测三相机视觉编码器输出、各层KV相对clean的差、L0和完整flow动作差；这些不能替代ASR。','',
      '## 闭环结果','']
    rows=[];clean_sets={}
    for m in [f'F_s{model_steps}',f'U_s{model_steps}']:
      d=read(root/'closedloop'/m/'clean/results.json')
      if d.get('complete'):clean_sets[m]={e['seed'] for e in d['episodes'] if e['success']}
    common=set.intersection(*clean_sets.values()) if len(clean_sets)==2 else None
    for model,clean in clean_sets.items():
      base=read(root/'closedloop'/model/'Random/results.json')
      rand={e['seed']:e['success'] for e in base.get('episodes',[])} if base.get('complete') else {}
      for path in sorted((root/'closedloop'/model).glob('*/results.json')):
        d=read(path)
        if not d.get('complete'):continue
        ep={e['seed']:e['success'] for e in d['episodes']};eligible=sorted(clean)
        assert clean.issubset(ep),'Unmatched episode set'
        failures=sum(not ep[s] for s in eligible)
        row=dict(model=model,condition=path.parent.name,N=len(ep),task_successes=sum(ep.values()),
            clean_successes=len(clean),attack_failures=failures,ASR=failures/len(clean) if clean else None,
            common_clean_ASR=sum(not ep[s] for s in common)/len(common) if common else None,
            delta_random=(sum((not ep[s])-(not rand[s]) for s in eligible)/len(clean)) if rand and clean else None)
        if eligible:
          rng=np.random.default_rng(202610251)
          picks=rng.integers(0,len(eligible),(10000,len(eligible)))
          fail=np.asarray([not ep[k] for k in eligible],float)
          row['ASR_CI95_low'],row['ASR_CI95_high']=map(float,np.quantile(fail[picks].mean(1),[.025,.975]))
          if rand:
            delta=np.asarray([(not ep[k])-(not rand[k]) for k in eligible],float)
            row['delta_CI95_low'],row['delta_CI95_high']=map(float,np.quantile(delta[picks].mean(1),[.025,.975]))
          else:row['delta_CI95_low']=row['delta_CI95_high']=None
        else:
          for key in ['ASR_CI95_low','ASR_CI95_high','delta_CI95_low','delta_CI95_high']:row[key]=None
        rows.append(row)
    if rows:
      lines += ['| 模型 | 条件 | SR | ASR | 相对Random | 共同clean ASR |','|---|---|---:|---:|---:|---:|']
      fmt=lambda x:'—' if x is None else f'{100*x:.1f}%'
      for d in rows:lines.append(f'| {d["model"]} | {d["condition"]} | {d["task_successes"]}/{d["N"]} | {fmt(d["ASR"])} | {fmt(d["delta_random"])} | {fmt(d["common_clean_ASR"])} |')
      with (root/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    else:lines+=['尚无正式闭环结果。']
    textures=read(root/'texture_manifest.json')
    if textures:
      import os
      os.environ.setdefault('MPLCONFIGDIR','/tmp/vision_freeze_mpl')
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
      from PIL import Image
      learned=[t for t in textures if t['id']!='Random']
      fig,axes=plt.subplots(2,3,figsize=(9,6),constrained_layout=True)
      for ax,item in zip(axes.ravel(),learned):
        ax.imshow(Image.open(item['path']),interpolation='nearest');ax.set_title(item['id']);ax.axis('off')
      out=root/'figures';out.mkdir(exist_ok=True)
      fig.savefig(out/'patch_comparison.png',dpi=150);plt.close(fig)
      lines+=['','![六张配对patch](../'+str((out/'patch_comparison.png').relative_to(S.ROOT))+')','']
    lines += ['','跨模型比较基于同一目标模型内的patch响应；不直接对两个模型不同坐标系的latent做逐元素差。即使解冻组patch迁移更强，也只支持视觉微调对这一配对实验的影响；“共同结构已被分离且足以攻击”仍需后续替换／保留干预验证。','',
      f'运行目录：`{root}`。入口：`robotwinprobe/vision_freeze_pipeline.py`；训练：`robotwinprobe/vision_freeze_ft.py`。原实验文件与失败记录不改写。','']
    (S.ROOT/'result/30_robotwin_vision_freeze.md').write_text('\n'.join(lines))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
