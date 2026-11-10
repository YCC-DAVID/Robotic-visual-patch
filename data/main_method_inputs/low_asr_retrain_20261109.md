# 低ASR任务：位置、可见性、条件审计与独立重训

用户要求：原主实验中ASR低于50%的任务重新训练，检查位置/遮挡和条件是否统一。范围为5个LIBERO任务（柜顶、酒瓶、开炉、碗放炉、抽屉）与RoboTwin place_container_plate。原240回合及其ASR完整保留，重训另存目录，不以新结果替换或隐藏原始结果。

## 已查清的证据

七项LIBERO在原位置的训练帧中，主相机均100%达到300px；没有双相机同时完全不可见的帧。腕相机可见性差异显著：

| 任务 | 腕视角零像素帧 | 腕视角至少100px帧 | 原位置 |
|---|---:|---:|---|
| 碗放盘子 | 7.1% | 92.9% | [0.09, 0.15] |
| 碗放柜顶 | 10.9% | 86.8% | [0.09, 0.15] |
| 酒瓶上架 | 77.3% | 17.7% | [-0.09, 0.24] |
| 打开炉子 | 37.3% | 56.5% | [-0.33, 0.03] |
| 碗放炉子 | 27.2% | 71.1% | [-0.09, 0.24] |
| 奶酪放碗 | 0.0% | 100.0% | [0.09, 0.15] |
| 打开中间抽屉 | 42.0% | 30.1% | [-0.03, 0.24] |

基于每帧homography重建的无遮挡投影，旧位置的腕视角零像素主要来自出画/镜头后方，未发现投影本应至少50px却完全被遮住的旧位置零帧。这个结论区分了视角与遮挡，并非声称每一帧都无遮挡。柜顶腕视角覆盖接近强任务，其低ASR不能仅用位置解释。

重新只按训练初态0–9的扫描数据排序：要求主相机稳定可见，依次最大化最差回合腕覆盖、回合宏平均覆盖、低分位和截断面积。只给抽屉换位置：候选90 `[-.03,.24]` → 候选238 `[.24,-.12]`；腕≥100px总帧覆盖30.1%→47.8%，回合宏平均27.9%→56.8%，主相机平均908→1400px。新位置仍有少量真实局部遮挡，未宣称完全可见。其他四弱任务没有更稳合法候选，保留原位，避免无证据换点。

RoboTwin原训练49个状态全属approach（query0:24、query1:24、query2:1），原采集轨迹有152个后续阶段query未纳入训练。method的65次测试推理中head相机始终看到patch，左右腕分别27/65、28/65次为0。全49训练帧随机纹理内部RGB RMSE最大约head .0703、left .0679、right .0534；代理误差高于LIBERO .02门禁。像素中心offset .5与0用9个训练状态random/checker实测，0全部更差，保留.5，不把未证实猜想改成修复。

## 条件与优化实现核对

七份LIBERO条件、500×8 batch序列、noise、EOT、prompt、geometry、PNG及源码hash全部复核匹配；未发现漏EOT、反向loss、训练漏跑或整体饱和。各任务训练diag和验证L0均上升，强任务验证L0约2.93/4.66，五弱约.136–.661。

已完成一张L40S上的短诊断：每任务训练/验证中位帧各1、四个noise（固定训练noise＋3独立draw），共224条件行、56组native/proxy比较、7组梯度。trained变化94–99%能量在实际7维，padding未主导；独立noise下实际执行窗口SSE为固定noise的.916–1.121倍，未见明显单noise过拟合。五弱full/执行窗口梯度cos：柜顶.988、酒瓶.909、开炉.954、碗放炉.502、抽屉.701，均有限、冻结模型参数grad0。随机纹理native/proxy速度方向cos中位.955–.995，相对SSE误差2.7–9.5%。据此保留原完整首步loss，避免同时更换目标。

相同公共配方：冻结π0.5 checkpoint、13cm、128×128、灰初始化、EOT范围、500×8、sign0.01、固定500终点、真实native相机输入、三条件配对评测。benchmark固有的相机数、动作维度/horizon/replan继续使用各自官方接口。新LIBERO作为独立seed重训（batch20261109、EOT2026110901）；RoboTwin保持原seed，主要改变训练阶段覆盖与训练/验证划分，逐项来源写进各自协议。不存在“跨benchmark相机/动作接口必须强行相同”的改动。

## 已启动的重训流程

LIBERO：新目录 `pi05probe/out/low_asr_retrain_v1`，5项全部重新优化纹理；四项同位置精确复用已核验标定，抽屉新位置重新渲染标定，clean轨迹/state/noise保持与旧池相同。新增权重、config及norm_stats明确SHA；源数据与EOT/训练源码hash冻结；代理RGB .02门禁保留。每项训练后立即clean/random/EOT各10回合并用原完整验证器检查，失败/缺回合不计最终ASR。

RoboTwin：新目录 `robotwinprobe/out/low_asr_retry_20261108`，同24个旧训练范围seed重跑完整native32查询轨迹至成功/400步，20个训练seed25000–25019与4个验证25020–25023；不按成功筛选。训练batch每阶段2帧，先均匀seed再均匀query；验证数据不进batch。缺任何阶段明确报错，保留各阶段初/最终train/val代理L0与fidelity诊断；固定原13cm位置、原完整loss及500终点，再做原26000–26009三条件30回合。

11月8日22:13 PST启动CPU协调器3891534，nnmc62 GPU5 UUID `GPU-d2b438d6-a3f4-ce04-95e3-2b61b295628a`，原生EGL6。GPU2被其他活跃项目任务重新占用，未分配；当前一张卡顺序执行RoboTwin，再LIBERO五项。每个新GPU阶段全fleet＋EGL UUID＋所选节点立即复查，显式CVD。遇资源占用会等待并重新完整预检，未知/坏卡/占用卡不计算，不停止任何现有任务；服务随客户端会话自然结束。

CPU代码复核已实际通过四个原位任务的validate_calibration：旧轨迹状态、图像、噪声和train/validation划分不变；新协议/selection哈希、5任务范围和三条件配对检查通过。GPU上Robo新scene检查通过、服务显式noise重复推理通过，已完成seed25000第一个native32 query采集。抽屉新位置仍待实际新标定验证。

目前尚无重训后ASR。两个原测试集已暴露，本轮明确标为纠正性复测；位置只从训练数据选，不能将反复复测同一测试集描述为全新的独立确认实验。

## 记录入口

- [CPU条件审计](../out/main_results/low_asr_audit/conditions.json)
- [位置/出画审计](../out/main_results/low_asr_audit/visibility/audit.json)、[确定的新位置](../out/main_results/low_asr_audit/visibility/selection.json)
- [GPU短诊断](../out/main_results/low_asr_audit/velocity_gpu/diagnostic.json)、[分解汇总](../out/main_results/low_asr_audit/velocity_gpu/summary.json)
- [RoboTwin阶段审计](../robotwinprobe/out/low_asr_retry_20261108/audit.json)、[offset反事实核验](../robotwinprobe/out/low_asr_retry_20261108/audit_offset.json)
- [重训总进度](../out/main_results/low_asr_retrain/progress.json)
- [LIBERO新协议](../pi05probe/out/low_asr_retrain_v1/protocol.json)、[RoboTwin新协议](../robotwinprobe/out/low_asr_retry_20261108/protocol.json)
