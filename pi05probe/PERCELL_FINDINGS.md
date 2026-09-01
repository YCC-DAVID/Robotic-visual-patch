# PERCELL_FINDINGS.md — per-patch(token 网格级)influence vs gradient vs attention

2026-08-24 · 本会话(wamattack-9f / ea73273f)。承接主力会话(dfbf6bdc)在 g0i 反投影后被
cyber 安全策略中断的收尾实验:**把 influence(ground truth)、gradient、attention 放到模型
原生 token 网格上逐格公平比较**,以判定"哪个量能作为廉价代理指导对抗 patch 选点"。

不动主力会话的 [`CONCLUSIONS.md`](CONCLUSIONS.md) / [`RESULTS.md`](RESULTS.md),本文单列 per-cell 这一轮。

---

## 为什么要做这个(它解决了什么老问题)

旧的 3×3 图(61 个近物体锚点)上,π0.5 的 gradient(+0.94/+0.73/+0.89)和 attention
(+0.82/+0.80/+0.80)对 influence 的秩相关**打平**,得不出"gradient 比 attention 更好"。
那个打平是三重混淆撑出来的(见 CONCLUSIONS §A):
1. **池子太小**:61 锚点全挤在物体旁一小圈 L 形,"离物体越近越强"一个趋势带动全部;
2. **attention 粒度假**:10cm 贴纸脚印跨 5 个 token 格,61 锚点去重只占 18 格、覆盖矩阵秩 41;
3. **attention 可挑层**:单层秩相关跨 −0.52~+0.95,+0.80 是手挑中层平均出来的。

**本轮一次性拆掉三者**:铺满整张桌的 token 网格、每格一张≈格尺寸的小 patch、attention 放回
原生网格且加不挑层的 rollout。

---

## 方法

- **地基:相机反投影**。token 格中心 → 桌面 z=0.9 世界坐标。π0.5 16×16(`g0i_project.py`,
  code=(0,1,0),world→格精确 45/61、±1 格 61/61);FastWAM 7×7(`fw_project.py`,同 code=(0,1,0),
  拿 78 锚点渲染真值校准,精确 77/78、±1 格 78/78)。相机/桌面两模型逐字相同。
- **per-cell 扫描**:每个桌面格贴一张 patch(π0.5 6cm≈格间距 5.7cm;FastWAM 13cm≈格间距 13.4cm),
  keepout 只**打标**不剔除(填满网格当 GT)。固定 ε/seed ⇒ 逐位确定(红线全为 0)。
  π0.5:`percell_dump.py`→`s2_scan_actions.py`(176 格×T 帧);FastWAM:`fw_scan.py --cells --patch-m`
  (35 格×10 帧)。
- **influence 标量**(GT):执行前缀先求和→取模→按帧求和 ×50(与 3×3 图同口径,对应 +0.94 那套)。
- **gradient**:帧0 逐像素 |∇action/∂pixel|(已有 `g0_grad_f0_*` / `fw_grad_f0`),按每格 patch 脚印求和。
- **attention**:π0.5 两套——现有法(destination 词×base256、head 求和、中层 L4-12 平均)+ rollout
  (逐层 0.5A+0.5I 复合、不挑层);FastWAM 一套(反向 cross-attn video×text,destination 名词列,
  rollout 不适用)。

---

## 结果一:π0.5 三任务(`fig_percell_pi05_3task.png`)

对 influence 的 Spearman:

| | 全网格(176) | | | 合法/可攻格(77) | | |
|---|---|---|---|---|---|---|
| 任务 | 梯度 | attn现有法 | attn rollout | 梯度 | attn现有法 | attn rollout |
| plate | +0.83 | +0.75 | +0.69 | **+0.66** | +0.32 | +0.12 |
| cabinet | +0.72 | +0.69 | +0.67 | **+0.37** | +0.12 | +0.03 |
| rack | +0.56 | +0.62 | +0.61 | **+0.41** | +0.27 | +0.23 |

- **全网格上三者接近**(rack 上 attention 甚至略高)——整张桌含"离物体近=高"这个谁都能抓的大趋势。
- **只看合法/可攻格,梯度三任务全面压过两套 attention**。旧的"没区别开"确认是近物体小池的混淆。
- **rollout 三任务 argmax 全落同一格 (12,1)** ——任务无关的结构性 sink,根本没追目的地。

## 结果二:跨模型 FastWAM(`fig_fw_percell_plate.png`,plate,7×7,35 格)

| 模型 | 网格 | 梯度↔FD(合法格) | attention↔FD(合法格) |
|---|---|---|---|
| π0.5 | 16×16 | +0.66 | +0.32 |
| FastWAM | 7×7 | **+0.72** | **−0.50** |

- 跨模型复制且**更强**:FastWAM 合法格上 attention **反相关**。
- 边界:FastWAM 合法格仅 9 个,Spearman 区间宽,−0.50 有噪声,但方向与 π0.5 一致。

## 结果三:FastWAM 未来预测头 Δfuture 探针(`fw_future_probe.py`)

敏感度 = Δ / 各自 seed 地板(SNR);红线 |Δaction|=|Δvideo|=0。

| 位置 | Δaction/地板 | Δfuture/地板 |
|---|---|---|
| hot(influence 最强格) | 2.55 | 1.10 |
| far_legal(远处低-influence 合法格) | 0.47 | 0.49 |

- **未来预测通路可攻**:hot 格 Δfuture/地板 1.10(后段 1.24),过自身地板 ⇒ 方案2 值得做;
- **但 action 头更敏感**(2.55 > 1.10),先攻 action 头性价比更高,未来头是第二通路;
- **两头都有空间选择性**(hot ≫ far_legal)。

**action 头 WAM÷π0.5 相对敏感度**(苹果对苹果,各自地板归一):FastWAM 最强格 SNR **2.55** vs
π0.5 **1.11** ⇒ **FastWAM action 头对 patch 约 2.3× 更敏感**。未来头无 π0.5 对照物(π0.5 没有该通路)。

---

## 总结论

**gradient 是无旋钮、跨模型稳定、跨任务稳定的 influence 廉价代理,可用来指导对抗 patch 选点;
attention 不行**(粒度假 + 可挑层 + rollout 是结构 sink;FastWAM 上甚至反相关)。
FastWAM 的未来预测头对 patch 敏感、可作第二攻击通路,但弱于 action 头;FastWAM action 头整体比
π0.5 敏感 ~2.3×。

## 诚实边界

- **单-ε 下 per-cell 绝对显著性弱**:ε 地板随轨迹变长而升(plate 231mm→cabinet 334→rack 441mm),
  过地板格 plate 5/176、cabinet 3/176、rack **0/176**。读的是**排序**不是单格幅度;要坐实单格得多-ε 平均。
- **argmax 格多在物体上**(occlusion 混淆、legal=False),rack/cabinet 上各量 argmax 不重合 ⇒
  "argmax 一致"只在 plate 成立,不是稳健论点。
- 未来头探针样本小(地板重采 4 次、3 帧、2 位置),数字粗、方向清楚。

## 产物

- 脚本:`percell_dump.py` `percell_attn.py` `percell_analyze.py` `percell_grid_fig.py`(π0.5);
  `probe/fw_project.py` `probe/fw_percell_analyze.py` `probe/fw_future_probe.py`;`fw_scan.py` 加了
  `--cells/--patch-m`(向后兼容)。
- 数据:`out/percell_{obs,actions,scores}_{plate,rack,cabinet}.npz`、`out/grad/percell_attn_*.npz`、
  `out/grad/pi05_cell_world.npz`;`probe/out/fw_{cell_world,percell_scan_plate,percell_scores}.npz`。
- 图:`out/fig_percell_{plate,cabinet,rack}.png`、`out/fig_percell_pi05_3task.png`、
  `probe/out/fig_fw_percell_plate.png`。
