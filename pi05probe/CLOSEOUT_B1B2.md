# CLOSEOUT_B1B2.md — ①换任务 / ②换措辞(PART B 的 B1/B2)定稿

状态:**①② 定稿**。四条指令面对逐位相同的一帧(前提检查 B 已过),模型确实听指令
(交叉评估 A 通过),四条改写句都不塌(B 通过),Spearman 已配上 B3 噪声地板(C)、
并可反投影到世界坐标(D,几何已验证)。

最后更新:2026-08-10 · 主机 `nnmc65`,GPU 3(L40S)· PyTorch 路径(与 §PT 一致)

配套产物:
`out/crosseval.json`(A+B 原始数)、`out/report_floor.txt`(C)、`out/reproject.txt`(D)、
`out/report_b1b2.txt`(单帧旧版,已被 C 取代)、`out/report_traj_minpair.txt`(最小对全轨迹)。

复现:
```
# 1) 起 torch server(py3.11)
/home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/run_demo.py --torch --server-only
# 2) 交叉评估 A + 改写成功率 B(py3.8,~15 min,90 rollout)
~/miniconda3/envs/openpi-libero/bin/python pi05probe/crosseval.py --episodes 10
# 3) 纯后处理(秒级)
/home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/report_floor.py   # C
/home/user1/miniconda3/envs/openpi-server/bin/python pi05probe/reproject.py       # D
```

---

## 收尾四项结论一览

| 项 | 是什么 | 结论 |
|---|---|---|
| **A · 交叉评估(前提检查:模型听不听指令)** | 给指令 A、用目标 B 的成功判据评,4×4 | ✅ **通过**:对角高、非对角全 0 |
| **B · 改写句成功率(哪些改写能进 B2)** | 5 个改写 + 原句,各 10 episode | ✅ **全 10/10**,五条改写都可用 |
| **C · B3 噪声地板(Spearman 的参照尺)** | 相邻帧地板 + 全轨迹聚合,替代单帧旧版 | ✅ 理想层 **2/18 → 10/18**,集中在 L07–L17 |
| **D · 世界坐标反投影(把网格换成米)** | depth+K+[R\|t] 反投影,seg 对准 | ✅ 几何已验证(桌面 z=0.8994);⚠️ 绝对峰值被 sink 主导,不作定位器 |

---

## A · 交叉评估 4×4(前提检查,go/no-go)

每格 = 10 episode 里的成功次数。用各 task 自己的 `.pruned_init`,seed=7+ep,PyTorch 策略。

| 下达指令＼评判 | turn_on_stove | bottle_on_rack | bowl_on_plate | bowl_on_cabinet |
|---|---|---|---|---|
| **turn on the stove** | **9** | 0 | 0 | 0 |
| **put the wine bottle on the rack** | 0 | **10** | 0 | 0 |
| **put the bowl on the plate** | 0 | 0 | **10** | 0 |
| **put the bowl on top of the cabinet** | 0 | 0 | 0 | **10** |

**读法**:对角线(自身成功率)9–10/10 正常;**非对角(交叉)全 0**。
⇒ 下达指令 A 从不(在 40 个 episode 里一次都没有)达成别的任务的目标
⇒ **模型确实由文本控制行为,不是按场景 affordance 广撒网**
⇒ **换任务(B1)的前提成立,①② 有资格定稿。**

（stove 那 1 个失败与 §PT 的 clean 成功率一致,不影响判据。）

---

## B · 改写句 clean 成功率(哪些改写能进 B2)

全在 `put the bowl on the plate` 场景、评 bowl_plate 目标,各 10 episode。

| 改写层 | 句子 | 成功率 |
|---|---|---|
| 原句 | `put the bowl on the plate` | 10/10 |
| L1 换动词 | `place / set / move the bowl on the plate` | 10/10 · 10/10 · 10/10 |
| L2 换句法(倒装) | `on the plate, put the bowl` | **10/10** |
| L3 加框架词 | `please put the bowl on the plate` | 10/10 |

**读法**:五条改写**全部不塌**,连最可疑的倒装 `on the plate, put the bowl` 也 10/10。
⇒ B2 比的是**"正常 vs 正常"**,不是"正常 vs 懵" ⇒ 五条改写都可进对比。
⚠️ 印证了计划里对 π0.5 的注记:它做了更广的语言协同训练,**设计上就更抗改写**。
所以下面"attention 对改写稳定"的结论要写成"**在 π0.5 上**稳定",不写成一般性结论。

---

## C · B1×B2 判读(配上 B3 地板,取代单帧旧版)

`report_b1b2.txt` 的所有 Spearman 是**单帧 t000、无地板**,所以"0.95 算高吗"没有参照。
本项改用 `attn_traj` npz(9 个 variant 沿 16 帧的 attention),**沿同 16 帧逐帧算再平均**
(与地板同口径,§A1),对照 B3 地板:

- **地板** = bowl_plate 的 `bowl` 图、相邻两帧 Spearman(该图自然漂移多少)。
- **换措辞** = bowl_plate 的 `bowl` 图 vs 5 个改写(名词不变,隔离改写效应)。
- **换任务(最小对)** = bowl_plate 的 `bowl` 图 vs bowl_cabinet(同操作对象,只换目的地)。

**判据(理想)**:换措辞 ≥ 地板(改写不动图) **且** 换任务 < 地板(换目的地动图)。

结果(逐层,详见 `report_floor.txt`):

| 层带 | 换措辞 vs 地板 | 换任务 vs 地板 | 判读 |
|---|---|---|---|
| L00–L06 | ≈ 地板 | ≈ 地板(不动) | 早层几乎不区分:任务也没动 |
| **L07–L13** | **≥ 地板(稳)** | **< 地板(动)** | ✅ **理想:换措辞稳、换任务动** |
| L15–L17 | ≥ 地板 | < 地板 | ✅ 理想(但整体 Spearman 已随深度下滑) |

⇒ **18 层里 10 层达到理想**(旧单帧版只有 2 层)。**信号集中在中后层 L07–L17。**

**两条定稿结论:**
1. **换措辞(②):attention 对同义改写稳定。** 五条改写的 `bowl` 图 Spearman 在多数层 ≥ 地板
   —— 改写扰动比单纯过一个时间步还小。(措辞收着说:**在 π0.5 上**。)
2. **换任务(①):attention 对任务敏感,锁"操作对象",也锁"目的地"。**
   - 最小对里换目的地,共享的 `bowl` 图在 L07+ 动幅**超过地板** ⇒ 任务语义确实进了图。
   - > ⚠️ **更正(2026-08-17,`make_task_attn_figs.py`)**:原来说"目的地没有被稳定编码成一个
     > 场景位置"是**错误推断**。`plate` 图与 `cabinet` 图重叠低,不是"没编码",而正是**期望的好结果**
     > —— 两张图在接近段(frame 0–9)的峰值**各自落在自己的目的地上**(L4/5/8,10/10、9/10 帧命中)。
     > 之前用 argmax 跨层跳来判"没编码",是把 sink 主导的绝对峰值当了定位器(见下 D 节的同类更正)。
   - B1 正交性(stove/bottle/bowl 物体集不相交 → 全名词图相关低于地板)**是弱证据**:
     它混入了"名词本来就不同"这个 confound,所以以最小对为准,不以正交性为准。

---

## D · 世界坐标反投影(§A-5)

链路:`16×16 cell → 224 输入像素 → 撤 resize(224→256) → 撤 180° 旋转 → raw256 →
depth_m + K + E 反投影 → 世界 3D`。E 已折叠 `diag(1,-1,-1,1)`(robosuite 相机看 -z)。

- **几何自检通过**:整幅反投影,桌面主平面 **z mean = 0.8994**(期望 0.900,Q7),
  xy 落在桌面范围内。⇒ 反投影正确,可留作 S2 influence 的世界坐标对照。
- ⚠️ **但绝对峰值不是可靠的逐指令定位器**:band-mean(L04–L12)后取 argmax,
  **四条指令、各名词 token 的峰值全塌到同一 register/sink 格 (7,9)**
  ⇒ 与 sink 诊断一致(`\n`/register 主导绝对量级)。可靠的逐指令定位要看
  **montage 的逐层 3×3 窗口**(§A2 第 5 步,对单个 sink 格更稳),不看 band-mean argmax。
  > ⚠️ **更正(2026-08-17)**:"四条指令峰值全落 (7,9)"是**单帧假象** —— `reproject.txt`
  > 用的是 `attn_b1b2.npz`(只有 t000 一帧)。在接近段(frame 0–9)上取多帧后,
  > 四条指令的峰值会**分开、各落在自己指令的物体上**(L8,见 `make_task_attn_figs.py`)。
  > 结论方向不变(band-mean argmax 不如逐层窗口稳),但"全塌同一格"这句只对单帧成立。
- 最小对目的地世界位移(plate 峰值 vs cabinet 峰值)≈ **0.244 m**,但因目的地图跨层不稳,
  **只作示意**,不作定量结论。

---

## 定稿后仍存在的边界(写清楚,免得被当成一般性结论)

1. **attention 侧到此为止。** 核心命题(attention vs influence、S4 交叉代价)仍需 S2 的 Δa —
   本文件只定稿了"陪跑"那一半。
2. **地板来自单条轨迹**(bowl_plate 的 16 帧)。信号跨层一致,但更稳的做法是多条轨迹各自出地板。
3. **π0.5 特有的抗改写**:②的稳定可能部分来自模型本身鲁棒,而非"attention 与语义无关"。
   结论已按此收窄措辞。
4. **绝对定位不可用**(见 D):所有"attention 落在哪"的结论都基于**相对/共享 token 比较**,
   不基于绝对峰值。

---

## 新增/改动的脚手架

| 文件 | 作用 | 能跑? |
|---|---|---|
| `crosseval.py` | 交叉评估 A + 改写成功率 B(py3.8 client,单 env 评 4 goal) | ✅ |
| `smoke_crosseval.py` | 红线1:烧算力前验证交叉评估谓词机制(纯 mujoco) | ✅ |
| `report_floor.py` | C:B1/B2 配 B3 地板 + 全轨迹聚合(取代单帧 report_b1b2) | ✅ |
| `reproject.py` | D:attention → 世界坐标,含桌面平面几何自检 | ✅ |
| `out/crosseval.json` | A 的 4×4 矩阵 + B 的成功率 | — |
| `out/report_floor.txt` / `out/reproject.txt` | C / D 完整数值 | — |
