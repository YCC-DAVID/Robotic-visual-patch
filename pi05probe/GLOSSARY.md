# 术语表

分两类:**来自计划文档**(`~/.claude/plans/enumerated-baking-ullman.md`)的编号,
和**本目录脚本自造**的词。后者以前只在脚本输出里有定义,容易看不懂,集中放这里。

---

## 一、来自计划文档的编号

| 记号 | 指什么 |
|---|---|
| `S0` … `S4` | 计划的执行阶段。S0 环境与事实核查;S0.5 前提检查;S1 场景 patch 注入;S2 探针扫描(Δa);S3 分析;S4 交叉代价 |
| `Q1` … `Q10` | S0 要回答的十个问题(相机/动作/归一化/chunk/ε/attention/场景注入/传感器/设 qpos/初始状态一致性) |
| `红线 1/2/3` | 计划里三条方法论红线。1:先跑"什么都不改"的重复;2:拆出的子模块必须 `.eval()`;3:batch 形状会改结果 |
| `B0` … `B4` | PART B 的子步骤。B0 attention 提取方式;B1 换任务;B2 换措辞;B3 噪声地板;B4 attention vs Δa |
| `§A` … `§E` | 计划 2026-08-03 追加的"补充规格"小节。§A attention 提取;§B 存储 schema;§C 跨指令比较的偏差;§D Δa 的度量;§E 追加禁止 |
| `POAP` | 对照方法(arXiv 2606.03556),未开源,按正文实现 |
| **最小对** | `put the bowl on the plate` vs `put the bowl on top of the cabinet` —— 同操作对象、只换目的地 |
| **噪声地板** | 一个"多少算大"的参照量。B3 的地板 = 同一条指令、**相邻两帧**之间的 attention 差异 |
| **influence / Δa** | 在场景某位置放 patch 引起的模型输出动作变化 |

---

## 一之补:2026-08-17 加密扫描 + 固定-ε rollout 引入的词

| 名字 | 指什么 |
|---|---|
| **加密扫描 / 加密锚点** | 在合法区靠近任务物体的一环上以 2 cm 间距补采的锚点(`make_fine_anchors.py`),号段从 **#1000** 起(与旧 6×6 网格的 0–35 号不撞)。补它是因为旧网格最近的合法点距物体 15 cm,已在 attention 塌掉的一侧 |
| **合并候选池 / 78 点池** | 旧网格里的 17 个合法点 + 加密的 61 个 = **78 个合法可贴位置**。2026-08-17 起所有排序/相关结论都在这个池上算(`report_fine_legal.py`) |
| **面积配对** | 两个投影可见面积几乎相同、但 influence 差几倍的锚点对(如 #1006 978 px 307 mm vs #1055 977 px 102 mm)。用来在**控制面积**下证明"是位置而非面积决定 influence/行为" |
| **偏相关(面积↔influence \| 距离)** | 控制"到最近任务物体的距离"后,面积与 influence 的秩偏相关。78 点池上 = **−0.48**,证明面积只是距离的影子 |
| **固定-ε / 零 ε 地板** | `serve_policy_fixed_noise.py` 在 server 侧把 flow-matching 的初始噪声 ε 钉死成一条固定向量 ⇒ clean 与 patched 共享同一 ε ⇒ **采样噪声地板 = 0**,`clean` vs `clean_repeat` 逐位相同。用来把上一轮埋掉信号的 28 mm 采样噪声消掉 |
| **路径发散 vs 放置误差** | rollout 偏移取两条轨迹公共长度算,patched 因扰动多走几步 ⇒ 峰值多半是"滞后";各自走到成功点时末端只差 ~19 mm(闭环修回)。峰值 67 mm 是路径发散,不是放置误差 |
| **跨模型复制(FastWAM)** | 在 FastWAM(Wan2.2 DiT + action expert)上复用**同一个 LIBERO 桌面、同一批 78 锚点、同一份 `scene_patch`**,只换模型,检验"面积不决定、位置决定"能否复制(`probe/fw_scan.py` / `fw_report.py`)。结论:距离主导逐字复制、attention 选不对复制;面积边际方向反号(RESULTS §17) |
| **反方向 cross-attention(FastWAM)** | π0.5 是"文本查询 × 图像键"(在图像上归一);FastWAM 视频 DiT 是"视频查询 × 文本键"(`wan_video_dit.py:265`)。取某词的图像图 = attention 矩阵**该词那一列**,再在 98 个 video token 上**重新归一**(该列本身不归一)。与扩散模型 cross-attn 可视化的通行做法一致 |
| **base 7×7 落格 / 格级命中检验** | FastWAM 视频 token 网格 f1×h7×w14=98,左 7 列(w0–6)= 主视角 base=**7×7=49 格**、≈32 px/格。attention 图只有这个粗分辨率。命中检验 = "attention 峰值格是否 = influence 最强锚点(#1007)的落格"。锚点落格由**渲染贴纸分割质心**求(`fw_anchor_cells.py`),不用拟合公式(rot180+镜像易错)。⚠️ 模型输入是 `agentview[::-1,::-1]`(rot180),叠回正立图要**左右翻列**;29/30 层峰值格压在物体本体(不许贴)⇒ 精确命中天然为 0,邻格命中才是可达上限 |

⚠️ 我(助手)在对话里还用过 `§1/§2/§3` 这种指**脚本输出里的小节**,那不是计划文档的编号。
以后一律写成"`diagnose_sink_cells.py` 输出的第 2 节",不用裸编号。

## 一之补二:2026-08-20 换 destination + 分阶段 + 梯度校验引入的词

| 名字 | 指什么 |
|---|---|
| **换 destination 扫描 / 跨任务对比** | 同一批 78 锚点、同一张探针纹理,只把 LIBERO-Goal 任务换成 bottle→rack、bowl→cabinet(`fw_scan.py --task`,产出 `fw_scan_rack.npz` / `fw_scan_cabinet.npz`),检验 influence 热区是否跟着目标物走(`fig_fw_crosstask_influence.png`) |
| **近机器人侧假设(不可检验)** | 猜想"influence 最强点在目标物靠机器人(基座 x=−0.66)的一侧"。最初以"三任务 I-max 全在远机器人侧"否证,后发现那是**池子偏置**(见下条),侧向假设在现有池上根本测不了 |
| **相机近侧(已撤回:池子偏置)** | 曾以"三任务 I-max 全落 +x 侧、agentview 相机在 (0.659,0,1.61) 的 +x 侧俯视"提出"离相机近像素占比大"的解释。用户指出合法池本身就偏:实测 **75/78 锚点在 x>0 半侧**(中位 x=+0.19)⇒ I-max 落 +x 是构成使然。控距目标后的偏秩相关 I↔x\|(−d_tgt) 三任务 = +0.23/−0.34/−0.63,等距环内 +x 半侧不强反弱 ⇒ 相机侧无池内优势。rack/cabinet 的负号疑似"离被抓物体近"混杂(x 低同时更近 bottle/bowl,rack 的 I↔(−d_src)=+0.73)。干净检验需在目标两侧对称补采锚点(`camtest.py`) |
| **rack 覆盖缺口** | rack 任务 clean rollout 在 10 个 replan 帧窗口内未完成(success=False,夹爪 8-9 帧才闭合)⇒ 该任务 influence 只覆盖 approach+grasp;且 78 点池距 rack 最近 48.8 cm ⇒ "近 rack"不可检验,量级也不可与另两任务直接比 |
| **分阶段 influence/attention** | 按 clean 动作的夹爪通道(dim6)两个翻转点 + z 走向把 10 帧切成 6 阶段(approach/grasp/lift+move/descend-place/release/retreat),逐阶段画图(`fw_phase_figs.py`)。发现:放置+松爪敏感度 ≈ 进场抓取的 3 倍,热区随任务进度迁移;attention 却平且不动 |
| **纯平面 attention 图 / 锚点落格上色** | 把 attention 画成与 influence 同款的世界坐标散点:每个锚点按它落格(`fw_anchor_cells.npz` 的 `cell`)的 7×7 attention 值上色(`fig_fw_phase_attention_plane.png`)。同格锚点同色 = 32 px 格子分辨率天花板的如实呈现 |
| **池外峰值** | attention 的 7×7 峰值格在全部 6 阶段都**不含任何合法锚点**(①-④在机器人/bowl 区,⑤-⑥在图像角落疑似 sink);池内 attention ≤1.5× uniform 且钉死在 #34 ⇒ attention 对"可贴区"完全无区分度 |
| **G0 梯度校验** | 梯度显著性实验的准入门:G0.1 梯度到达像素叶子 / G0.2 解析 vs 中心差分(h∈{1e-2,1e-3,1e-4},固定 ε,~10% 线)/ G0.3 NaN·用时·显存·3ε 稳定性。三项全过才许写 G1(`g0_gradcheck.py` → `out/grad/FINDINGS_grad.md`) |
| **用户-L≡0 问题** | 梯度 spec 里 L=Σw_k‖a−a_clean‖²(a_clean=同一固定 ε 的 clean 前向 detach)在未扰动点 x 处 ∇L=2Jᵀ(a−a_clean)**恰好为 0**(a 与 a_clean 逐位相同)⇒ 不能直接当 backward 标量。G0.2 里演示了这一点;一阶正确的替代是**通道和标量** |
| **通道和标量 s_c** | s_c = Σ_{k<EX} a[k,c](c=0,1,2 平移三通道分开)。对每个 c 单独 backward 得 g_c,合成 gmag=√(Σ_c g_c²) 作像素显著性 ⇒ 等价于 FD 的一阶方向导数结构,且满足"三通道分开"的要求 |
| **绕开 @torch.no_grad(不改模型)** | `sample_actions` 唯一的梯度阻断是装饰器(`pi0_pytorch.py:376`)+ policy 输出 `.detach()`(`policy.py:98`)。做法:把函数体逐字复刻到 `g0_gradcheck.py::sample_actions_grad`,在 `torch.enable_grad()` 下调用未装饰的子方法;模型文件零改动 |

---

## 二、本目录脚本自造的词

### 图类型(`make_attn_maps.py`,写进文件名)

设 `A` 为该层该视图下、逐行归一化后的逐 token 注意力图,形状 `(Z, 256)`,
`Z` = 该指令的真实 token 数,256 = 16×16 图像 patch。

| 名字 | 定义 | 用途 |
|---|---|---|
| `allmax` | `A.max(axis=0)` —— 对**全部 Z 个 token** 取逐元素 max | POAP 的定位量。⚠️ 跨指令**有偏**(Z 不同,token 越多 max 期望越高) |
| `allsum` | `A.sum(axis=0)` —— 对全部 token 求和 | §B1 要求的稳健变体。⚠️ 同样跨指令有偏 |
| `noun` | `A[名词行].sum(axis=0)` | **跨指令比较只能用这个**(或逐 token 图) |
| `verb` | `A[主动词行].sum(axis=0)` | |
| `func` | `A[其余行].sum(axis=0)`,含 `<bos>` 和 `\n` | 文本侧 sink 集中在这里 |

`allsum` / `noun` / `verb` / `func` 是**同一个操作**(在 token 轴求和),只是行子集不同。

### 归一化口径(`--renorm`)

§A3 要求 head 求和后、token 归约前做**逐行归一化**(每个 text token 各自归一化)。
分母有三种选法,写进文件名:

| 口径 | 分母 |
|---|---|
| `img512`(默认) | base 256 格 + left_wrist 256 格。`right_wrist` 是补零且 `image_mask=False`,不进分母 |
| `base` | 只用 base 的 256 格 |
| `none` | 不归一化(用来验证 sink 的影响) |

### 时间步

| 记号 | 含义 |
|---|---|
| `t000` | 共享帧:`set_init_state(共享状态)` + 10 步 dummy warmup 之后那一帧。**四条指令面对的这一帧逐位相同**,唯一变量是文本 |
| `k` / `t` | 沿 rollout 采帧时:`k` = 第几个采样点(replan 边界序号),`t` = 环境步数 |

### sink 与伪影(`diagnose_sink_cells.py`)

| 名字 | 定义 |
|---|---|
| **文本侧 sink** | 某个 **query 行**吃掉大量注意力质量。实测是 `\n`(不是 BOS),占 base 图质量 0.3806–0.3833,是其余 token 的 11–18 倍,且 23 条指令几乎不变 |
| **图像侧 sink / register patch** | 某些 **key 列**(固定的几个 image patch)被**所有** query 注意。ViT 里的已知现象("registers")。比文本侧更麻烦,因为 3×3 窗口是在图像格上滑的 |
| **key 侧 vs query 侧** | key 侧 = 那几个 patch 本身在吸引所有查询;query 侧 = 只是某个 token 的偏好。判别法:对**全部 token 行取 min**,连最小值都高就是 key 侧 |
| **伪影格** | 定量定义:每个 variant 的图各自 min-max 归一化到 `[0,1]`,再**对 variant 取 min**,取该层最高的 8 格。min 高 ⇒ 对每一条指令都亮 ⇒ 与指令无关 |
| **伪影 mask** | 上面那张"跨指令 min"图本身,存在 `out/sink_cells.npz` |
| **命中伪影** | 选出的 3×3 窗口所覆盖的 9 格里,**至少有一格是伪影格** |
| **贴边** | 3×3 窗口贴着图像边界。窗口中心只能取 1..14,所以中心的行或列 `∈ {1, 14}` 就是贴边 |
| **唯一位置数** | 该层下,23 个 variant 选出的窗口中心一共有几个**不同**位置。等于 1 ⇒ 所有指令选出同一个窗口 ⇒ 定位与指令无关 |
| **可用中间层带** | 实测约 L04–L12:青框逐指令不同、且落在被指名的物体上。出了这个带(L00–L02、L14–L17、rollout)窗口落在跨指令不变的格上 |

### 提取方式

| 名字 | 定义 |
|---|---|
| **逐层图** | 单层内 head 求和。montage 里 `L00`–`L17` 那些行 |
| **`roll` / rollout** | Attention rollout:每层 head **平均** → `0.5A + 0.5I` → 行归一化 → **18 层矩阵累乘**。**不是层,也不是求和**,是路径复合。所以没有层号 |
| **层平均** | 把 18 张逐层图直接平均。**当前没算**,是零成本后处理,可作 rollout 的对照 |

### 画图约定

| 名字 | 含义 |
|---|---|
| **青框** | 图上那个青色方框 = **实际选出的 3×3 窗口**(§A2 第 5 步)。青色是刻意选的,`hot` 色标里没有青,不撞色 |
| 色标 | `hot`:黑 → 红 → 黄 → 白,白最高。⚠️ 白端和场景里的亮物体撞色,考虑换 `inferno`(结尾是黄) |
| 显示归一化 | **每张图各自 min-max**。所以白 ≠ 跨图可比;原始 min/max 在字幕和 `manifest.csv` 里 |
| 上采样 | 最近邻(`np.kron`),16×16 → 448×448,每格 28×28 像素。不用双线性 —— 那会造出不存在的平滑,看起来像定位更准 |
| 朝向 | 模型输入是 `obs[...][::-1,::-1]`。LIBERO 原始 buffer 是 bottom-up 的 `R=V(A)` ⇒ 喂给模型的是 `V(H(R))=H(A)`,即**左右镜像的正立图**。热图叠在模型输入上自洽,但要对世界坐标就得把镜像解掉 |
