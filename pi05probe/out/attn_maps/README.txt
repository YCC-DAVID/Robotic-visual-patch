PART B attention 热力图
==============================================================================

命名: t{时间步}_L{层}_{图类型}_{variant}_{view}.png
  层     = 00..17 逐层,或 rollout(全层累乘,第二种提取方式)
  图类型 = allmax  全部 token 逐元素 max —— POAP 的定位量
           allsum  全部 token 求和 —— §B1 要求的稳健变体
           noun    名词子词 token 求和 —— **跨指令比较必须用这个**
           verb    主动词 token
           func    功能词(含 <bos> 与 \n;attention sink 在这里)
  view   = base(agentview,主视角) | wrist(robot0_eye_in_hand)

⚠️ 跨指令比较不能用 allmax/allsum
  各指令真实 token 数 Z 不同(实测 6/8/9/10),token 越多 max 的期望越高
  ⇒ allmax 图**跨指令有偏**。B1/B2 一律用 noun 或逐 token 图。

⚠️ SSIM 请用 saliency.npz,不要用 PNG
  PNG 是叠加图(灰度底图 + hot colormap,alpha 0.55,**每图各自 min-max 归一化**),
  在它上面做 SSIM 会同时吃进底图和 colormap。
  saliency.npz 的 key 是 "{variant}|{view}|{layer}|{图类型}",值是 (16,16) float32,
  **未做 min-max 显示归一化**,只做了下面那步逐行归一化。
  每张图的原始 min/max 也都记在 manifest.csv 里,信息没丢。

归一化口径(§A3 第 2 步,逐行,在 token 归约之前)
  本次用 renorm=img512
    img512 = 每个 text token 在【base 256 + left_wrist 256】上归一化
             (right_wrist 补零且 image_mask=False,不进分母)
    base   = 只在 base 的 256 内归一化
    none   = 不归一化(用于验证 sink 的影响)
  顺序承重:head 求和 → 逐行归一化 → token 归约。若先 max 再归一化,
  sink token 的绝对量级会压过其他 token。

⚠️ 朝向(见 orientation_check.png)
  模型输入是 obs[...][::-1, ::-1](examples/libero/main.py:115)。
  LIBERO 原始 buffer 是 bottom-up 的 R=V(A) ⇒ 喂给模型的是 V(H(R))=**H(A)**,
  即左右镜像的正立图。热图叠在模型输入上是自洽的(同一坐标系),
  但要跟世界坐标 / LIBERO 自然朝向对照时,必须把这个镜像解掉。

⚠️ 时间步
  目前只有 t000 = 共享帧(set_init_state + 10 步 dummy warmup 之后)。
  四条 B1 指令面对的这一帧已验证**逐位相同**(S0.5 检查 B),唯一变量是文本。
  多时间步要沿 rollout 采帧(渲染在 py3.8、模型在 py3.11,需跨进程),另做;
  那一步同时给出 B3 的噪声地板 —— **没有地板,现在所有相关系数都缺一个"多少算大"的参照**。

上采样: 最近邻(np.kron),保留 patch 边界。16×16 → 448×448,每个 patch 28×28 像素。
        没做双线性 —— 那会造出不存在的平滑,看起来像定位更准。

层数据来源: out/attn_b1b2.npz 的 head_sum[layer, token, view, 16, 16](head 已按 §A2 求和)
            per_head[...] 也存着,可查 head 间方差。
