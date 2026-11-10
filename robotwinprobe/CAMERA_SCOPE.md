# Patch训练的相机范围

用户于2026-10-27要求所有当前方向的patch训练纳入腕部。

默认让同一张世界纹理Q经模型所有原生相机的几何、遮挡与成像代理进入模型，联合梯度回传至Q。不能将“推理提供了腕部图像”当作“腕部参与patch优化”。只有明确命名的相机消融才保留clean腕部或单路梯度；已有消融结果不改写。

| 路径 | 训练范围 | 核查入口 |
|---|---|---|
| RoboTwin π0.5，历史20cm及后续阶段纹理 | head、left wrist、right wrist均依赖同一Q | `render_train.py`、`crossarch_proxy.images`、`crossarch_backend.Backend.prepare` |
| RoboTwin ImageWAM，早期20cm及原生AE阶段/完整flow目标 | head、left wrist、right wrist均依赖同一Q | `imagewamprobe/train_patch.py`、`crossarch_proxy.images`、`crossarch_wam.compose` |
| RoboTwin FastWAM，10cm及条件20cm分支 | head、left wrist、right wrist均依赖同一Q | `wam_flow_patch.gradient`、`fastwam_adapter.Backend.prepare` |
| LIBERO FastWAM旧base-only | 仅base代理，wrist clean；作为已完成消融保留 | `fastwam_libero_trial.py`，第41号报告 |
| LIBERO FastWAM当前both | 原base代理不变，新增逐帧wrist代理；共享Q，两路梯度求和已核验 | `fastwam_libero_both.py`，第42号报告 |

RoboTwin缺任一原生视角，或LIBERO缺base/wrist之一时，模型输入适配器应报错。输入数量检查不替代梯度接线检查；某路实际纹理梯度允许因物理遮挡或不可见而为零，不人为插入可见patch。

本方向尚未开展LIBERO ImageWAM试验，不能将LIBERO FastWAM的base-only遗漏归到此前RoboTwin ImageWAM上。
