# AttenA+ 在线成功 BC：本地接点审计

2026-09-09。只做规划和源码阅读；本文件未修改训练实现、配置或服务器。历史 clean BC 源码锁为 `01d770db3988da7862454e97434d4ff08f726fa2`，归档见 [source-lock](../bc4-u5-final-closeout-20260908/bc/source-lock.json)。

## 结论

**可以从干净 BC 独立加 AttenA+，只复用通用逐动作加权损失，不必继承 DVAC 的 V、历史统计、两级归一化或完整 batch 预处理。** 作者权重从 normalized target action 计算，因此最小接点在 `forward_actor` 已拿到 `prepared['actions']` 后；权重由独立 helper 产生，传入 SFT 的 `action_weights`，在 `[B,50,14]` 未归约 FM 误差上相乘。

关键语义是：**本项目 π0.5 的 action 是绝对关节目标，normalized action 的模长不是物理速度。** 严格按作者 normalized-action 幅值做，是一版“作者式幅值代理”；按相邻原始关节目标做差分，则是另外一版 RoboTwin 运动适配，不能无声替换后仍称逐字复现。

## 1. 采集标签与 state

| 对象 | 当前实际语义 | 证据 |
|---|---|---|
| replay `action` | rollout 经 output transform 后，实际提交环境的 command；flatten 保存，训练恢复 `[B,50,14]` | clean `SuccessEpisodeCollector.append`，`online_bc.py:43–65`；模型 `predict_action_batch`，`openpi_action_model.py:915–948` |
| `model_action` | 原始模型空间输出；在线成功 collector **不保留**它 | collector 只保留 `observation/*`、tokenized prompt，再自行设置 `action`；同文件 `:54–65` |
| `observation/state` | query 前环境的原始状态输入，保存在每个 query record | 模型 `obs_processor :814–835`、克隆 `to_process_obs :944–948`；collector `:56–65` |
| `query_idx / episode_id / policy_version` | 可以区分 query 顺序、episode 和采集版本；完整 episode 入池后按 query 均匀有放回抽样 | collector `:65–69`；replay `sample :125–132` |

这里的 state 来自 RoboTwin `joint_action.vector`：左臂 6 关节＋左夹爪＋右臂 6 关节＋右夹爪。RoboTwin 历史源码 `get_left_arm_jointState` 读取的是 `get_drive_target()`，不是逐关节真实实测 qpos；若将来使用“首动作−state”，应称**命令变化代理**，不能声称实测速度。

证据：[clean collector](../bc4-u5-final-closeout-20260908/bc/source/rlinf/data/online_bc.py)、[clean model](../bc4-u5-final-closeout-20260908/bc/source/rlinf/models/embodiment/openpi/openpi_action_model.py)、[vector_env update_obs](/C:/Users/86136/Documents/rl/docs/fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sources/robotwin/robotwin/envs/vector_env.py:56)、[Robot state getter](/C:/Users/86136/Documents/rl/audits/20260719-robotwin-performance-analysis/source/robotwin_robot.py:528)。Robot state getter 是已存历史依赖快照，实施时核一次实际依赖哈希即可。

## 2. 模型标签是如何产生的

clean 模型 `prepare_dagger_sft_batch :627–672`：

1. 从 replay 取 observation 和 action；本流程没有 `model_action`，走 `else` 分支。
2. env action 恢复为 `[B,action_chunk,-1]`，放入 `obs_dict['actions']`。
3. 调用原始 `input_transform`，取得模型空间 `processed_obs['actions']`。
4. 返回 observation 和模型标签，进行原生 flow matching；不得把 AttenA+ 的权重乘到动作标签、噪声或 `u_t` 上。

本轮用本地 Git 的 `git show 01d770db:...` 核过该提交中的注册和 wrapper：

| 设置 | clean π0.5 值 | 含义 |
|---|---|---|
| `config_name` | `pi05_sidney_robotwin` | 精确复用 Sidney 适配 |
| `adapt_to_pi` | `False` | 不翻转关节，不做额外夹爪坐标变换 |
| `extra_delta_transform` | `False` | **不执行 DeltaActions**，模型标签仍是绝对 qpos |
| `use_quantile_norm` | `False` | 使用 checkpoint 的 mean/std normalization |
| wrapper 顺序 | AlohaInputs → Normalize → ModelTransformFactory | 若模仿作者 normalized-action 权重，应取 transform 之后的 `prepared['actions']` |
| 模型内部动作宽度 | 32；环境有效宽度 14 | 权重信号排除 padding；监督损失仍只算 14 维 |

已有相同注册和 wrapper 的可点击文件：[Sidney 注册](/C:/Users/86136/Documents/rl/references/sz_sidney_pi05_adapter_20260903/work/rlinf/models/embodiment/openpi/dataconfig/__init__.py:455)、[Aloha delta 开关](/C:/Users/86136/Documents/rl/references/sz_sidney_pi05_adapter_20260903/work/rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_dataconfig.py:84)、[wrapper](/C:/Users/86136/Documents/rl/references/sz_sidney_pi05_adapter_20260903/work/rlinf/models/embodiment/openpi/__init__.py:116)。`D32 / use_relative_actions=False` 另见 [转换契约](/C:/Users/86136/Documents/rl/references/sz_sidney_pi05_adapter_20260903/work/rlinf/utils/ckpt_convertor/openpi/lerobot_pi05_to_openpi_rlinf.py:21) 及 [parity](/C:/Users/86136/Documents/rl/references/sz_sidney_pi05_adapter_20260903/work/toolkits/lerobot/sidney_pi05_parity.py:247)。外部 OpenPI `ModelTransformFactory` 内部 padding 函数未在本次本地审计重新展开；部署前一次 shape 检查即可，不影响已确认的有效监督边界。

## 3. 双臂维度不能照搬单臂 first6

- 有效关节索引：`[0,1,2,3,4,5,7,8,9,10,11,12]`。
- 夹爪索引：`6,13`；从速度/幅值信号排除它们，与作者排除夹爪的动机一致；最终每个时间步权重可以仍乘包含夹爪的全部 14 维 FM 误差。
- 如果只抄作者 `slice(0,6)`，本项目只看左臂。双臂合成 12 维、两臂各算再合成，或选任务活动臂，是需要明确写在方法配置里的适配选择。
- 直接把 6 维 norm 扩为 12 维，会改变信号量级与 clip 占比；不能称只修了维数、强度完全相同。首轮应记录上下限饱和率，以实际数据判断权重有没有退化成常数。

源代码明确描述 `[6,1,6,1]`：[Aloha decode](/C:/Users/86136/Documents/rl/worktrees/pi05-stage/rlinf/models/embodiment/openpi/policies/aloha_policy.py:197)（同段已与 clean Git 对照）；RoboTwin 执行代码直接用 `actions[:,:6], actions[:,6], actions[:,7:13], actions[:,13]`：[动作拆分](/C:/Users/86136/Documents/rl/docs/fastwam-robotwin-rlinf-grpo/evidence/discussion-audit-20260905/sources/robotwin/envs/_base_task.py:2093)。

## 4. FM 权重接点与可以复用的代码

clean `sft_forward :408–424` 的顺序：原生逐元素 MSE → 截取 `action_chunk=50, action_env_dim=14` → `masked_fm_loss`。

当前 collector 创建全 1 的 `[50,14]` command mask。它表示整个提交命令是监督标签，**不表示机器人在成功终止前执行完了每个物理插值步**。不引入基于猜测的尾部 padding mask。

可复用旧 DVAC 对 `masked_fm_loss` 的小扩展：`W[B,H]` detach、检查有限且非负、广播到 D，然后仍以有效元素数作分母：

$$
L=\frac{1}{B}\sum_b\frac{\sum_{h,d}m_{bhd}W_{bh}\ell_{bhd}}{\sum_{h,d}m_{bhd}}.
$$

这不是除以权重和；保留作者可能改变平均 loss 尺度的行为。不要自动套 DVAC new 的均重 1 约束。

| 组件 | 处理建议 |
|---|---|
| `SuccessEpisodeCollector / SuccessReplay` | 直接继承 clean；episode success 筛选和 `max_success_chunks=3` 整条过滤已有实现 |
| `masked_fm_loss` 的 `action_weights` | 可仅移植通用可选参数及 shape/finite/detach 检查；未启用时回到 clean loss |
| `forward_actor` | 原 `prepare_dagger_sft_batch` 后取 normalized labels 算 W，再传 SFT；无第二次模型前向 |
| `sft_forward` | 只把可选 W 交给 weighted helper；保持 FM、D14 mask、梯度累积和优化器不变 |
| `prepare_replay_batch` | 这是 DVAC new 为完整 batch 归一化新增的通用 hook；作者式逐 query AttenA+ **不需要**，最小版可以不移植 |
| DVAC signal/history/config | 不移植 `dvac_v`、moments、recent5、two-level helper、`dvac_new.pt`；AttenA 使用独立配置身份 |
| checkpoint/采集元数据 | 原成功池和 RNG 可以继承；记录 AttenA 方法设置用于恢复校验，不伪装旧 DVAC 状态 |

具体可复用实现：[weighted loss](/C:/Users/86136/Documents/rl/local_scripts/bc_dvac_new_packet/src/rlinf/data/online_bc.py:14)、[SFT 传权重](../bc4-u5-final-closeout-20260908/dvac/source/rlinf/models/embodiment/openpi/openpi_action_model.py:418)、[新 batch hook](/C:/Users/86136/Documents/rl/local_scripts/bc_dvac_new_packet/src/rlinf/workers/actor/fsdp_dagger_policy_worker.py:469)。

## 5. 实现前只需锁清的事项

1. 首版采取**作者 normalized-action 幅值**还是**RoboTwin 原始 command 差分**。前者源码更贴近；后者更接近运动语义，但额外设计更多。
2. 双臂信号怎么合成以及对应尺度。采用作者式策略/阈值也要记录12维合成后的饱和率，避免静默全体落到clip下限。
3. 复用作者 `/clip*2` 还是另做真实均值归一。后者属于对照消融，不能默认混入作者复现。

无需重新讨论的部分：成功筛选和长度过滤、4/U5对照预算、原始模型起点、M10/C50、B1024/micro32、学习率/优化器/评估协议均可继承现有已确认设置。若要独立归因重加权，clean 对照同样开启长度过滤；历史未过滤 clean 仅作附加参照。

检查保持少量但有效：禁用/全1权重恢复 clean loss 与梯度；作者小张量公式一致；gripper/padding排除；microbatch拆分不改变 per-query 权重；实际 replay 小样本检查权重范围/均值/饱和率。规划阶段无需新增模型训练或长期smoke。
