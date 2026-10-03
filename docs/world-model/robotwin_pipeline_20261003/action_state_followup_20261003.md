# OpenDW × Sidney：动作分布、state 与时间合同补充

核验日期：2026-10-03。只讨论本次 RoboTwin `adjust_bottle`、Sidney π0.5、OpenDW 发布 Robotwin bundle；接口基础见 [C32 决策](c32_adapter_decisions.md)。以下区分源码事实、尚未公开的训练信息和实验假设，不把“接线正确”写成“生成动力学已正确”。

## 1. 可以先跑；剩下的问题主要是模型是否理解我们的行为

**推荐仍是：Sidney 产生原始绝对关节命令，取前 32 条；OpenDW 用配套统计重新归一化并生成三图；下一次 state 取该 chunk 最后一条命令。** 不额外加入 state 预测器、关节平滑或速度转换。这个接口有直接的官方依据：OpenDW demo 以绝对 qpos 序列驱动生成，完成后 `base_state=action_abs[-1]`；我们原生 RoboTwin 的 policy state 本来读取 drive target。[OpenDW demo L1605–1629](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/playground/online_demos/robotwin_online_demo.py#L1605-L1629)、[RoboTwin getter L528–558](https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/envs/robot/robot.py#L528-L558)。

简单例子：state 的第一个数是 `0.20 rad`，动作命令它到 `0.30 rad`。完整执行后，我们给策略的命令 state 是 `0.30`；碰到瓶子时实际关节可能只有 `0.28`。**这里不需要把 `0.28` 猜出来填给原本读取命令的策略，但生成图必须合理体现这次动作和接触。** 若图里机械臂根本没动、瓶子却自动摆正，state 接线仍可正确，WM 动力学已经失真。

“末动作即下一 state”只适用于本轮定义的完整 C32 边界。原生环境的 TOPP 失败、提前终止会改变实际执行；本轮 WM 在 chunk 末更新命令态，不声称复刻了这些物理事件。其成功是视觉 RM 判定，最终效果仍由相同控制协议下的原生评估判断。

## 2. 新发现：发布 bundle 的推理配方与公开默认训练配方不同

| 项目 | 发布 Robotwin bundle 的官方推理路径 | 当前公开默认训练数据路径 |
|---|---|---|
| 动作意义 | 14D absolute qpos，配套 z-score | 从下一行 `robot.state` 构造动作，再默认转 `delta_first_frame` |
| delta 的意义 | 本次不用 | 每个未来状态减去当前起点，不是逐帧差分，更不是 rad/s |
| 维度 | 发布 bundle 推理按 14D，运行时从权重识别 | 14D 排成 16D，添加终止位，再 padding 至 32D |
| 归一化 | action/state 分开使用 mean/std，结果 clip 到 ±5 | 默认 q01/q99 分位归一化至 [-1,1] |
| 配置优先级 | `zscore_14d + absolute` 是本次显式选择 | dataset 参数 `action_type="state"` 被丢弃；真正生效的是每个 dataset 的 metadata |

证据：官方 runtime [L320–349](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L320-L349)、[动作归一化 L426–456](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L426-L456)；训练 [AddAction/DeltaAction L172–240](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/transform/action.py#L172-L240)、[ROBOTWIN_META L13–20](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/data_source.py#L13-L20)、[忽略参数 L192–218](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/dw_dataset.py#L192-L218)、[实际 transform 顺序 L775–798](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/dw_dataset.py#L775-L798)。

**这不证明发布权重训错了，也不要求现在改用 delta。** 它证明仓库同时保留不同数据/推理合同：本次推理应跟发布 bundle；以后微调该 bundle，不能照抄默认训练入口就称“同配方继续训练”。届时须显式对齐 action/state 维度、absolute/delta、统计文件格式及图像布局。

公开 loader 能确认动作来自“下一行 state”；目前未核到该发布 checkpoint 的完整训练 manifest、Dexdata 导出器及每条数据的采样时间。因而 **`robot.state` 在该训练集究竟采自 drive target 还是 measured qpos、是否包含我们 Sidney 的失败轨迹，仍未证实**。不要用我们原生环境的 getter 代替这个缺失的训练来源证据。官方 [JSONL 格式](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/README.md#L101-L123) 只给字段格式；[Dexmal 数据卡](https://huggingface.co/datasets/Dexmal/robotwin2-full) 说明 clean/randomized 两类目录，未补齐上述来源链。

配套 [norm_stats.json](https://huggingface.co/Dexmal/DW05-Robotwin/blob/6ab5f9e2636610cba440d08264663efe70c3f761/norm_stats.json) 记录 27,500 episodes、6,075,103 transitions，action 的 stepwise stats 为 32×14。这是**归一化统计文件的计数**，不能据此断言最终 checkpoint 的训练样本量或成功/失败覆盖。

## 3. C32 对齐的是索引；物理速度还不能直接画等号

| 数字 | 准确含义 | 不能混成什么 |
|---|---|---|
| H50 | Sidney 一次预测 50 条候选动作 | 不代表必须执行 50 条 |
| C32 | 执行前 32 条，再获取新观察 | 不代表 32 个物理仿真 tick |
| stride 4 | OpenDW 每 4 行动作对应一个生成图像位置 | 不代表控制频率是 4 Hz |
| 9 帧 | 起点 + 第 4、8、…、32 个动作后的图像 | 不是每条动作都有精确生成图像 |

公开训练 loader 以 `num_frames=33`、ratio 4 得到 chunk32 与 8 张未来图；图像索引明确是 `start+(offset+1)*4`。[dataset L223–227](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/dw_dataset.py#L223-L227)、[future image L559–571](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/data/dataset/dw05/transform/multimodal.py#L559-L571)。

原生 RoboTwin 每条 qpos 命令会经过 TOPP，左右臂各自得到一串中间位置/速度，再按 `1/250` 步长执行；中间步数随路径变化。因此“32 条命令”本身不能换算成固定几秒。OpenDW 数据的行间时间也未有上述训练 manifest 支持；GIF 播放帧率更不能当控制频率。[TOPP L1589–1631](https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/envs/_base_task.py#L1589-L1631)、[执行循环 L1708–1737](https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/envs/_base_task.py#L1708-L1737)。

**当前做法：按官方每 32 条 → 8 张未来图接通；记录命令变化量，不伪称物理速度已一致。** 原策略通常连续移动，WM 可能尚可泛化；若 RL 产生大跳变、频繁往返或极快夹爪切换，原生 TOPP 与训练视频中的动态就可能分离。此处是待观察的模型分布风险，不是已确认本次训练发生的问题。

正式 rollout 建议 **384=12×32**，避免 400 的尾部 16 条需要额外截断/补齐语义；smoke 用 32。384 比原 400 少 16 条，原始策略与 RL 检查点必须使用同为 C32/384 的评估才可直接比较；既有 C50/400 成绩保留原协议标签。

## 4. 其他公开项目怎么处理；哪些可借

| 实现 | 动作/state 的真实做法 | 对本次有用的结论 |
|---|---|---|
| OpenDW demo | 外部绝对动作；上一 state 与动作序列配对；最后一条动作成为下一 base_state | 直接支持当前命令态更新方案；不等于已经验证接触动力学 |
| RISE | `next_state=action_pred[:,-1]`，写回策略 observation；这里是模型空间动作，源码还将 decoded 版本注释掉 | “末动作更新 state”有公开先例；必须继承各自归一化，不能直接复制 tensor |
| VLA-MBPO | LIBERO WM step 保留旧 state；该 π0.5 分支关闭离散 state 输入 | 它用不读取 state 的策略绕过；不能给读取 state 的 Sidney 原样套用 |
| WEAVER | 联合生成未来图像与 proprio；另有 π0.5 joint-velocity → position adapter，15Hz 控制转 5Hz WM | 它处理的是不同合同，既有 adapter 的机器人和数据不匹配本次绝对14D；不需要为本次照搬 |
| WMPO | 用目标策略在原生环境采得的成功/失败行为微调 WM，再做 GRPO | 直接处理“专家轨迹与当前策略行为不同”这个更重要的差距 |
| WoVR/PACE | 先用 base policy 轨迹训练 WM；策略更新后补采一次/低频数据再更新 WM | 固定 OpenDW 跑通后若出现 WM 高分、原生退化，这是比盲目多训策略更针对性的下一阶段 |

RISE 固定源码：[区分 action_pred/decoded L525–529](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/rlinf/models/embodiment/openpi_action_model.py#L525-L529)、[next_state L784–794](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/rlinf/models/embodiment/openpi_action_model.py#L784-L794)、[写回 L313–322](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/rlinf/envs/roborl/roborl_env_lerobot.py#L313-L322)。VLA-MBPO：[旧 state L219–225](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/src/openpi/training/libero_env.py#L219-L225)，配置边界见[此前发布审计](../robotwin_options_20261003/vla_mbpo.md)。

WEAVER [源码转换 L68–139](https://github.com/arnavkj1995/WEAVER/blob/3e41e90738fea9a7b142db6b2fd3d9ae72b7a82e/weaver/robot/actions.py#L68-L139) 有“累加速度命令”与“用动力学 adapter 预测位置”两条路径；它不能证明简单积分适用于任意 controller。[论文 §4 与附录 A1.2](https://arxiv.org/html/2606.13672v1) 明确 15→5Hz、动作表示和 learned adapter；其速度动作与可观测关节位置之间确实需要映射，和本次 absolute command → command state 的接口问题不同。若将来换成速度控制或实测 proprio 输入，再考虑这种 adapter 或联合 state WM。

WMPO [§3.2 Policy Behavior Alignment](https://arxiv.org/html/2511.09515v1) 明确成功演示不足以忠实模拟失败，需目标策略行为数据。WoVR [§4.3 PACE](https://arxiv.org/html/2602.13977v1) 将随策略变化补数据、微调 WM 作为显式阶段。它们支持进一步做策略分布对齐，**不支持“只要换上现成 WM 就能复制论文收益”**。

## 5. 本轮只增加观测，不偷偷改变策略动作

已新增本地 service 草稿中的只读 [action telemetry](../../../local_patches/opendw_smoke_20261003/tools/opendw_action_telemetry.py)。每个 C32 在 `row_started` 记录：原始 action/state 的总范围与逐维范围、起点到第一条命令的差、相邻命令绝对变化的 p50/p95/p99/max，以及 action/state 分开的归一化越界统计。

原因是官方 z-score 有 **±5 裁剪**。假如策略给某关节一个超范围命令，WM 接收到的是截断后的条件，而回填 state 仍代表原始命令，就可能产生分歧。telemetry 记录逐维 `|z|>5` 次数、总比例和裁剪对应的原始量差；它复用 `policy.norm_stats` 经官方解析后的 `mean/std`，按同一 float32 公式计算，不把 JSON 的 `global_mean`、stepwise stats 或策略自己的统计擅自混入。[统计解析 L67–104](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L67-L104)、[实际公式 L229–238](https://github.com/dexmal/OpenDW/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L229-L238)。

**此项目前是风险观测，尚无本次运行发生裁剪失配的证据。** 不新增 joint clip、不平滑动作、不改原有 gripper clip、不改下一 state。接通后把这些记录与少量生成视频、RM 曲线、原生 C32 基线放在一起看：能区分“输入超出训练范围”“接触生成失真”“视觉 RM 假阳性”，而不把所有退化归咎于 state。

服务、helper 和 CPU 契约测试已写入本地 patch，服务器部署/实际测试由执行主任务统一回执；本专题没有启动远程训练或 GPU 测试。

## 6. 决策边界

**现在清楚可实现：** absolute14D、独立归一化、命令态回填、C32/8 个未来采样点、384 整块预算。**需要实际结果判断：** Sidney 的动作幅度/失败状态是否在发布 WM 覆盖内，生成接触是否可信，视觉奖励能否区分假成功。**以后继续训 WM 前必须补齐：** bundle 对应训练 manifest、导出 state 来源、采样时间和与推理一致的数据 transform。

本轮没有找到“必须先训 state 预测器”的证据；同样没有找到“发布 OpenDW 已经针对我们 Sidney 在线失败分布完成适配”的证据。先按明确合同跑 smoke 与基础闭环，再用观测决定是否做任务/策略轨迹微调，符合现有授权且不额外拼入新算法。
