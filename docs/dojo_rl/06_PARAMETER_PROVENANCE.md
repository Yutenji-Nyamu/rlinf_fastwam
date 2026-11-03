# 参数依据、预算单位与并行候选

2026-11-03。以下为规划，不是已运行配置。用户已接受官方Dojo π0.5、单任务、Clean GRPO、N10、H50/C50、冻结VLM/训练动作专家、双卡预算。扔瓶子优先、叠碗保留。每轮256局按本轮讨论列为候选，不声称是当前药瓶实配。

## 先固定控制来源

- 算法/优化器锚点：当前正在用的 `docs/experiments/entropy-replay-20261102/grpo-config-grpo_clean.yaml`，其真实为 **64环境×2批=128局**，G8，N10、noise0.5、global512、micro32、update_epoch2。
- 更早9月2日Clean Control确有 **64×4=256局**；但当时N5、noise0.3、global1024，不能把这些混拼后说“全部沿用当前配置”。旧证据在工作区 `docs/rlinf-shenzhen-pi05-robotwin/evidence/pi05-grpo-matched-u2-live-20260902-brief/raw/control/runtime/resolved.yaml`。
- 本候选明确采用：**现行N10 Clean算法设置 + 本轮提出的256局预算 + Dojo模型/环境契约**。这不是把旧full256实配原样迁移。

## 每项依据

| 层次/项目 | 候选设置 | 依据及适配边界 |
|---|---|---|
| 基座 | Dojo官方π0.5，seed1/checkpoint59999 | 原生Dojo历史；始终固定同一份权重和assets，不拿RoboTwin权重代替 |
| 模型后端 | 同权重JAX→PyTorch | RLinf `jax_to_openpi`路径；保留架构、tokenizer、数据变换，比较同观察/同噪声的实际动作 |
| 相机 | 头部、左腕、右腕；原图颜色/缩放/padding | 原生XPolicyLab/OpenPI；不只对齐tensor shape，保留训练时预处理 |
| 状态与动作 | 真实14维、模型32维；ALOHA排列 | Dojo模型内部左6+左夹爪+右6+右夹爪；仿真原生左6+右6+两夹爪；互逆重排 |
| 动作语义 | 臂关节delta，夹爪absolute，`adapt_to_pi=False` | 已读原Dojo `LeRobotAlohaDataConfig`与ALOHA输出；不复制Sidney绝对关节设置 |
| 归一化 | 所选checkpoint的arx_x5_sim q01/q99统计 | 模型训练契约；不采用其他项目重新计算的统计 |
| 生成与执行 | N10、预测H50、执行C50 | 用户已接受、原Dojo基线；每次生成50个动作不等于一次仿真物理步 |
| 物理/动作步长 | 任务原生dt、控制器、渲染/传感器配置 | Dojo；不因提速改物理时间、夹爪行程或末端控制器 |
| 时限 | 扔瓶子700、叠碗800原生动作 | Dojo任务源码；不照搬药瓶200动作 |
| 可训练部分 | VLM冻结、动作专家训练、无value head | 用户已接受；现行Clean `train_expert_only=true` |
| 采样探索 | Flow-SDE，noise_level0.5，训练N10 | 现行Clean/πRL；普通基线评估保留原生N10 ODE，不把新增探索造成的采样SR变化当转换错误 |
| 概率记录 | 原去噪chain/噪声位置、旧logprob和模型输入 | 复用Clean OpenPI实现；概率在模型归一化空间计算，执行用反变换动作 |
| 组大小 | G=8，同初态、不同策略随机数 | 现行Clean；布局seed与策略noise seed分离 |
| 每轮采样 | 256计划回合=32组×8 | 本轮候选，来自用户讨论及历史full预算；坏组缺测另记，禁止伪造失败补满 |
| 环境并发 | 首试32/卡，总64；4批组成一轮 | 历史原生25环境约45–48GiB提供量级依据，32尚未验收；一次接近目标测试 |
| 降级候选 | 16/卡，总32；8批组成一轮 | 如32吞吐/稳定性不合适，按实测收敛；仍256局，不改G8，不逐级从1试起 |
| 更新batch | global512个动作块，micro32/卡 | 现行Clean；双卡每rank256，累积8个micro；环境并发和训练micro互不等同 |
| 数据复用 | 每轮采到的块训练2遍 | 现行 `algorithm.update_epoch=2`；不是一轮只做2次optimizer.step |
| 优化器 | AdamW lr5e-6，β0.9/0.95，eps1e-8，wd0.01 | 现行Clean；不采用SFT的2.5e-5及6万更新预算 |
| 更新限制 | PPO/GRPO ratio clip±0.2，grad clip1 | 现行Clean，记录KL/clip比例，不额外加U、温度trick或Q |
| 其他损失 | KL系数0、entropy bonus0 | 现行Clean；γ0.99/λ0.95字段保留但当前GRPO按整局得分分组，不把它们解释成在训GAE critic |
| 成败信号 | 完整成功首次1，其余0，Score只记日志 | 成功事实来自Dojo，单次奖励/终止映射遵循GRPO；不引入VLM奖励 |
| 无区分组 | 组均值筛选[0.1,0.9]、原组内标准化 | 现行Clean；G8稀疏奖励下排除全败/全成；不按成功筛训练布局 |
| 结束mask | 已结束后不再给训练块；末块实际执行数单独保存 | 适配边界；保持现行chunk-level策略概率定义，见07，不偷偷改成逐动作独立PPO |
| 分布式 | 双卡原FSDP full_shard、按阶段offload模型 | 现行Clean；仿真进程常驻，避免每轮重新建Isaac |
| 软件/进程 | 原Dojo/Isaac环境独立于PyTorch RL环境 | 既有已跑通部署；用本机批量进程接口衔接，不直接替换共享torch/driver |
| 普通评估 | 目标128局，32固定验证布局×4策略噪声重复；中评先32局 | 项目配对评估经验，报告独立布局数；32布局需通过原生有效性检查，非成功筛选 |
| 评估频率 | 候选0/2/5，其后每5轮，中评32局、末评128局 | 沿当前每5轮，加一个早期2轮检查；新环境总ETA据首次完整周期估算 |
| 总预算/保存 | 候选200轮；先完整短测，最新完整点滚动、小日志保留，RL峰值另存保护 | 沿用户历史200轮与权重保护规则；本轮尚未授权启动，不把前20轮观察误当已批准只跑20轮 |

## 并发与时间：能保持什么、不能同时保持什么

**256局是总工作量，32/卡是同时做多少，4批是串行批数。** 32→16环境，只增加串行批数，保持32组G8；不要采用48环境×若干批后悄悄变成240或288局。

旧Dojo原生单任务25环境有真实运行依据；另一个“6任务×4环境=24”方案约72GiB，是6套Isaac加共享JAX模型，不能用其显存直接估单任务32环境。旧JAX模型仍逐环境推理。新RLinf PyTorch可合批，须在同噪声动作对照后使用；模型合批与仿真原生多环境是两层并发。

容量粗算：原25环境约45–48GiB，线性外推32约58–62GiB，只能用于选择首测点。RL的chain缓存、PyTorch模型/offload、双卡更新都有额外开销，最终看**完整采样→更新峰值、有效局/小时和reset时间**。不为了显存数字增加无效任务。

按当前actor源码把轨迹展平为“每个动作块一个训练样本”，保留固定长度padding与mask时：

| 设置 | 计划局数×最大块数 | global512、学习2遍的optimizer更新次数/轮 |
|---|---:|---:|
| 当前药瓶128局、200动作 | 128×4=512 | 2 |
| Dojo扔瓶子256局、700动作 | 256×14=3584 | 14 |
| Dojo叠碗256局、800动作 | 256×16=4096 | 16 |

这是固定布局的批次数推算，成功提前结束/全成全败组减少有效损失，不必然减少已填充batch的计算。**保留每份数据学习2遍，不等于保留每轮2次更新**；强行只更新2次将少学大量采样块。这里保持每份有效样本复用程度，接受Dojo每轮更多工作，先测墙钟再估总时长。

相同256局且都跑满，700/800动作比200动作多3.5/4倍动作；相对当前128局药瓶则是7/8倍最大动作预算。真实时间还受提前成功、原生并行、reset、取图影响，不能据此直接报耗时翻倍数。

## 源码定位

- 当前Control：`rlinf/workers/actor/embodied_fsdp_actor_worker.py::process_rollout_batch/run_training`，按块展平、整除检查、update_epoch循环、reward filter。
- 当前Control：`rlinf/algorithms/utils.py::preprocess_embodied_advantages_inputs/calculate_scores`；`advantages.py::compute_grpo_advantages`。
- 当前Control：`rlinf/models/embodiment/openpi/openpi_action_model.py`，chain与有效14维概率；无须另写GRPO公式。
- 原Dojo：`XPolicyLab/policy/Pi_05/openpi/src/openpi/training/config.py::LeRobotAlohaDataConfig` 与 `pi05_base_aloha_full_sim_arx-x5_seed_1`；配套 `policies/aloha_policy.py`。

本轮从现场正在跑的Control和已跑通Dojo工程直接读取上述源码；不是仅按较新的本地RLinf主分支猜接口。
