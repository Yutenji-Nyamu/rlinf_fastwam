# π0.5 在线成功 BC＋AttenA+：迁移规划

2026-09-09。状态：**研究与规划，尚未实现或启动训练**。目标是在干净在线成功 BC 上独立加入 AttenA+，与 DVAC 的重加权作比较。之前确实讨论过，见 [09-04 §8.2](history/RESEARCH_DISCUSSION_20260904.md#82-attena)；本次实读论文 v3、实时锁定作者仓库及真实 OpenPI 子模块，补齐具体接点。

## 1. 建议先做什么

**先移植作者 OpenPI 的逐动作加权公式，保持现有在线采集、成功池、模型与优化器；不把 DVAC 的两级归一化一起带过去。** 信号从同一个监督动作标签算出，额外计算只是向量范数和逐元素运算，不需要新的模型前向、critic、奖励或 denoising trace。

但有一个必须讲清的迁移点：作者 LIBERO 的动作本来是运动增量；本项目 Sidney π0.5 输出绝对关节目标。照作者代码算 normalized action 的模长，在我们这里是**动作幅值代理，不是物理速度**。建议先保留这种明确命名的作者式基线；将“原始关节命令差分”单列为运动语义适配，不能偷偷替换后称完全复现。

当前首版候选：`4/U5、B1024/micro32、LR2.5e-5、原始模型＋空池、仅成功episode入池、max_success_chunks=3`；其余沿用 clean 4/U5。方法取作者实际默认 `inverse_squared / clip_max=2 / epsilon=1e-3`。还需在实现前用少量已存成功 query 检查双臂12维信号的 clip 饱和程度；本规划没有把一个尚未看过分布的信号宣布为已验证。

## 2. 论文与作者代码究竟怎么做

论文用动作运动量作先验，较小的运动量对应较高的监督权重；公式是单个未来动作位置的标量权重乘原监督误差。π0.5 的论文结果为 LIBERO 96.85%→97.95%（+1.10个百分点），不是我们在线成功池的实验结果。它报告的是原训练数据上的加权微调；不能把已有论文收益直接当作在线 self-BC 的预期增益。[论文 v3 §3、Table 3](https://arxiv.org/html/2605.13548v3#S3)

### 2.1 真实链路和公式

设 `a[B,H,D]` 是**进入模型的、已预处理及归一化的监督目标动作**，`J` 是连续动作维度集合。作者 OpenPI 中：

$$
s_{bh}=\max(\|a_{bh,J}\|_2,\epsilon),\qquad
\widetilde w_{bh}=\operatorname{clip}(s_{bh}^{-2},1/c,c).
$$

可选的作者缩放为：

$$
w_{bh}=\widetilde w_{bh}\cdot 2/c.
$$

然后：

$$
\mathcal L=\operatorname{mean}_{b,h,d}\big[w_{bh}\,(v_\theta(x_t,t)_{bhd}-(\epsilon-a)_{bhd})^2\big].
$$

必须区分两个“速度”：`s` 取自动作标签；FM target 是 `noise-actions`，只作为监督误差目标。**权重不从随机 noise、FM target、预测误差、模型梯度或注意力矩阵计算。** 该作者实现不做相邻时间差分、不除以物理 dt；第一个动作位置与其他位置同样计算，不需要补前一个动作。[作者 JAX 实际实现 pi0.py:204–273](https://github.com/DaojiePENG/openpi/blob/fb3954ba7c0a2d0852b7f15c4445bcdb2ec7bee8/src/openpi/models/pi0.py#L204)

数据链：LIBERO data transform → Normalize → model transform → `compute_loss(actions)`；LIBERO 本身已是 delta，预注册 π0.5 关闭额外 DeltaActions。不是拿 raw qpos 直接套上面的阈值。[data_loader.py:185–190](https://github.com/DaojiePENG/openpi/blob/fb3954ba7c0a2d0852b7f15c4445bcdb2ec7bee8/src/openpi/training/data_loader.py#L185)、[config.py:326–342](https://github.com/DaojiePENG/openpi/blob/fb3954ba7c0a2d0852b7f15c4445bcdb2ec7bee8/src/openpi/training/config.py#L326)

### 2.2 应以真实配置为准的地方

| 项目 | 本次核对结论 | 对我们的处理 |
|---|---|---|
| main 仓库 | `58bea853373dfd98b24ac995bad3e203761d4629`，2026-05-14；09-09实时读取仍相同 | 锁源码，避免依赖 README 文字 |
| OpenPI 子模块 | `fb3954ba7c0a2d0852b7f15c4445bcdb2ec7bee8` | 以它的函数、注册值为主要移植依据 |
| π0.5 实际策略 | `pi0_config.py:41`、`training/config.py:774` 都是 **inverse_squared** | 首版用 inverse_squared；旧 integration.md 的 inverse 已过时 |
| 数值参数 | clip2、eps1e-3、normalize=True；alpha5只用于 exp_decay | 默认 inverse_squared 时没有额外 alpha 参数 |
| `normalize=True` | 实际只是 **除clip再乘2**，没有求样本、chunk或batch均值 | c=2时正好不改变权重；不得宣传均重1 |
| c=1 的陷阱 | 若保留作者 `normalize=True`，最终全2，并非 clean 全1 | 做零干预验证用 disabled/明确全1，不能机械设c=1 |
| JAX / PyTorch | 已改的是 JAX `pi0.py`；同提交 PyTorch `forward` 仍直接返回逐元素 MSE | 不能只打开作者 config flag 就认为我们的PyTorch/RLinf已启用 |
| gripper | 从信号范数中排除，但位置权重广播至整个动作监督误差 | 我们也排除6、13作为信号；对应位置的夹爪监督仍被加权 |

作者实际注册配置：[config.py:767–797](https://github.com/DaojiePENG/openpi/blob/fb3954ba7c0a2d0852b7f15c4445bcdb2ec7bee8/src/openpi/training/config.py#L767)；PyTorch未归约返回：[pi0_pytorch.py:317–374](https://github.com/DaojiePENG/openpi/blob/fb3954ba7c0a2d0852b7f15c4445bcdb2ec7bee8/src/openpi/models_pytorch/pi0_pytorch.py#L317)。

OFT/WAM不能当成同一个训练脚本照搬：OFT加权动作回归L1；我们保持π0.5 FM MSE。论文 WAM 在RoboTwin仅微调action head、冻结其余网络，预算也不同，不应据此改我们的冻结策略、优化器或在线预算。作者通用 PyTorch helper允许单独传FM target、padding mask、L1/MSE以及每样本归约；这只是接口可用性，不等于它已自动接入我们的actor。[作者通用helper:138–198](https://github.com/DaojiePENG/AttenA-Plus/blob/58bea853373dfd98b24ac995bad3e203761d4629/attena/velocity_attention.py#L138)、[论文Appendix E](https://arxiv.org/html/2605.13548v3#A5)

本轮也实际展开了两个子模块的helper：OFT函数默认 `inverse/clip10/normalize=False`，WAM函数默认 `inverse/clip2/normalize=True`；**这些是函数默认，不代表其正式实验命令**。WAM将 `ground_truth_actions` 与FM监督项 `target_actions` 分开，且仍硬编码first6。它们进一步说明不能从“同作者、同方法”推断所有默认及双臂处理一致；π0.5首版以实际OpenPI注册为准。[OFT:58–117](https://github.com/DaojiePENG/openvla-oft/blob/f6b63b95f3623dae4d14fd8322d399bd462f3cbf/prismatic/training/train_utils.py#L58)、[WAM:12–102](https://github.com/DaojiePENG/FastWAM/blob/1510b6c32af8336025a05bbb205940916249a3b4/src/fastwam/losses/action_loss.py#L12)

## 3. RoboTwin需要的最小适配

| 事项 | 首版建议 | 依据和限制 |
|---|---|---|
| action域 | 原有 `prepare_dagger_sft_batch` 得到的 normalized actions | 最贴近作者实际OpenPI接点；保持Sidney原mean/std、无额外delta |
| 双臂维度 | `J=[0,1,2,3,4,5,7,8,9,10,11,12]` | `[6joint,grip,6joint,grip]`，不能只用first6，不能用first12（包含左夹爪且遗漏右末关节） |
| 两臂合成 | 12维一次L2 norm；作为明确的维度适配 | 不偷偷加入双臂平均、RMS或活动臂挑选；这些会改变尺度，是后续消融 |
| 信号名称 | `normalized_action_norm` | 在绝对qpos场景不称实际速度，不声称对精细阶段已标定 |
| 第一动作 | 与chunk内其他动作完全同公式 | 作者无差分，因此没有首步补0/复制等边界设计 |
| 权重范围 | c=2时 `[0.5,2]`；无额外中心化/按权重和归约 | 保留作者式分配与loss总量；记录实际均值、分位数、饱和率 |
| 梯度 | 输入标签及w detach；FP32算范数和映射 | 稳定性保护，不引入可训练参数 |
| mask | 沿用当前真实FM mask与D14监督边界 | 排除模型padding维；不猜测成功终止前实际执行了多少插值步 |

12维L2通常比6维大，因此同clip2下可能更多位置落到下限：inverse_squared 的下限饱和条件是 `s≥√2`，上限条件是 `s≤1/√2`。**先看实际分布；若几乎全0.5，公式虽然执行了，却几乎只缩放整个loss。** 不能用“weight有限”代替有效重加权诊断。

为什么把物理速度差分放到后面：如果改成原始 command 相邻差分，其单位、双臂聚合和阈值需要重新定义；原eps/clip不能当作跨域校准。首步可取 `a0-state`，但当前RoboTwin state getter历史实现是drive target，因此它也只是命令变化，不能称测得速度；物理dt还涉及实际控制与插值时序，不能拿ODE的 `1/M` 代替。差分方案保持FM目标原样，只额外计算w；不得顺手把模型监督标签全部改成delta。具体接点和语义证据见 [LOCAL_CODE_AUDIT.md](evidence/attena-plus-plan-20260909/LOCAL_CODE_AUDIT.md)。

## 4. 哪些DVAC组件可复用

**复用通用的最后一段，替换信号和映射。**

| 组件 | 处理 |
|---|---|
| 在线采集、success replay、query有放回抽样、长度过滤 | 直接继承clean BC已有行为 |
| `masked_fm_loss(action_weights=...)` | 仅移植已测可选权重参数、shape/finite/detach检查、D14 mask；原有效元素数分母不变 |
| 模型SFT可选 `action_weights` 传递 | 可复用，无权重时保持clean路径 |
| DVAC V、tail3、历史5轮、moments、two-level、`dvac_new.pt` | 全部不需要 |
| DVAC new 的完整1024预处理hook | 作者权重逐query独立，**不需要**；直接在原transform后算，避免重复预处理 |
| 权重均值、分位数、ESS、clip上下限比例 | 可复用诊断思路；ESS只是系数集中程度，不当成独立样本数 |

建议从clean BC `01d770db3988da7862454e97434d4ff08f726fa2` 建独立分支，最小改动为：独立纯helper＋独立配置 → `forward_actor` 已准备好的actions算W → SFT接收W → 现有FM loss按位置加权。`enabled=False`保持原行为；配置禁止同时启用DVAC，恢复时校验方法身份。不要为了共享而把两种映射揉进同一段条件分支。

在线数据流建议：

`原始模型采集 → 成功且≤3chunk整episode入池 → 按query抽样 → 原input transform → 动作标签norm算W → 原FM误差×W → 原Adam更新`。

同一条query在原transform无随机动作增强时，W固定，不随batch其他样本变化。这与DVAC new的batch外层动态分配是方法差异，首版不应消除它。

## 5. 怎么公平比较两种重加权

建议围绕4/U5设三组，且**三组都打开相同长度过滤**：

| 组 | 监督权重 | 作用 |
|---|---|---|
| clean BC＋长度过滤 | 1 | 分离长度筛选收益 |
| DVAC new＋长度过滤 | 现有two-level1/1 | 当前对照，已完成结果可先作参照 |
| AttenA+＋长度过滤 | 作者式inverse_squared、clip2 | 比较另一种信号与映射的完整方法 |

统一原始模型、空池、N/U、batch、学习率、seed、环境采样、成功判定、M10/C50、fixed32协议。历史未过滤clean和旧DVAC继续画作参照，但不把全部差异独立归因于重加权。

这里比较的是**完整加权方法**，还不是纯“DVAC信号 vs 动作信号”：两者既换信号又换映射/归一化。若需分清哪部分有用，后续才做两类附加对照：同映射换信号；或AttenA权重加独立、明确的均值归一。后者会改作者算法，名称和报告应注明。

在线self-BC还会因为策略变化采到不同的成功池，这是在线方法总效果的一部分。若想单独比较权重如何分配，先在同一批冻结query上画两个W与动作运动的关系，不额外训练；最终用同seed与共同采集预算下的完整fixed曲线比较，不只取最高点。正式轮数与资源由后续实现/启动授权锁定，本轮不新增实验。

## 6. 实现前仅需解决与验证这些点

1. **确认首版语义**：推荐作者式 `normalized_action_norm`；raw command差分作为另一版适配。它的优点是忠实源码、改动少，限制是当前绝对qpos语义。
2. **确认双臂12维下不是clip常数**：用少量已有成功query，输出norm与W的分位数、上下限比例、均值；若退化再讨论尺度，不能默默调阈值。
3. **保留作者缩放还是另做均值归一**：推荐首版保留 `/clip*2`（clip2时恒等），明确不保证均重1；再以消融隔离loss总量变化。

实现后的少量有效检查：禁用/全1恢复clean loss与梯度；小张量与作者公式一致；双夹爪/padding从信号排除而损失覆盖正确；microbatch拆分不改变w；一个实际batch检查有限值、范围和饱和。随后短smoke只需完整采集→入池→Adam更新→退出并确认资源释放，不长期盯跑。

本轮源码快照、锁与细节：[evidence/attena-plus-plan-20260909](evidence/attena-plus-plan-20260909/UPSTREAM_AUDIT.md)。规划保留旧BC/DVAC实现及实验原状；没有把论文离线微调的训练预算或模型结构搬入在线BC。
