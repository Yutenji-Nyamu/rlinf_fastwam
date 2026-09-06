# 历史讨论：RLinf / RoboTwin / π0 成功过滤在线 BC 与 DVAC

本文保留 2026-09-04 首轮广搜与讨论，不再维护为当前计划；其中建议与预算未获实施批准。
当前唯一入口：[干净在线 BC 独立上下文](../00_RESEARCH_AND_PLAN.md)。

日期：2026-09-04。状态：研究与讨论，不是已批准的实施／训练合同。

## 1. 结论与当前范围

先实现成功过滤在线 BC，再在同一条监督训练链上接 DVAC，这个顺序合理。
可以复用 RLinf 的主体设施，不需要为了在线 BC 换到另一个训练框架。
但“RLinf 已有 DAgger 和 only_success”不等于 RoboTwin π0 已有可直接启动的自主成功 BC 配方。

建议首版：**同步按轮采集；按 episode 成功筛选；累计成功回放；从同一个 π0 SFT 起点持续做原生 FM 更新**。
先不引入老师、Q/V、RynnValue、额外环境随机化或新的渲染依赖。用户确认前不确定最终超参数、不启动实验。

本次已读用户提供的[另一窗口 RLinf 能力整理](C:/Users/86136/Documents/seek/0904_rlinf_current_weighted_bc_capabilities.md)，把它作为检索线索；重新核对了官方源码。
RLinf 调研锁为 `dc9b87cc49334c7516487ead68ebeb060fd7c090`（09-04 11:16:55 UTC），不是对深圳运行版本的升级。
本轮没有登录服务器，也没有刷新训练、GPU、checkpoint 或服务器 Git；根交接中的运行数字仍是旧快照。
源码锁、逐项证据、未验证边界见[源码审计](../evidence/ONLINE_BC_SOURCE_AUDIT_20260904.md)。

## 2. 哪些近年工作最值得借

检索以 2025-09 至 2026-09 为主。区分“直接匹配”“正式发表”“代码可用”，不把近期预印本都称为高影响力成果。

| 工作 | 时间／地位 | 实际可借的东西 | 不能误认为什么 |
|---|---|---|---|
| [Hi-ORS](https://arxiv.org/html/2510.26406v1)／[代码](https://github.com/hiors-project/hiors) | 2025-10，预印本；清华深圳／腾讯 Robotics X | π0 在线成功轨迹池→FM 监督更新；最直接的 π0 参考 | 论文含人工纠正；不是已经验证纯自主 RoboTwin 同样有效 |
| [SEIL](https://arxiv.org/html/2509.19460v1)／[代码](https://github.com/Jasper-aaa/SEIL) | 2025-09，预印本 | LIBERO 中 train→record→success/select→train 的完整分轮思路 | backbone 是 BAKU，不是 π0；还有 EMA、位置扰动和 selector |
| [SARM／RA-BC](https://qianzhong-chen.github.io/sarm.github.io/)／[LeRobot 接法](https://huggingface.co/docs/lerobot/en/sarm) | ICLR 2026；Stanford／Berkeley／xdof.ai | 进度标注→chunk 权重→π0/π0.5 加权 BC，已有维护中的实现 | 它提供的不是我们要的完整自主采集外循环 |
| [AttenA+](https://arxiv.org/abs/2605.13548)／[作者代码](https://github.com/DaojiePENG/AttenA-Plus) | 2026-05，预印本；已找到作者代码 | 动作时间步权重及 OpenPI 接口；接在未归约动作损失上 | README 的 NeurIPS 2026 占位 badge 不能当录用证明 |
| [RynnValue](https://github.com/alibaba-damo-academy/RynnValue) | 2026-08，较新预印本／作者代码 | 外部进度信号、离线 IQL Q/V→加权 π0 FM 的实际桥接 | 现成 offline IQL 不等于 online IQL；在线脚本也不等于纯成功 BC |
| [RL-100](https://lei-kun.github.io/RL-100/)／[代码](https://github.com/Lei-Kun/RL-100) | Science Robotics 2026（作者主页和项目已确认） | 数据轮次、合并、来源 manifest、BC→RL 的工程组织 | 主要是 diffusion/flow RL，不是 π0 成功 BC 的最小主体 |
| [What Matters for Batch Online RL?](https://arxiv.org/html/2505.08078v1) | ICLR 2026；Stanford；初稿 2025-05，早于一年窗口 | 系统比较 all-data IL、filtered-IL、value-based learning；预算和方法边界很有参考价值 | 论文说将开源；本轮未找到可核实的作者代码入口，不能当现成代码源 |

补充检索了 iRe-VLA 等 RL/SFT 交替路线，但不列为首版主体：时间更早、方法更重，当前问题已有更直接的实现依据。

### 2.1 最贴近的 Hi-ORS，代码实际做了什么

`reject_sampling_pi0.yaml` 虽挂在 `sac_pi0` agent 下，实际设 `rl_steps=0`、只启用 online successful data 的 SFT。
actor 先临时收齐 episode，成功后将其中记录放进 successful buffer；learner 从池中采样、调用 π0 FM。
这证明“成功采集＋π0 BC”有直接公开实现，而不是只存在于论文示意图。

不能整库复制：其代码有双臂任务专用的左臂七维 loss、no-op 筛选、人工干预和异步服务。
我们应借成功池和监督目标，不照搬它的任务维度、过滤阈值或部署设施。
论文中移除人工纠正的困难任务结果也提醒：自主 BC 的前提是初始策略能采到有用成功样本。
依据：[固定配置](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/config/algo/reject_sampling_pi0.yaml)、[收集与 learner](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/examples/train_rlpd.py)、[SFT loss](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/serl_launcher/serl_launcher/agents/continuous/sac_pi0.py#L464)。

## 3. RLinf 有什么，以及不能直接拼接的地方

### 3.1 可以复用

1. Env / Rollout / Actor 分工、参数同步、评估、checkpoint 和现有 RoboTwin rollout。
2. `EmbodiedDAGGERFSDPPolicy` 的 SFT backward、optimizer、累计数据训练结构；它跳过 advantage 计算。
3. π0 的 `prepare_dagger_sft_batch` 和 `sft_forward`：已经能以观测＋动作 chunk 计算 FM loss。
4. DAgger 的在线 LeRobot 导出、成功判定、内存回放和归档组织，可作为接口参考。

专家是可选的：rollout 仅在 `expert_model is not None` 时执行专家分支或重标注。
但普通 DAgger replay 接收端调用 `extract_intervene_traj(mode="all")`，因此**只删专家配置并不能得到普通自主 BC**。
也不应该把自主动作伪装成 expert/intervention 来绕过这个判断。
依据：[rollout](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/rlinf/workers/rollout/hf/huggingface_worker.py#L506)、[actor](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/rlinf/workers/actor/fsdp_dagger_policy_worker.py#L300)。

### 3.2 新发现：RoboTwin 与逐帧 LeRobot builder 的粒度不匹配

固定源码中：

- `robotwin_env.chunk_step()` 一次执行整块动作，`obs_list` 只放一个 chunk 后的观测。
- 它返回长度 C 的 term/trunc，结束标志放在最后一格。
- `EmbodiedLerobotTrajectoryBuilder.append_chunk_episode_data()` 则按 `len(obs_list)` 循环，读取同索引的单个动作和 term/trunc。

所以在 C50 下直接连接，源码路径只走索引0：取首个动作，却得到块后观测，也读不到放在49的终止标志。
这是**源码级接口不匹配**；没有在服务器启动此组合，不能称为本次实测的运行故障。
同样，`only_success` 要求 `is_success && done_by_term`，也不适合不加分析地继承 `ignore_terminations=True` 的 GRPO 配置。
依据：[RoboTwin 322–390](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/rlinf/envs/robotwin/robotwin_env.py#L322)、[builder 737–903](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/rlinf/data/schema/embodied_trajectory_builder.py#L737)。

### 3.3 首版实现方向：沿现有 query/chunk 粒度

```text
同一个 π0 SFT 起点
    ↓
冻结本轮行为策略，收集 N 次完整尝试
    ↓
episode 成功？──否→计入尝试预算，不进入本方法的 BC 池
    │是
    ↓
累计成功池：(决策前观测，实际执行动作 chunk，有效长度)
    ↓
采样 minibatch → 原生 π0 FM → 同步新策略 → 下一轮
```

建议在独立 `codex/` 工作树新增明确命名的 success-BC actor／collector 小适配，复用现有 supervised update 和 replay。
不重写 scheduler，不把 BC 硬塞进 PPO ratio，不为了适配 LeRobot 而新增每个 primitive action 的渲染。
后续确需逐帧数据时再补一个明确的 primitive collector；不能用重复末帧伪造逐帧图像。

实际执行前必须刷新目标服务器源码与依赖，核对哪些上述组件已在现役 source-lock 中。
**调研基线 dc9b87c 不自动成为部署基线；优先窄移植必要组件，不整体追 main。**

## 4. “在线 BC”究竟在学什么

普通 BC 是学示范动作；这里的示范来自模型自己已经成功的真实执行。
在线体现在每轮新数据由当前策略收集，不要求边走一个动作边反向传播。

对 π0，训练目标应保留它原生的 **flow-matching loss**，不是“完整跑一次去噪再对最终动作直接 MSE”。
记 `ell_FM(o,A)` 为原生 FM 损失，则成功 BC 学：

`L_success_BC = E[(o,A) ~ cumulative_success_buffer] ell_FM(o,A)`。

训练时重新采样 FM noise/time；监督的 A 是已执行的成功动作，不重新查询一个老师动作来替换。
动作先经过现有 action transform／normalization；物理动作与 normalized model action 都保留来源语义，不能二次归一化。

### 4.1 为什么可能提升

同一个初始条件附近，当前策略有的采样成功、有的失败。只增加成功行为的概率，能把“偶尔做对”变成“更常做对”。
多个成功路径、恢复行为也能进入训练数据，改善纯离线数据没有覆盖的状态。
因此初始策略不必是完美老师，但必须已有一定成功覆盖。

### 4.2 为什么也会较早饱和

- 成功轨迹里的所有动作不一定都好：绕路、停顿、先犯错后补救也会被一起学习。
- 它不直接利用失败轨迹中的好前缀；难任务若几乎没有成功，会缺少新监督。
- 反复采到相同成功模式，继续模仿可能只提高拟合，不能扩大成功区域。

这些是方法机制的局限，不是预先认定你的 π0 会失败。Batch Online RL 的系统研究中，value-based 方法更会利用新增数据，但那也不是本项目或大 π0 上的普遍结论。
依据：[该论文 §4–5](https://arxiv.org/html/2505.08078v1#S4)。

## 5. 每轮多少数据、多少轮：先统一计数

### 5.1 有依据的预算，不混淆口径

| 参考 | 采集／轮数 | 训练预算与解释 |
|---|---|---|
| Batch Online RL，仿真 | 200 rollouts/轮，10–20轮 | 2,000–4,000次尝试；模型不是 π0；不能照搬其 LR=3e-4 |
| SEIL 主配置描述 | 两个模型各25次 rollout/轮，再选15条扩充池；消融报告1/4轮并比较到第5轮 | 这是少样本多任务 BAKU；不同实验的选择数不同，不能把 selector 示例 `num_to_save=25` 当论文统一预算 |
| Hi-ORS 公开配置 | 异步持续采集；successful buffer 至少10条**记录**后开训 | `rl_steps=0`；`sft_steps=100`只是全SFT循环模式；每50 learner steps发布参数；不是每采50条就训100步 |
| RLinf RoboTwin DAgger 示例 | 64 env，rollout_epoch=1，200-action窗口，max_epochs=8000 | GB1024、update_epoch=1；有专家；64个窗口不自动等于64个完整episode |
| LeRobot RA-BC 文档示例 | 固定数据集，无在线采集轮数 | batch32、40,000 optimizer steps；不是每个在线轮要做40,000步 |

来源：[Batch Online RL §4/Appendix B](https://arxiv.org/html/2505.08078v1#S4)、[SEIL §V-A3](https://arxiv.org/html/2509.19460v1#S5.SS1.SSS3)、[Hi-ORS 配置](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/config/algo/reject_sampling_pi0.yaml)、[RLinf 配置](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml)、[RA-BC 示例](https://huggingface.co/docs/lerobot/en/sarm#step-5b-train-policy-with-ra-bc)。

### 5.2 我们可以怎么起步（候选，不是训练合同）

为了衔接已有实验，**每轮256次尝试**是自然候选，并行数保持目标 π0 Control 的设置，通过串行批次收齐。
若采用100轮，总交互预算是25,600次尝试；若先讨论20轮规模，则是5,120次。两者都不是“论文标准轮数”。

推荐固定**尝试数**，而不是“每轮必须凑齐256次成功”：后者会让低成功率方法消耗更多交互。
例如仅用于算账，成功率20%时，一轮256次约留下51次成功，而不是256次。

对 H=200、C=50、每次尝试最多4个 query 的候选协议：

- 每轮最多1,024个有效 query 记录；不能把它叫1,024条轨迹。
- 20%成功且成功episode接近全长时，新增成功 query 大约205条；提前结束则更少。
- 若 GB=1024、每轮 U=2 optimizer calls，则每轮呈现2,048个 chunk 样本，可从累计池有放回采样；这**不等于成功数据只遍历2次**。
- 该 `GB1024/U2` 只是与已有 GRPO 更新计数接近的算账示例，尚不能判断是否适合 BC。BC 的 LR、更新次数、warmup 必须按 BC 自己的 loss／有效数据量确定。

我倾向累计成功池、每轮固定 U 次更新，先不混回原始专家数据。这样第一条方法就是清楚的“自主成功 BC”。
新增多少成功数据、池里有多少独立样本、抽样呈现多少次，应同时记录。
如果以后保留较近窗口或混专家数据，应成为明确的方法／配置变化，不悄悄加入。

从原 SFT 初始化一次，之后延续模型、optimizer 和 scheduler，不每轮重新从 SFT 开始训。
初始权重建议与 π0 PPO/GRPO 对照一致；不要把已训练 GRPO 权重当作 BC 起点而仍称同起点比较。

## 6. 必须一次做对的少数实现细节

1. **标签与对齐**：保存 o_t 和它产生的执行 chunk；不用块后 o_(t+C) 配前一块动作，不用老师重标注。
2. **episode 与终止**：用任务的 episode 成功判据；timeout不自动等于失败或成功；成功后的重复动作不进入监督。实际执行长度／有效 mask 要来自环境可核实信息，不能把未执行后缀当成功证据。
3. **动作坐标与维度**：保持 π0 的 C、有效14维、三相机、语言、delta/absolute 和 norm；模型补齐的维度与越界位置不计 loss。
4. **采样模式**：现有 DAgger 强制 `mode="eval"`，它与 GRPO 的 train sampler 可能不同；需显式选择 BC 的行为采样器。eval mode 不自动等于完全无随机，不能只看名字。
5. **成功池及恢复**：不启用 GRPO 的组内优势或混合成功组过滤；100%成功组也有 BC 价值。保存模型同时保存池的来源/索引/轮次，否则 resume 后训练数据分布会改变。
6. **多卡归约**：各 rank 无成功时要共同跳过或从已有池采样，不能部分rank继续backward；时间mask和权重的分母要跨 microbatch/rank正确汇总。

第一批检查只需围绕这些实际边界：真实 episode 导出复读、单次 FM 梯度更新、一次保存恢复。
不预设长串 smoke 或大量候补实现；本轮没有执行这些检查，启动前另列精确配置与授权。

## 7. 成功过滤＋DVAC：接在哪里，为什么值得试

基线回答“这条经验是否可学”；DVAC 再回答“成功经验中的哪些动作位置值得投入更多学习”。
这是比“所有成功动作一样重要”更细的训练信号，适合成为下一步研究。

最小接口是未归约 FM loss `ell[b,h,d]` 加上 detached 的时序权重 `w[b,h]`：

`L_DVAC = sum(valid[b,h,d] * w[b,h] * ell[b,h,d]) / sum(valid[b,h,d] * w[b,h])`。

`w=1` 精确退化到带相同有效mask的成功 BC；成功数据、模型、采样器、optimizer预算不变。
权重由哪个模型、在哪个阶段算，以及权重映射／截断／归一化，留到下一次 DVAC 专题讨论后定。
不自动沿用过去 PPO/GRPO 的 advantage 权重，也不自动把旧 RLT 的“向冻结π0锚定”目标搬来。

我倾向首个候选是在 rollout 生成对应 action chunk 时计算／记录 DVAC，让信号与真正成功的那次生成对应；BC 内部用于 FM 训练的新噪声不应悄悄改变数据的 DVAC 标签。
如果选择训练时重算，则它是另一种“当前学习难度”语义，需要明确说明。

**机制边界**：DVAC 的不稳定性本身不等于任务价值或严格因果 credit；episode 成功提供总体结果证据，DVAC 提供内部学习分配线索。
“难但做对的位置多学一些”是可研究的机制，不必为首版加入额外 critic 或反事实验证系统。
若大量高权重集中在无意义抖动/停顿，再针对观察到的问题修正信号解释。

## 8. 后续 RA-BC、AttenA+、IQL 怎样接入

| 方法 | 额外信号 | 接入层与范围 |
|---|---|---|
| 成功 BC | episode success | 成功池＋原生 FM，无 critic |
| 成功＋DVAC | 对应生成过程的内部信号 | 成功池内的时序 FM 权重；具体映射待讨论 |
| 成功＋AttenA+ | 动作运动强度 | 时序 FM 权重；作者代码可参考 |
| RynnValue＋RA-BC | chunk 前后外部进度差 | sample/chunk 权重；可复用同一 collector/BC actor |
| RynnValue＋IQL | 奖励／进度、Q、V、next_obs、done | 新增价值学习与加权 FM；复杂度实质高于前三种，不是只换weight文件 |

### 8.1 RA-BC

LeRobot 已有独立 `SampleWeighter` 和 `RABCWeights`，再以 `reduction="none"` 接训练损失。
进度改善明显的 chunk 给满权；小幅改善给软权；倒退置零。统计与阈值取决于数据单位。
RynnValue 输出剩余秒数时，进度改善可表示为 `remaining(t) - remaining(t+C)`；不能反号，也不能直接套 SARM 的0–1进度阈值0.01。
在 query 数据里，C50 是50个 primitive actions，通常只跨**一个**相邻 query；不能错误地向后取50个 query。
如果给成功池加 RA-BC，应命名为“成功＋RA-BC”；若学全部rollout，则是另一个数据选择条件。
依据：[LeRobot 权重实现](https://github.com/huggingface/lerobot/blob/3f2c29ef7e44b1ddccbcda3b6a63939e53639e9e/src/lerobot/rewards/sarm/rabc.py)、[RynnValue](https://github.com/alibaba-damo-academy/RynnValue)。

### 8.2 AttenA+

已找到作者公开的 `attena/velocity_attention.py` 和 OpenPI 集成文档；不再列为“没找到代码”。
但它用选定动作维度的模长作为 speed，再做逆速度类映射。
在 RoboTwin 绝对关节目标、相对初态 delta、标准化动作上，动作模长不自动等于物理速度；双臂也不能只抄前6维。
首要适配是明确量纲与运动维度，而不是拷贝公式后把权重都截到上限。
其 `weights / clip_max * 2` 也不是严格均值一；与 DVAC 对比时要区分相对分配和整体 loss 缩放。
依据：[作者实现](https://github.com/DaojiePENG/AttenA-Plus/blob/58bea853373dfd98b24ac995bad3e203761d4629/attena/velocity_attention.py#L106)。

### 8.3 IQL 与更大实验矩阵

RynnValue 已有 `pi-rl/scripts/train_iql.py`：Q/V 更新→advantage weight→π0加权 FM，是更直接的 VLA 接法。
其在线 Franka 脚本含 EXPO-FT／SAC 等不同路径，不能根据 `train_online` 名字认为已经实现在线 IQL 或纯成功 BC。
后续 online IQL 要让每轮新transition进入Q/V训练，处理跨chunk reward／discount／terminal；原则上应使用失败经验，不沿用成功池作为唯一critic数据源。
RLinf 原生 IQL 数学可以参考，但其现成 D4RL/MLP 离线 recipe 不能直接替代 π0 actor。
依据：[RynnValue IQL](https://github.com/alibaba-damo-academy/RynnValue/blob/10e0d333f5f3811d0d130587e50f1faf48da49e5/pi-rl/scripts/train_iql.py)。

PPO/GRPO、在线 BC、SAC/RLT 应共享任务／初始化／数据计数和评估协议，而不是强行共享同一个 loss。
下一步建议先讨论两件事：确认累计成功池与原 SFT 起点；确定每轮尝试数和 BC 自己的更新预算。
然后才能给出窄移植的文件级实施合同；DVAC 接口预留即可，不抢先规定最终方法。
