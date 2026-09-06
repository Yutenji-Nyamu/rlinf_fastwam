# RLT Stage 1 parallel / AR：决策与来源

最后更新：2026-08-23  
状态：决策审计快照；current AR 已由用户锁定并完成实现，终态见
[`04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md`](04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md)。

当前总计划与全部实施依据请回到
[`00_INDEX_AND_MIGRATION_PLAN.md`](00_INDEX_AND_MIGRATION_PLAN.md)。本文件主要展开唯一的方法选择；工程适配
章节只保留为理解该选择的背景。

## 1. 你现在只需决定一件事

两个算法都固定从深圳已经验证的 RLinf `7d07a421...` 开始，模型都先保持深圳和旧 AutoDL 使用的
**精确 π0**。当前唯一会改变实验定义、需要用户确认的是：

> RLT Stage 1 是恢复旧版 parallel reconstruction，还是采用 current 官方修正后的 causal AR
> reconstruction 并重训 Stage 1？

推荐后者：**精确 π0 不变、冻结 VLA/token-only 边界不变，只采用 current AR RLT token 目标并重训
Stage 1。**

DSRL 当前没有同等级的方法选择；其方法级不变量保持不变，但接口、路径、rank 与资源必须按 current 重新解析。

## 2. 新版为什么需要适配

新版变化分成两类：

| 类型 | 例子 | 是否改变算法 | 是否需要用户决定 |
|---|---|---:|---:|
| 工程接口变化 | storage/schema 路径、actor worker 基类、`PolicyOutput`、trajectory builder、平台 API | 否 | 否，由实现侧适配 |
| RLT Stage 1 目标变化 | parallel reconstruction 改为 teacher-forced causal AR；仍输出一个 `z_rl` token | 是 | **是，本轮唯一核心决定** |

因此，“要新适配”不等于“两个算法都要重写”。DSRL 主要是把同一算法重新接到新接口；RLT 除了接口重接，
还要明确是否接受新版 Stage 1 目标。

### 八项适配分别是怎么来的

| 项目 | current 相比旧基线发生了什么 | 为什么不能原抄 | 我们怎么接 |
|---|---|---|---|
| schema / replay | `Trajectory` 与 replay 从旧单文件重组为 `data/schema`、`data/storage/replay` | 旧 import 和旧文件内插类会直接失效 | 只改新路径和导出；DSRL 专用 ring 放进新 storage |
| rollout 数据流 | 旧大容器拆成 `PolicyOutput → ChunkStepResult → EmbodiedTrajectoryBuilder → Trajectory` | 旧代码写入 `rollout_results` 的位置已不再是数据所有者 | RLT 的当前/下一 `z_rl`、reference chunk 和 DSRL macro transition 在 builder 边界接入 |
| actor / platform / checkpoint | `EmbodiedFSDPActor` 拆到独立模块；同步与清缓存改走 `Worker.torch_platform`；SAC 的 save/load 内容本身基本未变 | 整文件搬旧 worker 会丢掉 current worker/platform 接缝；只靠 generic save/load 又会漏算法私有状态 | 继承 current base，先走 `super().save/load`，再挂旧已验证的算法 sidecar |
| RTC / 动作合同 | 旧基线已经有 singular `forward_inputs["action"] / ["model_action"]`；current 后来为 eval-only RTC 新增 top-level plural `model_actions` 与 `rtc_context` | 整文件覆盖会回退 current API；但把 RTC 当作 RLT/DSRL 默认能力也不对 | 保留 current 字段和“一次 decode”合同；首版 RLT/DSRL 明确关闭 RTC |
| RLT RoboTwin route | current simulator transition route 只显式识别 `MANISKILL_RLT`；其余落到 real-world route | RoboTwin 既没有 ManiSkill `critical_phase`，也没有 real-world 人工切换 | 增加 opt-in RoboTwin 分派及 `FullTaskRLTRoute`，复现旧 full-task warm-up→student 语义 |
| DSRL compact ring | current 通用 replay 以 trajectory 为中心；旧 DSRL 的固定 transition 容量、有放回采样和 RNG 恢复没有上游化 | 强塞通用 buffer 会改变采样单位和恢复语义 | 在 current storage 下保留独立 compact transition ring |
| critic-only FP32 shadow | current SAC 通用实现给整个 target model 建 FP32 EMA shadow | 对冻结的 3.5B π0 也复制一份既占内存又无算法作用 | 保留官方“FP32 防止小 tau 被 BF16 舍入”的目的，但 allowlist 到 critic 参数 |
| strict resume | current base 能恢复模型、优化器、target 与通用 replay，但不知道 RLT/DSRL 私有 phase/counter；这与旧 base 基本相同 | 只恢复 generic 状态会让 schedule 或 Gaussian→learned phase 悄悄从头开始 | 重新接入旧已验证 sidecar；RLT 的 pending budget 按 step-boundary 语义从累计计数重算，DSRL ring 单独恢复 RNG |

前五项主要是 current 的接口或合同变化。后三项更准确的因果链是：旧 RoboTwin 私有 port 已经实现了 compact
ring、critic-only shadow 和算法 sidecar，但它们没有被 upstream 吸收；current 只改变了 data 路径、actor 基类
和 platform 接缝，因此需要**重接**，不是重新设计。它们都不改变损失函数。

RTC 也不是普通 π0 训练/推理的默认开关。generic model 默认 `rtc_enabled=false`，只有专用 RTC eval runner/YAML
同时打开 runner 与 model 开关时才启用；训练路径不启用。current RLT Stage 2 和 DSRL 也没有官方完整 RTC 集成，
所以首版只保持 API 不回退，不把 RTC 混进算法迁移。

## 3. 已经清晰、不需要你决定的部分

| 项目 | 首版怎么做 | 性质 |
|---|---|---|
| Git 基线 | RLT、DSRL 各自从 `7d07a421...` 建独立 worktree/branch | 工程隔离 |
| 模型身份 | 两者均使用 `7d07` 内仍保留的精确 π0 wrapper：`model_type: openpi` | 保持深圳/AutoDL 可比性；不是退回旧仓 |
| RLT Stage 2 | 继续旧成功的 AC + BC/Q，不切 current 新增 TD3 | 保持算法定义 |
| RLT 任务路线 | reference warm-up 后 student 做完整任务；无 expert；不伪造 ManiSkill critical phase | 直接迁旧成功语义 |
| RLT schedule/replay engine | 复用 current 已有通用引擎，只补 RoboTwin 分派、executed-action 与 truncation 接缝 | 新接口适配 |
| DSRL 主体 | 32D latent 在 `H=50` 上重复、Gaussian→learned、macro transition、动态 UTD20、10-Q | 直接迁旧成功语义 |
| DSRL replay | 新 storage 层放专用 compact transition ring | 工程实现选择，不改变方法 |
| 恢复能力 | 功能链闭合后补算法状态恢复；任何长 formal 前必须完成 | 工程完整性，不是科学选择 |

π0.5 / `openpi_rlinf`、RLT TD3、DSRL on `openpi_rlinf` 都放到以后作为独立 baseline，不混进首轮迁移。

## 4. 唯一疑点：RLT Stage 1

### 旧版在做什么

一个 RL token 压缩视觉/语言 prefix；decoder 用一组可学习位置 token，并行重建整段 prefix。它对应旧
AutoDL 成功实验和旧 Stage 1 checkpoint。

用三个 prefix embedding `x1,x2,x3` 举例：encoder 先压成 `z_rl`；旧 decoder 输入
`[z_rl, p1, p2, p3]`。`p1…p3` 只是“第几个位置”的固定可学习标签，不含本样本内容；decoder 一次输出
`x1_hat,x2_hat,x3_hat`。因此本样本的所有信息都必须经过 `z_rl`。它是合理的强瓶颈 autoencoder，
并不是“偷看未来”。

### current 在做什么

encoder 仍把 prefix 压成 single RL token；decoder 训练时再读取“前一个真实 prefix embedding”，用 causal
mask 逐位置预测下一个 embedding。RLT 论文 Eq. (2) 明确写成
`d([z_rl, stopgrad(z_1:i-1)]) -> z_i`，因此这是论文原始的 autoregressive reconstruction，而不只是新版
代码风格；RLinf 官方提交
[`8587bac4...`](https://github.com/RLinf/RLinf/commit/8587bac453f03951eeaf4a871c050bf1647e7276)
将旧逻辑明确标为 reconstruction fix，并增加 causal-mask 单测。论文原文：
[`arXiv:2604.23073, Sec. IV-A`](https://arxiv.org/html/2604.23073#S4.SS1)。

仍用三个 token：current decoder 输入可理解为 `[z_rl, stopgrad(x1), stopgrad(x2)]`，causal mask 保证：

- 预测 `x1` 只能看 `z_rl`；
- 预测 `x2` 能看 `z_rl,x1`；
- 预测 `x3` 能看 `z_rl,x1,x2`，不能看 `x3` 或未来。

这里的 teacher forcing 只发生在 Stage 1 训练。实现仍把整段 shifted tensor 一次送进 decoder，不是 Python
逐 token 循环；Stage 2 rollout 只用 encoder 产出的 `z_rl`，decoder 不参与动作推理，所以不会增加在线推理
时延。`stopgrad` 则保证 reconstruction target 不反向改写冻结的 π0 embedding。

两种目标对 `z_rl` 的压力不同：旧 parallel 要它携带整段全部样本信息；current AR 允许后一个位置借助真实前缀，
更强调“给定过去后仍需由 `z_rl` 补充的信息”。因此不能把旧 Stage 1 数值直接当成 current AR 数值，也不能在
没有对照实验时声称哪一个在 RoboTwin 上一定更好。

### 两种路线各自有什么依据

| 路线 | 直接依据 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| 旧 parallel | AutoDL 旧分支完成 Stage 1，重建诊断有效，并接着跑出成功的 Stage 2 控制 | 它是可训练、可控制的历史成功 baseline | 它符合论文 Eq. (2)，或优于 AR |
| current AR | RLT 论文 Eq. (2) 明写只读真实前缀；RLinf `8587bac4` 以 reconstruction fix 合入，并加“改未来 token 不影响过去预测”的 causal 单测 | 它是论文对齐且由 current 官方维护的目标 | 它在 RoboTwin 上成功率一定更高 |

current 还改变了 decoder 参数结构，旧 `prefix_token_embed/prefix_pos_enc` 被 teacher-input/causal 结构替代，
所以完整旧 Stage 1 checkpoint 不能 strict load 成一个已经训练好的 current AR checkpoint。理论上可以只热启一部分
encoder，但那会制造“旧目标 encoder + 新目标 decoder”的第三种混合 baseline；首版不推荐，直接重训更清楚。

### 三种可能路线

| 路线 | 改什么 | 好处 | 代价/问题 |
|---|---|---|---|
| A. 恢复旧 parallel | 额外保留旧 decoder 兼容路径 | 最接近旧数值，理论上可继续研究旧 checkpoint 兼容 | 保留官方已修逻辑；长期维护两套 decoder；与 current 官方目标不同 |
| **B. current AR（推荐）** | 精确 π0、数据、冻结边界、Stage 2 都不变；只换 RLT token 重建目标 | 只吸收一项明确上游修正，不引入 π0.5/TD3 混杂 | 旧 Stage 1 checkpoint 不兼容，需重训 Stage 1 |
| C. official-current 全套 | 同时换 π0.5、`openpi_rlinf`、joint VLA/RLT 训练 | 最贴近 current 官方新 recipe | 已不是旧 π0 RLT 的普通迁移，无法直接归因或比较 |

推荐 B。依据是它与论文 Eq. (2) 及 current 官方实现一致，同时仍保持精确 π0、冻结边界和旧 Stage 2 AC。
这不是“AR 已被证明在 RoboTwin 上成功率更高”：目前没有同数据同 seed 的 parallel-vs-AR 对照。旧 AutoDL
Stage 1 历史运行约 29 分钟，只说明重训成本不大；深圳实际时长要在运行时实测，不能先承诺。

## 5. 已按真实调用链拆开的实施方式

### RLT

```text
config
  → 精确 π0 wrapper 提取 prefix、训练/输出 z_rl
  → rollout 取得 z_rl + proprio14 + reference chunk
  → RoboTwin full-task route 选择 reference 或 student chunk
  → canonical 14D 动作只 decode 一次并执行
  → PolicyOutput / trajectory builder 连接当前与下一状态
  → current transition replay / schedule
  → current AC critic + BC/Q actor update
  → 权重同步与算法状态 checkpoint/resume
```

- 直接复用：current RLT token 模块（若选 B）、AC objective、schedule/replay 通用引擎。
- 迁旧语义：精确 π0 冻结开关、14D adapter、full-task route、time-limit bootstrap、strict resume。
- 新接口适配：`PolicyOutput`、trajectory builder、new schema/storage、worker/platform API。
- 方法决策：只有 Stage 1 parallel 或 AR。

旧 RLT 已验证的恢复范围是模型/优化器、replay 内容、update/累计 ingest/warm-up anchor、跨 rank manifest 与
合同。它没有保存 `pending_update_budget`、`transitions_since_train`、`episodes_since_train`；checkpoint 位于完整
outer-step 边界，恢复时把三者置零，再由累计计数与 `update_step` 重算后续预算。generic replay 也没有保存当前
generator state，所以不声称未来采样序列能 bitwise 续接。若补这个能力，应标成额外可靠性增强。

### DSRL

```text
config
  → 精确 π0 + Gaussian/learned 32D latent
  → denoise 得到 H=50 chunk，执行 N=20
  → trajectory builder
  → first-done macro transition 投影
  → compact replay
  → SAC：动态 UTD20、10-Q、actor update
  → critic-only FP32 target、phase/schedule 同步与 strict resume
```

- 直接复用：official Gaussian policy、compact encoders、SAC objective。
- 迁旧语义：latent/phase、macro projection、动态 UTD、critic-only shadow、完整恢复。
- 新接口适配：new schema/storage、actor base、platform 与 checkpoint hooks。
- 方法决策：首版没有；按旧成功实现迁移即可。

## 6. 推荐结论

首轮实现建议锁定为：

1. RLT：`7d07` + 精确 π0 + current AR Stage 1 + frozen/token-only + 重训 Stage 1 + 旧 AC Stage 2。
2. DSRL：`7d07` + 精确 π0 + 保持旧成功方法不变量，按 current 接口、路径和 rank 重新接线。
3. π0.5、joint RLT/VLA、TD3 均另开参考轨，不进入首轮。

在用户确认 RLT 第 1 条后，再生成精确文件级实施 packet；本页不授权创建服务器 worktree、修改代码或运行。
