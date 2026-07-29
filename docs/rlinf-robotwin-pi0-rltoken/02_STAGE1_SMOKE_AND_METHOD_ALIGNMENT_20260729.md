# π0 × RoboTwin × RLT：Stage 1 smoke、数据来源与方法对齐

> 状态：2026-07-29 Stage 1 最小 smoke 已完成。
> 本文集中回答本轮用户问题；逐条命令、错误、修复和复测见
> [`evidence/IMPLEMENTATION_LOG.md`](evidence/IMPLEMENTATION_LOG.md) A031–A038。

## 1. 先给结论

本轮约定的三道最小门都已通过：

1. clean-50 单 episode 的 raw RoboTwin → Aloha → LeRobot 数据与时序合同通过；
2. S1-A：两张 A800、每 rank batch 1、两步真实 forward/backward/update、step 2
   save 和新进程 reload-only 通过；
3. S1-B：两张 A800、每 rank batch 16、global batch 32 的单步正式容量门通过，
   每卡峰值 `26,447 MiB`，无 OOM。

这说明当前代码、数据接口、两卡拓扑、token-only loss、optimizer 和 checkpoint 主链能够
运行。它**不等于 RL token 已经训好**：正式 clean-50 的 2,000-step endpoint 尚未启动，
两步 loss 也不能证明表征收敛或下游控制有效。

本轮没有启动 Stage 2、没有安装依赖、没有做 batch sweep，也没有删除磁盘内容。

## 2. 结果与配置在哪里

| 材料 | 用途 |
|---|---|
| [`source_configs/robotwin_rlt_stage1_sft_openpi.yaml`](evidence/stage1_smoke_20260729/source_configs/robotwin_rlt_stage1_sft_openpi.yaml) | 正式 2k source config |
| [`source_configs/robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke.yaml`](evidence/stage1_smoke_20260729/source_configs/robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke.yaml) | S1-A source override |
| [`s1a_resolved.yaml`](evidence/stage1_smoke_20260729/s1a_resolved.yaml) | 实际 S1-A 完整 resolved config，SHA `2aa7400e...49d` |
| [`s1b_resolved.yaml`](evidence/stage1_smoke_20260729/s1b_resolved.yaml) | 实际 S1-B 完整 resolved config，SHA `5b984a68...723a` |
| [`exact_commands.txt`](evidence/stage1_smoke_20260729/exact_commands.txt) | 两个实际启动命令和停止条件 |
| [`stage1_postcheck.json`](evidence/stage1_smoke_20260729/stage1_postcheck.json) | metric、显存、RAM、wall time、checkpoint 文件汇总 |
| [`runtime/`](evidence/stage1_smoke_20260729/runtime/) | 三次运行的 driver log 和每秒资源 CSV |

大 checkpoint 不复制进文档仓，服务器位置为：

```text
/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1/s1a/
robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1/
checkpoints/global_step_2
```

## 3. Stage 1 应该用专家数据，还是当前模型 rollout

### 3.1 论文与 RLinf 的事实

RLT 论文的 Stage 1 明确在小规模 task-specific demonstration dataset 上训练 RL-token
encoder/decoder；VLA 对 reconstruction loss 冻结，VLA 的 action SFT 是可选项。之后才冻结
VLA 与 RL-token module，进入 online actor-critic。[RLT 论文](https://arxiv.org/html/2604.23073)

RLinf 的可执行 ManiSkill 示例也是 demonstration-first：参考数据是 400 条成功
`PegInsertionSideWideClearance-v1` episode；其重建脚本用 Panda motion-planning solver
采集，并只保存可成功 replay 的 episode。[RLinf RLT 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)

因此，**Stage 1 不要求由当前 π0 自己 rollout 数据**。成功专家 demonstration 是论文与
ManiSkill 示例都支持的直接用法。

### 3.2 两类数据分别解决什么问题

| 数据 | 主要覆盖 | 本项目放在哪一阶段 |
|---|---|---|
| 专家成功 demonstration | 任务成功走廊、目标物体/接触阶段、正确动作与视觉状态 | Stage 1 学压缩 π0 内部表示 |
| frozen π0 warm-up rollout | 当前部署策略实际到达的状态、偏差和失败 | Stage 2 预填 replay、先训 Q |
| RLT student rollout | actor 改动作后产生的新状态分布 | Stage 2 online replay |
| human intervention | 人修正后的状态/动作 | 论文可选；本项目首版关闭 |

专家数据的优点是 50 条都在成功分布上，不会让 Stage 1 主要看到早期失败和无关背景；缺点是
它不覆盖当前 π0 的典型失败状态。RLT 的分工就是：Stage 1 先从 demonstration 学一个
task-specific bottleneck，Stage 2 再用 reference/student rollout 覆盖部署分布并学习 Q。

所以不应在 Stage 1 前额外要求 π0 rollout，也不应把 clean-50 当作 Stage 2 online replay。

### 3.3 clean-50 原来是做什么的

RoboTwin 的 `demo_clean` 本来就是模拟器专家成功数据。官方数据采集流程执行 task script，
结合 cuRobo planner 自动寻找可成功的随机 seed，并保存 HDF5 observation/action、
instruction、video 和辅助轨迹。[RoboTwin 数据采集文档](https://robotwin-platform.github.io/doc/usage/collect-data.html)

RoboTwin π0 文档也以 `process_data_pi0.sh ... demo_clean 50` 为训练数据转换示例。
[RoboTwin π0 文档](https://robotwin-platform.github.io/doc/usage/Pi0.html)

在本项目中它的用途是：

```text
clean-50 expert observations
-> frozen π0 image-prefix embeddings
-> RL-token encoder bottleneck
-> decoder reconstruction target
```

action 字段仍必须通过 14D、时序和 normalization 合同；但
`rlt_train_vla=false` 时，action 不进入梯度。

## 4. 50 条够不够

可以先用 50，且这是首版最简单、最诚实的方案。

它足够：

- 建立 `adjust_bottle` 单任务 RL-token artifact；
- 做 converter、loader、两卡反传和 checkpoint smoke；
- 做低预算 2k Stage 1 与后续 Stage 2 pilot。

它不够支持：

- 与 RLinf ManiSkill 400 成功 episode 的等规模复刻；
- held-out 泛化结论；
- data-scaling 规律；
- 论文的真机效果复现。

因此固定表述为“单任务、低预算 RLT 移植”。不复制 50 条凑 400，不先做
10/25/50 或 50/100/200/400 sweep。只有 2k endpoint 的 reconstruction 诊断或 Stage 2
明确失败，才重开数据量。

本轮只转换了 episode 0 验证格式，**正式 Stage 1 仍会使用全部有效 50 条**。

## 5. 单 episode 数据合同结果

版本化 canonical smoke 数据位于：

```text
/root/autodl-tmp/datasets/robotwin2/canonical/
pi0-aloha-clean50-contract-ep0-v1
```

转换事实：

| 层 | 结果 |
|---|---|
| raw RoboTwin | `T=140`，双臂各 6D + 双 gripper，三相机可解码为 `240×320×3` |
| Aloha intermediate | 139 rows，`state/action=[139,14]`，三相机 `480×640×3` |
| LeRobot | 1 episode、139 frames、50 FPS，三相机 + state + action + task |

时序逐元素检查：

```text
processed qpos[t] == raw state[t]       max_abs_error = 0
processed action[t] == raw state[t + 1] max_abs_error = 0
```

OpenPI 分布式 loader 在两个模拟 rank 上又检查了：

```text
images  [B,3,224,224] × 3
state   [B,32]
actions [B,50,32]
tokens  [B,48]
```

其中 S1-A 的 `B=1/rank`，S1-B 的 `B=16/rank`，所有张量 finite；stats 始终来自同一 π0
checkpoint，SHA-256 为
`649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a`。

## 6. Stage 1 到底在重建什么，怎么知道训好

当前直接目标是 frozen π0 的三相机 image-prefix hidden embeddings：

$$
\mathcal{L}_{\rm RLT}
=
\frac{
\sum_{b,i}m_{b,i}
\left\|\hat z_{b,i}-\operatorname{sg}(z_{b,i})\right\|_2^2
}{
\sum_{b,i}m_{b,i}
}.
$$

这里：

- prefix 是 `[B,768,2048]`；
- encoder 把它压成一个 `[B,2048]` 的 `z_rl`；
- decoder 用 `z_rl` 重建 768 个 hidden tokens；
- `sg` 表示 stop-gradient，π0 不被 reconstruction 更新；
- 不是 RGB reconstruction，也不是 action reconstruction。

`rlt_train_vla=false`、`alpha=0` 与论文高层语义兼容，因为论文明确把 VLA action SFT 写成
可选；但它与 RLinf ManiSkill 的 joint VLA+RLT 示例不同。这样做的目的不是省掉 RLT，而是
保护已验证的 RoboTwin π0 行为，只增加 task-specific representation。

“训好”分两层：

1. **直接表示证据**：固定 prefix 上的 masked MSE 相比初始化下降且有限；endpoint
   reload 后相同输入的 `z_rl/loss` 一致；true-`z_rl` reconstruction 优于 shuffled/zero
   `z_rl`，避免 decoder 忽略瓶颈；
2. **控制证据**：Stage 2 pilot 的 Q、actor deviation 和成功率/速度；reconstruction
   下降本身不能保证控制更好。

两步 smoke 只证明 loss 可算、梯度可走、checkpoint 可加载。正式 2k 后才做第 1 层判定，
Stage 2 后才能做第 2 层判定。

还要公开一个复现差异：论文描述 autoregressive reconstruction；RLinf 当前实现是无 causal
mask 的 parallel reconstruction。首版继承 RLinf，不把它写成论文逐算子严格复刻。

## 7. Stage 1 并行、batch 与实际资源

### 7.1 参数含义

`2 ranks + no_shard` 是 data parallel：

- 两张 GPU 各有完整 π0、RLT module 和 optimizer 副本；
- 每张卡处理不同样本；
- backward 时同步梯度；
- 它不是把 743M token module 拆到两张卡上。

OpenPI loader 收到 global loader batch 后再按 world size 切分，因此实际是：

| 配置 | per-rank micro | global | accumulation | 用途 |
|---|---:|---:|---:|---|
| S1-A | 1 | 2 | 1 | correctness/save/reload |
| S1-B/formal | 16 | 32 | 1 | 正式容量 |

### 7.2 smoke 实测

| run | metric | 每卡 GPU peak | 相关 RSS peak | wall |
|---|---|---:|---:|---:|
| S1-A step 1 | loss 5.15，grad 2.42 | 23,073 MiB | 39.44 GiB | — |
| S1-A step 2 + save | loss 5.21，grad 2.33 | 同一 run | 同一 run | 总计 155 s |
| reload-only | 直接 2/2、exit 0 | 20,485 MiB | 39.15 GiB | 122 s |
| S1-B micro16 | loss 5.18，grad 2.30 | 26,447 MiB | 39.09 GiB | 121 s |

两个实际训练 run（S1-A/S1-B）均 `vla_loss=0`、无 OOM/NaN/crash；reload-only 没有
新增 optimizer step 或训练 metric。S1-B 在 80GB A800 上仍约有 53GB 显存余量，
所以：

- 正式 `micro16/global32` 已通过；
- 不需要降到 micro8；
- 也不为了“吃满显存”增加 batch；
- 后续可以根据正式 run 的稳定吞吐调整，但不是 Stage 1 启动前 blocker。

### 7.3 smoke 发现并关闭的 LR scheduler 问题

smoke-time config 写的是 `lr=2.5e-5`、absolute `min_lr=2.5e-6`，但 S1-A step 2 日志为
`6.25e-8`。这不是数值噪声：RLinf 在 AdamW param group 中写 `2.5e-5`，optimizer
defaults 仍是 `1e-3`；Transformers 4.53.2 用
`min_lr / optimizer.defaults["lr"]` 得到 `.0025`，再乘 param-group base LR，floor 正好是
`6.25e-8`。

同时，RLinf 是 `optimizer.step()` 后执行 `scheduler.step()` 再记录 LR，所以日志显示的是
下一次 update 将用的 LR：

- S1-A 第 1 次 update 使用 warm-up 初始 LR 0；第 2 次使用 `2.5e-5`，因此有一次非零更新；
- S1-B 只有一次 update，使用 LR 0；它仍完整验证 micro16 的 forward/backward、gradient、
  optimizer/scheduler 调用和显存，但不能作为非零参数 delta 证据。

最窄修复只改当前 RoboTwin Stage 1 source config 为 `min_lr_rate=.1`，不改通用 AdamW
builder，也不修改历史 smoke evidence。无模型、无数据 batch、无 GPU 的 repository-function
contract 已验证：

| schedule | step 0 | step 1 | step 100 | step 1050 | step 2000 |
|---|---:|---:|---:|---:|---:|
| fixed formal 2k | 0 | `2.5e-7` | `2.5e-5` | `1.375e-5` | `2.5e-6` |

当前 source config SHA-256 为
`8340ef4e953877de510da18548d0a69802104b7b2f8218698cd0fb586b49a8f2`；机器可读证据为
[`lr_scheduler_contract.json`](evidence/stage1_smoke_20260729/lr_scheduler_contract.json)。
这关闭了正式 2k 的 scheduler blocker，不需要重跑两卡 smoke。

### 7.4 743M 为什么不等于很难训

当前 RLT module 为 `743,094,272` 参数；RLinf ManiSkill 同结构约 `744,667,136`。参数大头
来自宽度 2048、encoder 2 层 + decoder 2 层、ratio-4 GeGLU，而不是“一个 token”本身。
论文没有公开其内部 module 参数量，因此不能叫“PI 官方 743M”。

它虽大，但：

- π0 冻结，不保存 π0 optimizer state；
- 两卡每卡峰值只有 26.4GB；
- 监督信号是 dense embedding MSE，不需要等稀疏 RL reward；
- 2k 是固定低预算 endpoint，不是大规模 VLA SFT。

当前 timing 不能可靠外推：S1-B 的冷启动单步约 20.7s，而 S1-A 第二步 training 子计时受
预取/异步影响仅约 0.17s，且紧接 checkpoint。保守上界
`2000 × 20.7s ≈ 11.5h`，真实 steady-state 很可能更低；正式 run 应在前 10–20 步根据
监控重算 ETA，而不是再做一轮吞吐 sweep。

## 8. 高层语义是否对齐论文与 ManiSkill

### 8.1 对齐的核心

| 方法不变量 | 本项目 |
|---|---|
| demonstration 上学习 RL-token bottleneck | 是，clean-50 |
| reconstruction 对 VLA hidden state stop-gradient | 是 |
| Stage 2 冻结 VLA 与 RL-token feature model | 是 |
| actor 输入 `z_rl + proprio + VLA reference chunk` | 是 |
| twin-Q/target TD over action chunks | 是 |
| actor 以 Q 改进、以 BC 锚定 reference | 是 |
| reference dropout 防止只复制 reference | 是 |
| train stochastic、eval deterministic mean | 是 |
| replay 混合 VLA warm-up 与 learner rollout | 是 |
| 高 UTD 和 2 critic : 1 actor 的论文主张 | 当前 Stage 2 candidate 对齐 |

### 8.2 必须公开的偏离

| 维度 | 论文 / RLinf | 本项目 |
|---|---|---|
| base VLA | 论文 π0.6；ManiSkill π0.5 | 现有 RoboTwin π0 SFT |
| Stage 1 VLA SFT | 可选；ManiSkill 开启 | 关闭，π0 全冻结 |
| decoder | 论文 autoregressive | 继承 RLinf parallel |
| phase route | 论文人工 critical phase；ManiSkill geometry gate | full-task，`C_t ≡ True` |
| intervention | 论文可用 human；ManiSkill 有 expert route | 首版都关闭 |
| rollout/update | 论文 async | 当前 RLinf 同步 worker |
| replay stride | 论文 stride 2 overlapping chunks | 当前 chunk boundary compact rows |
| data scale | ManiSkill 400 success demos | clean-50 |

所以准确名称仍是：

> RLT × π0 × RoboTwin 的单任务低预算 RLinf 移植。

不是论文严格复刻，也不是 ManiSkill 环境参数的机械搬运。

## 9. Stage 2 参数逐项来源与解释

### 9.1 一张来源表

| 参数 | 当前 candidate | 直接来源 | 结论 |
|---|---:|---|---|
| BC/Q endpoints | `7/.05 → 2.5/.45` | RLinf ManiSkill YAML | 数值原样继承 |
| reference dropout | `.5` | 论文机制 + ManiSkill 数值 | 继承 |
| fixed std | `.002` | 论文小固定 Gaussian + ManiSkill 数值 | 继承 |
| actor/critic LR | 各 `1e-4` | ManiSkill | 继承 |
| grad clip | 各 `10` | ManiSkill | 继承 |
| batch | global/micro `512/128` | ManiSkill MLP head | 继承；不是 π0 batch |
| tau | `.005` | ManiSkill/SAC worker | 继承 |
| per-cycle cap | `400` | ManiSkill YAML | **继承，不是本项目新发明** |
| UTD / critic:actor | `5 / 2` | RLT 论文 | 论文导向 |
| `train_every_transitions` | `1` | 本项目适配 | 使当前同步 worker 实现有效 UTD5 |
| warm-up rows | `500/rank` | 本项目低预算 heuristic | ManiSkill 是 10k/rank |
| critic floor | `5k updates` | 本项目低预算 heuristic | ManiSkill 是 30k |
| BC/Q 时间轴 | warm-up 5k、ramp 10k | 本项目缩放 | ManiSkill 是 20k/50k |
| gamma | `.99` | RoboTwin primitive-step 时间尺度适配；C10 target 跨满长 chunk 使用 $\gamma^{10}$ | ManiSkill 是 `.96` |
| replay window | 15k/rank | bounded pilot heuristic | ManiSkill cache10k、sample window50k |
| `H/C/D` | `50/10/14` | H/D 来自 RoboTwin π0；C 来自 RLT chunk interface | 显式混合适配 |

这也纠正一个容易混淆的点：`cap=400` 本身是 ManiSkill 继承值；把 5k floor 分批执行后
产生约 4,600 pending，是它和本项目 5k floor/UTD5 组合后的结果。

### 9.2 UTD、warm-up、cap

当前同步 worker 的 desired critic updates 是：

$$
N_{\rm desired}
=
N_{\rm critic-floor}
+
\left\lfloor
\frac{N_{\rm new\ macro\ transitions}}
{\text{train\_every\_transitions}}
\right\rfloor
\times\text{update\_epoch}.
$$

| 来源 | `update_epoch` | `train_every` | 有效 macro-UTD | critic:actor |
|---|---:|---:|---:|---:|
| 论文 | — | — | 5 | 2 |
| ManiSkill YAML + 当前 worker | 5 | 5 | 1 | 4 |
| RoboTwin candidate | 5 | 1 | 5 | 2 |

因此当前值更接近论文，不是 ManiSkill 可执行 YAML 原值。动机是 RoboTwin interaction 和 π0
query 昂贵，希望一条 transition 多做几次小 MLP update；风险是低多样性 replay 上的 Q
过拟合、梯度长期被 clip 和 wall-clock 增大。

`warmup_min_size=500/rank` 表示两个 rank 都至少有 500 行 compact transition，通常约
1,000 global rows；然后先做 5,000 critic update，route 才允许 student 接管。这不是
把 ManiSkill 的 10k/30k 严格按 env 数缩放，只是低预算 heuristic。

`max_updates_per_train_step=400` 是一个 `collect → train` 外层周期最多做 400 次 update，
不是总训练上限。第一次 ready 时 5k floor 只执行 400，约 4,600 update 留为 pending。
满长失败 rollout：

```text
4 env × 200 primitive / C10 = 80 new macro rows
80 × UTD5 = 400 new desired updates
```

正好等于 cap，因此 pending 约保持 4,600，不承诺自动清历史 debt。cap 的目的只是避免
首次连续 5k update 把采样长时间停住。

论文没有给出本项目可直接照抄的 `500 rows/5k updates`；RLinf ManiSkill 的确切可执行值
是 `10k rows/rank + 30k updates`。在 4-env 低预算 pilot 中直接照搬会显著增加启动成本，
所以当前建议保留 500/5k，但在 Stage 2 packet 中把它标为待明确批准，而不是伪装成论文值。

### 9.3 actor loss、dropout 和 std

$$
\mathcal{L}_{\rm actor}
=
-w_Q Q_1(s,\pi(s))
+w_{\rm BC}\operatorname{MSE}(\pi(s),a_{\rm ref}).
$$

- `BC 7 / Q .05`：早期强贴 π0，避免用尚不可靠的 Q 大幅改动作；
- `BC 2.5 / Q .45`：后期仍留安全锚点，但增强回报引导；
- dropout `.5`：一半样本把 actor 输入的 reference chunk 清零，迫使 actor 也使用
  `z_rl + proprio`；BC target 不清零；
- fixed std `.002`：train 时在 actor mean 周围加入固定 Gaussian 探索，eval 直接取 mean；
  std 不学习，entropy alpha 为 0；
- LR `1e-4`：actor 与 critic 各自 Adam 的常数学习率；
- clip `10`：每个 optimizer 的 global gradient norm 上限；
- ratio 2：每两个 critic optimizer steps 做一个 actor step。

BC/Q endpoint、dropout、std、LR、clip 都有 ManiSkill 依据；只有时间轴 5k/10k 与 ratio2
不是 ManiSkill YAML 原值，前者是低预算缩放，后者来自论文。

### 9.4 `H/C/D` 与 route

- `H=50`：π0 一次 query 预测 50 个 primitive actions；
- `C=10`：actor/Q/replay 只处理并执行前 10 个，之后重新观察与 query；
- `D=14`：12 个双臂 joint + 2 个 gripper。

`C=10` 是 macro-control 长度。相对既有 RoboTwin `C=50`，它最多使 π0 query 频率增加
5 倍，但缩短 TD 决策跨度并更快闭环。

ManiSkill 可以用 peg grasp/near-hole 等几何条件定义 critical phase；`adjust_bottle` 首版
没有可靠 predicate 或人类 phase label，所以明确使用 full-task：

| mode | ready | 控制者 | replay |
|---|---:|---|---:|
| train | 否 | frozen π0 reference | 写 |
| train | 是 | stochastic RLT student | 写 |
| eval | 任意 | deterministic student mean | 不写 |

这避免编造 gate，但 student 最终负责全任务，难度高于只优化关键接触阶段。

## 10. 调用流与数据流

### 10.1 Stage 1

```text
LeRobot clean-50
-> FSDPVlaSftWorker.build_dataloader
-> OpenPI RoboTwin Aloha transforms + checkpoint norm_stats
-> OpenPIActionModel._build_rlt_prefix_cache
-> select image-only prefix [B,768,2048]
-> RLTTokenTransformer.loss
-> masked reconstruction MSE
-> optimizer only sees rlt_module.*
-> FSDP DCP + full_weights checkpoint
```

主要文件：

```text
examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
rlinf/workers/sft/fsdp_vla_sft_worker.py
rlinf/models/embodiment/openpi/openpi_action_model.py
rlinf/models/embodiment/modules/rlt_token_transformer.py
rlinf/runners/sft_runner.py
```

### 10.2 Stage 2

```text
RoboTwin raw observation
-> frozen Stage 1 feature model
-> {z_rl, proprio14, reference chunk10x14}
-> route(reference pre-ready / actor when ready)
-> canonical 14D chunk
-> one shared RoboTwin decode
-> env executes 10 primitive steps
-> compact current/action/reward/next transition
-> per-rank replay
-> twin-Q TD updates
-> actor -Q + BC updates
-> rollout-side MLP weight sync
```

π0/Stage 1 feature model在 Stage 2 始终冻结；同步的是小型 actor MLP，不是再次同步整个 π0。

## 11. 磁盘占用到底对应哪些旧实验

以下是 2026-07-29 只读审计，没有删除；A–P 精确命令、两次无副作用失败和窄修复见
[`evidence/DISK_AUDIT_COMMANDS_20260729.md`](evidence/DISK_AUDIT_COMMANDS_20260729.md)。

### 11.1 `/root/autodl-tmp/RLinf`：约 149.3 GiB

其中共享 `.venv` 约 13.7GiB，当前 RLT 正在使用，不能删。`logs` 约 135.56GiB：

| run | 归属 | 约 GiB | checkpoint |
|---|---|---:|---|
| `20260714_170304` | π0 PPO smoke | 9.69 | step 1 |
| `20260714_181545` | π0 PPO baseline | 19.38 | step 10、20 |
| `20260715_113256` | π0 GRPO smoke | 9.68 | step 1 |
| `20260715_132507` | π0 GRPO formal | 96.81 | step 10…100，共 10 个 |

保守人工候选是：GRPO formal 保留 step100、PPO 保留 step20 和所有 commands/config/metrics/
TensorBoard/log；删除 GRPO step10…90、PPO step10、两个 smoke DCP，可回收约
`116.18 GiB`。如果以后确定 PPO 也不恢复，step20 还可另回收约 `9.69 GiB`。

### 11.2 `/root/autodl-tmp/RoboTwin`：约 111.8 GiB

`policy/Motus_old_20260618_111133` 约 78.82GiB，确实是旧 Motus TTS/VTTS 与 OPD/GKD online
distillation：

- `logs_single_20260602_170538`：约 48.37GiB，40 个约 1.30GB checkpoint；
- `logs_single_20260601_082941`：约 13.30GiB，11 个 checkpoint；
- `logs_his`：约 17.08GiB，主要是 25,222 张 PNG 与 CSV/log。

如果保留每个 run 的最终 endpoint、代码/config/log，删 49 个 intermediate checkpoint，
估计可回收约 `59.25 GiB`；PNG 若已不再做逐帧可视化，可再回收约 `16.9 GiB`
（约 `18.15 GB` decimal），同时保留
CSV/log。整目录只有在决定彻底退役 Motus 线后才讨论。

`policy/ACT` 约 16.49GiB，不是 Motus；它对应 2026-05-28
`beat_block_hammer clean50` imitation：

- `processed_data` 约 14.62GiB，可由原始 50 条重建；
- `act_ckpt` 约 1.88GiB，含 best/last 与多个中间 checkpoint。

若该实验退役，可优先讨论 processed_data；checkpoint 可保留 best+last+stats+plots，四个
中间点约可回收 1.25GiB。不要删除整个 `policy/ACT`，其代码有被跟踪内容和用户改动。

另外：

- `/root/autodl-tmp/RoboTwin/assets` 与 `/root/autodl-tmp/RoboTwin_RLinf/assets`
  各约 `15.52 GiB`，是两份物理副本；
- 当前 RLT Stage 2 使用 `/root/autodl-tmp/RoboTwin_RLinf/assets`；
- 只有旧 standalone Motus/ACT 路线都退休后，才审阅旧
  `/root/autodl-tmp/RoboTwin/assets`。

本轮 S1-A checkpoint 新增约 `20.56 GiB`，所以终态可用空间从约 694GiB 降到
`673GiB`、64% used。该 checkpoint 是本轮 reload 证据，近期应保留。

## 12. 正式 Stage 1 前还需要什么

文档 QA 新发现的 LR scheduler blocker 已用 `min_lr_rate=.1` 和 CPU contract 关闭；当前
Stage 1 算法与 batch 没有未解析 blocker。但正式 2k 仍需要一个独立批准 packet：

1. 将剩余 49 条按同一 converter 合同转成版本化 full clean-50 canonical 目录；
2. 记录有效 episode/frame 数和 dataset manifest，不复制、不划 val；
3. 把正式 source config 的实际 dataset path compose 成新的 resolved YAML，并 hard-fail
   `min_lr` 存在或 `min_lr_rate != .1`；
4. 明确 2k 命令、输出目录、约 21GiB endpoint 磁盘增量和停止条件；
5. 正式 run 前 10–20 步用已有 monitor 重算 ETA；无异常继续到固定 step 2000；
6. endpoint 做 fixed-prefix loss、true/shuffled/zero `z_rl`、reload 与 manifest 验收；
7. 不按 Stage 2 成绩回头挑 Stage 1 checkpoint。

Stage 2 参数中的 UTD5/ratio2、500/5k warm-up、gamma .99 和 15k/rank replay 仍是正式
Stage 2 packet 的显式审批项，不阻塞 full clean-50 转换和 Stage 1 2k。

## 13. 本轮问题对应位置

| 用户问题 | 回答位置 |
|---|---|
| 50 条是否够 | §4 |
| 专家数据还是模型 rollout | §3 |
| clean-50 原用途 | §3.3 |
| reconstruction 是什么、怎么训好 | §6 |
| Stage 1 batch/并行/显存 | §7 |
| LR scheduler 发现与修复 | §7.3 |
| 743M 是否太大、多久 | §7.4 |
| 论文/ManiSkill 高层语义 | §8 |
| BC/Q、dropout、std、LR、clip 依据 | §9.1、§9.3 |
| UTD、warm-up、cap、pending | §9.2 |
| `H/C/D`、route | §9.4 |
| 代码与数据调用流 | §10 |
| RLinf/RoboTwin 磁盘归属与候选 | §11 |
| 正式 Stage 1 下一步 | §12 |
