# 深圳 current RLinf × RLT / DSRL：实现结果与配置建议

日期：2026-08-23  
状态：两条 current-base 代码增量已实现、轻量检查、commit 并普通 push；真实模型/RoboTwin/Ray/GPU smoke 尚未启动。

## 1. 一页结论

| 项目 | RLT | DSRL |
|---|---|---|
| current base | official RLinf `7d07a421...` | official RLinf `7d07a421...` |
| 分支 | `codex/sz-rlt-pi0-robotwin-ar` | `codex/sz-current-dsrl-pi0-robotwin` |
| commit | `bdd875283b3f3516c439e5c79c902cf5c2da58b6` | `4b609178d10d2534f3f972435ad972e4e015c392` |
| 规模 | 10 files，`+1229/-24`；其中生产代码7 files，`+509/-24` | 7 files，`+1456/-121`；其中生产代码4 files，`+890/-121` |
| 轻量检查 | diff/compile/Ruff/2×Hydra PASS；focused + upstream AR `15 passed` | diff/compile/Ruff/Hydra PASS；focused `7/7 passed` |
| 真实 smoke | 未运行 | 未运行 |
| 现场缺口 | 深圳缺 OpenPI/LeRobot processed clean-50；ACT HDF5 不能冒充 | 代码/现有模型与资产路径无已知静态缺口 |

两条补丁都是 opt-in 新配置/分支，没有改 canonical、PPO/GRPO worktree 或正在运行的 GRPO。

## 2. 推荐配置：为什么不是照抄深圳 PPO/GRPO 四卡

PPO/GRPO 是大规模 on-policy rollout：深圳配置为4卡、128 train env，并在每个 outer step 处理大量 trajectory；
RLT Stage 2 与 DSRL 是小 actor/critic + replay 的 off-policy 训练，历史瓶颈主要是仿真、主机内存或大量小 SAC
更新。给它们加到4卡会复制冻结 π0 和 worker，未必缩短墙钟，且会降低每卡有效工作量。

因此，第一条可比较主线都推荐 **2×H100**，而不是为了把卡占满而直接扩到4卡。真实 smoke 等当前 GRPO
释放主机内存后，优先用 GPU2–3；这与 GPU 型号无关，只是避免碰 GPU4–7 的现运行。

### 2.1 RLT Stage 1：current AR token reconstruction

| 项 | smoke 建议 | 首个 formal 建议 | 依据 |
|---|---:|---:|---|
| GPU | 2×H100 | 2×H100 | 旧 AutoDL 2×A800 已完整跑通；current AR 的真实显存尚未测，不先猜测扩 batch |
| micro/global batch | `16/32` | `16/32` | 旧成功配置；保持每 optimizer step 的样本量可比较 |
| optimizer steps | `2` | `2000` | smoke只覆盖 forward/backward/save；旧 formal 固定2k endpoint |
| 模型/目标 | exact π0；frozen VLA；current causal AR；`rlt_alpha=0` | 同左 | 用户已锁 current AR；其余保持旧成功算法边界 |
| 数据 | processed clean-50 的最小真实 batch | 全部有效 clean-50 | 论文与旧成功线都用 task-specific demonstrations |

旧 parallel Stage 1 在2×A800上每卡峰值 `26,447 MiB`，matched RSS约 `38.51 GiB`、cgroup anon约
`39.53 GiB`；2,000步用时 `28m54s`。这些只是旧实现的资源先验，current AR 是否更高必须由本次两步 smoke
实测。首轮不把 batch 翻倍：那会同时改变每步样本量和2k训练总样本数，不是纯资源调节。

当前阻塞点很具体：服务器没有 `ROBOTWIN_RLT_CLEAN50_PATH` 所需的 OpenPI/LeRobot processed clean-50；
已有 ACT processed HDF5 不同 schema，不能直接绑定。AutoDL 原路线现已核清为“official pinned raw clean-50
下载 -> RoboTwin official raw-to-Aloha converter -> official Aloha-to-LeRobot converter”。首条深圳基线推荐重走
这条约285 MiB raw下载路线，不先写新的XPolicyLab bridge；完整依据见
[`05_PRE_SMOKE_DATA_RESOURCE_AND_GRPO_EXIT_DECISION_20260823.md`](05_PRE_SMOKE_DATA_RESOURCE_AND_GRPO_EXIT_DECISION_20260823.md)。

### 2.2 RLT Stage 2：AC + BC/Q

| 项 | smoke 建议 | 首个 formal 建议 | 依据 |
|---|---:|---:|---|
| GPU | 2×H100 | 2×H100 | 旧成功 Stage 2 是2卡；GPU并非主要瓶颈 |
| train env | 4 | 8 | 旧4-env pilot先闭环；旧8-env×250 cycles formal已自然完成 |
| eval env | 4 | fixed 20 或另行锁定的深圳 fixed set | smoke只看机制；formal评估口径在启动packet锁定 |
| actor global/micro | `512/128` | `512/128` | current upstream RLT Stage2，旧 RoboTwin runs沿用 |
| H/C/D | `50/10/14` | 同左 | exact π0 horizon、每次执行chunk与RoboTwin canonical action合同 |
| 预算 | fresh one-cycle → checkpoint → fresh-process resume one-cycle | 250 cycles | 一次覆盖真实链和严格恢复；formal沿旧成功端点 |

旧2×A800、8 env formal 的显存峰为 `19.37/19.51 GiB`，平均 GPU util约 `30.0%/27.7%`；matched
RSS/Env RSS/cgroup anon峰约 `87.42/62.05/82.43 GiB`。250 cycles耗时 `10h46m16s`；续训的粗均值约
`147.8 s/cycle`。这说明扩到4卡的收益依据弱；如果以后要提速，应先研究2卡下仿真并发，而不是复制四份模型。

### 2.3 DSRL

| 项 | smoke 建议 | 首个 formal 建议 | 依据 |
|---|---:|---:|---|
| GPU / train env | 2×H100 / 4 env | 2×H100 / 4 env | 旧 AutoDL 成功线的精确拓扑；H100显存占比已接近一半 |
| H/N/latent | `50/20/32D` | 同左 | π0预测H50；每次执行N20；一份32D latent重复到H |
| global/micro batch | `256/64` | 同左 | 旧成功配置；generic YAML按实际actor world-size分片 |
| replay / warm-up | 25k / `4` | 25k / `500` | smoke把阈值缩到一轮能触发更新；formal保持旧方法 |
| update | `UTD20 × globally-new macros` | 同左 | 旧成功 RoboTwin DSRL；不把 current 固定 `update_epoch=200` 冒充UTD20 |
| eval | 4个机制样本 | `4 env × 3 epochs = 12`，stochastic | 旧 formal 口径 |
| 预算 | fresh step1 → checkpoint → fresh-process resume step2 | 约200 outer steps | 机制闭环后再复制旧formal endpoint |

旧2×A800 formal 常态约 `31–32 GiB/card`，DCP峰 `41,455/41,400 MiB`；换算到80 GiB H100约为
40%常态、52%保存峰，正好符合“一半左右”的资源目标。旧墙钟主要受 UTD20 的小模型更新约束：典型 train
约 `233.1s`、rollout约 `32.8s`，最近50个无eval cycle均值约 `272.5s`。增加 env 会同时增加 macro 和
20倍更新量，不会免费提速，因此不推荐首轮上4卡或更多env。

## 3. 与 AutoDL 旧 RLinf 成功实现相比，训练行为哪里变了

### 3.1 RLT

真正改变算法表征的只有一项：Stage 1 从旧 official parallel reconstruction 改为 current official causal
AR reconstruction。decoder在 causal mask 下使用前序真实 prefix embedding，因此旧 Stage 1 checkpoint不能直接
复用，必须重训；得到的 `z_rl` 也不是旧数值逐位复刻。这个变化来自 current upstream 的明确修正。

其余高层训练行为保持旧成功线：exact π0、冻结VLA只训token、clean-50、Stage 2 frozen feature、FullTask
reference warm-up→student、AC+BC/Q、H50/C10/D14、pure truncation bootstrap。current RLinf 已经吸收
schedule、transition replay和AR主体，本次只补RoboTwin action adapter/route/truncation/sidecar等接缝。

由新版 API 导致的工程变化是：接 `PolicyOutput → ChunkStepResult → Builder`，保留 current 的 expert-ref
修正、`model_actions` 与RTC字段，并挂到 current actor/checkpoint hooks。RTC 是新版能力，但首版显式关闭；
不是把旧算法改成RTC。resume sidecar只保存 current generic checkpoint不知道的schedule/contract状态。

### 3.2 DSRL

DSRL 的方法行为没有换：Gaussian→learned phase、32D latent、N20 macro、成功reward0否则-1、纯truncation
bootstrap、compact ring、warm-up500、动态UTD20、10-Q、stochastic eval全部来自旧成功实现。

变化主要是 current API 接线：ring迁到新版 `data/storage/replay`，动作血缘接 current Builder，保存/恢复接
current worker hooks。critic-only FP32 shadow不是新算法；它保留 Polyak EMA 的FP32精度，只不再为冻结3.5B π0
建立无用的额外FP32副本。current RTC字段同样保留但显式关闭。

恢复行为比旧版更严格：fresh run不变，但current分支不隐式导入缺私有sidecar的旧2卡checkpoint；缺phase、
ring cursor/RNG或critic shadow状态就拒绝续训，避免“权重恢复了、算法阶段却从头开始”。

## 4. 当前停点与下一次真实 smoke 前的条件

最终证据刷新已定位 GRPO：它停在完整 Step52，Step53 rollout 4/4 后 Ray 的用户态 memory monitor 发现
节点用量 `1914.32/2015.51 GB` 越过95%阈值，主动杀4个worker，driver随后返回255。不是数值错误、kernel
OOM或RLT/DSRL组件混淆；8卡与主存已释放，本实现批次没有重启它。完整证据见
[`../rlinf-shenzhen-pi0-ppo-rlt/23_GRPO_STEP52_EXIT255_FINAL_REFRESH_20260823.md`](../rlinf-shenzhen-pi0-ppo-rlt/23_GRPO_STEP52_EXIT255_FINAL_REFRESH_20260823.md)。

RLT Stage 1仍需取得正确的 OpenPI/LeRobot processed clean-50。Mihomo当前已恢复，刷新后的供应侧配额为
500 GiB、剩余约499.999 GiB、2026-11-23到期；但数据下载仍属于下一批明确操作，不在本代码实现批次里执行。

DSRL资源与资产前提目前满足，RLT则要先补数据。随后分别给出 exact command、visible GPUs、resolved config、
输出目录、样本/更新预算、预期资源、监控项和停止
条件，得到批准后才启动真实 smoke。

## 5. 证据入口

- [RLT 实施流水账](evidence/RLT_IMPLEMENTATION_LEDGER_20260823.md)
- [RLT 完整 patch](evidence/patches/rlt_current_port.patch)
- [DSRL 实施流水账](evidence/DSRL_IMPLEMENTATION_LEDGER_20260823.md)
- [DSRL 完整 patch](evidence/dsrl_current_port.patch)
- [smoke前数据、真实AutoDL资源与GRPO退出决策](05_PRE_SMOKE_DATA_RESOURCE_AND_GRPO_EXIT_DECISION_20260823.md)
- [逐点来源与迁移主图](00_INDEX_AND_MIGRATION_PLAN.md)
- [旧 RLT 资源与结果](../rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md)
- [旧 DSRL 资源与结果](../rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_CLOSEOUT_REPORT_STEP198_20260729.md)
