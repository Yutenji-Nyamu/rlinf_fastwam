# 深圳 current RLinf × RoboTwin π0 × RLT / DSRL：实施依据主入口

最后更新：2026-08-24  
状态：**RLT 与 DSRL current-base 增量均已实现、push。RLT Stage 1 已完成2k；Stage 2 v4于2026-08-24 19:57 CST自然完成Step250/250并exit0，最终fixed20=18/20、`global_step_250`已存在。DSRL v2也已自然完成Step200、保存最终checkpoint并exit0。**

本文件是本专题唯一当前结论和路由。用户只读本文件即可了解首版要做什么、每一点依据什么、current
RLinf 为什么需要适配，以及哪些内容明确不改。源码级细节、历史统计和逐命令证据下沉到文末分文档。

## 0. 当前结论

- 唯一共同 base：深圳已验证的 RLinf `7d07a4212ee6858cc333e1d4fab7a37256d1f839`；不追浮动 `main`。
- 两算法各建一棵独立 branch/worktree；旧 RLT 分支的祖先已含 DSRL 热路径，禁止整支 cherry-pick。
- 两算法首版均保持 **exact π0 + legacy `model_type: openpi` + RoboTwin `adjust_bottle`**，不混入 π0.5。
- RLT 已按用户决定实现：**current causal AR Stage 1 + frozen VLA/token-only + fresh Stage 1 + AC/BC/Q Stage 2**。
- DSRL：复用 current 官方 SAC/DSRL 主体，迁回旧成功 RoboTwin 的 latent、macro transition、动态 UTD20、
  compact ring、critic-only shadow 与 strict resume 合同；没有新的方法选择。
- RTC、joint VLA/RLT、RLT TD3、DSRL on `openpi_rlinf` 均不进入首版；它们以后只能作为独立 reference track。
- 实现结果、commit、资源依据和推荐2卡配置见
  [`04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md`](04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md)。
- AutoDL clean-50精确来源、三阶段逐轮采样/耗时/资源、1卡与2卡判断，以及GRPO exit255定因见
  [`05_PRE_SMOKE_DATA_RESOURCE_AND_GRPO_EXIT_DECISION_20260823.md`](05_PRE_SMOKE_DATA_RESOURCE_AND_GRPO_EXIT_DECISION_20260823.md)。
- 真实 smoke 结果、strict resume、资源峰与formal参数判断见
  [`06_REAL_SMOKE_RESULT_AND_PARAMETER_DECISION_20260823.md`](06_REAL_SMOKE_RESULT_AND_PARAMETER_DECISION_20260823.md)。
  RLT Stage 1/2 与DSRL均已按已批准packet完成smoke；formal已经用户授权并执行，不重复smoke。
- 双 job 运行层究竟改了什么、RLT Step25卡点、A/B、两行修复和v4恢复见
  [`10_RLINF_MULTI_JOB_RUNTIME_AND_RLT_CHECKPOINT_DIAGNOSIS_20260824.md`](10_RLINF_MULTI_JOB_RUNTIME_AND_RLT_CHECKPOINT_DIAGNOSIS_20260824.md)。
  已证实的是plural `build_optimizers()`漏掉既有`warmup_optimizer_state()`，使fresh Adam在保存前全空，
  DCP lazy-init又把`step`推进到1；A0原代码也能保存，所以不声称empty state确定性造成旧hang。两行修复
  让保存前state完整且保持`step=0`；formal v4已在同样pre-update条件下完整保存Step25并进入Step26，
  实践终验通过。当前完整产物、训练指标与资源终态见
  [`11_RLT_DSRL_FORMAL_ARTIFACT_AND_METRIC_REFRESH_20260824.md`](11_RLT_DSRL_FORMAL_ARTIFACT_AND_METRIC_REFRESH_20260824.md)。

## 1. 依据怎么排优先级

后文每一个实施 ID 都引用这些来源，不用“感觉应该这样”作依据。

| 标签 | 精确来源 | 决定什么，不决定什么 |
|---|---|---|
| `PAPER-RLT` | [RLT 论文 Eq. (2)](https://arxiv.org/html/2604.23073#S4.SS1) | 决定 RLT reconstruction 的公式；不决定 RoboTwin 工程接口 |
| `CUR-7D07` | official RLinf `7d07a421...`；AR fix [`8587bac4...`](https://github.com/RLinf/RLinf/commit/8587bac453f03951eeaf4a871c050bf1647e7276)、data refactor `d3aff547...`、RTC [`657faae5...`](https://github.com/RLinf/RLinf/commit/657faae51ee266cadc5d3e64f88a7dc2a4bd409b) | 决定 current API、schema、worker 生命周期与仍受维护的官方实现 |
| `OLD-RLT` | private RLT 主实现 `1a923a23...`、artifact/strict binding `3b610cb4...`、final `2b8199d8...`；相对 fork 点 `48a775db...` | 决定旧成功 exact-π0 RoboTwin 的 adapter、route、AC、truncation、resume 行为 |
| `OLD-DSRL` | private DSRL 实现 `6817c73b...`、final `48a775db...` 及旧 focused tests/smoke/formal | 决定旧成功 RoboTwin latent、macro、ring、UTD、shadow、resume 行为 |
| `SZ-BASE` | 深圳 `7d07` 上已经跑通的 exact-π0 SFT/eval/PPO/GRPO 配置与现场 | 决定模型身份、RoboTwin `H=50,D=14` 合同、深圳路径和资源；不反推算法公式 |
| `SCOPE` | 用户当前要求：基于新 RLinf 迁移旧成功 RLT/DSRL，保持可比、分支隔离 | 决定实验边界；不替代源码或论文证据 |

冲突时按以下规则处理：

1. 论文与 official 明确修复决定“当前官方算法定义”，所以 RLT AR 优先于旧 parallel。
2. current 源码决定接口和生命周期；旧文件不得覆盖 current worker、Builder、RTC 等 API。
3. official current 未覆盖的 RoboTwin 专用语义，以旧成功 port、测试和真实运行作为 behavior oracle。
4. 深圳现场只决定路径、资源和并发；A800 的路径、2卡分片或 checkpoint 不直接迁移。
5. 任何新增方法、fallback 或阈值若不在上述来源中，必须先单独说明并取得决定，不能混入“适配”。

## 2. 先看两条完整调用链

```text
RLT Stage 1:
clean-50 + exact π0 prefix → current AR encoder/decoder → z_rl artifact → strict manifest

RLT Stage 2:
exact π0 observation/prefix → z_rl + proprio + reference chunk → FullTask route
→ canonical 14D student/reference action → decode exactly once → RoboTwin
→ current Builder/transition replay → AC/BC/Q update → generic checkpoint + RLT sidecar

DSRL:
exact π0 observation → Gaussian or learned 32D latent → repeat to H=50 → flow action → execute N=20
→ current Builder stores latent lineage → first-done macro projection → compact transition ring
→ current SAC/10-Q + dynamic UTD20 → target critic/FP32 shadow → generic checkpoint + DSRL sidecar
```

## 3. 两算法共同实施点

| ID | 实施点与依据 | current 情况、适配及原因 |
|---|---|---|
| `C01` | 两棵独立 worktree（`SCOPE + OLD-RLT + OLD-DSRL`） | RLT fork 点已经含 DSRL 对共享 OpenPI/SAC 热路径的修改；都从 clean `7d07` 开始，避免祖先污染和互相覆盖 |
| `C02` | 保持 exact π0（`SCOPE + OLD-* + SZ-BASE`） | current 新 RLT 示例主要转向 π0.5/`openpi_rlinf`；首版仍用 current 中保留的 legacy `openpi`，只迁算法，不同时换模型 |
| `C03` | 接 current schema/Builder（`CUR-7D07` 的 `d3aff547...`） | current 已变为 `PolicyOutput → ChunkStepResult → EmbodiedTrajectoryBuilder → Trajectory`；首版消费它并用 fixture 验证，不预先改 schema/Builder |
| `C04` | 接 current actor/platform（`CUR-7D07`） | actor 基类、import 和设备操作入口已变；旧 worker 只逐 symbol rebase，保留 `Worker.torch_platform`，禁止整文件覆盖 |
| `C05` | 保留 RTC API、首版关闭（`CUR-7D07` 的 `657faae5...`） | RTC generic 默认关闭且是专用 eval 路径；current 没有 RLT Stage 2 或 DSRL 的完整 RTC 链，打开会变成另一个控制实验 |
| `C06` | generic checkpoint + 私有 sidecar（`CUR-7D07 + OLD-*`） | generic 能保存模型/优化器/target等，但不知道 RLT schedule anchor 或 DSRL phase/ring/shadow；复用 current 主体后挂旧已验证的私有状态 |

## 4. RLT：逐点实施依据

| ID | 实施点与直接依据 | current 情况、具体适配及为什么 |
|---|---|---|
| `R01` | current causal AR（`PAPER-RLT + 8587bac4...`） | current `rlt_token_transformer.py` 已按真实前缀 causal reconstruction；**不改该文件、不搬旧 parallel decoder**，Stage 1 fresh 重训。旧 parallel 是当时 official，不是旧 port 自创；但已被上游明确修复 |
| `R02` | exact-π0 prefix 合同（`CUR-7D07 + OLD-RLT`） | legacy wrapper仍支持 RLT；锁 image-only prefix `768×2048`、mask 与 exact-π0 preprocessing，并用真实模型 fixture 重核。不能套 π0.5 示例的 `1024/image_only=false` |
| `R03` | clean-50 与同一 norm（`OLD-RLT + SZ-BASE`） | 复用旧 Stage 1 的 RoboTwin demonstration/norm 语义，只把路径和 loader 字段解析到深圳 current；不把 online replay 或 π0.5 dataconfig 混入 |
| `R04` | frozen VLA/token-only（`OLD-RLT`） | current legacy wrapper默认可算 `rlt_loss + rlt_alpha*vla_loss`；迁 `rlt_train_vla=false` freeze hook并锁 `rlt_alpha=0`，确保仅 RLT module 训练，避免 AR 迁移无声变成 joint VLA 训练 |
| `R05` | current-AR artifact 合同（`CUR-7D07 + OLD-RLT`） | 更新 manifest/source/prefix/norm/adapter；保留 finite loss、π0 delta0、strict reload。旧 shuffled/zero-z 优劣只适用于 parallel 强瓶颈，在 teacher-forced AR 下不作硬 gate |
| `R06` | Stage 2 feature 挂点（`CUR-7D07 + OLD-RLT`） | Stage 1 actor目录只挂到 `rollout.rlt_feature_model.model_path`；小 MLP actor单独训练。防止把整套 Stage 1 wrapper误作 Stage 2 actor或漏载 token module |
| `R07` | model `H=50`、Stage 2 `C=10,D=14` canonical adapter + 一次 decode（`OLD-RLT + SZ-BASE`） | current没有 RoboTwin专用 adapter。迁 raw-template/state decode context；route 只在 canonical `[C,14]` 学习空间选择动作，之后 output transform 一次。环境动作与 replay canonical action必须分开 |
| `R08` | FullTask route（`OLD-RLT`） | current simulator route正式识别 ManiSkill critical phase，普通 RoboTwin没有该语义；在 current route abstraction 增 `FullTaskRLTRoute`：warm-up reference、之后 student，eval student-only，不伪装 ManiSkill |
| `R09` | RoboTwin transition显式 opt-in（`OLD-RLT + CUR-7D07`） | current simulator dispatch不自动识别普通 RoboTwin；补 `algorithm.rlt_transition_replay.enable` capability，默认和其他环境行为不变 |
| `R10` | 三种动作血缘（`CUR-7D07 + OLD-RLT`） | 环境执行 decoded action；计划令 `forward_inputs["action"]`/Trajectory 保存 routed canonical action；RTC top-level `model_actions`不消费。先以 fixture 建立这一目标合同，不宣称 current 天然保证 |
| `R11` | pure time-limit bootstrap（`OLD-RLT`） | current 对任意 `done` 回填当前 obs；旧成功语义要求 pure truncation用真实 next obs并按 termination mask bootstrap。只换 mask 不够，next obs也必须一起重接 |
| `R12` | current schedule/replay + AC/BC/Q（`CUR-7D07 + OLD-RLT`） | current 的 schedule、transition linker/replay和 AC 主体直接复用；`rlt_mlp_policy.py` old/current blob一致。只补 RoboTwin接缝，不切新 TD3；2卡→4卡只重算并发/分片，不改全局算法预算 |
| `R13` | strict resume（`OLD-RLT + CUR-7D07`） | generic漏 `update_step`、累计 ingest、warm-up anchor和合同 manifest。重接旧 sidecar；旧实现不保存 pending budget，checkpoint在完整 outer-step 边界，恢复时 cycle-local计数归零后按累计量重算 |
| `R14` | current YAML、工具与 focused tests（全部来源） | 新建深圳 Stage1/2 YAML；四个旧工具从 `decode(z,seq_len)` 改成 AR `decode(z,target,mask)`；迁 freeze/adapter/route/truncation/resume测试并继续跑 current causal单测，不增加经验阈值 gate |

### RLT 已锁定的选择

用户已选择 **current AR**。理由是论文 Eq. (2) 一致、current 唯一维护、无需维护已修旧 decoder，同时保持 exact π0、
数据、冻结边界和 AC Stage 2 不变。代价是旧 parallel Stage 1 checkpoint不能复用，需 fresh 重训。

这项推荐**不等于**“AR 已证明在 RoboTwin 上优于 parallel”：目前没有同数据同 seed 的受控 A/B。完整来源与
三路线比较见 [RLT Stage 1 parallel / AR 决策证据](01_ONE_PAGE_DECISION_GUIDE.md)。

## 5. DSRL：逐点实施依据

| ID | 实施点与直接依据 | current 情况、具体适配及为什么 |
|---|---|---|
| `D01` | official Gaussian/encoder/SAC 主体（`CUR-7D07 + OLD-DSRL`） | `gaussian_policy.py`、`compact_encoders.py` old/current blob相同，SAC objective可复用；**不改**这些算法本体。`openpi_rlinf`没有同等DSRL接口，首版继续 exact π0 legacy wrapper |
| `D02` | Gaussian→learned persistent phase（`OLD-DSRL`） | current缺旧 RoboTwin warm-up phase；逐 symbol迁 `dsrl_gaussian_warmup`、persistent phase及跨rank同步。global resident达到500后切 learned，不在resume后从头开始 |
| `D03` | 32D latent重复到 `H=50`（`OLD-DSRL`） | 环境仍执行 π0 flow action；学习动作是一份32D latent在H上精确重复，ring压回单个32D。不能换成14D env action或50份独立latent |
| `D04` | 64×64主相机 critic 输入（`OLD-DSRL`） | 三相机继续给冻结π0，小actor/critic只看主相机。统一预处理helper识别ring内已处理BF16样本，避免二次normalize/resize |
| `D05` | Builder动作血缘（`CUR-7D07 + OLD-DSRL`） | 环境执行 `PolicyOutput.actions`，Builder记录 `forward_inputs["action"]`；DSRL故意把后者写为repeat-H latent。首版不改Builder，用fixture证明env action、latent、RTC `model_actions`不串位 |
| `D06` | `N=20` first-done macro语义（`OLD-DSRL`） | current generic replay按trajectory，不能直接表达旧RoboTwin macro。新projector保留：first-done inclusive、删terminal后padding、成功reward=0否则-1、成功不bootstrap、纯truncation继续bootstrap、discount=`0.999^20` |
| `D07` | compact transition ring（`OLD-DSRL + CUR-7D07`） | `d3aff547...`只重组data目录，generic仍trajectory-centric。将旧固定global容量、环形覆盖、均匀有放回、cursor/resident/total/RNG exact resume迁到新 `data/storage/replay/dsrl_transition.py`，generic buffer不改 |
| `D08` | global计数 + 动态 UTD20（`OLD-DSRL`） | current固定 `update_epoch` 不等价于每条新macro做20次更新。各actor rank all-reduce local-new/resident；warm-up前只采集，之后 `updates=20×global_new` |
| `D09` | 10-Q与梯度隔离（`CUR-7D07 + OLD-DSRL`） | 保留current SAC/10-Q数学；critic clip前清actor旧grad，actor clip前清Q parameter grad。`dQ/da`已传播到actor，不改变目标，只避免FSDP全模型clip混入错误参数 |
| `D10` | critic-only FP32 shadow（official目的 + `OLD-DSRL` 私有收窄） | current仍为完整target model建立额外全模型FP32 accumulator。保留完整target/checkpoint，只将额外shadow allowlist到 image encoder、state encoder、Q head，避免冻结3.5B π0的无效FP32复制 |
| `D11` | phase/ring/shadow strict resume（`OLD-DSRL + CUR-7D07`） | generic漏phase、`update_step`、pending count、ring cursor/RNG和critic shadow。先恢复generic target，再恢复/校验shadow、phase/counter与ring；fresh formal缺sidecar fail closed |
| `D12` | `save_full_model_weights=false` 透传（`OLD-DSRL + CUR-7D07`） | current SAC主模型save未透传该YAML字段；补一个窄接缝，默认仍true保护非DSRL配置，深圳DSRL显式false，只避免额外导出完整3.5B actor，不影响DCP恢复 |
| `D13` | stochastic eval与4卡分片（`OLD-DSRL + SZ-BASE`） | 旧formal明确 `dsrl_eval_deterministic=false`，新YAML显式保持；global capacity25k在4 ranks分为4×6250，global batch256/micro64和UTD20不变。旧2卡checkpoint不能称为4卡strict resume |

DSRL 没有待选的新方法；只是保持方法级不变量，并按 current 的 schema、worker、rank和深圳路径重新接线。
资源占用和是否采用4卡属于后续启动 packet，不是代码语义。

## 6. 首版明确不改、明确延后

### No-touch

- current `rlt_token_transformer.py`、`rlt_mlp_policy.py`、RLT AC loss/schedule/replay核心。
- official `gaussian_policy.py`、`compact_encoders.py`、SAC loss/entropy/10-Q/Polyak公式。
- current `PolicyOutput`、`ChunkStepResult`、`Trajectory`、Builder、EnvWorker、runner、FSDP strategy；除非
  fixture证明真实合同缺口，才扩大修改面并补依据。
- generic trajectory replay；RLT直接复用，DSRL另建opt-in compact ring。

### 首版延后

- RTC算法集成、π0.5/`openpi_rlinf`、RLT TD3、DSRL deterministic-eval对照。
- 旧 parallel Stage 1兼容路径、旧A800 checkpoint跨2→4卡导入、generic RLT replay bitwise RNG resume。
- 任何没有论文/current/旧成功实现依据的fallback、诊断硬gate或新算法变体。

## 7. 实施后的集中验收

| 覆盖 ID | 高信息量检查 |
|---|---|
| `C03/R07/R10/D03/D05` | 同一current Builder fixture证明：RLT env decoded action vs canonical replay action；DSRL env action vs repeat-H latent；RTC字段不串位 |
| `R01-R06` | current causal单测、真实exact-π0 prefix shape、仅RLT module可训练、finite loss、π0 delta0、save→fresh strict reload |
| `R08-R13` | FullTask真值表、RoboTwin capability、pure truncation真实next obs/bootstrap、sidecar manifest roundtrip |
| `D06-D08` | first-done/reward/discount；capacity5/sample8有放回；ring RNG roundtrip；4-rank global count/capacity/UTD预算 |
| `D09-D12` | actor/critic clip allowlist、critic-only shadow命中/EMA/load顺序、`save_full_model_weights`透传 |
| 回归 | current official RLT/DSRL compose与深圳 PPO配置仍能解析，证明opt-in分支不污染既有路径 |
| 真实链 | 经批准后各做 fresh one-cycle → checkpoint → 新进程resume one-cycle；不在代码批次中擅自启动 |

## 8. 文档路由

- **当前唯一入口：本文件。**
- [实现结果、commit、配置与资源建议](04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md)：本轮完成态与下一次真实 smoke 的直接入口。
- [smoke前数据、逐轮资源与GRPO退出决策](05_PRE_SMOKE_DATA_RESOURCE_AND_GRPO_EXIT_DECISION_20260823.md)：回答clean-50、1卡/2卡、采样/更新耗时、代理与exit255。
- [真实smoke结果与formal参数判断](06_REAL_SMOKE_RESULT_AND_PARAMETER_DECISION_20260823.md)：canonical数据、RLT Stage 1/2、DSRL fresh/resume、资源和下一步选择。
- [formal启动packet与双训练并发决策](07_FORMAL_LAUNCH_AND_CONCURRENCY_DECISION_20260823.md)：RLT一键Stage1→Stage2、DSRL成功endpoint预算、持久Ray与4--5/6--7隔离合同。
- [双 formal 并发运行层与早期启动历史](08_RLT_DSRL_CONCURRENT_FORMAL_RESOLUTION_20260824.md)：persistent Ray、双worktree code sync、namespace/GPU/输出隔离与旧actor清理。
- [v3事故前训练指标、产物与资源现场](09_FORMAL_LIVE_METRICS_AND_RESOURCE_REFRESH_20260824.md)：RLT Stage1完整曲线、Stage2 v3 checkpoint卡点、DSRL成功率/优化量及双训练资源历史。
- [当前双job判断、checkpoint A/B、两行修复与v4恢复](10_RLINF_MULTI_JOB_RUNTIME_AND_RLT_CHECKPOINT_DIAGNOSIS_20260824.md)：本轮给用户阅读的短主说明。
- [RLT/DSRL formal产物、指标与资源终态刷新](11_RLT_DSRL_FORMAL_ARTIFACT_AND_METRIC_REFRESH_20260824.md)：RLT v4 Step85中间态、DSRL Step200终态、checkpoint内容与四张主图。
- [AutoDL 250/480 与深圳预算一页对齐](12_RLT_AUTODL_250_VS_480_AND_SHENZHEN_BUDGET_20260824.md)：区分两算法Stage、250正式基线、250→480延长线和深圳当前250预算。
- [RLT v4 Step250终态、资源图与单卡判断](13_RLT_STAGE2_V4_LIVE_STEP235_AND_RESOURCE_DECISION_20260824.md)：最终fixed20/checkpoint、显存/利用率和250→480边界。
- [RLT v4 Step250轻量终态包与三张主图](14_RLT_STAGE2_V4_FINAL250_LIGHT_ARTIFACTS_20260824.md)：完整日志、TensorBoard、逐步CSV、fixed-20、资源与精确resolved合同；不含checkpoint权重。
- [双 formal 细粒度操作流水账](evidence/FORMAL_CONCURRENCY_OPERATION_LEDGER_20260824.md)：每次启动、失败、停止、窄修和现场结果。
- [formal现场分析流水账](evidence/FORMAL_LIVE_ANALYSIS_LEDGER_20260824.md)：本轮只读命令、轻量下载、解析口径与最终现场刷新。
- [RLT checkpoint与双job调查流水账](evidence/RLT_CHECKPOINT_AND_MULTI_JOB_INVESTIGATION_LEDGER_20260824.md)：v3精确停止、A0/B/A′、代码修复、push和v4逐操作记录。
- [AutoDL并行与AutoDL/深圳GRPO内存机制核对](evidence/AUTODL_PARALLELISM_AND_GRPO_MEMORY_NOTE_20260823.md)：训练env是否真并发、轮末offload、锯齿与两次run的12倍simulator规模差。
- [RLT Stage 1 parallel / AR 决策与来源](01_ONE_PAGE_DECISION_GUIDE.md)：只在你想深入唯一方法选择时看。
- [逐文件 readiness 与 checkpoint/replay/shadow 工程附录](02_CURRENT_PORT_FILE_BY_FILE_READINESS_AUDIT.md)：实现时查 exact file/symbol。
- [旧/新源码、代码量与迁移详细审计](03_DETAILED_SOURCE_AND_MIGRATION_AUDIT.md)：追溯 commit、LoC 和历史判断。
- [规划审计流水账](evidence/PLANNING_AUDIT_LEDGER_20260823.md)：精确只读命令与结果。
- [共同实施流水账](evidence/IMPLEMENTATION_LEDGER.md)：现场放行、双账号与系统审计、最终汇总。
- [RLT 实施流水账](evidence/RLT_IMPLEMENTATION_LEDGER_20260823.md)；[RLT 完整 patch](evidence/patches/rlt_current_port.patch)。
- [DSRL 实施流水账](evidence/DSRL_IMPLEMENTATION_LEDGER_20260823.md)；[DSRL 完整 patch](evidence/dsrl_current_port.patch)。
- [旧 RLT 行为 oracle](../rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md)。
- [旧 DSRL 行为 oracle](../rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md)。

## 9. 当前实现终态与下一步边界

- RLT：branch `codex/sz-rlt-pi0-robotwin-ar`，实现commit `bdd875283b3f3516c439e5c79c902cf5c2da58b6`，
  formal protocol commit `f3ea5f691b99fe39e024e5571c0e6ee3d83c51b4`，checkpoint合同修复commit
  `8bbd0216113ec6eca8bbc67e79d9d642d48b8ed1`；focused + upstream AR
  `15 passed`，Hydra/compile/Ruff/diff check通过。
- DSRL：branch `codex/sz-current-dsrl-pi0-robotwin`，commit
  `4b609178d10d2534f3f972435ad972e4e015c392`；7 files，`+1456/-121`；focused `7/7 passed`，
  Hydra/compile/Ruff/diff check通过。
- canonical clean-50已按official pin与两段official converter生成：50 episodes、7,188 frames、14D。
- RLT Stage 1 physical GPU4--5 current-AR两步、Stage 2 fresh/resume均exit0；Stage 2 sidecar从
  `update_step=8`精确恢复并增长到28。显存峰Stage 1约24.1 GiB/card，Stage 2约17.7 GiB/card。
- DSRL physical GPU6--7 fresh/resume均exit0；累计1,520 critic updates、ring resident76，strict phase/ring/
  critic-shadow恢复通过；显存峰约35.0 GiB/card。
- 两棵服务器正式worktree仍clean；formal使用一套persistent Ray，RLT/DSRL按GPU4--5/6--7分卡并发。
- RLT Stage1正式2,000/2,000已exit0并保存完整`global_step_2000`。Stage2 v3仅完整到Step24；
  Step25 fixed-20=`0/20`后卡在PyTorch DCP/FSDP optimizer-state提取阶段，`global_step_25`只有
  incomplete manifest、不可恢复；该run已按owned PGID和exact namespace精确停止，事故目录原样保留。
- A0/B/A′都在DSRL继续运行、shared Ray不动时完整保存；确认原plural builder保存前两optimizer全空，
  DCP内部lazy-init后为`step=1`，构造期warmup后保存前后均为`step=0`。修复仅1 file/+2 LOC，未改算法参数。
- Stage2 v4从fresh启动，继续使用GPU4--5、8 env、250 cycles、UTD5、GB512/MB128、fixed20/save25；
  图表快照完整到Step85，Step25/50/75均完整保存且fixed20=`0/20`。14:49 CST现场到Step110，仍为
  reference warm-up：min-rank replay=`8160/10000`、global total=`16512`、`actor_switch_rate=0`、
  `update_step=0`；进程继续运行，动态step以现场为准。
- DSRL v2已自然完成Step200/200并exit0；Step65/130/195/200均有约33.59 GB strict-resume checkpoint，
  最终`update_step=104120`。16个stochastic fixed12均值78.65%，末三点均值83.33%；GPU6--7和主存已释放。

用户已授权formal。冻结参数未变：RLT Stage 1为2卡/MB16/GB32/2k、Stage 2为2卡/8 train env/250
cycles；DSRL为2卡/4 env/GB256/MB64/UTD20/200 cycles。当前运行路径、共享Ray合同与问题解决见08号
文档；事故修复与v4见10号文档，当前产物与指标见11号文档；GRPO仍不重启。
