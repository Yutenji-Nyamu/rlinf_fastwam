# Fast-WAM + RoboTwin + RLinf + GRPO/PPO 总文档

更新时间：2026-08-22

## 0. 当前路由：深圳 H100 × current RLinf GRPO 迁移规划

official standalone与四任务64条采集已经完成，当前讨论已进入“把Fast-WAM接入深圳current RLinf并训练
`move_stapler_pad` GRPO”的新阶段。短主文档是：

- [`12_SHENZHEN_CURRENT_RLINF_GRPO_PORT_PLAN_20260831.md`](12_SHENZHEN_CURRENT_RLINF_GRPO_PORT_PLAN_20260831.md)：
  source lock、旧adapter真实增量、current必需适配、128/256预算换算、联合环境、最小实施面与待决策项。

下列 `09`–`11` 继续作为已经完成的 official standalone 事实源；旧 AutoDL 集成材料只提供可追溯依据，
不再作为深圳 current 训练的执行入口。

## 0.1 已完成路线：深圳 H100 × official current-HEAD standalone

用户当前任务是在 `SZ-H100` 单独部署 Fast-WAM official RoboTwin inference。该任务不复用下文旧
AutoDL runtime，也暂不进入 RLinf 集成；当前唯一执行入口是：

- [`09_SHENZHEN_OFFICIAL_STANDALONE_PLAN.md`](09_SHENZHEN_OFFICIAL_STANDALONE_PLAN.md)：
  current official source/release lock、旧→新差异、隔离布局、资产复用边界、网络/存储预算、official
  evaluator 命令与验收。
- [`evidence/SHENZHEN_STANDALONE_OPERATION_LEDGER_20260822.md`](evidence/SHENZHEN_STANDALONE_OPERATION_LEDGER_20260822.md)：
  从首条服务器操作开始的细粒度流水账；未执行项保持 `PLANNED`，实际结果只按 live 输出填写。
- [`10_SHENZHEN_S4_S5_EXECUTION_PACKET.md`](10_SHENZHEN_S4_S5_EXECUTION_PACKET.md)：
  source-locked render/expert/evaluator命令语义与本轮实际结果。
- [`11_SHENZHEN_FROM_ACT_TO_OFFICIAL_FASTWAM_NOTES.md`](11_SHENZHEN_FROM_ACT_TO_OFFICIAL_FASTWAM_NOTES.md)：
  从native RoboTwin/ACT成功到Fast-WAM official推理的最短复现摘要；区分official路径、深圳隔离/复用、
  自身额外操作问题与真实MPLib×NumPy兼容修复。
- [`../rlinf-shenzhen-pi0-ppo-rlt/13_DVAC_SIGNAL_PI0_FASTWAM_OBSERVATION_PLAN.md`](../rlinf-shenzhen-pi0-ppo-rlt/13_DVAC_SIGNAL_PI0_FASTWAM_OBSERVATION_PLAN.md)：
  current π0与official Fast-WAM共用的DVAC raw signal、Position/Residual/S/I离线分解、Fast-WAM
  `H=32/M=10/C=24`插桩点和视频对齐边界；两侧default-off实现已完成，真实采集未启动。
- [`evidence/SHENZHEN_DVAC_IMPLEMENTATION_LEDGER_20260822.md`](evidence/SHENZHEN_DVAC_IMPLEMENTATION_LEDGER_20260822.md)：
  Fast-WAM exact-7 telemetry实现、CPU检查、独立审查、local commit、独立Git push边界与P0草案的逐操作账。

深圳 source lock 是 Fast-WAM `7faa71108368fbb3b6885649f112af607427a2d4`，HF release revision 是
`8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`。旧 `07_OFFICIAL_STANDALONE_RUNBOOK.md` 及本文件下面
第 1–10 节主要描述 `AUTODL-A800@45d8e145...` 的历史 standalone/RLinf 集成，它们仍可提供
setuptools、Warp、ModelScope 等故障线索，但其中的机器路径、代理、runtime 与默认参数不是深圳命令。
特别地，current source 的 action shift 默认已变为 1；旧 release 推理必须显式
`EVALUATION.sigma_shift=5.0`。

2026-08-22深圳current-HEAD standalone已闭环：source/env/vendor assets、official HF checkpoint/stats、
ModelScope T5/VAE/tokenizer均完成；没有下载5B DiT。`FW-SZ-400`真实render通过、expert因
MPLib 0.2.1 × NumPy 2.2.6 segfault而保留为`PARTIAL / FAIL`；NumPy窄降到1.26.4后没有另跑expert，
而是直接执行official evaluator。`FW-SZ-500` run=`fw-sz-500-20260822_045105`在physical GPU 3
完成`adjust_bottle / demo_clean / unseen / 1 episode`，exit 0且1/1 success，accepted seed=`4300001`。
成功视频已轻量复制到
[`evidence/fastwam_adjust_bottle_official_seed42_20260822.mp4`](evidence/fastwam_adjust_bottle_official_seed42_20260822.mp4)；
checkpoint、Wan组件、assets和env等大物全部留在深圳服务器。本结果只证明该seed的一次成功轨迹，不外推成功率。

2026-08-22随后在独立branch `codex/sz-fastwam-dvac-observe`完成default-off action-denoising telemetry；
telemetry commit为`fc652fb49cd32350eca15734b5c7124c0b8c2c02`，exact7 files、`+675/-3`。其后完成
real-query parity工具，local head=`c63dc9b5384d6637a93cc862dbe2815d0332801d`；accepted seed锁为`4300001`，
CPU checks与两轮独立审查均通过，worktree clean，未运行GPU/model/simulator。两commit均尚未push：需先建立
用户自己的Fast-WAM fork并登记第二把repo-scoped key。

## 1. 项目目标与当前阶段

目标是在不破坏现有 π0 + RoboTwin + RLinf + GRPO 基线的前提下，实现：

```text
Fast-WAM policy
  + RoboTwin 2.0
  + RLinf rollout / actor / sync / checkpoint
  + critic-free GRPO / observation-critic PPO
```

RLinf 迁移已经完成主体编码、独立 worktree 同步和联合环境验证；官方 standalone 阶段 0 也已完成：

- π0 + GRPO 基座已经跑通 100 steps，是系统基座。
- 阶段 A–E、F1/F2、checkpoint 下载和 F4 均已完成。CuRobo 0.7.8 的无上界 Warp 依赖曾被解析为不兼容的 1.15.0，现已固定 `warp-lang==1.11.1`；F4 在 seed 0 完成 1/1 专家 episode，并保存 141 帧视频。此前错误强制 Hugging Face 组件源导致的 401 已通过恢复官方 ModelScope 路径解决。2026-07-17 20:05 已从服务器只读核实官方 policy 成功 run：`adjust_bottle` 1/1，成功视频存在，resolved `H/N/S=32/24/10`、scheduler infer shift=5.0。
- Fast-WAM GRPO接入已写入并同步到服务器独立 worktree `/root/autodl-tmp/RLinf_fastwam_rlinf`；原 π0 worktree/venv 未改。P2一步真实runner已通过，`move_stapler_pad`旧formal训练仍在运行。
- 2026-07-19已新增π0训练动态对齐的GRPO独立配置，以及从官方base冷启动的Fast-WAM PPO首版。PPO采用不读取动作的last-video-cache-V critic；formal现与新GRPO同为`4×32/global512/update2`，smoke保持4路并发并缩为`4×1/global32/update1`。静态/CPU链路已验收，真实两卡PPO smoke尚未执行。
- 当前设计默认不整体升级 RLinf、不修改 π0 工作树和环境。

## 2. 当前推荐方案

| 主题 | 当前推荐 |
|---|---|
| 系统基座 | 从已验证 π0 RLinf commit 派生独立分支/worktree |
| 官方 standalone 环境 | 已跑通的干净 Python 3.10、Torch 2.7.1+cu128 `FastWAM-official`；今后只作 oracle，不安装 RLinf/联合依赖 |
| 后续 RLinf 集成环境 | 第三个独立联合 venv：只把 π0 `freeze/list` 当依赖参考，在 Fast-WAM worktree 最终路径从空创建 Python 3.11 venv；不复制、迁移或直接运行 π0 venv，也不在现役 venv替换Torch；固定最终Torch后重编CUDA extension |
| 模型代码 | 官方 Fast-WAM 保持独立，作为模型、预处理和 checkpoint 真相 |
| RLinf 承载 | 已冻结为内置 `rlinf/models/embodiment/fastwam/`，并通过官方 `register_model` 注册；模型本体仍从独立 Fast-WAM 仓库导入。该落点同时对应 RLinf 官方新增模型文档、社区 Fast-WAM、Motus 和 LaWAM；`RLINF_EXT_MODULE` 仅保留为外部依赖场景的备选，不同时维护 |
| RoboTwin | 复用现有 RLinf RoboTwin env/reward；新增三相机、14D qpos model adapter |
| 第一算法 | critic-free GRPO |
| 训练范围 | 第一版只训练 action expert；冻结 proprio、video expert、VAE、T5 和 normalization stats |
| 概率路径 | eval/train 共用一个 ODE/SDE 去噪 core；train 仅在选中一步注入随机 transition，behavior old logprob 与 actor replay 对齐 |
| 首个工程 smoke | `N=24` 时显式使用 192-step episode/rollout，避免 200-step 尾部静默丢失 |
| stochastic step | 已冻结为从全部 `k∈[0,S-1]` 均匀采样；首步按 RLinf OpenPI 的 next-schedule denominator 保持有限，不另设 `ignore_first/ignore_last` |
| 首个非零探索 | 已冻结 `noise_level=0.1`，只作为稳定性与真实 ratio 的保守工程 smoke，不预先当作最终论文超参 |
| 权重同步 | 当前为 RLinf 原生 CPU-staged bucket 512 MiB；本机GPU CUDA IPC被容器拒绝，故只更换transport，bucket算法与同步张量不变 |
| RLinf 版本 | 不为版本号先升级；缺什么能力再按证据处理 |

仍需现场证据或后续实验决定的事项：

- 联合 runner 的实际 FSDP partition、梯度/冻结 hash、actor-rollout 同步等价和 GPU/RAM 峰值；`H/N/S=32/24/10`、infer shift=5.0 与联合配置均已现场确认。
- 是否在首轮稳定后对 `noise_level` 做 `0.1/0.3` 小范围比较；这不改变已经冻结的首个 smoke，也不改变概率公式。
- replay 图像首版保留 BF16 还是改存 uint8；当前推荐先以 BF16 保证 parity，资源实测后再优化。
- 双 A800 的最终 actor/rollout/FSDP 布局。
- 第一项有学习余量的 RoboTwin 任务。

第一版只训练 action expert 的依据不是主观简化：社区 Fast-WAM×RLinf 的有效 GRPO 实验虽然把 `proprio_encoder` 放进 allowlist，但 rollout 在 `no_grad` 下缓存了已拼接的 context，actor replay 没有重新经过 proprio encoder，所以其实际成功结果是 action-only；Motus 已跑通的首轮工程链也以 action expert 为主。官方 Fast-WAM SFT 则训练完整 MoT + proprio，并不能证明“冻结 video expert、只训练 action + proprio”这一组合。待 action-only 全链通过后，再把 proprio 作为单独实验变量引入。

为减少首轮变量，horizon 概率定义也已收敛：完整 `H=32` latent 采样，只执行并计算前 `N` 的 logprob/loss，与社区 Fast-WAM 和 π0 的 horizon/chunk 分离方式一致；其他定义后置。

用户已确认首个工程 smoke 在 `N=24` 时统一使用 192 steps。正式学习协议是否长期保持 192，还是以后实现带 mask 的最后一个短 chunk，作为后续单独实验变量，不阻塞首次接入。

## 3. 本轮实际代码审计结论

2026-07-18 的正式效果与逐层实现审计集中在 [05_IMPLEMENTATION_PLAN.md](05_IMPLEMENTATION_PLAN.md) 第27节；2026-07-19的新GRPO参数与PPO实现集中在第28节。当前主结论：`adjust_bottle` 官方为100/100饱和任务，现有约97% train rollout曲线没有足够GRPO对比信号且并非固定eval；没有发现可直接解释“不涨”的P0概率链迁移缺陷。当前训练任务已切换为`move_stapler_pad` clean；域随机化和no-std GRPO仍分别后置为单变量。

- **π0 基座**：`env_obs → policy preprocessing → denoise chain/behavior logprob → tensor-only forward_inputs → trajectory stack/flatten → actor default_forward → GRPO loss` 已经是可直接复用的外层骨架；Fast-WAM 不需要改 env worker、actor loss、advantage 或 checkpoint runner。
- **Motus**：不是“只有失败经验”。其官方推理复用、batch rollout、model-side chain replay、FSDP、同步、checkpoint、GRPO/PPO backward 基本跑通；失败的是学习质量、critic 和高维 chunk logprob 稳定性。因此它是重要工程迁移模板，不是算法效果基线。
- **社区 Fast-WAM×RLinf**：LIBERO 上的 critic-free GRPO、root FSDP2（禁用block/expert wrap）与bucket sync真正跑通并产生过提升；RLinf pin仍会单独wrap非tied Embedding，不能把它误称为纯root-only。LIBERO adapter、proprio伪训练、catch-all fallback、参数别名和泛化退化也不能照搬。
- **官方 Fast-WAM**：三相机拼图、14D qpos、prompt、normalization、H=32、scheduler、checkpoint schema 是唯一模型真值。
- **Fast-WAM batch 边界**：官方 public `infer_action()` 只支持 B=1，但 prompt/proprio、VAE batch返回、video/action expert、MoT KV cache和scheduler均保留batch维。首版只扩展编排层，生产B>1不做逐环境fallback；VAE先用官方wrapper，显存不足只允许配置化固定分块且失败即停。
- **Flow-SDE 权威分工**：官方 Fast-WAM scheduler决定shifted timestep、signed delta和effective `sigma_shift`；RLinf server pin OpenPI决定mean/std、首步分母、单stochastic transition和replay。社区Fast-WAM只作已跑通的拼接参考，其漏传`shift_override`和`0.98` hard clamp不照搬。
- **checkpoint/export**：目标RLinf pin自带`no_dist=True`的DCP→model-state转换器；regular checkpoint显式不保存full weights，deploy export做canonical prefix与官方schema严格校验，不另写matching-world-size gather。
- **完整改动面**：GRPO初始实现的registry、HF mode capability和FSDP2兼容开关保持不变；PPO增量只进入Fast-WAM builder/core/policy/export及独立配置/launcher/tests。env、trajectory、actor、loss、syncer、runner和DCP manager仍零diff。逐文件总表见实施计划第25节，PPO增量见第28节。
- **LaWAM**：提供 qpos/EE 边界、dtype、注册和 planner timeout 等较窄的反例；其 end-pose/planner 数据流不迁移到 Fast-WAM qpos 路径。

五份既有日志均已完成结构索引和关键章节交叉核查；详细来源职责见 [01_REFERENCE_MATRIX.md](01_REFERENCE_MATRIX.md)。

## 4. 文档层次

```text
项目自动规则
└── AGENTS.md

跨任务连续性
├── PROJECT_CONTEXT.md    稳定背景、路径、长期原则
└── HANDOFF.md            当前进度入口、权威动态交接和下一步

Fast-WAM 迁移主线
├── 00_INDEX.md           本总文档：目标、当前结论、文档索引
├── 05_IMPLEMENTATION_PLAN.md
│                         唯一 RLinf 迁移实施过程文档：阶段、调用链、逐文件台账与集中验收
└── 07_OFFICIAL_STANDALONE_RUNBOOK.md
                          阶段 0 官方推理命令附录

按需技术附录
├── 01_REFERENCE_MATRIX.md
├── 02_INTERFACE_CONTRACTS.md
├── 03_CODE_AND_DIFF_MAP.md
├── 04_BASELINE_AND_RLINF_VERSION.md
├── 06_TEST_ACCEPTANCE_MATRIX.md
└── SOURCE_LOCK.yaml

实验与审计证据
├── C:\Users\86136\Documents\rl\audits\
│                         π0 训练历史快照、日志、资源与图表
└── evidence/             Fast-WAM 迁移的小型校验证据
```

不存在第二份 RLinf 集成计划。`07` 只把 `05` 的阶段 0 展开成可逐段验收的命令，不定义另一条迁移路线；其他技术附录只给总文档和实施计划提供证据。

## 5. 每份文档的唯一职责

| 文档 | 唯一职责 | 何时更新 |
|---|---|---|
| `AGENTS.md` | 必须遵守的项目规则 | 用户长期工作方式变化时 |
| `PROJECT_CONTEXT.md` | 稳定背景、路径和长期决策 | 稳定事实或长期默认改变时 |
| `HANDOFF.md` | 动态状态入口和下一步 | 一轮实验、实施或交接结束时 |
| `00_INDEX.md` | 项目总览、当前共识、待决策和关键风险 | 每轮形成重要结论时 |
| `05_IMPLEMENTATION_PLAN.md` | 唯一实施过程文档：执行顺序、调用链、逐文件台账和阶段状态 | 实施设计、顺序或阶段进度变化时 |
| `07_OFFICIAL_STANDALONE_RUNBOOK.md` | 阶段 0 官方推理的逐段命令与验收 | 官方入口、依赖或本机执行事实变化时 |
| `01_REFERENCE_MATRIX.md` | 参考材料提供什么 | 新增/淘汰重要来源时 |
| `02_INTERFACE_CONTRACTS.md` | tensor、policy、replay、checkpoint 契约 | 技术接口改变时 |
| `03_CODE_AND_DIFF_MAP.md` | 来源代码到目标模块的映射 | 目标代码结构改变时 |
| `04_BASELINE_AND_RLINF_VERSION.md` | 某次版本评估的证据快照 | 重新评估 RLinf 版本时 |
| `06_TEST_ACCEPTANCE_MATRIX.md` | 必须通过的最小测试 | 验收标准改变时 |
| `SOURCE_LOCK.yaml` | commit/path 的机器可读锁 | 来源版本改变时 |
| `evidence/` | 小型可复核证据 | 实际审计或测试产生证据时 |
| `evidence/IMPLEMENTATION_LOG_20260717.md` | 本轮代码、服务器同步、环境与验证日志 | 本轮实施产生新事实时 |
| 项目 `audits/` | 训练历史快照、日志摘要和可视化 | 每次服务器只读实验审计后 |

维护原则：

- 同一事实只在一个权威文档详细描述；其他文档只写摘要和链接。
- 不因普通讨论机械更新所有文件。
- 重要结论、用户决定和关键风险更新总文档；具体接口或测试再更新对应附录。
- 不复制源码、大日志、模型、数据集或凭据进文档。

## 6. 新任务最小读取路径

Fast-WAM 迁移的新任务默认只需：

1. `AGENTS.md`。
2. `PROJECT_CONTEXT.md`。
3. `HANDOFF.md`。
4. 本文件 `00_INDEX.md`。

准备实施时再读 [05_IMPLEMENTATION_PLAN.md](05_IMPLEMENTATION_PLAN.md)；执行官方 standalone 时只需再读 [07_OFFICIAL_STANDALONE_RUNBOOK.md](07_OFFICIAL_STANDALONE_RUNBOOK.md)。遇到具体设计问题时，才按本索引打开对应技术附录。无需每次完整读取全部材料。

## 7. 目标架构

```text
Fast-WAM official
  ├── checkpoint / stats / resolved config
  └── deterministic inference fixture
                  ↓
RLinf 内置 Fast-WAM adapter
  ├── builder
  ├── RoboTwin adapter
  ├── BasePolicy wrapper
  ├── Flow-SDE rollout/replay
  ├── optional observation-side PPO value head
  └── canonical model tree + official deploy export
                  ↓
current RLinf + RoboTwin + GRPO/PPO
  ├── EnvWorker / group / reward
  ├── actor / rollout / Ray
  ├── GRPO advantage or GAE/value loss
  ├── bucket weight sync
  └── DCP / logs / monitoring
```

社区 Fast-WAM+LIBERO 代码作为已跑通的 Flow-SDE/FSDP/sync 参考，但不整体合并。Motus 作为已跑通大部分 RLinf 工程链的迁移模板；LaWAM 作为更窄的 env/action/dtype 参考。二者都不替代官方 Fast-WAM 数学和预处理。

## 8. 技术附录导航

- [01_REFERENCE_MATRIX.md](01_REFERENCE_MATRIX.md)：官方、社区、π0、Motus、LaWAM 分别提供什么。
- [02_INTERFACE_CONTRACTS.md](02_INTERFACE_CONTRACTS.md)：输入输出、Flow-SDE、proprio replay、FSDP 和 checkpoint 契约。
- [03_CODE_AND_DIFF_MAP.md](03_CODE_AND_DIFF_MAP.md)：内置 Fast-WAM 模型结构、逐 symbol 来源和最小 RLinf diff。
- [04_BASELINE_AND_RLINF_VERSION.md](04_BASELINE_AND_RLINF_VERSION.md)：为什么暂不升级 RLinf。
- [05_IMPLEMENTATION_PLAN.md](05_IMPLEMENTATION_PLAN.md)：唯一实施过程文档；含自上而下调用链、连贯编码批次、目标文件与集中验收。
- [06_TEST_ACCEPTANCE_MATRIX.md](06_TEST_ACCEPTANCE_MATRIX.md)：必要验收项。
- [07_OFFICIAL_STANDALONE_RUNBOOK.md](07_OFFICIAL_STANDALONE_RUNBOOK.md)：阶段 0 官方推理逐段操作与验收。
- [08_JOINT_ENV_REPRODUCTION.md](08_JOINT_ENV_REPRODUCTION.md)：联合 venv 的实际构建顺序、版本 pin、问题修复与复现边界。
- [SOURCE_LOCK.yaml](SOURCE_LOCK.yaml)：锁定的仓库、commit 和路径。

## 9. 关键风险

| 风险 | 必要控制 |
|---|---|
| 污染基准环境 | π0 golden 与 `FastWAM-official` 都只读；联合依赖只进入第三个独立 worktree/venv |
| adapter 与官方推理漂移 | 黄金 fixture 和 deterministic parity |
| old/new logprob 不一致 | optimizer 前 ratio≈1、clip≈0 |
| batch/FSDP 数值微差被高维 logprob 放大 | 同进程和真实 rollout→FSDP actor 两层 parity；不伪造 old logprob |
| 把社区 proprio allowlist 误当成已验证训练 | 第一版 action-only；之后单独启用并检查 grad/update |
| 官方模块别名重复注册 | FSDP 前收敛成唯一 `mot.*` 参数树，并做推理/state-dict parity |
| 内置模型未注册或未收到 mode | 在 `rlinf/models/__init__.py` 用官方 registry 注册；worker 使用默认关闭的 capability shim，不再增加模型名硬编码 |
| `N=24` 静默丢弃 200-step 尾部 | 首个 smoke 显式用 192；正式协议单独决定是否实现 masked short chunk |
| 双 A800 或主机 RAM 不足 | 当前并发固定4 env、micro/forward均2并启用env/rollout offload；新GRPO只增加顺序rollout和梯度累积。只在真实smoke证据后调整资源参数，不改算法链 |

其他问题在实际出现时再分析，不预先维护多套兜底。

## 10. 接下来

深圳official standalone当前无需再装环境、下载checkpoint或补跑专家采集；若下一阶段要接入latest RLinf，
从current RLinf base新建独立branch/worktree并另做source/interface审计。以下条目是旧AutoDL集成线的历史待办，
不属于本次深圳standalone续跑命令。

1. 当前旧`move_stapler_pad` formal先自然运行；新任务必须现场刷新进程/step，不能根据本文假定训练仍在。
2. 后续GRPO单变量运行使用新`pi0_aligned`配置：4 env×32 epoch、group4、global512、update2；旧formal不改，以便直接对照。
3. PPO已完成代码和静态测试，但尚未通过真实两卡runner。现役driver结束后先执行`4×1/global32/update1`独立PPO一步smoke，重点检查old/new value、value/head梯度、两卡资源、同步和step1 DCP；通过后才启动`4×32/global512/update2` formal。
4. GRPO/PPO共用官方Fast-WAM conditioning、scheduler、H/N/S和Flow-SDE核心；PPO只增加observation-side value路径，不把selected-k action或噪声喂给critic。
5. fresh DCP resume、更新后actor→rollout同步和官方deploy export parity仍是共同的后续验收，不因静态测试通过而省略。

每次推进一个连贯实现批次，完成后做适量必要检查并更新文档；不提前铺设多套兜底方案。
