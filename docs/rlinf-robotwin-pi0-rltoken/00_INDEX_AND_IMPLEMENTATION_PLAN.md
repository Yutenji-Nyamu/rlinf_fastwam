# π0 × RoboTwin × RLToken / RLT：上下文索引与实施主计划

> 状态：2026-07-30 实现候选版 v11（Stage 1 artifact 已验收；Stage 2 fresh smoke
> 已通过，resume 按用户要求省略；formal pilot 待预算批准）
> 本文件是 RLToken / RLT × π0 × RoboTwin 的唯一专题设计与实施计划。
> 动态服务器状态、命令和结果只写入 [`evidence/IMPLEMENTATION_LOG.md`](evidence/IMPLEMENTATION_LOG.md) 与根 [`HANDOFF.md`](../../HANDOFF.md)，不在本文件复制成第二份真相。
> DSRL、QAM、Fast-WAM 和旧附件均不进入 RLT 默认上下文；其历史只在明确追溯时按索引读取。

## 0. 状态词与阅读方式

本文只使用四种状态：

| 状态 | 含义 |
|---|---|
| **冻结** | 主体实现按这一条执行，不再同时维护备选路线 |
| **待事实验证** | 由真实数据、固定张量或源码检查决定，不需要用户凭感觉拍板 |
| **待运行批准** | 会改变预算、方法解释或启动服务器作业，必须进入审批 packet |
| **延后** | 不阻塞首版；只有明确失败证据才重开 |

实现者不应把“候选超参”“旧服务器快照”或“历史经验”误读为已冻结配置。正式
Stage 1、Stage 2 smoke 和 formal run 均以当次展示的完整 resolved config 为准。

## 1. 精确上下文路由

### 1.1 每个 RLT 新任务默认只读这四个入口

按顺序：

1. [`AGENTS.md`](../../AGENTS.md)：工作区执行、授权和文档规则；
2. 本地工作区根 `PROJECT_CONTEXT.md`：跨专题稳定原则；该文件不复制进服务器代码 worktree；
3. [`HANDOFF.md`](../../HANDOFF.md)：只选择 RLT 专题行，读取当前停点与授权；
4. **本文件**：RLT 唯一规范、调用流、改动面和开放决策。

默认不全文读取 DSRL 计划、Fast-WAM 文档、旧交接或七份用户历史材料。

### 1.2 按动作触发的条件材料

| 当前动作 | 额外读取 | 读取目的 |
|---|---|---|
| 继续实现、复盘命令、声称服务器/测试状态 | [`evidence/IMPLEMENTATION_LOG.md`](evidence/IMPLEMENTATION_LOG.md) 的索引与相关批次 | 精确命令、文件、结果、错误、修复和时间边界 |
| 查看 Stage 1 smoke、专家数据解释、参数来源与磁盘审计 | [`02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md`](02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md) | 本轮结论与 evidence 索引；不替代本文件的规范 |
| 查看 full clean-50、正式 Stage 1 指标、endpoint 与 artifact 验收 | [`03_STAGE1_FORMAL_TRAINING_20260729.md`](03_STAGE1_FORMAL_TRAINING_20260729.md) | Stage 1 完成、资源、重建对照与交付包 |
| 追溯 Stage 2 参数与运行前配置 | [`04_STAGE2_PRE_SMOKE_PACKET_20260729.md`](04_STAGE2_PRE_SMOKE_PACKET_20260729.md) | 历史批准包、参数来源、resolved config 与预期合同 |
| 查看 Stage 2 fresh smoke、资源或选择 formal 预算 | [`05_STAGE2_FRESH_SMOKE_RESULT_20260730.md`](05_STAGE2_FRESH_SMOKE_RESULT_20260730.md) | 当前结果、产物、限制与30/60-cycle决策 |
| 处理 AutoDL Git/GitHub 网络 | [`06_AUTODL_NETWORK_PLAYBOOK.md`](06_AUTODL_NETWORK_PLAYBOOK.md) | 大陆线路短探针、有界 push 与故障分流 |
| 改方法语义或判断是否忠于论文 | [RLT 论文 v2](https://arxiv.org/html/2604.23073)、[Physical Intelligence 项目页](https://www.pi.website/research/rlt/) | 方法不变量、作者实验和公开限制 |
| 改 Stage 1/Stage 2 算法代码 | [RLinf RLT 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html) 与第 1.3 节锁定源码 | 可执行复现、真实 config 和 symbol |
| 下载/转换 demonstration | RoboTwin [π0 数据文档](https://robotwin-platform.github.io/doc/usage/Pi0.html)、官方 HF 单文件 metadata、账本 A008/A009 | revision、schema、动作语义、空间和覆盖风险 |
| 构造旧路径回归或资源审批 packet | DSRL 当前计划/证据中的对应小节 | 只复用 RoboTwin π0 基础设施、回归格式和资源治理 |
| 追溯旧附件、早期 RLT/QAM 推导 | `../rlinf-robotwin-pi0-traditional-rl/01_FULL_REFERENCE_HISTORY_20260728.md` | 非规范历史 |
| 声称进程、GPU/RAM、HEAD、dirty tree、checkpoint“当前” | 新的服务器只读现场检查 | 动态事实不能靠旧文档或 Memory |

### 1.3 实现时的源码/symbol 索引

本地只读参考镜像锁定在
`.research-rlinf@c5ca51cc21c007a41d287159f9e1b14e0200000e`。RLT 主体来源是
`5769c6eb`，ManiSkill route/transition/schedule 来源是 `3d93750d`。实际开发基线已经按
用户指定冻结为服务器
`48a775db09c16c455aeba7b0600c920e7c80d534`；它包含 clean π0 基线
`6d0db56b` 及其后的 Fast-WAM/DSRL 提交，因此 RLT 必须依靠独立 branch/worktree、
config opt-in 和 legacy 回归隔离，而不能把它描述成纯官方 π0 commit。

| 问题 | 首要文件 / symbol |
|---|---|
| Stage 1 config 与训练量 | `examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml` |
| Stage 2 config、schedule、sync | `examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml` |
| prefix、RLT loss、feature/reference、OpenPI transform | `rlinf/models/embodiment/openpi/openpi_action_model.py` |
| RL token encoder/decoder | `rlinf/models/embodiment/modules/rlt_token_transformer.py` |
| Stage 2 actor/Q 输入和 fixed Gaussian | `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` |
| reference/student/expert 选择 | `rlinf/algorithms/rlt/route.py` |
| feature → actor → route | `rlinf/algorithms/rlt/rollout.py` |
| current/next feature linker | `rlinf/algorithms/rlt/transition.py` |
| replay ingest、reward/TD、schedule、RLT checkpoint | `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` |
| 基础 SAC save/load、target、replay | `rlinf/workers/actor/fsdp_sac_policy_worker.py` |
| rollout weight/version apply | `rlinf/workers/rollout/hf/huggingface_worker.py` |
| outer step、load、首轮 sync、save 边界 | `rlinf/runners/embodied_runner.py` |
| canonical training action 与 env action 分流 | `rlinf/workers/env/env_worker.py` |
| RoboTwin 三相机/state/action transform | `robotwin_aloha_dataconfig.py`、`aloha_policy.py`、现有 RoboTwin π0 config/env |

### 1.4 明确排除与文档分离

- `docs/fastwam-robotwin-rlinf-grpo/`：整个专题不作为 RLT 算法、参数、环境或分支来源。
- DSRL：只按需复用系统合同和治理；不继承 latent、reward、replay、target 或超参。
- QAM：不进入本批实现。
- `docs/project-history/` 与长历史文件：只作追溯，不决定当前实现。
- 七份旧用户材料没有 RLT 实现；其中可复用的只是“fixed-input parity、参数 delta、sync、DCP、资源记录”一类工程经验，已沉淀在工作区规则与验收合同中，不再在本文件逐份复述。

因此不再新建第二份“RLT 上下文计划”：当前 `00_INDEX_AND_IMPLEMENTATION_PLAN.md`
同时承担精确索引和唯一实施计划；无关内容已经由专题目录与历史索引物理隔离。

## 2. 首版范围与冻结决定

| 项目 | 首版决定 | 状态 |
|---|---|---|
| 任务 | 只做 `adjust_bottle` | 冻结 |
| Stage 1 数据 | 官方 `clean_50` 解包后的全部有效 episode | 冻结 |
| 数据规模解释 | 低预算单任务移植；不冒充 ManiSkill 400 条等规模复刻 | 冻结 |
| Stage 1 模型 | 现有 RoboTwin π0 SFT 初始化；π0 全冻结，只训练 RLT encoder/decoder | 冻结 |
| Stage 1 切分 | 首版不设 held-out val；使用固定 endpoint，不按下游 RL 结果挑 checkpoint | 冻结 |
| normalization | Stage 1 loader/model、Stage 2 feature/reference、action decode 共用现有 π0 checkpoint 的同一 `norm_stats.json` | 冻结 |
| feature | image-only prefix、mask 开启、真实 image prefix `[B,768,2048]`、`z_rl=2048` | 冻结；已由现有 RoboTwin π0 checkpoint 探针验证 |
| VLA horizon | `H=50` | 冻结 |
| actor/env chunk | `C=10`、14D canonical action | 冻结为实现主线；运行前批准成本解释 |
| Stage 2 | 同步 worker、4 train env、`rollout_epoch=1` | 实现主线；运行 packet 再确认 |
| route | train 全任务 reference warm-up → student；全程记录 compact transition | 冻结 |
| eval | deterministic student mean，不受 train-ready gate 替换成 reference；不写 replay | 冻结 |
| expert/human | 不接 intervention、人工 phase 或人工 reward | 冻结 |
| replay | 复用 RLinf compact transition；首轮 bounded run，不改通用 hard capacity | 冻结 |
| resume | 首批包含最小 RLT trainer-state/contract 修复 | 冻结 |
| 隔离 | 独立 `codex/rlt-pi0-robotwin` branch/worktree；新行为 config opt-in，legacy 默认不变 | 冻结 |

服务器上的 worktree 不是重新下载两份 Git 历史：同一 object store 挂出两个并列工作目录，
每个目录有独立 branch、index 和工作文件。DSRL 用
`cd /root/autodl-tmp/RLinf_fastwam_rlinf`，RLT 用
`cd /root/autodl-tmp/RLinf_rlt_pi0_robotwin`；分别提交各自分支、写各自 output。两者只读
复用同一个已验证 `.venv`，用显式 `PYTHONPATH` 选择当前 worktree，不复制/重命名环境，也
不需要从云端重新拉一份仓库。

50 条的结论是：**足够开始首版实现、Stage 1 和低预算 pilot；不保证复现 ManiSkill
400-demo 或论文真机效果。** 首版不复制样本凑 400、不做 10/25/50 sweep，也不预先收集
100/200/400。只有固定 endpoint 后 token/Stage 2 明确失败，才重开数据量。

## 3. 来源优先级与复现差异

### 3.1 到底抄谁

| 来源 | 直接继承 | 不能直接继承 |
|---|---|---|
| RLT 论文 v2 / 项目页 | RL token bottleneck、stop-gradient reconstruction、冻结 feature 后 online actor-critic、reference pass-through、chunk TD、warm-up | 未公开训练代码；π0.6、真机 critical phase、人类 supervision 不能冒充本项目现状 |
| RLinf ManiSkill RLT | Stage 1 模块、RLT MLP、twin-Q/BC、route 框架、compact transition、同步 schedule、SAC/DCP 主链 | Panda 8D delta action、两相机、geometry gate、现有 resume 缺口、任务超参 |
| 现有 RoboTwin π0 | 三相机、14D state/action、32D OpenPI padding、H50、Aloha dataconfig、output transform、EnvWorker、sync/DCP | PPO/GRPO/DSRL 的算法状态、reward/replay/target 语义 |

优先级是：论文定方法不变量；RLinf 定可执行算法骨架；RoboTwin π0 定环境和系统接口。

### 3.2 必须公开记录的差异

| 维度 | 论文 | RLinf ManiSkill | 本项目首版 |
|---|---|---|---|
| VLA | π0.6 | π0.5 | 现有 π0 RoboTwin SFT |
| Stage 1 | reconstruction，可选联合 SFT | reconstruction + `rlt_alpha * vla_loss` | π0 全冻结，`rlt_alpha=0`，只训 token 模块 |
| decoder | 论文写 autoregressive | RLinf 为 parallel reconstruction | 继承 RLinf，不在首版重写 |
| language | 每任务固定，RL token 步骤可去语言 | config 可选 | `rlt_image_only=true` |
| actor phase | 关键精细阶段 | peg/grasp/hole predicate | `C_t ≡ True` 的 full-task 简化 |
| intervention | 人可选介入并替换 reference | ManiSkill 默认关闭 expert | 完全关闭 |
| transition 密度 | 论文描述 stride-2 overlapping chunk subsampling | RLinf 当前按 rollout chunk boundary 存 compact transition | 继承 RLinf；stride-2 延后 |
| action | 论文中的同一机器人动作域 | 8D `pd_joint_delta_pos` | output-transform 前 14D canonical，再统一 decode 为 absolute qpos |
| demonstration | 小型 task-specific 数据 | 官方参考集 400 成功 episode | clean-50 全部有效 episode |

所以首版名称应是“RLT × π0 × RoboTwin 的低预算 RLinf 移植”，不能称为论文严格复刻。

## 4. 两阶段结构与张量合同

### 4.1 Stage 1

```text
RoboTwin demonstration
  -> frozen π0 image-prefix embeddings [B,768,2048]
  -> append learnable RL token
  -> RLT encoder
  -> z_rl [B,2048]
  -> RLT decoder reconstructs stop-gradient prefix target
```

首版：

```text
rlt_train_vla = false
rlt_alpha = 0
rlt_input_dim = 2048
rlt_embed_dim = 2048
rlt_prefix_seq_len = 768
rlt_image_only = true
rlt_use_mask = true
```

2026-07-29 使用现有 `adjust_bottle` π0 checkpoint、其
`physical-intelligence/robotwin/norm_stats.json` 和 synthetic 三相机 observation
执行无训练探针，得到：

```text
full prefix:  [1,816,2048], mask true=773
language:     [1,48]
image prefix: [1,768,2048], mask true=768
state:        [1,32]
```

因此 `768` 是 image-only token length，`2048` 是真实 hidden width；这两个值已经写入
config。单条真实 clean-50 的 OpenPI distributed loader 与 reconstruction 主链先由
S1-A/S1-B 验证；随后全部50个有效 episode 的固定2k endpoint 已完成并通过 artifact 验收。

### 4.2 Stage 2

每个 macro observation：

| 字段 | 合同 |
|---|---|
| `z_rl` | `[B,2048]`，frozen Stage 1 feature |
| `proprio` | `[B,14]` |
| `ref_chunk` | `[B,C=10,D=14]`，output-transform 前 canonical reference |
| student action | `[B,10,14]`，同一 canonical 域 |
| reward chunk | `[B,10]` |
| replay action | route 实际选中的 canonical `[B,10,14]` |
| env action | decode 后 absolute qpos `[B,10,14]` |

Actor 读取 `ref_chunk + z_rl + proprio`；critic 读取
`z_rl + proprio + action_chunk`。Stage 2 只同步 MLP actor；Stage 1 feature model始终冻结。

## 5. Stage 1 数据与 normalization 合同

### 5.1 数据流

```text
RoboTwin official adjust_bottle clean-50
  -> official raw RoboTwin -> Aloha HDF5 converter
  -> official Aloha HDF5 -> LeRobot converter（全新 repo_id）
  -> pi0_aloha_robotwin dataconfig
  -> 三相机 + fixed prompt + state[14] + action[14]
  -> explicit checkpoint norm_stats
  -> frozen π0 + new RLT module
```

RoboTwin source action 是下一拍 absolute qpos；现有 dataconfig 将 12 个 joint 转成 delta，
两个 gripper 保持 absolute。这个边界必须与现有 π0 checkpoint 一致。

### 5.2 50 条如何使用

- converter 合同检查只转换 1 episode；
- 正式 Stage 1 使用解包后全部有效 clean-50；
- 不做 45/5 或 40/10 episode split，不随机拆 frame；
- 不做 early stopping 或 best-checkpoint search；在 Stage 1 审批 packet 冻结一个
  steps/batch/LR/endpoint；
- 固定训练 batch 可检查 reconstruction 和冻结合同，但不宣称 held-out generalization；
- 实际有效 episode/frame 数必须在转换后报告；若名义 50 与实际明显不符，停止补样并重新确认。

### 5.3 `norm_stats` 只有一份真相

[RLinf 官方 RLT 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)
要求 Stage 1、Stage 2 feature model 和 OpenPI assets 使用同一份 stats。对本项目而言，π0
base 又是冻结的，因此运行时唯一来源固定为现有 π0 SFT checkpoint 内的
`physical-intelligence/robotwin/norm_stats.json`：

1. Stage 1 dataloader/model 显式指向该绝对路径；
2. Stage 2 frozen feature/reference 显式指向同一路径；
3. canonical decode 使用同一 transform；
4. resolved config 和日志记录绝对路径、SHA256、key 与 shape。

clean-50 可以计算诊断统计，但**不能静默替换运行 stats**。比较必须在
`AlohaInputs/DeltaActions` 后、`Normalize` 前，只看有效前 14D；不能把 raw absolute qpos
与 checkpoint 的 delta-action stats 直接比较。key/shape/non-finite 是 hard fail；分布差异
只记录到 Stage 1 审批材料，除非用户明确改变 frozen-base 合同。

### 5.4 Stage 1 最小验收

- 一个真实 batch 的三相机、prompt、state/action shape/dtype 正确；
- prefix width、真实 `L`、mask 与 `z_rl` shape 有记录；
- reconstruction loss 有限并下降；
- optimizer 只包含 RLT encoder/decoder；
- π0 参数 delta 为 0；
- 产出一个稳定的 Stage 1 contract manifest：checkpoint ID、RLT config、norm-stats hash、
  prefix/mask 和 action contract；Stage 2 不靠可变路径名猜身份。

## 6. 动作坐标与单次 decode

### 6.1 高层不变量

必须始终成立：

```text
reference action domain
= student action domain
= BC target domain
= critic action domain
= replay action domain
```

ManiSkill 的 reference/student 本来都能直接送入 `pd_joint_delta_pos`。RoboTwin 的 OpenPI
output transform 还会用当前 processed state 做 `delta -> absolute`，所以不能把已经变成
absolute qpos 的 reference 与 tanh canonical student 直接 route。

### 6.2 冻结方案

```text
frozen π0 raw action [B,H=50,32]
      |
      +--> canonical ref = first [C=10,D=14]
      |
RLT actor -> canonical student [B,10,14]
      |
FullTask route chooses canonical ref/student
      |
      +--> forward_inputs["action"] / replay / BC / Q
      |
clone raw 32D reference template for the same observation
overwrite first 14D with selected canonical action
      |
same OpenPI output transform(exact processed state)
      |
RolloutResult.actions -> RoboTwin absolute qpos [B,10,14]
```

尾部 18D 使用同一次 π0 raw reference 作为模板，不手填猜测值；环境仍只消费前 14D。

### 6.3 ephemeral decode context

首版接口必须满足：

- 同一次 feature extraction 只调用一次 `obs_processor/input_transform`；
- 可选 `return_decode_context=True` 返回
  `(rlt_obs, decode_ctx)`，旧 `extract_rlt_obs()` API 保持兼容；
- `decode_ctx` 至少含同一次预处理得到的 `observation.state` 和 raw 32D action template；
- `decode_rlt_action(selected_canonical, decode_ctx)` 不再接 raw `env_obs`，也不重复预处理；
- `decode_ctx` 是 no-grad/ephemeral，不进入 MLP obs、`forward_inputs`、transition 或 replay；
- train/eval/reference/student 共用这一个 route 后 decoder。

固定 observation 下，reference canonical → 新 decoder 的 env action 必须与旧 π0 直接
output transform 等价；同时断言 input transform 只执行一次。

## 7. Stage 2 调用流、route、replay 与 reward

### 7.1 完整调用流

```text
RoboTwin EnvWorker raw obs
  -> Frozen Stage1 π0 + RLT feature
       z_rl[2048], proprio[14], canonical ref[10,14], decode_ctx
  -> RLT MLP
       canonical student[10,14]
  -> FullTaskRLTRoute
       train: ref/student + record
       eval: deterministic student + no record
  -> shared canonical decoder
       absolute env qpos[10,14]
  -> RoboTwinEnv.chunk_step
       reward[10], termination, truncation, final/next obs
  -> pending-transition linker
  -> compact replay
  -> twin-Q target + Q1/BC actor update
  -> sync Stage2 MLP/version to rollout
```

### 7.2 route 真值表

训练时令 ManiSkill critical mask `C_t ≡ True`，保留 learner-ready `R_t`：

| mode | ready | 执行控制器 | `record_transition` |
|---|---:|---|---:|
| train | 0 | reference | true |
| train | 1 | stochastic student | true |
| eval | 任意 | deterministic student mean | false |

Eval 故意不让 ready gate 把控制器暗换成 reference，否则同一个 `eval/success` 曲线会中途
改变含义。日志仍记录 `learner_ready` 与 controller；step-0 student 是明确标注的初始点。
π0 reference 由独立的 C10 reference-only sanity 报告。

### 7.3 compact replay

每条 macro transition：

```text
curr {z_rl, proprio, ref_chunk}
+ executed canonical action[10,14]
+ reward[10]
+ termination/truncation
+ next {z_rl, proprio, ref_chunk}
```

`forward_inputs["action"]` 是训练/replay canonical action；
`RolloutResult.actions` 是 env absolute qpos。现有 EnvWorker 已能分开两者，不新增 worker。

首版：

- 将 transition replay/route 改为 config capability，不把 RoboTwin 伪装成 `MANISKILL_RLT`；
- 继承 RLinf chunk-boundary linker，不实现论文 stride-2 overlapping subsampling；
- `max_num_samples` 当前没有形成 hard capacity，不冒充已生效；
- bounded smoke/pilot 用总 transition 停止条件，`sample_window_size` 不小于本次累计
  transition，避免无意引入 recency replay；
- 正式长跑若将超过内存安全线，再单独实现 default-`None`、config-opt-in oldest eviction。

### 7.4 reward 与 bootstrap

- RoboTwin sparse success：成功 primitive slot 为 `+1`，其余为 0；
- RLT worker聚合完整 reward chunk：$\sum_{i=0}^{C-1}\gamma^i r_i$；
- target bootstrap 使用 chunk 对应折扣；
- success termination：不 bootstrap；
- 纯 time-limit truncation：bootstrap；
- success 与 truncation 同时出现：success 优先，不 bootstrap；
- final next observation 必须来自正确的 final obs，不得被 auto-reset obs 替换。

新增 `bootstrap_on_truncation` 只在 RoboTwin RLT config opt-in，legacy ManiSkill 默认不变。

### 7.5 schedule 与 warm-up 的精确语义

当前同步 worker：

- `total_transitions_added/episodes_added` 先按 actor rank 本地累计，再做 `SUM all-reduce`；
- replay readiness 使用各 actor rank `replay.total_samples` 的 `MIN all-reduce`；
- 因而 `warmup_min_size` 是**每 actor rank 最小 replay 条数**，不是 global transition；
- warm-up 触发时保存的 transition/episode anchors 是 global `SUM`；
- learner ready 由 `update_step >= warmup_post_collect_updates` 决定；
- online desired updates 由 global totals、global anchors、`update_step` 和 resolved schedule 推导。

例如两 actor ranks 若希望约 1k global transition 后开始 critic warm-up，现有语义下候选配置
约为 `warmup_min_size=500` per rank，并等待较慢 rank 达标；实际 ready 时仍要报告 observed
global total。该数值只是解释示例，正式 packet 再冻结。

论文支持、但必须和 RLinf ManiSkill 可执行 YAML 区分的结构参数：

- twin Q，target 取 min；
- reference dropout 0.5；
- UTD 5 critic updates / new transition；
- actor 每 2 次 critic update 更新一次；
- actor loss 为 `-q_weight * Q1 + bc_weight * BC(reference)`；
- fixed entropy alpha 为 0。

其中 UTD5 与 critic:actor=2 来自 RLT 论文；当前 RLinf ManiSkill YAML 在同步 worker 下实际是
`update_epoch=5/train_every_transitions=5`，即有效 macro-UTD1，并设置
`critic_actor_ratio=4`。本项目候选 `5/1` 与 ratio2 更接近论文，但比 ManiSkill YAML
激进，必须作为 RoboTwin 低预算 candidate 在 Stage 2 packet 中显式批准，不能描述为
“ManiSkill 原值”。

`gamma`、两段 warm-up、batch、LR、tau、fixed std、BC/Q 数值权重和总预算不冒充论文默认，
统一在 formal packet 冻结一组，不先做 sweep。

## 8. Resume：ManiSkill 现状与本项目合同

### 8.1 ManiSkill 当前怎么做

当前 `RLTACFSDPPolicy` 没有 RLT 专属 save/load override，继承的 SAC 基类能保存/加载：

- actor/Q、optimizer、scheduler；
- target model与可选 entropy state；
- 每 rank replay payload、index、size 和 total samples。

但它不保存 `update_step`、lifetime totals 或 warm-up anchors。直接 load 后这些从 0/None
重新开始，route 会退回 reference warm-up，actor/critic cadence 和 BC/Q ramp 也会回退。
runner 从目录恢复的 outer `global_step` 不是 RLT `update_step`，不能互相猜。

### 8.2 最小 source-of-truth state

每 actor rank 保存：

```text
sac_components/rlt_trainer_state/checkpoint_rank_<rank>.pt
```

schema：

```text
schema_version
rank
saved_runner_step                       # 仅审计
actor_world_size
rlt_resume_contract                    # canonical sorted JSON
rlt_resume_contract_sha256
update_step
local_total_transitions_added
local_total_episodes_added
global_warmup_ready_total_transitions
global_warmup_ready_total_episodes
```

只保存独立原始状态。以下全部重算，不另存第二份真相：

- train-ready；
- BC/Q ramp；
- pending update budget；
- `transitions_since_train/episodes_since_train`；
- rollout version。

### 8.3 resume contract fingerprint

不能只校验 schedule。Stage 2 checkpoint 不内嵌 frozen Stage 1 feature model，而 replay 中的
`z_rl` 来自它；若 resume 时换了 feature/stats/action adapter，旧新 replay 会混入不同坐标系。

`rlt_resume_contract` 至少包含：

1. **schedule**：per-rank replay threshold、post-collect updates、train cadence、
   update epoch/cap、critic/actor ratio、BC/Q schedule；
2. **distributed/sync**：actor world size、syncer 类型、`weight_sync_interval=1`、
   patch `init_sync.enabled=true`；
3. **feature/action/replay**：Stage 1 manifest ID、norm-stats SHA256、image-only/mask/prefix/
   `z_dim`、canonical adapter version、H/C/D、route、transition schema、termination/truncation
   bootstrap 语义；
4. **优化语义**：loss/Q aggregation、`bootstrap_type`、target-update freq/type、
   actor/critic optimizer、global/micro batch、Q/BC fallback weights。

使用 canonical sorted JSON + SHA256，不能用 Python `hash()`。不必每次读取并 hash 整个
多 GB checkpoint；使用 Stage 1 生成时的稳定 manifest/checkpoint ID 和小文件 hash。

### 8.4 save → load → first rollout

```text
安全 checkpoint 边界
  -> rank 0 原子写 completion=false，barrier
  -> super.save: model/optim/scheduler/target/replay
  -> all ranks 写临时 RLT state 并原子 replace
  -> 跨 rank 校验 update_step/anchors/contract
  -> rank 0 最后写 completion manifest

resume init
  -> rollout/env/actor init（rollout.version 此时仍为 0）
  -> super.load 恢复 SAC 主体和每-rank replay
  -> collective 校验完整性/schema/rank/world-size/fingerprint
  -> 恢复 update_step/local totals/global anchors
  -> 重算 ready/ramp/pending
  -> 在任何 env action 前做一次 full initial weight sync
  -> sync version = restored update_step
  -> 首个 route 与不中断训练处于同一阶段
```

正式 RLT config 固定：

```yaml
runner:
  weight_sync_interval: 1
weight_syncer:
  patch:
    init_sync:
      enabled: true
```

或经过证明的等价 full initial sync。首个 rollout 前必须满足 rollout 参数与 actor rollout
state dict 一致，且 `rollout.version == restored update_step`。

### 8.5 兼容与明确排除

- fresh run 不调用 load，不受影响；
- `resume_dir` 表示连续训练：缺 RLT state、部分 rank 文件、旧 schema、world-size 或
  fingerprint 不匹配均在首 rollout 前 collective fail；
- weights-only 初始化、evaluation migration 和 legacy ManiSkill checkpoint 必须走显式
  独立模式，不能静默清零 schedule；
- 首版只支持相同 actor world size、同步安全边界 checkpoint；
- RNG bitwise continuation、跨 world-size、async 和 mid-update resume 延后。

### 8.6 最小验收

- 两个 local totals 不相等的 ranks round-trip 后，global SUM、MIN replay、`update_step`、
  anchors 一致；
- pre-ready 与 post-ready checkpoint 的首个 route 都与不中断路径一致；
- 构造 update-cap backlog，验证 pending 由公式重建而不是从文件读取；
- 恢复后第一次 schedule 不得重置 warm-up anchors；
- 故意让 rollout 初始参数不同，首个 env step 前 full sync 后参数 hash 与 version 对齐；
- 缺 state/部分 checkpoint/错误 rank/world-size/fingerprint 均 fail closed；
- 服务器只做一次 post-ready fresh save → resume → continue，属于同一个 smoke 合同。

## 9. 最小代码改动

| 文件 | 调用链位置 | 计划改动 |
|---|---|---|
| `examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml` | demonstration → Stage 1 | 新增；clean-50、显式 checkpoint stats、frozen π0、image-only token |
| `examples/sft/config/robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke.yaml` | Stage 1 最小真实反传/save | 新增；2 steps、micro/global 1/2 |
| `examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml` | env/feature/MLP/worker 组装 | 新增；同步、H50/C10、z2048、P/D14、full-task、compact replay、resume/sync contract |
| `examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke.yaml` | fresh + resume smoke | 新增；最小 collect/update/eval/DCP 预算 |
| `openpi_action_model.py` | Stage 1 prefix/loss；Stage 2 feature/ref/decode | 窄改；显式 base freeze、raw canonical ref、ephemeral decode context、统一 decoder |
| `rlt/rollout.py` | feature → actor → route → env | 窄改；canonical route 后单次 decode，分开 replay/env action |
| `rlt/route.py` | controller 选择 | 新增 config-driven `FullTaskRLTRoute`；train ready gate，eval student-only |
| `rlt/transition.py` | current/next linker | transition replay 改为 config capability，不硬编码 env enum |
| `fsdp_rlt_ac_policy_worker.py` | replay/reward/schedule/checkpoint | truncation bootstrap；RLT state、contract fingerprint、completion manifest |
| `tests/.../test_robotwin_rlt_contract.py` | 集中验收 | 一组测试覆盖 route、action parity、end mask、schedule/resume/first-sync 合同 |
| `toolkits/rlt/probe_robotwin_rlt_prefix_contract.py` | 无训练真实 checkpoint 门禁 | 新增；prefix/freeze/z/loss/canonical parity |
| `evidence/IMPLEMENTATION_LOG.md` | 证据 | 每批命令、diff、结果、错误、修复和复测 |

预计不改：

- `rlt_token_transformer.py`：已有可配置 input projection 和 reconstruction；本项目配置为
  `2048→2048`；
- `rlt_mlp_policy.py`：canonical tanh actor/twin-Q 接口可复用；
- `replay_buffer.py`：首轮不做通用 hard capacity；
- RoboTwin dataconfig/env、`env_worker.py`、`huggingface_worker.py`、runner 和共享 DCP/sync
  主链；若实现事实证明必须改，先记录精确原因，不预设第二套框架。

## 10. 实施顺序与审批停点

### 阶段 0：实现授权与基线

已完成：

1. 刷新服务器 repo/branch/dirty tree、进程、资源、checkpoint 和数据目标；
2. 从用户指定 `48a775db...` 建立
   `/root/autodl-tmp/RLinf_rlt_pi0_robotwin` 和
   `codex/rlt-pi0-robotwin`；
3. 没有切换/复用 DSRL worktree，没有修改共享 `.venv` 或 editable-install；
4. clean-50 原始 ZIP 已按锁定 revision 下载并独立校验；先以 episode 0 验证
   raw → Aloha → LeRobot converter、三相机、14D 与时序合同，再把全部50条转换为
   7,188-frame canonical 数据。

### 阶段 1：连贯主体实现

已连贯完成 Stage 1 config、Stage 2 config、normalization、canonical decode、
route/transition、bootstrap 和最小 resume。逐项命令、修改、问题和修复见实施账本。

### 阶段 2：服务器集中前置检查

已使用服务器 π0 `.venv` 完成：

1. **已完成**：四份新 config 原生 compose/resolve；ManiSkill RLT、旧 π0 PPO、
   Fast-WAM GRPO 和 DSRL 回归；
2. **Stage 1 smoke 已完成**：真实 checkpoint prefix/mask、单 episode clean-50 loader、
   两卡 micro/global 1/2 两个 optimizer step（其中一次非零 LR）/save/new-process
   reload，以及正式 micro/global 16/32 单步前反传容量门均通过；S1-B 的唯一 update 使用
   warm-up 初始 LR 0，不声称参数 delta；正式2k endpoint、严格 reload、π0 delta=0 和
   true/shuffled/zero 诊断也已完成；
3. **已完成真实 checkpoint 与模拟器执行合同**：canonical reference decode 与旧
   `output_transform(raw_template)[:C,:D]` parity；fresh smoke 已走通真实
   RoboTwin collect、canonical action、eval 与 DCP；
4. **已完成单元合同**：route、current/next、termination/pure truncation；
5. **已完成 Stage 2 fresh smoke**：真实两 rank replay、8 critic/4 actor updates、
   target/sync、deterministic student eval 与 `global_step_1` completion 通过；
6. **部分完成**：Stage 1 artifact 新进程严格 reload 通过；Stage 2 多 rank DCP 已保存，
   但用户明确省略本轮 resume，因此不声称 save→load→continue 已验证；
7. **已完成**：Stage 2 当前代码的 Ruff、py_compile、whitespace 和19个集中单测；
   artifact preflight 与 formal/fresh/resume compose/audit 也通过，且没有启动 Ray。

Windows 本机只编辑文档、代码和 diff，不运行这些项目检查。

### 阶段 3：Stage 1 已验收；Stage 2 fresh smoke 已完成

Stage 1 已按展示后的 resolved packet 执行，结果见
[`02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md`](02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md)。
Stage 2 运行前材料集中在历史批准包
[`04_STAGE2_PRE_SMOKE_PACKET_20260729.md`](04_STAGE2_PRE_SMOKE_PACKET_20260729.md)，包括：

- 完整 resolved config；
- 精确命令与输出/checkpoint 目录；
- Stage 1/Stage 2 artifact ID、norm-stats hash、resume-contract hash；
- 两卡/4-env 资源计划与监控；
- transition/update/eval 预算；
- stop conditions。

用户随后批准 fresh-only 主链 smoke 并明确省略 resume。fresh 于2026-07-30 exit0，
完整结果见
[`05_STAGE2_FRESH_SMOKE_RESULT_20260730.md`](05_STAGE2_FRESH_SMOKE_RESULT_20260730.md)。

fresh smoke 实际覆盖：

```text
reference collect
-> per-rank replay ready
-> critic warm-up
-> update
-> MLP/version sync
-> deterministic student eval
-> DCP
```

只缩短 warm-up、总更新、save/val interval 和 eval episodes；不缩模型、不改 H/C、动作域、
route、loss、sync 或 DCP 主链。resume 未运行。

## 11. 运行前剩余的不确定性

### 11.1 不需用户逐项选择，由事实检查解决

- clean-50 全量转换后的实际有效 episode/frame 数；单 episode loader 已接受；
- 全量 converter instruction/seed 与 manifest；单 episode 三相机、prompt、
  state/action14、50 FPS 和动作时序已通过；
- checkpoint stats 的 key/shape/hash 与 clean-50 诊断分布；
- clean-50 真实 batch 与已验证 image-only prefix `L=768`、width `2048` 的一致性；
- canonical reference/decode parity 与单次预处理；
- termination/truncation/final-observation mask；
- trainer-state 文件、跨 rank completion 和 contract fingerprint。

### 11.2 Stage 1 启动前一次冻结

- 实际使用的全部有效 episode/frame 数；
- Stage 1 steps、global/micro batch、LR 和固定 endpoint；
- resolved config 必须只有 `min_lr_rate=.1`、没有 absolute `min_lr`；CPU scheduler
  contract 已验证 2k 关键步，正式 packet 复核其 hash；
- resolved dataset/checkpoint/stats/manifest/output 路径；
- 若实际有效 episode 明显不是名义 50，是否仍按全部有效数据继续。

不做 2k/5k/10k sweep，不按 Stage 2 成功率挑 Stage 1 checkpoint。

### 11.3 Stage 2 smoke/formal 前一次批准

1. 接受 C10 相对既有 C50 最多约 5 倍 π0 query 和不同反馈频率；
2. 接受 full-task 相对论文 critical-phase 的方法偏离；
3. primitive `gamma`、per-rank collect threshold、post-collect critic updates；
4. batch、sample window、actor/Q LR、tau、fixed std、BC/Q 权重；
5. 总 transitions、episodes/resets、updates、eval seeds/episodes、GPU-hours 与 stop condition。

这里只批准一组推荐值，不先做超参 sweep。先做小规模 frozen-reference C10 sanity；只有出现
灾难性控制退化，才重开 C10/C50。

### 11.4 延后项

- 额外 100/200/400 demos、data-scaling、held-out Stage 1 split；
- joint-SFT、autoregressive decoder、论文 stride-2 overlapping transition；
- RoboTwin critical-phase predicate、human/expert intervention、第二任务；
- hard replay capacity、RNG bitwise resume、跨 world-size、async；
- 完整 C10/C50 对照和大规模超参 sweep。

## 12. 当前停点

主体实现位于从 `48a775db...` 建立的独立
`codex/rlt-pi0-robotwin` worktree。Stage 2 artifact/预算加固代码提交为
`3b610cb4685a1d41c97da64df67ab86561697dfd`；19个集中单测、Ruff、py_compile、
formal/fresh/resume compose/audit 和 launch-script 语法/monitor 自测均通过。实现保持
config opt-in，没有切换或修改主 π0/DSRL worktree，也没有修改共享环境。

full clean-50 已按锁定 ZIP 转换为 50 episodes / 7,188 frames / 50FPS 的版本化 canonical
数据，dataset manifest SHA-256 为 `12ce2ed6...f86c`；正式 global32 loader 两 rank 各
local16 通过。source config 已改为版本化路径并提交
`4ac48d54c63b3a83d99f551fb54f738297525acf`；正式 resolved config SHA-256 为
`5aa824fc...d67e`。

正式2k run 已 exit0，固定 endpoint 为 `global_step_2000`。最后100步 reconstruction
loss 均值0.555，相对最初100步3.902下降85.8%；每卡峰值26,447MiB。artifact 验收用真实
batch 得到 fresh/endpoint/shuffled/zero loss
`5.1977/0.5338/1.7118/2.1027`，非RLT tensor变化数0，manifest与full-weights SHA已固定。
Stage 2 三份 bound config 已替换所有 `UNRESOLVED` 并通过 hard audit。

当前精确停点是：**fresh smoke 已通过；resume 按用户要求省略；不启动 formal pilot，
等待用户在30-cycle phase-transition pilot 与60-cycle初步趋势 pilot之间批准总预算。**
formal source `max_steps=0`，因此预算决定前继续 fail closed。结果、资源和限制见
[`05_STAGE2_FRESH_SMOKE_RESULT_20260730.md`](05_STAGE2_FRESH_SMOKE_RESULT_20260730.md)。
最新现场状态和具体授权以根 [`HANDOFF.md`](../../HANDOFF.md) 与
[`evidence/IMPLEMENTATION_LOG.md`](evidence/IMPLEMENTATION_LOG.md) 为准。
