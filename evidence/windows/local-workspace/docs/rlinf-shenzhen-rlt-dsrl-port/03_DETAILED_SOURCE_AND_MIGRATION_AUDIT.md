# 深圳 current RLinf × RoboTwin π0 × RLT / DSRL：详细源码与迁移审计

最后更新：2026-08-23  
状态：**只读源码与历史证据审计快照；两条实现现已完成并push，终态见
[`04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md`](04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md)。**

本文件是详细证据附录；唯一当前主入口是
[`00_INDEX_AND_MIGRATION_PLAN.md`](00_INDEX_AND_MIGRATION_PLAN.md)，RLT Stage 1 选择的展开说明见
[`01_ONE_PAGE_DECISION_GUIDE.md`](01_ONE_PAGE_DECISION_GUIDE.md)。下文保留源码、LoC 与迁移依据的完整审计。
第二轮按真实调用链逐文件复核见
[`02_CURRENT_PORT_FILE_BY_FILE_READINESS_AUDIT.md`](02_CURRENT_PORT_FILE_BY_FILE_READINESS_AUDIT.md)。
逐命令证据见
[`evidence/PLANNING_AUDIT_LEDGER_20260823.md`](evidence/PLANNING_AUDIT_LEDGER_20260823.md)。旧 AutoDL 文档只作
行为、配置和结果 oracle，不继续充当新 runtime。

## 0. 先给结论

可以看见并恢复之前 AutoDL 上成功的 RLT 和 DSRL；两者的旧源码分支、实现提交、最终结果和配置均已锁定。
它们都可以迁到深圳已经跑通 PPO/GRPO 的 current RLinf base，但不能把旧分支整体 cherry-pick 到新版：

- **DSRL 的 current 算法主体可复用，但 RoboTwin 私有接缝不能省略。** 需要把旧成功实现的 storage、
  macro-transition、动态 UTD、critic-only shadow 与 strict resume 合同重接到新版 data/actor API。
- **RLT 的文本迁移不大，设计分歧比代码量更重要。** 新版已经修复 RLT token reconstruction、增加正式
  schedule/transition replay 和 TD3，但官方示例同时从精确 π0 切到了 `openpi_rlinf` π0.5。若无声照搬官方
  新 YAML，就会同时更换模型、Stage 1 目标和 Stage 2 变体，不能再和旧 AutoDL 成功结果直接比较。
- 建议两算法均从深圳 source lock `7d07a421...` 建**相互独立的 branch/worktree**。RLT 首条主线保持
  `adjust_bottle + 精确 π0 + legacy openpi wrapper` 的模型身份与冻结 VLA/token-only 训练边界，但使用新版
  自回归 RLT reconstruction；Stage 1 在深圳重训，Stage 2 保持旧已验证 AC+BC/Q。π0.5/`openpi_rlinf`
  与 TD3 分别作为后续独立参考轨。
- DSRL 首条主线继续使用 `model_type: openpi`。`openpi_rlinf` 当前没有 DSRL 的 `use_dsrl` / SAC forward
  支持；强行迁过去不是普通适配，而是另一个新实现项目。

## 1. 四个必须分开的源码锚点

| 角色 | 精确 revision | 用途 |
|---|---|---|
| 旧 official common base | `6d0db56bf26f972cd27fa29535f5eb939e80e5bf` | 衡量旧 AutoDL 私有实现从哪里出发 |
| 旧 DSRL final | `48a775db09c16c455aeba7b0600c920e7c80d534` | DSRL 成功行为与结果 oracle；实现主体 commit 为 `6817c73b...` |
| 旧 RLT final | `2b8199d8ab2e7b110994fd3234bf7007196c3af9` | RLT 成功行为与结果 oracle；RLT 自身增量必须从 `48a775db...` 计算 |
| 深圳 current official base | `7d07a4212ee6858cc333e1d4fab7a37256d1f839` | 新实现唯一共同 base；深圳 canonical/PPO/GRPO 已围绕它验证 |

`7d07a421...` 是 `6d0db56...` 的严格后继，中间有 55 个 official commits。上游 `main` 会继续前移；本专题
所说的“current”专指深圳已经安装和验证的 `7d07a421...`，不在迁移中无声 rebase 到浮动 main。

旧 RLT branch 是从含 DSRL 历史的 `48a775db...` 分叉的。因此：

- 算 RLT 自身增量：`48a775db... → 2b8199d8...`；
- 算 DSRL 自身增量：实现 commit `6817c73b...` 相对其有效 official base；
- 新版实现时 RLT 与 DSRL **都独立从 `7d07a421...` 开始**，不把两套算法再次串成祖先关系。

官方 old→深圳 current 的主比较入口：
[RLinf 6d0db56...7d07a421](https://github.com/RLinf/RLinf/compare/6d0db56bf26f972cd27fa29535f5eb939e80e5bf...7d07a4212ee6858cc333e1d4fab7a37256d1f839)。

## 2. 旧成功实现到底有多少代码

### 2.1 精确统计

| 算法 | 统计边界 | runtime 生产代码 | config / tests / tools | 非文档总计 |
|---|---|---:|---:|---:|
| DSRL | 实现 commit `6817c73b...` | 3 files，`+1016/-130` | configs `+253`；tests `+334` | 不含文档为 `+1603/-130` |
| RLT | `48a775db... → 2b8199d8...` | 7 Python files，`+960/-24` | configs `+573`；seed `+27`；tests `+597`；tools `+1243` | 20 files，`+3400/-24` |

这说明两个算法都不是“把三四千行算法重新写一遍”：

- DSRL 真正 runtime 约一千行，配置和测试约六百行；
- RLT runtime 不到一千行。其 worker 的 `+752` 中约 `693` 行是严格 trainer-state、manifest、合同指纹与
  fresh/resume 一致性；π0/RoboTwin 主路径本身约两三百行。

### 2.2 旧 DSRL runtime 的三个文件

| 旧文件 | 增量 | 真正职责 |
|---|---:|---|
| `rlinf/data/replay_buffer.py` | `+409` | RoboTwin macro transition 投影；compact flat ring；capacity/rank/RNG/schema checkpoint |
| `openpi_action_model.py` | `+112/-54` | Gaussian warm-up、32D latent、learned actor phase、三相机冻结 π0 推理与 phase 同步 |
| `fsdp_sac_policy_worker.py` | `+495/-76` | global warm-up、动态 UTD20、梯度隔离、10-Q、critic-only FP32 target shadow、完整恢复 |

旧 DSRL **没有重写**官方 `gaussian_policy.py`、`compact_encoders.py` 或 SAC objective。它是在官方已有
LIBERO DSRL 骨架上补齐 RoboTwin 的交互粒度、在线训练和可靠恢复语义。

旧成功终态：formal 按授权在 step 198 收尾，最新 DCP195；共 792 train episodes、5,185 个有效 macro
transitions、94,260 次 SAC updates；learned-actor train `624/740=84.32%`，step65–195 fixed eval 汇总
`123/132=93.18%`。

### 2.3 旧 RLT runtime 的七个文件

| 旧区域 | 旧增量要点 |
|---|---|
| `algorithms/rlt/{route,rollout,transition}.py` | RoboTwin full-task reference→student route、transition replay 选择、executed-action 合同 |
| legacy `openpi_action_model.py` | token-only/frozen VLA 开关、14D canonical adapter、decode context 与 route 后单次 decode |
| `fsdp_rlt_ac_policy_worker.py` | compact transition、time-limit bootstrap、schedule 计数与严格 trainer-state/resume |
| init/seed | 注册与 fixed-20 seeds |

旧 RLT 没有重写 official RLT token transformer、RLT MLP policy 或 AC objective。

还要注意一个 ancestry 边界：RLT 的实际 fork 点 `48a775db...` 已经包含旧 DSRL 对 legacy OpenPI 与 SAC worker
的 `+607` 行共享热路径修改。它们不属于 RLT 增量，多数受 `use_dsrl` gate 保护，**不应迁入 RLT branch**；
但旧 RLT 确实曾在这套增强后的 SAC 基类上运行，所以 current port 必须用旧 250→480 的行为/resume 测试重新
确认 replay、target 和 save/load，而不能只根据 RLT 自身 `960` 行 patch 是否可应用来判断等价。

旧成功锚点：8-env×250 formal 自然 exit 0，fixed-20 为 `18/20`；resume 250→480 自然 exit 0，十次
fixed-20 聚合 `178/200=89%`，最后一次 `17/20`。这些只作行为和结果 oracle，不直接证明新版实现正确。

## 3. 深圳 current RLinf 新了什么

全仓 old→current 是 669 files、约 `+58,769/-5,824`，但大头是新增 generation/OpenPI 栈，不能拿这个数字
冒充 RLT/DSRL 迁移量。与本专题直接相关的是以下变化。

| official commit / 变化 | 对本任务的意义 |
|---|---|
| `d3aff547...` data module restructuring | replay 从 `rlinf/data/replay_buffer.py` 移到 `rlinf/data/storage/replay/buffer.py`；`Trajectory` 移到 `data/schema/embodied_types.py` |
| `9ad44393...` actor worker split | `EmbodiedFSDPActor` 迁到 `embodied_fsdp_actor_worker.py`；旧 worker import 不能原抄 |
| `8587bac4...` RLT reconstruction fix | 由 parallel reconstruction 改为 teacher-forced causal autoregressive；single RL token；补 mask/dtype/device 合同 |
| `c704688c...` RLT OpenPI migration | 官方 RLT 示例迁到 JAX-aligned vendored `openpi_rlinf` π0.5；这是并行新 wrapper，不是简单改名 |
| `13e5b652...` RLT TD3 | 新增 Stage 2 TD3 变体；不是旧 AC 成功实验的自动替代 |
| route/intervention 更新 | expert takeover 时同步修正 `ref_chunk`；ManiSkill critical-phase 路线更完整 |
| rollout/schema 更新 | `EmbodiedRolloutResult/rollout_results` 演进为 `EmbodiedTrajectoryBuilder/trajectory_builders` 与 `PolicyOutput` |
| runner sync 修复 | `set_global_step(...)` 后显式等待再同步，降低 step/version 竞态 |
| RTC / 动作合同 | 旧基线已有 singular `action/model_action` 训练合同；current 为 eval-only RTC 新增 top-level plural `model_actions` 与 `rtc_context`。迁移保留 API，但首版不启用 RTC |

### 3.1 DSRL 算法主体稳定，但私有 RoboTwin 合同仍需重接

以下 old/current Git object 完全相同或算法语义无变化：RoboTwin env、RoboTwin adjust-bottle env/PPO YAML、
official LIBERO DSRL YAML、`gaussian_policy.py`、`compact_encoders.py`、legacy OpenPI 的 DSRL 主体。

current SAC worker 的 DSRL diff 主要是新 import、actor 基类和平台抽象。all-model FP32 target shadow 早在旧
official base 就存在，current 只是继续保留；但：

- shadow 覆盖全模型，旧 RoboTwin DSRL 特意只维护 critic，避免复制巨大冻结 π0；
- official save/load 仍不保存 `update_step`、DSRL phase、FP32 shadow、pending counter 与精确 RNG state；
- official 仍按固定 `update_epoch: 200`，没有旧 port 的“每个新增有效 macro transition 对应 UTD20”。

因此新版没有替代旧 RoboTwin DSRL patch，迁移仍有必要。

### 3.2 RLT 相关变化是真正的语义变化

current 已吸收或新增：

- 自回归 RLT reconstruction 与 single RL token；
- transition replay/schedule/global ingest counters；
- actor BC/Q weight schedule；
- ManiSkill reference/critical-phase/expert 路线；
- `openpi_rlinf` π0.5 Stage 1/2 与 TD3 variant；
- official `robotwin_sft_openpi_rlinf.yaml` 与 π0 RoboTwin eval config，可复用其三相机、`H=50/D=14` 数据合同。

current **没有**吸收旧 RoboTwin 成功 patch 的：

- `rlt_train_vla`；
- `rlt_action_adapter=robotwin_aloha_canonical_v1`；
- `FullTaskRLTRoute`；
- RoboTwin `rlt_transition_replay` 与 pure time-limit truncation bootstrap；
- route 后单次 canonical decode；
- `rlt_resume` trainer-state/manifest/合同指纹。

还有一个关键分派缺口：current 的 `use_simulator_transition_replay(cfg)` 只认 ManiSkill RLT env；普通 RoboTwin
会落到 Realworld route，不会仅靠一份 YAML 自动获得 simulator transition replay 与 schedule。这是 RLT 新 port
必须显式解决的核心接缝。

## 4. RLT：需要先做的设计选择

### 4.1 不应把三个变量一次全换掉

可以把候选路线写成三层：

| 路线 | 模型身份 | RLT reconstruction / Stage 1 | 用途 |
|---|---|---|---|
| 历史精确复现 | 精确 π0 + legacy `openpi` | 旧 parallel decoder；冻结 VLA/token-only | 只在必须复刻旧数值时使用 |
| **推荐 current 主线** | **精确 π0 + legacy `openpi`** | **current AR decoder；冻结 VLA/token-only；Stage 1 重训** | 只更新明确的上游 correctness，同时保持深圳 π0 baseline 身份 |
| official-current 参考轨 | π0.5 + `openpi_rlinf` | current AR；joint `rlt_loss + alpha*vla_loss` | 研究新版官方 recipe，不能和旧 π0 结果直接等价 |

推荐中间路线的原因：

1. 深圳 PPO/GRPO 和现有 SFT/eval 都已经锁定精确 π0；不应把模型升级混进“算法 port”。
2. legacy `openpi` 在 `7d07` 仍保留 current RLT transformer、`use_rlt`、`rlt_alpha`、`extract_rlt_obs`，不是退回旧仓。
3. current AR reconstruction 是明确的上游 correctness 修复，旧 Stage 1 checkpoint 的 parallel decoder 参数/schema
   不应硬塞进来。clean-50、norm stats 与 14D action 合同可以复用，Stage 1 应重训。
4. `rlt_train_vla=false / rlt_alpha=0` 继续冻结现有 π0，只训练 RLT token 模块，避免在“RLT port”里同时改变
   PPO/GRPO 已验证的 base policy。
5. 旧 AutoDL Stage 1 只约 29 分钟；相比维护 legacy decoder 兼容层，重训更清楚。

需要用户最终确认的不是“用新还是旧仓”——仓统一用 `7d07`；而是首条 RLT baseline 是否接受：

> 精确 π0 模型身份与冻结/token-only 边界不变，但采用 current AR RLT token 目标并重训 Stage 1。

### 4.2 Stage 2 首版保持 AC，不切 TD3

旧成功结果是 AC actor + critic、BC/Q loss。current 新增 TD3 是有价值的上游 baseline，但它改变 policy/update
objective。首版迁移保持 AC，闭合旧行为 oracle后再增加 TD3 config；不能用“新版有 TD3”当作替换旧算法的理由。

### 4.3 RoboTwin route 应保持 full-task，而不是伪造 critical phase

旧 `adjust_bottle` 没有 ManiSkill peg-insertion 的 `in_critical_phase` 和 expert takeover 语义。首版应明确实现：

```text
reference warm-up → student full-task rollout
train/eval 均不使用 expert
只记录真实执行的 chunk/action
route 完成后只 decode 一次
```

若以后研究 phase-based RLT，应另加真正可审计的 RoboTwin phase 定义，不能为了复用 ManiSkill route 而把
`critical_phase=True` 当成无含义常量。

### 4.4 current schedule 可复用，strict resume 仍要补

current 已有 warm-up、transition/episode counters、pending update budget 和 max-updates 控制；这些不重写。
old→current 的 generic SAC save/load 内容基本未变，仍不保存 RLT 的 update/schedule anchor。为支持长 formal 的
fresh→resume 等价性，应把旧已验证 sidecar 重接到 current worker：

- runner step、update step；
- total transitions / warm-up anchor；
- `pending_update_budget`、`transitions_since_train`、`episodes_since_train` 不直接落盘：旧成功实现以完整
  outer-step 为 checkpoint 边界，恢复时置零，再由已恢复的 totals / warm-up anchor / `update_step` 重算；
- replay 内容与 seed；旧 RLT 的 generic replay 未保存当前 generator state，因此不把恢复描述成未来采样序列的
  bitwise 精确续接；
- Stage 1/model/norm/action/route 合同指纹；
- 多 rank 完整性 manifest。

这部分是旧 RLT 最大代码块，也是“可跑”和“可可靠续训”的主要差别。

## 5. DSRL：迁移设计

### 5.1 保持 legacy OpenPI wrapper

current `openpi_rlinf` 没有 DSRL `use_dsrl` / SAC forward 支持。首版保持：

```text
model_type: openpi
frozen exact π0
32D latent repeated across H=50
Gaussian warm-up → learned actor phase
10-Q critic
```

把 DSRL 搬到 `openpi_rlinf` 会同时要求重写 latent injection、SAC forward、checkpoint 与 rollout contracts；这不属于
本轮普通迁移。

### 5.2 macro transition 的算法合同必须原样保留

RoboTwin 一次 policy query 执行一段 action chunk，不能把完整 trajectory 数当 replay transition 数。迁移必须继续：

- 以 first-done 截断，删除 terminal 后 padding；
- 成功 reward 为 0，其余有效 macro step 为 -1；
- success termination 不 bootstrap；纯 time-limit truncation 继续 bootstrap；
- macro discount 为 `gamma ** N`；
- global warm-up 按所有 actor ranks 的**有效 macro transition**计数；
- 更新预算为 `UTD20 × 本轮新增有效 macro transition`，不退回固定 `update_epoch=200`。

### 5.3 replay 在新 data 架构下有两种实现方式

| 方案 | 优点 | 风险/代价 |
|---|---|---|
| 新建 `data/storage/replay/dsrl_transition.py` | 语义清楚，保留 compact ring、capacity、RNG 与 strict resume；不污染 generic buffer | 需迁约 400 行并适配新 schema |
| 把每个 macro transition 编成 length-1 generic trajectory | 复用 official storage/checkpoint | 必须证明 capacity 是 transition 容量、采样分布/RNG/恢复完全等价；现有 generic checkpoint只恢复 seed，不恢复精确 RNG state |

当前推荐第一种。它不是重复造一套通用 replay，而是为 DSRL 明确保留“固定容量、均匀有放回、逐 transition、
可精确恢复”的小后端，并通过 `storage/replay/__init__.py` 导出；`ReplayBufferDataset` 只依赖 `is_ready/sample`，可继续复用。

### 5.4 target shadow 与 resume

all-model FP32 shadow 不是 current 新增：旧 official base 已经如此，旧 RoboTwin 私有 DSRL port 才把它收窄，
而这项私有改进没有 upstream。迁移应把旧已验证语义重新接回 current：

- 只 shadow critic/target-Q；
- FP32 EMA 累积，再写回运行 dtype；
- 保存/恢复 shadow、phase、update step、pending count、replay cursor/resident/total 与 RNG；
- FSDP reload 后不允许 policy phase 回到 Gaussian warm-up。

## 6. 逐文件迁移地图

### 6.1 RLT

| 旧语义 | current 目标位置 | 处理方式 |
|---|---|---|
| full-task route | `rlinf/algorithms/rlt/route.py` | 在 current route abstraction 内新增 RoboTwin/full-task variant；保留 intervention/ref_chunk 新合同 |
| simulator transition switch | `algorithms/rlt/transition.py` 与 rollout 调用 | 不再用旧 env type 硬补丁；增加显式、opt-in 的 RoboTwin transition mode |
| 14D adapter / decode context | legacy `openpi/openpi_action_model.py` | 逐 symbol 迁；保留 current RTC 字段与动作合同，但首版 RTC 关闭 |
| schedule | current `fsdp_rlt_ac_policy_worker.py` | 直接复用 current；不复制旧 schedule 主体 |
| compact/truncation | current replay mixin/worker | 只补缺失的 executed-action、terminal/truncation 合同 |
| strong resume | current RLT worker + sidecar | 迁旧实现已验证的模型/优化器、replay 内容、update/warm-up counters、manifest 与合同；generic replay RNG 精确续接若新增，单列为可靠性增强 |
| Stage 1/2 configs | 新 RoboTwin YAML | 三列 provenance：current official、旧成功语义、深圳路径/资源；不搬 A800 路径 |

### 6.2 DSRL

| 旧语义 | current 目标位置 | 处理方式 |
|---|---|---|
| projection + flat ring | 新 `data/storage/replay/dsrl_transition.py` | 适配 `schema.embodied_types.Trajectory`；独立导出 |
| latent/phase | legacy `openpi/openpi_action_model.py` | 保留 current RTC/动作字段与现有 transforms，但 DSRL 首版 RTC 关闭 |
| worker wiring/UTD | current `fsdp_sac_policy_worker.py` | opt-in DSRL 分支；保留新基类和 `Worker.torch_platform` |
| critic-only shadow/resume | 同 worker + sidecar | 替换 DSRL 下 all-model shadow；完整状态恢复 |
| configs/tests | 新深圳 RoboTwin DSRL YAML + focused tests | 参数来源明确，不照搬 A800 GPU/path/monitor |

## 7. 新实现的预估代码量

以下是**规划区间，不是已产生的 diff**；精确数字只能在实现完成后统计。

| 路线 | 预估 runtime 增量 | 为什么可能小于旧实现 |
|---|---:|---|
| RLT current-π0 主线，不含 strict resume | 约 `250–350` 行 | current 已有 schedule/transition replay/AR reconstruction；主要剩 adapter、route 与 truncation |
| RLT current-π0 主线，含 strict resume | 约 `900–1000` 行 | strict trainer-state/manifest 本身接近七百行；需要对接 current SAC/data/checkpoint API |
| DSRL current-base port | 约 `800–1100` 行 | 算法本体可复用，但 compact replay、动态 UTD、critic-only shadow 与完整恢复仍需保留 |

RLT 若暂时不做 strict resume，能少数百行，但只能算机制 smoke 版本，不适合作为 250/480-cycle formal 的可靠实现。
DSRL 若 generic replay 的固定 fixture 能证明完全等价，可能进一步缩短；在证明前不把它写成既定事实。

## 8. Git / worktree 拓扑

建议在同一个服务器 Git object store 下建立两棵独立 worktree：

```text
/data/chenyiteng/projects/rlinf-shenzhen/RLinf                  # canonical 7d07, detached/clean
/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-current
  branch: codex/sz-current-rlt-pi0-robotwin
/data/chenyiteng/projects/rlinf-shenzhen/worktrees/dsrl-pi0-robotwin-current
  branch: codex/sz-current-dsrl-pi0-robotwin
```

优点：共享 Git objects、不重复 clone；RLT 与 DSRL 的 OpenPI/SAC worker 修改互不污染；可以分别 review、push、
回滚或继续开发。两支都从 `7d07a421...` 创建，不能从 PPO、GRPO、旧 DSRL 或旧 RLT branch 相互叠加。

服务器 GitHub 账号级 key 与 `gh` 登录已存在；创建 branch/worktree 不需要网页操作。首次普通 push 前只需确认
remote 和 branch target，不需要再建立 deploy key。

## 9. 建议实施顺序与少量高信息检查

### P0：共同 source lock（一次）

- live 确认 `7d07a421...`、canonical clean、目标 worktree 不存在；
- 建两支独立 branch/worktree；
- 从第一条修改开始分别维护 implementation ledger。

### P1：RLT current-π0 主线

1. 在 current legacy π0 上接 RoboTwin 14D adapter、full-task route 和 transition 合同；
2. 基于 official RoboTwin SFT config 新增 RLT Stage 1 config，采用 current AR token 目标；
3. 迁 strict resume 的最小状态集；
4. 迁两组旧合同测试，并改到新 `PolicyOutput/TrajectoryBuilder/schema/storage` 接口；
5. 一个集中静态/fixture 批次；
6. 经批准后：一次真实 Stage 1 短 smoke、一次 Stage 2 fresh one-cycle + resume one-cycle；通过后再给 formal packet。

### P2：DSRL current-base port

1. 先迁 macro projection + 独立 compact replay 与 fixed fixtures；
2. 迁 π0 latent/phase；
3. 迁 worker 的 global count、UTD20、gradient isolation、critic-only shadow 与 resume；
4. 一个集中静态/fixture 批次；
5. 经批准后：一次 fresh one-cycle + resume one-cycle；通过后再给 pilot/formal packet。

### P3：可选参考轨

- RLT π0.5 / `openpi_rlinf` official-current baseline；
- RLT TD3 Stage 2；
- DSRL on `openpi_rlinf`（这是新研究实现，不是迁移）。

真实 smoke/formal 前另给完整 resolved config、精确命令、输出目录、sample/update 预算、GPU/RAM 预估、监控项与
停止条件，并等用户批准。本轮不创建这些运行。

## 10. 当前真正需要用户决定的设计问题

只有一项：

1. **RLT 首条主线是否采用：精确 π0 与冻结/token-only 边界不变 + current AR RLT reconstruction + Stage 1 重训？**
   这既不是旧 decoder 数值级复刻，也不是 π0.5 模型升级；推荐采用。三条路线的通俗比较见
   [`01_ONE_PAGE_DECISION_GUIDE.md`](01_ONE_PAGE_DECISION_GUIDE.md)。

以下不再作为用户的方法选择：

- RLT strict resume 是长 formal 前必须补齐的工程完整性；
- DSRL 采用独立 compact transition backend，以保留旧成功语义与精确恢复；
- 两算法使用独立 branch/worktree，首轮均保持精确 π0；
- 实现顺序暂按 RLT 后 DSRL，若用户改变任务优先级再调整。

## 11. 当前授权边界与下一断点

本轮已授权的是调查、整理上下文和规划。已完成：

- 旧分支/commits/results/LoC 只读审计；
- 深圳服务器 current source/worktree 只读刷新；
- official old→current 关键接口/语义审计；
- 第二轮 RLT/DSRL 逐文件调用链、RTC、checkpoint/replay/shadow 因果复核；
- 本专题计划与逐操作账。

尚未执行：服务器 branch/worktree 创建、代码修改、依赖安装、测试、模型下载、smoke 或 formal。

下一断点：先讨论并锁定第 10 节第 1 项；确定后依据
[`02_CURRENT_PORT_FILE_BY_FILE_READINESS_AUDIT.md`](02_CURRENT_PORT_FILE_BY_FILE_READINESS_AUDIT.md) 生成第一批
精确文件级实施 packet。

## 12. 参考入口

- 旧 RLT 主入口：[`../rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md`](../rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md)
- 旧 RLT 250 结果：[`../rlinf-robotwin-pi0-rltoken/15_STAGE2_FORMAL_8ENV250_FINAL_RESULT_20260730.md`](../rlinf-robotwin-pi0-rltoken/15_STAGE2_FORMAL_8ENV250_FINAL_RESULT_20260730.md)
- 旧 RLT 480 结果：[`../rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md`](../rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md)
- 旧 DSRL 主入口：[`../rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md`](../rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md)
- 旧 DSRL closeout：[`../rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_CLOSEOUT_REPORT_STEP198_20260729.md`](../rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_CLOSEOUT_REPORT_STEP198_20260729.md)
- 深圳 current 主入口：[`../rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md`](../rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md)
- current official RLT guide：[`../../.tmp/rlinf_7d07_source_20260823/docs/source-en/rst_source/examples/embodied/rlt.rst`](../../.tmp/rlinf_7d07_source_20260823/docs/source-en/rst_source/examples/embodied/rlt.rst)
- current official DSRL guide：[`../../.tmp/rlinf_7d07_source_20260823/docs/source-en/rst_source/examples/embodied/dsrl.rst`](../../.tmp/rlinf_7d07_source_20260823/docs/source-en/rst_source/examples/embodied/dsrl.rst)
