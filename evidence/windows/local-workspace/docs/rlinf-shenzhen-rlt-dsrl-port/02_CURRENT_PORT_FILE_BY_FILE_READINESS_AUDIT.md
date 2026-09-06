# RLT / DSRL current-base 逐文件迁移复核

最后更新：2026-08-23  
状态：只读审计完成；尚未创建服务器 worktree、修改代码或运行测试/实验。

唯一当前主入口为
[`00_INDEX_AND_MIGRATION_PLAN.md`](00_INDEX_AND_MIGRATION_PLAN.md)；本文件只作逐文件工程附录。

## 0. 本轮复核后的简明结论

1. **current RLinf 的正式 RLT token decoder 已经是 causal AR。** 旧 AutoDL port 使用 parallel，不是我们自行
   发明了一套旧目标，而是继承了当时 RLinf 官方唯一实现；upstream 在旧实验结束后才用 `8587bac4...` 修正。
2. **RTC 不是 π0 默认训练/推理。** generic model 默认关闭，训练路径不使用，只有专用 RTC eval runner/YAML
   才启用。RLT/DSRL 首版只保留 current API，不打开 RTC。
3. **generic SAC checkpoint 内容从旧 base 到 `7d07` 基本未变。** 变化主要是 data import、actor 基类位置和
   `Worker.torch_platform`；strict resume 是把旧私有 sidecar 重新接入，不是为新 checkpoint 格式重写。
4. **DSRL compact ring 与 critic-only FP32 shadow 也不是 current 新问题。** 它们是旧 RoboTwin 私有 port 已经
   验证、但从未 upstream 的增量；current data 重构后要换宿主路径并重新接线。
5. 逐文件看完后，首版仍只有一个科学定义需要用户确认：**RLT Stage 1 用旧 parallel，还是 current AR**。
   推荐 current AR。其余是有直接旧实现/current API依据的工程迁移。

## 1. RLT reconstruction 的准确时间线

| 时间/commit | 当时发生了什么 | 对本任务的含义 |
|---|---|---|
| 2026-07-06 `5769c6eb...` | RLinf 首次加入 RLT；decoder 用 learned position queries 并行重建 prefix | 这是旧 AutoDL 开发时继承的官方实现 |
| 2026-07-29 `1a923a23...` | 我们增加 RoboTwin adapter、full-task route、transition/truncation 与 resume 接缝 | 没有修改 RLT token transformer/decoder |
| 2026-07-31 `2b8199d8...` | 旧 RLT 分支收尾 | 仍是 parallel decoder |
| 2026-08-10 `8587bac4...` | upstream 以 `fix: update the logic of reconstruction in RLT` 改成 teacher-forced causal AR，并增加 causal 单测 | `7d07` 已包含；旧私人分支没有同步这项修复 |

旧实验当时已经明确记录“论文 AR、RLinf parallel、首版继承 RLinf”，不是事后才发现：

- [`旧主计划：decoder 选择`](../rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md)
- [`旧 Stage 1 方法对齐`](../rlinf-robotwin-pi0-rltoken/02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md)
- [`旧实施账本`](../rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md)

current AR 的直接依据：

- [RLT 论文 Eq. (2)](https://arxiv.org/html/2604.23073#S4.SS1)
- [upstream reconstruction fix](https://github.com/RLinf/RLinf/commit/8587bac453f03951eeaf4a871c050bf1647e7276)
- [对应 issue](https://github.com/RLinf/RLinf/issues/1391) 与 [PR](https://github.com/RLinf/RLinf/pull/1425)

因此推荐 current AR 的理由是“论文一致 + current 唯一维护路径 + 不维护已修旧 decoder”，不是声称 AR 已在
RoboTwin 上实证优于 parallel；目前没有同数据、同 seed 的受控 A/B。

## 2. 不要把四类变化混在一起

| 类别 | 例子 | 迁移方式 |
|---|---|---|
| current 真正改了接口 | `schema/storage`、`PolicyOutput → ChunkStepResult → Builder`、actor/platform 路径 | 按 current API 重接旧语义 |
| current 真正修了算法 | RLT parallel → causal AR | 用户确认后采用 current 正式实现，Stage 1 重训 |
| current 新增但本算法不用 | eval-only RTC、π0.5 `openpi_rlinf`、RLT TD3 | 保留兼容接口；首版关闭/不选 |
| 旧私有增量未 upstream | full-task route、14D adapter、truncation、strict resume、DSRL ring、critic-only shadow、动态 UTD20 | 从旧已验证实现做 symbol-level rebase |

这就是“新版影响迁移”的准确边界：只有前两类由新版直接造成；第四类不是新版制造障碍，而是从 clean current
开始时不能遗漏旧成功实现的必要增量。

## 3. RTC 与动作空间合同

### 3.1 RTC 到底是什么、默认是否启用

RTC 用上一次 action chunk 尚未执行的尾部去约束下一次重新采样，使重叠 chunk 更连续。它由
[`657faae5...`](https://github.com/RLinf/RLinf/commit/657faae51ee266cadc5d3e64f88a7dc2a4bd409b)
加入，定位是 evaluation/deployment：

- `OpenPi0Config.rtc_enabled` generic 默认是 `False`；
- 只有 `runner.rtc.enabled=true` 才选择 RTC 专用 env/rollout worker；
- 模型还要同时 `rtc_enabled=true`，并且 `mode="eval"`；
- 训练路径不启用 RTC。

所以“新版 π0 有 RTC”正确，“默认训练和推理都用 RTC”不正确。

### 3.2 RLT / DSRL 要不要接 RTC

首版**不接算法语义，只保留 current API**：

- DSRL 在 `use_dsrl` 分支先产生 32D latent，再调用普通 flow sampler；当前代码会绕开 RTC 分支。
- RLT Stage 2 由 `rlt_mlp_policy` 产生 canonical chunk，再由 route/decoder 执行；它没有 current 官方的
  `rtc_context → flow guidance` 完整链。
- 若将来研究 RLT/DSRL + RTC，应作为单独 control-mode baseline，重新定义 previous model-space chunk、route 后
  chunk 与执行 horizon 的关系，不能只切一个 YAML 开关。

另一个纠正：旧 base 已经有 singular `forward_inputs["action"]` 和 `forward_inputs["model_action"]`。RTC 新增的是
top-level plural `result["model_actions"]` 与 `rtc_context`。迁移时应保留：

- `policy_output.actions`：真正送入环境的动作；
- `forward_inputs["action"]`：学习算法要回放的动作；DSRL 中是 latent，RLT 中是 routed canonical action；
- `forward_inputs["model_action"]`：π0 原始模型输出；
- top-level `model_actions`：RTC worker 的 current API，首版存在但不消费。

## 4. RLT 逐文件实施地图

先锁一个高风险边界：旧 RLT fork 点 `48a775db...` 已经包含此前 DSRL 对 legacy OpenPI/SAC 共享热路径的修改。
“RLT 自身 7 个 runtime 文件、约 `+960/-24`”是相对这个 fork 点的增量。current RLT worktree 不能整体
cherry-pick旧分支或覆盖整份 OpenPI/SAC worker，否则会把 DSRL latent/phase/shadow 一起带入；只取下表的
RLT-gated symbols。

| current 文件 | 动作 | 精确内容 | 依据/不能做什么 |
|---|---|---|---|
| `rlinf/models/embodiment/modules/rlt_token_transformer.py` | **不改** | 直接使用 current causal AR、single `z_rl` 和现有 mask/dtype/device 合同 | 不搬旧 parallel decoder，不加载旧 Stage 1 decoder 权重 |
| `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py` | **不改** | 直接复用 current Stage 2 policy | 不恢复旧 token 数字段 |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | symbol-level port | 加 `rlt_train_vla`、`rlt_action_adapter`；token-only Stage 1；14D canonical `ref_chunk`；保存 decode context；route 后只做一次 output transform | 在 current 文件上补方法；不能整文件覆盖 RTC、新 preprocessing 或其他 current 分支 |
| `rlinf/models/embodiment/openpi/__init__.py` | 小改 | `use_rlt && !rlt_train_vla` 时冻结 base VLA，只开放 `rlt_module` | 首版锁 `rlt_train_vla=false, rlt_alpha=0` |
| `rlinf/algorithms/rlt/route.py` | 小改 | 在 current route abstraction 内加 `FullTaskRLTRoute`；warm-up 用 reference，之后 student 完整任务；eval 始终 student | 保留 current expert takeover 对 `ref_chunk` 的修正；不伪造 ManiSkill critical phase |
| `rlinf/algorithms/rlt/transition.py` | 小改 | `algorithm.rlt_transition_replay.enable` 显式 opt-in RoboTwin simulator transition | 不靠 env type 误判成 real-world route |
| `rlinf/algorithms/rlt/rollout.py` | 小改 | feature model 提取 canonical obs + decode context；route canonical action；route 后 decode 一次送环境 | `forward_inputs["action"]` 仍保留 canonical learning action |
| `rlinf/algorithms/rlt/__init__.py` | 极小改 | 只导出新增 full-task route | 不搬其他旧模块 |
| `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` | 复用主体 + 窄补 | current AC loss/schedule/replay 主体不抄；只补 pure time-limit bootstrap、old strict-resume sidecar/manifest、合同 fingerprint | 保留 current worker/platform；generic replay RNG 不冒充 bitwise resume |
| `data/schema/embodied_types.py`、`embodied_trajectory_builder.py` | **首版不改** | 直接消费 current `PolicyOutput/ChunkStepResult/Trajectory` | 用 fixture 证明 canonical action、reward、done、next obs 对齐；发现合同缺口才改 schema |
| RoboTwin Stage 1/2 YAML | 新文件 | current official 字段 + 旧成功算法语义 + 深圳路径/4卡资源三列 provenance | Stage 1 fresh AR；Stage 2 AC，不切 TD3/π0.5 |
| focused tests | 迁旧 + 补 current fixture | freeze、adapter/decode-once、route、transition opt-in、truncation、resume manifest；current AR causal 单测继续跑 | 不增加经验阈值 gate |
| `toolkits/rlt/*` | 更新四个旧工具 | 删除 `rlt_num_rl_tokens` 假设；把旧 `decode(z, seq_len)` 改为 current `decode(z, target_embeddings, mask)`；更新 Stage 1 manifest | 工具也必须跟随 AR API，不能只改 runtime |

`schema`、Builder、EnvWorker、HF rollout worker、runner、train entrypoint 与 FSDP strategy 均列为首版 no-touch；
只有 fixture 证明 current 合同无法承载旧语义时才扩大修改面。

### RLT 最容易漏的五个点

1. current legacy OpenPI 默认仍算 `rlt_loss + rlt_alpha * vla_loss`；旧成功 baseline 是 frozen/token-only，所以必须
   恢复 `rlt_train_vla=false` 且锁 `rlt_alpha=0`，否则“采用 AR”会无声变成 joint VLA/RLT 训练。
2. RoboTwin canonical action 是 `H=50, D=14` 的学习空间；route 在 canonical 空间决定 reference/student，最后
   才用保存的 raw template/state 做一次 output transform。先 decode 再 route或 route 后重复 decode 都不等价。
3. current 对任何 `done` 都把 `next_obs=curr_obs`；旧 RoboTwin port 对 pure time-limit truncation 保留真实
   `next_obs`，critic bootstrap 看 `termination` 而非 `done`。这项要原样重接。
4. 旧 RLT resume 不直接保存 pending budget；checkpoint 位于完整 outer-step 边界，恢复 totals/warm-up anchor/
   `update_step` 后将 cycle-local 三项置零，再由 current schedule 重算。
5. 旧 parallel Stage 1 的 shuffled/zero-`z_rl` reconstruction probe 在 AR teacher forcing 下不再适合作硬 gate：decoder
   本来就能读取前序真实 embedding。首版硬合同应改为 finite loss、base 真冻结、current causal 单测和
   save→fresh reload；旧 probe 只保留为诊断。

## 5. DSRL 逐文件实施地图

| current 文件 | 动作 | 精确内容 | 依据/不能做什么 |
|---|---|---|---|
| 新 `rlinf/data/storage/replay/dsrl_transition.py` | 新文件 | 迁 `project_dsrl_trajectory` 与固定容量 compact ring；适配 current `schema.embodied_types.Trajectory` | 不把旧类重新塞回 generic `buffer.py` |
| `rlinf/data/storage/replay/__init__.py` | 小改 | 导出 DSRL projection/ring | generic replay 保持 untouched |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | symbol-level port | 64×64 main-image preprocessing；Gaussian→learned persistent phase；latent H-repeat；显式 eval stochasticity 开关 | 保留 current RTC/动作返回字段；旧 formal 锁 `dsrl_eval_deterministic=false`；DSRL 分支仍显式绕开 RTC |
| `rlinf/workers/actor/fsdp_sac_policy_worker.py` | 主要 rebase 文件 | opt-in ring；Builder trajectory→macro transition；global resident/new count；动态 `UTD20 × global_new`；梯度隔离；critic-only FP32 shadow；sidecar/save/load | 保留 current actor base、platform API、generic non-DSRL 行为 |
| `gaussian_policy.py`、`compact_encoders.py` | **不改** | old/current 算法 object 一致，直接复用 | 不重写 official DSRL 本体 |
| `embodied_types.py`、Builder | **首版不改** | 把 current Builder 产物作为 projection 输入 | fixture 覆盖 first-done、leading bootstrap、reward/termination/truncation shape |
| RoboTwin DSRL YAML | 新文件 | exact π0、`N=20/H=50/latent32/10-Q/UTD20`、固定 global replay capacity、4卡 rank 合同 | 不照搬 A800 路径；不使用 fixed `update_epoch: 200` 替代动态 UTD |
| focused tests | 迁旧 | capacity 5/sample 8 有放回；ring roundtrip 后下一批逐 tensor 相同；critic shadow allowlist/EMA/resume；phase/global count/Builder fixture | 先测确定合同，不加额外机制 gate |

current Builder 的关键动作血缘必须用 fixture 明确：EnvWorker 执行 `PolicyOutput.actions`（env-space），但
`ChunkStepResult.actions` 取 `policy_output.forward_inputs["action"]`。DSRL 分支故意把后者覆写为在 `H=50`
上重复的 32D latent，因此 `Trajectory.actions` 才是 macro projector 的输入；top-level `model_actions` 与
`forward_inputs["model_action"]` 都不是 DSRL replay latent。

### DSRL compact ring：新版改了什么、为什么仍要迁

upstream `d3aff547...` 做的是 data 目录重构：旧 `data/replay_buffer.py` 基本 rename 到
`data/storage/replay/buffer.py`，`Trajectory` 搬到 `data/schema/embodied_types.py`。generic replay 算法没有变成
DSRL 所需 ring；旧私人 `DSRLTransitionReplayBuffer` 也从未 upstream。

两者语义不同：

| 合同 | current generic replay | 旧 DSRL compact ring |
|---|---|---|
| 存储单位 | trajectory | 每 `N=20` 执行段一个 macro transition |
| 容量 | trajectory/window 语义 | 固定 global transition capacity，按 rank 分片并环形覆盖 |
| 采样 | requested 过大时受 resident/window 限制 | 均匀、有放回，可 sample batch > resident |
| payload | 可保留完整 trajectory/multi-camera | 只保留 64×64 main image、state、32D latent、reward/done/discount |
| resume RNG | 保存 seed，load 后重新 seed | 保存 generator 当前 state，下一批可逐 tensor 续接 |

因此迁 ring 不是为新版重复造通用设施；是避免把 DSRL 的采样单位、容量、内存和恢复语义换掉。

### DSRL FP32 shadow：新版改了什么、为什么仍要收窄

all-model FP32 shadow 在旧 official `42530b72...` 就已存在，原因是 `tau≈0.005` 的 Polyak 增量可能在 BF16
舍入中消失。old/current upstream 都遍历整个 target model；旧 RoboTwin 私有 port 才收窄为：

- `critic_image_encoder`；
- `critic_state_encoder`；
- `q_head`。

首版应重接这项私有修正。完整 target model 和 target checkpoint 仍保留；收窄的只是额外 FP32 EMA accumulator，
不会把 target model 变成“只有 critic”。加载顺序必须是先恢复 target model，再恢复/校验 FP32 shadow。

## 6. checkpoint：到底变没变、适配是否有依据

old `6d0` 到 current `7d07` 的 `fsdp_sac_policy_worker.py` 在 checkpoint 相关语义上没有改版；差异主要是 import、
actor 类位置与 platform API。current generic save/load 能恢复：

- online model、actor/critic optimizer 与 scheduler；
- entropy alpha 及其 optimizer/scheduler；
-完整 target model；
- generic replay 内容。

它仍不知道：

| 算法 | generic 漏掉的私有运行态 | 首版恢复方式 |
|---|---|---|
| RLT | `update_step`、累计 transitions/episodes、warm-up anchor、合同 fingerprint、rank/world-size 完整性 | 旧 manifest sidecar 做 current symbol-level rebase；cycle-local pending 不落盘，按 step-boundary 重算 |
| DSRL | Gaussian/learned phase、`update_step`、pending local new count、critic FP32 shadow；generic replay也没有 ring cursor/RNG | DSRL sidecar + ring 自身 checkpoint |

依据不是推测：旧 RLT 有 focused roundtrip/fail-closed tests，并完成 250→480 真实 resume；旧 DSRL 有 ring 下一批
逐 tensor 续接和 shadow resume tests。首版复用这些合同，只替换 current 的 import/hook/字段名。

另有一个容易漏的 current 接缝：generic SAC 主模型 save 调用没有把 YAML 的
`actor.fsdp_config.save_full_model_weights` 传下去，默认会额外写 full actor weights；旧 RoboTwin DSRL port 已补传，
且配置为 `False`。迁移必须保留这个窄修，不然 checkpoint 体积会无声放大。

## 7. 现在仍有哪些“模糊点”

### 7.1 需要用户做方法决定：只有一项

**RLT Stage 1 parallel 还是 current AR。** 推荐 current AR + exact π0 + frozen/token-only + fresh Stage 1。

### 7.2 不需要用户选择，但实现后必须用 fixture 闭合

1. current Builder 是否精确保持：env 执行动作、RLT canonical replay action、DSRL latent replay action三者不串位。
2. RoboTwin pure truncation 的 `next_obs`、termination bootstrap 与 done 索引是否在 current Builder 下仍逐步对齐。
3. DSRL macro projection 是否继续 first-done 截断、`N=20` 折扣和 leading-bootstrap shape 合同。
4. 4卡下 replay global capacity 如何按 rank 分片；checkpoint 必须锁 actor world size，不能把旧2卡 checkpoint当作
   4卡 strict resume。
5. DSRL critic-only allowlist 在 current 参数名下是否恰好命中三组件；不完整就 fail-fast。

### 7.3 两个可选可靠性增强，不进入首版方法

- 为 current generic RLT replay 增加精确 RNG-state 恢复。旧成功 RLT 本来就没有，首版不做。
- 兼容没有 DSRL sidecar 的旧 checkpoint。推荐 current formal 从 fresh 开始；若以后需要旧权重导入，只提供显式
  legacy opt-in，不能称为 strict resume。

## 8. 推荐锁定与下一实施批次

建议现在锁定：

1. RLT：`7d07` exact π0 + current AR + frozen/token-only + fresh Stage 1 + old AC Stage 2；RTC off。
2. DSRL：`7d07` exact π0 + 旧成功 macro/UTD/ring/shadow/resume 语义；RTC off。
3. 两支独立 worktree；首批代码只做上述 source-locked rebase 与 focused fixtures。

用户确认 RLT AR 后，下一份 packet 再列精确 branch/worktree、修改顺序、resolved config、检查命令与停止条件。
本文件不授权服务器写操作、测试、smoke 或训练。
