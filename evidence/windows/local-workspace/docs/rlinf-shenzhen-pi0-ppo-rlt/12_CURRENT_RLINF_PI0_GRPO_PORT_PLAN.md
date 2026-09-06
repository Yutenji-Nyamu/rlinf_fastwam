# Current RLinf 上的 π0 × RoboTwin GRPO 小迁移计划

> 日期：2026-08-22  
> 状态：config-only迁移、current compose/合同检查、commit与普通push均已完成；真实训练尚未执行。
> 2026-08-22 用户进一步选择**不另跑smoke，直接做与深圳PPO资源/采样密度对齐的formal-100**；当前
> `32×8/B512`是此前的old-global-budget候选，不再是选定formal参数。选定值、batch/filter源码语义、预算和
> 可直接执行命令见第12--13节。  
> 目标 base：official RLinf `7d07a4212ee6858cc333e1d4fab7a37256d1f839`。本文不记录服务器动态状态。

## 0. 核心判断

这次迁移**总体很小**，但更准确地说，不是“把旧 GRPO Python 实现搬到新版”，而是：

```text
current official π0 + RoboTwin PPO recipe
  + 旧 AutoDL 已验证的 GRPO 配置差异
  -> current-base π0 + RoboTwin GRPO recipe
```

旧 AutoDL 的 π0 GRPO baseline 没有个人仓 tracked feature commit；真实成功形态是
`official 6d0db56...` 的既有 GRPO/π0/RoboTwin 代码，加两份留在旧机器工作树中的 untracked YAML。
对 `6d0db56... -> 7d07a421...` 的逐文件复核表明：

- `compute_grpo_advantages` 函数内容未变；
- `compute_grpo_actor_loss_fn` 函数内容未变；
- RoboTwin `robotwin_env.py`、official `robotwin_adjust_bottle_ppo_openpi.yaml`、`model/pi0.yaml`
  和 patch syncer 均逐 blob 相同；
- embodied actor 已从旧的混合 `fsdp_actor_worker.py` 拆到
  `embodied_fsdp_actor_worker.py`，data/schema 也重构了，但 GRPO 的 batch reshape、reward-group filter、
  advantage 分派、`compute_values = (adv_type == "gae")` 和 actor-only loss 主链仍保留；
- current runner 的 global-step `.wait()`、新 schema/trajectory builder 和当前 OpenPI wrapper 应当保留，
旧文件不能覆盖回来。最初计划用一次独立one-step闭合current调用链；用户在看过current-base增量、历史
AutoDL成功依据和深圳PPO现场后，已选择把第一个outer step直接包含在formal-100中，不再为它另建smoke
run。这个决定不改变算法实现，只改变执行批次。

所以首批**预期只有 1 个 tracked 实现文件：一份 current-base YAML；不预期改 Python**。之所以仍不能把
旧 YAML 原样复制后宣告完成，是因为新 runner/worker/schema 已换代，而且旧 YAML 混有 AutoDL 路径、两卡
placement、旧 offload/batch/eval 选择；必须在 current source 上重新 compose，并用一次真实 optimizer-step
闭合当前调用链。可以把它概括为：**代码改动小，兼容验证有必要。**

## 1. 目标与非目标

### 1.1 本批目标

- 精确 π0，不换成 π0.5、OpenVLA-OFT 或 Fast-WAM；
- RoboTwin `adjust_bottle`，沿用 current π0 PPO 已验证的 3-camera、14D、H=C=50 接口；
- sparse task reward、chunk-level reward/logprob、group-relative advantage、actor-only clipped policy loss；
- 从 current official exact SHA 建独立 branch/worktree；
- 一批完成recipe、current compose和少量合同检查；真实调用链由用户选定formal-100的首个outer step闭合，
  不再另建独立smoke。

### 1.2 明确不进入本批

- Fast-WAM × RLinf GRPO/PPO；它有独立模型 adapter、scheduler/replay、registry/FSDP 接缝；
- DVAC telemetry、global-zscore/R-only weighting、control trace；它们是 baseline 之后的方法增量；
- DSRL、RLT、OGPO 的 actor/replay/route/critic 改动；
- 把 AutoDL 的 GPU、路径、venv、proxy、monitor 或 100-step 参数当作深圳默认；
- formal 训练或效果比较。旧 run 只提供行为与预算 oracle，不是新 main 的成功阈值。

三条容易混淆的线严格分开：

| 线 | 旧实现身份 | 是否进入本批 |
|---|---|---|
| π0 GRPO baseline | `6d0db56...` official core + untracked RoboTwin/π0 GRPO YAML | **是**；本次只恢复这一条 |
| Fast-WAM GRPO/PPO | personal feature `768e0243e4dafedea6c92b3f37b652c51efb5a2e`，含 Fast-WAM adapter/core/registry/FSDP/config | 否；独立模型迁移线 |
| π0 GRPO + DVAC | baseline 之上的 `dvac_telemetry.py`、`dvac_train_weighting.py` 及 worker/OpenPI opt-in 改动 | 否；待 baseline 验收后另开方法 branch |

## 2. Source locks 与证据职责

| 来源 | 精确位置/锁 | 本计划使用方式 |
|---|---|---|
| 旧 official code base | RLinf `6d0db56bf26f972cd27fa29535f5eb939e80e5bf` | 旧 π0 GRPO Python 行为真值 |
| 旧 baseline YAML | [`tmp/idea2_train_impl_source/rlinf/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline.yaml`](../../tmp/idea2_train_impl_source/rlinf/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline.yaml) | 只恢复旧 baseline 配置差异；不把相邻 DVAC 源码当 baseline |
| 旧 smoke YAML | [`tmp/idea2_train_impl_source/rlinf/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_a800_2gpu_smoke.yaml`](../../tmp/idea2_train_impl_source/rlinf/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_a800_2gpu_smoke.yaml) | 只核 one-step 预算与历史命名 |
| 旧真实 run | [`audits/20260717-084926-grpo-current/resolved-config.yaml`](../../audits/20260717-084926-grpo-current/resolved-config.yaml)、[`command.txt`](../../audits/20260717-084926-grpo-current/command.txt)、[`analysis.json`](../../audits/20260717-084926-grpo-current/analysis.json) | resolved/命令/100-step 产物与指标证据 |
| 历史叙述 | [`pi0 + ppo_grpo.md`](../../exports/dsrl_pi0_robotwin_formal_v1_work_materials_20260729/historical_source_materials/pi0%20+%20ppo_grpo.md) | 线索与操作背景；不覆盖源码和 resolved 证据 |
| current official | RLinf `7d07a4212ee6858cc333e1d4fab7a37256d1f839` | 新实现唯一 base；本地 Git object 已用于 exact `git show/diff` |
| current 迁移总图 | [`02_OFFICIAL_SOURCE_AND_PORT_MAP.md`](02_OFFICIAL_SOURCE_AND_PORT_MAP.md)、[`10_CURRENT_RLINF_PORT_AND_GIT_STRATEGY.md`](10_CURRENT_RLINF_PORT_AND_GIT_STRATEGY.md) | 上游结构和 Git 共同约束 |

旧 baseline YAML 虽位于 Idea2/DVAC 的提取树中，但文件本身是历史 π0 GRPO baseline；该提取树中相邻的
`dvac_telemetry.py`、`dvac_train_weighting.py`、DVAC YAML 和 worker/model 改动不属于本次 source lock。

实施前仍要在服务器对目标 canonical clone 做一次 `rev-parse/status`，确认 worktree 从精确
`7d07a421...` 创建；本文的本地 object audit 不替代当时的服务器 source truth。

## 3. 旧 → current 逐接口结论

| 接口 | 旧 `6d0db56...` → current `7d07a421...` | 本批动作 |
|---|---|---|
| GRPO advantage | `rlinf/algorithms/advantages.py::compute_grpo_advantages` 内容相同；仍按 `view(-1, group_size)` 做组内 mean/std | 原样使用 current；不迁代码 |
| GRPO actor loss | `rlinf/algorithms/losses.py::compute_grpo_actor_loss_fn` 内容相同；仍调用 clipped PPO actor loss | 原样使用 current；不迁代码 |
| registry | current 只为 `grpo_video`/OPD 增加旁路；普通 `grpo` 预处理与 score 路线未被替换 | 选 `adv_type=grpo`，不选新方法 |
| actor worker | embodied 主体由旧 `fsdp_actor_worker.py` 拆为 current `embodied_fsdp_actor_worker.py`；普通 GRPO 的 reshape/filter/adv/loss 主链保持 | 不复制旧 worker；compose + formal首步验证current worker |
| data/schema | `embodied_io_struct.py` 收敛为 `data/schema/embodied_*` + trajectory builder/storage | baseline 没有自定义 payload，故无需迁 schema；保留 current |
| runner | current 显式等待 actor/rollout `set_global_step(...).wait()` | 保留 current correctness fix |
| RoboTwin env | `rlinf/envs/robotwin/robotwin_env.py` blob 相同；同一 reset state 会按 `group_size` 连续重复 | 不迁代码；检查 `total_num_envs % group_size == 0` |
| π0 model config | `examples/.../model/pi0.yaml` blob 相同；GRPO 默认 value head off | 保留 current model config |
| OpenPI wrapper | current 增加默认关闭的 RTC 与 `model_actions` 输出；普通训练 likelihood/action transform 未被替换 | 不打开 RTC，不回滚wrapper；真实model-load由formal首步确认 |
| PPO recipe | current 与 old official 的 RoboTwin π0 PPO YAML blob 相同 | 用它作新 YAML 的结构母版 |
| FSDP defaults | old/current diff 仅注释/空白，不是运行语义变化 | 使用 current defaults |
| placement | current official RoboTwin/GRPO configs仍使用 `actor, env, rollout: <ranks>` | 只写深圳当次获批 ranks，不复制 `0-1` |

## 4. 可以复用、必须适配、无需迁移

### 4.1 可直接复用的算法/接口合同

- `adv_type=grpo`、`loss_type=actor`、`group_size=8`；
- `normalize_advantages=true`；
- sparse reward 下过滤全成/全败组的历史设定：`filter_rewards=true`、区间 `[0.1, 0.9]`；
- chunk-level reward/logprob、token-mean loss、clip high/low `0.2`、`update_epoch=2`、LR `5.6e-6`；
- `add_value_head=false`、不计算 GAE/value loss；注意 `critic.use_critic_model=false` 在 current PPO 中本来
  就是 false，它指独立 critic worker，真正关闭 π0 value head 的字段是 `actor.model.add_value_head=false`；
- π0 SFT checkpoint、`pi0_aloha_robotwin`、3 images、14D、H=C=50、Flow-SDE 和当前 normalization/action transforms；
- train group 内相同 reset state、eval `group_size=1` 的语义。

其中 group size、filter、clip、LR 是“历史 π0 GRPO baseline 定义 + current official π0 GRPO 仍支持”的
候选方法字段；它们可以作为 baseline 首版，但仍应在 resolved packet 中显式冻结。不能把历史 100-step
效果反推成这些参数在深圳最优。

### 4.2 必须按 current/深圳重做的工程字段

- 从 current PPO YAML 重建文件，不复制旧 YAML 的 defaults/path 注释和 AutoDL 绝对路径；
- canonical worktree、RoboTwin compatibility assets、π0 checkpoint 和结果目录；
- 物理 GPU → logical rank placement；
- `total_num_envs`、`rollout_epoch`、micro/global batch、offload；
- fixed eval 的分母、seed、并发、保存/评估间隔；旧 baseline `val_check_interval=-1` 且 eval 非 fixed，
  不能拿来定义深圳 baseline 的 held-out 指标；
- source-lock manifest、resolved config、完整命令和外置输出路径。

### 4.3 无需迁移

- `advantages.py`、`losses.py` 的普通 GRPO 函数；
- `robotwin_env.py`、`model/pi0.yaml`、patch syncer；
- 旧 actor worker、旧 embodied I/O/schema 文件；
- 旧 monitor、Ray 清理脚本、AutoDL 网络/venv/路径；
- DVAC 的 telemetry/weighting/control-trace；
- Fast-WAM 的 model registry、adapter、video/action scheduler/replay、FSDP 特例和 export；
- 旧 DSRL/RLT/OGPO 分支的任何祖先增量。

## 5. 预算事实与首轮候选，不把旧参数冒充深圳默认

旧 AutoDL 成功 run 的精确 resolved 事实是：

```text
2 × A800
16 train env × 16 rollout epochs = 256 trajectories / outer step
group size 8 = 32 groups / outer step
H=C=50, max episode 200 -> 每条最多 4 个 chunk records
最多 1024 records / outer step
global batch 512, micro 32, update epoch 2
-> 最多 2 global batches/epoch × 2 epochs = 4 distributed optimizer steps
eval disabled; save every 10; 100 outer steps
```

该 run 自然完成 100/100，step 100 train-rollout success 为 `0.984375`；它没有 held-out fixed eval，不能
当作 98.4% 泛化成功率。该数字及两卡资源只证明旧协议能运行，不自动决定深圳配置。

历史AutoDL真正提供的缩放规律不是“GRPO必须固定B512”，而是**同机PPO和GRPO保持同一全局采样、batch和
update预算**：

| AutoDL 2卡 | PPO | GRPO |
|---|---:|---:|
| train env × rollout epoch | `32×8` | `16×16` |
| trajectories / max chunk records | `256 / 1024` | `256 / 1024` |
| global / micro batch | `512 / 32` | `512 / 32` |
| update epoch / optimizer calls | `2 / 4` | `2 / 4` |
| per-rank trajectories / max records | `128 / 512` | `128 / 512` |
| actor microsteps / rank / outer step | `32` | `32` |

两者只用不同的并发env/顺序wave来满足G8分组，训练数据量和actor工作量保持一致。这直接支持在深圳也先
以**深圳PPO本身**为资源/采样母版，而不是把旧机器的B512绝对值搬过来。

当前选定的PPO-matched formal候选为：

| 项 | 选定值 | 来源/含义 |
|---|---:|---|
| actor/env/rollout ranks | physical `4--7` | 与已跑深圳PPO相同；启动前等PPO退出并确认释放 |
| train env / rollout epoch | `128 / 4` | `512 trajectories/outer step`；每rank 32 env、128 trajectories |
| group size | `8` | 16 groups/wave、64 groups/outer step；128可整除8 |
| trajectories / max chunk records | `512 / 2048` | 与深圳PPO相同；H=C50、episode200 |
| micro/global batch | `32 / 2048` | 与深圳PPO相同；每rank local batch 512 |
| update epoch | `2` | 每epoch一个global batch，精确2次optimizer-step调用/outer step |
| eval | fixed64、group1、每10步 | 与深圳PPO同口径 |

`global_batch_size`三个候选的含义必须分清：

| GBS | 每epoch global batches | optimizer calls/outer step | 对齐对象 |
|---:|---:|---:|---|
| **2048** | 1 | 2 | **深圳PPO；本轮选定** |
| 1024 | 2 | 4 | 保留AutoDL每outer-step的4次update，但不再与深圳PPO一致 |
| 512 | 4 | 8 | 只保留旧绝对batch，样本翻倍后update也翻倍；不适合作为PPO-matched baseline |

因此“尽量贴近深圳PPO”的唯一自然答案是`2048`。此前`32×8/B512`仍可用于未来做旧global-budget对照，
但不再作为本轮formal。128个train simulator意味着主存形态也会更像深圳PPO；这正是资源对齐的一部分，
不能再引用`32 env`去预估低主存。

## 6. 预期改动文件

### 6.1 首批 tracked diff

| 文件 | 预计改动 | 状态 |
|---|---|---|
| `examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml` | 从 current official PPO recipe 派生，只加入 GRPO 方法差异和获批深圳 overlay | **唯一预期实现文件** |

首批不预建第二份runtime YAML：formal步数、save/eval interval和输出路径由一份source-locked config加
明确CLI override形成resolved packet，避免baseline/runtime两份长期漂移。

### 6.2 仅在真实错误出现时才考虑的条件目标

- `rlinf/workers/actor/embodied_fsdp_actor_worker.py`；
- `rlinf/workers/env/env_worker.py`；
- `rlinf/models/embodiment/openpi/openpi_action_model.py`。

当前没有直接证据要求修改这三个文件，故不提前加 compatibility shim、fallback 或新算法代码。若 compose
或formal首步暴露确定问题，先保留单一失败证据，再只改实际断裂的current symbol。

服务器外置的 resolved YAML、命令、日志、DCP 和资源 CSV 属于运行证据，不进入算法 diff；本地专题账本
继续由 [`evidence/00_SERVER_OPERATION_LEDGER_INDEX.md`](evidence/00_SERVER_OPERATION_LEDGER_INDEX.md) 路由。

## 7. 推荐 branch/worktree

不从运行中的 PPO branch 或任一 AutoDL 方法分支派生；直接从 exact official base 建立：

```text
branch:   codex/sz-7d07a421-grpo-pi0-robotwin
worktree: /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421/
base:     7d07a4212ee6858cc333e1d4fab7a37256d1f839
```

拟执行命令（**仅记录，尚未执行**）：

```bash
git -C /data/chenyiteng/projects/rlinf-shenzhen/RLinf \
  worktree add \
  -b codex/sz-7d07a421-grpo-pi0-robotwin \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421 \
  7d07a4212ee6858cc333e1d4fab7a37256d1f839
```

它与 PPO worktree 共享 Git object database，但 checkout、branch 和 dirty tree 相互独立。venv、cache、assets
和 output 不由 worktree 自动隔离：本批若不新增依赖，可复用同一 source-locked RLinf venv；启动时必须从
GRPO worktree cwd/PYTHONPATH 进入，并把 run 放在 repo 外独立目录。

公开 clone/fetch 和本地 branch/worktree 创建不需要 GitHub 登录。用户现已选择repo-scoped deploy key，
专用keypair已在服务器生成；待用户在个人仓网页登记公钥并启用写权限后，再锁GitHub host key、做
`git ls-remote`、添加`personal` remote和创建本worktree。完整流水见
[`evidence/10_GITHUB_CONNECTION_LEDGER_20260822.md`](evidence/10_GITHUB_CONNECTION_LEDGER_20260822.md)。

## 8. 一批连贯实现与少量高信息量检查

实施批次保持短：

1. live 确认 canonical clone exact base/clean，创建上述 branch/worktree；
2. 从 current PPO YAML 派生唯一 GRPO YAML；不复制旧 worker/source；
3. 做一次 Hydra `--cfg job --resolve`，保存 exact resolved config；
4. 对 resolved config 做一张窄 diff 表：只允许 GRPO 方法字段、获批资源/path/eval 字段变化；原 PPO YAML
   必须仍可 compose 且未修改；
5. 按用户最新选择，不另建one-step smoke；把PPO-matched三项资源值写入tracked YAML并做一次CPU-only
   `--cfg job --resolve`，确认formal输出目录尚不存在；
6. PPO释放physical 4--7和主存后直接启动formal-100。第一个outer step就是同一正式run的真实调用链
   验证；若出现确定性错误，保留该formal输出中的最小证据后做窄修复，不另造多级gate。

如果上述全通，就完成“current RLinf π0 GRPO baseline 工程迁移”；此后 DVAC/其他 idea 必须从这个已验收
baseline commit 另开 branch，不把方法增量塞回 baseline。

## 9. 实施前需要用户决定/授权

下列决定已由用户完成：只做π0 sparse-reward GRPO baseline；继续使用既有独立branch/worktree；采用
physical 4--7、`128 env×4 epochs`、G8、B2048/mb32/update2、fixed64、100步；不另跑smoke。

剩余只是执行前现场条件：停止并释放本轮owned PPO/Ray；确认4--7空闲、主存已恢复、目标formal输出不存在、
worktree exact head/clean；完成三行tracked配置更新、CPU-only compose和普通push。Fast-WAM与DVAC仍是独立
GPU/代码线，不混入GRPO baseline。

## 10. 上下文入口

- 专题总入口：[`00_INDEX_AND_IMPLEMENTATION_PLAN.md`](00_INDEX_AND_IMPLEMENTATION_PLAN.md)
- official 调用链/旧→新 port map：[`02_OFFICIAL_SOURCE_AND_PORT_MAP.md`](02_OFFICIAL_SOURCE_AND_PORT_MAP.md)
- current RLinf/Git 总策略：[`10_CURRENT_RLINF_PORT_AND_GIT_STRATEGY.md`](10_CURRENT_RLINF_PORT_AND_GIT_STRATEGY.md)
- 当前 PPO 参数与资源解释：[`11_FORMAL_CONFIG_AND_MEMORY_ANALYSIS.md`](11_FORMAL_CONFIG_AND_MEMORY_ANALYSIS.md)

## 11. 2026-08-22 实施结果

- branch：`codex/sz-7d07a421-grpo-pi0-robotwin`；base精确为`7d07a421...`。
- commit：`554c6dc8d586162d9444c01fa88308ed4f5203d0`；已普通push到
  `Yutenji-Nyamu/rlinf_fastwam`同名branch，remote/upstream SHA一致，worktree clean。
- 实际diff与预判一致：只新增
  `examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi.yaml`，176行；没有迁移旧Python
  worker/schema/model代码。
- current Hydra compose、预算断言和实际`compute_grpo_advantages` CPU小张量调用均通过。首次实现时冻结的
  `32 train env×8 rollout epochs`、G8、global/micro=`512/32`、update2、fixed64，是old-global-budget
  历史候选；2026-08-22后续决策已由第12--13节的PPO-matched formal取代。
- 旧one-outer-step packet已生成但从未启动；保留为历史证据，不再作为当前启动入口。后续源码深读还纠正了
  其中“filter后optimizer-step数数据相关”的表述：current worker只修改`loss_mask`，不删除batch，因此
  optimizer-step调用数由records/GBS/update2确定；filter只改变有效梯度贡献。

逐命令、配置hash、push验收和完整待审启动包见
[`evidence/12_GRPO_CURRENT_IMPLEMENTATION_LEDGER_20260822.md`](evidence/12_GRPO_CURRENT_IMPLEMENTATION_LEDGER_20260822.md)。
- Fast-WAM 独立专题：[`../fastwam-robotwin-rlinf-grpo/00_INDEX.md`](../fastwam-robotwin-rlinf-grpo/00_INDEX.md)
- DVAC 独立专题：[`../rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md`](../rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md)

## 12. 2026-08-22 PPO-matched formal-100 决策

### 12.1 readiness结论

- **实现仍是干净的current-base增量**：`554c6dc8...`的parent精确为official `7d07a421...`，diff只有
  新增一份176行YAML；branch、upstream与personal remote SHA一致，worktree clean。
- **算法Python不是我们的移植代码**：旧`6d0db56...`与current `7d07a421...`中的普通GRPO advantage和
  actor loss语义相同；current worker已经包含G8 reshape、reward mask、actor-only loss和仅GAE计算value。
- **没有发现要求先做独立smoke的源码或配置阻塞项**。用户已明确选择直接formal-100；因此第一个真实
  outer step留在同一个formal run中。若current运行链出现问题，它会在独立formal输出内留下完整日志，
  再按一个实际原因做窄修复即可。
- 这不是声称current GRPO已经跑过：真实Ray/RoboTwin/FSDP路径仍要由formal的首步首次验证；只是没有再为
  同一参数另建一次一次性run。

### 12.2 四组参数的精确对照

| 预算 | AutoDL PPO 2卡 | AutoDL GRPO 2卡 | 深圳 PPO 4卡 | 深圳 GRPO 4卡（选定） |
|---|---:|---:|---:|---:|
| train env / per-rank | `32 / 16` | `16 / 8` | `128 / 32` | `128 / 32` |
| rollout epochs | 8 | 16 | 4 | 4 |
| trajectories / step | 256 | 256 | 512 | 512 |
| trajectories / rank | 128 | 128 | 128 | 128 |
| max primitive slots / step | 51,200 | 51,200 | 102,400 | 102,400 |
| max chunk records / step | 1,024 | 1,024 | 2,048 | 2,048 |
| GRPO groups / step | — | 32 | — | 64 |
| global / micro batch | `512 / 32` | `512 / 32` | `2048 / 32` | `2048 / 32` |
| update epochs | 2 | 2 | 2 | 2 |
| optimizer-step calls / outer step | 4 | 4 | 2 | 2 |
| grad accumulation / call / rank | 8 | 8 | 16 | 16 |
| actor microsteps / rank / outer step | 32 | 32 | 32 | 32 |
| eval | 关闭 | 关闭 | fixed64/10步 | fixed64/10步 |

历史AutoDL中PPO与GRPO已经证明：方法切换时可以保留全局采样、batch、update2和每rank actor工作量，只调整
env/wave来满足G8。深圳选定配置应用同一原则：GRPO除G8方法合同外，资源轴与已经成功运行的深圳PPO一致。

### 12.3 可直接执行的formal resolved草案

```text
source/head       554c6dc8d586162d9444c01fa88308ed4f5203d0
physical GPUs     4,5,6,7
fresh model       task-matched pi0 SFT @92684e50
train             128 env × 4 epochs × max200, G8, H=C50
actor             global2048 / micro32 / update2, lr5.6e-6, clip_grad1
eval              fixed64, group1, every10
runner             max_steps100, save10, no resume
method             grpo + actor-only + filter[0.1,0.9]
output             /data/chenyiteng/results/rlinf-shenzhen/grpo/runs/
                   grpo-formal100-current-4gpu128train64eval-ppo-matched-v1
```

在PPO exact owned Ray退出、physical 4--7空闲、主存恢复、目标输出不存在后，current head可通过以下显式
overlay直接得到上述resolved语义；没有Python改动：

```bash
source /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/activate
unset CUDA_VISIBLE_DEVICES
export REPO_PATH=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/grpo-pi0-robotwin-7d07a421
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export ROBOTWIN_PATH=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$REPO_PATH:$ROBOTWIN_PATH${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1

/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python \
  "$REPO_PATH/examples/embodiment/train_embodied_agent.py" \
  --config-path "$REPO_PATH/examples/embodiment/config" \
  --config-name robotwin_adjust_bottle_grpo_openpi \
  'cluster.component_placement={actor\,\ env\,\ rollout:4-7}' \
  'runner.logger.log_path=/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/grpo-formal100-current-4gpu128train64eval-ppo-matched-v1' \
  'runner.max_steps=100' \
  'runner.val_check_interval=10' \
  'runner.save_interval=10' \
  'runner.resume_dir=null' \
  'env.train.total_num_envs=128' \
  'env.train.rollout_epoch=4' \
  'actor.global_batch_size=2048' \
  'env.train.assets_path=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support' \
  'env.eval.assets_path=/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support' \
  'actor.model.model_path=/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50'
```

完整100步预算为：51,200 train trajectories、最多10,240,000 primitive action slots、6,400 GRPO groups、
最多204,800 chunk records、200次distributed optimizer-step调用、409,600 actor record presentations、
640 fixed eval episodes和10份checkpoint。

深圳PPO同为512 trajectories/step，实测普通step约26分钟且rollout占绝大部分；因此本配置应按
**约43--46小时、172--184 H100 GPU-hours**排期，而不是此前`32×8`候选的20--24小时。旧GRPO的9.7-GiB
DCP不包含current深圳保存布局的全部口径；深圳PPO每份checkpoint实测17.21 GiB，其中7.514 GiB为
`full_weights.pt`。GRPO使用同一current保存链，启动前按17--18 GiB/份、10份约172 GiB估算，并为
日志/TensorBoard/video把本run预留提高到**至少220 GiB**。128个train env与深圳PPO相同，主存也应按PPO已观察到的
EnvWorker主导增长排期；fresh process会释放旧PPO占用，但不能预期GRPO只用0.8--1.1 TiB。显存预计不高于
同配置PPO的现场约69 GiB/卡，value head关闭可能略低；这些是资源排期，不是经验硬gate。

## 13. reward filter 与 batching 源码语义纠正

2026-08-22进一步只读源码核查确认：

1. `embodied_fsdp_actor_worker.py:236--282`的reward filter只把全成/全败group对应位置写成false
   `loss_mask`；**没有删除trajectory或chunk record，也没有缩小tensor batch**。
2. 同文件`507--564`先把固定rollout展开，再按`global_batch_size/world_size`切batch；每个batch都会执行
   `optimizer_step()`。因此`128×4/H=C50/B2048/update2`是每rank 512 records、每epoch恰好一个batch，
   **每outer step精确2次optimizer-step调用**。
3. `compute_grpo_advantages`先在每个G8内计算mean/std，再乘`loss_mask`。PPO actor loss的all-false mask
   路径返回masked zero，`loss_mask_count`也有`or 1`保护。若某一步64个group全部同质，代码仍会执行两次
   optimizer-step调用，但有效梯度/学习信号可以为零；准确说法是“零有效学习”，不是“零次update”。
4. embodied loss因存在`max_episode_steps/loss_mask_sum`会走`masked_mean_ratio`：在完整2048-record tensor上
   对masked贡献置零后取mean，并不会只在有效group内重新归一化。因此有效group越少，梯度贡献也会自然
   缩小；这是filter的真实优化影响，不是动态batch缩小。
5. 所以filter不会让B2048在运行中变成B1024/B512，也不会让optimizer-step数数据相关。它只改变2048条
   record中哪些位置参与loss。此前文档中的“filter后updates可少于上限”已由本节纠正。

该结论也解释了为何PPO-matched应选B2048：B1024/B512不会补偿filter，只会把同一固定rollout拆成4/8次
optimizer-step调用，改变优化协议。
