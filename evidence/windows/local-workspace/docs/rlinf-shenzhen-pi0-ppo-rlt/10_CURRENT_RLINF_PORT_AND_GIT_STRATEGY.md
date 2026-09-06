# Current RLinf 迁移与 Git 拓扑策略

> 日期：2026-08-22  
> 范围：把 AutoDL 时期基于旧 RLinf 的 GRPO、RLT、DSRL、OGPO 开发，按可追溯增量迁移到深圳当前
> official RLinf；同时确定深圳、AutoDL 与 GitHub 的长期仓库/branch/worktree 关系。  
> 本文只做源码与 Git 规划，不声称任何服务器进程、GPU、内存或网络的当前状态。

## 0. 先给结论

可以迁移，而且正确问题不是“把旧仓库整体升级”，而是：

```text
旧 official base
  + 某算法分支相对该 base 的可审计增量
  -> 在 current official exact SHA 上重新实现同一合同
  -> current compose/import/checkpoint/simulator 重新验证
```

推荐方案是：

1. 继续使用同一个个人 GitHub 仓库
   [`Yutenji-Nyamu/rlinf_fastwam`](https://github.com/Yutenji-Nyamu/rlinf_fastwam)；不另建深圳专用仓库。
2. AutoDL 旧分支保持冻结，绝不 rebase、force-push 或改写；它们是历史 source/result oracle。
3. 深圳在一份本机 canonical clone 上，从 current official exact SHA 为每个算法建立**独立新 branch +
   独立 worktree**。worktree 不能跨机器共享；AutoDL 与深圳只是通过 GitHub 共享 commit。
4. 不 merge 旧分支，不 cherry-pick 整条旧历史。核心框架接缝采用 manual port；只有完全自包含、未被
   upstream 改语义的 commit 才考虑 `cherry-pick -x`。
5. 研究优先级建议 `RLT -> GRPO -> DSRL -> OGPO`；若只按工程风险由低到高，则是
   `GRPO -> RLT -> DSRL -> OGPO`。Fast-WAM × RLinf 的 GRPO/PPO 是另一条移植线，应在 standalone
   Fast-WAM RoboTwin inference 闭环后再做。

## 1. 2026-08-22 远端源码真值

本节来自当日 `git ls-remote`、本地 partial Git object audit 和 exact commit diff；网页说明不替代 SHA。

### 1.1 official 与个人分支

| 身份 | ref | exact HEAD | 角色 |
|---|---|---|---|
| old official/common base | historical commit | `6d0db56bf26f972cd27fa29535f5eb939e80e5bf` | AutoDL 开发共同旧基线 |
| current official | `RLinf/RLinf:main` | `7d07a4212ee6858cc333e1d4fab7a37256d1f839` | 深圳 current source lock |
| personal main | `main` | `8138d6700e3838250c1139289ebfba43d48ff7de` | Fast-WAM GRPO/PPO 集成终态，不是新 base |
| DSRL | `codex/dsrl-pi0-robotwin` | `48a775db09c16c455aeba7b0600c920e7c80d534` | AutoDL DSRL 历史终态 |
| RLT | `codex/rlt-pi0-robotwin` | `2b8199d8ab2e7b110994fd3234bf7007196c3af9` | AutoDL RLT 历史终态 |
| OGPO | `codex/ogpo-pi0-robotwin` | `5d5c84e3ac4efa1713a4139a05ac1b776e634ed3` | AutoDL OGPO 实现终态 |

个人仓当前**没有**专门的 `pi0 + RoboTwin GRPO` 分支，也没有 tracked
`robotwin_adjust_bottle_grpo_openpi_a800_2gpu_{baseline,smoke}.yaml`。历史证据表明这两份配置留在
AutoDL 旧公共树的 untracked 文件中；因此 π0 GRPO 没有可直接 cherry-pick 的个人 commit。

另一方面，个人 `main@8138d670` 确实包含发布过的 **Fast-WAM + RoboTwin + GRPO/PPO** 集成：

- feature commit `768e0243e4dafedea6c92b3f37b652c51efb5a2e`；
- 相对 `6d0db56` 的非文档增量为 26 files、`+6287/-2`；
- 这不是 π0 GRPO 配置分支，不能把两者混称为同一迁移任务。

### 1.2 旧分支真实祖先关系

```text
official 6d0db56
├─ 768e024  Fast-WAM + RoboTwin PPO/GRPO
│  └─ 8138d67  personal main/docs
│     └─ 6817c73  DSRL method port
│        └─ ... -> 48a775d  DSRL closeout
│           └─ 1a923a2  RLT method port
│              └─ fixes/config/eval -> 2b8199d  RLT closeout
└─ 5d5c84e  OGPO+CA port

official 6d0db56 -> many upstream commits -> current 7d07a421
```

这意味着：

- merge DSRL 整分支，会把 Fast-WAM 一起带入；
- merge RLT 整分支，会把 Fast-WAM、DSRL 和大量历史实验文档一起带入；
- 只有 OGPO 是直接从 `6d0db56` 长出的单独 feature branch；即使如此，它仍不能整体 merge 到 current，
  因为 upstream 已改变中央 worker/data/config 接缝。

方法增量的第一权威 commit 是：

| 增量 | exact feature commit | parent |
|---|---|---|
| Fast-WAM RoboTwin GRPO/PPO | `768e0243e4dafedea6c92b3f37b652c51efb5a2e` | `6d0db56bf26f972cd27fa29535f5eb939e80e5bf` |
| DSRL RoboTwin π0 | `6817c73b298ff9df78d371d4b139e4e0fa8ea529` | `8138d6700e3838250c1139289ebfba43d48ff7de` |
| RLT RoboTwin π0 | `1a923a2305b9bd01647fb66509f19caceff1a310` | `48a775db09c16c455aeba7b0600c920e7c80d534` |
| OGPO RoboTwin π0 | `5d5c84e3ac4efa1713a4139a05ac1b776e634ed3` | `6d0db56bf26f972cd27fa29535f5eb939e80e5bf` |

分支 HEAD 还包含后续 config、artifact binding、评估协议和实验文档；迁移时既要看上述 feature commit，
也要按 branch range 检查后续**非文档**修复，不能只看最后一个 HEAD 或只看最初 feature commit。

### 1.3 current official 的直接相关变化

| upstream commit | current 变化 | 迁移含义 |
|---|---|---|
| `d3aff547c06d66e2e792a62a71122abed86e8aee` | data/schema/storage 重构；replay 移到 `rlinf/data/storage/replay/` | DSRL/OGPO replay 不能按旧目录和旧 payload 盲搬 |
| `cb63097e5471c380bb8e51deb7f483df8017849b` | JAX-aligned PyTorch π0 SFT 与 RoboTwin 路线 | RLT Stage 1 可依 current official 精确 π0 loader，而非维持旧 adapter |
| `8587bac453f03951eeaf4a871c050bf1647e7276` | RLT causal autoregressive reconstruction、route/transition 与单测修复 | 旧 reconstruction 不回迁；RoboTwin 接缝应叠在新 RLT 上 |
| `c704688c9fafe7dfd7267011c3a241adcd8cd203` | RLT/OpenPI 迁入 `openpi_rlinf` 路线 | 旧 OpenPI glue 可作行为 oracle，不能定义新 public contract |
| `fbc72dd6eef6c6ecf69b26a67d13325fcf4de74c` | runner 等待 `set_global_step` 完成再同步 | 所有新算法分支必须保留该 correctness fix |
| `9ad44393d15b0e93461d7415591110678ae17ef6` | Embodied FSDP actor 拆分 | DSRL/OGPO actor dispatch、继承与 checkpoint 接缝需重核 |
| `13e5b652349cc0715267a560004af23e2925489c` | RLT TD3 Stage 2 | 是未来能力，不自动替换本轮 AC baseline |

current 的原 `compute_grpo_advantages` 和 GRPO actor loss 主体没有被改写；upstream 只在旁边增加
`grpo_video`、OPD 等新注册项。因此 π0 GRPO 的主要迁移面是 config/placement/runtime，而不是重写
GRPO 数学。

## 2. 迁移风险矩阵

三方 virtual merge 只用于测**文本冲突**：它不证明 import、shape、normalization、route、bootstrap、
checkpoint 或 simulator 语义正确。

| 路线 | source-locked 旧增量 | 对 current 的机械结果 | 应直接保留 | 必须按 current 重做 | 风险/建议 |
|---|---|---|---|---|---|
| π0 GRPO | 无远端 feature commit；AutoDL 有未跟踪 config/历史 resolved 证据 | official GRPO 主函数未变；current 仍无 `adjust_bottle + pi0 GRPO` recipe | GRPO objective、group advantage、历史有效预算仅作候选 | 从 current π0 PPO YAML 建 config；重新定深圳 placement/offload；补 config contract | **低**。manual reconstruction；不 cherry-pick |
| Fast-WAM GRPO/PPO | `768e024`; 26 non-doc files, `+6287/-2` | virtual merge 仅 `rlinf/models/__init__.py` 文本冲突，但 registry、FSDP、rollout 已发生语义变化 | Fast-WAM adapter/core/export 与集中 tests 的方法合同 | model registry、rollout/FSDP、current data/model interfaces、H100 config | **高，独立旁线**。standalone inference 先闭环 |
| RLT | `48a775db..2b8199d8`; 20 non-doc files, `+3400/-24`; 主 feature `1a923a2`，后续 `c22ba19/4ac48d5/3b610cb/46a2d19` | 7 个旧 core 文件三方文本可 clean；整分支唯一冲突是无关旧 `HANDOFF.md` modify/delete | `rlt_mlp_policy`、RoboTwin H/C/14D、full-task reference→student、transition/replay/resume 合同、seed/eval 意图 | causal reconstruction、new route、`openpi_rlinf` encode/decode、new schema/storage、Stage1/2 binding | **中等文本 / 高语义**。保留 upstream RLT，manual port RoboTwin/π0 接缝 |
| DSRL | `8138d670..48a775db`; 8 non-doc files, `+1607/-131`; 主 feature `6817c73` | replay 自动映射到新 path；`fsdp_sac_policy_worker.py` 1 个内容冲突；旧 monitor 在 current 已删除 | compact transition、reward/discount/truncation、target shadow/update-step resume、latent actor/Q 合同与 tests | storage path/schema、SAC worker 接缝、OpenPI ownership、current monitoring | **中高**。手工迁 3 个 framework seam，不能 merge 累积祖先 |
| OGPO | `6d0db56..5d5c84e`; 26 files, `+6121/-57` | `config.py`、`env_worker.py`、`huggingface_worker.py` 3 个内容冲突；其余 clean 不等于语义 clean | `algorithms/ogpo/core.py`、critic/modules、same-chain scorer/sampler、tests | config/dispatch、primitive trace、rollout payload、storage/checkpoint、OpenPI/current actor split | **高**。pure modules 可移，framework seams manual port；formal 前另闭合 replay save RAM 问题 |

## 3. 各算法建议迁移顺序

### 3.1 RLT：当前研究主线，先迁合同而不是旧实现文件

建议的新分支第一批只做以下一条连贯主路径：

1. 以 current official RLT Stage 1/2、causal reconstruction、route/transition 和 tests 为主体；
2. 加 RoboTwin `adjust_bottle` Stage 1 数据/config，以及 canonical 14D、H=50 的 π0 transform；
3. 加 full-task frozen reference → student route；
4. 加显式 executed-action transition/replay、time-limit truncation bootstrap；
5. 加 Stage 1 artifact → Stage 2 binding 与 source-locked resume；
6. 最后才带回 fixed-seed eval 与旧结果 oracle。

不要把旧 `rlinf/algorithms/rlt/{route,transition,...}.py` 整文件覆盖 current；尤其不能覆盖
`8587bac` 的 reconstruction/route 修复。旧 `1a923a2` 应逐 symbol 阅读并登记：

```text
source commit/path/symbol
-> preserved behavioral contract
-> current target path/symbol
-> adaptation reason
-> focused test
```

仍需用户拍板的方法定义：Stage 1 是 current official 的
`rlt_loss + alpha * vla_loss` 联合目标，还是明确复刻旧 frozen/token-only 变体。二者必须做成不同命名，
不能在迁移时无声混合。

### 3.2 GRPO：区分 π0 baseline 与 Fast-WAM integration

π0 GRPO 没有完整 Git feature commit。正确恢复顺序是：

1. 从 AutoDL 旧 live tree 或已归档 resolved YAML 恢复精确历史 config，形成 evidence-only snapshot；
2. 以 current official `robotwin_adjust_bottle_ppo_openpi.yaml` 为结构基准；
3. 只加入 GRPO 必需差异：`adv_type/loss_type/group_size`、critic/value 关闭、rollout group/有效 batch；
4. A800 的 GPU placement、offload、env 数、micro/global batch 不迁为深圳默认；
5. compose 后做一次 algorithm/config diff，证明没有混入 Fast-WAM、DSRL、RLT 或 OGPO。

Fast-WAM GRPO/PPO 则另建分支。只有 standalone Fast-WAM 的 source/model/RoboTwin inference 成功后，
才把 `768e024` 中的 adapter/core/tests 映射到 current registry/FSDP/rollout；不把它当作 π0 GRPO 的前置。

### 3.3 DSRL：保留算法合同，重接 storage/worker/OpenPI

推荐把旧增量拆成：

- 可复用合同：H=50、N=20、32D latent、canonical 14D env action、compact transition、宏观 reward 与
  `gamma^N`、target shadow、update-step/replay resume；
- current 接缝：`rlinf/data/storage/replay/`、current SAC worker、current OpenPI ownership 和 actor split；
- 验收：reward 落在 final chunk 时不会被 `rewards[:,0]` 丢掉；compact replay 不保存三相机全 payload；
  target shadow/load 顺序正确；legacy PPO/GRPO 默认路径不变。

旧 `examples/embodiment/monitor_resources.py` 不迁；深圳已有独立运行证据/资源记录方式，应按 current
observer 重写，而非解决 modify/delete 冲突。

### 3.4 OGPO：最后迁，先纯算法再中央接缝

建议两层提交：

1. 纯模块：OGPO core、critic、EMA/modules、same-chain sampler/scorer 与纯函数/tests；
2. current framework seam：config validator、runner/dispatch、primitive env trace、rollout payload、actor、
   replay/storage/checkpoint。

旧 OGPO formal 已暴露 replay sidecar save 后 actor RSS 阶梯式增加。它是旧运行的已确认风险，不等于
深圳 current 一定同因；但新 OGPO formal 前必须让 checkpoint probe 证明“保存一次/两次后 RSS 可控”，
不能把旧的一次性全 replay clone 原样带回。

## 4. 三种 Git 组织方案

| 方案 | 好处 | 坏处 | 判断 |
|---|---|---|---|
| **同一 GitHub 个人仓 + 每台机器一份 clone + 每算法独立 worktree** | 一套 upstream history；对象共享，省 clone/磁盘；old/new `range-diff` 最方便；PR/branch 集中；算法目录物理隔离 | worktree 只隔离文件，不自动隔离 venv/cache/output；错误选 base 仍会继承别的算法；canonical `.git` 不可随意移动/删除 | **推荐** |
| 新建“深圳 RLinf”仓库 | 权限、issue、release 完全独立；心理边界清晰 | 复制 upstream history；remote/PR/source authority 变多；AutoDL→深圳比较与后续 upstream sync 更麻烦；机器名侵入代码身份 | 当前没有收益足以抵消成本 |
| 每算法一个仓库 | 最强权限/生命周期隔离；适合未来独立开源项目 | 重复对象与依赖；共享修复需多次同步；难证明各算法来自同一 exact base；branch/remote 管理碎片化 | 仅当算法将独立发布、独立团队维护时采用 |

关键概念：**worktree 是同一台机器上一份 `.git` object database 的多个 checkout**。深圳和 AutoDL
不能成为“同一个 worktree”；它们分别有自己的 clone/worktree，通过 GitHub commit 对齐。

## 5. 推荐的具体拓扑

保持深圳现有路径语义：

```text
GitHub
├─ upstream: RLinf/RLinf
└─ personal: Yutenji-Nyamu/rlinf_fastwam

SZ-H100
/data/chenyiteng/projects/rlinf-shenzhen/RLinf/                 # canonical .git
└─ ../worktrees/
   ├─ ppo-pi0-robotwin/                                        # 当前 formal，先不动
   ├─ rlt-pi0-robotwin-<base8>/
   ├─ grpo-pi0-robotwin-<base8>/
   ├─ dsrl-pi0-robotwin-<base8>/
   └─ ogpo-pi0-robotwin-<base8>/

AUTODL-A800
/root/autodl-tmp/...                                            # 历史 clone/worktrees，冻结

WIN-LOCAL
C:/Users/86136/Documents/rl/                                   # 文档、diff、轻量证据
```

remote 不必为了美观重命名；深圳现有 `origin=official` 时，只新增 `personal` 即可。公开 clone/fetch 不需
GitHub 登录。到第一次 push 再由用户选择 repo-scoped deploy key、个人 SSH key或最小权限 token；凭据
不写 remote URL、文档或命令账。

新 branch 用 exact base 缩写区分旧/新，而不是只写易过期的 `latest`：

```text
codex/sz-<base8>-rlt-pi0-robotwin
codex/sz-<base8>-grpo-pi0-robotwin
codex/sz-<base8>-dsrl-pi0-robotwin
codex/sz-<base8>-ogpo-pi0-robotwin
```

若创建时 official 仍是 `7d07a421...`，`<base8>` 就是 `7d07a421`。branch 中的 `sz` 用于和已冻结
AutoDL v1 清楚分代；机器路径、CUDA、代理和输出目录仍不写死进算法代码。未来 idea 从**已经验收的
对应 baseline port commit**开新 branch，不从另一算法分支或运行中的 dirty worktree起步。

每个 worktree 单独保存：

- source-lock manifest；
- implementation ledger；
- resolved configs 与少量测试证据；
- 独立的外置 run directory。

worktree 不自动隔离 Python 环境。若算法不增加依赖，可以复用同一 source-locked base venv，但必须从
当前 worktree 的 cwd/PYTHONPATH 启动；一旦依赖版本不同，就建独立 venv。checkpoint、assets、cache 和
logs 均留在 repo 外，不能由 Git worktree 共享规则隐式决定。

## 6. merge、cherry-pick、patch 与 manual port 怎么选

| 手段 | 本项目用法 | 原因 |
|---|---|---|
| merge old branch | **禁止** | 会引入旧 official 历史和累积的 Fast-WAM/DSRL/RLT 文档/代码；冲突解决也无法证明保留了 current semantics |
| rebase old branch onto current | **禁止用于旧 branch** | 改写历史 SHA，破坏 AutoDL source/result provenance；还会把累积祖先伪装成新实现 |
| cherry-pick whole range | **不采用** | RLT 会连带 DSRL/Fast-WAM；DSRL/OGPO 中央接缝已变化 |
| `cherry-pick -x` 单 commit | **仅限自包含、语义未变的 leaf commit** | 保留来源；但仍须 current tests。配置/seed/tool 也要先确认无旧 path/runtime 假设 |
| raw patch / `git apply` | **只作冲突探针或草稿** | clean apply 只证明文本可落，不证明新 schema/action/checkpoint 语义 |
| manual contract port | **核心推荐** | 可以显式保留 current upstream correctness fixes，并逐 symbol 记录旧行为合同和必要适配 |

建议的新 branch 提交序列保持可审阅：

1. `test/contracts`: 先落不依赖 runtime 的行为合同/fixtures；
2. `port/core`: 纯算法模块；
3. `port/current-seams`: current schema/worker/model 接缝；
4. `config/robotwin-pi0`: source-locked recipe；
5. `test/integration`: compose/import/checkpoint/load/simulator 的集中验证。

不为每个小函数设置流程 gate；上述是 Git review 边界，不是五轮零碎实现。实际开发仍应一次完成一个
连贯主体批次，再做少量高信息量检查。

## 7. 实际启动顺序与验收边界

### 7.1 建议顺序

```text
当前 PPO formal 自然结束/形成明确停止点
  -> 锁定 new-port base SHA 与 PPO runtime oracle
  -> RLT current-main port（当前研究主线）
  -> π0 GRPO current-main 小迁移
  -> DSRL current-main port
  -> OGPO current-main port
  -> standalone Fast-WAM 成功后，再决定 Fast-WAM × RLinf GRPO/PPO port
```

若目标临时改成“最快得到第二个 current RLinf baseline”，GRPO 可以先于 RLT；它不应被设成 RLT 的
额外工程 gate。

### 7.2 每个 port 的最低完成定义

1. exact current official base、old source commit/path/symbol 与新 HEAD 均可追溯；
2. 旧→新 changed/unchanged contract 列表完整，legacy/current PPO 路径默认不变；
3. compose/import 与聚焦数学/shape/schema/checkpoint tests 通过；
4. 真实 source-locked checkpoint/model load；
5. 用户审阅 resolved config、命令、输出、预算、资源与停止条件后，才做真实 smoke；
6. smoke 闭合 rollout→训练→save→必要的 reload/eval；
7. 只有预算与协议另行对齐并获批后，才命名 pilot/formal 或比较算法效果。

## 8. 现在需要讨论的三个决定

1. **优先顺序**：按当前研究主线先做 RLT，还是先用低风险 GRPO 快速得到第二个 current baseline；本文
   推荐不改变既定路线，先 RLT。
2. **RLT Stage 1 定义**：current official joint `rlt_loss + alpha*vla_loss`，还是旧 frozen/token-only
   replication；推荐把前者作为 current-upstream baseline，后者若要保留则另命名 ablation。
3. **Fast-WAM 范围**：当前只完成 standalone official inference，还是之后继续迁 `768e024` 的 RLinf
   GRPO/PPO integration；二者不要和 π0 GRPO 混成一个 branch。

在这三个决定之外，仓库拓扑无需再等待：**同一 personal repo、AutoDL 旧 branch 冻结、深圳 current
base 每算法独立新 branch/worktree** 是当前最稳妥且最容易理解的长期结构。
