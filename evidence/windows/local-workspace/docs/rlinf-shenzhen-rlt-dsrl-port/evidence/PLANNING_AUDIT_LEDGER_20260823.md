# 深圳 current RLinf × RLT / DSRL 规划审计流水账

日期：2026-08-23  
范围：只读历史证据、Git 对象、官方 current 源码与深圳服务器现场；本轮未修改服务器代码、未创建 worktree、
未运行测试/训练。

## 1. 授权与目标

用户要求：恢复旧 AutoDL 成功 RLT/DSRL，实现增量定量；对照深圳较新 RLinf；识别迁移适配和设计问题；建立
独立上下文/规划文档。

本轮按 planning-only 处理。所有动态 Git/worktree 事实以深圳 live probe 为准；旧文档与 Memories 只作线索，
随后均由本地 Git objects 或专题证据复核。

## 2. 工作区规则与专题入口恢复

读取：

```powershell
Get-Content PROJECT_CONTEXT.md
Get-Content HANDOFF.md
Get-Content docs/rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md
Get-Content docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
Get-Content docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md
```

结果：确认深圳 ordinary work 用 `chenyiteng`；新算法各自独立 branch/worktree；本轮没有服务器写/运行授权；
旧 RLT/DSRL 专题为历史 oracle。

## 3. 本地 Git 对象恢复

为避免触碰现有 dirty worktree，建立只读审计用 bare repo：

```text
C:\Users\86136\Documents\rl\.tmp\rlinf_rlt_dsrl_port_audit_20260823.git
```

获取并锁定 refs：

```text
official old base       6d0db56bf26f972cd27fa29535f5eb939e80e5bf
official current pin    7d07a4212ee6858cc333e1d4fab7a37256d1f839
personal DSRL final     48a775db09c16c455aeba7b0600c920e7c80d534
personal RLT final      2b8199d8ab2e7b110994fd3234bf7007196c3af9
```

另建立 detached current source 阅读树：

```text
C:\Users\86136\Documents\rl\.tmp\rlinf_7d07_source_20260823
```

结果：两个临时目录合计约 36 MiB；未修改旧 `.rlt-impl-worktree`、`.dsrl-impl-worktree` 或用户 dirty tree。

## 4. ancestry 与 commit 范围核验

执行类别：

```bash
git merge-base 6d0db56... 7d07a421...
git rev-list --count 6d0db56...7d07a421...
git merge-base 6d0db56... 48a775db...
git merge-base 48a775db... 2b8199d8...
git log --oneline --ancestry-path 6d0db56...7d07a421...
```

结果：

- current 是 old 的严格后继，55 commits；
- DSRL 的 runtime 有效 base 是 `6d0db56...`；
- RLT branch 从 DSRL final `48a775db...` 分叉，所以 RLT 自身 diff 只能用 `48a775..2b8199`。

## 5. 旧 DSRL 精确增量

执行类别：

```bash
git show --numstat --format= 6817c73b...
git diff --numstat 8138d670...48a775db... -- rlinf examples tests
git diff --name-status 6d0db56...7d07a421... -- <DSRL target paths>
git merge-tree 8138d670... 7d07a421... 6817c73b...
```

结果：实现 commit 10 files、`+3277/-130`，其中 docs `+1674`；剔除 docs 后：

```text
runtime 3 files     +1016/-130
configs             +253
tests               +334
runtime+support      +1603/-130
```

文本冲突预测：legacy OpenPI 主块可自动合并；worker import hunk需按新 actor base 适配；旧
`replay_buffer.py` 对 current 是 modify/delete，因为 current 已迁目录。结论是文本冲突低、语义复验中等。

## 6. 旧 RLT 精确增量

精确命令：

```bash
git --git-dir=.tmp/rlinf_rlt_dsrl_port_audit_20260823.git \
  diff --numstat 48a775db09c16c455aeba7b0600c920e7c80d534 \
  2b8199d8ab2e7b110994fd3234bf7007196c3af9 -- rlinf examples tests toolkits
```

结果：20 files、`+3400/-24`：

```text
runtime Python 7 files  +960/-24
configs 6 files         +573
seed JSON               +27
tests 2 files           +597
toolkits 4 files        +1243
```

worker `+752/-14` 中约 693 行属于 strict resume/manifest/contract；不要把它全部称为算法主体。

附加 ancestry 检查：RLT fork 点 `48a775db...` 相对 clean official `6d0db56...`，legacy
`openpi_action_model.py` 与 `fsdp_sac_policy_worker.py` 还继承了旧 DSRL `+607` 行共享热路径改动。这些不计入
RLT LoC，也不应复制进 current RLT；迁移时改用行为测试确认 RLT 对新 SAC/replay/save-load 基类的真实依赖。

## 7. official old→current 聚焦审计

读取/比较的 current 关键路径：

```text
rlinf/algorithms/rlt/{route,rollout,transition}.py
rlinf/models/embodiment/openpi/openpi_action_model.py
rlinf/models/embodiment/openpi_rlinf/**
rlinf/workers/actor/{embodied_fsdp_actor_worker,fsdp_sac_policy_worker,fsdp_rlt_ac_policy_worker,fsdp_rlt_td3_policy_worker}.py
rlinf/data/storage/replay/{buffer,dataset}.py
rlinf/data/schema/embodied_types.py
docs/source-en/rst_source/examples/embodied/{rlt,dsrl}.rst
examples/sft/config/robotwin_sft_openpi_rlinf.yaml
examples/embodiment/config/*rlt* and *dsrl*
```

主要结果：

- DSRL objective/modules/YAML 基本未变；current `openpi_rlinf` 无 DSRL 支持；
- RLT reconstruction、wrapper、route、transition API、data/schema 与 worker base 均有相关变化；
- current RLT schedule 已覆盖旧 schedule 的大部分，不应复制；
- current generic checkpoint 仍未保存 DSRL/RLT 的算法计数和 strict resume sidecar；
- ordinary RoboTwin env 不会自动进入 current ManiSkill simulator RLT route。

## 8. 深圳服务器 live 只读审计

本地命令文件：
[`../../../local_scripts/remote_commands/shenzhen_rlt_dsrl_port_readonly_audit_20260823.sh`](../../../local_scripts/remote_commands/shenzhen_rlt_dsrl_port_readonly_audit_20260823.sh)

通过已验证固定 host key 的 Paramiko 密码通道，以 `chenyiteng` 执行；密码只在当前进程内注入。脚本只运行：

```bash
date --iso-8601=seconds
id
hostname
git -C <canonical> rev-parse HEAD
git -C <canonical> status --short --branch
git -C <canonical> remote -v
git -C <canonical> worktree list --porcelain
git -C <each-worktree> branch --show-current
git -C <each-worktree> rev-parse HEAD
git -C <each-worktree> status --short --branch
wc -l / sha256sum <relevant-current-files>
git branch -a --list '*rlt*' '*dsrl*'
```

现场时间：`2026-08-23T08:38:55+00:00`。结果：

- identity：`chenyiteng` UID1003，组含 `sudo/labdata`，host `admin`；
- canonical `/data/chenyiteng/projects/rlinf-shenzhen/RLinf`：`7d07a421...`、detached、clean；
- GRPO worktree：`554c6dc8...`、clean；π0 DVAC：`800baf80...`、clean；PPO：`7d07a421...`、clean；
- current RLT AC/TD3、DSRL/SAC、new replay/schema 与 `openpi_rlinf` files 均在；
- 服务器只有旧 remote RLT/DSRL refs，尚无 current Shenzhen RLT/DSRL port worktree/branch。

没有执行安装、fetch/pull、branch/worktree 创建、代码写入、测试或进程操作。

## 9. 本轮形成的设计判断

1. 两算法共用 source lock，但不共用实现 branch。
2. RLT 推荐“精确 π0 + current AR reconstruction”，Stage 1 重训；π0.5/TD3 分轨。
3. DSRL 继续 legacy OpenPI；compact transition backend 独立放进 new storage 层。
4. 旧代码逐 symbol 迁移；不整支 cherry-pick，不带回旧 import/schema/platform API。
5. 少量高信息检查：一个集中 fixture 批次 + 一个 fresh/resume 真实 smoke；运行前另给 packet 并取得批准。

## 10. 产物

- 主计划：[`../00_INDEX_AND_MIGRATION_PLAN.md`](../00_INDEX_AND_MIGRATION_PLAN.md)
- RLT Stage 1 决策证据：[`../01_ONE_PAGE_DECISION_GUIDE.md`](../01_ONE_PAGE_DECISION_GUIDE.md)
- 第二轮逐文件复核：[`../02_CURRENT_PORT_FILE_BY_FILE_READINESS_AUDIT.md`](../02_CURRENT_PORT_FILE_BY_FILE_READINESS_AUDIT.md)
- 本流水账：`PLANNING_AUDIT_LEDGER_20260823.md`
- 只读服务器脚本：[`../../../local_scripts/remote_commands/shenzhen_rlt_dsrl_port_readonly_audit_20260823.sh`](../../../local_scripts/remote_commands/shenzhen_rlt_dsrl_port_readonly_audit_20260823.sh)

## 11. 第二轮：RLT reconstruction 与 RTC 来源复核

只读命令类别：

```text
git show -s <official RLT-add / AR-fix / RTC / data-refactor commits>
git show 1a923a23 -- rlinf/algorithms/rlt/* openpi/* fsdp_rlt_ac_policy_worker.py
rg old RLT docs for "论文 autoregressive / RLinf parallel"
rg current OpenPI/eval worker for rtc_enabled / rtc_context / model_actions / use_dsrl
```

锁定结果：

- official `5769c6eb...`（2026-07-06）先加入 parallel RLT；旧私人 `1a923a23...` 只增 RoboTwin port，
  没改 decoder；official `8587bac4...`（2026-08-10）才改成论文一致 AR。
- 旧文档当时已明确记录论文/代码差异并选择继承 official，不是事后补写理由。
- RTC commit=`657faae5...`，generic `rtc_enabled=False`；专用 eval runner 还需 runner/model 双开，训练不用。
- singular `forward_inputs[action/model_action]` 在旧基线已存在；RTC 新增的是 top-level plural
  `model_actions` 与 `rtc_context`。
- current DSRL 分支先于 RTC 分支执行；current RLT Stage 2 也没有完整 RTC guidance 链。首版均 RTC off。

## 12. 第二轮：checkpoint、replay 与 FP32 shadow 因果复核

只读命令类别：

```text
git diff 6d0db56...7d07a421 -- fsdp_sac_policy_worker.py
git show 6817c73b -- replay_buffer.py openpi_action_model.py fsdp_sac_policy_worker.py
git show 1a923a23 -- fsdp_rlt_ac_policy_worker.py
rg current schema/storage/Builder and old focused tests
```

纠正后的结论：

1. generic SAC save/load old→current 语义未换版；current 仍漏算法私有 state。迁移是重挂旧 sidecar。
2. 旧 RLT sidecar不保存 pending budget；完整 outer-step checkpoint 恢复后将 cycle-local counters 置零，
   由 totals/warm-up anchor/update step重算。generic RLT replay RNG不承诺 bitwise续接。
3. data refactor `d3aff547...` 主要是 rename/restructure；DSRL compact ring从未 upstream，应迁到
   `data/storage/replay/dsrl_transition.py`，不改 generic buffer。
4. all-model FP32 shadow早在 official `42530b72...` 存在；critic-only是旧私人 DSRL 修复未 upstream，
   current port需重接而不是重新设计。
5. current SAC online-model save没有透传 `save_full_model_weights`；旧 DSRL已补，current port不能漏。

## 13. 第二轮：逐文件 readiness 审计

RLT 已按 `no-touch / symbol-level port / new-update` 分类。额外确认：

- no-touch：current AR token transformer、RLT MLP、current AC/schedule主体、schema/Builder/runner/env/FSDP；
- symbol-level：legacy OpenPI freeze+14D adapter、route/transition/rollout、RLT worker truncation+sidecar；
- update：两份 RoboTwin config、focused tests、四个 `toolkits/rlt`；旧工具的 parallel decode API在 AR 下必改；
- 旧 RLT fork 点已含 DSRL共享热路径，严禁整分支 cherry-pick。

DSRL 已分类：

- no-touch：official Gaussian/compact encoders、SAC objective、schema/Builder/runner/FSDP；
- new：`data/storage/replay/dsrl_transition.py` 与 export；
- symbol-level：OpenPI phase/preprocess/latent，SAC worker ring/UTD/grad/shadow/sidecar；
- update：深圳 config 与两个旧 focused tests。

Builder 的关键合同已记录：EnvWorker执行 top-level env action；Builder记录 `forward_inputs[action]`。DSRL
故意把后者覆写为32D latent，RLT则保存routed canonical 14D action。实现后用fixture核验，不预先修改schema。

本轮只编辑本地计划、决策说明、逐文件复核和交接路由；未执行服务器写入、worktree、测试或实验。

## 14. 第三轮：实施依据主入口收束

用户要求：下次正式实现前，自顶向下说明每个实现点的来源、current 状态、适配内容与原因；提供一个主文档，
由它链接分文档，不能要求用户先读多份长文。

本轮操作与结果：

1. 完整复核 `PROJECT_CONTEXT.md`、`HANDOFF.md` 和本专题 `00/01/02`；没有刷新动态训练状态，因为本轮仅处理
   source-locked 静态迁移设计，没有将旧快照称为当前服务器现场。
2. 将原长主计划用 `apply_patch` 移为
   `03_DETAILED_SOURCE_AND_MIGRATION_AUDIT.md`，保留全部 commit、LoC、历史差异和迁移审计，未丢弃证据。
3. 新建同路径 `00_INDEX_AND_MIGRATION_PLAN.md` 作为唯一主入口；建立
   `PAPER-RLT / CUR-7D07 / OLD-RLT / OLD-DSRL / SZ-BASE / SCOPE` 六类依据标签和冲突优先级。
4. 在主入口逐点登记共同 `C01-C06`、RLT `R01-R14`、DSRL `D01-D13`；每项都写明依据、current事实、
   适配动作与原因，并反向映射到集中验收。
5. `01_ONE_PAGE_DECISION_GUIDE.md` 改标题为 RLT Stage 1 parallel/AR 决策来源，并指回唯一主入口；
   `02_CURRENT_PORT_FILE_BY_FILE_READINESS_AUDIT.md` 明确降为逐文件工程附录；修正“四个/五个”标题计数。
6. 更新 `HANDOFF.md` 路由停点：当前仍未实施；RLT AR为推荐但尚待显式锁定；DSRL没有新的科学选择。

设计收束：

- RLT 的唯一方法选择是 current AR 或旧 parallel；推荐 current AR，但没有把推荐冒充用户确认。
- DSRL 的算法本体取 current，RoboTwin 私有方法合同取旧成功实现；storage/worker/schema只按current重接。
- 实现开始后新增 `evidence/IMPLEMENTATION_LEDGER.md`，不再建立并列高层计划。
- 本轮没有创建服务器 branch/worktree，没有改算法代码，没有运行测试、smoke或训练。
