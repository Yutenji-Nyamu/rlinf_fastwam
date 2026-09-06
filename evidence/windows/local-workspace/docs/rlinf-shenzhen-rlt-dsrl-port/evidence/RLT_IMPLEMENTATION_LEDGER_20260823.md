# 深圳 current RLinf × RoboTwin π0 RLT 实施流水账

开始时间：2026-08-23  
机器：`SZ-H100`  
日常账号：`chenyiteng`  
状态：已获最小 current-AR port、CPU 前置检查、commit 与普通 push 授权；真实 smoke 禁止启动。

## 0. 本轮冻结边界

- base：official RLinf `7d07a4212ee6858cc333e1d4fab7a37256d1f839`；先核 clean，再建独立 branch/worktree。
- 方法：exact π0、current causal AR、frozen VLA/token-only、AC Stage 2、RTC off。
- 修改：只做主计划 `C01-C06 / R01-R14` 有直接依据的 symbol-level 增量；不整文件覆盖 current。
- 检查：仅 CPU 级 AST/compile、目标 Ruff、小 fixture、Hydra compose、official AR 单测。
- 禁止：加载真实 π0/RoboTwin、启动 Ray、占用任何 GPU、启动真实 smoke、安装依赖、干预 GPU4-7 GRPO。

唯一依据入口：[`../00_INDEX_AND_MIGRATION_PLAN.md`](../00_INDEX_AND_MIGRATION_PLAN.md)。

## 1. 操作记录

### RLT-000：只读准备与现场停点

本地只读：

```text
git show/diff 7d07a421... 48a775db... 2b8199d8...
Get-Content 专题 00/01/02 与旧 RLT 主计划/实施记录
```

结果：

- current/old commit 对象均在本地只读镜像中；旧 RLT 自身 runtime 增量是 7 个 Python 文件，不能整支
  cherry-pick，因为旧 fork point 已含 DSRL 对共享热路径的修改。
- current `route.py` 新增了 expert takeover 后 `ref_chunk` 修正；current `transition.py` 已使用
  Builder/`PolicyOutput`；current OpenPI 已有 RTC `model_actions`；这些必须保留。
- 目标 runtime 最小集锁定为 OpenPI 2 文件、RLT algorithm 4 文件、RLT AC worker 1 文件；current
  `rlt_token_transformer.py`、`rlt_mlp_policy.py`、Builder/schema/env/runner/FSDP 均 no-touch。
- root 现场刷新确认：GRPO 正在 GPU4-7 正常运行到 Step52；GPU0-3 空闲，但 host available 仅约
  143 GiB、cgroup约1.914 TiB且swap已满。因此本轮任何真实模型、仿真、Ray或GPU检查都禁止执行。

### RLT-010：从 clean current base 创建独立 worktree

执行：固定 host-key 的 Paramiko 密码通道，普通账号 `chenyiteng`；密码只进入当前交互进程。远程脚本
`tmp/sz_rlt_create_worktree_20260823.sh` 先检查 canonical exact HEAD/clean、目标 branch/path 均不存在，再执行：

```bash
git -C /data/chenyiteng/projects/rlinf-shenzhen/RLinf worktree add \
  -b codex/sz-rlt-pi0-robotwin-ar \
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421 \
  7d07a4212ee6858cc333e1d4fab7a37256d1f839
```

结果：

```text
RLT_WORKTREE_CREATED
HEAD=7d07a4212ee6858cc333e1d4fab7a37256d1f839
BRANCH=codex/sz-rlt-pi0-robotwin-ar
PATH=/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-pi0-robotwin-ar-7d07a421
```

创建后 worktree clean；canonical、GRPO worktree和运行进程均未修改。

### RLT-020：迁移六个 current-safe core/adapter 增量

先从本地只读镜像生成旧 RLT 自身 `48a775db... -> 2b8199d8...` 的六文件 path-limited diff，排除
包含大段旧 worker 恢复验证器的第七文件；随后在 RLT worktree 执行：

```bash
git apply --3way /tmp/rlt_old_six_core.patch
```

目标：

```text
rlinf/algorithms/rlt/__init__.py
rlinf/algorithms/rlt/rollout.py
rlinf/algorithms/rlt/route.py
rlinf/algorithms/rlt/transition.py
rlinf/models/embodiment/openpi/__init__.py
rlinf/models/embodiment/openpi/openpi_action_model.py
```

结果：3-way clean；增量约 `+208/-10`。逐 symbol 复核确认：

- `FullTaskRLTRoute`、显式 `rlt_transition_replay` capability、RoboTwin canonical action adapter、
  token-only freeze 与 decode-once 路径已存在；
- current RTC `model_actions`、默认 `rtc_enabled=False` 和 current route 的 expert-takeover `ref_chunk`
  修正仍在；
- current causal AR `rlt_token_transformer.py` 未修改，旧 parallel decoder 未迁移。

### RLT-030：最小 worker 适配

对 `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` 做 symbol-level 增量：

1. compact transition 只保留 action/reward/done/termination/truncation/intervention 与
   `curr_obs/next_obs`，不保留旧 logprob/value/version 或 bulky forward payload；
2. pure time-limit truncation 在 critic target 中按 termination bootstrap，并将真实 final obs 写入
   replay next obs；真实 termination 仍不 bootstrap；
3. RLT 私有 sidecar 保存/恢复 `update_step`、本 rank lifetime totals、global warm-up anchors、
   schema/rank/world-size 与完整 contract SHA256；completion manifest 锁 rank set 与跨 rank schedule
   摘要；恢复后 cycle-local counters 和 pending budget 归零重算；
4. contract 为空或包含 `UNRESOLVED_*` 顶层字段时 fail closed。

第一次把人工摘录片段写成 patch 时缺少统一 diff range，`git apply --check` 返回 malformed/garbage；
未修改 worktree。修复方法是从 exact pre-edit 文件重新生成标准 unified patch，先 `git apply --check`
再 apply；通过后未保留失败版本到仓库。

有意没有迁移旧 port 的冗长 per-key/path/global-step/file-SHA 验证树。current 版本保留算法恢复所需的
source-of-truth、same-world-size、contract fingerprint、completion/rank-set 和跨 rank schedule 一致性；
不承诺 replay RNG bitwise 连续，这与旧 RLT 已声明的边界一致。

### RLT-040：两份方法 base config 与集中 fixture

新增：

```text
examples/sft/config/robotwin_rlt_stage1_sft_openpi_current_ar.yaml
examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_current.yaml
tests/unit_tests/test_robotwin_rlt_current_port.py
```

两份 YAML 只冻结算法/模型/任务语义：exact π0、current AR、token-only、RTC off、H=50、C=10、
D=14、z=2048；Stage 2 为 full-task、compact transition、pure truncation bootstrap、initial sync。
placement 使用 `all`，Stage 2 `max_steps=0`，不把 H100 的 2/4 卡拓扑、env 并发或正式预算锁入
base；AutoDL 已成功的 batch/env 数仅作为可覆盖 candidate。

深圳精确模型和 norm 默认路径：

```text
/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
.../physical-intelligence/robotwin/norm_stats.json
```

RoboTwin compatibility/assets 默认：

```text
/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
```

Stage 1 数据使用必填 `ROBOTWIN_RLT_CLEAN50_PATH`。窄层级只读盘点
`/data/chenyiteng/datasets` 与 `/data/chenyiteng/models` 未发现 OpenPI/LeRobot processed clean-50；
现有 ACT processed clean50 不符合 Stage 1 数据合同，因此没有伪装成默认值。真实 smoke 前需取得或转换
official `RLinf/RoboTwin-adjust_bottle-official-demo_clean50-Pi0_processed-data`。

### RLT-050：CPU focused checks

环境：既有 `/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin`；未安装依赖、未加载模型/仿真/Ray。

执行集合：

```text
git diff --check + git diff --cached --check
python -m py_compile（7 runtime + 1 fixture）
ruff check（同一目标集）
Hydra compose Stage1/Stage2 base，并 resolve 全配置
pytest -q tests/unit_tests/test_robotwin_rlt_current_port.py \
  tests/unit_tests/test_rlt_token_transformer.py
```

结果：

```text
diff/compile/Ruff: PASS
Stage1 compose: openpi, H50/C10/D14, current AR, train_vla=False, RTC=False
Stage2 compose: full_task, compact=True, bootstrap_on_truncation=True,
                RTC=False, initial_sync=True, max_steps=0
pytest: 15 passed in 8.88s
```

16 条 warning 均来自现有 torch JIT deprecation；无 test failure/error。official AR 四个小测试覆盖 causal
future mask、padding mask、shape 与 encoder/decoder gradient，current AR 文件本身未修改。

### RLT-060：最终 staged patch

最终 staged diff：

```text
10 files changed, 1229 insertions(+), 24 deletions(-)
```

完整无生成物 patch：

[`patches/rlt_current_port.patch`](patches/rlt_current_port.patch)（54,052 bytes）。

截至本条：未启动真实 π0、RoboTwin、Ray 或 GPU smoke；未占用 GPU0-3；未干预 GPU4-7 GRPO。

### RLT-070：commit 与普通 push

最终再次执行 `git diff --cached --check`，确认 exact base 与 branch 后：

```bash
git commit -m "feat(rlt): port RoboTwin pi0 RLT to current RLinf"
git push -u personal codex/sz-rlt-pi0-robotwin-ar
```

结果：

```text
commit=bdd875283b3f3516c439e5c79c902cf5c2da58b6
branch=codex/sz-rlt-pi0-robotwin-ar
remote=personal/codex/sz-rlt-pi0-robotwin-ar
push=success (ordinary new-branch push; no force)
worktree=clean
```

GitHub 比较入口：

```text
https://github.com/Yutenji-Nyamu/rlinf_fastwam/pull/new/codex/sz-rlt-pi0-robotwin-ar
```

### RLT-080：canonical clean-50 与 current-AR Stage 1 smoke

- canonical输入按official pin与official两段converter重建为50 episodes、7,188 frames、50fps、14D；路径
  `/data/chenyiteng/datasets/robotwin2/canonical/pi0-aloha-clean50-v1`。
- 第一次launcher仅在Hydra precompose暴露placement key应为`actor,env,rollout`；模型/Ray未启动。三个
  launcher统一使用转义后的精确key，失败attempt可恢复保留。
- physical GPU4--5、2 ranks、MB16/GB32、current causal AR、frozen exact π0，真实完成2/2 steps并保存
  `global_step_2`；loss与RLT loss约4.39、VLA loss0。full weights为9,556,454,857 bytes，manifest锁定
  source/data/norm/config SHA。
- GPU峰24,105 MiB/card、cgroup峰46.97 GiB、host available最低1,953.20 GiB；exit0、OOM0。

### RLT-090：Stage 2 fresh/resume 与 strict state

- physical GPU4--5、2 ranks、4 train+4 fixed eval、20 primitive/C10、GB512/MB128。
- fresh：8 global transitions、8 critic/4 actor updates，`update_step=8`，step60.067s，exit0。
- fresh-process resume：新增8 transitions、20 critic/10 actor updates，累计16、`update_step=28`，
  step62.964s，exit0。
- sidecar逐rank通过：fresh local4/rank、resume local8/rank，world-size、runner step、update step与manifest
  SHA均保持；两个checkpoint各约51 MiB。
- 显存峰fresh 16,937/17,233 MiB，resume 17,230/17,690 MiB；cgroup峰均约42 GiB，无OOM/worker crash。
- 完整命令、resolved YAML、日志、CSV和postflight已收进共同轻量证据包；formal尚未启动。
