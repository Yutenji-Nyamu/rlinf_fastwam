# π0 × RoboTwin × DSRL 实施与调试账本

本账本是操作级记录，不代替唯一设计计划。不得记录密码、token 或其他凭据。时间均为 Asia/Shanghai。

## 授权与停止点

- 2026-07-28：用户授权直接在服务器实现 RoboTwin DSRL、运行 smoke 前的 compose/import/compile/集中基础测试、修复本任务问题，并在完成后提交和推送。
- fresh/resume smoke 与正式训练仍是强制停止点：执行前必须展示完整 resolved config、精确命令、输出目录、预计资源和停止条件，等待用户批准。
- 不在本授权内：删除数据、停止无关进程、重装依赖、下载大模型、覆盖既有 checkpoint。

## OP-001：恢复单一事实源与精简根目录

- 时间：2026-07-28 14:00–14:20
- 操作：
  1. 完整读取根目录 `AGENTS.md`、`PROJECT_CONTEXT.md`、`HANDOFF.md`。
  2. 完整读取当前唯一计划 `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md`。
  3. 将原 `AGENTS.md`、`PROJECT_CONTEXT.md`、`HANDOFF.md` 无损移动为 `docs/project-history/*_SNAPSHOT_20260728.md` / `HANDOFF_TIMELINE_THROUGH_20260728.md`。
  4. 新建简短的根目录规则、长期上下文、当前交接入口及 `docs/project-history/00_INDEX.md`。
- 结果：当前入口只保留跨专题规则、动态授权和路由；Fast-WAM 具体值与约 29 KiB 历史时间线仍可追溯，但不再默认加载。
- 问题与修复：第一次 `apply_patch` 尝试仅移动文件，因空 hunk 被拒绝；第二次在移动时同步修改归档标题，成功完成，未删除历史。

## OP-002：服务器只读身份、仓库、资源与产物基线

- 观察边界：2026-07-28 14:22:12+08:00
- 连接：Paramiko 密码认证，`look_for_keys=False`、`allow_agent=False`；密码只在当前进程，未写入文件或命令。
- 所有远端检查设置 `PYTHONDONTWRITEBYTECODE=1`，没有执行项目 import。

### OP-002.1 身份探针

执行：

```bash
hostname
whoami
pwd
date -Ins
uname -srmo
```

结果：

- 实例 `autodl-container-nekaqbwt43-6ce5babb`，用户 `root`。
- 目标主仓确认位于 `/root/autodl-tmp/RLinf`。

### OP-002.2 Git 基线与用户内容保护

执行：

```bash
cd /root/autodl-tmp/RLinf
git rev-parse HEAD
git branch --show-current
git status --short --branch --untracked-files=all
git diff --name-status
git diff --cached --name-status
git worktree list --porcelain
git branch -vv
git stash list
git remote -v
```

结果：

- HEAD `6d0db56bf26f972cd27fa29535f5eb939e80e5bf`，分支 `local/openpi-a800-2gpu-migration`，无 upstream。
- tracked/staged diff 与 stash 均为空。
- 发现 5 个本轮前的未跟踪内容，必须原样保留：
  - `examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_a800_2gpu_baseline.yaml`
  - `examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_a800_2gpu_smoke.yaml`
  - `examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline.yaml`
  - `examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_a800_2gpu_smoke.yaml`
  - `local_scripts/monitor_pi0_resources_2gpu.py`
- `origin` 指向上游 `RLinf/RLinf`；`personal` 指向 `Yutenji-Nyamu/rlinf_fastwam`。
- 第二 worktree `/root/autodl-tmp/RLinf_fastwam_rlinf` 位于 `feat/fastwam-robotwin-grpo`、HEAD `8138d670`，tracking `personal/main`，完全 clean。

### OP-002.3 进程与资源

执行：

```bash
ps -eo pid,ppid,lstart,etime,%cpu,%mem,rss,stat,cmd --sort=-rss
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
free -h
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
df -hT / /root/autodl-tmp
```

结果：

- 无训练、Ray、RoboTwin 或 torchrun 进程；`/tmp/ray/session_latest` 是 7 月 19–20 日陈旧会话。
- 两张 A800 80GB 均为 `0 MiB`、`0%`。
- cgroup RAM 上限 240 GiB，当前约 4.3 GiB，`max/oom/oom_kill` 均为 0。
- 数据盘约 1.9 TiB，已用 994 GiB，可用 851 GiB。
- 历史 PPO/GRPO formal 内存峰值接近 242,000 MiB；这是不照搬大环境并发、首版保持 4 train env 的资源依据。

### OP-002.4 可复用产物与 DSRL 缺口

只读核对 `find/stat/du`、模型索引 JSON、配置和关键源码后确认：

- π0 RoboTwin SFT：
  `/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle`
  完整存在，约 7.6 GiB；两份 safetensors shard 和
  `physical-intelligence/robotwin/norm_stats.json` 均存在。
- RoboTwin 资产 `/root/autodl-tmp/RoboTwin_RLinf`、RLinf `.venv` 可复用。
- 历史 PPO DCP 有 step 1/10/20；GRPO DCP 有 step 1、10…100。算法状态不同，均不得作为 DSRL resume，只能参考资源占用与启动路径。
- LIBERO DSRL 主链源码与配置均已跟踪；不存在 RoboTwin DSRL config/run/checkpoint。
- 审计结束时 Git 状态与开始一致，未写服务器。

## 后续操作模板

每个远端写入或测试继续使用下列字段：

- `OP 编号 / 时间`
- `目的与精确命令`
- `修改文件或输出目录`
- `返回码与最小关键输出`
- `发现的问题及原因`
- `窄修复`
- `复测结果`
- `Git 状态与是否触及既有用户内容`

## OP-003：选择干净发布基线并创建功能分支

- 时间：2026-07-28 14:30
- 目的：避免触及 `/root/autodl-tmp/RLinf` 中 5 个既有未跟踪文件，同时继承云端已发布的 Fast-WAM 增量。
- 远端脚本：`local_scripts/remote_commands/op003_create_dsrl_branch.sh`
- 执行逻辑：

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
cd /root/autodl-tmp/RLinf_fastwam_rlinf
case "$(git rev-parse HEAD)" in
  8138d670*) ;;
  *) echo "unexpected HEAD" >&2; exit 1 ;;
esac
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git switch -c codex/dsrl-pi0-robotwin
git status --short --branch --untracked-files=all
gh --version | sed -n '1p'
gh auth status --hostname github.com
```

- 返回码：0。
- 结果：
  - 新分支 `codex/dsrl-pi0-robotwin` 已在干净 worktree 上创建。
  - `gh 2.4.0` 可用；账号 `Yutenji-Nyamu` 已认证，HTTPS Git 已配置。
  - 分支创建后 worktree 仍无 tracked 或 untracked 改动。
- 安全说明：分支名遵循当前 Codex 分支前缀规则；没有改 remote、云端仓库名或默认分支，没有 push。
- 非阻塞提示：本机 Paramiko 依赖打印 Blowfish deprecation warning；连接、主机指纹核验与命令执行均成功，不影响服务器项目。

## OP-004：环境与存储只读审计

- 时间与观察边界：2026-07-28 14:35–14:50；核心 `df/du` 容量快照为 14:43:00+08:00。
- 目的：确认最近实验使用的 Python 环境、π0 `.venv` 来源和可复用性、DSRL package delta，以及数据盘主要占用和只读清理候选。
- 边界：Paramiko 使用 `look_for_keys=False`、`allow_agent=False`；远端命令均设置 `PYTHONDONTWRITEBYTECODE=1`。未安装、复制、改名、删除文件，未运行重型项目 import，未产生新 `pycache`。

### OP-004.1 命令类别

身份、容量和仓库：

```bash
date -Is
hostname
id
df -hT / /root/autodl-tmp
free -h
git -C <repo> rev-parse HEAD
git -C <repo> branch --show-current
git -C <repo> status --short --untracked-files=all
git -C <repo> remote -v
```

环境来源、路径和只读元数据：

```bash
find <repo> -maxdepth 2 -name '.venv*' -o -name '*golden*'
stat <env>
readlink -f <env>
du -sh <env>
cat <env>/pyvenv.cfg
sed -n '1,220p' examples/embodiment/run_fastwam_robotwin_{ppo,grpo}.sh
<env>/bin/python -B -c '<sys.prefix/site/importlib.metadata/find_spec probe>'
rsync -ani --delete --exclude='__pycache__/' --exclude='*.pyc' \
  /root/autodl-tmp/backups/RLinf-pi0-venv-golden-20260717/ \
  /root/autodl-tmp/RLinf/.venv/
```

存储分层和大文件定位：

```bash
du -x -B1 --max-depth=1 /root/autodl-tmp
du -x -B1 --max-depth=1 <large-directory>
du -x -B1 --max-depth=2 <run-directory>
find <log-roots> -xdev -type f -size +1G -printf '%s %TY-%Tm-%TdT%TH:%TM:%TS %p\n'
```

最后用显式路径做无 import 的解析探针：

```bash
PYTHONPATH=/root/autodl-tmp/RLinf_fastwam_rlinf:/root/autodl-tmp/RoboTwin_RLinf \
  /root/autodl-tmp/RLinf/.venv/bin/python -B \
  -c '<importlib.util.find_spec probe>'
```

### OP-004.2 环境结论

- 最近 Fast-WAM GRPO/PPO 的 launcher 和日志 `python_interpreter_path` 都指向 conda `/root/autodl-tmp/conda/envs/FastWAM-RLinf`；当前 DSRL worktree 没有自己的 `.venv`。
- `/root/autodl-tmp/RLinf/.venv` 是 uv 创建的 Python 3.11.14 环境；7 月 14–15 日 π0 PPO/GRPO 日志明确使用它。核心版本为 Torch 2.6.0、OpenPI 0.1.0、Ray 2.55.1。
- golden `/root/autodl-tmp/backups/RLinf-pi0-venv-golden-20260717` 与当前 `.venv` 的关键 hash 相同；排除 `pycache` 后，rsync dry-run 共 68 条差异且全是目录 mtime，非目录 mtime/内容差异为 0。
- 显式 `PYTHONPATH` 探针把 `rlinf` 解析到 `/root/autodl-tmp/RLinf_fastwam_rlinf/rlinf`，OpenPI/SAPIEN/TOPPRA/Torch/OmegaConf 解析到 π0 `.venv`。
- DSRL PyTorch 路径没有新增第三方依赖；`distrax`、TensorFlow Probability、TorchRL/TensorDict 虽不存在，但当前 port 不 import。package delta 为零，不安装依赖。
- 不把复制到新路径的 venv 当 live env：console-script shebang 硬编码原 `.venv`；golden 只作恢复基线。

### OP-004.3 存储结论

- `/root/autodl-tmp` 约 1.9 TiB，已用 994 GiB、可用 851 GiB，无需为开发或 smoke 立即清理。
- 最大项：`RLinf_fastwam_rlinf` 377.08 GiB、`RLinf` 149.3 GiB、`RoboTwin` 111.8 GiB、旧 RLinf 整仓备份 110.94 GiB、models 90.7 GiB、conda 48.0 GiB、`RLinf_old` 30.90 GiB、cache 27.54 GiB。
- Fast-WAM worktree 和 RLinf 主仓的大头都是重复 DCP。正式 run 只留最后 checkpoint、删除中间 checkpoint 与 smoke DCP，分别预计可回收 296.26 GiB 和 116.18 GiB；精确目标已写入唯一实施计划第 13.2 节。
- `/root/autodl-tmp/cache/uv_python` 必须保留，因为 π0 `.venv/bin/python` 是指向它的绝对链接。`cache/uv` 和 `cache/pip` 只是可再生候选，不能据此删除整个 `cache`。
- 当前授权不包含删除；本次只形成候选清单。

### OP-004.4 无害错误与修复

1. 第一批远端只读命令用 `bash -lc` 包裹 Python `repr`，嵌套引号及一次 heredoc 尾部分号导致 shell syntax error；目标命令体没有执行。随后改用“命令正文 base64 编码、远端解码后交给 bash”的只读封装并重跑成功。
2. 一次本地 Python 构造 grep 正则时引号未闭合，解释器在 SSH 连接前报错；简化正则字符串后重跑成功，服务器未收到该命令。

- 结束状态：未写服务器，未改变任何环境、仓库、日志、checkpoint 或缓存。

## OP-005：分支补丁实现、静态检查与独立审查

- 时间：2026-07-28 14:50–15:30。
- 目的：在不运行本机项目代码的前提下，形成 RoboTwin π0 DSRL 的完整分支补丁，并在写入服务器前做静态检查和双人代码审查。
- 本地暂存区：`C:\Users\86136\Documents\rl\.dsrl-impl-worktree`；只编辑和检查文本，没有执行项目 import、compile、pytest、Hydra compose 或模型初始化。
- 服务器写入前刷新：2026-07-28 15:18:42+08:00，`/root/autodl-tmp/RLinf_fastwam_rlinf` 仍在 `codex/dsrl-pi0-robotwin`，HEAD `8138d6700e3838250c1139289ebfba43d48ff7de`，worktree clean。

### OP-005.1 连贯实现范围

1. `openpi_action_model.py`
   - 增加统一的 DSRL 64×64 主相机预处理；raw rollout image 与 replay 中已预处理 BF16 image 走同一入口。
   - Gaussian warm-up 采样一份 32D latent 并沿 H=50 重复；learned actor 保持同一 latent contract。
   - phase 使用 persistent buffer，可随现有 selective parameter sync 同步；新增配置默认值保持旧 LIBERO DSRL 行为不变。
2. `replay_buffer.py`
   - 在共享 trajectory replay 旁增加 opt-in 的 `DSRLTransitionReplayBuffer`，不替换原 buffer。
   - 投影 RoboTwin chunk：first done inclusive、success 优先、success reward=0、未成功 reward=-1、纯 truncation bootstrap、`discount=gamma**N`，并删除 terminal 后 padding。
   - flat ring 保存 compact transition、cursor/resident/total/RNG；同 rank/world-size/capacity/schema 才允许恢复。
3. `fsdp_sac_policy_worker.py`
   - 仅在 `replay_buffer.type=dsrl_transition` 时启用新 replay、global warm-up 和 query-transition UTD。
   - target FP32 shadow 只覆盖 critic image/state encoder 与 Q head；保存/恢复 shadow、`update_step`、phase、pending transition count 和 replay。
   - 旧 checkpoint 缺 trainer state 时才兼容重建；新 checkpoint 严格核对 online/target/trainer phase。
4. 配置与测试
   - 新增正式配置和继承它的薄 smoke 配置；正式值为 H=50、N=20、4 train env、global batch 256、warm-up 500、UTD 20、global replay 25,000、10-Q。
   - 新增 transition projection/ring checkpoint 与 target-shadow/resume/gradient-isolation 单测。
   - smoke 固定的是 environment reset seeds；policy 仍是官方语义的 stochastic evaluation，因此 smoke 只验调用链，不声称 fresh/resume 配对效果结论。

### OP-005.2 静态检查与预应用

执行：

```powershell
git status --short
git diff --check
git diff --stat
rg -n ".{100}" <changed-python-yaml-test-paths>
```

以及服务器只读预应用：

```bash
cd /root/autodl-tmp/RLinf_fastwam_rlinf
git apply --check --whitespace=error-all -
```

- 第一轮 `git apply --check` 返回 0；它只验证补丁，不写服务器。
- `git diff --check` 返回 0。`rg` 仅报告原文件已有长行，没有发现新增 diff whitespace 错误。
- Windows Git 提示“下次触碰时 LF 可能转为 CRLF”；补丁当前仍由 Git diff 生成，服务器预应用成功，不做无关的整文件换行重写。

### OP-005.3 审查发现与窄修复

1. resume 使用 `torch.load(..., map_location=self.device)`，而 worker 的 device 可能是整数 GPU id；改为先加载到 CPU，再由现有 `load_state_dict` 放置，消除恢复时崩溃。
2. 旧 SAC 路径对整个模型做 gradient clip：
   - critic backward 前清除上轮残留 actor grad；
   - actor backward 后、actor clip 前清除 Q 参数的 incidental grad；
   - 两个 helper 都只在 `use_dsrl` 下生效，非 DSRL 路径完全不变。
3. 主模型 checkpoint 调用没有透传 `save_full_model_weights`，会忽略 YAML 的 `false` 并重复导出冻结 π0；现已显式透传，默认仍为 `true`，不改变旧配置。
4. missing-phase 兼容原先可能掩盖新 checkpoint 损坏；现只在 trainer-state 文件不存在时启用，并同时验证 online、target、trainer 三方 phase。
5. DSRL shadow 依赖原参数名，`compile_model=True` 会改变名字；首版显式拒绝该组合，避免运行到首次 EMA 才失败。
6. 无 OpenPI 的普通单测环境只跳过两条 projection 用例；flat ring 用例仍可运行。正式服务器有 OpenPI，后续会执行全部用例。
7. 官方 DSRL 有 random crop 与 color jitter；首版只启用 RLinf 已存在的 random crop，不额外引入新 augmentation 管线，差异已写入唯一实施计划。

### OP-005.4 无害问题

- 一次 PowerShell `rg` 命令引号未闭合、一次从错误工作目录读取文件、一次文档 patch 上下文不匹配；均在本地命令解析或 patch 匹配阶段失败，没有远端写入。
- intent-to-add 后 `git diff --check` 曾发现 Markdown 行尾空格和 smoke 文件末尾多余空行；已用窄 patch 修正并复查为 0。

- 当前停点：补丁仍只在本地专用 worktree；等待最终 P0/P1 审查结论后，再次执行最新补丁的远端 `git apply --check`，随后才实际写入功能分支。

## OP-006：服务器应用、基础测试、问题修复与复测

- 时间：2026-07-28 15:31–15:44。
- 目标 worktree：`/root/autodl-tmp/RLinf_fastwam_rlinf`。
- 分支与基线保护：应用前再次确认分支 `codex/dsrl-pi0-robotwin`、HEAD `8138d6700e3838250c1139289ebfba43d48ff7de`、worktree clean；最新完整补丁再次 `git apply --check --whitespace=error-all` 返回 0。
- 实际写入：只应用 OP-005 列出的 3 个修改文件和 7 个新增文件；没有触碰 `/root/autodl-tmp/RLinf` 主 worktree 的既有 PPO/GRPO 未跟踪文件。

### OP-006.1 第一轮单测

命令环境：

```bash
PYTHONPATH=/root/autodl-tmp/RLinf_fastwam_rlinf:/root/autodl-tmp/RoboTwin_RLinf
PYTHONDONTWRITEBYTECODE=1
/root/autodl-tmp/RLinf/.venv/bin/python -B -m pytest \
  tests/unit_tests/test_dsrl_transition_replay.py \
  tests/unit_tests/test_dsrl_target_shadow_resume.py -q
```

结果：7 passed / 1 failed。唯一失败是 phase-mismatch 测试夹具没有先模拟真实 runner 已加载 target model 的步骤，所以先被更早的 shadow/target 权重一致性保护拦截。

窄修复：在该测试中先把 source target state 加载到 resumed target，再故意把 target phase 改成 0。没有放松生产代码检查。复测结果：8 passed，3 个 warning 均来自既有可选依赖的 deprecation 信息。

### OP-006.2 Hydra 组合问题与修复

第一轮 programmatic compose：

- formal 配置 compose、resolve、no-Ray validator 成功。
- smoke 失败：父 formal YAML 含 `hydra.searchpath`；Hydra 禁止 secondary config 改 searchpath。

窄修复：从新 formal YAML 删除不必要的两行 `hydra.searchpath`。配置和 defaults 均在 `train_embodied_agent.py` 的同一 `config/` 根下，不依赖该 searchpath；这样 formal 仍可直接启动，thin smoke 也能继承 formal，避免复制整份 YAML。

最终验证：

1. 原生 `train_embodied_agent.py --cfg job --resolve`：
   - `robotwin_adjust_bottle_dsrl_openpi`：通过；
   - `robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke`：通过；
   - 既有 `robotwin_adjust_bottle_ppo_openpi`：通过。
2. `validate_cfg()` 会创建 `Cluster` 并启动 Ray，不能直接当纯配置检查。测试只 mock `rlinf.config.Cluster` 和 placement 的 world-size 查询，保留其余 validator；formal/smoke 用 2 ranks，legacy PPO 用配置原有 8 ranks，三者均通过。
3. 旧 PPO 配置打印“defaults 缺 `_self_`”警告，是既有文件行为；新 DSRL formal/smoke 无该警告。

### OP-006.3 静态、导入与格式检查

- 5 个变更 Python/test 文件 `ast.parse`：通过。
- `replay_buffer`、OpenPI action model、SAC worker 从目标功能分支路径导入：通过。
- `ruff check`：通过。
- `ruff format --check` 第一轮只建议重排新增代码的换行；使用服务器环境中的 Ruff 0.15.17 格式化 5 个 Python 文件并同步回本地补丁。
- 格式化后 `ruff check`、`ruff format --check`、`git diff --check`：全部通过。
- 格式化后再次运行原生 Hydra 三配置和全部 8 个单测：全部通过。

### OP-006.4 资源与进程复核

- 检查后没有 `raylet`、`gcs_server` 或 plasma store 进程；两张 A800 均为 0 MiB / 0%。
- cgroup `memory.current` 一度约 238 GiB；拆分后匿名内存约 294 MiB、file cache 约 236 GiB，其中约 235 GiB 为 inactive file。主机仍约 980 GiB available，`oom=0`、`oom_kill=0`、`max=0`。
- 结论：这是可回收页缓存，不是测试进程泄漏；很可能与本轮此前的大范围存储扫描/共享容器缓存有关。不执行 drop-cache 或其他系统级清理，smoke 前重新刷新。
- 没有复制或改名 venv，没有安装包，没有删除日志/checkpoint/cache，没有启动模型、RoboTwin、Ray 或训练。

### OP-006.5 无害命令问题

1. 两次 PowerShell 远端命令把 `$(git ...)` 放入本地双引号，先在本地展开并导致 helper argparse 拒绝；服务器没有执行写入。之后改为无命令替换的远端正文。
2. 一次增量 patch 命令先 `git apply --check -`、随后又对同一个 stdin 执行 `git apply -`；第一步消费了 stdin，第二步报 `unrecognized input`，未应用。随后分成单次 apply 并成功。
3. 第一次 no-Ray 配置脚本暴露 smoke 的 Hydra searchpath 问题后退出；没有进入 cluster/model/env 初始化。删除无必要 searchpath 后重跑成功。

- 当前结果：服务器功能分支存在预期的 10 个未提交文件；代码、单测、原生 Hydra compose/resolve、no-Ray validator、legacy PPO 配置回归、Ruff 和 whitespace 检查均已通过。下一步更新交接与批准材料，核对精确提交范围后 commit/push；fresh/resume smoke 仍未执行。

## OP-007：精确提交与云端推送

- 时间：2026-07-28 15:48–15:50。
- 提交前现场：分支 `codex/dsrl-pi0-robotwin`，基线仍为 `8138d6700e3838250c1139289ebfba43d48ff7de`；只存在 3 个预期修改文件和 7 个预期新增文件。
- 本地补丁与服务器 10 个目标文件逐一 SHA-256 相同；最大新增文档约 61 KiB，没有 checkpoint、模型、日志或二进制文件。
- 凭据扫描没有发现密码、token、API key 或服务器地址值；历史索引中的 `secret_present: true` 只是说明原始本机材料含敏感内容，实际敏感内容没有复制进仓库。
- 使用精确 10 路径执行 `git add -- ...`；暂存后 `git diff --cached --check` 返回 0，没有额外 staged/untracked 文件。
- 使用 `git commit -s -m 'feat(embodiment): add pi0 RoboTwin DSRL port'` 创建带 DCO sign-off 的实现提交：
  - commit：`6817c73b298ff9df78d371d4b139e4e0fa8ea529`
  - 范围：10 files，3277 insertions，130 deletions。
- 使用 `git push --set-upstream personal codex/dsrl-pi0-robotwin` 推送到 `Yutenji-Nyamu/rlinf_fastwam`；远端分支与本地 commit hash 完全一致。
- 没有改 GitHub 仓库名、remote URL、默认分支；没有创建 PR。仓库改名 `rlinf_exp` 继续作为单独待讨论动作。
- 推送后服务器 worktree clean，功能分支 tracking `personal/codex/dsrl-pi0-robotwin`。
- 本 OP 的文档闭环会作为后续 docs-only 提交推送；fresh/resume smoke 和正式训练仍未执行。

## OP-008：DSRL actor 资源监控兼容

- 时间：2026-07-28 15:52–15:55。
- 现场发现：分支已有 `examples/embodiment/monitor_resources.py`，可复用历史 2 秒 CSV/peak 监控，但 actor RSS matcher 只包含 `EmbodiedFSDPActor`；DSRL worker 的进程名是 `EmbodiedSACFSDPPolicy`，不适配会漏报 actor RSS。
- 窄修改：保留原 matcher，并把 `EmbodiedSACFSDPPolicy` 加入同一 tuple；CSV schema、采样、GPU/cgroup/OOM、PID 跟随和其他 worker 分类均不改。
- 检查：
  - `git diff --check`：通过；
  - 脚本 `--help`：通过；
  - Ruff 排除文件既有的 import-order `I001` 后通过。
- 文件在本改动前已存在 Ruff import-order 和 whole-file format 差异；它们与本次三行 matcher 扩展无关，不做无关的整文件重排。
- 无害问题：第一份手写增量 patch 的 hunk 行数/上下文不匹配，`git apply` 报 corrupt patch 且未写入；更正为现场精确上下文后成功。一次包含嵌套 Python `-c` 的 helper 命令被本地 argparse 拒绝，服务器未执行该次命令；随后用无嵌套引号的 `--help`/grep 检查通过。
- fresh/resume smoke 仍未启动；该修改只保证获批后资源证据不会漏掉 DSRL actor。

## OP-009：smoke 完整批准材料

- 时间：2026-07-28 15:55–16:03。
- 使用当前服务器功能分支重新 compose formal/smoke；只 mock 会启动 Ray 的 `Cluster/placement` 资源发现，其余 `validate_cfg` 正常执行，生成过程中断言 Ray 始终未初始化。
- 生成并核对：
  - `FORMAL_VALIDATED_RESOLVED_20260728.yaml`：SHA-256 `f128166c80846ab3dddaa8e3b773b9c62db2cdb6aecaaffed452145863ef1422`；
  - `FRESH_SMOKE_VALIDATED_RESOLVED_20260728.yaml`：SHA-256 `2c616159726dc9d39fbe4d011909dcba963c95c355539c606c718a6ed25da390`；
  - `RESUME_SMOKE_VALIDATED_RESOLVED_20260728.yaml`：SHA-256 `c81e722d72dc06976f1011f42de1e9701c18d33857dfa1cc296af5216a491d46`。
- fresh/resume 完整配置 diff 严格只有 `max_steps 1→2` 和 `resume_dir null→DCP1`；两者都已经包含实际固定 output override。
- 输出根固定为 `/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1`，现场确认当前不存在。
- 新增 `SMOKE_APPROVAL_20260728.md`，包含完整配置链接、fresh/resume 精确命令、PID/driver/resource 日志、DCP 路径、资源估计、3 小时单阶段上限、停止条件和最低通过结论。
- 资源估计参考同机历史 π0 smoke 的实际峰值：29–40 GiB GPU/卡、112–125 GiB cgroup、约 27–29 分钟；DSRL 首轮保守预留 60 GiB GPU/卡、180 GiB 非可回收工作集、每阶段 30–90 分钟。
- 云端状态：实现与交接提交先推送到 `c9cf4316`；监控兼容 `15892e45` 和批准包 `4021245e` 随后创建。服务器本地 DNS 指向的 GitHub edge 连续出现 HTTP2 framing/443 timeout；公共 DNS 返回的备用 edge 经独立 TLS 探测通过后，使用一次性 `git -c http.curloptResolve=...` 推送成功。没有改 remote、全局 DNS/Git 配置或提交历史。
- 推送后本地/远端均为 `4021245e68372c497286ffaf5258f42cf2f98303`，服务器 worktree clean。
- fresh/resume smoke 仍未执行；当前等待用户明确批准。

## OP-010：fresh/resume smoke、恢复连续性与资源复核

- 时间：2026-07-28 17:04–17:47。
- 完整逐命令、PID、时间、指标、DCP/replay/shadow、资源和无害问题记录在
  [`SMOKE_EXECUTION_LOG_20260728.md`](./SMOKE_EXECUTION_LOG_20260728.md)，本节只保留交接摘要。
- 执行前 live preflight：服务器 worktree 位于 `codex/dsrl-pi0-robotwin`，HEAD/upstream
  `2d942b714b004de9a7efdbd4a7e2efaac3ef6d01`，clean；两张 A800 空闲；目标 venv、
  RoboTwin、π0 SFT、assets、seeds 和两份 resolved YAML 哈希均匹配批准包。
- fresh：
  - 40 条 macro transitions，800 次 UTD20 updates；
  - Gaussian collect → learned actor/Q/temperature → target EMA → sync → 4-episode eval →
    DCP1 全链完成；
  - loss/Q/alpha/gradient 均 finite；DCP1 trainer state 为 phase 1、`update_step=800`，
    两 rank FP32 shadow 相同。
- resume：
  - 从 DCP1 只运行一个 cycle，新增 37 条 transitions、740 次 updates；
  - 两 rank `update_step 800→1540`，phase、shadow、replay 内容及 replay sampling RNG
    精确连续；learned rollout 新 latent 全在 `[-1,1]`；
  - 新 replay 首次真实覆盖 success→reward 0/termination，同时纯 truncation 仍 bootstrap；
  - DCP2 完整写出；resume 前 DCP1 的 11 文件 SHA-256 在 resume 后逐项不变。
- 参数证据：DCP1→DCP2 的 778 个冻结 π0 tensor、4,028,019,472 参数 bitwise 不变；
  小 actor 32 个 tensor 中 27 个变化，小 critic 156 个中 119 个变化，两个 optimizer
  moments 均 finite 且非零。
- 资源：两轮 GPU 峰均约 34.8 GB/卡；训练期平均 GPU 利用率约 22–25%；SAC 约
  1.94–1.95 updates/s，是主要墙钟瓶颈。cgroup 曾触顶且 `memory.events max` 增长，
  但匿名工作集约 34–41 GB，其余主要为 file cache，`oom=0`、`oom_kill=0`。
- 终态：fresh/resume driver、monitor、Ray/worker 均退出，两卡显存归零，worktree clean，
  run root 约 63 GB；没有启动正式训练。
- 并发建议：第一版保留 2 GPU、4 env、UTD20、micro 64。micro 128 只做可选窄吞吐 A/B；
  formal 启动前必须替换 `max_steps=-1`、`val_check_interval=-1`、`save_interval=10`
  和相对 `save_path` 占位值，并重新给出正式批准包。
