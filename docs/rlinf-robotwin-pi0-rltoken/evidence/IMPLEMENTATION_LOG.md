# π0 × RoboTwin × RLT：实施与验证流水账

> 状态：2026-07-29 主体实现与 pre-smoke 检查完成，等待 Stage 1 packet 批准。
> 用户已授权服务器只读刷新、独立 RLT branch/worktree、主体实现和正式 smoke 前的必要检查。Stage 1/Stage 2 训练与 smoke 尚未授权；下载/转换数据、安装依赖和清理磁盘也不在本批授权内。
> SSH 凭据只在当前进程读取并使用，本文不记录地址、账号或密码。

## 0. 记录合同

后续每个批次必须记录：

1. 动作时间与授权边界；
2. 文件在调用链中的位置、直接上游和直接下游；
3. 来源 commit/path/symbol；
4. 新增/修改文件与关键 `+/-`；
5. 原样命令或去除凭据后的可复现命令；
6. stdout/stderr/exit code 和产物；
7. 错误、诊断、修复、复测；
8. 未验证项和下一停点。

长 stdout、resolved config、资源 CSV/PNG 或训练日志可以保存为独立 evidence 文件；本账本给出精确索引。

## 1. 2026-07-28 本轮操作索引

| ID | 类型 | 结果 |
|---|---|---|
| A001 | 读取根上下文、RLT 计划和本地锁定 RLinf 源码 | 完成 |
| A002 | Paramiko 只读身份探针 | 成功 |
| A003 | 首次远端批量审计 wrapper | 失败；远端 shell quoting 在执行主体前报错，无写入 |
| A004 | 凭据块安全解析重试 | 三次本地/认证层失败；完成原因定位，无远端写入 |
| A005 | 直接 Paramiko channel 的服务器只读审计 | 成功 |
| A006 | RLT commit/config、checkpoint/norm stats、数据根和 import 审计 | 成功 |
| A007 | `6d0db56b` 与 `8138d670` ancestry/diff | 成功 |
| A008 | RLT/RoboTwin 官方资料和数据源核验 | 完成 |
| A009 | 第二轮 ManiSkill 数据、动作/route/replay 与 resume 源码对照 | 完成；仅本机/官方资料只读 |
| A010 | 第三轮上下文精简、clean-50/stats/decode/eval 与 resume 状态机收口 | 完成；计划 v4，未访问或修改服务器 |
| A011 | 2026-07-29 实现授权、上下文重载与执行边界 | 完成 |
| A012 | 服务器现场、磁盘、Git/worktree 与基线 `48a775db` 只读刷新 | 完成 |
| A013 | 从 `48a775db` 创建独立 RLT branch/worktree | 完成 |

## 2. A001：本机上下文与源码

读取：

- `PROJECT_CONTEXT.md`
- `HANDOFF.md`
- `docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md`
- `.research-rlinf/AGENTS.md`
- `.research-rlinf` 中 RLT route、transition、rollout、OpenPI、MLP、worker、replay 和 YAML

本地锁定参考：

```text
.research-rlinf@c5ca51cc21c007a41d287159f9e1b14e0200000e
RLT subject commits:
5769c6eb feat: add RLT algorithm (#1324)
3d93750d feat(embodiment): add RLT algorithm in Maniskill (#1352)
```

关键新证据：

- 同步 RLT schedule 直接使用 `replay_buffer.total_samples` 判断 warm-up readiness。
- EnvWorker 已把 `forward_inputs["action"]` 用于训练记录，把
  `RolloutResult.actions` 送环境；不需要修改 EnvWorker 才能分离 canonical/env action。
- actor/replay 只需要 `ref_chunk[C=10,14]`；π0 生成的 H=50 不应全部塞入 compact replay。

## 3. A002～A004：登录和失败留痕

### A002：身份探针

使用低层 `paramiko.Transport`、process-only password、关闭 key/agent 自动路径、有限重试和 host-key fingerprint 核对。身份探针成功：

```text
hostname: autodl-container-nekaqbwt43-6ce5babb
remote uid: 0
```

### A003：第一次批量审计失败

最初把多行 remote script 再包进 `bash -lc <Python repr>`，远端在执行审计主体前报：

```text
bash: -c: line 1: syntax error near unexpected token 'then'
```

诊断：Python `repr()` 不是 shell-safe quoting。修复：后续通过 Paramiko channel
`exec_command(remote_script)` 直接提交多行只读命令，不再套第二层 `bash -lc`。

### A004：凭据块解析失败

为避免打印附件凭据，解析器只输出标签命中和长度。连续保留以下失败：

1. 中文 regex 经 PowerShell pipe 编码成非法 `??`，Python `re` 报
   `nothing to repeat`；未连接服务器。
2. 密码标签与值分行，inline regex 未命中；未连接服务器。
3. 误把 SSH command 行选作密码，服务器返回 `Authentication failed`。

修复：定位包含 `ssh -p` 的行，选择其后第一个非空行作为 process-only password；不打印、不落盘。随后 A005/A006 均认证成功，说明第三次是本地候选选择错误，不是有效密码失效。

## 4. A005：服务器现场只读审计

远端命令族：

```bash
date -Is
hostname
id -u
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
free -h
df -h /root/autodl-tmp
pgrep -af 'python|ray|RoboTwin|train'

git -C <repo> branch --show-current
git -C <repo> rev-parse HEAD
git -C <repo> status --short --untracked-files=no
git -C <repo> remote -v
git -C <repo> worktree list --porcelain
```

观察边界：`2026-07-28T15:12:20+08:00`。

结果：

- GPU0/GPU1：NVIDIA A800-SXM4-80GB，均 0 MiB、0%。
- RAM：1.0 TiB total、36 GiB used、964 GiB available。
- `/root/autodl-tmp`：1.9 TiB total、994 GiB used、851 GiB available。
- 仅见 Jupyter/TensorBoard 等常驻服务，未见 Ray 或训练进程。
- `/root/autodl-tmp/RLinf`：
  `local/openpi-a800-2gpu-migration@6d0db56bf26f972cd27fa29535f5eb939e80e5bf`，
  tracked tree clean。
- `/root/autodl-tmp/RLinf_fastwam_rlinf`：
  `codex/dsrl-pi0-robotwin@8138d6700e3838250c1139289ebfba43d48ff7de`，
  tracked tree clean。
- 本轮没有创建或切换 RLT branch。

## 5. A006：RLT、环境、checkpoint 和数据

执行的只读检查：

```bash
git log --oneline -- \
  rlinf/algorithms/rlt \
  rlinf/models/embodiment/modules/rlt_token_transformer.py \
  rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py

git show HEAD:examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml
git show HEAD:examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml
git show HEAD:examples/embodiment/config/env/robotwin_adjust_bottle.yaml

find <checkpoint> -maxdepth 6 -type f -name norm_stats.json
find <likely-lerobot-roots> -maxdepth 4 \
  -type f \( -name info.json -o -name norm_stats.json \)
find /root/autodl-tmp -maxdepth 6 -iname '*adjust_bottle*'

PYTHONDONTWRITEBYTECODE=1 /root/autodl-tmp/RLinf/.venv/bin/python -B \
  -c '<version/find_spec/RLT import probe>'
```

结果：

- 当前 server HEAD 含 RLT 两笔主体 commit：`5769c6eb`、`3d93750d`。
- π0 环境：
  Python 3.11.14、Torch 2.6.0+cu124、Ray 2.55.1；
  OpenPI/RLinf/OmegaConf/SAPIEN/TOPPRA 和 RLT imports 成功。
- Fast-WAM joint env 与 official env 的 `openpi=None`；不作为 RLT 基座。
- checkpoint 存在：
  `/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle`，
  `action_dim=32`、`action_horizon=50`。
- 目标 normalization 存在：
  `<checkpoint>/physical-intelligence/robotwin/norm_stats.json`。
- 常见 LeRobot cache/data 根没有发现原 `adjust_bottle` SFT dataset；只确认数据当前未出现在已审计路径，不能证明服务器任意深层路径绝对不存在。
- `RoboTwin` 树中可见官方/随附 LeRobot 转换相关脚本；本轮未执行转换或下载。

## 6. A007：开发基线 ancestry

命令：

```bash
git merge-base 6d0db56b 8138d670
git merge-base --is-ancestor 6d0db56b 8138d670
git log --oneline --graph 6d0db56b..8138d670
git diff --stat 6d0db56b..8138d670
git diff --stat 6d0db56b..8138d670 -- <pi0/rlt/robotwin paths>
```

结果：

- merge-base 是 `6d0db56b`，且它是 `8138d670` ancestor。
- 后两笔 commit 只加入 Fast-WAM 集成/发布材料；指定 π0/RLT/RoboTwin 路径无 diff。
- RLT 拟从 clean `6d0db56b` 建独立 worktree，不基于 DSRL/Fast-WAM branch。

## 7. A008：官方资料与数据源

只读核验：

- RLinf RLT 文档与两笔主体 commit；
- Physical Intelligence RLT 论文/项目页；
- RoboTwin 官方仓库和 π0 数据流程；
- `TianxingChen/RoboTwin2.0` 数据仓及 `adjust_bottle` clean-50 文件。

结论：

- 官方可下载目标：
  `dataset/adjust_bottle/aloha-agilex_clean_50.zip`。
- 官方支持 raw RoboTwin → Aloha HDF5 → LeRobot 两段转换。
- 首版只计划单任务 clean-50；不整仓下载、不先用 randomized-500、不把第三方预转换数据作为主线。
- 下载、解压和转换均属于后续写操作，需另行展示 revision、hash、路径、空间与覆盖风险后取得授权。

## 8. 本轮结束状态

- 已更新唯一 RLT 计划为 v3，逐点回答第二轮问题并进一步收口首版范围。
- 未改服务器、未建分支、未下载数据、未运行项目。
- 下一动作需要用户选择/授权：是否进入 RLT 代码实现批次；正式 smoke 仍有单独审批停点。

## 9. A009：第二轮 ManiSkill 对照与设计收口

只读核验范围：

- RLinf RLT 文档、ManiSkill Stage 1/Stage 2 config、collector 与数据 metadata；
- `rlt/route.py`、`rlt/rollout.py`、OpenPI ManiSkill/RoboTwin transform；
- `fsdp_rlt_ac_policy_worker.py`、SAC 基类 checkpoint 和 replay checkpoint；
- RLT 论文 v2 的动作域、critical phase、intervention 与 Stage 1 训练范围。

新增结论：

1. 官方 ManiSkill 参考集是 400 条成功 episode、28,681 frames、10 Hz；RoboTwin clean-50 是低预算首版，不冒充等规模复刻。
2. ManiSkill 的 reference/student 已处在兼容的 `pd_joint_delta_pos` 环境动作域；RoboTwin output transform 还做 state-dependent delta-to-absolute，因此 canonical route + 单次 decode 是必要 adapter。
3. ManiSkill route 可写成
   `record=C_t`、`actor_switch=C_t AND ready`；RoboTwin FullTask 令
   `C_t=True`，并固定 expert/human 关闭。
4. compact transition/linker/chunk TD/total-sample schedule 可直接复用；首轮
   `sample_window_size` 不小于 bounded run 总 transition，hard capacity 继续推迟。
5. 当前 ManiSkill RLT checkpoint 没有保存 `update_step`、lifetime totals 或 warm-up anchors。第二轮将最小 trainer state 收口为首批正确性修复；ready/ramp/pending 只派生，不重复保存。

本项没有创建分支、编辑服务器、下载数据或运行项目。

## 10. A010：上下文精简与实现前第三轮收口

本轮只读/本地文档范围：

- 根 `AGENTS.md`、`PROJECT_CONTEXT.md`、`HANDOFF.md`；
- RLT 唯一计划、实施账本与 workspace 全部专题文档清单；
- RLT 论文 v2、Physical Intelligence 项目页、RLinf RLT 官方文档；
- `openpi_action_model.py`、`rlt/rollout.py`、`rlt/route.py`、
  `rlt/transition.py`、RLT/SAC worker、`embodied_runner.py`、
  `huggingface_worker.py`、patch weight syncer 和 ManiSkill RLT configs。

上下文整理：

1. 将根 `HANDOFF.md` 从单一“主任务 DSRL”改成 DSRL/RLT 并行专题路由，分别记录授权，
   避免 RLT 窗口误继承 DSRL 的服务器写入权限。
2. RLT 默认读取收缩为
   `AGENTS.md -> PROJECT_CONTEXT.md -> HANDOFF RLT row -> RLT 00 plan`；
   evidence、论文、RLinf 源码、数据材料和 DSRL 基础设施均改为 trigger-based。
3. Fast-WAM、QAM、七份旧附件、根历史和 DSRL 算法正文明确排除于 RLT 默认上下文。
4. 未创建第二份并行计划；原 981 行计划重构为 638 行 v4，删除聊天式 Q&A 与逐份历史摘要，
   保留来源索引、冻结合同、调用流、改动矩阵、验收和审批停点。

本轮新增/修正的设计合同：

1. **clean-50**：首版使用全部有效 episode，无 held-out split、重复凑 400 或 data sweep；
   固定 endpoint，不宣称 held-out generalization 或等规模复刻。
2. **stats 单源**：Stage 1 loader/model、Stage 2 feature/reference 和 canonical decode 显式使用
   现有 π0 checkpoint 的同一 `norm_stats.json`；clean-50 stats 只在 delta-action 边界诊断。
3. **decode context**：同一次 OpenPI preprocessing 产出 ephemeral processed state 与 raw 32D
   action template；route 后只做一次 output transform，context 不进入 replay。
4. **eval**：始终执行 deterministic student mean，不让 train-ready gate 暗换成 reference；
   reference C10 作为独立 sanity。
5. **transition 密度**：首版继承 RLinf chunk-boundary compact transition；论文 stride-2
   overlapping subsampling 明确登记为延后差异。
6. **warm-up 语义修正**：当前 `warmup_min_size` 比较各 actor rank replay size 的 MIN，
   是 per-rank threshold，不是 global transition；global lifetime totals/anchors 才是 SUM。
7. **resume contract**：除 `update_step`、local lifetime totals 和 global anchors 外，增加
   canonical JSON/SHA256 合同，锁住 schedule、world size/sync、Stage 1 manifest、
   norm-stats、prefix/mask、H/C/D、canonical adapter、route/replay/bootstrap。
8. **首 rollout**：resume 后任何环境动作前必须 full initial sync；
   `weight_sync_interval=1`、patch `init_sync.enabled=true`，rollout version 必须等于恢复的
   `update_step`。

本轮没有创建 RLT branch/worktree，没有修改或访问服务器，没有下载/转换数据，也没有运行
Hydra compose、import/compile、测试、smoke 或训练。

## 11. A011：2026-07-29 实现授权与上下文重载

观察时间：2026-07-29（Asia/Shanghai，本机）。用户本轮明确要求“开始实现”，并要求每个
操作、命令、文件、问题、修复和结果均持续写入本账本。

本批授权解释：

- 允许服务器只读身份/资源/磁盘/Git 现场刷新；
- 核验用户指定提交 `48a775db09c16c455aeba7b0600c920e7c80d534` 后，从该提交建立
  独立 `codex/rlt-pi0-robotwin` branch 与独立 worktree；
- 允许 RLT 主体实现和正式 smoke 前的 compose/import/compile/集中测试；
- 允许在实现验证完成后提交并推送独立分支；
- 不启动 Stage 1、Stage 2 smoke、pilot 或正式训练；
- 不下载/转换 clean-50，不安装/修改共享环境，不删除任何磁盘内容。

按 `AGENTS.md` 重新完整读取：

```powershell
Get-Content PROJECT_CONTEXT.md
Get-Content HANDOFF.md
Get-Content docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
Get-Content docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
```

注意：RLT 计划文件包含混合换行；普通 `Get-Content` 报告 431 个数组元素，但按
``CRLF|LF|CR`` 分割为 638 个真实文本行。为避免漏读，随后使用 `-Raw` 和显式正则分割
读取第 431～638 行。没有因此修改文件。

同时只读搜索 workspace Memory 注册表中的 `RLT/RLToken/worktree/DSRL/disk` 条目，用于找回
“独立 worktree、config opt-in、共享 venv 不复制”的旧工程约束；这些事实仍以本轮服务器
现场和当前专题文档为准。

执行计划冻结为：

1. 只读刷新服务器磁盘、进程、GPU/RAM、Git/worktree 和指定基线；
2. 先把刷新结果写账本，再创建 worktree；
3. 连贯完成配置、canonical adapter、route/transition、bootstrap、resume 和集中测试；
4. 服务器执行必要预检，失败时一次定位一个原因并窄修复；
5. 提交/推送并准备独立的 Stage 1/Stage 2 smoke 审批材料。

本项没有访问服务器、创建分支、修改算法代码或运行项目检查。

## 12. A012：服务器现场、磁盘与指定基线刷新

### 12.1 身份与动态资源

使用 `local_scripts/remote_exec_autodl.py`，从用户附件中只在当前 PowerShell 进程提取密码，
设置临时 `SEETA_SSH_PASSWORD`，完成命令后立即删除该环境变量。先执行：

```bash
date -Is
hostname
id -u
pwd
```

结果：`2026-07-29T11:52:32+08:00`，
`autodl-container-nekaqbwt43-6ce5babb`，UID 0，cwd `/root`。host-key 固定校验与密码认证成功。
stderr 只有本机 Paramiko/cryptography 的 Blowfish deprecation warning，不影响连接。

完整刷新脚本：

```text
local_scripts/remote_rlt_20260729_readonly_refresh.sh
```

核心命令：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
free -h
df -hT /root/autodl-tmp
pgrep -af 'ray::|raylet|gcs_server|train_embodied_agent|RoboTwin|robotwin|torchrun'
git -C /root/autodl-tmp/RLinf status --short
git -C /root/autodl-tmp/RLinf worktree list --porcelain
git -C /root/autodl-tmp/RLinf show -s \
  48a775db09c16c455aeba7b0600c920e7c80d534
```

观察边界：`2026-07-29T11:53:10+08:00`。

- GPU0/GPU1 均为 A800-SXM4-80GB，0 MiB、0%；
- RAM 1.0 TiB total、65 GiB used、777 GiB available；
- `/root/autodl-tmp`：1.9 TiB total、1.2 TiB used、694 GiB available、63%；
- 进程匹配只命中本次审计 shell 本身，未见 Ray、RoboTwin 或训练进程；
- 主 worktree 仍为 `local/openpi-a800-2gpu-migration@6d0db56b`，保留 5 个既有未跟踪
  PPO/GRPO config/`local_scripts/`，本轮未触碰；
- DSRL worktree 为 clean
  `codex/dsrl-pi0-robotwin@48a775db09c16c455aeba7b0600c920e7c80d534`；
- 目标 RLT path、本地 branch、`personal` remote-tracking branch 均不存在。

### 12.2 磁盘结构

执行：

```text
local_scripts/remote_rlt_20260729_disk_inventory.sh
```

该脚本只使用有界 `du/find/stat`。所有 `du` 均 exit 0。

主要 top-level 实占：

| 路径 | bytes | 约 GiB |
|---|---:|---:|
| `/root/autodl-tmp/RLinf_fastwam_rlinf` | 572,891,234,304 | 533.6 |
| `/root/autodl-tmp/RLinf` | 160,262,770,688 | 149.3 |
| `/root/autodl-tmp/RoboTwin` | 120,085,757,952 | 111.8 |
| `RLinf_wamppo_backup_20260714_step57_lastdcp40` | 119,126,163,456 | 110.9 |
| `models` | 97,418,907,648 | 90.7 |
| `conda` | 51,531,620,352 | 48.0 |
| `RLinf_old_20260618_085536` | 33,182,851,072 | 30.9 |
| `cache` | 29,566,889,984 | 27.5 |
| `RoboTwin_RLinf` | 16,669,966,336 | 15.5 |
| `backups` | 14,657,212,416 | 13.7 |
| `experiment_exports` | 407,134,208 | 0.38 |

`RLinf_fastwam_rlinf` 的源码本身只有几十 MiB；572.9 GB 几乎全部在 `logs/`。最大运行根：

| 运行根 | bytes | 约 GiB | 时间语义 |
|---|---:|---:|---|
| 20260718 move_stapler GRPO formal | 202,395,021,312 | 188.5 | 历史 |
| 20260728 DSRL formal | 100,791,369,728 | 93.9 | 最近，先保留 |
| 20260719 move_stapler PPO formal | 86,813,331,456 | 80.8 | 历史 |
| 20260728 DSRL smoke | 67,187,593,216 | 62.6 | 最近，先保留 |
| 20260718 adjust_bottle GRPO formal | 57,827,512,320 | 53.9 | 历史 |
| 20260719 move_stapler PPO smoke | 28,935,467,008 | 27.0 | 历史 |
| 20260718 adjust_bottle GRPO smoke | 28,912,521,216 | 26.9 | 历史 |

因此用户提到的“60 多”对应 DSRL smoke 单个 run root 约 62.6 GiB，不是服务器总占用。
本轮未删除任何文件。当前建议先保留最近 DSRL formal 三个 DCP、DCP195、smoke、轻量 export、
π0 SFT/norm stats、RoboTwin assets、当前 `.venv`、golden venv backup 和 `cache/uv_python`；
历史 Fast-WAM/PPO/GRPO 中间 DCP、旧整目录、legacy conda env 与可再生 cache 只列为后续
逐项审批候选，不能按顶层目录直接删除。

### 12.3 `48a775db` 基线事实

执行：

```text
local_scripts/remote_rlt_20260729_base_diff.sh
```

结果：

- `48a775db` 存在，提交标题为 `docs(dsrl): record closeout publication`；
- 旧 clean 基线 `6d0db56b` 是其 ancestor；
- `6d0db56b..48a775db` 包含 Fast-WAM、DSRL 代码及 DSRL 文档/实验记录；
- RLT 重叠路径只有 `openpi_action_model.py` 与 `fsdp_sac_policy_worker.py`；
- RLT route/rollout/transition、Stage 1/Stage 2 ManiSkill configs 和 RLT worker 均存在。

这意味着 `48a775db` 不是“纯官方 π0”提交。由于用户明确指定从它开始，本批不擅自退回
`6d0db56b`；隔离措施是独立分支/worktree、RLT-only config、独立输出，以及对 ManiSkill RLT
和 RoboTwin π0 PPO/GRPO legacy 路径做回归。

## 13. A013：创建独立 RLT branch/worktree

脚本：

```text
local_scripts/remote_rlt_20260729_create_worktree.sh
```

等价关键命令：

```bash
git -C /root/autodl-tmp/RLinf cat-file -e \
  48a775db09c16c455aeba7b0600c920e7c80d534^{commit}
git -C /root/autodl-tmp/RLinf worktree add \
  -b codex/rlt-pi0-robotwin \
  /root/autodl-tmp/RLinf_rlt_pi0_robotwin \
  48a775db09c16c455aeba7b0600c920e7c80d534
```

脚本在写操作前再次拒绝已有 path/branch；两个条件均未触发。结果：

```text
/root/autodl-tmp/RLinf_rlt_pi0_robotwin
branch: codex/rlt-pi0-robotwin
HEAD: 48a775db09c16c455aeba7b0600c920e7c80d534
tracked/untracked status: clean
```

没有重新 clone 仓库。三个 worktree 共享 Git object database，但各有独立 checkout/index：

```text
/root/autodl-tmp/RLinf                         -> 6d0db56b, 主 π0
/root/autodl-tmp/RLinf_fastwam_rlinf           -> 48a775db, DSRL
/root/autodl-tmp/RLinf_rlt_pi0_robotwin        -> 48a775db, RLT
```

后续显式使用共享只读环境，不复制/安装：

```bash
PYTHONPATH=/root/autodl-tmp/RLinf_rlt_pi0_robotwin:/root/autodl-tmp/RoboTwin_RLinf \
PYTHONDONTWRITEBYTECODE=1 \
/root/autodl-tmp/RLinf/.venv/bin/python -B ...
```

Worktree 只隔离源码和 Git 状态；共享 `.venv`、模型/数据、Ray/GPU、端口和 output path 仍需
显式隔离。本批不会与 DSRL 同时启动任务，并为 RLT 使用独立 experiment/output 名。

## 14. A014：精确基线镜像、传输故障与本地编辑副本

Windows 本机没有 `48a775db` 对象；为避免从其他提交近似实现，先在服务器从精确 commit 生成
只包含 Git 对象的 bundle：

```bash
git -C /root/autodl-tmp/RLinf bundle create \
  /root/autodl-tmp/experiment_exports/rlt_base_48a_20260729.bundle \
  48a775db09c16c455aeba7b0600c920e7c80d534
stat --format='%s' \
  /root/autodl-tmp/experiment_exports/rlt_base_48a_20260729.bundle
sha256sum \
  /root/autodl-tmp/experiment_exports/rlt_base_48a_20260729.bundle
```

结果：

- 大小：`15,745,909` bytes；
- SHA256：`e7b33c7ece3963c913fdc461dc1c48c6363a88c8140a82cd687b8db764350819`。

第一次通过 `remote_exec_autodl.py download` 直接 SFTP 下载时，60 秒命令超时，本地留下
`5,931,008` bytes 的不完整临时文件。该文件未被当作有效输入。随后尝试过两个后台下载办法：

1. PowerShell `Start-Process`：因为向新进程重复传入大小写不敏感的 `Path/PATH` 环境键而失败；
2. `cmd /c start`：参数引号没有按预期传入；
3. 独立 Python 启动器：新解释器加载 `cryptography` 的 `_rust` DLL 失败。

这些尝试都没有修改服务器代码或 Git 状态。最终改用可校验的分块传输：

```bash
split -b 4000000 -d -a 2 \
  /root/autodl-tmp/experiment_exports/rlt_base_48a_20260729.bundle \
  /root/autodl-tmp/experiment_exports/rlt_base_48a_20260729.bundle.part
sha256sum \
  /root/autodl-tmp/experiment_exports/rlt_base_48a_20260729.bundle.part*
```

四块大小依次为 `4,000,000 / 4,000,000 / 4,000,000 / 3,745,909` bytes，SHA256 依次为：

```text
58dcf17344d5e889bfb6b4b64a604bdd123e11d7366449a3eb9f076dc53b2238
fbf898cc01100d08bb14d6798129a3fca0dd539a11e48dd73a629874d0425535
52610202df9064413cba4a4e297aa31dd7d6cac94a8745956c55ddc3c85ad842
50ef07c6cbb74f8b13085d73a8f90e613389aed4923137b3ea38ed28e807023b
```

逐块下载并核验后，Windows 使用二进制拼接生成
`.tmp/rlt_base_48a_20260729.complete.bundle`；总 SHA256 与服务器 bundle 一致。随后：

```powershell
git clone .tmp/rlt_base_48a_20260729.complete.bundle .rlt-impl-worktree
git -C .rlt-impl-worktree switch -c codex/rlt-pi0-robotwin `
  48a775db09c16c455aeba7b0600c920e7c80d534
git -C .rlt-impl-worktree rev-parse HEAD
git -C .rlt-impl-worktree status --short
```

本地 staging clone 的 HEAD 精确为 `48a775db...` 且初始 clean；它只用于阅读、`apply_patch`、
生成 diff 和文档，不在 Windows 运行项目 compose/import/test。服务器独立 worktree 仍是唯一执行现场。
本地完整读取了该 clone 的 `AGENTS.md`。

## 15. A015：RLT 主体第一批代码编辑

编辑位置：

```text
C:\Users\86136\Documents\rl\.rlt-impl-worktree
```

所有写入均由 `apply_patch` 完成。第一批修改：

| 文件 | 修改 | 兼容边界 |
|---|---|---|
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | 新增 Stage 1 真冻结、canonical action adapter、一次 preprocessing 的 decode context 与单次 output transform | 新字段默认保留旧语义；identity adapter 不变 |
| `rlinf/models/embodiment/openpi/__init__.py` | 权重加载后按 `rlt_train_vla=false` 调用冻结 | 仅 RLT opt-in |
| `rlinf/algorithms/rlt/rollout.py` | canonical student/reference route 后再统一 decode；replay 仍存 canonical action | identity adapter 走原路径 |
| `rlinf/algorithms/rlt/route.py` | 新增 `FullTaskRLTRoute`：train 未 ready 用 reference、ready 用 student；eval 恒 deterministic student | ManiSkill route 未改 |
| `rlinf/algorithms/rlt/transition.py` | 显式 transition-replay capability；缺省仍由 ManiSkill 枚举判断 | 旧 config 不变 |
| `rlinf/algorithms/rlt/__init__.py` | 导出 full-task route | 仅符号增加 |
| `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py` | compact replay、pure truncation 的真实 next obs/bootstrap、最小 per-rank RLT trainer state 和 strict resume contract | 全部由新 config opt-in |

实现中发现并当场修正的两个设计风险：

1. `rlt_alpha=0` 不是冻结。若 π0 参数仍进入 AdamW，即使梯度为零也可能被 weight decay 改动；
   因此 Stage 1 增加 `rlt_train_vla=false`，冻结全部 π0 参数并只打开 `rlt_module.*`。
2. pure truncation 不能只把 TD mask 从 `done` 改成 `termination`；replay 中的 `next_obs` 也必须使用
   linker 保存的真实 final observation，否则会从错误状态 bootstrap。

本批结束时：

```powershell
git -C .rlt-impl-worktree diff --stat
git -C .rlt-impl-worktree diff --check
```

得到 7 个 tracked 文件修改、`704 insertions / 24 deletions`；`diff --check` 无空白错误。
Windows 提示未来可能把 LF 转为 CRLF，这是 working-copy 提示，不是内容错误；在生成/应用服务器 diff
前仍需检查行尾和补丁一致性。此时尚未完成配置、测试，也尚未把 diff 应用到服务器。

## 16. A016：配置、合同测试和真实前缀探针

继续只在 `.rlt-impl-worktree` 使用 `apply_patch` 完成第二批实现。

新增配置：

| 文件 | 用途 | 主要来源与适配 |
|---|---|---|
| `examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml` | clean-50 的固定 endpoint Stage 1 候选 | 结构沿用 ManiSkill RLT Stage 1；数据、π0 checkpoint、RoboTwin stats、`H=50/C=10/D=14` 沿用当前 RoboTwin π0 |
| `examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml` | 正式 pilot 候选，不是本轮直接执行命令 | RLT AC/MLP 优化器和 actor-weight schedule 沿用 ManiSkill；2 GPU/4 env、RoboTwin env/model/action 语义沿用已验证 π0/DSRL |
| `examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke.yaml` | 一步 Stage 2 smoke 候选 | 只缩小 horizon 内运行规模、warm-up 和 replay window，不改变模型、adapter、loss 或 resume 合同 |

Stage 1 当前冻结为：

- converter 只转 1 episode 做格式合同检查；
- 训练读取解包后的全部有效 clean-50，不复制到 400，也不划 45/5；
- `max_steps=2000` 固定一个 endpoint，不按下游 RL 成绩挑 checkpoint；
- `rlt_train_vla=false`，只训练 `rlt_module.*`；
- 同一 RoboTwin normalization stats 同时绑定 Stage 1 manifest 和 Stage 2 resume contract；
- `rlt_prefix_seq_len=768`、`rlt_input_dim=2048` 是待真实 checkpoint 探针确认的候选值，
  不是凭文档直接当作已验证事实。

Stage 2 当前冻结为：

- full-task route，训练在 ready 前用 reference，ready 后用 stochastic student；eval 始终用
  deterministic student；
- `robotwin_aloha_canonical_v1` adapter：replay/critic 使用 canonical `C x D` action，
  环境侧只做一次 output transform；
- 2 actor GPU、2 env GPU、每个 env rank 2 个环境，总计 4 env；
- `H=50/C=10/D=14`，`z=2048`，MLP critic；
- `update_epoch=5`、`train_every_n_rollouts=1`，即每个新 transition 约 5 次更新；
- 每 rank warm-up 500 transitions，再累计 5000 transitions 线性降低 BC；每次触发最多消费
  400 个 pending updates；
- `gamma=0.99`、`tau=0.005`、actor/critic LR 均为 `1e-4`、global/micro batch
  `512/128`；
- replay `sample_window_size=15000`，compact transition replay opt-in；
- Stage 1 manifest ID/SHA 和 normalization SHA 暂时显式写成 `UNRESOLVED`；因此配置可 compose，
  但 save/resume 会 fail closed，直到 Stage 1 真实产物生成。

新增：

```text
tests/unit_tests/test_robotwin_rlt_contract.py
toolkits/rlt/probe_robotwin_rlt_prefix_contract.py
```

测试覆盖 route truth table、legacy builder、transition capability、冻结、canonical decode、
world-size=1 的 save/load/continue、缺文件/合同不一致、stale completion marker、compact replay 和
pure truncation。探针只加载现有 π0 checkpoint 和 stats，用 synthetic 三相机 observation 检查
真实 prefix/mask/state；可选 `--with-rlt` 再检查 trainable names、`z_rl` 和初始 reconstruction
loss，不启动训练。

并行代码审阅发现并修正：

1. canonical decode 最初返回了 output transform 后完整 `H x 32` action；已改成写入模板、
   transform 一次后再切回环境实际需要的 `C x D`。
2. compact replay 不能删除 `intervene_flags`，因为 intervention-aware actor loss 仍读取它；
   已保留这个很小的 tensor。
3. checkpoint complete marker 不能只在保存成功后写；否则覆盖同一步 checkpoint 时可能误用旧
   `complete=true`。现在 rank 0 在 base checkpoint 写入前原子写 `complete=false`，全部 rank
   完成后才原子改为 `complete=true`。
4. secondary Hydra config 不能覆盖 `hydra.searchpath`；Stage 2 formal config 已和已验证 DSRL
   路线一样移除该项，避免 compose 阶段失败。

明确推迟：RNG bitwise resume、跨 world-size、async resume，以及
`rollout_epoch>1` 时旧 worker `done_idx=t+1` 的潜在对齐问题。当前候选固定
`rollout_epoch=1`，但多 rank save→load→continue 仍必须进入正式 smoke 合同。

## 17. A017：完整 diff 传输、服务器格式化与聚焦回归

为同时传输 tracked 和 untracked 文件，先用 `apply_patch` 扩展
`local_scripts/remote_exec_autodl.py`，新增 `--stdin-git-full-diff`：

```text
temporary GIT_INDEX_FILE
git read-tree HEAD
git add -A
git diff --cached --binary HEAD
```

该机制不修改真实 index。第一次在 PowerShell 中内嵌远端引号时发生本地 parser error；远端没有
执行任何命令。随后改成独立 shell command file，再依次执行：

```bash
git apply --check
git apply
```

完整补丁在 `/root/autodl-tmp/RLinf_rlt_pi0_robotwin` 检查和应用成功；DSRL 和主 π0
worktree 均未修改。后续两项小修复分别生成：

```text
.tmp/rlt_incremental_incomplete_marker.patch
.tmp/rlt_incremental_remove_searchpath.patch
```

每个都先 `git apply --check`，再应用到服务器 RLT worktree。

服务器固定运行边界：

```bash
cd /root/autodl-tmp/RLinf_rlt_pi0_robotwin
export PYTHONPATH=/root/autodl-tmp/RLinf_rlt_pi0_robotwin:/root/autodl-tmp/RoboTwin_RLinf
export PYTHONDONTWRITEBYTECODE=1
/root/autodl-tmp/RLinf/.venv/bin/python -B ...
```

执行 `local_scripts/remote_rlt_20260729_static_checks.sh`：

1. 9 个改动/新增 Python 文件 AST compile 全部通过；
2. 第一次 `ruff check` 只报告新增测试文件 import sorting；
3. 执行 `ruff check --fix` 修复 1 项；
4. 执行 `ruff format`，6 个文件被格式化、3 个无需变化；
5. 重新检查：AST 9/9、Ruff `0.15.17` check 全通过、9 个文件 format check 全通过、
   `git diff --check` 通过。

服务器 formatter 的 9 个精确结果随后机械同步回 Windows staging clone，避免本地和服务器产生
两套 diff；这是 formatter 输出同步，没有在 Windows 运行项目代码。

执行 `local_scripts/remote_rlt_20260729_focused_tests.sh`：

```text
imported rlinf:
/root/autodl-tmp/RLinf_rlt_pi0_robotwin/rlinf/__init__.py

16 x tests/unit_tests/test_robotwin_rlt_contract.py
4  x tests/unit_tests/test_dsrl_transition_contract.py
4  x tests/unit_tests/test_dsrl_target_shadow_contract.py
```

结果：`24 passed`，耗时 `8.96s`；只有 1 条既有 deprecation warning。RLT 新测试和基线中的
DSRL transition/target-shadow 回归均通过。

## 18. A018：原生 compose 与无 Ray 配置验证

执行 `local_scripts/remote_rlt_20260729_native_compose.sh`，使用 RLinf 自身 Hydra 入口分别
compose/resolve 7 份配置，并把 resolved YAML 写到：

```text
/root/autodl-tmp/experiment_exports/rlt_pre_smoke_20260729
```

结果：

| 配置 | resolved 行数 | SHA256 |
|---|---:|---|
| RoboTwin Stage 1 candidate | 105 | `498f1e4d759566caadcb4c5039ae4651ef0833e1a5197430160e4bb27cf5ea4e` |
| RoboTwin Stage 2 candidate | 340 | `b2eb1512703ded4b0a9f1af5a2024f29e79938be5269992116e9192f644be8d8` |
| RoboTwin Stage 2 smoke candidate | 340 | `8e600c55803e7f7fc2856abb16731eb16b1b8f56450ca6816007d966caacc22d` |
| legacy ManiSkill RLT Stage 1 | 未单独计入本地表 | `cd5df33...` |
| legacy ManiSkill RLT Stage 2 | 未单独计入本地表 | `75fb58...` |
| legacy RoboTwin π0 PPO | 未单独计入本地表 | `1185ea...` |
| legacy RoboTwin π0 DSRL | 未单独计入本地表 | `ee42e1...` |

7 份配置均 compose/resolve 成功；总耗时约 `66.5s`。stderr 只有继承的 Hydra `_self_`
warning 和 TensorFlow CPU 信息，没有配置失败。

随后执行 `local_scripts/remote_rlt_20260729_no_ray_validate.sh`。脚本保留真实
`rlinf.config.validate_cfg`，只把会连接集群的 `Cluster` 和 placement 查询替换为静态、无副作用
实现。它首先确认 Ray 未初始化，逐个验证 resolved YAML，再次确认 Ray 未初始化。

结果：

- 首轮 7/7 通过 `validate_cfg`；
- RLT Stage 1：actor world size 2，batch `32/16`；
- RLT Stage 2/formal smoke：actor/env world size `2/2`，4 env，batch `512/128`，
  `num_action_chunks=10`；
- legacy ManiSkill RLT、RoboTwin π0 PPO 和 RoboTwin π0 DSRL 同时通过；
- `ray.is_initialized()` 前后均为 `false`。

本项是 config/preflight 验证，不是模型加载、模拟器运行、smoke 或训练证明。

为了覆盖用户点名的既有 GRPO 路径，又追加原生 compose：

```text
robotwin_adjust_bottle_grpo_fastwam_a800_2gpu
```

生成 `legacy_robotwin_pi0_grpo_fastwam_resolved.yaml`，302 行，SHA256：

```text
3be0a6ea732ce7ff988721b5b7ed330ac5a4222ed774ea1a773366cf00bb70b9
```

第一次把该文件交给无 Ray validator 时失败：

```text
NotImplementedError: Model Type: fastwam_robotwin not supported
```

定位结果不是 YAML 或 RLT 回归，而是检查 harness 只导入了 `rlinf.config`；生产
`train_embodied_agent.py` 会先导入 rollout worker，再由 `rlinf.models` 的 import side effect
注册 custom Fast-WAM model。修复检查脚本，使其显式 `import rlinf.models` 后重跑：
8/8 通过，Ray 仍未初始化。没有为此改 Fast-WAM、DSRL 或通用 config 代码。

## 19. A019：真实 π0/RLT 合同探针

先后台执行：

```bash
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/RLinf/.venv/bin/python -B \
  toolkits/rlt/probe_robotwin_rlt_prefix_contract.py \
  --device cuda:0 \
  --expected-hidden-width 2048 \
  --expected-image-tokens 768
```

它只加载现有 checkpoint、stats 和 synthetic 三相机 observation，不启动训练/Ray/模拟器。
结果：

```text
full prefix       [1,816,2048], bfloat16, mask true 773
language tokens   [1,48]
image prefix      [1,768,2048], mask true 768
processed state   [1,32]
norm-stats SHA256 649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
```

候选 config 的 `L=768/input=2048` 因而从“推断”升级为真实 checkpoint 事实。

再以同一命令增加 `--with-rlt`，结果：

```text
z_rl                         [1,2048]
initial reconstruction loss  5.168433666229248
trainable tensor count       62
trainable parameter count    743,094,272
all trainable names          rlt_module.* only
```

该结果证明冻结/形状/初始 loss 可计算，但不证明 loss 会下降或 Stage 1 已训练。它还推翻了
配置注释中“small RLT module”的说法：7.43 亿可训练参数配合 `no_shard`，正式
micro16/global32 不能未经真实 micro1/global2 forward→backward→optimizer smoke 就冻结。

两个后台进程都正常退出；检查后两张 GPU 回到 `0 MiB/0%`，未见 Ray、RoboTwin 或训练进程。

## 20. A020：最终只读审阅、窄修复和第二轮回归

三路只读审阅分别覆盖 config、data/action/route/replay 和 resume。审阅未发现核心调用链的
新 defect，但给出一个代码级 resume blocker 和两个配置门禁：

1. resume fingerprint 未包含 `algorithm.bootstrap_type`，改变 terminal bootstrap 语义仍可能
   通过合同；
2. Stage 1 token module 为 7.43 亿参数，正式 batch 需要先做最小显存 smoke；
3. Stage 2 smoke 原 `warmup_min_size=4/rank` 恰好等于 20 primitive steps 的理论最大
   采样量；任一提前终止都可能导致没有 learner update。

用 `apply_patch` 做以下窄修复：

- resume fingerprint 增加 loss/Q 聚合、bootstrap、target update、Q/BC 权重、
  actor/critic optimizer 和 global/micro batch；
- 新增 bootstrap 语义变化必须 fail-closed 的单元测试；
- Stage 1 删除冗余 `hydra.searchpath` 和未被实现消费的 `rlt_encoder_type`；
- 新增薄
  `robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke.yaml`：
  2 steps、micro/global `1/2`、warm-up 1、step 2 保存；
- Stage 2 smoke 的 `warmup_min_size` 改为 2/rank；其余模型、H/C/D、route、loss、
  update 和 DCP 主链不变；
- 正式 Stage 2 config 明确 replay 15k 是 per rank，并明确 cap400 的语义：在满长失败
  rollout 下 steady UTD5 成立，但 pending 约 4600，不承诺把初始 floor debt 清零；
- 探针增加真实 canonical reference：
  `decode(ref_chunk, context)` 与旧 `output_transform(raw_template)[:C,:D]` 的 allclose 门禁。

第一次手写 incremental patch 有两次 `git apply --check` 失败：先是新增文件 hunk 行数错误，
再是相邻 hunk header 计数错误。两次都停在 `--check`，服务器文件没有变化。为避免继续手算
hunk，改用可复核流程：

1. 服务器 alternate index 导出当前完整 diff，85,113 bytes，SHA256
   `97d9aee6b2cc33ed8caccff640e52fe7a04e0faf97b1b154d6153173da66727f`；
2. Windows 从精确 48a bundle 创建一次性临时 clone并应用该 diff，得到服务器当前态；
3. 把本地新文件机械复制到临时 clone，以服务器当前态作为 index 自动生成 incremental diff；
4. 得到 11,312 bytes patch，SHA256
   `7a51f77da1eb1a191edbe63a450e50b9099f7f8e1bf04f2508d312ef7e0333d4`；
5. 服务器 `git apply --check` 成功后再 `git apply` 成功。

随后服务器执行：

```text
ruff check --fix
ruff format
AST compile
ruff check
ruff format --check
git diff --check
```

结果：Ruff 只重排 1 个 worker 文件；复查 9/9 Python 文件通过。通过逐文件 SHA256 对比，
只把该 formatter 结果机械同步回 Windows。

重新执行集中测试：

```text
17 x test_robotwin_rlt_contract.py
4  x test_dsrl_transition_replay.py
4  x test_dsrl_target_shadow_resume.py
```

结果：`25 passed, 1 warning in 9.14s`。新增 bootstrap fingerprint 测试通过。

重新 compose 四份候选配置：

| 配置 | 行数 | SHA256 |
|---|---:|---|
| Stage 1 formal candidate | 104 | `bb0cc71cc69cf1a90e495f493f720c5ce864cc2ede6c13a8295e17156d6b7615` |
| Stage 1 smoke candidate | 104 | `7eee3a33d57275d732a88d1f7e0e028e109cd7a286c252687e29c15098712a79` |
| Stage 2 formal candidate | 339 | `bdc1ffa9d475457579522964b056677c1328aba502333633a13ea7f467917c88` |
| Stage 2 smoke candidate | 339 | `197900afbaa783e53f68b4f7097fba048623a0199325a086f659cbe71e540077` |

最终 9 份 resolved config 全部通过真实 `validate_cfg`，其中包括两份 ManiSkill RLT、
RoboTwin π0 PPO、Fast-WAM GRPO、DSRL 和四份 RLT candidate；全过程
`ray.is_initialized() == false`。

最后重跑增加 parity 的真实 RLT probe。由于 token module 每次新建时随机初始化，本次
initial reconstruction loss 为 `5.223368167877197`；该值与第一次 `5.1684` 的用途都只是
确认有限、可计算，不作跨初始化比较。新增门禁结果：

```text
canonical reference shape  [1,10,14]
decoded reference shape    [1,10,14]
legacy decode max abs      0.0
legacy decode parity       true
```

即真实 checkpoint 下
`decode(ref_chunk, decode_context)` 与旧 reference 的
`output_transform(raw_template)[:10,:14]` 完全一致。探针正常退出，未启动训练。

## 21. A021：证据固化与最终静态现场

把四份最终 candidate resolved YAML、两份 probe JSON 和 GRPO legacy resolved YAML
从服务器机械下载到：

```text
docs/rlinf-robotwin-pi0-rltoken/evidence/rlt_pre_smoke_20260729/
```

候选 config 的本地 SHA 与服务器输出逐一一致。新增
`01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md`，集中记录每组参数来源、clean-50 能与不能
证明什么、Stage 1/Stage 2 fresh/resume 验收、未解析变量和磁盘结构。专题主计划、packet、
完整账本和小型 resolved/probe 证据随后机械复制进 RLT staging clone，并通过生成补丁：

```text
124,487 bytes
SHA256 3a9a70e32a97b4259ccc97d6340eca22177657bc8638ca03b43f3c089ded1e2a
```

在服务器 RLT worktree 先 `git apply --check`，再应用。第一次文档补丁保留了 Markdown
hard-break 的两个尾随空格，`git apply` 只警告但仍成功；为保持 `git diff --check` 干净，
使用 `apply_patch` 删除这些尾随空格并把账本状态改为“等待 Stage 1 packet 批准”，再用
自动生成的 3,296-byte incremental patch
（SHA256 `9b3455ce4c666beb18a752426129546e38ca73a109020215a98f2a87a7ae953d`）
检查、同步到服务器。

根 `HANDOFF.md` 已更新当前路由、授权边界和下一步；没有把无关 QAM/DSRL 文档复制进
RLT 专题默认上下文。

最终只读现场边界：`2026-07-29T13:18:06+08:00`。

- GPU0/GPU1：A800-SXM4-80GB，均 `0 MiB/0%`；
- RAM available 778 GiB；
- `/root/autodl-tmp` 可用 694 GiB，63% used；
- 未见 Ray、RoboTwin、SFT、embodied training 或 prefix-probe 进程；
- 主 π0 worktree 的 4 个既有 config 与 `local_scripts/` 仍保持 untracked、未触碰；
- DSRL worktree仍 clean；
- RLT worktree仍是独立 `codex/rlt-pi0-robotwin@48a775db...`，只有本批预期代码、
  config、test、toolkit 和专题文档变更。

截至本项仍没有下载/转换 clean-50，没有运行 Stage 1/Stage 2 smoke、训练、Ray、
RoboTwin simulator 或 checkpoint save/load。

## 22. A022：提交与首次推送

提交前用 alternate index 纳入 tracked + untracked 全量变更，执行：

```text
git diff --cached --check
name-status/stat
新增文件 1 MiB size guard
private-key/AWS/OpenAI-key/SEETA env assignment 敏感模式扫描
git remote -v
git branch -vv
```

结果：

- 23 个文件，`5,146 insertions / 24 deletions`；
- full diff whitespace check 通过；
- 最大新增文件约 43 KiB，没有大文件/checkpoint；
- credential pattern scan 通过；
- 变更范围只有 RLT code/config/test/toolkit 和本专题文档/小型证据。

服务器 RLT worktree 执行：

```bash
git add -A
git diff --cached --check
git commit -m "feat(rlt): port RoboTwin pi0 RL-token training"
```

得到 implementation commit：

```text
1a923a2305b9bd01647fb66509f19caceff1a310
```

随后：

```bash
git push -u personal codex/rlt-pi0-robotwin
```

新 remote branch 推送成功并建立 tracking：

```text
personal/codex/rlt-pi0-robotwin
```

没有创建 PR，也没有合并到主 π0、DSRL 或 personal main。此后仅允许追加本账本 closeout
和最终交接，不再改变已验证算法/config。

## 23. A023：closeout commit 推送故障与恢复

把 A022 追加为独立 docs commit：

```text
3a23317d81828293f0a0c0e93e516b8f5af2e2c5
docs(rlt): record pre-smoke implementation evidence
```

第一次普通 push 失败：

```text
fatal: ... Error in the HTTP2 framing layer
```

随后两次 HTTP/1.1 重试在本地 SSH 调用 50–60 秒超时后仍留有服务器 push/helper 进程。
每次都先只读确认：

- remote-tracking ref 仍为 `1a923a...`，没有部分更新；
- 精确 PID 仍停在 `git push` / `git-remote-https`；

等待约两分钟仍无进展后，只终止这两个精确 PID。没有停止训练（现场无训练），也没有删除、
reset 或改写 Git ref。SSH Git 探针把 GitHub ED25519 host key 加入服务器
`known_hosts`，但服务器没有 SSH public-key 权限，因此没有切换 remote protocol。

继续检查：

- `curl` 到 `github.com` / `api.github.com` 均 HTTP 200；
- `gh auth status` 确认既有账号和 HTTPS credential helper 正常；
- 没有输出或保存 token。

最后使用服务器侧 `timeout 25s` 包住一次有界 `GIT_TRACE/GIT_CURL_VERBOSE` 普通 push。
trace 中 Authorization 按 Git 默认显示为 `<redacted>`；确认 1,550-byte pack 已完整上传并收到
GitHub HTTP/2 200。推送成功：

```text
1a923a23..3a23317d
codex/rlt-pi0-robotwin -> personal/codex/rlt-pi0-robotwin
```

因此算法 implementation commit 和 docs closeout commit 都已在远端。该网络故障没有改变
代码、测试结果、branch 历史或 smoke 授权边界。

## 24. A024：最终 remote/worktree 对齐

把 A023 作为最后一个 branch-side publication recovery 记录提交：

```text
cfa556550efa7da1779a0d29c3a34b00a7f17ed8
docs(rlt): record publication recovery
```

使用服务器侧 `timeout 25s git push`，本次 4.1 秒成功：

```text
3a23317d..cfa55655
codex/rlt-pi0-robotwin -> personal/codex/rlt-pi0-robotwin
```

最终只读核验边界：`2026-07-29T13:35:17+08:00`。

- server HEAD 与 upstream 都是 `cfa556550efa7da1779a0d29c3a34b00a7f17ed8`；
- `HEAD...@{u}` 为 `0 0`；
- RLT worktree status clean；
- DSRL worktree status clean；
- 两张 GPU 均 `0 MiB/0%`；
- 未见 Ray、RoboTwin、SFT、embodied training 或 probe 进程。

本节是 push 完成后的根工作区最终交接记录；不再为了记录“最后一次记录提交”递归制造新
branch commit。远端 branch 内的账本截至 A023，根工作区本文件截至 A024。

## 25. A025：2026-07-29 Stage 1 配置复核与数据/磁盘刷新

用户要求在 smoke 前继续逐项讨论 Stage 1/Stage 2 配置，并确认、下载 clean-50；下载属于
服务器写操作，必须和代码实现一样记录命令、结果、问题与解决。本轮授权解释为：

- 可以按此前中国大陆 AutoDL 的成功经验，把锁定的 clean-50 单文件下载到
  `/root/autodl-tmp`；
- 可以创建只用于下载的目标目录、Hugging Face cache、互斥锁、下载脚本和日志；
- 可以做 ZIP 大小、SHA256、完整性和只读目录/schema 检查；
- 本轮仍不解压、不转换数据，不启动 Stage 1/Stage 2 smoke、训练、Ray 或模拟器；
- 不删除、覆盖或清理服务器任何既有文件。

按工作区规则重新完整读取：

```text
PROJECT_CONTEXT.md
HANDOFF.md
docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md
evidence/IMPLEMENTATION_LOG.md 的数据、配置、磁盘和最终状态批次
```

同时只读搜索 Memory 注册表的 RLT/clean-50/ManiSkill/worktree 条目；动态事实仍以下面的
服务器现场为准。

新增本地可审计脚本：

```text
local_scripts/remote_rlt_20260729_prestage1_refresh.sh
local_scripts/remote_rlt_20260729_clean50_download.sh
```

第一个脚本只执行身份、GPU/RAM/磁盘、进程、Git/worktree、目标 ZIP、网络能力和主要目录
分层审计。第二个脚本锁定：

```text
repo: TianxingChen/RoboTwin2.0
revision: 9dc9299c163db059931898a9f0852098a61155a1
file: dataset/adjust_bottle/aloha-agilex_clean_50.zip
expected bytes: 298659710
expected SHA256:
  5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e
target:
  /root/autodl-tmp/datasets/robotwin2/source/
  9dc9299c163db059931898a9f0852098a61155a1/
  dataset/adjust_bottle/aloha-agilex_clean_50.zip
```

密码继续只从既有用户附件中定位 `ssh -p` 后第一条非空值，注入当前 PowerShell
`SEETA_SSH_PASSWORD`；命令结束后立即删除变量，不打印、不写脚本或文档。执行：

```powershell
python local_scripts/remote_exec_autodl.py run `
  --command-file local_scripts/remote_rlt_20260729_prestage1_refresh.sh
```

观察边界：`2026-07-29T14:42:39+08:00`。结果：

- 身份仍为预期 AutoDL 容器/UID 0；两张 A800 80GB 均 `0 MiB/0%`；
- RAM available 778 GiB；`/root/autodl-tmp` 仍约 1.9 TiB 总量、1.2 TiB 已用、
  694 GiB 可用、63%；
- 进程匹配只有本次审计 shell，没有 Ray、RoboTwin、SFT/RLT/DSRL 训练；
- RLT HEAD/upstream 均为 `cfa55655...`、ahead/behind `0/0`、worktree clean；
- DSRL worktree clean；
- 两个约定目标路径均不存在；有界 `find` 也没有找到同名 ZIP，因此现场确认
  **clean-50 尚未下载**；
- 所有大小写 proxy 与 `HF_ENDPOINT` 均未设置；共享 π0 venv 已安装
  `huggingface_hub 0.36.2`，不需安装依赖；
- 本项没有创建服务器数据目录或修改服务器文件。

磁盘分层的新证据：

- `RLinf_fastwam_rlinf` 533.6 GiB 几乎全是 `logs/`：move-stapler GRPO formal
  188.5 GiB、DSRL formal 93.9 GiB、move-stapler PPO formal 80.8 GiB、DSRL smoke
  62.6 GiB、adjust-bottle GRPO formal 53.9 GiB，其余两个历史 smoke 各约 27 GiB；
- `RLinf` 149.3 GiB 中 logs 135.6 GiB、共享 `.venv` 13.7 GiB；logs 主要是
  2026-07-15 adjust-bottle OpenPI GRPO baseline 96.8 GiB，以及 2026-07-14/15
  PPO/GRPO smoke/formal；
- `RoboTwin` 111.8 GiB 中 `policy/Motus_old_20260618_111133` 78.9 GiB、
  `policy/ACT` 16.5 GiB、assets 15.5 GiB；不是本次 RLT 数据；
- `RLinf_wamppo_backup_20260714_step57_lastdcp40` 110.9 GiB 中 logs 56.0 GiB，
  另有四套约 13.6～13.9 GiB 的 venv/venv backup；这是 7 月 13～14 日
  adjust-bottle PPO/旧 Motus/LaWAM 迁移备份；
- `models` 90.7 GiB 中 Motus 60.1 GiB、Fast-WAM 23.1 GiB、当前 RLT 必需的
  π0 RoboTwin SFT checkpoint 7.5 GiB；
- `conda` 48.0 GiB 主要是五个独立旧环境；`cache` 27.5 GiB 主要是可再生的 uv/pip
  下载 cache，但 `cache/uv_python` 是共享 π0 venv 的解释器依赖，不能盲删；
- `RLinf_old_20260618_085536` 30.9 GiB 是 6 月旧 π0 PPO repo/logs/venv；
  `RoboTwin_RLinf` 15.5 GiB 几乎全为当前 RLinf 模拟器 assets；
  `backups` 13.7 GiB 是当前 π0 venv golden rollback。

本项只做审计和本地记账；下载的每条实际命令、PID、日志、校验和问题继续记入下一批。

## 26. A026：clean-50 锁定下载、独立校验与日志收尾修复

### 26.1 上传与启动

下载脚本的网络边界：

```bash
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1
```

没有启用 `/etc/network_turbo`，没有并行启动 `wget`/`aria2`/第二个 Hugging Face downloader，
没有安装依赖。共享 π0 venv 内的 `huggingface_hub 0.36.2` 直接调用：

```python
hf_hub_download(
    repo_id="TianxingChen/RoboTwin2.0",
    filename="dataset/adjust_bottle/aloha-agilex_clean_50.zip",
    repo_type="dataset",
    revision="9dc9299c163db059931898a9f0852098a61155a1",
    local_dir="/root/autodl-tmp/datasets/robotwin2/source/"
              "9dc9299c163db059931898a9f0852098a61155a1",
)
```

先通过 SFTP 上传无凭据脚本：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_clean50_download.sh `
  /root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.sh
```

再执行
`local_scripts/remote_rlt_20260729_start_clean50_download.sh`。远端启动：

```bash
chmod 700 /root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.sh
nohup bash /root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.sh \
  >/root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.log 2>&1 \
  </dev/null &
```

结果：

```text
start: 2026-07-29T14:45:21+08:00
PID: 602939
log: /root/autodl-tmp/tmp/rlt_clean50_download_20260729_v1.log
lock: /root/autodl-tmp/tmp/rlt_clean50_download.lock
```

脚本在创建目标前持有 `flock -n`；若目标已经存在但 hash 不同则 fail closed，不覆盖、改名或
删除旧文件。

### 26.2 第一次状态检查与独立校验

执行：

```powershell
python local_scripts/remote_exec_autodl.py run `
  --command-file local_scripts/remote_rlt_20260729_status_clean50_download.sh

python local_scripts/remote_exec_autodl.py run `
  --command-file local_scripts/remote_rlt_20260729_verify_clean50_download.sh
```

状态检查时间 `14:46:32+08:00`：下载 PID 已退出，目标存在。独立校验时间
`14:47:23+08:00`，没有依赖 producer 脚本的成功判断：

```text
path:
  /root/autodl-tmp/datasets/robotwin2/source/
  9dc9299c163db059931898a9f0852098a61155a1/
  dataset/adjust_bottle/aloha-agilex_clean_50.zip
bytes:
  298659710
SHA256:
  5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e
unzip -tqq:
  PASS
archive:
  207 files, 450331107 uncompressed bytes
  50 _traj_data/episode*.pkl
  50 episode video files
  50 instructions/episode*.json
```

下载树总实占 298,668,032 bytes；除目标 ZIP 外只有 Hugging Face 的 1-byte `.gitignore`
和 125-byte download metadata，没有残留 partial 大文件。数据盘仍显示约 694 GiB 可用、
63% used。

### 26.3 问题、诊断与窄修复

问题：v1 producer 日志已经记录正确的 `actual_size`、`actual_sha256`、ZIP 完整性和
archive listing，但缺少脚本末尾预期的 `SUCCESS` 行。数据独立校验全部通过，所以这是
**脚本收尾证据不完整**，不是数据损坏或下载失败。

初始怀疑是 `set -o pipefail` 下的：

```bash
unzip -Z1 "$target" | head -30
```

可能让上游收到 broken pipe。为消除这个风险，将它改成完整消费输入的：

```bash
unzip -Z1 "$target" | awk 'NR <= 30 {print}'
```

同时把“目标已存在且大小/hash 正确”的幂等路径补上 `unzip -tqq` 和终态
`SUCCESS`。不过随后单独复现旧 pipeline 的返回码为 0，因此不能把 SIGPIPE 声称为已证实
根因；v1 缺终态行的精确原因保持未证实，不继续为已通过独立校验的数据扩大调查。

把修正版作为新文件上传：

```text
/root/autodl-tmp/tmp/rlt_clean50_download_20260729_v2.sh
```

执行 `local_scripts/remote_rlt_20260729_close_clean50_download.sh`，保留 v1 日志不覆盖。
v2 检查到目标已正确存在，重新做大小、SHA256 和 ZIP 完整性校验，不重新下载，结果：

```text
producer_v2_rc=0
ALREADY_VALID
SUCCESS 2026-07-29T14:48:56+08:00
```

最终结论：clean-50 原始 ZIP 已锁定、完整下载且可复核；尚未解压、转换或用于训练。

### 26.4 解压前 archive 安全与结构检查

执行：

```powershell
python local_scripts/remote_exec_autodl.py run `
  --command-file local_scripts/remote_rlt_20260729_inspect_clean50_archive.sh
```

脚本把完整 archive listing 保存到：

```text
/root/autodl-tmp/tmp/rlt_clean50_archive_listing_20260729.txt
```

只读检查时间 `2026-07-29T14:54:26+08:00`。结果：

- 207 个路径全部位于唯一顶层目录 `aloha-agilex_clean_50/`；
- 绝对路径、Windows drive 路径、反斜杠和 `../` traversal 数均为 0；
- 50 个 `_traj_data/episode*.pkl`；
- 50 个 `data/episode*.hdf5`；
- 50 个 `video/episode*.mp4`；
- 50 个 `instructions/episode*.json`；
- 另有 `seed.txt` 和 `scene_info.json`。

这把先前简单的“50 trajectory/video/instruction”计数补全为 50 组
`pkl+hdf5+mp4+instruction`。ZIP 内已有 `data/episode*.hdf5` 不代表可以跳过官方
RoboTwin→Aloha 转换：是否已是 Stage 1 converter 接受的 schema，仍要在单 episode
解压后按 key/shape/时序事实判断，不能仅凭扩展名决定。

## 27. A027：smoke 前逐参数复核与事实纠正

用户逐项询问 Stage 1/Stage 2 的并发、ManiSkill 来源、replay、UTD、warm-up、cap、
actor loss、H/C/D、route、743M 模块和训练效果。本项没有启动项目或修改服务器代码；
只读对照：

```text
examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml
examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
examples/sft/config/robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke.yaml
examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml
examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml
examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_a800_2gpu_smoke.yaml
rlinf/models/embodiment/modules/rlt_token_transformer.py
rlinf/models/embodiment/openpi/openpi_action_model.py
rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py
rlinf/workers/actor/fsdp_sac_policy_worker.py
rlinf/data/replay_buffer.py
```

同时复核 RLT 论文 v2、RLinf RLT 官方文档、RoboTwin π0 数据文档与 ManiSkill 官方说明。
新增/纠正结论：

1. Stage 1 smoke 已保留正式两卡/两 rank、同模型/loss/no-shard/optimizer/save 主链；
   micro1/global2 能测真实分布式同步，但不能证明 formal micro16/global32 的 activation
   memory。推荐同一 smoke 增加一条 formal-batch 单步 gate，不改模型。
2. “官方 token 模块大小”必须分三层：PI 论文未公开；RLinf ManiSkill 代码严格为
   744,667,136 参数；本项目真实 probe 为 743,094,272。差异只有约 0.21%，743M 不是
   RoboTwin 意外膨胀，也不能称为 PI 官方大小。
3. token-only Stage 1 的 `rlt_loss` 是 frozen π0 image-prefix embeddings 的 masked MSE，
   **不是 action reconstruction**。action 字段仍需通过 loader/schema，但
   `rlt_train_vla=false` 时不进入梯度。
4. RLinf decoder 是无 causal mask 的 parallel reconstruction；RLT 论文写的是
   autoregressive decoder。这是既有 RLinf 复现差异，首版不重写。
5. ManiSkill YAML 的 `update_epoch=5/train_every_transitions=5` 按当前 worker 公式实际为
   macro-UTD1，`critic_actor_ratio=4`；本项目 `5/1` 为 UTD5、ratio2。后者与论文明确的
   UTD5 和 2 critic : 1 actor 一致，但不是 ManiSkill YAML 原值。
6. 500 rows/rank + 5k critic updates 是预算 heuristic，不是把 ManiSkill
   10k/30k 按 64→4 env 严格等比例缩放。
7. cap400 是每个 collect→train 周期的上限。首次 floor5k 留约4600 pending；满长失败周期
   80 new macro rows × UTD5=400，pending 会保持而非清空。该设计避免首次 5k 连续 burst，
   不承诺 catch-up。
8. replay 15k/rank 是 compact tensor cache/recent sampling window，不是删除 lifetime
   index 的 hard capacity；只有 bounded pilot 每 rank 不超过15k时才是全量 replay。
9. clean-50 在 token-only Stage 1 提供的是成功轨迹 observation/prompt frame 分布；
   首版除 fixed-prefix loss 下降、π0 delta0、reload 一致外，应增加 true-z 对
   shuffled/zero-z 的 post-hoc reconstruction 对照，防止 decoder 忽略瓶颈。
10. 当前没有真实 Stage 1 step timing。只能报告
    `2000 × steady_step_time + startup/save`；2/5/10/20 秒每 step 分别约
    1.1/2.8/5.6/11.1 小时。

据此使用 `apply_patch` 更新：

```text
docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md
HANDOFF.md
```

改动只修正来源标签、数据动态状态、术语、两阶段 smoke 建议与磁盘归属；没有修改当前
Stage 2 candidate 数值，也没有为 unresolved 的 UTD1/UTD5 选择偷偷切换配置。现有 Stage 1
默认 dataset path 与版本化 canonical 规划不一致，明确留到 converter 产出路径冻结后再改
source YAML、重新 compose/hash，不建立兼容 symlink。

## 28. A028：文档收口与跨专题动态状态校正

在准备交付前执行本地文档只读检查：

```powershell
rg -n -C 3 "ZIP尚未下载|RLT.*下载|clean-50" HANDOFF.md
Get-Content -LiteralPath `
  "docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md" `
  -Encoding utf8 | Select-Object -Last 36
```

发现根 `HANDOFF.md` 的 RLT 段已经记录 ZIP 下载完成，但 QAM 段仍保留“ZIP 尚未下载”的
旧动态状态。该 ZIP 是 RLT/QAM 共用的只读 raw source，因此这不是 QAM 设计变化，而是根
交接页同一物理对象的状态矛盾。使用 `apply_patch` 只更新 QAM 的动态事实：

- ZIP 已由 RLT owner 下载到版本化 source 目录；
- size、SHA-256、ZIP 完整性和 archive 路径安全检查已通过；
- QAM 本身没有解压、转换或创建 sidecar；
- 单 episode schema/mask 合同仍未验收。

保留本账本 A021 等带时间戳的“当时尚未下载”记录；它们是实施历史，不能改写成事后状态。
本次收口没有修改算法代码、source/resolved YAML、服务器环境、解压目录或训练产物，也没有
启动 compose、import、测试、smoke 或训练。

## 29. A029：文档证据同步前检查

服务器同步前先上传并执行只读脚本：

```powershell
python local_scripts/remote_exec_autodl.py run `
  --command-file local_scripts/remote_rlt_20260729_docs_preflight.sh
```

`2026-07-29T15:06:06+08:00` 的结果：

- worktree 为 `/root/autodl-tmp/RLinf_rlt_pi0_robotwin`；
- branch 为 `codex/rlt-pi0-robotwin`；
- HEAD 为 `cfa556550efa7da1779a0d29c3a34b00a7f17ed8`；
- upstream ahead/behind 为 `0/0`；
- `git status --short` 无输出；
- `HANDOFF.md`、专题索引、配置复核和本账本四个目标文件全部已跟踪。

之后通过四次 `put` 把同名本地文档上传到该独立 worktree；没有上传 checkpoint、数据 ZIP、
运行产物或本地 operational scripts。上传后第一次误写成：

```powershell
python local_scripts/remote_exec_autodl.py run `
  --command "bash /root/autodl-tmp/tmp/rlt_docs_precommit_20260729.sh"
```

helper 的 `run` 没有 `--command` option，`argparse` 将此前缀匹配成
`--command-file`，因此返回本地 `FileNotFoundError`；远程预提交脚本没有开始执行，
已上传的四份文档未被回滚或覆盖。核对 `run --help` 后改为它定义的位置参数：

```powershell
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_docs_precommit_20260729.sh"
```

复测通过：

- 临时 alternate index 的 `git diff --cached --check HEAD` 无输出；
- 只有四份预期 Markdown 为 modified；
- diff 为 706 insertions / 65 deletions，最大变化是细粒度实施账本和逐参数配置复核；
- 没有新文件越过 1 MiB guard；
- credential pattern scan 为 `CREDENTIAL_PATTERN_SCAN_OK`；
- 没有改动实际 Git index。

密码仍只从用户既有附件中定位后注入当前 PowerShell 进程的
`SEETA_SSH_PASSWORD`，每组命令在 `finally` 中删除；没有打印、写入脚本、文档或服务器。

## 30. A030：文档提交、推送与下载阶段停点

将提交逻辑写入
`local_scripts/remote_rlt_20260729_docs_commit.sh`，上传为服务器临时脚本后执行。脚本先锁定
branch，要求 `git status --short` 的路径集合严格等于四份预期 Markdown，再执行：

```bash
git add -- \
  HANDOFF.md \
  docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md \
  docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
git diff --cached --check
git commit -m "docs(rlt): lock clean50 and review pre-smoke config"
timeout 30s git push personal codex/rlt-pi0-robotwin
git status --short
git rev-list --left-right --count HEAD...@{upstream}
```

结果：

```text
[codex/rlt-pi0-robotwin e02ce9c4] docs(rlt): lock clean50 and review pre-smoke config
4 files changed, 754 insertions(+), 65 deletions(-)
COMMIT e02ce9c495f58dc6eaae6d8828c3810703ae56c0
PUSH_OK
0  0
```

remote 为 `personal=https://github.com/Yutenji-Nyamu/rlinf_fastwam.git`；推送范围为
`cfa55655..e02ce9c4` 的独立 `codex/rlt-pi0-robotwin` 分支。commit 后
`git status --short` 无输出，upstream ahead/behind 为 `0/0`。

本节自身随后作为 ledger-only 收尾提交同步；该收尾提交的最终 hash 由 Git history 和本轮
交接给出，避免在提交内容中制造不可满足的自引用 hash。下载阶段的最终停点不变：

- raw ZIP 已下载、固定 revision/hash，并通过 ZIP/path/成员计数验收；
- 未解压、未转换、未创建 canonical dataset；
- 未修改 source/resolved config；
- 未运行 Stage 1/Stage 2 smoke 或训练；
- 下一次服务器写操作必须先获得“单 episode 解压/converter 合同 + Stage 1 S1-A/S1-B”
  明确批准。

## 31. A031：Stage 1 smoke 授权、范围与上下文重载

用户在 2026-07-29 明确授权自行完成 Stage 1 的全部必要 smoke，并要求：

- 继续使用细粒度流水账，记录每条指令、结果、问题、原因、修复和复测；
- 监控 GPU 显存、主机内存和进程资源，作为后续并行度与 batch 调整依据；
- smoke 保持克制，只做简洁且会改变决策的必要检查；
- 回答 demonstrations/rollout 数据、高层论文/ManiSkill 对齐、Stage 1 并行和 Stage 2
  参数来源问题；
- 只讨论 `RLinf`/`RoboTwin` 历史空间是否可清，不删除、移动或覆盖。

本轮授权边界解释为：

1. 可以选择性解压并完成单 episode 数据/schema/converter 合同；
2. 可以创建全新、版本化 canonical smoke 数据目录和 manifest；
3. 可以 compose 最终 S1-A/S1-B resolved config，启动两卡 Stage 1 smoke；
4. 可以执行最小保存/新进程重载、trainable-set 和参数更新链检查；
5. 遇到 blocker 可以做一个由证据直接支持的窄修复并复测；
6. 不自动扩展到 2,000-step Stage 1 正式训练、Stage 2、超参 sweep、依赖安装或磁盘清理。

按工作区规则重新完整读取：

```text
PROJECT_CONTEXT.md
HANDOFF.md
docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md
evidence/IMPLEMENTATION_LOG.md 的 Stage 1 implementation/probe/config/download 批次
```

同时只读搜索 Memory 注册表中 RLT、Stage 1、clean-50、worktree 和 smoke 规则；动态
进程、资源、Git、数据和日志仍只以本轮服务器现场为真。

最小运行序列冻结为：

```text
现场刷新
-> 单 episode schema/converter 合同
-> S1-A：2 ranks，micro/global 1/2，2 optimizer steps，save/reload
-> S1-B：2 ranks，formal micro/global 16/32，1 optimizer step
-> 终态资源/Git/产物检查
```

本轮不增加 data-scaling、held-out eval、模型结构消融、单卡替代、Stage 2 或长训。

## 32. A032：Stage 1 smoke 前服务器现场刷新

新增并上传只读脚本：

```text
local_scripts/remote_rlt_20260729_stage1_refresh.sh
/root/autodl-tmp/tmp/rlt_stage1_refresh_20260729.sh
```

执行：

```powershell
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_refresh_20260729.sh"
```

密码只注入该 PowerShell 进程的 `SEETA_SSH_PASSWORD`，在 `finally` 删除。观察时间
`2026-07-29T15:37:29+08:00`：

- 两张 A800-SXM4-80GB 均为 `0 MiB/0%`；
- host RAM available `774 GiB`，无 swap；
- `/root/autodl-tmp` 可用 `694 GiB`、63% used；
- 进程匹配只有本次刷新 shell，无 Ray、RoboTwin、SFT、RLT、DSRL 或 torchrun；
- RLT worktree branch 为 `codex/rlt-pi0-robotwin`，HEAD
  `e4127fd49e38362161eac08c551a7a98c11e9802`，ahead/behind `0/0`，clean；
- DSRL worktree clean；
- ZIP 大小、SHA256 和 `unzip -tqq` 再次通过；
- versioned raw/intermediate/canonical、S1-A/S1-B output 六个目标均不存在，不会覆盖旧 run；
- 共享解释器解析到
  `/root/autodl-tmp/cache/uv_python/install/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11`；
- 运行时已有 `torch 2.6.0+cu124`、`datasets 3.6.0`、`lerobot 0.1.0`，没有安装依赖。

刷新脚本只在 `/root/autodl-tmp/checkpoints` 搜索 `norm_stats.json`，该项无输出；实际 Stage 1
config 指向 `/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/...`，后续
packet 用精确路径单独校验，不把这次范围过窄的 `find` 解释为 checkpoint 缺失。

## 33. A033：clean-50 单 episode 选择性解压与两级官方 converter 合同

为避免把全部 50 条数据提前解包，只处理 `adjust_bottle` 的 `episode0`。先只读检查
RoboTwin converter 源码：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_converter_inventory.sh `
  /root/autodl-tmp/tmp/rlt_converter_inventory_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_converter_inventory_20260729.sh"

python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_converter_source.sh `
  /root/autodl-tmp/tmp/rlt_converter_source_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_converter_source_20260729.sh"
```

确认两级官方路径为：

```text
/root/autodl-tmp/RoboTwin/policy/pi0/scripts/process_data.py
/root/autodl-tmp/RoboTwin/policy/pi0/scripts/convert_aloha_data_to_lerobot_robotwin.py
```

源码 SHA-256 分别以 `b462...` 和 `b8f...` 开头。第二级脚本在目标目录已经存在时会执行
递归删除，且内部使用未排序遍历和随机 instruction；因此本次只允许写入预检已确认不存在的
版本化目录，并由外层固定 NumPy seed。没有修改 RoboTwin 工作树。

执行选择性解压：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_extract_contract_ep0.sh `
  /root/autodl-tmp/tmp/rlt_extract_contract_ep0_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_extract_contract_ep0_20260729.sh"
```

源为已固定 SHA 的 clean-50 ZIP；新目标为：

```text
/root/autodl-tmp/datasets/robotwin2/raw/\
9dc9299c.../adjust_bottle/contract_ep0
```

结果：

- raw episode 长度 `T=140`；
- 左、右臂均为 `(140, 6)`，左右 gripper 均为 `(140,)`；
- head/left/right 三路压缩图像均可解码为 `(240, 320, 3)`；
- instruction 候选数为 `100`；
- HDF5 为 `9,104,360` bytes，SHA-256 以 `6e5a49e2...` 开头；
- 目标总大小约 `8.8 MiB`。

随后按官方 π0 数据链运行 raw RoboTwin → Aloha：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_process_contract_ep0.sh `
  /root/autodl-tmp/tmp/rlt_process_contract_ep0_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_process_contract_ep0_20260729.sh"
```

新目标为：

```text
/root/autodl-tmp/datasets/robotwin2/intermediate/\
9dc9299c.../adjust_bottle/pi0-aloha-clean50-contract-ep0-v1
```

结果为 `139` rows、`state/action` 均为 `[139, 14]`。逐元素合同检查：

```text
processed qpos[t] == raw state[t]       max_abs_error = 0
processed action[t] == raw state[t + 1] max_abs_error = 0
```

三路处理后图像均为 `(480, 640, 3)`，100 条 instruction 候选仍保留；输出 HDF5 为
`14,231,682` bytes，SHA-256 以 `ee5938...` 开头。

最后运行 Aloha → LeRobot：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_lerobot_contract_ep0.sh `
  /root/autodl-tmp/tmp/rlt_lerobot_contract_ep0_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_lerobot_contract_ep0_20260729.sh"
```

新 canonical smoke 数据集为：

```text
/root/autodl-tmp/datasets/robotwin2/canonical/\
pi0-aloha-clean50-contract-ep0-v1
```

结果：

- `1 episode / 139 frames / 50 FPS`；
- 包含 action、state、head/left/right image、task 等列；
- 固定 seed 后选择的 prompt 为
  `Lift the smooth green plastic bottle head-up from the table.`；
- parquet 为 `35,036,057` bytes，SHA-256 以 `382cb...` 开头；
- canonical 目录约 `34 MiB`。

本批没有展开其余 49 条 episode，也没有复制样本凑 400。这里验证的是格式与时序合同，
不把单条数据称为 Stage 1 正式训练集。

## 34. A034：OpenPI 分布式 loader 合同、一次环境错误与窄修复

先检查 RLinf wrapper 与已安装 OpenPI loader 源码，再用同一个 canonical episode 模拟
S1-A 和 S1-B 两种 batch：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_openpi_loader_source.sh `
  /root/autodl-tmp/tmp/rlt_openpi_loader_source_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_openpi_loader_source_20260729.sh"

python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_loader_contract.py `
  /root/autodl-tmp/tmp/rlt_loader_contract_20260729.py
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_loader_contract.sh `
  /root/autodl-tmp/tmp/rlt_loader_contract_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_loader_contract_20260729.sh"
```

源码事实是：RLinf 传入 `micro_batch_size * world_size`；OpenPI 检测到
`torch.distributed` 后构造 `DistributedSampler`，再把传入 batch 除以 world size。因此：

| 配置 | loader 传入值 | rank 数 | 每 rank 实际 batch | accumulation |
|---|---:|---:|---:|---:|
| S1-A | 2 | 2 | 1 | 1 |
| S1-B | 32 | 2 | 16 | 1 |

第一次 loader 合同命令失败在取 batch 之前：

```text
RuntimeError: Unable to initialize backend cuda ...
Set JAX_PLATFORMS=cpu
```

原因是脚本清空了 `CUDA_VISIBLE_DEVICES` 以做 CPU-only loader 检查，但 JAX 仍尝试初始化
CUDA plugin；这不是数据/schema 错误，也没有启动训练。窄修复只是在合同脚本增加：

```bash
export JAX_PLATFORMS=cpu
```

没有安装、升级或修改项目依赖。相同命令复测成功：

- 两个模拟 rank 均使用 `DistributedSampler`，dataset length 均为 `139`；
- S1-A 每 rank 图像为 `[1, 3, 224, 224]`，state `[1, 32]`，
  actions `[1, 50, 32]`，token/mask `[1, 48]`；
- S1-B 每 rank 图像为 `[16, 3, 224, 224]`，state `[16, 32]`，
  actions `[16, 50, 32]`，token/mask `[16, 48]`；
- 三路图像、state、action 均 finite；
- normalization stats 从
  `/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/`
  `physical-intelligence/robotwin/norm_stats.json` 加载，文件 SHA-256 为
  `649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a`；
- 整个双 batch 合同约 `73.9 s`，没有占用 GPU。

`no_shard` 在这里表示两张 GPU 各保留完整模型/optimizer 副本并同步 data-parallel 梯度，
不是参数分片。RLT module 显式使用 BF16，reconstruction MSE 在 FP32 计算；YAML 中
`precision: null` 不等于 token module 以 FP32 保存。当前自定义 Stage 1 forward 会关闭
gradient checkpointing，因此不把它预设为显存不足时的兜底。若 S1-B OOM，只允许按证据
做一次窄调整：先将 per-rank micro batch 从 16 降到 8，同时用 accumulation 2 保持
global batch 32；不做 batch sweep。

## 35. A035：S1-A/S1-B 最终 resolved packet

新增并上传：

```text
local_scripts/remote_rlt_20260729_stage1_final_packet.sh
/root/autodl-tmp/tmp/rlt_stage1_final_packet_20260729.sh
```

执行：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_stage1_final_packet.sh `
  /root/autodl-tmp/tmp/rlt_stage1_final_packet_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_final_packet_20260729.sh"
```

脚本先锁定独立 branch、clean worktree、所有输入存在以及两个 run 目标均不存在，再通过
Hydra `--cfg job --resolve` 生成：

```text
/root/autodl-tmp/experiment_exports/rlt_stage1_smoke_20260729_v1/
  s1a_resolved.yaml
  s1b_resolved.yaml
  exact_commands.txt
```

观察时间为 `2026-07-29T15:57:11+08:00`。最终关键值：

| 项 | S1-A | S1-B |
|---|---:|---:|
| ranks | 2 | 2 |
| per-rank micro batch | 1 | 16 |
| global batch | 2 | 32 |
| accumulation | 1 | 1 |
| optimizer steps | 2 | 1 |
| optimizer schedule length | 2 | 2000 |
| LR warm-up | 1 | 100 |
| save interval | 2 | -1 |

两份配置都断言：

- placement 为 GPU `0-1`；
- dataset 为
  `/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-contract-ep0-v1`；
- model 为
  `/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle`；
- norm stats 为该 checkpoint 下的
  `physical-intelligence/robotwin/norm_stats.json`；
- `rlt_train_vla=false`；
- `sharding_strategy=no_shard`。

resolved SHA-256：

```text
S1-A 2aa7400eb1355bcb1b84cdb431c6110f6f6bde378861379dbffce340befae49d
S1-B 5b984a6865df3d0f2aed8e957a4ba8f7f040ef1010ce606b5150714f9811723a
commands 1ea9f65590f211c027641fadc98fea142a0b4cd64b2da3a8cd7472cf1d22dc2b
```

停止条件也随 packet 固定：S1-A 在 step 2 保存并由新进程 reload-only 成功后停止；S1-B
只跑一个 optimizer step、不保存；任一 rank 失败、loss 非有限、OOM 或出现非 RLT
trainable parameters 都停止。S1-B 只有发生 OOM 时才允许一次
`micro16 → micro8 + accumulation2` 窄重试。

## 36. A036：S1-A 两卡两步、checkpoint 与 reload-only

新增并上传资源监控、启动和状态脚本：

```text
local_scripts/remote_rlt_20260729_resource_monitor.sh
local_scripts/remote_rlt_20260729_start_s1a.sh
local_scripts/remote_rlt_20260729_status_s1a.sh
/root/autodl-tmp/tmp/rlt_stage1_resource_monitor_20260729.sh
/root/autodl-tmp/tmp/rlt_stage1_start_s1a_20260729.sh
/root/autodl-tmp/tmp/rlt_stage1_status_s1a_20260729.sh
```

启动命令：

```powershell
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_start_s1a_20260729.sh"
```

远程脚本在预检 branch、clean tree、输入存在、输出不存在以及无相关训练/Ray 进程后，用
`nohup` 启动最终 packet 中的精确命令；外层设置 30 分钟硬超时。资源 monitor 每秒记录
GPU 0/1 显存与利用率、主机 available RAM、cgroup memory、相关进程 RSS 和 GPU compute
process 数。driver PID 为 `611074`，monitor PID 为 `611075`。

轮询只调用状态脚本，没有向运行中的训练写入控制信号。观察到：

- 两个 worker 的 `local_batch_size` 均为 `1`；
- 两个 rank 都从固定 checkpoint 路径加载 norm stats；
- `NO_SHARD` warning 是 PyTorch 对该旧接口的弃用提示，不是本轮失败；
- FSDP AMP 关闭，但 RLT module 自身仍按实现显式使用 BF16；本轮不借 warning 改动数值
  精度；
- gradient checkpointing 按 resolved config 关闭；
- 没有 traceback、OOM、RayTaskError 或非有限 metric。

训练以 exit code `0` 完成。去重后的两步 metric：

| step | loss / rlt_loss | vla_loss | grad norm | LR |
|---:|---:|---:|---:|---:|
| 1 | 5.15 | 0 | 2.42 | `2.5e-5` |
| 2 | 5.21 | 0 | 2.33 | `6.25e-8` |

两步只是执行合同；不同 shuffled batch 上 `5.15 → 5.21` 不应用来判断收敛。第一步显示
`time/training≈18.6 s`；第二步日志的 training 子计时受异步/缓存影响而显示约 `0.172 s`，
同时外层 step 包含 checkpoint 保存约 `34.1 s`，因此不把第二个数字外推为正式吞吐。
本轮日志和先前 probe 证明 optimizer/trainable names 只含 `rlt_module.*`，但没有保存逐参数
π0 before/after snapshot，因此不声称已经数值验证 π0 delta=0。

checkpoint 位于：

```text
/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1/s1a/
robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1/
checkpoints/global_step_2
```

包含：

| 文件 | bytes |
|---|---:|
| `actor/dcp_checkpoint/.metadata` | 542,278 |
| `actor/dcp_checkpoint/__0_0.distcp` | 6,262,795,292 |
| `actor/dcp_checkpoint/__1_0.distcp` | 6,261,725,940 |
| `actor/model_state_dict/full_weights.pt` | 9,551,212,074 |

合计 `20.56 GiB`，`du -sh` 为 `21G`。

随后新增并上传：

```text
local_scripts/remote_rlt_20260729_start_s1a_reload.sh
local_scripts/remote_rlt_20260729_status_s1a_reload.sh
```

以相同配置、新 Ray/worker 进程和：

```text
+runner.resume_dir=<上述 global_step_2>
runner.max_steps=2
```

启动 reload-only。driver PID 为 `621054`；exit code 为 `0`。resolved config 日志记录精确
`resume_dir`，新进程初始化后进度直接显示 `2/2 [00:00<?, ?it/s]`，没有执行第三步、没有
写第二份 checkpoint。这证明该 checkpoint 可由同 world size 的新进程加载模型、optimizer
和 scheduler，并满足本轮最小 endpoint 合同；不声称 RNG bitwise、跨 world size 或
OpenPI dataloader cursor 恢复，也不声称同一 fixed prefix 在保存前后的 `z_rl/loss`
数值等价；后者留到正式 2k endpoint。

## 37. A037：S1-B 正式 batch 单步显存门

新增并上传：

```text
local_scripts/remote_rlt_20260729_start_s1b.sh
local_scripts/remote_rlt_20260729_status_s1b.sh
```

启动：

```powershell
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_start_s1b_20260729.sh"
```

driver PID 为 `629633`。两个 worker 都报告 `local_batch_size: 16`，即真正测试了
2 ranks × 16 = global batch 32，而不是复用 S1-A 的小 batch。单步以 exit code `0`
完成：

```text
time/step = 20.7 s
time/training = 20.7 s
loss = rlt_loss = 5.18
vla_loss = 0
grad_norm = 2.30
learning_rate = 2.5e-7
```

所有 metric 有限；没有 OOM、traceback 或 rank failure。`save_interval=-1` 生效，输出下
checkpoint 文件数为 `0`。因此不执行预案中的 micro8 重试，也不继续尝试更大 batch。

## 38. A038：资源汇总、一次 postcheck 断言错误与终态

新增并上传：

```text
local_scripts/remote_rlt_20260729_stage1_postcheck.sh
/root/autodl-tmp/tmp/rlt_stage1_postcheck_20260729.sh
```

第一次执行该汇总脚本返回非零，但三个 smoke 进程早已正常退出。失败文本是：

```text
RuntimeError: reload log lacks DCP load marker
```

原因是汇总脚本过度假设这版 Ray 会把内部固定短语 `loading DCP checkpoint` 转发到
driver log。只读 `grep` 显示实际日志保留的是精确 `resume_dir` 和 `Global Step 2/2`，
而非该内部短语。窄修复仅调整 postcheck 证据断言为：

1. reload resolved config 中必须出现精确 `global_step_2` 路径；
2. 新进程必须 exit `0`；
3. 进度必须直接为 `2/2`；
4. checkpoint DCP metadata/full weights 必须仍存在。

没有重跑 S1-A、reload 或 S1-B。相同 postcheck 复测通过，生成：

```text
/root/autodl-tmp/experiment_exports/rlt_stage1_smoke_20260729_v1/
stage1_postcheck.json
```

资源结果：

| run | wall | 每卡 GPU peak | 相关 RSS peak | host available 最低 | 从首样本最大下降 |
|---|---:|---:|---:|---:|---:|
| S1-A 两步+保存 | 155 s | 23,073 MiB | 39.44 GiB | 737.97 GiB | 37.82 GiB |
| reload-only | 122 s | 20,485 MiB | 39.15 GiB | 739.47 GiB | 36.05 GiB |
| S1-B micro16 一步 | 121 s | 26,447 MiB | 39.09 GiB | 741.10 GiB | 33.98 GiB |

三次运行每卡利用率采样峰值都到 `100%`，GPU compute process 峰值均为 `2`。S1-B 比
S1-A 每卡峰值只增加约 `3.3 GiB`，在 80GB A800 上留有约 53GB 余量；因此正式
micro16/global32 已通过容量门，不需要为了“吃满显存”扩 batch。cgroup current 含服务器
大文件 page cache，峰值约 240GiB，不等价于训练进程私有内存；决策以相关 RSS 和 host
available 为主。

终态 `2026-07-29T16:11:32+08:00`：

- 两卡均 `0 MiB/0%`，无相关训练、Ray、raylet 或 GCS 进程；
- host available `776 GiB`，无 swap；
- `/root/autodl-tmp` 余 `673 GiB`、64% used；相较 smoke 前约减少 21GiB，主要就是
  S1-A checkpoint；
- RLT worktree 仍为 `codex/rlt-pi0-robotwin`、HEAD
  `e4127fd49e38362161eac08c551a7a98c11e9802`、clean、upstream `0/0`；
- S1-A/S1-B resolved SHA 保持不变；
- `stage1_postcheck.json` SHA-256 为
  `8f84e8c5297f1ea8bd6eb55fa6b8c19bd6eea31be66d0a0d8f12fd8c41870acd`；
- DCP metadata SHA-256 为
  `09a51c2530d095d838b41eb928729daf15b7d82f233e0e247153927cdf9d590d`。

## 39. A039：历史磁盘只读审计与两次命令问题

本轮另行对用户点名的 `RLinf`、`RoboTwin/Motus/ACT` 做只读归属审计。使用的命令类型为：

```bash
du -x -B1 --max-depth=<n> <精确目录> | sort -n
find <精确 run/checkpoint 目录> -maxdepth <n> -type f -printf '%p\t%s\n'
git -C <精确 repo> status --short
git -C <精确 repo> log --oneline -- <相关路径>
stat <精确 checkpoint/manifest/config>
```

所有目标都写成 `/root/autodl-tmp/...` 下的精确路径；没有运行 `rm`、`mv`、`truncate`、
压缩、重命名或 checkpoint 转换。

审计中有两次无副作用问题：

1. 第一次命令在本地 helper 参数解析/引号层失败，未连接到服务器执行；
2. 一次只读 `awk` 汇总表达式语法错误，未写入文件；随后改为逐目录
   `du/find/stat` 统计并复测。

结论：

- `/root/autodl-tmp/RLinf/logs` 为 `135.562 GiB`，对应四个 2026-07-14/15 的
  adjust-bottle π0 PPO/GRPO smoke/formal；保留 GRPO step100、PPO step20 和小型元数据后，
  旧中间/smoke DCP 是约 `116.177 GiB` 的人工候选；
- `RoboTwin/policy/Motus_old_20260618_111133` 为 `78.824 GiB`，确属 TTS/VTTS 与
  OPD/GKD online distillation 历史；49 个中间 checkpoint 约 `59.253 GiB`，PNG 历史约
  `16.9 GiB`（约 `18.15 GB` decimal），均只列候选；
- `RoboTwin/policy/ACT` 为 `beat_block_hammer clean-50` imitation，不是 Motus；
  processed data `14.616 GiB` 可由仍存在的 raw 50 重建，四个中间 checkpoint 约
  `1.251 GiB`；
- `/root/autodl-tmp/RoboTwin/assets` 与 `/root/autodl-tmp/RoboTwin_RLinf/assets`
  各 `15.521 GiB`，是两个实体副本；
  当前 RLT 用后者，前者只有在旧 standalone Motus/ACT 路线退役后才讨论。

完整解释和保留/候选边界写入
[`../02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md`](../02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md)
§11。本轮没有执行任何清理。

## 40. A040：本地 evidence 收口与上下文索引更新

通过 helper 的 `get REMOTE LOCAL` 子命令，逐一下载小型、可审阅的 Stage 1 证据：

```text
两份服务器 source config
S1-A/S1-B resolved YAML
exact_commands.txt
stage1_postcheck.json
S1-A/reload/S1-B driver.log
S1-A/reload/S1-B resources.csv
```

本地目标为：

```text
docs/rlinf-robotwin-pi0-rltoken/evidence/stage1_smoke_20260729/
```

每个 `get` 均建立独立 Paramiko 连接、验证固定 host-key；密码只在当前 PowerShell
`SEETA_SSH_PASSWORD` 中存在并在 `finally` 删除。大 checkpoint 未下载。

下载后用 `Get-FileHash -Algorithm SHA256` 校验核心文件与服务器输出一致，并新增
`evidence/stage1_smoke_20260729/README.md` 作为证据索引。随后使用 `apply_patch` 更新：

```text
docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md
docs/rlinf-robotwin-pi0-rltoken/02_STAGE1_SMOKE_AND_METHOD_ALIGNMENT_20260729.md
docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
HANDOFF.md
```

`00` 保持唯一设计规范；`01` 保留配置来源和 Stage 2 pre-smoke 语义；`02` 集中回答本轮
专家数据、50 条、Stage 1 结果、参数来源、调用流和磁盘问题；原始 runtime evidence
不复制进正文。根 `HANDOFF.md` 的 RLT 行已从“smoke 前”改为“Stage 1 smoke 已通过，
停在 full clean-50 2k endpoint 前”。

## 41. A041：服务器文档同步、预提交 QA 与证据口径收窄

第一次尝试把 preflight 写成内联 PowerShell/remote shell 字符串时，本地 quoting 解析失败；
命令没有连接服务器，也没有远端写入。随后改用版本化脚本：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_stage1_docs_preflight.sh `
  /root/autodl-tmp/tmp/rlt_stage1_docs_preflight_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_docs_preflight_20260729.sh"
```

`2026-07-29 16:29` preflight 确认 branch
`codex/rlt-pi0-robotwin`、HEAD `e4127fd49e38362161eac08c551a7a98c11e9802`、upstream
`0/0`，且拟新增 evidence 目录不存在。随后逐文件使用 helper `put LOCAL REMOTE` 上传第一版
18 个 docs/evidence 文件。

使用 alternate index 的预提交审查第一次只因三份 Ray/RLinf 原始 driver log 有行尾空格
而失败。没有改训练文本、metric 或控制字符；本地仅机械删除：

```text
[ \t]+(?=\r?$)
```

匹配的行尾空白，然后重传：

```text
runtime/s1a_driver.log
runtime/s1a_reload_driver.log
runtime/s1b_driver.log
```

独立文档 QA 随后指出并修正了这些口径问题：

1. smoke 只证明 trainable-set 为 `rlt_module.*`，没有逐参数 π0 delta=0；
2. reload-only 只证明同 world-size checkpoint 可加载，没有 fixed-prefix `z_rl/loss`
   保存前后等价；
3. 原 `exact_commands.txt` 缺 reload/status/postcheck 细节，故保持原文件/hash不变，新增
   [`stage1_smoke_20260729/exact_commands_addendum.md`](stage1_smoke_20260729/exact_commands_addendum.md)；
4. 磁盘审计 A–P exact commands 单列为
   [`DISK_AUDIT_COMMANDS_20260729.md`](DISK_AUDIT_COMMANDS_20260729.md)；
5. 修正 assets 绝对路径、GiB/GB、reload-only 无训练 metric 和 primitive $\gamma$ 术语。

本节只记录文档/证据同步；没有启动训练、Ray、RoboTwin 或 simulator。

## 42. A042：LR scheduler 根因、最窄配置修复与 CPU contract

文档 QA 注意到 smoke-time source/resolved 均声明：

```text
lr=2.5e-5
min_lr=2.5e-6
```

但 S1-A step 2 日志为 `train/learning_rate=6.25e-8`。这不会推翻 forward/backward/save/load
或 micro16 容量结论，却会阻塞正式 2k 的 schedule 语义。

第一次只读源码搜索把带 `|` 的 regex 直接经过 PowerShell native argv 传递；服务器没有
`rg`，且 shell 把 regex 中的 `|` 解释为管道，产生 `rg: command not found` 等只读错误。
没有写入。随后上传并执行：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_lr_scheduler_audit.sh `
  /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_audit_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_audit_20260729.sh"
```

源码链闭合为：

```text
robotwin_rlt_stage1_sft_openpi.yaml
-> fsdp_model_manager.py build_optimizer:
   param group lr=2.5e-5，但 AdamW constructor 没有顶层 lr
-> optimizer.defaults["lr"]=1e-3
-> fsdp/utils.py get_lr_scheduler
-> Transformers 4.53.2:
   min_lr_rate = min_lr / optimizer.defaults["lr"] = .0025
-> real floor = 2.5e-5 * .0025 = 6.25e-8
```

将服务器 source config 下载到 `C:\tmp` 的第一次本地 `get` 因 sandbox 用户无该路径写权限
失败；服务器未改。改为 workspace 内
`local_scripts/remote_commands/robotwin_rlt_stage1_sft_openpi.formal_current.yaml` 后下载成功，
原文件 SHA-256 为
`0fa01fa8c6f8624438a3d27288ecb848336cd2857599bc4b1a1d369dfc563cb3`。

最窄修改只在 RoboTwin RLT Stage 1 config 中：

```diff
- min_lr: 2.5e-6
+ min_lr_rate: 0.1
```

不修改通用 optimizer builder，避免波及其他 FSDP workload 和多 param-group 语义。一次内联
PowerShell preflight 又在本地 `$()` quoting 解析阶段失败、未连接；随后通过精确脚本：

```powershell
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_fix_preflight_20260729.sh"
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_commands/robotwin_rlt_stage1_sft_openpi.formal_current.yaml `
  /root/autodl-tmp/RLinf_rlt_pi0_robotwin/examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_fix_verify_20260729.sh"
```

preflight 锁定 branch/HEAD、旧 hash 和该 config 在上传前无 diff；上传后 `git diff --check`
通过，新 source SHA-256：

```text
8340ef4e953877de510da18548d0a69802104b7b2f8218698cd0fb586b49a8f2
```

最后运行无模型、无数据 batch、`CUDA_VISIBLE_DEVICES` 为空的 CPU scalar contract：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_lr_scheduler_contract.sh `
  /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_contract_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_contract_20260729.sh"
```

第一次 probe 的 Python exact-float dictionary equality 在旧 floor
`6.250000000000001e-08` 上失败；未生成 evidence。断言窄改为 `math.isclose` 后，同一
probe 通过：

```text
legacy smoke 0 -> 2.5e-5 -> 6.25e-8
fixed smoke  0 -> 2.5e-5 -> 2.5e-6
fixed formal step 0/1/100/1050/2000
             0/2.5e-7/2.5e-5/1.375e-5/2.5e-6
```

输出
[`stage1_smoke_20260729/lr_scheduler_contract.json`](stage1_smoke_20260729/lr_scheduler_contract.json)
SHA-256 为
`e68a7da1457e32538995f39b41f23a21b34d248e2ed3fa37f1177476e7c614df`。

RLinf 在 `optimizer.step()` 后调用 `scheduler.step()` 再记录 LR，因此 smoke 证据需精确
解释为：S1-A 第一次 update 用 LR 0、第二次用 `2.5e-5`，有一次非零更新；S1-B 唯一
update 用 LR 0，只证明正式 batch 的前反传/optimizer 链和容量，不证明参数 delta。无需
重跑两卡 smoke；正式 2k packet 必须 hard-fail absolute `min_lr` 或 scheduler contract
不一致。

## 43. A043：22 文件统一预提交审查

把 scheduler 修复、CPU contract、command addendum 和磁盘 exact-command 附件加入 expected
scope 后，重新上传本地文档与 review 脚本。alternate-index 审查精确锁定 22 个拟提交文件，
避免把服务器其他 dirty/untracked 内容误加入真实 index。

第一次统一审查发现
`00_INDEX_AND_IMPLEMENTATION_PLAN.md` 链接到服务器 worktree 不存在的
`../../PROJECT_CONTEXT.md`。这是文档链接问题，不是训练或 evidence 问题；只读 `ls` 确认
`AGENTS.md/HANDOFF.md` 存在而 `PROJECT_CONTEXT.md` 只属于本地工作区根。窄修复把它改为
plain-text 本地上下文入口，不把本地治理文件复制进服务器代码仓。

再次执行：

```powershell
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_docs_review_20260729.sh"
```

结果：

```text
UTF8_LINK_CREDENTIAL_JSON_OK files=22
PRECOMMIT_REVIEW_OK
```

检查内容包括：

- alternate index staged path 与 22-file expected set 完全相同；
- `git diff --cached --check`；
- 所有文本 strict UTF-8、单文件小于 1MiB；
- 所有相对 Markdown 链接存在；
- 无 password/API-key/bearer-like 内容；
- postcheck 与 LR scheduler JSON 机器断言；
- S1-A/S1-B/original-command/addendum/LR-contract/postcheck SHA 输出。

本次审查没有改变真实 Git index、没有启动训练、没有加载模型/GPU，也没有删除服务器文件。

## 44. A044：主提交推送与 17:03 最终现场

在真实 index 为空且 HEAD 仍为 `e4127fd49e38362161eac08c551a7a98c11e9802` 时，提交脚本：

1. 只 `git add -f` A043 审核通过的 22 个 expected paths；
2. 要求 staged path 集合与 expected 完全相同；
3. 要求 `git diff --cached --check` 通过；
4. 要求没有额外 unstaged tracked 或 untracked non-ignored 文件；
5. commit 后推送精确的 `personal/codex/rlt-pi0-robotwin`。

执行：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_stage1_commit_push.sh `
  /root/autodl-tmp/tmp/rlt_stage1_commit_push_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_commit_push_20260729.sh"
```

结果：

```text
commit c22ba19af0b6dfd130e289f02efd3a42ce5e938f
fix(rlt): validate Stage 1 smoke scheduler
22 files changed, 4102 insertions(+), 127 deletions(-)
push e4127fd4..c22ba19a
ahead/behind 0/0
remote head c22ba19af0b6dfd130e289f02efd3a42ce5e938f
```

随后上传并执行只读终态脚本：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_stage1_delivery_refresh.sh `
  /root/autodl-tmp/tmp/rlt_stage1_delivery_refresh_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_delivery_refresh_20260729.sh"
```

`2026-07-29T17:03:46+08:00`：

- RLT branch `codex/rlt-pi0-robotwin`，HEAD=remote=`c22ba19a...`，clean，upstream `0/0`；
- DSRL worktree 仍为 `codex/dsrl-pi0-robotwin` 且 clean；
- 两卡 `0 MiB/0%`，没有 SFT/Ray/RoboTwin 相关进程；
- host available `966 GiB`、无 swap；
- `/root/autodl-tmp` 可用 `673 GiB`、64% used；
- S1-A checkpoint 仍为 `21G`；
- 当前 formal source config hash `8340ef4e...f2`；
- Git evidence 和 export 的 LR contract hash 都为 `e68a7da1...14df`。

本节和根 `HANDOFF.md` 的 17:03 状态将作为 ledger-only 收尾提交；该收尾提交自身的 hash
由最终 Git history/交接给出，避免在提交内容中形成不可满足的自引用。
