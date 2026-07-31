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

## 45. A045：Stage 2 resume-fingerprint 审阅意见的当前源码复核

收尾时一份并行只读审阅意见称 `_rlt_resume_contract()` 可能没有纳入
`algorithm.bootstrap_type`。由于动态源码事实优先于审阅摘要，本轮没有据此直接修改代码，
而是对最终服务器 HEAD 做定点只读检查。

第一次把带引号的 `grep -A/-B` 命令直接作为 PowerShell native argv 传给 helper，在本地
argparse 阶段被拆成多余参数而失败；远端未执行、未写入。随后上传：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_resume_contract_inspect.sh `
  /root/autodl-tmp/tmp/rlt_resume_contract_inspect_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_resume_contract_inspect_20260729.sh"
```

当前
`rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py::_rlt_resume_contract()` 的
`optimization` payload 已明确包含：

```text
bootstrap_type
target_update_freq
target_update_type
actor_optim
critic_optim
global_batch_size
micro_batch_size
```

同时还包含 loss/Q aggregation、gamma/tau、update cadence、route、transition/replay、
feature/actor model、world size 和 syncer。故该审阅意见不适用于当前 HEAD；设计文档和
HANDOFF 中“bootstrap/optimizer/batch 已进入 resume fingerprint”的表述保持不变。
本次没有代码修改、没有启动 Stage 2 smoke，也没有扩大测试。

## 46. A046：正式 Stage 1 2k 启动授权与执行边界

2026-07-29，用户明确授权本轮完成：

1. 将锁定 revision 的 `adjust_bottle/clean_50` 剩余 49 条按已经通过的 converter 合同转为
   版本化 full clean-50 canonical 数据；
2. 生成 full dataset manifest，重新 compose/hash 正式配置；
3. 以两张 GPU、`micro_batch_size=16`、`global_batch_size=32`、固定 step 2000 endpoint
   启动 Stage 1；
4. 启动后只检查数据加载、连续训练 step、loss、GPU/RAM 与 checkpoint 路径等早期健康
   信号，不持续在线轮询；
5. 对旧 Motus 实验 checkpoint，以及 RLinf PPO/GRPO 旧 run 的获批 DCP 做精确、定向清理，
   保留 Motus 官方/base 权重以及 PPO baseline、GRPO formal 的最终 checkpoint，并保留
   smoke/formal 的轻量日志、配置和指标。

本轮不启动 Stage 2，不改变 RLT 高层语义，不做额外 batch/data-scaling sweep，不安装或复制
环境，不停止无关进程。所有动态状态先以新的服务器只读探针刷新；任何删除都必须先解析为
精确绝对路径并验证处在已授权实验根目录内，再执行并记录实际回收量与不可恢复边界。

## 47. A047：19:09 正式运行前现场探针

上传并执行只读脚本：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_formal_preflight.sh `
  /root/autodl-tmp/tmp/rlt_stage1_formal_preflight_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_formal_preflight_20260729.sh"
```

SSH 密码仍只从既有 attachment 提取到当前 PowerShell 进程的
`SEETA_SSH_PASSWORD`，两个 helper 调用结束后在 `finally` 中移除；没有输出或持久化密码。

第一次脚本于 `19:09:47+08:00` 执行到 RLinf 历史 run 时退出。原因是脚本把审计材料中的
时间戳简称误写成完整目录名，而现场目录实际还带实验 slug；失败前只有 `date/id/git/df/
free/nvidia-smi/find/sha256sum` 等只读命令，未清理、转换或启动训练。窄修复仅把四个 run
改为各自唯一的 `*<timestamp>*` glob，并断言恰好解析为四个目录。

`19:10:18+08:00` 复测通过，现场为：

- RLT `codex/rlt-pi0-robotwin@66dc388e...`，upstream `0/0`，输出为空即 clean；
- DSRL `codex/dsrl-pi0-robotwin@48a775db...`，输出为空即 clean；
- 两张 A800 80GB 均 `0 MiB/0%`，没有实际 SFT/Ray/RoboTwin 训练进程；
- `/root/autodl-tmp` 可用 `673G`、64% used；host available RAM `965GiB`；
- full canonical、formal experiment 和 formal export 三个目标均不存在；
- source ZIP 为 `298,659,710 B`，SHA-256
  `5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e`；
- formal source config、scheduler contract、norm stats 的 SHA-256 分别为
  `8340ef4e...a8f2`、`e68a7da1...14df`、`649ed92b...f6a`。

只读清单进一步确认：RLinf 将删除 12 个获批旧 DCP（两个 smoke、PPO step10、GRPO
step10–90），保留 PPO step20 与 GRPO step100；Motus 目标精确为两个
`Motus_old_20260618_111133/logs_single_*` 下共 51 个 `opd_checkpoints/turn_switch/*.pt`
实验输出。`policy/Motus` 下没有匹配的大权重文件；本轮仍只会触碰上述两个旧实验输出根，
不会扫描删除任何模型根或官方/base checkpoint。

## 48. A048：19:12 获批旧 checkpoint 定向清理

上传并执行：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_approved_cleanup.sh `
  /root/autodl-tmp/tmp/rlt_stage1_approved_cleanup_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_approved_cleanup_20260729.sh"
```

脚本在删除前 hard-fail 检查：

- 12 个 RLinf DCP 的精确绝对路径均存在且 `realpath` 位于对应
  `/root/autodl-tmp/RLinf/logs/*/checkpoints/global_step_*`；
- PPO step20 与 GRPO step100 两个保留 endpoint 存在且不在删除数组；
- Motus 只从两个精确
  `Motus_old_20260618_111133/logs_single_*/opd_checkpoints/turn_switch` 父目录解析
  `.pt`，数量必须恰为 51，且每个 `realpath` 仍在原父目录；
- 独立 evidence 根
  `/root/autodl-tmp/experiment_exports/rlt_pre_stage1_cleanup_20260729` 必须原先不存在。

按用户本轮最新、直接的“实验目录下的大 checkpoint 删除，只保留 Motus 原本官方
checkpoint”授权，51 个 OPD/GKD 实验 `.pt` 全部删除；没有把旧讨论中的“每 run 保留一个
endpoint”覆盖到这次更明确的新指令。Motus 官方/base 权重位于独立模型根，不在两个
`logs_single_*` 删除范围内；`logs_his` PNG、CSV/log、代码/config 均未触碰。

执行结果：

| 类别 | 删除数 | 现场记录 bytes | 约 GiB |
|---|---:|---:|---:|
| RLinf 旧 DCP | 12 个目录 | 124,744,056,832 | 116.18 |
| Motus OPD 实验 checkpoint | 51 个文件 | 66,219,777,050 | 61.67 |
| 合计 | 63 项 | 190,963,833,882 | 177.85 |

删除后逐项断言目标不存在；PPO step20 与 GRPO step100 分别仍约 9.7G；两个 Motus 实验
目录剩余 `.pt` 数为 0。`/root/autodl-tmp` 从 673G available、64% used 变为
851G available、54% used。

审计产物位于：

```text
/root/autodl-tmp/experiment_exports/rlt_pre_stage1_cleanup_20260729/
  summary.tsv
  deleted_rlt_dcp.tsv
  deleted_motus_checkpoints.tsv
  df_before.txt
  df_after.txt
  SHA256SUMS
```

其中两个删除清单 SHA-256 分别为
`49536ff4b3e2f0964bdc296821beb2b8d2970f417a770c9fea489f698abec066` 与
`d91e1aecae4af136351964ab5ff08d36ab997f95e2c16be043f85406a577e744`。
这些删除不可恢复，除非服务器另有外部快照；清单只保留审计元数据，不包含被删权重内容。

## 49. A049：full clean-50 转换脚本固化与 extract 两次窄失败

新增并上传四个相互独立的正式转换脚本，避免一个总脚本跨阶段覆盖失败证据：

```text
/root/autodl-tmp/tmp/rlt_extract_clean50_full_20260729_v1.sh
/root/autodl-tmp/tmp/rlt_process_clean50_full_20260729_v1.sh
/root/autodl-tmp/tmp/rlt_lerobot_clean50_full_20260729_v1.sh
/root/autodl-tmp/tmp/rlt_manifest_clean50_full_20260729_v1.sh
```

上传后先对四个文件统一执行 `bash -n`，语法检查通过；再逐阶段运行。设计边界是：raw、
intermediate、canonical 均只在同一文件系统的唯一 `mktemp` staging 中生成，完整校验后才
用 `mv -T` 提升；最终 target 只接受原先不存在的版本化路径；所有阶段共用 non-blocking
`flock`；LeRobot converter 的 `HF_LEROBOT_HOME` 只指向全新 staging parent，因此其内部
递归清理逻辑无法碰到最终 canonical。

第一次 extract 于 `19:19:15+08:00` 在创建 staging 前失败：

```text
ValueError: expected 207 files, got 202
```

原因是历史审计的 207 指 ZIP 总 entries，而脚本先过滤了目录；真实结构为 207 entries =
202 files + 5 directories，其中 202 files = 50×`pkl+hdf5+mp4+instruction` + `seed.txt`
+ `scene_info.json`。窄修复为分别断言 207 entries/202 files；未写 raw target。

第二次于 `19:19:54+08:00` 通过 archive 合同、完成 staging 解压后，在 episode 0 首帧检查
失败：

```text
ValueError: episode 0: bad head_camera frame 0: (240, 320, 3)
```

原因是把已经通过的 **intermediate** 480×640 输出尺寸误套给 raw HDF5；既有 ep0 合同的
raw 正确尺寸为 240×320。窄修复只把 raw 首末帧 gate 与 raw validation metadata 改为
`[240,320,3]`，intermediate/canonical 的 480×640 合同不变。失败 staging 精确为：

```text
/root/autodl-tmp/datasets/robotwin2/raw/9dc9299c163db059931898a9f0852098a61155a1/
adjust_bottle/.clean50-v1.extract.kOi5x9
```

final raw target 仍不存在；后续只会在验证该绝对 staging 位于上述 raw parent 且 final 不存在
后清除这个由本轮创建的失败临时目录，再执行修订版，不会清理任何其他 raw/canonical 数据。

## 50. A050：full clean-50 转换、manifest 与正式 global32 loader

修订版四个 wrapper 在服务器的最终 SHA-256 为：

```text
extract   ecb21f6ba8e4cbc56e9e0e95c1e6a1225ff50883d7b8c1e1330ece84392440d8
process   04249ea1d150d63d1a42f23f9b60a39529124927034deb3056a6bf55f97596bf
lerobot   bcd75635be00741d27ff7e5c6aa9b251f4499a06810b5c62c23e825e453deb97
manifest  55f0e2d4db42f53e4e028c5b235b6971bba46c9101824687260698acfc0da9c3
```

先精确清除了 A049 的 430M 失败 extract staging，再于
`19:23:00–19:23:08+08:00` 成功完成 raw：

```text
/root/autodl-tmp/datasets/robotwin2/raw/
  9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle/clean50-v1
```

50 个 episode 的四组文件、joint/gripper finite、三相机首末帧 240×320、instruction 均通过；
总有效训练 row 为 7,188，raw validation SHA-256 为
`09130eb8e6158960ade2367dc8e214146e98b842ae7109866c57d07d524437d2`。

第一次 raw→Aloha converter 已报告 0–49 全部成功，但 validation 在读取 gripper 时失败：

```text
ValueError: 2 indexing arguments for 1 dimensions
```

根因是对 h5py dataset 直接写 `dataset[:, None]`；此前 ep0 合同正确做法是先 `[:]` 读成
NumPy，再扩维。final intermediate 未提升。精确删除本轮 683M 失败 staging 后，仅把两侧
gripper 改为 `np.asarray(dataset[:]).reshape(-1,1)`，于
`19:24:34–19:25:07+08:00` 复测通过：

```text
/root/autodl-tmp/datasets/robotwin2/intermediate/
  9dc9299c163db059931898a9f0852098a61155a1/adjust_bottle/pi0-aloha-clean50-v1
```

官方一级 converter SHA-256 为
`b462918bf3f41f6d2fc30c3498381ac3cc7d8ce7a8bd6333fafb925e7d9d5590`；
50/50 episode、7,188 rows、三相机首末帧 480×640、14D state/action 均闭合，
`qpos-raw_t` 与 `action-raw_{t+1}` 全量最大绝对误差均为 0；intermediate validation
SHA-256 为 `a6990d6979706119979ca0409f3d14c6b5d45c58d7e84179e485ae01f35fbddd`。

`19:25:21–19:28:05+08:00`，在独立
`.pi0-aloha-clean50-v1.hfhome.BK76GA` 中运行已通过 ep0 的二级 converter：

```text
/root/autodl-tmp/RoboTwin/policy/pi0/examples/aloha_real/
  convert_aloha_data_to_lerobot_robotwin.py
SHA-256 b8f0829329e099b7246b3d6467cec3ea4d60767eedd219b825d0b7f26bb7c373
```

50 个 episode 全部完成后，canonical validation 证明 50 episodes、7,188 frames、50 FPS、
14D state/action finite、三相机列无缺失、episode/frame/timestamp 与 task metadata 闭合，
再提升到：

```text
/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
```

现场大小约 1.6G；canonical validation、`info.json`、`episodes.jsonl`、`tasks.jsonl` 的
SHA-256 分别为 `3be98ee9...2c73`、`9561367f...4a05`、`1e3ecae1...b611`、
`b8017646...aff7`。

dataset manifest 独立于训练 config/checkpoint，避免 hash 循环；它再次 hard-fail 固定 ZIP
size/hash，记录真实 HF source `TianxingChen/RoboTwin2.0@9dc929...`、filename、两级
converter/wrapper SHA、seed、RoboTwin HEAD/dirty、actual tasks/episode metadata、per-episode
row、metadata hash 和 55 个 canonical file 的排序 digest。路径与 SHA-256：

```text
/root/autodl-tmp/datasets/robotwin2/manifests/pi0-aloha-clean50-v1.json
12ce2ed68632e2b18cf96f52b717edec00bcebb6cc0a446f83da1670d81ef86c
```

最后只跑一次无 GPU 模型加载的正式 loader 合同：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_loader_contract.py `
  /root/autodl-tmp/tmp/rlt_loader_full_clean50_20260729.py
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_loader_full_clean50_global32_20260729.sh"
```

两 rank 都得到 `DistributedSampler`、local batch16、dataset length7,188、三相机
`[16,3,224,224]`、state `[16,32]`、action horizon `[16,50,32]` 和 tokenized prompt
`[16,48]`；这里 32 是 OpenPI pad 后宽度，canonical 原始合同仍是 14D。stats 明确从既有
checkpoint 加载，SHA-256 仍为 `649ed92b...f6a`。

## 51. A051：正式配置收口、提交与 2k 启动

只把 source config 的默认 dataset 从旧占位路径改为：

```text
/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1
```

其他参数不变；新 source SHA-256 为
`c293bc476ec7458c6bfc5c5c59393e48b286f3e12007f3039ccc282e30645a4c`。
运行前 compose 使用与实际训练完全相同的 model/dataset/stats/log env 和 experiment name；
机器断言包括：无 `min_lr` key、`min_lr_rate=.1`、2k max/save/optimizer steps、无 val、
micro/global16/32、no-shard、frozen VLA、dataset/model/stats/output 精确路径。resolved
SHA-256：

```text
5aa824fc9ac5cc361dace2b1162b2ef1bdf52adab3c775b8cd2e1ae468dfd67e
```

完整批准包在：

```text
/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/
  formal_resolved.yaml
  source_config.yaml
  dataset_manifest.json
  exact_command.txt
  prelaunch_provenance.tsv
  PRELAUNCH_SHA256SUMS
```

配置单文件经 `git diff --check`、精确 staged-path 检查后提交并推送：

```text
4ac48d54c63b3a83d99f551fb54f738297525acf
chore(rlt): bind Stage 1 to clean50 dataset
```

推送到 `personal/codex/rlt-pi0-robotwin` 后 HEAD=remote、upstream `0/0`、worktree clean。

完整 resolved packet、精确命令、输出、资源预期和停止条件再次在聊天中展示后，于
`2026-07-29T19:34:31+08:00` 后台启动。启动命令主体为：

```bash
timeout --signal=TERM --kill-after=120s 64800s \
  /root/autodl-tmp/RLinf/.venv/bin/python -B examples/sft/train_vla_sft.py \
  --config-path /root/autodl-tmp/RLinf_rlt_pi0_robotwin/examples/sft/config \
  --config-name robotwin_rlt_stage1_sft_openpi \
  runner.logger.experiment_name=robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1
```

driver PID `650254`，resource monitor PID `650255`。目录：

```text
run:
  /root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1
runtime/evidence:
  /root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/runtime
expected endpoint:
  /root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/
  robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
```

`19:35:33` 两个 FSDP worker 均已建立，读取 full clean-50 与同一 stats，local batch 均为
16；`19:36:19` 每卡已分配 17,705MiB、resource monitor 观察到 matched RSS 峰值约
38.5GiB，driver 仍 running、错误计数 0，处于模型初始化而尚无 optimizer metric。
后续只观察到连续 finite optimizer step 和稳定资源后停止本轮轮询，不等待 2k 完成。

## 52. A052：19:38 一次性早期健康快照与停止轮询

为避免用滚动进度条或人工抄数，上传并运行
`rlt_capture_early_health_20260729.sh`。脚本只读 driver log/resource CSV/Git/GPU，
hard-fail：

- driver 不存活或 Git HEAD/clean 漂移；
- 少于 20 个 optimizer step；
- loss/rlt loss/grad/LR/time 任一非 finite；
- `loss != rlt_loss` 或 `vla_loss != 0`；
- OOM、CUDA error、Traceback、NCCL error、ChildFailed 或 killed 信号。

输出写入：

```text
/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/
  early_health.json
  early_health.json.sha256
```

`2026-07-29T19:38:41.986924+08:00` 快照：

| 项 | 结果 |
|---|---:|
| state | running |
| latest step | 172/2000 |
| step1 | loss5.20，LR2.5e-7，冷首步20.1s |
| step10 | loss5.19，LR2.5e-6，0.780s |
| step20 | loss5.16，LR5e-6，0.783s |
| step50 | loss4.51，LR1.25e-5，0.777s |
| step172 | loss1.05，LR2.49e-5，grad1.03，0.774s |
| median last20 | 0.777s/step |
| GPU now/peak | 26,447MiB each；100%/94% 当前利用率 |
| matched RSS peak | 40,378,024KiB，约38.5GiB |
| error counts | 全0 |

所有采样点均 `loss=rlt_loss`、`vla_loss=0`。按最近20步中位数计算的纯训练剩余时间约
1,420s，即23.7分钟，尚未计 endpoint save；这替代了冷 S1-B 单步给出的11.5小时保守上界，
但仍只是现场估计，不是完成承诺。`early_health.json` SHA-256 为
`eb18a1622e53d202f10d9a05e1f64d47d85b8945a0947d3dd45f1cb116dc2f4f`。

按用户要求，本轮到此停止主动轮询，不等待2k。下一次询问时必须重新读取服务器现场，不能把
step172、PID、显存或 ETA 当成持续当前值。

## 53. A053：文档收口、提交与有限网络重试

把本轮正式转换、清理、配置、启动和一次性健康快照整理进专题 SSOT、HANDOFF 和机器证据。
第一次文档审查发现 `exact_command.txt` 被仓库的 `*.txt` 规则忽略；没有改写忽略规则，而是
把该精确命令作为证据文件单独 `git add -f`。第一次提交前的
`git diff --cached --check` 又发现正式训练交接文档两行末尾各有两个空格；仅对本轮精确
staged paths 执行 `git restore --staged`，保留 worktree 内容，删除尾随空格后重新运行完整
审查。第二次审查于 `2026-07-29T19:49:02+08:00` 通过：

```text
FORMAL_DOCS_REVIEW_OK files=14
```

审查覆盖 UTF-8、Markdown 相对链接、JSON/YAML 解析、证据 SHA 与口令/私钥模式。随后提交：

```text
d7c3ca7e2ddfc8d0b3c376ec6d30ba89b965a5dc
docs(rlt): record formal Stage 1 launch
14 files changed, 1876 insertions(+), 48 deletions(-)
```

第一次 `git push` 在等待约 131 秒后因 GitHub 443 连接超时失败；该失败只影响 Git 同步，
不影响已经独立运行的 Stage 1 driver。一次内联重试命令在 Windows helper 参数解析阶段失败，
未产生远端操作。随后上传窄重试脚本
`/root/autodl-tmp/tmp/remote_rlt_20260729_retry_docs_push.sh`，脚本先 hard-fail 检查精确 HEAD、
clean worktree 和 upstream ahead=1，再给 `git push` 240 秒上限。有限重试成功：

```text
4ac48d54..d7c3ca7e  codex/rlt-pi0-robotwin -> codex/rlt-pi0-robotwin
remote refs/heads/codex/rlt-pi0-robotwin =
d7c3ca7e2ddfc8d0b3c376ec6d30ba89b965a5dc
```

到此不再轮询训练；下次查看实验时从服务器现场重新核对 driver/exit、最终日志、资源尾部、
`global_step_2000` 完整性和固定 prefix 的 endpoint 质量证据。
### A054 — 2026-07-29 Stage 1 收尾与 Stage 2 pre-smoke 授权、现场刷新

- 用户授权边界：
  - 可执行简洁的 Stage 1 endpoint reload、fixed-prefix、瓶颈对照、冻结性与 artifact manifest 检查；
  - 可筛选、打包并下载 Stage 1 小型高信息量证据；
  - 可完成 Stage 2 smoke 之前的配置审计、compose、静态/单元测试和机器检查；
  - **不得在本轮直接启动 Stage 2 smoke**；必须先给出 resolved config、精确命令、输出目录、资源预估和停止条件，等待用户批准。
- 按工作区规则重新完整读取：
  - `PROJECT_CONTEXT.md`
  - `HANDOFF.md`
  - `docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md`
  - `docs/rlinf-robotwin-pi0-rltoken/03_STAGE1_FORMAL_TRAINING_20260729.md`
  - 本实施账本最近条目。
- 使用密码附件仅向当前进程注入凭据，通过
  `local_scripts/remote_exec_autodl.py` 做只读身份探针和现场刷新；未打印或落盘保存密码。
- 现场观察时间：`2026-07-29T20:48:19+08:00`。
- Git：
  - worktree：`/root/autodl-tmp/RLinf_rlt_pi0_robotwin`
  - branch：`codex/rlt-pi0-robotwin`
  - HEAD：`6df42bf488ef10d9c7eb2f89584bc5ab7543a08a`
  - porcelain：空，服务器 worktree clean；
  - 与 upstream：`0 behind / 1 ahead`，ahead 为此前收尾文档提交，尚未因远端网络恢复而推送。
- 运行现场：
  - 没有存活的 RLT 训练进程；旧 PID 文件不代表存活进程；
  - 两张 GPU 均为 `0%` util、约 `0 MiB` 占用；
  - 主机内存总计约 `1.0 TiB`，available 约 `982 GiB`；
  - `/root/autodl-tmp`：约 `1.9 TiB` 总量、`1019 GiB` 已用、`826 GiB` 可用、`56%`。
- Stage 1 正式运行：
  - `runtime/exit_code.txt = 0`
  - 结束时间 `2026-07-29 20:03:22+08:00`
  - 唯一 endpoint：
    `/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000`
  - endpoint 总量约 `20.56 GiB`，包括 `full_weights.pt` 与两 rank DCP shards；没有临时 checkpoint 残留。
- 本条仅记录现场与授权，没有启动模型、测试、smoke 或训练。

### A055 — 服务器当前配置取证与一次 SFTP 瞬时失败

- 目标：以服务器 branch 当前文件为准，取回 ManiSkill RLT、RoboTwin Stage 2
  formal/smoke、Stage 1 source config 和既有 prefix probe，避免把本机旧 worktree
  当作动态真相。
- 第一次命令：在一个 PowerShell 循环中连续调用
  `local_scripts/remote_exec_autodl.py get`。
- 结果：
  - 第一个 ManiSkill YAML 下载成功；
  - 第二次新建 SSH connection 时 Paramiko 报
    `SSHException: Error reading SSH protocol banner`；
  - 没有改动服务器文件。
- 诊断：不是认证失败或文件缺失，而是短时间连续新建 SSH transport 的 banner
  瞬时失败。最小修复是不重试并发连接，而是在同一个已校验 host-key 的 Paramiko/SFTP
  session 中批量获取。
- 新增临时、无凭据工具：
  `.tmp/remote_get_current_rlt_files.py`。第一次运行因脚本目录在 `.tmp`、Python
  没把仓库根加入 `sys.path`，报
  `ModuleNotFoundError: No module named 'local_scripts'`；随后仅增加仓库根
  `sys.path`，复测成功。
- 成功取回目录：
  `.tmp/rlt-stage2-prep-server/`。
- 当前服务器关键 SHA-256：
  - RoboTwin Stage 2 formal：
    `426c09e2d9b036c566560124059917e9c22059457e75035e924fb678f6018637`
  - RoboTwin Stage 2 smoke：
    `02715a69ac5ff76fb6d3d7250b447dd0131f3621ae87271d54e9fe4ef6712aa8`
  - Stage 1 source config：
    `c293bc476ec7458c6bfc5c5c59393e48b286f3e12007f3039ccc282e30645a4c`
  - prefix probe：
    `ac0555274f6347b0bd089a8d626477d1a8b26d870731f235bd2e0c25b8d9df33`
- 发现：本机 `.rlt-impl-worktree` 中 RoboTwin Stage 2 两份 YAML 与服务器逐字一致；
  ManiSkill 文件不一致，因此后续参数溯源固定使用刚取回的服务器版本
  `bb5c01c0db25fcd962b5fa21d2fe60505ed73ed86bb8875e19167378d6456457`。

### A056 — 新增 Stage 1 artifact acceptance probe

- 新增文件：
  `toolkits/rlt/validate_robotwin_rlt_stage1_artifact.py`。
- 调用链和检查：
  1. 用原 π0 checkpoint、正式 Stage 1 架构和 seed 0 构造 fresh control；
  2. `torch.load(..., mmap=True, weights_only=True)` 打开正式
     `full_weights.pt`；
  3. 对 endpoint 与 fresh/base 的完整 state-dict key set 做严格相等检查；
  4. 对所有非 `rlt_module.*` 张量逐元素 `torch.equal`，作为 π0 delta=0 hard gate；
  5. 从 clean-50 OpenPI loader 取固定 seed 的真实 batch=4，构造同一 image-prefix；
  6. 先测 fresh seed-0 proxy loss，再 `strict=True` 加载 endpoint；
  7. 检查 reload 前后 frozen prefix 完全一致；
  8. 比较 endpoint true-`z_rl`、batch-shuffled-`z_rl` 和 zero-`z_rl`
     reconstruction；
  9. 输出 `validation.json` 和供 Stage 2 绑定的
     `stage1_artifact_manifest.json`，后者记录 endpoint/full-weight SHA、DCP
     inventory、dataset/config/stats/validation identities。
- 这个 probe 不做 backward、不改模型、不产生新 checkpoint；它是一次单卡 artifact
  验收，不是新训练实验。
- 上传前 guard：
  - server HEAD
    `6df42bf488ef10d9c7eb2f89584bc5ab7543a08a`
  - branch `codex/rlt-pi0-robotwin`
  - worktree clean；
  - 无 RLT 训练/验收进程。
- 静态检查过程：
  - `python -m py_compile`：通过；
  - `git diff --check`：通过；
  - 第一次 `ruff check`：仅报 import block `I001`；
  - 手动调整一次仍被 `I001` 拒绝；
  - 用仓库同版 `ruff check --fix` 做机械排序，再运行
    `ruff check`、`py_compile`、`git diff --check`：全部通过；
  - 格式化后的服务器文件已取回本机对应 RLT worktree，避免两端漂移。

### A057 — Stage 1 acceptance v1 启动失败与不覆盖修复

- 新增可复现包装器：
  - `local_scripts/remote_rlt_20260729_stage1_artifact_acceptance.sh`
  - `local_scripts/remote_rlt_20260729_start_stage1_artifact_acceptance.sh`
- v1 目标目录：
  `/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v1`
- v1 启动前两卡都是 `0 MiB / 0%`，正式 endpoint、dataset manifest 和输入配置均存在。
- v1 PID：`718791`。
- v1 结果：在加载 Python/模型之前立即退出，`exit_code=127`；
  `driver.log` 为 0 bytes，GPU 始终为空。
- 原始错误：
  `/root/autodl-tmp/tmp/rlt_stage1_artifact_acceptance_20260729_v1.sh:
  line 47: /usr/bin/time: No such file or directory`。
- 原因：包装器错误假设镜像提供 GNU `/usr/bin/time`；这不是模型、checkpoint 或 CUDA
  错误。
- 处理：
  - 不安装新依赖；
  - 保留整个 `artifact_acceptance_v1` 失败目录和 exit code；
  - validation Python 使用标准库
    `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` 记录进程 max RSS；
  - 移除外部 `/usr/bin/time`；
  - 新建不覆盖的 `artifact_acceptance_v2`。
- v2 重新上传后再跑 `ruff check --fix`、`ruff check`、`py_compile` 和
  `git diff --check`，全部通过。
- v2 PID：`719016`。
- v2 输出目录：
  `/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/artifact_acceptance_v2`
- 当前状态：已启动，尚未把运行中状态误报为通过；等待 `exit_code.txt`、
  `validation.json` 和 manifest 后再验收。

### A058 — Stage 1 artifact acceptance v2 完成

- v2 时间：
  - start：`2026-07-29T21:04:56+08:00`
  - finish：`2026-07-29T21:06:54+08:00`
  - validation 内部 wall：`104.814 s`
- 退出：`exit_code=0`；退出后两张 GPU 均为 `0 MiB`，无验收进程。
- 产物：
  - `artifact_acceptance_v2/validation.json`
  - `artifact_acceptance_v2/stage1_artifact_manifest.json`
  - `artifact_acceptance_v2/artifact_sha256.txt`
  - `artifact_acceptance_v2/driver.log`
  - `artifact_acceptance_v2/stderr.log`
  - `artifact_acceptance_v2/resources_before_after.txt`
  - `artifact_acceptance_v2/launch_provenance.txt`
- 所有 hard gates 为 true：
  - full state dict strict reload；
  - 非 `rlt_module.*` 的 π0 tensors delta=0；
  - reload 前后 fixed frozen prefix 完全相等；
  - 所有 loss finite；
  - endpoint loss 优于 fresh seed-0 matched proxy；
  - true `z_rl` 优于 batch-shuffled `z_rl`；
  - true `z_rl` 优于 zero `z_rl`。
- fixed real batch 的结果：
  - fresh proxy loss：`5.1976585388`
  - endpoint/true-z loss：`0.5337553024`
  - shuffled-z loss：`1.7118018866`
  - zero-z loss：`2.1026575565`
  - endpoint/fresh：`0.10269`
  - true/shuffled：`0.31181`
  - true/zero：`0.25385`
- 参数冻结/更新：
  - non-RLT changed tensor count：`0`
  - RLT changed tensor count：`54 / 62`
  - Stage 1 trainable parameter count：`743,094,272`
- 资源：
  - CUDA peak allocated：`9,055,774,720 bytes`，约 `8.43 GiB`
  - CUDA peak reserved：`9,367,977,984 bytes`，约 `8.72 GiB`
  - process max RSS：`20,728,152 KiB`，约 `19.77 GiB`
  - 主机 available 从约 `982.88 GiB` 到约 `982.69 GiB`，无内存压力；
  - v2 不做 backward，因此不能拿这组 8.4GiB 替代 Stage 1 正式训练的
    26,447MiB/card 峰值。
- artifact identity：
  - manifest ID：
    `robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1`
  - manifest SHA-256：
    `6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433`
  - validation SHA-256：
    `90385a9ffd812e5806cd97db7537a4e1d8c62f873711a18e72c0c010d4de66cc`
  - full weights SHA-256：
    `7dddc268733b978bf382cda77257371cf9de4155f60ec3094cc8ffcfd6d74bd0`
- 结论边界：
  - 现在可以把 Stage 1 称为“artifact 验收通过”，并绑定 Stage 2；
  - 这证明 checkpoint 可加载、π0 未改、重建已学习且 decoder 使用了 sample-specific
    bottleneck；
  - 它仍不证明 `z_rl` 一定提升控制成功率，后者只能由 Stage 2 smoke/pilot 给出。

### A059 — Stage 1 内存图表生成与双视口 QA

- 数据源固定为正式运行的 `runtime/resources.csv`，共 `1,455` 个采样点；曲线只为显示做降采样，
  卡片中的峰值和最小值仍由全部采样点计算。
- 新增独立内联可视化：
  `E:/Codex/home/visualizations/2026/07/28/019fa752-79fd-7de3-b66f-5ac4f0a72bfc/rlt-stage1-memory-breakdown.html`。
- 图中明确拆分：
  - 仅匹配训练 rank 的 RSS；
  - 整个容器 cgroup anonymous memory；
  - cgroup accounted total 与其中可回收的 file cache；
  - 主机 available memory。
- 使用技能提供的 `render.py` 生成预览；第一次浏览器检查的是预览包装页，
  截图正常，但 DOM 查询落在包装层，`querySelector("main")` 返回空。这不是图表缺失。
- 窄修复：改为直接打开原始 HTML，再以系统 Chrome/Playwright 做桌面和移动端 QA。
- QA 结果：
  - 桌面 `1440×1100`：`scrollWidth=clientWidth=1440`；
  - 移动端 `390×844`：`scrollWidth=clientWidth=390`；
  - 两端均有 3 个 SVG 图、无 console error、无横向溢出；
  - 桌面截图 `127,983 bytes`，移动端截图 `110,344 bytes`；
  - 人工查看移动端截图，标题、四张指标卡、三张曲线和解释文字均完整可读。

### A060 — Stage 1 服务器证据包 v1 失败与 v2 窄修复

- 新增并执行：
  `local_scripts/remote_rlt_20260729_package_stage1_evidence.sh`。
- v1 在复制
  `/root/autodl-tmp/RLinf_rlt_pi0_robotwin/local_scripts/remote_rlt_20260729_stage1_artifact_acceptance.sh`
  时以 exit 1 停止；原始错误为 `cp: cannot stat ... No such file or directory`。
- 原因：两个验收启动包装器是本地实施工具，没有被提交到服务器 worktree；正式运行证据、
  验收 Python 和 checkpoint 均未缺失。
- 处理：
  - 保留服务器 `download_bundle_v1` 作为失败证据，不删除、不覆盖；
  - 新增 `remote_rlt_20260729_package_stage1_evidence_v2.sh`；
  - v2 只从服务器仓库复制确实存在的验收 Python，本地包装器在下载后加入最终 ZIP。
- v2 成功生成：
  `/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1/rlt_stage1_formal_high_info_20260729_v2.tar.gz`
- v2 大小 `327,611 bytes`（`du -h` 为 `320K`），SHA-256：
  `07b24ab71a0cff2ed6a6ccaee0abed828def2ec870acc35ea6ba9fd9e11b3166`。
- 服务器 tar 包含 34 个文件：完整 driver/resources/TensorBoard、source/resolved config、
  数据和运行 provenance、v1 失败、v2 成功验收、manifest/validation 与验收源码；
  不含 20.56 GiB checkpoint。

### A061 — Stage 1 证据下载、校验与最终 ZIP

- 通过 host-key 校验的单次 Paramiko SFTP 下载服务器 tar；密码只在当前 PowerShell
  进程环境中短暂注入，`finally` 中清除。
- 本地 tar：
  `exports/rlt_stage1_formal_high_info_20260729_v2.tar.gz`；
  下载后 SHA 与服务器的
  `07b24ab71a0cff2ed6a6ccaee0abed828def2ec870acc35ea6ba9fd9e11b3166`
  一致。
- 解包前先检查所有 archive member 均以 `download_bundle_v2/` 开头、没有绝对路径或
  `..`；随后解到独立目录，不覆盖已有文件。
- 对服务器生成的 `CONTENTS_SHA256.txt` 做本机逐文件复验：
  `SERVER_BUNDLE_CONTENTS_OK files=34`。
- 下载后加入：
  - 正式精确命令；
  - 验收的两个本地包装器；
  - 训练状态图；
  - 内存桌面/移动端图；
  - 中文 README。
- `LOCAL_ADDITIONS_SHA256.txt` 的 7 个文件也逐文件复验通过。
- 最终用户包：
  `exports/rlt_stage1_formal_high_info_20260729_v2.zip`
  - 大小：`630,816 bytes`
  - SHA-256：
    `9d9e2c38789897479a27cc04ed15034a9d65175284c837f3c1c6f54ca0c2daa8`
  - 同目录有 `.zip.sha256`。

### A062 — Stage 2 方法源复核与 PDF 下载失败边界

- 重新以三层来源核对 Stage 2：
  - RLT 论文/Physical Intelligence 项目页决定方法高层语义；
  - 服务器当前 RLinf ManiSkill RLT YAML 决定可执行参考实现；
  - 已验证 RoboTwin π0 配置决定任务、相机、H=50、14D canonical action、
    normalization 与 env/rollout 接口。
- 论文/官方页面确认的高层项包括：frozen Stage 1 feature、reference-conditioned
  actor、BC+Q actor objective、高 UTD=5、每两个 critic update 做一个 actor update、
  fixed-small exploration、critical-phase/人工 intervention 为可选机器人设定。
- 服务器 ManiSkill YAML 的当前 SHA-256 固定为
  `bb5c01c0db25fcd962b5fa21d2fe60505ed73ed86bb8875e19167378d6456457`；
  其可执行 schedule 是 `update_epoch=5`、每 5 个 transition 触发，等效 macro-UTD=1，
  `critic_actor_ratio=4`，不能误写成论文 UTD5/ratio2。
- 为保留论文离线副本，先后尝试：
  - PowerShell `Invoke-WebRequest` 下载 arXiv PDF；
  - Windows `curl.exe` 下载同一 PDF。
- 两次均未产生可用文件：
  - 前者为 TLS receive 失败；
  - 后者为 Schannel `SEC_E_NO_CREDENTIALS`。
- 未绕过证书、未安装工具、未保留 0-byte 假 PDF。参数复核继续使用已打开的论文
  HTML、Physical Intelligence 项目页和 RLinf 官方文档；这不影响本轮代码检查。

### A063 — Stage 2 fail-closed artifact/预算加固

- 修改文件：
  - `examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp.yaml`
  - `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py`
  - `tests/unit_tests/test_robotwin_rlt_contract.py`
- 新增文件：
  - `toolkits/rlt/preflight_robotwin_rlt_stage2_artifact.py`
  - `toolkits/rlt/audit_robotwin_rlt_stage2_resolved.py`
- 配置修复：formal source 的 `runner.max_steps` 从会实际执行的 `-1` 改为
  fail-closed `0`；正式 collection-cycle 预算必须在 smoke 后由启动命令显式覆盖。
  dedicated smoke config 仍显式覆盖为 `1`，不受影响。
- resume/artifact 合同新增 hard gates：
  - manifest 必须 `accepted=true`、`schema_version=1`；
  - manifest 内 Stage 1 model path 必须与实际 feature model 路径一致；
  - stats SHA、canonical adapter、H/C/D、`z_rl` 和 prefix shape 必须同时与
    manifest 和 resolved config 一致。
- 新 preflight 以流式 SHA-256 验证 9.55GB `full_weights.pt`，避免把“目录存在”
  当成 artifact 身份；新 resolved-audit 同时审 formal/fresh/resume 三份绑定后配置。
- 上传前执行
  `local_scripts/remote_rlt_20260729_stage2_hardening_upload_guard.sh`，逐项核对 branch、
  HEAD、三个被修改文件的服务器 baseline SHA、目标新文件不存在且目标路径无 dirty
  change；结果 `STAGE2_HARDENING_UPLOAD_GUARD_OK`。随后才上传精确五个文件，
  未覆盖其他用户修改。

### A064 — Stage 2 集中测试首次失败、原因与窄修复

- 服务器检查命令固定在
  `local_scripts/remote_rlt_20260729_stage2_hardening_checks.sh`：
  `py_compile`、`ruff check`、`git diff --check`，再运行
  `tests/unit_tests/test_robotwin_rlt_contract.py`。
- 第一次结果：
  - 静态检查全部通过；
  - 单测 `16 passed / 3 failed`。
- 三个失败都来自同一新增 model-path gate：测试的 resumed worker 使用了自己的临时
  Stage 1 model 目录，而 source worker 的 manifest 仍记录另一个临时目录。该失败证明
  hard gate 确实生效，不是训练实现回归。
- 窄修复：测试 helper 新增 `_bind_same_stage1_artifact`，在模拟合法 resume 时把 source
  与 resumed worker 的 Stage 1 model/stats identity 对齐；不放宽生产代码验证。
- 只重新上传测试文件后再次运行同一脚本：
  - `ruff`、`py_compile`、`git diff --check` 全通过；
  - `19 passed in 8.39s`；
  - 终标记 `STAGE2_HARDENING_CHECKS_OK`。
- 本项没有加载 π0、没有启动 Ray/RoboTwin，也没有运行 smoke。

### A065 — Stage 1 绑定、三份 Stage 2 compose 与资源现场

- 执行
  `local_scripts/remote_rlt_20260729_stage2_bind_compose_preflight.sh`。
- 执行前 hard fail 检查：
  - branch/HEAD 精确为
    `codex/rlt-pi0-robotwin@6df42bf488ef10d9c7eb2f89584bc5ab7543a08a`；
  - 无 RLT/Ray 进程；
  - 新 evidence root、计划 smoke root 和待批准 pilot root 均不存在。
- 新建的仅是轻量 evidence root：
  `/root/autodl-tmp/experiment_exports/rlt_stage2_pre_smoke_20260729_v1`。
  计划 smoke root
  `/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1`
  仍不存在。
- artifact preflight 通过：
  - manifest SHA
    `6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433`
  - full weights `9,551,212,074 bytes`
  - full-weights SHA
    `7dddc268733b978bf382cda77257371cf9de4155f60ec3094cc8ffcfd6d74bd0`
  - H/C/D=`50/10/14`、`z_rl=2048`、prefix=`768×2048`、stats SHA 全一致。
- 使用 `--cfg job --resolve` compose，不初始化模型、Ray 或模拟器：
  - formal bound SHA：
    `4cbb7c7c03457276723293845c14ddc3a2f960badabd38cc40b2c9c592baf59c`
  - fresh bound SHA：
    `c45743c1c797a9010d9a0f0c36a41c4cbabf4fd8f69e39707cb501e7b3d5c229`
  - resume bound SHA：
    `f91688d21c7d6180dacb169824210b415e49f1c7d26a27d2a917f562ab24c82a`
  - resolved audit SHA：
    `a26d8db55d4ac306a618511ed971b325f219f0bb270d2bb26b638d540038469f`
  完整 SHA 位于 evidence root 的 `SHA256SUMS`，不以省略值作机器合同。
- resolved audit 全部通过；推导 smoke 合同：
  - 两 rank、global/micro=`512/128`，gradient accumulation=`2`；
  - fresh 满长 collection 为 global 8 rows、每 rank 4 rows；
  - fresh 做 8 critic / 4 actor updates；
  - 新进程 resume 满长做 20 critic / 10 actor updates，终态
    `update_step=28`、pending=`20`。
- 运行耗时 `39s`，主要是流式计算 9.55GB checkpoint SHA 和 Hydra import；
  仅出现 TensorFlow CPU/oneDNN 提示，无模型/Ray/模拟器启动。
- 资源前后：
  - 前：两卡 `0MiB/0%`；host available 约 `982.2GiB`；
    `/root/autodl-tmp` available 约 `826GiB`；
  - 后：两卡各约 `4MiB/0%`；host available 变化约 `0.5GiB`；
    磁盘仅增加约百 KB evidence。
- 输出中最重要的 11 个文件已通过同一 host-key 校验的 Paramiko/SFTP session
  下载到本机 `.tmp/rlt-stage2-pre-smoke-v1/`；密码仍只存在当前进程并在
  `finally` 清除。

### A066 — pre-smoke evidence 本地校验、一次 PowerShell 解析失败与修复

- 目标：把服务器已下载的绑定/resolved/audit/source/resource 文件整理到
  `docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_pre_smoke_20260729/`，
  但不把未经校验的 SFTP 字节直接当成证据。
- 第一次 PowerShell 命令在执行任何目录创建/复制前就被 parser 拒绝：
  `"SHA mismatch for $rel: ..."` 中变量后紧跟冒号，被 PowerShell 解释成无效 drive
  变量引用。没有文件被创建或覆盖。
- 窄修复：
  - 字符串变量改为 `${rel}`；
  - SHA 行匹配从易误转义的正则改为 `EndsWith("  $rel")`。
- 第二次结果：
  - 逐个对照服务器 `SHA256SUMS` 验证10个下载文件；
  - 新建独立 evidence 目录；
  - `STAGE2_EVIDENCE_COPY_OK verified=10 files=11`。
- source/resolved 文件保持服务器原字节；只把编码后的临时下载名改为可读证据名。

### A067 — Stage 2 小模型与 replay 资源预算实测

- 新增并执行：
  `local_scripts/remote_rlt_20260729_stage2_model_budget.sh`。
- guard：branch/HEAD 精确、pre-smoke evidence 存在、输出不存在、无 RLT/Ray 进程。
- 在服务器共享 venv 的 CPU 上实际构造
  `RLTMLPPolicy(z=2048,proprio=14,C=10,D=14,3×256,twin-Q)`，没有加载 π0、
  GPU、Ray 或模拟器。
- 结果：
  - actor optimizer group：`767,512`
  - critic/twin-Q group：`1,394,690`
  - model total：`2,162,202`
  - model+target FP32 raw tensors：`17,297,616 bytes`
  - model/target/grad/Adam 粗上界：`43,244,040 bytes`
  - compact replay raw estimate：`18,359 bytes/row`
  - smoke 64 rows/rank：`1,174,976 bytes/rank`
  - formal 15k rows/rank：`275,385,000 bytes/rank`
- 输出：
  `/root/autodl-tmp/experiment_exports/rlt_stage2_pre_smoke_20260729_v1/model_replay_budget.json`；
  下载 SHA-256
  `9913e641cb909535b2fdc31cd022a3daa79d86d8a133bb47029a2482e5537d2d`。
- 这个 replay 数只含 tensor payload，不含 Python object、allocator、trajectory staging、
  dataloader batch 或 frozen π0，因此只能用于确认“小 MLP/replay 不是资源主项”。

### A068 — Stage 2 代码提交

- 提交前服务器精确 status 只有本轮三个 tracked changes 和三个新 toolkit files：
  formal config、RLT worker、contract test、Stage1 acceptance、Stage2 artifact preflight、
  resolved audit；没有其他 dirty/untracked 文件。
- 新增并执行
  `local_scripts/remote_rlt_20260729_stage2_code_commit.sh`：
  - hard-code 对比精确 `git status --short`；
  - 对5个 Python 文件做 `py_compile`/Ruff；
  - 运行19个 RLT contract tests；
  - `git diff --check` 与 staged diff check；
  - 只 `git add` 六个精确目标文件。
- 结果：
  - `19 passed in 8.20s`
  - Ruff/compile/diff 全通过；
  - commit：
    `3b610cb4685a1d41c97da64df67ab86561697dfd`
  - message：
    `fix(rlt): bind Stage 1 artifact before Stage 2`
  - `6 files changed, 1176 insertions(+), 16 deletions(-)`；
  - commit 后 worktree clean。
- 本项尚未 push；后续文档收口后统一做有限网络同步。

### A069 — fresh/resume 精确 launch wrappers 与 monitor 自测

- 新增但**未执行 smoke**：
  - `remote_rlt_20260729_stage2_resource_monitor.sh`
  - `remote_rlt_20260729_start_stage2_smoke_fresh.sh`
  - `remote_rlt_20260729_start_stage2_smoke_resume.sh`
  - `remote_rlt_20260729_stage2_launch_scripts_validate.sh`
- wrapper hard gates：
  - clean/upstream0/0、code commit ancestor、code commit 后无非 docs diff；
  - source/worker/monitor/manifest/stats SHA；
  - Stage1 artifact preflight 与 fresh/resume resolved SHA；
  - 无 RLT/Ray/GPU process、host available RAM、disk、output 不存在；
  - resume 额外检查 fresh exit0、DCP1 completion/rank state/replay。
- 第一次尝试把多行 remote `awk` validation 塞进 PowerShell 双引号字符串，在本地 parser
  阶段因嵌套引号报 `UnexpectedToken`；没有建立 SSH 连接或上传文件。
- 窄修复：把验证命令写为独立 Bash 文件，不再跨 PowerShell/SSH 两层拼接复杂引号。
- 服务器验证：
  - `bash -n` 三个脚本通过；
  - monitor 用不存在 PID 做一次单样本执行；
  - CSV `22 fields / 2 rows`，每行 field 数相同；
  - 计划 smoke run/evidence root 仍不存在。
- 首次成功后又给 fresh/resume 加入 monitor SHA hard gate；为保留 v1 selftest，
  第二次使用不覆盖的 `rlt_stage2_monitor_selftest_20260729_v2.csv` 复测。
- 最终脚本 SHA：
  - monitor：
    `925cb515a4ecd6dbfcb192168c63644e1b2b2d691f6a4d50fdc3ddd8a5bbd96b`
  - fresh：
    `13ea8602d257bc084b3441ecd7a2220dcfb2d0aab6869dd7fb5ac0445fdeb7c6`
  - resume：
    `b76b67c47cd1e629990157ad5c4f1d8628bd253a26cf533989aadfde166b896c`

### A070 — 当前 Stage 2 packet 与文档分层

- 新增当前唯一 Stage 2 审批文件：
  `docs/rlinf-robotwin-pi0-rltoken/04_STAGE2_PRE_SMOKE_PACKET_20260729.md`。
- packet 逐组覆盖：
  - 论文/ManiSkill/RoboTwin/项目适配/工程治理来源；
  - runner/topology/env/feature/action/route/TD/target/actor loss/optimizer/replay/
    UTD/warm-up/cap/resume；
  - formal 与 smoke 唯一差异；
  - fresh/resume 调用流、expected updates、resolved SHA、精确脚本、输出；
  - GPU/RAM/disk/time 预算、停止条件和结论边界；
  - formal 总 cycle 继续 fail closed，smoke 后才在约30/60-cycle pilot 中决策。
- 更新：
  - `00_INDEX_AND_IMPLEMENTATION_PLAN.md`：路由到当前 Stage2 packet，Stage1/artifact
    状态改为已完成；
  - `01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md`：明确为历史参数来源快照，不再用
    unresolved config 启动；
  - `03_STAGE1_FORMAL_TRAINING_20260729.md`：补齐2k指标、内存口径、artifact验收和用户包；
  - 根 `HANDOFF.md`：当前停点改为“Stage2 packet 已准备、smoke 未启动”。
- evidence 目录加入机器原始文件、source/resolved、两份 SHA 清单、资源预算和8个可复现
  脚本；大 checkpoint 仍只留服务器。

### A071 — 本地文档与证据 QA，以及根目录 Git 边界

- 对当前 RLT 文档、Stage 2 evidence 和 Stage 1 下载包做只读一致性检查，结果：
  `RLT_DOC_QA_OK markdown=8 hashes=21 json=3`。
- 检查内容包括：
  - 8 份 Markdown 的相对链接均能解析到现有文件；
  - 3 份 JSON 可解析，formal/fresh/resume resolved YAML 可解析；
  - Stage 2 bound config 中不存在未解析的 artifact 占位符；
  - `LOCAL_SHA256SUMS` 中 21 个文件逐一复算一致；
  - 文档、脚本和证据中没有密码、SSH 密码变量值或附件原文。
- 第一次把根目录 `git diff --check` 混入 QA 时，Git 因仓库探测/ownership 边界报错；
  后续只读核对确认：
  - `C:/Users/86136/Documents/rl/.git` 存在，但这是一个尚无 commit 的文档聚合仓；
  - sandbox 用户与目录 owner 不同，普通 Git 命令会触发 `dubious ownership`；
  - 使用单次命令参数
    `git -c safe.directory=C:/Users/86136/Documents/rl ...` 可只读访问，不修改全局配置；
  - 因当前文档全是 untracked，根仓的 `git diff --check` 本身不能覆盖这些文件，不能把它
    当作有效 whitespace 证据。
- 因此根目录只承担 SSOT/下载包，最终 Git diff、精确 dirty-set、commit 和 push 以服务器
  `codex/rlt-pi0-robotwin` 独立 worktree 为准；同步前仍会对所有待上传文本做独立
  UTF-8、trailing-whitespace、link、JSON/YAML 和 SHA 检查。
- 内存可视化已做桌面 `1440×1100` 与移动端 `390×844` QA：3 个 SVG、无 console error、
  无横向溢出；它只读取 Stage 1 `resources.csv`，没有修改实验产物。

### A072 — Stage 2 文档原子同步与提交工具准备

- 新增本地、无凭证的单次实施工具：
  - `local_scripts/sync_rlt_stage2_docs.py`：通过已固定 host-key 的 Paramiko 连接，把6个
    SSOT/交接文件与22个 Stage 2 evidence 文件上传到独立 staging，不直接碰 Git worktree；
    每个文件使用 `.part -> posix_rename`，最后生成并复核 `UPLOAD_SHA256SUMS`；
  - `remote_rlt_20260729_stage2_docs_upload_guard.sh`：上传前检查 branch/HEAD/clean、
    RLT/Ray/GPU/RAM/disk、smoke 路径和 staging 路径；
  - `remote_rlt_20260729_stage2_docs_deploy_review.sh`：先复核 staging 全文件 SHA，再只复制
    manifest 内路径，随后检查 dirty-set、UTF-8、链接、JSON/YAML、resolved binding、
    21个 evidence hash、凭证模式和8份脚本的 `bash -n`；
  - `remote_rlt_20260729_stage2_docs_commit_push.sh`：只暂存 upload manifest 中的路径，
    校验 staged diff 后提交并有限时 push。
- 本地 `sync_rlt_stage2_docs.py` 已通过 AST parse；四个新工具均通过严格 UTF-8、
  trailing-whitespace 和精确 private-key marker 检查。
- 第一次本地 Bash 检查误判为可用：`Get-Command bash` 找到了 WSL shim，但三次实际启动均
  返回 `Bash/Service/CreateInstance/E_ACCESSDENIED`；该命令随后仍打印了
  `LOCAL_BASH_N_OK`，所以这个标记被判定为无效，不作为证据。
- 同一复合检查中的第一次 secret scan 又因 pattern 以 `-----BEGIN` 开头且未加 `--`，
  被 `rg` 当成命令行选项；加 `rg --` 后会命中 review 脚本自身的检测正则，仍不能当成泄露。
  最终改成精确 fixed-string 私钥头检查，并把真正的 Bash `-n` 留给服务器。
- 以上两个失败都发生在 SSH/上传之前，没有改变服务器；后续服务器 guard、staging、
  deploy、review、commit/push 的实际结果将另记，不提前写成成功。

### A073 — 文档 deploy review v1 的 pathspec 失败与窄修复

- 服务器上传 guard 于 `2026-07-29T22:26:03+08:00` 通过：
  - branch/HEAD：
    `codex/rlt-pi0-robotwin@3b610cb4685a1d41c97da64df67ab86561697dfd`
  - upstream left/right：`2/0`
  - 两卡 `0 MiB / 0%`
  - host available：`1,001,745,600 KiB`
  - `/root/autodl-tmp` available：`886,849,052,672 bytes`
  - smoke run/evidence 与 staging v1 均不存在。
- 单一 Paramiko/SFTP session 把28个文件原子上传到
  `/root/autodl-tmp/tmp/rlt_stage2_docs_upload_20260729_v1`；远端逐文件 SHA 通过，
  upload manifest SHA：
  `b2a22b1313adfe5d3787e402e84b5f4069948b3c0753c5b403caabc2eff60c24`。
- v1 deploy 已把这28个精确文件复制到 worktree；随后：
  - 28个 upload SHA 全部通过；
  - 内容 QA 通过：`files=28 markdown=7 hashes=21`；
  - 8份 evidence Bash 全部 `bash -n` 通过；
  - 但 review 最后 exit 1，因此没有 commit/push。
- 新增只读诊断
  `remote_rlt_20260729_stage2_docs_review_diagnose.sh` 后定位为唯一原因：
  - 旧 gate `git diff ... ':(exclude)docs/**'` 仍看到根目录 `HANDOFF.md`，故失败；
  - 同一 diff 再精确排除 `HANDOFF.md` 后为空；
  - 不是代码、配置、脚本语法、证据 SHA 或 smoke 失败。
- 这个问题同时会让正式 docs commit 后的 fresh/resume launcher 自我拒绝，所以不能只修改
  review。窄修复同时作用于：
  - docs review/commit gate；
  - fresh launcher；
  - resume launcher；
  - evidence 中的两份 launcher 副本。
- 新 gate 只允许 `docs/**` 与根目录唯一的 `HANDOFF.md` 在 code commit 之后变化；
  branch clean/upstream、code commit ancestor、config/worker/artifact/stats/monitor SHA、GPU/Ray、
  输出不存在等 hard gates 均保留，因此没有放宽算法代码或运行配置。
- 修复后的 launcher identity：
  - fresh：
    `6e0f1c7ce5497bd3d5a2bef539bbea5e3fc964a5d8259b16f472cf353d19e27a`
  - resume：
    `6494eaeb8cb2e6c1decee07d97798a0aba368ee15b0491faf3ecdfe6a9ff054c`
  - monitor 不变：
    `925cb515a4ecd6dbfcb192168c63644e1b2b2d691f6a4d50fdc3ddd8a5bbd96b`
- `04` packet、evidence README 和 `LOCAL_SHA256SUMS` 已同步新 identity。v1 staging 和失败
  dirty state 不删除、不覆盖；v2 guard 只接受 dirty path 是 v1 manifest 的子集，然后使用
  新 staging
  `/root/autodl-tmp/tmp/rlt_stage2_docs_upload_20260729_v2` 覆盖同一精确文件集合。
- 当前仍未提交、未 push、未启动 smoke；v2 staging/deploy/review、服务器 `/tmp` launcher
  替换与 v3 syntax/hash 检查尚待执行。

### A074 — 文档 v2 全量复核与服务器 launcher v3 就绪

- v2 guard 于 `2026-07-29T22:32:25+08:00` 通过：
  - HEAD 仍是 Stage 2 code commit，upstream left/right 仍为 `2/0`；
  - 两卡仍 `0 MiB / 0%`；
  - host available `968,385,896 KiB`，磁盘 available
    `886,848,389,120 bytes`；
  - dirty path 全部属于 v1 upload manifest，没有夹入代码或无关文件；
  - v2 staging 与 smoke run/evidence 仍不存在。
- v2 单连接原子上传28个文件到
  `/root/autodl-tmp/tmp/rlt_stage2_docs_upload_20260729_v2`；所有远端 SHA 通过，
  manifest SHA：
  `7888f139e22a60113fa605663ccfd9c2807ee4c9d0cc1e262c71eb74332e527a`。
- v2 deploy/review 通过：
  - 28个 upload SHA 全部通过；
  - `STAGE2_DOCS_CONTENT_QA_OK files=28 markdown=7 hashes=21`；
  - 8份 evidence Bash 全部 `bash -n`；
  - code commit 后排除 `docs/**` 与 `HANDOFF.md` 时没有其他 diff；
  - `STAGE2_DOCS_DEPLOY_REVIEW_OK`。
- Git status 默认只显示24个 path；缺少的4个是 `.gitignore` 覆盖的
  `LOCAL_SHA256SUMS.txt`、`SHA256SUMS.server.txt`、`resources_before.txt`、
  `resources_after.txt`。它们是 packet 链接的机器证据，不能静默漏掉，因此提交脚本窄改为
  对 upload manifest 的28个精确 path 使用 `git add -f`，并要求 staged name list 与
  manifest 完全相等；没有扩大到目录级强制暂存。
- 实际供未来批准后执行的服务器 `/root/autodl-tmp/tmp` fresh/resume launcher 使用
  `.v3.part` 上传，先检查旧 SHA、新 SHA 和 `bash -n`，再原子替换；monitor 未改。
- 随后运行只做 syntax/hash/root-absence 的 v3 验证：
  - fresh：
    `6e0f1c7ce5497bd3d5a2bef539bbea5e3fc964a5d8259b16f472cf353d19e27a`
  - resume：
    `6494eaeb8cb2e6c1decee07d97798a0aba368ee15b0491faf3ecdfe6a9ff054c`
  - monitor：
    `925cb515a4ecd6dbfcb192168c63644e1b2b2d691f6a4d50fdc3ddd8a5bbd96b`
  - 结果：`STAGE2_LAUNCH_SCRIPTS_V3_OK`。
- v3 检查没有执行 launcher 主体，没有创建 smoke 输出或 runtime evidence；当前仍未提交、
  未 push、未启动 Stage 2 smoke。

### A075 — 首次 docs commit 被 staged whitespace gate 拦截

- 提交脚本先按 upload manifest 对28个精确路径执行 `git add -f`，随后
  `git diff --cached --check` 在 commit 前 exit 2。
- 唯一错误是 Stage 2 packet 顶部两行 blockquote 末尾的 Markdown 双空格：
  - `04_STAGE2_PRE_SMOKE_PACKET_20260729.md:4`
  - `04_STAGE2_PRE_SMOKE_PACKET_20260729.md:5`
- 本地 QA 曾把 Markdown 双空格视为合法 hard break，因此没有提前拦截；Git 的 staged gate
  更严格且正确地成为最终准入标准。
- 处理：
  - 去掉这两个不必要的 hard-break 空格；连续 blockquote 行本身仍保持所需排版；
  - 不 `reset --hard`、不丢弃已 staged 的其余文件；
  - 原子更新 staging/worktree 中的 packet 与本流水账，并同步更新 upload manifest；
  - 重新运行全量内容 QA，再由同一 commit 脚本重新暂存28个精确路径。
- 当前没有形成 commit、没有 push、没有改变算法代码，也没有启动 smoke。

### A076 — Stage 2 packet commit 成功，GitHub 网络发布暂缓

- 去掉两处双空格、原子更新 packet/ledger 与 staging manifest 后，重新执行全量 review：
  - update 后 manifest SHA：
    `84858557cb02a0d3b61ae070035337e5dc985f8e96ccf889d22680ef0ec80d9e`
  - 28个 upload SHA、内容/链接/JSON/YAML/21个 evidence SHA、8个 Bash syntax 均通过；
  - dirty set 精确为28个 manifest path。
- 第二次提交尝试通过 staged diff gate并成功生成：
  ```text
  92e02d9e51c47422696f5ed17a2f15165a6331a6
  docs(rlt): prepare Stage 2 smoke approval
  28 files changed, 3952 insertions(+), 120 deletions(-)
  ```
- 同一脚本中的首次 push 被服务器侧60秒 timeout 终止，exit 124；commit 已存在且
  worktree clean。
- 随后执行一次有界恢复脚本：
  - 开始时 HEAD 相对 upstream left/right 为 `3/0`；
  - 30秒 `ls-remote` 未得到结果；
  - 最多240秒的 push 实际在约129秒报
    `Failed to connect to github.com port 443: Connection timed out`，exit 128；
  - 没有遗留 push 进程，没有改 remote/ref/commit。
- 按本项目此前成功经验再做一次窄网络探针，而不是继续盲重试：
  - `https://github.com`：10秒连接超时，HTTP `000`；
  - `https://api.github.com`：HTTP `200`；
  - 因 Git smart-HTTP 所需主站不可达，脚本返回
    `NETWORK_PROBE_UNAVAILABLE_NO_PUSH`，没有发起第三次 push。
- 当前结论：
  - commit 与全部证据安全保存在服务器独立 branch；
  - upstream publication 暂缓是外部网络状态，不是代码/测试/配置失败；
  - 不安装代理、不改 remote、不输出 credential，也不继续占用连接；
  - 网络恢复后从精确 HEAD/clean/ahead 状态做一次 push 即可，不重做实施或 packet。
- 本节与 `HANDOFF.md` 的 publication 状态作为终端 closeout 追加；为避免“记录最后一次
  记录提交”递归制造无限 commit，本轮最终 closeout commit SHA 由聊天交接给出。
- Stage 2 smoke、Ray、RoboTwin、模型加载和训练仍均未启动。

### A077 — 最终只读审计的 schema 打印失败与通过结果

- A076/HANDOFF closeout 已作为本地服务器 commit：
  `4f3062762043558a22c375eb415e636e08de9369`
  （`docs(rlt): record Stage 2 publication state`）；commit 后 clean、相对 upstream
  left/right `4/0`。
- 第一次最终只读审计在以下项目全部通过后，因摘要打印器 exit 1：
  - branch/HEAD/clean/ahead；
  - 无 RLT/Ray/push 进程；
  - 两卡空闲；
  - host/cgroup/disk；
  - Stage 1 manifest/validation、Stage 2 三份 resolved/audit、三份 launcher/monitor SHA。
- 唯一错误为 Python 读取 `validation.json` 时误用不存在的顶层键
  `all_gates_passed`，触发 `KeyError`。实际 schema 是：
  `accepted`、`gates`、`metrics`、`reload_contract`；这不是 artifact 内容错误。
- 窄修复只改本地只读审计脚本的字段名，没有改 repo、artifact、checkpoint 或配置。
- `2026-07-29T22:47:49+08:00` 复跑得到
  `STAGE2_FINAL_READONLY_AUDIT_OK`：
  - HEAD `4f306276...9369`、clean、left/right `4/0`；
  - 无 RLT/Ray/push 进程；
  - 两张 A800 均 `0 MiB / 0%`；
  - host available `968,499,880 KiB`，约 `923.63 GiB`；
  - cgroup current/anon/file 约 `214.45/0.289/212.54 GiB`，file cache 是主体；
  - cgroup `oom=0`、`oom_kill=0`；
  - `/root/autodl-tmp` available `886,847,287,296 bytes`，约 `825.94 GiB`，使用率56%；
  - Stage 1 accepted/all-gates 均 true，fresh/true/shuffled/zero loss
    `5.1976585388/0.5337553024/1.7118018866/2.1026575565`，
    non-RLT changed `0`；
  - endpoint `22,076,275,790 bytes`，约 `20.56 GiB`；
  - Stage 2 smoke run root 与 runtime evidence root 均不存在。
- 本节是终端审计记录；后续只需将它形成一个 ledger-only 本地 commit。由于 GitHub 主站
  网络已确认不可用，本轮不再 push，也不再运行会改变实验状态的操作。

### A078 — 2026-07-30 fresh-only Stage 2 smoke 授权与执行边界

- 用户明确批准执行一个简洁的 Stage 2 主链 smoke，并冻结本轮范围：
  - 只做 fresh；
  - 省略 resume；
  - 不增加额外实验或大批检查；
  - 尽量在约30分钟内返回；
  - 继续记录 GPU、host/cgroup RAM、磁盘和错误；
  - 同步总结中国大陆服务器的 Git/GitHub 网络经验，形成一份短副本。
- 本轮重新完整读取根 `PROJECT_CONTEXT.md`、`HANDOFF.md`、RLT 当前索引与
  `04_STAGE2_PRE_SMOKE_PACKET_20260729.md`；动态状态仍以新服务器现场为准。
- 算法和正式 batch 不缩：继续使用2 GPU/4 env、H50/C10/D14、frozen Stage 1、
  batch512/128、8 critic/4 actor updates、deterministic student eval 和 DCP1。
- 只调整执行治理：
  - 不运行 resume wrapper；
  - 先做一次短 GitHub 网络探针和有界 push，不重复长时间盲试；
  - fresh driver 目标 wall-clock 约30分钟，若必须修改原90分钟 hard timeout，只复制为
    fresh-only 运行包装并记录新 SHA，不改算法 config；
  - 保留原 artifact/config hash、空闲 GPU/Ray、输出不存在和资源下限 hard gates。
- 当前尚未刷新服务器、尚未创建 smoke 目录或启动进程；下一条先记录 live
  Git/network/process/GPU/RAM/disk/path 现场。

### A079 — fresh smoke 启动前现场与3秒 Git 推送恢复

- 执行只读脚本
  `local_scripts/remote_rlt_20260730_fresh_smoke_live_audit.sh`，现场时间
  `2026-07-30T00:14:24+08:00`。
- Git：
  - branch `codex/rlt-pi0-robotwin`
  - HEAD `6fd3ee7106fb82f06eda82603c41a09767151709`
  - worktree clean
  - 相对 upstream left/right `5/0`
  - upstream/remote head 均为
    `d7c3ca7e2ddfc8d0b3c376ec6d30ba89b965a5dc`
- 网络现场：
  - 所有大小写 HTTP(S)/ALL proxy 环境变量均未设置；
  - Git `http.version` 为默认值；
  - `https://github.com`、API、raw 均 HTTP 200；
  - connect/total 分别约 `0.107/0.903s`、`0.111/0.397s`、
    `0.295/1.659s`；
  - `ls-remote` 正常返回旧 remote head。
- 资源/路径：
  - 无 RLT/Ray/push 进程；
  - 两张 A800 均 `0 MiB / 0%`；
  - host available `1,030,629,608 KiB`；
  - cgroup current/anon/file
    `228,076,175,360 / 309,866,496 / 226,026,766,336 bytes`，
    raw total 几乎全是可回收 file cache；
  - `oom=0`、`oom_kill=0`；
  - `/root/autodl-tmp` available `886,847,029,248 bytes`，使用率56%；
  - smoke run/evidence root 均不存在。
- source formal/smoke config、worker、Stage 1 manifest、fresh launcher 和 monitor SHA
  与批准包完全一致。
- 因主站已经恢复，执行
  `remote_rlt_20260730_bounded_push_before_smoke.sh`：
  - exact HEAD、clean、left/right `5/0`、主站HTTP200 hard gate；
  - `GIT_TERMINAL_PROMPT=0`；
  - push 最长60秒。
- push 实际3秒成功：
  `d7c3ca7e..6fd3ee71`；随后 HEAD=remote、left/right `0/0`。
- 结论：前一轮慢是大陆服务器到 `github.com:443` 的瞬时链路/路由不可达，不是 Git
  历史、认证或大 pack。当前默认直连本身就是已验证的成功路径；不安装代理、不固定
  HTTP/1.1、不改 remote。以后先做7–10秒主站/API/raw探针，仅在主站200时做一次有界
  push，失败即延期，避免把单次连接拖到数分钟。
- 当前满足原 fresh launcher 的 upstream `0/0` 门；尚未创建 smoke 目录或启动 driver。

### A080 — fresh v1 launcher exit127、定位与不覆盖重试准备

- 执行批准包精确命令：
  `bash /root/autodl-tmp/tmp/remote_rlt_20260729_start_stage2_smoke_fresh.sh`。
- artifact preflight 和 resolved compose 已先完成；约22.7秒后 launcher 自身 exit 1，
  原因是启动后2秒的 `kill -0 729402` 发现 driver 已退出，没有误报为“运行中”。
- 只读检查时间 `2026-07-30T00:16:50+08:00`：
  - driver start/finish 都是 `00:16:16+08:00`；
  - `exit_code=127`；
  - driver.log 401 bytes；
  - 无 Ray/RLT 进程；
  - 两卡 `0 MiB / 0%`；
  - smoke run root 内没有模型/checkpoint 文件；
  - monitor 只有表头和两条空闲样本，GPU最高4MiB，`oom/oom_kill=0`。
- 原始错误：
  `timeout` 把
  `python -B ... runner.resume_dir=null`
  整串当成单一可执行文件名，报 `No such file or directory`。
- 原因：outer launcher 在未引用 heredoc 中展开 `"${fresh_cmd[@]}"`；展开结果写入
  `run_foreground.sh` 后不会再次按 shell 参数数组解析，因此命令被合并。
- 窄修复：
  - 不改 YAML、模型、batch、环境、更新数或算法；
  - 只把生成的 `run_foreground.sh` 中 timeout 后的命令改为逐参数显式 Bash 命令；
  - old launcher SHA
    `6e0f1c7ce5497bd3d5a2bef539bbea5e3fc964a5d8259b16f472cf353d19e27a`；
  - fixed launcher SHA
    `473f339af5123802526dae93fe2fde7289fe52f32efb80b581ded073eaabd985`。
- 为避免覆盖失败证据，下一步只在以下 hard gates 通过后：
  - 把失败 run/evidence 分别原子移动到后缀
    `_failed_launcher_127`；
  - 把 old launcher 及 SHA 放入失败 evidence；
  - 对 fixed `.part` 先做 SHA 和 `bash -n`，再原子替换 `/tmp` launcher；
  - 原批准包 run/evidence 路径重新为空后做一次 fresh 重试。
- 当前没有删除失败文件，没有启动第二次 driver。

### A081 — 重试准备 gate 修正与 fresh driver 正常启动

- fixed launcher `.part` 上传后，第一次执行 retry-prep 脚本 exit 1，且没有输出。
- 使用同一无凭证脚本经 `bash -x -s` 只读定位；失败点是
  `test -d /root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1`。
- 原因：exit127 发生在训练程序启动前，旧 launcher 只建立了
  `experiment_exports/.../fresh_runtime`，训练 run root 从未创建。此前 failure inspector
  已输出 `smoke_files=NONE`，但 retry-prep 错把“没有 run root”当成“必须移动 run root”。
- 窄修复：
  - 要求 run root 必须不存在；
  - 不创建伪失败 run root；
  - 只归档实际存在的 evidence root；
  - 其余 exact HEAD/clean/upstream、exit127、错误文本、old/new launcher SHA、
    `bash -n` 和目标不存在 gate 不变。
- 第二次 retry-prep 通过：
  - failed run：`NOT_CREATED`
  - failed evidence：
    `/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1_failed_launcher_127`
  - fixed `/tmp` launcher SHA：
    `473f339af5123802526dae93fe2fde7289fe52f32efb80b581ded073eaabd985`
  - 标记：`RLT_FRESH_RETRY_PREPARED`
- 重新运行同一 fresh 命令后，launcher 正常返回：
  - driver PID `730146`
  - monitor PID `730147`
  - runtime：
    `/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1/fresh_runtime`
  - run root：
    `/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1`
  - expected DCP：
    `.../robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1/checkpoints/global_step_1`
- launcher 返回前 `kill -0` 已通过；当前只称为“已启动”，尚未称为 smoke 通过。

### A082 — fresh smoke 完成、资源与精简 postcheck

- 第一次健康检查在启动后约73秒完成；driver/monitor 仍活跃，日志已进入 RoboTwin/π0
  初始化。两卡当时约947MiB；日志打印 Vulkan fallback 和 Curobo import traceback。
- 没有仅凭 `traceback` 字样停止任务：resolved task 使用 `planner_backend=mplib`，
  且 driver 仍活跃；继续按实际退出码、rollout/eval/update/DCP 判断。
- `2026-07-30T00:25:27+08:00` driver 自行 exit0，monitor 同步退出：
  - started `00:22:45`，总 wall-clock 约162秒；
  - RLinf cycle `58.145s`；
  - train rollout 和 deterministic eval 各完成1 epoch；
  - 保存 `global_step_1`。
- fresh hard contract：
  - global lifetime transitions `8`；
  - replay rank0/rank1 各4 rows；
  - critic/actor updates `8/4`；
  - train `actor_switch_rate=0`；
  - completion manifest `complete=true`、world size2、rank0/1；
  - 两 rank 保存后均为 `update_step=8`、`warmup_transitions=8`。
- metric table 的 `rlt/update_step=0` 是本 cycle 更新前快照；同表报告
  `updates_to_run=8`，DCP completion manifest 的保存后值8是恢复合同的权威值。
- 数值健康：
  - actor/critic loss `2.189/0.017`；
  - actor/critic grad norm `5.055/4.591`，均低于 clip10；
  - CUDA OOM、NCCL fatal、NaN metric、Ray actor death、SIGSEGV 均为0。
- 资源 monitor 73点、2秒间隔、覆盖164秒：
  - GPU峰值 `15,375/15,368 MiB`，util峰值 `96%/93%`；
  - matched RSS峰值 `47,664,296 KiB`（45.46GiB）；
  - cgroup anon峰值 `44,007,514,112 bytes`（40.99GiB）；
  - cgroup file峰值 `226,419,949,568 bytes`（210.87GiB）；
  - host available最低 `1,010,300,000,000 bytes`（940.92GiB）；
  - disk available最低 `886,794,207,232 bytes`（825.89GiB）；
  - cgroup oom/oom_kill 增量 `0/0`。
- checkpoint `52,542,741 bytes`，包含两 rank DCP、8.66MB聚合 actor权重、两 rank
  replay、trainer state、target model；run总计 `52,542,809 bytes`。
- postcheck Bash 由 Windows PowerShell 5.1 写出时带 UTF-8 BOM，远端第一行报告
  `#!/usr/bin/env: No such file or directory`；后续 Bash 内容仍执行并 exit0。该一次性
  helper 随后用 `apply_patch` 去掉 BOM；这不影响服务器运行、checkpoint或上述结果。
- 用户明确要求省略 resume，因此没有调用 resume wrapper，也没有把“DCP完整”误写成
  “新进程恢复继续已通过”。

### A083 — 成功证据下载与上下文收口

- 通过逐连接、固定 host-key 的 Paramiko/SFTP 下载11个小文件到
  `docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_fresh_smoke_20260730/`：
  driver log、resources CSV、resolved config、provenance、Stage1 preflight、
  exact command、起止/exit、resources-before 与 trainer completion。
- 所有密码仍只存在当前 PowerShell 进程的 `SEETA_SSH_PASSWORD`，每组连接结束后在
  `finally` 清除；未下载52.5MB checkpoint。
- 新增：
  - `05_STAGE2_FRESH_SMOKE_RESULT_20260730.md`；
  - `06_AUTODL_NETWORK_PLAYBOOK.md`；
  - fresh evidence README、postcheck summary 和 SHA256SUMS。
- `04_STAGE2_PRE_SMOKE_PACKET_20260729.md` 明确降为历史批准包；RLT 索引与根
  `HANDOFF.md` 改为 fresh通过、resume省略、formal budget待批准。服务器同步、
  文档QA与Git结果另记。

### A084 — closeout 原子同步与服务器 QA

- 本地先完成：
  - 3份 JSON 解析；
  - 11个服务器原始文件 SHA 逐一复算；
  - 9份相关 Markdown 的相对链接检查；
  - Markdown trailing-whitespace 检查；
  - 标记 `RLT_STAGE2_LOCAL_QA_OK markdown=9 hashes=11 json=3`。
- `2026-07-30T00:38:24+08:00` 服务器上传 gate 通过：
  - branch `codex/rlt-pi0-robotwin`；
  - HEAD `6fd3ee7106fb82f06eda82603c41a09767151709`；
  - clean、upstream `0/0`；
  - 无 RLT/Ray进程，两卡 `0MiB/0%`；
  - host available `1,029,260,896 KiB`；
  - disk available `886,794,207,232 bytes`；
  - fresh exit0、completion保存态 `update_step=8`。
- 第一次 staging v1 原子上传21个精确文件，manifest SHA
  `54beb68d7ae21220a02d424ff16bd1984e555706e5ed3b111f44f52eda7726aa`；
  逐文件远端 SHA 通过。
- 第一次 deploy review 在复制后、Git stage/commit前被 whitespace gate 拦截：
  `exact_command.txt:1` 保留原始 shell 命令的行尾空格。该文件是按 SHA 固定的原始机器
  证据，不应为了手写文档风格而改字节。
- 窄修复只调整 review helper：
  - Markdown/JSON/YAML/TSV 继续做 trailing-whitespace gate；
  - 原始 driver log、CSV、command/time/resource文本保持服务器下载字节和 SHA；
  - 重跑时只接受现有 dirty path 是 upload manifest 的子集，仍拒绝任何范围外变化。
- 第二次 deploy review 于 `00:40:39+08:00` 通过：
  - 21个 upload SHA、JSON字段、Markdown links、fresh 11-file SHA均通过；
  - visible dirty 14个，其余是 `.gitignore` 覆盖的原始 evidence，后续只按21个 exact
    manifest path 用 `git add -f`；
  - 标记 `RLT_STAGE2_FRESH_CLOSEOUT_DEPLOY_REVIEW_OK`。
- 为把本节本身纳入同一个最终文档提交，下一步重新生成不可覆盖的 v2 staging 和 manifest，
  再做同一 review、精确 stage、commit与一次有界 push。最终 commit SHA 在聊天交接，
  不为记录自身 SHA 再制造递归 ledger-only commit。
- v2 staging manifest SHA 为
  `8e97b08bd17d05ccabe37e9d11f0d3df2d68d23f10bf88c61e8e486a21c3f36e`；
  `00:42:10+08:00` 第二次全量 deploy review 通过。
- 首次 commit/push helper 已精确 stage 21个 manifest path，但在 commit 前被
  `git diff --cached --check` 拦截：
  - `05_STAGE2_FRESH_SMOKE_RESULT_20260730.md`；
  - `06_AUTODL_NETWORK_PLAYBOOK.md`；
  - `postcheck_summary.json`；
  三者各多一个 EOF 空白行。exit2发生在 commit/push前，所以没有新 commit、没有网络写入。
- 使用 `apply_patch` 只删除三个 EOF 空白行；不改运行证据值或结论。由于本节也需随最终
  commit 保存，下一步生成 v3 staging，复核已有 staged path 仍全部属于同一 manifest，
  再重新 stage exact 21 paths。

### A085 — 主文档提交、push 与最终只读停点

- v3 staging 21文件 manifest SHA：
  `d48f1bafb564f3877b31b8891984bb5faa45c568252cc2a9bbad393aa507a54c`。
- `2026-07-30T00:44:33+08:00` v3 deploy review 通过：
  - 21个文件逐 SHA；
  - 既有21个 staged path 和4个 worktree-visible dirty path 均是 manifest 子集；
  - JSON/link/fresh evidence SHA 与 credential marker 复核通过。
- exact 21 paths 再次 `git add -f` 后，staged path 与 manifest 完全相等；
  text-only `git diff --cached --check` 通过。形成并推送主提交：
  - commit `9bb2dd78feff7133780c3df6a88618d10168c4e4`；
  - message `docs(rlt): record Stage 2 fresh smoke`；
  - `21 files changed, 1717 insertions(+), 40 deletions(-)`；
  - remote head等于本地HEAD，left/right `0/0`。
- 第一次最终只读审计脚本在输出前 exit1。`bash -x` 定位到“无匹配进程”时
  `pgrep | grep | wc` 在 `set -o pipefail` 下仍返回非零；这是审计 helper 逻辑问题，
  不是服务器状态或实验失败。
- 窄修复为 `mapfile < <(pgrep ... || true)` 后，`00:46:17+08:00` 最终审计通过：
  - branch `codex/rlt-pi0-robotwin`；
  - HEAD `9bb2dd78...c4e4`、clean、left/right `0/0`；
  - RLT/Ray进程0，两卡 `0MiB/0%`；
  - fresh exit0、保存态 `update_step=8`、checkpoint `52,542,741 bytes`；
  - formal source `max_steps=0`；
  - host available `1,029,204,604 KiB`；
  - disk available `886,792,486,912 bytes`；
  - 标记 `RLT_STAGE2_FRESH_FINAL_CLOSEOUT_AUDIT_OK`。
- 本节和 HANDOFF 中的主提交身份将形成一次终端 ledger-only closeout；该 closeout
  commit 自身 SHA 只在聊天交接，不再为记录自身制造第三个递归提交。

### A086 — RoboTwin Stage 2 formal 100-cycle 授权与方法边界

- 用户于2026-07-30明确授权启动100-cycle正式 pilot，要求：
  - 确认正式训练健康启动后即返回；
  - 不持续轮询；
  - 记录精确命令、配置、问题、产物与资源监控位置；
  - 次日再按用户请求刷新实验。
- 用户随后追问“为什么是 ManiSkill”。执行侧明确澄清：
  - 本次训练始终是 RoboTwin `adjust_bottle`；
  - ManiSkill 只因用户问“官方训多少步”而做只读来源核实，不进入运行配置；
  - 当前锁定 RLinf ManiSkill YAML 是
    `max_epochs=5000/max_steps=-1`，EmbodiedRunner 每 epoch 为一个 outer step，
    即5000 cycles；它还有64 train env、500 primitive/cycle和不同任务，不能把规模
    直接当成本项目预算；
  - 本项目只 compose/run
    `robotwin_adjust_bottle_rlt_stage2_ac_mlp`，并显式覆盖
    `runner.max_steps=100`。
- RoboTwin formal 冻结项不变：2 GPU、4 train/eval env、H50/C10/D14、accepted
  clean-50 Stage1 artifact、RoboTwin π0 stats、full-task route、compact replay、
  batch512/128、UTD5、critic:actor=2、500 rows/rank、5k critic floor、cap400、
  eval/save每10 cycles。
- 100 cycles 的解释：约第26个满长 cycle 才开始 student control；约第51个 cycle
  越过15k update 的完整 BC/Q ramp。100-cycle pilot 因而比30/60更能观察稳定
  student阶段，但仍远小于 ManiSkill 5000-cycle参考，不能称等规模复刻。
- 当前尚未做服务器写操作；下一步先刷新 branch/dirty/upstream、进程、GPU/RAM/disk、
  Stage1 binding与目标路径，再生成唯一 RoboTwin resolved config和启动包。

### A087 — formal100 启动前服务器现场

- 执行
  `local_scripts/remote_rlt_20260730_formal100_live_audit.sh`；现场时间
  `2026-07-30T01:09:02+08:00`。
- Git：
  - branch `codex/rlt-pi0-robotwin`；
  - HEAD `2df23e7f4b3d19d4f0dedab32168767a32845a58`；
  - worktree clean；
  - upstream-only/head-only `0/1`；
  - 唯一 ahead commit 只改 `docs/**`/`HANDOFF.md`；相对 Stage2 code commit
    `3b610cb4...` 没有任何 code/config diff。
- 进程/资源：
  - 无 RLT/Ray/目标 formal 进程；
  - 两卡 A800均 `0MiB/0%`；
  - host available `1,029,219,316 KiB`；
  - cgroup current/anon/file
    `211,436,257,280 / 310,644,736 / 209,387,597,824 bytes`，raw总量主要是
    可回收 file cache；
  - `oom=0`、`oom_kill=0`；
  - `/root/autodl-tmp` available `886,792,167,424 bytes`。
- 新目标 run/evidence root 均不存在，不会覆盖 smoke、Stage1、DSRL 或其他实验。
- source identity：
  - RoboTwin formal config SHA
    `f089f333839c99b87d546e8bcf0d5bddbb7da380e8cc1597e1de4c4450592850`；
  - RLT worker SHA
    `71cccde9b7f18ab63a10817f75b7d5a4d5f5c8d9cadfef99da20690d327c4766`；
  - artifact preflight SHA
    `3278a8cbdf766d30309856eac2a4eb5f8cc3c792986e230c2ef022b615553bb6`；
  - monitor SHA
    `925cb515a4ecd6dbfcb192168c63644e1b2b2d691f6a4d50fdc3ddd8a5bbd96b`；
  - Stage1 manifest/stats SHA仍为
    `6ca58f26...12433` / `649ed92b...ce4f6a`。
- source formal 继续 `max_steps=0` fail closed；只有本次命令显式覆盖100。

### A088 — formal100 resolved config与启动包

- 执行
  `local_scripts/remote_rlt_20260730_prepare_formal100.sh`，只创建新的 runtime
  evidence，不启动 Ray/训练。
- artifact preflight重新通过；compose使用唯一配置：
  `robotwin_adjust_bottle_rlt_stage2_ac_mlp` 加四个显式 override：
  - `runner.max_steps=100`；
  - 新 run root；
  - 新 experiment name；
  - `runner.resume_dir=null`。
- 完整 resolved config SHA：
  `efff00b71d8ab618f4a77c082cbec8fd65fda9abe2573def31e0aca980e50178`。
- 机器断言：
  - `env_type=robotwin`、task=`adjust_bottle`、Aloha；
  - resolved文本不存在 `UNRESOLVED`，本地副本也不存在 `maniskill`；
  - 100 cycles、eval/save 10、4 train/eval env、200 primitive/cycle；
  - H/C/D=`50/10/14`、z=2048、batch512/128、fp32；
  - UTD5、ratio2、500 rows/rank、5k critic floor、cap400、15k replay/rank；
  - Stage1 model/manifest/stats精确绑定。
- 路径：
  - run：
    `/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1`；
  - runtime：
    `/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime`；
  - experiment：
    `robotwin_adjust_bottle_rlt_stage2_formal_100c_v1`；
  - hard timeout `50,400s`（14小时）。
- Windows 下载6个小型 prelaunch 文件后，第一次本地解析因 Windows Python 无
  `omegaconf` 报 `ModuleNotFoundError`；文件下载本身已完成。改用现有 PyYAML复核
  同一字段并通过，没有安装依赖或改服务器包。

### A089 — launcher 自匹配失败、窄修复与 driver 启动

- 首次执行
  `bash .../runtime/launch_background.sh` 静默 exit1；只读 inspector确认：
  - run root未创建；
  - 无 driver/monitor PID、start/exit/log/resources文件；
  - 无相关进程，两卡仍0；
  - 所以训练没有短暂启动，也没有需要终止的后代。
- 原因：进程 gate 的 pattern 包含 `rlt_stage2_formal_100c`，而 launcher 自己的 argv
  路径也含这段文本，故把自己计为 active process。
- 窄修复只删除 pattern中的实验路径词，保留
  `train_embodied_agent.py|raylet|gcs_server`：
  - old SHA
    `a71c36d46ab8d99b47956efc79400ad4dc8c6cb0c98ec6dc7ac4b79134f89509`；
  - fixed SHA
    `a54c70c2c4691836d50f8663cf29e524ae9f2a57c1aefe80938d4e42c86c8537`；
  - `bash -n`通过，Python逐行比较确认只有第14行变化；
  - old script归档为
    `launch_background.failed_self_match.sh`，另存 failure说明；
  - resolved config、formal command、算法和预算均未改。
- 重跑同一 launcher成功：
  - driver PID `744880`；
  - monitor PID `744881`；
  - started at `2026-07-30T01:15:54+08:00`。
- 43秒首次状态：
  - driver/monitor均alive，Actor/Env/Rollout两 rank均已创建；
  - fatal scan 0、OOM增量0；
  - GPU尚在加载期，峰值 `1,020/995MiB`；
  - host available最低约954GiB，cgroup anon峰约26.6GiB；
  - Curobo/Vulkan提示与已通过的 fresh smoke相同；resolved实际使用MPLib。
- 73秒复核仍在 frozen π0初始化：driver/monitor alive、fatal0、OOM0，anon/RSS继续
  上升说明模型正在装载；尚未把“进程存在”夸大为已进入第一轮 rollout。

### A090 — formal100 健康启动门与停止轮询

- `2026-07-30T01:18:07+08:00`，启动后约132秒再次做一次有界只读检查：
  - driver PID `744880` 与 monitor PID `744881` 均存活；
  - 没有 `finished_at.txt` 或 `exit_code.txt`；
  - log `28,255 bytes / 523 lines`，fatal scan为0；
  - 两个 rollout rank 均加载 RoboTwin stats；
  - 两个 actor rank 的FSDP初始化完成，per-rank replay size 15,000生效；
  - 出现 `Generating Rollout Epochs: 0%`，第一个真实 RoboTwin formal rollout已经开始。
- 该观察窗内资源：
  - 两卡显存样本约 `15,208/15,291MiB`，GPU util峰 `31%/28%`；
  - cgroup anon峰 `50,966,687,744 bytes`（约47.47GiB）；
  - matched RSS峰 `54,293,092KiB`（约51.78GiB）；
  - host available最低 `1,002,230,000,000 bytes`（约933.4GiB）；
  - disk available最低 `886,792,028,160 bytes`（约825.9GiB）；
  - cgroup OOM/oom_kill增量 `0/0`。
- Curobo/Vulkan提示与fresh smoke相同；resolved planner为MPLib，且本次已经越过模型初始化
  进入rollout，不将其视为fatal。
- 通过SFTP只下载启动时间、driver/monitor PID和launcher失败说明；密码只注入当前
  PowerShell进程并在 `finally` 清除。增长中的driver log/resources CSV继续留在服务器，
  避免把早期快照误作最终结果。
- 启动记录与小型证据索引：
  - `docs/rlinf-robotwin-pi0-rltoken/07_STAGE2_FORMAL_100C_LAUNCH_20260730.md`；
  - `docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_formal_100c_20260730/`。
- 按用户要求，健康启动门通过后停止持续轮询；下一次仅在用户请求时刷新现场。

### A091 — formal100 本地证据QA与交接收口

- 第一次PyYAML合同检查错误地把嵌套字段写成 `env.env_type`，只读检查报
  `KeyError: env_type`；查看resolved结构后改为权威路径
  `env.train.env_type` 与 `env.train.task_config.task_name`，未修改配置本身。
- 修正后的本地QA：
  - `launch_health_summary.json` 与 `stage1_binding_preflight.json` 均可解析；
  - resolved为 `runner.max_steps=100`、RoboTwin、`adjust_bottle`；
  - resolved不存在 `UNRESOLVED` 或 `maniskill`；
  - `SHA256SUMS.txt` 的10个服务器原始/下载文件逐一复算通过；
  - HANDOFF、专题索引和formal启动记录的相对Markdown链接全部存在；
  - 5个新增/更新的交接Markdown均为UTF-8、无BOM、无行尾空白。
- 本地Git由不同Windows SID拥有；第一次 `-c safe.directory` 使用反斜杠未匹配，
  改用 `C:/Users/86136/Documents/rl` 后只读命令正常。该本地文档树在当前Git视图中为
  untracked，因此没有把空的 `git diff --check` 当作新文件文本QA，改用上述显式逐文件检查。
- 未stage、commit或push本地文档；未再访问增长中的服务器日志，也未停止/改变正式进程。

### A092 — formal100 完成态刷新授权与上下文重载

- 用户于 2026-07-30 要求只读查看 formal100 的实验进展/产物/训练指标/资源指标与
  可视化，并追问：
  - 两卡并行是否顶满；
  - RLinf ManiSkill 训练量换算到本项目约多少；
  - 100 cycles 是否太少，或 RLT 的采样效率是否已足够；
  - 是否有成功率指标，训练效果如何。
- 按根 `AGENTS.md`，重新完整读取 `PROJECT_CONTEXT.md`、`HANDOFF.md` 和当前 RLT
  单一事实源 `00_INDEX_AND_IMPLEMENTATION_PLAN.md`；按需读取 ledger 的 formal
  A086–A091、正式 resolved config、RLinf 锁定的 ManiSkill YAML 和 worker schedule。
- 同时核对官方 RLT 方法页、论文摘要和 RLinf RLT 文档。外部来源只用于方法语义和公开
  参考量级；本次动态状态、实际 counters 与资源仍以服务器现场和日志为准。
- 本轮授权是只读审计和本地文档/图表整理；没有授权启动 eval、续训、resume、删除产物
  或修改服务器 Git，因此这些动作均未执行。

### A093 — 服务器完成态只读审计

- 新增只读脚本
  `local_scripts/remote_rlt_20260730_formal100_status_audit.sh`，通过既有
  `local_scripts/remote_exec_autodl.py` 执行。密码仍只注入当前 PowerShell 进程的
  `SEETA_SSH_PASSWORD`，Paramiko 使用固定 host key、`look_for_keys=False`、
  `allow_agent=False`，并在 `finally` 清除环境变量。
- 精确调用结构：
  ```powershell
  python local_scripts/remote_exec_autodl.py `
    --script local_scripts/remote_rlt_20260730_formal100_status_audit.sh
  ```
- 观察时间 `2026-07-30T09:54:24+08:00`。现场结果：
  - started `01:15:54+08:00`、finished `03:47:38+08:00`、`exit_code=0`；
  - driver/monitor PID已正常退出，无目标RLT/Ray进程，两卡均 `0MiB/0%`；
  - log `923,824 bytes / 6,049 lines`，进度 `Global Step 100/100`；
  - CUDA OOM、NCCL fatal、Ray actor death、NaN metric均为0；
  - 两条Curobo/Vulkan traceback来自既知非致命初始化探测，实际MPLib run已完成；
  - branch `codex/rlt-pi0-robotwin`，HEAD
    `2df23e7f4b3d19d4f0dedab32168767a32845a58`，worktree clean，
    upstream/head-only为 `0/1`，领先项仍为既有docs-only commit；
  - 10个 `global_step_{10,20,...,100}` completion manifest全部
    `complete=true`、world size 2；
  - 最终 rank replay为3,921/3,900 rows，合计7,821；
  - 最终 trainer state为400 train episodes、`update_step=34,800`，
    warm-up anchor为56 episodes/1,043 transitions。
- checkpoint completion中的 `update_step` 序列为
  `0, 2,800, 6,800, ..., 34,800`；student首次在cycle27接管，完整BC/Q ramp在
  cycle52结束。

### A094 — 产物枚举第一次过宽、窄化复查与cgroup查询

- 第一次只读 artifact inspector 递归枚举整个 checkpoint，意外打印了数千个 trajectory
  文件并尝试展开很大的 replay metadata；终端输出被截断，末尾一个 `awk` 片段语法错误
  exit2。该命令没有写服务器、没有改变run或进程，但信息密度差，不能作为最终证据。
- 修复方法：改用窄 Python inspector：
  - 排除 trajectory payload 的逐文件打印；
  - 只统计文件数、字节数和扩展名；
  - 对每个 checkpoint 只读取小型 completion manifest；
  - 对step100只打印非trajectory文件和两rank metadata/state摘要。
- 窄化结果：
  - run总计 `1,541,864,422 bytes`；
  - 最终checkpoint `237,293,208 bytes`；
  - 最终trajectory/replay `184,948,025 bytes / 7,823 files`；
  - 全run `.pt` 共42,931个，主要是各checkpoint replay snapshot；
  - media image/video文件数0。
- 第一次单独查询 `memory.max` 时，PowerShell 到远端CLI的引号传递导致参数被拆开，
  helper报 `unrecognized arguments`，没有产生远端状态变化。改用 PowerShell
  here-string 传脚本文本后查询通过，`memory.max=257698037760 bytes=240GiB`。

### A095 — 有界证据下载、TensorBoard/资源解析和统计结果

- 新增
  `local_scripts/download_rlt_formal100_status.py`，通过同一受控SFTP连接只下载：
  完整driver/metrics日志、TensorBoard event、2秒资源CSV、exit/finish、10份completion
  manifest、step100两rank trainer state和replay metadata。没有下载checkpoint权重或
  7,823个trajectory文件。
- 下载目标：
  `docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_formal_100c_20260730/status_20260730_0954/`。
- 新增并执行：
  ```powershell
  python local_scripts/analyze_rlt_formal100_status.py
  ```
  脚本使用TensorBoard event和资源CSV生成：
  - `status_summary.json`；
  - `selected_scalars.csv`；
  - `resources_30s.csv`；
  - `visual_data.json`；
  - 四张PNG：success/schedule、optimization、resource、checkpoint growth。
- 成功率统计：
  - train `30/400=7.5%`；
  - reference控制阶段 `10/104=9.62%`；
  - student阶段 `20/296=6.76%`；
  - student前/后20 cycles为 `3/80=3.75%` / `10/80=12.5%`；
  - deterministic eval十次依次为
    `0,0,0,0,0,1/4,2/4,0,0,0`，合计 `3/40=7.5%`。
- 优化统计：
  - actor loss首/末 `0.5913/0.00320`；
  - critic loss首/末 `0.002409/0.000502`；
  - BC loss首/末 `0.08452/0.009321`；
  - actor/critic grad峰值 `3.456/0.942`，均低于clip10；
  - 全部87个更新点finite。
- 资源CSV共4,004个样本，覆盖9,104秒：
  - GPU显存峰17,543/17,626MiB；
  - active mean util 25.59%/26.17%，两卡峰值均100%；
  - matched RSS峰51.78GiB，env-worker RSS峰23.59GiB；
  - cgroup anon/file/current峰47.47/195.05/约240GiB；
  - `memory.events` 的max增量23,372，high/OOM/OOM-kill增量0/0/0；
  - host available最低933.40GiB；
  - disk available减少约1.483GiB，与run产物量一致。
- 判断：两卡使用对称但GPU没有顶满，主要等待rollout/simulator；同时cgroup曾被可回收
  file cache顶到240GiB上限，因此不能仅凭显存余量直接增加env并行。

### A096 — ManiSkill量级对照与效果解释

- 重新读取锁定源码
  `.research-rlinf/examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml`：
  5,000 outer cycles、64 train env、256 eval env、500 primitive steps/env/cycle、
  C=10、10k replay rows/rank、30k critic floor、cap400。
- 不同口径的只读量级计算：
  - outer cycles：ManiSkill为本次50倍；
  - train env-cycles：`5000*64 / (100*4)=800`倍；
  - 最大primitive/action slots和macro chunks约2,000倍；
  - 仅按cap推导的critic/actor update上界相对本次约57.5/28.7倍。
- 这些不是等价训练预算：ManiSkill是64-way GPU simulator、不同任务、critical-phase
  route和macro-UTD1；本项目是4 env RoboTwin、full-task route和macro-UTD5。
- 效果结论保持两层：
  - 工程/数值：健康，且已有学习信号；
  - 科学效果：未验收。cycle70的2/4和endpoint100的0/4都只有4条，不能选best或判断
    稳定退化；train与eval也不是严格A/B。
- 建议的下一道门是同一组至少20个固定seeds，独立评估frozen reference、
  checkpoint70和checkpoint100。该建议没有在本轮执行。

### A097 — 可视化、报告、索引与证据收口

- 使用visualization工作流生成一个自包含的内联结果面板：
  `E:/Codex/home/visualizations/2026/07/28/019fa752-79fd-7de3-b66f-5ac4f0a72bfc/rlt-formal100-overview.html`。
- 新增正式完成报告：
  `docs/rlinf-robotwin-pi0-rltoken/08_STAGE2_FORMAL_100C_RESULT_20260730.md`。
  报告逐项回答成功率、训练健康、并行利用、RAM/cgroup、产物、ManiSkill量级与
  “100 cycles是否足够”。
- evidence目录新增 `README.md` 与 `SHA256SUMS.txt`；后者固定29个下载/派生文件的
  SHA-256，README和SHA文件自身不递归写入清单。
- 根 `HANDOFF.md` 与RLT唯一索引更新为完成态：formal100 exit0；下一道门为
  reference/ckpt70/ckpt100固定种子复评，尚未授权或启动。
- 本轮没有stage、commit或push这些本地文档，也没有向服务器写入新文件。

### A098 — 图表目视检查与最终本地QA

- 逐张使用本地图片查看器检查四张PNG：
  - 成功率图的reference/student边界、cycle27切换、cycle52 ramp结束和10个eval点可读；
  - 优化图的log loss、Q值和clip10梯度线可读；
  - 资源图能同时看出两卡对称、GPU间歇利用、RSS阶梯和240GiB cgroup上限；
  - checkpoint图能看出10个点线性增长且由replay主导。
- visualization render第一次把standalone目标写到 `C:/tmp`，当前Windows ACL返回
  `PermissionError`；只影响本地预览文件，没有影响fragment或项目文件。改写到获准的
  visualization目录后成功生成46,269-byte standalone包装，并确认fragment只有一个根节点、
  无外层`html`标签、script开闭匹配。
- 第一次Markdown文本QA发现完成报告首个blockquote行保留两个hard-break空格；
  `TRAILING_WHITESPACE` gate正确拦截。使用 `apply_patch` 去除该非必要空格后复跑。
- 最终命令级QA结果：
  ```text
  RLT_FORMAL100_LOCAL_QA_OK files=5 hashes=29 report_links=5 json=ok html_render=ok
  ```
  覆盖UTF-8无BOM、无行尾空白、29个SHA逐一复算、summary JSON、报告5个相对链接、
  `git diff --check` 和内联可视化render。
- 没有stage、commit或push；服务器run、checkpoint和进程状态保持只读。

### A099 — 2026-07-30 formal规模纠正与本轮授权边界

- 用户指出：上次把100-cycle运行作为正式训练批准时，没有完整对齐论文、ManiSkill示例
  和RoboTwin实际训练规模，属于必须纠正的工作失误；后续formal应尽量一次拿到完整结果，
  checkpoint间隔也必须随总预算缩放，不能撑爆存储。
- 用户同时要求解释：
  - 成功率图每个元素；
  - 每cycle的rollout数量；
  - env并行能否从4增到8；
  - 195GiB file cache是否虚高/可清；
  - 每50 cycles评20条是否合理；
  - cycle和既有step口径的区别。
- 本轮授权是本地文档设计与只读来源复核；没有授权服务器代码修改、资源smoke、新formal、
  resume、删除或覆盖产物，因此未访问或改变服务器动态状态。
- 纠正规则写入根 `PROJECT_CONTEXT.md`：任何formal启动前必须批准cycles、episodes、
  action slots、transitions、updates、eval、checkpoint、wall-clock和GPU-hours多轴表；
  否则只能叫pilot，不能用“睡一晚”定义训练量。

### A100 — 上下文、图表和调用链只读复核

- 按工作区规则完整读取根 `PROJECT_CONTEXT.md`、`HANDOFF.md` 和RLT唯一事实源
  `00_INDEX_AND_IMPLEMENTATION_PLAN.md`，再按需读取：
  - `08_STAGE2_FORMAL_100C_RESULT_20260730.md`；
  - `evidence/IMPLEMENTATION_LOG.md` 的A086–A098；
  - Stage2 source config；
  - 锁定的RLinf ManiSkill YAML、runner、env worker和RLT policy worker。
- 使用本地图片查看器检查用户附图
  `C:/Users/86136/AppData/Local/Temp/codex-clipboard-2531812a-a330-40c2-b9d2-28c7238a060f.png`。
- 源码结论：
  - `cycle`就是runner一次`global_step`，顺序为sync→collect→replay ingest→train→
    optional eval/save；
  - formal100每cycle发起一次4-env batched rollout，每env一条episode，共4条；
  - 每env最多200 action slots、C10，因此最多20 macro rows/env、80 rows/cycle；
  - 实际100 cycles为7,821 rows，低于8,000上限是episode提前done；
  - eval每10 cycles做4条deterministic episodes，红色stem不是误差条；
  - 蓝线10-cycle rolling完整窗口是40条train episodes；
  - 紫/橙虚线分别是cycle27 student接管和cycle52 BC/Q ramp完成。

### A101 — 论文、RLinf ManiSkill和RoboTwin预算口径纠正

- 复核官方一手来源：
  - RLT论文HTML：`https://arxiv.org/html/2604.23073`；
  - RLinf RLT文档：
    `https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html`；
  - Physical Intelligence项目页：`https://www.pi.website/research/rlt`。
- 关键更正：
  - 论文公开训练量是每任务约400–1000 episodes、macro-UTD5、critic:actor=2；
  - formal100有400条train episodes，处在论文episode范围下界，不能仅凭
    `5000/100=50`说它科学上少50倍或完全无效；
  - 但旧run的长期schedule被压缩、稳定student段很短、每点eval仅4条，并且启动前没有
    完成多轴规模审批，所以永久降级为pilot。
- 锁定ManiSkill YAML的5000是runner outer cycles，且每cycle为64 env×最多500 control
  slots；它还有critical-phase gate、可能expert takeover、macro-UTD约1和ratio4。
  当前RoboTwin是full-task、无人/无expert、UTD5、ratio2，不能机械按一个字段等价。
- 字面串行对齐是RoboTwin `max_steps=5000`，但4/8 env会产生20k/40k train episodes，
  预计约5天或6–8天；匹配ManiSkill episode/action容量还会分别需要80k/200k cycles。
  这些互相冲突，证明不存在唯一“等价step”。

### A102 — 下一次完整formal预算与并行设计

- 独立做三路只读复核：
  - ManiSkill outer-cycle/episode/macro/update/eval精确语义；
  - RoboTwin formal100真实采样、并行、资源和checkpoint线性外推；
  - 完整phase覆盖所需的schedule/cycle预算。
- 冻结为同一2,000-episode科学预算的两个实现：
  - 首选：8 env×250 cycles、cap1600、eval/save interval25；
  - 资源门失败回退：4 env×500 cycles、cap800、eval/save interval50。
- 两者共同预计：
  - 最大400k action slots；
  - 约39.1k full-task macro rows；
  - 约125k/62.5k critic/actor updates；
  - 约820条student-controlled episodes；
  - 10个periodic eval点×20条；
  - 10个full checkpoints。
- 长schedule恢复为10k replay rows/rank、30k post-collect update floor和20k/50k
  actor weight warm-up/ramp；保持论文UTD5/ratio2及既有H50/C10/D14。
- cap选择依据：
  - 4 env稳态约391 desired updates/cycle，cap800同时维持UTD5并清理30k floor debt；
  - 8 env稳态约782，cap400会把有效UTD降到约2.56，cap800只维持稳态，cap1600才同时
    清理启动debt。
- “从头”定义为Stage2 actor/Q fresh、empty replay、counter/anchor归零和
  `resume_dir=null`；继续复用已验收Stage1 step2000 artifact/stats/adapter，不重训Stage1。

### A103 — 评估、checkpoint与内存设计

- 用户建议的“降低频率、每点20条”被采用：
  - 4 env每50 cycles或8 env每25 cycles，均为每200条train episodes评一次；
  - 10个监控点共200条；
  - 正式效果另做reference/endpoint相同52个held-out seeds的paired eval；
  - periodic结果不用于挑primary endpoint。
- 发现当前fixed-reset机制不能仅靠`rollout_epoch=5`保证20个不同seeds；正式前需增加
  config-opt-in的4 env×5 epochs seed-bank/epoch-offset，并验证20 unique IDs、分母20和
  seed-bank SHA。长期常驻20个eval env因RAM风险不作为首选。
- checkpoint保存固定为总cycle的1/10：4env方案每50、8env方案每25，恰好10个。
  按formal100 replay大小外推，相同39.1k rows的最终full checkpoint约0.9GiB、10个累计
  约5.2GiB、约21.5万replay小文件。预计每rank19.5k rows低于50k窗口，不依赖hard
  eviction；禁止临时延长预算。
- 195GiB file cache是真实cgroup page cache但多数可回收，不等于进程RSS；anon/RSS峰为
  47.5/51.8GiB，host available最低933GiB且OOM0。`memory.events:max +23372`说明确实
  触达240GiB并发生回收，不能完全忽略，也不应手工`drop_caches`。
- 8-env只通过资源smoke决定：监控GPU、env RSS、anon斜率、PSI、memory events、
  macros/s、updates/s、cycle wall-clock、20-seed eval峰值和save成本。

### A104 — 文档、长期记忆与当前停点

- 新增
  `09_STAGE2_NEXT_FORMAL_SCALE_DESIGN_20260730.md`，集中记录本轮所有解释、来源和下一次
  formal候选，不把新计划散落进旧100-cycle报告。
- 更新：
  - 根 `PROJECT_CONTEXT.md`：formal多轴规模审批稳定规则；
  - 根 `HANDOFF.md`：100-cycle降级为pilot、新formal候选和未授权停点；
  - RLT `00_INDEX_AND_IMPLEMENTATION_PLAN.md`：v14路由、批准门和当前停点；
  - `08_STAGE2_FORMAL_100C_RESULT_20260730.md`：标题/预算口径勘误与新设计链接。
- 按用户“精简记住”要求新增Memory扩展注记：
  `extensions/ad_hoc/notes/20260730-104619-formal-scale-approval-gate.md`；只记录跨项目
  formal规模硬门和本run永久归类为pilot，不保存动态服务器事实。
- 当前没有服务器写操作，没有启动8-env smoke或训练，没有stage/commit/push。

### A105 — 本地文本与链接QA

- 第一次QA发现新设计文档第3行保留Markdown hard-break的两个尾随空格；gate输出
  `QA_BAD ...:3`。使用`apply_patch`去掉尾随空格，并把RLT索引中两条历史入口从
  “formal 100-cycle”改为“历史100-cycle pilot”；没有改变历史服务器目录名。
- 复跑PowerShell QA，覆盖6个变更文档：
  - UTF-8无BOM；
  - 无行尾空白；
  - 相对Markdown `.md` 链接全部存在；
  - 新设计、旧pilot勘误、索引、HANDOFF和账本关键词一致；
  - `git diff --check`没有报告错误。
- 最终输出：
  ```text
  RLT_NEXT_FORMAL_LOCAL_QA_OK
  NEW_DOC_LINES=300
  ```
- 根文档树在当前Git视图中仍为用户既有untracked结构；本轮没有stage、commit或push。

### A106 — 8-env × 250-cycle formal授权与边界

- 用户明确选择首选方案 `8 train env × 250 cycles`，其余已冻结设计由实现者按现有
  09号设计文档执行，并授权：
  - 实现20条周期评估的最小适配；
  - 运行一次克制的8-env资源门；
  - 资源门通过后启动fresh Stage 2 formal；
  - 健康启动后停止轮询，等待用户下一次主动检查。
- 仍然不授权删除/覆盖既有实验，也不把8-env失败静默改成4-env；资源门失败时必须停住
  报告。
- 用户重新提供AutoDL登录口令。口令只注入当前受控Paramiko进程，设置
  `look_for_keys=False`、`allow_agent=False`；未写入脚本、文档、Git、日志、Memory或
  命令行。本账本只记录凭据治理，不记录秘密本身。

### A107 — 上下文恢复、计划与并行只读审阅

- 按工作区规则完整恢复根 `PROJECT_CONTEXT.md`、`HANDOFF.md`、RLT唯一事实源和实施账本，
  并读取09号正式规模设计；历史100-cycle只作为pilot测量依据。
- 工作计划冻结为六步：
  1. 服务器现场刷新；
  2. 20条固定评估适配与测试；
  3. 8-env资源smoke；
  4. 冻结正式resolved packet；
  5. 启动并观察至少一个完整cycle；
  6. 更新账本、交接与启动报告。
- 两路并行只读源码审阅得到一致结论：
  - formal预算算术闭合；
  - 现有 `4 env × rollout_epoch=1 × fixed=true` 只能真实评4条；
  - 不能只把epoch改成5，否则同4个reset ID重复5次；
  - 无需修改通用EnvWorker/RoboTwinEnv，使用RLT专属exact-20 seed bank和现有游标轮转
    即可保持legacy调用不变。

### A108 — 凭据环境缺失与进程内恢复

- 第一次连接准备检查发现本地当前进程没有上轮临时SSH口令；没有尝试猜测、读取磁盘或把
  口令写进文件，因而先向用户报告阻点。
- 用户重新提供后，在持久Node子进程环境对象中设置临时
  `SEETA_SSH_PASSWORD`，所有远端调用继续走：
  ```text
  python local_scripts/remote_exec_autodl.py run --command-file <script>
  python local_scripts/remote_exec_autodl.py put <local> <remote>
  ```
  口令没有出现在这些命令参数中；任务结束前清空进程内变量。

### A109 — 2026-07-30 11:02服务器只读刷新

- 新增并执行：
  ```text
  local_scripts/remote_rlt_20260730_formal250_readonly_refresh.sh
  ```
- 现场结果：
  - repo `/root/autodl-tmp/RLinf_rlt_pi0_robotwin`；
  - branch `codex/rlt-pi0-robotwin`；
  - HEAD `2df23e7f4b3d19d4f0dedab32168767a32845a58`，相对upstream
    left/right=`0/1`，多出的提交只含文档；
  - worktree clean，无真实RLT/Ray训练进程，两卡 `0 MiB/0%`；
  - cgroup current约193.97GiB，其中anon约0.29GiB、file cache约191.98GiB，
    PSI为0、OOM为0；
  - `/root/autodl-tmp` 可用约824GiB；
  - Stage 1 endpoint与manifest存在，manifest SHA仍为
    `6ca58f26...12433`；
  - 新smoke/formal目标目录均不存在。
- 服务器原文件hash也被固定，后续写入均以hash和clean-tree双重guard拒绝漂移。

### A110 — canonical服务器快照、检索小问题与评估方案

- 通过SFTP只读下载服务器canonical快照到：
  `.tmp/formal250_server_snapshot/`，包括source config、RoboTwin env和seed test；没有用
  本地较旧worktree文件覆盖服务器。
- 第一次本地Git检查把 `safe.directory` 参数按PowerShell错误组合，命令没有修改文件；
  改成正确参数形式后取得A109的HEAD/clean结论。
- 第一次 `rg` 使用PowerShell不支持的路径glob，返回Windows路径语法错误；改为
  `rg --glob ... local_scripts` 后正常定位既有compose/test/launch脚本。
- 源码事实：
  - PPO配置使用官方 `eval_seeds.json`、128个固定eval env，一次已覆盖大量独立seed；
  - Fast-WAM GRPO专属配置为2 eval env、`fixed=false`，评估口径不同；
  - 本formal要求每点20条但只开4 eval env，因此使用
    `auto_reset=false/use_fixed_reset_state_ids=false/rollout_epoch=5`，让每个epoch显式
    reset并推进seed游标；
  - 两个EnvWorker各分得10个互不重叠seed，5个epoch后游标恰好回绕，下一评估点复用同一
    20条paired bank。
- 该变更只影响新RLT overlay的评估；训练仍使用原 `train_seeds.json`，没有改变RL采样。

### A111 — exact-20 seed bank与两份opt-in配置

- 使用 `apply_patch` 新增本地上传材料：
  - `eval_seeds_adjust_bottle_rlt_periodic20_v1.json`；
  - `robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250.yaml`；
  - `robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env_resource_smoke.yaml`。
- seed bank的20个ID严格来自官方 `adjust_bottle` eval bank前20个：
  `Count=20`、`Unique=20`、`MatchesOfficialFirst20=True`；SHA-256为
  `fb9c3353e27b83aad6fe7ff778437d960b084de9d981c2af68615d52769952a7`。
- formal overlay只覆盖已批准差异：
  `250/25/25` cycles/eval/save、8 train env、cap1600、10k rows/rank、30k critic
  floor、20k/50k actor schedule、50k replay window/rank和20-seed eval。
- smoke overlay继承formal完整拓扑、batch、replay、模型和评估，仅把串行门禁缩为3 cycles、
  readiness 1 row/rank、floor/cap1600、末轮eval/save。
- 第一次PowerShell把不同对象类型连续送入默认table formatter，导致hash行显示为空；
  文件未受影响。改为逐行字符串输出后取得四个明确hash。

### A112 — 服务器guarded安装与连续seed游标测试收紧

- 新增并执行：
  ```text
  local_scripts/remote_rlt_20260730_formal250_prepare_upload_dir.sh
  local_scripts/remote_rlt_20260730_formal250_apply_seed_eval_config.sh
  ```
- 四个文件先SFTP到
  `/root/autodl-tmp/tmp/rlt_formal250_upload_20260730`；安装脚本逐项断言：
  branch/HEAD、clean tree、原test/base config hash、新目标不存在和上传件hash，然后才用
  `install -m 0644` 写入RLT worktree。
- 初版测试已覆盖20 unique、worker不重叠和两事件相同；审阅时发现“两事件相同”是两次
  重建helper，尚未直接证明游标跨事件回绕。随即用 `apply_patch` 把helper改成让同一
  `current_seed_index` 连续运行两个event，再经SFTP和双hash guard更新服务器test。
- 最终test SHA-256：
  `4d764fc0a0e042e1b71a93d98d5e9a2627e6ed8a899c9dee8a92528303ad3c0f`。

### A113 — 集中单测、compose与legacy不变验证

- 新增并执行：
  ```text
  local_scripts/remote_rlt_20260730_formal250_validate_configs.sh
  ```
- 脚本实际运行：
  - `py_compile` 新seed test；
  - `ruff check` 新seed test；
  - `pytest -q test_robotwin_seed_partition.py test_robotwin_rlt_contract.py`；
  - 原生compose/resolve legacy base、formal 8×250和resource smoke；
  - OmegaConf逐项断言预算、H/C/D、UTD、ratio、schedule、replay、batch、Stage 1绑定、
    20-seed路径/数量/hash和legacy默认；
  - `git diff --check`。
- 结果：
  - `27 passed in 8.90s`；
  - seed bank `20/20 unique`；
  - legacy/formal/smoke resolved SHA-256分别为
    `c0ae5510...9dcd61`、`4fd111af...9dd8d`、`eb38e0bf...32ce5`；
  - `FORMAL250_CONFIG_VALIDATION_OK`。
- stderr仅有既有Ruff preview提示、TensorFlow/oneDNN信息和本地Paramiko旧Blowfish弃用提示，
  无测试/compose错误。此时服务器只有预期四项diff，正式训练尚未启动。

### A114 — 服务器提交与有界Git网络处理

- 新增并执行：
  ```text
  local_scripts/remote_rlt_20260730_formal250_commit_eval_config.sh
  local_scripts/remote_rlt_20260730_formal250_bounded_push.sh
  ```
- 服务器提交成功：
  ```text
  46a2d19b feat(rlt): add 8-env formal evaluation protocol
  ```
  完整HEAD为`46a2d19bae629eaa57830f5faeac71ac81a1a494`，worktree clean。提交仅含
  两份RLT专属overlay、独立20-seed bank和seed游标单测，没有修改通用运行时代码。
- push严格遵循大陆网络短探针：先探测`github.com`，9秒后HTTP000、curl exit28；
  脚本在真正push前有界退出，没有反复重试。结论只是当时GitHub线路不可达，不归因为仓库、
  认证或pack。服务器分支相对upstream ahead2；本次formal精确使用上述clean本地HEAD，
  远端push延后。

### A115 — 8-env资源门packet与进程gate窄修复

- 新增并依次执行：
  ```text
  local_scripts/remote_rlt_20260730_resource_smoke_8env_prepare.sh
  local_scripts/remote_rlt_20260730_resource_smoke_prepare_diagnose.sh
  local_scripts/remote_rlt_20260730_process_gate_probe.sh
  ```
- prepare在创建evidence前连续三次fail closed。原因不是已有训练，而是Paramiko远端
  `bash -c`命令文本本身包含`train_embodied_agent.py`，最初的`pgrep/ps`模式把当前
  准备shell误认为训练进程；加入self-PID过滤后，process-substitution子shell仍会命中。
- 最终gate改为：
  ```text
  ps -eo pid,comm,args |
    awk '$2 ~ /^python/ && $0 ~ /train_embodied_agent[.]py/'
  pgrep -ax raylet
  pgrep -ax gcs_server
  ```
  只把Python训练进程或exact-name Ray服务视为冲突。探针证明自身shell不再命中；
  三次失败均在写evidence和启动Ray前发生，没有残留run。

### A116 — 首次smoke launcher exit127与保留证据

- 新增并执行：
  ```text
  local_scripts/remote_rlt_20260730_resource_smoke_8env_launch.sh
  local_scripts/remote_rlt_20260730_resource_smoke_failed_launch_audit.sh
  local_scripts/remote_rlt_20260730_resource_smoke_archive_failed_launcher.sh
  local_scripts/remote_rlt_20260730_resource_smoke_launcher_syntax_check.sh
  ```
- 第一次launcher在训练程序启动前exit127，run root不存在。原因是Bash数组在heredoc中
  被扩展成一个带空格的单一“可执行文件名”，不是模型、配置或资源故障。
- 失败evidence保留到：
  `/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1_failed_launcher_127`。
- 窄修复为把每个argv逐项写入`exact_command`与运行脚本；随后`bash -n`和命令段检查
  通过。算法、resolved config、预算与hash均未改变。

### A117 — 8-env资源smoke正式执行

- 修复后重新执行同一launcher，并用以下只读脚本检查：
  ```text
  local_scripts/remote_rlt_20260730_resource_smoke_live_audit.sh
  local_scripts/remote_rlt_20260730_resource_smoke_tensorboard_progress.sh
  local_scripts/remote_rlt_20260730_resource_smoke_tb_selected.sh
  local_scripts/remote_rlt_20260730_resource_smoke_artifact_probe.sh
  ```
- run root：
  `/root/autodl-tmp/experiments/rlt_stage2_resource_smoke_8env3c_20260730_v1`；
  runtime：
  `/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime`；
  resolved SHA-256：
  `bbcdcdcb22ca93f106dea0786a57086a2e8317caaefd610d18cabe6c4e2ff6aa`。
- 运行从11:17:27到11:33:33，共966秒，exit0；完成3 cycles、24 train episodes、
  472 macro transitions、3200/1600 critic/actor updates。末轮exact-20
  deterministic eval为`2/20=10%`。
- `global_step_3` checkpoint completion=true、world size2、两rank
  `update_step=3200`。actor loss为`0.244/0.102/0.0425`，critic loss为
  `9.33e-4/1.89e-4/3.31e-4`；actor/critic grad峰3.048/1.022，均finite且低于clip10。
- console出现既有可选Curobo导入`curobo.types.math`提示；env worker、真实train/eval、
  optimizer和save均成功，因此记录为非阻塞环境提示，不为它扩大依赖或修改算法。

### A118 — 资源门postflight与8-env放行

- 执行：
  ```text
  local_scripts/remote_rlt_20260730_resource_smoke_postflight.sh
  ```
  生成
  `/root/autodl-tmp/experiment_exports/rlt_stage2_resource_smoke_8env3c_20260730_v1/runtime/postflight_summary.json`；
  其正式packet副本SHA-256为
  `26c0152e4c31cfa902b28ff262ef791477ff963f07692a966960448dc364f8b9`。
- 两卡显存峰17,169/17,252MiB、利用率峰均100%；matched RSS峰51.88GiB、
  env RSS峰15.07GiB、cgroup anon峰47.47GiB、file峰192.19GiB、
  current峰240.00GiB，host available最低934.03GiB。
- `memory.events:max`增加7,867，说明file cache确实触发回收；但high/OOM/OOM-kill增量
  均为0、最终PSI为0，anon与RSS没有失控。没有手工`drop_caches`。根据预先冻结的门禁，
  8-env通过，不触发4-env回退。
- run总大小63,548,290 bytes。静态机器合同和运行时`num_trajectories=20`共同证明每次
  周期评估使用20条，不把`2/20`当作正式性能结论。

### A119 — 250-cycle formal packet、启动与首周期健康门

- 新增并执行：
  ```text
  local_scripts/remote_rlt_20260730_formal_8env250_prepare.sh
  local_scripts/remote_rlt_20260730_formal_8env250_launch.sh
  local_scripts/remote_rlt_20260730_formal_8env250_live_audit.sh
  ```
- packet逐项锁定clean HEAD、accepted Stage 1 artifact、8-env smoke postflight SHA、
  250 cycles、8 env、2,000 train episodes、最大400k primitive slots、
  预计39,105 macro rows、约125,525/62,762 critic/actor updates、每25 cycles
  20条eval与1个checkpoint，共10点/200条eval/10个checkpoint。
- 正式resolved SHA-256为
  `586644cd69461016c1dd8c653da0eea12b01c61f2d0a9b4901654d90800f2a3e`。
  run root：
  `/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1`；
  experiment：
  `robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1`；
  runtime：
  `/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime`。
- 2026-07-30T11:37:25+08:00成功启动，driver/monitor PID为`154857/154858`；
  正常停止由250 cycles控制，外层64,800秒timeout只作18小时故障保险。
- 2026-07-30T11:41:49+08:00首个完整cycle健康闭合：8条trajectory均
  `episode_len=200`，总return1.25；新增151 macro transitions，两rank replay
  平均75.5、最慢rank71。10k rows/rank warm-up尚未满足，故reference route、
  `update_step=0`、本轮updates=0均符合冻结合同。
- 同一快照两卡约17,111/17,194MiB，cgroup anon约32.4GiB，
  OOM/OOM-kill为0、PSI为0；没有CUDA OOM、NCCL fatal、NaN或rank death。
  达到用户要求的“确认正常启动”门后停止Codex主动轮询；服务器2秒资源monitor随driver
  继续运行并在driver结束时退出。

### A120 — 启动记录、交接与本地QA收口

- 新增
  `docs/rlinf-robotwin-pi0-rltoken/10_STAGE2_FORMAL_8ENV250_LAUNCH_20260730.md`，
  集中记录exact-20适配、资源门、两次准备问题、正式预算、resolved SHA、精确命令、
  运行路径、首周期和下次检查口径。
- 更新RLT唯一索引与根`HANDOFF.md`：历史100-cycle仍为pilot；当前动态停点改为
  8-env formal已健康启动、停止主动轮询。没有把动态状态写回`PROJECT_CONTEXT.md`。
- 本地QA逐文件检查UTF-8严格解码、无BOM、无行尾空白和相对Markdown链接存在，四份文档
  全部通过。第一次直接运行Git QA因Codex sandbox用户与仓库owner不同而触发
  `dubious ownership`，没有修改仓库；改为单次命令参数
  `git -c safe.directory=C:/Users/86136/Documents/rl -C ... diff --check`
  后exit0，没有写全局Git配置。
- 根文档树在当前本地Git视图中本来就是用户既有untracked结构，本轮不stage、不commit、
  不push；因此未跟踪文件的空白与链接由上述显式逐文件gate覆盖。没有改动或清理用户其他
  dirty/untracked内容。

### A121 — 2026-07-30 formal250运行中只读刷新

- 用户要求简要查看当前实验、产物、训练指标和资源指标；本轮授权仅为只读服务器检查与
  本地报告/可视化，不停止进程、不修改配置、不启动第二个run。
- 按工作区规则完整重读根`PROJECT_CONTEXT.md`、`HANDOFF.md`和RLT唯一事实源，再读
  10号启动记录与账本A114–A120。
- 首次无口令OpenSSH身份探针：
  ```text
  ssh -p 36406 -o BatchMode=yes -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=NUL -o ConnectTimeout=10 \
    root@connect.bjb1.seetacloud.com "hostname; pwd; id -u"
  ```
  返回`Permission denied (publickey,password)`；这只证明BatchMode不能回答password，
  没有把它误诊为服务器、网络或认证配置故障。随后把用户已提供的口令重新注入当前受控
  Node/Paramiko子进程，继续使用`look_for_keys=False/allow_agent=False`等价的既有helper；
  口令不写入脚本、文档、Git或输出，任务结束前再次清空。
- 使用`apply_patch`新增只读审计脚本：
  `local_scripts/remote_rlt_20260730_formal_8env250_status_audit.sh`。通过
  ```text
  python local_scripts/remote_exec_autodl.py run \
    --command-file local_scripts/remote_rlt_20260730_formal_8env250_status_audit.sh
  ```
  在服务器执行；exit0、stdout 116,069 bytes，结束标记存在。TensorFlow oneDNN信息只来自
  只读EventAccumulator加载，没有训练错误。
- 服务器审计时间为2026-07-30 12:53:54+08:00：
  - branch/HEAD=`codex/rlt-pi0-robotwin@46a2d19b...a494`，worktree clean，
    upstream left/right=`0/2`；
  - driver/monitor/唯一训练进程均存活，Ray `raylet/gcs_server`存在；
  - `finished_at.txt`和`exit_code.txt`不存在，符合运行中状态；
  - resolved SHA仍为
    `586644cd69461016c1dd8c653da0eea12b01c61f2d0a9b4901654d90800f2a3e`；
  - console完成cycle34并开始35；后续下载时console已到36、TensorBoard落盘到35；
  - CUDA OOM、NCCL fatal、Ray death和NaN metric均为0。
- 两个日志traceback用单独只读grep复核，均是两EnvWorker初始化时可选Curobo
  `curobo.types.math`缺失；后续真实rollout、eval和checkpoint正常，故仍是已知非阻塞提示。

### A122 — 高信息量证据下载、指标分析与可视化

- 使用`apply_patch`新增：
  ```text
  local_scripts/download_rlt_formal250_status.py
  local_scripts/analyze_rlt_formal250_status.py
  ```
- 下载器通过一次Paramiko/SFTP连接只读取得runtime小文件、metrics、唯一TensorBoard
  event、step25 completion、双rank replay metadata和双rank trainer state：
  ```text
  python local_scripts/download_rlt_formal250_status.py \
    docs/rlinf-robotwin-pi0-rltoken/evidence/
    stage2_formal_8env250_20260730/status_20260730_1254
  ```
  结果为18个文件、962,772 bytes；没有下载trajectory replay payload或大checkpoint。
- 本地执行：
  ```text
  python local_scripts/analyze_rlt_formal250_status.py \
    docs/rlinf-robotwin-pi0-rltoken/evidence/
    stage2_formal_8env250_20260730/status_20260730_1254
  ```
  生成`status_summary.json`及三张PNG：
  `success_and_phase.png`、`replay_and_timing.png`、`resource_profile.png`。
  第一次成功率图使用完整250-cycle横轴，当前35点过于拥挤；目视检查后窄改为当前50-cycle
  视窗，并把预计replay门写入图内，再次生成。没有改变任何原始证据或服务器状态。
- 冻结证据时间12:56:18：
  - TensorBoard最新完整cycle35，console最新完整cycle36；差1个正常flush；
  - 280 train episodes、20 eval episodes、5,237 global macros；
  - min replay/rank 2,599，约26%，当前斜率预计cycle135过10k门；
  - train reference 42/280=15.0%，最近10 cycles 8/80=10.0%；
  - cycle25 deterministic untrained-student eval 0/20；尚无优化tags，因为critic/actor
    updates均为0；
  - 最近10 cycle总耗时120.68s，rollout120.22s；cycle25 eval+save总耗时397.52s，
    eval本身272.81s。
- 2,085个资源样本、median间隔2秒：
  - GPU峰19,641/19,976MiB，active mean util30.1%/31.2%；
  - matched RSS/env RSS峰50.02/21.43GiB；
  - cgroup current/anon/file峰237.91/45.32/190.76GiB；
  - 本run high/max/OOM/OOM-kill增量均为0，审计时memory PSI为0；
  - host available最低936.22GiB，数据盘available 824.21GiB。
- cycle25 exact-20 eval后env/matched RSS出现约5GiB永久阶梯；最近300个样本分别在
  19.41–21.26/43.52–45.37GiB内周期波动而非继续上升。只经历一次eval，故把cycle50后
  是否再次阶梯列为后续观察项，不提前宣称泄漏或完全稳定。
- step25 checkpoint为140,307,188 bytes、3,722文件，completion=true、world size2、
  update0；双rank state存在，replay size为1,874/1,835。12:53 run root总计
  140,630,856 bytes。

### A123 — 运行中报告、路由更新与本地QA

- 使用`apply_patch`新增
  `docs/rlinf-robotwin-pi0-rltoken/11_STAGE2_FORMAL_8ENV250_STATUS_20260730.md`，
  汇总冻结时点、阶段解释、成功率、replay/耗时、资源、checkpoint和后续判读门；更新根
  `HANDOFF.md`、RLT唯一索引和本账本。索引从v15升到v16，并只路由动态报告，不在设计文档
  固化过时cycle数字。
- 本地QA实际执行：
  ```text
  python -               # compile()检查两个分析/下载脚本，不落pyc
  python -               # UTF-8/BOM/尾空白/冲突标记/fence/相对链接/JSON/PNG/hash
  git -c safe.directory=C:/Users/86136/Documents/rl diff --check
  ```
- 第一次JSON检查误按服务器checkpoint目录层级查找
  `checkpoints/global_step_25/checkpoint_completion.json`，返回`FileNotFoundError`。
  原因是下载器有意把18个小证据文件平铺并重命名；用`rg --files`核对真实目录后，改查
  `rlt_completion_step25.json`和两个`replay_rank_*_metadata_step25.json`，不是证据缺失。
- 最终QA全部通过：两个Python脚本可编译；7个本轮文本文件严格UTF-8、无BOM、无尾空白、
  无冲突标记，Markdown fence与相对链接闭合；5个JSON可解析；3张PNG存在且非空；
  `download_manifest.json`所列18个文件的字节数与SHA-256逐项匹配，总计962,772 bytes；
  `git diff --check` exit0。三张图均已目视检查。
- 本轮服务器动作始终只读：没有改配置、停止/重启进程、清cache、创建额外run或下载大权重。

### A124 — 2026-07-30 14:05 formal250第二次只读刷新

- 用户再次要求按同一口径查看实验。本轮完整重读`PROJECT_CONTEXT.md`、根`HANDOFF.md`和
  RLT唯一事实源，再读取上一份状态报告与本账本相关批次；授权仍为只读现场检查、本地证据/
  报告，不含进程控制、配置修改或第二个run。
- 登录口令只由当前PowerShell进程从本任务已有的用户消息动态取得，注入
  `SEETA_SSH_PASSWORD`后立即清除；没有写入脚本、文档、Memory、Git或输出。最初尝试用
  `read_thread(turnLimit=50)`恢复当前任务消息，工具明确限制`turnLimit<=10`；改用分页后，
  当前API只返回压缩后的两页且未含授权消息。随后只读使用本任务现有rollout日志中的原始
  用户消息完成process-only注入，没有人工复制或新建凭据文件。
- 执行：
  ```text
  python local_scripts/remote_exec_autodl.py run \
    --command-file \
    local_scripts/remote_rlt_20260730_formal_8env250_status_audit.sh
  ```
  服务器审计时间`2026-07-30T14:05:16+08:00`，exit0。driver/monitor
  `154857/154858`、唯一训练进程及Ray仍存活；`finished_at.txt`/`exit_code.txt`不存在。
  branch/HEAD仍为
  `codex/rlt-pi0-robotwin@46a2d19bae629eaa57830f5faeac71ac81a1a494`，
  worktree clean，upstream计数`0 behind/2 ahead`。
- 日志检查：CUDA OOM、NCCL fatal、Ray death和NaN均0；两个traceback仍只是两EnvWorker
  的已知可选Curobo导入提示。审计开始时console到cycle66/67附近；后续SFTP冻结的
  TensorBoard和console均闭合到cycle68。

### A125 — cycle68小证据、图与补充内存探针

- 通过一次只读SFTP执行：
  ```text
  python local_scripts/download_rlt_formal250_status.py \
    docs/rlinf-robotwin-pi0-rltoken/evidence/
    stage2_formal_8env250_20260730/status_20260730_1405
  ```
  得到23个文件、1,779,880 bytes：runtime日志/资源、唯一TensorBoard event、cycle25/50
  completion、双rank replay metadata和trainer state；没有大模型权重或replay payload。
- 执行：
  ```text
  python local_scripts/analyze_rlt_formal250_status.py \
    docs/rlinf-robotwin-pi0-rltoken/evidence/
    stage2_formal_8env250_20260730/status_20260730_1405
  ```
  生成`status_summary.json`与三张PNG。目视检查发现replay图标题仍硬写“cycle25”，而图中
  已有cycle25/50两个红色点；用`apply_patch`改为通用
  `red diamonds mark 20-episode eval + checkpoint`并重新生成，数据与原证据未改。
- 冻结结果：cycle68/250、544条train、40条eval、10,153条global macros；train
  `84/544=15.44%`，最近10 cycles `11/80=13.75%`；cycle25/50 student eval均
  `0/20`。min replay/rank `5,067/10,000`，critic/actor/update_step仍全0，无优化tag。
  cycle25/50 checkpoint均complete；cycle50大小229,067,493 bytes、7,463文件、
  replay rank0/rank1为3,704/3,746。
- 资源CSV共3,920行，覆盖11:37:25–14:05:48，median/max间隔2/3秒。GPU峰
  19,776/19,976MiB、active mean31.14%/30.91%；磁盘剩823.99GiB。与12:56末值相比，
  env RSS、matched RSS、cgroup anon分别从19.70/43.83/38.20GiB升到
  29.47/53.76/48.11GiB；约69.5分钟净增9.77/9.92/9.91GiB，呈EnvWorker主导阶梯。
- 为避免把冻结CSV外的PSI猜成0，追加一次只读现场探针：
  ```text
  date --iso-8601=seconds
  cat /sys/fs/cgroup/memory.current
  grep 'anon|file|inactive_file|active_file|slab' /sys/fs/cgroup/memory.stat
  cat /sys/fs/cgroup/memory.events
  cat /sys/fs/cgroup/memory.pressure
  grep 'MemAvailable|SwapTotal|SwapFree' /proc/meminfo
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu ...
  ```
  14:08结果为current239.97GiB、anon49.37GiB、file188.30GiB；
  PSI some/full avg60均0.09%；OOM/OOM-kill仍0。相对run首行的`memory.max`新增5,911，
  全部集中在14:03–14:05，证明已开始触发240GiB限额回收，不能继续称为“无回收压力”。

### A126 — cycle68判断与文档收口

- 新增
  `12_STAGE2_FORMAL_8ENV250_STATUS_CYCLE68_20260730.md`，更新RLT索引和根`HANDOFF.md`。
  报告把数值/显存/磁盘健康与内存风险分开：当前无OOM、吞吐未恶化，但anon/RSS不只在
  eval时增长，风险升级为“疑似EnvWorker retained allocation/泄漏”的橙色观察项。
- cycle25评估前后5分钟中位数出现约5.1–5.3GiB阶梯；cycle50即时只约0.5GiB，但随后
  普通cycle仍继续增长。因此不把每次eval固定解释为5GiB，也不提前断言已证明泄漏。
- 本轮没有越过只读边界停止进程。下一高信息量节点为cycle75后的anon、`memory.max`、
  PSI和cycle time；若持续同步恶化，则已接近批准的
  `sustained memory pressure/anon growth`停止条件，不能等到OOM后才处理。

### A127 — cycle68本地QA

- 本地执行只读QA：两个Python工具使用内存内`compile()`检查；7个本轮相关文本逐个检查
  严格UTF-8、无BOM、无尾空白、无冲突标记、Markdown fence与相对链接；6个JSON解析；
  23项下载manifest逐文件复核字节数与SHA-256；三张PNG存在且逐张目视检查。
- 结果：全部通过，manifest总计1,779,880 bytes；三张PNG大小分别
  134,682/179,262/533,811 bytes。最后执行
  `git -c safe.directory=C:/Users/86136/Documents/rl diff --check`，exit0。

### A128 — 2026-07-30 cycle100任务恢复与只读边界

- 用户连续要求按同一口径再次查看formal250，并追加“找问题、成功率是否下降、训练细指标
  是否正常、为什么还没有critic/actor优化、训练量是否太少”；同时再次要求流水账保存
  **完整命令**，不能只写命令名。
- 按工作区规则完整读取根`PROJECT_CONTEXT.md`、根`HANDOFF.md`和RLT唯一事实源
  `00_INDEX_AND_IMPLEMENTATION_PLAN.md`，再读取12号报告和账本A120–A127。服务器动作
  授权仍仅为只读状态、日志、资源和产物检查；不停止进程、不改配置、不清cache、不启动
  第二个run。
- 本轮先复核现有cycle68路由、账本尾部和证据目录。完整本地只读命令为：

  ```powershell
  Get-Date -Format o
  Get-ChildItem -LiteralPath `
    'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_formal_8env250_20260730/status_20260730_1514' |
    Select-Object Name,Length,LastWriteTime
  Get-Content -Raw -Encoding utf8 -LiteralPath `
    'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_formal_8env250_20260730/status_20260730_1514/status_summary.json'
  Get-ChildItem -LiteralPath 'docs/rlinf-robotwin-pi0-rltoken' |
    Sort-Object Name | Select-Object Name,Length,LastWriteTime
  rg -n "cycle68|status_20260730_1405|A12[0-9]|实施账本|Formal 8-env" `
    'docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md' `
    'HANDOFF.md' 'docs/rlinf-robotwin-pi0-rltoken' -g '*.md'
  Get-Content -Raw -Encoding utf8 -LiteralPath `
    'docs/rlinf-robotwin-pi0-rltoken/12_STAGE2_FORMAL_8ENV250_STATUS_CYCLE68_20260730.md'
  Get-Content -Encoding utf8 -LiteralPath 'HANDOFF.md' | Select-Object -First 190
  Get-Content -Encoding utf8 -LiteralPath `
    'docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md' |
    Select-Object -Last 240
  ```

- 为缩短服务器输出并使后续状态刷新可复用，使用`apply_patch`新增只读脚本
  `local_scripts/remote_rlt_20260730_formal_8env250_compact_status.sh`。脚本只执行
  identity/Git/process/runtime/error计数、最后三条Global Step、GPU/cgroup/PSI/disk和
  checkpoint metadata读取，不执行任何写操作。

### A129 — 15:14/15:18 cycle99冻结现场与第一轮问题定位

- AutoDL口令只从当前任务已有用户消息的rollout日志动态提取，注入当前PowerShell子进程
  的`SEETA_SSH_PASSWORD`，Paramiko helper完成固定host-key校验和password auth，命令结束
  立即删除环境变量。没有把口令写入脚本、文档、Memory、Git或输出。
- 15:18复核使用的完整命令如下；15:14首次调用使用同一命令和同一只读脚本：

  ```powershell
  $rollout = 'E:\Codex\home\sessions\2026\07\28\rollout-2026-07-28T14-03-48-019fa752-79fd-7de3-b66f-5ac4f0a72bfc.jsonl'
  $raw = Get-Content -Raw -Encoding utf8 -LiteralPath $rollout
  $match = [regex]::Match(
    $raw,
    'ssh -p 36406[^\r\n]{0,1000}?】\u3010?([A-Za-z0-9]{8,64})】'
  )
  if (-not $match.Success) {
    throw 'Authorized credential was not found in the current rollout log.'
  }
  $env:SEETA_SSH_PASSWORD = $match.Groups[1].Value
  try {
    python 'local_scripts/remote_exec_autodl.py' run `
      --command-file `
      'local_scripts/remote_rlt_20260730_formal_8env250_compact_status.sh'
  }
  finally {
    Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    $raw = $null
    $match = $null
  }
  ```

- 15:14冻结目录的完整下载与分析命令：

  ```powershell
  python 'local_scripts/download_rlt_formal250_status.py' `
    'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_formal_8env250_20260730/status_20260730_1514'
  python 'local_scripts/analyze_rlt_formal250_status.py' `
    'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_formal_8env250_20260730/status_20260730_1514'
  ```

  下载结果为28个小型文件、2,574,944 bytes，包含cycle25/50/75三个checkpoint的小
  metadata/state，不含大replay/model payload。分析生成`status_summary.json`和三张PNG。
- 冻结到cycle99/250：792条train、14,865条macro、min replay/rank
  `7,353/10,000`，train `112/792=14.14%`，最近10 cycles `12/80=15%`，
  cycle25/50/75 eval均0/20；critic/actor/update_step均0。driver、monitor、唯一训练
  进程和Ray均存活；CUDA OOM/NCCL fatal/Ray death/NaN均0。
- cycle68→99的68.8分钟内Env RSS、matched RSS、anon分别增加
  8.58/8.25/9.01GiB；但15:14前最近30分钟暂时平台，故当时没有抢先断言停止条件已经
  确定触发，而是等待cycle100这个下一次exact-20自然节点。

### A130 — TensorBoard 57-tag全量审计与成功率统计

- 使用两个并行只读审计交叉检查：一个枚举全部TensorBoard tags和finite/step合同，一个
  专门做成功率分段、Wilson区间、Fisher/线性趋势、reward/transition/timing相关。主任务
  又用以下完整命令独立复算全量scalar摘要：

  ```powershell
  @'
  from pathlib import Path
  import math
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  root = Path(
      r"docs/rlinf-robotwin-pi0-rltoken/evidence/"
      r"stage2_formal_8env250_20260730/status_20260730_1514"
  )
  event = next(root.glob("events.out.tfevents.*"))
  acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
  acc.Reload()
  print(f"event={event.name}")
  for tag in sorted(acc.Tags().get("scalars", [])):
      values = [float(e.value) for e in acc.Scalars(tag)]
      finite = sum(math.isfinite(v) for v in values)
      print(
          f"{tag}\tn={len(values)}\tfinite={finite}"
          f"\tfirst={values[0]:.8g}\tlast={values[-1]:.8g}"
          f"\tmin={min(values):.8g}\tmax={max(values):.8g}"
          f"\tmean={sum(values)/len(values):.8g}"
      )
  '@ | python -
  ```

- cycle99成功率独立统计的完整命令：

  ```powershell
  @'
  from pathlib import Path
  import math
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  from scipy.stats import linregress, fisher_exact

  root = Path(
      r"docs/rlinf-robotwin-pi0-rltoken/evidence/"
      r"stage2_formal_8env250_20260730/status_20260730_1514"
  )
  event = next(root.glob("events.out.tfevents.*"))
  acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
  acc.Reload()
  vals = [float(e.value) for e in acc.Scalars("env/success_once")]
  counts = [round(v * 8) for v in vals]

  def wilson(k, n, z=1.96):
      p = k / n
      d = 1 + z * z / n
      c = (p + z * z / (2 * n)) / d
      h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
      return c - h, c + h

  for a, b in [
      (1, 25), (26, 50), (51, 75), (76, 99),
      (1, 49), (50, 99), (1, 68), (69, 99), (90, 99),
  ]:
      k = sum(counts[a - 1:b])
      n = (b - a + 1) * 8
      lo, hi = wilson(k, n)
      print(
          f"cycles={a:02d}-{b:02d}\tsuccess={k}/{n}"
          f"\trate={k/n:.4%}\twilson95=[{lo:.4%},{hi:.4%}]"
      )
  first_k = sum(counts[:49])
  last_k = sum(counts[49:])
  odds, p = fisher_exact(
      [[first_k, 49 * 8 - first_k], [last_k, 50 * 8 - last_k]]
  )
  print(f"Fisher first49_vs_last50 odds={odds:.4f} p_two_sided={p:.4f}")
  reg = linregress(range(1, len(vals) + 1), vals)
  print(
      f"cycle_rate_linear_slope={reg.slope:.8f}"
      f"_fraction_per_cycle p={reg.pvalue:.4f} r={reg.rvalue:.4f}"
  )
  print("per_cycle_counts=" + ",".join(map(str, counts)))
  '@ | python -
  ```

- cycle99有57个scalar tags，非eval每项99点、eval每项3点，全部finite。分段成功率
  17.0%/14.5%/13.5%/11.46%名义下降，但逐cycle趋势`p=0.1825`、前49对后50
  Fisher `p=0.2637`，rolling-10又回到15%；不足以证明真实下降。
- 并行审计最初把临时JSON写到`C:\tmp`时得到`PermissionError`；原因是该子任务的受控
  写权限不包含目标。它改为工作区`.tmp`后成功，分析结束删除临时JSON和临时脚本。随后
  直接`git status`又遇到仓库owner导致的`dubious ownership`；没有改全局配置，改用
  `git -c safe.directory=C:/Users/86136/Documents/rl ...`后成功。

### A131 — 调度源码复核与“critic-only”措辞纠正

- 为回答“为什么还不优化、何时开始SAC”，执行以下完整只读命令：

  ```powershell
  Get-Content -Encoding utf8 -LiteralPath `
    '.rlt-impl-worktree/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py' |
    Select-Object -Skip 1400 -First 190
  Get-Content -Encoding utf8 -LiteralPath `
    '.rlt-impl-worktree/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py' |
    Select-Object -Skip 240 -First 175
  rg -n "def update_one_epoch|critic_actor_ratio|update_actor" `
    '.rlt-impl-worktree/rlinf/workers/actor/fsdp_sac_policy_worker.py' `
    '.rlt-impl-worktree/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py'
  Get-Content -Encoding utf8 -LiteralPath `
    '.rlt-impl-worktree/rlinf/workers/actor/fsdp_sac_policy_worker.py' |
    Select-Object -Skip 700 -First 135
  Get-Content -Encoding utf8 -LiteralPath `
    '.rlt-impl-worktree/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py' |
    Select-Object -Skip 1550 -First 250
  rg -n -C 4 `
    "min_replay_buffer_size|warmup_required_updates|max_updates_per_train_step|bc_weight|q_weight|reference_dropout|actor_update|critic.*actor|actor.*critic" `
    'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_formal_8env250_20260730/status_20260730_1514/resolved.yaml'
  ```

- 源码确认：min rank低于10k时`skip_reason=1`且optimizer完全不动；过门后
  `run_training()`立即调用`update_one_epoch(train_actor=True)`，每步critic、每2步actor。
  `ready_for_online`只控制30k update后是否把rollout交给student。因此早先口头使用的
  “30k critic-only warm-up”不精确，已经在聊天中主动纠正为：
  **约cycle135开始critic+actor离线更新，约cycle154达到30k后student接管。**
- 同一轮读取train seed实现：

  ```powershell
  rg -n "use_fixed_reset_state_ids|seed|reset|success_once|episode_len" `
    '.rlt-impl-worktree/rlinf/envs/robotwin' `
    '.rlt-impl-worktree/rlinf/workers/env' -g '*.py'
  Get-Content -Encoding utf8 -LiteralPath `
    '.rlt-impl-worktree/rlinf/envs/robotwin/robotwin_env.py' |
    Select-Object -Skip 410 -First 105
  Get-Content -Encoding utf8 -LiteralPath `
    '.rlt-impl-worktree/rlinf/workers/env/env_worker.py' |
    Select-Object -Skip 720 -First 40
  ```

  一次直接`Get-Content train_seeds.json`打印了极大的多任务seed bank，工具输出被截断；
  没有修改文件。有效结论仍由局部源码得到：seed bank先确定性shuffle/分rank，随后每个
  EnvWorker顺序推进；当前指标不记录逐episode seed ID，所以不能做seed-conditioned归因。

### A132 — cycle100自然节点、第四个checkpoint与最新版冻结

- 15:22:14再次执行A129的完整process-only Paramiko命令；服务器现场已经完成cycle101，
  cycle100 checkpoint完整。CUDA OOM/NCCL/Ray death/NaN仍0。
- 使用以下完整命令冻结最新版：

  ```powershell
  $rollout = 'E:\Codex\home\sessions\2026\07\28\rollout-2026-07-28T14-03-48-019fa752-79fd-7de3-b66f-5ac4f0a72bfc.jsonl'
  $raw = Get-Content -Raw -Encoding utf8 -LiteralPath $rollout
  $match = [regex]::Match(
    $raw,
    'ssh -p 36406[^\r\n]{0,1000}?】\u3010?([A-Za-z0-9]{8,64})】'
  )
  if (-not $match.Success) {
    throw 'Authorized credential was not found in the current rollout log.'
  }
  $env:SEETA_SSH_PASSWORD = $match.Groups[1].Value
  try {
    python 'local_scripts/download_rlt_formal250_status.py' `
      'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_formal_8env250_20260730/status_20260730_1522'
  }
  finally {
    Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
    $raw = $null
    $match = $null
  }

  python 'local_scripts/analyze_rlt_formal250_status.py' `
    'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_formal_8env250_20260730/status_20260730_1522'
  ```

- SFTP得到33个小型文件、2,652,489 bytes和manifest；本地分析生成三张图及summary。
  最新TensorBoard完整到cycle100：800条train、80条eval、15,025条macro，min replay
  7,433/10,000；train112/800，最近10为11/80，四次eval均0/20；仍无optimizer update。
- cycle100 checkpoint为408,791,932 bytes、15,038文件，completion=true、world size2、
  contract SHA一致、双rank trainer state完整、replay 7,433/7,592。run root为
  1,097,692,292 bytes。process/Git未变化，训练仍运行。

### A133 — cycle100成功率复算与内存停止门判断

- 对最新版事件文件执行以下完整finite和趋势复算：

  ```powershell
  @'
  from pathlib import Path
  import math
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  from scipy.stats import fisher_exact, linregress

  root = Path(
      r"docs/rlinf-robotwin-pi0-rltoken/evidence/"
      r"stage2_formal_8env250_20260730/status_20260730_1522"
  )
  event = next(root.glob("events.out.tfevents.*"))
  acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
  acc.Reload()
  tags = sorted(acc.Tags().get("scalars", []))
  bad = []
  points = 0
  for tag in tags:
      events = acc.Scalars(tag)
      points += len(events)
      for event_item in events:
          if not math.isfinite(float(event_item.value)):
              bad.append((tag, event_item.step, event_item.value))
  print(f"tags={len(tags)} points={points} bad_nonfinite={len(bad)}")
  vals = [float(e.value) for e in acc.Scalars("env/success_once")]
  counts = [round(v * 8) for v in vals]

  def wilson(k, n, z=1.96):
      p = k / n
      d = 1 + z * z / n
      c = (p + z * z / (2 * n)) / d
      h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
      return c - h, c + h

  for a, b in [
      (1, 25), (26, 50), (51, 75), (76, 100),
      (1, 50), (51, 100), (91, 100),
  ]:
      k = sum(counts[a - 1:b])
      n = (b - a + 1) * 8
      lo, hi = wilson(k, n)
      print(
          f"cycles={a}-{b} success={k}/{n} rate={k/n:.4%}"
          f" wilson95=[{lo:.4%},{hi:.4%}]"
      )
  first = sum(counts[:50])
  last = sum(counts[50:])
  odds, p = fisher_exact([[first, 400 - first], [last, 400 - last]])
  reg = linregress(range(1, 101), vals)
  print(f"first50_vs_last50 fisher_odds={odds:.4f} p={p:.4f}")
  print(
      f"linear_slope_fraction_per_cycle={reg.slope:.8f}"
      f" p={reg.pvalue:.4f} r={reg.rvalue:.4f}"
  )
  print(
      "eval=" + ",".join(
          f"cycle{e.step + 1}:{float(e.value):.4f}"
          for e in acc.Scalars("eval/success_once")
      )
  )
  '@ | python -
  ```

  结果为57 tags、5,028点、非finite 0；四段为17.0%/14.5%/13.5%/11.0%，
  前50对后50 Fisher `p=.1851`、线性趋势`p=.1270`，仍不足以称真实下降。
- 评估后内存阶梯使用以下完整命令复算；列名先统一换算GiB，评估结束时间来自
  `time/step`的TensorBoard wall time：

  ```powershell
  @'
  from pathlib import Path
  import pandas as pd
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  root = Path(
      r"docs/rlinf-robotwin-pi0-rltoken/evidence/"
      r"stage2_formal_8env250_20260730/status_20260730_1522"
  )
  resource = pd.read_csv(root / "resources.csv")
  for column in [
      "cgroup_anon_bytes", "cgroup_file_bytes", "cgroup_current_bytes"
  ]:
      resource[column] /= 1024 ** 3
  for column in ["env_rss_kib", "matched_total_rss_kib"]:
      resource[column] /= 1024 ** 2
  accumulator = EventAccumulator(
      str(next(root.glob("events.out.tfevents.*"))),
      size_guidance={"scalars": 0},
  )
  accumulator.Reload()
  times = {e.step + 1: e.wall_time for e in accumulator.Scalars("time/step")}
  columns = [
      "env_rss_kib", "matched_total_rss_kib", "cgroup_anon_bytes",
      "cgroup_file_bytes", "cgroup_current_bytes", "cgroup_max_events",
  ]
  for cycle in [25, 50, 75, 100]:
      event_time = times[cycle]
      pre = resource[
          (resource.unix_time >= event_time - 300)
          & (resource.unix_time <= event_time - 60)
      ]
      post = resource[
          (resource.unix_time >= event_time + 60)
          & (
              resource.unix_time
              <= min(event_time + 360, resource.unix_time.iloc[-1])
          )
      ]
      print(
          f"cycle={cycle} pre_n={len(pre)} post_n={len(post)}"
          f" post_span_s={0 if post.empty else int(post.unix_time.iloc[-1] - post.unix_time.iloc[0])}"
      )
      for column in columns:
          before = pre[column].median()
          after = post[column].median()
          print(
              f"  {column}: pre={before:.3f}"
              f" post={after:.3f} delta={after-before:+.3f}"
          )
  '@ | python -
  ```

- 四个eval后的Env RSS/matched/anon中位阶梯分别约
  `+5.75/+5.97/+5.96`、`+1.22/+1.25/+1.25`、
  `+3.92/+3.52/+4.22`和`+2.21/+2.21/+2.22`GiB。cycle75/100同时发生
  file cache回收和`memory.max`增加24,149/13,146。
- 15:22冻结current/peak：Env RSS40.21/40.24GiB、matched64.17/64.20GiB、
  anon59.29/59.68GiB、file177.96/191.01GiB、cgroup239.60/240GiB；
  本run max事件累计95,824，OOM/OOM-kill为0。cycle100现场PSI some/full avg60均0.04%。
  GPU峰19,776/19,976MiB、disk余823.28GiB。
- 与cycle99前30分钟暂时平台的反证相比，cycle100又重现完整阶梯。因此本轮把风险从
  “接近”升级为 **已经满足冻结的`sustained memory pressure/anon growth`停止定义**。
  但用户本轮只授权只读检查，故没有擅自TERM；建议用完整checkpoint100作安全停点，
  等待用户明确授权。

### A134 — cycle100报告、路由与待QA

- 使用`apply_patch`新增
  `13_STAGE2_FORMAL_8ENV250_STATUS_CYCLE100_20260730.md`，集中回答：
  成功率名义下降及统计不确定性、57-tag细指标、哪些正常/异常、为什么无loss/Q/grad、
  精确SAC开始和student切换时点、250-cycle量级限制、资源停止判断和checkpoint100产物。
- 更新RLT唯一索引到v17并把12号报告降为cycle68历史追溯，13号成为最新状态入口；更新根
  `HANDOFF.md`的RLT专题行和动态状态。没有把动态实验数字写入`PROJECT_CONTEXT.md`。
- 本轮所有服务器调用均只读；没有修改server worktree/run、停止或重启进程、删除文件、
  清cache、下载大权重或启动额外实验。

### A135 — cycle100图形复核与本地QA

- 逐张使用原分辨率目视检查最新版
  `success_and_phase.png`、`replay_and_timing.png`和`resource_profile.png`。前两张文字、
  图例、点线和四个eval节点清晰。第一次资源图检查发现底部子图左上角的事件增量文字与
  图例重叠；数据没有错误，但影响阅读。
- 使用`apply_patch`把
  `local_scripts/analyze_rlt_formal250_status.py::plot_resources`中的事件文字从axes坐标
  `(0.01, 0.96), va=top`移动到`(0.01, 0.02), va=bottom`，然后完整重跑：

  ```powershell
  python 'local_scripts/analyze_rlt_formal250_status.py' `
    'docs/rlinf-robotwin-pi0-rltoken/evidence/stage2_formal_8env250_20260730/status_20260730_1522'
  ```

  再次目视检查确认图例和事件文字不再重叠；原始日志、CSV、TensorBoard和下载manifest
  未改动。
- 最终本地QA完整命令：

  ```powershell
  @'
  from pathlib import Path
  import hashlib
  import json
  import re

  root = Path.cwd()
  text_files = [
      root / "HANDOFF.md",
      root / "docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md",
      root / "docs/rlinf-robotwin-pi0-rltoken/13_STAGE2_FORMAL_8ENV250_STATUS_CYCLE100_20260730.md",
      root / "docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md",
      root / "local_scripts/analyze_rlt_formal250_status.py",
      root / "local_scripts/remote_rlt_20260730_formal_8env250_compact_status.sh",
  ]
  errors = []
  for path in text_files:
      raw = path.read_bytes()
      try:
          text = raw.decode("utf-8", errors="strict")
      except UnicodeDecodeError as exc:
          errors.append(f"{path}: utf8 {exc}")
          continue
      if raw.startswith(b"\xef\xbb\xbf"):
          errors.append(f"{path}: BOM")
      for number, line in enumerate(text.splitlines(), 1):
          if line.rstrip(" \t") != line:
              errors.append(f"{path}:{number}: trailing whitespace")
      if any(marker in text for marker in ("<" * 7, "=" * 7, ">" * 7)):
          errors.append(f"{path}: conflict marker")
      if path.suffix == ".md" and text.count("`" * 3) % 2:
          errors.append(f"{path}: unbalanced fenced code")

  link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
  for path in text_files[:4]:
      text = path.read_text(encoding="utf-8")
      for target in link_pattern.findall(text):
          target = target.strip().strip("<>")
          if target.startswith(("http://", "https://", "#", "mailto:")):
              continue
          target = target.split("#", 1)[0]
          if target and not (path.parent / target).resolve().exists():
              errors.append(f"{path}: missing link {target}")

  evidence = (
      root
      / "docs/rlinf-robotwin-pi0-rltoken/evidence/"
      / "stage2_formal_8env250_20260730/status_20260730_1522"
  )
  for path in evidence.glob("*.json"):
      json.loads(path.read_text(encoding="utf-8"))
  manifest = json.loads(
      (evidence / "download_manifest.json").read_text(encoding="utf-8")
  )
  total = 0
  for item in manifest:
      path = evidence / item["local_name"]
      raw = path.read_bytes()
      total += len(raw)
      if len(raw) != item["bytes"]:
          errors.append(f"{path}: bytes mismatch")
      if hashlib.sha256(raw).hexdigest() != item["sha256"]:
          errors.append(f"{path}: sha mismatch")
  for name in (
      "success_and_phase.png", "replay_and_timing.png", "resource_profile.png"
  ):
      path = evidence / name
      if not path.exists() or path.stat().st_size <= 0:
          errors.append(f"{path}: missing/empty")
  for script in (
      "local_scripts/analyze_rlt_formal250_status.py",
      "local_scripts/download_rlt_formal250_status.py",
      "local_scripts/remote_exec_autodl.py",
  ):
      source = (root / script).read_text(encoding="utf-8")
      compile(source, script, "exec")
  print(
      json.dumps(
          {
              "text_files": len(text_files),
              "manifest_entries": len(manifest),
              "manifest_bytes": total,
              "json_files": len(list(evidence.glob("*.json"))),
              "errors": errors,
          },
          ensure_ascii=False,
      )
  )
  if errors:
      raise SystemExit(1)
  '@ | python -

  git -c safe.directory='C:/Users/86136/Documents/rl' diff --check
  ```

- 结果：6个本轮文本文件严格UTF-8、无BOM/尾空白/冲突标记，Markdown fence和相对链接
  闭合；15个JSON可解析；33项manifest逐项字节数和SHA-256一致，总计2,652,489 bytes；
  三张PNG非空且已目视检查；三个Python工具内存编译通过；`git diff --check` exit0。

### A136 — 流水账自检误报与最终复测

- A135把QA程序本身完整写入Markdown后，第一次最终复测报告本账本含“conflict”和
  “unbalanced fence”。只读定位命令最初尝试用带PowerShell backtick的`rg`正则，因shell
  预处理得到`unclosed group`；改用以下无歧义命令定位：

  ```powershell
  Select-String -LiteralPath `
    'docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md' `
    -SimpleMatch `
    -Pattern '<<<<<<<','=======','>>>>>>>','text.count("```")' |
    Select-Object LineNumber,Line
  ```

- 原因不是实际冲突或fence缺失，而是账本内嵌的QA源码恰好逐字包含自己要搜索的七字符
  conflict marker与三个backtick，形成自指误报。使用`apply_patch`把等价检查改为
  `("<" * 7, "=" * 7, ">" * 7)`和`text.count("`" * 3)`，命令语义不变。
- 最终复测把冲突检查限定为“行首真实marker”、fence检查限定为“行首fence delimiter”，
  从而允许流水账忠实保存失败命令本身。6个文本、33项manifest、2,652,489 bytes、JSON、
  PNG、Python compile和`git diff --check`全部通过，`errors=[]`。

### A137 — cycle200只读刷新、下载中断修复与可视化重做

- 重新完整读取根`PROJECT_CONTEXT.md`、`HANDOFF.md`和RLT唯一专题计划后，按授权仅做
  服务器只读检查。密码仍只从当前任务进程注入`SEETA_SSH_PASSWORD`，命令结束立即移除；
  没有写入脚本、账本、Memory或输出。身份探针完整命令为：

  ```powershell
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run whoami
  ```

  返回`root`，同时由helper验证既有SSH host-key SHA256。
- 第一次现场命令误把远端命令直接放在主parser后、遗漏`run`子命令，返回
  `invalid choice`；第二次又因PowerShell把带空格的远端`printf`拆成多个参数而返回
  `unrecognized arguments`。没有连接后写操作。修复为无空格身份探针，以及把复杂远端
  只读命令完整保存到`.tmp/remote_rlt_formal250_status_query.sh`后执行：

  ```powershell
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file '.\.tmp\remote_rlt_formal250_status_query.sh'
  ```

  该脚本只执行`date/ps/grep/nvidia-smi/cat/find/du/df`，不修改服务器。
- 原有完整证据下载命令为：

  ```powershell
  python '.\local_scripts\download_rlt_formal250_status.py' `
    '.\.tmp\formal250_live_20260730_195508'
  ```

  大陆线路下120秒超时，driver/resources等文件已下载但TensorBoard未完成；进程没有
  留存。另一个并行只读快照
  `.tmp/formal250_live_subagent_20260730_195741`成功取得完整TensorBoard和resources，
  用于cycle196预检查。
- 20:16服务器自然完成cycle200：20条fixed-seed deterministic eval为`16/20=80%`，
  train为`4/8`，并开始保存checkpoint200。只读tail命令为：

  ```powershell
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file '.\.tmp\remote_rlt_formal250_tail_query.sh'
  ```

  返回`global_step_200`、eval五个epoch全部完成、`Saving checkpoint at step 200`和
  completion marker存在。
- 为冻结cycle200，先用SFTP分别下载TensorBoard和resources：

  ```powershell
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root get `
    '/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1/tensorboard/events.out.tfevents.1785382662.autodl-container-nekaqbwt43-6ce5babb.154864.0' `
    '.\.tmp\formal250_live_cycle200_20260730_201644\events.out.tfevents.1785382662.autodl-container-nekaqbwt43-6ce5babb.154864.0'

  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root get `
    '/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime/resources.csv' `
    '.\.tmp\formal250_live_cycle200_20260730_201644\resources.csv'
  ```

  event文件788,379 bytes完成，但未压缩resources在120秒时仅1,146,880 bytes，故不能
  使用。原因是小带宽SFTP，不是训练或CSV错误。
- 使用`apply_patch`新增`.tmp/download_remote_gzip.py`，通过同一已校验Paramiko连接在
  远端只读执行`gzip -c`，只把压缩stdout写到本地；没有创建远端临时文件。复测命令：

  ```powershell
  python '.\.tmp\download_remote_gzip.py' `
    '/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime/resources.csv' `
    '.\.tmp\formal250_live_cycle200_20260730_201644\resources.csv'
  ```

  30.8秒取得495,063-byte压缩流并恢复为2,271,685-byte完整CSV。
- 通用`analyze_rlt_formal250_status.py`要求driver等全套文件，直接用于最小快照时报
  `FileNotFoundError: driver.log`；该失败未改数据。改用本轮新增的动态边界复算脚本：

  ```powershell
  python '.\.tmp\plot_rlt_formal250_verified.py' `
    '.\.tmp\formal250_live_cycle200_20260730_201644' `
    'E:\Codex\home\visualizations\2026\07\28\019fa752-79fd-7de3-b66f-5ac4f0a72bfc' `
    --label cycle200-verified
  ```

  脚本从真实tag推导P2/P3/P4起点为136/155/192，不再硬编码预测边界；严格使用完整cycle
  `step+1`、train每cycle 8条、eval每点20条，并输出success/optimization/resources三张
  PNG和JSON。
- cycle200复算结果：
  - P1/P2/P3/P4成功率为
    `156/1080`、`22/152`、`94/296`、`55/72`；
  - deterministic eval cycle150/175/200为`1/20`、`6/20`、`16/20`；
  - critic/actor累计`75,215/37,608`，pending 0；
  - actor/critic loss`-0.0661/0.00149`，grad`3.189/0.189`；
  - Q0/Q1/Q(data)`0.214/0.206/0.181`，BC/Q权重`2.5/0.45`。
- 重做inline图时先用`render.py`包装，再用系统Chrome headless分别按736px和360px宽度
  实际打开。第一次视觉QA发现phase table的Unicode en dash在既有shell编码链中变成
  `鈥?`并吞掉第二个模板变量；使用`apply_patch`改成ASCII`-`，重新生成并复测：

  ```powershell
  python 'E:\Codex\home\plugins\cache\openai-bundled\visualize\1.0.14\skills\visualize\scripts\render.py' `
    'E:\Codex\home\visualizations\2026\07\28\019fa752-79fd-7de3-b66f-5ac4f0a72bfc\rlt-formal250-cycle200-verified.html' `
    '.\.tmp\rlt-formal250-cycle200-standalone.html'
  ```

  最终两种宽度均无JS error、无水平溢出；四行table为
  `1-135/136-154/155-191/192-200`，success、eval和updates均可见。三张PNG也逐张用原分辨率
  目视确认图例、阶段线、eval标签和资源双轴无重叠。
- 20:23:28最终服务器只读现场：driver仍存活，console到cycle202；
  checkpoint25–200共8个，step200约747MiB且completion完整；磁盘余约821GiB。
  fatal扫描为空，OOM/OOM-kill为0。两卡历史显存峰19.37/19.51GiB，平均利用率
  30.3%/27.5%。但anon峰77.44GiB、matched RSS峰82.41GiB、Env RSS峰57.18GiB，
  相对cycle162继续增加约7.4GiB并依靠file cache回收维持240GiB上限，所以原停止定义
  仍成立。本轮没有停止、重启、resume、删文件、清cache或启动第二个run。
- 第一次本地文档QA又使用了整文件`text.count("```")`，被A136内嵌的旧QA源码触发
  `unbalanced fences`误报；改为只计数匹配`^\s*```(?:\w+)?\s*$`的真实行首fence后为
  404个、偶数。最终四个UTF-8文档无BOM/尾空白、line-start fences闭合，五个图/JSON/HTML
  产物非空且summary JSON可解析，`git diff --check` exit0。

### A138 — 250/250自然完成、最终证据冻结与双宽度可视化QA

- 按根`HANDOFF.md`授权只做服务器只读刷新。身份凭据仍只注入当前进程环境变量，
  `look_for_keys=False`、`allow_agent=False`，命令结束后移除；没有写入脚本、账本、
  Memory或输出。
- 第一轮状态脚本执行`ps -p 154857`时，因为训练进程已经自然退出且脚本使用`set -e`，
  在后续完成标记前提前exit1。该失败没有写服务器。窄修复是在只读查询中的非零探针后加
  `|| true`，并以`finished_at/exit_code/Global Step/checkpoint completion`共同判断：

  ```powershell
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file '.\.tmp\remote_rlt_formal250_final_query.sh'
  ```

  23:03+08:00返回`250/250`、`exit_code=0`、finished
  `2026-07-30T22:23:41+08:00`，driver不存在、两卡现场显存约5MiB。最后一次`pgrep`
  命中了查询命令自身，因为正则包含run路径；结合PID命令行、driver PID缺失、GPU为0和
  finished/exit证据，确认没有残留训练/Ray进程。
- 最终只读下载优先沿用压缩stdout办法，避免大陆小带宽SFTP对2–3MB文本反复超时：

  ```powershell
  python '.\.tmp\download_remote_gzip.py' `
    '/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1/tensorboard/events.out.tfevents.1785382662.autodl-container-nekaqbwt43-6ce5babb.154864.0' `
    '.\.tmp\formal250_final_20260730_230104\events.out.tfevents.1785382662.autodl-container-nekaqbwt43-6ce5babb.154864.0'

  python '.\.tmp\download_remote_gzip.py' `
    '/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime/resources.csv' `
    '.\.tmp\formal250_final_20260730_230104\resources.csv'

  python '.\.tmp\download_remote_gzip.py' `
    '/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime/driver.log' `
    '.\.tmp\formal250_final_20260730_230104\driver.log'
  ```

  原始/压缩字节分别为event `1,054,383/246,728`、resources
  `2,810,818/613,037`、driver `1,947,559/83,174`。三个最终文件SHA-256分别为
  `4dea11b3...17e2`、`b7f0862d...ded2`、`e829d2ce...792d`。
- 使用与cycle200相同、从真实tag动态推导阶段的复算脚本：

  ```powershell
  python '.\.tmp\plot_rlt_formal250_verified.py' `
    '.\.tmp\formal250_final_20260730_230104' `
    'E:\Codex\home\visualizations\2026\07\28\019fa752-79fd-7de3-b66f-5ac4f0a72bfc' `
    --label cycle250-final
  ```

  返回P2/P3/P4边界`136/155/192`。P1/P2/P3/P4 train成功率为
  `156/1080`、`22/152`、`94/296`、`375/472`；final eval为
  `18/20`，最后20/10/5 train cycles为83.75%/85%/90%。
- 最终 critic/actor updates为`102,260/51,130`，pending 0；actor/critic loss
  `-0.08119/0.001286`，grad `2.821/0.170`，Q0/Q1/Q(data)
  `0.2515/0.2383/0.2123`，BC/Q权重`2.5/0.45`。全量finite、fatal扫描为空。
- 资源峰值为GPU显存19.37/19.51GiB、matched RSS87.42GiB、Env RSS62.05GiB、
  anon82.43GiB、file191.01GiB、cgroup current240GiB；OOM/OOM-kill为0。
  退出后matched RSS归零、Env RSS近零、anon约0.3–0.6GiB，证明大部分retained
  anonymous memory由worker持有并在退出时释放，但不撤销运行期内存压力风险判定。
- 交互片段通过官方visualization render helper包装：

  ```powershell
  python 'E:\Codex\home\plugins\cache\openai-bundled\visualize\1.0.14\skills\visualize\scripts\render.py' `
    'E:\Codex\home\visualizations\2026\07\28\019fa752-79fd-7de3-b66f-5ac4f0a72bfc\rlt-formal250-cycle250-final.html' `
    '.\.tmp\rlt-formal250-cycle250-final-standalone.html'
  ```

  再用系统Chrome headless分别按736px和360px实际打开。两种宽度均无JS error、无水平
  溢出、四行阶段表完整；右侧cycle250、18/20标签未裁切。三张静态PNG也按原分辨率目视
  检查，阶段、eval、优化和资源释放曲线均与同一summary JSON一致。
- 最终checkpoint25–250共10个、总计约5.2GiB；`global_step_250`约876MiB，
  completion marker和两个rank trainer state均完整，`update_step=102260`。本轮没有
  停止、启动、resume、删除、覆盖、清cache或发起新实验。

### A139 — Stage 2 final高信息量材料包

- 用户在Stage 2 `250/250`顺利结束后，明确要求收集训练日志、指标、资源、规划文档等
  高信息量产物并在本机打包；这授权只读下载服务器小文件和创建本地交付包，不授权删除、
  覆盖远端产物或搬运大checkpoint。
- 新任务开始先完整读取根`PROJECT_CONTEXT.md`、`HANDOFF.md`和RLT唯一专题计划；确认
  当前停点仍是formal自然完成，final checkpoint保留服务器，不启动第二个run或resume。
- 首轮本地盘点命令：

  ```powershell
  Get-ChildItem -LiteralPath 'exports' -Force |
    Sort-Object LastWriteTime |
    Select-Object Name,Length,LastWriteTime

  Get-ChildItem -Recurse -File `
    -LiteralPath 'exports\rlt_stage1_formal_high_info_20260729_v2' |
    Select-Object FullName,Length

  rg --files .tmp |
    rg 'formal250|rlt_stage2_formal_8env250|source_config|resolved|provenance|runtime'
  ```

  结果确认可沿用Stage 1包的`README + runtime + TensorBoard + config/provenance +
  visuals + SHA256`结构；本地已经有final event、完整driver和resources，以及已冻结
  resolved config、预算、精确命令和provenance。
- 本包冻结为“非模型高信息量包”：包含完整训练日志、TensorBoard event、资源CSV、
  source/resolved config、budget/provenance/command/stop条件、final checkpoint的小型
  completion/trainer-state元数据、最终统计与三张图、专题计划/参数依据/启动/最终报告和
  本流水账。明确排除十个checkpoint权重、replay payload、数据集、Stage 1大模型和重复
  中间状态快照，以控制体积并避免把恢复材料误当作可独立resume包。
- 使用全新、不覆盖的本地staging目录：

  ```text
  exports/rlt_stage2_formal_8env250_high_info_20260730_v1/
  ```

  创建`runtime/tensorboard`、`config`、`dependencies`、`results`、`visuals`、
  `docs/evidence`和`tools`子目录后，使用`Copy-Item -LiteralPath`逐项复制：
  - final `.tmp/formal250_final_20260730_230104`中的完整driver、resources和event；
  - `.tmp/formal250_live_20260730_195508`中的resolved、exact command、provenance、
    budget、stop conditions和started-at；
  - formal source overlay与独立20-seed bank；
  - Stage 1高信息包中的accepted artifact manifest；
  - cycle250 summary JSON、三张PNG和交互HTML；
  - `00/04/05/09/10/15`六份关键文档、完整实施账本和两份分析脚本。

  第一轮26个复制文件合计7,455,594 bytes。随后用`apply_patch`新增包内`README.md`、
  `METRIC_GLOSSARY.md`和不含checkpoint payload的
  `results/final_run_and_checkpoint_audit.json`；最终payload为29个文件、
  7,469,394 bytes。
- 第一轮secret/text QA命令使用以下类别：

  ```powershell
  Get-ChildItem -Recurse -File -LiteralPath $root |
    Select-String -Pattern `
      'BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|SEETA_SSH_PASSWORD\s*=\s*[''"][^$<]'

  Get-ChildItem -Recurse -File -LiteralPath $root -Filter '*.json' |
    ForEach-Object {
      Get-Content -Raw -Encoding utf8 -LiteralPath $_.FullName |
        ConvertFrom-Json | Out-Null
    }
  ```

  最初过宽的`SEETA_SSH_PASSWORD\s*=\s*[^$<]`误报账本内两处安全的
  `$env:SEETA_SSH_PASSWORD = $match.Groups[1].Value`示例；收紧为只检查带引号的静态
  赋值后secret hits为0。text QA又发现服务器原始`config/exact_command.txt`有一处行尾
  空格；这是原始证据字节，不修写，只精确白名单该一个文件/一处，其他23个文本均UTF-8
  无BOM且无行尾空白。6个JSON全部可解析。
- 使用PowerShell逐项生成`PACKAGE_MANIFEST.json`和`CONTENTS_SHA256.txt`：

  ```powershell
  $payloadFiles = Get-ChildItem -Recurse -File -LiteralPath $root |
    Where-Object { $_.Name -notin @('PACKAGE_MANIFEST.json','CONTENTS_SHA256.txt') } |
    Sort-Object FullName

  $payload = foreach ($file in $payloadFiles) {
    [ordered]@{
      path = $file.FullName.Substring($root.Length + 1).Replace('\','/')
      bytes = $file.Length
      sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).
        Hash.ToLower()
    }
  }
  ```

  由于Windows PowerShell的`Set-Content -Encoding utf8`给manifest加了BOM，立即使用
  `.NET UTF8Encoding($false)`做等内容无BOM机械重写，再重新生成contents hashes。
  最终manifest校验29个payload/7,469,394 bytes，contents校验30个文件；逐项存在性、
  byte count和SHA-256均一致，errors=0。
- 压缩命令：

  ```powershell
  Compress-Archive `
    -LiteralPath `
      'exports\rlt_stage2_formal_8env250_high_info_20260730_v1' `
    -DestinationPath `
      'exports\rlt_stage2_formal_8env250_high_info_20260730_v1.zip' `
    -CompressionLevel Optimal

  Get-FileHash -Algorithm SHA256 -LiteralPath `
    'exports\rlt_stage2_formal_8env250_high_info_20260730_v1.zip'
  ```

  在同步更新专题计划第12节的final停点后，对staging中的该计划做`Copy-Item -Force`，
  重新生成manifest/contents，并用`Compress-Archive -Update`更新同一新建交付ZIP。
  最终ZIP大小2,286,757 bytes，SHA-256为
  `158282a1e38151b39d4e9ba1f6d173855c0bade413c803c29827d325e2771b96`；同名
  `.zip.sha256`已写入。
- 为让包内账本payload字节数与最终manifest一致，最后只在staging账本副本中做等长度数字
  更正并再次更新ZIP。
- 最后用全新`.tmp/rlt_stage2_formal_8env250_high_info_20260730_v1_zipqa3`目录执行
  `Expand-Archive`，从解压副本重新检查manifest逐文件byte/SHA和contents逐文件SHA：
  31个总文件、29个payload、30条内部hash，外部ZIP hash一致，errors=0。包内账本记录到
  压缩前完整QA；ZIP自身hash由本工作区权威账本和包外`.sha256`保存，避免自引用循环。

### A140 — 从global_step_250继续约10小时的只读可行性审查

- 用户追加询问能否“什么都不改”从250断点继续到次日、约10小时。本轮把它解释为：
  模型、算法、环境、batch、UTD、schedule、eval/save cadence等训练语义不变；只允许
  改变继续训练必需的`resume_dir`、绝对总终点`runner.max_steps`和新的输出目录。该问题
  先做只读审查，尚未取得启动续训的明确授权。
- runner源码核对命令：

  ```powershell
  rg -n "max_steps|resume_dir|global_step|start_step|load_checkpoint|resume" `
    '.rlt-impl-worktree\rlinf\runners\embodied_runner.py'

  Get-Content -Encoding utf8 `
    '.rlt-impl-worktree\rlinf\runners\embodied_runner.py' |
    Select-Object -Skip 160 -First 35

  Get-Content -Encoding utf8 `
    '.rlt-impl-worktree\rlinf\runners\embodied_runner.py' |
    Select-Object -Skip 470 -First 105
  ```

  结果：worker初始化后先从`resume_dir/actor`加载checkpoint，再从目录名
  `global_step_250`恢复`global_step=250`；训练循环是
  `range(start_step, max_steps)`，所以`max_steps`是绝对总cycle，不是新增cycle数。
  继续时不能仍填250，否则循环为空。
- RLT resume源码核对确认load前会fail-closed验证：completion=true、schema、两actor
  ranks/world size、目录step、rank state存在及SHA、Stage 1 manifest/model、norm stats、
  H/C/D/z、route/replay/bootstrap、优化器/schedule/batch和weight-sync fingerprint。
  随后SAC基类恢复model/optimizer/scheduler/target/replay，RLT层恢复
  `update_step=102260`、lifetime totals和warm-up anchors，再重算ready/ramp/pending。
  `max_steps`、eval/save interval和输出路径没有放进算法resume fingerprint，因此只改总
  终点/输出不会伪造合同不匹配。
- 从最终TensorBoard读取真实稳定阶段用时：

  ```powershell
  @'
  from pathlib import Path
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  import numpy as np
  p = next(Path(".tmp/formal250_final_20260730_230104").glob("events.out.tfevents.*"))
  ea = EventAccumulator(str(p), size_guidance={"scalars": 0})
  ea.Reload()
  rows = [(v.step + 1, v.value) for v in ea.Scalars("time/step")]
  for start, end in [(192, 250), (201, 250), (226, 249)]:
      values = np.array([value for cycle, value in rows if start <= cycle <= end])
      print(start, end, len(values), values.mean(), np.median(values), values.sum() / 3600)
  '@ | python -
  ```

  P4 cycle192–250平均156.48秒；最近50个含cycle225/250评估与保存的平均为152.25秒；
  最近24个普通cycle平均139.71秒。评估/保存cycle约427–449秒。按相同每25-cycle
  eval/save cadence，新增230 cycles约9.7小时，加checkpoint加载与启动约为10小时。
  因而建议绝对总终点`max_steps=480`，即从250再跑230 cycles，而不是500。
- 新增230 cycles对应约1,840 train episodes、最多368,000 primitive action slots；
  按P4实测约109 macro transitions/cycle，预计新增约25k transitions、
  约125k critic与62.5k actor updates。global step 275–475每25步评估/保存，480作为
  train-end再评估和保存，总计新增10个评估点与10个checkpoint。
- 主要风险不是显存，而是resume主链此前按用户要求被省略，未真实执行过
  save→load→continue；以及前一run的EnvWorker/anon在10.8小时中持续增长并触及cgroup
  240GiB。若后续获准启动，应把正式续训进程的首个完整cycle设为健康门，必须看到
  checkpoint preflight通过、replay/`update_step=102260`连续、ready=1、ramp=1、
  student控制、首轮更新与sync正常，再让同一进程继续；不另造改变定义的smoke。

### A141 — 最终交付元数据回填与工作区QA

- 包内manifest自检完成后，最终ZIP因staging账本副本的一处等长度数字更正而发生最后一次
  hash变化；用`apply_patch`把工作区`HANDOFF.md`与本账本中的旧大小/hash统一改为最终值，
  未再修改或重压交付包。
- 最终核验命令：

  ```powershell
  $zip = Resolve-Path `
    'exports/rlt_stage2_formal_8env250_high_info_20260730_v1.zip'
  $shaFile = Resolve-Path `
    'exports/rlt_stage2_formal_8env250_high_info_20260730_v1.zip.sha256'
  $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $zip
  [pscustomobject]@{
    Zip = $zip.Path
    Bytes = (Get-Item -LiteralPath $zip).Length
    SHA256 = $hash.Hash.ToLower()
    Recorded = (Get-Content -Raw -LiteralPath $shaFile).Trim()
  }
  ```

  结果：`Bytes=2286757`，计算值与包外记录均为
  `158282a1e38151b39d4e9ba1f6d173855c0bade413c803c29827d325e2771b96`。
- 同一批QA尝试在文档根目录执行`git diff --check`，返回
  `Not a git repository`。原因是该根目录是交接/文档工作区而不是RLT实现worktree；
  这不是内容校验失败，也没有因此切换、初始化或修改任何Git仓库。RLT实现仍由独立
  `.rlt-impl-worktree`维护。
- 首次把`foreach (...) { ... } | Format-Table`直接写入一行QA命令时，PowerShell解析为
  `EmptyPipeElement`，整行在执行任何检查前即退出。修复为先把循环结果赋给`$docRows`，
  再单独执行`$docRows | Format-Table`；复测得到ZIP实际/记录hash一致，三个修改文档均为
  UTF-8无BOM且尾随空白行数为0。

### A142 — formal `global_step_250 → 480` 续训授权、准备、启动与首cycle健康门

- 用户明确批准：保持模型、算法、环境、UTD、batch、BC/Q schedule和评估/保存频率不变，
  从`global_step_250`继续到绝对总终点`runner.max_steps=480`，使用新输出目录；启动后只
  观察到正常开始便停止轮询。该授权包含服务器prepare、启动和必要健康检查，不包含删除、
  覆盖原formal250产物或后续持续监控。
- 密码仍只注入每次Paramiko helper的当前进程环境；helper固定host-key并关闭key/agent
  自动尝试。密码未写入本账本、脚本、Git、Memory或服务器文件。所有远程调用统一为：

  ```powershell
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file '<本条对应脚本>'
  ```

- 首先执行只读脚本
  `local_scripts/remote_rlt_20260730_resume480_readonly_preflight.sh`。2026-07-30
  23:44:00+08:00现场：
  - repo为`codex/rlt-pi0-robotwin`，HEAD
    `46a2d19bae629eaa57830f5faeac71ac81a1a494`，worktree clean，ahead upstream 2；
  - 原formal为`exit_code=0`、22:23:41完成；`global_step_250`为879,468,486 bytes、
    34,864 files，completion SHA
    `5d4185b3782aa227c22be257328302fbe6a05f7bc0aede54114f1303b444ae15`；
  - completion为`complete=true`、world-size2、saved runner step250、
    `update_step=102260`、contract SHA
    `82cd409bef1549afb3feb41fa5a80ed08d207d112e4dfe8020af14f49cad1fc9`；
  - rank0/1 state SHA与manifest一致，replay分别17,170/17,681条；
  - 无train/Ray/GPU进程；两卡0MiB，cgroup anon约0.29GiB、OOM/OOM-kill为0，
    数据盘可用879,603,601,408 bytes。cgroup current约158GiB主要是可回收file cache，
    host MemAvailable约979GiB，不清cache。
- 新运行固定为：

  ```text
  source checkpoint:
    /root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1/
    robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1/checkpoints/global_step_250
  run root:
    /root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1
  experiment:
    robotwin_adjust_bottle_rlt_stage2_formal_resume250_to480_v1
  runtime/evidence:
    /root/autodl-tmp/experiment_exports/
    rlt_stage2_formal_resume250_to480_20260730_v1/runtime
  ```

  复用`robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250`，CLI仅显式覆盖新log root/name、
  `runner.max_steps=480`和`runner.resume_dir=<上述checkpoint>`。resolved SHA-256为
  `cbbfffda43a6ca17ee938da21d7f71ccb70ba394d1247b8e5ae8d3f48dda5787`。
  新旧resolved机器diff只包含这四项和由新`RLT_LOG_ROOT`派生的train/eval
  `video_base_dir`两项；algorithm/actor/env控制语义/rollout/weight-sync均完全相同。
- `local_scripts/remote_rlt_20260730_resume250_to480_prepare.sh`逐次问题与窄修复：
  1. 第一次prepare在resolved diff处fail closed：除预期四项外还发现两个派生视频输出
     目录。它们必须随新run root隔离，否则会写回原formal目录；将精确允许集合扩为这六项，
     没有放宽算法字段。
  2. 第二次prepare到Stage 1 binding preflight时，工具按设计拒绝覆盖第一次已成功生成的
     JSON。一次把多行远程检查直接放入`run '<command>'`的本地调用又因PowerShell/argparse
     引号拆分失败，未建立远程命令、未改服务器；随后改用
     `.tmp/remote_rlt_resume480_partial_preflight_inspect.sh`，确认JSON
     `passed=true`、SHA
     `9085afbc1d6046a78df3d0bec532854fbf2e66ef251821ce2939323d03511cd7`。
     prepare改为存在时逐字段验证并复用，不删除或覆盖。
  3. 第三次prepare语义校验通过，但脚本末尾回显发现`${formal_cmd[@]}`在生成的heredoc中
     被折叠成一个带空格的“可执行文件名”。这时尚未调用launcher；改成逐参数显式命令，
     同时只允许复用本次已知pre-launch文件，拒绝driver/PID/start/exit/resources等启动
     痕迹。
  4. 第四次prepare全部通过：source checkpoint/state/replay、Stage 1 artifact、
     config parity、`bash -n`和foreground命令回显均成功，23:50:51标记
     `RLT_STAGE2_RESUME250_TO480_PREPARED`。前三次均在启动前退出，未产生训练进程。
- 续训增量预算：230 cycles、1,840 train episodes、最多368,000 primitive action
  slots、预计约25,070 macro transitions、约125,350 critic/62,675 actor updates；
  275–475每25 cycles以及最终480共10次fixed-20评估和10个checkpoint。12小时
  `timeout --signal=TERM --kill-after=180s 43200s`只是故障保险，预计约10小时完成。
- 用
  `local_scripts/remote_rlt_20260730_resume250_to480_launch.sh`执行唯一launcher。
  2026-07-30 23:51:08+08:00启动成功：

  ```text
  driver PID   657385
  monitor PID  657386
  target       480
  ```

  第一次状态刷新时driver/Ray正在初始化；23:52:48日志明确出现
  `Resuming training from checkpoint directory .../global_step_250`。RoboTwin可选
  Curobo planner的导入traceback与原formal250启动日志逐字同类，随后环境/rollout继续，
  不是fatal；resume行之后没有fingerprint、trainer-state、CUDA/NCCL/Ray fatal。
- 首个完整续训cycle在日志中显示`Global Step: 251/480`，step time144.16秒，
  ETA约9h10m；8条student rollout成功7条。首cycle TensorBoard step为250（runner
  zero-based记录），精确值：

  | 指标 | 值 |
  |---|---:|
  | 恢复后的pre-update `rlt/update_step` | 102,260 |
  | 本cycle critic / actor updates | 510 / 255 |
  | ready / ramp / actor switch | 1 / 1 / 1 |
  | BC / Q weight | 2.5 / 0.45 |
  | pending debt | 0 |
  | actor / critic loss | -0.08148 / 0.001327 |
  | actor / critic grad norm | 2.8280 / 0.16977 |
  | train success | 7/8 |

  这证明模型/optimizer/target/replay/schedule与`update_step`从250状态继续，并在同一进程
  执行新更新；不声称RNG bitwise continuation，因首版合同明确不保存逐位RNG状态。
- `local_scripts/remote_rlt_20260730_resume250_to480_health_gate.sh`的第一次精确
  TensorBoard读取发生在异步logger尚未flush后半组tag时，因找不到`rlt/update_step`
  fail closed；训练不受影响。用`.tmp/remote_rlt_resume480_tb_tags.sh`只读列出tag，
  一次过早复测仍因同一异步flush顺序空匹配退出；继续等待后发现第二cycle部分tag已写入。
  健康脚本改为选择**精确step250**而不是latest，并对float32 `q_weight`使用`1e-6`
  容差；23:58:44返回
  `RLT_RESUME250_TO480_FIRST_CYCLE_HEALTH_PASS`。
- 健康门终态：driver/monitor仍存活，resource monitor已有201行；两卡现场约
  17.0/17.1GiB，cgroup anon约35.0GiB，OOM/OOM-kill仍为0。原250 checkpoint只作load
  source；新TensorBoard从step250记录，后续完整曲线必须拼接旧run 0–249和新run
  250–479。按用户要求到此停止主动轮询，进程继续后台运行。
- 本地交接QA首次又把`foreach {...}`结果直接接到管道，PowerShell在执行任何检查前报
  `EmptyPipeElement`；没有改文件。改为先赋值`$fenceRows`再格式化后，4份Markdown和5份
  本轮shell脚本均为UTF-8无BOM、LF、尾随空白0；Markdown fence计数均为偶数，当前
  HANDOFF/专题计划不再含“续训尚未启动”的旧状态。

### A143 — formal续训480终态只读验收、指标/资源分析与制图

- 用户要求“同样地”查看当前训练。本轮授权解释为：服务器只读刷新进程、完成状态、日志、
  TensorBoard、resources和checkpoint；允许在本地下载小型高信息量证据、分析、制图和更新
  文档；不启动/停止进程，不删除、覆盖、清cache或修改服务器产物。
- 第一轮现场状态命令：

  ```powershell
  $env:SEETA_SSH_PASSWORD='<process-only secret>'
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file `
    '.\local_scripts\remote_rlt_20260730_resume250_to480_status.sh'
  Remove-Item Env:SEETA_SSH_PASSWORD
  ```

  `2026-07-31T09:22:05+08:00`现场已显示：driver/monitor退出、`exit_code=0`、最终
  `480/480`、GPU归零、10个checkpoint存在。该状态脚本输出较长，因此随后增加窄的
  final audit 和 export index，不依赖被截断的console做最终统计。
- 使用`apply_patch`新增完整只读脚本：

  ```text
  local_scripts/remote_rlt_20260731_resume480_final_audit.sh
  local_scripts/remote_rlt_20260731_resume480_export_index.sh
  ```

  final audit 精确读取identity、Git/dirty/upstream、resolved及provenance hash、
  start/finish/exit、进程/Ray/GPU、fatal、TensorBoard tags/stats/eval、resource CSV、
  10个checkpoint的completion/state/replay、总字节/文件数、live cgroup和磁盘；
  export index只枚举小文件并展示traceback上下文与final completion。
- 第一次调用final audit时误用了旧环境变量名：

  ```powershell
  $env:AUTODL_PASSWORD='<process-only secret>'
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file `
    '.\local_scripts\remote_rlt_20260731_resume480_final_audit.sh'
  Remove-Item Env:AUTODL_PASSWORD
  ```

  helper在建立任何SSH连接前返回
  `SEETA_SSH_PASSWORD is required in the current process`；因此没有远程操作或状态变化。
  修复为正确的进程变量后原样重试：

  ```powershell
  $env:SEETA_SSH_PASSWORD='<process-only secret>'
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file `
    '.\local_scripts\remote_rlt_20260731_resume480_final_audit.sh'
  Remove-Item Env:SEETA_SSH_PASSWORD
  ```

  `2026-07-31T09:24:59+08:00`精确终态：
  - `23:51:08 → 09:17:41`，`9h26m33s`，`exit_code=0`；
  - driver/monitor/Ray/train进程均不存在，两卡0MiB；
  - final completion为`complete=true`、world-size2、step480、
    `update_step=215055`，rank0/1 state SHA分别为
    `193dfa8d...28e87c1`和`ebd426d9...90af444`；
  - 10个checkpoint为275–475每25步加480，总计
    11,967,404,588 bytes / 482,255 files；final480为
    1,415,135,834 bytes；
  - 新run root共11,970,639,364 bytes / 482,258 files；数据盘可用
    867,102,216,192 bytes；
  - OOM/OOM-kill为0。cgroup live current约180.3GB，其中file约177.2GB十进制、
    anon约0.31GB；进程已退出，说明主要是page cache。
- 为区分两个启动traceback是否fatal，执行：

  ```powershell
  $env:SEETA_SSH_PASSWORD='<process-only secret>'
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file `
    '.\local_scripts\remote_rlt_20260731_resume480_export_index.sh'
  Remove-Item Env:SEETA_SSH_PASSWORD
  ```

  traceback只来自两rank导入未使用的可选Curobo planner失败，随后立即加载norm stats、
  恢复250 checkpoint并完整训练到480。CUDA OOM、NCCL、Ray death、NaN/Inf、训练
  ERROR均为0；故记录为非阻塞告警，不安装额外依赖。
- 使用`apply_patch`新增
  `local_scripts/download_rlt_resume480_status.py`，只下载driver、resources、TensorBoard、
  config/provenance、runtime状态和10组completion/replay metadata。实际命令：

  ```powershell
  $env:SEETA_SSH_PASSWORD='<process-only secret>'
  python '.\local_scripts\download_rlt_resume480_status.py' `
    '.\exports\rlt_stage2_formal_resume250_to480_high_info_20260731_v1'
  Remove-Item Env:SEETA_SSH_PASSWORD
  ```

  结果为50个远端文件、5,874,658 bytes；SFTP只建立一次连接。脚本生成
  `download_manifest.json`，逐项保存remote path、local name、byte count和SHA-256。
  本地复核为0 missing/0 mismatch；加manifest自身后文件夹51个文件、约5.62MiB。
  不下载checkpoint权重或replay trajectory `.pt`。
- TensorBoard tag与step口径检查：

  ```powershell
  @'
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  from pathlib import Path
  for p in [
      Path("exports/rlt_stage2_formal_8env250_high_info_20260730_v1/runtime/tensorboard"),
      Path("exports/rlt_stage2_formal_resume250_to480_high_info_20260731_v1"),
  ]:
      event = next(p.glob("events.out.tfevents.*"))
      ea = EventAccumulator(str(event), size_guidance={"scalars": 0})
      ea.Reload()
      print(event)
      print("\n".join(sorted(ea.Tags().get("scalars", []))))
  '@ | python -
  ```

  两段各89个相同scalar tags，分别覆盖step0–249和250–479；按cycle=step+1，无缺口、
  无重叠覆盖1–480。优化tag从cycle136出现是SAC阶段边界，不是日志缺失。
- 使用`apply_patch`新增可复现绘图脚本
  `exports/rlt_stage2_formal_resume250_to480_high_info_20260731_v1/tools/`
  `plot_rlt_complete480.py`。执行：

  ```powershell
  python `
    '.\exports\rlt_stage2_formal_resume250_to480_high_info_20260731_v1\tools\plot_rlt_complete480.py' `
    '.\exports\rlt_stage2_formal_8env250_high_info_20260730_v1' `
    '.\exports\rlt_stage2_formal_resume250_to480_high_info_20260731_v1' `
    '.\exports\rlt_stage2_formal_resume250_to480_high_info_20260731_v1\visuals'
  ```

  生成success/update、optimization和resource三张全程图，以及
  `rlt-summary-complete480.json`。图只使用真实TensorBoard/CSV，不制造cycle0或插值成功率。
- 第一次图QA发现两个表达问题：
  1. success图把final TensorBoard的pre-update `update_step=214580`标成终值；而final
     cycle随后执行475 updates，checkpoint终值是215055。修复为同cycle
     `update_step + critic_updates_run`，并加cycle-grid一致性断言；
  2. 狭窄P2/P3阶段标题横向重叠，final480 eval标签贴右边界。修复为缩短阶段名、P3纵向
     错层和final标签左移。

  每次窄修复后均原样重跑绘图，并用本地图片查看工具分别按original/high检查三张PNG。
  最终success图显示完整20个eval点、P1–P4、resume虚线和精确215055；optimization图
  无断线或轴误导；resource图明确显示进程重启处RSS归零与file cache保留。
- 指标终审：
  - train P1/P2/P3/P4为`14.44%/14.47%/31.76%/89.32%`；
    续训251–480为`1690/1840=91.85%`，最后50/20/10 cycles为
    `92.50%/93.75%/91.25%`；
  - 续训10个fixed-20 eval合计`178/200=89%`，前后5点为90%/88%，final
    `17/20=85%`；相对cycle250只少1条，判定高位平台波动，不判定明确退化；
  - critic/actor updates从102260/51130连续到215055/107528；新增global replay
    22559条，新增critic112795，严格满足UTD5，pending0；
  - final actor/critic loss为`-0.132776/0.000813`，grad为`2.160/0.133`；
    Q0/Q1/Qdata为`0.3683/0.3489/0.3306`。grad远低于clip10、critic loss稳定，
    Q缓升但eval平台化，提示收益递减而非当前数值故障。
- 资源终审基于续训14,936行、原run17,032行2秒CSV：
  - 续训GPU active mean为29.0%/30.1%，P95/peak均100%，显存峰19.37/19.56GiB；
  - matched/Env RSS峰78.75/52.46GiB，cgroup anon/file/current峰
    73.03/165.89/240.00GiB；
  - `memory.max`增量11,485，但OOM/OOM-kill为0，PSI当前均0；
  - 训练结束后matched RSS归零、anon约0.29GiB，约165GiB file cache不是活进程泄漏；
  - 磁盘可用减少11.64GiB至807.55GiB；230 cycles粗均值147.8秒，未见整体减速。
- 阅读文档时第一次误用旧的短目录名
  `docs/rlt-pi0-robotwin`，三个`Get-Content/Get-ChildItem`只在本地返回
  `PathNotFound`，没有写操作。根据`HANDOFF.md`路由修正为
  `docs/rlinf-robotwin-pi0-rltoken`并显式`-Encoding utf8`后成功读取。
- 使用`apply_patch`创建
  `17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md`，并更新本账本、专题索引和
  根交接。完整结果、图、服务器/本地产物路径和当前停点均写入17号报告。本轮没有执行
  stop/start/delete/cache-drop、服务器代码修改、Git commit或push。
- 最终本地QA命令对`download_manifest.json`的50个条目逐项重算byte/SHA，errors=0；
  HANDOFF、专题索引、17号报告、本账本和快照README均为UTF-8无BOM、Markdown fence为
  偶数、尾随空白为0并以换行结束。credential字符串全文搜索无匹配；`rg`因此按设计返回
  exit1，不是QA失败。加入绘图脚本、summary、README和三张PNG后，快照最终为57个文件、
  7,654,908 bytes。首次QA时17号报告仍把“下载核心51文件”与“完整文件夹含图”写成同一
  口径；根据实测数字窄修正为下载核心51文件约5.62MiB、完整文件夹57文件约7.30MiB，
  未修改服务器或重新下载。
- 文档和图QA后，为保证“当前”结论不依赖09:24快照，新增并执行只读最小复核：

  ```powershell
  $env:SEETA_SSH_PASSWORD='<process-only secret>'
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file `
    '.\local_scripts\remote_rlt_20260731_resume480_final_live_check.sh'
  Remove-Item Env:SEETA_SSH_PASSWORD
  ```

  `2026-07-31T09:44:57+08:00`仍为exit0、finish09:17:41、driver/monitor/train/Ray全0、
  两卡0MiB/0%，final completion=true/step480/update215055/world-size2、OOM/OOM-kill0，
  数据盘可用867,102,216,192 bytes；故最终聊天结论使用该时点。
- 最小live复核写入文档后再次跑Markdown终态QA时，一次又把PowerShell
  `foreach {...}`直接接管道，parser在执行任何检查前返回`EmptyPipeElement`；没有文件
  或服务器操作。立即改为`$finalQaRows = foreach (...) {...}`再单独
  `$finalQaRows | ConvertTo-Json`，复测5份文档仍全部BOM=false、trailing=0、
  newline=true且fence计数为0/26/8/308/2（均为偶数）。

### A144 — 480终态后的Git/云端同步

- 用户明确要求检查未推送内容并与云端Git简要同步；该授权包含只读审计、把当前RLT
  最终文档形成docs-only提交、把已有RLT提交推送到既有`personal`远端，以及push后
  ahead/behind复核。不包含清理本地dirty镜像、改remote、安装代理、force push、删除或
  重写历史。
- 新任务先完整读取`PROJECT_CONTEXT.md`、`HANDOFF.md`和RLT专题SSOT，再读取
  `06_AUTODL_NETWORK_PLAYBOOK.md`。大陆网络沿用无proxy、默认Git HTTP、HTTPS
  `personal` remote、短探针和有界push，不因一次超时改网络配置。
- 本地Git审计：

  ```powershell
  Get-ChildItem -Force -LiteralPath .
  git -C '.\.rlt-impl-worktree' branch --show-current
  git -C '.\.rlt-impl-worktree' rev-parse HEAD
  git -C '.\.rlt-impl-worktree' status --short --branch
  git -C '.\.rlt-impl-worktree' remote -v
  git -C '.\.rlt-impl-worktree' rev-list --left-right --count `
    '@{upstream}...HEAD'
  ```

  根文档目录第一次直接调用Git被`dubious ownership`拒绝；没有写配置。按既有安全做法
  使用单命令`git -c safe.directory=<当前绝对路径> -C . ...`复核，确认根`.git`只是
  `master`无任何commit、无remote、全部文件未跟踪，不能从这里push。
  `.rlt-impl-worktree`是从本地bundle建立的旧开发镜像：
  `codex/rlt-pi0-robotwin@48a775db...`，origin指向本地bundle，ahead/behind 0/0；
  它仍保留实施期dirty/untracked代码与配置，但服务器权威分支已经把这些实现形成提交并
  继续演进。因此本轮不提交、不清理、不覆盖该镜像，也不把它误称为待推送云端提交。
- 使用`apply_patch`新增只读
  `local_scripts/remote_rlt_20260731_git_sync_audit.sh`，精确远端调用为：

  ```powershell
  $env:SEETA_SSH_PASSWORD='<process-only secret>'
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file `
    '.\local_scripts\remote_rlt_20260731_git_sync_audit.sh'
  Remove-Item Env:SEETA_SSH_PASSWORD
  ```

  `2026-07-31T09:57:21+08:00`服务器真值：
  - repo `/root/autodl-tmp/RLinf_rlt_pi0_robotwin`；
  - branch `codex/rlt-pi0-robotwin`，HEAD
    `46a2d19bae629eaa57830f5faeac71ac81a1a494`，worktree clean；
  - upstream `personal/codex/rlt-pi0-robotwin`，left/right `0/2`；
  - 未推送提交为`2df23e7f docs(rlt): close Stage 2 fresh handoff`和
    `46a2d19b feat(rlt): add 8-env formal evaluation protocol`；
  - 两提交共改6个允许文件、210 insertions/1 deletion，包含8-env250/resource-smoke
    config、20-seed bank、seed分区测试、HANDOFF和RLT账本，没有实验产物或checkpoint。
- 同次网络诊断：无proxy环境变量，Git HTTP为DEFAULT；GitHub main在10秒内
  HTTP000 timeout，API为HTTP200，raw收到数据但10秒timeout，`timeout 15 git
  ls-remote`返回124。按网络手册这是主站smart-HTTP链路暂不可用，不是认证、仓库或提交
  失败；因此没有立即盲push或改proxy/remote。
- 为纳入刚完成的cycle480终态，新增服务器暂存目录
  `/root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/staging`。命令：

  ```powershell
  $env:SEETA_SSH_PASSWORD='<process-only secret>'
  python '.\local_scripts\remote_exec_autodl.py' `
    --host connect.bjb1.seetacloud.com --port 36406 --user root `
    run --command-file `
    '.\local_scripts\remote_rlt_20260731_git_sync_prepare_staging.sh'

  python '.\local_scripts\remote_exec_autodl.py' ... put `
    '.\HANDOFF.md' `
    '/root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/staging/HANDOFF.md'
  python '.\local_scripts\remote_exec_autodl.py' ... put `
    '.\docs\rlinf-robotwin-pi0-rltoken\00_INDEX_AND_IMPLEMENTATION_PLAN.md' `
    '<staging>/docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md'
  python '.\local_scripts\remote_exec_autodl.py' ... put `
    '.\docs\rlinf-robotwin-pi0-rltoken\evidence\IMPLEMENTATION_LOG.md' `
    '<staging>/docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md'
  python '.\local_scripts\remote_exec_autodl.py' ... put `
    '.\docs\rlinf-robotwin-pi0-rltoken\17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md' `
    '<staging>/docs/rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md'
  Remove-Item Env:SEETA_SSH_PASSWORD
  ```

  目录是新建且fail-if-exists；四次SFTP只写隔离staging，没有直接覆盖repo。密码仅在当前
  PowerShell进程环境中，未写入脚本、账本或staging。
- 第一次apply/commit脚本先通过四文件SHA和UTF-8/BOM/final-newline/trailing-whitespace，
  但Markdown fence检查错误地使用`text.count("```")`，把正文内联提到的三个反引号也
  当成围栏，因账本总计为奇数而fail closed。失败发生在`git add/commit`之前；repo只留下
  四个允许docs文件的未暂存工作树版本，没有提交、push或代码/config变化。修复为只统计
  正则`(?m)^````匹配的行首围栏，并把重入门从“必须完全clean”收紧为“允许且只允许这四个
  已验证docs路径”，以便在不reset用户内容的情况下继续同一事务。
- 修复后原样重跑apply/commit，四文件QA、精确allowlist和staged diff gate均通过，形成
  docs-only提交：

  ```text
  d81267b57eb5ce13e6452139aaeba02af3911624
  docs(rlt): record cycle 480 formal closeout
  ```

  提交修改4个文件、2773 insertions/17 deletions，其中新增17号最终报告；服务器
  worktree随后clean，相对远端由ahead2变为ahead3。该提交没有修改算法、配置、checkpoint、
  数据集或实验产物。
- 服务器随后执行
  `local_scripts/remote_rlt_20260731_git_sync_bounded_push.sh`。脚本先短探针
  `github.com`和`git ls-remote`；主站仍HTTP000且`ls-remote`返回124，因此按网络手册
  输出`PUSH_SKIPPED_NETWORK_UNAVAILABLE`并以75退出，没有进入认证或pack阶段，也没有
  改远端。
- 为避免用旧Windows开发镜像倒灌服务器，服务器从云端已知基线
  `9bb2dd78feff7133780c3df6a88618d10168c4e4`到当前HEAD生成仅含ahead3的增量bundle：

  ```text
  /root/autodl-tmp/experiment_exports/rlt_git_sync_20260731_0957_v1/
    codex-rlt-ahead3.bundle
  bytes=69651
  sha256=fe9ec6782030b46ed81e845ebd56178590055499194fbbed186f5e14912f9404
  ```

  bundle下载到
  `C:\Users\86136\Documents\rl\.tmp\rlt_git_sync_20260731_0957_v1\`后SHA一致。本机另建
  全新bare relay：先从GitHub只读fetch精确云端HEAD，再`git bundle verify`并fetch
  bundle；`merge-base --is-ancestor`确认`9bb2dd78 -> d81267b5`是fast-forward，
  `git fsck`通过。旧`.rlt-impl-worktree`全程只读，未参与fetch、commit、bundle或push。
- Windows第一次relay push在任何写入前失败：

  ```text
  error: cannot spawn sh: No such file or directory
  fatal: could not read Username for 'https://github.com': terminal prompts disabled
  ```

  push后`git ls-remote`仍为`9bb2dd78...`。只读诊断确认Git为
  `E:\long\Git\cmd\git.exe`、Credential Manager 2.5.0可单独运行、`sh.exe`存在于
  `E:\long\Git\bin`；加入该路径后Git trace仍显示失败发生在内部
  `run_command: 'git credential-manager get'`的shell启动层。没有修改global/system
  Git配置。
- 拟改用已配置Credential Manager在当前PowerShell进程内直接提供一次认证头：不打印、
  不落盘、push后清除。但Windows sandbox在启动Credential Manager前以
  `CreateProcessAsUserW failed: 5`拒绝；按规则申请一次沙箱外执行后，安全审查又因
  “未把具体提交批次和具体GitHub目的地写成精确授权”拒绝。两次均发生在读取凭据和push
  之前，不能用替代路径绕过。当前云端仍为`9bb2dd78...`，服务器为clean
  `d81267b5...`、ahead3；下一步只等待用户精确批准把这3个提交fast-forward到
  `Yutenji-Nyamu/rlinf_fastwam`的`codex/rlt-pi0-robotwin`。
- 本专题机器角色由此冻结为：
  1. 服务器独立worktree是compose/test/smoke/train与“实际跑过”的运行时真值；
  2. GitHub同名分支是源码和文档的持久、协作与灾备真值；
  3. Windows根目录主要承载交接文档、diff、图和小型证据，RLT本地镜像应可从云端重建；
  4. `.tmp`下bundle/bare relay只作一次性传输，不是开发真值；
  5. 任何本地dirty镜像都不得反向覆盖服务器，除非先逐提交证明来源、差异和
     fast-forward关系。

  因此“本机不运行项目”不等于“可以不维护”：长期源码副本应跟云端一致或明确标记为
  disposable；未提交文档在同步前仍可能是唯一副本。冗余清理须等云端核验后先列精确
  目录、大小和可恢复性，再另行取得删除授权。
- 为把上述边界形成可精确授权的服务器提交，使用`apply_patch`新增：

  ```text
  local_scripts/remote_rlt_20260731_git_sync_prepare_authority_docs.sh
  local_scripts/remote_rlt_20260731_git_sync_commit_authority_docs.sh
  ```

  prepare脚本要求branch不变、HEAD精确为`d81267b5...`、worktree clean和新staging
  不存在；commit脚本只允许`HANDOFF.md`与本账本两条路径，执行UTF-8/BOM/newline/
  whitespace/fence、`diff --check`和staged allowlist后形成docs-only提交。
  本地首次QA发现commit脚本为了断言“文档不含密码”而把密码原文本身写进否定检查，导致
  `CredentialLiteral=true`。该脚本尚未上传或执行；立即用`apply_patch`删除该原文断言，
  保留不依赖任何秘密值的通用格式检查。修复后再执行Bash语法和文本QA，确认本批脚本无
  BOM、无CRLF、无尾随空白，也不再包含密码原文。
