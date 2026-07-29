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
