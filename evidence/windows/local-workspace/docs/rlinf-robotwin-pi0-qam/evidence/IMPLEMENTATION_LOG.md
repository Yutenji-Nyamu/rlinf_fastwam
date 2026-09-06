# QAM × π0 × RoboTwin：实施与验证流水账

最后更新：2026-07-31。

本文记录从上下文准备开始的每个操作批次。它不是计划；当前规范见
[`../00_INDEX_AND_IMPLEMENTATION_PLAN.md`](../00_INDEX_AND_IMPLEMENTATION_PLAN.md)。

## 0. 记录规则

每个实施/测试/运行批次至少记录：

1. ID、时间、授权边界和停点；
2. 目的；
3. 去凭据后的精确命令、cwd、环境；
4. 目标文件、直接上游/下游；
5. 来源 commit/path/symbol；
6. 修改文件与关键增删；
7. stdout/stderr、退出码、产物路径和 hash；
8. 问题、一个直接原因、窄修复与复测；
9. Git 前后状态；
10. 未验证项与下一停点；
11. 运行批次的 GPU/RAM/cgroup/磁盘、吞吐、OOM 和残留进程。

密码只写作 `[PROCESS-ONLY SECRET REDACTED]`，绝不记录明文。

状态词只使用：

- `冻结`
- `待事实验证`
- `待运行批准`
- `延后`

## QAM-CTX-0001：根上下文与既有专题路由

时间：2026-07-28 18:30–18:50 CST。

授权：本地只读和文档整理；无代码实施、服务器写入、测试或运行授权。

目的：

- 读取工作区稳定规则和当前交接；
- 把 DSRL/RLT 的工程经验与算法语义分开；
- 确认 QAM 应拥有独立 SSOT。

读取：

```text
PROJECT_CONTEXT.md
HANDOFF.md
docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-traditional-rl/evidence/IMPLEMENTATION_LOG.md
docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md
docs/rlinf-robotwin-pi0-traditional-rl/01_FULL_REFERENCE_HISTORY_20260728.md
```

结果：

- 冻结复用：独立 branch/worktree、config-opt-in、server-only tests、细粒度账本、
  fixed-input/gradient ownership/resume/sync、smoke 批准包；
- 禁止继承：DSRL/RLT 的算法变量、reward、H/N、UTD、capacity、checkpoint schema；
- 文档结构确定为一个主计划、一个来源地图、一个动态流水账。

问题/修复：无。

Git：仓库尚无 commit，现有工作区文件均为用户内容；未更改全局 `safe.directory`，
后续只在单条 Git 命令使用 `-c safe.directory=...`。

停点：进入官方 source lock 和服务器只读盘点。

## QAM-CTX-0002：官方 QAM source lock

时间：2026-07-28 18:35–19:05 CST。

授权：互联网和本地临时只读研究 clone；无 vendoring 或依赖安装。

去凭据命令：

```powershell
git ls-remote https://github.com/ColinQiyangLi/qam.git HEAD refs/heads/main refs/tags/*
git clone --filter=blob:none --no-checkout https://github.com/ColinQiyangLi/qam.git .tmp/qam-official-2726d767
git -C .tmp/qam-official-2726d767 checkout 2726d767c9a0a7a46d49693f0391f73dc2cf58ac
git -C .tmp/qam-official-2726d767 rev-parse HEAD
```

注：第一次尝试把临时 clone 放在 `C:\tmp`，因该路径写权限失败；窄修复为放到工作区
`.tmp/`，未修改权限。

结果：

- official `main` HEAD：
  `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`；
- 无 tag/release；
- 临时 clone HEAD exact，working tree clean；
- 核对 `agents/qam.py`、`utils/networks.py`、`utils/datasets.py`、
  `experiments/reproduce.py`；
- 论文/项目页/仓库一致指向作者实现；
- QSM May fix 不涉及 `agents/qam.py`。

确认的 QAM core：

- behavior flow + fine flow；
- memoryless SDE；
- target-critic terminal action gradient；
- target-slow-flow lean VJP；
- AM regression；
- 10-Q pessimistic target；
- plain/FQL/edit 路由；
- 官方 manipulation 只验证 5×5D chunk。

未执行：JAX 环境安装、官方训练、数值 parity。

停点：把官方实现作为 P1 oracle，不直接复制 JAX 工程。

## QAM-CTX-0003：AutoDL 身份与现场只读盘点

时间：2026-07-28 18:56–19:04 CST。

授权：服务器只读；禁止停止进程、写文件、建分支、下载、测试和训练。

认证：

```text
Paramiko.Transport(host, port)
start_client(timeout=20)
auth_password(user, [PROCESS-ONLY SECRET REDACTED])
host key SHA256 verified
hostname; pwd; id -u
```

结果：

```text
hostname = autodl-container-nekaqbwt43-6ce5babb
pwd      = /root
uid      = 0
```

Git/代码：

```text
/root/autodl-tmp/RLinf
  branch = local/openpi-a800-2gpu-migration
  HEAD   = 6d0db56bf26f972cd27fa29535f5eb939e80e5bf
  dirty  = 只有已知本地 A800 PPO/GRPO config 与 local_scripts 未跟踪项

/root/autodl-tmp/RLinf_fastwam_rlinf
  branch = codex/dsrl-pi0-robotwin
  HEAD   = d664bf349b63b75f41d51c8295cb0a330780d783
  dirty  = clean
```

QAM 定向检查：

- 无 QAM branch/worktree；
- `git grep` 未见 QAM/Adjoint Matching；
- `/root/autodl-tmp` 未见候选 QAM clone；
- SAC-Flow、NFT、OpenPI explicit velocity 路径存在。

模型/数据：

- `adjust_bottle` π0 SFT checkpoint 存在；
- 两个 model shard 约 4.28 GB 和 3.78 GB；
- norm stats SHA-256：
  `649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a`；
- 定向搜索未发现 clean-50/LeRobot RL 数据；
- 已知 PPO/GRPO run roots 未发现 replay/trajectory/dataset 产物。

资源/进程：

- DSRL formal v1 已于约 18:54 CST 启动，driver PID `70062`；
- 两张 A800 80 GB 当时各约 14.9 GB 已用；
- cgroup memory limit 240 GiB，`oom=0`、`oom_kill=0`；
- `/root/autodl-tmp` 当时约 789 GB 可用。

操作问题：

- 第一条组合审计命令在本地参数解析阶段因引号失败，未建立远程连接；
- 原因：PowerShell → argparse → remote shell 的嵌套引号；
- 窄修复：用 PowerShell here-string 保存远端只读命令，再作为单个参数传入；
- 复测：身份、Git、进程、资源、模型和数据审计全部退出码 0。

未执行：

- 未写服务器；
- 未停止/改变 DSRL；
- 未创建 branch/worktree；
- 未运行 import/test；
- 未下载数据。

停点：任何未来“当前状态”都要重刷；QAM 实施不得使用正在运行的 DSRL worktree。

## QAM-CTX-0004：RLinf 调用链与关键语义核对

时间：2026-07-28 19:00–19:20 CST。

授权：本地代码副本和已保存 audit 只读。

来源：

```text
.research-rlinf/rlinf/models/embodiment/openpi/openpi_action_model.py
.research-rlinf/rlinf/workers/actor/fsdp_nft_policy_worker.py
.research-rlinf/rlinf/workers/actor/fsdp_sac_policy_worker.py
.research-rlinf/rlinf/models/embodiment/flow_policy/flow_policy.py
.research-rlinf/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml
audits/20260719-robotwin-performance-analysis/source/rlinf_robotwin_env.py
```

确认：

- OpenPI 显式 velocity 入口存在且不必通过普通 sampling；
- OpenPI 与 QAM 时间反向、velocity 符号相反；
- π0 内部 `[50,32]` 与 RoboTwin active 14D 必须显式投影；
- 当前 RoboTwin `chunk_step` 只给 chunk-final observation；
- NFT 适合借显式 velocity/FSDP 接线；
- SAC-Flow 适合借 off-policy 外壳；
- PPO/GRPO 只提供 RoboTwin/π0 数据面，不提供 QAM loss；
- DSRL 的 latent/reward/H/N 不是 QAM 语义。

派生的首要风险：

1. 若 Q 评分 50 步而只执行前 N 步，suffix 非因果；
2. 若把 QAM `t` 直接传 OpenPI，向量场方向错误；
3. 若通过 no-grad rollout sampling 做 adjoint，VJP 图不存在；
4. 若只存 final observation，官方 overlapping primitive Q-chunk 不能 exact 复现；
5. 若没有 RL transition dataset，π0 checkpoint 不能替代 QAM offline 数据。

问题/修复：无代码修改；把这些风险提升为主计划的 P1/P2 parity gate。

## QAM-CTX-0005：交付前现场轻量刷新

时间：2026-07-28 19:21 CST。

授权：服务器只读；不读取训练数据、不改进程。

检查：

```text
date / hostname / pwd / id -u
两棵 worktree 的 branch / HEAD / scoped status
tracked QAM symbol 与浅层 QAM 路径扫描
DSRL formal driver 存活
nvidia-smi
cgroup memory.current / memory.max / memory.events
df -h /root/autodl-tmp
```

结果：

- 身份不变；
- `/root/autodl-tmp/RLinf@6d0db56b...` 的已知未跟踪项不变；
- `/root/autodl-tmp/RLinf_fastwam_rlinf@d664bf34...` clean；
- QAM symbol/path 扫描仍为空；
- DSRL driver PID `70062` 已运行约 1,644 秒，仍存活；
- GPU0/GPU1 分别约 `29761/29475 MiB` 已用；
- cgroup `memory.current=247216775168`、
  `memory.max=257698037760` bytes，`oom=0`、`oom_kill=0`；
- `/root/autodl-tmp` 约 789 GB 可用。

结论：QAM 实施仍未物化；当前资源由 DSRL formal 使用，不启动 QAM probe。

问题/修复：无。

## QAM-DOC-0001：建立专题 SSOT、来源索引和账本

时间：2026-07-28 19:13–19:32 CST。

授权：本地文档写入。

新增：

```text
docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-qam/01_CONTEXT_AND_SOURCE_MAP.md
docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
```

计划更新：

```text
HANDOFF.md
```

QA：

- UTF-8 replacement character：0；
- 三份专题 Markdown 均以 newline 结束；
- code fence 和 `$$` display-math delimiter 成对；
- 相对本地链接全部存在；
- 明文密码、host/port credential command、私钥：未发现；
- scoped Git status 只包含本条列出的四个文档；
- 独立交叉审阅发现并窄修：
  - 补齐官方 `behavior-FM + fine-AM` actor objective；
  - 区分 Q tilt `inv_temp` 与 EMA `tau`；
  - 锁定 pre-update-online-parameter EMA；
  - 区分 mean-Q terminal gradient 与 pessimistic TD bootstrap；
  - 补齐 exact `t+h` reverse recurrence 和 final-valid gate；
  - 修正 `target_actor_fast` 为“随机初始化但未同步/更新/使用”；
  - 把名义 50-step 提交与真实 executed length 分开；
  - 消除 `HANDOFF.md` 中 DSRL “未启动/已启动”的时间冲突。
- 修复后复审：上述实质问题均无残留。

结果：

- QAM SSOT、上下文来源地图、动态账本和根路由均已建立；
- 没有修改 `PROJECT_CONTEXT.md`，因为长期执行规则已经存在，无需复制；
- 没有修改代码、配置、依赖或服务器状态。

当前停点：P0 完成；等待方法讨论和“开始实现”授权。

## QAM-CTX-0006：官方当前性、模型与调用链复核

时间：2026-07-29 10:58–11:25 CST。

授权：公开一手来源与本地锁定源码只读；不安装、不运行官方训练。

来源/命令：

```text
arXiv 2601.14234 submission history / v4 HTML
ICLR 2026 virtual poster 10006800
QAM official project page
GitHub ColinQiyangLi/qam README and locked source paths
git ls-remote https://github.com/ColinQiyangLi/qam.git HEAD
```

`git ls-remote` 结果：

```text
2726d767c9a0a7a46d49693f0391f73dc2cf58ac    HEAD
```

源码复核：

```text
agents/qam.py:
  critic_loss
  adj_matching
  actor_loss
  _update
  sample_actions
  compute_flow_actions
  create/get_config
utils/networks.py:
  MLP
  ActorVectorField
  Value
utils/datasets.py:
  Dataset.sample_sequence
  ReplayBuffer
main.py:
  offline update
  offline -> online preload
  primitive transition insertion
  H-step sampling/update
```

结果：

- arXiv 当前仍是 v4（2026-05-18），ICLR 2026 Poster；
- 仓库 HEAD 未漂移，GitHub 显示 15 commits、无 tag/release；
- README 的 05-09 修复针对 QSM baseline，不是 QAM core；
- 正式实验全部是 OGBench/MuJoCo 仿真，无真机或 π0/VLA；
- observation 是 flat state；behavior/fine 是两个 4×512 MLP；critic 是 10 个
  4×512 MLP；
- 官方 plain 对应 trainable behavior FM + target-slow EMA + 独立完整 fine flow；
  主复现 `residual=False`；
- 官方 update 数据流确认为 behavior FM、fine SDE/terminal mean-target-Q gradient/
  target-slow VJP/AM、10-Q pessimistic TD、联合 Adam、pre-update-online EMA；
- 官方 online replay 插入 primitive transition，并重叠抽 H-step sequence；online
  restore 不受支持。

问题/处理：

- 直接打开固定 commit 的 `raw.githubusercontent.com` 首次返回 cache miss；改由 GitHub
  blob/raw 页面和已锁定本地研究 clone交叉读取，未改变 source pin。
- 没有运行官方代码；本条只证明 source semantics，不是数值 parity。

## QAM-CTX-0007：服务器、生产环境与 worktree 只读刷新

时间：2026-07-29 11:10–11:14 CST。

授权：服务器只读；密码只注入当前进程，认证后移除；不改进程、文件、branch、env 或数据。

去凭据命令形状：

```text
python local_scripts/remote_exec_autodl.py \
  --host connect.bjb1.seetacloud.com --port 36406 --user root \
  run '<identity/process/GPU/cgroup/disk/worktree/data/env read-only probe>'
```

检查项：

```text
hostname; pwd; id -u; date
pgrep -af train/ray/dsrl
nvidia-smi --query-gpu=...
memory.current / memory.max / memory.events
df -h /root/autodl-tmp
git -C /root/autodl-tmp/RLinf worktree list --porcelain
限定 maxdepth 的 QAM/RLT/clean-50/env 路径扫描
shared .venv pyvenv.cfg / python -V / du -sh / pip list
```

结果：

- 身份：`autodl-container-nekaqbwt43-6ce5babb`、`/root`、UID 0；
- DSRL driver、Ray 与监控已退出；进程扫描只匹配本次 probe shell；
- GPU0/GPU1：A800 80GB，均 `0 MiB`、utilization `0%`；
- cgroup：`memory.max=257698037760`，`oom=0`、`oom_kill=0`；
- `/root/autodl-tmp` 约 694GB 可用；
- baseline：
  `local/openpi-a800-2gpu-migration@6d0db56bf26f972cd27fa29535f5eb939e80e5bf`；
- DSRL：
  `codex/dsrl-pi0-robotwin@acc7c14b93aec8eb2f2e8f32e4072be3957b761b`；
- 无 RLT/QAM worktree、目标 clean-50 ZIP/partial 或 QAM 数据目录；
- shared `.venv`：Python 3.11.14，约 14GB；torch 2.6.0、JAX/JAXlib 0.5.3、
  Flax 0.10.2、Optax 0.2.8；OGBench/Distrax 未安装。

问题/修复：

- 首次用内嵌 `python -c` 查询 package metadata 时，远端 shell 引号被 helper
  重解释，产生 `SyntaxError`；身份、资源和路径输出仍有效。
- 窄修为只读
  `python -m pip list --format=freeze | grep -Ei ...`，复测成功。
- 由绝对 shebang/解释器链接和“生产端无新增依赖”共同决定：不复制/改名 shared
  `.venv`，不在其中安装官方 QAM。官方 JAX oracle 使用独立 source tree/CPU venv。

## QAM-DATA-0001：clean-50 source、converter 与 provenance 锁定

时间：2026-07-29 11:08–11:32 CST。

授权：官方文档/API、锁定源码和服务器路径只读；未下载、解压或转换。

数据 source lock：

```text
repo: TianxingChen/RoboTwin2.0
revision: 9dc9299c163db059931898a9f0852098a61155a1
file: dataset/adjust_bottle/aloha-agilex_clean_50.zip
size: 298,659,710 bytes
expected SHA-256:
  5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e
```

converter source：

```text
upstream RoboTwin:
  13c3c47ff4312dd62484bcd51be034af55c062d1
server RoboTwin:
  c3ddfa8b97d5519efa828b075999bd0006778e5e
```

本任务涉及的四个 converter 在两 commit 间相同。SHA-256：

```text
process_data_pi0.sh:
  868c92fbb76b9b7b2a8d1ecd630237df8d320c7aa3a37b5bd0e704d8c0d49bbe
scripts/process_data.py:
  b462918bf3f41f6d2fc30c3498381ac3cc7d8ce7a8bd6333fafb925e7d9d5590
generate.sh:
  d0a065051bbc1db08bd0486150b5908e15c6682df4df343c9723708d5bbf1eee
LeRobot converter:
  b8f0829329e099b7246b3d6467cec3ea4d60767eedd219b825d0b7f26bb7c373
```

字段结论：

以下是官方 converter、文件命名与元数据的静态结论，不是 archive 实物验收；ZIP 尚未
下载，单 episode schema 必须在下载后确认：

- clean-50 是成功 expert trajectory，不是原生 RL transition dataset；
- raw 有逐帧三相机/qpos 和 episode boundary；
- converter 用 `qpos[t+1]` 作为 `obs[t]` 的 action proxy；
- reward、terminal、truncation 未原生记录；
- success demo 可派生 terminal `0,...,0,1`，但必须标 provenance；
- LeRobot 丢最后一帧 next observation，QAM sidecar 必须从 raw HDF5 建 frame index；
- failure/timeout/observed reward/end 由 online replay 提供。

冻结数据布局：

```text
/root/autodl-tmp/datasets/robotwin2/
  source/<revision>/
  raw/<revision>/adjust_bottle/
  canonical/pi0-aloha-clean50-v1/
  rlt/
  qam/transitions-v1/
```

问题/处理：

- 官方 converter 对已存在 `repo_id` 会删除重建，episode 遍历和 instruction 选择默认
  不完全确定。规划只增加固定排序/seed/source→episode manifest 的薄 wrapper，不建
  复杂兜底。
- RLT/QAM 必须确定唯一下载/转换 owner；另一方只读最终 manifest，不能并发覆盖。

## QAM-DOC-0002：方法教学、环境/数据和逐文件计划 v2

时间：2026-07-29 11:25–11:36 CST。

授权：本地专题文档写入；无代码、配置、服务器写入。

新增：

```text
docs/rlinf-robotwin-pi0-qam/02_METHOD_AND_PORT_DECISION_GUIDE.md
```

修改：

```text
docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-qam/01_CONTEXT_AND_SOURCE_MAP.md
docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
HANDOFF.md
```

文档分工：

- `00`：唯一规范性 SSOT；
- `01`：source/data/live 精确索引；
- `02`：术语、官方/计划调用链、B/F/C/M 用户决策教学；
- 本文件：逐命令、结果、问题、修复和复测。

实质更新：

- 回答官方 repo/论文状态、仿真/真机、模型结构和“主要抄谁”；
- 解释 behavior/fine、FM、SDE、adjoint、VJP、AM、10-Q、target、EMA、
  temperature、plain/F/E；
- 明确官方 plain 是 B2+F1，B1/F2 是 π0 适配；
- 明确官方是 C0 flat state，没有视觉 encoder 可抄；
- 解释 primitive-faithful 与 macro-QAM、H_model/N、time/sign 和 active projection；
- 将 D1 冻结为 shared behavior data + derived sidecar + authoritative online replay；
- 冻结生产/shared venv 与独立 JAX oracle 环境边界；
- 给出新增/窄改/conditional/明确不改的逐文件候选清单与双重 opt-in；
- 删除 D1/D2 的并列未决状态；保留 macro/primitive 由用户决定，B/F/C/N
  由最小 probe 后决定。

当前停点：

- 没有创建 QAM branch/worktree；
- 没有修改 RLinf 源码或 config；
- 没有下载数据/依赖；
- 没有运行 compose/import/test/smoke/training；
- 等待文档 QA 与用户下一项方法选择。

## QAM-DOC-0003：专题文档机械 QA 与一致性收口

时间：2026-07-29 11:33–11:36 CST。

授权：本地 Markdown 只读检查与窄文档修正；不是项目代码测试，无服务器写入。

检查范围：

```text
HANDOFF.md
docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-qam/01_CONTEXT_AND_SOURCE_MAP.md
docs/rlinf-robotwin-pi0-qam/02_METHOD_AND_PORT_DECISION_GUIDE.md
docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
```

检查项与结果：

- 五个文件均可严格 UTF-8 解码；`U+FFFD=0`、无 BOM、文件末尾均有 LF；
- code fence 与 `$$` 显示公式分隔符全部成对；
- 无重复标题；
- 七个相对 Markdown 链接目标全部存在；
- 外部 HTTP 链接只做来源锁定，不把在线可达性混入本地机械检查。

问题/修复：

- 第一次 PowerShell 汇总把 `foreach` 代码块直接接到管道，解析时报
  `EmptyPipeElement`；改为先收集 `$qaRows` 再输出，复测通过。该失败发生在解析阶段，
  未产生写入。
- 澄清“上下文共几份”：自动生效的 `AGENTS.md` + 两个根路由/规则文件 + 四份 QAM
  专题文件，共七份有效文档；若只数主动读取项则为六份，QAM 专题自身仍只有四份。
- 将主计划中可能误读为已选 B1 的句子改为“π0 SFT 是初始化来源；若选 B1 才冻结”。
- 修正教学调用链中的时间映射文字为
  `t_qam -> t_pi0=1-t_qam`，并将 replay 名称与候选文件
  `qam_transition_replay.py` 对齐。

最终状态：

- 机械格式 QA 通过；随后进行的实质一致性审计与修正记录在 QAM-DOC-0004；
- 没有新增代码、配置、数据、环境、分支或服务器运行；
- 当前唯一适合用户现在决定的方法门仍是 M1 primitive-faithful 或 M2 macro-QAM。

## QAM-DOC-0004：方法、数据与 RLinf 调用链实质一致性审计

时间：2026-07-29 11:36–11:53 CST。

授权：本地锁定源码与专题文档只读审查、窄文档修正；无项目代码、配置或服务器写入。

精确源码检查：

```powershell
rg -n 'class EmbodiedRunner|def run\(|rollout_worker|actor_worker|sync|update_weights|weight' `
  .research-rlinf/rlinf/runners/embodied_runner.py `
  .research-rlinf/rlinf/workers/rollout/hf/huggingface_worker.py `
  .research-rlinf/rlinf/workers/actor/fsdp_nft_policy_worker.py `
  .research-rlinf/examples/embodiment/train_embodied_agent.py

Get-Content .research-rlinf/rlinf/runners/embodied_runner.py
Get-Content .research-rlinf/rlinf/workers/rollout/hf/huggingface_worker.py
rg -n -C 5 'def sync_model_to_rollout|weight_syncer|state_dict\(' `
  .research-rlinf/rlinf/workers/actor/fsdp_actor_worker.py `
  .research-rlinf/rlinf/workers/actor/fsdp_sac_policy_worker.py
```

源码结论：

- `EmbodiedRunner` 调 actor→rollout weight sync；不是 actor worker 直接调用 env；
- rollout worker 通过 `hf_model.predict_action_batch()` 生成动作，并将同步权重应用到
  `hf_model`；
- actor worker 已有 rollout state-dict 与 parameter-name filter 接口；
- 因此 QAM active inference route 必须成为 OpenPI `hf_model` 的 opt-in 子模块/route；
  critic、target、optimizer 和 replay 必须留在 actor worker。

实质问题与窄修：

- 统一 A/B 与 M1/M2 命名；未选前把 replay、contract 和 P3 都写成条件路线；
- 不再称服务器 common baseline 为 clean；实施前重新读取 exact commit 的 HEAD/status；
- 明确官方 B2+F1 只指拓扑/update；官方 slow/fast 独立初始化，π0 双拷 SFT 是适配；
- 明确 B1 不用 clean-50 更新 behavior；数据只作 derived-success 接口/critic 温启动；
- 纠正调用链为 rollout `predict_action_batch` 出动作、actor 更新后 filtered sync；
- 冻结同步所有权：F1 只同步 active fine；F2 同步 adapter，仅 B2+F2 再同步 online
  behavior；target/critic/optimizer/replay 不进 rollout；
- 增加实际执行长度 $L<N$ 的 `executed_action_mask`；$L{:}N$/suffix/padding
  terminal gradient 必须为 0，bootstrap 使用 $\gamma^L$；
- 明确 clean-50 没有 query boundary：M1 可建 overlapping primitive window，M2 只能
  建带 provenance 的 deterministic `derived_macro_segment`；
- 将 clean-50 字段降格为 converter/metadata 预期；source lock 不等于 archive/schema
  已验证，下载后只做 hash/archive/单 episode schema 与 mask 前检；
- 解释 C1/C2 下 10-Q 的独立性：共享 encoder + 10 head 是相关 ensemble 适配，
  10 套 encoder+head 才更接近官方但成本高；
- 增加 MIT provenance 规则：实质翻译保留 attribution，clean rewrite 保留
  公式/官方 symbol/本地 symbol 映射；
- 删除教学文档中重复的逐文件矩阵；唯一规范性矩阵只留主计划 §9。

修改：

```text
docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-qam/01_CONTEXT_AND_SOURCE_MAP.md
docs/rlinf-robotwin-pi0-qam/02_METHOD_AND_PORT_DECISION_GUIDE.md
docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
HANDOFF.md
```

未改：

- `HANDOFF.md` 中 DSRL 历史由对应窗口收口；QAM 窗口不并发重写其他专题；
- RLinf/RoboTwin/OpenPI 源码、config、环境、数据、分支/worktree；
- 服务器进程与运行产物。

## QAM-DOC-0005：normalized Q action 与 env action 合同及最终 QA

时间：2026-07-29 11:55–12:05 CST。

授权：本地锁定源码只读审查与专题文档窄修；无项目代码/配置/服务器写入。

精确源码检查：

```powershell
Get-Content .research-rlinf/rlinf/models/embodiment/openpi/openpi_action_model.py
Get-Content .research-rlinf/rlinf/models/embodiment/openpi/__init__.py
Get-Content .research-rlinf/rlinf/models/embodiment/openpi/policies/aloha_policy.py
```

问题：

- 旧图把 $P_N$ 的 `[N,14]` 直接连到 env，但实际代码先保留 normalized
  `outputs["actions"]`，再经 `output_transform` 的 `Unnormalize + AlohaOutputs`
  生成 env action；
- `forward_inputs["model_action"]` 与 `forward_inputs["action"]` 已分别保存 transform
  前后动作；若不明确坐标，normalized replay、env execution 和 Q gradient 会混用。

修复：

- `model_action`：normalized `[H_model,32]`；
- $P_N$：选择 normalized `[N,14]`，供 critic/VJP/replay；
- `output_transform`：独立生成 env `[N,14]`；
- 实际执行后用同一 $L$/mask 裁两套 action；replay 保存两者，critic 只读 normalized；
- Q gradient 不穿 NumPy/robot transform；
- resume/manifest 保存 norm stats 与 output-transform fingerprint；
- P2、Q3、方法教学、来源索引和 `HANDOFF.md` 同步更新。

最终本地 Markdown QA：

- 五个文件严格 UTF-8，`U+FFFD=0`，均以 LF 结尾；
- code fence、`$$`、行内 `$` 与反引号全部闭合；
- 无重复标题；
- 八个相对 Markdown 链接目标全部存在；
- 过时词扫描无匹配；
- 实质复查确认动作坐标合同修复后无剩余阻塞冲突。

QA 中的问题/处理：

- 朴素行内 `$` 计数器曾把 code span 中的 `$qaRows` 误报为数学分隔符不平衡；
  改为先移除成对 code span 后复测通过，文档本身无错误；
- 过时词 `rg` 无匹配时按工具约定返回 1；用显式 no-match 分支复测并返回 0。

边界：

- 这是文档 QA，不是项目代码测试；
- 本轮未运行 compose/import/compile/pytest/smoke/training；
- 所有项目测试仍只允许在服务器实施阶段执行。

## QAM-CTX-0008：服务器身份、worktree、资源、数据与环境再次只读刷新

时间：2026-07-29 12:33–12:34 CST。

授权：服务器只读。密码只注入当前 helper 进程，关闭 key/agent 探测；未创建、修改、
下载、安装、停止或启动任何服务器对象。

命令流水：

```text
尝试 1：
python local_scripts/remote_exec_autodl.py ... run <PowerShell 多行内嵌命令>

结果：
本地 argparse 返回 "unrecognized arguments"；命令尚未发送到服务器，不是 SSH
认证失败。

尝试 2：
python local_scripts/remote_exec_autodl.py ... run "<PowerShell 单行内嵌命令>"

结果：
仍在本地 argparse 分词阶段返回 "unrecognized arguments"；同样未触发 SSH。

窄修与复测：
python local_scripts/remote_exec_autodl.py ... \
  run-file local_scripts/remote_rlt_20260729_readonly_refresh.sh

随后用 apply_patch 临时建立：
.tmp/qam_readonly_refresh_20260729.sh

再执行：
python local_scripts/remote_exec_autodl.py ... \
  run-file .tmp/qam_readonly_refresh_20260729.sh

最后用 apply_patch 删除临时脚本。
```

说明：

- 改用 `run-file` 只为避免 Windows/argparse/远端 shell 的三层引号重解释；
- helper 使用低层 Paramiko Transport、password auth、keepalive 与只对
  banner/EOF/timeout 的有界重试；
- `CryptographyDeprecationWarning` 不影响本次认证和命令结果；
- 密码未写入命令、脚本、日志或专题文档。

结果：

- 身份：
  `autodl-container-nekaqbwt43-6ce5babb`、`/root`、UID 0；
- 未发现训练/Ray 常驻进程；进程 grep 只匹配本次 probe shell；
- 两张 A800 80GB 均为 `0 MiB / 0%`；`/root/autodl-tmp` 约 694GB 可用；
- baseline：
  `local/openpi-a800-2gpu-migration@6d0db56bf26f972cd27fa29535f5eb939e80e5bf`；
  dirty 只含已知未跟踪 PPO/GRPO YAML 与 `local_scripts/`；
- DSRL：
  `codex/dsrl-pi0-robotwin@48a775db09c16c455aeba7b0600c920e7c80d534`；
- RLT worktree 已建立：
  `codex/rlt-pi0-robotwin@48a775db09c16c455aeba7b0600c920e7c80d534`；
- QAM branch/worktree 仍不存在；
- `/root/autodl-tmp/datasets`、目标 clean-50 ZIP/partial、QAM sidecar 均不存在；
- QAM oracle source/venv 均不存在；
- shared `/root/autodl-tmp/RLinf/.venv`：Python 3.11.14、约 14GB；
  torch 2.6.0、JAX/JAXlib 0.5.3、Flax 0.10.2、Optax 0.2.8、Ray 2.55.1、
  openpi-client 0.1；OGBench/Distrax 未安装。

结论：

- 上一份“无 RLT worktree”的 11:10 快照已过时，QAM 专题动态状态更新为本条；
- QAM 仍只有本地规划文档，没有服务器实现环境或数据实物；
- 生产继续复用 shared venv 且不改包；官方 oracle 仍规划为独立环境；
- 本条是只读现场证据，不授权后续创建 worktree、下载、安装或测试。

## QAM-DOC-0006：初学者方法、环境与数据教学补全

时间：2026-07-29 12:34–12:43 CST。

授权：本地专题文档整理；无项目代码/配置和服务器写入。

修改：

- `02_METHOD_AND_PORT_DECISION_GUIDE.md`
  - 新增最小词典，区分 observation/action/reward/return、actor/critic、target/EMA、
    bootstrap，以及 flow terminal 与 episode terminal；
  - 解释官方 QAM 与经典 Diffusion Policy 的关系及模型规模；
  - 解释 Q 是独立 off-policy critic、与 SAC 的共同点/差异和一次 Plain QAM update；
  - 用二维例子解释 VJP，明确它仍是反向传播但不构造完整 Jacobian；
  - 用数字例子解释 10-Q、mean-minus-std TD bootstrap、target critic 和 terminal
    action gradient；
  - 展开 Plain/QAM-F/QAM-E，并与官方 `residual=True`、本项目 F1/F2 分开；
  - 补充 B/F/C 所需的 action expert、prefix、proprio、Q head 和共享 encoder 术语；
  - 补充 behavior FM、critic TD、online replay 各自的数据最低合同与 clean-50 gap；
  - 用“代码桌面/工具箱”解释 commit、branch、worktree、venv、PYTHONPATH 与污染边界；
  - 将服务器现场刷新为 12:33–12:34 的结果。
- `00_INDEX_AND_IMPLEMENTATION_PLAN.md`
  - 只更新 QAM 专题的动态服务器快照和 P0 事实，不改变方法选择或实施授权。
- `01_CONTEXT_AND_SOURCE_MAP.md`
  - 更新动态资产表，并补充初学者问题到方法指南的索引。
- 根 `HANDOFF.md`
  - 只更新 QAM 路由的动态快照；未修改其他专题的方法语义。

边界：

- 没有新增专题文件，避免形成并行计划；
- 没有修改任何实现代码、配置或依赖；
- 没有运行本地项目测试；只运行了文档机械 QA；
- B1/B2、F1/F2、C1/C2、M1/M2 与执行 `N` 均未在本条替用户拍板。

QA 流水：

1. 首次直接执行 `git diff -- ...` 返回 “Not a git repository”。后续核对表明当前
   sandbox 用户触发 Git `dubious ownership`，且根仓本身是无可用 tracked baseline 的
   文档容器；没有修改 global `safe.directory`，也没有用该失败结果声称 diff clean。
2. 首轮 display-math 检查把账本 code span 中记载的三个字面 `$$` 当成公式分隔符，
   误报账本不平衡。
3. 窄修 QA 逻辑：先排除 fenced code 和 inline code，再统计 display/inline math；
   复测五个文件通过：
   - strict UTF-8，`U+FFFD=0`；
   - LF-only 且末尾有 LF；
   - code fence、display math、inline math 成对；
   - 无重复标题；
   - 相对 Markdown 链接目标全部存在。
4. 当前态过时词扫描在 `HANDOFF.md`、`00`、`01`、`02` 中未发现 `11:10`、
   `无 RLT/QAM` 或旧 DSRL `acc7c14b...`；历史账本保留旧快照作为时间证据。

## QAM-DOC-0007：Plain 主线收束、critic 适配与 clean-50 充分性

时间：2026-07-29 13:00–13:29 CST。

授权：继续讨论与整理本地 QAM 专题文档；无项目代码/配置和服务器写入。

用户本轮决定：

- 首版先做 Plain、action-space QAM；
- 目标是保留 QAM 主干高层语义，在现有 π0 baseline 上先跑通并验证是否涨点；
- 继续讲清 critic、VJP、10-Q 和 clean-50 数据边界，再决定生产整包。

只读来源复核：

```text
官方论文：
https://arxiv.org/html/2601.14234v4

锁定官方源码：
.tmp/qam-official-2726d767/agents/qam.py
.tmp/qam-official-2726d767/utils/networks.py
.tmp/qam-official-2726d767/utils/datasets.py
.tmp/qam-official-2726d767/main.py

本地 RLinf/OpenPI 副本：
.research-rlinf/rlinf/models/embodiment/openpi/openpi_action_model.py

核对方式：
git -C .tmp/qam-official-2726d767 rev-parse HEAD
git -C .tmp/qam-official-2726d767 status --short
rg -n "class QAM|critic_loss|actor_loss|adj_matching|ensemblize|class Value|sample_sequence" \
  .tmp/qam-official-2726d767
rg -n "_build_prefix_cache|get_velocity|get_value_from_vlm|freeze_vlm" \
  .research-rlinf/rlinf/models/embodiment/openpi/openpi_action_model.py
```

结果：

- 官方本地 clone 为锁定 commit
  `2726d767c9a0a7a46d49693f0391f73dc2cf58ac`，只读检查时 clean；
- 官方 critic 把 flat observation 与 flattened action chunk 拼接，交给 10 个各自
  独立初始化的完整 4×512 LayerNorm MLP；没有视觉 encoder，也不是共享 trunk +
  10 个线性头；
- 官方 `sample_sequence` 构造重叠 primitive H-step action/reward/mask/valid/next
  observation；online replay 也存 primitive transition；
- π0 prefix cache 只含三相机/语言；proprio/noisy action/time 进入 action expert
  suffix，因此 C1 必须在 frozen prefix feature 外显式拼 normalized 14D proprio；
- 现有 scalar value 路线只输出 $V(o)$，不读候选 action，不能提供 QAM 所需
  $\nabla_a Q(o,a)$。

关键方法纠正：

- Q 不给每个 noisy/denoising step 产生 FM 标签；
- 数据动作直接产生 behavior FM 标签；
- target-Q ensemble mean 只在 clean final action 产生一次 action gradient；
- behavior/target-behavior 的 reverse VJP 把该终点方向搬回各 flow time；
- 各时刻得到的是 AM velocity-correction 监督，fine flow 做局部参数反传；
- 10-Q 的 `mean-rho*std` 只用于 TD bootstrap，terminal adjoint 默认只对 target-Q
  ensemble mean 求 action gradient。

生产 v1 收束推荐（等待用户整包确认）：

```text
Plain
+ B1 frozen SFT behavior
+ F1 full fine action expert
+ C1 frozen pi0 prefix/proprio + 10 independent full Q MLP
+ M2 macro replay
+ N=20（H_model 保持 50）
```

准确名称：

> Plain-QAM π0 adaptation（frozen behavior + macro transition）

收束理由：

- B1 用已完成 SFT/FM 的 π0 作为稳定 behavior prior，省去 behavior optimizer 与
  target-slow 状态；这是显式 π0 适配；
- F1 保留完整、多步、与 behavior 同构的 fine action expert，不用 F2 限制 QAM
  主干表达力；冻结 VLM/prefix 可共享，只复制 action expert；
- C1 最接近官方“固定 state 表示 + 10 个独立完整 critic”的高层拓扑，同时保留
  RoboTwin 三视角和 proprio；不在 10-Q 前放共享可训练 bottleneck；
- M2 对齐当前一 query→chunk-final observation 的系统合同，代价是明确的 macro
  adaptation；
- N20 在约 200 primitive-step episode 中最多约 10 个 macro 决策，避免 N50 只有约
  4 个 transition；相较 N5/N10 又控制 π0 query 开销。

失败触发 fallback：

1. F1 实测超显存才退 F2；
2. C1 prefix 接口/可分性失败才转 C2；
3. M2 拿不到真实 final transition 或始终学不出 Q 才升级 M1；
4. N20 fixed-base 控制明显下降才回 N50；credit 太粗且吞吐有余才缩 N10；
5. frozen behavior 明显失配才升 B2。

clean-50 结论：

- 它是 50 个成功 episode，不是 50 个 tensor 样本；
- 可用于动作/norm 合同、成功正例、behavior FM（若 B2）和可选 critic 温启动；
- 当前只有元数据/converter source lock，ZIP 未下载，schema 未实物验收；
- 缺原生 reward/end/failure/timeout/query boundary/policy version；派生成功 sidecar
  只能让 TD 机械可运行，不能独自提供可信 action-Q；
- 推荐先用 frozen SFT 在线采集 success/failure/timeout macro transition，`tau=0`
  训练 10-Q；只有 Q 能基本区分 executed/轻微扰动 action 且 action-gradient finite，
  才打开 `tau>0` AM。

本轮修改：

- `00_INDEX_AND_IMPLEMENTATION_PLAN.md`
  - 记录 Plain 已确认；
  - 把生产路线收束为 `B1+F1+C1+M2+N20` 推荐包；
  - 将 P2/P3 和预计文件面改为推荐路线优先、fallback 失败触发；
  - 补充 FM/AM、C1、10-Q 和 clean-50 的规范边界。
- `02_METHOD_AND_PORT_DECISION_GUIDE.md`
  - 纠正“每步 FM 监督”说法；
  - 展开 VJP/adjoint、10-Q/target/bootstrap 的完整数据流；
  - 写明 B1+F1 与 C1 推荐及其失败门；
  - 写明 clean-50 的机械可训练与可信训练边界。
- `01_CONTEXT_AND_SOURCE_MAP.md`
  - 新增本轮四个核心问题到已有章节的索引。
- 根 `HANDOFF.md`
  - 将 Plain 决定、生产推荐包和 clean-50/online warm-up 门写入当前 QAM 路由。

边界：

- 新增文档：0；
- 未新增或修改实现代码、配置、依赖；
- 未创建分支/worktree；
- 未下载 clean-50；
- 未运行项目 compose/import/test、smoke 或训练；
- 本轮没有用旧服务器快照声称动态状态“当前”，也没有新的服务器写入或进程操作。

文档 QA：

```text
目标：
HANDOFF.md
00_INDEX_AND_IMPLEMENTATION_PLAN.md
01_CONTEXT_AND_SOURCE_MAP.md
02_METHOD_AND_PORT_DECISION_GUIDE.md
evidence/IMPLEMENTATION_LOG.md

检查：
strict UTF-8、BOM/U+FFFD、LF-only、末尾 LF、fenced code、
排除 fenced/inline code 后的 display/inline math、重复标题、相对 Markdown 链接。

结果：
QA_PASS all 5 files
```

补充一致性扫描：

- 当前态文档未再出现“比较 F1/F2、比较 C1/C2 后再选”“按用户选择 M1/M2”或
  “待方法选择”等旧并列措辞；
- 历史账本保留 `QAM-DOC-0006` 当时“尚未拍板”的原始时间证据，不回写历史；
- P2 的名义 `N=50` 只作非终止 correctness bridge，已明确不覆盖生产推荐 `N=20`。

## QAM-DOC-0008：B1+F1 冻结、官方 Q 数据流、VLA 近邻与 planned-N 因果修正

时间：2026-07-29 14:35–15:31 CST。

授权：继续讨论、公开资料只读核查与本地 QAM 文档整理；无服务器写入、代码实现、
数据解压/转换、依赖安装、测试、smoke 或训练授权。

用户本轮方向：

- 首版继续收束为 Plain QAM；
- 选择 B1 frozen SFT behavior + F1 完整 fine action expert；
- 讲清 official Q 数据、VJP、critic 输入、clean-50 价值和公开近邻实现；
- 沿调用链复核仍未清楚的工程合同；
- 不新增文档，不继续堆并行候选和大而全检查。

只读来源与核对范围：

```text
官方 QAM：
https://arxiv.org/html/2601.14234v4
https://github.com/ColinQiyangLi/qam
agents/qam.py::critic_loss / adj_matching / actor_loss / sample_actions
utils/datasets.py::Dataset.sample_sequence
main.py offline/online replay loop

VLA / 独立近邻：
https://arxiv.org/html/2605.00416              # LWD
https://arxiv.org/html/2606.08015              # Q-VGM
https://github.com/rl2-vla/qam/tree/rl2-train
https://github.com/yonghdong/trqam
https://github.com/microsoft/soc-fine-tuning-sd
https://github.com/HaoyunT/lwd_wall_repo

本地审阅：
rg -n "mask/L|executed_action_mask|executed_length|L\\{:}N|planned|未下载|F2" \
  HANDOFF.md docs/rlinf-robotwin-pi0-qam/*.md
Get-Content -Encoding utf8 逐节读取 00/01/02/HANDOFF/账本
```

官方 Q 数据流结论：

1. offline 当前 action 是数据中真实记录的 action，直接进入
   `Q_online(s,a_data)`；
2. online 当前 action 由 active fine policy 生成，但必须先经环境执行并得到
   reward/next-state/end，形成 transition 后才进入 replay；
3. TD target 的 next action 在 update 时由当前 fine policy 在 logged `s'` 生成，只用于
   `target-Q(s',a')` bootstrap，不能凭空创造 transition；
4. 这是 SAC 类 off-policy critic 的常见 current-action/next-action 分工；
5. 官方 OGBench `play/navigate` 是大规模、覆盖多种行为和结果的 transition 数据，不是
   clean-50 式同一任务纯成功 demo；默认 offline dataset 预装 replay，再追加 online
   transition；
6. 专家数据不是没用：官方用它训练 behavior FM、Q 和 AM 的状态分布；在 B1 下
   clean-50 不再训练 behavior，但仍有 action/norm/schema、高回报正例和有限 warm-start
   价值，不能单独约束可信 $\nabla_aQ$。

VJP 教学修正：

- 可以理解为“求中间 noisy action 对终点 Q 上坡方向的敏感度”；
- 仍须先 forward 生成终点和保存各 $x_t$；
- 仍须从终点逐时间步 reverse recurrence；
- VJP 的“直接”只表示直接计算每步 $J^\top g$，不显式构造完整 Jacobian，不表示跳过
  forward 或一跳跨过全部 flow time。

公开近邻结论：

- 官方 QAM 仍是 Plain 数学、update 顺序和 oracle 的唯一主来源；
- LWD 是最接近本任务的 VLA/真机方法证据：$\pi_{0.5}$ 类 actor、QAM、在线冻结 VLM、
  更新 action expert、VLM state feature + action-chunk critic、mixed offline/online
  replay；但未找到可直接抄的完整公开实现，critic 也不是 Plain 10-Q；
- Q-VGM 的 frozen prefix RL token + proprio + action-sensitive Q 支持 C1 输入设计，但
  它不是 adjoint matching；
- RL2-VLA fork 只在预计算 π0/VLA latent 上训练官方小 MLP QAM，不端到端更新 π0；
- TRQAM 是 JAX/low-dim 的交叉检查和未来稳定性参考；
- 非官方 LWD reproduction 未实现完整 reverse-adjoint VJP，不作为主抄对象。

审阅发现并修正的关键因果缺口：

旧计划把执行后才知道的 `executed_length=L`/mask 送入 critic，并用 replay L 裁
update-time fine candidate 的终端梯度。这会泄露中途 success/end outcome，且新候选没有
对应的 realized L。

首版合同改为：

```text
Q input:
    state feature + proprio + planned fixed-N normalized action chunk

realized L:
    只进入 per-chunk return、gamma^L、end/bootstrap 与 execution provenance

terminal gradient:
    planned N 内不按事后 L 裁剪
    仅 π0 model suffix N:50 与 14D 外 padding 经 P_N^T 保证为 0
```

online replay 保存 planned normalized/env chunks 和 realized L。clean-50 的 M2
warm-start 只构造从 episode 终点向前对齐的完整 fixed-N chunk，丢弃不足 N 的开头
remainder，并标 `derived_terminal_aligned_macro`；不再用 partial/executed mask 作为
critic 输入。

B1+F1 参数所有权冻结为：

```text
原 SFT action expert = frozen behavior f_beta
clone(SFT action expert) = 独立 trainable fine f_theta
同一 frozen VLM/prefix = 两者共享
```

生产 B1 无 behavior FM/optimizer/target-slow；F1 不就地更新原 expert，只由 AM 更新并
负责 rollout/TD next action。FM 只在 P1 官方 B2+F1 oracle 和未来 B2 fallback 中存在。

仍待实施证据而非继续方法规划的事项：

1. F1 精确参数/optimizer/峰值显存；
2. C1 pooled feature 的 rollout capture、next-feature、cache fingerprint；
3. M2 payload 是否无损提供 planned chunks、realized L、final observation/end 和
   policy version；
4. 两 GPU 的 critic FSDP/DDP、replay batch owner 与同步；
5. `gamma/rho/batch/warm-up/UTD/capacity/offline-online ratio`；
6. PyTorch/FSDP 联合或分离 optimizer，但必须共享同一 pre-update snapshot。

文档修改：

- **新增文档：0**。
- `00_INDEX_AND_IMPLEMENTATION_PLAN.md`
  - 将 B1+F1 和生产 C1+M2+N20 提升为主线，B2/F2/C2/M1 降为失败触发；
  - 写死 F1 是独立 action-expert copy、B1 生产无 FM；
  - 修正 planned-N/realized-L critic、replay、梯度、target 和 clean-50 分段合同；
  - 删除过时服务器快照/F2 预先 probe，补 C1 feature 与两卡 critic ownership 缺口；
  - 更新 shared ZIP 已下载/archive 已验、QAM 未解压/schema 的当前边界。
- `01_CONTEXT_AND_SOURCE_MAP.md`
  - 按动态/方法/授权拆分可信顺序；
  - 新增 LWD、Q-VGM、RL2-VLA、TRQAM、Microsoft adjoint 与非官方 reproduction 索引；
  - 删除旧服务器快照和未执行下载命令，改为稳定路径与 archive 当前状态；
  - 压缩远程认证重复教程。
- `02_METHOD_AND_PORT_DECISION_GUIDE.md`
  - 新增 B1/F1/C1/M2/N20、FM/AM/TD/VJP/VLM/MLP 直白词典；
  - 新增 official logged-current/generated-next/online-executed 三类 action 数据流；
  - 补 VJP “先 forward、再逐步 reverse、只省 Jacobian”边界；
  - 补真机 vision+proprio 与 official flat observation 的 privilege 区分；
  - 修正 C1、M2、clean-50 和计划调用链的 planned-N 因果合同；
  - 删除 dated server snapshot，合并重复的文件/验证/当前决策章节。
- 根 `HANDOFF.md`
  - 更新 QAM 路由停点和 shared ZIP 状态；
  - 将 QAM 当前节压成方法源、冻结主线、数据、环境、剩余事实和授权九条；
  - 删除相互冲突的旧动态快照和“等待 macro/primitive/整包确认”。

边界：

- 未修改任何 RLinf 实现代码、配置或依赖；
- 未创建/切换 branch 或 worktree；
- 未运行本地项目测试；机械文档 QA 在本批末尾执行并记录；
- 未连接服务器或把旧 snapshot 称为 live current；
- 历史账本的“当时未下载/未确认”保持 append-only，不回写。

机械 QA：

```text
目标：
HANDOFF.md
00_INDEX_AND_IMPLEMENTATION_PLAN.md
01_CONTEXT_AND_SOURCE_MAP.md
02_METHOD_AND_PORT_DECISION_GUIDE.md
evidence/IMPLEMENTATION_LOG.md

检查：
strict UTF-8、BOM/U+FFFD、LF-only、末尾 LF、fenced code、
排除 fenced/inline code 后的 display/inline math、重复标题、相对 Markdown 链接。

首次结果：
QA_PASS all 5 files
```

一致性扫描：

- 当前态四份文档不再出现“等待 macro/primitive”“等待整包确认”“ZIP 尚未下载”、
  “M2 候选”“F2 零初始化预先 probe”或把 realized `L/mask` 作为 Q 输入的旧主张；
- `executed_action_mask` 只在主计划中以“禁止作为 Q 输入”的纠错语境出现；
- 本批只做文档 QA，不把它称为项目测试或 pre-smoke。

## QAM-DOC-0009：online-only、SDE/ODE 与 planned-N 第二次收口

时间：2026-07-29 15:34–16:21 CST。

授权：继续讨论、公开资料只读核查与本地 QAM 文档整理；无服务器写入、代码实现、
数据解压/转换、依赖安装、测试、smoke 或训练授权。

本批目标：

- 随设计收敛删除 clean-50/QAM sidecar 等不再使用的生产分支；
- 解释 π0 ODE 与 QAM memoryless SDE 并修正歧义措辞；
- 核对官方 10-Q 是否共享 encoder，以及 C1 的对齐/偏离边界；
- 将 planned fixed-N、realized L/mask 的因果合同写到可直接实施的粒度；
- 不新增文档，不扩展检查矩阵。

只读一手核对：

```text
QAM paper v4:
Eq. 3/5/7             flow ODE 与同边缘 SDE family
Eq. 19/21/22          AM-training memoryless SDE 与 behavior reverse adjoint
Eq. 23                TD next action 来自 fine ODE
Eq. 24/25             离散 SDE 与 reverse VJP
Eq. 26                10-Q pessimistic TD

locked qam@2726d767:
agents/qam.py::adj_matching
agents/qam.py::sample_actions / compute_flow_actions
agents/qam.py::critic_loss / create
utils/networks.py::Value / ensemblize

公开 VLA 近邻：
LWD arXiv 2605.00416
```

核对结论：

1. fine policy 是 velocity field，不是“SDE 模型”。rollout/evaluation/TD next action
   使用 fine ODE；memoryless SDE 只在 AM 训练中生成辅助轨迹，不注入环境。
2. 论文理想式写 fine SDE；锁定代码前 `T-1` 步使用 fine SDE，最后边界步显式使用
   behavior ODE。P1 oracle 以锁定代码为准。
3. 删除 SDE 改成纯 ODE training path 会失去当前 Plain-QAM Eq. 21 的直接语义/保证，
   必须另称 ODE-path adaptation/ablation。
4. 官方 `Value(encoder=None)` 直接读取 flat observation + action；`ensemblize` 建 10 套
   独立 MLP 参数。官方没有共享可训练视觉 trunk，也没有 encoder 可抄。
5. C1 高层上仍对齐“固定 observation map + 10 个独立 action-conditioned Q”；额外边界是
   10-Q std 不能发现 frozen π0 encoder 的共同 representation error。
6. 生产 v1 改为 online-only：SFT π0/F1 真执行 RoboTwin，产生全部 critic transition；
   `collect -> q_only -> am_on`。clean-50 不进 v1 Q loss、不解压/转换、不建 sidecar。
7. 这比派生 clean-50 label 更符合 transition 因果语义，但不复现官方丰富 OGBench
   offline-to-online 实验协议；准确名称是
   `Plain-QAM π0 online adaptation (frozen behavior + macro transition)`。
8. planned fixed-N action 是决策时变量，可进 Q；realized L/mask 是执行后 outcome，
   只进 return、$\gamma^L$、end/bootstrap 与 provenance。把 L/mask 喂 Q 会泄露未来，
   也无法给 update-time 新候选定义输入。

文档修改：

- **新增文档：0**。
- `00_INDEX_AND_IMPLEMENTATION_PLAN.md`
  - 主线改为 `Plain+B1+F1+C1+M2+N20+online-only`；
  - 增加 AM-training SDE 与 rollout/TD fine ODE 的规范边界；
  - 删除 D1、clean-50 sidecar/转换目录和 QAM dataset loader/prepare script 计划；
  - Q0 改为 online transition payload probe；
  - 清理 P0/P1、文件矩阵、未决参数和下一授权包中的 clean-50 依赖。
- `01_CONTEXT_AND_SOURCE_MAP.md`
  - clean-50 revision/hash 只作为 RLT-owned 可选诊断资产保留；
  - 精简 converter provenance，并明确不再是 QAM 实施步骤；
  - 增加官方 ODE sampler / AM-training SDE 索引与问答路由。
- `02_METHOD_AND_PORT_DECISION_GUIDE.md`
  - 将 SDE/ODE 分成执行路、AM 辅助轨迹和 reverse-adjoint 三条路；
  - 将 B1/B2/F1/F2 从长篇候选压成一张决策表和已选参数所有权；
  - 补官方 `encoder=None`、10 套独立 MLP 与 C1 共同表示错误边界；
  - 增加 query/planned chunk/realized L/mask/semi-MDP 的因果教学；
  - 将 D1/clean-50 长章节压成 online-only 数据表与冷启动门序。
- 根 `HANDOFF.md`
  - QAM 当前节改为 online-only、SDE/ODE、C1 uncertainty 与 planned-N/L 最新停点。

明确删除的计划产物：

```text
rlinf/data/datasets/robotwin_qam.py
examples/embodiment/prepare_robotwin_qam_dataset.py
qam/transitions-v1 clean-50 sidecar
offline/online sampling-ratio path
```

边界：

- 未修改任何 RLinf 实现代码、配置或依赖；
- 未创建/切换 branch 或 worktree；
- 未连接服务器，未把旧动态快照称为 current；
- 历史账本保持 append-only，旧 D1 决策只作为当时记录保留；
- 机械 QA 与一致性扫描在本批末尾执行后补记。

机械 QA：

```text
目标：
HANDOFF.md
00_INDEX_AND_IMPLEMENTATION_PLAN.md
01_CONTEXT_AND_SOURCE_MAP.md
02_METHOD_AND_PORT_DECISION_GUIDE.md
evidence/IMPLEMENTATION_LOG.md

检查：
strict UTF-8、BOM/U+FFFD、LF-only、末尾 LF、fenced code、
排除 fenced/inline code 后的 display math、重复标题、相对 Markdown 链接。

结果：
QA_PASS all 5 files
```

一致性扫描：

```text
rg current docs:
\bD[12]\b
offline+online 已选
prepare_robotwin_qam_dataset
robotwin_qam.py
qam/transitions-v1
fine SDE + terminal

结果：
CONSISTENCY_PASS no D1/D2 or removed-file production remnants
```

本批仍只是本地文档 QA，不是项目 compose/import/test、pre-smoke 或训练证据。

## QAM-DOC-0010：fixed-N macro 现场修正与五项实施问题收口

时间：2026-07-29 18:42–20:12 CST。

授权：继续 QAM 方法讨论、本地专题文档整理、公开一手资料只读核对和服务器只读审查；
无服务器写入、代码实现、branch/worktree 创建、依赖安装、数据解压/转换、项目测试、
smoke 或训练授权。

本批目标：

- 回答 F1 资源、C1 feature capture、M2 payload、两卡 ownership 和数值超参分别是什么；
- 对照官方 fixed-H replay/discount/valid 语义，纠正此前 variable-L macro 假设；
- 解释 10-Q head diversity、state representation separability 和 action-value
  separability 的不同来源；
- 将当前三份 QAM 文档压回一条 fixed-N 主线，不新增文档和并行兜底。

### 1. 服务器只读身份与基线

认证路径：复用 `local_scripts/remote_exec_autodl.py`，由当前进程注入密码，
`Paramiko.Transport.start_client()` 后核对 pinned SHA256 host key，再
`auth_password()`；未使用 key/agent，未把凭据写入命令文件、文档或日志。

本批命令体如下；`<current-process-secret>` 只表示运行时注入，实际值不落账：

```powershell
$env:SEETA_SSH_PASSWORD='<current-process-secret>'
python local_scripts/remote_exec_autodl.py run @'
hostname
pwd
id -u
date --iso-8601=seconds
git -C /root/autodl-tmp/RLinf rev-parse HEAD
git -C /root/autodl-tmp/RLinf branch --show-current
git -C /root/autodl-tmp/RLinf status --short
test -d /root/autodl-tmp/RLinf_qam_pi0_robotwin
'@
Remove-Item Env:SEETA_SSH_PASSWORD
```

结果：

```text
hostname: autodl-container-nekaqbwt43-6ce5babb
pwd: /root
uid: 0
time: 2026-07-29T19:22:03+08:00
RLinf HEAD: 6d0db56bf26f972cd27fa29535f5eb939e80e5bf
branch: local/openpi-a800-2gpu-migration
status: only known untracked A800 configs/local_scripts
QAM worktree: absent
```

### 2. RoboTwin fixed-N payload 审查

只读定位命令：

```text
rg -n "def chunk_step|venv.step|chunk_rewards|chunk_terminations|chunk_truncations" \
  /root/autodl-tmp/RLinf/rlinf/envs/robotwin

rg -n "gen_sparse_reward_data|TOPP|n_steps_to_run|_elapsed_steps" \
  /root/autodl-tmp/RoboTwin_RLinf/envs

rg -n "final_obs|final_observation|_handle_auto_reset|model_weights_id|versions" \
  /root/autodl-tmp/RLinf/rlinf
```

核对结果：

1. `RoboTwinEnv.chunk_step` 只调用一次 `venv.step(chunk_actions)`；上层只拿 chunk-final
   observation，termination/truncation 只标最后 slot。
2. qpos 路径 `gen_sparse_reward_data` 先用全部 planned waypoints 生成一条 TOPP
   trajectory，再执行到 success 或轨迹结束；因此一次 query 的完整 planned chunk
   参与了低层规划。
3. `_cal_chunk_rewards()` 当前把 `n_steps_to_run=0`，`_elapsed_steps` 按配置 chunk width
   增加；success reward 因而对齐到 query final slot。现有 payload 没有可解释为
   planned-action 索引的真实 `L`。
4. `EnvOutput.final_obs`、`_handle_auto_reset`、`infos["final_observation"]` 和
   `model_weights_id/versions` 基础设施已经存在。
5. 原计划的 `gamma^L + executed-prefix mask` 没有现场依据。当前推荐 M2 修正为每
   query 一条 fixed-N macro transition：

```text
(s, planned_N_action, reward_vector_N, s_next, end, policy_version)
R_macro = sum(gamma_slot ** i * reward_vector_N[i], i=0..N-1)
Gamma_N = gamma_slot ** N
y = R_macro + Gamma_N * bootstrap_mask
    * (mean(target_Q) - rho * std(target_Q))
```

`gamma_slot` 是逻辑 planned waypoint/reward-slot clock，不是测得的 simulator primitive
duration。没有 `executed_length` 或 `executed_prefix_mask`。success 不 bootstrap；timeout 只有拿到
true-final feature 才 bootstrap。首版优先 `auto_reset=false`；若配置不支持，再做
QAM-only final-feature payload 扩展。

### 3. 官方 fixed-H 语义重新核对

一手来源：

```text
QAM paper v4:
Eq. 21/22     adjoint matching
Eq. 23–26    fine next action、10-Q TD、target
Table 6      common hyperparameters

locked qam@2726d767:
agents/qam.py::critic_loss / update
utils/datasets.py::sample_sequence
utils/networks.py::Value / ensemblize
```

结论：

- 官方 theory 是 single-step MDP；manipulation 设 `H=5`，正常时尝试 open-loop 执行
  H 步，episode done 会提前清空余项；online replay 逐 primitive 存 transition，
  update 时取重叠 fixed-H window。
- action window flatten 后进入 Q，return 是 H-step discounted return，bootstrap 固定
  使用 `gamma**H`。
- terminal 出现在前 H−1 槽时，`valid[..., -1]=0`，整条 critic/FM loss 被 gate；
  terminal 正落最后槽仍 valid，只以 final mask 关闭 bootstrap。官方不缩短成 `L`
  再用 `gamma**L`。
- 官方 `masks` 是 TD bootstrap indicator，不是 executed-action mask。
- 因此 fixed-N M2 比旧 variable-L 方案更接近官方的 fixed-width action 与固定
  bootstrap-coefficient 形式；transition/reward clock、overlap 和 valid 仍不同，且当前
  success reward 只对齐 query final slot，必须称 macro adaptation。production M2 的完整
  query transition 不使用 official `sequence_valid`。

### 4. F1 静态资源 probe

checkpoint：

```text
/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
```

只读 header 统计范围：

```text
include:
  gemma_expert.model
  action_in_proj
  action_out_proj
  state_proj
  action_time_mlp*

exclude:
  gemma_expert.lm_head
```

输出：

```text
F1 tensors: 173
F1 parameters: 314,713,120
  gemma_expert.model: 311,464,960
  action_time_mlp*: 3,147,776
  action_in_proj: 33,792
  action_out_proj: 32,800
  state_proj: 33,792
checkpoint storage: 635,998,336 bytes (~606.5 MiB)
dtype numel:
  BF16: 311,427,072
  F32: 3,286,048
excluded lm_head: 263,323,648 parameters
```

静态预算：

- FP32 Adam 两个 moments 约 2.344 GiB 总量，两卡理想等分约 1.172 GiB/卡；
- 连同 param/grad/master 的额外静态量约 1.8–2.7 GiB/卡，取决于 FSDP mixed-precision
  所有权；
- 该 allowlist 只是 header-derived candidate，仍须 runtime load/use/parity；必须实测
  frozen-prefix recompute、AM-SDE trajectory、behavior VJP、FSDP sync 和 optimizer
  step 的 peak，静态统计不能代替真实峰值。

### 5. C1、两卡和数值收口

C1 已有只读 probe：

```text
full prefix:  [1,816,2048] BF16
image prefix: [1,768,2048]
mask true: 773
```

当前 C1 推荐设计：

- rollout 在 prefix 已算出的位置 detach feature；
- contextualized token 按三个 camera position block 和 language position block 分组做
  mask-aware mean，缓存 `[4,2048]` BF16；这叫 position-block-preserving pooling，
  不是 attention 后的纯 source feature。四块 valid count 必须大于 0，critic 前 cast
  FP32；
- 复核发现 pooled feature 只够 10-Q，不够原 OpenPI action expert：AM 的 current
  behavior/fine forward 和 TD 的 next fine ODE 都需要完整 frozen prefix KV；
- 因此同一 replay 增加去重 observation store。transition 以 `obs_id/next_obs_id` 引用
  canonical 三相机 uint8、task/prompt ID、proprio 与 transform fingerprint；actor worker
  采样时重算 frozen prefix KV。不存 full prefix token/KV；
- replay 同时存 current/next pooled critic view、normalized proprio、planned
  normalized/env action 和 fingerprint。50k current+next pooled feature 原始量约
  1.64 GB decimal；canonical image shape/bytes 与 recompute 吞吐仍须 P2 实测。

三种可分性不能混为“随机初始化”：

1. head diversity：官方和本计划都由 10 套独立参数/RNG 起步；所有 head 看同一 minibatch
   和同一 TD target，没有 per-head bootstrap resampling。初始化只提供起点，不保证
   训练后仍有 useful disagreement。
2. state representation separability：官方来自 low-dimensional simulator observation；
   本计划来自 frozen three-view/language feature + proprio。P2 只用 simulator
   object pose/task stage 作诊断标签检查明显 alias，privileged label 不进 Q。
3. action-value separability：需要同一/相近 state representation 下有局部 action 变化和
   不同真实后果；只看不同 state/action/outcome 时 Q 仍可能学成 $V(s)$。`q_only` 后用
   同一 seeded/snapshotted state 的 base/`+dQ/da`/`-dQ/da` 真实执行排序检查。

当前两卡推荐从已验证 DSRL 路径
`fsdp_sac_policy_worker.py::DSRLTransitionReplayBuffer(capacity, seed, rank, world_size)`
复用以下所有权：

- 每 rank local replay shard，local batch = global batch / world size；
- 每 rank 为本地 batch 计算完整逻辑 10-head ensemble；FSDP 可物理分片，不拆成 5+5；
- 同 rank 内 head checksum 必须不同；跨 rank 对应 head checksum 必须由 broadcast/
  `sync_module_states` 保持相同；
- FSDP/DDP 同步 critic/fine gradient；target EMA 维持一个逻辑状态；
- replay 按 rank checkpoint，resume world size 必须一致；
- rollout 只同步 active F1 inference weights。

首次 probe-config 候选不做网格：

```text
K=10
flow_steps=10
gamma_slot=0.99
Gamma20=0.8179069
rho=0.5
target_ema=0.005
grad_clip=1.0
critic_lr=3e-4
fine_lr=2e-5
global_batch=64
local_batch=32
replay_capacity=measure canonical observation bytes first;
                target 50000 global only if budget fits
collect_warmup=512 global macro transitions
UTD=1 logical update/new macro
inv_temp: q_only=0, am_on=0.5
```

critic/fine 使用分离 optimizer，但 loss 从同一 pre-update snapshot 计算；target EMA 读取
pre-update online 参数。`global_batch=64` 和 replay 50k 都是 resource probe 起点，不是
已验证数值；正式 smoke 前仍须把最终值放进 resolved config 批准包。

复核还发现两项必须进入 P2、不能靠文档静默决定的合同：

1. $P_N^\top$ 只保证 terminal direct-Q gradient 的 `N:50`/14D 外为 0；frozen behavior
   reverse VJP 因 token/维度耦合可让 intermediate noisy-state adjoint 在这些坐标非零，
   不能逐 flow-time 强制裁零。
2. 官方 TD next action、rollout action 和 `clip_adj=True` terminal-Q input 使用
   `[-1,1]` clamp。π0 端先量 normalized active action 越界率；current/next/terminal-Q
   与实际 env action 必须共用一个 canonical clamp 合同。若有实质越界，由用户在
   “执行也 clamp”与“明确偏离官方不 clamp”间决定。

确认强度同步修正：用户已确认的是 `Plain+action-space+adjust_bottle+B1+F1`；
`C1+fixed-N M2+N20+online-only` 是当前单一推荐，仍待用户整包确认。DSRL 只证明两卡
工程模式，C1/F1/QAM FSDP 仍需上述 probes。

### 6. 本批文档改动与边界

- **新增文档：0**。
- `00_INDEX_AND_IMPLEMENTATION_PLAN.md`
  - 主线从 variable-L 修正为 M2 fixed-N macro；
  - 写入 F1 精确参数/静态 optimizer 预算、C1 dual-view observation contract、
    position-block pooling 与三种可分性；
  - 形成两卡 replay/critic ownership、fixed-N target、分离 optimizer 和首组 probe
    数值的单一推荐；除已确认的 B1+F1 外，仍保留整包确认门；
  - replay schema、调用链、文件矩阵、验收矩阵和剩余事实同步收口。
- `01_CONTEXT_AND_SOURCE_MAP.md`
  - 增加 checkpoint/prefix probe、RoboTwin chunk/final-obs/version 现场代码定位；
  - 增加 DSRL per-rank replay 可复用边界；
  - 删除把 L/mask 裁 action 的当前主张。
- `02_METHOD_AND_PORT_DECISION_GUIDE.md`
  - 教学层明确官方 `gamma**H + valid gate`；
  - 区分 head/state/action 三种 separability，说明随机初始化只解决第一种；
  - 重写 fixed-N、四类 mask、M2 target 与官方对齐/偏离；
  - 用一张表解释五个实施问题、当前决定和仅剩证据。
- 根 `HANDOFF.md`
  - 刷新 19:22 CST server identity/baseline/worktree 状态；
  - 写入确认/推荐边界、fixed-N 修正、F1/C1/两卡/数值停点。

边界：

- 未修改任何 RLinf 实现代码、config、checkpoint、数据或依赖；
- 未创建 branch/worktree，未解压/转换数据；
- 服务器审查全为只读；
- 未运行 compose/import/compile、项目测试、smoke 或训练；
- 本批只在最后运行本地 Markdown 机械 QA，不把它称为项目测试。

### 7. 本地文档机械 QA

执行：

```powershell
# 对 HANDOFF、QAM 00/01/02、账本共 5 个文件检查：
# strict UTF-8、无 BOM/U+FFFD、纯 LF、final LF、代码围栏/显示数学分隔符成对、
# 文件内标题不重复、相对 Markdown 链接目标存在。
# 另检查 QAM 专题文件总数与 tab，并扫描当前态 SSOT 的过时路线措辞。
```

结果：

```text
QA_PASS all=5 qam_files=4 utf8_no_bom_lf_links_balanced_no_tabs
TAB_SCAN no matches
CONSISTENCY_SCAN only expected statements that clean-50 does not enter v1 Q loss
```

第一次在补写本节后复跑时，账本文字把显示数学分隔符本身作为普通示例写了一次，触发
`UNBALANCED .../IMPLEMENTATION_LOG.md`。原因只是 QA 说明文字造成计数为奇数；将其改写成
“显示数学分隔符”后用同一检查复跑，得到上面的最终通过结果。

Git 机械检查：

```powershell
git -c safe.directory='C:/Users/86136/Documents/rl' status --short
git -c safe.directory='C:/Users/86136/Documents/rl' diff --check
```

结果与限制：

- 当前本地 snapshot 在 Git 视角把整个工作区显示为 untracked，`git diff --check` 没有可比较
  的 tracked baseline；未修改全局 `safe.directory`，也未把该空输出当作 diff 通过证据；
- 因而本批可声称的只有上述文件级 Markdown QA 与当前态一致性扫描，不声称仓库级 diff、
  项目测试或实现验证通过。

## QAM-DOC-0011：论文 critic 模态、M2 高层语义与文档职责收束

时间：2026-07-29 20:12–20:46 CST。

授权：继续 QAM 方法讨论、公开一手资料只读核对和本地专题文档整理；无服务器写入、
代码实现、branch/worktree 创建、依赖安装、项目测试、smoke 或训练授权。

### 1. 本批问题

- 论文是否声称视觉 Q、privileged Q，或根本未规定 observation 模态；
- 官方 fixed-H 与计划 fixed-N M2 在高层是否足够对齐，还有哪些真实风险；
- C1 dual-view 和两卡 10-Q ownership 是否仍需用户作方法选择；
- 现有四份 QAM 文档是否准确、精简，有无应迁移或删除的重复内容。

### 2. 一手资料结论

核对：

- [QAM arXiv v4](https://arxiv.org/html/2601.14234v4)；
- 锁定版
  [`agents/qam.py`](https://github.com/ColinQiyangLi/qam/blob/2726d767c9a0a7a46d49693f0391f73dc2cf58ac/agents/qam.py)、
  [`utils/networks.py`](https://github.com/ColinQiyangLi/qam/blob/2726d767c9a0a7a46d49693f0391f73dc2cf58ac/utils/networks.py) 与
  [`experiments/reproduce.py`](https://github.com/ColinQiyangLi/qam/blob/2726d767c9a0a7a46d49693f0391f73dc2cf58ac/experiments/reproduce.py)；
- [OGBench 项目页](https://seohong.me/projects/ogbench/) 与
  [Cube state observation](https://github.com/seohongpark/ogbench/blob/master/ogbench/manipspace/envs/cube_env.py#L731-L769)。

结论：

- 论文方法只定义抽象 MDP state `s` 和 `Q(s,a)`，未规定 pixels、visual encoder、
  privileged state 或 observation encoder；理论形式对模态中立，但论文没有视觉 critic
  实验证据；
- 官方复现实验选用不带 `visual-*` 前缀的 OGBench/MuJoCo state 环境。actor 与 critic
  读取同一 observation，故不是 critic 独享隐藏信息的 asymmetric/privileged critic；
- OGBench manipulation 的 nonvisual state 实际拼入 proprio 与 simulator-derived
  block position/orientation。相对真机视觉部署可称依赖仿真真值，但不能改写成
  “QAM 算法规定 privileged critic”；
- 论文明确 10 critics 与悲观 target；锁定代码确认 10 套独立参数的 MLP 读取同一
  observation/action。独立性来自参数和初始化，不来自十种 observation 或视觉 encoder。

### 3. M2 对齐结论与剩余风险

把 query 边界定义为 macro state，把 planned `N=20` chunk 定义为连续 macro action，
则 fixed-N M2 仍是一个定义明确的 action-conditioned MDP/Q 问题。它保留 Plain QAM 的：

- 当前 action 的真实环境后果与 TD critic；
- 当前 fine policy 生成的 next action；
- 10-Q `mean-rho*std` pessimistic bootstrap；
- endpoint target-Q mean action gradient；
- frozen behavior reverse VJP 与逐 flow-time AM。

它不等于官方实验协议：官方是 primitive replay、重叠 fixed `H=5` window 和
`gamma**H`；计划是 query-level、非重叠 `N=20` macro、logical slot reward clock。
因此成果名保持
`Plain-QAM pi0 online adaptation (frozen behavior + fixed-N macro transition)`，
不称 exact reproduction。

高层方法已足够对齐，首版不因协议差异预先升级 M1。真正剩下三项风险：

1. 官方 manipulation Q 的 action 约为 `5*5=25D` 且有大规模多样 transition；计划 Q
   直接评价 `20*14=280D` chunk，并从稀疏 online outcome 学动作梯度。Q 可能退化为
   近似 `V(s)`，或在数据外给出任意梯度；
2. `gamma_slot**20` 与 final-slot reward 是 query-clock 适配，不是测得的 simulator
   primitive 时间。success/live/timeout、true-final observation 和 bootstrap mask 必须
   在同一时钟下自洽；
3. normalized current action、TD next action、terminal-Q action 与 env 实际执行 action
   必须使用同一 canonical clamp/transform，否则 Q 的梯度针对的是另一种动作。

故 `am_on` 前保留最小高信息门：真实 success/failure/timeout 覆盖、相近 state 下的
action/outcome 变化、finite 且非平凡的 action gradient，以及同一 snapshot 上
base/`+dQ/da`/`-dQ/da` 的真实执行排序。失败时先补实际执行的 frozen-SFT transition
或窄扰动，不给 clean-50 伪造 reward。

### 4. 工程 ownership 与确认边界

- C1 仍是首版主线。用户已把 pooling、dual-view store/recompute 和两卡 ownership
  交由实施侧按 probe 落定；这些不再作为方法选项反复询问；
- 每个 rank 对本地 batch 计算完整逻辑 10-Q、跨 rank 同步对应参数/梯度/target，
  replay 按 rank 保存且 resume world size 一致；不拆成每卡 5 个互不一致的 Q；
- 用户已确认 `Plain+action-space+adjust_bottle+B1+F1`。本批没有把问题回答自动解释成
  formal implementation/smoke 授权；仍待确认的是
  `fixed-N M2+N20+online-only` 这一套训练协议，尤其 online-only 数据预算。

### 5. 文档整理

新增文档：**0**。

修改：

- `00_INDEX_AND_IMPLEMENTATION_PLAN.md`
  - 补正论文模态中立、官方非视觉 state 实验、symmetric actor/critic 与 simulator truth
    的边界；
  - C1 只保留规范性公式、selected design 和 probe，三种可分性教学统一指向 `02`；
  - 把确认强度改为 C1/两卡工程细节已委托，M2/N20/online-only 待协议确认。
- `01_CONTEXT_AND_SOURCE_MAP.md`
  - 增加 OGBench 项目与 cube state 一手代码来源；
  - 补正视觉/privileged 表述；
  - 将过时的“五个用户实施问题”索引改成实施合同路由。
- `02_METHOD_AND_PORT_DECISION_GUIDE.md`
  - 解释论文未规定模态、官方 state 实验和“仿真真值但非 asymmetric critic”的区别；
  - 明确 query-boundary macro-MDP 足以承载 QAM 高层语义，并突出 280D online data
    geometry 风险；
  - 删除与主计划重复的五项实施问题表，只保留三项方法风险与规范文档指针。
- 根 `HANDOFF.md`
  - 同步论文/实验边界与最新确认强度。
- 本账本
  - 追加本批来源、判断、修改和 QA；历史批次不回写，以保留决策演化证据。

职责复核后仍保留四份专题文档：

- `00`：唯一当前规范；
- `01`：来源、代码定位和动态资产；
- `02`：术语与方法教学；
- 本账本：append-only 操作/证据历史。

默认上下文只需 `00 + 账本最新批次`；发生来源争议才读 `01`，需要教学解释才读 `02`。
未发现值得再拆成第五份文档的内容；账本较长是预期的历史属性，不迁回当前 SSOT。

### 6. 边界

- 本批未连接服务器，也未刷新任何动态服务器事实；
- 未修改 RLinf 实现、config、checkpoint、数据或环境；
- 未运行 compose/import/compile、项目测试、smoke 或训练；
- 下一节只记录本地 Markdown 机械 QA，不把它称为实现验证。

### 7. 本地文档机械 QA

执行一：对 `HANDOFF.md`、QAM `00/01/02` 和本账本运行 inline PowerShell 文件检查：

```text
strict UTF-8 decode
no BOM / U+FFFD / CR / tab
final LF
balanced fenced-code and display-math delimiters
no duplicate heading in current-state docs
all relative Markdown link targets exist
recursive QAM Markdown file count == 4
```

结果：

```text
QA_PASS all=5 qam_files=4 strict_utf8_no_bom_lf_final_lf_fences_math_headings_links_no_tabs
```

执行二：对 `HANDOFF.md` 与 QAM `00/01/02` 扫描旧确认措辞和旧“五个用户问题”措辞；
再将 `00/01/02` 按空行切段、空白归一化，比较长度至少 120 字符且不属于表格/代码块的
跨文档完全相同段落。

结果：

```text
STATUS_CONSISTENCY_PASS no stale confirmation/five-question wording
PARAGRAPH_DEDUP_PASS exact_cross_doc_groups=0 min_chars=120
DOC_SIZE lines=245 bytes=19271 HANDOFF.md
DOC_SIZE lines=955 bytes=48441 00_INDEX_AND_IMPLEMENTATION_PLAN.md
DOC_SIZE lines=534 bytes=25699 01_CONTEXT_AND_SOURCE_MAP.md
DOC_SIZE lines=1171 bytes=56206 02_METHOD_AND_PORT_DECISION_GUIDE.md
DOC_SIZE lines=1760 bytes=71914 evidence/IMPLEMENTATION_LOG.md
```

结论：当前态文档没有检测到完全重复长段；剩余少量主题重叠是有意的“00 写规范、
01 写来源、02 作教学”。账本长度来自 append-only 历史，不应为了表面精简删除旧证据。

补写本节后以同一文件集合复跑核心机械检查：`QA_PASS_FINAL all=5 qam_files=4`。

## QAM-DOC-0012：实施就绪度与端到端调用链审查

时间：2026-07-30 09:44–10:14 CST。

授权：继续 QAM 设计讨论、公开/本地锁定源码只读审查和本地专题文档整理；无服务器
连接或写入、branch/worktree 创建、代码实现、依赖安装、项目测试、smoke 或训练授权。

### 1. 审查范围

- 配置入口、policy/F1、rollout、env payload、replay、critic/AM、两卡同步、resume 和
  集中验收的完整调用链；
- 官方 Plain QAM 与 B1+F1+C1+fixed-N M2+online-only 的语义边界；
- 仍需用户拍板的方法事项与只能在实现中取得的事实证据；
- QAM `00/01/02/账本` 的职责、材料覆盖和冗余。

本地只读证据：

```text
QAM source:
  .tmp/qam-official-2726d767
  HEAD 2726d767c9a0a7a46d49693f0391f73dc2cf58ac

RLinf exact common baseline:
  git show 6d0db56bf26f972cd27fa29535f5eb939e80e5bf:<path>
```

精确检查了：

- `examples/embodiment/train_embodied_agent.py` 的 worker route 和
  `use_training_pipeline` gate；
- `rlinf/models/embodiment/openpi/openpi_action_model.py` 的
  `predict_action_batch()`、`model_action/action`、prefix/velocity API；
- `rlinf/data/embodied_io_struct.py` 的 `EnvOutput`、`Trajectory`、
  `append_transitions()`；
- `rlinf/workers/env/env_worker.py` 的 `prepare_actions()`、true-final observation、
  mixed done/live 与 transition append；
- 官方 `agents/qam.py`、`utils/networks.py`、`utils/datasets.py` 的 update、AM、
  10-Q 和 fixed-H 语义。

### 2. 就绪度结论

算法主干、适配边界、模块所有权和预计文件已经收束，设计约 **90%** 完成。开始实现前
只剩一个用户级方法确认：

```text
fixed-N M2 + N=20 + online-only
clean-50 不进入 v1 Q loss
成果名为 Plain-QAM pi0 online adaptation
         (frozen behavior + fixed-N macro transition)
```

确认后仍须另行授权“开始实现 P1”。F1 显存、C1 capture、payload、clamp、
两卡 FSDP/resume 和最终 batch/replay 数值都是实施 probe，不再升级成并行用户选项。

### 3. 调用链发现与修正

#### 3.1 P2/P3 顺序

旧 P2 写了 `q_only` 后真实 `±dQ/da` 排序，但 critic/replay/trainer 到 P3 才实现，
顺序不可执行。修正为：

- P2：time/sign、投影、prefix/C1、真实 π0 VJP、F1 资源；
- P3：critic/replay/trainer、synthetic target 和单 update；
- P4 的单独获批 q-only 诊断：训练出 Q 后再做真实 `±dQ/da` 排序，之后才允许
  `am_on`。

#### 3.2 UTD 的全局计数

`UTD=1` 不能等同于每个 runner outer cycle 一次 update。锁定：

```text
global_new_macros
  = all-reduce(sum(per-rank schema-valid replay inserts))
update_credit
  += global_new_macros * UTD
consume 1 credit
  = all ranks complete one synchronized global-batch critic logical update
```

进入 q_only 时建立 anchor；collect warm-up 不默认追补为突发更新。global/local inserts、
credit、anchor、critic/fine update counters 均进入 checkpoint。

#### 3.3 phase 与 policy version

- `collect`：只入 replay；
- `q_only`：只更新 critic/target，完全跳过 AM/fine optimizer，F1 与 behavior bitwise
  相同；
- `am_on`：critic/fine 从同一 pre-update snapshot 计算后分别 step；
- `collect→q_only` 可按 global valid replay 阈值单调切换；
- `q_only→am_on` 只由获批 config/resume 显式切换，不按 runner step 静默打开；
- policy version 表示 fine 权重版本，只在 F1 实际更新并同步时增加，不能在 q_only
  期间随 runner step 虚增。

#### 3.4 AM 张量域

生产 AM 锁定在完整 `[50,32]` flow state 上按官方 reduction 计算。terminal Q gradient
先由 `P_N^T` 从 `[N,14]` 嵌回；direct inactive gradient 为零，reverse VJP 后不再裁剪。
只在 `[20,14]` 上做 AM 会切断 π0 flow 内部耦合，不属于当前 v1。

#### 3.5 通用 policy API

trainer-only critic 不进入通用 `ForwardType`。从默认修改矩阵删除
`base_policy.py`；只有 FSDP root-dispatch probe 证明直接 velocity/prefix API 不可用时，
才增加一个最窄 `QAM_FLOW` route，绝不加入 QAM-Q route。

### 4. payload 事实门

exact baseline 已能传 current/next observation、processed current conditioning、
model/env action、reward、termination/truncation 和 rollout version，但仍须 Q0/P2 证明：

1. `Trajectory.append_transitions()` 会删除 `task_descriptions`，trajectory 又没有
   terminal-kind/env-info 字段；固定任务 prompt/tokenized prompt 是否足够，以及
   time-limit/other truncation 能否无损区分；
2. `env_output.dones.any()` 是 batch-wide 分支；`_build_chunk_final_obs()` 的 per-env
   fallback 是否在 mixed done/live batch 正确；
3. rollout env action 后还经过 `prepare_actions()`；它是否只 reshape/identity，否则
   replay 必须携带实际 `exec_actions` provenance。

只有对应 probe 失败才物化：

- `rlinf/data/embodied_io_struct.py` 的 QAM-only prompt/terminal/final-view metadata；
- `rlinf/workers/env/env_worker.py` 的逐 env final/terminal/exec-action payload；
- `rlinf/workers/rollout/hf/huggingface_worker.py` 的 final feature capture。

`embodied_runner.py` 仍默认不改。

### 5. 文档与材料审查

新增文档：**0**。

当前四份专题文档继续保留：

- `00`：唯一规范、调用链、schema、文件矩阵和验收门；
- `01`：官方 commit/path/symbol、RLinf/RoboTwin 证据和资产 provenance；
- `02`：用户按需读取的术语/方法教学；
- 本账本：append-only 命令、证据和决策演化。

本批只修改：

- `00_INDEX_AND_IMPLEMENTATION_PLAN.md`
  - 修正 P2/P3/P4 顺序；
  - 增加 phase、fine-policy-version、UTD credit、完整 AM tensor/reduction 合同；
  - 收窄调用链和修改矩阵，补齐 conditional payload 文件；
  - 扩充 Q0/Q6 的 mixed-batch、terminal-kind、exec-action、phase/UTD/version 验收。
- 本账本：追加本批审查与 QA。

`01/02` 无需修改；它们不在默认上下文。默认读取集仍是
`PROJECT_CONTEXT + HANDOFF + 00 + 账本最新批次`。无需新增第五份文档，也不删除历史账本。

### 6. 边界

- 本批没有连接服务器，未刷新 GPU/process/HEAD/worktree 动态事实；
- 未修改任何 RLinf 实现、config、checkpoint、数据或环境；
- 未运行项目 compose/import/compile/test、RoboTwin 执行、smoke 或训练；
- 下一节只记录本地 Markdown/一致性 QA。

### 7. 本地文档与一致性 QA

对 `HANDOFF.md`、QAM `00/01/02` 和本账本检查：

```text
strict UTF-8
no BOM / U+FFFD / CR / tab
final LF
balanced fenced code and display math
current-state docs have no duplicate heading
all relative Markdown link targets exist
recursive QAM Markdown file count == 4
```

结果：

```text
QA_PASS all=5 qam_files=4 strict_utf8_no_bom_lf_final_lf_fences_math_headings_links_no_tabs
```

另对 `00/01/02` 按空行切段并归一化空白，比较至少 120 字符、非表格/代码块的跨文档
完全相同段落；并定向扫描 P2/P3/P4、q-only、UTD、full-state AM、conditional
`base_policy.py`、fixed-N/online-only 的当前措辞。

```text
PARAGRAPH_DEDUP_PASS exact_cross_doc_groups=0 min_chars=120
CONSISTENCY_PASS P2_P3_P4_phase_UTD_full_AM_policy_API_protocol_boundary
DOC_SIZE lines=321  bytes=24672 HANDOFF.md
DOC_SIZE lines=1019 bytes=52632 00_INDEX_AND_IMPLEMENTATION_PLAN.md
DOC_SIZE lines=534  bytes=25699 01_CONTEXT_AND_SOURCE_MAP.md
DOC_SIZE lines=1171 bytes=56206 02_METHOD_AND_PORT_DECISION_GUIDE.md
```

账本是 append-only 历史，补写本节前为 1,957 行、79,957 bytes；它不进入默认完整
上下文，只读最新批次。补写后以同一文件集合复跑核心机械检查，结果仍为
`QA_PASS_FINAL all=5 qam_files=4`。

第一次复跑命令把 PowerShell 的 `Test-Path -LiteralPath` 误连成不存在的
`Test-Path-LiteralPath`，在链接检查首项即退出，未产生文件结论；只修正该命令拼写后
原样复跑通过。记录该问题后，一次试图缩短复跑命令的写法又在 PowerShell 解析阶段因
逻辑表达式括号不足退出，同样没有读取或修改文件；改回复用已通过的完整命令后再次通过。
最终补写前账本为 2,002 行、81,783 bytes。

Git 只读检查仍显示当前本地 snapshot 在 Git 视角为整体 untracked，因此本批不声称
仓库级 tracked diff 通过；可声称的是上述文件级 QA、当前态一致性审查和精确列出的
文档修改。

## QAM-IMPL-0001：实施授权、连接环境恢复与服务器开工前刷新

时间：2026-07-31 10:50–11:05 CST。

授权：

- 用户确认开始实现当前 `Plain+B1+F1+C1+fixed-N M2+N20+online-only` 主线；
- 允许创建独立 QAM branch/worktree、修改代码/config/docs，并自行运行正式 smoke 前测试；
- 正式 smoke/training 仍须提交完整配置、命令、资源、输出和停止条件后等待批准；
- 不影响 DSRL/RLT，不删除磁盘资产；本轮磁盘只读盘点后仅给候选。

### 1. 本地 Paramiko helper 恢复

第一次调用：

```text
python local_scripts/remote_exec_autodl.py run <readonly-command>
```

结果：本地 Microsoft Store `python.exe` 在 helper 启动前报
`系统无法访问此文件`；未建立 SSH，不能解释成认证或服务器失败。

只读定位：

```powershell
Get-Command python -All
Get-Command py
codex_app.load_workspace_dependencies
```

结果：

- `python` 只指向不可用的 WindowsApps stub；
- `py.exe` 没有已注册解释器；
- Codex bundled Python 位于
  `C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`，
  但不含 Paramiko。

第一次安装目标：

```powershell
<bundled-python> -m pip install --disable-pip-version-check --no-input `
  --target C:\tmp\codex-paramiko-3.5.1 paramiko==3.5.1
```

结果：包下载和 staging 成功，最终创建 `C:\tmp\codex-paramiko-3.5.1` 时
`PermissionError: [WinError 5]`；没有形成可用目标目录。

窄修复与复测：

```powershell
<bundled-python> -m pip install --disable-pip-version-check --no-input `
  --target C:\Users\86136\Documents\rl\.tmp\local-paramiko-3.5.1 paramiko==3.5.1
$env:PYTHONPATH='<workspace>\.tmp\local-paramiko-3.5.1'
<bundled-python> -c "import paramiko; print(paramiko.__version__)"
```

结果：`paramiko==3.5.1` import 成功。它是本地一次性 SSH helper 依赖，不属于 RLinf 或
服务器 π0 环境；密码仍只在调用进程的 `SEETA_SSH_PASSWORD` 中存在并在 `finally` 清除。

### 2. 服务器身份、Git、进程与资源刷新

通过固定 host-key 的低层 `Paramiko.Transport.start_client/auth_password` helper 执行：

```bash
date -Is
hostname
pwd
id -u
git -C /root/autodl-tmp/RLinf worktree list --porcelain
for d in /root/autodl-tmp/RLinf \
         /root/autodl-tmp/RLinf_fastwam_rlinf \
         /root/autodl-tmp/RLinf_rlt_pi0_robotwin \
         /root/autodl-tmp/RLinf_qam_pi0_robotwin; do
  # branch / HEAD / status or MISSING
done
ps -eo pid,ppid,etime,%cpu,%mem,rss,args --sort=-rss |
  grep -E 'train_embodied_agent|ray::|raylet|gcs_server|qam'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
free -h
df -h /root/autodl-tmp
```

结果（2026-07-31 11:00:28+08:00）：

```text
identity = autodl-container-nekaqbwt43-6ce5babb /root uid=0
base     = local/openpi-a800-2gpu-migration@6d0db56b...
DSRL     = codex/dsrl-pi0-robotwin@48a775db... clean
RLT      = codex/rlt-pi0-robotwin@2b8199d8... clean
QAM branch/worktree = absent
base dirty = only the five known untracked A800 configs/local_scripts
train/Ray/QAM processes = none
GPU0/GPU1 = 0/81920 MiB, utilization 0
cgroup current/max = 180272386048 / 257698037760 bytes
host available = 979 GiB
cgroup oom/oom_kill = 0/0
disk = 1.9T total, 1.1T used, 808G available, 57%
```

cgroup current 约 168 GiB 是训练退出后仍在的可回收 file cache；不存在 live RSS/GPU
竞争，可以开工。

第一次 shared-venv 模块探针因远端 heredoc 内嵌 f-string 引号被 native 参数层剥离，
在最后一步报 `NameError: present is not defined`；前面的身份/Git/资源命令均已完成。
只把输出改成 `print(module, bool(find_spec(module)))` 后复测：

```text
Python 3.11.14
torch 2.6.0
numpy 1.26.4
jax/jaxlib 0.5.3
flax 0.10.2
optax 0.2.8
pytest 9.0.3
hydra-core 1.4.0.dev1
omegaconf 2.4.0.dev11
```

决定：

- 生产 QAM 复用 shared `/root/autodl-tmp/RLinf/.venv`，不复制、不安装、不修改；
- 是否建立独立 oracle venv 只由锁定官方 source 的实际额外依赖决定，绝不向 shared
  venv 安装缺失包；
- 下一步先记录磁盘只读盘点，再创建独立 QAM worktree。

## QAM-DISK-0001：开工前磁盘只读盘点

时间：2026-07-31 11:05–11:22 CST。

授权边界：只读统计；没有删除、移动、压缩或覆盖任何服务器资产。

### 1. 执行命令

第一层按字节统计：

```bash
df -B1 /root/autodl-tmp
df -i /root/autodl-tmp
find /root/autodl-tmp -mindepth 1 -maxdepth 1 -xdev -printf '%p\0' |
  sort -z |
  while IFS= read -r -d '' p; do
    du -sx --block-size=1 "$p"
  done
```

随后只对第一层大项定向展开：

```bash
du -x --block-size=1 --max-depth=2 /root/autodl-tmp/RLinf_fastwam_rlinf |
  sort -nr | head -80
du -x --block-size=1 --max-depth=2 \
  /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40 |
  sort -nr | head -80
du -x --block-size=1 --max-depth=2 /root/autodl-tmp/experiments |
  sort -nr | head -80
du -x --block-size=1 --max-depth=2 /root/autodl-tmp/models |
  sort -nr | head -80
du -x --block-size=1 --max-depth=2 /root/autodl-tmp/RLinf |
  sort -nr | head -80
du -x --block-size=1 --max-depth=2 /root/autodl-tmp/RoboTwin |
  sort -nr | head -80
du -x --block-size=1 --max-depth=2 /root/autodl-tmp/conda |
  sort -nr | head -80
du -x --block-size=1 --max-depth=2 /root/autodl-tmp/backups |
  sort -nr | head -80
```

为给日志目录补充时间证据，第一次本地调用把
`$(stat -c '%y' "$p")` 内嵌进 PowerShell native 参数，命令在 SSH 前被本地参数解析
破坏；没有执行远端命令、也没有产生服务器改动。窄修复为去掉 command substitution，
改成两条普通远端命令后复测：

```bash
du -sx --block-size=1 "$p"
stat --format='%y %n' "$p"
```

### 2. 第一层结果

`/root/autodl-tmp` 文件系统为 1.9 TiB 总量、约 1.1 TiB 已用、808 GiB 可用，
使用率 57%；inode 使用率 2%。第一层 `du` 可归属总量为
`1,112,879,120,384` bytes（约 1.013 TiB）：

| 路径 | bytes | 约 GiB | 主要归属 |
|---|---:|---:|---|
| `RLinf_fastwam_rlinf` | 572,891,234,304 | 533.6 | 历史 FastWAM、DSRL 日志/checkpoint |
| `RLinf_wamppo_backup_20260714_step57_lastdcp40` | 119,126,163,456 | 110.9 | 旧 WAM-PPO 备份及四份 venv |
| `models` | 97,418,907,648 | 90.7 | Motus/FastWAM/π0 SFT 模型 |
| `experiments` | 63,884,136,448 | 59.5 | RLT stage1/stage2 运行产物 |
| `RoboTwin` | 53,865,926,656 | 50.2 | 仿真器、assets、旧 policy |
| `conda` | 51,531,620,352 | 48.0 | RoboTwin/FastWAM 等 conda env |
| `RLinf` | 35,522,256,896 | 33.1 | 共享 venv 与旧 PPO/GRPO baseline |
| `cache` | 31,358,480,384 | 29.2 | 通用缓存 |
| `LaWAM` | 14,758,912,000 | 13.7 | LaWAM 项目资产 |
| `backups` | 14,657,212,416 | 13.65 | shared venv golden backup |

其余第一层目录单项均小于 3.5 GB。

### 3. 主要实验归属

`RLinf_fastwam_rlinf` 的 533.6 GiB 几乎全部是 `logs/`：

| 日志目录 | 约 GiB | 归属/判断 |
|---|---:|---|
| `20260718_100910-robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu` | 188.4 | 7 月 18 日旧 GRPO 正式运行 |
| `20260728_dsrl_pi0_robotwin_n20_formal_v1` | 93.9 | DSRL 正式运行；含可恢复 DCP195，当前保留 |
| `20260719_124315-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu` | 80.8 | 7 月 19 日旧 PPO 正式运行 |
| `20260728_dsrl_pi0_robotwin_n20_smoke_v1` | 62.6 | 旧 DSRL smoke，较强清理候选 |
| `20260718_020324-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu` | 53.9 | 7 月 18 日旧 GRPO 正式运行 |
| 两个 7 月 18/19 日 GRPO/PPO smoke | 各约 26.9 | 旧 smoke，较强清理候选 |

`RLinf_wamppo_backup_20260714_step57_lastdcp40` 的 110.9 GiB 中，四份
`.venv*` 各约 13.6–13.9 GiB，总计约 54.9 GiB；其余是约 56 GiB 的旧
log/baseline/smoke。它整体是很早的备份，属于强清理候选，但删除前仍需用户确认
是否保留 step57/last-DCP40 恢复点。

`experiments` 的 59.5 GiB 主要为：

- RLT stage1 formal：约 20.6 GiB；
- RLT stage1 smoke s1a：约 20.6 GiB，大小近似重复，是候选；
- RLT resume250→480：约 11.6 GiB；
- RLT formal250：约 5.1 GiB；
- RLT formal100 pilot：约 1.5 GiB。

`models` 中约 60.1 GiB 属于 Motus、23.1 GiB 属于 FastWAM、约 7.5 GiB 是
本次 QAM 必需的 `adjust_bottle` π0 SFT checkpoint；后者不可列入清理候选。
`RLinf/.venv` 约 13.7 GiB 是 QAM 将复用的生产环境，也不可删除。

结论：空间不是当前开工阻塞；不需为了 QAM 复制约 14 GiB 的 shared venv。
若用户之后授权清理，优先按“旧 smoke → 旧重复 venv 备份 → 明确不再需要的旧正式
日志”逐项核对，而不是按目录总量批量删除。本批没有实施清理。

## QAM-IMPL-0002：创建隔离 QAM branch/worktree

时间：2026-07-31 11:24–11:32 CST。

目标：

```text
base commit = 6d0db56bf26f972cd27fa29535f5eb939e80e5bf
branch      = codex/qam-pi0-robotwin
server      = /root/autodl-tmp/RLinf_qam_pi0_robotwin
local copy  = C:\Users\86136\Documents\rl\.qam-impl-worktree
```

### 1. 服务器 worktree

第一次本地调用把多行 shell here-string 直接作为 helper 的位置参数：

```powershell
<bundled-python> local_scripts/remote_exec_autodl.py run $remote
```

PowerShell native 参数传递将其拆成多个参数，`argparse` 报
`unrecognized arguments`；连接函数尚未执行，因此没有到达服务器、没有创建分支或
目录。窄修复是把完全相同的远端脚本放入本地临时 command file，再使用 helper
已有的 `--command-file`：

```powershell
<bundled-python> local_scripts/remote_exec_autodl.py run `
  --command-file .tmp/qam_create_worktree.sh
```

远端完整脚本：

```bash
set -euo pipefail
base=/root/autodl-tmp/RLinf
target=/root/autodl-tmp/RLinf_qam_pi0_robotwin
commit=6d0db56bf26f972cd27fa29535f5eb939e80e5bf
branch=codex/qam-pi0-robotwin

printf 'PRECHECK\n'
test ! -e "$target"
test "$(git -C "$base" rev-parse "$commit^{commit}")" = "$commit"
test -z "$(git -C "$base" show-ref --verify --hash "refs/heads/$branch" || true)"
git -C "$base" status --short
git -C "$base" worktree add -b "$branch" "$target" "$commit"
printf 'POSTCHECK\n'
git -C "$target" branch --show-current
git -C "$target" rev-parse HEAD
git -C "$target" status --short
git -C "$base" worktree list --porcelain
```

结果：

```text
branch = codex/qam-pi0-robotwin
HEAD   = 6d0db56bf26f972cd27fa29535f5eb939e80e5bf
status = clean
worktree list 同时保留 base、DSRL、RLT，并新增 QAM
```

precheck 中用于确认“不存在”的 `show-ref --verify` 将预期的
`fatal: ... not a valid ref` 写到 stderr；`|| true` 后脚本继续且最终退出码为 0。
这是负向存在性探针的噪声，不是创建后的失败；创建后的 `worktree list` 已显示有效
QAM branch。后续存在性探针改用安静模式避免该噪声。

### 2. Windows 代码副本

只读确认 `.research-rlinf` 已含 exact common commit，且目标目录和本地同名 branch
均不存在后执行：

```powershell
git -C .research-rlinf worktree add `
  -b codex/qam-pi0-robotwin `
  ..\.qam-impl-worktree `
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
git -C .qam-impl-worktree branch --show-current
git -C .qam-impl-worktree rev-parse HEAD
git -C .qam-impl-worktree status --short
```

结果同样为 `codex/qam-pi0-robotwin@6d0db56...`、clean。后续 Windows 只编辑
`.qam-impl-worktree` 中的代码副本，compose/import/test 只在服务器 QAM worktree
执行；DSRL/RLT worktree 均不作为 patch 目标。

## QAM-IMPL-0003：Plain-QAM 数学核心与服务器 PyTorch 前置测试

时间：2026-07-31 11:32–11:48 CST。

### 1. 新增代码

仅在 QAM worktree 新增：

| 文件 | 新增内容 |
|---|---|
| `rlinf/algorithms/qam/__init__.py` | QAM 核心公开接口 |
| `rlinf/algorithms/qam/core.py` | FM、flow ODE/SDE path、terminal mean-Q adjoint、behavior VJP 反传、AM、10-Q pessimistic TD、critic MSE、pre-update EMA |
| `tests/algorithms/qam/test_core.py` | 6 个聚焦测试，覆盖梯度边界、时间点、最后一步 ODE、10-Q 公式、FM/AM 和 EMA 时序 |

实现锁定为官方 plain QAM 的关键语义：

- critic bootstrap 用 `mean - rho * population_std`；
- terminal adjoint 只对 target-Q ensemble mean 求动作梯度，不对 std 求梯度；
- 前 `K-1` 步 fine SDE，最后一步 behavior ODE，最后一份噪声不消费；
- forward fine/base 在 `t`，reverse behavior VJP 在 `t+h`；
- path、target adjoint 和 critic 参数不接收 AM 梯度，fine flow 在每个时刻做局部反传；
- target slow/critic EMA 从 optimizer 更新前的 online 参数快照计算。

代码通过 SFTP 只写入服务器
`/root/autodl-tmp/RLinf_qam_pi0_robotwin` 的同名路径；没有触碰 base、DSRL 或
RLT worktree。随后执行：

```bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv

cd "$repo"
test "$(git rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=
"$venv/bin/python" -m pytest -q tests/algorithms/qam/test_core.py
```

结果：

```text
......                                                                   [100%]
6 passed in 1.93s
```

这是 CPU 小张量前置测试，不是 smoke，也没有启动 Ray、RoboTwin 或 GPU 训练。

## QAM-IMPL-0004：官方 JAX 数值 oracle

时间：2026-07-31 11:48–12:29 CST。

目标是把锁定官方仓库
`ColinQiyangLi/qam@2726d767c9a0a7a46d49693f0391f73dc2cf58ac`
的小网络 B2+F1 数值一次性导出为 `.npz`，生产测试以后只用 PyTorch 读取 fixture，
不把 JAX 安装进共享 π0 环境。

### 1. 新增文件

| 文件 | 用途 |
|---|---|
| `tests/algorithms/qam/oracle/export_official_fixture.py` | 从锁定官方 source 构造 FM、AM、critic、Adam 与 pre-update EMA 数值 |
| `tests/algorithms/qam/oracle/requirements.lock.txt` | 8 个直接依赖的精确版本 |
| `tests/algorithms/qam/oracle/README.md` | oracle 来源、生成与消费边界 |
| `tests/algorithms/qam/oracle/qam_official_2726d767_v1.npz` | 553 个官方 JAX 数组 |
| `tests/algorithms/qam/oracle/resolved-freeze.txt` | 独立 oracle venv 的 39 包完整 freeze |

### 2. 官方 source 获取：失败、定位与修复

本地镜像是 shallow clone；先执行只读核对及 bundle：

```powershell
git -C .tmp/qam-official-2726d767 rev-parse --is-shallow-repository
git -C .tmp/qam-official-2726d767 rev-parse HEAD
git -C .tmp/qam-official-2726d767 bundle create `
  .tmp/qam-official-2726d767.bundle --all
git -C .tmp/qam-official-2726d767 bundle verify `
  .tmp/qam-official-2726d767.bundle
```

结果：HEAD 正确、`is-shallow-repository=true`；本地 bundle 表面验证通过，但传到服务器
后从 bundle clone 失败：

```text
Could not read f93efd...
Failed to traverse parents of commit ...
remote did not send all necessary objects
```

失败发生在 venv 创建前；Git 自动清除了未完成的 clone 目标。失败 bundle 仍保留为
`/root/autodl-tmp/oracles/qam-official-2726d767.bundle`
（11,443,043 bytes），本轮没有删除授权。

按既定大陆网络短流程执行一次有界诊断和恢复：

```bash
set -euo pipefail
source_dir=/root/autodl-tmp/oracles/qam-2726d767
url=https://github.com/ColinQiyangLi/qam.git
commit=2726d767c9a0a7a46d49693f0391f73dc2cf58ac

env | grep -iE '^(http|https|all)_proxy=' || true
git config --get http.version || printf 'DEFAULT\n'
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'main code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com || true
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'api code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://api.github.com || true
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'raw code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://raw.githubusercontent.com || true
GIT_TERMINAL_PROMPT=0 timeout 15 git ls-remote --heads "$url"

git init "$source_dir"
git -C "$source_dir" remote add origin "$url"
GIT_TERMINAL_PROMPT=0 timeout 120 \
  git -C "$source_dir" fetch --depth=1 origin "$commit"
git -C "$source_dir" checkout --detach FETCH_HEAD
test "$(git -C "$source_dir" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$source_dir" status --short)"
```

结果：无 proxy，Git HTTP version 为 `DEFAULT`；main/API/raw 均 HTTP 200，
`ls-remote` 的 main 正是锁定 commit；单次 depth-1 fetch 成功，source detached
HEAD 正确且 clean。

### 3. 独立环境：失败、修复与验证

第一次使用：

```bash
python3.11 -m venv /root/autodl-tmp/venvs/qam-oracle-2726d767
```

服务器没有 bare `python3.11` 命令，退出 127；没有创建 venv。窄修复为只借用 shared
venv 的解释器创建一个 `include-system-site-packages = false` 的独立环境：

```bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
source_dir=/root/autodl-tmp/oracles/qam-2726d767
venv_dir=/root/autodl-tmp/venvs/qam-oracle-2726d767

/root/autodl-tmp/RLinf/.venv/bin/python -m venv "$venv_dir"
"$venv_dir/bin/python" -m pip install \
  --disable-pip-version-check --no-input --no-cache-dir \
  -r "$repo/tests/algorithms/qam/oracle/requirements.lock.txt"
```

直接依赖安装成功。第一次版本打印访问了不存在的
`ml_collections.__version__`，仅验证脚本在最后报错，环境已经完整；改用标准 metadata
后复核：

```bash
"$venv_dir/bin/python" - <<'PY'
from importlib.metadata import version
for name in (
    "numpy", "jax", "jaxlib", "flax", "optax", "distrax",
    "tensorflow-probability", "ml-collections",
):
    print(name, version(name))
PY
```

结果精确为 NumPy 1.26.4、JAX/JAXLIB 0.4.26、Flax 0.8.4、Optax 0.2.2、
Distrax 0.1.5、TensorFlow Probability 0.24.0、ml-collections 0.1.1。
oracle source 为 28,323,840 bytes，独立 venv 为 765,292,544 bytes；shared
`/root/autodl-tmp/RLinf/.venv` 没有执行任何 install。

### 4. 生成与冻结

```bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
source_dir=/root/autodl-tmp/oracles/qam-2726d767
venv_dir=/root/autodl-tmp/venvs/qam-oracle-2726d767
output="$repo/tests/algorithms/qam/oracle/qam_official_2726d767_v1.npz"

JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=2 \
  "$venv_dir/bin/python" \
  "$repo/tests/algorithms/qam/oracle/export_official_fixture.py" \
  --source "$source_dir" --output "$output"
"$venv_dir/bin/python" -m pip freeze --all |
  LC_ALL=C sort > \
  "$repo/tests/algorithms/qam/oracle/resolved-freeze.txt"
sha256sum "$output" \
  "$repo/tests/algorithms/qam/oracle/resolved-freeze.txt"
stat --format='%s %y %n' "$output"
wc -l "$repo/tests/algorithms/qam/oracle/resolved-freeze.txt"
```

结果：

```text
fixture arrays = 553
fixture size   = 213610 bytes
fixture sha256 = 42343edd3dea673447e3eeedd3c74c5401752ec9d87ec8a26442dc0c718c55ab
freeze lines   = 39
freeze sha256  = 6c97deb560680f018abd9913c0cbceabf6f5a62c3d295982d16b9501044d0a6c
```

fixture 和 freeze 随后通过 SFTP 下载到 Windows QAM 代码副本，字节数与 SHA-256
复核一致。该批没有启动 GPU/Ray/仿真。

本批还修正了一处本地命令包装问题：PowerShell `finally` 中的环境变量清理会覆盖
`$LASTEXITCODE`。早期两个远端非零退出因此在外层工具显示为 0，但远端 stderr 和
实际失败均已保存且没有被误当成成功继续。后续统一先保存：

```powershell
$code = 0
try {
  & <bundled-python> local_scripts/remote_exec_autodl.py run `
    --command-file <script>
  $code = $LASTEXITCODE
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD,Env:PYTHONPATH `
    -ErrorAction SilentlyContinue
}
if ($code -ne 0) { exit $code }
```

### 5. 官方 JAX → PyTorch fixture parity

新增 `tests/algorithms/qam/test_official_fixture.py`，测试只依赖 NumPy、PyTorch 和
静态 `.npz`，不 import JAX。它从 manifest 重建官方 actor/10-Q MLP，并覆盖：

- FM、fine ODE、10-Q、pessimistic bootstrap、TD 和 critic loss；
- memoryless SDE path、最后一步 behavior ODE、endpoint Q 动作梯度、reverse
  behavior VJP、AM 和总 loss；
- online slow/fast/critic 原始与 clip 后梯度、global norm、Optax Adam 首步状态、
  PyTorch Adam 参数增量、pre-update EMA 和 target 零梯度。

同步测试和更新后的 oracle README 到服务器 QAM worktree 后执行：

```bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv

cd "$repo"
test "$(git rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git branch --show-current)" = codex/qam-pi0-robotwin
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=
"$venv/bin/python" -m pytest -q \
  tests/algorithms/qam/test_official_fixture.py
```

结果：

```text
...                                                                      [100%]
3 passed in 2.43s
```

这表明本批锁定的小网络 FM/AM/critic/optimizer/EMA 数值在既定容差内跨框架一致。

## QAM-IMPL-0005：固定 N 动作合同与 rank-local replay

时间：2026-07-31 12:05–12:35 CST。

新增：

| 文件 | 新增内容 |
|---|---|
| `rlinf/algorithms/qam/contracts.py` | `H=50/32D → N=20/14D` canonical clamp、投影和 $P_N^\top$；fixed-slot return；结束/bootstrap 语义；raw-policy 与 macro-transition 合同 |
| `rlinf/data/qam_transition_replay.py` | 有界 ring、raw observation 内容去重/引用计数、rank/world fingerprint、均匀采样、原子 checkpoint 和精确 RNG/cursor 恢复 |
| `tests/embodiment/test_robotwin_qam_contract.py` | 5 个合同/replay 聚焦测试 |

审查时发现模型配置中的 π0 prefix width 不能静默固定为 2048，因此在服务器测试前将
critic feature 合同从 `[4, 2048]` 窄修正为 `[4, D]`，要求 `D>0` 且当前/下一
observation 的宽度完全一致。四个 block 仍固定对应三相机和语言；这里的“双视图”是
同一 observation 的 critic pooled feature 与 policy raw conditioning 两种表示，
不是把三相机改成两相机。

代码通过 SFTP 只同步到 QAM worktree 后执行：

```bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv

test "$(git -C "$repo" rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git -C "$repo" branch --show-current)" = codex/qam-pi0-robotwin
mkdir -p \
  "$repo/rlinf/algorithms/qam" \
  "$repo/rlinf/data" \
  "$repo/tests/embodiment"

cd "$repo"
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=
"$venv/bin/python" -m pytest -q \
  tests/embodiment/test_robotwin_qam_contract.py
```

结果：

```text
.....                                                                    [100%]
5 passed in 1.66s
```

仍是 CPU 前置测试；shared venv 未安装依赖，GPU/Ray/RoboTwin 未启动。

## QAM-IMPL-0006：π0 B1+F1+C1 模型适配与真实单批量 probe

时间：2026-07-31 11:35–11:49 CST。

### 1. 代码改动

| 文件 | 改动 |
|---|---|
| `rlinf/models/embodiment/base_policy.py` | 新增 opt-in `ForwardType.QAM_FLOW` |
| `rlinf/models/embodiment/modules/qam_modules.py` | F1 action expert copy、QAM/π0 时间和速度映射、动态 C1 pooling、projection fingerprint、canonical rollout clamp |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | B1 完整冻结、F1 fine ODE、root-forward QAM conditioning/velocity/sample 入口、QAM rollout payload |
| `rlinf/models/embodiment/openpi/__init__.py` | SFT load 后初始化 F1；完整 QAM checkpoint 保留 F1；partial F1 fail closed；transform/norm fingerprint |
| `tests/embodiment/test_qam_openpi_adapter.py` | 动态 pooling、F1 copy、时间/速度翻转、fingerprint、canonical action 测试 |

关键边界：

- `openpi.use_qam=false` 时不构造 F1、不新增 state-dict/payload keys，legacy 分支保持原样；
- F1 只复制 `gemma_expert.model` 与 action/state/time projections，排除 VLM 和未使用的
  `lm_head`；
- SFT checkpoint 必须在 behavior 权重加载后再复制 F1；若 checkpoint 带部分而非完整
  `qam_fine.*` keys 则拒绝；
- `t_pi0=1-t_qam`、`f_qam=-v_pi0`；
- C1 按运行时 prefix width 返回 `[B,4,D]`，不硬编码 1024/2048；
- actor/rollout 都注册同构 F1；critic、target、replay 和 critic optimizer 不注册进
  policy；
- raw fine endpoint 保留在生成链；只把 `[0:N,0:14]` clamp 后的同一 canonical
  model action 同时交给 Q payload 和现有 `output_transform`，H suffix 和 14D 外 padding
  原样保留。

模型侧 5 个文件通过 SFTP 只同步到 QAM worktree 后执行：

```bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv

cd "$repo"
test "$(git rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git branch --show-current)" = codex/qam-pi0-robotwin
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=
"$venv/bin/python" -m pytest -q \
  tests/embodiment/test_qam_openpi_adapter.py
```

结果：

```text
.....                                                                    [100%]
5 passed, 3 dependency deprecation warnings in 8.23s
```

### 2. probe 前现场

执行：

```bash
date -Is
hostname
ps -eo pid,ppid,etime,%cpu,%mem,rss,args --sort=-rss |
  grep -E 'train_embodied_agent|ray::|raylet|gcs_server|qam' |
  grep -v grep || true
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
du -sh \
  /root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
```

结果（2026-07-31 11:42:45+08:00）：无训练/Ray/QAM 进程；两张 A800 均
`0/81920 MiB, util=0`；SFT checkpoint 为 7.6 GiB。probe 复用 shared π0 venv，
没有安装依赖，也没有启动仿真或写训练输出。

### 3. 基础真实模型 probe

probe 从已经验证的 DSRL N20 resolved config 只抽取 π0/transform/checkpoint 基线，在
进程内关闭 DSRL/RLT/NFT/value head 并打开 QAM，固定 `H=50,D=32,N=20,active=14`。
远端完整入口：

```bash
set -euo pipefail
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv

cd "$repo"
test "$(git rev-parse HEAD)" = \
  6d0db56bf26f972cd27fa29535f5eb939e80e5bf
test "$(git branch --show-current)" = codex/qam-pi0-robotwin
test -z "$(
  ps -eo args |
    grep -E 'train_embodied_agent|ray::|raylet|gcs_server' |
    grep -v grep || true
)"
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
export PYTHONPATH="$repo:/root/autodl-tmp/RoboTwin:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
"$venv/bin/python" -u /root/autodl-tmp/qam_real_model_basic_probe.py
```

第一次本地包装把 `shell_command.timeout_ms` 设为 10 秒；helper 被本地工具终止等待时，
远端 Python 已经启动并继续加载。只读复核见该进程 PID 54514、GPU 尚未占用；约 52 秒
后进程自行退出且 GPU 回到 0，没有可回收输出，也没有遗留进程。此后改用可 yield/wait
的长命令 cell 重跑同一 probe，不启动第二个并发进程。

可靠重跑结果：

```text
F1 parameters                 = 314,713,120
F1 dtype numel                = BF16 311,427,072; FP32 3,286,048
trainable tensors             = 173, all qam_fine.*
F1/base parameter copy        = exact; max_abs=0
base/fine velocity parity     = exact; max_abs=0
prefix output                 = [1,816,2048]
C1 feature                    = [1,4,2048]
block lengths                 = [256,256,256,48]
block valid counts            = [256,256,256,5]
load / conditioning / two-v   = 58.73s / 0.340s / 0.051s
single-GPU peak allocated     = 7,795,550,720 bytes
single-GPU peak reserved      = 7,902,068,736 bytes
QAM_REAL_MODEL_BASIC_OK       = 1
```

### 4. action VJP、F1 backward 与 clamp probe

第二个真实 probe 仍只用一个固定三相机 observation、一个 batch，不进入环境。它执行
10 步 fine Euler ODE，随后用 active `[20,14]` terminal adjoint 做一次 frozen behavior
action VJP，再做一次 F1 local velocity backward。命令入口与上节相同，只把 Python
文件换成：

```bash
"$venv/bin/python" -u \
  /root/autodl-tmp/qam_real_model_gradient_probe.py
```

结果：

```text
10-step sample                         = 0.258s
raw active overflow                    = 20 / 280 = 7.142857%
raw active max |a| / excess            = 1.291547 / 0.291547
behavior action VJP                    = 0.064s
VJP peak delta                         = 107,261,952 bytes
VJP inactive-coordinate max after pull = 0.723803
F1 local backward                      = 0.067s
F1 backward peak delta                 = 637,071,872 bytes
F1 grad tensors / finite / norm        = 173 / true / 35.6748
QAM_REAL_GRADIENT_OK                   = 1
```

解释：

- `P_N^\top` 的 terminal direct adjoint 在 inactive 坐标确实为零，但 behavior VJP 后
  inactive noisy-action 坐标可非零，验证了不能在每个 reverse flow time 再强制裁零；
- base 没有任何参数梯度，F1 的 173 个 tensor 都得到 finite gradient，梯度所有权正确；
- raw endpoint 有 7.14% 实质越界，不能再当成数值噪声。当前代码采用更贴近官方的
  canonical clamp，并让 Q 与环境执行同一动作；正式 smoke 前必须把“保留该 clamp”
  列为用户确认项。若选择不 clamp，必须显式报告为 π0-QAM action-bound 偏离，不能只
  clamp Q 而让环境执行 raw action。

两次成功 probe 退出后，GPU 回到约 4 MiB context baseline；没有训练/Ray 进程。

## QAM-IMPL-0007：QAM worker、10-Q critic 与 launch-closed 配置

时间：2026-07-31 11:50–12:08 CST。

本批新增：

| 文件 | 内容 |
|---|---|
| `rlinf/workers/actor/fsdp_qam_policy_worker.py` | QAM 专属 FSDP worker；rank-local replay、完整 10-Q/target-Q、TD、AM/VJP、pre-update EMA、phase/version、两卡同步和 sidecar resume |
| `rlinf/models/embodiment/modules/qam_critic.py` | 10 套完全独立的 FP32 Q MLP；每套为官方式 4×512、GELU-tanh、LayerNorm、Xavier-uniform/zero-bias |
| `examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml` | 双重 opt-in 的生产源配置；`max_steps=0` 保持 launch-closed |
| `tests/workers/test_qam_worker_helpers.py` | critic、terminal clamp、prefix、UTD、phase、三相机、配置和 synthetic M2 ingest 测试 |

本批修改：

| 文件 | 内容 |
|---|---|
| `examples/embodiment/train_embodied_agent.py` | 只在 `loss_type=embodied_qam` 时选择 QAM worker |
| `rlinf/config.py` | QAM 双开关及 fixed-N、B1+F1、C1、online-only、phase、数据 payload 和运行合同 |

收口值：

- actor `global_batch_size=64`，两卡每 rank 32；当前 worker 不再伪装成 micro-batch
  切分，故 `micro_batch_size=32`；
- replay `4096/rank`；按现场一份三相机 observation `2,764,800 bytes` 估算，raw
  observation 约 `11.3 GB/rank`、两卡约 `22.6 GB`，不再沿用不现实的
  `50,000/rank`；
- `rollout.collect_transitions=true`；raw replay 明确使用 main + 两个 wrist camera，
  `extra_view_images` 不被误当成第三路有效输入；
- 当前 payload 没有可信 timeout 种类，所有 truncation 保守归为
  `other_truncated` 且不 bootstrap；
- terminal-Q 对 raw AM endpoint 使用 canonical clamp；raw endpoint 仍保留在 SDE
  path，critic、rollout 与环境执行只读同一 `[20,14]` canonical action；
- source config 为 `phase=collect, inv_temp=0, am_evidence_passed=false`，不能直接进入
  AM，也不能在 `max_steps=0` 时误启动正式运行。

本地只做 `git diff --check` 静态检查，没有在 Windows 运行项目测试。

## QAM-PRE-0001：前置测试前服务器现场刷新

时间：2026-07-31 12:09 CST。授权：只读身份、进程、GPU、磁盘和 Git 刷新。

第一次本地 helper 包装把带 `$d` 的远端 `for` 循环放进 PowerShell 双引号；PowerShell
提前展开变量，`argparse` 返回：

```text
unrecognized arguments: ]; then printf ...
```

连接函数尚未执行，因此这次没有到达服务器、也没有服务器副作用。修复为 UTF-8
`--command-file` 后执行的完整远端指令：

```bash
set -eu
echo '=== identity ==='
hostname
pwd
id -u
date '+%F %T %Z'
echo '=== processes ==='
pgrep -af 'train_embodied_agent|ray::|rlt|dsrl|qam' || true
echo '=== gpu ==='
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
echo '=== disk ==='
df -h /root/autodl-tmp
echo '=== qam git ==='
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin branch --show-current
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin rev-parse HEAD
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin status --short
echo '=== other heads/status counts ==='
for d in \
  /root/autodl-tmp/RLinf_dsrl_pi0_robotwin \
  /root/autodl-tmp/RLinf_rlt_pi0_robotwin
do
  if [ -d "$d" ]; then
    printf '%s\n' "$d"
    git -C "$d" branch --show-current
    git -C "$d" rev-parse --short HEAD
    git -C "$d" status --short | wc -l
  fi
done
```

结果：

```text
host/uid/time = autodl-container-nekaqbwt43-6ce5babb / 0 /
                2026-07-31 12:09:29 CST
training/Ray = none
GPU0/GPU1   = 0 MiB / 0 MiB, utilization 0%
disk        = 1.9T total, 1.1T used, 807G free, 57%
QAM         = codex/qam-pi0-robotwin @
              6d0db56bf26f972cd27fa29535f5eb939e80e5bf
RLT         = codex/rlt-pi0-robotwin @ 2b8199d8, clean
```

`pgrep` 唯一输出是包含查询正则自身的临时 shell。没有 RLT 训练进程，故本批服务器
测试不会与其争用 GPU；没有修改 RLT worktree。

## QAM-PRE-0002：完整代码同步、CRLF 修复与集中前置测试

时间：2026-07-31 12:10–12:13 CST。

Paramiko 只复用一个已校验 host-key 的 password-auth Transport，并把以下清单用 SFTP
覆盖到 `/root/autodl-tmp/RLinf_qam_pi0_robotwin` 的同名相对路径：

```text
examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml
examples/embodiment/train_embodied_agent.py
rlinf/algorithms/qam/__init__.py
rlinf/algorithms/qam/contracts.py
rlinf/algorithms/qam/core.py
rlinf/algorithms/qam/UPSTREAM_NOTICE.md
rlinf/config.py
rlinf/data/qam_transition_replay.py
rlinf/models/embodiment/base_policy.py
rlinf/models/embodiment/modules/qam_critic.py
rlinf/models/embodiment/modules/qam_modules.py
rlinf/models/embodiment/openpi/__init__.py
rlinf/models/embodiment/openpi/openpi_action_model.py
rlinf/workers/actor/fsdp_qam_policy_worker.py
tests/algorithms/qam/oracle/export_official_fixture.py
tests/algorithms/qam/oracle/qam_official_2726d767_v1.npz
tests/algorithms/qam/oracle/README.md
tests/algorithms/qam/oracle/requirements.lock.txt
tests/algorithms/qam/oracle/resolved-freeze.txt
tests/algorithms/qam/test_core.py
tests/algorithms/qam/test_official_fixture.py
tests/embodiment/test_qam_openpi_adapter.py
tests/embodiment/test_robotwin_qam_contract.py
tests/workers/test_qam_worker_helpers.py
```

SFTP 输出逐项报告 `PUT <relative-path> <bytes>`；24/24 成功。随后执行：

```bash
set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "$repo"
export PYTHONPATH="$repo"
echo '=== git diff check ==='
git diff --check
echo '=== compile ==='
"$venv/bin/python" -m compileall -q \
  rlinf/algorithms/qam \
  rlinf/data/qam_transition_replay.py \
  rlinf/models/embodiment/modules/qam_critic.py \
  rlinf/models/embodiment/modules/qam_modules.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/algorithms/qam \
  tests/embodiment/test_qam_openpi_adapter.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py
echo '=== qam pytest ==='
"$venv/bin/python" -m pytest -q \
  tests/algorithms/qam/test_core.py \
  tests/algorithms/qam/test_official_fixture.py \
  tests/embodiment/test_qam_openpi_adapter.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py
```

第一次在 `git diff --check` 停止；服务器把 Windows SFTP 副本中的 CRLF `\r` 判成每行
尾随空白。没有进入 compile 或 pytest。修复只对上述清单中的文本文件机械执行
`CRLF/CR -> LF`，不改内容；`.npz` 保持原二进制，再次 SFTP 同步 23 个文本文件。

原样复测结果：

```text
git diff --check = pass
compileall         = pass
pytest             = 31 passed, 3 dependency deprecation warnings in 8.44s
```

shared venv 没有安装或修改任何依赖；本批没有启动 Ray、RoboTwin、smoke 或训练。

## QAM-PRE-0003：Hydra compose、legacy gate、lint 与格式复测

时间：2026-07-31 12:14–12:17 CST。

### 1. 配置 compose 与合同验证

SFTP 上传 `/root/autodl-tmp/qam_config_probe.py`，随后执行：

```bash
set -eu
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "$repo"
export PYTHONPATH="$repo"
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
"$venv/bin/python" -u /root/autodl-tmp/qam_config_probe.py
```

probe 用 Hydra compose `robotwin_adjust_bottle_qam_openpi`，调用完整
`validate_cfg()`，再断言 launch gate、phase、双开关、batch、replay 和 transition
collection；随后单独 compose 既有 `robotwin_adjust_bottle_ppo_openpi`，证明 QAM
validator 在 `use_qam=false` 时直接返回、不对 legacy 增加约束。

结果：

```text
QAM_COMPOSE_OK {
  max_steps: 0, phase: collect,
  global_batch: 64, micro_batch: 32, envs: 2,
  use_qam: true, use_dsrl: false, use_rlt: false
}
LEGACY_QAM_OFF_OK {loss_type: actor_critic, use_qam: false}
```

`validate_cfg()` 的既有 `Cluster()` 路径临时启动了一个 local Ray instance 来解析两卡
placement。脚本退出后立即执行：

```bash
date '+%F %T %Z'
pgrep -af 'raylet|gcs_server|dashboard|monitor.py|log_monitor.py' || true
ls -1dt /tmp/ray/session_* 2>/dev/null | head -3 || true
```

12:14:50 CST 没有 Ray 服务进程；只留下本次普通 `/tmp/ray/session_*` 日志目录，没有
删除。没有训练进程或 GPU 占用。

### 2. Ruff

在 shared venv 中对 18 个本批 Python 文件执行：

```bash
/root/autodl-tmp/RLinf/.venv/bin/ruff check <18 changed Python files>
/root/autodl-tmp/RLinf/.venv/bin/ruff format --check <18 changed Python files>
```

第一次 `ruff check` 全部通过；全量 format-check 报 14 个文件会重排，其中包含
`rlinf/config.py`、`openpi_action_model.py` 和训练入口等大段既有文件。为避免把一个
QAM 分支扩成 legacy 全文件格式化，只对 13 个**本批新建 Python 文件**运行
`ruff format`：11 个被机械重排、2 个不变。格式后的服务器文件通过同一 Paramiko
Transport SFTP 拉回本地副本。

最终结果：

```text
ruff check changed files       = All checks passed
ruff format --check new files  = 13 files already formatted
```

随后原样重跑 `QAM-PRE-0002` 的 `git diff --check + compileall + 5-file pytest`：

```text
git diff --check = pass
compileall         = pass
pytest             = 31 passed, 3 dependency deprecation warnings in 8.24s
```

## QAM-IMPL-0008：FSDP-compatible VJP 与流式 AM

时间：2026-07-31 12:32–12:48 CST。授权：继续 QAM 实施和正式 smoke 前测试；
不得改 DSRL/RLT，不得启动正式 smoke。

两卡真实模型预检确认 PyTorch FSDP `FULL_SHARD + use_orig_params=True` 不支持在
FSDP forward 后调用 `torch.autograd.grad()`。首版 core 的 input-only VJP 在普通
PyTorch 数值上正确，但不能直接进入生产 FSDP worker。

窄修复：

```text
rlinf/algorithms/qam/core.py
  reverse_behavior_adjoint(..., use_backward_vjp=False)
    默认仍走 autograd.grad，官方数值 oracle 不变
    生产 opt-in 时逐 reverse step：
      fresh detached state.requires_grad_(True)
      reverse_drift.backward(gradient=adjoint)
      VJP = state.grad.detach()

  adjoint_matching_step_loss(...)
    先 no-grad behavior target
    再做一个 fine forward
    返回该 flow-time 的 batch-mean AM contribution

rlinf/algorithms/qam/__init__.py
  导出 adjoint_matching_step_loss

rlinf/workers/actor/fsdp_qam_policy_worker.py
  B1 worker 显式 use_backward_vjp=True
  K=10 个 AM contribution 各自立即 backward，梯度累积后只 optimizer.step 一次

tests/algorithms/qam/test_core.py
  backward-VJP 与 autograd.grad 数值 parity
  frozen behavior 无参数梯度
  逐 step loss/梯度与完整官方 reduction parity
```

逐 step 累积没有除以 `K`，因为它与官方 reduction 严格相同：

$$
\sum_i \operatorname{mean}_b \sum_d r_{ibd}^2
=
\operatorname{mean}_b \sum_i \sum_d r_{ibd}^2.
$$

没有使用 FSDP `no_sync()`：当前正确性优先，每步允许一次 reduce-scatter；避免
`no_sync()` 暂留完整未分片梯度的显存代价。behavior 已由 B1 合同冻结，所以普通
backward 不会产生 behavior 参数梯度。target critic 非 FSDP，terminal
`dQ/da` 仍保留 `autograd.grad()`。

同步方式：以上 4 个文本文件经已校验 host-key 的 Paramiko/SFTP 上传到独立 QAM
worktree；密码仅存在于当前进程环境变量。服务器 `ruff format` 最终机械重排 1 个
文件，4 个文件随后 SFTP 拉回本地副本。shared pi0 venv 未安装、卸载或升级依赖。

## QAM-PRE-0004：真实两卡 FSDP 失败链、修复与完整 K=10 复测

时间：2026-07-31 12:21–12:48 CST。目的：在不启动 Ray、RoboTwin 或训练循环的
前提下，用真实 π0 checkpoint 验证两卡 FSDP、B1/F1、C1、K=10 VJP/AM、10-Q
同步和 target EMA。

每轮使用同一条有界命令；batch=32 轮只额外设置
`QAM_PROBE_BATCH=32`：

```bash
set -u
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
venv=/root/autodl-tmp/RLinf/.venv
cd "$repo"
export PYTHONPATH="$repo"
export EMBODIED_PATH="$repo/examples/embodiment"
export REPO_PATH="$repo"
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# batch=32 轮：export QAM_PROBE_BATCH=32
date '+%F %T %Z'
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
status=0
timeout --signal=TERM --kill-after=20s 360s \
  "$venv/bin/python" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=2 \
  /root/autodl-tmp/qam_two_gpu_fsdp_probe.py || status=$?
pgrep -af 'qam_two_gpu_fsdp_probe|torch.distributed.run' || true
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
exit "$status"
```

### 失败 1：tied embedding 跨 FSDP ownership boundary

12:21 首轮在 conditioning 的 `embed_tokens` 报：

```text
RuntimeError: 'weight' must be 2-D
```

只读来源检查：

```bash
grep -R -n 'embed_tokens\|lm_head\|tie_word\|tie_weights' \
  /root/autodl-tmp/RLinf/.venv/lib/python3.11/site-packages/openpi/models_pytorch/{gemma_pytorch.py,pi0_pytorch.py}

PYTHONPATH=/root/autodl-tmp/RLinf_qam_pi0_robotwin:/root/autodl-tmp \
CUDA_VISIBLE_DEVICES= \
/root/autodl-tmp/RLinf/.venv/bin/python -u \
  /root/autodl-tmp/qam_inspect_shared_params.py
```

结果只有一组共享参数：

```text
paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight
paligemma_with_expert.paligemma.lm_head.weight
```

原因：既有 `_no_split_names` 把 `lm_head` 包成 child FSDP，而 tied embedding 留在
root；`FULL_SHARD + use_orig_params=True` 把同一参数拆到两个 ownership boundary。
窄修复只在 QAM 下给 tied PaliGemma head 唯一 wrap name，使它与 embedding 一起
留在 root；独立 action expert head 继续按 legacy 规则包装。toy test、ruff 和
真实两卡 conditioning 随后通过。

### 失败 2：FSDP 不支持 `autograd.grad`

12:30 复测越过 conditioning 后，在 behavior input VJP 报：

```text
RuntimeError: A leaf node was passed to _will_engine_execute_node ...
running autograd.grad(). This is currently not supported.
```

原因和修复即 `QAM-IMPL-0008`。先在服务器运行：

```bash
cd /root/autodl-tmp/RLinf_qam_pi0_robotwin
export PYTHONPATH=$PWD
/root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  tests/algorithms/qam/test_core.py \
  tests/algorithms/qam/test_official_fixture.py \
  tests/embodiment/test_qam_openpi_adapter.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py
```

结果：`34 passed, 3 dependency deprecation warnings in 8.36s`。

### probe harness 的两个非生产失败

12:38 的两卡 probe 已完成 10 次 VJP 和 10 次 streaming AM，随后 probe 自己调用
PyTorch 原生 `FSDP.clip_grad_norm_()` 时拒绝 BF16/FP32 混合梯度：

```text
ValueError: Requires uniform dtype across all gradients but got
{torch.float32, torch.bfloat16}
```

F1 的 BF16 主干与 FP32 projection 是既有预期；生产
`FSDPStrategy.clip_grad_norm_()` 已按 dtype 分组并在 FP32 中合成全局范数。probe
因此改为直接调用该生产函数。12:41 首次调用误传顶层 `cfg`，报
`ConfigAttributeError: Key 'optim' is not in struct`；worker 实际传 `cfg.actor`，
probe 同步修正。这两次均在 optimizer step 前退出，无权重或 checkpoint 产物；
elastic launcher 清理子进程，下一轮前两卡显存均为 0。

### 通过结果

12:43，batch=1：

```text
QAM_TWO_GPU_FSDP_OK=1
trainable_name_count = 173
fine_grad_norms      = [3546.109619140625, 3546.109619140625]
critic checksum      = 两 rank 完全相同
target checksum      = 两 rank 完全相同
peak allocated       = 7,987,814,400 / 7,988,862,976 bytes
```

12:46，按正式配置每 rank batch=32：

```text
QAM_TWO_GPU_FSDP_OK=1
fine_grad_norms = [889.205810546875, 889.205810546875]
critic checksum = 两 rank 完全相同
target checksum = 两 rank 完全相同
peak allocated  = [14,148,494,848, 14,148,494,848] bytes
```

两轮均实际覆盖：

1. 真实 SFT π0 load 和两卡 `FULL_SHARD + use_orig_params=True`；
2. tied PaliGemma root ownership 与 independent F1 head child ownership；
3. 三视角 prefix conditioning 和 C1 `[B,4,2048]`；
4. 完整 K=10 behavior backward-VJP；
5. 完整 K=10 streaming AM backward、173 个 F1 gradient finite；
6. RLinf mixed-dtype global grad norm、clip 和 AdamW step；
7. 10 个独立 FP32 Q 的跨 rank gradient average、step 和 pre-update EMA；
8. 两 rank critic/target checksum 一致。

12:48 最终清场与回归：

```bash
/root/autodl-tmp/RLinf/.venv/bin/ruff check \
  rlinf/algorithms/qam/{__init__.py,core.py} \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/algorithms/qam/test_core.py
/root/autodl-tmp/RLinf/.venv/bin/ruff format --check <same files>
git diff --check
/root/autodl-tmp/RLinf/.venv/bin/python -m compileall -q <QAM paths>
/root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q <same five test files>
pgrep -af 'qam_two_gpu_fsdp_probe|torch.distributed.run|raylet|gcs_server' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
```

结果：

```text
ruff/check/format = pass
git diff --check  = pass
compileall         = pass
pytest             = 34 passed, 3 warnings in 8.40s
training/Ray       = none（pgrep 仅匹配查询 shell 自身）
GPU0/GPU1          = 4 MiB / 4 MiB, utilization 0%
```

未执行：RoboTwin env、Ray worker lifecycle、正式 smoke、训练、checkpoint 写入或
Git push。没有修改 DSRL/RLT worktree，没有删除磁盘文件。

## QAM-IMPL-0009：UTD anchor 与 terminal replay 因果性收口

时间：2026-07-31 12:49–13:02 CST。授权：继续 QAM 实施和正式 smoke 前测试。

最终调用链审查发现并窄修两类问题：

1. `collect` 阶段只负责积累 replay，不应同步积累一笔稍后在 `q_only` 全部追补的
   optimizer debt。现在切入 `q_only` 时以当前 global insert count 建 anchor，
   `pending_update_credit=0`；之后才按新增 global macro 与 UTD=1 计 credit。
2. RoboTwin trajectory 在某个 env 已 terminal 后仍可能带固定长度 padding slot。
   replay 现在按 env 维护 `alive`，保留第一条真实 terminal transition、跳过其后的
   padding；若 `terminated=true` 与 `truncated=true` 同时出现，success termination
   优先且不 bootstrap。

改动只落在：

```text
rlinf/workers/actor/fsdp_qam_policy_worker.py
tests/workers/test_qam_worker_helpers.py
```

集中测试增加 collect→q_only anchor、首 terminal 保留/后续 padding 丢弃、同时 done 的
success 优先三类断言。没有增加 heuristic timeout 推断：当前 payload 无法区分 time-limit，
因此所有非 success truncation 继续保守地不 bootstrap。

## QAM-PRE-0005：最终集中回归与静态审查

时间：2026-07-31 13:02–13:08 CST。全部在服务器独立 QAM worktree、shared π0 venv
中执行；未启动 RoboTwin/Ray 正式运行。

完整命令：

```bash
cd /root/autodl-tmp/RLinf_qam_pi0_robotwin
export PYTHONPATH=$PWD
/root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  tests/algorithms/qam/test_core.py \
  tests/algorithms/qam/test_official_fixture.py \
  tests/embodiment/test_qam_openpi_adapter.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py

/root/autodl-tmp/RLinf/.venv/bin/ruff check \
  rlinf/algorithms/qam \
  rlinf/data/qam_transition_replay.py \
  rlinf/models/embodiment/modules/qam_critic.py \
  rlinf/models/embodiment/modules/qam_modules.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/algorithms/qam \
  tests/embodiment/test_qam_openpi_adapter.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py
/root/autodl-tmp/RLinf/.venv/bin/ruff format --check <同一批本次新增 Python 文件>
/root/autodl-tmp/RLinf/.venv/bin/python -m compileall -q \
  rlinf/algorithms/qam \
  rlinf/data/qam_transition_replay.py \
  rlinf/models/embodiment/modules/qam_critic.py \
  rlinf/models/embodiment/modules/qam_modules.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py
git diff --check
```

结果：

```text
pytest             = 36 passed, 3 dependency deprecation warnings, 8.16s
ruff check         = pass
ruff format-check  = pass
compileall         = pass
git diff --check   = pass
```

两轮独立静态 review 继续检查 Plain-QAM loss/update、FSDP collective、10-Q ownership、
replay causal fields、legacy opt-in 和 DCP sidecar；本批发现的 UTD/padding 问题修复后，
正常 fresh 路径未留 P0/P1。正式 env payload、真实 DCP 产物和 resume 仍由获批 smoke
验证，不能用 synthetic tests 替代。

## QAM-PRE-0006：resolved config、失败 overlay 与审批证据

时间：2026-07-31 13:08–13:15 CST。目标：形成可审计 smoke 配置，不执行 smoke。

最初尝试新增 secondary smoke YAML，只覆盖 formal source。Hydra compose 在该 secondary
config 再次出现 `hydra.searchpath` 时拒绝覆盖；失败发生在配置合成阶段，没有创建 Ray、
模型、env、run root 或 checkpoint。原因是 Hydra 的 search path 只能由 primary config
定义，不是 QAM 字段错误。

窄解决不是放宽 Hydra，而是删除**本批自己新增、仍未跟踪**的单个失败 overlay：

```bash
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
target="$repo/examples/embodiment/config/robotwin_adjust_bottle_qam_openpi_qonly_smoke.yaml"
test -f "$target"
rm -- "$target"
test ! -e "$target"
```

该文件未提交、无用户内容；本机也删除同一工作副本。正式入口继续只有
`robotwin_adjust_bottle_qam_openpi.yaml`，smoke 用显式 CLI overrides，并把
compose→`validate_cfg()` 后的完整结果固化。执行：

```bash
cd /root/autodl-tmp/RLinf_qam_pi0_robotwin
export PYTHONPATH=$PWD
export EMBODIED_PATH=$PWD/examples/embodiment
export REPO_PATH=$PWD
/root/autodl-tmp/RLinf/.venv/bin/python -B /root/autodl-tmp/qam_config_probe.py
bash -n /root/autodl-tmp/qam_resource_monitor_20260731_v1.sh
diff -u \
  /root/autodl-tmp/qam_source_resolved_20260731_v1.yaml \
  /root/autodl-tmp/qam_qonly_smoke_resolved_20260731_v1.yaml \
  > /root/autodl-tmp/qam_source_to_qonly_smoke_20260731_v1.diff
```

固定证据：

| artifact | SHA-256 |
|---|---|
| source post-validation resolved | `ae1faf2e177f6ca5abce17a27056c191c11aaaf0c0d30a6063cb14f17f0dfdfd` |
| q-only smoke post-validation resolved | `ce8661de889992357f473068e481ac5b6c56f44fb9eddeaee3e20858db9cefee` |
| source→smoke exhaustive diff | `ea19fd759f653f2ee924ca45a3a67524a6a9d5e9dc6a34779f3396c56a37c998` |
| 2-second resource monitor | `01f0e4087f58a6a6c72e1865cb683639df377bb9bf3a85a54f4ae634bd0282d7` |

又用真实 CLI 的 `--cfg job --resolve` 重放同一 override。其 SHA 为
`4bd8ac...`，与 post-validation artifact 不同；穷尽 diff 只有
`actor.fsdp_config.grad_scaler.{init_scale,growth_interval}` 和
`runner.{per_worker_log,per_worker_log_path}` 四个字段。它们由运行入口随后调用的
`validate_cfg()` 注入，因此 post-validation artifact 才是运行时真值；没有算法或预算
差异。

13:11 发现服务器 QAM worktree 中有四份 11:11 CST 从本机复制过去的专题文档旧副本，
它们不是代码 commit 的一部分，并会让运行 worktree dirty。逐一核对目标后只删除这四份
由本批生成的重复副本及空目录：

```text
docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-qam/01_CONTEXT_AND_SOURCE_MAP.md
docs/rlinf-robotwin-pi0-qam/02_METHOD_AND_PORT_DECISION_GUIDE.md
docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
```

本机 canonical 文档全部保留，故可恢复；没有删除实验、checkpoint、cache 或用户文件。

## QAM-GIT-0001：首个实现提交与双机 tree 对齐

时间：2026-07-31 13:03–13:15 CST。

本机独立 code worktree 的第一次 `git commit` 因该 worktree 没有 author identity 而
拒绝，index 和文件未丢失。未修改 global/repository config；只在该次命令用现有身份：

```bash
git -c user.name='Zhou Yiming' \
  -c user.email='149066435+YimingZhou2002@users.noreply.github.com' \
  commit -m 'feat(qam): add pi0 RoboTwin plain QAM adaptation'
```

服务器同样精确 `git add` 24 个目标文件、`git diff --cached --check` 后使用 command-scoped
identity 提交。结果：

```text
local  commit = c32d044bcb559aa9c618dcb74c23263592ee0b50
server commit = 7bc5f87086035087adf6d44ddda76eb5a9e54ee8
tree hash     = 6dc9124ba63b5712918ba2dbdcffde203cfb5eed
files         = 24
diffstat      = 7,079 insertions / 3 deletions
```

commit ID 因提交元数据不同而不同，但 tree byte-identical。服务器 worktree 随后 clean。
未 push；本轮用户没有授权要求发布远端分支。

## QAM-PRE-0007：13:16–13:21 正式 smoke 前现场刷新

通过已锁定 host-key 的 Paramiko/密码认证执行只读审计。第一版脚本错误沿用了不存在的
`assets/aloha-agilex/adjust_bottle/norm_stats.json`；其余检查已完成，但
`sha256sum` 报 No such file，而且脚本只有 `set -u`，所以错误地返回 0。

只读 `grep/find` 确认 source config 的实际路径是
`physical-intelligence/robotwin/norm_stats.json`。脚本改为 `set -eu`、修正路径并把
process query 改成不会匹配自身的 exact/pattern 后，原样全量复测通过：

```text
time              = 2026-07-31 13:21:12 CST
host / uid        = autodl-container-nekaqbwt43-6ce5babb / 0
server branch     = codex/qam-pi0-robotwin
server HEAD       = 7bc5f87086035087adf6d44ddda76eb5a9e54ee8
server tree       = 6dc9124ba63b5712918ba2dbdcffde203cfb5eed
server status     = clean
training/Ray      = none
GPU 0/1           = 0 MiB, utilization 0%
disk              = 1.9 TiB total / 1.1 TiB used / 807 GiB free / 57%
inode             = 2%
smoke run root    = absent
source YAML SHA   = d3da1b66d24233300e2a5cebebdf9cb9bcb9e17db959c72e9efcf85dcff1cc6f
norm stats SHA    = 649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a
model index SHA   = 79b9eae15b87f8757471b1040bd27fba4b7731feb302c347bbdc55e4765f0311
monitor SHA       = 01f0e4087f58a6a6c72e1865cb683639df377bb9bf3a85a54f4ae634bd0282d7
shared venv       = 14 GiB; unchanged
oracle venv       = 730 MiB; isolated
```

包版本为 Python 3.11.14、Torch 2.6.0、Ray 2.55.1、
Hydra 1.4.0.dev1、OmegaConf 2.4.0.dev11。该刷新没有启动、停止或修改任何进程。

## QAM-PRE-0008：fresh q-only 正式 smoke 批准包（尚未执行）

状态：**等待用户批准**。本节和配套 launcher 的形成、上传、`bash -n` 不等于运行；
本轮没有启动 RoboTwin、Ray、driver 或 env。

### 1. 冻结身份与配置

| 项 | 值 |
|---|---|
| server cwd | `/root/autodl-tmp/RLinf_qam_pi0_robotwin` |
| branch | `codex/qam-pi0-robotwin` |
| server HEAD | `7bc5f87086035087adf6d44ddda76eb5a9e54ee8` |
| tree | `6dc9124ba63b5712918ba2dbdcffde203cfb5eed` |
| shared venv | `/root/autodl-tmp/RLinf/.venv`（只读复用） |
| source YAML SHA | `d3da1b66d24233300e2a5cebebdf9cb9bcb9e17db959c72e9efcf85dcff1cc6f` |
| source resolved SHA | `ae1faf2e177f6ca5abce17a27056c191c11aaaf0c0d30a6063cb14f17f0dfdfd` |
| smoke resolved SHA | `ce8661de889992357f473068e481ac5b6c56f44fb9eddeaee3e20858db9cefee` |
| source→smoke diff SHA | `ea19fd759f653f2ee924ca45a3a67524a6a9d5e9dc6a34779f3396c56a37c998` |
| norm stats SHA | `649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a` |
| model index SHA | `79b9eae15b87f8757471b1040bd27fba4b7731feb302c347bbdc55e4765f0311` |
| resource monitor SHA | `01f0e4087f58a6a6c72e1865cb683639df377bb9bf3a85a54f4ae634bd0282d7` |
| launcher SHA | `220fa7f3dab9814be0917070e803602237a21c94592a75945dca60364f782d5a` |

完整文件：

- [launch-closed source resolved](qam_source_resolved_20260731_v1.yaml)；
- [fresh q-only smoke resolved](qam_qonly_smoke_resolved_20260731_v1.yaml)；
- [source→smoke 穷尽 diff](qam_source_to_qonly_smoke_20260731_v1.diff)；
- [经 `bash -n` 的启动脚本](qam_qonly_smoke_launch_20260731_v1.sh)。

diff 只改变派生 video 路径、唯一 run/experiment 名、`max_steps/save_interval`、
`collect→q_only`、warm-up/replay ready/update cap、batch `64/32→2/1`，并关闭额外
rank-0 full-weight 导出；N20、K10、10-Q、F1/C1/M2、2 GPU/2 env、UTD1、
replay4096/rank 和 `inv_temp=0` 均不变。

### 2. 精确启动

用户批准后，先再次运行 `QAM-PRE-0007` 同级只读现场刷新；只有 branch/HEAD/tree/clean、
所有 hash、run/runtime root 不存在、无 Ray/training、两卡各 ≤100 MiB 且利用率 0、
数据盘可用 ≥100 GiB 全部通过，才执行：

```bash
nohup bash /root/autodl-tmp/qam_qonly_smoke_launch_20260731_v1.sh \
  >/root/autodl-tmp/qam_qonly_smoke_launcher_20260731_v1.log 2>&1 &
echo $!
```

launcher 内部的训练命令是：

```bash
cd /root/autodl-tmp/RLinf_qam_pi0_robotwin
export PYTHONPATH=/root/autodl-tmp/RLinf_qam_pi0_robotwin:/root/autodl-tmp/RoboTwin_RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf_qam_pi0_robotwin/examples/embodiment
export REPO_PATH=/root/autodl-tmp/RLinf_qam_pi0_robotwin
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

timeout --signal=TERM --kill-after=180s 7200s \
  /root/autodl-tmp/RLinf/.venv/bin/python -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf_qam_pi0_robotwin/examples/embodiment/config \
  --config-name robotwin_adjust_bottle_qam_openpi \
  runner.logger.log_path=/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1 \
  runner.logger.experiment_name=robotwin_adjust_bottle_qam_qonly_smoke_20260731_v1 \
  runner.max_steps=1 \
  runner.save_interval=1 \
  runner.resume_dir=null \
  runner.ckpt_path=null \
  algorithm.qam.phase=q_only \
  algorithm.qam.warmup_global_inserts=2 \
  algorithm.qam.min_replay_per_rank=1 \
  algorithm.qam.max_updates_per_step=2 \
  actor.global_batch_size=2 \
  actor.micro_batch_size=1 \
  +actor.fsdp_config.save_full_model_weights=false
```

### 3. 输出与预算

| 口径 | 预算 |
|---|---:|
| outer cycle / train episodes | 1 / 2 |
| requested action slots | 最多 400 |
| global macro replay inserts | 预计 2–20 |
| critic optimizer updates | **恰好 2** |
| fine/AM updates / fine policy version | 0 / 0 |
| eval episodes | 0 |
| checkpoint | 1 个 `global_step_1` DCP |
| wall-clock / GPU-hours hard limit | 2h / 4 |
| GPU memory stop | 60 GiB/卡 |
| host anon stop | 180 GiB |
| 新增磁盘 stop | 25 GiB |

run root：

```text
/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1/
  robotwin_adjust_bottle_qam_qonly_smoke_20260731_v1/
  checkpoints/global_step_1
```

runtime evidence：

```text
/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime
```

其中记录 resolved、diff、provenance、exact command、budget、stop conditions、driver
console、2 秒资源 CSV、PID 和 exit code。`save_full_model_weights=false` 只省略额外的
rank-0 完整模型导出；distributed DCP 及 QAM rank sidecar/replay 仍保存。

### 4. 停止和最低通过条件

立即停止：loss/Q/TD/grad/action 任一 NaN/Inf；CUDA OOM；NCCL/Ray fatal 或 rank death；
cgroup OOM/OOM-kill；QAM payload/合同失败；冻结 behavior 变化；30 分钟无新进度；
超过上述 GPU/anon/disk/hard-time 预算；DCP 或 sidecar 不完整。

最低通过：

1. 两 rank replay 各至少 1 条、global inserts 至少 2；
2. `updates_run=critic_updates=2`，critic loss/Q/TD/grad finite；
3. `fine_updates=0`、`fine_policy_version=0`；
4. 对应的 critic/target state 跨 rank 一致，第 2 次 update 后 pre-update target EMA
   产生可解释的数值变化；
5. `global_step_1` DCP、两 rank sidecar 和两 replay 完整，保存的 step/counters/contract
   在两 rank 一致；
6. 无 OOM、fatal、payload mismatch 或残留训练/Ray 进程。

本 smoke 只证明
`RoboTwin → macro replay → TD next fine ODE → 10-Q → target EMA → DCP` 的 fresh
端到端链。它不证明 resume、AM live-worker、策略收益或成功率上涨；这些分别另行审批。

## QAM-REVIEW-0001：exact resume 缺口与不过度加固边界

时间：2026-07-31 13:22–13:35 CST。只发生在本机 code 副本，没有同步服务器。

并行静态 review 指出：当前 sidecar 保存 critic/target/optimizer/counters 和 replay，
replay 自身也保存 ring RNG；但 worker 尚未保存 Python/NumPy/Torch rank-local process
RNG，也没有 QAM 两 rank 同代 completion manifest。因此当前实现足够验证 fresh save，
不能声称 fresh→resume 的下一次随机 TD/AM update exact 连续。

曾在本机起草 RNG+manifest 补丁；第一稿增长超过 600 行，并引入多层通用校验，超出用户
“检查适量、不要臃肿护栏”的要求，而且中途缩减时形成了不可用半成品。该探索从未上传、
测试或提交。随后只用 `apply_patch` 把
`qam_transition_replay.py` 和 `fsdp_qam_policy_worker.py` 精确恢复到 code HEAD，
`git diff --exit-code -- <两文件>` 返回 0。

决策：不把未审清的大补丁塞进 fresh smoke。当前批准包明确只做 fresh `q_only`；
fresh 通过后，再以独立窄批次补 rank-local current-device RNG、step/snapshot identity
和 rank-0 completion manifest，并为 resume 单独提交配置、预算和批准。

## QAM-PRE-0009：launcher 上传、语法检查与最终未启动证明

时间：2026-07-31 13:37–13:38 CST。

先验证服务器目标 launcher、run root 和 runtime root 均不存在，再用同一已锁定
Paramiko/SFTP 连接上传：

```text
local:
  docs/rlinf-robotwin-pi0-qam/evidence/
  qam_qonly_smoke_launch_20260731_v1.sh
server:
  /root/autodl-tmp/qam_qonly_smoke_launch_20260731_v1.sh
```

随后只执行：

```bash
bash -n /root/autodl-tmp/qam_qonly_smoke_launch_20260731_v1.sh
sha256sum /root/autodl-tmp/qam_qonly_smoke_launch_20260731_v1.sh
stat --format='%s %y %n' /root/autodl-tmp/qam_qonly_smoke_launch_20260731_v1.sh
```

结果：`bash -n` 通过；大小 7,377 bytes；SHA-256
`220fa7f3dab9814be0917070e803602237a21c94592a75945dca60364f782d5a`。

13:38:22 CST 最终全量只读复核：

```text
branch/HEAD/tree/status = codex/qam-pi0-robotwin /
                          7bc5f870... /
                          6dc9124b... /
                          clean
training/Ray            = none
GPU0/GPU1               = 0 MiB / 0 MiB, utilization 0%
disk                    = 1.9 TiB total / 1.1 TiB used / 807 GiB free / 57%
run root                = absent
runtime root            = absent
source/resolved/diff/norm/model-index/monitor/launcher hashes = all exact
```

正式 smoke 仍未执行。

## QAM-GIT-0002：云端发布与一次性 AutoDL turbo

时间：2026-07-31 14:05–14:07 CST。用户明确授权当前 QAM Git 分支发布，并新增长期
偏好：每个完成最小验证的连贯改动批次由实施侧主动 commit/push，不再逐次等待发布确认。

先按既有大陆网络短流程只读检查：

```bash
env | grep -iE '^(http|https|all)_proxy=' || true
git config --get http.version || printf 'DEFAULT\n'
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'main code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'api code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://api.github.com
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'raw code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://raw.githubusercontent.com
timeout 15 git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin \
  ls-remote --heads personal codex/qam-pi0-robotwin
```

结果：无 proxy、Git HTTP default；main HTTP000/7 秒 connect timeout、API HTTP200、
raw HTTP000/10 秒 timeout，`ls-remote` 返回 124。身份、branch、HEAD/tree 和 clean
均已通过，所以这是 GitHub 主站/smart-HTTP 路由问题，不是密码、GitHub认证、仓库或
commit 错误。

随后只在子 shell 临时启用官方加速：

```bash
(
  set +u
  source /etc/network_turbo >/dev/null 2>&1
  set -u
  timeout 15 git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin \
    ls-remote personal refs/heads/codex/qam-pi0-robotwin
  GIT_TERMINAL_PROMPT=0 timeout 60 \
    git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin \
    push --set-upstream personal \
    HEAD:refs/heads/codex/qam-pi0-robotwin
  timeout 15 git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin \
    ls-remote personal refs/heads/codex/qam-pi0-robotwin
)
```

远端分支此前不存在；push 创建：

```text
Yutenji-Nyamu/rlinf_fastwam
codex/qam-pi0-robotwin
remote HEAD = 7bc5f87086035087adf6d44ddda76eb5a9e54ee8
ahead/behind = 0/0
```

父 shell 在 source 前后均为 `proxy=NONE`；没有打印具体代理端点、写 shell 配置、改
remote/Git HTTP config、force push 或重写历史。正式 smoke 仍未启动。

## QAM-DISK-0002：清理候选逐目录现场审计

时间：2026-07-31 14:11–14:15 CST。目的只是回答磁盘归属和可恢复性；未删除、移动、
压缩或改写任何实验资产。

第一次发现命令先列出 Fast-WAM smoke、RLT 目录和旧 backup 顶层，用来锁定精确目标。
随后在同一只读 Paramiko 路径执行下面的完整深审命令：

```bash
set -u
targets=(
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_013332-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke
  /root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260719_120513-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke
  /root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1
  /root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40
)
date
df -h /root/autodl-tmp
for path in "${targets[@]}"; do
  test -d "$path" || { echo "MISSING $path"; continue; }
  stat --format='%y %n' "$path"
  du -sx --block-size=1 "$path"
  du -x --block-size=1 --max-depth=3 "$path" 2>/dev/null |
    sort -nr | head -35
  find "$path" -xdev -type d -name 'global_step_*' -print | sort
  find "$path" -xdev -type f \
    -printf '%s\t%TY-%Tm-%Td %TH:%TM:%TS\t%p\n' 2>/dev/null |
    sort -nr | head -25
  find "$path" -xdev -type f -name '*.mp4' -printf x 2>/dev/null | wc -c
  find "$path" -xdev -type f -name '*.distcp' -printf x 2>/dev/null | wc -c
  find "$path" -xdev -type f -name 'full_weights.pt' -printf x 2>/dev/null | wc -c
  find "$path" -xdev -type f -iname '*resolved*.yaml' -printf x 2>/dev/null | wc -c
  find "$path" -xdev -type f -name '*.log' -printf x 2>/dev/null | wc -c
done
backup=/root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40
find "$backup" -mindepth 1 -maxdepth 1 \
  -printf '%y\t%TY-%Tm-%Td %TH:%TM:%TS\t%p\n' | sort
find "$backup" -mindepth 1 -maxdepth 1 -type d -name '.venv*' \
  -exec du -sx --block-size=1 {} + | sort -nr
du -x --block-size=1 --max-depth=1 "$backup/logs" 2>/dev/null |
  sort -nr | head -30
git -C "$backup" rev-parse --is-inside-work-tree 2>&1 || true
```

为了补足旧 backup 的 Git 归属和两次 Fast-WAM smoke 的实际终态，又执行：

```bash
backup=/root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40
grpo=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_013332-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke
ppo=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260719_120513-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke
git -C "$backup" rev-parse --show-toplevel 2>&1 || true
git -C "$backup" rev-parse --absolute-git-dir 2>&1 || true
test -e "$backup/.git" && ls -ld "$backup/.git" || echo NO_OWN_DOT_GIT
du -x --block-size=1 --max-depth=1 "$backup/logs/logs_his" 2>/dev/null |
  sort -nr | head -35
ppo_root="$backup/logs/20260713_124420-robotwin_adjust_bottle_ppo_openpi_a800_2gpu_baseline-step0-to-60/robotwin_ppo_openpi_a800_2gpu_baseline/checkpoints"
du -x --block-size=1 --max-depth=1 "$ppo_root" 2>/dev/null | sort -nr
find "$ppo_root" -xdev -type f \
  \( -name '*.distcp' -o -name 'full_weights.pt' -o -name '.metadata' \) \
  -printf '%s\t%p\n' 2>/dev/null | sort -nr
for run in "$grpo" "$ppo"; do
  sed -n '1,10p' "$run/command.txt"
  sed -n '1,160p' "$run/metrics.log"
  grep -Ei \
    'global_step|checkpoint|success|reward|loss|oom|nan|fatal|traceback|error|peak|finished|complete' \
    "$run/run_embodiment.log" | tail -80 || true
  cat "$run/resource_monitor/peak.txt" 2>/dev/null || true
done
```

现场结果：

| 目标 | 精确大小 | 里面实际是什么 | 结论 |
|---|---:|---|---|
| DSRL N20 smoke | 67,187,593,216 B / 62.57 GiB | `global_step_1/2` 各约 31.28 GiB；每步含两份约 8.09 GB actor local shard、两份约 8.07 GB target、两份约 617 MB replay、trainer state；0 视频、无 full weights | fresh→resume 证据；正式 DSRL DCP195 独立存在。确认不再亲自重载 DCP1→2 后可清 |
| Fast-WAM GRPO smoke | 28,912,521,216 B / 26.93 GiB | step1 两个约 14.45 GB DCP shard，外加小型 command/config/log/metrics/resource；0 视频、无 full weights | 完成 1 step、8 trajectories、success 1.0、DCP1；后有 formal，强候选 |
| Fast-WAM PPO smoke | 28,935,467,008 B / 26.95 GiB | step1 两个约 14.46 GB DCP shard，外加小型证据；0 视频、无 full weights | 完成 1 step、4 trajectories、success 0.25、actor/value update、DCP1；`explained_variance=nan` 来自小样本统计 warning，不是 loss NaN；后有 formal，强候选 |
| RLT Stage-1 smoke | 22,076,305,408 B / 20.56 GiB | step2：9.55 GB `full_weights.pt` + 6.263/6.262 GB DCP shards；无视频 | 两步 S1-A/reload 证据，已被正式 S1 step2000 endpoint 取代；最安全候选 |
| 旧 WAM/PPO backup | 119,126,163,456 B / 110.94 GiB | 四套 venv 共 58.93 GB；logs 60.16 GB，其中旧历史 PPO/GRPO 28.94 GB、7月13 PPO step20/40 20.81 GB、CPU-transport smoke 10.40 GB；101 个视频、多个旧 Motus/LaWAM run | 不建议整目录盲删；先决定是否放弃 step40 和旧环境，再做定向清理 |

旧 backup 的四套环境分别为 14,936,498,176、14,678,839,296、
14,657,064,960、14,657,060,864 bytes；大量 TensorFlow、Torch、Open3D 和
Flash-Attn 二进制重复。7月13主 PPO 的 DCP20/DCP40 各 10,402,308,096 bytes，
没有 step57 checkpoint；“step57”只是运行最后完成 step，真正可恢复点是 DCP40。

一个容易误判的现场项也已纠正：`git -C <backup> rev-parse --is-inside-work-tree`
返回 true，不代表 backup 自带 Git。`--show-toplevel` 实际返回
`/root/autodl-tmp`、git-dir 为 `/root/autodl-tmp/.git`，backup 自己没有 `.git`。

四个前置 smoke 合计 147,111,886,848 bytes，约 137.01 GiB。当前磁盘仍是
1.9 TiB 总、1.1 TiB 已用、807 GiB 可用、57%；本轮零删除。

## QAM-SSH-0001：SSH 真实稳定性、窄重试与命令文件

时间：2026-07-31 14:15–14:18 CST。

历史和本轮证据表明，已校验 host-key 的 Paramiko 密码路线总体可用；已知异常主要是
SeetaCloud 网关在**密码认证前**偶发 banner/EOF/timeout，不是持续密码错误。本轮连续
多次只读连接在 1–3 秒内完成。旧 helper 已经使用低层 `Transport` 和
`auth_password()`，但此前没有真正实现文档曾写到的 bounded retry/keepalive。

因此只修改 `local_scripts/remote_exec_autodl.py`：

```text
+ import socket
+ CONNECT_ATTEMPTS = 3
+ CONNECT_BACKOFF_SECONDS = (1.0, 3.0)
+ KEEPALIVE_SECONDS = 30
+ 只把 Transport/start_client 阶段的 banner/EOF/socket timeout/reset/SSHException
  纳入最多三次重试
+ host-key mismatch 仍直接 SystemExit
+ auth_password 在重试循环外只执行一次；失败立即 close/raise
+ auth success 后 transport.set_keepalive(30)
```

它不会重试或重放 `exec_command()`，所以已经发出的有副作用命令不会因网络错误被自动
执行第二次。密码仍只来自当前进程 `SEETA_SSH_PASSWORD`，不写入 helper、文档或仓库。

本轮两次尝试把含 `$file`/带空格日期格式的 shell 直接作为 PowerShell CLI 参数传给
helper，均在本机 argparse 前失败，服务器没有执行命令。原因是 PowerShell 提前展开和
拆词；修复为 UTF-8 command-file 后成功。后续规则是：只有最简单的单行命令可直接传；
包含变量、引号、管道、循环或多行结构统一用 `--command-file`。

修后实际身份探针：

```bash
hostname
pwd
id -u
TZ=Asia/Shanghai date '+%F %T %Z'
```

结果：

```text
autodl-container-nekaqbwt43-6ce5babb
/root
0
2026-07-31 14:18:04 CST
```

本次未触发重试，说明直连当时正常；后续 smoke 启动仍优先复用一次连接完成只读前检，
正式命令发出后不靠自动重放兜底。

## QAM-GIT-0003：QAM 文档、流水账与 SSH helper 云端固化

时间：2026-07-31 14:18–14:22 CST。根协调目录自身是无 commit/无 remote 的本地索引，
因此为了落实“每个连贯改动主动 push”，把本轮 QAM 专属材料和无凭据 helper 加到现有
QAM 云端分支，而不是只留在本机。

上传前对以下 11 个文件做精确密码、private-key 和 GitHub token 模式扫描，零命中：

```text
PROJECT_CONTEXT.md
HANDOFF.md
local_scripts/remote_exec_autodl.py
docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md
docs/rlinf-robotwin-pi0-qam/01_CONTEXT_AND_SOURCE_MAP.md
docs/rlinf-robotwin-pi0-qam/02_METHOD_AND_PORT_DECISION_GUIDE.md
docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md
docs/rlinf-robotwin-pi0-qam/evidence/qam_qonly_smoke_launch_20260731_v1.sh
docs/rlinf-robotwin-pi0-qam/evidence/qam_qonly_smoke_resolved_20260731_v1.yaml
docs/rlinf-robotwin-pi0-qam/evidence/qam_source_resolved_20260731_v1.yaml
docs/rlinf-robotwin-pi0-qam/evidence/qam_source_to_qonly_smoke_20260731_v1.diff
```

服务器先验证 branch、HEAD `7bc5f870...` 和 clean，再 `mkdir -p` 精确目标目录，并逐文件
SFTP `put`。服务器检查：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python -m py_compile \
  local_scripts/remote_exec_autodl.py
bash -n \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_qonly_smoke_launch_20260731_v1.sh
git diff --check
grep -R -n -F '<exact-password-redacted>' \
  PROJECT_CONTEXT.md HANDOFF.md local_scripts/remote_exec_autodl.py \
  docs/rlinf-robotwin-pi0-qam
```

全部通过。只 `git add` 上述 11 个路径，提交：

```text
3e7f26eb0cc38cc2f44e4145af480a71a2948262
chore(qam): publish implementation docs and ssh helper
```

默认直连 `timeout 15 git ls-remote` 再次返回 124；当时 HEAD clean、相对 upstream
ahead 1。随后只在子 shell `source /etc/network_turbo`，执行一次有界 push 并复核：

```text
remote: 7bc5f870... -> 3e7f26eb...
server HEAD: 3e7f26eb...
ahead/behind: 0/0
parent proxy before/after: 0/0
```

没有改 remote、Git HTTP 配置、shell 配置或历史，也没有启动 smoke。随后只再提交这段
自描述流水和对应 HANDOFF 更新；该最后 docs-only commit 本身由 Git 历史记录。

## QAM-PRE-0010：讨论前最终未启动快照

时间：2026-07-31 14:26:45 CST。执行：

```bash
repo=/root/autodl-tmp/RLinf_qam_pi0_robotwin
git -C "$repo" branch --show-current
git -C "$repo" rev-parse HEAD
git -C "$repo" status --short
git -C "$repo" rev-list --left-right --count '@{upstream}...HEAD'
pgrep -af 'train_embodied_agent.py|ray::|raylet|gcs_server' || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits
for path in \
  /root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1 \
  /root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1; do
  test -e "$path" && echo "PRESENT $path" || echo "ABSENT $path"
done
df -h /root/autodl-tmp
```

结果：branch `codex/qam-pi0-robotwin`、HEAD `3fc1acdede14...`、clean、
ahead/behind `0/0`；`pgrep` 只匹配到包含搜索字符串的本次只读 shell 自身，没有真实
training/Ray 进程；GPU0/1 均 0 MiB、0%；两个 smoke 目标目录均不存在；磁盘
1.9 TiB / 1.1 TiB / 807 GiB / 57%。正式 smoke 仍未启动。

## QAM-SMOKE-0001：授权后前检、运行代码锁与启动

时间：2026-07-31 14:56:43–14:57:35 CST。用户明确批准本批准包中的 fresh
`q_only` smoke；没有批准正式训练、resume smoke 或 `am_on` smoke。

第一条远端命令仍是 `hostname; pwd; id -u`，随后在同一只读前检中核对：

```text
host = autodl-container-nekaqbwt43-6ce5babb
uid = 0
branch = codex/qam-pi0-robotwin
HEAD = de99f969975564c8fec53de09c595b6670e3a416
tree = 3a495f06cc79a5f00a7543b21e4a6953039f7ea9
status = clean
ahead/behind = 0/0
GPU0/GPU1 = 0 MiB, 0%
training/Ray = none
run root/runtime root = absent
disk available = 866,263,560,192 B
cgroup anon/file = 311 MB / 183.84 GB
cgroup oom/oom_kill = 0/0
```

服务器 runtime HEAD 比最初实现 commit 多了已经审查过的 QAM 文档与无凭据 SSH
helper；`git diff 7bc5f870... de99f969...` 没有 runtime code/config 变化。旧 launcher
仍硬锁旧 HEAD/tree，因此只更新以下 provenance 常量，不改训练配置：

```text
expected_head: de99f969975564c8fec53de09c595b6670e3a416
expected_tree: 3a495f06cc79a5f00a7543b21e4a6953039f7ea9
launcher SHA-256:
92a76d47615e34c02f6c33d33fb84d194a03e5310aaf4d3cb0550aeed18d79b4
```

上传后执行 `bash -n`、源/resolved/diff/monitor SHA 和 run-root 不存在检查均通过。
启动命令为：

```bash
bash /root/autodl-tmp/qam_qonly_smoke_launch_20260731_v1.sh
```

launcher 内真正送入训练程序的完整命令、所有环境变量和覆盖项原样保存在
`qam_qonly_smoke_20260731_v1/exact_command.txt`；resolved config、穷尽 diff 和
provenance 位于同目录。本次精确预算仍为 2 GPU、2 env、1 outer cycle、2 episode、
最多 400 requested slots、预期 2–20 global macro、恰好 2 次 critic update、
0 fine update、0 eval 和 1 个 DCP。

## QAM-SMOKE-0002：监控权限故障、窄恢复与永久修复

训练 driver 正常启动，但原资源 monitor 立即返回：

```text
/root/autodl-tmp/qam_qonly_smoke_launch_20260731_v1.sh: line 181:
/root/autodl-tmp/qam_resource_monitor_20260731_v1.sh: Permission denied
monitor_exit_code = 126
```

原因是 SFTP 上传后的脚本 mode 为 `644`，launcher 直接把路径当可执行文件调用。
算法、配置和 driver 均未受影响。15:00:27 CST 在训练仍运行时执行唯一窄恢复：

```bash
bash /root/autodl-tmp/qam_resource_monitor_20260731_v1.sh \
  82368 \
  /root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime/resources.csv \
  2 \
  >/root/autodl-tmp/experiment_exports/qam_qonly_smoke_20260731_v1/runtime/monitor_recovered.log \
  2>&1 &
```

其中 `82368` 是 driver PID；恢复 monitor 自然跟随 driver 退出。launcher 的永久修复仅为：

```diff
-"$monitor" "$driver_pid" "$runtime_root/resources.csv" 2
+bash "$monitor" "$driver_pid" "$runtime_root/resources.csv" 2
```

15:11:27 CST 又用一个 3 秒 `sleep` 假 driver 做窄复测：

```bash
sleep 3 &
probe_pid=$!
bash /root/autodl-tmp/qam_resource_monitor_20260731_v1.sh \
  "$probe_pid" /root/autodl-tmp/qam_resource_monitor_retest_20260731.csv 1
```

结果 `MONITOR_STATUS=0`、CSV 共 5 行（header + 4 sample）、GPU/内存/磁盘字段完整；
临时 CSV 随后删除。无需为这一 launcher 权限问题重跑 11 GB smoke。

## QAM-SMOKE-0003：fresh q_only 结果、checkpoint 与资源

训练自然完成，driver exit 0；启动到完成约 3 分 14 秒，其中模型/环境初始化约
2 分 31 秒，唯一训练 cycle 为 42.753 秒：

| 项 | 实测 |
|---|---:|
| rollout | 2 trajectories；每条 200 slots；18.872 s |
| 环境结果 | return 0；reward 0；success 0 |
| replay | global 20；每 rank 10；每 rank 10 个不同 planned chunk |
| critic | exactly 2 updates；loss 23.505；pre-clip grad norm 286.3 |
| Q | mean -1.992；10-head std 3.610；TD target mean -0.822 |
| fine/F1 | 0 update；policy version 0 |
| UTD credit | 剩余 18；符合 20 inserts - 2 updates |
| 保存 | `global_step_1` DCP + 2 sidecar + 2 replay |

两次 update 后数值全 finite；随机冷启动 Q 的 loss/未裁剪 grad 较大，本轮只证明调用链，
不作为收敛结论。配置中的 critic grad clip 为 1.0。日志中的 Curobo/pytorch3d traceback
是 RoboTwin planner 的既有可选 Curobo 导入提示；TOPP/SAPIEN 实际 rollout、Q update
和保存均继续完成，未安装新依赖或改环境。

checkpoint 精确目录：

```text
/root/autodl-tmp/experiments/qam_qonly_smoke_20260731_v1/
  robotwin_adjust_bottle_qam_qonly_smoke_20260731_v1/
  checkpoints/global_step_1
```

主要文件：

```text
4,988,130,752 B  __1_0.distcp
4,987,882,268 B  __0_0.distcp
  822,350,722 B  qam_components/rank_0.pt
  822,350,722 B  qam_components/rank_1.pt
    7,283,083 B  qam_components/replay_rank_0.pt
    7,283,083 B  qam_components/replay_rank_1.pt
    1,112,122 B  .metadata
```

run 共 11,636,440,837 B，另有两个 16,304/15,036 B train video、metrics 和
TensorBoard 小文件。没有 full-weight 导出。

服务器 validator 逐 tensor/transition 检查：

- 2 个 sidecar/replay 均 complete，900 个 sidecar tensor 和 replay tensor 全 finite；
- 对应 critic、target、optimizer 跨 rank digest 相同；
- 每 rank 的 10 个 Q 首层 digest 均不同，确为 10 套独立 Q；
- online critic 与 target 已有非零 EMA 差：
  `max_abs=0.0005989075`、`L2=3.26567`；
- counters 两 rank 完全一致：`q_only`、critic 2、fine 0、version 0、
  local/global inserts 10/20、pending 18；
- action normalized 精确在 `[-1,1]`；两 rank 9/9 相邻链完整；
- 每 rank 最后一条为 generic `other_truncated`，故 9 条可 bootstrap、1 条终止。

`runner_global_step=0` 与目录 `global_step_1` 不是丢步：runner 在 cycle 开头把零基
step 0 传给 worker，cycle 完成后自增到 1 再保存；resume 后 runner 从目录恢复为 1，
下一 cycle 开头会把 worker 覆盖为 1。

恢复 monitor 仅覆盖 rollout/update/save/teardown 的最后 22 秒，不能声称包含初始化峰值：

```text
samples = 11
GPU peak = 23,486 / 23,677 MiB
GPU util peak = 100% / 100%
cgroup anon peak = 36,866,805,760 B (34.34 GiB)
QAM process RSS peak = 34,575,536 KiB (32.98 GiB)
cgroup current peak = 235,295,363,072 B，主要为 file cache
OOM/OOM-kill delta = 0/0
captured min disk available = 851,218,075,648 B
```

完整小型 runtime evidence 已下载到新增目录
`evidence/qam_qonly_smoke_20260731_v1/`，含 driver log、resolved config、精确命令、
资源 CSV/summary、sidecar validation 和 artifact inventory；不复制 11 GB
checkpoint。服务器压缩包 SHA-256 为
`27969cc30356825857a2b6e58c306cb9d64c0cd273640159c8fc73db855bb746`。

本 smoke 的结论边界：真实
`RoboTwin -> macro replay -> next-action fine ODE -> 10-Q -> target EMA -> DCP`
已通；没有验证 hardened resume、AM、效果或涨点。

## QAM-CLEANUP-0001：四组旧 smoke checkpoint 定向删除

时间：2026-07-31 15:06:42–15:06:45 CST。用户只授权删除四组已经确认是 smoke
的最大 checkpoint/DCP；为避免和 QAM smoke 的保存 I/O 叠加，实际删除等 smoke
自然结束后才执行。完整可执行脚本新增为
`qam_delete_old_smoke_checkpoints_20260731.sh`，完整 stdout/stderr 为
`qam_old_smoke_checkpoint_cleanup_20260731.txt`。

脚本先对每个 root/target 做 `readlink -f` 精确相等、目录存在、非 symlink、
target 必须位于含 `smoke` 的 root 下且以 `checkpoints` 结尾的检查；四项全部通过后才逐项：

```bash
rm -rf -- "$target"
test ! -e "$target"
```

删除量：

| smoke | 仅删除的 checkpoints | bytes |
|---|---|---:|
| DSRL N20 fresh→resume | `.../robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_smoke/checkpoints` | 67,186,950,144 |
| Fast-WAM GRPO step1 | `.../robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke/checkpoints` | 28,912,402,432 |
| Fast-WAM PPO step1 | `.../robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu_smoke/checkpoints` | 28,935,348,224 |
| RLT Stage-1 S1-A | `.../robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1/checkpoints` | 22,076,280,832 |
| 合计 |  | 147,110,981,632 B / 137.01 GiB |

四个 smoke root 及 command/config/log/metrics/TensorBoard/resource/小型文档均保留；
DSRL formal 和 RLT Stage-1 formal root 均再次验证存在。磁盘 available 从
854,626,586,624 B 增至 1,001,737,560,064 B，使用率 57% 降到 50%。被删的四组
smoke checkpoint 不可恢复，只能重跑；所有 formal DCP 未动。

## QAM-DISK-0003：旧 WAM/PPO backup 的实验归属

`RLinf_wamppo_backup_20260714_step57_lastdcp40` 是 7 月 13–14 日迁移前的标准
π0 PPO `adjust_bottle` baseline 快照，不是 QAM/Fast-WAM。它使用 2×A800、
3 相机、14D、H=50、PPO/GAE（$\gamma=0.99$、$\lambda=0.95$、clip 0.2、
update epoch 2），16 env × rollout epoch 16，即每 outer step 约 256 episode。

目标 60 step，完整完成到 step 57；step 58 的 256 个 rollout 已收完，但 update 前
Ray 因 host memory 230.89/240 GiB 超过 95% 阈值杀掉 EnvWorker。两个 env worker
当时约 84.44/82.38 GiB，属于 SAPIEN/renderer native memory 增长，不是 NaN。
57 个完整 step 约 23.5 小时；没有固定 seed eval，不能称 step57 为 best 或证明涨点。

真正可恢复的只有 DCP20/DCP40，各 10,402,308,096 B；DCP40 比最后完整训练落后 17
次 update，且不含环境状态与 step58 rollout。整个 110.94 GiB backup 还混有四套
重复 venv（58.93 GB）、其他 PPO/GRPO/Motus/LaWAM logs 和 101 个视频，所以本轮没有
对它执行任何删除。若以后明确放弃旧协议，可先保留小型证据 capsule，再优先考虑被
DCP40 支配的 DCP20 与重复 venv；DCP40 是否放弃仍需用户另行决定。

## QAM-FORMAL-0001：18 小时阶段预算的当前结论

正式训练尚未获批，也尚未启动。官方 QAM 没有我们的三段：

```text
官方：1,000,000 offline joint updates
   -> online 500,000 primitive steps
   -> 前 5,000 primitive 只 collect
   -> 后 495,000 按 UTD=1 joint critic + FM + AM
```

因此官方 online 约为 1% collect、99% joint，`q_only=0`；我们的
`collect -> q_only -> am_on` 是移除大规模 offline buffer 后，为冷启动视觉 Q 增加的
在线安全适配，不能声称阶段比例复刻官方。

首个 18 小时候选应按 transition/update 和证据门定，不先按 runner step 猜：

| 阶段 | 候选目标 | 达标条件 | 时间上限/转移 |
|---|---:|---|---|
| collect | 512 global valid macro；0 update | replay 真实结果覆盖足够启动 | 暂留最多 3 h，提前完成则时间给 AM |
| q_only | 先看 256，目标 512–1,024 critic updates | Q/梯度 finite、非零、对动作敏感，真实 `+dQ/da` 不劣于 `-dQ/da` | 最多约 4 h；不达标则整晚只能叫 Q pilot |
| 诊断/阶段 DCP/eval | 约 1 h 总壳 | 明示 gate | 不能省略后静默开 AM |
| am_on | 吃掉剩余约 10 h | fine version 增长、AM/adjoint finite、固定 seed 不退化并争取涨点 | 目标至少 512、争取不低于 1,024 fine updates |

宏 transition 目标比例约 `512 : 512–1024 : >=1024`，即约 `1 : 1–2 : >=2`；
这是端口稳定性预算，不是官方比例。fresh smoke 实测每 cycle 为 20 global macro，
故 collect 512 理论约 26 cycles；但 smoke 是 batch 1/rank，只能说明一 cycle 的
真实生命周期，不能外推正式 batch 32/rank 的 Q/AM update 吞吐。

当前两卡并行保持 2×A800、2 env（每 actor rank 1 env）、global/local batch 64/32、
每 rank 完整逻辑 10-Q、F1 FSDP full-shard、UTD1、每 cycle update cap 32。它按
π0 计算图设计，不按官方小 MLP 的 batch256/作业拓扑照搬；现有显存证据宽裕，首要未知
更可能是 batch32 的 Q/AM 计算时间、RoboTwin rollout 与 host memory。

正式阶段不能忽略 resume：当前 phase 是显式 config，需
`collect DCP -> q_only resume -> 诊断 -> am_on resume` 才能保留 replay、Q/target、
optimizer 和 counters。fresh q-only smoke 可以不测 resume，但正式三段必须先补
rank-local RNG/跨 rank completion，再做单独获批的 fresh→resume smoke。还需一个很短
的 production-batch q-only throughput 点和一个 AM smoke，才能把 18 小时时间壳换算成
可信 cycles。另一个正式前语义点是：当前 RoboTwin 只暴露 generic truncation，200-step
样本保守地不 bootstrap；若不补可信 timeout/final-observation 类型，Q 会对超时样本偏低。

## QAM-PRE-0011：RoboTwin truncation 与 true-final observation 现场定性

时间：2026-07-31 15:34–15:36 CST。fresh smoke 显示每 rank 最后一条 generic
truncation 被 v1 保守当作 terminal；正式前需判断能否安全 bootstrap。

第一次只读搜索错误地假设服务器有 `rg`，并对
`/root/autodl-tmp/RoboTwin_RLinf` 执行 `git status`。该目录实际受上层
`/root/autodl-tmp/.git` 管理，因而输出了大量无关 sibling 状态；随后
`rg: command not found`、远端 exit127。没有写文件、没有改进程。修复是按本机规则在
`rg` 不存在时改用限定目录和 `--include='*.py'` 的 `grep`，并停止把该子目录的上层
Git 状态当作 RoboTwin 自身状态。

成功的只读链依次检查：

```bash
grep -R -n -E \
  'def chunk_step|truncat|terminated|_elapsed_steps|max_episode_steps|final_observation' \
  --include='*.py' \
  /root/autodl-tmp/RoboTwin_RLinf/envs \
  /root/autodl-tmp/RLinf_qam_pi0_robotwin/rlinf/envs/robotwin \
  /root/autodl-tmp/RLinf_qam_pi0_robotwin/rlinf/workers/env

sed -n '1735,1810p;1945,1990p;2295,2340p;2440,2490p' \
  /root/autodl-tmp/RoboTwin_RLinf/envs/_base_task.py
sed -n '280,435p' \
  /root/autodl-tmp/RLinf_qam_pi0_robotwin/rlinf/envs/robotwin/robotwin_env.py
sed -n '450,525p;1115,1150p' \
  /root/autodl-tmp/RLinf_qam_pi0_robotwin/rlinf/workers/env/env_worker.py
```

证据闭环：

1. RoboTwin sparse `gen_sparse_reward_data()` 只在成功时设
   `termination=1`，只在 `take_action_cnt >= step_lim` 时设 `truncation=1`；
2. RLinf `RoboTwinEnv.chunk_step()` 还显式执行
   `_elapsed_steps += chunk_width` 和
   `truncation |= (_elapsed_steps >= max_episode_steps)`；
3. 当前 train resolved config 是 `auto_reset=false`、`ignore_terminations=false`；
4. EnvWorker 在 `auto_reset=false` 时把 `env_output.obs`，也就是本 query 执行后的
   `obs_list[-1]`，直接保存为 transition `next_obs`；不会混入 reset observation；
5. 若 success 与 time limit 同槽，success termination 仍应优先且不 bootstrap。

因此对当前锁定的 RoboTwin sparse route，`truncated && !terminated` 可以可靠标为
`time_limit_truncated`，并用保存的 true query-final observation bootstrap；无需继续
保守地降级成 `other_truncated`。这不是根据 episode 长度猜测，而是由环境源代码的
truncation 定义和 `auto_reset=false` 的 payload 路线共同证明。补丁应只修改 QAM
分类与现有聚焦测试，不改通用 EnvWorker/RoboTwin 行为；新的 fresh→resume smoke
需验证最后一条 replay 的 `time_limit_truncated=1`、`next_state_valid=1`、
`bootstrap_mask=1`。

## QAM-RESUME-0001：exact-resume 窄加固与两次静态 P1 修正

时间：2026-07-31 15:39–16:05 CST。授权：继续 QAM 实现与服务器前置测试；没有批准
第二次 smoke、resume smoke、`am_on` 或正式训练。

只修改四个已有文件：

```text
rlinf/data/qam_transition_replay.py
rlinf/workers/actor/fsdp_qam_policy_worker.py
tests/embodiment/test_robotwin_qam_contract.py
tests/workers/test_qam_worker_helpers.py
```

最终职责限定为：

1. replay/sidecar 使用临时文件、flush/fsync 和原子替换；
2. rank 0 广播一个 snapshot ID，replay、sidecar、completion manifest 绑定同一代；
3. 覆盖同一路径前先删除旧 completion manifest；若新 DCP/QAM sidecar 中途失败，旧
   manifest 不会把“新 F1 + 旧 QAM state”误报为完整；
4. 所有 rank 的 local write 和 load preflight 先汇总；compact signature 比较
   phase、contract、prefix、critic shape、runner/fine/critic/global-insert/UTD counters，
   防止 resume 后各 rank 进入不同 update 次数而 collective 死锁；
5. 保存并在 load 最后恢复每 rank 的 Python/NumPy/Torch process RNG；replay 继续恢复
   自己的 sampler generator；
6. fresh `q_only` 的 warm-up rows 不形成旧 update debt，但 crossing batch 中超过
   warm-up 阈值的 rows 正常获得 UTD credit；
7. 仅对 QAM 分类函数应用 `QAM-PRE-0011` 结论：
   `truncated && !terminated` 是可 bootstrap 的 time limit；success 同槽仍优先且不
   bootstrap。

第一次静态审查发现 warm-up crossing P1：若一次从 0 插入 20 条、warm-up=2，早期候选
把 anchor 直接设成 20，导致 0 update。窄修为按本批前后 global total 计算越过阈值的
增量；因此该例获得 18 条 credit，smoke cap=2 时仍执行 2 次 update。正式
`500 -> 520`、warm-up=512 时只给后 8 条 credit。

第二次静态审查发现两个 checkpoint P1：

- compact signature 原先未比较 global insert/UTD/版本 counters，可能让两 rank resume
  后执行不同次数的 collective update；
- 同一路径重存若在新 QAM sidecar 前失败，旧 complete manifest 可能仍存在。

两处均按上面的第 3、4 点窄修。为避免实现臃肿，随后删除 manifest 中重复的 rank/file-name
列表和 15 字段展开状态，改为确定性文件名 + 一个 compact signature。没有 checksum
大文件、incomplete marker、路径 step regex、旧 checkpoint 迁移器或额外测试框架。
机械格式化后的最终实现提交相对基线为 4 files、`+475/-79`；两轮最终静态审查均为
P0=0、P1=0。

正式 v1 阶段也据此简化：fresh `q_only` 在同一进程内先收 512 条 warm-up macro，阈值前
不更新；随后才按新 rows、UTD=1 更新 Q。因此默认不再需要
`collect DCP -> q_only resume`，只保留诊断通过后的 `q_only -> am_on` 明示 resume。
独立 `collect` phase 仍是可选的纯收数入口。

## QAM-PRE-0012：现场刷新、代码同步与服务器集中回归

15:44 的只读前检首先误用了 Windows Store 的 `python.exe` alias，报
“系统无法访问此文件”；这一步没有建立 SSH 连接，也没有任何服务器变化。修复为使用
Codex bundled Python：

```text
C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\
dependencies\python\python.exe
```

随后通过新增的
`evidence/qam_resume_hardening_preflight_20260731.sh` 执行完整只读探针。远端 payload：

```bash
hostname
pwd
id -u
date '+%F %T %Z'
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin branch --show-current
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin rev-parse HEAD
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin status --short
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin \
  rev-list --left-right --count '@{upstream}...HEAD'
ps -eo pid,etimes,%cpu,%mem,rss,args --sort=-rss |
  grep -E 'rlt_stage2|qam_|ray::|train_embodied_agent' |
  grep -v grep | head -20 || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader
free -h
df -h /root/autodl-tmp
```

结果：

```text
host/root uid     = autodl-container-nekaqbwt43-6ce5babb / 0
time              = 2026-07-31 15:44:01 CST
branch/HEAD       = codex/qam-pi0-robotwin
                    de99f969975564c8fec53de09c595b6670e3a416
dirty/ahead-behind= empty / 0 0
QAM/RLT/Ray train = none
GPU0/GPU1         = 0 MiB, 0%
host available    = 977 GiB
disk              = 933 GiB available, 50% used
```

代码先用以下远端命令检查再应用本机 QAM sparse worktree diff：

```bash
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin apply --check -
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin apply -
```

静态 P1 修正后，只用 SFTP 覆盖同一独立 worktree 的两个精确文件：

```text
.qam-impl-worktree/rlinf/workers/actor/fsdp_qam_policy_worker.py
 -> /root/autodl-tmp/RLinf_qam_pi0_robotwin/rlinf/workers/actor/fsdp_qam_policy_worker.py
.qam-impl-worktree/tests/workers/test_qam_worker_helpers.py
 -> /root/autodl-tmp/RLinf_qam_pi0_robotwin/tests/workers/test_qam_worker_helpers.py
```

服务器测试由新增的
`evidence/qam_resume_hardening_server_tests_20260731.sh` 固定，完整命令为：

```bash
cd /root/autodl-tmp/RLinf_qam_pi0_robotwin
export PYTHONPATH=$PWD
git branch --show-current
git rev-parse HEAD
git status --short
git diff --check
/root/autodl-tmp/RLinf/.venv/bin/ruff check <4 changed files>
/root/autodl-tmp/RLinf/.venv/bin/ruff format --check <4 changed files>
/root/autodl-tmp/RLinf/.venv/bin/python -m py_compile <4 changed files>
/root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  tests/algorithms/qam/test_core.py \
  tests/algorithms/qam/test_official_fixture.py \
  tests/embodiment/test_qam_openpi_adapter.py \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py
pgrep -af 'train_embodied_agent|qam_|rlt_stage2|raylet|gcs_server' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

第一次执行在 `ruff format --check` 停下：worker 有一处会被重排，逻辑测试尚未开始。
执行：

```bash
/root/autodl-tmp/RLinf/.venv/bin/ruff format \
  /root/autodl-tmp/RLinf_qam_pi0_robotwin/rlinf/workers/actor/fsdp_qam_policy_worker.py
```

结果 `1 file reformatted`，并把机械格式化后的文件同步回本机副本。随后一次试图“重定向、
cat、转交 exit code”的 PowerShell→remote 双层引号拼接失败，远端报
`unexpected EOF while looking for matching '"'`；没有执行测试、没有写训练产物。解决是
停止复杂 wrapper，直接执行：

```bash
bash /root/autodl-tmp/qam_resume_hardening_server_tests_20260731.sh
```

最终结果：

```text
git diff --check  = pass
ruff check        = pass
ruff format-check = 4 files already formatted
py_compile        = pass
pytest            = 39 passed, 3 dependency deprecation warnings, 8.35s
QAM/RLT/Ray train = none（pgrep 只命中当前测试 shell）
GPU0/GPU1         = 4 MiB / 4 MiB, utilization 0%
```

本批没有启动第二次 smoke。唯一正式 fresh `q_only` smoke 仍是
`QAM-SMOKE-0001` 至 `0003` 的那一次；3 秒 monitor 假 driver 只验证脚本调用权限，也
不是 smoke。旧 smoke checkpoint 清理、测试和文档整理均未触碰 DSRL/RLT worktree、
formal checkpoint 或 shared π0 venv。

同一路径 launcher 有两个可区分版本：实际执行 smoke 的 SHA-256 是
`92a76d47615e34c02f6c33d33fb84d194a03e5310aaf4d3cb0550aeed18d79b4`；事后把 monitor
调用改为显式 `bash` 的未来运行版 SHA-256 是
`af554628da63e739c29c72b5a6574ade255ee25ab2866ff107b02361b1207572`。后者没有被用来
重跑本次 smoke。

## QAM-FORMAL-0002：对 18 小时阶段入口的简化修正

`QAM-FORMAL-0001` 中“必须 collect DCP→q_only resume”和“generic truncation 不
bootstrap”已被后续事实替代：

```text
fresh q_only process
  -> 前 512 global macro：只 collect，0 update
  -> 阈值后的新 macro：UTD=1 q_only
  -> Q 动作梯度诊断门
  -> 明示 q_only→am_on resume（仍需单独批准）
```

因此 18 小时时间壳仍是 collect 最多约 3 h、q_only 最多约 4 h、诊断/保存约 1 h、
通过门后把余下约 10 h 给 am_on；但第一、二段可在同一 fresh process 内连续完成。
锁定 sparse route 的 pure truncation 使用 true query-final observation bootstrap。
正式 cycles 仍不能由 batch1 smoke 推断，必须等获批的 production-batch 短吞吐点。

## QAM-GIT-0004：exact-resume 实现提交与推送

为把实现与运行附件分开，先只 stage 四个已经通过服务器回归的代码/测试文件：

```bash
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin diff --check -- <4 files>
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin add -- <4 files>
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin diff --cached --check
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin diff --cached --stat
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin \
  commit -m 'feat(qam): harden macro replay resume'
```

第一次 PowerShell inline wrapper 因 `$repo`/引号被本地解释，helper 在参数解析阶段报
`unrecognized arguments`；没有建立 SSH 连接，也没有 stage/commit。改为 PowerShell
single-quoted here-string 后成功：

```text
commit = 851db175fb8e9743585bbbdcd90298741fa910e0
stat   = 4 files changed, 475 insertions(+), 79 deletions(-)
```

推送前执行最短网络检查：

```bash
env | grep -iE '^(http|https|all)_proxy=' || true
git config --get http.version || printf 'DEFAULT\n'
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 10 \
  -w 'main code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com
timeout 15 git ls-remote --heads personal codex/qam-pi0-robotwin
git rev-list --left-right --count '@{upstream}...HEAD'
GIT_TERMINAL_PROMPT=0 timeout 60 \
  git push personal HEAD:codex/qam-pi0-robotwin
git rev-list --left-right --count '@{upstream}...HEAD'
git ls-remote --heads personal codex/qam-pi0-robotwin
```

结果：无 proxy、Git HTTP 默认、GitHub main HTTP 200（connect 0.118 s、total
0.991 s）、`ls-remote` 成功；push 前 `0/1`，push 后 `0/0`，remote head 精确为
`851db175fb8e9743585bbbdcd90298741fa910e0`。本次直连成功，没有启用学术加速，也没有
修改持久 Git/proxy 配置。

## QAM-GIT-0005：运行附件与文档提交前检查

文档包只包含 HANDOFF/SSOT/账本、两个精确服务器脚本、定向清理脚本与 stdout，以及
116 KiB 的 smoke runtime evidence；没有 checkpoint、视频或模型。对该 allowlist 搜索
当前密码和 private-key header，结果为空。

第一次对全部 staged 文件执行 `git diff --cached --check`，因不可变
`driver.log` 保留上游日志本身的行尾空格而失败；这不是代码/Markdown 生成错误。没有为了
通过 Git whitespace gate 改写原始运行证据。修复是保持原始 log 字节不变，只对
HANDOFF、SSOT、账本、shell/JSON/YAML/CSV 等非原始日志执行 whitespace check，再提交
完整附件。

## QAM-FORMAL-0003：连续阶段方案获批与实施起点

时间：2026-07-31 16:43 CST 起。用户明确授权启动正式训练，并要求只确认健康启动后
退出，不持续监控。获批的方法级入口为：

```text
fresh one-process formal
  -> 512 global macro warm-up；0 update
  -> exactly 512 critic-only updates；继续按 UTD=1 收集
  -> 第 513 个 logical update 起 joint critic + AM
  -> outcome coverage/action sensitivity 只记录，不阻塞切换
  -> NaN/Inf/OOM/fatal 仍 fail-fast
  -> inv_temp=1.0
```

`inv_temp=1.0` 是官方 Plain-QAM 正式任务使用的最低档；官方源码 fallback 是 0.3，
而此前 0.5 只是无直接来源的插值。首跑选 1.0 的同时保留 target-Q mean、10-Q、
fine grad clip 1.0 和 fine LR 2e-5。

本轮开始前的本机镜像事实：

```text
local QAM HEAD = c32d044b
local dirty    = exact-resume 四文件，475 insertions / 79 deletions
server/cloud expected HEAD from prior ledger = 851db175
```

这些 dirty 修改是上一批已在服务器提交和推送的 exact-resume 内容，不是本轮未识别的
用户改动。下一步先通过已验证 Paramiko helper 做身份、server HEAD/tree、进程/GPU/RAM/
磁盘只读探针；再让本机镜像与 `851db175...` 对齐，之后才实施自动 AM 边界。密码只在
当前 helper 进程环境中注入，不写入命令、文件或本账本。

17:11 CST 身份与现场只读探针：

```bash
hostname; pwd; id -u; date '+%F %T %Z'
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin status --short --branch
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin log -8 --oneline --decorate
pgrep -af 'python.*(main|train)|ray::|qam|rlt' || true
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
df -h /root/autodl-tmp
cat /sys/fs/cgroup/memory.{current,max}
```

结果：

```text
host/uid       = autodl-container-nekaqbwt43-6ce5babb / 0
server branch  = codex/qam-pi0-robotwin
server HEAD    = ced8672f322187b71939bd2859842619c6284d05
server tree    = clean
QAM/RLT/Ray    = none
GPU0/GPU1      = 0 MiB / 0 MiB；utilization 0%
MemAvailable   = 1,023,924,948 kB
cgroup current/max = 166,121,512,960 / 257,698,037,760 B
disk           = 1.9T total；912G used；933G free；50%
```

本机第一次调用 helper 使用 PATH 中不可执行的 `python.exe`，在建立 SSH 连接前报
“系统无法访问此文件”；改用 Codex workspace dependency 的 Python，不安装任何依赖，
随后同一 Paramiko Transport/password/host-key 路线成功。

服务器 HEAD 已包含上一批完整序列：
`7bc5f870 implementation -> 851db175 exact resume -> c2800669 smoke evidence ->
e5323c91 handoff -> ced8672f ledger fix`。本机四个 exact-resume 文件与服务器逐文件
SHA-256 完全一致，因此可在其上做窄增量，不覆盖未知改动。

### 自动 AM 边界代码修改

本轮只改四个服务器代码文件：

| 文件 | 修改 |
|---|---|
| `rlinf/config.py` | 新增正整数 `q_only_updates_before_am` 合同；移除静态 `am_evidence_passed` 阻塞 |
| `fsdp_qam_policy_worker.py` | 用 pre-update `critic_updates` 判定；1–512 只 Q，第 513 次起 joint；记录 configured/effective phase；checkpoint schema 2 固定该阈值 |
| QAM source YAML | 新增 `q_only_updates_before_am: 512`，移除旧 evidence gate；仍保持 launch-closed `phase=collect, inv_temp=0` |
| worker helper tests | 增加 511/512 边界、负计数和无人工 evidence gate 的 config 合同；checkpoint fixture 携带阈值 |

四文件先 SFTP 到
`/root/autodl-tmp/qam_formal_patch_20260731/`，逐文件 SHA-256 验证后才 `install`
覆盖 repo 目标。第一次 inline PowerShell 把远程 `$repo` 和引号在本机拆开，helper 在
参数解析阶段报 `unrecognized arguments`，没有建立 SSH 连接、没有服务器改动。随后改用
新增的精确脚本 `qam_formal_patch_apply_20260731.sh`。该脚本第一版误把八位 HEAD
补成了未经验证的完整 SHA，远程 `test` 立即失败、仍无 repo 改动；用
`git rev-parse HEAD` 取得真实完整 SHA 后修正并复跑。

最终应用结果：

```text
4 staged-upload SHA-256 = OK
server diff --check     = pass
server diff stat        = 4 files changed, 86 insertions(+), 8 deletions(-)
```

### 自动阶段切换的服务器验证

新增并执行
[`qam_formal_schedule_server_tests_20260731.sh`](qam_formal_schedule_server_tests_20260731.sh)。
完整远端入口为：

```bash
bash /root/autodl-tmp/qam_formal_schedule_server_tests_20260731.sh
```

脚本内依次执行：

```bash
cd /root/autodl-tmp/RLinf_qam_pi0_robotwin
export PYTHONPATH="$PWD:/root/autodl-tmp/RoboTwin_RLinf"
git diff --check
/root/autodl-tmp/RLinf/.venv/bin/python -m compileall -q \
  rlinf/config.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py
/root/autodl-tmp/RLinf/.venv/bin/python -m ruff check \
  rlinf/config.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/workers/test_qam_worker_helpers.py
/root/autodl-tmp/RLinf/.venv/bin/python -m ruff format --check \
  rlinf/config.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/workers/test_qam_worker_helpers.py
/root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  tests/workers/test_qam_worker_helpers.py \
  tests/algorithms/qam/test_core.py \
  tests/algorithms/qam/test_official_fixture.py
```

结果：

```text
git diff --check = pass
compileall       = pass
ruff check       = pass
ruff format      = pass
pytest           = 30 passed, 1 dependency deprecation warning
```

随后同一脚本分别 compose launch-closed source 与 formal overrides，运行
`validate_cfg(OmegaConf.load(...))`，机器断言输出：

```text
QAM_FORMAL_CONFIG_OK 500 am_on 512 512 1.0 64 32
```

该 validator 沿用项目 `Cluster()`，因此短暂创建 local Ray；脚本结束后再次 `pgrep`
确认 raylet/gcs/training 均已退出。这里没有启动 RoboTwin rollout，不是第二次 smoke。

### resolved config 与代码提交

formal 附件：

| 文件 | SHA-256 |
|---|---|
| QAM source YAML | `0aca13bfd8b24c4f08dc867599c9cedc55f0be7c379f822a661ab71a626b112d` |
| [`qam_source_resolved_20260731_formal_v1.yaml`](qam_source_resolved_20260731_formal_v1.yaml) | `45bea3edcd28d9b7d8475ce66fe7ef1cf9533dcbc795a78aa94fd210ad4310b4` |
| [`qam_formal_resolved_20260731_v1.yaml`](qam_formal_resolved_20260731_v1.yaml) | `c26133cd7462d7c30d5779b9a6bba224209ec0781ea003fb99cc1d74e7644915` |
| [`qam_source_to_formal_20260731_v1.diff`](qam_source_to_formal_20260731_v1.diff) | `851dd01876ce4cfbc4893981a360eba9c11fd02bfed504791da91fcf3fb0a07c` |
| [`qam_formal_launch_20260731_v1.sh`](qam_formal_launch_20260731_v1.sh) | `c6c772ea0624a6152896a5704f02ae84f24d00cf2a21d6f0eb2a3206062c9901` |

source→formal 的算法/预算变化只有：

```text
max_steps                 0 -> 500
phase               collect -> am_on
inv_temp                0.0 -> 1.0
save_full_model_weights default -> false
run/experiment/video paths -> 唯一 formal 路径
```

`warmup_global_inserts=512`、`q_only_updates_before_am=512`、UTD1、
batch64/32、2 GPU/2 env、N20、K10、10-Q、F1/C1/M2 已在 source 中一致。

用
[`qam_formal_code_commit_push_20260731.sh`](qam_formal_code_commit_push_20260731.sh)
固定 stage allowlist、commit 和有界网络流程。核心命令：

```bash
git add -- \
  examples/embodiment/config/robotwin_adjust_bottle_qam_openpi.yaml \
  rlinf/config.py \
  rlinf/workers/actor/fsdp_qam_policy_worker.py \
  tests/workers/test_qam_worker_helpers.py
git diff --cached --check
git commit -m 'feat(qam): schedule in-process AM activation'
```

结果：

```text
commit = 4a15699e10971e306ed756dcbbf8aa65632553d5
tree   = d86082209c07866c13fef7e9051355cf54e6511c
```

Git 网络事件：默认直连的 GitHub homepage 返回 HTTP 200，但在 10 秒总时限处超时；
紧接的 direct `git ls-remote` 也在 15 秒内超时。这证明当时 smart-HTTP 链路不稳定，
不是 branch/commit 错误，也没有循环盲试。按既有授权只在一个子 shell 内执行：

```bash
source /etc/network_turbo
GIT_TERMINAL_PROMPT=0 timeout 60 \
  git push personal HEAD:codex/qam-pi0-robotwin
git rev-list --left-right --count '@{upstream}...HEAD'
git ls-remote --heads personal codex/qam-pi0-robotwin
```

push 成功；本地/上游 `0/0`，remote branch 精确指向 `4a15699e...`。子 shell 退出后未保留
proxy、未改 remote 或 Git config。

## QAM-FORMAL-0004：正式启动与一次性健康门

时间：2026-07-31 17:32–17:35 CST。启动前 launcher 自行 fail-closed 检查：

```text
branch/HEAD/tree = codex/qam-pi0-robotwin /
                   4a15699e10971e306ed756dcbbf8aa65632553d5 /
                   d86082209c07866c13fef7e9051355cf54e6511c
worktree         = clean
existing Ray/train = none
GPU0/GPU1        = idle
disk available   = 1,001,735,888,896 B，约 933 GiB
run/runtime root = both absent
all pinned hashes = pass
```

正式 detached 启动命令：

```bash
nohup bash /root/autodl-tmp/qam_formal_launch_20260731_v1.sh \
  >/root/autodl-tmp/qam_formal_supervisor_20260731_v1.log 2>&1 &
echo $! >/root/autodl-tmp/qam_formal_supervisor_20260731_v1.pid
```

launcher 内部的精确训练命令已原样写入
`/root/autodl-tmp/experiment_exports/qam_formal_20260731_v1/runtime/exact_command.txt`；
其等价入口是：

```bash
cd /root/autodl-tmp/RLinf_qam_pi0_robotwin
export PYTHONPATH=/root/autodl-tmp/RLinf_qam_pi0_robotwin:/root/autodl-tmp/RoboTwin_RLinf
export EMBODIED_PATH=/root/autodl-tmp/RLinf_qam_pi0_robotwin/examples/embodiment
export REPO_PATH=/root/autodl-tmp/RLinf_qam_pi0_robotwin
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

timeout --signal=TERM --kill-after=180s 55677s \
  /root/autodl-tmp/RLinf/.venv/bin/python -B \
  examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf_qam_pi0_robotwin/examples/embodiment/config \
  --config-name robotwin_adjust_bottle_qam_openpi \
  runner.logger.log_path=/root/autodl-tmp/experiments/qam_formal_20260731_v1 \
  runner.logger.experiment_name=robotwin_adjust_bottle_qam_formal_20260731_v1 \
  runner.max_steps=500 \
  runner.save_interval=25 \
  runner.resume_dir=null \
  runner.ckpt_path=null \
  algorithm.qam.phase=am_on \
  algorithm.qam.inv_temp=1.0 \
  algorithm.qam.warmup_global_inserts=512 \
  algorithm.qam.q_only_updates_before_am=512 \
  algorithm.qam.min_replay_per_rank=32 \
  algorithm.qam.max_updates_per_step=32 \
  actor.global_batch_size=64 \
  actor.micro_batch_size=32 \
  +actor.fsdp_config.save_full_model_weights=false
```

启动身份：

```text
supervisor PID = 103802
timeout PID    = 103857
inner Python   = 103859
run root       = /root/autodl-tmp/experiments/qam_formal_20260731_v1
runtime        = /root/autodl-tmp/experiment_exports/qam_formal_20260731_v1/runtime
hard deadline  = 2026-08-01 09:00 CST
```

17:35:53 CST 只做一次健康探针，命令类别为：

```bash
ps -o pid,ppid,stat,etime,rss,cmd -p <supervisor/driver>
pgrep -af 'train_embodied_agent.py|raylet|gcs_server|QAMPolicyWorker|EnvWorker|RolloutWorker'
tail -n 180 <runtime>/driver.log
grep -E 'global_step|rollout|trajectory|success|critic|qam/|checkpoint|saved' \
  <runtime>/driver.log | tail -n 80
grep -E 'Traceback|CUDA out of memory|OutOfMemory|NCCL|NaN|Inf|SIGTERM|Killed' \
  <runtime>/driver.log | tail -n 60
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.events
df -h /root/autodl-tmp
```

健康结果：

```text
supervisor/driver/Ray        = alive
actor/rollout/env ranks      = 2/2/2 alive
真实完成 rollout cycles      >= 3
global total inserts         = 60
local replay size/rank       = 30
effective phase              = collect/warm-up
critic/fine/AM updates       = 0 / 0 / 0
fine policy version          = 0
GPU memory                   = 23,259 / 23,781 MiB
GPU util（瞬时）             = 0% / 99%
cgroup current               = 198,789,279,744 B
cgroup OOM / OOM-kill        = 0 / 0
disk available               = 933 GiB
```

日志中的 Curobo/pytorch3d import traceback 是 RoboTwin 可选 planner 探测；与 fresh smoke
相同，实际 TOPP 路径随后连续完成 rollout，因此不归类为本次 fatal。Hydra future warning
和 SAPIEN Vulkan ICD warning 同样没有阻止环境交互。当前只有 60/512 warm-up transition，
所以 `critic_updates=0`、`fine_updates=0` 正是批准配置的预期，不是空转或训练故障；
尚无 Q 学习、AM、生效涨点或 checkpoint 结论。

按用户要求，确认健康启动后不持续在线盯盘，也不发送停止信号。下一次只有在用户要求时
才重新连接，届时必须 live 刷新进程、最新完整 cycle、global inserts、critic/fine update、
effective phase、success、checkpoint、GPU/cgroup 和 fatal 扫描。

## QAM-GIT-0006：formal 启动材料同步

只同步 HANDOFF、QAM SSOT/账本和 7 个 formal 脚本/resolved/diff 附件；不触碰运行代码、
shared venv、DSRL/RLT worktree 或活进程。第一次把含 shell `for`/`$file` 的多行命令
作为 PowerShell 普通参数传给 helper，`argparse` 在本机报
`unrecognized arguments: $f; fi; done`；没有建立 SSH 连接、没有服务器改动。改用
`remote_exec_autodl.py run --command-file` 后只读预检成功：

```text
HEAD/upstream = 4a15699e10971e306ed756dcbbf8aa65632553d5 / same
server tree   = clean
formal PIDs   = 103802 / 103857 / 103859 alive
```

本机将精确 allowlist 打成 323,584 B tar，SHA-256 为
`e2531aa63ae12a01b6dc334c9d196b04b7c2d434d2eafa2dc826a9a8eee55ca1`，用一次 SFTP
上传到 `/root/autodl-tmp/qam_formal_docs_sync_20260731_v1.tar`。远端先验证 archive
及内部 10 个文件逐文件 SHA，再从独立 staging `qam_formal_docs_sync_20260731_v1/`
用 `install -D -m 0644` 覆盖精确目标。结果：

```text
archive SHA + 10 file SHA = pass
Markdown/script/YAML diff --check = pass
password/private-key scan         = empty
Git status                        = 3 modified docs + 7 new evidence files
unexpected code/config changes    = none
```

提交前只 stage 这 10 个路径并复核 cached allowlist；push 仍沿用“直连短探针，明确超时才
在单个 child shell 临时 `source /etc/network_turbo`”的既有有界流程。

实际 docs commit 为
`d6fa0f0f4915587ae5e6a03c580fea7938acd3ca`（`docs(qam): record formal launch`），
10 files、1,439 insertions、28 deletions；提交后 tree clean。push 前 GitHub main
HTTP 200、0.94 秒，但 direct `ls-remote` 在 15 秒超时（exit124），因此仅在该次
remote child shell 内启用 `/etc/network_turbo`。一次 push 成功：

```text
personal/codex/qam-pi0-robotwin = d6fa0f0f4915587ae5e6a03c580fea7938acd3ca
ahead/behind                    = 0/0
persistent proxy/remote/config  = unchanged
```

## QAM-FORMAL-0007：首轮 formal 异常、最小修复与 09:00 续跑

### 1. 现场结论与直接失败

2026-07-31 23:53–2026-08-01 00:14 CST 使用固定 host-key 的 Paramiko password-auth
helper 做身份探针与只读审查。首轮 formal 已于 18:05 异常退出，并非配置的正常停止：

```text
last complete outer cycle = 51 / 500
global inserts            = 1005
critic updates            = 493
fine / AM updates         = 0 / 0
fine policy version       = 0
exit code                 = 255
fatal                     = QAM frozen-prefix replay round-trip changed the critic feature
OOM / OOM-kill            = 0 / 0
```

自动 schedule 在 cycle 52 把 critic update 从 493 推向 513；第 513 次 logical update
首次进入 AM，校验在 fine optimizer 前抛错。因而阶段配置只负责暴露此前未执行的 AM 路径，
不是主动停止条件。`global_step_50` 的 DCP/两 rank sidecar/replay/complete manifest 结构完整，
但旧 replay 只保存了固定 prompt，无法无损恢复实际 rollout language，故没有冒险 resume。

根因位于 π0/RoboTwin observation 适配层：RoboTwin 每 episode 从
`description/task_instruction/adjust_bottle.json` 随机选实际指令；rollout frozen feature
使用该指令，通用 trajectory 随后删除字符串，而 QAM replay 旧实现用固定
`adjust the bottle` 重建。官方 QAM 直接读 flat simulator state，没有视觉/语言 prefix
或这项 round-trip；Plain-QAM 的 critic、endpoint gradient、VJP 与 AM 数学未改。

### 2. 两个窄代码修复

首个提交 `e49bba1d9e92808847c76641c3e7a65d2b4e2160`：

- `contracts.py` 新增固定 256-byte UTF-8 prompt tensor encode/decode；
- `openpi_action_model.py` 只在 QAM opt-in 分支把当次实际 prompt 放进 forward payload；
- `fsdp_qam_policy_worker.py` 从 payload 无损恢复 prompt，并让 AM conditioning 显式走
  eval 后恢复原模式；
- 两个测试文件覆盖英文/中文 prompt round-trip 与真实 ingest prompt；
- 5 files，`+132/-8`，不修改 DSRL/RLT/PPO/GRPO、shared venv 或旧 config。

第一次真实 AM fixcheck 证明 prompt 合同已修，但旧逐元素 `allclose(2e-3)` 仍把同一输入在
rollout 单样本和 actor batch32 的 BF16/FSDP 数值差异当成 fatal：

```text
max_abs=0.25
mean_abs=0.00856103
```

第二个提交 `d5f6d7d1da0fc355a71ca653be027282cad040d2` 仅修改同一校验：按每个样本、
每个 prefix block 检查 `max_relative_l2 <= 0.1` 且 `min_cosine >= 0.995`，并把实际
mean-abs/relative-L2/cosine 写进 AM metrics。该门仍能发现 block/prompt/相机合同错配，
但不再要求不同 batch kernel 逐元素近似相等；QAM loss、更新顺序和正式超参均未变。

两次修改后都在服务器执行同一集中命令：

```bash
cd /root/autodl-tmp/RLinf_qam_pi0_robotwin
PYTHONPATH=/root/autodl-tmp/RLinf_qam_pi0_robotwin \
  /root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py \
  tests/embodiment/test_qam_openpi_adapter.py
```

结果均为 `31 passed, 3 dependency warnings`；分别用时 8.92 s 与 8.41 s。

默认直连 `ls-remote` 两次 15 s 超时；没有改 remote/Git config。仅在各自 child shell
临时 `source /etc/network_turbo`，两次 fast-forward push 后复核 remote HEAD，最终为
`d5f6d7d1da0fc355a71ca653be027282cad040d2`，server tree clean，代理未持久化。

### 3. 一次 AM 的真实验证

第二次 fixcheck 使用 formal 的两卡、2 env、global/local batch `64/32`，只把测试预算改为：

```text
runner.max_steps=4
warmup_global_inserts=32
q_only_updates_before_am=1
max_updates_per_step=2
save_interval=100
video=false
```

四个 outer cycle 是让每 rank replay 达到正式 local batch32 所需的最小收集量；实际只执行
2 次 critic update，其中第 1 次 q_only、第 2 次 joint critic+AM。结果自然 exit0：

```text
critic_updates          = 2
am_updates / fine       = 1 / 1
fine_policy_version     = 1
am_loss                 = 0.011
terminal_adjoint_norm   = 0.098
fine_grad_norm(preclip) = 10.055
OOM / OOM-kill          = 0 / 0
```

没有保存大 checkpoint。第一次失败 fixcheck 和第二次通过 fixcheck 分别保存在
`qam_prompt_fixcheck_20260801_v1`、`v2` 的独立 run/runtime，均未覆盖首轮 formal。

### 4. fresh formal v2 启动

用户再次授权后，使用新增脚本
`evidence/qam_formal_launch_20260801_v2.sh` fresh 启动；除代码 HEAD、独立输出路径和
重新计算的 wall timeout 外，正式配置保持已批准版本：

```text
512 global macro collect, 0 update
-> 512 critic-only updates, UTD=1
-> logical update 513 起 joint critic + AM
2 GPU / 2 env
global/local batch = 64 / 32
inv_temp = 1.0
max_updates_per_step = 32
save_interval = 25
max_steps = 500 safety ceiling
hard deadline = 2026-08-01 09:00 CST
```

启动现场：

```text
launch time       = 2026-08-01 00:43:47 CST
wall limit        = 29,773 s
supervisor PID    = 183126
HEAD / tree       = d5f6d7d1... / d63c2a03...
run root          = /root/autodl-tmp/experiments/qam_formal_20260801_v2
runtime evidence  = /root/autodl-tmp/experiment_exports/qam_formal_20260801_v2/runtime
disk available    = 967,284,490,240 B
```

00:46:39 的一次启动健康探针确认 supervisor/driver/Ray 和两 rank actor/rollout/env 存活，
首个真实 rollout 已在 18.72 s 内完成；GPU 为 24,628/24,165 MiB、util 55%/74%，
cgroup anon 约 27.8 GiB，OOM/OOM-kill 为 0/0，fatal 扫描为空。Curobo/pytorch3d 是既有
可选 planner 探测，实际 mplib/TOPP rollout 已完成。按用户要求，此后不持续监控；下一次
只有收到请求才重新连接并刷新当前状态。

## QAM-FORMAL-0005：非阻断诊断与无超时 step25→100 续训（2026-08-01 00:57–01:10 CST）

用户澄清 09:00 是按步数估算的完成时间，不是 hard cutoff；并把此前无官方依据的
relative-L2/cosine 经验阈值阻断正式训练定性为严重事故。长期规则：结构/字段/shape
错误和 NaN/Inf 可 fail-fast；经验数值诊断默认只记录/告警，不得擅自成为 formal gate。

Paramiko 只读审查确认 v2 `global_step_25` 完整：DCP 两 shard 各约4.99GB、两 rank
sidecar/replay 均存在，manifest 为 `complete=true`、schema2、world_size2、snapshot ID
一致且无 tmp。随后按 `driver.pid` 精确核对命令和 PGID，执行：

```bash
kill -TERM -- -183159
```

首版停止脚本因错误假设命令字符串顺序而自行拒绝，未发信号、训练未受影响；拆成两个独立
标识匹配后成功。v2 driver exit134、monitor exit0；这是预算语义纠正，不是训练异常。

代码只改 `rlinf/workers/actor/fsdp_qam_policy_worker.py`：shape 与非有限诊断仍抛错；
`max_relative_l2>0.1` 或 `min_cosine<0.995` 仅每 rank 首次 warning，原 metrics 持续记录，
不中断、不跳过 AM、不改变 loss。服务器测试：

```bash
cd /root/autodl-tmp/RLinf_qam_pi0_robotwin
PYTHONPATH=/root/autodl-tmp/RLinf_qam_pi0_robotwin \
  /root/autodl-tmp/RLinf/.venv/bin/python -m pytest -q \
  tests/embodiment/test_robotwin_qam_contract.py \
  tests/workers/test_qam_worker_helpers.py \
  tests/embodiment/test_qam_openpi_adapter.py
# 31 passed, 3 warnings in 8.23s
```

提交 `9e2abc04e8c178575d9b800154d69b9123e73ecb` 已经临时
`/etc/network_turbo` 子 shell fast-forward 推送并复核 remote HEAD；没有持久化 proxy。

新增精确 launcher：`evidence/qam_formal_resume25_to100_launch_20260801_v3.sh`。核心覆盖：

```text
resume_dir = v2/.../checkpoints/global_step_25
runner.max_steps = 100（绝对终点）
runner.save_interval = 25
warmup_global_inserts / q_only_updates_before_am = 512 / 512
UTD = 1；global/local batch = 64/32；inv_temp = 1.0
无 timeout / wall-clock kill
```

cycle100 的依据：collect 25 cycles 实测约14分钟；collect+q_only 约30–45分钟；一次真实
joint update 约25–30秒，AM 时每 cycle 通常约20 updates。因此 step25→100 估计约
7.5–9小时，接近09:00但允许吞吐误差；各阶段计数未改。

01:06:02 supervisor PID209536 启动。01:09:48 首个恢复 cycle 完成：step26/100、
replay256/rank、global total inserts512、q-only anchor512、critic/fine/policy version均0；
GPU30,497/30,437MiB、util48%/42%，OOM/OOM-kill0/0，进程无 exit。该证据确认 replay/
counters live resume 连续；环境进程重新初始化，不声称轨迹 bitwise 连续。按用户要求不再盯。

## QAM-FORMAL-0006：step100→380 约12小时续训（2026-08-01 10:33–10:57 CST）

### 1. v3 完成事实与预算依据

10:33 现场只读审查确认 v3 已于03:39自然完成 `100/100`，driver/monitor exit0。最终计数为
global inserts `1974`、critic updates `1462`、fine updates/policy version `950/950`、
pending0，满足 `1974-512=1462` 与 `1462-512=950`。step100 checkpoint
complete/schema2/world2，大小13,154,614,481 B（12.25 GiB）；GPU峰43,083/43,251 MiB，
OOM/OOM-kill0/0。

成熟 AM 段 step50→75 用时63.3分钟、step75→100用时67.0分钟，即
`2.53–2.68 min/cycle`。约12小时对应新增269–285 cycles，取中间且对齐 checkpoint 间隔的
280 cycles，因此绝对终点为380；`max_steps=300` 只会从100新增200 cycles，约8.5小时。
这只是基于实测吞吐的步数估计，不是 wall-clock deadline。

### 2. 只读前检

通过进程内密码 Paramiko 和固定 host-key，于10:49:24执行：

```bash
hostname
pwd
id -u
git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin rev-parse HEAD
test -z "$(git -C /root/autodl-tmp/RLinf_qam_pi0_robotwin status --short)"
pgrep -af '[t]rain_embodied_agent.py|[r]aylet|[g]cs_server' || true
tr -d '\n' < .../global_step_100/actor/qam_components/complete.json
test -f .../global_step_100/actor/dcp_checkpoint/.metadata
test -z "$(find .../global_step_100 -name '.tmp-*' -o -name '*.tmp')"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
df -B1 --output=avail /root/autodl-tmp
awk '/oom |oom_kill / {print}' /sys/fs/cgroup/memory.events
```

结果：hostname `autodl-container-nekaqbwt43-6ce5babb`、pwd `/root`、uid0；QAM worktree
HEAD `24cbc8d20d19161c46da9940b5731127530e911d` 且 clean；无训练/Ray；step100 manifest
complete/schema2/world2；两卡0 MiB/0%；磁盘 available918,612,430,848 B；OOM/OOM-kill0/0。

### 3. 新增文件与精确启动

新增 `evidence/qam_formal_resume100_to380_launch_20260801_v4.sh`。相对 v3 只改变恢复起点、
独立输出名和绝对终点 `runner.max_steps=380`；阶段门槛512/512、UTD1、N20、2 GPU/2 env、
global/local batch64/32、`inv_temp=1.0`、save interval25、update cap32、无 full weights 与无
wall-clock kill 均保持不变。脚本保存完整 resolved command、budget 和 provenance。

启动命令：

```bash
bash -n /root/autodl-tmp/qam_formal_resume100_to380_launch_20260801_v4.sh
nohup bash /root/autodl-tmp/qam_formal_resume100_to380_launch_20260801_v4.sh \
  >/root/autodl-tmp/qam_formal_resume100_to380_supervisor_20260801_v4.log \
  2>&1 </dev/null &
```

10:50:38启动：supervisor PID380815、driver PID380841。运行根为
`/root/autodl-tmp/experiments/qam_formal_resume100_to380_20260801_v4`；runtime 为
`/root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime`。

### 4. 一次性启动健康门与问题记录

10:53:08 log 确认从完整 step100 resume。10:56:56首个新 cycle 完成：`101/380`、global
inserts `1994`、critic `1482`、fine/policy version `970/970`、phase2、pending0；相对 step100
分别精确增加20/20/20，未重置或追补。GPU36,089/36,265 MiB、util76%/76%，
OOM/OOM-kill0/0，无 exit/fatal。按用户要求至此停止查看。

一次本机状态调用因默认 `python.exe` 不可访问而在建立 SSH 前失败；未向服务器发送命令。
随后改用 Codex bundled Python 与既有 Paramiko 依赖目录，沿用同一固定 host-key helper，
只读探针成功。没有修改服务器训练、配置或依赖。

### 5. Git 发布

四个精确文件以提交 `6aa4ec95d51cab3a5f890317386d941a83bd70db` 发布。命令为：

```bash
git add -- HANDOFF.md \
  docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md \
  docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md \
  docs/rlinf-robotwin-pi0-qam/evidence/qam_formal_resume100_to380_launch_20260801_v4.sh
git diff --cached --check
git commit -m "docs(qam): record resume to cycle 380"
GIT_TERMINAL_PROMPT=0 timeout 60 git push personal HEAD:codex/qam-pi0-robotwin
```

默认直连在60秒有界窗口内未完成；随后只在子 shell 临时
`source /etc/network_turbo` 并重试一次，push 成功，未持久化 proxy/Git 配置。远端
`personal/codex/qam-pi0-robotwin` 与本地均为 `6aa4ec95...`，server tree clean。

## QAM-FORMAL-0007：用户停止、cycle247 收尾与轻量产物（2026-08-01 17:28–17:35 CST）

### 1. 授权与停止前快照

用户明确要求“停止训练，总结可能不 work 的原因，简要收集产物”。本轮授权包含精确停止
当前 QAM formal，未授权删除 checkpoint、清 cache、恢复或运行新实验。

先通过进程内密码 Paramiko 做身份与 PID/命令只读核对：

```bash
hostname
pwd
id -u
date '+%F %T %Z'
ps -o pid,ppid,pgid,sid,etime,cmd -p 380841
grep -a 'Global Step:' \
  /root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime/driver.log \
  | tail -n 1
```

17:28:22 结果为 hostname `autodl-container-nekaqbwt43-6ce5babb`、pwd `/root`、uid0；
PID380841 的完整命令明确包含
`qam_formal_resume100_to380_20260801_v4`、`runner.max_steps=380`、resume step100、
512/512 schedule、batch64/32 和 `inv_temp=1.0`。停止前最后完整记录为 `247/380`。

### 2. 精确优雅停止

远端执行的核心脚本如下；它先验证目标命令再只向该 driver 发 TERM，没有发送 broad
`pkill`、没有碰其他 worktree 或训练：

```bash
runtime=/root/autodl-tmp/experiment_exports/qam_formal_resume100_to380_20260801_v4/runtime
pid=$(cat "$runtime/driver.pid")
cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
case "$cmd" in
  *qam_formal_resume100_to380_20260801_v4*) ;;
  *) echo TARGET_MISMATCH; exit 42 ;;
esac
kill -TERM "$pid"
for _ in $(seq 1 30); do
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  sleep 2
done
```

精确结果：

```text
STOP_REQUEST_TIME=2026-08-01 17:28:58 CST
TARGET_PID=380841
TERM_SENT=yes
DRIVER_EXITED=yes
STOP_CHECK_TIME=2026-08-01 17:29:02 CST
GPU0/GPU1=0/0 MiB
```

17:29:55 二次只读复核：driver 与 monitor 均不存活；run-specific/Ray 过滤只命中正在执行
检查的 shell 自身；两卡0MiB/0%。monitor exit code 为0。driver runtime exit code 为134，
日志末尾同一时刻明确记录 `SIGTERM received`；因此它是用户授权停止的信号退出，不是训练
自行 fatal。没有调用 KILL，也没有删除、覆盖或另存 checkpoint。

### 3. 最终指标与恢复边界

停止后重新下载完整 v4 `driver.log/resources.csv`，与已有 v2/v3 日志合并解析。最终为：

```text
last complete cycle       247/380
success                   86/494 = 17.41%
collect/q_only/am_on      16.00% / 11.54% / 18.37%
last 10/20/50 cycles      35.0% / 22.5% / 24.0%
global inserts            4861
critic/fine/policy        4349 / 3837 / 3837
pending credit            0
recent20 critic loss      0.00376
recent20 Q/TD/std         0.03065 / 0.03090 / 0.01815
recent10 AM/adjoint/grad  7.2708 / 0.00474 / 3356.71 pre-clip
GPU peak MiB              43567 / 43693
host anon peak            44.99 GiB
OOM/OOM-kill              0/0
```

schedule 算术严格满足 `4861-512=4349`、`4349-512=3837`。最新可恢复点为
`global_step_225`：completion `complete=true`、schema2、world2、snapshot
`29fad04b897a403891289193ef20bd3c`；大小15,030,394,539 bytes，completion SHA-256
`6fda6d37543194988ed6a0f49e31118774e90c822b7316d040c6e7dc261cd5e7`。cycle226–247
日志保留，但其参数更新不在 checkpoint225。

### 4. 不 work 风险的证据分层

直接证据是 critic 已 TD 自洽，但 action gradient 尚未被真实反事实验证：低 loss、Q≈TD、
head std 小不能证明 $\nabla_aQ$ 指向更高成功率；terminal adjoint 从 AM 前10轮均值0.0097
降到末10轮0.00474；fine pre-clip grad 仍远大于 clip1.0。资源和数值稳定，不是直接原因。

主要强推断依次为：online-only replay 在当前策略附近覆盖不足；N20 的280D macro action 与
末端 sparse reward 造成粗信用；C1 frozen pooled feature 可能共同丢失姿态/接触信息；长期
强裁剪可能压平 reward 强弱；B1 frozen behavior 缺少官方 B2 的 FM/slow EMA update。
仍未验证的是动作扰动 Q 排序与真实执行、C1 success/failure 可分性，以及 fine 相对 base
的动作漂移。

最终 online train curve 也不是纯失败：am_on 18.37% 高于 collect16.00%，末50轮24%。但
每 cycle 仅2个非独立 train episode，且没有同 seed frozen-base/held-out eval，所以这些
末段数字既不能证明稳定涨点，也不能推翻上述风险。完整简报新增为
`evidence/QAM_FORMAL_STOP247_CLOSEOUT_20260801.md`。

### 5. 轻量产物收集

先只读盘点：v4 runtime约1.3MiB，v4 run/checkpoint约67GiB；checkpoint正文不下载。随后
在新目录生成不可变轻量包：

```text
/root/autodl-tmp/experiment_exports/qam_pi0_robotwin_formal_stop247_20260801
/root/autodl-tmp/experiment_exports/qam_pi0_robotwin_formal_stop247_runtime_20260801.tar.gz
```

包内包含 v1–v4 runtime、四份 launcher、resolved/source diff、checkpoint 文件大小清单、
各 checkpoint completion JSON、post-stop snapshot 和逐文件 SHA-256。明确排除 DCP shard、
rank sidecar、replay tensor、视频、SFT 模型、venv、数据集和 cache。服务器目录2.7MiB，
压缩包231,672 bytes。

服务器与 Windows 下载副本 SHA-256 均为：

```text
6740c3e71f6b963940498cec214b7448cd483b847baa4e303d869548b44d14ab
```

Windows 副本：

`exports/qam_pi0_robotwin_formal_stop247_runtime_20260801.tar.gz`

下载后重新计算 hash，`match=True`。该包用于审计、复盘和重画曲线，不能独立 resume。

Windows 另生成 high-info work-materials 包，包含最终报告、机器可读指标 JSON、cycle247
曲线 PNG 与 runtime hash，不包含大 checkpoint：

```text
exports/qam_pi0_robotwin_formal_stop247_high_info_20260801_v1.zip
SHA-256 28fa0708f6bee368928dbc365e4267a23ac25767b85dcab8e671d89082be20c2
```

ZIP 共5个文件、103,673 bytes；重新打开归档列出5个 entry，JSON 可解析，hash
`match=True`。

### 6. 文档发布

服务器 worktree 操作前为 `codex/qam-pi0-robotwin@e964b44c...`、clean、upstream `0/0`。
只同步并暂存 `HANDOFF.md`、SSOT、实施账本和本次新增收尾报告，`git diff --cached --check`
通过；首个提交为：

```text
f587f1cfa9d90d53c185226c502913ff6eb27c0e
docs(qam): close formal run at cycle 247
```

默认直连 `ls-remote/push` 在有界窗口内无输出并留下 ahead1，没有循环重试。随后仅在一次
子 shell 内 `source /etc/network_turbo`，执行同一 fast-forward push 并复核：

```text
LOCAL_HEAD=f587f1cfa9d90d53c185226c502913ff6eb27c0e
REMOTE_HEAD=f587f1cfa9d90d53c185226c502913ff6eb27c0e
ahead/behind=0/0
```

子 shell 退出后未持久化 proxy、Git config 或 remote；没有 force push。
