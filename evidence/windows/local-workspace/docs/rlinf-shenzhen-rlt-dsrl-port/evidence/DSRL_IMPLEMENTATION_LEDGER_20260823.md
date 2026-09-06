# 深圳 current RLinf × RoboTwin π0 DSRL 实施流水账

日期：2026-08-23  
机器：`SZ-H100`  
账号：`chenyiteng`  
专题入口：[`../00_INDEX_AND_MIGRATION_PLAN.md`](../00_INDEX_AND_MIGRATION_PLAN.md)

本账只记录 DSRL 独立实现；不记录密码、token 或私钥。所有远端密码仅由当前进程的无回显提示注入，
SSH 固定校验既有 host-key `SHA256:qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY`。

## 授权与硬边界

- 用户已授权：在深圳 current `7d07a421...` 的独立 worktree/branch 实现最小 DSRL port，执行
  CPU 级 AST/compile/Ruff、focused fixture 和 Hydra compose，完成后 commit 并普通 push。
- 当前禁止：加载真实 π0、启动 RoboTwin/Ray、占用 GPU、运行真实 smoke、停止或干预 GPU4--7 的
  GRPO，以及安装依赖、下载模型或更改系统服务。
- 现场由主任务刷新：GRPO 在 GPU4--7 正常运行到 Step52；GPU0--3 空闲，但 host available 约
  143 GiB、cgroup 约 1.914 TiB 且 swap 已满。因此本轮所有检查必须保持 CPU/轻量，不借空闲 GPU
  扩大检查。

## DSRL-001：来源恢复与最小补丁冻结

- 完整读取：根 `PROJECT_CONTEXT.md` / `HANDOFF.md`、专题 `00` / `02`、旧 DSRL 主计划与实施账。
- 源码锚点：
  - current host：official RLinf `7d07a4212ee6858cc333e1d4fab7a37256d1f839`；
  - 旧成功实现：`6817c73b298ff9df78d371d4b139e4e0fa8ea529`，最终文档态 `48a775db...`；
  - current data/schema 重构：`d3aff547...`；RTC：`657faae5...`，本算法首版关闭。
- 旧两组 focused tests 已逐行复核：macro projection/ring，以及 critic-only shadow/resume/梯度隔离。
- 冻结的最小生产面：
  1. 新 `rlinf/data/storage/replay/dsrl_transition.py`；
  2. 小改 `rlinf/data/storage/replay/__init__.py`；
  3. symbol-level 改 `rlinf/models/embodiment/openpi/openpi_action_model.py`；
  4. symbol-level 改 `rlinf/workers/actor/fsdp_sac_policy_worker.py`；
  5. 新深圳 RoboTwin DSRL YAML；
  6. 两个 current-schema focused test 文件。
- 明确 no-touch：Builder/schema、EnvWorker、runner、RTC worker、Gaussian policy、compact encoders、
  SAC/10-Q 数学和 generic trajectory replay。
- 与旧 port 的一个有意收窄：首版不提供“缺 DSRL sidecar 的 legacy DCP”兼容恢复；current formal
  从 fresh 开始，DSRL checkpoint 缺私有 phase/ring/shadow sidecar 时 fail closed。

## DSRL-002：服务器 worktree preflight

- 时间：2026-08-23 19:29 CST（服务器输出 UTC `11:29`）。
- 完整命令文件：
  `local_scripts/remote_commands/shenzhen_dsrl_impl_create_worktree_20260823.sh`。
- 写入前检查：
  - canonical `/data/chenyiteng/projects/rlinf-shenzhen/RLinf` 精确为
    `7d07a4212ee6858cc333e1d4fab7a37256d1f839`，detached、clean；
  - 目标目录和分支均不存在；
  - `/data` 为 ext4 3.5 TiB，已用338 GiB、可用3.0 TiB；
  - 既有 GRPO、π0-DVAC、PPO worktree 未改动。
- 执行：
  `git -C <canonical> worktree add -b codex/sz-current-dsrl-pi0-robotwin
  /data/chenyiteng/projects/rlinf-current-dsrl/RLinf-7d07-dsrl-robotwin 7d07a421...`。
- 结果：返回码0；新 worktree HEAD/branch精确匹配、worktree clean。remote保持既有
  `origin=RLinf/RLinf`、`personal=Yutenji-Nyamu/rlinf_fastwam`，没有修改Git配置。

## DSRL-003：current源码现场复核与补丁准备

- 本地独立审计 worktree：`.tmp/dsrl_current_impl_20260823`，精确基于 `7d07a421...`；未触碰
  RLT agent 的共享源码目录。
- 已完成的最小生产增量：
  - current `Trajectory` → first-done DSRL macro projector 与 compact CPU ring；
  - legacy OpenPI 内的 Gaussian warm-up phase、repeat-H 32D latent、compact main-image helper、显式
    stochastic eval 开关；
  - current SAC worker 的 opt-in ring ingest、global计数、动态 UTD20、actor/critic grad隔离、
    critic-only FP32 target shadow；
  - strict phase/counter/shadow sidecar，ring 自身 strict cursor/RNG checkpoint，及
    `save_full_model_weights` 窄透传；
  - 拓扑中立的深圳 source-aligned YAML 与两组 focused tests。
- 有意未实现：旧 checkpoint 缺 sidecar 的隐式兼容、RTC、π0.5、generic replay改写、额外机制 gate。
- 本地 `git diff --check` 返回0；本机 Windows Python app alias不可执行，因此 AST/compile、focused
  tests、Ruff和Hydra compose全部转到既有服务器venv做轻量 CPU 检查，不安装依赖。

## DSRL-004：补丁同步与轻量检查

已完成。完整补丁通过既有固定 host-key 的 Paramiko 进程直接流入独立 worktree；远端先检查 exact
HEAD/clean、再 `git apply --check`，不会写 canonical worktree，也不会触发模型、Ray 或 GPU。

- 第一次调用把CLI参数误写成带`SHA256:`前缀；helper自身会补前缀，因此在认证/执行前按预期拒绝为
  `expected SHA256:SHA256:...`。改为传helper要求的裸摘要后，现场返回的完整fingerprint仍精确匹配既有
  固定值；未关闭host-key校验，也未切换认证路线。

- 第一次 `py_compile` 返回0；Ruff只报两份新 focused test 缺仓库标准 copyright header（`CPY001`）。
  这是代码风格失败，不涉及算法或依赖；仅补标准header，随后完整重跑，不新增检查机制。
- 首轮修复后结果：Ruff返回0；两组 focused tests 共7项全部通过（20.17秒）；Hydra完整resolve返回0。
- 配置在commit前按专题边界收窄为拓扑中立的 `robotwin_adjust_bottle_dsrl_openpi.yaml`：placement为
  `all`；global batch 256和global capacity 25000由实际actor world-size动态分片。1/2/4张visible GPU
  均可compose，formal推荐拓扑留到启动packet讨论，不写进算法base。
- generic YAML 显式 `rtc_enabled: false`；这不是删除 current RTC API，只锁定 DSRL 首版不用该专用
  eval控制轨。
- 配置重命名的第一次手写增量patch因hunk计数错误报 `corrupt patch at line 22`；第二次语法有效但上下文
  过短未应用。远端文件只完成了同目录精确rename，内容仍原样。随后从SFTP只读取回该文件，由Git生成
  exact no-index diff并在本地副本上 `git apply --check` 通过，再应用到远端；未删除或覆盖其他文件。
- 最终完整复测（显式 `CUDA_VISIBLE_DEVICES=""`，既有venv，无安装）：
  - 6个production/test Python文件 `py_compile`：通过；
  - 同6文件 Ruff：通过；只有repo配置中4条preview rule无效的既有warning；
  - 两组 focused tests：7/7通过，25.00秒；16条warning均来自环境内deprecated TorchScript API；
  - generic DSRL YAML Hydra compose/resolve：通过，并断言 DSRL ring、UTD20、warm-up、N20/H50、
    Gaussian phase、stochastic eval、RTC off和`save_full_model_weights=false`；
  - `git diff --check`：通过。
- 完整待提交patch：`evidence/dsrl_current_port.patch`，73,471 bytes；最终stat为7 files，
  `+1456/-121`（含225行YAML与341行focused tests）；production四文件为`+890/-121`，与旧成功
  runtime约`+1016/-130`同量级，但没有重写current generic replay或引入legacy兼容层。
- 第一次commit被Git在写对象前拒绝：新worktree没有author identity，报
  `fatal: unable to auto-detect email address`；没有commit、没有push。只读查询个人仓库既有DSRL与GRPO
  提交，作者均为 `Yutenji-Nyamu <1842710211@qq.com>`；因此只对本worktree设置同一repo-local
  `user.name/user.email`，不改global配置，再重试同一精确7文件提交。

## DSRL-005：commit、push与当前断点

- 精确staged文件：上述7个目标文件；`git diff --cached --check`返回0，未纳入日志、pycache、模型、
  checkpoint、Hydra输出或其他worktree文件。
- commit：`4b609178d10d2534f3f972435ad972e4e015c392`，消息
  `feat(dsrl): port RoboTwin DSRL to current RLinf`。
- push：`personal` → `Yutenji-Nyamu/rlinf_fastwam` 新分支
  `codex/sz-current-dsrl-pi0-robotwin`，普通push成功；服务器独立worktree clean。
- 当前断点：实现、轻量检查、commit/push均完成；**没有运行真实π0、RoboTwin、Ray或GPU smoke**。
  下轮只能先讨论并冻结可见GPU拓扑、并发/outer-step预算、输出路径、预期资源、监控项和停止条件；
  按专题/用户边界另获真实smoke批准后再启动。

## DSRL-006：repo-local upstream 收口

- 发现：push已成功，但新建本地branch尚未记录upstream；这不影响远端提交存在，却会让后续普通
  `git status/pull/push`缺少默认比较对象。
- 仅在DSRL独立worktree执行：
  `git branch --set-upstream-to=personal/codex/sz-current-dsrl-pi0-robotwin codex/sz-current-dsrl-pi0-robotwin`。
- 精确验证结果：
  - local HEAD：`4b609178d10d2534f3f972435ad972e4e015c392`；
  - remote-tracking HEAD：`4b609178d10d2534f3f972435ad972e4e015c392`；
  - upstream：`personal/codex/sz-current-dsrl-pi0-robotwin`；
  - ahead/behind：`0/0`；
  - worktree：clean。
- 本步只修改该worktree的repo-local branch tracking metadata；没有修改源码、提交或远端内容，也没有
  运行测试、模型、Ray、RoboTwin或GPU任务。

## DSRL-007：physical GPU 6--7 fresh/resume smoke

- 2 ranks、4 train env、H50/N20、GB256/MB64、UTD20、ring25k、RTC/eval off。
- fresh：40 global macro transitions、800 critic updates，`global_step_1`，总时429.239s，其中rollout
  31.533s、training338.1s；自然退出0。
- fresh-process resume：真实提前终止后新增36 transitions、720 updates，累计`update_step=1520`与
  ring resident76，保存`global_step_2`；总时375.282s，其中rollout34.598s、training307.9s；退出0。
- strict state逐rank验证learned phase、ring cursor/RNG、pending0以及critic-only FP32 shadow 156 tensors；
  fresh resident20+20，resume 36+40。
- 显存峰fresh 34,395 MiB/card、resume 35,843 MiB/card；cgroup峰53.80/48.16 GiB，host available最低
  1,947.84/1,948.91 GiB，OOM/kill均0。
- 训练、恢复、checkpoint与资源链均已通过；formal尚未启动。完整脚本和取证入口见共同
  `IMPLEMENTATION_LEDGER.md` 的IMPL-009/011。
