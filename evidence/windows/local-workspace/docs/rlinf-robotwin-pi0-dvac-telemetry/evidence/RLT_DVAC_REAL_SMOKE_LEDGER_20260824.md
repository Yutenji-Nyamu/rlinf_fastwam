# RLT teacher-DVAC 真实 smoke 流水账（2026-08-24）

## 1. 本轮请求与授权

- 用户授权：对已经实现并推送的 RLT teacher-DVAC 做简洁检查，然后执行真实 smoke。
- 用户要求：除为 smoke 缩短预算所必需的覆盖项外，参数尽量对齐此前成功的 RoboTwin π0 RLT。
- 用户同时要求：把已经停止的 global-z DVAC `[0,2]` g49 运行整理成轻量高信息量 ZIP。
- 本轮不启动正式 RLT 训练；正式预算另行讨论。历史事实需要保持为：先完成 fresh 250-cycle formal，再续训至 cycle 480；最终成功终点是 cycle 480，而不是把 480 当作 smoke 配置。

## 2. 操作边界

- Windows：阅读、文档、轻量证据与 ZIP；不在本地运行 RLinf/RoboTwin 项目测试。
- AutoDL：刷新进程/GPU/RAM/Git 现场；复核 resolved config；运行少量高信息量项目检查；启动并观察本次 RLT-DVAC smoke。
- 不安装依赖，不下载模型，不删除历史产物，不停止无关进程，不覆盖已有运行目录。
- 资源监视只记录，不自动改变训练行为；仅在 CUDA OOM、worker fatal、非有限 loss/gradient 或本次进程异常退出时停止本次 smoke。

## 3. 流水账

### 3.1 上下文恢复

- 完整读取根目录 `PROJECT_CONTEXT.md`、`HANDOFF.md` 和专题单一事实源 `00_INDEX_AND_PLAN.md`。
- 完整读取 RLT-DVAC 方案与既有实现账本：
  - `28_RLT_DVAC_IMPLEMENTATION_AND_RECORDING_PLAN_20260823.md`
  - `evidence/RLT_DVAC_IMPLEMENTATION_LEDGER_20260824.md`
- 复核历史 RLT 结果：Stage 2 fresh formal 先到 cycle 250，随后从 250 续训并完整结束于 cycle 480；最终记录为 3,840 train episodes、57,410 global replay rows、215,055 critic updates、107,528 actor updates，最终 fixed-20 为 17/20。
- 本机 Git 只读状态首次因 Windows sandbox 用户与仓库所有者不同触发 `dubious ownership`；改用每条命令局部的 `git -c safe.directory=...` 读取，没有写全局 Git 配置。

### 3.2 AutoDL 现场刷新

- 指令：通过固定host-key的低层Paramiko执行
  `tmp/rlt_dvac_live_audit_20260824.sh`、`tmp/rlt_dvac_config_and_smoke_audit_20260824.sh`和
  `tmp/rlt_smoke_source_values_20260824.sh`。
- 身份探针：`hostname=autodl-container-nekaqbwt43-6ce5babb`、`pwd=/root`、`uid=0`；容器PID 1启动于
  `2026-08-24 16:03:26 CST`。
- 启动前资源：两张A800均`0/81920 MiB`、util 0%，无GPU compute process、训练或Ray；cgroup current
  `384,425,984 bytes`，`oom=0, oom_kill=0`；`/root/autodl-tmp`余约`735 GiB`。
- RLT-DVAC worktree：
  `/root/autodl-tmp/RLinf_rlt_teacher_dvac`，branch=`codex/rlt-teacher-dvac-weighting`，
  HEAD=`275ab4535f8363e15b839cbb64e7d997a056d01e`，worktree clean，与upstream ahead/behind=`0/0`。
- 成功RLT base：`/root/autodl-tmp/RLinf_rlt_pi0_robotwin@2b8199d8...`；Stage-1 step2000、Stage-2
  step250和resume step480 checkpoint仍在服务器。

### 3.3 smoke 前检查与 resolved 合同

- 指令：执行`tmp/rlt_dvac_smoke_precheck_20260824.sh`，只创建本次新的runtime/evidence目录，不改source。
- 结果：
  - `tests/unit_tests/test_rlt_dvac_weighting.py`：`4 passed in 2.10s`；
  - Stage-1 binding preflight：`passed=true`，确认H50/C10/D14、`z_rl=2048`、prefix=`[768,2048]`、
    Stage-1完整权重与norm stats；
  - Hydra compose/resolve成功。
- resolved主体仍为成功RLT：2×A800、8 train env、4 eval env×5 epoch=20 fixed episodes、H50/C10、
  episode上限200、replay window 50,000、原actor/critic/BC-Q目标与优化器。
- 方法增量：`mode=apply`、L2/L3/L4采集、selected L3、applied horizon C10、`z_clip=2`、
  `strength=.5`即`[0,2]`。
- smoke-only覆盖：`max_steps=1`、`val/save=1`、`warmup_min_size=2`、
  `warmup_post_collect_updates=8`、`max_updates_per_train_step=20`、actor Q schedule warmup/ramp=`4/8`、
  raw trace interval=`1`。它们只让单cycle内实际进入apply；正式配置仍是`10000/30000/1600`。
- 输出：
  `/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v1`；resolved配置和精确命令
  位于同名`experiment_exports/.../runtime/`。

### 3.4 真实 smoke

- 首次启动器调用没有启动项目：复用的资源monitor文件当前没有executable位，而启动器误用`test -x`。
  因monitor实际以`bash monitor.sh`调用，窄修启动器为`test -f`；训练参数、source和输出目录均未变化。
- 修复后于`2026-08-24T17:18:19+08:00`启动；driver PID=`4038`、只记录的monitor PID=`4039`。
- 启动后Ray、两个actor rank、两个rollout rank和两个EnvWorker正常创建；历史已有的可选Curobo import warning
  不影响RoboTwin rollout继续。
- 首个8-env rollout于119.15秒自然完成。随后两rank均写出actor update `0/2/4/6`的DVAC NPZ，证明
  collection、baseline freeze和真实apply分支已经进入；最终评估、checkpoint与exit仍待自然完成。

### 3.5 产物核对与结论

- `17:39--17:49 CST`用`tmp/rlt_dvac_smoke_poll_concise_20260824.sh`、
  `tmp/rlt_dvac_smoke_worker_probe_20260824.sh`与Ray task只读探针持续刷新：两卡始终有计算、两个actor
  rank和rollout/env worker均alive，cgroup约40 GiB，OOM/OOM-kill为0；没有人工中断。
- 指令：把`tmp/rlt_dvac_smoke_npz_probe_20260824.py`经SSH stdin交给服务器RLinf venv Python读取
  8个已写NPZ。结果：两个rank均有update=`0/2/4/6`；每个trace的
  `teacher_dvac_v=[4,3,50]`、`selected_v/z/weights=[4,10]`、
  `student_actions/ref_chunk=[4,10,14]`，全部numeric值finite。该证据确认真实teacher endpoint采集、
  冻结global-z映射和student-Q反向加权分支都已进入，不只是compose成功。
- `2026-08-24T17:52:06+08:00`进程自然异常退出，exit=`255`。第一条决定性错误是：
  - rank 1：`SeqNum=44, OpType=ALLREDUCE, NumelIn=105`，运行`1,800,045 ms`后watchdog timeout；
  - rank 0：已到`SeqNum=45, OpType=ALLREDUCE, NumelIn=1`，运行`1,800,090 ms`后timeout；
  - 两个actor随后被NCCL watchdog以SIGABRT终止，main process因worker execution failure退出。
- 时间解释：8-env rollout和8个NPZ均在约`17:22`完成；上述collective从该时刻挂起并在30分钟watchdog
  到期后退出。因此这次33分47秒wall time不是“评估变慢”；本次根本没有进入fixed-20 eval，也没有创建
  `global_step_1` checkpoint。
- 只读源码定位：
  - `process_train_metrics()`把每个rank本地metric dict排序、拼成一个tensor后直接`all_reduce_dict()`；
  - 新增`_rlt_dvac_context_metrics()`会根据rank-local batch是否含positive/non-positive reward以及
    student/reference route，条件性添加metric key；
  - 因此两rank可用不同key数量进入同一个collective。观察到的rank1 105元素reduce与rank0已前进至
    1元素barrier，和这一schema分歧完全一致。当前最可能触发字段是正/非正reward分组；本次32个抽样
    `actor_switch`均为false。
- 结论：这是新telemetry的**分布式固定schema缺口**，不是DVAC数学、RLT objective、显存或RAM问题。
  窄修方向是让所有条件组始终输出同一组固定sum/count字段（或固定键及presence count），再重跑同一
  smoke；不需要改`[0,2]`、H50/C10、actor/critic/BC-Q或成功RLT主体参数。本轮遵守边界，未改source、
  未重跑。
- 评估路径源码同时确认：`rlt_dvac_mode=apply`时`get_rlt_obs()`无论train/eval都会设置
  `collect_endpoint_previews=True`，而runner eval复用同一rollout model。因此修复后若按当前配置正式跑，
  fixed-20 eval也会计算teacher-DVAC；是否给eval单独关闭采集是正式运行前的工程开销决策，不在本轮
  自动改变。
- 资源CSV离线汇总：899条约2秒样本、区间2029秒；GPU显存峰值`17,111/17,193 MiB`，两卡util峰值
  100%；cgroup峰值`54.68 GiB`，host available最低`910.64 GiB`；high/max/OOM/OOM-kill增量全0。
  退出后driver/monitor和所有匹配worker均消失，两卡回到`0 MiB`。
- 8个NPZ合计320个weight的min/p05/median/mean/p95/max为
  `0/0.2584/0.9810/1.0247/1.8670/2`，per-query weight ESS均值`0.8891`，top-20% weight mass
  均值`0.2904`；0/2边界命中率`0.94%/3.44%`。这证明`[0,2]`不是只在配置中存在。
- 指令：使用单个固定host-key Paramiko SFTP连接下载driver、两rank actor stderr、resolved/command/
  preflight/unit logs、resources和8个NPZ；本地证据目录共24个文件、约0.30 MiB：
  [rlt_dvac_real_smoke_8env1c_20260824](rlt_dvac_real_smoke_8env1c_20260824/README.md)。不含checkpoint、
  视频或批量Ray日志；本次也没有checkpoint可下载。

### 3.6 v1根因的窄修复

- 只修改三处：`dvac_weighting.py`新增空组也可返回固定sum/count的helper；worker的positive/nonpositive与
  student/reference四组指标始终输出同一schema；单测新增一项空组/非空组覆盖。
- 远端指令：对修改文件运行`ruff check`、`git diff --check`和
  `pytest -q tests/unit_tests/test_rlt_dvac_weighting.py`。
- 结果：ruff/diff通过，`5 passed in 1.97s`。未改DVAC公式、RLT objective、`[0,2]`或smoke配置。
- commit：`513dbcb7f31ebb639afa5267dff218028d07187b`，message=
  `fix(rlt): keep DVAC metric schema rank-stable`；worktree clean。

### 3.7 同合同重跑v2

- 输出改为新目录`rlt_teacher_dvac_w0to2_smoke_8env1c_20260824_v2`，其余resolved合同与v1一致。
- 启动：`2026-08-24T18:09:38+08:00`；结束：`18:17:51+08:00`；wall约8分13秒；driver `exit=0`。
- 训练链：8-env rollout用时121.4秒；8 critic / 4 actor update完成；两个rank都写update 0/2/4/6的
  DVAC trace；最终metric all-reduce自然返回。
- 评估链：4 eval env×5 epoch共20 episodes，用时268.2秒；success 0/20。该数字只说明1-cycle未训练student，
  不用于比较方法效果。
- 保存链：`global_step_1`共约54 MiB，模型、target、replay、两rank trainer state和complete manifest齐全。
  两rank`update_step=8`；嵌套`rlt_dvac_baseline`完全一致：count=1440、mean=-4.8754106、std=0.5060390、
  frozen=true；complete manifest=`complete=true`。

### 3.8 v2方法与资源后检

- 最终metrics：actor/critic grad norm=`3.804/3.259`；actor/critic loss finite；run级weight
  p05/median/mean/p95=`0.263/0.959/0.991/1.820`，ESS=`0.880`，top-20% mass=`0.298`。
- 本机使用bundled Python只读聚合8个NPZ：320个weight的min/p05/median/mean/p95/max=
  `0/0.3007/0.9962/1.0108/1.8251/2`；per-query ESS均值=`0.8871`，per-query top-20% mass均值=`0.2946`；
  down/up=`50.94%/49.06%`。metrics与NPZ数值范围一致；前者是run聚合，后者是每两次update抽样。
- 218条资源样本：GPU峰值`17,111/17,193 MiB`，cgroup峰值`57.18 GiB`，host available最低
  `907.45 GiB`，OOM/OOM-kill=`0/0`。退出后GPU为0且无匹配训练/Ray进程。

### 3.9 轻量证据与云端同步

- 指令：一个固定host-key Paramiko SFTP连接下载resolved、driver/metrics、生命周期、resources、8个NPZ和
  两rank5 KiB trainer-state/complete manifest；不下载54 MiB模型/replay checkpoint正文。
- 本地入口：
  [v2 README](rlt_dvac_real_smoke_8env1c_20260824_v2/README.md)与
  [analysis_summary.json](rlt_dvac_real_smoke_8env1c_20260824_v2/analysis_summary.json)。
- GitHub直连push两次无进展后停止；随后复用既有一次性loopback CONNECT relay，Git TLS与凭据仍由服务器Git
  端到端处理。push成功：`275ab453..513dbcb7`；最终personal branch与worktree ahead/behind=`0/0`。
- 最终现场核对首次尝试把含`$(git status ...)`的远端shell放进PowerShell双引号；PowerShell在本机提前解析，
  只产生本机`dubious ownership`与helper参数错误，远端命令未执行、环境未改变。改用
  `tmp/rlt_dvac_smoke_v2_postpush_audit_20260824.sh`命令文件后通过：HEAD=upstream=`513dbcb7...`、
  ahead/behind=`0/0`、两卡`0 MiB`、cgroup current约9.46 GiB、OOM/OOM-kill=0；进程匹配仅为本次只读探针自身。

### 3.10 本轮停止点

- RLT teacher-DVAC真实smoke已通过；source、tests、训练、eval、指标归约、权重trace、资源和checkpoint均闭环。
- 本轮没有启动正式RLT训练。若下一轮完整对齐历史预算，必须明确使用历史
  `fresh 1--250 -> resume 251--480`两段合同。
