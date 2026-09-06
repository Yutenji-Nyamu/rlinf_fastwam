# RLT teacher-DVAC 实现与 smoke 前检查流水账（2026-08-24）

## 1. 本轮授权与边界

- 目标：把已讨论的 DVAC action-level 训练权重迁移到此前成功的 RLT Stage 2。
- 已授权：创建独立 worktree/branch、实现、少量必要的 smoke 前检查、普通 commit 与 push、只读审计服务器是否存在漏推的有用改动。
- 本轮不做：正式 smoke、正式训练、停止或修改正在运行的 global-z 训练、安装依赖、下载模型、删除或覆盖历史产物。
- 运行隔离：新 RLT worktree 不复用当前 global-z 训练的源码目录；项目相关检查以不启动 Ray、环境或模型为限，资源现场不允许时继续收窄。

## 2. 冻结实现语义

1. 冻结的 pi0 teacher 在原有参考动作 ODE 中旁路产生 `V_L2/V_L3/V_L4 [B, 50]`，不增加一次 teacher forward。
2. RLT student 实际输出 `C=10`，首版用 `V_L3[:, :10]`。
3. 训练基线为 warmup replay 上的 global z-score；权重映射为 `w=1+0.5*clip(z,-2,2)`，即 `[0, 2]`。
4. 权重只作用于 student action 经 critic 返回 actor 的 Q 梯度；student 前向动作、Q 数值、critic TD loss 与 BC 分支保持原逻辑。
5. `off | observe | apply` 由配置控制；原成功 RLT 配置默认不启用。

## 3. 逐指令记录

### L001 上下文恢复（本机，只读）

- 指令/操作：完整读取 `PROJECT_CONTEXT.md`、`HANDOFF.md`、专题 `00_INDEX_AND_PLAN.md`，以及 RLT-DVAC 规划文档 26、28 和既有规划账本。
- 结果：确认成功 RLT Stage 2 基线记录为服务器 `/root/autodl-tmp/RLinf_rlt_pi0_robotwin`、分支 `codex/rlt-pi0-robotwin`、历史 HEAD `2b8199d8...`；这些动态事实仍需现场复核。
- 设计结论：实现目标和挂点已足够清晰，可以进入实现；不需要再新增算法选择。

### L002 并行只读源码复核（本机）

- 指令/操作：分别复核 RLT actor/replay/state 流程、既有 GRPO-DVAC 可复用代码、Git/worktree 审计范围。
- 结果：
  - 可窄复用：OpenPI endpoint preview、`V_L2/V_L3/V_L4` population variance、global-z 数学和 straight-through 公式。
  - RLT 新增：冻结 warmup moments、RLT action-gradient helper、RLT 专属记录；不复制 PPO ratio/clip、advantage、log-prob、GRPO writer 和 5-step rolling baseline。
  - RLT actor 的默认 Q 与 CrossQ 路径可共用同一 `pi_for_q`；BC 继续使用原 `pi`。

### L003 当前训练与资源刷新（AutoDL，只读）

- 远端指令：`remote_exec_autodl.py run --command-file tmp/idea2_global_z_w0to2_live_refresh_20260824.sh`。
- 时间：`2026-08-24T14:50:57+08:00`。
- 结果：global-z 正常完成 `47/100`；wrapper/driver/observer 与 6 个核心 worker 存活；fatal `0`、OOM/OOM-kill `0/0`；两条 traceback 均是已知可选 Curobo import。
- 资源：GPU 约 `22.5/22.1 GiB`（随后 Git 审计瞬时约 `26.3/26.1 GiB`）；cgroup `243.77/257.70 GB`，处于高水位；数据盘可用约 `630 GB`。
- 决策：本轮只允许 Git、源码传输、语法/纯函数级轻检查；不启动 Ray、环境、模型或 tensor-heavy 项目检查。

### L004 RLT/Git 现场刷新（AutoDL，只读）

- 远端指令：`remote_exec_autodl.py run --command-file tmp/rlt_dvac_readonly_audit_20260824.sh`。
- 结果：
  - RLT base：`/root/autodl-tmp/RLinf_rlt_pi0_robotwin`；branch `codex/rlt-pi0-robotwin`；HEAD `2b8199d8ab2e7b110994fd3234bf7007196c3af9`；tracked/untracked 均 clean；upstream `personal/codex/rlt-pi0-robotwin`；ahead/behind `0/0`。
  - 新路径 `/root/autodl-tmp/RLinf_rlt_teacher_dvac` 不存在，可以安全创建。
  - 已列出的 DSRL、DVAC-v2/v3/global-z、OGPO、QAM worktree 均 clean 且与各自 upstream `0/0`。
  - `RLinf_idea2_dvac_pi0_robotwin` 没有配置 upstream；`RLinf` 主 worktree 有若干旧未跟踪 YAML/local scripts。这两处是否属于漏推的有用源码，后续单独只读核对，不混入 RLT 提交。

### L005 创建隔离 worktree（AutoDL，已授权写入）

- 远端指令：`remote_exec_autodl.py run --command-file tmp/rlt_dvac_create_worktree_20260824.sh`。
- 操作：从精确 base `2b8199d8ab2e7b110994fd3234bf7007196c3af9` 创建
  `/root/autodl-tmp/RLinf_rlt_teacher_dvac`，branch=`codex/rlt-teacher-dvac-weighting`。
- 结果：新 worktree 初始 clean；只增加 Git worktree 元数据，没有切换或改写当前 global-z 训练使用的 source。

### L006 主体实现（本机副本编辑 → SFTP 小文件上传）

- 本机副本：`tmp/rlt_dvac_impl_source/`；所有文本修改均用 `apply_patch`，再由
  `tmp/rlt_dvac_batch_transfer_20260824.py put` 上传到新 worktree。
- 新增/修改共7个提交文件：
  1. `rlinf/algorithms/rlt/dvac_weighting.py`：population variance、冻结global-z moments、`[0,2]`
     映射、student-action straight-through梯度挂点和轻量权重统计；
  2. `openpi_action_model.py`：在原M4 teacher ODE旁路产生`V_L2/L3/L4 [B,50]`，`off`时不采集；
  3. `rollout.py`、`transition.py`：可选`teacher_dvac_v/version/route`进入compact transition/replay，原必填
     `RLT_OBS_KEYS`不变；
  4. `fsdp_rlt_ac_policy_worker.py`：warmup moments跨rank冻结/恢复、取L3前C10、default/CrossQ仅Q分支接
     `pi_for_q`、BC与critic原样、训练指标与低频NPZ；
  5. opt-in `8env250_teacher_dvac_w0to2.yaml`：继承成功RLT配置，只新增DVAC块与实验名；
  6. `test_rlt_dvac_weighting.py`：公式、映射、反向倍率和optional replay字段测试。
- 结果：π0 teacher仍冻结且不增加forward；student前向动作、Q前向值、BC、critic TD loss均保持原路径，只有
  `Q -> action_h -> student`的反向贡献乘`w_h`。

### L007 独立代码审查与窄修复

- 审查发现并修复两项真实兼容问题：
  1. 共享sync/async mixin原先会在`mode=off`也调用新hook。修复为共享off no-op；async只在显式非off时拒绝，
     成功的同步RLT路径实现完整DVAC逻辑。
  2. `rlt_dvac={}`原先无条件进入resume合同，会改变旧off配置的hash。修复为只有非off时写入合同，旧RLT
     checkpoint恢复合同保持原样。
- 同轮收口：明确producer固定`L=(2,3,4)`顺序；raw trace中的student/reference动作写成`[B,10,14]`；增加
  `transition=True` optional telemetry往返测试。
- 审查结论：同步正式主路径shape为`[B,4,50,14] -> [B,3,50] -> L3前10 [B,10]`，与student
  `[B,10,14]`一致；baseline冻结/恢复、default/CrossQ与BC/critic边界未见阻塞问题。

### L008 静态检查与格式化（AutoDL）

- 远端指令：`remote_exec_autodl.py run --command-file tmp/rlt_dvac_review_fix_check_20260824.sh`。
- 结果：
  - `ruff format`格式化1个文件；随后`ruff check`全部通过；
  - `git diff --check`通过；
  - 6个相关Python文件用`compile()`做无pycache语法检查，全部`SYNTAX_OK`；
  - opt-in YAML此前已由PyYAML解析，确认`strength=0.5`、`z_clip=2.0`。
- 同时现场：global-z wrapper/driver/observer均alive；GPU约`27.1/26.5 GiB`；cgroup约`244.0 GB`；
  `oom/oom_kill=0/0`。没有启动Ray、RoboTwin或模型。

### L009 CPU-only 窄单测（AutoDL）

- 指令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
  tests/unit_tests/test_rlt_dvac_weighting.py`，60秒timeout。
- 首次问题：命令没有先进入新worktree，Python解析到另一份RLinf，collection报
  `ModuleNotFoundError: rlinf.algorithms.rlt.dvac_weighting`；代码未执行。
- 修复：只把cwd与`PYTHONPATH`指向`/root/autodl-tmp/RLinf_rlt_teacher_dvac`后重跑同一文件。
- 结果：`4 passed in 1.91s`；cgroup前后约`244.03/244.02 GB`，GPU不可见且没有新增OOM事件。

### L010 提交与云端推送

- 提交前：精确add 7个文件，`git diff --check`通过；没有把其他worktree或运行产物带入。
- commit：`275ab4535f8363e15b839cbb64e7d997a056d01e`，message=
  `feat(rlt): add teacher DVAC Q-gradient weighting`。
- 云端：`personal/codex/rlt-teacher-dvac-weighting`；本地branch已设upstream，ahead/behind=`0/0`，worktree clean。
- 问题与处理：push本身成功，但脚本末尾直连GitHub的`ls-remote`无输出并留下只读Git子进程；只读状态已先证明
  upstream `0/0`，随后按完整cmdline核对并只对该精确PID发TERM，未触及训练进程。

### L011 服务器“漏推”只读审计

- 指令：`remote_exec_autodl.py run --command-file tmp/forgotten_cloud_readonly_audit_20260824.sh`。
- 结果：
  - DVAC inference `61996e15...`虽然本地未设置upstream，但`personal/codex/idea2-dvac-pi0-robotwin`的
    remote-tracking SHA完全相同；不是漏推。
  - RoboTwin control-trace `43696bba...`与`origin/codex/idea2-dvac-control-trace`为`0/0`；仅有预期
    `RoboTwin_RLinf/assets`链接；不是漏推。
  - DSRL、DVAC v2/v3/global-z、OGPO、QAM此前现场均clean且upstream `0/0`。
  - 唯一未纳入云端提交的是老main worktree中的4个早期A800 PPO/GRPO baseline/smoke YAML和一个10 KB资源监视
    helper。它们属于历史迁移材料，不是本次RLT或当前global-z源文件；本轮不把它们混入任何现有分支。

### L012 当前停止点

- `2026-08-24T15:26:29+08:00`只读现场：global-z已打印完整Global Step 48；三根主进程alive；
  `memory.current=245,256,945,664 bytes`，`oom/oom_kill=0/0`。
- 最终复核`2026-08-24T15:37:12+08:00`：RLT worktree clean、upstream `0/0`、没有遗留RLT Git子进程；
  global-z仍为完整g48且三根主进程alive，GPU约`28.1/26.7 GiB`，cgroup约`240.62 GB`，OOM仍为0。
- RLT-DVAC状态：实现、审查、静态检查、4个CPU-only单测、commit与push均完成；**尚未运行真实smoke**。
- 首版低频NPZ记录的是actor-update replay样本中的V/z/w/student/reference/route/version，不包含collection-side
  原始`endpoint/x_chain`。训练与smoke不依赖后者；若以后要复核完整teacher去噪链，再单独加collection trace。

### L013 真实smoke暴露的telemetry schema修复

- v1在两rank均完成8次update后卡于最终metric all-reduce：rank1归约105元素，rank0已进入下一次1元素barrier。
- 根因是新增positive/nonpositive reward、student/reference route指标按rank-local数据条件增删key。
- 窄修为固定sum/count schema，跨rank归约后再派生mean；新增空组/非空组测试。
- 远端结果：ruff/diff通过，`5 passed in 1.97s`；commit=`513dbcb7f31ebb639afa5267dff218028d07187b`，
  已push到`personal/codex/rlt-teacher-dvac-weighting`，worktree clean且upstream `0/0`。

### L014 真实smoke闭环

- 同resolved合同v2于18:09:38启动、18:17:51自然exit0：8 train episodes、20 fixed eval episodes、
  8 critic/4 actor update、两rank8个DVAC trace、完整g1 checkpoint均完成。
- 权重真实非均匀且命中0/2；run ESS=.880，NPZ per-query ESS均值=.887；GPU峰值约17.1/17.2 GiB，
  cgroup峰值57.18 GiB，OOM/OOM-kill=0。
- 完整结果见[32号文档](../32_RLT_DVAC_REAL_SMOKE_RESULT_20260824.md)和
  [真实smoke流水账](RLT_DVAC_REAL_SMOKE_LEDGER_20260824.md)。正式训练未启动。
