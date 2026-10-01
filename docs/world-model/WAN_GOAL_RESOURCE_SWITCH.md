# 深圳3：Wan Goal 借卡与 RLT 归还方案

**08:57最新：用户起床要求继续WM。新v3 owner已重新借原RLT4–7，绑定最新完整CP125；独立π05 smoke实际placement通过。新的唯一归还链为WM→原Dojo→原四RLT。专题`WAN_GOAL_RESUME_20261001.md`。prepare兼容v2的精确外部RLT归还，不修改历史final，不重放旧stop/restore；9项CPU检查通过。以下暂停/优先级为历史。**

**00:46最新：用户要求暂停WM、恢复原RLT。s185–s189已完成WM/Dojo精确退出与四RLT恢复派发；00:50 root状态USER_RESTORED_RLT。WM自动继续已删除，不再自动启动WM或Dojo。以下为历史调度设计，当前进度见`WAN_GOAL_PAUSE_20261001.md`。**

**2026-10-01最新授权覆盖下文旧恢复顺序：WM第一，Dojo第二，RLT暂不安排。** 新owner不再等待RLT首轮，以精确身份、旧owner终态和释放回执完成交接；使用`--skip-prior-first-round --defer-rlt-restore`。WM结束续原Dojo；Dojo最终结束后保留RLT恢复清单，不自动派发。下文旧“Dojo→RLT”流程仅作为历史与可选能力。00:18已进入OFT smoke启动，运行路由为`continuation-20260930-wan-goal-v1`。

更新：2026-09-30。本文记录实现依据与切换规划；独立胶水已上传深圳3，s024核验原v3源码，s033通过7项服务器CPU检查。尚未发送本次切换信号，现场进程、checkpoint和磁盘状态仍须在实际切换前重新核验。

本轮最新授权：材料准备齐后借深圳3 GPU 4、5、6、7，先做 Wan Goal OFT smoke，再做批准配置的 π0.5 GRPO 正式实验；**WM 完成或故障后恢复 Dojo，Dojo 最终结束再归还原四个 RLT**。共享 Ray、其他用户、GPU 0–3 与无关实验保持原状。停止 RLT 时使用最近的完整 checkpoint，不等待额外补存。

## 1. 当前资源归还机制

本地交接最后记录的 Dojo 路由：

- 项目：`/data/chenyiteng/projects/robodojo-openwam-sz3`。
- 原正式评测：`runs/sz3_pi05_official_6300_n4_dual_20260929_r2`。
- 当前路由以该 run 下 `active-continuation.json` 为准；历史记录是 `continuation-20260930-single-v3`。
- 历史冻结源码：`scripts/eval-recovery-20260930-v3`。
- 历史借卡 cycle：`rlt-cycle-sz3-20260930-single-v3`。
- 预期用户 UID `20001`、主机 `h100-gpu01`；SSH host key 仍须按项目既有固定记录核验。

现有 `continue_pipeline.py` 的 `finally` 会依次清理本次 Dojo、验证四卡释放、调用原 cycle 的 `resume`，并观察四个 RLT 的恢复首轮。它没有“暂停归还”“把 GPU 交给另一个实验”的开关。

**因此不能停止 Dojo 后看到空卡就启动 Wan。** 此时旧 pipeline 仍有归还责任，会启动 RLT，与 Wan 争抢资源。旧 watchdog 只请求结束绑定的 Dojo controller，恢复 RLT 的责任在外层 pipeline。

## 2. 最少改动的切换顺序

1. **借卡前准备齐**：Wan/OFT/π0.5 的代码、依赖、权重、数据、实际解析配置、启动入口和退出清理均已就绪；产物放 `/data`。先冻结新的独立运行目录、日志路径、预算和退出条件，再开始切换。
2. **保留 Dojo 现场**：只读保存当前 active pointer、pipeline 状态、plan、controller/worker 身份、原生结果计数及视频清单。不要删除、移动、重建原正式评测目录，也不要改原任务/seed/回合预算。
3. **精确停止当前 Dojo controller**：读取当前 active pointer，通过当前版本 `hang_watchdog.bound_run()` 验证绑定关系；写一个全新、唯一的授权切换 intent，再调用 `signal_bound(..., SIGTERM)`。不要重放历史停止脚本。
4. **让旧 pipeline 完成归还**：等待原 `dojo-release.json`、`resumed-dispatched.json`、`rlt-first-round.json` 和 pipeline 终态。四个 RLT 均恢复首轮后，再借卡。若归还进入 `RESOURCE_RETURN_NEEDS_ATTENTION`，先按该 cycle 回执处理，不跳过并启动 Wan。
5. **新建一个“先 WM、后 Dojo”的 continuation/cycle**：复用 `reborrow_cycle.py` 的动态发现逻辑，使用新的唯一 cycle、明确的 predecessor cycle、新的 continuation 和独立 WM 子目录；读取刚恢复四个 RLT 的实际 driver/config/namespace/checkpoint。冻结 helper 和 plan。沿用原 Dojo run 的任务、seed、预算、权重、W1×N4 设置和结果/manifests；原脚本与旧 cycle 保留。
6. **精确停这四个 RLT**：执行新 cycle 的 `stop`，获取完整 `rlt-stopped.json`，核实四卡无遗留进程后，新 outer 按已批准顺序运行 OFT smoke 和 π0.5 GRPO。WM 阶段不使用 Dojo 的 `EVALUATING` phase。
7. **WM 完成或故障后续 Dojo**：新 outer 记录 WM 退出状态，只清理本次 WM 的进程及 Ray actors；确认四卡释放并保存 `wm-release.json`。这里**不调用 RLT resume**，而是启动原 Dojo controller，从保留的原生结果与 manifests 续跑。WM 失败也走此路径；若清理未通过，记录 `RESOURCE_RETURN_NEEDS_ATTENTION`，不要启动另一个占卡任务。
8. **Dojo 最终结束后归还 RLT**：仍由这个新 outer 执行既有 Dojo `finally` 的 cleanup/release/resume，并验证四个 RLT 均从记录的完整 checkpoint 恢复、推进一轮。一次借卡 cycle 只有这个 owner 负责最终归还。

这条路线会有一次短暂的 RLT 归还再借出，但完整复用现有恢复机制。`select_recovery()` 会把当前配置的 `runner.resume_dir` 加入 checkpoint 候选，因此不需要等待短暂恢复的 RLT 再保存新 checkpoint。

若要直接 Dojo → Wan → Dojo，需另行实现并验证“归还责任移交”：旧 pipeline 明确接受新的 owner、取消自己的恢复动作、新 owner 持久接管原 cycle。当前代码没有这项协议；不能用杀外层 pipeline、`SIGSTOP` 挂起进程或抢占空卡代替。暂停 outer 不能保证暂停点在 `finally` 前，也不能提供重启后可核验的归还责任。

### 新 continuation 的最小代码改动位置

在复制出的新 `continue_pipeline.py` 中，保留已有的借卡前核验、RLT stop、Dojo 启动及最终 RLT 归还。在 `rlt-stopped.json` 已存在后、原 `state('EVALUATING')` 和 `dojo-controller` 启动前，插入一个有独立退出处理的 WM 阶段：

```text
验证所有准备材料 → 新 cycle 精确停止四个 RLT
    → RUNNING_WM：OFT smoke；符合验收再 π0.5 GRPO
    → 无论 WM 成败，清理已登记的 WM 进程/Ray actors
    → WM 释放通过：保存回执，转入 EVALUATING，续原 Dojo
    → Dojo 结束：原 cleanup → 新 cycle 恢复 RLT → 首轮验收
```

WM 的异常应在该局部阶段记录，不能直接向外抛出并进入旧的“立即恢复 RLT” `finally`。`managed_identities()` 和最终释放检查要包含 WM 已登记身份，并保留独立 WM 清理回执；原 Dojo guard 不能代替这一步。全局 owner 的终止与“仅结束 WM 并续 Dojo”应区分信号/意图，不能让一个旧 `stop_requested` 标志阻止后面的 Dojo 启动。

新 outer 仍用现有的 `pipeline.lock` 排除双 owner；必须在旧 outer 退出后才创建新 active pointer。WM phase 下旧 watchdog 自动空闲；进入 Dojo `EVALUATING` 前保存新 controller identity 与匹配的冻结 plan，使现有 `bound_run()` 能验证新的 parent/controller/config/cycle。保留 `continue_pipeline.py` 入口名，当前 watchdog 按该名称定位 outer。

## 3. 可复用文件与不能直接重放的入口

以下均为本地审查源，执行前须核对远程冻结版本和 SHA：

| 文件 | 可复用内容 | 使用限制 |
|---|---|---|
| `local_scripts/dojo_stability_20260930/stop_sz3_for_lanes.py` | 唯一 intent、`bound_run`、pidfd TERM 的停止模式 | 硬编码历史 v2，不能原样运行 |
| `local_scripts/dojo_stability_20260930/sz3-v3/hang_watchdog.py` | 第 41–75 行绑定当前 owner/controller；第 140–147 行重新验证后 pidfd 发信号 | 只有当前 Dojo 的 `EVALUATING` 状态可绑定；不可改 active pointer 指向 WM |
| `local_scripts/dojo_stability_20260930/sz3-v3/continue_pipeline.py` | 第 163–177 行清理、释放、恢复；第 83–134 行首轮观察 | `finally` 无条件承担归还；不要编辑正在运行的副本 |
| `local_scripts/dojo_stability_20260930/sz3-v3/reborrow_cycle.py` | 第 28–73 行从前一 cycle 实际恢复出的四个 RLT 动态生成新 plan | 需新文件、新 STAGE 和明确 predecessor；原文件路径为 Dojo v3 专用 |
| `local_scripts/dojo_stability_20260930/sz3-v3/rlt_cycle_sz3.py` | checkpoint 检验、精准 stop、幂等 resume、首轮 status | 必须冻结到新 cycle；不可重放旧 stop；`prepare` 含历史路径，不作为通用入口 |
| `local_scripts/dojo_stability_20260930/sz3-v3/process_guard.py` | UID、boot/start、已知进程树等身份核验方式 | 主要靠 `DOJO_*` 环境标识找进程，原样不能证明 Wan/Ray 已清理 |

现有 watchdog 与新的 WM 无需争用归还责任：旧 Dojo 结束并归还后，watchdog 转入 `IDLE_RLT_OR_FINISHED`；新 continuation 的 WM phase 也保持空闲。Dojo 续跑进入 `EVALUATING` 后，watchdog 按原 run 下新的 active pointer 动态绑定，所有身份与哈希检查仍须通过。WM 使用独立子目录和进程标识。

## 4. 进程、checkpoint 与释放回执

停止或清理前记录并复核：host/UID、boot ID、PID 与 `/proc` start time、命令哈希、父子关系；Ray 还要记录 namespace、job ID、actor ID、actor 所属 UID。目标 GPU 是物理 4–7，不能仅凭 CUDA 局部编号推断。

`rlt_cycle_sz3.py` 的 stop 会验证当前四个 driver、配置哈希、Ray namespace/job、进程树和四卡占用归属；写入 stop intent 后才发信号。停止完成后重新选择最近可验证的 checkpoint，保存到 `rlt-stopped.json`。不要只记一个 `global_step_*` 文件夹名。

完整 checkpoint 的现有验证包括：`complete.json`、resume contract、模型及 optimizer/scheduler/RNG DCP、trainer 累计计数、target model、replay metadata/index 与轨迹文件。存在 checkpoint 但全部损坏时不会静默新开训练。本次归还要求 `recovery.mode=resume_checkpoint`。

WM 的阶段释放回执至少包含 cycle、WM 终态、管理进程身份、Ray 清理及四卡释放证据，用来准许启动 Dojo；**它不触发 RLT resume**。最终 Dojo 释放回执继续沿用现有 helper 接口，至少包含：

- `cycle_id` 精确等于新 cycle 名。
- `terminal_status` 使用 helper 支持的 `completed`、`failed`、`timed_out` 或 `not_started`；授权中止原因另记字段，不能随意传 `paused`。
- `all_workers_stopped=true`，`managed_processes` 为本次实际管理进程身份，验证均已退出。
- 附本次 Ray namespace/actor 清理和物理四卡无计算进程的核验结果。

必须先完成 WM 清理，才能续 Dojo；必须等 Dojo 最终清理完成，才能提交最终 RLT release receipt。全局 `ray stop`、广泛 `pkill`、GPU reset 不属于此次复用方案。

恢复入口模板（占位符须由冻结 plan 解析，不直接复制执行）：

```text
<RLT_PY> -u -B <NEW_CYCLE>/rlt_cycle_sz3.py --cycle-dir <NEW_CYCLE> resume --release-receipt <NEW_CONTINUATION>/dojo-release.json
<RLT_PY> -u -B <NEW_CYCLE>/rlt_cycle_sz3.py --cycle-dir <NEW_CYCLE> status
```

现有 RLT Python 路由是 `/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/bin/python`，执行前按实际 plan 确认。`resumed_name()` 已处理多次恢复导致目录名过长，应直接复用。

**恢复验收不是仅看进程存在。** 现有 `status()` 第 735–765 行逐卡要求：恢复 driver 身份仍匹配；日志明确载入该 checkpoint；update step 与 replay 样本数不低于 checkpoint；真实日志 round 大于 checkpoint 的累计 step。四个 `first_round_verified=true` 才确认归还完成。若 checkpoint 处于 replay warmup，允许该首轮尚未做 optimizer update，但要如实记录 online/update 状态。

## 5. Dojo 原结果与两层日志

暂停前后记录原任务/seed 的已提交原生结果数、成功数、结果文件路径及视频清单；退出清理后再做稳定快照。保留结果 JSON、resume manifest、视频与原日志，之后继续 Dojo 时从其既有记录续跑。最后尚未提交的回合可能重跑，不能把“已启动”算“已完成”。

旧 pipeline 可能把授权停止的 Dojo 记成 `failed`；保留该原始状态，并另写“为已授权 WM 借卡而中止”的事实与 intent 路径。不要修改原生结果伪装评测已完成。WM 完成或故障后的默认恢复对象是 Dojo；Dojo 完成或故障退出时才按其既有机制恢复 RLT。

- 粗日志：准备完成、停止 Dojo 的授权原因、原结果保留、旧 cycle 归还、新借卡、OFT/π0.5 实验结论、WM 清理、Dojo 恢复后首个新增结果、Dojo 最终退出与 RLT 恢复首轮；记录异常及处理。
- 细日志：固定源 SHA、解析配置、命令与退出码、run/cycle 路由、身份与 checkpoint 清单、每个 intent/release/resume/status 回执路径，以及前后 GPU 状态。

仍需现场核验：当前 active continuation 是否仍为 v3；四个实际恢复 RLT 的任务/配置/namespace/checkpoint；是否有其他占用；新 Wan 运行是否使用共享 Ray 及其独立 namespace；Dojo 新 continuation 首个新增原生结果是否继续原计数，最终预算仍为 6,300。本文不提供未经现场验证的 PID 或 checkpoint 名。

## 6. 独立胶水源码与检查

已在 `local_scripts/wan_goal_20260930/resource_switch/` 实现本地独立入口，详见该目录 `README.md`。现有冻结 v3 的六个运行文件按 SHA 复用；只复制并调整新的 continuation，RLT helper 在 prepare 时原样冻结到新 cycle。

新增能力：动态 pidfd TERM 旧 controller；精确 prepare 新 cycle；WM argv 入口、唯一 `WM_OWNER_TOKEN`、独立 Ray 地址/namespace/temp 路由；按 token 和已登记 UID/boot/start/hash 清理；WM 清理回执与四卡释放二次核验；通过后续原 Dojo。WM 命令失败也可在清理通过后续 Dojo；清理无法证明完成时保留现场并报告。

Windows仅完成AST/JSON解析，没有本地项目测试；服务器s033的7项CPU检查通过。`audit_base.py`提供live v3哈希核验与仅供审查的candidate清单；尚未开始真实借卡或GPU恢复验收，不能把这些CPU检查当作归还成功。
