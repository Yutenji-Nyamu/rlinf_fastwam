# Rynn WMRL owner 接入审查

2026-10-05。仅审查本地已部署版本的源码副本；未操作服务器、旧训练或 RLT。当前实际身份、路径与 hash 必须以根任务新取的回执为准。

## 最小改动结论

保留原四卡借还、共享 Ray、精确 namespace/job/token/PID 清理和 graphics scope。新实验采用独立 owner/输出/namespace；用同一 owner 管理 WM6/7 与 Rynn4/5 四个 CPU-first HTTP 服务。无需另加一个 Rynn 守护进程。

现有 `opendw_multigpu_owner.py` 的 `owner_main`、`Catalog`、`cleanup`、`resource_snapshot` 已按服务清单启动、监测、记身份、停止和确认卡位。只要 Rynn 实现 `/health` 与同步 `/offload` 合同，可复用这些流程。服务健康响应必须包含 `ok/pid/physical_gpu/is_offloaded`；服务初次 CPU-ready 之前不能建立 CUDA 上下文。

## 明确需要改的 WM 专用假设

| 本地文件与位置 | 现假设 | 新实验适配 |
|---|---|---|
| `multigpu/opendw_multigpu_owner.py:154` | env.service_urls 等于所有 plan.services | 比较 `kind=wm` 的服务；独立 reward_service_urls 对照 `kind=rynn_success` |
| 同文件 `:172` | 服务物理卡恰为 [6,7] | WM 恰为 [6,7]、RM 恰为 [4,5]，各 key/loopback 端口唯一；保留 interpreter/cwd/输出/遮罩校验 |
| 同文件 `:126` | 分数差奖励、coef1、threshold0.9 | 新授权 reward/done 合同独立核验；其余 actor/GRPO/N64/G8/R8/C32/L384 不变 |
| `batch16/batch16_formal_owner.py:18,45,88` | 来源固定 formal-v2，包括 CP 与 graphics scope | 新来源固定为 live B16 owner 的已验证回执；不能改旧文件的常量再用于旧 owner |
| 同文件 `:115,145,186` | 每个服务都是 B16 WM、GPU [6,7] | WM gate 只检查两 WM；Rynn gate 独立记录实际 batch、输出完整性、显存/装卸；不能把 HTTP 队列大小当实际推理 batch |
| `formal/opendw_formal_owner.py:163` | formal protocol 完全等于旧奖励配置 | 新协议允许清晰列出的奖励字段差异；其余保持源配置实际值 |
| `multigpu/rlinf/envs/world_model/opendw_robotwin_env.py:202` | env offload 只等 WM | 同步等待对应 Rynn 完成卸载，才结束 env offload；让现 runner 的训练前等待屏障覆盖 Rynn |

Rynn 必须被登记为 `service_<key>`，继承 owner token/phase。不要单独启动一个未登记的 4/5 卡服务，否则异常归还 RLT 前无法证明它已退出。

## 正常完整 CP 后切换

1. 实现与 CPU 检查完成后，验证旧 owner/driver identity、旧 owner-plan hash 和最新**完整**正常 checkpoint：两 FSDP rank、full_weights 与对应保存完成日志；只见目录不能视为完成。
2. 准备新配置、源文件 pin 和新 owner 目录，旧训练保持运行。不要在旧 source_sha256 引用的文件上原位覆盖，否则会破坏旧冻结链；新源码使用新路径或独立 repo。
3. 使用 `batch16_handoff.py:187–263` 已验证模式：持有旧 lifecycle 的 `operation.lock`，重新确认未开始 RLT return，记录 handoff intent，再对精确旧 owner 发 SIGTERM。
4. 旧 owner 按原流程清 driver/actors/WM；其归还 RLT 尝试因锁被阻止。`direct_continuation.py:165` 要求旧 final 的 recovery_error 是 BlockingIOError、所有登记进程已退出、GPU 上无己方残余，并确认 return 未开始。这个 BlockingIOError 是有意交接证据，不是应盲修重跑的故障。
5. 调用 `rlt_returned_cycle.prepare_adopted` 和 combined `prepare_adopted`，沿原 stopped RLT/checkpoint/namespace 建新借用链；复用现已激活 scope。新 owner 接受身份与 adoption 后才记录转移完成并释放锁。
6. 切换失败时由唯一交接主管执行精确新旧 cleanup，再走原恢复 RLT 分支；不得旧 owner、新 owner、手工恢复三方竞争。

`batch16_handoff.py` 与 `save_current_checkpoint.py` 中的 PID、SOURCE_OWNER 等是旧 formal-v2 的定值，**不得原样执行**。优先使用正常 CP；外部 SIGSTOP＋强制演员保存仅在确有必要且当前两 rank 确认空闲时另适配。

## 初始化与预算

续最新完整 CP 能保留模型、优化器、调度器和全局步数，是改奖励的最小运行变更。新实验名明确“从 B16 CPx 切换 Rynn”；同时记录 `rynn_round = global_step - x`。`runner.resume_dir` 会从路径中的 `global_step_x` 恢复计数，保持原 `max_steps=200` 意味着剩余 `200-x` 轮，不应静默变成另加200轮。

从原 SFT 重新开始适合与旧 RM 做同起点对照，但它是另一条实验、会丢掉当前学习状态并增加总预算。不能让短 smoke 的随机/单轮权重替代原定 formal CP。若根任务选择从 SFT 正式开始，应在新计划明确写出理由和预算。

## 根任务需要刷新或拉取的最少远端信息

- 当前 B16 owner-plan、owner-identity、formal/driver-identity、state；final/error 是否存在。
- 该 plan 的 lifecycle plan 及两 child plan；return-started/resumed-dispatched/gpu*-launched 是否存在。
- 最新已完成正常 CP 路径、两个 rank 文件和 full_weights 大小/mtime、保存完成日志及global step；不要为这次审查重 hash 20GB 权重。
- plan 中实际 base_owner_module、base_formal_wrapper、graphics runtime、lifecycle 模块及 hashes；本地相关源码是否逐字一致。
- 当前 4–7 计算与图形进程身份，且 0–3 上无此次 owner 登记进程；保留其他用户与无关进程。

服务生命周期的少量针对性检查应覆盖：typed 服务路由、CPU-ready 无 CUDA、某个 Rynn 退出时 owner 的失败清理、同步 offload 后训练可用、未知/错卡位拒绝。无需重复旧借还全部验证。
