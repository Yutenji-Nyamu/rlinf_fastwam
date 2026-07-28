# DSRL × π0 × RoboTwin smoke 执行流水账（2026-07-28）

> 性质：本文件只记录已经实际执行的操作、观察结果、问题与修正；设计与命令基线仍以
> `SMOKE_APPROVAL_20260728.md` 为准，不在此重复形成第二份计划。

## 0. 已批准范围与边界

- 目标分支：`codex/dsrl-pi0-robotwin`
- 服务器工作树：`/root/autodl-tmp/RLinf_fastwam_rlinf`
- 实验根目录：`/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_smoke_v1`
- 顺序：fresh 单轮 smoke 通过后，才从 `global_step_1` 做 resume 单轮 smoke。
- 关注：训练主链能否跑通、DCP/replay/target-shadow 恢复、GPU 显存、GPU 利用率、cgroup 内存与异常事件。
- 禁止：不进入正式训练；不改仓库名或目录名；遇到失败只保留最小证据并先定位一个直接原因。
- 凭据：SSH 密码只注入当前客户端进程，不写入本文件、脚本或服务器。

## 1. 操作流水

### OP-001　恢复上下文并确认授权边界

- 时间：2026-07-28 17:04 CST
- 操作：完整读取 `PROJECT_CONTEXT.md`、`HANDOFF.md`、权威 smoke 批准包和实施流水账尾部。
- 结果：用户已明确批准执行 fresh 与 resume smoke；正式训练仍未获批准。
- 决定：先做服务器只读 preflight，fresh 通过后才启动 resume；全程采集资源与 DCP 证据。

### OP-002　建立独立 smoke 执行流水账

- 时间：2026-07-28 17:05 CST
- 操作：新建本文件。
- 结果：后续每个服务器命令、主要输出、错误、修正和资源结论均追加到这里。
- 备注：本地 Git 容器仓库本身为未初始化提交状态且全部项目材料显示为未跟踪；这不影响服务器目标 worktree 的分支/提交核验，服务器状态以 live preflight 为准。

### OP-003　服务器只读 preflight

- 时间：2026-07-28 17:06 CST
- 命令文件：本地 `.tmp/remote_dsrl_smoke_preflight.sh`，经带主机指纹校验的 Paramiko helper 执行；未向服务器写入。
- 身份：`root@autodl-container-nekaqbwt43-6ce5babb`。
- Git：
  - branch：`codex/dsrl-pi0-robotwin`
  - HEAD：`2d942b714b004de9a7efdbd4a7e2efaac3ef6d01`
  - upstream：`personal/codex/dsrl-pi0-robotwin`
  - worktree：干净；远端分支哈希与 HEAD 相同。
- 配置证据：
  - fresh SHA-256：`2c616159726dc9d39fbe4d011909dcba963c95c355539c606c718a6ed25da390`
  - resume SHA-256：`c81e722d72dc06976f1011f42de1e9701c18d33857dfa1cc296af5216a491d46`
  - 目标 run root 不存在，满足 fresh 防覆盖条件。
- 运行时：Python `3.11.14`，PyTorch `2.6.0+cu124`，CUDA `12.4`，Ray `2.55.1`，Hydra `1.4.0.dev1`；目标 `.venv`、RoboTwin、模型、配置及监控脚本均存在。
- 进程/GPU：无目标训练、Ray 或资源监控进程；两张 A800 80 GB 均为 `0 MiB / 0%`。
- cgroup：
  - `memory.current=222,568,542,208` B，但 `inactive_file=219,224,809,472` B、`anon=307,798,016` B；
  - `oom=0`、`oom_kill=0`、`max=0`，判定高表面占用主要是可回收文件缓存，不是启动阻塞。
- 磁盘：`/root/autodl-tmp` 共 1.9 TB，已用 994 GB，余 851 GB；最大目录为目标 worktree 378 GB、主 RLinf 150 GB、RoboTwin 112 GB、旧 WAM-PPO 备份 111 GB、models 91 GB。
- 结果：`PREFLIGHT_OK=1`，允许进入已批准的 fresh smoke。

## 2. Fresh smoke

### FRESH-001　启动

- 时间：2026-07-28 17:07:08 CST
- 精确参数：按 `SMOKE_APPROVAL_20260728.md` 的 fresh 命令执行；服务器同时把 shell-escaped 命令保存为 `fresh_command.txt`。
- driver PID：`40184`
- resource monitor PID：`40185`
- 日志：`fresh_driver.log`
- 资源采样：`resource_monitor/fresh/`，间隔 2 秒。
- 状态：已启动，等待模型/Ray/RoboTwin 就绪与首轮 rollout/update。

### FRESH-002　启动期采样

- 时间：2026-07-28 17:08:50 CST（启动后 102 秒）
- 进程：driver、Ray GCS/raylet、2 个 actor、2 个 rollout、2 个 env worker 均已创建；driver 存活。
- 日志进展：两侧 rollout worker 已加载 RoboTwin normalization，并明确打印 DSRL 冻结 Gemma expert 与 π0 projection 层。
- 暂见告警：
  - SAPIEN 自动补 Vulkan ICD；
  - RoboTwin `planner.py` 的可选 Curobo/PyTorch3D 导入失败。
  这两项来自环境启动时的可选路径，driver 未退出；继续观察实际 task reset/step 后再判定是否无害。
- GPU：当时两卡各约 `7,689 MiB`，尚处模型初始化阶段；无 OOM。
- cgroup：
  - 监控器记录到保守 `memory.current` 峰值 245,760 MiB（等于 cgroup 上限），但 `oom=0`、`oom_kill=0`；
  - 同时现场 `anon≈18.6 GB`、`inactive_file≈211.5 GB`，说明峰值仍主要由可回收文件缓存构成；
  - 内存 PSI `avg10=0.53`，有轻度回收压力但尚无失活或 OOM。

### FRESH-003　首轮 rollout 完成

- 时间：2026-07-28 17:11:51 CST（启动后 284 秒）
- 调用链进展：
  - online actor 两个 rank 已完成 DSRL 模型/FSDP 初始化；
  - rollout `1/1` 于 37.64 秒完成；
  - actor 正在接收 rollout transitions，说明三相机冻结 π0 → RoboTwin step → compact replay 投影主链已越过首次环境交互。
- 资源：
  - 截至该时刻单卡峰值约 `33,507 / 33,485 MiB`，两卡合计峰值 `66,992 MiB`；
  - 当前约 `28.6 GB/卡`，采样中可见有效 GPU 利用率；
  - cgroup 仍无 OOM/kill，现场 `anon≈33.9 GB`、`inactive_file≈201.9 GB`。
- 结果：rollout 阶段通过；等待 SAC 更新、eval 与 DCP。

### FRESH-004　补齐 run 内自描述证据

- 时间：2026-07-28 17:11:38 CST
- 操作：不改代码和配置，只把 fresh/resume resolved YAML 复制到 run root，并写入 `run_provenance.txt`（branch、HEAD、upstream、干净状态、运行时版本、哈希、driver cwd）。
- 结果：证据哈希仍匹配批准包；driver cwd 实测为 `/root`。
- 说明：resolved config 的 RoboTwin `save_path` 是相对路径 `./data`。本轮 fresh 已从 `/root` 启动；该任务实际 `save_data=false`，训练视频也单独使用绝对 log path。为了不在 fresh/resume 对之间引入 cwd 差异，resume 将保持同一 cwd；正式运行前再把相对路径改为绝对实验目录。

### FRESH-005　SAC、eval 与 DCP1 完成

- 时间：2026-07-28 17:20:03 CST
- 生命周期：
  - 资源监控覆盖 365 个 2 秒样本、775 秒；
  - 一轮 RLinf step 为 534.49 秒，其中 rollout 39.45 秒、SAC `run_training` 409.41 秒、weight sync 10.20 秒、eval 47.34 秒；
  - eval 本体 1 个 rollout epoch，日志耗时 29.77 秒；
  - driver 正常消失，monitor 自动补终态样本，两卡显存归零。
- 数据/更新：
  - `global_new_transitions=40`
  - `global_resident_transitions=40`
  - `planned_optimizer_updates=800 = 20 × 40`
  - 两个 replay rank 各 resident/inserted/cursor `20/20/20`。
- 训练指标（均 finite）：
  - critic loss `0.01672`，critic grad norm `0.1955`；
  - actor loss `-17.6198`，actor grad norm `4.4349`，entropy `21.8306`；
  - alpha loss `35.0944`，alpha `0.9263`，alpha grad norm `22.8680`；
  - 10 个 Q head 与 `q_pi` 均 finite。
- eval：4 条轨迹，`success_once=0.75`、`success_at_end=0.75`。这是 smoke 的小样本健康信号，不解释为算法效果。
- DCP1：`global_step_1` 完整写出，目录约 32 GB；online local shard、alpha DCP、target、trainer state、flat replay 均有两个 rank 的预期产物，且无残留 `.tmp`。

### FRESH-006　DCP/replay/资源离线验收

- 时间：2026-07-28 17:20:40 CST
- trainer state：
  - 两 rank 均为 schema 1、world size 2、phase 1、`update_step=800`、pending new=0；
  - critic-only FP32 shadow 为 156 个 tensor、2,672,362 参数、约 10.69 MB；
  - 两 rank shadow SHA-256 同为 `8e377d3f1311bcb0b76325dfef158669998a14832e9a6137df00b0c97f9d64ad`，全部 finite。
- flat replay：
  - 每 rank 分配 `616,737,500` B（全局约 1.233 GB），有效 20 行约 0.987 MB；
  - main image BF16 `[12500,3,64,64]`，state FP32 `[12500,14]`，latent BF16 `[12500,32]`；
  - reward 全为 -1、termination 全 false、continuation 全 true；truncation 同时含 false/true；
  - discount 全为 float32 `0.999^20 = 0.9801888466`，证明纯 truncation 仍 bootstrap。
- 边界：本轮训练 rollout 没有 success termination，因此 success→reward 0 的真实环境样本未被触发；该分支只由此前集中单测覆盖，不能从本次 replay 过度声称。
- 资源：
  - 单卡显存峰值 `34,563 / 34,532 MiB`，两卡合计 `69,093 MiB`；
  - actor/rollout/env RSS 峰值分别约 `32.15 / 30.15 / 9.96 GB`；
  - cgroup `memory.current` 曾触及 240 GB 上限，但现场峰值期匿名内存约 41 GB，其余主要是可回收 file cache；
  - 全程 `oom=0`、`oom_kill=0`，无 CUDA OOM、NaN/Inf 或 worker crash。
- 结论：fresh smoke 通过，允许进入 resume smoke。

## 3. Resume smoke

### RESUME-001　防覆盖 gate 与启动

- 时间：2026-07-28 17:22:32–17:25:14 CST
- gate：
  - DCP1 的 online local shard、alpha DCP、target、trainer state、replay 两 rank 文件均完整；
  - DCP2、resume 日志/PID/资源目录均不存在；
  - 无残留目标训练/Ray/monitor 进程，两卡显存为 0；
  - 对 DCP1 全树生成 `ckpt1_before_resume.sha256`，32 GB 读取耗时约 162 秒，用于 resume 后证明源 checkpoint 未被改写。
- resume 参数：`resume_dir=.../global_step_1`、`runner.max_steps=2`，因此 runner 只执行 global step 1 这一轮并保存 DCP2。
- driver PID：`53085`
- resource monitor PID：`53086`
- 启动时间：2026-07-28 17:25:14 CST
- 重点失败信号：legacy/non-bitwise fallback、phase/shadow/replay layout mismatch、DCP1 hash 改变、DCP2 不完整。

### RESUME-002　恢复完成并进入第二轮更新

- 时间：2026-07-28 17:30:43 CST
- 日志：runner 明确打印从 `global_step_1` 恢复；阻塞式 `actor.load_checkpoint(...).wait()` 已返回，随后 rollout `1/1` 在 39.21 秒完成并进入 `EmbodiedSACFSDPPolicy.run_training`。
- 恢复异常：未出现 legacy/non-bitwise fallback、load failure、layout/phase/shadow mismatch。
- 说明：FSDP strategy 的细粒度 local-shard/DCP load 日志只在 class logger 已设置时输出，本次 driver log 未出现这两行；不把“日志行缺失”误判为未加载，最终以 DCP1→DCP2 state/replay 精确连续性为强证据。
- 资源：checkpoint load/同步期单卡峰值约 34.74 GB；训练时约 27.92 GB/卡；cgroup OOM/kill 仍为 0。

### RESUME-003　第二轮更新、eval 与 DCP2 完成

- 时间：2026-07-28 17:37:46 CST
- 生命周期：
  - 一轮 RLinf step 为 `511.17` 秒，其中 rollout `40.47` 秒、SAC `run_training` `381.04` 秒、weight sync `10.05` 秒、eval `50.91` 秒；
  - driver 正常退出，资源监控覆盖 353 个 2 秒样本、752 秒；
  - DCP2 在 17:37:13 开始保存并完整落盘，目录约 32 GB，无残留 `.tmp`。
- 数据/更新：
  - `global_new_transitions=37`
  - `global_resident_transitions=77`
  - `planned_optimizer_updates=740 = 20 × 37`
  - rank 0 从 20 增至 40 条，rank 1 从 20 增至 37 条。
- 训练指标（均 finite）：
  - critic loss `0.00923`，critic grad norm `0.0926`；
  - actor loss `-11.7986`，actor grad norm `5.1551`，entropy `21.8353`；
  - alpha loss `30.0623`，alpha `0.7933`，alpha grad norm `20.7417`；
  - 10 个 Q head、`q_pi` 与 replay/TensorBoard 指标均 finite。
- 训练 rollout：4 条轨迹，`success_once=0.25`。eval 为另 4 条固定小样本轨迹，`success_once=0`；两者都只作为链路健康信号，不据此判断算法效果。
- 资源：
  - 单卡显存峰值 `34,789 / 34,737 MiB`，两卡合计 `69,474 MiB`；
  - actor/rollout/env RSS 峰值分别约 `32.86 / 32.06 / 12.26 GB`；
  - 全程 `oom=0`、`oom_kill=0`，无 CUDA OOM、NaN/Inf 或 worker crash。

### RESUME-004　trainer state 与 flat replay 连续性验收

- 时间：2026-07-28 17:39 CST
- trainer state：
  - 两 rank 的 `update_step` 均从 `800` 精确恢复并推进到 `1540 = 800 + 740`；
  - phase 始终为 learned phase 1，pending new 始终为 0；
  - FP32 target shadow 两 rank 相同，SHA-256 从 DCP1 的 `8e377d...d64ad` 变为 DCP2 的 `ecec15...1f14`，证明 resume 后 EMA 沿恢复状态继续推进，而非被 fresh shadow 覆盖；
  - 未触发 legacy/non-bitwise 兼容恢复告警。
- replay：
  - DCP1 的全部 40 条有效前缀在 DCP2 中逐字段 bitwise 不变；
  - 新增 37 条 learned latent 的绝对值最大值为 `0.99609375`，均位于 `[-1,1]` 且非全零，证明 resume rollout 没有退回 Gaussian phase；
  - 两 rank 的 replay sampling RNG 均可从 DCP1 状态按 740 次本地 batch 抽样精确重放到 DCP2；
  - 新样本仍全部满足 32D latent、14D state、`continuation == ~termination` 和 float32 `discount=0.999^20`。
- 成功/截断语义：
  - rank 1 的新增 replay 首次出现真实 `reward=0`、`termination=true` 样本，覆盖 success 投影；
  - 新增样本同时含纯 truncation，且 continuation 仍为 true，验证 time-limit truncation 继续 bootstrap；
  - 因而 fresh 阶段“success 分支仅有单测证据”的边界已由 resume 真实环境样本补齐。
- 结论：resume 主链、phase、target shadow、optimizer update step、replay 内容与 replay RNG 连续性全部通过。

### RESUME-005　冻结基座、DCP1 防污染与终态验收

- 时间：2026-07-28 17:40–17:47 CST
- 冻结基座：
  - DCP1 online 与 target 的 778 个冻结 π0 tensor 完全相同；
  - DCP1→DCP2 的 778 个冻结 π0 tensor、共 4,028,019,472 参数，changed tensor 数为 0；
  - 小 actor 32 个 tensor 中 27 个变化，小 critic 156 个 tensor 中 119 个变化；
  - actor/critic 两个 optimizer 在 DCP1 和 DCP2 中均有完整、finite 且非零的一阶/二阶 moments。
- DCP1 防污染：
  - resume 前生成的 11 文件全树 SHA-256 manifest 在 resume 后逐项 `OK`；
  - 证明 resume 只新建 DCP2，没有就地改写 DCP1。
- 终态：
  - fresh/resume driver 和 monitor 均已退出，二次使用避开自匹配的进程扫描确认没有目标 Ray/worker/monitor 残留；
  - 两张 A800 均为 `0 MiB / 0%`；
  - 服务器功能分支、HEAD 与 upstream 均为 `2d942b714b004de9a7efdbd4a7e2efaac3ef6d01`，worktree clean；
  - `/root/data` 不存在，确认相对 `save_path: ./data` 没有产生旁路数据；
  - smoke run root 共约 63 GB，磁盘剩余约 789 GB；`/dev/shm` 120 GB 全空。
- 结论：fresh 与 resume smoke 全部通过；未启动正式训练。

## 4. 资源与并行度结论

### 4.1 实测吞吐与瓶颈

| 项目 | fresh | resume |
|---|---:|---:|
| 新增 macro transitions | 40 | 37 |
| optimizer updates | 800 | 740 |
| rollout | 39.45 s | 40.47 s |
| SAC training | 409.41 s | 381.04 s |
| updates/s | 1.95 | 1.94 |
| 完整 RLinf step | 534.49 s | 511.17 s |
| GPU 峰值 | 34.56 / 34.53 GB | 34.79 / 34.74 GB |
| 训练期两卡平均利用率 | 22.34% / 24.58% | 25.25% / 24.42% |

主瓶颈是 UTD20 的 SAC 更新，不是 rollout。增加 env 会同时增加 transition 和
`20 × transition` 的更新量，并扩大同一旧策略产生的数据 burst；它不会免费提高每条样本吞吐。

### 4.2 内存解释

- cgroup 原始 `memory.current` 两轮都曾触及 240 GB 上限，但进程匿名工作集峰值约
  34–41 GB，其余主要是 checkpoint/model 读取形成的可回收 file cache。
- `oom=0`、`oom_kill=0`，不能把 raw cgroup 峰值解释为泄漏。
- 但 `memory.events max` 在 fresh 由 134,909 增至 250,878，resume 又增至 477,728，
  说明确实发生了频繁回收/限流；因此也不能因为匿名内存较小就直接扩大 env 并发。

### 4.3 第一版正式配置建议

1. 保持 **2 GPU + 4 train env + UTD20 + micro batch 64** 作为已经实测通过的主线；不扩大
   env，也不下调 UTD。
2. `micro_batch_size=128` 有潜在吞吐收益：两卡下可把 gradient accumulation 从 2 降到
   1，且显存有余量。但两轮 smoke 不能证明其峰值和数值轨迹安全；如果追求墙钟，应另做一次
   同 global batch 256 的窄吞吐 A/B。若不追加 A/B，正式首跑保持 64。
3. 当前 formal resolved config 仍是占位值：`max_steps=-1`、`val_check_interval=-1`、
   `save_interval=10`。正式 packet 前必须显式冻结预算；不能直接用它启动。
4. 按本次平均每 cycle 约 38.5 transitions / 770 updates，训练内评估建议
   `val_check_interval=13`，约每 10k updates 跑 `4 env × 3 rollout_epoch = 12` episodes，
   接近官方 10 episodes / 10k updates。
5. 每个 DCP 约 32 GB。`save_interval=10` 若长跑会快速耗尽磁盘；首次 100k requested
   primitive-interaction 审阅段约需 130 cycles，建议在下一轮讨论 `13`（恢复更密、约
   320 GB）与 `26`（约 160 GB、恢复间隔更长）的取舍，并明确保留策略后再运行。
6. 正式 packet 把 RoboTwin `save_path: ./data` 改为 run-scoped 绝对路径；本次
   `save_data=false` 且已确认没有旁路产物，但不应把 launcher cwd 继续留作隐含条件。

## 5. 问题与修正索引

1. fresh 启动时出现 SAPIEN Vulkan ICD 自动补全和可选 Curobo/PyTorch3D import
   warning；真实 reset/step、训练和 eval 均通过，判定为本次路径无害。
2. resume driver log 没有 class logger 的细粒度 local-shard/DCP 两行；没有据此猜测，
   改用 trainer/shadow/replay/RNG/DCP2 连续性作强证据，全部通过。
3. 一次本地 helper 命令因嵌套 awk 引号被 argparse 拒绝，服务器未执行；随后改用完整
   command-file 取得同一只读数据。
4. 第一份终态 `pgrep` 输出匹配到正在运行的审计脚本自身；随后用 bracketed pattern
   避免自匹配，确认目标进程残留为 0。
