# OGPO × π0 × RoboTwin：逐命令与变更索引

本文件从 2026-08-07 首次 RoboTwin smoke 开始，作为
[`IMPLEMENTATION_LOG.md`](IMPLEMENTATION_LOG.md) 的逐命令附录。主流水账保留问题、原因与修复叙事；
本文件记录实际执行细节。密码只通过执行进程的 `SEETA_SSH_PASSWORD` 注入，所有记录均以
`<process-only secret>` 代替，不保存凭据。

## 记录合同

每次服务器操作记录：操作 ID、开始/结束时间、完整无密码 launcher、command-file 路径与运行时
SHA-256、远端 cwd/目标、exit code、原始 stdout/stderr、服务器副作用、结果、问题与后续处理。
command-file 在执行后不得原地改写；确需修正时创建新文件和新操作 ID。

## 首次 RoboTwin smoke

授权边界：用户于 2026-08-07 明确授权启动首次 smoke，要求 GPU 显存/利用率与 CPU/RSS/RAM
监控；并行轴保持两卡和 8 个 train env，串行训练预算压到 1 个 paired update。调参、pilot 与 formal
预算留到 smoke 后另行讨论。

### 操作流水

| ID | 时间 | 命令与 SHA | 远端目标 | Exit | 结果与原始输出 |
|---|---|---|---|---:|---|
| SMK-0001 | 19:41:22–19:41:24 +08:00 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260807_readonly_refresh.sh`；command SHA `e22dced1…c9d` | 身份、GPU/RAM/磁盘、进程、Git、checkpoint/runtime 只读刷新 | 0 | 两卡空闲、MemAvailable 982 GiB、数据盘可用 816 GiB、目标 `5d5c84e3` clean；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0001_readonly_refresh.log)，SHA `44636009…d9d` |
| SMK-0002 | 19:49:03–19:49:10 +08:00 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260807_smoke_prepare_v1.sh`；command SHA `bf604786…13a` | live preflight、Hydra resolve、runtime packet | 0 | resolved SHA `ebd163f6…f4eb`；创建 evidence runtime，未启动训练、未创建 run root；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0002_prepare.log)，SHA `b2c19ea0…82fb` |
| SMK-0003 | 约 19:50:23–19:50:27 +08:00 | 两次 `remote_exec_autodl.py put`；本地 monitor SHA `40e65af8…368`、run SHA `ba3cf334…9ad` | 上传 immutable monitor/run scripts 到 runtime | 0/0 | 两个 SFTP put 成功；[输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0003_upload.log)，SHA `7ea08137…553` |
| SMK-0004 | 19:51:07 +08:00 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260807_smoke_verify_upload_v1.sh`；command SHA `9f6009e7…971` | 远端 SHA 与 `bash -n` | 0 | 两脚本字节数 3,688/2,412，SHA 精确匹配且语法通过；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0004_verify_upload.log)，SHA `c50280ab…5c8e` |
| SMK-0005 | 19:53:39–19:53:44 +08:00 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260807_smoke_launch_v1.sh`；command SHA `136f646c…013c` | 启动唯一 smoke driver 与 1 秒资源监控 | 0 | driver PID `75160`、monitor PID `75161`；启动前再次校验 clean HEAD、resolved/run/monitor SHA、无冲突进程和空闲 GPU；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0005_launch.log)，SHA `db25d5d9…ecda` |
| SMK-0006 | 19:53:59–19:54:01 +08:00 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260807_smoke_status_v1.sh`；command SHA `1eabc24b…99fe` | 首个运行态快照 | 0 | driver/monitor 均 alive；CSV 已有 15 行；Ray 正常启动、两卡已分配但模型尚未上卡；cgroup `oom=0/oom_kill=0`；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0006_status_initial.log)，SHA `85fe5984…ca04` |
| SMK-0007 | 19:55:45–19:55:47 +08:00 | 与 SMK-0006 相同的 status launcher，输出改为 `smk0007_status_progress1.log`；command SHA `1eabc24b…99fe` | 初始化进度快照 | 0 | driver/monitor alive、CSV 97 行；两卡各 `8267 MiB`，cgroup 约 `47.19 GB`，`oom=0/oom_kill=0`；8 个环境已分组，仍在模型/环境初始化；Curobo 可选导入警告未使进程退出；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0007_status_progress1.log)，SHA `bc76b7cb…dc22` |
| SMK-0008 | 19:57:03–19:57:05 +08:00 | 与 SMK-0006 相同的 status launcher，输出改为 `smk0008_status_progress2.log`；command SHA `1eabc24b…99fe` | 首个 rollout 快照 | 0 | `Generating Rollout Epochs 0/1`；8 个 env 正在 `interact`；两卡约 `21677/21775 MiB` 且出现真实计算利用率，cgroup 约 `40.41 GB`，`oom=0/oom_kill=0`；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0008_status_progress2.log)，SHA `661f7470…0e37` |
| SMK-0009 | 19:58:06–19:58:08 +08:00 | 与 SMK-0006 相同的 status launcher，输出改为 `smk0009_status_progress3.log`；command SHA `1eabc24b…99fe` | rollout/eval/checkpoint 进度 | 0 | train rollout `1/1` 用时 `54.05 s`；post-update eval `1/1` 用时 `14.86 s`；出现 `Saving checkpoint at step 1`；两卡约 `25.7 GiB`、cgroup 现场约 `55.31 GB`，`oom=0/oom_kill=0`；checkpoint 尚在保存，未提前判定完成；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0009_status_progress3.log)，SHA `a27f4c7d…b559` |
| SMK-0010 | 19:59:11–19:59:13 +08:00 | 与 SMK-0006 相同的 status launcher，输出改为 `smk0010_status_progress4.log`；command SHA `1eabc24b…99fe` | 终态进程与 metric 快照 | 0 | smoke `19:53:43–19:58:13`、exit `0`；driver/monitor 均退出、无残留匹配进程、两卡 `0 MiB`、OOM 增量 0。最终 metrics：global rows `80`，`updates_run=actor_updates=critic_updates=policy_version=1`，pending updates `0`，ratio `1.0`；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0010_status_progress4.log)，SHA `d56646f9…b7aa` |
| SMK-0011 | 约 20:03:37–20:03:39 +08:00 | `remote_exec_autodl.py put local_scripts/analyze_ogpo_smoke_resources.py …/runtime/analyze_ogpo_smoke_resources.py`；local SHA `7feaca97…aafc` | 上传只读 CSV 汇总器 | 0 | SFTP put 成功；仅新增 4,388-byte evidence helper；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0011_upload_analyzer.log)，SHA `36a6887c…8b7` |
| SMK-0012 | 约 20:03:57–20:03:59 +08:00 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260807_smoke_postflight_v1.sh`；command SHA `c1015c40…155c` | 首次 postflight | 3 | 未进入 CSV/sidecar 读取：helper 将整段脚本置于 `bash -c` argv，脚本内 `pgrep` 因而匹配自身并误报残留进程。没有训练/代码副作用；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0012_postflight.log)，SHA `5c912722…687e` |
| SMK-0013 | 约 20:04:41–20:04:44 +08:00 | `remote_exec_autodl.py put local_scripts/remote_ogpo_20260807_smoke_postflight_v1.sh …/runtime/remote_ogpo_20260807_smoke_postflight_v1.sh`；local SHA `c1015c40…155c` | 上传 immutable postflight v1 | 0 | SFTP put 成功；v1 原字节保留，不原地改写已执行 command-file；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0013_upload_postflight.log)，SHA `3041675b…dd57` |
| SMK-0014 | 约 20:05:16–20:05:21 +08:00 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260807_smoke_postflight_launch_v2.sh`；command SHA `475278ba…9847` | SHA 锁定后以文件路径执行 v1 | 0 | 复测通过：CSV 207 samples/271 s、两卡峰值 `50701/51311 MiB`、cgroup 峰值 `60.14 GB`、OOM 增量 0；manifest + 两 rank sidecar 可读且 counters/replay 完整，DCP 两片非空；Git clean/upstream `0/0`；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0014_postflight_v2.log)，SHA `3d4bbb08…bfb4` |
| SMK-0015 | 约 20:07:09–20:07:12 +08:00 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260807_smoke_evidence_inventory_v1.sh`；command SHA `9b765b49…3c90` | 服务器 evidence 尺寸与 SHA inventory | 0 | runtime 全部小文件、metrics/TensorBoard/manifest SHA 已锁定；checkpoint 总体约 14.09 GB，未下载大文件；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0015_evidence_inventory.log)，SHA `566e23b6…2f99` |
| SMK-0016 | 约 20:08:35 +08:00 | 直接 `& local_scripts/download_ogpo_20260807_smoke_evidence_v1.ps1`；script SHA `03cd5db4…4e2a` | 首次本地证据下载 launcher | 实际失败（外层误返 0） | Windows ExecutionPolicy 在脚本 body 前拒绝执行；无 GET、无服务器或证据目录副作用。工具输出于 20:10 补存且明确标注来源；[记录](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0016_download_evidence.log)，SHA `7fdf2561…0fa1` |
| SMK-0017 | 20:08:38–20:09:17 +08:00 | `powershell.exe -NoProfile -ExecutionPolicy Bypass -File local_scripts/download_ogpo_20260807_smoke_evidence_v1.ps1`；script SHA `03cd5db4…4e2a` | 下载小型证据 | 0 | 21 个文件逐个 `GET_OK` 且 local SHA 与 SMK-0015 remote SHA 全一致；不含 DCP/sidecar 大文件；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0017_download_evidence_v2.log)，SHA `166691d1…c92f` |
| SMK-0018 | 20:10:08–20:10:14 +08:00 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260807_smoke_tensorboard_check_v1.sh`；command SHA `a2e0d986…06e4` | 服务器 TensorBoard scalar contract | 0 | step80 的 rows/update/version/pending/loss 均与 metrics/sidecar 一致；一次 eval event 已落盘，`eval/num_trajectories=0` 表示 C10 截断内无完整 200-step episode，不表示 eval 未运行；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_smk0018_tensorboard_check.log)，SHA `e2701cb4…94c0` |

#### SMK-0001 完整无密码 launcher

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'
try {
  & 'C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
    'local_scripts\remote_exec_autodl.py' run `
    --command-file 'local_scripts\remote_ogpo_20260807_readonly_refresh.sh' 2>&1 |
    Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0001_readonly_refresh.log'
  $taskRc=$LASTEXITCODE
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
}
exit $taskRc
```

服务器副作用：无。重要结果：`autodl-container-nekaqbwt43-6ce5babb`；GPU 0/1 均为
`0 MiB/0%`；无相关训练/Ray 进程；目标 branch/upstream clean；checkpoint 与 Robotwin norm SHA
仍匹配；共享 Python/Torch/Ray/JAX/Flax runtime 可用。

#### SMK-0002 完整无密码 launcher

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'
try {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
    --command-file 'local_scripts\remote_ogpo_20260807_smoke_prepare_v1.sh' 2>&1 |
    Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0002_prepare.log'
  $taskRc=$LASTEXITCODE
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
}
exit $taskRc
```

服务器副作用：新建
`/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime`，保存 source/resolved
config、exact command、provenance、启动前资源和停止条件；没有启动 Ray/RoboTwin/GPU compute，
没有创建 `/root/autodl-tmp/experiments/ogpo_robotwin_smoke_20260807_v1`。

#### SMK-0003 完整无密码 launcher

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'
try {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' put `
    'local_scripts\remote_ogpo_20260807_smoke_monitor_v1.sh' `
    '/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime/remote_ogpo_20260807_smoke_monitor_v1.sh'
  $monitorRc=$LASTEXITCODE
  if ($monitorRc -ne 0) { exit $monitorRc }
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' put `
    'local_scripts\remote_ogpo_20260807_smoke_run_v1.sh' `
    '/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime/remote_ogpo_20260807_smoke_run_v1.sh'
  $runRc=$LASTEXITCODE
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
}
exit $runRc
```

服务器副作用：只新增上述两个 runtime script；未启动进程。

#### SMK-0004 完整无密码 launcher

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'
try {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
    --command-file 'local_scripts\remote_ogpo_20260807_smoke_verify_upload_v1.sh' 2>&1 |
    Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0004_verify_upload.log'
  $taskRc=$LASTEXITCODE
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
}
exit $taskRc
```

服务器副作用：无；仅校验上传字节与 shell 语法。

#### SMK-0005 完整无密码 launcher

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'
try {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
    --command-file 'local_scripts\remote_ogpo_20260807_smoke_launch_v1.sh' 2>&1 |
    Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0005_launch.log'
  $taskRc=$LASTEXITCODE
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
}
exit $taskRc
```

服务器副作用：创建 smoke run root；以 `nohup` 启动唯一 driver 和独立 1 秒监控进程；写入 PID、启动时间、driver/monitor 日志。训练命令外层为 `timeout --signal=TERM --kill-after=120s 1800s`。

#### SMK-0006 完整无密码 launcher

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'
try {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
    --command-file 'local_scripts\remote_ogpo_20260807_smoke_status_v1.sh' 2>&1 |
    Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0006_status_initial.log'
  $taskRc=$LASTEXITCODE
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
}
exit $taskRc
```

服务器副作用：无；只读 PID、marker、CSV、driver tail、输出树、相关进程、GPU 与 cgroup memory events。

#### SMK-0007～0010 完整无密码 launcher

四次均复用 SMK-0006 的完整 launcher，只替换 `Tee-Object -FilePath`：

```text
SMK-0007 -> exports\ogpo_smoke_20260807_smk0007_status_progress1.log
SMK-0008 -> exports\ogpo_smoke_20260807_smk0008_status_progress2.log
SMK-0009 -> exports\ogpo_smoke_20260807_smk0009_status_progress3.log
SMK-0010 -> exports\ogpo_smoke_20260807_smk0010_status_progress4.log
```

command-file 始终为 `local_scripts\remote_ogpo_20260807_smoke_status_v1.sh`，SHA
`1eabc24bdc538f7dc76ac7503228170c67cc9b4766727fb8dae1d0b68e4b99fe`；四次均为只读状态查询。

#### SMK-0011～0015 完整无密码 launcher

所有命令都在同一进程级 secret wrapper 中运行，并在 `finally` 删除环境变量：

```powershell
# SMK-0011
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' put `
  'local_scripts\analyze_ogpo_smoke_resources.py' `
  '/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime/analyze_ogpo_smoke_resources.py'

# SMK-0012：首次 postflight，因 bash -c argv 自匹配而 exit 3
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_smoke_postflight_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0012_postflight.log'

# SMK-0013
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' put `
  'local_scripts\remote_ogpo_20260807_smoke_postflight_v1.sh' `
  '/root/autodl-tmp/experiment_exports/ogpo_robotwin_smoke_20260807_v1/runtime/remote_ogpo_20260807_smoke_postflight_v1.sh'

# SMK-0014：v2 只校验上述远端文件 SHA 后 exec bash <path>
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_smoke_postflight_launch_v2.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0014_postflight_v2.log'

# SMK-0015
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_smoke_evidence_inventory_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0015_evidence_inventory.log'
```

服务器副作用：SMK-0011/0013 只在 runtime evidence 目录新增两个 SHA 锁定 helper；SMK-0014
写入 `resource_summary.json` 和 `checkpoint_summary.json`；其余只读。没有修改代码 worktree、
checkpoint 或训练输出。

#### SMK-0016～0018 完整无密码 launcher

```powershell
# SMK-0016：被 Windows ExecutionPolicy 在脚本 body 前拒绝
& 'local_scripts\download_ogpo_20260807_smoke_evidence_v1.ps1' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0016_download_evidence.log'

# SMK-0017：同一下载脚本，显式使用子 PowerShell Bypass
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  'local_scripts\download_ogpo_20260807_smoke_evidence_v1.ps1' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0017_download_evidence_v2.log'

# SMK-0018
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_smoke_tensorboard_check_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_smoke_20260807_smk0018_tensorboard_check.log'
```

SMK-0017 的脚本逐项列出 21 个 `remote_exec_autodl.py get <remote> <local>`，并把每个 local SHA
与 SMK-0015 的 remote SHA 比较；完整文件表即
[`download_ogpo_20260807_smoke_evidence_v1.ps1`](/C:/Users/86136/Documents/rl/local_scripts/download_ogpo_20260807_smoke_evidence_v1.ps1)。
服务器副作用：SMK-0016/0017 无；SMK-0018 只读 TensorBoard event。SMK-0017 在本机新增
`exports/ogpo_smoke_20260807_v1/` 的 21 个小型证据文件，不含约 14 GB checkpoint。

### Smoke resolved packet

live preflight 与 Hydra compose 后的最终 packet 已保存并下载：resolved SHA
`ebd163f647d2a9399fdca007099fac550f6f344392162adc36eede129671f4eb`。核心 override 为 train 8 env、
C10、B64/G8、`start_training_rows=79`、`total_online_rows=80`、capacity80、UTD1、baseline eval off、
终点一次 4-env C10 eval、interval/final checkpoint 80、actor total training steps 1；runner nominal 1 cycle、
hard cap 2，外层 wall timeout 1,800 秒。完整内容见
[`resolved.yaml`](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_v1/resolved.yaml)、
[`exact_command.txt`](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_v1/exact_command.txt) 和
[`stop_conditions.txt`](/C:/Users/86136/Documents/rl/exports/ogpo_smoke_20260807_v1/stop_conditions.txt)。

### 问题、原因、修复与复测

按发生顺序填写；没有发生的问题不预先建立兜底项。

1. SMK-0012 的 postflight v1 用 `pgrep -af` 搜索残留 actor/env/rollout 名称；helper 把整段脚本放入
   `bash -c` argv，搜索因而命中 postflight 自身。它在任何 CSV/sidecar 读取前 exit 3。保留 v1 原字节，
   上传后由只含路径和 SHA 的 v2 launcher 执行；SMK-0014 复测通过。
2. SMK-0016 直接执行 `.ps1` 被 Windows ExecutionPolicy 拒绝，脚本 body 未开始；外层 wrapper 的
   null `LASTEXITCODE` 还错误返回 shell 0，所以账本按实际行为标记失败，不采用该 exit。SMK-0017
   显式 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File` 后，21 个 GET 和 SHA 校验全部通过。
3. RoboTwin 启动时打印 Vulkan ICD/Curobo/pytorch3d 可选 planner import traceback；两个 EnvWorker
   随后进入 `interact`、完成 80 rows 并 clean exit，因此这不是本次 ALOHA task 的致命依赖错误，未安装
   新依赖或增加兜底。
4. 终态 metrics 显示 actor loss/grad norm 0、critic loss/grad 非零。只读源码复核表明最可能是首次
   actor-first update 读取随机 10Q，严格 CA 将全部候选 veto；本次又无 success BC。由于当前日志没有
   `ca_nonzero_fraction`，账本把它记为“系统 smoke 通过、有效 actor 学习信号未覆盖”，不假称权重已
   有效变化，也不在本次授权内自动重跑。
5. monitor 是 launcher 独立启动、以 driver PID 消失为停止条件；本版没有 supervisor `wait` 它并写
   `monitor_exit_code.txt`，所以 exact monitor exit code 未捕获。可验证事实是 monitor PID 已消失、stderr
   log 为空、CSV header+207 rows 全部有效且最后三行已回到 0 compute/0 GPU memory。以后 launcher
   应在不改变采样内容的前提下持久化 monitor exit；本次不补造该值。

## 一日预算 formal：逐命令流水

本节记录用户于 2026-08-07 授权启动的正式训练。算法与系统配置沿用已通过 smoke 的版本，只允许四项
预算语义变化：`total_online_rows=35000`、`start_training_rows=10000`、paired
`UTD=0.1`（代码字段为 `utd_q=0.1` 与 `utd_pi=0.1`）、`replay_capacity=40000`。

### FRM-0001：只读现场刷新

第一次 launcher 在本机读取附件第 3 行（索引 2），该行为空，因
`Missing process-only SSH password` 在建立 socket 前 exit 1；服务器没有收到连接，也没有副作用。原命令
与下方成功版相同，仅 `$lines[5]` 写成 `$lines[2]`，未产生输出文件。检查附件时只输出每行索引、长度和
是否为空，没有输出内容；确认实际密码位于索引 5 后重试。

成功重试时间：服务器 `2026-08-07T23:01:14+08:00`；exit 0；command-file SHA256
`e22dced16d2fd924a09776a7f22b583a6042ef7366d23c2bdd0106ee52024c9d`。

```powershell
$credPath='E:\Codex\home\attachments\a0e17fb0-c2f9-4800-b353-c3a3d2090c83\pasted-text.txt'
$lines=Get-Content -Encoding UTF8 -LiteralPath $credPath
$pw=$lines[5].Trim()
if (-not $pw) { throw 'Missing process-only SSH password' }
try {
  $env:SEETA_SSH_PASSWORD=$pw
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
    --command-file 'local_scripts\remote_ogpo_20260807_readonly_refresh.sh' 2>&1 |
    Tee-Object -FilePath 'exports\ogpo_formal_20260807_frm0001_readonly_refresh.log'
  $taskRc=$LASTEXITCODE
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
  $pw=$null
}
exit $taskRc
```

结果：目标 worktree 为 clean、已跟踪 upstream 的 `codex/ogpo-pi0-robotwin`，HEAD
`5d5c84e3ac4efa1713a4139a05ac1b776e634ed3`；无相关训练、Ray 或 GPU compute 进程；两张
A800-SXM4-80GB 均为 0 MiB；系统 available RAM 约 981 GiB；`/root/autodl-tmp` 可用约
802 GiB；SFT 模型存在且 RoboTwin norm SHA256 仍为
`649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a`。原始输出为
`exports/ogpo_formal_20260807_frm0001_readonly_refresh.log`，SHA256
`5dcb821e0e35f41882387b96adad821f5db766eac94917107db2f4bc905d5055`。服务器副作用：无。

### FRM-0002～0014 操作索引

| ID | 服务器时间/本机日志末次写入 | 操作与 exit | 结果/服务器副作用 | 原始输出 SHA256 |
|---|---|---|---|---|
| FRM-0002 | 23:07:21 | 4 个 formal 脚本逐一 `bash -n`，exit 0 | 只读 stdin 解析；服务器无文件 | `8dd575f983b5faa89b57465bea14e976e748c357939756a392e64fa2ee8351e0` |
| FRM-0003 | 23:07:48 | 初次 compose-only prepare，exit 1 | 创建 runtime partial：`source_config.yaml`、`resolved.yaml`；Python assertion line 54 失败；未建 run root、未启动进程 | `17a4785085d61a9ccd7ca8118c3e983b1dfcb43bead00e8bf86bbfedc809ee7b` |
| FRM-0004 | 23:09:00 | partial 只读检查，exit 0 | 确认 resolved 四项预算及所有调度值正确；placement 重载类型为 `DictConfig` | `0e1b5d8fab2c2519f0e89dc288a7743466a531e8a4407a52293f04dff1de2d1b` |
| FRM-0005 | 23:10 前 | resume prepare `bash -n`，exit 0 | 只读 stdin 解析；命令无 stdout/stderr，因此未生成空 log 文件 | 无输出文件 |
| FRM-0006 | 23:10:38 | narrow resume prepare，exit 0 | 不覆盖 resolved；补齐 exact command、provenance、stop conditions、resources-before | `161f7dd653afbb577d5ee7839170c642d8d3a462c372f8f4b41ca01b62606f51` |
| FRM-0007 | 23:11:38 | SFTP PUT 6 个 runtime scripts，全部 exit 0 | runtime 目录新增 prepare/resume/inspect/run/monitor/status 固定字节 | `a3dafc979c9ac8d04f06a02c2304a2ab6af8a10a6d085287fec3b9cd321dfbc1` |
| FRM-0008 | 23:12:32 | runtime SHA、`bash -n`、Git/进程/GPU preflight，exit 0 | 只读；确认没有 smoke timeout、没有第五项预算 override | `10f1b37e9ee436d2dbde488d5515bcff7df29659dfac0d7da174f65f16ed736d` |
| FRM-0009 | 23:13:40 | SFTP PUT launch v1，exit 0 | runtime 新增 SHA 锁定 launch script | `34a05df1d6acbbf4fe791e38317c0b24e41fedf501191d20806337d6140fdd10` |
| FRM-0010 | 23:13:50 | SHA-verified launch，exit 0 | `nohup` 启动 driver PID 103679 与 monitor PID 103680；创建 formal run root | `9d8cc454b96d9cd49c83b8e900e4938580cc2068a0c64ec827c9f9b7aa5125f2` |
| FRM-0011 | 23:14:07 | 初始只读 status，exit 0 | driver/monitor alive；Ray 初始化；14 个资源样本；GPU compute 尚未创建 | `100dc5569185b4f38554e6dc5bb146beb70b9b595b1e843483816bfc59177cce` |
| FRM-0012 | 23:14:55 | progress status 1，exit 0 | actor/rollout/env workers 已创建；两卡 compute context；49 个资源样本；OOM 0 | `ecd21c2ac0deb0d8887934047ad00fbf4403637a63789bc30301a50e7c5a9b54` |
| FRM-0013 | 23:16:01 | progress status 2，exit 0 | norm stats 加载；GPU 约 8.27 GiB/卡；FSDP 模型继续初始化；OOM 0 | `bb0b2dc28403aaaac38066f3578e46607b6fe46a88d4faafcef4c24f3c8838bc` |
| FRM-0014 | 23:17:10 | progress status 3，exit 0 | baseline eval 已进入；8 个 compute-app rows；GPU 约 28.60/28.04 GiB；cgroup约48.8 GiB；OOM 0 | `d53a455ca1ec9cc2a3c8c317ac2c91dea9fd3a0573c4a6bdf03c75b4bf368cd5` |
| FRM-0015 | 23:19:06 | stable status，exit 0 | baseline 2/5；240 个资源样本；显存约28.47/27.91 GiB；OOM 0 | `e15b3a66b2eddae6e160326e802c02e8264c6c79de1caebce8cc4e05814da81e` |
| FRM-0016 | 23:20:42 | concise health，exit 0 | baseline 4/5；313 个资源样本；driver/monitor alive；OOM 0 | `d606cf520e12ca97a1a72c73c9cf54c39c5b7eb22a137585cd4be38b3cc932d0` |
| FRM-0017 | 23:21:50 | concise health，exit 0 | baseline 5/5于4:28完成；切到`Generating Rollout Epochs` | `be0ceb4fffa65febfa37951b6a36293318ca8cbaa464a11c1235678797271225` |
| FRM-0018 | 23:23:06 | concise health，exit 0 | online rollout持续；422个资源样本；GPU利用率100%样本；OOM 0 | `ce2221da23c350acb6aef5366af9c2ae91a6c63753a1775cd58fbf2d6072d194` |
| FRM-0019 | 23:24:23 | concise health，exit 0 | online rollout继续；479个资源样本；driver/monitor alive；OOM 0 | `43afa1c5af705ab1c23e02bb477a5f5233da4965087aaf2ddaccb1c2289d41ec` |
| FRM-0020 | 23:25:13 | live TensorBoard/process read，exit 0 | baseline step0有20 trajectories、success_once/end=.05；无train scalar尚属10k warmup前；TensorFlow只打印初始化warning | `d639b35601d135f1a12d9ead560e60b94959509a20ec27d555d201f70d1753fb` |
| FRM-0021 | 23:26:07 | active-method只读 probe，exit 0 | actor=`recv_rollout_trajectories`、rollout=`generate`、env=`interact`；8 compute rows；显存约24.19/24.04 GiB；OOM 0 | `852422dac77d34daef9d9bff51bb044189bc94b538648c2c76a8d98a845b6ca1` |
| FRM-0022 | 23:31:57 | immutable evidence manifest，exit 0 | 只读10个固定文件的size/SHA；不读取仍增长的driver/CSV | `e581772a7a43ae558e20addc8027f5cf013217be54806c3bc159ec281dd2cadc` |
| FRM-0023 | 23:32:24 | SFTP GET 10个immutable evidence，全部exit 0 | 本机新增`exports/ogpo_formal_20260807_v1/`；local SHA逐项匹配FRM-0022 | `459ef4f40ceaa57b99219edc0094c0dfeaf52600eca28f6a072a4feae2ba59e6` |
| FRM-0024 | 23:32:48 | final health handoff，exit 0 | 首个online wave 8:20完成；global rows1600/local800；updates0；第二wave开始；859 samples、OOM0 | `160ccb45cd75c533df98ec0d04e0a7783db892cd6dad18424784317fa8413130` |
| FRM-0025 | 23:33:01 | final live TensorBoard read，exit 0 | train scalars逐项确认rows1600、replay800/rank、success0、updates/version0；进程/GPU/monitor持续 | `f444a99448c5b0bb3619e5a690a0ca25b119fc82dd531c6e455a21af4df918b8` |
| FRM-0026 | 2026-08-08 10:07:09–10:07:11 | 复用formal health只读刷新，exit 0；command SHA `6e554212…94cf` | driver/monitor均alive、exit pending；最新完整batch为30,720 rows、2,073 paired updates、success rows 2,765/rank、BC loss .027、critic loss .011；29,613个资源样本，显存现场57,231/57,464 MiB、cgroup 118.47 GB、OOM 0；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0026_health_refresh.log)，SHA `bff25b6d…a4d5` | `bff25b6d9a781ffefb964cfefa41d6b413196c3a28181d0dccb077974c60a4d5` |
| FRM-0027 | 10:09:53–10:09:55 | 两次`remote_exec_autodl.py run 'bash -n' --stdin-file ...`；artifact/metrics脚本SHA `547e53fe…3fbc` / `523da349…c5e6` | 两个只读取formal现场的审计脚本均通过远端bash语法检查；未执行脚本主体、无服务器副作用；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0027_audit_syntax.log)，SHA `1156240e…e954` | `1156240e01d6a7273f913c810b90ae02aeb1af4336e81e4439ec2039e298e954` |
| FRM-0028 | 10:10:24–10:10:30 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260808_formal_artifact_audit_v1.sh`；command SHA `547e53fe…3fbc` | 全量只读现场审计：latest完整TB batch 30,730/35,000 rows与2,073/2,500 paired updates；20,081-row eval成功率5%→35%；16个训练点均无NaN/Inf。资源29,764样本，GPU峰57,231/57,676 MiB、cgroup峰118.99 GB、OOM0。run root仅metrics/TB/config三文件、无checkpoint（符合50k interval且尚未final）；两处Traceback均为启动时未使用的Curobo planner可选导入，训练继续。Git HEAD/clean/upstream与immutable SHA不变；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0028_artifact_audit.log)，SHA `8346698a…5333` | `8346698a380f5c8b2313ebfed33bccad4d417246ba039bac921159c8190c5333` |
| FRM-0029 | 10:10:46–10:10:52 | `remote_exec_autodl.py run --command-file local_scripts/remote_ogpo_20260808_formal_metrics_dump_v1.sh`；command SHA `523da349…c5e6` | 只读导出全部`train/ogpo/*`与`eval/*` scalar历史；22个row/counter点、16个post-warmup优化点、2个eval点。输出尾含TensorFlow初始化stderr，故作为原始合并日志而非纯JSON；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0029_metrics_dump.json)，SHA `7df2951e…f4e2` | `7df2951e1c41d34faa517d3f1c6aedf2ef7d5f4bd172b3f5ee8e7d4a9182f4e2` |
| FRM-0030 | 10:11:14–10:11:23 | 五次`remote_exec_autodl.py get`，逐项exit 0 | SFTP只读下载当前`resources_1s.csv`、`driver.log`、`metrics.log`、TB config/event到`exports/ogpo_formal_20260808_live/`，共约7.16 MB；未下载模型、replay或checkpoint。逐文件size/SHA见本节下方；[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0030_download_live_evidence.log)，SHA `79077355…b7f7` | `79077355b6df085a076fd81d7bf71b5fe64a664f7336484471692c1290a6b7f7` |
| FRM-0031 | 10:23:06–10:23:07 | 复用formal health只读刷新，exit 0；command SHA `6e554212…94cf` | driver/monitor alive、exit pending；当前处于update burst；30,336个资源样本，现场显存57,231/57,254 MiB、GPU85%/84%、cgroup123.09 GB、OOM0；run root仅metrics/TB/config，无checkpoint符合未final。[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0031_health_final_refresh.log)，SHA `50144f1f•5228` | `50144f1f29625d1629ae4013861f928f7a8c6090d16c0ebce90dbd2e8b25228b` |
| FRM-0032 | 10:23:20–10:23:24 | 复用formal live-metrics只读刷新，exit 0；command SHA `90b78421•0560` | 最新完整点32,330/35,000 rows、2,233/2,500 paired updates、policy version2233、replay16,165/rank、success2,765/rank；actor loss/grad `4.00156e-5/0.20675`、BC`.0265566`、ratio`.930057`；critic loss/grad`.0121004/.55523`、Q/TD`.0359843/.0358568`；所有scalar有限。[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0032_live_metrics_final_refresh.log)，SHA `f72b5ba2•3f2` | `f72b5ba21c505e2b37acb9a23caea742cbe91e9bbafb176651ccebd6d809b3f2` |
| FRM-0033 | 10:48:54 | artifact-audit只读刷新；远端脚本正常输出，但本机wrapper exit 1 | 本机设置`$ErrorActionPreference='Stop'`后，把TensorFlow正常stderr warning当作`NativeCommandError`并提前退出；已取得14,682-byte部分日志，服务器训练、代码与产物均无副作用。[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0033_artifact_audit_refresh.log) | `2e06e04be4e56f8a4aaf6af26d61a9fd6f4e81cff3f2ea1f8a1b264bd78bafbf` |
| FRM-0034 | 10:49:13–10:49:18 | 去掉本机stop-on-stderr后重跑同一artifact audit，exit 0；command SHA `547e53fe…3fbc` | 完整115,668-byte只读审计；确认进程、Git/immutable SHA、指标、资源和产物正常，未改服务器状态。[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0034_artifact_audit_refresh.log) | `3c4d8315c3c9b0c0b9a459e3e1b0445ad34c6b2602a49b65d882759b6c2bf11d` |
| FRM-0035 | 10:53:38 | formal health只读刷新，exit 0；command SHA `6e554212…94cf` | 确认最新完整点33,861 rows/2,386 paired updates，driver/monitor alive、exit pending；下一轮rollout已开始；无OOM或fatal。[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0035_health_refresh.log) | `7d2573067575e9aa6cd2004647b43b07f134bf8346e16eb033a16d278287d50c` |
| FRM-0036 | 10:54:05 | formal live-metrics只读刷新，exit 0；command SHA `90b78421…0560` | 逐tag确认33,861/35,000 rows、2,386/2,500 updates；actor PPO loss/combined grad `4.08862e-5/.21105`、BC`.02723`、ratio`.94328`；critic loss/grad`.009026/.37958`、Q/TD`.023373/.023332`，全部有限。[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0036_live_metrics_refresh.log) | `964a9283d25e6fcb18e172bbfad6a8e8cc018dd531e4d7e5f4b19bc6de7a094f` |
| FRM-0037 | 11:00:22 | formal health只读刷新，exit 0；command SHA `6e554212…94cf` | 第25个8-env wave已用7:42完成模拟，随后进入最后update burst；现场cgroup125.65 GB、GPU约57.0/57.3 GiB、OOM0；最新完整TB点仍为33,861/2,386。[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0037_health_refresh.log) | `de1d338ac73a16eb71615c964d31a4303189bf3308973db4ba8a08760e8b43d0` |
| FRM-0038 | 11:08:11 | formal health只读刷新，exit 0；command SHA `6e554212…94cf` | driver/monitor仍alive、exit pending；最后update burst继续，两卡85%/90%、显存57,231/57,254 MiB、cgroup125.69 GB（约117.06 GiB）、OOM/oom_kill0；尚未落盘新的batch表/checkpoint/final-eval。[原始输出](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0038_health_refresh.log) | `ef2ed74b0ce1ace8a4c956b5abaeabe12763d98e53aaec533c017f5013558ff5` |

FRM-0011～0021 之间按30、45、50、50、50、55、55秒做本机有界等待，没有在等待阶段向服务器发命令
或产生副作用。

### FRM-0002～0006 精确 launcher

所有 launcher 都从已授权附件中读取 secret 到当前 PowerShell 进程，只把
`SEETA_SSH_PASSWORD` 传给 Paramiko helper，并在 `finally` 删除；`<bundled-python>` 是
`C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`。

```powershell
# FRM-0002：$files 中依次为 prepare_v1、run_v1、monitor_v1、status_v1。
$credPath='E:\Codex\home\attachments\a0e17fb0-c2f9-4800-b353-c3a3d2090c83\pasted-text.txt'
$lines=Get-Content -Encoding UTF8 -LiteralPath $credPath
$pw=$lines[5].Trim()
$files=@(
  'local_scripts\remote_ogpo_20260807_formal_prepare_v1.sh',
  'local_scripts\remote_ogpo_20260807_formal_run_v1.sh',
  'local_scripts\remote_ogpo_20260807_formal_monitor_v1.sh',
  'local_scripts\remote_ogpo_20260807_formal_status_v1.sh'
)
$taskRc=0
try {
  $env:SEETA_SSH_PASSWORD=$pw
  & {
    foreach ($file in $files) {
      Write-Output "CHECKING`t$file"
      & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run 'bash -n' --stdin-file $file
      $fileRc=$LASTEXITCODE
      Write-Output "EXIT`t$fileRc"
      if ($fileRc -ne 0) { $taskRc=$fileRc; break }
    }
  } 2>&1 | Tee-Object -FilePath 'exports\ogpo_formal_20260807_frm0002_bash_syntax.log'
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
  $pw=$null
}
exit $taskRc
```

FRM-0003、0004、0006 使用同一个 secret wrapper，内部命令分别为：

```powershell
# FRM-0003
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_prepare_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260807_frm0003_prepare.log'

# FRM-0004
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_partial_inspect_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260807_frm0004_partial_inspect.log'

# FRM-0005
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run 'bash -n' `
  --stdin-file 'local_scripts\remote_ogpo_20260807_formal_prepare_resume_v2.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260807_frm0005_resume_syntax.log'

# FRM-0006
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_prepare_resume_v2.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260807_frm0006_prepare_resume.log'
```

FRM-0003 的失败原因不是配置值错误：原 source 的 placement 是带冒号的 quoted string；Hydra
`--cfg job` 的 YAML printer 输出时丢失引号，证据文件被 OmegaConf 重载后成为
`{'actor, env, rollout': '0-1'}`。FRM-0004 证实预算和运行参数均正确。FRM-0006 只把断言改为匹配该
证据表示，并保留 live run 从原 source YAML compose；没有删除 partial、重写 resolved 或改变训练命令。

### FRM-0007～0010 精确 launcher

```powershell
# FRM-0007：同一 secret wrapper；逐项 SFTP PUT 并逐项检查 LASTEXITCODE。
$runtime='/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime'
$files=@(
  'remote_ogpo_20260807_formal_prepare_v1.sh',
  'remote_ogpo_20260807_formal_prepare_resume_v2.sh',
  'remote_ogpo_20260807_formal_partial_inspect_v1.sh',
  'remote_ogpo_20260807_formal_run_v1.sh',
  'remote_ogpo_20260807_formal_monitor_v1.sh',
  'remote_ogpo_20260807_formal_status_v1.sh'
)
foreach ($name in $files) {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' put `
    "local_scripts\$name" "$runtime/$name"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# FRM-0008
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_verify_runtime_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260807_frm0008_verify_runtime.log'

# FRM-0009
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' put `
  'local_scripts\remote_ogpo_20260807_formal_launch_v1.sh' `
  '/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime/remote_ogpo_20260807_formal_launch_v1.sh'

# FRM-0010：launcher SHA 锁定后从远端文件执行。
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_launch_exec_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260807_frm0010_launch.log'
```

launch v1 SHA256 为 `be32fe047759654056aa80c7bd9e7379cf88f0bbec4ccf9b845a6bb41bb64a52`；
它只在重新核对 clean HEAD/upstream、resolved/exact/provenance/stop/run/monitor SHA、空 GPU、无 Ray、
run root 不存在后，使用 `nohup bash <run>` 和 `nohup bash <monitor>` 启动两个进程。formal run 没有外层
wall timeout，以 35,000 replay rows 自然停止。

### FRM-0011～0021 精确只读状态 launcher

FRM-0011～0015 均复用以下完整命令，只替换输出文件名：

```powershell
$credPath='E:\Codex\home\attachments\a0e17fb0-c2f9-4800-b353-c3a3d2090c83\pasted-text.txt'
$lines=Get-Content -Encoding UTF8 -LiteralPath $credPath
$pw=$lines[5].Trim()
try {
  $env:SEETA_SSH_PASSWORD=$pw
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
    --command-file 'local_scripts\remote_ogpo_20260807_formal_status_v1.sh' 2>&1 |
    Tee-Object -FilePath '<operation-specific-output.log>'
  $taskRc=$LASTEXITCODE
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
  $pw=$null
}
exit $taskRc
```

输出映射：FRM-0011 `status_initial`、0012 `status_progress1`、0013 `status_progress2`、0014
`status_progress3`、0015 `status_stable`，完整文件名与 SHA 见上表。

FRM-0016～0019 使用相同 secret wrapper，把 command-file 改为
`local_scripts\remote_ogpo_20260807_formal_health_v1.sh`，输出依次为`health_progress`、
`health_baseline_complete`、`health_first_rollout`、`health_first_rollout2`。FRM-0020/0021 仍用同一
wrapper，精确内部命令分别为：

```powershell
# FRM-0020
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_live_metrics_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260807_frm0020_live_metrics.log'

# FRM-0021
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_active_methods_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260807_frm0021_active_methods.log'
```

这些命令只读取PID、driver/TensorBoard、1秒CSV、进程方法、GPU和cgroup memory events；没有修改训练
状态。10k warmup前本就没有optimizer update，故健康启动判据是baseline完整结束、online
`generate/interact`持续、driver/monitor alive、资源曲线连续且OOM=0，不额外等待数小时到首次update。

FRM-0022、0024、0025 使用相同secret wrapper，command-file依次为
`remote_ogpo_20260807_formal_immutable_manifest_v1.sh`、`formal_health_v1.sh`、
`formal_live_metrics_v1.sh`，输出文件名与SHA见上表。FRM-0023 的精确SFTP主体为：

```powershell
$runtime='/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime'
$localRoot='exports\ogpo_formal_20260807_v1'
$files=@(
  'source_config.yaml','resolved.yaml','exact_command.txt','run_provenance.tsv',
  'stop_conditions.txt','resources_before.txt','launched_at.txt','started_at.txt',
  'driver_pid.txt','monitor_pid.txt'
)
New-Item -ItemType Directory -Force -Path $localRoot | Out-Null
foreach ($name in $files) {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' get `
    "$runtime/$name" "$localRoot\$name"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

FRM-0023只从服务器读取小型固定证据，不下载正在增长的driver.log、TensorBoard event或resource CSV，
也不暂停monitor。下载后本机逐项SHA与FRM-0022完全一致。

### FRM-0026–0032 精确 launcher

本组继续使用前文定义的同一process-only secret wrapper；密码未写入下列命令、脚本、
日志或环境持久层。各次wrapper内部命令为：

```powershell
# FRM-0026
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_health_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0026_health_refresh.log'

# FRM-0027
$auditScripts=@(
  'local_scripts\remote_ogpo_20260808_formal_artifact_audit_v1.sh',
  'local_scripts\remote_ogpo_20260808_formal_metrics_dump_v1.sh'
)
foreach ($file in $auditScripts) {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run 'bash -n' --stdin-file $file
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} 2>&1 | Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0027_audit_syntax.log'

# FRM-0028
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260808_formal_artifact_audit_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0028_artifact_audit.log'

# FRM-0029
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260808_formal_metrics_dump_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0029_metrics_dump.json'

# FRM-0031
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_health_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0031_health_final_refresh.log'

# FRM-0032
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_live_metrics_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0032_live_metrics_final_refresh.log'
```

FRM-0030在同一wrapper中的五次SFTP GET为：

```powershell
$pairs=@(
  @('/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime/resources_1s.csv',
    'exports\ogpo_formal_20260808_live\resources_1s.csv'),
  @('/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime/driver.log',
    'exports\ogpo_formal_20260808_live\driver.log'),
  @('/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1/metrics.log',
    'exports\ogpo_formal_20260808_live\metrics.log'),
  @('/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1/tensorboard/config.yaml',
    'exports\ogpo_formal_20260808_live\tensorboard_config.yaml'),
  @('/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1/tensorboard/events.out.tfevents.1786115648.autodl-container-nekaqbwt43-6ce5babb.103684.0',
    'exports\ogpo_formal_20260808_live\events.out.tfevents.1786115648.autodl-container-nekaqbwt43-6ce5babb.103684.0')
)
foreach ($pair in $pairs) {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' get $pair[0] $pair[1]
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

五个下载文件的本地快照大小/SHA为：

| 文件 | bytes | SHA256 |
|---|---:|---|
| `resources_1s.csv` | 6,797,552 | `6b7c584f688c33184e46f95111361b87fe8166dd9971b05acbb489cd039a9efe`（增长中快照） |
| `driver.log` | 168,109 | `ec5e4fc49095149f6cda7211001eed587e260d8f8ea6b3e3c64a1ae7a22ad11a`（增长中快照） |
| `metrics.log` | 129,293 | `c3bb07c6a474ec603b758bba72b38aa3f92749244bbf0b8c37abc4bcf71ccced`（增长中快照） |
| `tensorboard_config.yaml` | 8,389 | `0ba5da258cb0639039361b5f536f42b6e9007e7b3797855014f4a9da0f009368` |
| `events.out.tfevents…103684.0` | 66,688 | `8683a81aca547239153f1475f74e7429d1836692ee011c8983eaf828e28a663b`（增长中快照） |

这些是10:11的有时间快照，不是终态产物；服务器原文件继续由formal/monitor追加，SFTP GET不改变服务器状态。

### FRM-0033–0038 精确 launcher

本组仍在前文定义的process-only secret wrapper内运行；下列命令不含密码。FRM-0033唯一额外设置
`$ErrorActionPreference='Stop'`，因此TensorFlow写到stderr的正常初始化提示被PowerShell升级为本机异常；
FRM-0034删除该设置后立即重跑，远端脚本和训练均无需修复。

```powershell
# FRM-0033：本机wrapper失败，服务器只读命令无副作用
$ErrorActionPreference='Stop'
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260808_formal_artifact_audit_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0033_artifact_audit_refresh.log'

# FRM-0034
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260808_formal_artifact_audit_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0034_artifact_audit_refresh.log'

# FRM-0035
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_health_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0035_health_refresh.log'

# FRM-0036
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_live_metrics_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0036_live_metrics_refresh.log'

# FRM-0037
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_health_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0037_health_refresh.log'

# FRM-0038
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_health_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0038_health_refresh.log'
```

FRM-0033–0038都只读服务器现场；没有SFTP PUT、配置改动、训练控制或产物写入。唯一失败是FRM-0033的
本机stderr处理，FRM-0034已用同一远端脚本完整闭合。

### Formal resolved packet

resolved SHA256：`77419258766880fca7b8d15d725dfb9f2fba5b88d2bfe21eb918fd760e9a7bc9`；
exact command SHA256：`1c5d3aed7a8d73a82cd67ee066b799d77c9180b33196f6683bdfa18eecd95888`。
唯一预算语义变化是 total 35k、warmup 10k、paired UTD 0.1、capacity 40k。paired UTD 在代码里由
`utd_q`/`utd_pi` 两个字段共同表达。其余继承 source：OGPO+CA、B64/G8、flat32/rank、H50/C10、K4、
train env8、eval env4×5、baseline/20k/final eval、checkpoint interval50k。训练预计 2,500 paired
updates、1,280,000 imagined group chains；唯一 checkpoint 是 exact35k final save。

### Formal 支持文件变更索引

本轮没有修改服务器RLinf代码或YAML；以下均为本机新增的运行/取证脚本，因文件为新增，行数等于
`+N/-0`。已上传到runtime的文件由FRM-0008/0010逐SHA锁定；只读查询脚本由Paramiko直接发送。

| 新文件 | +/- | bytes | SHA256 | 用途 |
|---|---:|---:|---|---|
| `formal_prepare_v1.sh` | +239/-0 | 9,567 | `008dca1e...cfeb22` | 初次compose；保留失败断言原字节 |
| `formal_prepare_resume_v2.sh` | +211/-0 | 9,013 | `44ab3b63...aee02` | 不重写resolved的窄恢复prepare |
| `formal_partial_inspect_v1.sh` | +47/-0 | 1,825 | `0420b8a5...aaa6` | 只读定位resolved证据表示 |
| `formal_run_v1.sh` | +53/-0 | 1,882 | `c61a712d...e0a2` | 无smoke timeout的正式driver |
| `formal_monitor_v1.sh` | +68/-0 | 3,775 | `e7c16a18...dad5` | 1秒GPU/RAM/process/disk CSV |
| `formal_status_v1.sh` | +40/-0 | 2,421 | `739c7c5b...684f` | 完整只读状态快照 |
| `formal_verify_runtime_v1.sh` | +74/-0 | 3,511 | `d604006d...ea8` | runtime逐SHA、syntax与空闲preflight |
| `formal_launch_v1.sh` | +67/-0 | 3,134 | `be32fe04...4a52` | 唯一driver+monitor nohup launcher |
| `formal_launch_exec_v1.sh` | +11/-0 | 382 | `97feae35...ab49` | 校验远端launch SHA后执行 |
| `formal_health_v1.sh` | +31/-0 | 1,797 | `6e554212...4cf` | concise progress/resource检查 |
| `formal_live_metrics_v1.sh` | +44/-0 | 1,477 | `90b78421...0560` | 只读TensorBoard最后scalar |
| `formal_active_methods_v1.sh` | +21/-0 | 842 | `aa21ce01...af3a` | 只读Ray当前method与compute-app |
| `formal_immutable_manifest_v1.sh` | +25/-0 | 599 | `ff9242b6...a33e` | 固定证据size/SHA清单 |

脚本完整文件名统一前缀为`local_scripts/remote_ogpo_20260807_`；表内省略共同前缀以便阅读。文档修改为：
`HANDOFF.md`更新当前运行与授权；主计划§6.9/§8–§10更新formal决策；参数来源§5–§6分开source与
running值；`IMPLEMENTATION_LOG.md`新增IMP-0036～0038；`CONTEXT_INVENTORY_LOG.md`新增CTX-0016；
本文件新增FRM逐命令、结果、问题与SHA。`PROJECT_CONTEXT.md`长期规则未变化，未修改。

### 2026-08-08 live audit 支持文件与本机派生命令

本轮没有修改服务器RLinf代码、YAML或运行中文件。新增的两个远程审计脚本通过Paramiko发送到stdin/
command channel执行，没有SFTP PUT到服务器；两个分析/渲染脚本仅在Windows本机运行。

| 新文件 | +/- | bytes | SHA256 | 用途 |
|---|---:|---:|---|---|
| `local_scripts/remote_ogpo_20260808_formal_artifact_audit_v1.sh` | +193/-0 | 7,979 | `547e53febb927b9af02720ff5e650fe3497c3f3c20cc4ff15a87602a8db33fbc` | 一次读取进程、Git/哈希、产物、TB、资源、异常 |
| `local_scripts/remote_ogpo_20260808_formal_metrics_dump_v1.sh` | +31/-0 | 975 | `523da34945e590d1383affe01dc8afc90c912ac4ddfc157308d6628ac918c5e6` | 只读导出全部OGPO/eval scalar历史 |
| `local_scripts/analyze_ogpo_formal_live.py` | +197/-0 | 6,746 | `278448409271e819b3b20ba04774873e4d36a23bd61bd43277cdef03088a3c8a` | 解析TB dump和resource CSV，生成可重算JSON |
| `local_scripts/render_inline_visualization_png.js` | +32/-0 | 1,505 | `f22c6486f0d046fc37b5b670db802bea06ce63477a1aaed2bdbc450f8605aa35` | 用已有Chrome将会话内联HTML渲染为独立PNG |

本机分析命令为：

```powershell
& '<bundled-python>' 'local_scripts\analyze_ogpo_formal_live.py' `
  --root 'exports\ogpo_formal_20260808_live'

& '<bundled-node>' 'local_scripts\render_inline_visualization_png.js' `
  'E:\Codex\home\visualizations\2026\08\05\019fd0f8-e1fa-76b1-816c-0ce8544216c4\ogpo-formal-live-20260808.html' `
  'E:\Codex\home\plugins\cache\openai-bundled\visualize\1.0.14\skills\visualize\assets\visualize.css' `
  'exports\ogpo_formal_20260808_live\ogpo-formal-live-20260808.png'
```

首次Python解析因metrics dump是UTF-16LE且前置TensorFlow/PowerShell日志而失败；脚本窄修为编码检测与
JSON marker搜索后重跑exit 0。首次PNG渲染因Playwright捆绑browser executable不存在而失败；窄修为显式
使用本机Chrome后重跑exit 0。两次都是本机派生工具问题，未影响服务器训练。

| 派生产物 | bytes | SHA256 |
|---|---:|---|
| `exports/ogpo_formal_20260808_live/analysis.json` | 330,822 | `01290a032b4b367af8f3447c4a531ba6b35b25a774684fbfb57a127adfa87414` |
| `exports/ogpo_formal_20260808_live/ogpo-formal-live-20260808.png` | 214,914 | `154d3f19fa585012659def3e2589958cb34d482a4704e713f6c7e7e8881b6d52` |
| `visualizations/.../ogpo-formal-live-20260808.html` | 17,317 | `8a57ea195a313b5811851d4ab8d58b873032c0cb98c0dc1c0ac31852c2656d33` |

2026-08-08文档修改为：`HANDOFF.md`将动态快照刷新到10:23；主计划§10更新进度、eval、优化/资源
指标与产物；`IMPLEMENTATION_LOG.md`新增IMP-0039；本文新增FRM-0026–0032、精确launcher、下载快照
哈希与本机派生索引。`PROJECT_CONTEXT.md`、参数来源、调用流和服务器RLinf代码均未修改。

### 2026-08-08 11:08 细粒度复盘文档变更

- `HANDOFF.md`：动态现场由10:23刷新到11:08，加入33,861/2,386、最后wave/update状态、最新资源、
  rollout账目和ETA；修正两处旧文本乱码。
- `00_INDEX_AND_IMPLEMENTATION_PLAN.md`：§10刷新到11:08，并加入Q排序可见性边界和下一轮最小指标集。
- `IMPLEMENTATION_LOG.md`：新增IMP-0040，记录数值诊断、ratio/EMA解释、warmup/UTD/total/capacity复盘、
  rollout口径与FRM-0033本机wrapper问题。
- 本文件：新增FRM-0033–0038逐操作表、完整无密码launcher、日志size/SHA和失败闭环。
- `PROJECT_CONTEXT.md`、参数来源、调用流、服务器RLinf代码、训练配置和运行产物均未修改。

### FRM-0039～0045：final循环、指标与产物

| ID | 服务器时间/本机结束 | 操作与exit | 结果/服务器副作用 | 原始输出SHA256 |
|---|---|---|---|---|
| FRM-0039 | 11:32:11–11:32:12 | formal health只读刷新，exit 0；health SHA `6e554212…94cf` | 第25 wave只写1,107 rows到34,968并跑110 updates；第26 wave已完成，final eval开始；进程alive/OOM0。服务器无副作用。[日志](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0039_health_refresh.log) | `15c97f3c2dc8b9f8009f908be08c843d32be111e79172526775289169c73d8d9` |
| FRM-0040 | 11:32:40–11:32:44 | live-metrics只读刷新，exit 0；metrics SHA `90b78421…0560` | final eval进行中；最新完整点34,968 rows/2,496 updates/pending`.8`，ratio`.94767`。服务器无副作用。[日志](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0040_live_metrics_refresh.log) | `43e5f31fff0893eb25fab0e8d74ed025190c691d4fa94cf4ae350cf04eb8b167` |
| FRM-0041 | 11:33:19 | 四次SFTP GET，均exit 0 | 只读下载当时的driver/resource/metrics/TB event到`exports/ogpo_formal_20260808_timing_1132/`，共约7.9 MB；用于时间分段，不下载checkpoint。[清单](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0041_timing_snapshot.log) | `e7528c15578998fa70f094f4920b789dac3d1c1771a704ba800f4b634f1c899e` |
| FRM-0042 | 11:38:27 | formal health final刷新，exit 0 | driver/monitor均已自然退出、exit0；final严格35k/2,500；eval 5 waves完成；GPU compute0、OOM0。[日志](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0042_health_refresh.log) | `aaa075f082cfbb07febf42bb1ee141e05fa7e83f7b1f32d2dc26089c6064c868` |
| FRM-0043 | 11:38:39–11:38:43 | final live-metrics，exit 0 | final eval `1/20=5%`；35k/2,500/pending≈0；final ratio`.94224`、BC`.02019`、critic loss`.00953`，全部有限。[日志](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0043_final_metrics.log) | `2c41449b18983abd6a28e8068ebe671dabb0193b59a046b46bafae19eefa9b10` |
| FRM-0044 | 11:39:17–11:39:21 | final artifact audit，exit 0；audit SHA `547e53fe…3fbc` | finished11:37:32、checkpoint `global_step_26/complete=true`、run root62.526 GB；44,623秒资源、GPU峰57,231/57,676 MiB、cgroup checkpoint瞬态峰226.47 GB、OOM0；Git/hash不变。服务器无副作用。[日志](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0044_final_artifact_audit.log) | `5f48b84aba25860f1f85e0db7ebddee61386761a18921aec9581afafb04ac1de` |
| FRM-0045 | 11:40:24 | 九次SFTP GET，均exit 0 | 只读下载final driver/resource/metrics/TB、exit/finished/resources-after与`complete.json`到`exports/ogpo_formal_20260808_final/`，共约8 MB；大checkpoint留服务器。[清单](/C:/Users/86136/Documents/rl/exports/ogpo_formal_20260808_frm0045_final_evidence_download.log) | `a47deb338b15a67aeef8589a911aa1d121156e8b603038abd604b53c17ae82c8` |

### FRM-0039～0045 精确 launcher

继续使用前文process-only secret wrapper；每条命令结束后`finally`删除`SEETA_SSH_PASSWORD`。run操作为：

```powershell
# FRM-0039
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_health_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0039_health_refresh.log'

# FRM-0040
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_live_metrics_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0040_live_metrics_refresh.log'

# FRM-0042
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_health_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0042_health_refresh.log'

# FRM-0043
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260807_formal_live_metrics_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0043_final_metrics.log'

# FRM-0044
& '<bundled-python>' 'local_scripts\remote_exec_autodl.py' run `
  --command-file 'local_scripts\remote_ogpo_20260808_formal_artifact_audit_v1.sh' 2>&1 |
  Tee-Object -FilePath 'exports\ogpo_formal_20260808_frm0044_final_artifact_audit.log'
```

FRM-0041的精确SFTP主体为：

```powershell
$out='exports\ogpo_formal_20260808_timing_1132'
$pairs=@(
  @('/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime/resources_1s.csv',"$out\resources_1s.csv"),
  @('/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime/driver.log',"$out\driver.log"),
  @('/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1/metrics.log',"$out\metrics.log"),
  @('/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1/tensorboard/events.out.tfevents.1786115648.autodl-container-nekaqbwt43-6ce5babb.103684.0',"$out\events.out.tfevents.103684.0")
)
foreach($pair in $pairs) {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' get $pair[0] $pair[1]
  if($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

FRM-0045进一步读取：

```powershell
$runtime='/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260807_v1/runtime'
$run='/root/autodl-tmp/experiments/ogpo_robotwin_formal_20260807_v1'
$ckpt="$run/robotwin_adjust_bottle_ogpo_ca_formal_35k_v1/checkpoints/global_step_26/actor/ogpo_components"
$out='exports\ogpo_formal_20260808_final'
$pairs=@(
  @("$runtime/driver.log","$out\driver.log"),
  @("$runtime/resources_1s.csv","$out\resources_1s.csv"),
  @("$runtime/exit_code.txt","$out\exit_code.txt"),
  @("$runtime/finished_at.txt","$out\finished_at.txt"),
  @("$runtime/resources_after.txt","$out\resources_after.txt"),
  @("$run/metrics.log","$out\metrics.log"),
  @("$run/tensorboard/config.yaml","$out\tensorboard_config.yaml"),
  @("$run/tensorboard/events.out.tfevents.1786115648.autodl-container-nekaqbwt43-6ce5babb.103684.0","$out\events.out.tfevents.103684.0"),
  @("$ckpt/complete.json","$out\checkpoint_complete.json")
)
foreach($pair in $pairs) {
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' get $pair[0] $pair[1]
  if($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
```

下载文件size/SHA已写入FRM-0045日志；`checkpoint_complete.json` SHA为`c3ec625c…c557`，记录
step26/rows35000/version2500和完整合同。

### 本机final时间与统计命令

- 第一次PowerShell汇总把`foreach {...}`直接接`| Format-Table`，parser报`EmptyPipeElement`，exit1、无文件。
  窄修为先把对象加入`$rows`数组，再`$rows | Format-Table`；从final driver得到
  `generate_rollouts=12,714.3s`、`actor/run_training=30,732.518s`、`eval`已记录两点565s，另加baseline
  268s；总wall由started/finished相减为44,622s。
- SciPy Fisher命令因bundled runtime没有`scipy`而exit1；没有安装依赖。随后只用Python标准库
  `math.comb`计算20k `7/20`与final `1/20`在独立episode假设下的双侧exact值约`.043596`，exit0。
- 训练episode统计从每个final driver metric block的第一个`success_once`读取train值，求和为
  57/208；同block若有第二个`success_once`属于eval，不混入训练。

成功时间汇总的精确主体（第一次失败只少了`$rows=@()`并把`foreach`结果直接接pipe）：

```powershell
$lines=Get-Content -Encoding UTF8 'exports/ogpo_formal_20260808_final/driver.log'
$rows=@()
$names=@('generate_rollouts','actor/run_training','eval','sync_weights','rollout/predict','env/interact','step')
foreach($name in $names) {
  $vals=@()
  $pat='(?<![A-Za-z0-9_./])'+[regex]::Escape($name)+'=([0-9.]+)'
  foreach($line in $lines) { if($line -match $pat) { $vals += [double]$Matches[1] } }
  $m=$vals | Measure-Object -Sum -Average -Minimum -Maximum
  $rows += [pscustomobject]@{Metric=$name;Count=$vals.Count;SumSec=$m.Sum;MeanSec=$m.Average}
}
$rows | Format-Table -AutoSize
```

Fisher两次命令为：

```powershell
# exit 1：本机没有SciPy；未安装
& '<bundled-python>' -c "from scipy.stats import fisher_exact; print(fisher_exact([[7,13],[1,19]]))"

# exit 0：仅标准库
& '<bundled-python>' -c "import math; den=math.comb(40,20); ps={x:math.comb(8,x)*math.comb(32,20-x)/den for x in range(9)}; obs=ps[7]; print(sum(p for p in ps.values() if p<=obs+1e-15))"
```

### 2026-08-08 11:39 final复盘文档变更

- `HANDOFF.md`：切换为formal exit0终态，加入208 episodes、5%→35%→5%、时间/资源和checkpoint。
- `00_INDEX_AND_IMPLEMENTATION_PLAN.md`：新增§6.10预算耦合规则与开放决策；§9冻结并行基线；§10写终态。
- `02_CALL_AND_DATA_FLOW.md`：新增§3.4外层cycle，明确rollout/ingest/UTD/eval/checkpoint顺序与26-wave实账。
- `03_PARAMETER_PROVENANCE.md`：补实测burst、wall分布、eval/checkpoint疏忽、UTD候选和并行结论。
- `IMPLEMENTATION_LOG.md`：新增IMP-0041，记录FRM-0039～0045、final结果与本机分析失败/修复。
- 服务器RLinf代码、formal配置、checkpoint和运行产物未修改；`PROJECT_CONTEXT.md`未修改。

## 2026-08-08 final轻量包与下一轮预算复盘

本组全部在Windows本机读取FRM-0045已下载证据；没有连接服务器、写服务器产物、下载checkpoint、
启动训练或改RLinf代码。`<bundled-python>`、`<bundled-node>`和`<visualize-css>`分别指本线程由
workspace dependencies返回的固定运行时路径。

| ID | 本机操作与exit | 精确结果/问题 |
|---|---|---|
| PKG-0001 | `& '<bundled-python>' -m py_compile local_scripts\build_ogpo_formal35k_package.py`，exit0 | 新生成器语法通过；输入未写。 |
| PKG-0002 | `& '<bundled-python>' local_scripts\build_ogpo_formal35k_package.py --immutable exports\ogpo_formal_20260807_v1 --final exports\ogpo_formal_20260808_final --package exports\ogpo_formal_35k_high_info_20260808_v1 --visual-dir 'E:\Codex\home\visualizations\2026\08\05\019fd0f8-e1fa-76b1-816c-0ce8544216c4'`，exit0 | 解析26个wave和33,707行resource，复制约8MB原始轻量证据并生成表格/JSON/HTML；不含checkpoint。 |
| PKG-0003 | `& '<bundled-node>' local_scripts\render_inline_visualization_png.js ...training-fragment.html '<visualize-css>' ...ogpo-training-overview.png`，exit1 | 当前进程未设置bundled `NODE_PATH`，报`Cannot find module 'playwright'`；未生成PNG、未改输入。 |
| PKG-0004 | `$env:NODE_PATH='<bundled-node-modules>'; & '<bundled-node>' local_scripts\render_inline_visualization_png.js ...training-fragment.html '<visualize-css>' ...ogpo-training-overview.png`，exit0 | 使用已有Chrome生成training PNG；process结束即丢弃NODE_PATH。 |
| PKG-0005 | 与PKG-0004相同，仅输入/输出改为`resources-fragment.html`/`ogpo-resource-overview.png`，exit0 | 生成resource PNG。视觉复核发现均匀降采样漏掉窄checkpoint cgroup peak。 |
| PKG-0006 | 生成器加入`cgroup_current/gpu0/gpu1` extrema保留后，重跑PKG-0002、0004、0005，均exit0 | 最终resource图显示原始CSV的226,473,750,528-byte峰；training图不变。两张PNG经`view_image(original)`人工复核，无裁切/重叠。 |
| PKG-0007 | `& '<bundled-python>' local_scripts\build_ogpo_formal35k_package.py --package exports\ogpo_formal_35k_high_info_20260808_v1 --finalize --zip exports\ogpo_formal_35k_high_info_20260808_v1.zip`，exit0；随后标准库`ZipFile.testzip()`，exit0 | 28 members，`testzip=None`；最终zip 1,803,223 bytes。metrics/eval CSV为26/3条数据，SUMMARY精确复核GPU/cgroup/checkpoint bytes。 |

### PKG变更与产物

| 文件 | +/- 或 bytes | SHA256 | 用途 |
|---|---:|---|---|
| `local_scripts/build_ogpo_formal35k_package.py` | +676/-0；29,044 bytes | `d5cbd0560c17a1eb187f931854b11eb4e4f2c208f69fb2cd70fe8b0420b29b15` | 只读解析final evidence，生成轻量包与图表 |
| `exports/ogpo_formal_35k_high_info_20260808_v1.zip` | 1,803,223 | `962f026a26f0d87983d3edff3899b4ea2fcbd9f6eeb744b79b52d12d8e31d0fd` | 28文件checkpoint-free交付包 |
| `visuals/ogpo-training-overview.png` | 110,973 | `05a4971c030aac1d6cf11093f61f296af6ba97e533fcdfbdfa33cabbdd2a8aab` | success/eval、ratio、BC/critic曲线 |
| `visuals/ogpo-resource-overview.png` | 314,779 | `486326417e7daf3fba1ea08120fc475e1e7c04e1cc7efb5a61fe69165f1c8408` | wall allocation、GPU memory/util、cgroup memory |
| `visualizations/.../ogpo-formal-35k-overview.html` | 44,150 | `eba62292ea9adb84576e8e52729ac9f58a65dcaa0b35014a650452697c5abcdb` | 会话内完整训练/资源图 |

本轮文档修改为：主计划§6.10/§9/§10登记checkpoint组成、官方调度对照和90k/85k候选；
`03_PARAMETER_PROVENANCE.md`更新source/formal/next三层取值与24h实测换算；`IMPLEMENTATION_LOG.md`
新增IMP-0042；本文件新增PKG逐命令、失败、修复和SHA。`PROJECT_CONTEXT.md`与服务器代码/运行产物未修改。

## 2026-08-08 formal v2：fresh 90k正式启动

本组服务器操作使用同一个process-only secret wrapper；`<bundled-python>`为本线程返回的固定Codex
Python。每个helper结束后都在`finally`删除`SEETA_SSH_PASSWORD`。stdout/stderr保留在本任务tool
output，未另存原始本机日志；这里不补造不存在的raw文件。

```powershell
$credPath='E:\Codex\home\attachments\a0e17fb0-c2f9-4800-b353-c3a3d2090c83\pasted-text.txt'
$nonEmpty=@(Get-Content -LiteralPath $credPath -Encoding utf8 |
  Where-Object { $_.Trim().Length -gt 0 })
$sshIdx=-1
for($i=0;$i -lt $nonEmpty.Count;$i++) {
  if($nonEmpty[$i] -match '^ssh -p 36406 root@connect\.bjb1\.seetacloud\.com$') {
    $sshIdx=$i; break
  }
}
if($sshIdx -lt 0 -or $sshIdx+1 -ge $nonEmpty.Count) {
  throw 'credential source layout mismatch'
}
$pw=$nonEmpty[$sshIdx+1].Trim()
try {
  $env:SEETA_SSH_PASSWORD=$pw
  & '<bundled-python>' 'local_scripts\remote_exec_autodl.py' <EXACT_ARGS>
} finally {
  Remove-Item Env:SEETA_SSH_PASSWORD -ErrorAction SilentlyContinue
  $pw=$null; $nonEmpty=$null
}
```

下表每行的“命令”就是替换上述`<EXACT_ARGS>`的完整helper参数；`bash -n`行还通过
`--stdin-file`把对应本地脚本正文送给远端bash，只做语法检查。

| ID / 时间(+08) | 精确helper命令 | exit | 结果、问题与方法 |
|---|---|---:|---|
| FR2-0001 / 13:16:28 | `run --command-file local_scripts\remote_ogpo_20260808_formal_v2_readonly_preflight.sh`（SHA`7df375e3…de79`） | 0 | 身份、Git、source/norm SHA、新路径、进程、GPU、RAM/cgroup、磁盘、v1 completion只读检查；两卡空闲、host available约1.055TB、disk798.51GB、OOM0。 |
| FR2-0002a | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_v2_readonly_preflight.sh` | 0 | preflight语法通过。 |
| FR2-0002b | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_prepare_v2.sh` | 0 | prepare SHA`6eaab998…e004`，语法通过。 |
| FR2-0002c | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_run_v2.sh` | 0 | run SHA`ed694eea…6ee3`，语法通过。 |
| FR2-0002d | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_monitor_v2.sh` | 0 | monitor SHA`505665da…c0a1`，语法通过。 |
| FR2-0003 / 约13:20:58 | `run --command-file local_scripts\remote_ogpo_20260808_formal_prepare_v2.sh` | 1 | Hydra compose成功并留下runtime中的source/resolved；Python第54行assert失败。未启动训练、未创建run root。 |
| FR2-0004a | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_partial_inspect_v2.sh` | 0 | partial inspect SHA`6aac3f2d…1a19`语法通过。 |
| FR2-0004b / 13:22:11 | `run --command-file local_scripts\remote_ogpo_20260808_formal_partial_inspect_v2.sh` | 0 | run root absent；runtime仅source/resolved；resolved SHA`352f8e80…f56a36`；placement是等价单项map，全部预算/fresh字段正确。 |
| FR2-0005a | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_prepare_resume_v3.sh` | 0 | resume SHA`384100bc…8d4f`语法通过。 |
| FR2-0005b / 13:23:37 | `run --command-file local_scripts\remote_ogpo_20260808_formal_prepare_resume_v3.sh` | 0 | 不覆盖续完packet；写exact command/provenance/stop/resources-before；resolved SHA不变。 |
| FR2-0006a | `put local_scripts\remote_ogpo_20260808_formal_run_v2.sh /root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime/remote_ogpo_20260808_formal_run_v2.sh` | 0 | SFTP上传run，SHA`ed694eea…6ee3`。 |
| FR2-0006b | `put local_scripts\remote_ogpo_20260808_formal_monitor_v2.sh /root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime/remote_ogpo_20260808_formal_monitor_v2.sh` | 0 | SFTP上传1秒monitor，SHA`505665da…c0a1`。 |
| FR2-0006c | `put local_scripts\remote_ogpo_20260808_formal_health_v2.sh /root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime/remote_ogpo_20260808_formal_health_v2.sh` | 0 | SFTP上传health，SHA`67b49777…90ff`。 |
| FR2-0007a | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_verify_runtime_v2.sh` | 0 | verify-runtime SHA`9274bbc4…9290`语法通过。 |
| FR2-0007b / 13:25:16 | `run --command-file local_scripts\remote_ogpo_20260808_formal_verify_runtime_v2.sh` | 0 | Git/新路径/无进程、run/monitor/health语法和所有override通过；输出9个immutable SHA及exact command。 |
| FR2-0008a | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_launch_v2.sh` | 0 | launch SHA`d94dbaeb…0158`语法通过。 |
| FR2-0008b | `put local_scripts\remote_ogpo_20260808_formal_launch_v2.sh /root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime/remote_ogpo_20260808_formal_launch_v2.sh` | 0 | launch上传成功；尚未执行。 |
| FR2-0008c | `run "test (sha256sum '/root/autodl-tmp/experiment_exports/ogpo_robotwin_formal_20260808_v2/runtime/remote_ogpo_20260808_formal_launch_v2.sh' \| awk '{print \\}') = 'd94dbaeb955460964d325b7be6a6f3ade5af0b3ea32e1baa2603695ceee20158'"` | 2 | PowerShell破坏远端`$(...)`转义，bash报`syntax error near unexpected token sha256sum`；launch未执行。改用独立command-file核验。 |
| FR2-0009a | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_verify_launch_upload_v2.sh` | 0 | verify-launch SHA`ba474bbe…3632`语法通过。 |
| FR2-0009b | `run --command-file local_scripts\remote_ogpo_20260808_formal_verify_launch_upload_v2.sh` | 0 | 远端launch SHA精确为`d94d…0158`且`bash -n`通过。 |
| FR2-0010a | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_launch_exec_v2.sh` | 0 | launch-exec SHA`e6fbd55c…c335`语法通过。 |
| FR2-0010b / 13:27:34 | `run --command-file local_scripts\remote_ogpo_20260808_formal_launch_exec_v2.sh` | 0 | 再核远端launch SHA后唯一启动；driver PID99676、monitor PID99677；5秒后均alive。 |
| FR2-0011 / 13:27:57 | `run --command-file local_scripts\remote_ogpo_20260808_formal_health_v2.sh` | 0 | 两进程alive、19 resource rows、run/TensorBoard已创建，尚在worker初始化，OOM0。 |
| FR2-0012 / 13:29:06 | 同FR2-0011 | 0 | norm stats已加载，4个compute进程；两卡约8.3GiB、cgroup约85GB、OOM0；Curobo为已知未使用可选导入。 |
| FR2-0013 / 13:30:01 | 同FR2-0011 | 0 | FSDP初始化继续；两卡约23.3GiB、cgroup约90.6GB、OOM0。 |
| FR2-0014 / 13:31:01 | 同FR2-0011 | 0 | baseline `Evaluating Rollout Epochs 0/5`开始；两卡约28GiB、cgroup约98GB、OOM0。 |
| FR2-0015 / 13:32:38 | `run --command-file local_scripts\remote_ogpo_20260808_formal_status_v2.sh`（SHA`cd467849…8c4e`） | 0 | eval 2/5，driver/monitor alive、OOM0。 |
| FR2-0016 / 13:33:59 | 同FR2-0015 | 0 | eval 4/5，资源平稳、OOM0。 |
| FR2-0017 / 13:34:56 | 同FR2-0015 | 0 | baseline 5/5完成，driver/monitor alive、OOM0。 |
| FR2-0018 / 13:35:42 | 同FR2-0015 | 0 | runner已切到首个`Generating Rollout Epochs 0/1`。 |
| FR2-0019a | `run 'bash -n' --stdin-file local_scripts\remote_ogpo_20260808_formal_startup_verify_v2.sh` | 0 | startup verify SHA`44785487…26e`语法通过。 |
| FR2-0019b / 13:36:35 | `run --command-file local_scripts\remote_ogpo_20260808_formal_startup_verify_v2.sh` | 0 | TB baseline20 episodes、success_once5%；首个train rollout已开始；410 resource samples，GPU峰28729/28164MiB、cgroup峰115.17GB、OOM0。停止主动观察，不停止后台训练。 |

### Formal v2 immutable packet

```text
resolved.yaml       352f8e80752d60624a0c53c62d21dcc10bdc8e6712a433c18d0eec0ee1f56a36
source_config.yaml  f777a0caceacf260c427093fdb0a493f6bd77c3d444cf233a57727ce9027c291
exact_command.txt   0faa8cdd976a400aec6ff48cff3c1c0e3b0336e3e885b77eacef00b0631452f6
run_provenance.tsv  4470be3551d00fbc4285c897be6c11390a9cfc03402cff744aca8fd386abf8a3
stop_conditions.txt 2eaf6e5dee226acdc891f0149553a012cb2e5f54d18111bfa44f2bba0171f1d2
run script           ed694eeadae6e6e8506bee45aeb500a20f2a9402f2db7eab60c4ba12aa556ee3
monitor script       505665dab6f3afc2082b156e7c6296a4f4002a1132e566624a886e500acdc0a1
health script        67b497778fb0718918c17e4392ebe0b58afabdb049c3b38d349a62c788c890ff
launch script        d94dbaeb955460964d325b7be6a6f3ade5af0b3ea32e1baa2603695ceee20158
```

### 本轮新增本机脚本

| 文件 | bytes | SHA256 | 用途 |
|---|---:|---|---|
| `remote_ogpo_20260808_formal_v2_readonly_preflight.sh` | 2,156 | `7df375e3…de79` | live identity/Git/resources/collision preflight |
| `remote_ogpo_20260808_formal_prepare_v2.sh` | 9,823 | `6eaab998…e004` | 首次compose；断言失败现场保留 |
| `remote_ogpo_20260808_formal_partial_inspect_v2.sh` | 1,353 | `6aac3f2d…1a19` | 只读核验partial packet |
| `remote_ogpo_20260808_formal_prepare_resume_v3.sh` | 9,113 | `384100bc…8d4f` | 不覆盖续完packet |
| `remote_ogpo_20260808_formal_run_v2.sh` | 2,067 | `ed694eea…6ee3` | exact formal driver |
| `remote_ogpo_20260808_formal_monitor_v2.sh` | 3,775 | `505665da…c0a1` | 1秒资源CSV |
| `remote_ogpo_20260808_formal_health_v2.sh` | 2,123 | `67b49777…90ff` | 详细启动健康检查 |
| `remote_ogpo_20260808_formal_verify_runtime_v2.sh` | 3,403 | `9274bbc4…9290` | immutable packet/SHA/override校验 |
| `remote_ogpo_20260808_formal_launch_v2.sh` | 3,086 | `d94dbaeb…0158` | 唯一后台driver+monitor启动 |
| `remote_ogpo_20260808_formal_verify_launch_upload_v2.sh` | 443 | `ba474bbe…3632` | 修复临时转义失败后的远端SHA核验 |
| `remote_ogpo_20260808_formal_launch_exec_v2.sh` | 382 | `e6fbd55c…c335` | SHA锁定后执行launch |
| `remote_ogpo_20260808_formal_status_v2.sh` | 1,459 | `cd467849…8c4e` | 紧凑状态快照 |
| `remote_ogpo_20260808_formal_startup_verify_v2.sh` | 2,274 | `44785487…26e` | TB baseline、train切换与资源摘要 |

本轮没有修改服务器RLinf源码、Git branch或checkpoint；只在新的formal v2 experiment/runtime root
写入已授权的配置、日志、监控和运行产物。`PROJECT_CONTEXT.md`不修改。

### Formal v2文档变更与本机QA

| 文件 | 本轮净行数变化 | 具体增加/修改 |
|---|---:|---|
| `HANDOFF.md` | 107→121（+14） | 路由切到v2 active；90k resolved、root、授权和13:36健康快照 |
| `00_INDEX_AND_IMPLEMENTATION_PLAN.md` | 665→695（+30） | §6.10落地决策/先例，§8阶段6，§9状态，§10动态快照 |
| `03_PARAMETER_PROVENANCE.md` | 377→392（+15） | v1/v2/source三层值、v2两阶段、burst先例和23.71h口径 |
| `IMPLEMENTATION_LOG.md` | 743→800（+57） | 固定授权边界及IMP-0043～0045 |
| `COMMAND_AND_CHANGE_INDEX.md` | 882→1003（+121） | FR2-0001～0019逐命令、immutable SHA、脚本表与本机QA |

- 首次本机`git diff --check/status`因仓库owner与Codex沙箱用户不同触发`dubious ownership`；没有改
  global Git config。改用单次`git -c safe.directory=C:/Users/86136/Documents/rl ...`后确认该工作区
  整个专题仍是既有untracked文档，故Git无法给出逐文件numstat；没有stage/commit。
- 随后的只读PowerShell文档QA第一次因字符串`"$file:..."`被解析为非法变量引用而在读文件前exit1；
  改成`"${file}:..."`后通过：五个文档均无trailing whitespace/conflict marker，且HANDOFF、resolved
  SHA、10-eval、IMP-0045、FR2-0019b五个关键文本存在。该失败与修复只影响本机QA命令，不影响服务器。
- 最终QA首次把远端stdout marker`OGPO_ROBOTWIN_FORMAL_V2_RUNTIME_VERIFIED`误当作必须逐字写进文档，
  因needle不存在exit1；改核文档中的等价叙述`runtime immutable审计于13:25:16通过`后exit0，无文件副作用。

## 2026-08-08 formal v2：21.8k只读状态、轻量下载与可视化

本组沿用上一节完整的process-only password wrapper；服务器只执行两条只读command-file，随后由同一
helper做10次SFTP `get`。没有`put`、远端写命令、停止/重启或大文件读取。`<bundled-python>`、
`<bundled-node>`、`<node-modules>`和`<visualize-css>`仍指workspace dependencies返回的固定路径。

| ID / 时间(+08) | 精确命令或操作 | exit | 结果、问题与方法 |
|---|---|---:|---|
| FR2-0020 / 18:05:14 | `run --command-file local_scripts\remote_ogpo_20260808_formal_health_v2.sh` | 0 | driver/monitor alive、exit pending；最近完整table 21,802 rows/590 updates；20k eval已完成；GPU约56.8GB/卡、cgroup155.33GB、OOM0。 |
| FR2-0021 | `apply_patch`新增`remote_ogpo_20260808_formal_live_audit_v2.sh`和本轮下载/生成脚本；随后`Get-FileHash -Algorithm SHA256 ...` | 0 | 只改本机新文件；live audit固定light artifact/checkpoint/TB/resource/exception/progress字段。未执行的TB dump原型在最终收口前删除。 |
| FR2-0022 / 18:06:58 | `run --command-file local_scripts\remote_ogpo_20260808_formal_live_audit_v2.sh` | 0 | 进程仍alive；TB精确21,802 rows、590/590 updates；eval 5%/5%/15%；无checkpoint；CUDA OOM/Ray death/NaN均0。12,611个当时resource rows，GPU峰57,231/56,834MiB、cgroup峰155.82GB、OOM0。stdout保留在task tool output，未另造远端报告。 |
| FR2-0023 | `& 'local_scripts\download_ogpo_20260808_formal_live_v2.ps1'`（密码仅在父进程env） | PowerShell策略拒绝；外层错误地显示0 | Windows execution policy在脚本正文运行前抛`PSSecurityException`；未连接服务器、未下载文件。原因是调用方式，不是SSH或训练问题。 |
| FR2-0024 / 约18:07–18:08 | `powershell -NoProfile -ExecutionPolicy Bypass -File local_scripts\download_ogpo_20260808_formal_live_v2.ps1` | 0 | 仅当前子进程绕过本机策略；10个SFTP `get`全部`GET_OK`，总约3.17MB。目标`exports\ogpo_formal_90k_live_20260808_1807\`；逐文件bytes/SHA/remote/local写入本次stdout并固化到package manifest。 |
| FR2-0025 | `<bundled-python> -c "...build_ogpo_formal35k_package.parse_metric_tables/parse_resources..."`；`rg -n -C 8 'Traceback|CUDA out of memory|ActorDiedError|RayTaskError|NaN' <snapshot>\driver.log` | 0 | 解析15 waves/120 episodes/24 success、wall分段和优化序列；仅两段已知Curobo可选导入Traceback，无训练异常。 |
| FR2-0026 | `<bundled-python> -m py_compile local_scripts\build_ogpo_formal90k_live_snapshot.py`；随后`<bundled-python> local_scripts\build_ogpo_formal90k_live_snapshot.py --snapshot exports\ogpo_formal_90k_live_20260808_1807 --visual-dir E:\Codex\home\visualizations\2026\08\05\019fd0f8-e1fa-76b1-816c-0ce8544216c4 --zip exports\ogpo_formal_90k_live_20260808_1807.zip` | 0/0 | 生成metrics/eval/resource CSV/JSON、running summary、两个HTML fragment和初版zip；不把v1 final常量带入v2。 |
| FR2-0027 | 依次执行`$env:NODE_PATH='<node-modules>'; <bundled-node> local_scripts\render_inline_visualization_png.js <training|resources fragment> <visualize-css> <snapshot>\visuals\<training|resources>.png` | 0/0 | 生成129,568-byte training PNG和320,223-byte resource PNG；只用本机Chrome和已下载CSV。 |
| FR2-0028 | `view_image(training.png, high)`与`view_image(resources.png, high)` | 0/0 | 人工核验曲线、图例、轴、峰值与wall allocation可读，无裁切；训练与资源图均通过。 |
| FR2-0029 | 重跑FR2-0026生成器以把PNG纳入manifest/zip；随后标准库`ZipFile.testzip()`并断言manifest含两PNG | 0 | `PACKAGE_OK 19 20`；19个manifest payload加manifest自身共20 members，最终zip 988,480 bytes，`testzip=None`。 |
| FR2-0030 | `<bundled-python> -c "...analyze_ogpo_formal_live.parse_resources..."`；`rg -n -i '<!doctype|<html|<head|<body' <two fragments>`；最终`Get-FileHash` | 0 | 资源trace无>5秒gap、两卡allocated约4.675 GPU-hour、util-equivalent合计6.50 GPU-hour、能耗约1.84kWh；两个fragment无完整HTML壳且均<2MB。 |
| FR2-0031 / 18:19:25 | `run --command-file local_scripts\remote_ogpo_20260808_formal_status_v2.sh` | 0 | 最终交付前只读刷新：driver/monitor alive、exit pending；23,125 rows/656 updates、ratio`.777`、最近burst66；下一rollout已完成；两卡56,765/56,834MiB、cgroup157.10GB、OOM0。18:08包保持冻结。 |
| FR2-0032 | PowerShell逐文件检查5份更新文档的conflict marker/trailing whitespace并用`rg -n -F`核关键章节；标准库重算19个manifest SHA并执行`ZipFile.testzip()` | 0 | `FINAL_QA_OK 19 20`；主计划§10、参数§6.3、IMP-0046、FR2 live节和HANDOFF动态现场均命中。 |

FR2-0024的10个下载目标固定为：

```text
runtime/driver.log                 -> driver.log
runtime/resources_1s.csv           -> resources_1s.csv
runtime/resolved.yaml              -> resolved.yaml
runtime/source_config.yaml         -> source_config.yaml
runtime/exact_command.txt          -> exact_command.txt
runtime/run_provenance.tsv         -> run_provenance.tsv
runtime/stop_conditions.txt        -> stop_conditions.txt
run_root/metrics.log               -> metrics.log
run_root/tensorboard/config.yaml    -> tensorboard_config.yaml
run_root/tensorboard/events...99682.0 -> events.out.tfevents
```

### FR2 live脚本与产物

| 文件 | bytes | SHA256 | 用途 |
|---|---:|---|---|
| `local_scripts/remote_ogpo_20260808_formal_live_audit_v2.sh` | 5,031 | `ffb52a7bd1908523521b3b4233b5d9a1ba9c231ab280c970b872fd393414b42f` | 单次只读现场/产物/TB/资源审计 |
| `local_scripts/download_ogpo_20260808_formal_live_v2.ps1` | 1,519 | `bf2827afe3cf3ab9b5a47d6c8882e0576f07044f63378f5869f04c00d5be228f` | 10项checkpoint-free SFTP下载清单 |
| `local_scripts/build_ogpo_formal90k_live_snapshot.py` | 14,052 | `24556683e6481c1898471475ad406a8365c7493cf3b1e5c81954bb1c649db688` | v2 running派生表、图和轻量包生成器 |
| `exports/ogpo_formal_90k_live_20260808_1807.zip` | 988,480 | `320de927c5282c4efea8ff1b3eac08f6d608fa2fdc56c94168296a4c1fe79a5c` | 20-member轻量快照；不含checkpoint/replay |
| `visuals/training.png` | 129,568 | `6caf20a65097b0a92c14e6cb474b97fa3f44723efd8d02ba0b8ffe79b42d5fa6` | train/eval、ratio、BC/critic、Q/TD |
| `visuals/resources.png` | 320,223 | `85e0cc7db65728b8969611c3de73d6653cbff3e9beef354d54f90644141b6dc7` | wall、GPU memory/util、cgroup memory |
| `visualizations/.../ogpo-formal-v2-live-training.html` | 11,422 | `6a312c47fd7a5f7d66f3b1c02cbdb032b917ddccb6cba0e4152bf2023b947f27` | 会话内训练图 |
| `visualizations/.../ogpo-formal-v2-live-resources.html` | 55,017 | `2da24139d7962d97352a84c6bc71d76ba4113a78b01be7d7a4e8df342c0121a4` | 会话内资源图 |

本组文档只更新主计划§10、参数来源§6.3、`HANDOFF`动态现场、`IMPLEMENTATION_LOG` IMP-0046和本节；
`PROJECT_CONTEXT.md`、服务器RLinf源码/config/runtime、正在运行的driver/monitor均未修改。

## 2026-08-09 formal v2：64,078-row终态、失败诊断与轻量包

本组继续使用process-only password与固定host-key的Paramiko helper。服务器调用均为只读shell或SFTP
`get`；没有`put`、停止/重启、恢复checkpoint、改代码/config、删除或覆盖服务器产物。命令中的
`<bundled-python>`/`<bundled-node>`/`<visualize-css>`是workspace dependency返回的固定绝对路径；
口令从当前进程环境注入且不写日志。

| ID / 时间(+08) | 精确命令或操作 | exit | 结果、问题与方法 |
|---|---|---:|---|
| FR2-0033 | `<bundled-python> local_scripts\remote_exec_autodl.py run --command-file local_scripts\remote_ogpo_20260809_formal_final_audit_v2.sh 2>&1 \| Tee-Object exports\ogpo_formal_20260809_fr2_0033_final_audit.log` | 0 | driver/monitor已退出、exit255；64,078 rows/2,703 updates；7个eval点；30k/60k checkpoint complete；Git clean/upstream0/0/remote HEAD同5d5c84e3；抓到Ray 228.21/240 GiB kill→NCCL timeout因果链和checkpoint前后RSS。 |
| FR2-0034 | `<bundled-python> local_scripts\remote_exec_autodl.py run --command-file local_scripts\remote_ogpo_20260809_formal_metrics_dump_v2.sh > exports\ogpo_formal_20260809_fr2_0034_metrics_dump.json 2>&1` | 0 | TensorBoard dump包含`env/`、`train/`、`eval/`、`time/`全部scalar；TensorFlow stderr被PowerShell合并在JSON后，生成器用`JSONDecoder.raw_decode`只读取首个JSON对象。 |
| FR2-0035 | `& local_scripts\download_ogpo_20260809_formal_partial_v2.ps1` | PowerShell策略拒绝 | 脚本正文和SSH连接前被`PSSecurityException`阻止；服务器无操作、本机目标未改。 |
| FR2-0036 | `powershell -NoProfile -ExecutionPolicy Bypass -File local_scripts\download_ogpo_20260809_formal_partial_v2.ps1 2>&1 \| Tee-Object exports\ogpo_formal_20260809_fr2_0036_download_partial.log` | 0 | 17个`GET_OK`：driver、1秒资源、resolved/source/exact/provenance/stop、start/finish/exit/resources before/after、metrics、TB config/event及两个checkpoint `complete.json`；不取checkpoint正文。 |
| FR2-0037 | `<bundled-python> -m py_compile local_scripts\build_ogpo_formal90k_partial_package.py`；随后`<bundled-python> local_scripts\build_ogpo_formal90k_partial_package.py --snapshot exports\ogpo_formal_90k_partial_64078_20260809_v2 --tb-dump exports\ogpo_formal_20260809_fr2_0034_metrics_dump.json --audit exports\ogpo_formal_20260809_fr2_0033_final_audit.log --visual-dir E:\Codex\home\visualizations\2026\08\05\019fd0f8-e1fa-76b1-816c-0ce8544216c4 --zip exports\ogpo_formal_90k_partial_64078_20260809_v2.zip` | 首次1，窄修后0 | 首次按UTF-8读取audit后因实际UTF-16/NUL导致`substring not found`；只增加BOM/NUL编码检测后重跑成功，未改变原始证据。 |
| FR2-0038 | 对training/resources各执行`$env:NODE_PATH='<node-modules>'; <bundled-node> local_scripts\render_inline_visualization_png.js <fragment> <visualize-css> <png>`；随后两次`view_image(..., high)` | 0 | 生成165,957-byte训练PNG和360,276-byte资源PNG；轴、图例、eval曲线、checkpoint RSS台阶、228 GiB阈值和GPU资源均可读，无裁切。 |
| FR2-0039 | 重跑FR2-0037把PNG纳入manifest/zip；`<bundled-python> -m zipfile -t exports\ogpo_formal_90k_partial_64078_20260809_v2.zip`；标准库逐项重算manifest SHA | 0 | 30个payload哈希一致，zip 2,430,833 bytes，`testzip=None`；SHA256 `13e4e7b258cb800f6d95a077c1e3303b8fe5ad87939b36eddedeaf9535c8bcc4`。 |
| FR2-0040 | `git -c safe.directory=C:/Users/86136/Documents/rl status --porcelain=v2 --branch`、`git ... remote -v`、`git ... ls-files` | 0 | 本地根仓为initial、无remote、0 tracked；近期实验证据未上云。服务器Git结果来自FR2-0033，目标实现已推且remote同SHA。 |
| FR2-0041 | `apply_patch`更新主计划/参数来源/HANDOFF/实施与命令账；`rg`核当前状态、低起点审计、失败根因；PowerShell检查conflict marker/trailing whitespace；`<bundled-python> -m zipfile -t ...`并重算manifest；核两个fragment无HTML壳 | 首次QA 1，窄修后0 | 首次误用系统`python.exe`，Windows返回`file cannot be accessed`；此前Git/doc检查已完成且无写副作用。改用workspace dependency给出的bundled Python后`Done testing`、`PACKAGE_QA_OK 30`，五份文档均`DOC_OK`。事实源由“v2 active 23k”切到“64,078 partial failure”；`PROJECT_CONTEXT.md`未改，服务器无写操作。 |

### FR2终态脚本与产物

| 文件 | bytes | SHA256 | 用途 |
|---|---:|---|---|
| `local_scripts/remote_ogpo_20260809_formal_final_audit_v2.sh` | 8,047 | `a8ecb84b14bebc6b1db8f8044af30a15d6e1804a2a6d64673777e346e6ec87ca` | 终态/故障/checkpoint/RSS/Git只读审计 |
| `local_scripts/remote_ogpo_20260809_formal_metrics_dump_v2.sh` | 1,028 | `ff3009d09625e8509236827c564768e238a1e942bf5e5bd231dedd3cc6292c85` | 全量主要TensorBoard scalar dump |
| `local_scripts/download_ogpo_20260809_formal_partial_v2.ps1` | 2,932 | `0b0942e0b5ec097ffc90bb6feadc6a7f8e68909fe7a361a08e29cf13031785ef` | 17项轻量SFTP清单 |
| `local_scripts/build_ogpo_formal90k_partial_package.py` | 18,920 | `fbd2eca656d5f5bfff3aeca3fd10c276c1f0b9c6bc6bf3d34802fc5b601cf1f5` | 终态解析、图、manifest与zip |
| `exports/ogpo_formal_90k_partial_64078_20260809_v2.zip` | 2,430,833 | `13e4e7b258cb800f6d95a077c1e3303b8fe5ad87939b36eddedeaf9535c8bcc4` | 30-file轻量终态包；无checkpoint/replay正文 |
| `ogpo-formal-v2-partial-training.html/png` | 15,072 / 165,957 | `56ee0f56…1884` / `073e8236…30a4` | 会话训练图与静态PNG |
| `ogpo-formal-v2-partial-resources.html/png` | 79,975 / 360,276 | `6df62085…5817` / `16eeaec4…905a` | 会话资源/失败图与静态PNG |

### 文档事实源变更索引

| 文件 | 明确更新 |
|---|---|
| `00_INDEX_AND_IMPLEMENTATION_PLAN.md` | §6.11新增跨算法首点协议表与matched A/B；§8/§9改为v2部分结束；§10以64,078-row终态替换过期live快照 |
| `03_PARAMETER_PROVENANCE.md` | §5/§6把v2状态改为2,703 updates，并登记checkpoint clone/RSS资源约束与`.05`结论边界 |
| `IMPLEMENTATION_LOG.md` | 新增IMP-0047终态/失败/包和IMP-0048低起点调查 |
| `HANDOFF.md` | 动态路由改为无进程、两个可恢复点、当前授权边界和代码/证据云端状态 |
| `COMMAND_AND_CHANGE_INDEX.md` | 新增FR2-0033～0041逐命令、失败窄修、脚本/产物SHA与本表 |

本轮未修改`PROJECT_CONTEXT.md`、服务器RLinf源码、resolved config、checkpoint或远端Git。
