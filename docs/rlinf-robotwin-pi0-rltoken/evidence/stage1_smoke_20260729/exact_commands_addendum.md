# Stage 1 smoke：执行命令补录

> 日期：2026-07-29
>
> 原始 [`exact_commands.txt`](exact_commands.txt) 保持服务器生成时的字节和 SHA-256
> `1ea9f65590f211c027641fadc98fea142a0b4cd64b2da3a8cd7472cf1d22dc2b` 不变。
> 本文从当时实际执行且仍保留的 start/status/postcheck 脚本、driver log 和实施账本恢复遗漏
> 命令；没有为补录而重跑 S1-A、reload-only 或 S1-B。状态脚本的精确调用次数没有单独保存，
> 因此这里只列实际使用的命令模板和已知首末结果，不伪造轮询次数。

所有远端调用都通过 `local_scripts/remote_exec_autodl.py`；密码只在调用进程环境中短暂注入，
结束后移除，不进入参数、脚本、日志、本文或 Git。

## 1. 脚本上传

上传本身只同步脚本，不启动作业。实际使用的命令为：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_resource_monitor.sh `
  /root/autodl-tmp/tmp/rlt_stage1_resource_monitor_20260729.sh

python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_start_s1a.sh `
  /root/autodl-tmp/tmp/rlt_stage1_start_s1a_20260729.sh
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_status_s1a.sh `
  /root/autodl-tmp/tmp/rlt_stage1_status_s1a_20260729.sh

python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_start_s1a_reload.sh `
  /root/autodl-tmp/tmp/rlt_stage1_start_s1a_reload_20260729.sh
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_status_s1a_reload.sh `
  /root/autodl-tmp/tmp/rlt_stage1_status_s1a_reload_20260729.sh

python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_start_s1b.sh `
  /root/autodl-tmp/tmp/rlt_stage1_start_s1b_20260729.sh
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_status_s1b.sh `
  /root/autodl-tmp/tmp/rlt_stage1_status_s1b_20260729.sh

python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_stage1_postcheck.sh `
  /root/autodl-tmp/tmp/rlt_stage1_postcheck_20260729.sh
```

## 2. S1-A 启动与状态

外层实际调用：

```powershell
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_start_s1a_20260729.sh"

python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_status_s1a_20260729.sh"
```

start wrapper 先检查 branch、clean tree、数据/stats/monitor 存在、输出不存在以及无相关
SFT/Ray 进程，然后以 `nohup` 启动原始 `exact_commands.txt` 中的 S1-A foreground 命令和
每秒资源 monitor；foreground 外层硬超时为：

```bash
timeout --signal=TERM --kill-after=60s 1800s
```

状态命令只读输出 `STATE/EXIT_CODE`、当前 GPU/RAM、driver tail、resource CSV 行和 checkpoint
文件，不发送信号。已知终态：driver PID `611074`、exit `0`、global step 2 checkpoint
存在。

## 3. S1-A 新进程 reload-only

外层实际调用：

```powershell
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_start_s1a_reload_20260729.sh"

python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_status_s1a_reload_20260729.sh"
```

start wrapper 的 branch/process/input/output 预检通过后，在新 Ray/worker 进程中实际运行的
foreground 命令为：

```bash
cd /root/autodl-tmp/RLinf_rlt_pi0_robotwin
export PYTHONPATH=/root/autodl-tmp/RLinf_rlt_pi0_robotwin:/root/autodl-tmp/RoboTwin_RLinf
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0,1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export ROBOTWIN_RLT_CLEAN50_PATH=/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-contract-ep0-v1
export ROBOTWIN_PI0_BASE_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle
export ROBOTWIN_PI0_NORM_STATS_PATH=/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle/physical-intelligence/robotwin/norm_stats.json

timeout --signal=TERM --kill-after=60s 1800s \
  /root/autodl-tmp/RLinf/.venv/bin/python -B examples/sft/train_vla_sft.py \
    --config-path /root/autodl-tmp/RLinf_rlt_pi0_robotwin/examples/sft/config \
    --config-name robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke \
    runner.logger.log_path=/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1/s1a_reload \
    runner.logger.experiment_name=robotwin_adjust_bottle_rlt_stage1_s1a_reload_only_v1 \
    +runner.resume_dir=/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1/s1a/robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1/checkpoints/global_step_2
```

`runner.max_steps=2` 由 smoke source config 继承，所以恢复后进度直接为 `2/2`，没有第三步和
第二份 checkpoint。已知终态：driver PID `621054`、exit `0`。

## 4. S1-B 启动与状态

外层实际调用：

```powershell
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_start_s1b_20260729.sh"

python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_status_s1b_20260729.sh"
```

start wrapper 的预检、30 分钟 hard timeout 和 resource monitor 与 S1-A 相同；foreground
训练命令就是原 `exact_commands.txt` 的 S1-B 命令。状态脚本只读输出
`STATE/EXIT_CODE/GPU/RAM/log tail/resource rows/checkpoint count`。已知终态：driver PID
`629633`、exit `0`、checkpoint file count `0`。

## 5. Postcheck 两次尝试

第一次实际执行：

```powershell
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_postcheck_20260729.sh"
```

训练与 reload 进程当时都已 exit 0；postcheck 自身因过度要求 Ray driver 转发内部短语而失败：

```text
RuntimeError: reload log lacks DCP load marker
```

窄修复只把 evidence assertion 从内部日志短语改成：

1. resolved log 有精确 `resume_dir`；
2. 新进程 exit 0；
3. progress 直接为 `2/2`；
4. DCP metadata 和 full weights 仍存在。

没有改训练代码、checkpoint 或 runtime log，也没有重跑 S1-A/reload/S1-B。修复后的脚本再次
用第 1 节的 `put` 命令上传，并用同一 `run` 命令复测；第二次 exit 0，生成
`stage1_postcheck.json`。

## 6. LR scheduler 发现、修复与 CPU contract

文档 QA 从 S1-A step 2 日志的 `6.25e-8` 发现，smoke-time source config 的绝对
`min_lr=2.5e-6` 没有实现声明的 floor。实际只读审计和修复命令入口为：

```powershell
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_lr_scheduler_audit.sh `
  /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_audit_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_audit_20260729.sh"

python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_fix_preflight_20260729.sh"
python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_commands/robotwin_rlt_stage1_sft_openpi.formal_current.yaml `
  /root/autodl-tmp/RLinf_rlt_pi0_robotwin/examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_fix_verify_20260729.sh"

python local_scripts/remote_exec_autodl.py put `
  local_scripts/remote_rlt_20260729_lr_scheduler_contract.sh `
  /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_contract_20260729.sh
python local_scripts/remote_exec_autodl.py run `
  "bash /root/autodl-tmp/tmp/rlt_stage1_lr_scheduler_contract_20260729.sh"
```

一次内联 PowerShell preflight 在本地 quoting 解析阶段失败，未连接、未写服务器；随后改为
上面的版本化 preflight 脚本。CPU contract 第一次因浮点数 exact-equality assertion 失败，
没有生成 evidence；把断言窄改为 `math.isclose` 后，同一 probe 通过。它只 compose config
并用一个 CPU scalar 调用仓库现有 scheduler，不加载模型、数据 batch 或 GPU，也不重跑 smoke。

修复后的正式 source config SHA-256 为
`8340ef4e953877de510da18548d0a69802104b7b2f8218698cd0fb586b49a8f2`；contract 见
[`lr_scheduler_contract.json`](lr_scheduler_contract.json)，SHA-256 为
`e68a7da1457e32538995f39b41f23a21b34d248e2ed3fa37f1177476e7c614df`。

## 7. 磁盘审计命令

完整 A–P 服务器命令、两次只读失败和窄修复单列在
[`../DISK_AUDIT_COMMANDS_20260729.md`](../DISK_AUDIT_COMMANDS_20260729.md)。该审计没有
删除、移动、改名、覆盖或压缩任何文件。
