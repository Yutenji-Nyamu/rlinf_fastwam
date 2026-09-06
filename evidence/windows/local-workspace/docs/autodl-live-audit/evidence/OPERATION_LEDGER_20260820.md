# AUTODL-A800 2026-08-20 只读现场盘点流水账

## 范围与边界

- 任务：使用用户明确提供的 SeetaCloud 密码 SSH 端点，刷新既有 AutoDL 容器现场。
- 机器标签：`AUTODL-A800`；不得与当前根交接中的 `SZ-H100` 混写。
- 当前授权：只读身份、资源、进程、Git 工作树、日志与 checkpoint 元数据检查；本地记录本次脱敏证据。
- 未授权且未执行：服务器写入、安装/下载、启动或停止进程、smoke、训练、删除/覆盖文件。
- 凭据：仅通过当前 PowerShell 进程的安全输入转成临时环境变量；本文和命令文件均不记录密码。

## 逐步记录

### A0820-001 — 读取工作区入口与当前专题事实源

- 位置：`WIN-LOCAL`。
- 读取：`PROJECT_CONTEXT.md`、`HANDOFF.md`、`docs/rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md`。
- 结果：根交接当前专题是 `SZ-H100` 新路线；本次用户指定端点是旧 `AUTODL-A800`，后续状态必须独立记录。
- 变更：无。

### A0820-002 — 核对既有密码 SSH helper 与历史工作规程

- 位置：`WIN-LOCAL`。
- 读取：`local_scripts/remote_exec_autodl.py` 及相关长期记忆条目。
- helper 默认端点：`connect.bjb1.seetacloud.com:36406`，用户 `root`。
- helper 行为：固定 SHA256 host key；低层 Paramiko `Transport.start_client()` + `auth_password()`；认证前最多 3 次有界重试（1/3 秒）；认证后 keepalive 30 秒；禁用密钥/agent 路线。
- 结果：端点和本次用户给出的 SSH 地址一致，可直接复用；无安装、无 helper 修改。

### A0820-003 — 定位本机执行时 Python

- 位置：`WIN-LOCAL`。
- 结果：使用 Codex bundled Python：`C:/Users/86136/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`；helper 可从既有 E 盘独立依赖目录导入 Paramiko。
- 变更：无；未安装 Anaconda 或新依赖。

### A0820-004 — 首次身份探针输入方式失败与窄修复

- 位置：`WIN-LOCAL -> AUTODL-A800`。
- 首次方法：非交互 PowerShell 从 stdin 读取密码，再设置进程内 `SEETA_SSH_PASSWORD` 并调用 helper。
- 结果：stdin 已关闭，命令在发起 SSH 前退出：`missing process-only SSH password`。这不是网络、host key 或服务器认证失败。
- 修复：改为 PTY 中 `Read-Host -AsSecureString`，只在当前 PowerShell 进程转成临时字符串并注入环境变量；命令结束后删除变量。
- 复测远端命令（凭据省略）：`hostname; pwd; id -u`。
- 复测结果：exit `0`；hostname `autodl-container-nekaqbwt43-6ce5babb`；cwd `/root`；UID `0`。
- 结论：固定 host key 校验和密码认证均成功，确认连接到既有目标容器。

### A0820-005 — 核对本地工作区基线

- 位置：`WIN-LOCAL`。
- 命令：`git -c safe.directory='C:/Users/86136/Documents/rl' status --short`、`rev-parse --show-toplevel`、`rev-parse --verify HEAD`。
- 问题：普通 `git status` 先被不同 Windows 用户所有权的 `safe.directory` 检查阻止；未修改全局 Git 配置，改为单命令临时 `-c safe.directory=...`。
- 结果：工作区 Git 尚无可解析 `HEAD`，现有目录整体为 untracked；本次未覆盖或清理任何既有文件。

## 待执行

- 无待执行的只读检查；任何恢复、实现、测试、smoke、训练或进程控制都需要进入相应专题并重新确认授权边界。

### A0820-006 — 一批完成系统、资源与产物粗盘点

- 位置：`WIN-LOCAL -> AUTODL-A800`。
- 精确远端命令文件：`local_scripts/remote_autodl_readonly_audit_20260820.sh`。
- 连接：固定 host key 的低层 Paramiko 密码路线；密码只在当前进程；exit `0`。
- 现场时间：`2026-08-20T12:16:30+08:00`。
- 系统：Ubuntu 22.04.5；144 CPU；约 1.0 TiB 主机内存，检查时约 984 GiB available；无 swap。
- cgroup v2：`memory.max=257698037760`（240 GiB），`memory.high=253403070464`（236 GiB），检查时 `memory.current=408723456`；`max/oom/oom_kill` 均为 `0`。
- 数据盘：`/root/autodl-tmp` 为 XFS，约 2.1 TiB，总使用约 1.3 TiB，剩余 824 GiB，inode 余量充足；根 overlay 剩余约 21 GiB。
- GPU：2 × NVIDIA A800-SXM4-80GB，driver `580.126.09`；两卡计算显存均为 `0 MiB`，利用率 `0%`，无 compute process。
- 进程：只有容器基础服务（Jupyter、TensorBoard、AutoPanel、proxy、sshd）和本次只读命令；没有训练、Ray、Torch/NCCL worker。
- 会话：无 tmux；screen 无 session。
- 容器基础服务启动时间：`2026-08-20 09:19:17+08:00`；数据盘上的旧工作树和实验目录仍保留。
- 问题：粗盘点的 Git/近期文件/checkpoint 输出较长，在工具显示层被截断；远端命令已完整结束，不是服务器或脚本失败。
- 处理：改用两个更窄的只读批次，只打印相关工作树与 OGPO 终态证据。

### A0820-007 — 相关工作树与运行目录定向复核

- 精确远端命令文件：`local_scripts/remote_autodl_targeted_audit_20260820.sh`；exit `0`。
- 复核时两张 GPU 仍为 `0 MiB / 0%`，没有计算进程。
- 相关 Git 工作树均无 tracked 改动、无 untracked 文件：
  - DSRL 共用树：`/root/autodl-tmp/RLinf_fastwam_rlinf`，`codex/dsrl-pi0-robotwin@48a775db09c16c455aeba7b0600c920e7c80d534`。
  - RLT：`/root/autodl-tmp/RLinf_rlt_pi0_robotwin`，`codex/rlt-pi0-robotwin@2b8199d8ab2e7b110994fd3234bf7007196c3af9`。
  - QAM：`/root/autodl-tmp/RLinf_qam_pi0_robotwin`，`codex/qam-pi0-robotwin@ff8e28ef6a4d485642e695b6b76c5c84187e134e`。
  - OGPO：`/root/autodl-tmp/RLinf_ogpo_pi0_robotwin`，`codex/ogpo-pi0-robotwin@5d5c84e3ac4efa1713a4139a05ac1b776e634ed3`。
  - 旧基础 RLinf：`/root/autodl-tmp/RLinf`，`local/openpi-a800-2gpu-migration@6d0db56bf26f972cd27fa29535f5eb939e80e5bf`，tracked clean、5 个既有 untracked 文件。
- 标识符核对：上述 RL 仓及 `/root/autodl-tmp` 根仓均不能解析 `8cde1ff^{commit}`；因此不能把用户粘贴经验中的 `8cde1ff` 归因到本机这些 RL 工作树或 checkpoint。
- OGPO 目录仍在：smoke apparent 14 GiB；formal v1 apparent 59 GiB；formal v2 apparent 145 GiB。

### A0820-008 — OGPO v2 终态与恢复点只读核验

- 精确远端命令文件：`local_scripts/remote_autodl_ogpo_terminal_check_20260820.sh`；exit `0`。
- v2 启动：`2026-08-08T13:27:34+08:00`；结束：`2026-08-09T06:03:27+08:00`；driver exit `255`；monitor exit `0`。
- frozen command 仍是 `90,000 total rows / 10,000 warmup / UTD-Q=UTD-PI=0.05 / capacity 100,000 / eval 10,000 / checkpoint 30,000`。
- `global_step_22`：54 GiB，6 个 checkpoint 文件，存在 `actor/ogpo_components/complete.json`。
- `global_step_43`：92 GiB，6 个 checkpoint 文件，存在 `actor/ogpo_components/complete.json`。
- driver 末尾仍是 Ray/NCCL worker 终止链；没有发现后续 resume 或新运行目录。
- 说明：初次尝试在 run 子目录读取 `metrics.log` 返回 missing；粗盘点已确认 metrics 位于 experiment 根目录，且根交接中的 64,078 rows / 2,703 updates 终态与现场最后指标一致。未为此增加无信息量的第四次日志扫描。

### A0820-009 — 两个 OGPO 恢复点完成清单

- 精确远端命令文件：`local_scripts/remote_autodl_ogpo_manifest_check_20260820.sh`；exit `0`。
- `global_step_22` 的 `complete.json`：`complete=true`，`global_online_rows=30959`，`policy_version=1047`，sidecars 为 `rank_0.pt/rank_1.pt`。
- `global_step_43` 的 `complete.json`：`complete=true`，`global_online_rows=60968`，`policy_version=2548`，sidecars 为 `rank_0.pt/rank_1.pt`。
- 两者合同均记录 2 ranks、H=50、执行 C=10、active action 14D、replay capacity 100,000、warmup 10,000、total 90,000、UTD-Q/PI 0.05。
- 结论：恢复点正文和完整性清单仍在；本次没有加载权重、恢复训练或改动任何服务器文件。

## 本地文件变更索引

- 新增本文：保存脱敏的完整操作、结果、问题、窄修复和复测证据。
- 新增 `local_scripts/remote_autodl_readonly_audit_20260820.sh`：系统/资源/工作树/产物粗盘点命令。
- 新增 `local_scripts/remote_autodl_targeted_audit_20260820.sh`：相关 RL 工作树和 OGPO 产物定向命令。
- 新增 `local_scripts/remote_autodl_ogpo_terminal_check_20260820.sh`：OGPO v2 终态与 checkpoint 元数据命令。
- 新增 `local_scripts/remote_autodl_ogpo_manifest_check_20260820.sh`：读取两个小型 `complete.json` 的命令。
- 已更新根 `HANDOFF.md`：只写本次 live 结论与下一步边界，没有复制长流水。
