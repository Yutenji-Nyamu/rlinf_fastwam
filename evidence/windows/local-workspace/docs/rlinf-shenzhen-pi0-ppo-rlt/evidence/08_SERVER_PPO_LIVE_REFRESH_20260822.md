# SZ-H100 PPO、整机与 Git 只读刷新（2026-08-22 14:59–15:02 CST）

边界：固定 host-key 的低层 Paramiko 密码认证；普通项目现场使用 `chenyiteng`，其他用户与系统状态使用
`toom` 的只读 sudo 视角。没有修改服务器文件、代码、Git、服务或训练进程，没有读取其他用户文件内容、
SSH key 正文、订阅 URL 或 token，也没有进行 GitHub 认证尝试。

## 1. 可复现命令

通用入口：

```powershell
C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  local_scripts/verified_password_ssh.py `
  --host 120.241.223.9 --port 22 --user chenyiteng `
  --host-key-sha256 qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY `
  run --command-file <COMMAND_FILE>

C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  local_scripts/verified_password_ssh_sudo_stdin.py `
  --host 120.241.223.9 --port 22 --user <ACCOUNT> `
  --host-key-sha256 qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY `
  --command-file <COMMAND_FILE>
```

| command file | SHA256 | 账号 / exit | 只读内容 |
|---|---|---|---|
| `shenzhen_ppo_server_git_refresh_20260822.sh` | `b7c50ac2966823263e84405e36567814bf4bf49623b7646debafdeafcd1beabc` | `chenyiteng / 0` | PPO、TensorBoard、产物、GPU/RAM、存储、网络、Git/SSH 文件名 |
| `shenzhen_other_users_readonly_refresh_20260822.sh` | `a0ac5e2b756f48a6917a4662a3387e4f007ca4eff0242174eeddb478134c5558` | `toom / 0` | 登录会话、按用户资源、GPU owner、系统与磁盘健康 |
| `shenzhen_mihomo_quota_sanitized_20260821.sh` | `9a0053c17275b32b1d1ce570cc883328a9c3bdb7a1e665214145d34bfadf71c2` | `chenyiteng / 0` | numeric quota/status；probe hash 预检后只读 sudo |
| `shenzhen_ppo_latest_metrics_tail_20260822.sh` | `d3eae7395259456c2aeb5a8d056fbb6bd4700522e3b3e98ce6c8607f98fa24a4` | `chenyiteng / 0` | 最新完整 console metric table |

密码只进入当前进程；以上 command file 均在 Windows 工作区，未上传到服务器。

## 2. PPO 当前状态与主要指标

15:02 CST 的事实：

- driver PID `1375834` 与 Ray 均 alive；console 已完整输出 `Global Step 33/100`，随后进入下一步
  `Generating Rollout Epochs 0/4`。
- fatal 扫描为 `0`：没有 traceback、CUDA OOM、Ray worker crash、NCCL failure、SIGKILL 或 no-space。
- GPU 4–7 的全部 12 个 compute PID 都属于 `chenyiteng` 的 PPO：4 个 EnvWorker、4 个 rollout
  worker、4 个 FSDP actor。物理 GPU 0–3 没有 compute process。

最近四张完整 console 表：

| Global Step | train success | KL | clip fraction | grad norm | critic EV | value loss | step time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 84.18% | 0.016 | 0.074 | 33.214 | 0.408 | 0.037 | 1,987.3 s，含定期 eval |
| 31 | 93.16% | 0.0057 | 0.026 | 21.589 | 0.456 | 0.018 | 1,585.6 s |
| 32 | 91.80% | 0.011 | 0.054 | 19.935 | 0.430 | 0.021 | 1,589.6 s |
| 33 | 87.50% | 0.013 | 0.044 | 34.882 | 0.420 | 0.031 | 1,542.6 s |

这些训练量均 finite；成功率有正常单步波动，KL/clip/critic EV 没有显示数值发散。最新 Step 33 中
rollout `1,517.3 s`，actor training `22.929 s`，仍是 simulator rollout 主导墙钟时间。

fixed-64 inline eval 的三次结果为：

```text
Step 10  58/64 = 90.625%
Step 20  62/64 = 96.875%
Step 30  58/64 = 90.625%
```

Step 30 回到 58/64，说明三点仍有方差，不能据此声称单调提升；但也没有出现 eval 崩坏。TensorBoard
event step 比 console 的 `Global Step` 显示小 1，本节统一按 console/checkpoint 的人类可读 step 命名，
避免两种索引混写。

## 3. 主要产物

run：
`/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1`

| 产物 | 当前现场 |
|---|---|
| run 总量 | `52G` |
| checkpoints | `global_step_10/20/30`，各约 `18G`；每份含 4 个 `.distcp`、1 个 `.metadata` 和完整 `model_state_dict` |
| train videos | `528` 个 MP4，合计 `134,495,132 B` |
| eval videos | `12` 个 MP4，合计 `1,615,112 B` |
| TensorBoard | 1 个 event file，约 `92.7 KB`；主要 train/eval/time scalars 均存在且 finite |
| 运行元数据 | `resolved.yaml`、`launch_manifest.txt`、`driver.pid`、`driver.log`、`metrics.log`、TensorBoard `config.yaml` |

Step 30 checkpoint 与第三次 eval/video 均已自然落盘。没有检查 checkpoint reload，因为本次只是运行现场审计，
产物结构也与 Step 10/20 一致。

## 4. 主存：同相位比较确认增长尚未平台化

15:00 CST，刚完成 Step 33、下一步 `0/4` 的现场：

```text
cgroup memory.current = 1,814,963,118,080 B = 1.651 TiB
host MemAvailable     =   361,730,492 KiB = 345.0 GiB
cgroup swap.current  = 0
memory.events high/max/oom/oom_kill = 0/0/0/0
```

四个 EnvWorker 的 PSS 为约 `398.8 / 417.2 / 388.3 / 370.6 GiB`，合计
`1,651,300,673 KiB = 1.538 TiB`；仍几乎全部是 private anonymous memory，不是 checkpoint/page cache。

与 11:52 CST **相同执行相位**（完整 Step 26、下一步 `0/4`）比较：

| 指标 | Step 26 next 0/4 | Step 33 next 0/4 | 7-step 变化 | 近期均摊/step |
|---|---:|---:|---:|---:|
| cgroup | `1,602,268,200,960 B` | `1,814,963,118,080 B` | `+198.09 GiB` | `+28.30 GiB` |
| 4×EnvWorker PSS | `1,450,109,793 KiB` | `1,651,300,673 KiB` | `+191.87 GiB` | `+27.41 GiB` |
| host MemAvailable | `562,086,232 KiB` | `361,730,492 KiB` | `-191.07 GiB` | `-27.30 GiB` |

因此，“早期增长已经平台化”现在可以排除：近期增长仍由 EnvWorker 私有匿名内存主导。系统尚未产生
memory pressure/OOM 事件，训练也仍健康，但仅剩约 345 GiB available，已从黄灯进入需要尽快决策的高风险
状态。表中的 per-step 只是最近 7 步同相位斜率；它不能线性保证未来速度，也不能用来断言精确 OOM step。

**需要用户决定**：继续让当前进程自然运行，还是利用已经完整落盘的 Step 30 checkpoint，规划停止/恢复或
降低 simulator 并发。此次审计没有控制进程，也不替用户作这个决定。

## 5. GPU、其他用户与整机

- 15:00 的相位快照中 GPU 4–7 约 `55–59 GiB/卡`；稍前同一 rollout 切换点约 `55–56 GiB/卡`。
  显存随 rollout 子阶段变化，仍留有约 `22–26 GiB/卡`；没有 GPU OOM。
- GPU 0–3 为 `69/4/4/4 MiB` 且无 compute PID；其他用户当前没有使用 GPU。
- `liwenbo` 总 RSS 约 `45 MiB`，仅见 tmux/watchdog 等轻量进程；10:29 快照约 `1.47 GiB`，现已明显下降。
  `zhangwei` 总 RSS 约 `218 MiB`，主要是 Codex/tmux，和 10:29 的约 `0.22 GiB` 基本相同；
  `xiongzizhen` 当前无进程统计项。没有读取任何人的项目文件或 shell history。
- 整机 load average `8.15/8.68/8.67`，相对 128 CPU 不高；PPO 是当前唯一明显重型工作负载。
- failed systemd unit 为 0；SSH、Mihomo、Docker、containerd active；当日没有新 NVRM Xid、OOM、
  I/O/filesystem/NVMe severe match。

## 6. 存储、网络与 quota

| mount | used / available | inode use | 相比 10:28 CST |
|---|---|---:|---|
| `/` | `48G / 236G` | 2% | 基本不变 |
| `/home` | `94G / 2.2T` | 1% | used 约 `+11G` |
| `/data` | `234G / 3.1T` | 1% | used 约 `+56G` |

空间变化与本轮已知的 Fast-WAM env/models、Step 30 新增约 18G checkpoint、视频和运行增长相容；三个
文件系统及 inode 均无压力，根盘没有被项目下载挤占。没有遍历其他用户目录来强行分摊磁盘变化。

Mihomo 为 active/enabled，监听 `127.0.0.1:7890`；显式加载 profile 后 GitHub/Hugging Face 均 HTTP 200。
15:01 sanitized quota：`80.499 GiB used / 19.501 GiB remaining`，到期
`2026-08-23 14:01:48 CST`。相比 12:56 的 19.531 GiB 只少约 0.030 GiB，没有出现续费扩容或延期变化。

## 7. 当前 RLinf Git/worktree 与 GitHub 连接准备

| tree | HEAD | branch/status | remotes |
|---|---|---|---|
| canonical `/data/chenyiteng/projects/rlinf-shenzhen/RLinf` | `7d07a4212ee6858cc333e1d4fab7a37256d1f839` | detached、clean | 只有 official `origin=https://github.com/RLinf/RLinf.git` |
| PPO worktree `.../worktrees/ppo-pi0-robotwin` | 同上 | `codex/sz-ppo-pi0-robotwin`、clean | 与 canonical 共享同一组 remotes |

`git worktree list` 目前只有以上两棵；尚无 GRPO worktree，也没有 `personal` remote。`chenyiteng` 的
`~/.ssh` 目录不存在，因此现场没有可供这个个人仓 push 的用户级 deploy key/known_hosts。此次只检查了
文件名/权限是否存在，没有创建 key、读取 key 正文、执行 `ssh -T` 或修改 Git 配置。

因此最干净的后续是：在同一 canonical clone 里从 current base 建 GRPO 独立 branch/worktree；第一次
需要 push 时再建立一个 repo-scoped deploy key、由用户把公钥加入个人 GitHub 仓库并允许写，然后添加
`personal` remote。worktree 本身不需要网页操作；网页只负责登记公钥。该动作尚未执行。

