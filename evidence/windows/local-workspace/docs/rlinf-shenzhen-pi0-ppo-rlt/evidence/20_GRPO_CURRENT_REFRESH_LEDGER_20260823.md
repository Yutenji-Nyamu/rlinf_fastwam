# 深圳 GRPO 当前现场刷新流水账（2026-08-23）

## 1. 本轮目标与边界

- 目标：只读刷新 `grpo-formal100-current-4gpu128train64eval-ppo-matched-v2` 的当前进度、完整训练指标、fixed-64 评估、GPU/RAM/Swap、进程健康与主要产物，并生成从 Step 1 到最新完整 step 的静态可视化。
- 授权边界：本轮仅做服务器与本地证据读取、轻量日志下载、离线解析和文档整理；不停止、不重启、不修改训练，不写服务器文件。
- 动态事实口径：服务器现场 > 当前交接文档 > 历史快照。
- SSH：普通账号 `chenyiteng`，Paramiko 密码认证，固定校验服务器 host key；密码仅进入调用进程环境。

## 2. 操作流水

### 2.1 恢复工作区规则与专题单一事实源

完整读取：

```powershell
Get-Content -LiteralPath PROJECT_CONTEXT.md -Raw
Get-Content -LiteralPath HANDOFF.md -Raw
Get-Content -LiteralPath docs/rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md -Raw
Get-Content -LiteralPath docs/rlinf-shenzhen-pi0-ppo-rlt/20_GRPO_STEP36_LIVE_METRICS_AND_RESOURCE_REFRESH_20260823.md -Raw
Get-Content -LiteralPath docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/18_GRPO_LIVE_REFRESH_20260823.md -Raw
```

结果：确认当前运行根目录、参数口径、Step 36 历史断点，以及本轮只能把新 SSH 现场称为“当前”。

### 2.2 复核既有采集与绘图实现

读取既有 `final_status`、产物审计、内存分解、SFTP 下载和 Step 36 绘图脚本；确认可以沿用同一解析口径，但需要去掉 Step 36 硬编码并覆盖完整新历史。

### 2.3 建立本轮只读采集脚本

新增：

- `local_scripts/remote_commands/shenzhen_grpo_v2_current_refresh_20260823.sh`
- `local_scripts/invoke_shenzhen_grpo_v2_current_refresh_20260823.ps1`

远端脚本仅执行 `ps`、`grep`、`stat`、`find`、`du`、`nvidia-smi`、`free`、`vmstat`、`df` 和 `/proc`/cgroup 读取；本地包装器在固定 host-key 校验后执行该脚本，并下载 `metrics.log`、`resource.csv`、`driver.log` 三个轻量文件。没有服务器写操作。

远端 command file SHA-256：

```text
2DB841FDF1684CD2EAEEFEA13A40C73E8EEFC7A70D42507F39F504048626D33F
```

## 3. 现场结果

执行入口：

```powershell
& 'C:\Users\86136\Documents\rl\local_scripts\invoke_shenzhen_grpo_v2_current_refresh_20260823.ps1'
```

Paramiko 连接使用普通账号 `chenyiteng` 和已锁定 host key；返回码 `0`。远端只读 command file 为
`local_scripts/remote_commands/shenzhen_grpo_v2_current_refresh_20260823.sh`。

现场时间 `2026-08-23T07:59:59+00:00`：

- driver/observer alive；actor/rollout/EnvWorker=`4/4/4`，GCS/raylet=`1/1`；
- 完整 Step 42，随后 Step 43 rollout 到 `2/4`；
- Step 42 success/KL/clip/grad/ratio-abs=`0.9628906/0.019/0.079/13.832/0.077`；
- Step 40 fixed-64=`0.984375=63/64`；
- fatal pattern=`0`，metrics nonfinite token=`0`，exit marker 均未出现；
- cgroup current=`1909614964736 bytes=1778.47 GiB`，swap current=`5.952 GiB`；host available约`270.8 GiB`；
- `memory.events` 的 high/max/oom/oom_kill 全0，memory PSI为0，三次`vmstat`的si/so均为0；
- anonymous/file/shmem/page-table/slab约`1694.1/78.5/23.2/3.6/1.85 GiB`；
- 4个EnvWorker RSS合计约`1655.1 GiB`，actor约`37.2 GiB`，rollout约`21.2 GiB`；RSS有共享页，未将三者相加冒充cgroup PSS；
- GPU4–7现场显存=`59901/60763/61571/60069 MiB`，利用率=`2/0/2/0%`；
- run=`69 GiB`；checkpoint Step10/20/30/40各约18 GiB；train/eval MP4=`680/16`；
- `/`可用234 GiB，`/data`可用约3.0 TiB。

下载结果：

```text
driver.log    322552 bytes
metrics.log   242876 bytes
resource.csv   83954 bytes
```

问题与处理：无SSH、认证、训练或下载错误；没有实施额外修复。

## 4. 离线解析、可视化与复核

新增动态绘图脚本：

```powershell
C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  -m py_compile local_scripts/render_shenzhen_grpo_live_current_20260823.py
C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  local_scripts/render_shenzhen_grpo_live_current_20260823.py
```

两条命令均退出0。后检：

- metric table 精确解析出连续 Step 1–42；全部目标标量有限；
- resource CSV 有988条数据行，首尾UTC=`2026-08-22T15:28:57`至`2026-08-23T07:59:14`，最大间隔61秒；
- 生成成功率、优化/耗时、资源三张独立PNG，并逐张视觉检查通过；
- GRPO最近5/10步success=`95.20%/94.96%`；Step40 fixed64=`63/64`；
- cgroup slope Step31–36=`+6.87 GiB/step`，Step37–42=`+21.81 GiB/step`，不能继续沿用“增长已明显放慢”的旧现场判断；
- 本地证据目录总量`1,071,051 bytes`，没有大文件或checkpoint落到C盘。

完整结论与图见 `22_GRPO_STEP42_LIVE_METRICS_AND_RESOURCE_REFRESH_20260823.md`。
