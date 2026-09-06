# Idea2 DVAC telemetry：2-GPU / 16-env 真实 smoke 执行流水账

日期：2026-08-20  
机器：AUTODL-A800  
状态：已自然完成；driver exit code 0，数据/资源/视频后检与最终只读审计通过。

## 1. 用户决策与本次边界

用户明确纠正：GPU/RAM监控只用于观察，不得触发告警、自动停止、timeout或任何运行行为变化；此前提出的
70/78 GiB、200/220 GiB和15分钟边界全部不进入本次运行。用户同时授权真实smoke，并希望串行度尽量小、
并行度贴近现有评估，允许参考16或32并发的历史记录。

本轮选择：

```text
2 × A800-80GB
total_num_envs = 16（每个GPU/rank 8 env）
rollout_epoch = 1
max_episode_steps = 200
H / C / M / D_active = 50 / 50 / 4 / 14
expected episodes = 16
expected policy queries <= 64
```

选择16而不是32：两卡16-env是本工作区已有实际运行记录的并发规模；它比原2-env候选更接近正常评估，
同时避免第一次真实telemetry链路直接翻倍到没有同样完整现场证据的32 env。资源监控仅旁路记录，run不会因
任何观测值被信号、kill或timeout。

## 2. REMOTE-SMOKE-001：启动前现场刷新

连接方式：固定host-key的低层Paramiko，密码只注入当前PowerShell进程；身份认证成功。精确command-file：
`tmp/idea2_presmoke16_live_audit.sh`。服务器时间2026-08-20 18:44:32 +08:00：

```text
host = autodl-container-nekaqbwt43-6ce5babb
pwd = /root
uid = 0
source HEAD = remote HEAD = 61996e15cc7f5a32bd6012b61b20893d94636c82
branch = codex/idea2-dvac-pi0-robotwin
source dirty count = 0
RoboTwin_RLinf = 481380fbd97cbf9ff830aedfb2279851e1e58969
checkpoint bytes = 8,067,761,500
2-env candidate SHA256 = 62207051678d2ad515c74aa574282b9bc04592f3b97ea10258d43263483b4f71
new 16-env config = absent
new 16-env output = absent
GPU0/1 = A800-80GB, 0 MiB, 0% util, 30/31 C
GPU compute processes = 0
matching eval/Ray processes = none
cgroup current = 2,007,707,648 bytes
cgroup high/max = 253,403,070,464 / 257,698,037,760 bytes
memory.events low/high/max/oom/oom_kill = all 0
/root/autodl-tmp free = 824G
/dev/shm free = 120G
PRESMOKE16_LIVE_AUDIT_PASS=1
```

该操作只读，没有写服务器文件或启动进程。

## 3. LOCAL-SMOKE-002：16-env配置与只观察监控脚本

使用`apply_patch`新建下列本地文件；此阶段没有接触服务器运行态：

```text
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/SMOKE_RESOLVED_2GPU_16ENV_V1.yaml
local_scripts/idea2_dvac_prepare_smoke_2gpu16env_v1.sh
local_scripts/idea2_dvac_validate_smoke_2gpu16env_v1.sh
local_scripts/idea2_dvac_observe_resources_2gpu16env_v1.sh
local_scripts/idea2_dvac_launch_smoke_2gpu16env_v1.sh
local_scripts/idea2_dvac_start_smoke_2gpu16env_v1.sh
```

与冻结的2-env候选做机械差异核对，唯一的评估语义变化是：

```text
total_num_envs: 2 -> 16
run_id / output_dir / log_path / launch_command: 切换到唯一的2gpu_16env_v1命名
```

其余模型、环境、任务、H/C/M/D、固定reset ID、checkpoint、norm stats、episode长度、相机、
offload和视频配置保持不变。脚本行为为：核心driver自然运行并自然退出；旁路observer每2秒读取
`nvidia-smi`、cgroup、`/proc`、`df`和进程RSS。没有资源阈值、告警动作、timeout或向driver发送信号。

本地问题与修复：最初编辑16-env YAML时误多写了一个`dsrl_hidden_dims: 128`元素；在任何上传、解析或
运行前即通过`apply_patch`删除。随后与2-env文件的机械差异确认没有其他参数漂移。验证脚本中的
禁止行为关键字检查最初使用`grep -E '\\b...\\b'`；为避免GNU ERE对`\\b`的解释差异，改成
POSIX字符边界`(^|[^[:alnum:]_])...([^[:alnum:]_]|$)`，尚未上传旧版本。

执行的哈希指令：

```powershell
Get-FileHash -Algorithm SHA256 `
  docs\rlinf-robotwin-pi0-dvac-telemetry\evidence\SMOKE_RESOLVED_2GPU_16ENV_V1.yaml, `
  local_scripts\idea2_dvac_prepare_smoke_2gpu16env_v1.sh, `
  local_scripts\idea2_dvac_validate_smoke_2gpu16env_v1.sh, `
  local_scripts\idea2_dvac_observe_resources_2gpu16env_v1.sh, `
  local_scripts\idea2_dvac_launch_smoke_2gpu16env_v1.sh, `
  local_scripts\idea2_dvac_start_smoke_2gpu16env_v1.sh
```

结果：

```text
config   d4b7393e1a02f118f6ea5ab2d303f92e83779d3e2a968a7920aebadbd35bc018
prepare  575274ca9d6de8aa01a8f98de0513cdd93cafd9f51439398504665ede5e141bb
validate 62f4aca1ec7e82a5533fff297193dcb5bda0b1fe198c237c3155a3474e14c7c5
observe  9c1b9a7d02a4a3742bb2b8adba44e6f83cd44a11c88f754011c3c507b19e72c1
launch   42b4dc995107852a963faf88fe725c449cb92609b33062df361cfa1579af2665
start    724fae3ce6ffbfc710193fceb5ccd7ac79b9024f708bc5d5a88100d690c8e1dd
```

## 4. REMOTE-SMOKE-003：准备、上传与无模型前置验证

### 4.1 创建唯一runtime目录

精确本地调用：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py run `
  --command-file local_scripts\idea2_dvac_prepare_smoke_2gpu16env_v1.sh
```

服务器脚本依次验证source HEAD、remote branch、clean tree、配置不存在、输出不存在、runtime目录不存在，
然后只创建runtime目录。结果：

```text
RUNTIME_DIR=/root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1
CONFIG_STATE=ABSENT
OUTPUT_STATE=ABSENT
PREPARE_SMOKE16_PASS=1
```

### 4.2 SFTP上传

通过同一条已认证Paramiko连接逐文件执行：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py put `
  docs\rlinf-robotwin-pi0-dvac-telemetry\evidence\SMOKE_RESOLVED_2GPU_16ENV_V1.yaml `
  /root/autodl-tmp/idea2_dvac_run_configs/idea2_dvac_sft_smoke_2gpu_16env_v1.yaml
& $codexPython local_scripts\remote_exec_autodl.py put `
  local_scripts\idea2_dvac_observe_resources_2gpu16env_v1.sh `
  /root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1/observe_resources.sh
& $codexPython local_scripts\remote_exec_autodl.py put `
  local_scripts\idea2_dvac_launch_smoke_2gpu16env_v1.sh `
  /root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1/launch.sh
& $codexPython local_scripts\remote_exec_autodl.py put `
  local_scripts\idea2_dvac_start_smoke_2gpu16env_v1.sh `
  /root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1/start.sh
```

SFTP均成功，无重试。服务器`sha256sum`与本地逐字节一致：

```text
config  d4b7393e1a02f118f6ea5ab2d303f92e83779d3e2a968a7920aebadbd35bc018
observe 9c1b9a7d02a4a3742bb2b8adba44e6f83cd44a11c88f754011c3c507b19e72c1
launch  42b4dc995107852a963faf88fe725c449cb92609b33062df361cfa1579af2665
start   724fae3ce6ffbfc710193fceb5ccd7ac79b9024f708bc5d5a88100d690c8e1dd
```

### 4.3 完整配置解析与脚本合同

精确本地调用：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py run `
  --command-file local_scripts\idea2_dvac_validate_smoke_2gpu16env_v1.sh
```

服务器依次执行：三支运行脚本`bash -n`；检索并拒绝独立的`kill/pkill/timeout/TERM/INT`行为词；
通过真正入口`eval_embodied_agent.py --cfg job --resolve`解析完整Hydra配置；用OmegaConf断言2卡placement、
16 env、1 epoch、200 slots、H/C/M/D=50/50/4/14、3相机、固定reset ID、视频和DVAC开关；最后再次
验证输出目录仍不存在。结果：

```text
SMOKE16_CONFIG_CONTRACT_PASS=1
EXPECTED_EPISODES=16
EXPECTED_POLICY_QUERIES=64
RESOURCE_MONITOR_MODE=OBSERVE_ONLY
CONFIG_SHA256=d4b7393e1a02f118f6ea5ab2d303f92e83779d3e2a968a7920aebadbd35bc018
ENTRYPOINT_CFG_RESOLVE_PASS=1
NO_MODEL_OR_ENV_RUN=1
```

解析入口导入TensorFlow时输出oneDNN/CPU指令提示；它不是错误，退出码为0，也没有加载checkpoint、启动
Ray或环境。

### 4.4 observer最小机制探针

精确调用：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py run `
  --command-file tmp\idea2_dvac_observer_dummy_test.sh
```

该探针令observer旁路观察一个自然存活5秒的`sleep`进程；没有向进程发送信号。结果：

```text
OBSERVER_DUMMY_PROBE_PASS=1
RESOURCE_GPU_ROWS=6
SAMPLES=3
OBSERVER_NATURAL_EXIT_AT=2026-08-20T18:57:40+08:00
```

两张GPU每轮各一行，因此3轮对应6行；driver自然消失后observer自然退出。至此仍未启动真实smoke。

## 5. REMOTE-SMOKE-004：真实smoke启动

用户已明确授权两卡16-env真实smoke。启动前在聊天中再次展示完整配置hash、输出目录、16 episodes/
最多64 queries预算、参数血缘、精确核心命令，以及observer只观察、不设阈值/timeout/signal的行为。

精确本地调用：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py run `
  --command-file local_scripts\idea2_dvac_start_smoke_2gpu16env_v1.sh
```

`start.sh`验证唯一output尚不存在、launcher PID文件尚不存在；用`nohup bash launch.sh`启动本次唯一
launcher。`launch.sh`再次验证source/remote commit、clean tree、配置hash、runtime和output，然后启动第5.3
节核心driver与旁路observer。没有timeout或信号动作。结果：

```text
LAUNCHER_PID=21052
LAUNCHER_STARTED_AT=2026-08-20T19:03:17+08:00
DRIVER_PID=21072
OBSERVER_PID=21073
STATUS=RUNNING
DETACHED_SMOKE_STARTED=1
```

本次启动只发生一次；后续只读轮询`status.env`、driver日志和资源CSV，等待driver自然结束。

## 6. REMOTE-SMOKE-005：自然完成与现场回收

只读轮询调用：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py run `
  --command-file tmp\idea2_dvac_smoke_status.sh
```

19:04:21首次快照显示driver/Ray双rank正在初始化，GPU仅585/551 MiB；19:05:03模型已加载，GPU约
11,973/11,944 MiB，driver进入`Evaluating Rollout Epochs 0/1`。两个EnvWorker都打印：

```text
ModuleNotFoundError: No module named 'curobo.types.math'; 'curobo.types' is not a package
```

没有据此停止或重启。只读检查精确源码：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py run `
  --command-file tmp\idea2_curobo_readonly_probe.sh
& $codexPython local_scripts\remote_exec_autodl.py run `
  "grep -nE '^except|traceback|CuroboPlanner' /root/autodl-tmp/RoboTwin_RLinf/envs/robot/planner.py | tail -n 20"
```

结果证明导入位于optional `try/except Exception`，异常时主动`traceback.print_exc()`后设置
`CuroboPlanner=None`；本次manifest的真实backend是`mplib`，不会实例化Curobo。历史成功运行也有同样提示。
所以它是噪声较大的可选后端提示，不是运行问题，不安装依赖、不改环境。

driver随后自然完成：

```text
LAUNCHER_STARTED_AT=2026-08-20T19:03:17+08:00
DRIVER_EXIT_CODE=0
LAUNCHER_FINISHED_AT=2026-08-20T19:07:11+08:00
STATUS=COMPLETE
wall ≈ 3 min 54 s
```

observer随driver自然退出；没有运行中信号、timeout、资源阈值或人工停止。

## 7. REMOTE-SMOKE-006：产物合同与信号sanity

### 7.1 现场与文件清单

精确调用：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py run `
  --command-file tmp\idea2_dvac_postcheck_inventory.sh
```

结果：

```text
source HEAD = remote HEAD = 61996e15cc7f5a32bd6012b61b20893d94636c82
source dirty count = 0
matching target processes = 0
GPU compute processes = 0
memory.events low/high/max/oom/oom_kill = all 0
output bytes = 10,129,300
output files = 206
trace NPZ / query CSV / episode CSV = 2 / 2 / 2
query PNG = 192
RLinf MP4 = 2
RoboTwin_RLinf native MP4 created during run = 0
```

两支trace约0.92 MB/支；output自然包含`run_manifest.json`、`resolved_config.yaml`、两套rank manifest、
TensorBoard config/event和两支RLinf视频。

### 7.2 Python合同核验

将`tmp/idea2_dvac_artifact_validate.py`用SFTP上传到本次runtime后执行：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py run `
  "/root/autodl-tmp/RLinf/.venv/bin/python /root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1/artifact_validate.py /root/autodl-tmp/RLinf_idea2_dvac_pi0_robotwin/outputs/idea2_dvac_sft_smoke_2gpu_16env_v1"
```

第一次validator把冻结合同中的语义名误写为`final_action_normalized`，而实际NPZ键是代码合同中的
`final_model_action`，因此在读取第一个NPZ时断言失败；产物没有损坏。只修改validator键名和对应allclose
检查，再次上传运行通过：

```text
query rows = 64, unique query_uid = 64
episode rows = 16, unique reset IDs = 16
each episode query_idx = 0,1,2,3
each episode action_slot_start = 0,50,100,150
rank00/rank01 = 32 queries + 8 episodes each
each video tile 0..7 = 4 queries
pre frames = 0..3; post frames = 1..4
query images = 192, all RGB 320×240
success_at_end = 14/16（87.5%，只作本次smoke事实，不作为official评估）
termination_reason = truncation × 16
DVAC_ARTIFACT_CONTRACT_PASS=1
```

两支NPZ均为：

```text
x_chain           [32,5,50,14] float32
z_endpoint        [32,4,50,14] float32
timesteps         [4] = [1.0,0.75,0.5,0.25]
final_model_action[32,50,14] float32
env_action        [32,50,14] float32
robot_state       [32,14] float32
```

全部数组finite，`x_chain[:,-1]`与`final_model_action`逐值相等。按DVAC总体方差（`ddof=0`，动作维求和）
只做非退化sanity，不据此下任务结论：64×50=3,200个`query×h`在L2/L3/L4均全部大于0且finite；中位数
分别约`0.00256 / 0.00978 / 0.0340`。raw trace保留完好，正式阈值与图仍留到后续讨论。

manifest逐项匹配source/RoboTwin/checkpoint/norm/seed hash，实际记录`H=50,D_model=32,D_active=14,
C=50,M=4,dtype=float32,flow_ode,mplib`；启动命令不含timeout/kill。

## 8. REMOTE-SMOKE-007：资源只观察结果

将`tmp/idea2_dvac_resource_summarize.py`上传runtime，对observer CSV/TSV离线汇总，写入
`runtime/resource_summary.json`。精确运行形式：

```text
/root/autodl-tmp/RLinf/.venv/bin/python resource_summarize.py \
  resource_monitor runtime/resource_summary.json
```

212 GPU行对应106个两秒time samples，elapsed 0..231秒：

```text
GPU0 peak = 19,132 MiB = 18.684 GiB; util peak 100%; temp peak 42 C
GPU1 peak = 19,102 MiB = 18.654 GiB; util peak 100%; temp peak 38 C
cgroup current peak = 59,298,377,728 bytes = 55.226 GiB
cgroup anon peak = 52.317 GiB; file peak = 1.263 GiB
memory.events first/last low/high/max/oom/oom_kill = all 0
largest RolloutWorker RSS peaks = 13.891 / 13.135 GiB
EnvWorker RSS peaks = 13.798 / 13.430 GiB
```

运行结束后GPU回到0 MiB、没有目标进程；这些数值只被记录，没有触发任何行为。

## 9. REMOTE-SMOKE-008：视频解码、问题与视觉抽查

精确调用：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py run `
  --command-file tmp\idea2_dvac_video_postcheck.sh
```

第一次后处理用当前服务器旧ffmpeg不支持的`-fps_mode passthrough`，在成功ffprobe seed0后停止；不影响
原MP4。改为该版本支持的`-vsync 0`并使用新目录。第二次因ffmpeg在shell`while read`中读取stdin，吞掉
下一条路径首字母`r`；再次不改原MP4。最终窄修复是在两个ffmpeg调用都加`-nostdin`并使用第三个唯一
probe目录；无需删除前两个失败目录，第三次通过。

两支视频精确一致的流合同：

```text
codec = h264, pix_fmt = yuv420p
resolution = 1280×480（每rank 8 env，2×4 tile）
decoded/container frames = 6/6
fps = 30
duration = 0.200 s
sizes = 65,026 / 62,261 bytes
```

共解出12帧和2张3×2时间contact sheet。视觉抽查显示：frame0是8个不同reset的初态；frame1..4依次
出现抓取/调整过程；frame5回到新的瓶子初态，符合auto-reset spill。rank00 episode0 q0保存的head PNG与
seed0 frame0 tile0视觉一致；图像清晰，没有黑帧、错tile或相机串线。视频“太快”的原因被实测锁定为
6帧按30 FPS播放，不是编码器漏帧。

通过同一Paramiko/SFTP下载到本地证据目录：两张contact sheet、两支原MP4及一张代表性query head PNG；
没有下载NPZ大张量或checkpoint。

## 10. LOCAL-SMOKE-009：小型证据同步

为使资源、来源和视频结论可在本地直接复核，又从服务器SFTP下载以下小文件；远端原件保留，未移动或
删除任何内容：

```text
SMOKE_2GPU16ENV_V1_DRIVER.log                 19,279 bytes
SMOKE_2GPU16ENV_V1_RESOURCE_SUMMARY.json       5,833 bytes
SMOKE_2GPU16ENV_V1_RESOURCES.csv              28,821 bytes
SMOKE_2GPU16ENV_V1_PROCESS_RSS.tsv           161,107 bytes
SMOKE_2GPU16ENV_V1_RUN_MANIFEST.json           2,401 bytes
SMOKE_VIDEO_SEED0_CONTACT.png              1,608,823 bytes
SMOKE_VIDEO_SEED1_CONTACT.png              1,575,950 bytes
SMOKE_VIDEO_SEED0.mp4                         65,026 bytes
SMOKE_VIDEO_SEED1.mp4                         62,261 bytes
SMOKE_QUERY_RANK00_EP000000_Q000_HEAD.png      44,402 bytes
```

全部位于`docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/`。原始10.1 MB run output、两支NPZ、192张PNG
和完整runtime仍留在服务器，不复制进本地文档仓。

## 11. REMOTE-SMOKE-010：最终只读收口

精确调用：

```powershell
& $codexPython local_scripts\remote_exec_autodl.py run `
  --command-file tmp\idea2_dvac_final_audit.sh
```

2026-08-20 19:24:51 +08:00结果：

```text
DRIVER_EXIT_CODE=0
STATUS=COMPLETE
SOURCE_CLEAN=1
MATCHING_PROCESSES=0
GPU_COMPUTE_PROCESSES=0
TRACE_SHARDS=2
QUERY_SHARDS=2
EPISODE_SHARDS=2
QUERY_IMAGES=192
MP4=2
MEMORY_EVENTS=low 0;high 0;max 0;oom 0;oom_kill 0
FINAL_AUDIT_PASS=1
```

本轮真实smoke及其后检至此结束；没有启动第二次模型/环境运行。

## 12. 补充只读指令索引

为避免正文摘要漏掉短探针，以下也属于本轮服务器指令：

```powershell
# 上传后逐字节核对
& $codexPython local_scripts\remote_exec_autodl.py run `
  "sha256sum /root/autodl-tmp/idea2_dvac_run_configs/idea2_dvac_sft_smoke_2gpu_16env_v1.yaml /root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1/observe_resources.sh /root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1/launch.sh /root/autodl-tmp/idea2_dvac_runtime/idea2_dvac_sft_smoke_2gpu_16env_v1/start.sh"

# optional Curobo源码补查
& $codexPython local_scripts\remote_exec_autodl.py run `
  "sed -n '325,430p' /root/autodl-tmp/RoboTwin_RLinf/envs/robot/planner.py | tail -n 30"
& $codexPython local_scripts\remote_exec_autodl.py run `
  "grep -nE '^except|traceback|CuroboPlanner' /root/autodl-tmp/RoboTwin_RLinf/envs/robot/planner.py | tail -n 20"

# CSV表头/代表行与全局manifest人工查看
& $codexPython local_scripts\remote_exec_autodl.py run `
  "head -n 3 <output>/dvac_telemetry/query_index_rollout_rank00.csv; head -n 3 <output>/dvac_telemetry/episode_index_env_rank00.csv; cat <output>/dvac_telemetry/run_manifest.json"

# 下载前大小核对
& $codexPython local_scripts\remote_exec_autodl.py run `
  "stat -c '%n %s' <runtime>/driver.log <runtime>/resource_summary.json <runtime>/resource_monitor/resources.csv <runtime>/resource_monitor/process_rss.tsv <output>/dvac_telemetry/run_manifest.json"
```

其中`<runtime>`与`<output>`分别是本文第4.1节和第5节锁定的完整绝对目录；实际命令使用的是展开后的
绝对路径。`tmp/idea2_dvac_smoke_status.sh`共只读调用三次（19:04、19:05、19:08），每次内容相同；
所有SFTP `put/get`源与目的逐项列在第4.2、9、10节。最终在本地PowerShell中清空
`SEETA_SSH_PASSWORD`及临时安全变量并退出会话，凭据从未进入文件、命令参数或账本。
