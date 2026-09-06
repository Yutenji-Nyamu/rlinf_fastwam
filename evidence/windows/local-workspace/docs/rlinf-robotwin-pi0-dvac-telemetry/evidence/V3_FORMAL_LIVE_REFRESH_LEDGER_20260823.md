# v3 formal 现场刷新逐指令账（2026-08-23）

范围：用户要求只读查看当前训练、主要产物、训练指标与资源指标，并更新简洁可视化。本文从本轮第一次
服务器操作开始记录；不控制训练、不改配置、不安装依赖、不删除或覆盖远端产物。

## LR-001 — 本地上下文恢复

- 命令：完整读取根目录`PROJECT_CONTEXT.md`、`HANDOFF.md`，以及专题单一事实源
  `docs/rlinf-robotwin-pi0-dvac-telemetry/00_INDEX_AND_PLAN.md`；搜索已有AutoDL Paramiko helper。
- 结果：当前目标锁定为AutoDL Idea2 v3 `[0,2]` formal；旧文档停点为完整g28、Step29 rollout 12/16，
  但所有动态状态必须以本轮服务器现场刷新替代。复用
  `local_scripts/remote_exec_autodl.py`的固定host-key、低层Paramiko密码路线。
- 问题：首次本地读取因Windows隔离进程创建错误`CreateProcessWithLogonW failed: 1056`未执行。
- 解决：不改变环境，直接重试同一只读读取；随后成功。长文件按行号分块读取至EOF，避免工具输出截断。

## LR-002 — 身份探针

- 远程命令：`hostname; pwd; id -u; date '+%Y-%m-%d %H:%M:%S %Z'`。
- 目的：在读取训练现场前确认仍连接到获批AutoDL容器和root账号。
- 结果：成功；`autodl-container-nekaqbwt43-6ce5babb`、`/root`、UID 0，服务器时间
  `2026-08-23 13:30:15 CST`。固定host key和密码认证均通过。

## LR-003 — 训练、资源与产物主快照

- 远程command-file：`tmp/idea2_v3_live_readonly_20260823.sh`。
- 指令内容：读取wrapper/driver/observer与匹配worker、`nvidia-smi`、cgroup memory、主机RAM、
  `metrics.log`最新表、driver进度/fatal关键词、双actor的DVAC CSV尾部、checkpoint/产物数量、资源CSV尾部、
  数据盘空间和inode。
- 结果：成功。13:31:51 CST时最新完整Global Step为36/100，Step37 rollout刚开始；三根控制进程、
  2 actor、2 rollout、2 env worker均有现场存活证据。g36 success=`0.95703125`、KL=`0.060`、
  query clip=`0.175`、pre-global-clip grad norm=`33.938`；核心数值finite。
- 资源：GPU0/1即时`26,287/26,390 MiB`；cgroup current=`250,490,183,680` bytes、max=`257,698,037,760`
  bytes，`oom=oom_kill=0`；匿名内存约164.83 GB、file约83.44 GB。host available约827 GiB；
  `/root/autodl-tmp`余679 GiB、inode使用2%。
- 产物：run约30 GiB、runtime约50 MiB；72个DVAC NPZ、2个runner CSV、1个control-trace MP4/
  frames CSV、1个TensorBoard event。checkpoint目录名需要第二条窄命令确认，因为首次按错误的
  `$RUN/checkpoints`层级搜索没有输出。
- 日志扫描：CUDA OOM、OutOfMemory、oom_kill、ActorDiedError、RayActorError、NCCL均为0；
  `Traceback`字样有4处，需读上下文判断是否仍是已知可选CuRobo导入提示。

## LR-004 — checkpoint层级与Traceback上下文窄查

- 远程command-file：`tmp/idea2_v3_artifact_refine_20260823.sh`。
- 指令内容：在run内搜索`global_step_*`、统计各checkpoint大小，列出双rank最新NPZ，打印4处Traceback
  上下文、wrapper/observer尾部以及关键文件mtime/size。
- 结果：成功。checkpoint实际位于run内同名子目录的`checkpoints/`，g10/g20/g30均存在且各约9.7 GiB；
  首次主快照按错误层级`$RUN/checkpoints`搜索为空的问题闭合。最新NPZ为双rank
  `rollout_step0035.npz`。4处Traceback全部是启动时可选CuRobo/pytorch3d导入提示，与此前成功run相同；
  当前EnvWorker持续运行，未形成fatal。

## LR-005 — 下载轻量现场材料

- 本地脚本：`tmp/idea2_v3_download_live_g36_20260823.py`；单次Paramiko连接、SFTP只读下载。
- 下载内容：metrics、双rank DVAC CSV/rolling state/manifest/latest NPZ、driver/resolved/launch、
  resource CSV和TensorBoard event；不下载checkpoint正文。
- 第一次结果：12个静态文件逐字节匹配；下载仍在追加的`resources.csv`时，SFTP开始前stat为
  7,097,435 bytes，完成后的本地副本为7,099,189 bytes，原脚本将正常增长误判为size mismatch并退出。
- 原因：observer每2秒持续追加资源行；不是传输损坏或训练异常。
- 修正：静态文件继续要求exact size；仅对明确的append-only资源CSV和TensorBoard event接受下载副本
  `local_size >= initial_remote_stat`，仍拒绝截短副本。
- 复测：成功；14项共8,888,481 bytes，目标目录
  `evidence/v3_formal_live_g36_20260823/raw`。

## LR-006 — 离线汇总与可视化

- 脚本：由已验证g28分析器机械复制为
  `evidence/v3_formal_live_g36_20260823/analyze_v3_g36.py`，只更新snapshot合同g36、最新zero-based
  `rollout_step0035`及输出名；历史三run和v2同elapsed资源输入保持不变。
- 命令：Codex bundled Python执行上述分析器。
- 结果：exit0；解析最新完整g36，主数值全部finite。生成四run训练、v3方法和资源三张PNG，以及训练、
  DVAC、horizon、资源CSV与`SUMMARY_G36.json`。
- 视觉后检：三张PNG均成功打开；标题、图例、坐标、曲线和g36终点可读。
- 主要结果：v3 g1–36 success相对原GRPO累计/最近5/最近10步为
  `+0.543/+0.703/+1.406pp`；g36 ESS=`0.897`、系数角=`18.23°`、positive/negative mean weight=
  `1.163/1.295`。GPU峰值`30.368/30.218 GiB`；cgroup峰值240 GiB，OOM/OOM-kill均0。

## LR-007 — 分析完成后的最终存活探针

- 远程command-file：`tmp/idea2_v3_final_ping_20260823.sh`。
- 结果：2026-08-23 13:36:53 CST，wrapper/driver/observer仍alive，最新完整g36，Step37 rollout=`4/16`；
  GPU0/1即时`24,779/24,919 MiB`；cgroup current=`251,741,786,112` bytes（234.453 GiB），
  `max=119918、oom=0、oom_kill=0`。本轮没有干预训练。

## LR-008 — 15:56同口径现场主刷新

- 远程command-file：再次执行`tmp/idea2_v3_live_readonly_20260823.sh`。
- 结果：2026-08-23 15:56:48 CST时最新完整Global Step=`41/100`，Step42 rollout=`15/16`；wrapper、
  driver、observer、2 actor、2 rollout、2 env worker均alive。g41 success=`92.188%`、KL=`0.030`、
  query clip=`15.204%`、pre-global-clip grad=`34.566`，主数值finite。
- 资源与产物：GPU即时`27,263/29,109 MiB`；cgroup current=`257,682,685,952` bytes（239.986 GiB）、
  limit=240 GiB、`oom=oom_kill=0`；82个NPZ，run约39 GiB、runtime约61 MiB，数据盘余约669 GiB。
  fatal关键词扫描仍为0；4个Traceback字样需要沿用先前上下文窄查确认。

## LR-009 — 本地Python入口问题与修正

- 第一次命令：直接用`python local_scripts/remote_exec_autodl.py ...`执行窄查。
- 结果：未连服务器；PowerShell命中失效的WindowsApps `python.exe`，报系统无法访问。训练不受影响。
- 第二次命令：切换到Codex bundled Python，但首次漏写helper要求的`run`子命令，argparse立即退出；仍未连服务器。
- 解决：使用bundled Python并改为`remote_exec_autodl.py run --command-file ...`；后续连接与命令成功。
  密码只进入该PowerShell子进程环境，命令结束即删除环境变量。

## LR-010 — g40 checkpoint、g42落盘与Traceback复核

- 远程command-file：`tmp/idea2_v3_artifact_refine_20260823.sh`。
- 结果：g10/g20/g30/g40 checkpoint均存在，各约9.7 GiB；双rank最新NPZ已经是
  `rollout_step0041.npz`，说明Global Step42完整数据已落盘。4处Traceback上下文仍全部是启动时可选
  CuRobo/pytorch3d导入提示，没有新增训练fatal。
- 随后command-file：`tmp/idea2_v3_final_ping_20260823.sh`。
- 结果：15:59:24 CST最新完整g42，下一轮rollout已从0/16开始；三根控制进程alive；GPU即时
  `22,211/22,058 MiB`，cgroup current=`250,306,531,328` bytes（233.116 GiB），
  `max=159401、oom=0、oom_kill=0`。

## LR-011 — g42轻量材料下载

- 本地脚本：`tmp/idea2_v3_download_live_g42_20260823.py`；复用固定host-key Paramiko单连接SFTP。
- 下载内容：metrics、双rank CSV/rolling state/manifest/g42 NPZ、driver/resolved/launch、资源CSV和
  TensorBoard event；不下载checkpoint正文。
- 活文件处理：metrics、双rank runner CSV、driver、资源CSV与TensorBoard event允许在初始stat后继续
  追加，但拒绝本地副本短于初始远端大小；NPZ、manifest、rolling state、resolved和launch仍逐字节匹配。
- 结果：成功；14项共`10,158,400` bytes，保存到
  `evidence/v3_formal_live_g42_20260823/raw`。

## LR-012 — g42离线汇总与视觉检查

- 脚本：机械复制g36分析器为
  `evidence/v3_formal_live_g42_20260823/analyze_v3_g42.py`，只更新g42快照合同、
  `rollout_step0041`和输出名；历史三run及v2资源对照输入未改。
- 命令：Codex bundled Python执行上述分析器。
- 结果：exit0；解析最新完整g42，主数值全部finite。生成四run训练、v3方法、资源三张PNG，以及训练、
  DVAC、horizon、资源CSV与`SUMMARY_G42.json`。
- 视觉后检：三张PNG均成功打开；标题、图例、坐标、曲线和g42终点可读。
- 主要结果：v3 g1–42 success相对原GRPO累计/最近5/最近10步为
  `+0.372/+0.313/+0.352pp`；g42 ESS=`0.876`、系数角=`20.00°`，positive/negative mean weight=
  `1.084/1.222`。GPU峰值`30.368/30.218 GiB`；cgroup峰值240 GiB，OOM/OOM-kill均0。

## LR-013 — 文档完成后的最终存活探针

- 远程command-file：再次执行`tmp/idea2_v3_final_ping_20260823.sh`。
- 结果：2026-08-23 16:05:05 CST，wrapper/driver/observer仍alive，最新完整g42，Step43 rollout=`4/16`；
  GPU0/1即时`24,725/24,959 MiB`；cgroup current=`251,835,691,008` bytes（234.540 GiB），
  `max=159401、oom=0、oom_kill=0`。本轮没有干预训练。

## LR-014 — 18:45同口径现场主刷新

- 远程command-file：再次执行`tmp/idea2_v3_live_readonly_20260823.sh`。
- 结果：2026-08-23 18:45:09 CST时最新完整Global Step=`48/100`，Step49 rollout约`12/16`；wrapper、
  driver、observer、2 actor、2 rollout、2 env worker均alive。g48 success=`84.375%`、KL=`0.0510`、
  query clip=`16.453%`、pre-global-clip grad=`41.155`，主数值finite。
- 资源与产物：GPU即时`29,497/27,825 MiB`；cgroup current=`249,252,188,160` bytes、limit=240 GiB，
  `max=165367、oom=0、oom_kill=0`；96个NPZ，run约39 GiB、runtime约67 MiB，数据盘余约669 GiB。
  fatal关键词扫描仍为0，Traceback字样仍为4处。

## LR-015 — g48 NPZ、checkpoint与Traceback复核

- 远程command-file：再次执行`tmp/idea2_v3_artifact_refine_20260823.sh`。
- 结果：g10/g20/g30/g40 checkpoint均存在且各约9.7 GiB；双rank最新NPZ均为
  `rollout_step0047.npz`。4处Traceback上下文仍全部是启动时可选CuRobo/pytorch3d导入提示，没有
  新增训练fatal。

## LR-016 — g48轻量下载、离线汇总与视觉检查

- 本地下载脚本：`tmp/idea2_v3_download_live_g48_20260823.py`；沿用固定host-key Paramiko单连接SFTP和
  活文件不截短规则，不下载checkpoint正文。
- 结果：成功下载14项共`11,585,849` bytes到
  `evidence/v3_formal_live_g48_20260823/raw`；静态文件逐字节匹配，活文件均不短于初始远端stat。
- 分析脚本：`evidence/v3_formal_live_g48_20260823/analyze_v3_g48.py`；只把g42分析器的快照合同、
  latest NPZ和输出名机械更新到g48，历史三run及v2资源对照输入不变。
- 结果：exit0；主数值全部finite。生成四run训练、v3方法和资源三张PNG，以及对应CSV和
  `SUMMARY_G48.json`；三张PNG均已打开后检，标题、坐标、曲线和g48终点可读。
- 主要结果：v3 g1–48 success相对原GRPO累计/最近5/最近10步为
  `+0.326/-0.703/+0.352pp`；g48 ESS=`0.906`、系数角=`17.37°`，positive/negative mean weight=
  `1.145/1.222`。GPU峰值`30.368/30.218 GiB`；cgroup峰值240 GiB，OOM/OOM-kill均0。

## LR-017 — 文档完成前的最终存活探针

- 远程command-file：再次执行`tmp/idea2_v3_final_ping_20260823.sh`。
- 结果：2026-08-23 18:50:06 CST，wrapper/driver/observer仍alive，最新完整g48，Step49 rollout=`15/16`；
  GPU0/1即时`30,521/29,152 MiB`；cgroup current=`248,252,477,440` bytes（231.203 GiB），
  `max=165367、oom=0、oom_kill=0`。本轮没有干预训练。
