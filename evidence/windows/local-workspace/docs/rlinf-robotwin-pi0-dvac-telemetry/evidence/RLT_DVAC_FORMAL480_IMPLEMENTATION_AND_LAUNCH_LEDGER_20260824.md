# RLT teacher-DVAC fresh-480正式训练实施与启动流水账（2026-08-24）

## 1. 用户决策与授权

- 用户明确要求正式训练**fresh直接跑到绝对cycle 480**，而不是先停止于250再另行恢复。
- 用户授权在参数讨论和简洁检查后启动正式训练。
- 主体参数继续对齐历史成功RLT；只允许DVAC方法块、实验名、fresh-480总预算以及对应保存/评估合同不同。
- 资源监视只记录，不自动回收、杀进程或改变训练行为。

## 2. 本轮待核对

- 刷新AutoDL现场进程、GPU/RAM/cgroup、磁盘和RLT-DVAC Git状态；
- 对照历史成功fresh-250与resume-480配置，明确fresh-480单段运行的精确预算差异；
- compose并保存resolved config、精确命令、source/config hash与输出目录；
- 启动后只确认Ray/worker、首轮rollout和资源正常，不等待480完成。

## 3. 逐指令记录

### 3.1 本地上下文与历史合同

1. 完整读取根`PROJECT_CONTEXT.md`、`HANDOFF.md`与当前专题`00_INDEX_AND_PLAN.md`；随后读取RLT-DVAC
   28/32号文档和历史RLT 17号终态。
   - 结果：历史完整长度是绝对cycle 480；原运行分为fresh 1--250与resume 251--480。
   - 本次用户新决策：fresh单进程直接到480；不把它写成历史逐点复现。
2. 对照本地历史resolved配置、当前RLT-DVAC overlay和Stage 1验收记录。
   - 结果：正式只需新增fresh480 config覆盖`max_steps/name/path`，算法Python不需要再改。

### 3.2 AutoDL现场与内存只读核对

通过固定host-key、纯密码Paramiko执行
`tmp/rlt_dvac_formal480_live_and_memory_audit_20260824.sh`。

关键指令包括：

```bash
hostname; pwd; id -u
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
cat /sys/fs/cgroup/memory.current
cat /sys/fs/cgroup/memory.high
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
cat /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/memory.pressure
pgrep -af 'train_embodied_agent.py|ray::|raylet|gcs_server'
git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac status --short
git -C /root/autodl-tmp/RLinf_rlt_teacher_dvac rev-parse HEAD
```

结果：

- 容器、用户与端点正确；两张A800空闲；无训练/Ray进程；磁盘余约735 GiB；
- cgroup current约10.16 GiB，`memory.high≈236 GiB`、`memory.max=240 GiB`、swap max=0，event全0；
- 当前内存主要是约9.79 GiB file cache，anon约0.31 GiB；PSI为0；
- `memory.reclaim`文件不存在；代码/脚本无`memory.reclaim/drop_caches/malloc_trim`写入；
- server worktree当时为`513dbcb7...`且clean/upstream 0/0。

问题：第一版inventory脚本使用了一个已过时的config文件名，`sed`在不存在路径处结束。

解决：只读列举服务器实际config文件，再执行
`tmp/rlt_dvac_formal480_config_read_20260824.sh`读取base、8env250和DVAC overlay。该问题发生在只读读取阶段，
没有修改配置、进程或输出。

### 3.3 新增fresh480 opt-in配置

使用`apply_patch`本地新建：

```text
tmp/rlt_dvac_impl_source/examples/embodiment/config/
robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2_fresh480.yaml
```

内容只继承已通过smoke的DVAC overlay并覆盖：

```yaml
runner:
  max_steps: 480
  resume_dir: null
  logger:
    log_path: ${oc.env:RLT_LOG_ROOT}
    experiment_name: robotwin_adjust_bottle_rlt_teacher_dvac_w0to2_formal_fresh480_v1
```

本地SHA256：`e92c99f62a52aeb0aac280618fed350216687b3b69027132d88b253875348f9f`。

通过SFTP上传后执行`tmp/rlt_dvac_formal480_config_precheck_commit_20260824.sh`，其中包含：

```bash
python -B examples/embodiment/train_embodied_agent.py \
  --config-path .../examples/embodiment/config \
  --config-name robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2_fresh480 \
  --cfg job --resolve
```

compose检查精确核对max480、resume null、2卡、8 train env、fixed20 eval、H50/M4/D14、student C10、
batch512/micro128、warmup/replay/保存区间、DVAC L3/strength0.5/zclip2/applied C10以及Stage 1 artifact hash。
结果：`FORMAL480_RESOLVED_CONTRACT_OK`。

随后在服务器执行：

```bash
git add examples/embodiment/config/robotwin_adjust_bottle_rlt_stage2_ac_mlp_8env250_teacher_dvac_w0to2_fresh480.yaml
git commit -m 'config(rlt): add teacher DVAC fresh 480 formal run'
git push personal codex/rlt-teacher-dvac-weighting
```

结果：commit=`a85b101bfd905f6d1e0700ae6c3ef1e4fb0ecec4`，普通非force push成功，worktree与upstream同步。

### 3.4 正式输出与启动脚本准备

上传并执行`tmp/rlt_dvac_formal480_prepare_20260824.sh`，新建以下独立目录：

```text
/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1
/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime
```

准备脚本保存resolved config/source/config hash，核对输出目录未复用、Stage 1/norm/manifest准确，并落盘
`run_foreground.sh`与`resource_monitor.sh`。结果：准备exit0；Stage 1 manifest SHA为
`6ca58f26...a3712433`，norm SHA为`649ed92b...ce4f6a`。

正式启动指令：

```bash
setsid bash runtime/run_foreground.sh >runtime/foreground.log 2>&1 < /dev/null &
setsid bash runtime/resource_monitor.sh >runtime/monitor.log 2>&1 < /dev/null &
```

资源monitor每2秒只读记录GPU、cgroup current/stat/events/PSI和进程RSS；没有告警阈值、自动signal或外层
timeout。

结果：

- driver wrapper PID=`106844`，Python driver PID=`106850`；
- monitor PID=`106845`；
- `started_at=2026-08-24T18:52:18+08:00`；
- 2 actor、2 rollout、2 env worker均创建；Stage 1、norm stats与replay cache加载成功。

### 3.5 首轮真实运行复核

第一次startup grep把历史可选Curobo planner traceback和配置文本中的`inf`也列入`FATAL`，造成噪声；进程、
退出标记和真实训练没有异常。

解决：只缩窄**读取脚本**的fatal匹配为CUDA OOM、NCCL error/timeout、RayTaskError、Worker crash、
FloatingPointError和memory-pressure kill；没有更改训练进程。

2026-08-24约19:03 CST再次执行只读startup probe，结果：

- 完成Global Step 1与2，正在第3轮rollout；每步8 trajectories；
- Step1/2 `success_once=0.125`，global min replay=71/142；
- `ready_for_online=0`且actor/critic update=0，符合原RLT每rank 10,000-transition warmup；
- GPU显存16,359/16,442 MiB；cgroup current约44.2 GiB；memory event全0；
- 精确fatal匹配为空；`exit_code/finished_at`仍不存在。

结论：正式fresh480配置已经进入连续真实rollout并正常累计replay，启动闭环通过。

## 4. 内存教学文档简化（用户反馈后）

用户反馈原34号把配置、源码、Linux、Ray和迁移步骤全部放在一起，首次阅读过长。本轮按“先结论、后证据”
重新分层：

1. 重新读取AutoDL/深圳对照证据、旧/current EnvWorker offload源码和两边resolved config；
2. 确认最关键差异是AutoDL `env.train.enable_offload=true`，深圳train/eval均为`false`；深圳每worker
   常驻`32 train + 16 eval`，AutoDL为8 train且eval未实例化；
3. 确认`clear_cache_freq=1`两边都有，不能将它写成锯齿差异来源；
4. 原34号路径保留，改成3分钟说明，避免已有链接失效；
5. 新建35号技术附录，承接调用链、真实数字、cgroup/Ray原理和本地证据；
6. 更新专题索引、28号计划和根HANDOFF的阅读顺序。
7. 同步修正专题索引中一条残留的“formal尚未启动”旧状态，改为fresh480已启动；没有刷新动态step。

复核后又补充三点：旧AutoDL两秒下降发生在236 GiB high水位以下且EnvWorker RSS同步下降，主体是应用
offload；顶层`env.enable_offload`不能替代train/eval nested键；深圳current RLT已启用train offload，说明
该机制能在深圳机器工作，差异集中在此前GRPO的配置与每worker `32+16`个常驻环境。

本轮只修改Windows本地文档，没有连接服务器、改变formal进程或修改训练配置。

### 4.1 用户进一步冻结迁移边界

用户指出，降低每worker常驻env数会改变既有并发与训练合同，不应作为本轮新增内存组件的建议；同时确认
train/eval两个nested offload值得保留，并将run级cgroup `MemoryHigh/MemoryMax`判断为AutoDL高水位行为的
主要原因。文档据此再次修正：

1. 删除“通过减少env解决”的路线，冻结env数、并发、rollout、batch和评估预算不变；
2. 将run级cgroup写为深圳侧主要新增组件，用于在Ray整机95%杀worker之前建立本run的高水位与最后边界；
3. 将train/eval nested offload写为已有配套机制，用于在轮末断开simulator环境引用、给内核和allocator
   提供可释放对象；
4. 区分两种观测：周期边界的局部下降主要由offload解释，高位平台/边界压力主要由cgroup解释；
5. 没有连接服务器、改变训练配置或干预当前formal进程。

## 5. 21:15--21:22 CST只读现场刷新与简图

### 5.1 服务器探针

使用进程内密码、固定host-key和既有Paramiko helper执行：

```text
python local_scripts/remote_exec_autodl.py run --command-file tmp/rlt_dvac_live_probe_20260824.sh
```

首轮本地入口`py -3.12`返回`No installed Python found!`，服务器尚未连接。改用Codex工作区内置Python后，
同一只读探针成功。身份仍为既有AutoDL容器root；wrapper/driver/monitor及2 actor、2 rollout、2 EnvWorker
均存活。

### 5.2 下载与离线分析

通过同一helper的SFTP `get`下载当时的`metrics.log`和`resources.csv`到
`evidence/rlt_dvac_formal_live_g65_20260824/raw/`。第一次连续下载在30秒本地等待边界处只得到64 KiB的
resources临时文件；单独重取同一路径后得到完整778,391-byte CSV。

本地分析脚本首次因内置Python没有`matplotlib`而退出，未生成错误图。确认Pandas/NumPy/Pillow可用后，
只将绘图后端窄改为Pillow；解析得到完整g1--66、3,821条有效资源样本，并输出step CSV、summary JSON与
四面板PNG。图像已人工查看，标题、坐标、图例和四条结论均可读。

### 5.3 最终存活复核

执行：

```text
python local_scripts/remote_exec_autodl.py run --command-file tmp/rlt_dvac_live_final_probe_20260824.sh
python local_scripts/remote_exec_autodl.py run --command-file tmp/rlt_dvac_latest_metric_probe_20260824.sh
```

21:22:39 CST最新完整g69/480并进入下一轮；每rank最小replay=`5,097/10,000`、actor/critic update=0、
baseline count约51,600且未冻结。g25/g50 checkpoint存在；六类精确fatal计数和cgroup
high/max/oom/oom_kill均为0。最终GPU使用16.06/16.34 GiB，cgroup current约57.17 GiB，数据盘余735 GiB。

本轮没有发送signal、修改服务器文件、改变配置或干预训练。

## 6. 2026-08-25 09:38--09:56 CST只读刷新与g342简图

### 6.1 现场探针

继续使用进程内密码、固定host-key、内置Python和既有Paramiko helper，执行：

```text
python local_scripts/remote_exec_autodl.py run 'bash -s' --stdin-file tmp/rlt_dvac_live_probe_20260825.sh
```

首轮探针时最新完整cycle为`g334/480`，wrapper/driver/monitor、2 actor、2 rollout和2 EnvWorker均存活；
DVAC baseline已冻结，`mode_apply=1`，说明训练已跨过replay warmup并真实进入student actor更新。六类精确
fatal匹配以及cgroup `high/max/oom/oom_kill`均为0。

### 6.2 原始数据下载、分析与图面复核

通过同一helper的SFTP `get`下载：

```text
/root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/metrics.log
/root/autodl-tmp/experiment_exports/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/runtime/resources.csv
```

本地以Pandas解析完整metric table，以Pillow绘制四面板；第一次图面检查发现底部摘要与横轴刻度重叠，
只调整绘图边距后重新生成并再次人工查看。最终再次下载两份动态文件，冻结到完整`g342`、23,757条资源
样本，输出：

```text
evidence/rlt_dvac_formal_live_g334_20260825/RLT_DVAC_LIVE_G342_SUMMARY.png
evidence/rlt_dvac_formal_live_g334_20260825/step_metrics.csv
evidence/rlt_dvac_formal_live_g334_20260825/summary.json
```

只纳入`g1--g342`完整table；未完成cycle不计。`g342`训练success=`7/8`，5步/10步均值=`92.5%/90.0%`；
最近fixed20为`g325=16/20`，此前`g300=18/20`。DVAC从`g136`起apply；最新
`p05/median/p95=0.319/0.952/1.861`、ESS=`0.896`、top20 weight mass=`0.289`，累计
`update_step=161,000`。

### 6.3 最终存活与资源复核

执行：

```text
python local_scripts/remote_exec_autodl.py run 'bash -s' --stdin-file tmp/rlt_dvac_live_final_probe_20260825.sh
python local_scripts/remote_exec_autodl.py run 'tail -n 170 .../metrics.log'
```

09:55 CST现场仍为完整`g342/480`并持续运行；两个EnvWorker进程存活。`g342` actor/critic grad norm为
`2.697/0.146`，loss为`-0.096/0.0011`。cgroup current约102.1 GiB、所有memory event为0；两卡瞬时
显存约19.58/19.66 GiB，数据盘剩余725 GiB。资源曲线呈阶梯上升，但峰值102.54 GiB仍远低于236 GiB
high水位。本轮未发送signal、未写服务器文件、未改变配置或进程。

### 6.4 本地文档收口中的两个窄问题

- 第一次合并文档patch仍假定`HANDOFF.md`首行日期为8月24日，但同工作区已被另一条有效记录更新为8月25日，
  因上下文不匹配整包未写入。重新读取精确行、拆分patch后成功，未覆盖该并行更新。
- 首次`git status`遇到Windows Codex sandbox用户与仓库owner不同导致的`dubious ownership`；只在该只读
  命令上使用`git -c safe.directory=C:/Users/86136/Documents/rl status ...`复核，没有修改全局Git配置。

## 7. 2026-08-25 10:20--10:37 CST：原RLT对照、fixed20与单卡审计

### 7.1 只读服务器刷新

使用进程内密码、固定host-key、既有低层Paramiko helper执行：

```text
python local_scripts/remote_exec_autodl.py run 'bash -s' \
  --stdin-file tmp/rlt_dvac_live_final_probe_20260825.sh
python local_scripts/remote_exec_autodl.py get \
  /root/autodl-tmp/experiments/rlt_teacher_dvac_w0to2_formal_fresh480_20260824_v1/metrics.log \
  docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/rlt_dvac_vs_original_live_g350_20260825/raw/dvac_metrics.log
```

10:36:43 CST最新完整`g357/480`，两个EnvWorker仍存活。cgroup current=`110,710,968,320`
bytes（103.09 GiB），`high/max/oom/oom_kill=0`；两卡瞬时显存=`19,578/19,660 MiB`，数据盘剩余
724 GiB。没有发送signal、修改服务器文件或改变训练参数。

### 7.2 原RLT与当前DVAC曲线

原RLT输入严格使用历史fresh1--250与resume251--480两份完整`driver.log`，当前输入使用本轮下载的完整
metric table。执行：

```text
python tmp/analyze_rlt_dvac_vs_original_20260825.py
```

首次版本尝试导入`matplotlib`，内置Python返回`ModuleNotFoundError: No module named 'matplotlib'`，没有生成
错误图，也未安装新依赖。随后只把渲染后端改为SVG，继续用Pandas解析；再用工作区已有Sharp做机械格式转换：

```text
node -e "const sharp=require('sharp'); sharp('...G357.svg').png().toFile('...G357.png')"
```

SVG与PNG均为1800×1640；Sharp只报告fontconfig cache不可写，但PNG正常生成。人工查看确认三面板标题、
坐标、图例、g136/g250边界和fixed20计数可读。数据校验同时确认：原RLT完整`g1--480`无缺口；当前
完整`g1--357`无缺口；两侧train每点均8 episodes；所有eval点均20 episodes。

输出目录：

```text
docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/rlt_dvac_vs_original_live_g350_20260825/
```

其中包括PNG/SVG、逐cycle CSV、summary JSON、当前resolved config和README教学说明。

### 7.3 逐叶配置与单卡只读审计

逐行比较当前live `runtime/resolved.yaml`、历史fresh250 resolved与resume480 resolved，并按源码核对
placement、world-size派生、gradient accumulation和checkpoint world-size合同。结果：

1. task、teacher/student、seed内容、并发、batch、replay、更新、Q/BC、eval/save、video/offload全部一致；
2. 预期差异仅为DVAC块、路径/命名和当前单进程fresh1--480；unexpected=`0`；
3. 当前`no_shard`两rank各持完整模型，A800-80GB单卡容量可行，但需改为单rank placement；
4. 单卡保持global/micro512/128时gradient accumulation由2变4；若保持两卡全局replay合同，per-rank
   readiness和cache/window应相应补偿；2-rank checkpoint不能原样resume到1-rank；
5. 单卡主存估计90--115 GiB，墙钟估计为当前两卡的1.5--2.2倍，正式结论仍需单卡smoke。

本节所有服务器操作均只读，未干预正在运行的formal。
