# v3 收尾与 global-z `[0,2]` 正式训练逐指令账

日期：2026-08-23  
机器：AutoDL / Windows 工作区（每条记录显式标注）

## 授权与边界

- 用户已明确授权：中断当前 R-only v3 `[0,2]` 训练；整理高信息量、轻量的日志/指标/图/配置 ZIP；
  启动一个 100-step 的 `global-z（v1信号）× [0,2]（v3权重区间）` 正式训练。
- 只向本次 v3 已确认的 process group 发信号，不操作其他任务。
- checkpoint 正文、大批原始 NPZ 和完整训练视频留在服务器；本地 ZIP 只收高信息量代表材料。
- 新实验保持历史成功 π0 GRPO 的任务、SFT、并发、采样、优化器、PPO clip 与保存节奏；只允许新增/修改
  DVAC 信号模式和权重区间所需配置。

## 流水账

### CL-001（Windows，只读）

- 指令：完整读取 `PROJECT_CONTEXT.md`、`HANDOFF.md` 与当前专题 `00_INDEX_AND_PLAN.md`；检索相关 memory
  索引。
- 结果：确认动态服务器事实必须现场刷新；v3 上次停点为完整 g48、Step49 rollout 15/16；当前用户授权覆盖
  v3 停止、轻量收尾与新 100-step 正式训练。
- 问题/处理：组合读取输出发生显示截断；专题主文档随后单独完整读取，不依赖截断段落执行。

### CL-002（AutoDL，只读现场刷新）

- 指令：经固定 Paramiko 路径执行 `tmp/idea2_v3_pre_stop_refresh_20260823.sh`；读取身份、精确PID/PGID、
  driver进度、checkpoint/NPZ数、GPU、cgroup、磁盘和Git状态。
- 结果：主机/UID=`autodl-container-nekaqbwt43-6ce5babb/0`；wrapper/driver/observer仍为
  `70610/70614/70615`、共同PGID=`70608`；双rank NPZ=`98`，即g1--g49均已完整写出；下一轮rollout
  正在进行。GPU约`26.5/27.0 GiB`，cgroup约`228.0 GiB`，`oom=oom_kill=0`；run/runtime约`39G/69M`；
  source HEAD=`eb2a091...`且clean。

### CL-003（AutoDL，按授权停止v3）

- 指令：先对精确PGID `70608`执行`kill -INT -- -70608`并等待30秒；随后只读验证。
- 结果：SIGINT已成功发出，但后台Python/Ray进程组继续运行，30秒后PGID内仍有302个进程。
- 问题/原因：后台启动链忽略/未处理SIGINT；不是密码、host-key或项目训练错误。
- 修复：对同一个已核对PGID执行`kill -TERM -- -70608`；9秒后整个组退出。
- 复测：PGID无成员，GPU0/1均为0 MiB，cgroup降至约67.8 GiB，`oom=oom_kill=0`；没有操作其他进程。
- 结果边界：v3最终计入完整g1--g49；被中止的下一轮rollout不计入训练结果。

### CL-004（Windows / AutoDL，只读下载与g49分析）

- 指令：用Paramiko `get`下载v3 `metrics.log`、双rank CSV/state/manifest、双rank
  `rollout_step0048.npz`、driver/launch/resolved/resource日志和TensorBoard event；运行
  `analyze_v3_g49.py`与`plot_full_grpo_vs_v3.py`。
- 结果：确认完整g1--g49、主要数值finite、fatal命中0；生成四run训练、方法诊断、资源和原GRPO完整100步
  对v3 g49共4张PNG及配套CSV/JSON。
- 视觉复测：4张PNG均已打开检查，坐标、图例和原GRPO/v3不同长度边界清楚。

### CL-005（Windows / AutoDL，新配置）

- 指令：从v1 global-z formal YAML复制新配置，只改`log_path`、`experiment_name`和
  `strength: 0.1 -> 0.5`，上传到residual child。
- 结果：source YAML SHA256=`48ae7a3965fa43844377862df4ec294b051b7b93cccde461d6c119cd62374149`；
  `z_clip=2`下公式精确映射到`[0,2]`，没有Python代码改动。

### CL-006（AutoDL，最小前测）

- 指令：执行`tmp/idea2_global_z_w0to2_pretest_20260823.sh`，包含synthetic forward/backward、Hydra
  `--cfg job --resolve`和关键字段对照。
- 结果：forward恒等；backward倍率为`[0,.5,1,1.5,2]`；compose通过；resolved SHA256=
  `8e7e807678be41d41d2d20644ea9105771a6c73f42f6986633370c0e0b1e6951`。
- 问题/处理：脚本里两个`pgrep`被Paramiko helper自身命令行命中，导致无副作用的提前停止；删除该不必要
  自匹配检查后原样复测通过。训练未在前测中启动。

### CL-007（AutoDL，提交与远端核验）

- 指令：提交新YAML为`afdaa2e2aa59aa16128e89f47eb4aaf7a64badd8`；直接HTTPS push瞬时挂起后，按既有
  AutoDL网络路线在单一subshell临时加载`/etc/network_turbo`，对`ls-remote/push/ls-remote`分别设
  15/40/15秒界限。
- 结果：普通非force push成功；远端`personal/codex/idea2-dvac-residual-downweight` HEAD复核为
  `afdaa2e2...`；父shell没有遗留proxy变量，服务器worktree clean。

### CL-008（AutoDL，启动前现场）

- 指令：读取目标目录、GPU、cgroup memory/current/events与数据盘余量。
- 结果：目标run目录不存在；GPU0/1约4 MiB；cgroup约68.1 GiB；`oom=oom_kill=0`；数据盘余约669 GiB。

### CL-009（AutoDL，正式训练唯一一次启动）

- 指令：执行`tmp/idea2_global_z_w0to2_start_formal_20260823.sh`，driver命令为：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python \
  /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf_idea2_dvac_residual_downweight/examples/embodiment/config/ \
  --config-name robotwin_adjust_bottle_grpo_openpi_dvac_global_z_w0to2_100step_formal
```

- 结果：`2026-08-23T19:27:49+08:00`启动；wrapper/driver/observer=`820640/820644/820645`，
  PGID=`820638`。run/runtime均使用独立`idea2_dvac_global_z_w0to2_formal_100step_2gpu16env_20260823`目录。

### CL-010（AutoDL，只读启动确认）

- 指令：执行`tmp/idea2_global_z_w0to2_startup_refresh_20260823.sh`，读取三根PID、六个核心worker、driver
  rollout进度、GPU、cgroup、memory events、产物与exit marker。
- 结果：`19:33:04 CST` wrapper/driver/observer及2 actor/2 rollout/2 env worker均alive；真实rollout已完成
  `1/16`。GPU0/1约`22.1/25.7 GiB`，cgroup约111.83 GiB，`oom=oom_kill=0`，没有exit marker；资源CSV
  287行并继续增长。
- 说明：可选`curobo.types.math`导入traceback与此前成功v2/v3相同，核心worker存活且rollout继续，不影响
  本次“已成功开始”的判断；此时尚无完整Global Step。
- 二次刷新：`19:40:32 CST`六个worker与三根控制进程继续存活，首轮rollout已到`5/16`；GPU约
  `25.9/25.5 GiB`，cgroup约119.7 GiB，`oom=oom_kill=0`，仍无exit marker。

### CL-011（Windows，轻量ZIP）

- 指令：用PowerShell `Compress-Archive`封装g49证据目录、25号收尾文档与本流水账；随后用
  `System.IO.Compression.ZipFile.OpenRead()`读取成员表，并计算SHA256。
- 结果：`exports/idea2_dvac_v3_formal_stop_g49_20260823.zip`，4,566,769 bytes，32个成员，展开后
  22,192,701 bytes；SHA256=`534C4541CFEF604725AFF660C38C190E985E13F639835C162620AA558389338C`。
- 边界：不含约9.7 GiB/个的checkpoint、其余96个step NPZ或完整视频正文；这些继续留在服务器。
