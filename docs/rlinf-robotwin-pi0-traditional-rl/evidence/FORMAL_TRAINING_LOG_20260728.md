# DSRL × π0 × RoboTwin 正式训练流水账（2026-07-28）

> 本文件是本次 formal run 的操作级事实源，持续记录授权、命令、配置、产物、进度、资源、问题与处理。设计与实现仍以
> [`00_INDEX_AND_IMPLEMENTATION_PLAN.md`](../00_INDEX_AND_IMPLEMENTATION_PLAN.md) 为唯一主计划；实现过程与 smoke 细节分别见
> [`IMPLEMENTATION_LOG.md`](./IMPLEMENTATION_LOG.md) 和
> [`SMOKE_EXECUTION_LOG_20260728.md`](./SMOKE_EXECUTION_LOG_20260728.md)。

## 0. 当前状态

- 状态：训练继续运行；截至 2026-07-28 21:03:56 CST，最新完整记录为 global step 30。
- 进度：global replay resident 1,062；约 21,240 requested primitive interactions；
  learned SAC 累计 11,800 optimizer updates。
- 进程：driver、2 actor、2 rollout、2 env workers 仍存活；`oom=0`、`oom_kill=0`，
  已报告 loss/Q/alpha/gradient 均有限。
- 最新静态状态报告：
  [`FORMAL_STATUS_REPORT_STEP30_20260728.md`](./FORMAL_STATUS_REPORT_STEP30_20260728.md)。
- 启动时间：2026-07-28 18:54:12 CST。
- 服务器：`root@autodl-container-nekaqbwt43-6ce5babb`。
- 分支：`codex/dsrl-pi0-robotwin`。
- HEAD/upstream：`d664bf349b63b75f41d51c8295cb0a330780d783`。
- driver PID：`70062`。
- 普通资源采样 PID：`70064`；cgroup 分项采样 PID：`70065`；采样间隔均为 2 秒。
- run root：
  `/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1`。
- 完整 resolved config：
  [`FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml`](./FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml)。
- 服务器原件 SHA-256：
  `e99c212d1743e285dcda23cb129e2ed96545cceb36bebe772ae69a693b9df595`。

## 1. 正式预算与依据

| 配置 | 正式值 | 依据 |
|---|---:|---|
| `runner.max_steps` | 650 | 这里是 collection cycles，不是 optimizer steps。按 smoke 平均 38.5 条 macro transitions/cycle、每条请求执行 `N=20`，约为 500,500 requested primitive interactions。 |
| `runner.val_check_interval` | 13 | `38.5 × UTD20 × 13 ≈ 10,010` optimizer updates，基本对齐官方每约 10k updates 做一次训练中评估。 |
| `runner.save_interval` | 65 | 约每 50k updates 保存一次；`65 % 13 == 0`，满足 runner 硬约束；650 cycles 正好 10 个 DCP。 |
| eval 数量 | 4 env × 3 epochs = 12 episodes | 小于一次正式大评估，但足以作为每约 10k updates 的方向监控；评估数据不进入 replay。 |
| train 并发 | 2×A800、4 env | fresh/resume 已实测通过；GPU 仍有余量，但 SAC 是主要墙钟瓶颈，且 cgroup 有文件缓存回收压力，所以不盲目扩 env。 |
| `H / N` | `50 / 20` | `H=50` 保持 π0 RoboTwin 基座去噪 horizon；每次只提交前 `N=20` 个动作，以更细反馈形成一条 macro transition。 |
| `utd_ratio` | 20 | 每新增一条 transition 做 20 次 SAC optimizer updates，保持 LIBERO DSRL 的高层训练语义。 |
| batch | global 256 / micro 64 | smoke 已验证显存、梯度与吞吐；正式运行不临时改并行或数值定义。 |
| replay | capacity 25,000 / warm-up 500 | capacity 对应约 `500k / UTD20` 的 transition 历史量级；warm-up 是开始 learned actor/Q 更新的独立阈值。 |
| discount | `gamma=0.999` | 一条 `N=20` macro transition 的非终止 discount 为 `0.999^20`；success 不 bootstrap，纯 time-limit truncation 继续 bootstrap。 |
| target / Q | `tau=0.005` / 10 Q heads | 保持现有 DSRL SAC 的 target EMA 与 Q ensemble 语义。 |
| eval seeds | `use_fixed_reset_state_ids=false` | formal 每次顺序消费新的 `eval_seeds.json` 条目；不是反复评同一组 smoke reset。策略仍保持现有 stochastic eval 语义。 |
| 输出路径 | run-scoped 绝对路径 | logger、train/eval video 和 RoboTwin train/eval `save_path` 全部隔离到本 run，消除 cwd 漂移和 train/eval 共用相对目录。 |

预算估计不是停止计数器：实际每轮 transition 数会随成功提前终止而变化。按 smoke 吞吐，预计约
0.49–0.51M optimizer updates、80–85 小时、160–170 GPU-hours；10 个 DCP 按约 32 GB/个估计约
320 GB。

## 2. 精确启动命令

服务器同时保存了 shell-escaped 原件 `formal_command.txt` 和逐行参数
`formal_overrides.txt`。启动命令为：

```bash
/root/autodl-tmp/RLinf/.venv/bin/python -B \
  /root/autodl-tmp/RLinf_fastwam_rlinf/examples/embodiment/train_embodied_agent.py \
  --config-path /root/autodl-tmp/RLinf_fastwam_rlinf/examples/embodiment/config \
  --config-name robotwin_adjust_bottle_dsrl_openpi \
  runner.logger.log_path=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1 \
  runner.logger.experiment_name=robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_formal_v1 \
  runner.max_steps=650 \
  runner.val_check_interval=13 \
  runner.save_interval=65 \
  runner.resume_dir=null \
  runner.ckpt_path=null \
  env.train.video_cfg.save_video=false \
  env.train.video_cfg.video_base_dir=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1/video/train \
  env.train.task_config.save_path=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1/robotwin_data/train \
  env.eval.video_cfg.save_video=false \
  env.eval.video_cfg.video_base_dir=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1/video/eval \
  env.eval.task_config.save_path=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1/robotwin_data/eval
```

运行环境保持 smoke 已验证组合：

```text
python=/root/autodl-tmp/RLinf/.venv/bin/python
Python=3.11.14
REPO_PATH=/root/autodl-tmp/RLinf_fastwam_rlinf
ROBOTWIN_PATH=/root/autodl-tmp/RoboTwin_RLinf
ROBOT_PLATFORM=ALOHA
CUDA_VISIBLE_DEVICES=0,1
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

## 3. 产物与监控

run root 内已创建：

- `FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml`：实际启动的完整解析配置；
- `formal_command.txt`、`formal_overrides.txt`：精确命令与覆盖项；
- `run_provenance.txt`：时间、身份、branch/HEAD/upstream、配置哈希和预算；
- `prelaunch_resource_snapshot.txt`：启动前 GPU、cgroup、PSI、磁盘与 `/dev/shm`；
- `formal_driver.log`、`formal.pid`、`formal_started_at.txt`；
- `tensorboard/config.yaml` 与 TensorBoard event；
- `resource_monitor/resources.csv`、`peak.txt`：两卡显存/利用率/功耗、cgroup 总量、各 worker RSS、OOM；
- `resource_monitor/cgroup_detail.csv`：`memory.current/max`、anon/file、active/inactive file、
  `memory.events max/oom`、memory PSI 和磁盘余量。

后续图表分开绘制：

1. GPU0/GPU1 显存；
2. GPU0/GPU1 利用率；
3. cgroup total/anon/file 与 240 GB limit；
4. `memory.events max` 增量、OOM、env/actor/rollout RSS。

其中 `inactive_file` 是 `file` 的子集，不能重复堆叠；cgroup 总量与进程 RSS 也不能相加。
当前覆盖到 step 20 / 20:09 的手机可读快照为：

- [`FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP20_20260728.png`](./FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP20_20260728.png)；
- [`FORMAL_OPTIMIZATION_TRENDS_STEP20_20260728.png`](./FORMAL_OPTIMIZATION_TRENDS_STEP20_20260728.png)；
- [`FORMAL_RESOURCE_CURVES_STEP20_20260728.png`](./FORMAL_RESOURCE_CURVES_STEP20_20260728.png)；
- [`FORMAL_STEP_TIMING_STEP20_20260728.csv`](./FORMAL_STEP_TIMING_STEP20_20260728.csv)。

截至 step 30 / 21:04 的横版大图为：

- [`FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP30_20260728_WIDE.png`](./FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP30_20260728_WIDE.png)；
- [`FORMAL_OPTIMIZATION_TRENDS_STEP30_20260728_WIDE.png`](./FORMAL_OPTIMIZATION_TRENDS_STEP30_20260728_WIDE.png)；
- [`FORMAL_RESOURCE_CURVES_STEP30_20260728_WIDE.png`](./FORMAL_RESOURCE_CURVES_STEP30_20260728_WIDE.png)。

启动期早期图只保留作初始化追溯，不用它替代最新训练全史。

## 4. 内存判定

启动前：

- `memory.current=230,775,824,384 B`，表面约 214.9 GiB；
- `anon=616,542,208 B`，仅约 0.57 GiB；
- `file=228,650,094,592 B`，其中 `inactive_file=194,582,913,024 B`；
- `oom=0`、`oom_kill=0`，memory PSI 的 10/60/300 秒均为 0；
- `/root/autodl-tmp` 剩余 789 GB，`/dev/shm` 剩余 120 GB。

结论是“可运行的黄灯”，不是 240 GB 被训练进程真实吃满，也没有泄漏/OOM 证据。高
`memory.current` 主要是模型/checkpoint 读取形成的可回收 file cache；smoke 的实际 anon 工作集峰值约
34–41 GB。但 smoke 中 `memory.events max` 大幅增长，说明确有频繁回收/限流，因此保持 4 env，不扩并发。

真正触发干预的信号是：anon/RSS 跨 cycle 单调增长且不回落、file cache 已回收仍贴顶、OOM 计数增加、
memory PSI 持续升高并伴随 cycle/DCP 明显变慢，或磁盘跌破安全余量。仅 raw
`memory.current` 触顶不单独判为失败。

## 5. 操作流水

### FORMAL-001　恢复上下文与授权

- 时间：2026-07-28 本轮开始。
- 操作：完整读取 `PROJECT_CONTEXT.md`、`HANDOFF.md` 和当前权威交接；确认用户已明确授权直接启动完整正式训练。
- 决定：不再增加算法测试；只做启动必需的配置冻结、live preflight、运行与监控。

### FORMAL-002　live preflight

- 时间：2026-07-28 18:47:32 CST。
- 第一次只读脚本把 normalization 文件误写为模型根下的 `assets/...`，因路径不存在立即退出；没有远程写入或进程变化。
- 修正：按真实模型布局改为
  `physical-intelligence/robotwin/norm_stats.json`，只修 preflight 路径，不改项目。
- 复核结果：
  - branch/HEAD/upstream 正确且 worktree clean；
  - 目标 run root 不存在；
  - 无训练/Ray 进程，两张 A800 均 `0 MiB / 0%`；
  - 模型、RoboTwin、正式源配置、normalization 均存在；
  - cgroup 无 OOM，磁盘与 `/dev/shm` 满足预算。

### FORMAL-003　冻结预算与完整配置

- 时间：2026-07-28 18:48–18:54 CST。
- 决定：采用 `650/13/65`，而不是视觉上更圆的 `650/10/50`。前者分别对齐约
  10k updates/eval、50k updates/DCP，并正好产生 50 次 eval、10 个 DCP；后者会把评估加密到约
  7.7k updates，并把 DCP 增至约 416 GB。
- 操作：用正式 source config 加窄 CLI overrides 做 Hydra `--cfg job --resolve`；断言预算、路径、并发、
  H/N、DSRL 维度、UTD、replay 与 fresh 状态均为预期值。
- 结果：resolved config 生成并通过断言，SHA-256 为
  `e99c212d1743e285dcda23cb129e2ed96545cceb36bebe772ae69a693b9df595`。

### FORMAL-004　启动正式训练与双层资源采样

- 时间：2026-07-28 18:54:12 CST。
- 操作：再次执行 branch/HEAD/clean、Ray/driver、GPU 空闲防覆盖 gate；将 prepare 目录原子移为独占
  run root；启动 driver、通用资源监控和 cgroup 分项监控。
- 结果：三个进程均成功启动，PID 分别为 `70062/70064/70065`；TensorBoard config/event 已创建。

### FORMAL-005　初始化观察

- 时间：2026-07-28 18:55–18:57 CST。
- 进展：driver、Ray GCS/raylet、2 actor、2 rollout 和 2 env workers 均存活；日志持续增长。
- 告警：RoboTwin `planner.py` 在模块加载时探测可选 Curobo/PyTorch3D 后端，打印
  `curobo.types.math`/`pytorch3d` 缺失 traceback。
- 判定：正式配置明确使用 `planner_backend=mplib`，fresh smoke 初始化时出现过相同告警并随后完成真实
  reset/step/rollout；因此当前先记为已知无害的可选后端告警，不安装包、不改环境。继续以首轮 rollout
  是否推进作为主链判据。
- 资源快照：两卡显存处于模型初始化阶段；`oom=0`、`oom_kill=0`。file cache 正在被回收，
  `memory.events max` 增长，符合 smoke 观察。

### FORMAL-006　warm-up 越过与 learned actor 接管

- 观察边界：2026-07-28 19:05–19:29 CST。
- step 1–12：global replay resident 从 40 增到 472；worker 每轮明确打印
  `DSRL replay warm-up: ... < 500 global transitions`，无 optimizer update。
- step 13：新增 40，resident 达 512；首次执行 `40 × UTD20 = 800` updates。未对前 472
  条 warm-up 数据追补更新，符合在线 DSRL 语义。
- phase：step 13 train rollout 仍是 update 前的 Gaussian collection；step 13 eval 已使用
  同步后的 learned actor，成功 1/12。step 14 是第一轮 learned-actor train rollout，
  成功 2/4；step 15 为 0/4。样本太小，只判调用链健康，不判效果趋势。
- 累计：step 13/14/15 分别新增 40/33/40 条并做 800/660/800 updates；至 step 15 为
  585 resident、2,260 learned updates。

### FORMAL-007　首三轮优化指标与墙钟

- step 13/14/15 的 critic loss 为 0.0210/0.0140/0.0260，alpha 为
  0.926/0.800/0.685，entropy 为 21.828/21.841/21.831；全部有限。
- Qπ 为 -2.684/-5.604/-8.222，Qdata 为 -2.645/-5.562/-8.188；10 个 Q heads 每轮
  保持紧密。未成功 reward=-1 且 `gamma^20≈0.98019`，因此早期 Q 向负值移动方向合理，
  当前不是发散证据。
- actor/critic/alpha 的裁剪前 grad norm：
  4.336/0.213/22.870、3.615/0.122/20.856、3.030/0.279/18.784。actor/alpha 按配置裁剪，
  critic 远低于阈值。
- 三轮 SAC 共 2,260 updates / 1,161.2 秒，即约 1.95 updates/s；普通 learned cycle
  约 35–38 秒 rollout，SAC 占不含 eval 墙钟约 91%。step 13 的 12-episode eval 为
  115.8 秒。
- 基于当前吞吐而不是 runner 的 warm-up 混合 ETA，估计全程约 79–84 小时，
  预计 2026-08-01 02:00–07:00 CST 完成；首个 step-65 DCP 约在 2026-07-29 02:00。

### FORMAL-008　资源与产物静态审阅

- 现场边界：主 poll 19:29:08，产物清点 19:31:39 CST。
- GPU0/1 当前 29,963/29,934 MiB，峰值 35,213/34,787 MiB；进入 SAC 后平均利用率约
  24.1%/24.7%。显存余量充足，但 UTD20 的小模型更新是墙钟瓶颈，不据此扩 env。
- cgroup current 约 231.3/240 GiB，其中 anon 约 38.9 GiB、file cache 约 190.4 GiB；
  anon 峰 47.7 GiB。`memory.events max` 仅在初始化期增加 93,384 后平台，当前 PSI
  avg10=0，OOM/OOM-kill 均为 0。判定为有回收历史的正常黄灯，不是泄漏。
- 磁盘尚余约 788 GB；run root 约 744 KiB。`save_interval=65`，所以当前没有 DCP，
  replay/actor/Q/target/temperature 暂时只存在于活进程。
- 报告与三张图已生成并逐图检查：
  [`FORMAL_STATUS_REPORT_STEP15_20260728.md`](./FORMAL_STATUS_REPORT_STEP15_20260728.md)。
- 绘图过程发现第一版 optimization 图把 Event 文件中最后一个 step 误当作首个 learned
  step，并把 step-13 eval 错标到 step 14；仅影响本地报告图，不影响训练。已改为合并
  TensorBoard step 13/14 与日志 step 15，逐 step 对齐后重新生成。
- 用户明确不要求持续在线监控。本轮不改配置、不干预训练；下次收到检查请求后再 live 刷新。

### FORMAL-009　状态产物同步与收尾

- 写入前 gate：服务器仍在 `codex/dsrl-pi0-robotwin@d664bf3`，upstream 相同、worktree
  clean；driver 和 2 actor/2 rollout/2 env workers 均存活。
- 仅上传 8 个文档/报告产物：根 `HANDOFF.md`、主计划、formal 流水账、step-15 状态报告、
  实际 resolved YAML 和三张 PNG；没有上传或修改生产代码、配置源、run root 或活进程状态。
- 校验：resolved YAML SHA-256 仍为
  `e99c212d1743e285dcda23cb129e2ed96545cceb36bebe772ae69a693b9df595`；三张 PNG
  均通过文件类型/非空检查；报告入口与 `FORMAL-008` 存在。
- 第一次 commit gate 被 `git diff --cached --check` 拒绝，因为报告首部使用了 Markdown
  的两个行尾空格；没有产生 commit。改成空引用行后重新上传，gate 通过。
- 结果：docs-only commit `1def9a24e46491f7801ad20badce6afd2fe81467`
  已推到 `personal/codex/dsrl-pi0-robotwin`；push 后 HEAD/upstream 相同且 worktree clean。
- 最后一次核验最初用 PowerShell 双引号拼接远程命令，本地提前展开了 `$()`，因此 helper
  在连接前拒绝参数；没有远程副作用。改用固定 command file 后，2026-07-28 19:44:02 CST
  确认 driver 仍存活、服务器 worktree clean。随后停止轮询。

### FORMAL-010　step-20 指标刷新、成功率与逐 step 用时

- 用户要求：再次审阅全部主要指标，绘制成功率/采样效率/优化/资源趋势；解释 eval 周期、
  是否能证明采样效率优势，以及总 step 数和每个 step 的实际用时。
- 现场边界：2026-07-28 20:08:11 CST；最新完整 metric table 为 step 20，资源 CSV
  覆盖到 20:09:12。
- 现场：driver、2 actor、2 rollout、2 env workers 存活；branch/HEAD/upstream 为
  `codex/dsrl-pi0-robotwin@95e62518`，worktree clean；OOM/OOM-kill 为 0。

#### FORMAL-010.1　服务器只读快照命令

密码只通过当前 PowerShell 进程的 `SEETA_SSH_PASSWORD` 注入；流水账按安全规则以
`<process-only secret>` 代替真实值。

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_metrics_refresh_snapshot.sh'
```

`remote_dsrl_metrics_refresh_snapshot.sh` 原样如下：

```bash
set -euo pipefail

REPO=/root/autodl-tmp/RLinf_fastwam_rlinf
RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1

echo "TIME=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "HOST=$(hostname)"
echo "USER=$(id -un)"

cd "$REPO"
echo "BRANCH=$(git branch --show-current)"
echo "HEAD=$(git rev-parse HEAD)"
echo "UPSTREAM=$(git rev-parse '@{upstream}')"
echo "STATUS_BEGIN"
git status --short
echo "STATUS_END"

PID=$(cat "$RUN/formal.pid")
kill -0 "$PID"
echo "DRIVER_PID=$PID"
echo "DRIVER_ALIVE=1"
pgrep -c -f '^ray::EmbodiedSACFSDPPolicy' | awk '{print "ACTOR_WORKERS=" $1}'
pgrep -c -f '^ray::MultiStepRolloutWorker' | awk '{print "ROLLOUT_WORKERS=" $1}'
pgrep -c -f '^ray::EnvWorker' | awk '{print "ENV_WORKERS=" $1}'

echo "LAST_GLOBAL_STEP"
grep 'Global Step:' "$RUN/formal_driver.log" | tail -n 1
echo "LAST_REPLAY"
grep 'sac/global_resident_transitions=' "$RUN/formal_driver.log" | tail -n 1
echo "LAST_LOG_TIME=$(stat -c '%y' "$RUN/formal_driver.log")"
echo "LOG_BYTES=$(stat -c '%s' "$RUN/formal_driver.log")"

echo "GPU"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
echo "CGROUP_CURRENT=$(cat /sys/fs/cgroup/memory.current)"
echo "CGROUP_MAX=$(cat /sys/fs/cgroup/memory.max)"
grep -E '^(anon|file|inactive_file|active_file) ' /sys/fs/cgroup/memory.stat
cat /sys/fs/cgroup/memory.events
cat /sys/fs/cgroup/memory.pressure

echo "CHECKPOINTS"
find "$RUN" -type d -path '*/checkpoints/global_step_*' -printf '%TY-%Tm-%Td %TH:%TM:%TS %p\n' | sort
echo "CHECKPOINT_SIZES"
find "$RUN" -type d -path '*/checkpoints/global_step_*' -print0 |
  xargs -0 -r du -sh
echo "EVENT_FILES"
find "$RUN/tensorboard" -maxdepth 1 -type f -name 'events.out.tfevents.*' -printf '%s %p\n'
echo "RUN_SIZE=$(du -sh "$RUN" | awk '{print $1}')"
df -BG --output=avail /root/autodl-tmp | tail -n 1 | awk '{print "DISK_FREE=" $1}'
sha256sum "$RUN/FORMAL_RUN_VALIDATED_RESOLVED_20260728.yaml"
```

#### FORMAL-010.2　日志与采样文件下载命令

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; $helper='C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py'; $run='/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1'; $dst='C:\Users\86136\Documents\rl\.tmp'; python $helper get "$run/formal_driver.log" "$dst/dsrl_refresh_formal_driver.log"; python $helper get "$run/metrics.log" "$dst/dsrl_refresh_metrics.log"; python $helper get "$run/tensorboard/events.out.tfevents.1785236070.autodl-container-nekaqbwt43-6ce5babb.70062.0" "$dst/dsrl_refresh_events.tfevents"; python $helper get "$run/resource_monitor/resources.csv" "$dst/dsrl_refresh_resources.csv"; python $helper get "$run/resource_monitor/cgroup_detail.csv" "$dst/dsrl_refresh_cgroup_detail.csv"; python $helper get "$run/resource_monitor/peak.txt" "$dst/dsrl_refresh_peak.txt"
```

#### FORMAL-010.3　图表、摘要和 timing CSV 生成命令

```powershell
python 'C:\Users\86136\Documents\rl\.tmp\build_dsrl_formal_step20_plots.py' --events 'C:\Users\86136\Documents\rl\.tmp\dsrl_refresh_events.tfevents' --success-out 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP20_20260728.png' --optimization-out 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_OPTIMIZATION_TRENDS_STEP20_20260728.png' --summary-out 'C:\Users\86136\Documents\rl\.tmp\dsrl_refresh_summary_step20.json' --timing-csv-out 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_STEP_TIMING_STEP20_20260728.csv'

python 'C:\Users\86136\Documents\rl\.tmp\build_dsrl_formal_resource_plot.py' --resources 'C:\Users\86136\Documents\rl\.tmp\dsrl_refresh_resources.csv' --cgroup 'C:\Users\86136\Documents\rl\.tmp\dsrl_refresh_cgroup_detail.csv' --out 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_RESOURCE_CURVES_STEP20_20260728.png'
```

上述命令是本次实际执行路径；验证后将两份脚本原样归档到
`evidence/tools/build_dsrl_formal_step20_plots.py` 和
`evidence/tools/build_dsrl_formal_resource_plot.py`，供下次从最新 event/CSV 重建，避免
只留下不可复现的 PNG。

归档后入口复核命令与结果：

```powershell
python -B 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_step20_plots.py' --help; python -B 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_resource_plot.py' --help
```

两者均退出 0 并展示预期参数。

#### FORMAL-010.4　关键结果

- 总预算是 650 collection cycles，不是 650 optimizer updates。eval 每 13 cycles，
  DCP 每 65 cycles。
- warm-up step 1–12 平均 40.1 秒；step 13 首次更新并评估为 571.4 秒；learned step
  14–20 平均 400.4 秒，范围 361.4–445.0 秒。完整逐 step 数值在 timing CSV。
- Gaussian train phase 为 2/52；learned train phase 为 8/28；最新 trailing-20 为 30%。
  它是积极的 run 内方向，不是 paired A/B。
- formal eval 当前只有 step 13 的 1/12；Wilson 95% 区间约 1.5%–35.4%。下一次为 step 26。
- critic loss 升到 0.880，是当前首要观察项；critic grad 4.84 低于 clip=10，Qπ/Qdata
  已从最低点回到 -11.25/-11.60，全部 finite。
- GPU 峰值未变；cgroup anon 约 40.9 GiB、低于初始化峰值，reclaim events 已平台，
  OOM=0，未改变并发。

#### FORMAL-010.5　可视化 QA 问题与处理

- 三张 PNG 均用本地 `view_image` 逐张检查，轴、图例和标注无重叠。
- inline success 图先检查 fragment：无 `doctype/html/body`、无转义引号或字面 `\n`，
  root id 唯一，文件约 12 KiB。
- 第一次 standalone render 目标写到 `C:\tmp`，因当前 sandbox 身份无该目录写权限而
  `PermissionError`；没有项目或服务器副作用。改写到项目 `.tmp` 后成功：

```powershell
python 'E:\Codex\home\plugins\cache\openai-bundled\visualize\1.0.14\skills\visualize\scripts\render.py' 'E:\Codex\home\visualizations\2026\07\27\019fa3db-1fba-7d22-a35c-eaa3b1ec680d\dsrl-success-sample-efficiency-step20.html' 'C:\Users\86136\Documents\rl\.tmp\dsrl-success-sample-efficiency-step20-preview.html'
```

- Playwright 默认 Chromium 未安装；改用系统 Chrome 的自动 preview 又在工具 30 秒边界超时。
  没有安装浏览器或依赖，也没有修改训练环境。保留通过的 fragment 结构检查与静态 PNG
  视觉检查，不把 preview 工具问题误记为训练问题。
- 文档末尾空白与 timing 行数检查最终输出
  `TRAILING_WHITESPACE=0`、`TIMING_ROWS=20`；报告首部最初的 Markdown 双空格已改成空引用行。

### FORMAL-011　step-20 文档同步、验证与云端收口

- 写入边界：只同步报告、图表、timing CSV 和复现这些报告所需的脚本；不改生产代码、
  source config、run root、训练进程或参数。
- 同步前 gate：服务器仍为
  `codex/dsrl-pi0-robotwin@95e6251841f6d7256ee2c13de053d4618e02e00e`，
  upstream 相同、worktree clean，driver 和 worker 均存活。
- 为避免聊天中省略命令细节，现场快照、服务器验证、commit/push 三份 shell 均原样归档到
  `evidence/tools/`。密码仍只通过当前 PowerShell 进程注入，记录中以
  `<process-only secret>` 代替。

精确上传命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; $helper='C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py'; $repo='/root/autodl-tmp/RLinf_fastwam_rlinf'; $pairs=@(
  @('C:\Users\86136\Documents\rl\HANDOFF.md',"$repo/HANDOFF.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\00_INDEX_AND_IMPLEMENTATION_PLAN.md',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_TRAINING_LOG_20260728.md',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_STATUS_REPORT_STEP20_20260728.md',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP20_20260728.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP20_20260728.png',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP20_20260728.png"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_OPTIMIZATION_TRENDS_STEP20_20260728.png',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_STEP20_20260728.png"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_RESOURCE_CURVES_STEP20_20260728.png',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP20_20260728.png"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_STEP_TIMING_STEP20_20260728.csv',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STEP_TIMING_STEP20_20260728.csv"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_step20_plots.py',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/build_dsrl_formal_step20_plots.py"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_resource_plot.py',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/build_dsrl_formal_resource_plot.py"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\remote_dsrl_metrics_refresh_snapshot.sh',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/remote_dsrl_metrics_refresh_snapshot.sh"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\remote_dsrl_step20_docs_validate.sh',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/remote_dsrl_step20_docs_validate.sh"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\remote_dsrl_step20_docs_commit.sh',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/remote_dsrl_step20_docs_commit.sh")
); foreach ($pair in $pairs) { python $helper put $pair[0] $pair[1]; if ($LASTEXITCODE -ne 0) { throw "upload failed: $($pair[0])" } }
```

精确验证与收口命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\remote_dsrl_step20_docs_validate.sh'

$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\remote_dsrl_step20_docs_commit.sh'
```

验证脚本检查 branch/HEAD/upstream、driver 存活、`git diff --check`、三张 PNG、20 行
timing 数据、报告入口和两份 Python 绘图入口，只有全部通过才打印 `VALIDATION=PASS`。

- 第一次执行收口脚本时，`git diff --cached --check` 把 timing CSV 的 21 个 Windows
  CRLF 行尾全部判为 trailing whitespace，脚本在 commit 前退出 1；没有 commit、push
  或训练副作用。此前 `TRAILING_WHITESPACE=0` 只检查了 Markdown，覆盖面不足。
- 修正只把该 CSV 从 CRLF 规范化为 LF，不改任一字段或数值；随后重新上传流水账和 CSV，
  再运行同一验证及收口脚本。该问题归类为报告产物跨平台行尾问题，不是训练异常。

修正后的精确重传命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; $helper='C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py'; $repo='/root/autodl-tmp/RLinf_fastwam_rlinf'; python $helper put 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_STEP_TIMING_STEP20_20260728.csv' "$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STEP_TIMING_STEP20_20260728.csv"; if ($LASTEXITCODE -ne 0) { throw 'CSV upload failed' }; python $helper put 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_TRAINING_LOG_20260728.md' "$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md"; if ($LASTEXITCODE -ne 0) { throw 'ledger upload failed' }
```

收口脚本只 stage 上述 13 个精确路径，要求 staged 数量恰为 13，再 commit、在 45 秒边界内
push，并最终复核 HEAD/upstream、clean worktree 和 driver 存活。实际 commit/push 结果在本轮
回复中报告；commit 自身的 hash 不写回同一个 commit，以避免自引用修改。

### FORMAL-012　step-30 只读刷新与横版可视化

- 用户要求：重新检查当前训练，并修正上一轮纵向长图在 Codex 消息中显示过小的问题。
- 现场边界：2026-07-28 21:03:56 CST；最新完整 step 30，资源 CSV 覆盖到 21:04:56。
- 服务器只读 gate：branch/HEAD/upstream 为
  `codex/dsrl-pi0-robotwin@723eb475fd6524851df8471b1f05f6f92aeea508`，
  worktree clean；driver、2 actor、2 rollout、2 env workers 存活。

服务器快照命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\remote_dsrl_metrics_refresh_snapshot.sh'
```

日志与采样文件下载命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; $helper='C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py'; $run='/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1'; $dst='C:\Users\86136\Documents\rl\.tmp'; python $helper get "$run/formal_driver.log" "$dst/dsrl_step30_formal_driver.log"; if($LASTEXITCODE -ne 0){throw 'driver download failed'}; python $helper get "$run/metrics.log" "$dst/dsrl_step30_metrics.log"; if($LASTEXITCODE -ne 0){throw 'metrics download failed'}; python $helper get "$run/tensorboard/events.out.tfevents.1785236070.autodl-container-nekaqbwt43-6ce5babb.70062.0" "$dst/dsrl_step30_events.tfevents"; if($LASTEXITCODE -ne 0){throw 'events download failed'}; python $helper get "$run/resource_monitor/resources.csv" "$dst/dsrl_step30_resources.csv"; if($LASTEXITCODE -ne 0){throw 'resources download failed'}; python $helper get "$run/resource_monitor/cgroup_detail.csv" "$dst/dsrl_step30_cgroup_detail.csv"; if($LASTEXITCODE -ne 0){throw 'cgroup download failed'}; python $helper get "$run/resource_monitor/peak.txt" "$dst/dsrl_step30_peak.txt"; if($LASTEXITCODE -ne 0){throw 'peak download failed'}
```

横版图与摘要生成命令：

```powershell
python -B 'docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_step20_plots.py' --events '.tmp\dsrl_step30_events.tfevents' --success-out 'docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP30_20260728_WIDE.png' --optimization-out 'docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_OPTIMIZATION_TRENDS_STEP30_20260728_WIDE.png' --summary-out '.tmp\dsrl_step30_summary_wide.json' --timing-csv-out '.tmp\dsrl_step30_timing_wide.csv' --layout landscape

python -B 'docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_resource_plot.py' --resources '.tmp\dsrl_step30_resources.csv' --cgroup '.tmp\dsrl_step30_cgroup_detail.csv' --out 'docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_RESOURCE_CURVES_STEP30_20260728_WIDE.png' --layout landscape

python -m py_compile 'docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_step20_plots.py' 'docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_resource_plot.py'
```

结果：

- step 26 formal eval 为 7/12；learned train phase 33/68，trailing-20 为 60%。
- step 30 为 28 transitions / 560 updates，整轮 322.5 秒；累计 11,800 updates，
  实测 1.950 updates/s。
- critic loss 在 step 25 达 1.929 后回落至 0.848；Q、10-head spread 和三个 grad
  均有限。alpha/entropy 降到 0.076/16.77，作为后续观察项。
- GPU/cgroup 无 OOM 或 PSI 压力；`memory.events max` 未再增加。env-worker RSS 从
  19:15 的约 11.9 GiB 升至 21:05 的约 18.3 GiB，是新的资源观察项，但总 anon
  仍低于初始化峰值。

可视化 QA：

- 第一版横版 success 图继承了旧 `ylim=0.62`，把 step 22 之后的 75%/100% 点裁掉；
  逐图检查后改为 `[-0.02,1.02]` 并重绘。
- 第一版横版 resource 图的窄面板仍有时间 tick 重叠；先用 `AutoDateLocator` 时产生
  interval warning，最终改为按总时长选择 30 分钟/1 小时/4 小时/12 小时 locator，
  重绘后无 warning。
- 两个已有绘图脚本只新增 opt-in `--layout landscape`；默认 portrait 不变。
- 三张最终 PNG 均为约 16:9 横版并已逐张视觉检查；成功率轴覆盖完整 0%–100%，资源图
  tick 无重叠。对话内 success 图使用相同 step-30 数据和轴语义。
- `py_compile` 生成的本地 `evidence/tools/__pycache__` 在确认解析路径位于工作区后删除，
  没有上传或触及服务器环境。
- 对话图 fragment 检查为 10,278 bytes、唯一 root、无 document wrapper/转义字面量，
  `render.py` 包装成功。第一次 inline `node -e` 因 Windows 参数引号被剥离而语法失败；
  第二次临时 `.cjs` 入口又因 sandbox 对 `C:\Users\86136` 的 `lstat` 权限失败。最终改为
  process-env 传 fragment 路径的 `node -e`，输出 `JS_SYNTAX=PASS`；两次失败均只有本地
  QA 进程，无项目或服务器副作用。

最终 fragment 检查命令：

```powershell
$env:VISUAL_PATH='E:\Codex\home\visualizations\2026\07\27\019fa3db-1fba-7d22-a35c-eaa3b1ec680d\dsrl-formal-step30-success.html'; node -e "const fs=require('fs'); const s=fs.readFileSync(process.env.VISUAL_PATH,'utf8'); const m=s.match(/<script>([\s\S]*?)<\/script>/); if(!m) throw new Error('missing script'); new Function(m[1]); console.log('JS_SYNTAX=PASS');"

python 'E:\Codex\home\plugins\cache\openai-bundled\visualize\1.0.14\skills\visualize\scripts\render.py' 'E:\Codex\home\visualizations\2026\07\27\019fa3db-1fba-7d22-a35c-eaa3b1ec680d\dsrl-formal-step30-success.html' 'C:\Users\86136\Documents\rl\.tmp\dsrl-formal-step30-success-preview.html'
```

最终本地 gate 对两个脚本运行 `-B --help`，检查三张 PNG 均为
`2430×1440`，确认无残留 `__pycache__`，并对本轮 9 个拟同步路径运行
`git -c safe.directory=... diff --check`；输出 `LOCAL_VALIDATION=PASS`。

#### FORMAL-012.1　文档同步与收口命令

写入前只读 gate 命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_step30_docs_preflight.sh'
```

该 command file 精确检查 branch、HEAD、upstream、clean worktree 和 formal driver PID，
全部通过后输出 `PREFLIGHT=PASS`。

九个文档/报告路径的精确上传命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; $helper='C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py'; $repo='/root/autodl-tmp/RLinf_fastwam_rlinf'; $pairs=@(
  @('C:\Users\86136\Documents\rl\HANDOFF.md',"$repo/HANDOFF.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\00_INDEX_AND_IMPLEMENTATION_PLAN.md',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_TRAINING_LOG_20260728.md',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_STATUS_REPORT_STEP30_20260728.md',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP30_20260728.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP30_20260728_WIDE.png',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP30_20260728_WIDE.png"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_OPTIMIZATION_TRENDS_STEP30_20260728_WIDE.png',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_STEP30_20260728_WIDE.png"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_RESOURCE_CURVES_STEP30_20260728_WIDE.png',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP30_20260728_WIDE.png"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_step20_plots.py',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/build_dsrl_formal_step20_plots.py"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_resource_plot.py',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/tools/build_dsrl_formal_resource_plot.py")
); foreach($pair in $pairs){python $helper put $pair[0] $pair[1]; if($LASTEXITCODE -ne 0){throw "upload failed: $($pair[0])"}}
```

服务器验证与 commit/push 命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_step30_docs_validate.sh'

$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_step30_docs_commit.sh'
```

validate script 要求 dirty path 恰为 9、`git diff --check` 通过、三张 PNG 均为
2430×1440、报告含 step 30/7-of-12/FORMAL-012，且两个 Python 入口在服务器 venv
下 `-B --help` 成功。commit script 只 stage 上述 9 个路径，再执行：

```bash
git diff --cached --check
test "$(git diff --cached --name-only | wc -l)" -eq 9
git commit -m "docs(dsrl): refresh formal step 30 report"
timeout --signal=TERM --kill-after=5s 45s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin
```

上传后的服务器验证已输出 `VALIDATION=PASS`；实际 commit/push hash 在本轮回复报告。

### FORMAL-013　step-53 只读刷新、第四次 eval 与资源复核

- 用户要求：按上一轮同样方式刷新正在运行的训练，继续给出横版成功率、优化和资源趋势。
- 第一现场边界：2026-07-28 23:02:13 CST；服务器最新日志为 step 52。下载 event
  期间 step 53 于 23:04:08 完成，因此最终报告统一使用 step 53 event 数据，资源 CSV
  覆盖到 23:05:42。
- 只读 gate：branch/HEAD/upstream 为
  `codex/dsrl-pi0-robotwin@b01661e8a6b3ca1b883fb61d4ade9a467ffd84b5`，worktree
  clean；driver PID 70062、2 actor、2 rollout、2 env workers 均存活。

现场快照命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\remote_dsrl_metrics_refresh_snapshot.sh'
```

快照输出 step 52、resident 1,658，GPU 当前约 32.4/32.4 GB；cgroup
`oom=0/oom_kill=0`、PSI avg10=0，尚无 checkpoint，磁盘可用 789 GB。

第一次追加日志检查直接把含分号和管道的远程 shell 作为 `run` 位置参数传入：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run "RUN=/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1; find \"$RUN\" -maxdepth 2 -type f -printf '%s %p\n' | sort -n; ..."
```

PowerShell/argparse 在引号层拆开参数，helper 报 `unrecognized arguments`，远程命令没有执行。
修复是把同一只读 shell 写入 `.tmp/remote_dsrl_step52_detail.sh`，再用
`--command-file`：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_step52_detail.sh'
```

该命令退出 0。日志中的两处 `Traceback` 都在 18:54 初始化阶段，来自可选
`CuroboPlanner` 缺 `curobo.types.math`；RoboTwin 随后正常进入本任务使用的 qpos 路径，
训练已经推进 53 步。定向复核命令：

```powershell
rg -n -i "(^|[^a-z])(nan|inf)([^a-z]|$)|traceback|cuda out of memory|outofmemory|oom" 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_formal_driver.log'
```

只返回上述四行初始化 traceback 标题，没有 NaN/Inf/OOM 或后续 traceback。

第一次下载按六个独立 SFTP 连接顺序执行：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; $helper='C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py'; $run='/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1'; python $helper get "$run/tensorboard/events.out.tfevents.1785236070.autodl-container-nekaqbwt43-6ce5babb.70062.0" 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_events.tfevents'; python $helper get "$run/resource_monitor/resources.csv" 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_resources.csv'; python $helper get "$run/resource_monitor/cgroup_detail.csv" 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_cgroup_detail.csv'; python $helper get "$run/resource_monitor/peak.txt" 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_peak.txt'; python $helper get "$run/formal_driver.log" 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_formal_driver.log'; python $helper get "$run/metrics.log" 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_metrics.log'
```

工具在 60 秒边界超时：event 和 resources 完整，cgroup 只下载 425,984 bytes。没有覆盖
远程文件。窄修复是在一个已验证 host-key 的 Paramiko/SFTP 会话里依次下载到
`.partial`，完成后才原子替换本地目标；实际入口和命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python -B 'C:\Users\86136\Documents\rl\.tmp\download_dsrl_step52_snapshot.py'
```

六个文件分别为 163,408、1,092,813、782,399、756、441,082 和 377,146 bytes，
全部完成。密码始终只在当前 PowerShell 进程存在。

横版图与摘要生成命令：

```powershell
python -B 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_step20_plots.py' --events 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_events.tfevents' --success-out 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP52_20260728_WIDE.png' --optimization-out 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_OPTIMIZATION_TRENDS_STEP52_20260728_WIDE.png' --summary-out 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_metrics_summary.json' --timing-csv-out 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_timing.csv' --layout landscape

python -B 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\tools\build_dsrl_formal_resource_plot.py' --resources 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_resources.csv' --cgroup 'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_cgroup_detail.csv' --out 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_RESOURCE_CURVES_STEP52_20260728_WIDE.png' --layout landscape
```

生成后 summary 显示 event 已含 step 53。为避免报告名与图内标题不一致，在确认三个
STEP53 目标均不存在后，用三个精确 `Move-Item -LiteralPath` 把本轮新生成的 STEP52
文件改名为 STEP53；没有移动历史 step-20/30 文件。

关键结果：

- step 53：resident 1,681、33,620 requested interactions、24,180 optimizer updates；
  learned train 101/160，trailing-20 为 75%。
- step 39/52 formal eval 为 8/12、10/12；四次 eval 单调
  `1/12 → 7/12 → 8/12 → 10/12`。
- critic loss 0.395，Q-head range 0.111；actor/critic/alpha grad
  0.54/2.96/0.072，均 finite。
- alpha/entropy 为 0.0061/-4.15；target entropy=-16，当前仍是朝目标下降，下一节点
  检查是否减速。
- 最近 step 40–53 平均 305.4 秒；排除 eval 为 296.0 秒。按该窗口估算剩余约
  50.7 小时；step 65 约 61 分钟后到达，另加首次 DCP 未知保存耗时。
- 资源 23:05：两卡约 31.4 GiB，峰 34.4/34.1 GiB；cgroup 234.7/240 GiB，
  anon 45.6 GiB、file 187.1 GiB、inactive file 155.4 GiB、PSI=0、OOM=0。
  env RSS 峰 21.5 GiB，仍是观察项，但总 anon 低于初始化峰值 47.8 GiB。

TensorBoard 最新 event wall time 读取第一次 20 秒超时，随后只提高本地读取超时到
60 秒，没有重下文件或触及训练：

```powershell
$env:TF_CPP_MIN_LOG_LEVEL='2'; python -B -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; import datetime; p=r'C:\Users\86136\Documents\rl\.tmp\dsrl_step52_events.tfevents'; e=EventAccumulator(p,size_guidance={'scalars':0}); e.Reload(); x=e.Scalars('time/step')[-1]; print('LATEST_STEP',x.step+1); print('WALL_TIME_CST',datetime.datetime.fromtimestamp(x.wall_time).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))"
```

输出 step 53、2026-07-28 23:04:08 CST。

可视化与报告 QA：

- 三张 PNG 均为 2430×1440 横版，已逐张用 `view_image(detail=original)` 检查；
  0%–100% 成功率完整、时间 tick 和图例无重叠。
- 新对话图
  `dsrl-formal-step53-success.html` 使用同一 step-53 数据，大小 10,615 bytes、唯一
  root、无 document wrapper、转义引号或字面 `\n`，JavaScript 语法通过。
- fragment 结构/语法和 standalone wrapper 命令：

```powershell
$p='E:\Codex\home\visualizations\2026\07\27\019fa3db-1fba-7d22-a35c-eaa3b1ec680d\dsrl-formal-step53-success.html'; $text=Get-Content -LiteralPath $p -Raw -Encoding UTF8; node -e "const fs=require('fs');const s=fs.readFileSync(process.argv[1],'utf8');const m=s.match(/<script>([\s\S]*?)<\/script>/);if(!m)throw new Error('script missing');new Function(m[1]);console.log('JS_OK');" $p

python -B 'E:\Codex\home\plugins\cache\openai-bundled\visualize\1.0.14\skills\visualize\scripts\render.py' 'E:\Codex\home\visualizations\2026\07\27\019fa3db-1fba-7d22-a35c-eaa3b1ec680d\dsrl-formal-step53-success.html' 'C:\Users\86136\Documents\rl\.tmp\dsrl-formal-step53-success-preview.html'
```

本轮只新增状态报告和三张图，并更新 HANDOFF、主计划和本流水账；没有修改 production
code、配置、run root、checkpoint 或训练进程。

#### FORMAL-013.1　step-53 文档同步命令

写入前只读 gate：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_step53_docs_preflight.sh'
```

七个精确文档/报告路径上传：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; $helper='C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py'; $repo='/root/autodl-tmp/RLinf_fastwam_rlinf'; $pairs=@(
  @('C:\Users\86136\Documents\rl\HANDOFF.md',"$repo/HANDOFF.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\00_INDEX_AND_IMPLEMENTATION_PLAN.md',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_TRAINING_LOG_20260728.md',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_STATUS_REPORT_STEP53_20260728.md',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP53_20260728.md"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP53_20260728_WIDE.png',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_SUCCESS_SAMPLE_EFFICIENCY_STEP53_20260728_WIDE.png"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_OPTIMIZATION_TRENDS_STEP53_20260728_WIDE.png',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_OPTIMIZATION_TRENDS_STEP53_20260728_WIDE.png"),
  @('C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_RESOURCE_CURVES_STEP53_20260728_WIDE.png',"$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_RESOURCE_CURVES_STEP53_20260728_WIDE.png")
); foreach($pair in $pairs){python $helper put $pair[0] $pair[1]; if($LASTEXITCODE -ne 0){throw "upload failed: $($pair[0])"}}
```

服务器验证与 docs-only commit/push：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_step53_docs_validate.sh'

$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_step53_docs_commit.sh'
```

validate 要求 dirty set 精确等于这 7 条、`git diff --check` 通过、三张 PNG 都是
2430×1440、报告/主计划/HANDOFF/FORMAL-013 入口存在且 driver 存活。commit 脚本只
stage 这 7 条，以 `docs(dsrl): refresh formal step 53 report` 提交，45 秒边界 push，
最后检查 HEAD/upstream、clean worktree、driver 和最新 step。

实际结果：

- preflight 在 step 55 输出 `PREFLIGHT=PASS`；上传 7 条后 validate 输出
  `VALIDATION=PASS`，dirty set 精确为 7。
- docs-only commit 成功，hash 为
  `4447d40211be8c78874bf4b000c871e7fbd93561`；服务器 worktree clean。
- commit 脚本内第一次 45 秒 push 没有完成。23:20 只读快照确认
  `HEAD=4447d402...`、`UPSTREAM=b01661e8...`，说明提交完整但尚未发布；训练已到
  step 56。
- 第二次前台重试使用 55 秒硬边界：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_step53_push_retry.sh'
```

输出 `PUSH_STATUS=124`。为区分“45/55 秒太短”和“出口不可达”，随后启动一个不占 SSH
会话、180 秒自终止的后台 push，并用只读状态脚本轮询：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_step53_push_background_start.sh'

$env:SEETA_SSH_PASSWORD='<process-only secret>'; python 'C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py' run --command-file 'C:\Users\86136\Documents\rl\.tmp\remote_dsrl_step53_push_background_status.sh'
```

后台最终退出 128，精确错误为：

```text
fatal: unable to access 'https://github.com/Yutenji-Nyamu/rlinf_fastwam.git/':
Failed to connect to github.com port 443 after 129645 ms: Connection timed out
```

这确认是服务器 GitHub 出口故障，不是 staged scope、commit、认证提示或训练错误。本机
备用发布检查执行 `gh --version; gh auth status`，发现 `gh` 未安装；没有擅自安装工具、
创建 PR 或改 Git 远端。最终停点为：服务器 commit `4447d402...` clean、upstream
`b01661e8...`，训练 step 57 继续。下次 live 刷新后只需重试：

```bash
timeout --signal=TERM --kill-after=5s 55s \
  git -c http.version=HTTP/1.1 push personal HEAD:codex/dsrl-pi0-robotwin
```

为让下一轮在服务器端也能直接看到该 blocker，本轮最后只重传更新后的 `HANDOFF.md`
和本流水账，并做第二个本地 docs-only commit；不再尝试 GitHub push。实际命令：

```powershell
$env:SEETA_SSH_PASSWORD='<process-only secret>'; $helper='C:\Users\86136\Documents\rl\local_scripts\remote_exec_autodl.py'; $repo='/root/autodl-tmp/RLinf_fastwam_rlinf'; python $helper put 'C:\Users\86136\Documents\rl\HANDOFF.md' "$repo/HANDOFF.md"; python $helper put 'C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\evidence\FORMAL_TRAINING_LOG_20260728.md' "$repo/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md"
```

服务器侧在 HEAD 仍为 `4447d402...`、upstream 仍为 `b01661e8...`、worktree 起始 clean
且 driver 存活的前提下，只 stage 这两条，运行 `git diff --cached --check`，然后：

```bash
git commit -m "docs(dsrl): record step 53 push blocker"
```

该 bookkeeping commit 的 hash 在本轮最终回复报告，不为写回自身再制造第三个 commit。

## 6. 停止与干预条件

发生下列任一项才停止并保留现场：

- CUDA OOM、cgroup `oom/oom_kill` 增加或 worker/driver crash；
- loss、Q、alpha、gradient 出现 NaN/Inf；
- 初始化完成后超过 20 分钟没有 rollout/update 进展；
- anon/RSS 持续异常增长并逼近上限，或 memory PSI/回收压力使 cycle、DCP 持续恶化；
- checkpoint 不完整、出现残留临时目录，或 resume/trainer/replay/shadow 保存失败；
- `/root/autodl-tmp` 可用空间低于 200 GB；
- 冻结 π0、N/H、reward/discount/replay 等关键训练语义出现漂移证据。
