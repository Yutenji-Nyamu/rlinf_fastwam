# DSRL × π0 × RoboTwin 正式训练流水账（2026-07-28）

> 本文件是本次 formal run 的操作级事实源，持续记录授权、命令、配置、产物、进度、资源、问题与处理。设计与实现仍以
> [`00_INDEX_AND_IMPLEMENTATION_PLAN.md`](../00_INDEX_AND_IMPLEMENTATION_PLAN.md) 为唯一主计划；实现过程与 smoke 细节分别见
> [`IMPLEMENTATION_LOG.md`](./IMPLEMENTATION_LOG.md) 和
> [`SMOKE_EXECUTION_LOG_20260728.md`](./SMOKE_EXECUTION_LOG_20260728.md)。

## 0. 当前状态

- 状态：训练继续运行；截至 2026-07-28 19:29:08 CST，最新完整记录为 global step 15，
  step 16 已开始。
- 进度：global replay resident 585；warm-up 已在 step 13 越过；learned SAC 累计
  2,260 optimizer updates。
- 进程：driver、2 actor、2 rollout、2 env workers 仍存活；`oom=0`、`oom_kill=0`，
  已报告 loss/Q/alpha/gradient 均有限。
- 本次静态状态报告：
  [`FORMAL_STATUS_REPORT_STEP15_20260728.md`](./FORMAL_STATUS_REPORT_STEP15_20260728.md)。
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
当前覆盖到 step 15 / 19:29 的手机可读快照为：

- [`FORMAL_PROGRESS_CURVES_STEP15_20260728.png`](./FORMAL_PROGRESS_CURVES_STEP15_20260728.png)；
- [`FORMAL_OPTIMIZATION_CURVES_STEP15_20260728.png`](./FORMAL_OPTIMIZATION_CURVES_STEP15_20260728.png)；
- [`FORMAL_RESOURCE_CURVES_STEP15_20260728.png`](./FORMAL_RESOURCE_CURVES_STEP15_20260728.png)。

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

## 6. 停止与干预条件

发生下列任一项才停止并保留现场：

- CUDA OOM、cgroup `oom/oom_kill` 增加或 worker/driver crash；
- loss、Q、alpha、gradient 出现 NaN/Inf；
- 初始化完成后超过 20 分钟没有 rollout/update 进展；
- anon/RSS 持续异常增长并逼近上限，或 memory PSI/回收压力使 cycle、DCP 持续恶化；
- checkpoint 不完整、出现残留临时目录，或 resume/trainer/replay/shadow 保存失败；
- `/root/autodl-tmp` 可用空间低于 200 GB；
- 冻结 π0、N/H、reward/discount/replay 等关键训练语义出现漂移证据。
