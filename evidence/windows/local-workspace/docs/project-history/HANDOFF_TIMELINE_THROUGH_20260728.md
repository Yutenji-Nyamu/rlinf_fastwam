# AutoDL / RLinf 交接历史时间线（截至 2026-07-28）

本文件是唯一动态入口；任何带时间戳的状态都只是最后已知快照，进程、GPU/RAM、日志、checkpoint、HEAD 和 dirty tree 必须现场只读刷新后才能称为当前。

## 2026-07-28 当前主题路由

- DSRL × π0 × RoboTwin：唯一计划为 `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md`。首版已冻结 H=50、N=20、4 train env、`rollout_epoch=1`、warm-up 500 global macros、UTD20、flat replay 25k，以及 narrow target-shadow resume 修复；N=50 暂缓。
- 当前只完成设计和本机文档更新，没有刷新或改动服务器。下一轮在本机只编辑代码/文档/diff，不运行 compose/import/compile/测试；上传 diff、服务器检查、smoke 和训练都要先展示完整配置/命令/输出/资源/停止条件并取得明确授权。
- DSRL smoke 保持 formal 两卡/4 env/N20/H50/batch256/micro64/UTD20/replay25k，只把 global warm-up 改为 4、fresh/resume 绝对 max steps 改为 1/2，并各跑 4 个 eval episodes。
- Fast-WAM × RoboTwin × RLinf：按需先读 `docs/fastwam-robotwin-rlinf-grpo/00_INDEX.md`；真正讨论实施时再读 `05_IMPLEMENTATION_PLAN.md` 和实施日志。
- 旧外部交接 `C:\Users\86136\Documents\Codex\2026-07-13\n\autodl-rlinf-fastwam-handoff-20260716.md` 只作历史 Fast-WAM 基线，不再默认读取，也不再称为当前权威。

## 新任务启动顺序

1. 读取本目录 `AGENTS.md`、`PROJECT_CONTEXT.md` 和本文件。
2. 只读当前主题对应的唯一专题文档；历史档和其他主题文档不默认加载。
3. 涉及动态服务器事实时，先只读刷新 HEAD/status、进程、GPU/RAM、checkpoint 和最新日志；未经新的明确授权不写服务器。

## 历史时间线（以下均不是当前现场）

## 2026-07-17 当前实施状态

- 23:38 CST 最终现场复核：无训练/Fast-WAM probe/Embodied workers 残留；两张 A800 均 `0 MiB / 0%`；主机内存 available 约 `940 GiB`。新任务仍须重新刷新。
- 原 RLinf 工作目录 `/root/autodl-tmp/RLinf` 未被本迁移修改，基线 HEAD 为 `6d0db56bf26f972cd27fa29535f5eb939e80e5bf`。
- Fast-WAM 官方仓库 `/root/autodl-tmp/FastWAM` 固定在 `45d8e1458921d83f8ad6cf9ce993d371208dabd0`；官方 `adjust_bottle` 单 episode 已成功。
- 独立服务器 worktree：`/root/autodl-tmp/RLinf_fastwam_rlinf`，分支 `feat/fastwam-robotwin-grpo`，仍以 RLinf pin 为 HEAD；18 个目标文件处于未提交工作区，包含 15 个新增、3 个既有文件小改。用户此前说 git 暂不处理，因此没有提交。
- 本地镜像 worktree：`C:\Users\86136\Documents\rl\.rlinf-fastwam-worktree`，分支 `codex/fastwam-robotwin-grpo`。
- 联合环境：`/root/autodl-tmp/conda/envs/FastWAM-RLinf`，Python 3.11、Torch 2.7.1+cu128；Fast-WAM、RLinf 为 editable，CuRobo v0.7.8 在最终 CUDA 12.8 环境重编译。
- 现役 π0 venv 未改；源与备份均约 14 GiB，完整备份为 `/root/autodl-tmp/backups/RLinf-pi0-venv-golden-20260717`，现场复跑 `rsync -aHn --delete` 输出 0 行。marker 与清单位于 `/root/autodl-tmp/fastwam-rlinf-setup/pi0-env-before-fastwam-20260717`。
- joint venv 已通过 `pip check`、RoboTwin render、Warp/CuRobo import 和真实 `adjust_bottle.play_once()`；2026-07-18 再次用数据盘联合环境的绝对 Python 路径验证两份 Hydra 配置、launcher 语法和资源监控自测。启动器不依赖 Conda base。

## 2026-07-18 首次 P2 smoke 现场结果

- run：`/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260718_004457-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke`。两卡 rollout/actor 模型均成功构建，但尚未进入 RoboTwin rollout、loss、backward、optimizer 或 DCP。
- 首个致命错误发生在初始 actor→rollout 权重广播：GPU bucket 进入 CUDA IPC 后，`torch.storage._new_shared_cuda()` 报 `pidfd_getfd: Operation not permitted`。这是当前 AutoDL 容器的 IPC 传输限制，不是 Fast-WAM checkpoint、模型前向或 OOM。
- 已在本地镜像和服务器独立 worktree 的 smoke/formal YAML 中把 `bucket_device` 从 `cuda` 改为 `cpu`；bucket 算法、同步集合、dtype、512 MiB 分桶和 `load_instant` 不变。服务器 Hydra compose 已确认两份配置均解析为 `bucket/cpu/536870912`。尚未代用户重启 smoke。
- 首次 run 的 226 个资源采样覆盖 00:45:04–00:53:06：无 cgroup OOM/OOM kill；模型就绪后显存约 GPU0 `59.06 GiB`、GPU1 `59.58 GiB`，约为同机 π0 GRPO 首步峰值的 `2.05×`。cgroup RAM 峰值 `237.259 GiB/245.760 GiB`，但起跑基线已 `158.401 GiB`，现场 `memory.stat` 又显示约 `160 GiB` 为可回收 file cache，故不能把 cgroup 总量全部解释为模型常驻内存。
- 该 run 没有进入真实 rollout/backward，因此没有 Fast-WAM 的实测 action 推理或一步训练耗时，也不能据此放行 formal。失败进程在 00:57 仍挂起并占用约 59/60 GiB 显存；按项目规则 Codex 未擅自 kill，下一步应先由用户停止旧 run，再重跑同一条 smoke 命令。

## 已完成的代码与数值验收

- 完整改动面：内置 `rlinf/models/embodiment/fastwam/` 六个文件、model/smoke/formal 三份 YAML、四份测试、专用 launcher 和通用资源监控；只小改 model registry、HF rollout worker mode capability 和 FSDP2 `cast_forward_inputs` 默认兼容开关。
- 原 π0/PPO/GRPO 路径默认行为不变：新 registry 为 lazy，worker capability 仅 Fast-WAM opt-in，FSDP 开关默认仍为 `true`。
- 真实 checkpoint 通过 public RLinf registry 加载：总参数 `12,406,348,522`；只有 canonical action expert 可训练，`824` 个 parameter tensors、`1,020,900,366` 参数。
- official B=1 max abs=`0.0`；B=2 permutation max/mean abs=`0.0/0.0`；rollout old 与 actor replay new logprob max/mean abs=`0.0/0.0`。
- `model_forward_batch_size=2` 对两次 B=1 的 max/mean abs=`0.008000642/0.000861770`，reference max=`0.508447`；判定为不同 batch shape 的 BF16 kernel 漂移。生产 batch2 保留，不退化为逐样本 fallback。
- train initial latent 逐样本不相同；共享 denoise `k` 只在 logical batch 内共享。固定 eval singleton latent 没有进入训练路径，因此不是训练萎缩。
- model-level probe 峰值 CUDA `23.8102 GiB`；这不是 FSDP runner 资源峰值。
- joint venv 最终结果：`50 passed, 1 warning`；Ruff、compileall、`git diff --check` 全通过。

第一次真实 replay probe 的 CPU/CUDA 报错来自测试直接把 CPU rollout dict 交给 policy，跳过了正式 `FSDPActor.train_micro_batch()` 的 `put_tensor_device` 边界。生产代码没有增加隐式搬运；修正 probe 后 old/new logprob 完全一致。

## 冻结设计

- RoboTwin absolute qpos：三相机、14D state/action、H=32、执行 N=24、S=10、192 env steps。
- 官方 Fast-WAM scheduler 决定 raw timestep/signed delta/shift=5；RLinf OpenPI 决定 Flow-SDE mean/std 与首步分母。
- 每个 logical model batch 共享均匀 `k∈[0,9]`；每条 trajectory 的 initial latent/SDE epsilon 独立；完整 chain `[B,11,32,14]` 保存为 FP32。
- eval seed0 singleton latent broadcast 只用于与官方 B=1 oracle 等价；train 使用逐样本独立随机性。
- 只训练 action expert；video expert、proprio encoder、VAE、T5 冻结。
- P2最初设计为16 train env、group8、rollout epoch1；真实资源迭代后已被下面“当前权威增量”的resolved配置覆盖。模型/概率定义未随资源调整而萎缩。
- 当前formal为4并发env×rollout epoch16=64 trajectories、group4/16 groups、512 actor transitions、global128/micro2、noise0.3、lr5e-6、100 steps、每10步DCP；rollout/env offload开启，CPU-staged bucket 512 MiB。
- `run_fastwam_robotwin_grpo.sh` 会同步启动只读资源监控；同一 run 目录保存 `resources.csv`、`peak.txt`、`monitor.log`、PID、命令和 resolved config。监控格式与旧 π0 PPO/GRPO 的 22 列资源 CSV 对齐。

## 尚未完成，必须诚实保留

首次 runner 只走到模型构建和初始同步，以下仍未完成：

1. 默认 P2：fresh `lr=1e-6`、1 global step、保存 DCP；检查首次 ratio/KL/clip、finite action-only grad、资源峰值和 checkpoint。需要纯诊断时可另开 fresh 进程覆盖 `actor.optim.lr=0`，但不再要求用户先跑两条复杂命令。
2. 一步 smoke 只验证初始 actor→rollout 同步；更新后权重同步必须由第二步或 resume P3 证明。
3. P3：从 `global_step_1` fresh resume，绝对 `max_steps=2`，再跑一步；随后 strict export 官方 `.pt`，比较恢复 RLinf action 与导出后官方 loader action。

可直接执行的完整命令在：

`docs/fastwam-robotwin-rlinf-grpo/evidence/IMPLEMENTATION_LOG_20260717.md` 第 7 节。

## 证据路径

- 实施日志：`C:\Users\86136\Documents\rl\docs\fastwam-robotwin-rlinf-grpo\evidence\IMPLEMENTATION_LOG_20260717.md`
- 服务器环境与检查：`/root/autodl-tmp/fastwam-rlinf-setup`
- resolved config：`/root/autodl-tmp/fastwam-rlinf-setup/resolved_smoke.yaml`
- 真实模型 probe：`real_fastwam_rlinf_probe.py` / `real_fastwam_rlinf_probe.log`
- 最终静态检查：`final_static_checks_20260717.log`

## SSH 续接规则

- 入口：`root@connect.bjb1.seetacloud.com:36406`。
- 用户给出密码时，批量终端用 Paramiko 当前进程密码认证，`look_for_keys=False`、`allow_agent=False`；先做 `hostname; pwd; id -u` 探针。不能把 OpenSSH `BatchMode=yes` 无法回答密码提示误判为服务器不可达。
- 密码不写入项目、日志、Memory 或 Git。新任务默认只读；本轮写权限不自动延续。

## 下一任务可直接使用的首句

请按 AGENTS.md 启动，读取 PROJECT_CONTEXT.md、HANDOFF.md、权威交接和 Fast-WAM 实施日志；先只读刷新 `/root/autodl-tmp/RLinf_fastwam_rlinf`、联合环境、GPU/进程及当前GRPO/PPO验收状态，再继续 Fast-WAM × RoboTwin × RLinf GRPO/PPO 工作。

## 2026-07-18 当前权威增量（覆盖上面的首次失败状态）

- 后续一步smoke已经完整通过；正式100-step run `logs/20260718_020324-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu` 正在运行。
- 09:12 CST 现场已完成step24、step25进行中；无fatal/OOM。总train rollout success约97.14%，5/24步全成功并零梯度；GPU峰63036/62511 MiB，cgroup RAM峰98.44%，OOM计数0。
- 当前曲线不是fixed deterministic eval。Fast-WAM论文的`adjust_bottle`为clean/randomized 100/100，首要问题是任务饱和和评估不可辨识，不是已证实的迁移代码错误。
- 逐层实现、配方对比、域随机化和换任务审计集中在`05_IMPLEMENTATION_PLAN.md`第27节；用户已选择seed就绪的`move_stapler_pad` clean，并拍板保持192步以做单变量任务对照。
- 当前服务器resolved formal配置是4 env×rollout_epoch16、group4、64 trajectories、noise0.3、lr5e-6、rollout offload开启、CPU bucket。保存Git/ZIP前必须以服务器YAML为准同步本地旧镜像。
- 服务器已新增`robotwin_move_stapler_pad` env与formal配置：相对当前adjust formal只改任务env引用和实验名，192步、4 env×16 epoch=64 trajectories、group4、noise0.3、lr5e-6及offload/sync参数均不变；launcher新增可选第二参数config-name且保持旧调用兼容。Hydra compose/数量关系检查通过。用户将自行停止旧run，下一条命令为`bash examples/embodiment/run_fastwam_robotwin_grpo.sh train robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu`。no-std GRPO、LR、randomized和proprio均作为后续单变量。

## 2026-07-19 当前权威增量（覆盖旧的“PPO尚未实现”状态）

- 11:01 CST只读现场：现役`move_stapler_pad`旧formal训练PID 915681仍存活，最新完整日志step为75/100；本轮同步新代码/配置与静态测试没有停止或重启它。动态step仍需新任务现场刷新。
- 服务器与本地镜像已新增GRPO π0对齐配置`robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu_pi0_aligned.yaml`。相对旧formal严格只改实验名、rollout epoch 16→32、global batch 128→512、update epoch 1→2；旧配置保留。新配置为128 trajectories、1024 transitions、4 optimizer updates、2048样本呈现。
- Fast-WAM PPO首版已经实现：PPO smoke/formal YAML、独立launcher、默认关闭的model head schema、builder/policy/core/export及集中测试。PPO从官方release base冷启动，不从GRPO checkpoint暖启。2026-07-19后续拍板把formal改为`4×32/global512/update2`，与新GRPO同为1024 transitions、4次optimizer update和2048次样本呈现；smoke改为`4×1/global32/update1`，保持4路并发但只做一次顺序采样和一次真实更新。
- critic固定为observation-side：官方最后一层`video_kv_cache[-1]["v"] [B,120,3072]`池化，`ValueHead 3072→1024→512→256→1`；feature不读action/k/SDE noise并detach。head参数跟随BF16模型，value输出转FP32；rollout old value、actor replay new value和bootstrap共用同一conditioning路径。
- 通用HF worker、EnvWorker、GAE/PPO loss、FSDP optimizer分组、bucket sync和DCP未新增PPO分支；依靠顶层`value_head`进入独立`value_lr`、同步和DCP。官方deploy export排除head，训练DCP保留。
- 最新现场验收：PPO预算调整已同步服务器；Hydra resolved断言smoke为4 trajectories/32 transitions/global32/update1、formal为128 trajectories/1024 transitions/global512/update2，formal与新GRPO同为4次optimizer update和2048次样本呈现。配置合同`7 passed`（含PPO↔GRPO与smoke↔formal严格diff），Fast-WAM集中测试`61 passed, 1 warning`；唯一warning仍是预期的单进程DCP提示。现役GRPO PID 915681在验证后仍存活，未停止或重启。
- 尚未执行真实两卡PPO smoke、head初始/更新后同步和DCP resume。现役训练结束并确认无其他driver后，可先运行：`bash examples/embodiment/run_fastwam_robotwin_ppo.sh smoke`。PPO formal默认命令为`bash examples/embodiment/run_fastwam_robotwin_ppo.sh train`；必须在smoke真实验证后再启动。
- 新GRPO配置的未来启动命令：`bash examples/embodiment/run_fastwam_robotwin_grpo.sh train robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu_pi0_aligned`；同样不得与现役driver并行启动。
- 详细逐文件依据、dtype修正和测试证据见`05_IMPLEMENTATION_PLAN.md`第28节与实施日志第12节。

## 2026-07-19 Git发布与PPO formal增量

- 服务器Fast-WAM集成worktree已配置repo-local作者`Yutenji-Nyamu <1842710211@qq.com>`；官方`origin=https://github.com/RLinf/RLinf.git`保留，新增`personal=https://github.com/Yutenji-Nyamu/rlinf_fastwam.git`。
- 已创建公开仓库`https://github.com/Yutenji-Nyamu/rlinf_fastwam`，默认分支为`main`。主体集成commit为`768e0243e4dafedea6c92b3f37b652c51efb5a2e`；Git发布流水账补充commit为`8138d6700e3838250c1139289ebfba43d48ff7de`，服务器分支`feat/fastwam-robotwin-grpo`跟踪`personal/main`且现场工作区干净。
- 上云范围为Fast-WAM adapter、三项RLinf opt-in兼容改动、model/env/GRPO/PPO配置、launcher、资源监控和集中测试；checkpoint、logs、权重、环境、cache、RoboTwin assets及官方Fast-WAM源码均未纳入。官方Fast-WAM仍作为锁定在`45d8e1458921d83f8ad6cf9ce993d371208dabd0`的外部editable依赖。
- Git操作流水账同时保存在服务器仓库和本地镜像的`docs/fastwam_rlinf_git_publication_20260719.md`。首次staged检查发现Windows CRLF与三处YAML行尾空格，机械规范化后`git diff --check`通过；PPO launcher执行位在首次push前由`100644`修正为`100755`并amend，错误版本未上传。27个主体文件共6348行新增/2行删除；无超过5 MiB文件、无常见凭据模式，两个launcher均通过`bash -n`。
- PPO formal run为`logs/20260719_124315-robotwin_move_stapler_pad_ppo_fastwam_a800_2gpu`。14:05 CST只读复核：driver PID 981955与monitor PID 981956存活，首个global step尚在rollout 26/32；fatal扫描为空，cgroup OOM/OOM-kill均0；当前GPU0/1约45880/39415 MiB，峰值62813/62289 MiB；当前cgroup RAM 74.82%，峰值98.33%。Git操作未停止、重启或改变训练配置值。

## 2026-07-27 DSRL / RLT / QAM 只读调查增量

- 21:59–22:01 CST 现场：没有 RLinf driver、Ray worker 或训练任务；两张 A800-SXM4-80GB 均 `0 MiB / 0%`。主机内存 available 约 `979 GiB`；当前 cgroup `memory.current` 约 `398 MiB`，OOM/OOM-kill 计数为 0。`/root/autodl-tmp` 剩余约 `851 GiB`。这是带时间戳快照，新任务仍需现场刷新。
- `/root/autodl-tmp/RLinf` 当前为 `local/openpi-a800-2gpu-migration@6d0db56bf26f972cd27fa29535f5eb939e80e5bf`；只有 4 份本地 A800 配置和 `local_scripts/` 未跟踪。本轮没有修改工作树。
- 当前 pin 已包含真实 DSRL 主体：π0/π0.5 OpenPI hook、LIBERO 同步/异步配置、Gaussian latent-noise actor、轻量 image/state encoder、10-Q critic、SAC replay/target worker、文档与 E2E 定义。关键提交为 `42530b72`、`41dcdf03`、`72d1e0de`。
- 当前 pin 也包含两阶段 RLT / RLToken 主体：RL-token encoder/decoder、Stage 1 π0.5 SFT 配置、Stage 2 MLP actor-critic、route/transition/replay 与 worker。关键提交为 `5769c6eb`、`3d93750d`、`828b1af1`。现有模板是 π0.5 + ManiSkill/真机，不是 π0 + RoboTwin。
- 对当前源码、镜像、refs 和 commit subject 的精确检索均未发现 QAM。这里的 QAM 是 *Q-learning with Adjoint Matching*；官方代码为 JAX/Flax + OGBench，不能冒充已在 RLinf 中实现。
- 历史产物仍在：π0 GRPO checkpoints 到 100；Fast-WAM GRPO checkpoints 到 70；Fast-WAM PPO checkpoints 到 30。PPO 日志在 step 34 记录 Raylet 意外退出；当前无 cgroup OOM 证据，根因仍未解决。
- 新建本主题唯一索引与计划：`docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md`。它登记七份用户历史文档的路径/hash/敏感标记，分别摘要用途与风险，并给出 DSRL → RLT → QAM 的分阶段路线。Fast-WAM 事实仍只链接既有 `00_INDEX.md` / `05_IMPLEMENTATION_PLAN.md`，没有复制第二套计划。
- 推荐先做 DSRL × π0 × RoboTwin：先保留现有 LIBERO DSRL oracle，再做 14D state、三相机 base-policy、latent-noise、chunk reward/replay 与 sync/DCP 的窄适配。RLT 排第二，先解决 π0.5→π0 token contract、Stage 1 数据和 RoboTwin route；QAM 作为 JAX 复现→PyTorch 数值核→SAC-Flow 桥梁→π0 的独立研究线。
- 本轮只做本机文件整理、服务器只读检查和公开材料核验；没有在服务器写代码、安装、下载、启动、停止或删除任何内容。下一次服务器写操作仍需用户明确授权。

## 2026-07-27 22:57 DSRL × π0 × RoboTwin 逐调用链规划增量

- 22:46–22:57 CST 再次只读现场：无 RLinf/Ray/RoboTwin 训练进程；两张 A800-SXM4-80GB 均 `0/81920 MiB、0%`。主机 available RAM 约 979 GiB；cgroup 上限 240 GiB、`memory.high=236 GiB`，OOM/OOM-kill 为 0；`/root/autodl-tmp` 剩余约 851 GiB。
- `/root/autodl-tmp/RLinf` 仍为 `local/openpi-a800-2gpu-migration@6d0db56bf26f972cd27fa29535f5eb939e80e5bf`，只有既有 4 份 A800 config 与 `local_scripts/` 未跟踪。当前 venv 为 Python 3.11.14、Torch 2.6.0+cu124、Ray 2.55.1；DSRL/SAC/RoboTwin imports 成功。历史 `pip check` 不一致仍在，不为本任务顺手升级。
- 透明说明：本轮 import probe 可能在 `.venv` 生成少量 OpenGL 等包的 `__pycache__/*.pyc` 与目录 mtime；排除 `__pycache__`/目录 mtime 后，现役 venv 与 golden backup 无实体文件差异。没有改源码、配置、依赖或权重，也未擅自清理；后续探针应加 `PYTHONDONTWRITEBYTECODE=1`。
- 已现场核验 RoboTwin SFT：`/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle` 约 7.6 GiB，两份 safetensors shard 完整；内部 model dim 32、horizon 50，RoboTwin norm stats 为 32D padding且前14D有效。它与 RoboTwin assets/seeds/launcher/monitor可直接复用。
- π0 PPO/GRPO DCP 不能直接作为 DSRL resume：没有 DSRL Gaussian actor、10-Q、target、temperature 和 replay。若以后做“GRPO-refined base + DSRL”，必须列为独立 warm-start 条件。
- LIBERO package、LIBERO π0 SFT 和历史 DSRL run 当前均不存在；不阻塞 RoboTwin 静态实现，但运行 LIBERO oracle需要另行安装/下载授权。
- 已确认新 YAML 不足以得到正确训练，至少有三个必修点：
  1. RoboTwin success reward 在 chunk 最后一格，当前 SAC 固定读第一格，直接拼接会让 critic看不到成功；
  2. 旧 `sample_window_size=15000` 按 trajectory cache slot 预分配，并重复存原始三相机/forward payload，历史 envelope 下可达约 14.48 TiB/actor rank；
  3. resume 在 target load 后不刷新 FP32 shadow，第一次 EMA 会用 fresh shadow 覆盖已恢复 target。
- 第一主体批次确定为：新 RoboTwin DSRL config + 配置化 macro reward/discount/mask + lean transition replay/capacity/readiness + target-shadow/update-step resume + 集中 contract tests。首版保持 frozen π0 三相机、DSRL小策略单主相机、一个32D latent repeat到H=50；不改 runner、denoise core、Gaussian actor 参数化或 RoboTwin base transforms。
- 执行 chunk主线暂推荐 `N=50`，保持已验证 RoboTwin控制频率且符合官方 π0 DSRL Aloha `query_freq=50`；`N=5/10` 是固定SFT/seeds对照。state/env action为14、latent/model dim为32、model horizon为50，三者不得混改。
- DSRL是 warm-up 后持续 collect→replay→SAC update 的在线 off-policy算法，不是一次性离线收集。Aloha-like主线 warm-up暂设1000个全局 query transitions；现有 `min_buffer_size=10` 的 trajectory-object语义必须改为 sample计数。
- 参数存在有依据的冲突，已显式拆分：RLinf/LIBERO parity 用 target entropy -16、latent magnitude1.0；官方Aloha-like配方用0.0、2.0。UTD必须按“gradient batches/new query transition”计算；不能把 `update_epoch=200` 或 `batch×updates/data`直接冒充官方UTD20。
- 唯一计划已扩展为逐调用链规格、逐文件 change matrix、参数来源、三层验收和资产复用表：`docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md` 第11节。服务器仍未写代码、未安装、未下载、未启动/停止训练；实施仍需用户明确授权。

## 2026-07-28 DSRL × π0 × RoboTwin 首版设计收敛增量

- 本轮完整复核用户问题清单、本地 `.research-rlinf` 锁定源码、官方 `swissai-dsrl` 可执行配置和 2026-07-27 服务器证据；没有连接或刷新服务器，因此没有把旧 GPU/进程/RAM 状态称为当前，也没有服务器写操作。
- 首版来源规则已固定：算法高层跟官方 π0 + LIBERO DSRL；模型、环境和系统合同跟现有 π0 + RoboTwin。固定 `state/env action=14`、latent `d=32`、model horizon `H=50`、execution `N=50`、三相机 frozen π0、单主相机 64×64 DSRL actor/critic、RoboTwin normalization 和 denoise steps 4。
- 不再要求 implementation 前跑 `N=5/10/50` gate。`N=50` 每个 200-step episode 只有 4 个 macro decisions，风险是 credit 粗和 reset 多，但不是错误；若可信主线 plateau，第一消融改为官方 LIBERO 的 `N=20`。
- reward/TD 语义唯一化：在 SAC/replay projection 边界生成 `success→0`、其他 `-1`，成功 termination 不 bootstrap，time-limit truncation bootstrap，discount 固定 `0.999**50`。不修改 `robotwin_env.py`，不再为 smoke 训练 env-native reward，也不需要 `effective_horizon/steps_executed`。
- 官方 LIBERO 配方核定为：标准高斯 latent warm-up 500 个全局有效 macro transitions；之后才切 learned tanh actor；global batch 256；每个新有效 macro 做 20 个 optimizer updates；target entropy -16、magnitude 1.0、Q10 mean、actor/critic/temp LR `1e-4/3e-4/3e-4`、initial alpha 1、gamma .999、tau .005。当前 RLinf 的 `min_buffer_size=10` trajectory objects、固定 `update_epoch=200` 和未训练 actor warm-up 都不是官方 parity。
- replay 不复用旧 `sample_window_size=15000`。第一批在 legacy buffer 旁增加 DSRL opt-in transition ring：全局 capacity 25k、两 actor ranks 各 12.5k；只存 curr/next 64×64 主相机、14D state、canonical 32D latent、macro reward、termination/truncation/discount。预计约 0.62 GB 全局；保存/恢复 ring cursor、resident/total、RNG、schema 和 world-size，resume 首版拒绝静默 world-size 变化。
- 采集系统默认 4 env×`rollout_epoch=1`；满长每轮新增 `D=16`，动态更新 `U=20D=320`。4-env profile 通过后才试 8 env；首版不追求 16/32。global batch 256、micro 64，只有资源证据不通过才降 micro 32。
- 交互公平主预算建议 10k total macros（约 500k requested primitive steps、2500 满长 episodes；warm-up 后约 190k updates），milestones 为 500/1k/2.5k；25k total macros 仅作接近官方 500k-update 的扩展。曲线同时报告 episode、valid macros、requested/actual primitive steps、optimizer updates、GPU-hours，不能用 runner global step 代替。
- scientific parity eval 改为 fixed-RNG stochastic actor；deterministic mean 单列部署诊断。官方学习后 control eval 仍调用 stochastic sampling。
- resume 缺陷处理升级为精确方案：FP32 shadow/EMA 只覆盖 target Q 实际读取的 critic image encoder、state encoder 和 Q head；新 checkpoint 持久化 shadow 与 `update_step`。旧 checkpoint 缺 shadow 时才从 loaded BF16 target 重建并明确告警，不能把这种兼容恢复称为 bitwise 连续。
- 唯一计划 `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md` 已回写所有收敛值、调用/数据流、逐文件改动和验收。首批预计修改新 RoboTwin DSRL YAML、OpenPI action model、SAC worker、replay buffer/adapter和集中测试；不改 runner 主链、RoboTwin env、Gaussian 参数化或 base transforms。真正服务器实施仍需用户明确授权，并在开始前重新只读刷新现场。

## 2026-07-28 DSRL 设计优先重构增量（覆盖上一个同日方案）

- 用户要求减少验收/协议膨胀，优先解决设计问题，并保留而不删除历史材料。旧 860 行计划已完整移动为非规范性历史参考：`docs/rlinf-robotwin-pi0-traditional-rl/01_FULL_REFERENCE_HISTORY_20260728.md`；精简的当前唯一计划仍为 `00_INDEX_AND_IMPLEMENTATION_PLAN.md`，只讨论 DSRL × π0 × RoboTwin。
- 修正 execution chunk 结论：model horizon `H=50` 固定，但正式训练的 `N` 不再先验冻结。DSRL 锁定的 OpenPI base 是 LIBERO `N=5` / Aloha-sim `N=10`，官方 DSRL 实验则是 LIBERO `N=20` / Aloha `N=50`，作者没有给出文字 rationale；现有 RoboTwin π0 为 `N=50`。实现和工程 smoke 先保持 50；正式 warm-up 前只做同 checkpoint/seeds 的冻结-base 20 vs 50 对照，再按 base 成功率、恢复能力、query/墙钟代价和反馈粒度暂停让用户冻结。
- 官方 LIBERO DSRL 是在线 off-policy，不是先收离线数据再训练：标准 Gaussian latent warm-up 500 个全局有效 macro transitions；之后循环采集与 SAC 更新；每条新 transition 做 20 次 optimizer updates，global batch 256。`max_steps=500000` 是 optimizer updates，理想满程约 25.5k macros、约 510k train primitive steps，不是 500k 环境步。
- 第一版并发保持 4 env×`rollout_epoch=1`，不照抄 PPO/GRPO 的 16×16 大批 on-policy 结构。N=50 时每轮最多 16 macros/320 updates；N=20 时每轮最多 40 macros/800 updates。global batch 256、micro 64；只有现场显存证据不通过才降 micro 32。
- reward 仍在 SAC/replay projection 边界统一为 success→0、其他→-1；success termination 不 bootstrap，time-limit truncation bootstrap，discount 为 `gamma` 的 N 次方。N=20/50 均整除 200，首版不增加 effective horizon。
- replay 主线改为 DSRL opt-in flat transition ring：global capacity 25k，只存 small actor/critic 所需的 curr/next 主相机 64×64、14D state、canonical 32D latent、reward/continuation/discount 等紧凑字段；legacy trajectory buffer 不变。
- resume 缺陷是 RLinf DSRL worker 通用问题，LIBERO 同样受影响，不是 RoboTwin 特有。复核实际 checkpoint 结构后采用完整但窄的首版：shadow 只覆盖 critic image/state encoder 和 Q head；新 checkpoint 保存/恢复这部分 FP32 shadow、`update_step`、Gaussian/learned phase 和 replay trainer state。旧 checkpoint 缺 shadow 时才从 loaded BF16 target 重建并打印非 bitwise 兼容恢复告警。target model 大结构和冻结 π0 不重写。
- 最小主体预计只改四处：新增 RoboTwin DSRL YAML；修改 OpenPI action model、SAC worker、replay buffer。`embodied_buffer_dataset.py` 若保持 `sample/is_ready` 接口则不改；runner、RoboTwin env、Gaussian policy、compact encoder 和 base transforms 保持。
- 集中检查收缩为三项：fixed observation/latent 的 frozen-base path；一条投影 transition + 一次真实 SAC update；replay save/load + resume 后 target update。实现开始后另建 `evidence/IMPLEMENTATION_LOG.md` 记录全部文件改动、命令、结果、问题、修复和资源峰值。
- 任何服务器写入、N 对照、smoke 或训练前都必须先只读刷新现场，并向用户展示完整有效配置、精确命令、输出目录、资源预期和停止条件，等待明确批准。smoke 必须记录 GPU/RAM 峰值和 OOM 计数。本轮没有刷新或改动服务器，没有运行 smoke。
