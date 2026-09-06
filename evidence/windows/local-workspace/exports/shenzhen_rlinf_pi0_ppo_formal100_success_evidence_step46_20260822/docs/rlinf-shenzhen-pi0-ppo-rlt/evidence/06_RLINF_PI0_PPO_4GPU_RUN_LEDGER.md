# 深圳 latest RLinf：4×H100 official-half π0 PPO 运行流水账

## 0. 边界与当前状态

- 机器/账号：`SZ-H100` / `chenyiteng`；固定 host-key 的低层 Paramiko 密码认证，凭据只注入当前
  进程，不写入脚本或账本。
- source lock：RLinf `7d07a4212ee6858cc333e1d4fab7a37256d1f839`；RoboTwin
  `RLinf_support@0008ae6800df9f75fc8de7098bacb01735fd8fd2`；π0 SFT
  `92684e50dca1a5f75adc8d332046c4cf4fa7a3d0`。
- 本账从原 fixed-8 草案改为物理 GPU 4、5、6、7 后开始；4 卡的唯一当前执行包是
  [`../10_RLINF_PI0_PPO_4GPU_OFFICIAL_HALF_EXECUTION_PACKET.md`](../10_RLINF_PI0_PPO_4GPU_OFFICIAL_HALF_EXECUTION_PACKET.md)。
- 截至 2026-08-22：Stage A SFT fixed-64 与 Stage B PPO one-step + inline fixed-64 均自然完成并
  退出 0；Stage B 已闭合精确一次 distributed optimizer step 和 `global_step_1`。Stage C 已由全新
  Python/Ray 生命周期成功 reload 该权重并完成 64 条 rollout，但按用户要求在指标汇总前对 exact owned
  driver 发出 SIGINT；因此它是**授权终止的 partial run**，既不是自发运行失败，也不是完成的评估。

## 1. 固定路径

```text
RLinf worktree
  /data/chenyiteng/projects/rlinf-shenzhen/worktrees/ppo-pi0-robotwin
RoboTwin compatibility tree
  /data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support
venv
  /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin
π0 SFT
  /data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50
resolved packet
  /data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821/resolved-packet-4gpu-official-half-v1
Stage A
  /data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed64-4gpu4567-official-half-v1
Stage B (completed)
  /data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1
Stage C (authorized stop before metric aggregation)
  /data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-reload-fixed64-4gpu4567-v1
```

## 2. 操作记录

### RL-SZ-R2-003 — 4 卡 placement 语义核对与最终决定

- 用户指定物理 H100 `4,5,6,7`，随后明确：官方为 8×80GB，本次 4×80GB 保持每卡负载，train/eval
  总并发各减半，直接做一次高信息量 smoke，不从很小并发防御性爬升。
- locked RLinf source 显示 worker 将 placement ID 直接写入自己的 `CUDA_VISIBLE_DEVICES`，并设置
  `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` 阻止 Ray 再映射。因此本裸机的正确合同是：

  ```text
  driver CUDA_VISIBLE_DEVICES = unset
  Ray/node driver sees        = physical GPU 0..7
  cluster placement           = physical GPU 4-7
  eval key                    = {env, rollout: 4-7}
  PPO key                     = {actor, env, rollout: 4-7}
  ```

- 不能使用常见的“外层 mask `4,5,6,7` + placement `0-3`”：在这版 RLinf 的 literal placement 与
  Ray override 组合下，它不能证明 worker 会落到物理 4–7，存在误用物理 0–3 的风险。

### RL-SZ-R2-004 — fixed-4 三次 compose-only 更正轨迹（均未运行）

这三次只处理 Hydra compose/placement/预算，没有启动 Ray、simulator、模型加载、GPU workload 或训练：

1. 第一版把包含逗号和空格的 placement dictionary 当成普通引号字符串传给 Hydra，override grammar
   解析失败；失败发生在 compose 阶段。
2. 第二版改为 Hydra 可解析的转义写法，compose 可过，但仍沿用外层 GPU mask + logical `0-3` 的常见
   映射。source audit 证明它不符合 locked RLinf 的 literal placement 合同，因此未执行。
3. 第三版取消 driver mask、直接 placement 物理 `4-7`，placement 语义正确；但它仍是 fixed-4 的极小
   预算。用户随后明确要求按 official 8 卡配置把总并发减半、保持每卡负载，故该版在真实运行前被
   official-half 包取代。

结论：fixed-4 是被纠正并冻结的 cold evidence，不是失败的模型/环境实验；三个 fixed-4 run root 均未被
用于真实运行，也没有据此判断性能或成功率。

### RL-SZ-R2-005 — official-half packet compose 与哈希闭合

- command file：
  `local_scripts/remote_commands/shenzhen_rlinf_r2_compose_four_gpu_official_half_oneopt_20260821.sh`
- command-file SHA256：
  `6501eb3c14b0bb677196f51a02ae75a2a0fb4184244d77ba4b7c995b5479791f`
- exit 0；marker：`R2_FOUR_GPU_OFFICIAL_HALF_PACKET_COMPOSE_OK`。
- official 8 卡到本次 4 卡的主要 resolved diff：train env `256→128`、eval env `128→64`；每卡仍为
  32 train / 16 eval env；actor micro batch 保持 32。smoke 时间轴为 rollout epoch 1、50 train
  steps、update epoch 1、global batch 128。
- 精确 one-step 记账：`128 env × 1 × 50/50 = 128 records`；4 ranks 各 32 records，恰好一个
  micro batch；`128/global_batch128 × update_epoch1 = 1` 次 distributed optimizer step。
- resolved artifacts：

  | 文件 | SHA256 |
  |---|---|
  | `sft-fixed64.resolved.yaml` | `b52e45cd42702f8c92937dc7e57c8f19dcbe855830f6cd0017a4d7d2b6618915` |
  | `ppo-oneopt.resolved.yaml` | `3a0967a4b7592cf068956febcf2c273c79c50c1cef20d068588dbc5d2d8a931d` |
  | `ppo-reload-fixed64.resolved.yaml` | `71da969514116ef528d1c4fc05f24effaf0fd2dcb0d0930f69454edb41e8649a` |
  | `budget-and-seeds.json` | `c8791f808f904e67367b6ea2b0e5a0a220d630919fd1634f864acba72d1d3689` |

- `budget-and-seeds.json` 保存全部 64 个 fixed reset IDs；A、B inline eval、C 必须复用同一列表。

### RL-SZ-R2-006 — SSH 认证前短暂网络超时

- official-half compose 的首次固定路线连接在 SSH 认证前连续命中 3 次 bounded pre-auth timeout；执行器
  按既定 1/3 秒退避达到上限后停止，随后 TCP/22 短时也不可达。
- 约 30 秒后 TCP/22 恢复；仍使用完全相同的 host、port、用户、固定 host-key 与密码 Paramiko 路线
  重试并成功。
- 没有改用 OpenSSH、agent/key、其他账号、忽略 host-key 或放宽认证条件；第一次失败发生在远端命令
  发出前，因此没有 compose 或其他副作用被重复执行。

### RL-SZ-R2-007 — 真实启动前 live preflight

- command file：
  `local_scripts/remote_commands/shenzhen_rlinf_r2_four_gpu_live_preflight_20260821.sh`
- SHA256：`0582ed02fdfaf84384e616d72676f59977b4719a3c060e3c12b734b569c50fbe`。
- exit 0；marker：`R2_FOUR_GPU_LIVE_PREFLIGHT_OK`。
- 现场断言：driver `CUDA_VISIBLE_DEVICES=None`；Ray NVIDIA manager 报 node GPU count 8、driver visible
  IDs `None`；8 卡均空闲，物理 4–7 无 compute app，无既有 `raylet`/`gcs_server`；A/B/C 三个目标
  output 均不存在。
- `/home` 与 `/data` 仍有 TiB 级可用空间；本次环境、结果和视频均留在 `/home/chenyiteng` 或
  `/data/chenyiteng`，未向 Windows C: 下载大运行产物。

### RL-SZ-R2-008 — Stage A：official π0 SFT fixed-64 真实评估

#### 启动与精确命令

- 本地 command file：
  `local_scripts/remote_commands/shenzhen_rlinf_r2_run_sft_fixed64_gpu4567_v1.sh`
- SHA256：`45316e3b58c5d6076061ae30b2fa45969fca921bbc9cadd3b0dfb861f00a3bc1`。
- 本地执行入口（密码由 no-echo prompt 注入，不写在命令中）：

  ```powershell
  C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
    local_scripts/verified_password_ssh.py `
    --host 120.241.223.9 --port 22 --user chenyiteng `
    --host-key-sha256 qw92OXne52y6NsQMQ6+PdjOu0Hosc4ZX78XmkYxcuEY `
    run --command-file local_scripts/remote_commands/shenzhen_rlinf_r2_run_sft_fixed64_gpu4567_v1.sh
  ```

- command file 先断言 source/model locks、目标 output 不存在、物理 4–7 无 compute process；随后
  `unset CUDA_VISIBLE_DEVICES`，并设置 `ROBOTWIN_PATH`、`ROBOT_PLATFORM=ALOHA`、`PYTHONPATH` 与
  OpenPI cache。主命令为：

  ```bash
  /usr/bin/time -v timeout --signal=INT --kill-after=120s 3600s \
    "$VENV/bin/python" "$ROOT/evaluations/eval_embodied_agent.py" \
    --config-path "$ROOT/evaluations/robotwin" \
    --config-name robotwin_adjust_bottle_openpi_eval \
    'cluster.component_placement={env\,\ rollout:4-7}' \
    "runner.logger.log_path=$RUN" \
    "rollout.model.model_path=$MODEL" \
    "env.eval.assets_path=$ROBOTWIN" \
    env.eval.total_num_envs=64 \
    env.eval.rollout_epoch=1 \
    env.eval.max_episode_steps=200 \
    env.eval.max_steps_per_rollout_epoch=200 \
    env.eval.use_fixed_reset_state_ids=true
  ```

#### 现场过程与资源

- 启动约为 2026-08-21 23:43:30 CST；Ray session：
  `/tmp/ray/session_2026-08-21_15-43-27_098471_1279722`。
- driver 与 worker 日志确认 placement 直接落在物理 GPU 4、5、6、7；物理 0–3 没有本次 RLinf compute
  process。
- rollout 实测 `167.79 s`；`/usr/bin/time` 端到端 wall 约 `5 min 58 s`。资源快照约为物理 4–7
  各 `11 GiB` 显存、host used `113 GiB`；未达到 80GB/card 或约 2TiB host RAM 的资源边界。

#### 自然完成与产物验收

- driver exit 0；完成 marker：`R2_SFT_FIXED64_GPU4567_OK`。
- fixed-64 结果：`eval/num_trajectories=64`，`eval/success_once=0.734375`，即 `47/64`。这是本次
  fixed seed SFT 起点评估，不外推为更多 seed 的总体成功率，也不是 PPO 改善证据。
- 输出与哈希：

  | 产物 | 服务器路径 | SHA256 / schema |
  |---|---|---|
  | resolved config | `/data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed64-4gpu4567-official-half-v1/resolved.yaml` | `b52e45cd42702f8c92937dc7e57c8f19dcbe855830f6cd0017a4d7d2b6618915` |
  | driver log | `/data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed64-4gpu4567-official-half-v1/driver.log` | `caf925f359c5fb1fd493032e59bdf1999c1e166f2d5def9ed40a007b4185ebe6` |
  | eval video | `/data/chenyiteng/results/rlinf-shenzhen/ppo/sft-fixed64-4gpu4567-official-half-v1/video/eval/seed_2/0.mp4` | `59ee32fe598a9830738766795ca5e638da00faec7c9ef6136ce37d2653bcb135`; H.264, 1280×960, 6 frames |

## 3. Stage B 与 Stage C

### RL-SZ-R3-001 — Stage B PPO one-step + inline fixed-64（完成）

- command file：
  `local_scripts/remote_commands/shenzhen_rlinf_r3_run_ppo_oneopt_4gpu128train64eval_v1.sh`。
- command-file SHA256：
  `010cf27ec3aec0e6e18178f258a4b26c7790b2b483d1239fab47b02b1a0e216b`。
- output：
  `/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1`。
- 沿已 resolved 的 official-half 配置运行：物理 GPU 4–7、128 train env、1 rollout epoch、每 env
  50 primitive steps、H/C=50/50、global batch 128、micro batch 32、update epoch 1；inline eval 为同一
  64 个 fixed reset IDs、最多 200 primitive steps。

#### 预算、耗时与资源

- train rollout 形成精确 `128` records；4 actor ranks 各 32 records，完成 **1 次 distributed
  optimizer step**。没有把 micro batch、rank 或 value-head 子计算重复计成多次 optimizer update。
- global step 实测 `682.349 s`；command 端到端 wall 约 `13 min 20 s`。
- rollout 阶段现场快照约为物理 4–7 各 `36.6 GiB` 显存、host used `124 GiB`；后续 inline eval/save
  阶段快照约为各 `52.4 GiB`、host used `232 GiB`。这些是离散现场快照，不冒充连续采样峰值。

#### 训练与 inline eval 指标

- train rollout：`success_once=0`。
- 唯一 update 的 actor 指标：`total_loss=0.123`、`policy_loss=-7.45e-09`、`grad_norm=510.6`；
  value loss 为 `0.123`。本账只记录 official logger 数值，不从单次 update 推断收敛或稳定性。
- inline fixed-64：`eval/success_once=0.8125`，即 `52/64`。它与 Stage A 使用同一 fixed seed 列表，
  但单次 smoke 的前后数值不单独构成稳定收益结论。

#### checkpoint、日志与视频验收

- driver exit 0；完成 marker：`R3_PPO_ONEOPT_4GPU128TRAIN64EVAL_OK`。
- `global_step_1` checkpoint 已完整落盘，`du -sh` 为约 `18G`；同时存在 DCP metadata/非空
  `*.distcp` 与可供 fresh reload 的 full weights：
  `/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1/robotwin_ppo_openpi/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt`。
- 产物哈希：

  | 产物 | 服务器路径 | SHA256 |
  |---|---|---|
  | resolved config | `/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1/resolved.yaml` | `3a0967a4b7592cf068956febcf2c273c79c50c1cef20d068588dbc5d2d8a931d` |
  | driver log | `/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1/driver.log` | `39e5827c26ac7580b5f10387b49308dfdd01786d669b3cb0e850951a2613e432` |
  | full weights | `/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1/robotwin_ppo_openpi/checkpoints/global_step_1/actor/model_state_dict/full_weights.pt` | `ed6a4cc0474d2b31a58de7671448ebbcd5c952756c84adb8c4753aaf0ffbb448` |
  | inline eval video | `/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-oneopt-4gpu128train64eval-v1/video/eval/seed_2/0.mp4` | `c5306d8360397b7346c72773b53a5dd664d8696fbf1a0292bd7599f56ba266aa` |

### RL-SZ-R3-002 — Stage C fresh process reload fixed-64（授权终止）

- command file：
  `local_scripts/remote_commands/shenzhen_rlinf_r3_run_reload_fixed64_gpu4567_v1.sh`。
- command-file SHA256：
  `3aac64f24dbfde3e8deee9d8737f1a167032daefd65761fd184b9b0335613cc4`。
- output：
  `/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-reload-fixed64-4gpu4567-v1`。
- Stage B 已自然退出后，本段由新的 SSH、Python 和 Ray 生命周期启动；实际 `resolved.yaml` 的
  `runner.ckpt_path` 指向 Stage B 的上述 `full_weights.pt`，不是继续使用进程内 actor，也不是把 inline
  eval 冒充 reload。
- full weights load 与 fresh Ray worker 启动通过；fixed-64 rollout progress 达到 `100%`，rollout 实测
  `166.56 s`，证明 checkpoint→fresh process→worker→simulator rollout 调用链已经实际运行。

#### 用户授权的 exact stop

- rollout 完成后、最终 metric table 汇总前，用户要求停止本段。目标 owned driver PID 为 `1352354`；
  stop helper 在发信号前读取 `/proc/1352354/cmdline`，同时核对
  `evaluations/eval_embodied_agent.py` 与 run name `ppo-reload-fixed64-4gpu4567-v1`，没有按模糊名称停止
  其他 Python/Ray 进程。
- stop command file：
  `local_scripts/remote_commands/shenzhen_rlinf_stop_owned_reload_fixed64_20260822.sh`；SHA256
  `a07c396ed0d255a7229599f52de6e129eca48e5b2215550237d509bd6c841505`。
- stop helper exit 0；exact output：

  ```text
  sending SIGINT to owned reload driver pid=1352354
  owned reload driver exited after SIGINT
  ```

- Stage C 主 run 因该 SIGINT exit 1，终端为 `KeyboardInterrupt`；没有出现
  `R3_RELOAD_FIXED64_GPU4567_OK` marker，也没有最终 metric table。
- 结论边界：该 exit 1 是用户授权终止的直接结果，**不记为自发 runtime failure**；但由于缺少聚合后的
  `eval/success_once`/trajectory metric table，**也不能称 Stage C fixed-64 评估完成或成功**。当前只确认
  fresh reload 与 64 条 rollout 机制已实际走通。

## 4. RL-SZ-R4-001 — official-time-axis PPO formal-100（后台运行中）

- 用户确认 Stage B 的 4 卡并发与资源占用满意，并明确授权立即启动正式 PPO；不继续等待 fresh-reload
  汇总，也不让 Fast-WAM 安装争用网络/CPU。
- command file：
  `local_scripts/remote_commands/shenzhen_rlinf_launch_ppo_formal100_4gpu128train64eval_v1.sh`；SHA256
  `369f202ec6a467ba6e7f30840a97adfa0662a2217e7264409b28880b960a97ee`。
- output：
  `/data/chenyiteng/results/rlinf-shenzhen/ppo/ppo-formal100-4gpu128train64eval-official-v1`。
- background driver PID：`1375834`；resolved SHA256：
  `48b4be79af300512d757ce47219ea2b27445155dbd6438be300459756b266926`。

### 精确训练预算

- physical placement：GPU `4-7`；train/eval 并发为 `128/64`。
- 恢复 official PPO 完整时间轴：train `rollout_epoch=4`、每 epoch/episode 最多 `200` primitive steps、
  `H=C=50`。每个 outer step 形成 `128×4×200/50=2048` records。
- actor `global_batch=2048`、`micro_batch=32`、world size 4、`update_epoch=2`，所以每个 outer step
  精确完成 2 次 distributed optimizer update。
- `runner.max_steps=100`、`val_check_interval=10`、`save_interval=10`；这不是此前的 50-step-prefix
  smoke。100 个完整 outer steps 预计是多日任务，不承诺约 12 小时完成。

### 启动确认（2026-08-22 00:37 CST）

- driver alive；Ray、4 actor ranks、4 rollout ranks 与 4 env ranks 均已创建，placement 日志逐项显示
  physical GPU 4、5、6、7。
- official SFT norm stats 与 actor/rollout 模型均已加载；日志已进入
  `Generating Rollout Epochs: 0/4`，即首个正式 outer step 的第一轮 rollout。
- `fatal_lines=0`：未发现 traceback、CUDA OOM、worker crash 或 SIGKILL。
- 现场资源：GPU 4–7 分别约 `18.49/18.37/18.72/17.89 GiB`；host used `69 GiB`、available
  `1917 GiB`。这是启动阶段快照，不冒充后续峰值。
- 按用户要求只确认正常启动后交接；不持续盯守。后续刷新必须先读 live PID/log/checkpoint/GPU/RAM，
  不从本段启动快照推断当前仍在运行或已完成。

## 5. RL-SZ-R4-002 — formal-100 Step 21/22 只读健康审计

### 审计命令与边界

- 现场时间：2026-08-22 09:55–10:08 CST；账号仍为 `chenyiteng`，固定 host-key 的 Paramiko
  密码路线未变。全部远端命令只读取目标 run、`/proc`、cgroup、GPU、文件元数据和 TensorBoard
  scalar；未停止进程、未修改服务器文件、未读取 checkpoint 权重内容、未下载大产物。
- 主要 command files 与 SHA256：

  | command file | SHA256 |
  |---|---|
  | `shenzhen_rlinf_formal100_live_audit_20260822.sh` | `41c1bd3de09d2c4a419bf502b65e8b65bddea4f19ffd1de67e294c4aeb86991a` |
  | `shenzhen_rlinf_formal100_readonly_artifact_metrics_audit_20260822.sh` | `ca4a9ebba42bb1cf6ef4e9820ca3616c69f33c5f62226e99835133465f549052` |
  | `shenzhen_rlinf_formal100_extract_step_metrics_20260822.sh` | `79b1a14bc8b29b53bde296f7a5cb3457063b28611f6d41b0877286f6346fb094` |
  | `shenzhen_rlinf_audit_formal100_step_current_20260822.sh` | `ddc36d3bce6678678b049d6c8d48bdea75bb53f264027a9c58e4e1ec90172951` |

### 训练与产物现场

- 09:58 CST 时 driver PID `1375834` alive，完整到 Global Step 21，Step 22 rollout 为 `2/4`；
  10:08:21 CST 再读时 Step 22 已完整，训练继续进入 Step 23。fatal/OOM/NCCL/worker-died/NaN/Inf
  扫描均为 0。
- Step 1→21 的 train success 为 `78.71%→88.48%`；后 5 步均值 `88.59%`。fixed-64 eval
  在 Step 10/20 分别为 `58/64=90.625%` 与 `62/64=96.875%`。Step 22 train success 为
  `92.77%`。
- Step 21 的 actor `KL=0.0130`、`clip_fraction=0.0549`、`ratio=1.0017`、
  `grad_norm=30.90`；critic `value_loss=0.0291`、`explained_variance=0.397`。Step 22 对应
  `KL=0.014`、`clip_fraction=0.053`、`grad_norm=28.01`、`value_loss=0.020`、
  `explained_variance=0.419`。全部 finite；数值侧未见发散。
- 09:58 CST run 为 `35G`；checkpoint `global_step_10`、`global_step_20` 各约 `17.21 GiB`，
  均有 DCP metadata、4 个非空 shard 和 `full_weights.pt`。已有 train/eval MP4 `344/8`，另有
  resolved config、metrics log 和 1 个 TensorBoard event。`/data` 约剩 `3.1 TiB`。
- 完整产物清单与逐步 scalar 留在
  `evidence/formal100_live_step21_20260822/`；本轮没有做额外 checkpoint hash/fresh-load，避免给
  运行中的训练增加约 16 GiB 顺序读取负担。

### 资源判断：数值绿、GPU 绿、主存黄灯（高优先级）

- 09:58 的 GPU 4–7 显存为约 `64.2–66.6 GiB/卡`；Step 22 边界后约
  `60.7–62.1 GiB/卡`，80GB 卡仍有余量。瞬时 utilization 会随 rollout epoch 边界降到低值，
  不把单点低利用率解释为空转。
- 主存是当前唯一高优先级风险。09:58 cgroup `memory.current` 为
  `1,528,978,501,632 B`（约 `1.391 TiB`），10:08 为 `1,535,529,234,432 B`
  （约 `1.397 TiB`）；同一时段 host available 约 `603.1→597.0 GiB`。四个 EnvWorker 的 RSS
  合计约 `1.32 TiB`。跨 Step 22 边界没有出现大释放；cgroup 的 `high/max/oom/oom_kill` 计数仍全 0，
  swap 仍为 0。
- 65 秒短窗曾出现 available 下降约 `2.5 GiB`，随后增长速度变慢，因此不能把该瞬时斜率线性外推成
  精确 OOM 时间；但从启动期 host used 约 `69 GiB` 到当前 cgroup 约 `1.4 TiB`，且 step 边界不回收，
  已足以判定“当前仍能正常训练，但资源稳定性不能过关”。是否在已有 `global_step_20` 安全点主动停止并
  定位 EnvWorker 内存，需要用户决定；本次审计未擅自干预。

## 6. RL-SZ-R4-003 — Step 23 中段与 EnvWorker `smaps` 只读复核

- 现场时间：2026-08-22 10:28–10:31 CST；完整到 Step 22，Step 23 rollout 已达 `3/4`，driver
  `1375834` alive，fatal 扫描仍为 0。Step 22 标量与上一条一致且全部 finite。
- GPU 4–7 瞬时约 `62.3–64.6 GiB/卡`；GPU 0–3 无 compute process，其他账号也没有 GPU app。
- 10:28 训练 session cgroup `memory.current=1,579,751,776,256 B=1.437 TiB`，host available
  `555.6 GiB`，swap 与 cgroup OOM events 仍为 0。相比 10:08，约 20 分钟内 cgroup 再增
  `41.2 GiB`、host available 同量下降；该区间位于 rollout 相位，不能以两点精确外推 OOM 时间。
- 四个 EnvWorker RSS 合计 `1,431,581,644 KiB=1.333 TiB`。逐 PID `smaps_rollup` 显示
  `Pss≈RSS`，且绝大部分为 `Pss_Anon/Private_Dirty`；file/shared 每 worker 不到约 1 GiB、`VmSwap=0`。
  因此主存主体不是 page cache 或共享映射，而是 EnvWorker 的真实私有匿名内存；当前还没有足够证据把
  根因进一步归到 SAPIEN、CuRobo 或某个 Python/CUDA 对象。
- 精确双账号、服务、网络、quota、其他用户和 Fast-WAM 断点见
  [`07_SERVER_LIVE_AUDIT_20260822.md`](07_SERVER_LIVE_AUDIT_20260822.md)。本次仍没有停止、改参、安装、
  下载或写远端。

## 7. RL-SZ-R4-004 — Step 40 checkpoint/eval 与内存停止点复核

- 现场时间：2026-08-22 18:03–18:10 CST；固定host-key的`chenyiteng`只读检查。command file
  `shenzhen_rlinf_formal100_step40_closeout_snapshot_20260822.sh` SHA256 为
  `f7ae802cb2b1c962518f75a064e421c7ec3e70e644d8697848ccb7f20fb9a02b`。
- 18:10已完整输出`Global Step 40/100`，`global_step_40`目录存在，fixed-64 eval=`62/64=96.875%`；
  当步train success=`92.578%`、KL=`0.0089`、clip fraction=`0.062`、grad norm=`30.747`、critic
  explained variance=`0.437`，全部finite。随后自然进入Step 41的`0/4`；driver仍alive、fatal=0。
- GPU 4–7约`54.3–55.0 GiB/卡`；GPU 0–3仍无compute process。cgroup
  `memory.current=1,920,436,883,456 B≈1.747 TiB`，host
  `MemAvailable=266,727,452 KiB≈254.4 GiB`，swap仅28 KiB，`high/max/oom/oom_kill=0/0/0/0`。
- 与16:20 Step 36后同为next `0/4`的点相比，4步内cgroup增加约`84.8 GiB`，MemAvailable下降约
  `75.8 GiB`；短窗折合约`+21.2/-18.95 GiB per step`，不能再称“增长正在平台化”。
- 数值与checkpoint侧正常，主存侧风险不通过。已向用户明确建议以Step 40为自然停止点，不再等到Step 60；
  本条仍为只读检查，未在没有用户停止授权时给driver发送信号。完整Step 1–40曲线与原始轻量证据见
  [`09_PPO_CURVES_LIVE_20260822.md`](09_PPO_CURVES_LIVE_20260822.md)。
- 18:26:42 CST补充只读点：Step41 rollout=`2/4`，driver alive、fatal/OOM仍0；cgroup约`1.769 TiB`、
  MemAvailable约`227.9 GiB`，GPU4–7约`67.8–68.7 GiB/卡`。该点不是step边界，不用于拟合同相位斜率；
  它只进一步支持“不等Step60”的停止建议。仍未发送任何signal。
- 18:44:55 CST再次执行同一只读command：Step41已经完整，正在Step42 rollout=`1/4`；driver仍alive，
  fatal与cgroup OOM事件仍为0。cgroup `memory.current=1,946,696,237,056 B≈1.770 TiB`，host
  `MemAvailable=237,033,552 KiB≈226.05 GiB`，GPU4–7约`64.45–65.57 GiB/卡`。本次仍未发送signal；
  继续等待到Step60已没有资源依据。
- 18:55:54 CST同一只读command：Step42 rollout=`3/4`；driver alive、fatal/OOM事件仍为0。cgroup
  `memory.current=1,941,020,966,912 B≈1.765 TiB`，host `MemAvailable=236,971,940 KiB≈226.0 GiB`，
  GPU4–7约`60.68–61.44 GiB/卡`。与18:44点相比只是同一步内约5.3 GiB波动，整机可用内存没有恢复；
  不把它解释成长期平台，仍未发送signal。
- 19:02:45 CST完整到Step42、进入Step43的`0/4`；driver alive、fatal/OOM事件仍为0。cgroup
  `memory.current=1,935,574,204,416 B≈1.760 TiB`，host `MemAvailable=236,328,504 KiB≈225.38 GiB`，
  GPU4–7约`54.29–55.23 GiB/卡`。相对Step40同为next-step `0/4`，两步后cgroup仍增加约14.1 GiB，
  host available再下降约29.0 GiB；故Step42边界也没有提供“可安全等到60”的新证据。仍未发送signal。

## 8. RL-SZ-R4-005 — Step 46完整指标、资源与产物只读刷新

- 现场时间：2026-08-22 20:58:16 CST；账号`chenyiteng`，仍走固定host-key的Paramiko密码路线。
  command file为
  `local_scripts/remote_commands/shenzhen_rlinf_formal100_latest_readonly_refresh_20260822.sh`，SHA256
  `a33f38347e6812c9552366cb0de2349e8c5f5ac2718e8ec29032b95098e87bcf`。本次只读`driver.log`、
  `metrics.log`、`/proc`、cgroup、GPU与文件元数据；没有stop/signal、GPU workload或服务器写入。
- driver PID `1375834` alive；最新完整为`Global Step 46/100`，现场正在Step47 rollout `2/4`；fatal
  扫描为0，cgroup `high/max/oom/oom_kill=0/0/0/0`。
- Step46 train success=`89.84%`；Step42–46五步均值=`90.82%`，最近10步均值=`91.68%`。分段均值从
  Step1–10的`79.18%`升至Step31–40的`91.68%`，Step41–46为`91.21%`：训练成功率已进入约
  `90–92%`高位平台并有采样波动，未见崩坏，但不能只凭训练rollout宣称统计收敛。
- Step46 `KL=0.020`、clip fraction=`0.063`、pre-clip grad norm=`30.992`、critic explained
  variance=`0.447`，全部finite。Step1–46范围分别为`0.0043–0.064`、`0.026–0.097`、
  `18.993–36.021`、`-0.240–0.490`；没有优化器发散形态。
- fixed-64仍只有Step10/20/30/40四个真实点：`58/62/58/62`。它确认没有held-out collapse，但呈交替
  波动而非单调改善；Step46不是保存/评估点，最新可恢复checkpoint仍为Step40。
- Step1–46 step/rollout/actor-training中位数=`1563.05/1535.15/22.94 s`；非评估步均值约
  `1545.35 s`，每10步的eval/save步均值约`1912.97 s`。计算瓶颈仍为simulator rollout。

### 资源与产物

- cgroup `memory.current=1,955,901,038,592 B=1821.58 GiB≈1.779 TiB`；host
  `MemAvailable=210,635,256 KiB=200.88 GiB`，约只剩整机总内存的10%；swap current仅
  `335,872 B`，memory events仍全0。四个EnvWorker RSS合计
  `1,803,294,536 KiB=1719.76 GiB≈1.679 TiB`，主存主体仍在EnvWorker。
- GPU4–7瞬时显存=`70,921/61,048/62,580/66,639 MiB`，总量均为`81,559 MiB`；GPU0–3为
  `69/4/4/4 MiB`且无compute app。该点是rollout中段瞬时值，不作显存峰值或利用率均值。
- run总量`74,093,166,525 B=69.01 GiB`；`global_step_10/20/30/40`四个checkpoint各
  `18,475,261,914 B=17.21 GiB`，各含`8,067,778,207 B=7.514 GiB`的`full_weights.pt`。
  train/eval MP4为`743/16`个，合计约`180.3/2.06 MiB`；另有完整metrics log、driver log与
  TensorBoard event。`global_step_50`尚不存在。
- run内及同级PPO结果根没有resource/monitor CSV或log。资源PNG只连接既有的**完整step边界离散点**；
  20:58的Step47中段点用独立红点表示，不把它伪装成逐step连续监控。该证据边界和Step1–46四张图见
  [`09_PPO_CURVES_LIVE_20260822.md`](09_PPO_CURVES_LIVE_20260822.md)。

本次结论：策略指标已经高位平台化、优化仍稳定；但主存继续逼近整机极限。若用户下一步选择停止，现有
自然恢复点是Step40，而不是未保存的Step46。本条仍未发送任何signal。
