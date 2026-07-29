# AutoDL / RLinf 当前交接入口

最后更新：2026-07-29。本文只保留并行专题路由、当前停点和授权边界；旧时间线见
`docs/project-history/00_INDEX.md`。进入某个窗口时只选择对应专题，不默认加载另一专题正文。

## 并行专题路由

| 专题 | 唯一事实源 | 实施账本 | 当前停点 |
|---|---|---|---|
| π0 × RoboTwin × DSRL | `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md` | 正式训练已在 step 198 收尾；可恢复 DCP 为 step 195 |
| RLToken / RLT × π0 × RoboTwin | `docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-rltoken/03_STAGE1_FORMAL_TRAINING_20260729.md` | full clean-50 Stage 1 2k 已启动；19:38 固定快照 step172、运行健康 |
| QAM × π0 × RoboTwin | `docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md` | 已确认 Plain/action/B1+F1；C1 工程细节已委托；fixed-N M2+N20+online-only 待协议确认；服务器只读接口/资源已刷新 |
| Fast-WAM × RoboTwin × RLinf | `docs/fastwam-robotwin-rlinf-grpo/00_INDEX.md` | 由该索引路由 | 非本窗口默认上下文 |
| 根历史 | `docs/project-history/00_INDEX.md` | — | 只作追溯 |

DSRL / RLT / QAM 的旧调查和七份历史材料保存在
`docs/rlinf-robotwin-pi0-traditional-rl/01_FULL_REFERENCE_HISTORY_20260728.md`，不作为当前规范。

## DSRL 当前状态与授权

- 冻结主线：H=50、N=20、4 train env、`rollout_epoch=1`、warm-up 500 global macro
  transitions、UTD20、flat replay 25k、DSRL-only narrow target-shadow resume；N=50 暂缓。
- 服务器独立 worktree：
  `/root/autodl-tmp/RLinf_fastwam_rlinf` 的 `codex/dsrl-pi0-robotwin`。
- 主体实现、flat replay、resume 修复、formal/thin-smoke config 和集中单测已完成；
  8 个单测、三配置 compose/resolve、no-Ray validator、旧 PPO 回归、imports、Ruff 和
  whitespace 均通过。
- 已推送 `personal/codex/dsrl-pi0-robotwin`，云端 HEAD
  `2d942b714b004de9a7efdbd4a7e2efaac3ef6d01`；当时服务器 worktree clean。
- fresh/resume 完整 packet：
  `docs/rlinf-robotwin-pi0-traditional-rl/evidence/SMOKE_APPROVAL_20260728.md`。
- fresh/resume 实际流水：
  `docs/rlinf-robotwin-pi0-traditional-rl/evidence/SMOKE_EXECUTION_LOG_20260728.md`。
- 2026-07-28 17:47 终态：fresh 40 transitions/800 updates、resume 37/740，DCP1/DCP2、
  target-shadow/replay/RNG 连续性、冻结 π0 bitwise 不变和真实 success/truncation 投影均通过；
  GPU 峰约 34.8 GB/卡，无 OOM/NaN/crash；两卡、Ray/worker/monitor 均已清空。
- 18:17 追加 fixed observation/fixed 32D repeat-H latent 单卡 parity：DSRL 入口与 base
  transform/denoise/output core 的六类输出均 bitwise exact、最大差异 0；这是 18:54
  正式训练启动前的历史验收快照。
- smoke 实际运行的代码快照为 `2d942b714b004de9a7efdbd4a7e2efaac3ef6d01`；formal 启动
  时加载的代码快照为 `d664bf349b63b75f41d51c8295cb0a330780d783`。后续只增加状态报告
  与图，不改变活进程已经加载的代码。约 63 GB smoke 产物保存在批准包固定 run root；
  下轮仍须 live 刷新分支最新 docs-only HEAD。
- 用户已授权并启动完整正式训练：2 GPU、4 env、UTD20、micro 64、
  `max_steps=650`、`val_check_interval=13`、`save_interval=65`，每次评估 12 episodes。
- 正式 run root：
  `/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1`；
  driver PID `70062`，通用/cgroup 分项 2 秒监控 PID 为 `70064/70065`。启动 resolved config
  SHA-256 为 `e99c212d1743e285dcda23cb129e2ed96545cceb36bebe772ae69a693b9df595`。
- 正式运行的精确命令、配置、参数依据、产物、资源解释、停止条件和逐操作状态统一记录在
  `docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md`；
  下轮先 live 刷新 driver/Ray、日志 global step、GPU/cgroup、磁盘和 DCP，再称为当前。
- 2026-07-28 19:29 现场：最新完整 step 15、step 16 已开始；resident 585，
  step 13 越过 warm-up，learned SAC 累计 2,260 updates。首轮 formal eval 为 1/12；
  指标有限、10-Q 紧密、无 OOM/NaN/crash。GPU 峰约 35.2/34.8 GB，anon 峰约 47.7 GiB，
  cgroup raw 贴顶主要为 file cache，回收事件已平台。当前没有 DCP，首个在 step 65。
- 本次静态报告与图：
  `docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP20_20260728.md`。
  用户不要求 Codex 持续在线监控；训练保持运行，下次请求时再做新的 live 刷新。
- 2026-07-28 20:08 现场：最新完整 step 20，resident 761，约 15,220 requested
  primitive interactions，learned SAC 累计 5,780 updates。Gaussian train phase 2/52，
  learned train phase 8/28，trailing-20 为 30%；formal eval 仍只有 step 13 的 1/12，
  下一次为 step 26。critic loss 0.880 是首要观察项，但 critic grad 4.84、Q/alpha/entropy
  均 finite，无 OOM/crash。逐 step 用时和完整命令见 formal 流水账 FORMAL-010。
- 2026-07-28 21:03 现场：最新完整 step 30，resident 1,062，约 21,240 requested
  interactions，累计 11,800 updates。第二次 formal eval（step 26）为 7/12；learned
  train phase 为 33/68、trailing-20 为 60%。critic loss 在 step 25 达 1.929 后回落到
  0.848，grad/Q 均有限；alpha/entropy 降至 0.076/16.77，作为后续观察项。无 OOM/crash，
  但 env-worker RSS 从 19:15 的约 11.9 GiB 阶梯升至 21:05 的约 18.3 GiB，下一次需确认
  是否继续增长。横版状态图和完整判断见
  `docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP30_20260728.md`。
- 2026-07-28 23:04 指标现场：最新完整 step 53，resident 1,681，约 33,620 requested
  interactions，累计 24,180 updates。step 39/52 formal eval 分别为 8/12、10/12，
  learned train phase 累计 101/160、trailing-20 为 75%。critic loss 约 0.395，
  10-Q/grad 均有限；alpha/entropy 为 0.0061/-4.15，仍高于 target entropy -16，需观察
  是否在目标附近减速。两卡当前约 31.4 GiB、历史峰约 34.4/34.1 GiB；cgroup anon
  45.6 GiB、PSI=0、OOM=0。env-worker RSS 峰值 21.5 GiB，仍增长但总 anon 低于初始化
  峰值。下一节点 step 65 同时 eval 和首个 DCP；完整判断见
  `docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP53_20260728.md`。
- step-53 报告已在服务器形成 docs-only commit
  `4447d40211be8c78874bf4b000c871e7fbd93561`，worktree clean；服务器到
  `github.com:443` 连续连接超时，故 upstream 仍为
  `b01661e8a6b3ca1b883fb61d4ade9a467ffd84b5`。训练未受影响，23:26 已推进到 step 57。
  下次刷新先核验 HEAD/upstream/clean 和 driver，再只重试该 commit 的 push，不重做报告。
- 2026-07-29 10:02 指标现场：最新完整 step 188/650，resident 4,959，约
  99,180 requested interactions、89,740 optimizer updates，已经到约 100k review 点。
  step 65–182 的 10 次 formal eval 合计 112/120=93.3%，step 130–182 合计
  57/60=95%；最新 step 182 为 10/12，最近训练 trailing-20 为 95%，目前没有退化证据。
  alpha/entropy 已稳定在约 0.0024/-16；critic loss 和 Q finite，但记录的裁剪前 critic
  grad 在 step 91–188 持续高于 clip=10，最新 29.18，作为当前首要优化观察项。
- step 65/130 两个约 32 GiB DCP 结构完整，保存额外耗时约 30/28 秒。两卡当前约
  31.6 GiB，历史 DCP 峰约 40.5/40.4 GiB；cgroup anon 54.3 GiB、PSI=0、OOM=0，
  env-worker RSS 约 28.5 GiB且增速已放缓。当前可以继续训练，但不扩大 env 并发。
  预计剩余约 35.9 小时，下一节点 step 195 同时 eval 和 DCP。完整横版报告见
  `docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_STATUS_REPORT_STEP188_20260729.md`。
- step-188 报告已形成 docs-only commit
  `dc6a3a430be9c3a5002b436c4aeeaa399509f334` 并成功推送到
  `personal/codex/dsrl-pi0-robotwin`，此前积压的 docs commits 也一并发布；
  推送后 HEAD=upstream、worktree clean，训练已推进到 step 192。训练配置、进程和
  run 产物均未改动；下轮仍先 live 刷新，不能把本条静态快照称为当前。
- 用户确认信息足够后，2026-07-29 10:49:33 向 formal driver 发 TERM；10:49:41
  driver、本次 Ray session 全部后代及两个资源 monitor 均退出，两卡显存归零，
  `GRACEFUL_STOP=PASS`。最后完整 cycle 为 step 198，TensorBoard 最后 flush 到
  step 197；最终累计 5,185 macros、103,700 requested interactions、94,260 updates。
- 可恢复终点是完整的 `global_step_195`：约 32 GiB、11 个必需文件、无临时残留；
  step 196–198 的日志/指标保留，但其参数更新不在 DCP195。三个 DCP 仍原地保存在
  formal run root，未删除或搬运。
- 最终效果：step 195 formal eval 11/12；step 65–195 合计 123/132=93.18%，
  step 130–195 合计 68/72=94.44%。完整解释、指标词典、资源/PPO-GRPO 对照和样本量
  核算见
  `docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_CLOSEOUT_REPORT_STEP198_20260729.md`。
- 服务器运行材料包保存在
  `/root/autodl-tmp/experiment_exports/dsrl_pi0_robotwin_formal_v1_20260729/`，
  本机副本为
  `exports/dsrl_pi0_robotwin_formal_v1_runtime_step198_20260729.tar.gz`，
  SHA-256 `f64762c1f95f881732facf9d7da2870dc1bfb96f9a2e4328180d145b8a7f877c`；
  本地工作材料包为
  `exports/dsrl_pi0_robotwin_formal_v1_work_materials_20260729.zip`，校验值在同名
  `.sha256` 文件。大 DCP 只留服务器，运行包包含完整 manifest 和恢复路径。

## RLT 当前状态与授权

- 唯一主线：单任务 `adjust_bottle`；Stage 1 使用全部有效 clean-50、无 val split、冻结 π0
  只训 RL token；checkpoint `norm_stats` 为 Stage 1/Stage 2/decode 单一真相。
- Stage 2：H=50/C=10、14D canonical action、`z_rl=2048`、full-task train route、
  deterministic student eval、无人/无 expert、compact replay。
- 首批正确性修复已实现：RLT trainer state、feature/action/schedule/optimizer/bootstrap
  resume fingerprint、per-rank warm-up、completion marker 和首 rollout full sync；
  Stage 1 accepted manifest/model path/full-weights/stats/H/C/D/z/prefix 又增加 fail-closed
  双向验证。不做 hard capacity、RNG exact、async 或跨 world-size。
- 服务器独立 worktree：
  `/root/autodl-tmp/RLinf_rlt_pi0_robotwin`，
  从 `48a775db09c16c455aeba7b0600c920e7c80d534` 建立；分支
  `codex/rlt-pi0-robotwin`。Stage 2 artifact/预算加固代码提交为
  `3b610cb4685a1d41c97da64df67ab86561697dfd`；没有切换 DSRL worktree，也没有
  复制/安装环境。
- full clean-50 已从
  `TianxingChen/RoboTwin2.0@9dc9299c163db059931898a9f0852098a61155a1`
  转为 50 episodes / 7,188 frames / 50FPS：
  `/root/autodl-tmp/datasets/robotwin2/canonical/pi0-aloha-clean50-v1`。
  dataset manifest 为
  `/root/autodl-tmp/datasets/robotwin2/manifests/pi0-aloha-clean50-v1.json`，
  SHA-256 `12ce2ed6...f86c`；全量动作时序 max-abs0，正式 global32 loader 两 rank 各
  local16 通过。
- Stage 1 smoke 已完成：S1-A 两卡 micro/global1/2 两个 optimizer step，loss
  `5.15/5.21`、vla loss 0、step2 DCP/full-weights 20.56GiB，新进程 reload-only exit0；
  S1-B 两卡 micro/global16/32 单步 loss5.18，显存峰 26,447MiB/卡，无 checkpoint/OOM。
  scheduler 在 step 前后记录方式意味着 S1-A 只有第 2 次 update 使用非零 LR，S1-B 的
  唯一 update 使用 LR 0；因此 S1-B 是正式 batch 的前反传/容量门，不是参数 delta 证据。
  两步 checkpoint 只作 smoke 证据，不是正式 feature artifact。
- 正式 source config SHA-256 `c293bc47...5a4c`，resolved config SHA-256
  `5aa824fc...d67e`；`min_lr_rate=.1`，2k/无 val/micro-global16/32/no-shard/仅 RLT
  trainable 均经机器断言。
- 正式 Stage 1 已 exit0 完成2,000步，总时长28m54s。运行根：
  `/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1`；
  runtime/evidence：
  `/root/autodl-tmp/experiment_exports/rlt_stage1_formal_20260729_v1`。
  唯一 endpoint 是
  `.../robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000`。
- 正式训练前100/后100 loss均值 `3.9023/0.5551`，下降85.8%；两卡各
  26,447MiB，matched RSS峰值38.51GiB，cgroup anon峰值39.53GiB，host available最低
  928.83GiB；无 OOM/CUDA/NCCL/rank-death。
- Stage 1 artifact acceptance 全门通过：
  fresh/endpoint/shuffled/zero reconstruction loss
  `5.1977/0.5338/1.7118/2.1027`，non-RLT changed tensor=0，RLT changed=54/62。
  manifest ID 为
  `robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1`，manifest SHA
  `6ca58f26...12433`，full-weights SHA `7dddc268...4bd0`。
- 高信息量 Stage 1 用户包：
  `exports/rlt_stage1_formal_high_info_20260729_v2.zip`，SHA-256
  `9d9e2c38789897479a27cc04ed15034a9d65175284c837f3c1c6f54ca0c2daa8`；
  不含20.56GiB checkpoint。
- Stage 2 formal source 现在 `max_steps=0` fail closed。正式 candidate 保持2GPU/4env、
  H50/C10/D14、batch512/128、UTD5/ratio2、500 rows/rank、5k critic floor、
  cap400、15k replay/rank；这些相对 ManiSkill/论文/RoboTwin 的来源与风险见
  `docs/rlinf-robotwin-pi0-rltoken/04_STAGE2_PRE_SMOKE_PACKET_20260729.md`。
- Stage 2 artifact preflight、formal/fresh/resume compose/audit、Ruff、py_compile和
  19个集中单测已通过；MLP仅2,162,202参数。fresh/resume launch script 服务器
  `bash -n` 和22-field resource-monitor单样本自测通过。
- 计划 smoke 路径
  `/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1` 和 runtime evidence
  `/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1`
  在最后预检时均不存在；**Stage 2 smoke 尚未启动**。
- 启动前按授权精确删除12个旧 RLinf DCP和51个 Motus OPD实验`.pt`，回收约177.85GiB；
  PPO step20、GRPO step100、轻量日志/配置/指标与Motus官方/base权重均保留。删除清单位于
  `/root/autodl-tmp/experiment_exports/rlt_pre_stage1_cleanup_20260729`，被删权重不可恢复。
- 当前授权停点：等待用户审批 `04_STAGE2_PRE_SMOKE_PACKET_20260729.md` 中的
  fresh→postcheck→新进程 resume 两阶段；没有审批不得运行。下一次执行前仍须刷新
  branch/upstream/dirty tree、进程、GPU/RAM/disk 和目标路径。formal pilot 总 cycle
  预算继续未批准；smoke 后再在约30-cycle phase-transition pilot 与约60-cycle
  初步趋势 pilot 之间决定。

## QAM 当前状态与授权

- 方法真值锁定 arXiv v4 与
  `ColinQiyangLi/qam@2726d767c9a0a7a46d49693f0391f73dc2cf58ac`：JAX/Flax、
  behavior FM/EMA + 独立 fine flow + 10-Q。论文只抽象写 state；官方复现实验使用
  actor/critic 对称的 nonvisual OGBench low-dimensional state，其中 manipulation
  observation 含 simulator-derived 物体真值。官方无 π0/RoboTwin/真机实现；LWD 只提供
  VLA/真机方向证据，不是可直接抄的 Plain 代码。
- 用户已确认 `Plain+action-space+adjust_bottle+B1+F1`，并把 C1 pooling/双视图与两卡
  ownership 交由实施侧按 probe 落定；当前仍待确认的单一训练协议是
  `fixed-N M2+N20+online-only`，准确名称
  “Plain-QAM π0 online adaptation（frozen behavior + fixed-N macro transition）”，
  P1 小网络 exact 复现官方 B2+F1/FM/AM/10-Q/EMA。
- 原 SFT action expert 是 frozen behavior，无 FM/optimizer；F1 是独立完整
  action-expert copy，只由 AM 更新。两者共享 frozen VLM/prefix。rollout、evaluation
  和 TD next action 用 fine ODE；memoryless SDE 只在 trainer 内生成 AM 辅助轨迹，
  锁定代码最后边界步用 behavior ODE。
- C1 是 frozen 三视角/语言 feature + normalized 14D proprio + planned fixed-N action，
  后接 10 个独立 Q。它高层对齐官方“固定 observation map + 独立 Q”，但 ensemble std
  不覆盖 frozen encoder 的共同表示错误。当前 C1 推荐的 pooling 是三相机+语言四块 mask-aware
  position-block mean `[4,2048]` BF16 storage/FP32 critic，另拼 proprio。该 pooled view
  只够 Q；AM current state 和 TD next action 还需 canonical 三相机/task/proprio
  `obs_id/next_obs_id`，由 actor worker 重算 frozen prefix KV，不存 full tokens/KV。
- 2026-07-29 19:22 CST 服务器只读核对确认：RoboTwin qpos 路径把完整 planned
  waypoints 先组成 TOPP trajectory，向上不提供 planned-action `realized L`，
  `_cal_chunk_rewards()` 的 `n_steps_to_run` 还是 0。故 M2 改为每 query 一条固定
  N=20 macro；$R_{\rm macro}=\sum_{i=0}^{19}\gamma_{\rm slot}^i r_i$，target
  bootstrap 使用 $\Gamma_{20}=\gamma_{\rm slot}^{20}$；无 executed-prefix mask。
  `slot` 是逻辑 planned waypoint/reward 位置，不是测得的 simulator primitive duration。
  `N:50` 与 14D 外 padding 只在 $P_N^\top$ 嵌入的 terminal direct-Q gradient 为 0；
  frozen behavior reverse VJP 后的 intermediate adjoint 不强制裁零。
- 当前 online-only 推荐的 critic transition 全来自 RoboTwin 在线真实执行：先
  `collect`，再 `q_only`，
  Q 有动作区分且 gradient finite 后才 `am_on`。clean-50 不进 v1 Q loss、不建 QAM
  sidecar、不阻塞实现；RLT 已下载 ZIP 只作可选诊断资产。
- endpoint target-Q mean action gradient 经 frozen behavior 的逐步 VJP 搬回各 flow
  time，形成 AM 而非 FM 标签。OpenPI 必须统一使用
  `t_qam=1-t_pi0`、`f_qam=-v_pi0`，Q gradient 不穿
  `Unnormalize+AlohaOutputs`。官方 `[-1,1]` action clamp 尚需 P2 先量 π0 越界率；
  current/next/terminal-Q 与 env 执行不得使用两套不同 action。
- 生产不复制/改名 shared 14GB venv，也不安装官方 QAM；独立
  `/root/autodl-tmp/RLinf_qam_pi0_robotwin` worktree 显式复用
  `/root/autodl-tmp/RLinf/.venv`。官方小网络 oracle 使用独立 pinned CPU-JAX venv。
- 同次只读 header 计数把 F1 candidate allowlist 收窄为 173 tensors / 314,713,120 参数，排除
  263,323,648 参数的 unused `gemma_expert.lm_head`；真实 AM-SDE/VJP/FSDP peak 仍待
  实现 probe，并须包含 frozen-prefix recompute。当前两卡推荐复用 DSRL 式 per-rank
  replay/local B，每 rank 对本地 batch 计算完整逻辑 10-Q、同步梯度/target、per-rank
  replay checkpoint，不拆 5+5 heads；同 rank heads 初始化不同，跨 rank 对应 head
  初始化必须相同。
- 第一组 probe 起点是 `K10/flow10/gamma_slot=.99/Gamma20=.8179069/rho=.5/EMA=.005/
  clip1/globalB64/warmup512/UTD1`，critic `3e-4`、fine `2e-5`、`inv_temp 0→0.5`；
  replay 50k 只作为 canonical observation bytes probe 后的目标。分离 optimizer 共享
  pre-update snapshot。事实门是 F1 peak、C1 dual-view store/recompute、final-view
  payload、critic FSDP/resume，以及 normalized action clamp/env parity。
- 19:22 CST 身份探针为 `autodl-container-nekaqbwt43-6ce5babb /root uid=0`；
  baseline 仍是
  `local/openpi-a800-2gpu-migration@6d0db56bf26f972cd27fa29535f5eb939e80e5bf`，
  仅见已知 untracked A800 configs/local scripts；QAM worktree 尚不存在。
- 当前只授权本地专题文档与服务器只读审查；未授权创建 QAM branch/worktree、改服务器
  代码、解压/转换数据、运行 compose/import/test、smoke 或训练。后续默认只读根上下文、
  QAM 主计划和账本最新批次；来源查 `01`，教学查 `02`。

## 共同执行边界

- 动态事实以新的服务器只读现场检查为准；旧快照只用于定位。
- Windows 本机只读写文档、代码和 diff；Hydra compose、项目 import/compile、测试、
  smoke 和训练在服务器进行，并遵守对应专题授权。
- smoke/formal 前必须展示完整 resolved config、精确命令、输出目录、资源计划、观测指标
  和停止条件，等待明确批准。
- 不删除用户数据、不停止无关进程、不重装依赖、不下载大模型、不覆盖既有 checkpoint。
- 仓库改名 `rlinf_fastwam` → `rlinf_exp` 仍只是待核实候选；未明确目标前不执行。

任何进程、GPU/RAM、日志、checkpoint、HEAD、dirty tree 或远端仓库状态，只有本轮现场刷新后
才能称为“当前”。
