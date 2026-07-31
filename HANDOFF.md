# AutoDL / RLinf 当前交接入口

最后更新：2026-07-31。本文只保留并行专题路由、当前停点和授权边界；旧时间线见
`docs/project-history/00_INDEX.md`。进入某个窗口时只选择对应专题，不默认加载另一专题正文。

## 并行专题路由

| 专题 | 唯一事实源 | 实施账本 | 当前停点 |
|---|---|---|---|
| π0 × RoboTwin × DSRL | `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md` | 正式训练已在 step 198 收尾；可恢复 DCP 为 step 195 |
| RLToken / RLT × π0 × RoboTwin | `docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md` | formal250及250→480续训均自然完成、exit0；final eval 17/20，续训eval合计178/200，final checkpoint完整 |
| QAM × π0 × RoboTwin | `docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md` | fresh q-only smoke 已 exit0；exact-resume 加固与 39 项服务器回归已完成，后续 smoke/正式训练未批准 |
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
- 2026-07-30 23:03+08:00 最终动态快照：source-aligned
  `8 env × 250 cycles` formal已自然完成，`250/250`、`exit_code=0`；
  `11:37:25–22:23:41`，wall-clock `10h46m16s`。driver/Ray/训练进程均已退出。
  实际阶段为P1 `1–135` reference collect、P2 `136–154` reference+SAC、
  P3 `155–191` student+ramp、P4 `192–250` stable student。
- train成功率P1/P2/P3/P4分别为
  `156/1080=14.44%`、`22/152=14.47%`、`94/296=31.76%`和
  `375/472=79.45%`；最后20/10/5 cycles为83.75%/85%/90%。同一20-seed
  deterministic eval在cycle200/225/250为`16/20`、`11/20`、`18/20`；
  endpoint明显改善，但结论只限单任务与该固定seed协议。
- 最终 critic/actor累计updates为`102,260/51,130`，pending debt为0；actor/critic
  loss为`-0.08119/0.001286`，grad为`2.821/0.170`，Q0/Q1/Q(data)为
  `0.2515/0.2383/0.2123`，BC/Q权重为`2.5/0.45`。全部finite且fatal扫描为空。
- 两卡显存峰19.37/19.51GiB、全程平均利用率30.0%/27.7%；matched RSS、Env RSS、
  cgroup anon峰为87.42/62.05/82.43GiB。运行期间cgroup current触及240GiB且
  `sustained memory pressure/anon growth`风险成立，但OOM/OOM-kill均为0；进程退出后
  matched RSS归零、Env RSS近零、anon降到约0.3–0.6GiB。
- `global_step_250` completion与两个rank trainer state完整；25到250共10个checkpoint，
  总计约5.2GiB，final约876MiB。最终报告见
  `docs/rlinf-robotwin-pi0-rltoken/15_STAGE2_FORMAL_8ENV250_FINAL_RESULT_20260730.md`。
- Stage 2 final高信息量非模型包：
  `exports/rlt_stage2_formal_8env250_high_info_20260730_v1.zip`，大小2,286,757 bytes，
  SHA-256
  `158282a1e38151b39d4e9ba1f6d173855c0bade413c803c29827d325e2771b96`；
  包含完整driver/TensorBoard/resources、source/resolved config、预算/provenance、
  Stage 1 manifest、最终统计/图、关键规划和流水账，不含checkpoint/replay，不能独立resume。
- 当前formal run root：
  `/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1`；
  runtime evidence：
  `/root/autodl-tmp/experiment_exports/rlt_stage2_formal_8env_250c_20260730_v1/runtime`；
  resolved SHA-256：
  `586644cd69461016c1dd8c653da0eea12b01c61f2d0a9b4901654d90800f2a3e`。
  正常停止为250 cycles，18小时timeout只作故障保险；运行已自然结束。
- 用户随后明确授权保持训练语义不变，从`global_step_250`续到绝对总终点480。
  新进程于2026-07-30 23:51:08+08:00启动，并于2026-07-31 09:17:41自然结束，
  `480/480`、`exit_code=0`、wall-clock `9h26m33s`：
  - run root：
    `/root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1`；
  - runtime：
    `/root/autodl-tmp/experiment_exports/rlt_stage2_formal_resume250_to480_20260730_v1/runtime`；
  - experiment：`robotwin_adjust_bottle_rlt_stage2_formal_resume250_to480_v1`；
  - driver/monitor PID：`657385/657386`，现均已退出；
  - resolved SHA：
    `cbbfffda43a6ca17ee938da21d7f71ccb70ba394d1247b8e5ae8d3f48dda5787`；
  - 12小时hard timeout未触发；275–475每25 cycles以及最终480完成fixed-20
    eval并保存，共新增10个checkpoint。
- 续训train为`1690/1840=91.85%`，最后50/20/10 cycles为
  `92.50%/93.75%/91.25%`。10个fixed-20 eval合计`178/200=89%`，前后5点为
  90%/88%，final为`17/20=85%`；相对cycle250的18/20只差1条，属于高位平台波动，
  不能判定明确退化，也没有继续上升证据。
- final critic/actor updates为`215055/107528`，pending0；新增global replay22559、
  critic112795，严格满足UTD5。actor/critic loss为`-0.132776/0.000813`，grad为
  `2.160/0.133`，Q0/Q1/Q(data)为`0.3683/0.3489/0.3306`；全部finite。
- 续训GPU显存峰19.37/19.56GiB，active mean util29.0%/30.1%；matched/Env/anon峰
  78.75/52.46/73.03GiB。cgroup current触及240GiB且`memory.max`增加11,485，但
  OOM/OOM-kill为0；训练结束后matched RSS归零、anon约0.29GiB，live约165GiB file
  cache不是活进程泄漏。
- 新run 10个checkpoint为275–475每25步加480，全部completion=true；总计约11.15GiB，
  final480约1.32GiB、`update_step=215055`。完整启动/健康门见
  `docs/rlinf-robotwin-pi0-rltoken/16_STAGE2_FORMAL_RESUME250_TO480_LAUNCH_20260730.md`，
  最终验收见
  `docs/rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md`。
- 本地高信息量快照位于
  `exports/rlt_stage2_formal_resume250_to480_high_info_20260731_v1`；远端下载核心加
  manifest为51个文件、约5.62MiB，加入分析/图/README后的完整文件夹为57个文件、
  7,654,908 bytes（约7.30MiB）；
  包含完整driver/TensorBoard/resources、config/provenance、completion/replay
  metadata、分析JSON和三张图，不含checkpoint权重，不能独立resume。
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
- Stage 2 formal source 现在 `max_steps=0` fail closed。已完成的100-cycle pilot使用
  2GPU/4env、H50/C10/D14、batch512/128、UTD5/ratio2、500 rows/rank、5k critic
  floor、cap400、15k replay/rank；这些历史参数相对 ManiSkill/论文/RoboTwin 的来源与风险见
  `docs/rlinf-robotwin-pi0-rltoken/04_STAGE2_PRE_SMOKE_PACKET_20260729.md`。
- Stage 2 artifact preflight、formal/fresh/resume compose/audit、Ruff、py_compile和
  19个集中单测已通过；MLP仅2,162,202参数。fresh/resume launch script 服务器
  `bash -n` 和22-field resource-monitor单样本自测通过；docs commit 后的治理 gate 已
  精确允许 `docs/**` 与根 `HANDOFF.md`，同时继续拒绝其他 code/config diff。
- Stage 2 运行代码与批准材料已发布到 `personal/codex/rlt-pi0-robotwin`；smoke source
  HEAD 为 `6fd3ee7106fb82f06eda82603c41a09767151709`。2026-07-30 运行前 main/API/raw
  与 `ls-remote` 全部恢复，一次有界 push 用时3秒，left/right `0/0`。此前变慢是
  `github.com:443` 的瞬时大陆线路不可达，不是仓库、认证或大 pack；短流程见
  `docs/rlinf-robotwin-pi0-rltoken/06_AUTODL_NETWORK_PLAYBOOK.md`。
- Stage 2 fresh smoke 已于
  `2026-07-30T00:22:45+08:00` 至 `00:25:27+08:00` exit0：
  - 真实 train/eval 各1 epoch；
  - global transitions 8、每 rank replay 4；
  - critic/actor updates `8/4`，保存后两 rank `update_step=8`；
  - train `actor_switch_rate=0`，deterministic student eval 已执行；
  - `global_step_1` completion=true、rank0/1完整；
  - actor/critic loss `2.189/0.017`，grad norm `5.055/4.591`；
  - CUDA OOM、NCCL fatal、NaN、Ray actor death与cgroup OOM均为0。
- smoke 两卡显存峰 `15,375/15,368 MiB`，matched RSS峰45.46GiB，cgroup anon峰
  40.99GiB，host available最低940.92GiB，磁盘 available最低825.89GiB。
  run/checkpoint约52.5MB。Curobo/Vulkan提示未阻止 mplib train/eval/DCP。
- smoke run：
  `/root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1/
  robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1`；runtime evidence：
  `/root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1/fresh_runtime`。
  第一次 launcher 在训练程序前因 heredoc 参数数组合并 exit127；失败 evidence 保存在
  同级 `_failed_launcher_127`，窄修复 launcher SHA 为
  `473f339a...abd985`，未改算法/config。
- 用户明确省略本轮 resume；因此 DCP completion 已验证，但不声称新进程
  save→load→continue 已通过。完整结果见
  `docs/rlinf-robotwin-pi0-rltoken/05_STAGE2_FRESH_SMOKE_RESULT_20260730.md`；结果与
  高信息量 evidence 主提交为
  `9bb2dd78feff7133780c3df6a88618d10168c4e4`。
- 启动前按授权精确删除12个旧 RLinf DCP和51个 Motus OPD实验`.pt`，回收约177.85GiB；
  PPO step20、GRPO step100、轻量日志/配置/指标与Motus官方/base权重均保留。删除清单位于
  `/root/autodl-tmp/experiment_exports/rlt_pre_stage1_cleanup_20260729`，被删权重不可恢复。
- 用户曾批准并已完成 **RoboTwin `adjust_bottle`** Stage 2 100-cycle pilot；ManiSkill
  只作参考实现/量级核对，没有进入 resolved config。RLinf ManiSkill 可执行参考是
  5,000 outer cycles、64 train env、500 primitive/cycle和不同任务，不能把它当成本次
  RoboTwin预算。
- formal source 继续 `max_steps=0` fail closed；批准命令显式覆盖
  `runner.max_steps=100`，resolved SHA为
  `efff00b71d8ab618f4a77c082cbec8fd65fda9abe2573def31e0aca980e50178`。
- 100-cycle run于 `2026-07-30T01:15:54+08:00` 启动，并于
  `2026-07-30T03:47:38+08:00` 正常结束：
  - `100/100` cycles，`exit_code=0`，wall-clock `9,104s=2h31m44s`；
  - run root
    `/root/autodl-tmp/experiments/rlt_stage2_formal_100c_20260730_v1`；
  - runtime
    `/root/autodl-tmp/experiment_exports/rlt_stage2_formal_100c_20260730_v1/runtime`；
  - 400 train episodes、40 deterministic eval episodes、7,821 macro transitions；
  - student 于 cycle 27 接管，BC/Q ramp 于 cycle 52 完成；critic/actor updates
    `34,800/17,400`；
  - train 全部 `30/400=7.5%`；student 前/后20 cycles为
    `3/80=3.75%` / `10/80=12.5%`；
  - deterministic eval 在 cycle 60/70 为 `1/4`、`2/4`，cycle 80/90/100
    均为 `0/4`，合计 `3/40=7.5%`；每点仅4条，不能据此选 best 或声称稳定改善；
  - 两卡显存峰值17,543/17,626MiB，active mean util 25.6%/26.2%，负载对称但未顶满；
  - cgroup anon峰值47.47GiB、file cache峰值195.05GiB、memory.max 240GiB；
    OOM/OOM-kill/fatal/NaN均为0，但扩大env并发前需单独验证cgroup压力；
  - 10个 checkpoint completion均完整；最终checkpoint约226.3MiB，run约1.44GiB，
    无视频/图片。
- 首次 launcher 因进程gate匹配自身路径而在训练启动前exit1；只删除该自匹配词后
  `bash -n`与逐行diff通过，算法、config、预算均未改，旧script和失败说明已归档。
- 预算口径已更正：旧100-cycle run永久归类为pilot。它的400条train episodes位于论文
  公开400–1000 episodes的下界，因此不是简单“少50倍”；真正缺陷是启动前没有完成
  多轴规模审批、长期schedule覆盖不足且每个eval点仅4条。完整formal审批不得再用
  “跑一晚”代替cycles/episodes/transitions/updates/eval/checkpoint预算。
- 首段完整2,000-episode预算已按首选方案执行：
  - `8 env × 250 cycles`、cap1600、eval/save每25 cycles；
  - 10k replay rows/rank、30k update floor、20k/50k actor schedule；
  - 预计约39.1k macro rows、125.5k/62.8k critic/actor updates；
  - 周期监控每点20个唯一fixed seeds，共10点；恰好10个full checkpoints。
- exact-20评估只通过RLT专属overlay和独立官方seed bank启用，没有修改通用
  EnvWorker/RoboTwinEnv，也不影响PPO、GRPO或旧RLT。集中测试为27 passed。
- 8-env资源门完成3 cycles、24 train episodes、472 macro transitions、
  3200/1600 critic/actor updates和20条deterministic eval；两卡显存峰
  `17,169/17,252MiB`，matched RSS峰51.88GiB、cgroup anon峰47.47GiB，
  high/OOM/OOM-kill增量均为0。完整资源门与formal启动记录见
  `docs/rlinf-robotwin-pi0-rltoken/10_STAGE2_FORMAL_8ENV250_LAUNCH_20260730.md`。
- `global_step_250`的结构resume已经由本次真实load→continue补齐：runner绝对总终点480，
  新旧resolved只变化resume/终点/新输出及两个派生视频目录；原formal250 checkpoint未覆盖。
- 当前授权停点：续训已自然完成，当前没有RLT训练/Ray进程，两卡空闲。本轮只读验收没有
  启动、停止、重启、删除、覆盖或清cache。独立held-out评估、EnvWorker内存定位、
  继续延长或扩展实验仍须另行明确授权。480最终报告见
  `docs/rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md`；
  首段250最终报告见
  `docs/rlinf-robotwin-pi0-rltoken/15_STAGE2_FORMAL_8ENV250_FINAL_RESULT_20260730.md`；
  cycle200历史报告见
  `docs/rlinf-robotwin-pi0-rltoken/14_STAGE2_FORMAL_8ENV250_STATUS_CYCLE200_20260730.md`；
  完整旧pilot结果见
  `docs/rlinf-robotwin-pi0-rltoken/08_STAGE2_FORMAL_100C_RESULT_20260730.md`；启动记录见
  `docs/rlinf-robotwin-pi0-rltoken/07_STAGE2_FORMAL_100C_LAUNCH_20260730.md`；下一次设计见
  `docs/rlinf-robotwin-pi0-rltoken/09_STAGE2_NEXT_FORMAL_SCALE_DESIGN_20260730.md`；
  当前formal启动记录见
  `docs/rlinf-robotwin-pi0-rltoken/10_STAGE2_FORMAL_8ENV250_LAUNCH_20260730.md`。
- 2026-07-31 Git收尾：服务器
  `codex/rlt-pi0-robotwin@2b8199d8ab2e7b110994fd3234bf7007196c3af9`已通过一次性
  AutoDL `/etc/network_turbo`子shell fast-forward推送到
  `personal/codex/rlt-pi0-robotwin`；push前远端为`9bb2dd78...`，首轮push后两端HEAD
  一致。没有持久化proxy、改remote或使用旧Windows dirty镜像；最终docs-only记录提交
  与复核见本专题账本A144。

## QAM 当前状态与授权

- 方法与命名已锁定为
  `Plain-QAM π0 online adaptation（B1 frozen behavior + F1 full fine expert + C1 +
  fixed-N20 M2 + online-only）`；算法真值是
  `ColinQiyangLi/qam@2726d767c9a0a7a46d49693f0391f73dc2cf58ac`。首版不做
  QAM-F/QAM-E/F2/C2/M1，clean-50 不进 Q loss。
- 独立服务器 worktree：
  `/root/autodl-tmp/RLinf_qam_pi0_robotwin`，branch
  `codex/qam-pi0-robotwin`，从 common commit `6d0db56...` 建立。生产继续只读复用
  `/root/autodl-tmp/RLinf/.venv`；官方 oracle 使用
  `/root/autodl-tmp/venvs/qam-oracle-2726d767`，shared venv 未安装或升级依赖。
- 实现已覆盖官方 JAX fixture parity、`t_qam=1-t_pi0`/速度翻转、B1/F1、C1 四块
  prefix+proprio、10 个完整 Q、fixed-N20 macro replay、collect/q_only/am_on gate、
  target EMA、FSDP-compatible VJP/streaming AM、rank sidecar/replay resume 与 legacy
  opt-in。replay 自身 RNG/world-size 与 phase/credit helper 已覆盖；fresh 后正在补
  worker rank-local process RNG、统一 snapshot ID 与跨 rank QAM completion manifest。
  旧 fresh checkpoint 缺这些字段，只作 fresh 证据，不能追认为 exact-resume 起点。
  实际 code files 和逐行流水只在 QAM SSOT/账本维护。
- 本轮 exact-resume、warm-up credit 与 time-limit 分类实现提交为
  `851db175fb8e9743585bbbdcd90298741fa910e0`，已直接推送
  `personal/codex/qam-pi0-robotwin`；该次 GitHub main HTTP 200、`ls-remote`
  成功，无需启用 `/etc/network_turbo`。最终动态 HEAD/dirty/ahead-behind 仍须在下次
  操作前现场刷新。
- 真实模型事实：F1 为 173 tensors / 314,713,120 参数；C1 block 长度
  `[256,256,256,48]`、valid `[256,256,256,5]`；raw active endpoint 越界
  `20/280=7.14%`，Q 与 env 已共用 canonical clamp；replay 4,096/rank，约
  11.3 GB raw observation/rank。
- 服务器集中回归为 `36 passed, 3 dependency warnings, 8.16s`；
  Ruff/format/compile/diff 均通过。
  两卡真实 `FULL_SHARD + use_orig_params`、K=10、batch=32/rank probe 峰值
  14,148,494,848 bytes/卡，F1 173 个梯度 finite、10-Q/target 跨 rank 一致。
- 2026-07-31 14:57–15:01 的 fresh `q_only` smoke 已自然 exit0：2 GPU/2 env、
  2 trajectories、20 global macro、每 rank replay 10、恰好 2 次 critic update、
  0 fine update、policy version 0；critic/target/optimizer 跨 rank 一致、10-Q 独立、
  全 tensor finite，保存约 11.64 GB `global_step_1`。
- 原资源 monitor 因 SFTP mode 644 被直接执行而 exit126；训练未受影响。launcher 已改为
  `bash monitor.sh`，3 秒假 driver 窄复测 exit0。恢复 monitor 覆盖训练尾部 22 秒，
  GPU 峰 23,486/23,677 MiB、cgroup anon 峰 34.34 GiB、OOM/OOM-kill 0/0；不把它
  声称为初始化峰值。
- 完整 source/smoke resolved、穷尽 diff、精确命令、driver log、资源 CSV/summary、
  sidecar validation 和 inventory 见 QAM 账本 `QAM-SMOKE-0001` 至 `0003` 以及
  `docs/rlinf-robotwin-pi0-qam/evidence/qam_qonly_smoke_20260731_v1/`。
- 按用户精确授权，smoke 结束后只删除 DSRL N20、Fast-WAM GRPO/PPO、RLT Stage-1
  四组旧 smoke 的 `checkpoints/`，回收 147,110,981,632 B / 137.01 GiB；小型证据与
  所有 formal DCP 保留。磁盘 available 增至约 933 GiB、使用率 50%。完整清单见
  `QAM-CLEANUP-0001`、`evidence/qam_delete_old_smoke_checkpoints_20260731.sh` 和
  `evidence/qam_old_smoke_checkpoint_cleanup_20260731.txt`。旧 WAM/PPO backup
  110.94 GiB 未动。
- 18 小时正式候选暂按 collect 512 macro、q_only 512–1,024 critic update、约 1 h
  诊断/阶段保存、通过 Q 动作梯度门后把余下约 10 h 给 am_on；这是 online-only
  稳定化适配，不是官方阶段比例。正式 v1 可在 fresh `q_only` 进程内完成前 512 条
  collect warm-up，不需要 collect→q_only resume；当前 batch1 smoke 不能给 batch32
  正式吞吐。
- 当前授权停点：可继续完成 exact-resume 代码与服务器前置测试；fresh→resume、
  production-batch q-only 诊断、`am_on` smoke 和正式训练均需分别展示完整配置/命令/
  预算并取得新批准。锁定的 sparse route 已证明 `truncated && !terminated` 是
  time limit，且 `auto_reset=false` 保存 true query-final observation；当前只需验证
  QAM-only 分类补丁，不再把 timeout 类型当开放设计分支。

## 共同执行边界

- 动态事实以新的服务器只读现场检查为准；旧快照只用于定位。
- Windows 本机只读写文档、代码和 diff；Hydra compose、项目 import/compile、测试、
  smoke 和训练在服务器进行，并遵守对应专题授权。
- 机器间Git角色固定为：服务器独立worktree是运行时真值，云端同名分支是持久协作与
  灾备真值，Windows源码镜像是可从云端重建的编辑副本，`.tmp` bundle/relay只是一次性
  传输物。不得用旧或dirty的Windows镜像反向覆盖服务器；本机冗余只在云端核验完成、
  精确列出目标/大小/可恢复性并另获删除授权后清理。
- smoke/formal 前必须展示完整 resolved config、精确命令、输出目录、资源计划、观测指标
  和停止条件，等待明确批准。
- 不删除用户数据、不停止无关进程、不重装依赖、不下载大模型、不覆盖既有 checkpoint。
- 仓库改名 `rlinf_fastwam` → `rlinf_exp` 仍只是待核实候选；未明确目标前不执行。

任何进程、GPU/RAM、日志、checkpoint、HEAD、dirty tree 或远端仓库状态，只有本轮现场刷新后
才能称为“当前”。
