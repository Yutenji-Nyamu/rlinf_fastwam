# AutoDL / RLinf 当前交接入口

最后更新：2026-07-28。本文只保留并行专题路由、当前停点和授权边界；旧时间线见
`docs/project-history/00_INDEX.md`。进入某个窗口时只选择对应专题，不默认加载另一专题正文。

## 并行专题路由

| 专题 | 唯一事实源 | 实施账本 | 当前停点 |
|---|---|---|---|
| π0 × RoboTwin × DSRL | `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md` | 截至 23:04 完整到 step 53，训练继续运行 |
| RLToken / RLT × π0 × RoboTwin | `docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md` | 规划 v4 已完成实现前语义收口；等待“开始实现”授权 |
| QAM × π0 × RoboTwin | `docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md` | 上下文/source lock v1 已完成；等待方法分叉选择与“开始实现”授权 |
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

## RLT 当前状态与授权

- 唯一主线：单任务 `adjust_bottle`；Stage 1 使用全部有效 clean-50、无 val split、冻结 π0
  只训 RL token；checkpoint `norm_stats` 为 Stage 1/Stage 2/decode 单一真相。
- Stage 2：H=50/C=10、14D canonical action、`z_rl=2048`、full-task train route、
  deterministic student eval、无人/无 expert、compact replay。
- 首批正确性修复：RLT trainer state、feature/action/schedule resume contract、per-rank
  warm-up 语义和首 rollout full sync；不做 hard capacity、RNG exact、async 或跨 world-size。
- 2026-07-28 15:12 的只读快照：`/root/autodl-tmp/RLinf@6d0db56b` clean、两卡 A800
  空闲、π0 `.venv` 的 RLT imports 成功、`adjust_bottle` SFT checkpoint 及
  `physical-intelligence/robotwin/norm_stats.json` 存在；常见数据根未找到原 LeRobot
  dataset。该状态已经过时风险，实施前须轻量刷新。
- 官方 clean-50 与两段 converter 已核验，但尚未下载、解压或转换。
- 本轮授权仅覆盖本地上下文/设计文档整理和只读材料审查；尚未授权创建服务器 RLT
  branch/worktree、改代码、下载数据、运行前置测试、smoke 或训练。

## QAM 当前状态与授权

- 官方方法锁定为 arXiv v4 与
  `ColinQiyangLi/qam@2726d767c9a0a7a46d49693f0391f73dc2cf58ac`；上游是
  JAX/Flax OGBench flat-state 实现，没有 π0、视觉/语言 VLA 或 RoboTwin 代码。
- 当前组合边界：官方 QAM 提供方法真值；RoboTwin π0 PPO/GRPO 提供环境、transform、
  checkpoint、rollout 和 sync 数据面；NFT/OpenPI 提供显式可微 velocity；SAC-Flow
  提供 off-policy 工程壳；DSRL/RLT 只提供治理、replay/resume/sync/验收范例。
- 首版只考虑 plain、action-space QAM。不得继承 DSRL 的 latent/reward/H/N/UTD/capacity，
  也不得用 PPO/GRPO/NFT/SAC loss 替代 adjoint matching。
- 已确认 OpenPI 与官方 QAM 的时间方向和 velocity 符号相反，必须显式使用
  `t_qam=1-t_pi0`、`f_qam=-v_pi0`；Q 只允许评分真实执行的 `N×14` action prefix，
  未执行 suffix/padding 的终端梯度必须为 0。
- 2026-07-28 18:56–19:21 的只读快照：服务器没有 QAM branch/worktree/代码；π0 SFT
  checkpoint 和 norm stats 存在；定向搜索未发现 clean-50 或可复用 PPO/GRPO replay；
  19:21 时 DSRL driver PID `70062` 仍在两卡运行，GPU 约 29.8/29.5 GB，cgroup
  memory 约 230.2/240 GiB 且无 OOM；本专题不干预，也不在该状态启动 QAM probe。
- 当前未决的前三项方法选择是：primitive-faithful 或 macro-QAM；frozen π0 behavior
  base 或保留 FM/EMA 的 trainable slow clone；完整 trainable fine expert copy 或
  residual adapter。先做官方 JAX→PyTorch 小网络 oracle 和真实 π0 VJP/显存探针，
  再据证据冻结。
- 当前只授权本地专题文档与服务器只读审查；未授权创建 QAM branch/worktree、改服务器
  代码、下载数据、运行 compose/import/test、smoke 或训练。
- 后续 QAM 窗口只需读根上下文、本节路由、QAM 主计划和账本最新批次；精确来源按
  `docs/rlinf-robotwin-pi0-qam/01_CONTEXT_AND_SOURCE_MAP.md` 查，不再默认加载 DSRL/RLT
  全文。

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
