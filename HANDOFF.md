# AutoDL / RLinf 当前交接入口

最后更新：2026-07-29。本文只保留并行专题路由、当前停点和授权边界；旧时间线见
`docs/project-history/00_INDEX.md`。进入某个窗口时只选择对应专题，不默认加载另一专题正文。

## 并行专题路由

| 专题 | 唯一事实源 | 实施账本 | 当前停点 |
|---|---|---|---|
| π0 × RoboTwin × DSRL | `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md` | 正式训练已在 step 198 收尾；可恢复 DCP 为 step 195 |
| RLToken / RLT × π0 × RoboTwin | `docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-rltoken/evidence/IMPLEMENTATION_LOG.md` | 独立分支主体实现和 pre-smoke 已完成；停在 clean-50/Stage 1 smoke 批准前 |
| QAM × π0 × RoboTwin | `docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md` | `docs/rlinf-robotwin-pi0-qam/evidence/IMPLEMENTATION_LOG.md` | 规划/source lock v2 与 D1 下载对象锁已完成；ZIP/schema 未验；等待 macro/primitive 选择及“开始实现”授权 |
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
  不做 hard capacity、RNG exact、async 或跨 world-size。
- 服务器独立 worktree：
  `/root/autodl-tmp/RLinf_rlt_pi0_robotwin`，
  从 `48a775db09c16c455aeba7b0600c920e7c80d534` 建立；主体实现基线为
  `cfa556550efa7da1779a0d29c3a34b00a7f17ed8`，后续 clean-50 与配置复核只提交文档证据。
  分支为 `codex/rlt-pi0-robotwin`，没有切换或修改 DSRL worktree，也没有复制/安装环境。
- 2026-07-29 14:42 现场：两张 A800 均 `0 MiB/0%`，无 Ray/RoboTwin/训练进程；
  RAM available 778 GiB；`/root/autodl-tmp` 可用 694 GiB、63% used。
- AST/Ruff/whitespace、25 个聚焦测试通过；9 份 candidate/legacy resolved config
  经原生 compose 和无 Ray `validate_cfg` 通过。真实 checkpoint 探针确认 image prefix
  `[1,768,2048]`、`z_rl=[1,2048]`、只有 `rlt_module.*` 可训练、token module
  743,094,272 参数，canonical reference decode 与旧 transform max-abs `0.0`。
- clean-50 原始 ZIP 已从
  `TianxingChen/RoboTwin2.0@9dc9299c163db059931898a9f0852098a61155a1`
  下载到 `/root/autodl-tmp/datasets/robotwin2/source/<revision>/dataset/adjust_bottle/`；
  `298659710` bytes、SHA256 `5554b6b3...be50e`、ZIP test、50 组
  `pkl+hdf5+mp4+instruction` 和 path safety 均通过。尚未解压或转换。
- 四份 source config 和完整 resolved YAML、参数来源、磁盘说明及 fresh/resume 验收见
  `docs/rlinf-robotwin-pi0-rltoken/01_CONFIG_PROVENANCE_AND_PRE_SMOKE_PACKET.md`。
- 当前下载授权已完成；尚未授权解压/转换、Stage 1/Stage 2 smoke、训练、安装依赖或磁盘
  清理。下一步先批准单 episode converter，以及 Stage 1 两卡
  S1-A micro1/global2 correctness + S1-B formal micro16/global32 batch-fit；Stage 2 必须等
  真实 Stage 1 endpoint/manifest/hash。Stage 2 当前 UTD5/ratio2 是论文导向 candidate，
  不是 RLinf ManiSkill YAML 的有效 UTD1/ratio4，正式 Stage 2 packet 再显式批准。

## QAM 当前状态与授权

- 官方方法锁定为 arXiv v4 与
  `ColinQiyangLi/qam@2726d767c9a0a7a46d49693f0391f73dc2cf58ac`；上游是
  JAX/Flax OGBench flat-state 仿真实现，没有真机、π0、视觉/语言 VLA 或 RoboTwin
  代码。2026-07-29 复核 HEAD 未漂移；官方 plain 结构是 trainable behavior FM/EMA +
  独立完整 fine flow + 10-Q，不是 residual 默认。
- 当前组合边界：官方 QAM 提供方法真值；RoboTwin π0 PPO/GRPO 提供环境、transform、
  checkpoint、rollout 和 sync 数据面；NFT/OpenPI 提供显式可微 velocity；SAC-Flow
  提供 off-policy 工程壳；DSRL/RLT 只提供治理、replay/resume/sync/验收范例。
- 用户已确认首版做 Plain、action-space QAM；QAM-F/QAM-E 延后。不得继承 DSRL 的
  latent/reward/H/N/UTD/capacity，也不得用 PPO/GRPO/NFT/SAC loss 替代 adjoint
  matching。核心语义是：数据动作只给 behavior FM；target-Q 只在 clean final action
  给一次 action gradient，再由 behavior VJP 搬回各 flow time 形成 AM 监督，不能称为
  “Q 给每个去噪步做 FM 标签”。
- 已确认 OpenPI 与官方 QAM 的时间方向和 velocity 符号相反，必须显式使用
  `t_qam=1-t_pi0`、`f_qam=-v_pi0`。Q/VJP/replay 使用 $P_N$ 选择的 normalized
  14D model action；环境另走既有 `Unnormalize+AlohaOutputs`。实际执行 $L<N$ 时两套
  action 都按 mask 记录，Q gradient 不穿 robot transform，$L{:}N$/suffix/padding
  终端梯度必须为 0。
- 用户已选 D1 offline+online。下载锁为
  `TianxingChen/RoboTwin2.0@9dc9299c.../dataset/adjust_bottle/aloha-agilex_clean_50.zip`，
  298,659,710 bytes，SHA-256
  `5554b6b30e37c6ed2f0bbc48079e8ad79d9512e9d4f910a5e71b0d5ad8fbe50e`。
  共享 raw ZIP 已由 RLT owner 下载到版本化 source 目录并完成 size/SHA-256、ZIP
  完整性和 archive 路径安全检查；QAM 本身仍未解压、转换或写 sidecar，单 episode
  schema/mask 合同也尚未验收。
  B2 时 clean-50 主要作 behavior/FM；B1 时只作 derived-success 接口/critic 温启动探针。
  reward/end 是带 provenance 的成功轨迹派生 sidecar，observed failure/timeout 以 online
  replay 为权威。RLT/QAM 共用只读 raw/canonical，各自维护 sidecar，实际下载/转换前
  确定唯一 owner；下载后只做 hash/archive/单 episode schema 与 mask 前检。
- 2026-07-29 12:33–12:34 的只读快照：DSRL/Ray/监控已退出；两张 A800 均
  `0 MiB/0%`；cgroup 无 OOM；约 694GB 可用。RLT worktree 已建立在
  `codex/rlt-pi0-robotwin@48a775db...`；QAM worktree、目标 ZIP 和 QAM 数据目录仍
  不存在；π0 SFT checkpoint/norm stats 存在。实施前仍须重刷。
- 生产端不复制/改名 14GB shared venv，也不往里安装官方 QAM；独立 QAM worktree 继续
  显式使用 `/root/autodl-tmp/RLinf/.venv`。官方 JAX 小网络 oracle 使用独立 pinned
  source tree/CPU venv，导出 `.npz` 供 PyTorch tests。
- 现有 runner/rollout 调度保持不变：rollout 的 OpenPI `hf_model` 仅在双重 opt-in 时
  注册实际推理需要的 active fine route；QAM actor worker 持有 critic/target/optimizer/
  replay，并只同步推理参数，trainer-only 状态不得进入 rollout。
- 生产 v1 当前收束推荐为 `B1+F1+C1+M2+N20`：frozen SFT behavior、完整 fine action
  expert、frozen π0 三视角 prefix+proprio 后接 10 个独立完整 Q MLP、macro replay、
  实际提交 20 步（模型宽度仍 50）。准确名称是
  “Plain-QAM π0 adaptation（frozen behavior + macro transition）”；该整包等待用户
  确认。P1 小网络仍无条件 exact 复现官方 B2+F1/FM/AM/10-Q/EMA；生产备选只在 F1
  超显存、C1 表示失败、M2 transition 不成立、N20 控制/credit 失败或 frozen behavior
  明显失配时按专题 §4.4 窄触发，不并行实现。
- clean-50 是 50 个成功 episode，适合动作/norm 合同、成功正例和可选 critic 温启动，
  但缺原生 reward/end/failure/timeout/query boundary，不能单独训练出可信 action-Q。
  生产先采 frozen-SFT online warm-up transition，`tau=0` 训练 10-Q；只有 Q 能基本区分
  executed/扰动 action 且 action gradient finite，才打开 AM。
- 当前只授权本地专题文档与服务器只读审查；未授权创建 QAM branch/worktree、改服务器
  代码、下载数据、运行 compose/import/test、smoke 或训练。
- 后续 QAM 窗口默认只读根上下文、本节路由、QAM 主计划和账本最新批次；来源争议查
  `01_CONTEXT_AND_SOURCE_MAP.md`，方法教学/选择查
  `02_METHOD_AND_PORT_DECISION_GUIDE.md`，不再默认加载 DSRL/RLT 全文。

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
