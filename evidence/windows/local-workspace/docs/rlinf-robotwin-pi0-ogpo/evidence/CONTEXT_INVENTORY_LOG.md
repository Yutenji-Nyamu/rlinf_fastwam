# OGPO 上下文整理账本

当前结论以最新条目 CTX-0015 为准；更早条目保留当时讨论轨迹，冲突内容不再进入实施计划。

本账本只记录 `OGPO × π0 × RoboTwin × RLinf` 专题的上下文发现、裁剪、来源核验与文档变更。
2026-08-07 已按授权在独立服务器 worktree 完成主体实现、定向测试和首次真实 RoboTwin smoke；
pilot 与 formal 训练均未运行。CTX-0007 中的旧设计已由后续条目明确取代。

## CTX-0001：建立专题前的入口与边界核对

- 日期：2026-08-05。
- 已完整读取：根 `PROJECT_CONTEXT.md`、根 `HANDOFF.md`。
- 已读取的专题入口：
  - `docs/fastwam-robotwin-rlinf-grpo/00_INDEX.md`；
  - `docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md` 的来源分类与可复用资产章节；
  - `docs/rlinf-robotwin-pi0-rltoken/00_INDEX_AND_IMPLEMENTATION_PLAN.md` 的上下文路由与来源优先级章节；
  - `docs/rlinf-robotwin-pi0-qam/00_INDEX_AND_IMPLEMENTATION_PLAN.md` 的来源、π0 合同、主线与收尾章节；
  - `docs/project-history/00_INDEX.md`。
- Git 基线：本机 `C:\Users\86136\Documents\rl` 是无提交的资料工作区，`master` 上所有内容均为未跟踪文件；不对它执行提交或推送。
- 授权边界：用户要求先维护、修剪上下文并建立 OGPO 专题文档；服务器未开，先讨论。

## CTX-0002：官方来源在线核验

- OGPO 项目页：`https://simchowitzlabpublic.github.io/ogpo-site/`。
- OGPO 论文：`https://arxiv.org/html/2605.03065v4`，v4 日期为 2026-06-26；项目页标注 ICML 2026 accepted。
- OGPO 官方代码：`https://github.com/simchowitzlabpublic/OGPO_public`，公开仓库、MIT；本轮已核对 README、`ogpo/agents/ogpo.py`、`ogpo/runners/online_rl_runner.py`、`ogpo/configs/algos/` 与 `scripts/README.md` 的入口。2026-08-05 在线参考 pin 为 `0b3be413cde766a41257c6b19c0c2b06393a557f`，当前无 release/tag。
- RLinf 官方 RoboTwin 页面：`https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html`；列出 π0 + PPO 配置 `examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml`。
- 2026-08-05 在线参考 pin：RLinf `36baa75031ff863449322b548be49bd5620698ef`；OpenPI `15a9616a00943ada6c20a0f158e3adb39df2ccac`；RoboTwin main `13c3c47ff4312dd62484bcd51be034af55c062d1`。RLinf 官方 RoboTwin 指南要求的 `RLinf_support` 分支对应 `0008ae6800df9f75fc8de7098bacb01735fd8fd2`，因此 RoboTwin 的 main 与集成分支必须分开记录。
- 直接 `git ls-remote` 曾被连接重置，但官方页面、raw 源码和精确 commit 路径随后均已核验；这些 pin 是本轮在线参考锁点，尚不是服务器实现基线。

## CTX-0003：第一轮裁剪结论

- 采用“按职责分层”，不使用会混淆语义与工程价值的一条总排名。
- OGPO 论文/官方代码只负责 OGPO 方法真值；RLinf π0 PPO、OpenPI 与 RoboTwin 官方实现负责目标系统与接口真值。
- DSRL、RLT、QAM、Fast-WAM PPO/GRPO 和操作流水账只按需要提供窄参考，不能改写 OGPO 的目标、损失、更新顺序或数据合同。
- QAM 运行没有证明“QAM 一般失败”，但没有建立可用 action-gradient 学习；它对 OGPO 的主要价值是负面证据：不要依赖 `dQ/da` 或穿过生成链的 BPTT。OGPO 官方方法本身正好采用 zeroth-order PPO，避免两者。
- Fast-WAM PPO/GRPO 历史没有形成严格可比、固定 seed 的算法失败结论；它们主要用于 logprob、同步、FSDP、Ray、指标与资源故障清单。

## CTX-0004：待补事项与停止边界

- 四个官方上游已记录 2026-08-05 精确参考 pin；真正实现前仍需明确选择 RLinf release/main 与 RoboTwin `RLinf_support` 的兼容组合，并把所选基线写入独立 worktree 的实施 source lock。
- 等服务器开机后，先只读刷新身份、worktree/HEAD/dirty、进程、GPU/RAM、磁盘、模型/数据/日志/checkpoint 路径，再决定实现基线。
- 在实现前讨论并冻结：首版是否严格复现 vanilla OGPO、π0 中“full GCP”对应的可训练参数边界、RoboTwin action chunk/transition 定义、critic observation 与 online-only/offline 初始化范围。

## CTX-0005：本机材料盘点与降温

- 2026-08-05 只读盘点：`docs/` 360 个文件 / 24.46 MiB；`exports/` 554 个文件 / 446.25 MiB；`audits/` 377 个文件 / 91.35 MiB；`.tmp/` 7,002 个文件 / 324.86 MiB。
- `.tmp/` 包含 bundle、源码镜像及浏览器 profile 噪声；不按文件数误判为 7,002 份有效证据，默认退出 OGPO 上下文。
- DSRL/RLT/QAM 的本机导出是轻量、高信息量包；大型 DCP、Stage checkpoint、模型、数据与完整 run root 没有全部复制到本机。服务器端是否仍存在必须等开机后现场核验。
- 7 份 E: 盘历史原始流水账仍存在；部分可能包含敏感登录记录。本轮只确认存在，不打开敏感内容，不把凭据复制进专题文档。
- `.dsrl-impl-worktree`、`.rlt-impl-worktree`、`.qam-impl-worktree`、`.rlinf-fastwam-worktree` 均含历史修改或未跟踪文件，只作证据源；OGPO 实现不得从这些 dirty worktree 直接起分支。

## CTX-0006：源码级调用流与 paper-code 差异核验

- 完整核对本机 `.research-rlinf` 的 π0 PPO RoboTwin 入口、runner、rollout/env、actor、OpenPI
  wrapper、trajectory、FSDP/sync/checkpoint 主链，并用 `RLinf@36baa750` 官方 raw 源码核对标准路径
  没有语义漂移；本机镜像只作离线 path/symbol 地图，不作为实现基线。
- 核对 OpenPI `Pi0Config`、prefix/suffix、action expert、action horizon/dim 与 freeze filter：OpenPI
  普通非-LoRA 默认 full fine-tune；`train_expert_only=true` 是 RLinf π0 PPO 的集成选择。
- 核对 RoboTwin `RLinf_support@0008ae68`：外部一次提交 `[H,14]` chunk，只返回 chunk-level
  reward/done/next observation；块内有效 policy waypoint 数没有暴露，低层控制步不能替代。因此
  首版不能伪造 effective-h。
- 通过官方网页和只读 raw 源码核对 `OGPO_public@0b3be413` 的 `ogpo.py`、`pg_helper.py`、
  `q_helper.py`、`datasets.py`、`online_rl_runner.py`、`ogpo.yaml` 与 state/image 脚本；源码只输出到
  当前进程，没有 clone、落盘代码包或模型下载。
- 发现并登记的高影响差异：
  - 论文 whole-chain product 与代码默认 per-denoise-step/per-action-dimension normalization 不同；
  - 官方主脚本是 OGPO+CA + BoN=8 + success-buffer critic update 的复合配置；
  - success buffer 不足时当前 runner 会在普通 replay 上做 BC，违反论文 success-only 定义；
  - practitioner guide 对同步更新推荐 `subsample`，当前同步脚本却覆盖为 `mean`；
  - sampler 的 old logp 使用 pre-clip final sample、stored chain 却是 post-clip，饱和时破坏
    `current==target -> ratio==1`；
  - rolling checkpoint 不保存 online RNG、环境状态、未完成 episode/action queue，不是精确续训。

## CTX-0007：主计划升级与首版决定

- 将唯一 SSOT 从 `00_INDEX_AND_CONTEXT_PLAN.md` 重命名为
  `00_INDEX_AND_IMPLEMENTATION_PLAN.md`；新增 `02_CALL_AND_DATA_FLOW.md` 作为事实附录，更新根
  `HANDOFF.md` 与 `01_REFERENCE_MATRIX.md` 路由，没有建立第二份规范性计划。
- 决定一套超集实现：主目标 OGPO+CA，vanilla 只作语义验真，OGPO+ 作消融；BoN 和 success-Q
  独立且首版关闭，无成功样本时 success-BC 为 0。
- 决定沿用 RLinf frozen VLM + full action expert/projection，关闭 PPO value head；critic 只看部署
  可见 frozen π0 feature + proprio，不新增 privileged state。
- 决定首版以 whole submitted RoboTwin chunk 构造 macro transition，使用 chunk reward 与
  `gamma^H`；为避免把现有 PPO 的 macro `gamma=.99` 误变成 `.99^H`，主配置由
  `gamma_macro` 反推 `gamma_primitive=gamma_macro^(1/H)`；严格 effective-h 留待环境接口显式扩展。
- 决定 whole-chain scorer 按官方代码 normalization 建 JAX-PyTorch fixture，但修复 final clip
  identity：likelihood 保存 raw latent chain，Q/环境另用 clipped/projected action。
- 决定 online/success replay 从空开始、完整 episode flush；长期 replay 不保存 behavior chain、
  value、old logp 或整块 `forward_inputs`。
- 本轮只修改本机文档；未连接服务器，未修改项目代码/配置/依赖/运行产物，未测试、smoke 或训练。

## CTX-0008：2026-08-07 服务器只读 source/resource freeze

- 使用现有 Paramiko helper、process-only credential 和固定 host-key 完成身份探针：当前容器
  `autodl-container-nekaqbwt43-6ce5babb`，工作目录 `/root`，UID 0；凭据未写入文件或文档。
- 资源：两张 NVIDIA A800-SXM4-80GB 均为 0 MiB/0% util；RAM 1.0 TiB、available 约 982 GiB；
  `/root/autodl-tmp` 可用约 816 GiB；无训练或 Ray driver，仅平台 Jupyter/TensorBoard 等常驻进程。
- RLinf 公共 worktree：`/root/autodl-tmp/RLinf@6d0db56bf26f...`，local `origin/main` 同 commit；
  含用户未跟踪 A800 PPO/GRPO configs 与 `local_scripts/`，不得直接修改。
- 独立 worktree：DSRL `48a775db`、QAM `ff8e28ef`、RLT `2b8199d8`，本轮 `git status --porcelain`
  均为空；它们只作窄参考，不作为 OGPO 分支父线。
- 目标 SFT 模型存在于 `/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle`，
  约 7.6 GiB；RoboTwin norm SHA-256 为
  `649ed92b431bd70627febdb00b2385e35fcab5088e72a4e4a4845585f8ce4f6a`。
- 共享 `/root/autodl-tmp/RLinf/.venv`：Python 3.11.14、Torch 2.6.0+cu124、Ray 2.55.1，JAX/
  Flax 已安装；console scripts 含该绝对 shebang，因此未来 worktree 复用环境并显式设置
  `PYTHONPATH/REPO_PATH`，不复制或改名 venv。服务器没有 OGPO clone，本轮未下载或安装。
- live config/source：π0 `H_model=50,D_model=32,D_env=14,K=4,train_expert_only=true`；
  `RoboTwinEnv.chunk_step` 只有块末 observation/synthetic final-slot reward，但 singleton
  `RoboTwinEnv.step` 已可用，为 OGPO-only primitive trace 提供现成底层接口。

## CTX-0009：用户讨论后的主线收敛（取代 CTX-0007 冲突项）

- 运行时只实现 OGPO+CA：success-only flow-BC + per-Q-head sign-consensus CA；不提供 vanilla/
  OGPO+ 配置、pilot 或消融。`use_success_buffer_q=false`，`BoN=1`。
- 取消 whole-submitted-chunk macro 近似和 `gamma_macro` 开根设计。π0 保留 `H_model=50`，执行
  `C=10`；EnvWorker opt-in 逐 waypoint 调用 singleton step，replay 存 primitive transition，sequence
  sampler 构造真实 `h<=10`、`Σ gamma^i r_i + gamma^h Q`。首版 primitive `gamma=.99`。
- C=10 的依据：RLT 在同一 π0/RoboTwin 路径完整验证；DSRL C=20 是工程旁证；QAM C=20/
  280D macro credit 的负面结果支持缩短 TD/action horizon。singleton trace 下 C=10/C=50 都会
  保存 primitive rows，因此不把 replay 行数当依据；C=10 的依据是每 episode 20 次而非 4 次闭环
  决策、140D 而非 700D critic action 输入，以及 RLT 的同系统完整先例。
- critic：10 个独立 FP32、5×512 + LayerNorm Q；输入为 QAM 已实现的三图像+语言四块 frozen
  prefix pooling、14D proprio、flattened `[10,14]` canonical action；target aggregation 固定 mean，
  因为当前官方 runnable/PaliGemma scripts 与论文超参表均用 mean，CA 仍读取完整 ensemble。
- actor：frozen VLM + full action expert/projections + EMA target expert，关闭 value head；logical VLM
  参数只一套，但 rollout 仍有 RLinf 所需 inference replica。G=32、UTD-Q/PI=1 抄 OGPO；K=4 抄
  resolved π0，不强改成官方小 actor 的 K=10。
- whole-chain raw tensor仍为 `[K+1,50,32]` 以运行 π0 velocity/scorer，但 likelihood、ratio、Q 和
  success BC 只聚合/监督实际执行的 C=10、active 14 coordinates。raw likelihood chain 与
  clipped/projected Q/env action 分开，修复官方 pre/post-clip identity mismatch。官方没有 H50/C10
  分离，因此 executed mask 明记为待 score-function fixture 的项目适配；若耦合情形不成立，smoke
  前收敛回 full raw-joint ratio，不提供双 runtime。
- replay primitive row 立即进入 online ring；episode 成功后只把 row IDs 加入 success view。长期
  不存 value/old-logp/full prediction/chain/forward payload；无成功样本时 BC=0，不用普通 replay
  fallback。
- 拟建 worktree `/root/autodl-tmp/RLinf_ogpo_pi0_robotwin`、branch
  `codex/ogpo-pi0-robotwin`，父 commit `6d0db56b`。当前未获创建或实现授权，未执行该动作。

## CTX-0010：简化与源码边界修正（取代 CTX-0009 冲突项）

- 用户确认 `C=10`。选择依据只保留同系统 RLT 先例、20 次闭环重规划、最长 10-step TD 和 140D
  critic action；撤回“QAM C20/280D 是缩短 horizon 的反证”这一无依据因果解释。
- actor PPO 对完整 raw `[K+1,H50,D32]` joint chain 计算 old/current likelihood；Q 与环境另取
  `[C10,D14]` 投影。删除 executed-coordinate likelihood、factorized/coupled toy 分支与双 runtime。
- EnvWorker 把 primitive trace 挂到现有 `Trajectory.forward_inputs`，完整 trajectory 到 actor 后批量
  写 replay；不新增逐 transition streaming 通道。首版配置使用完整 200-step episode、
  `auto_reset=false`，因此不保存 pending episode。
- 最小文件链以现有 `EmbodiedRunner.run` 为骨架；runner、`RoboTwinEnv`、`embodied_io_struct.py` 不改。
  新建 OGPO actor worker、core、replay、critic、OpenPI sampler/scorer，其他只做 opt-in thin hook。
- 验收收敛为四组：OGPO 数学、RoboTwin primitive trace、一次 opt-in synthetic update 且旧 PPO 不变、
  一次 save/load。QAM/Fast-WAM 历史不预先展开为诊断树或 gate，遇到同类具体问题再窄读。
- 官方 final-action boundary mismatch 只作窄记账修正：old/current 始终评分同一 raw chain，Q/env
  使用另行 clipped/projected action；不扩展为新方法或额外路线。
- 用户说明服务器当前已关闭；CTX-0008 的资源/进程数据降为历史快照，开机后再刷新。

## CTX-0011：服务器刷新、根上下文压缩与官方参数复核

- 日期：2026-08-07 13:39（Asia/Shanghai）。按附件提供的入口完成只读 SSH 身份探针；口令只进入
  当前进程环境变量，未写入脚本、文档或日志。服务器在线，容器为
  `autodl-container-nekaqbwt43-6ce5babb`；2×A800 80GB 均空闲，RAM available 约 981 GiB，
  `/root/autodl-tmp` available 约 816 GiB；未发现相关训练或 Ray 进程。
- 服务器 `/root/autodl-tmp/RLinf` 当前 HEAD 为 `6d0db56bf26f972cd27fa29535f5eb939e80e5bf`，
  branch 为 `local/openpi-a800-2gpu-migration`，且有既存未跟踪配置/脚本；DSRL/QAM/RLT worktree
  均 clean。OGPO 目标 worktree 和官方 OGPO clone 均不存在，故仍不得在公共 dirty tree 实现。
  目标 SFT checkpoint、norm SHA 与共享 Python/Torch/Ray/JAX/Flax 运行时仍在。
- 根 `HANDOFF.md` 的 484 行历史专题内容会污染当前路由，已原样归档为
  `docs/project-history/HANDOFF_SNAPSHOT_20260807_PRE_COMPACTION.md`，根文件压缩为只含稳定授权、
  当前 OGPO 路由、live server truth 与下一步；`PROJECT_CONTEXT.md` 无需改动。
- 官方 paper v4 与固定 commit `0b3be413cde766a41257c6b19c0c2b06393a557f` 参数复核后修正：
  PaliGemma runnable 的 `G=32` 不能直接视为 3B π0 的默认值，首版改为建议 `G=8`（LIBERO
  VLA 先例，待用户锁定）；`gamma` 公式与 primitive 单位已对齐，但数值由 `.99` 重开为建议
  `.999`（200-step/长时域先例，待用户锁定）；`UTD-Q=UTD-PI=1` 是每新增 primitive row 各一
  次更新 credit，不是每条完整 trajectory 一次。
- OGPO 的 tapered sampler 是四个 flow step 都注入 SDE 噪声，默认/机器人配置 `sigma_init=.01`；
  RLinf PPO 的 `noise_level=.5` 与“一次随机 denoise step”语义不能复用为 OGPO sigma。保留 π0
  prefix cache、velocity、batching 与 raw chain，新增隔离 sampler/scorer。
- TD bootstrap 首版使用 ensemble `mean`；CA 仍读取全部 10 heads。官方 PaliGemma 的
  8-next-action Q variance reduction 明确关闭，避免把另一条可选方差缩减轴误当成 CA/TD mean。
  官方 optimizer reset 把 actor/critic 同设 `4.5e-5` 与配置/论文 critic LR `3e-4` 不一致；首版采用
  `critic_lr=3e-4`，actor 沿 RLinf π0 PPO optimizer contract 使用 `5.6e-6`。
- raw-chain 修正保持为一个 runtime：EMA 对 raw latent chain 生成并记录 old log-prob，online 对
  同一 raw chain 重评分；Q 取 normalized `[C10,14]` projection，环境取现有 output-transform/14D
  execution copy。OpenPI transform 本身不 clip；若执行路径另有 bound，也不写回 raw chain。该做法沿用
  RLinf π0 PPO 已有“raw chain 与 env action 分属不同张量”的所有权，不把 projected action 写回
  likelihood chain；只保留 same-chain ratio fixture，不设 gate 或替代路线。
- `clean-50` 不属于 OGPO actor/critic 初始化或 replay 设计，已从 active OGPO 计划删除。QAM、
  Fast-WAM、GRPO 只作 P1 工程接口/历史定位参考；其失败结果不再作为 OGPO horizon、目标或参数的
  因果证据。CTX-0003/0009 的“负面证据”措辞与 CTX-0010 的“服务器关闭”状态均由本条覆盖。
- 本轮新增 `03_PARAMETER_PROVENANCE.md`，同步修订专题主计划、reference matrix、call/data flow、
  根 handoff 与 project-history index；新增三个无凭据的本地只读探针脚本：
  `remote_ogpo_20260807_readonly_refresh.sh`、`remote_ogpo_20260807_parameter_probe.sh`、
  `remote_ogpo_20260807_official_oracle_probe.sh`。
- 本轮服务器动作全部只读；未创建/修改服务器文件、worktree、环境或运行产物，未执行测试、
  smoke 或训练。

## CTX-0012：训练时间轴、串行/并行轴与数据比例复核

- 日期：2026-08-07 15:15（Asia/Shanghai）。只读核对 OGPO paper v4、固定 commit
  `0b3be413` 的 `main.py`、`online_rl_runner.py`、`ogpo.py`、PaliGemma scripts 与公开参数表；
  同时核对本机 RLinf π0 PPO 配置和同机 RLT 资源先例。未新增服务器动作。
- 统一 step 单位：`online_steps/start_training/eval/save` 按 primitive environment rows；BC/Q/CalQL
  与 LR schedule 按 optimizer steps；UTD-Q/PI 是每新增 primitive row 的 update credits。官方可选
  BC-Q、CalQL、Q-only warmup、BC refine 均关闭，主线只保留 20k pure collection 和后续 steady
  OGPO+CA 两阶段。
- 当前正式规模单一建议为 total 250k primitive rows：20k 纯收集，约 230k joint Q/actor credits；
  依据是论文中最接近的 image+language LIBERO 250k，而非 Robomimic PaliGemma 的 2M–3M。log/
  eval/checkpoint 建议 5k/20k/50k，eval 20 episodes；算法与 constant LR 不在里程碑切换。
- 论文 Algorithm 2 写 Q-first，released fused `_update` 实际 actor→EMA→critic→target-Q。本项目按
  可执行实现优先改为 actor-first，不保留顺序开关；完整 episode 到达后批量结算 credits 的 RLinf
  调度适配保持不变。
- 官方参数改写为 `offline_ratio=0.0 (online fraction=1.0)`：Q transition batch、actor imagined-state
  batch 都只来自 online replay；success view 是其在线成功子集。SFT checkpoint 只初始化 actor，
  不含 TD transition，BC-Q/CalQL/Q-only warmup 也不自动启用。
- 两个 likelihood normalization 的真实语义已补公式：K4/H50/D32 时 ratio 为 joint likelihood ratio
  的 `1/(5×1600)=1/8000` 次方，即 normalized whole-chain score ratio；它保留所有 chain 因子和
  坐标，但不是论文未归一化 importance ratio 的字面值，`.01` clip 与该官方尺度配套。
- gamma 建议仍为 primitive `.999`，依据改为 terminal 0/1 sparse reward、LIBERO/VLA 近邻和单位
  变化：`.99^200=.134`，`.999^200=.819`；不再只用“200 步算长任务”解释。旧 PPO macro `.99`
  的名义 primitive 等价约 `.999799` 只用于对照，不写入 OGPO。
- 双 A800 首版资源建议：train 8 env×1 wave、eval 4×5、FSDP/rollout world size 2、pipeline false、
  B64/G8、10Q vectorized，candidate microbatch 目标 32/rank 待 probe；同一 state 的 G8 不跨 rank
  拆分。K4、C10 和 optimizer credits 串行，B/G/head/env 轴并行。
- G32 不是近乎免费的 batch 扩展：B64 下 G8/G32 为 512/2048 chains per actor update，只有 B 个
  frozen prefix 能复用，action-expert generation、online scorer/backward 和 Q scores 的 candidate
  主体为四倍。250k/UTD-PI1/B64/G8 已对应 117.76M imagined chains，正式启动前必须实测一个
  full update，才能把逻辑预算换算成 wall-clock/GPU-hours。
- 同步修订 `00_INDEX_AND_IMPLEMENTATION_PLAN.md` §6.5–§6.9/§7.1/§9–§10、
  `02_CALL_AND_DATA_FLOW.md` §3.3/§7 与 `03_PARAMETER_PROVENANCE.md` §1/§3/§5–§9；没有新增
  第二套计划、运行 gate、服务器文件或代码实现。

## CTX-0013：实施后上下文裁剪（取代旧“未实现/runner 不改/资源待测”状态）

- 日期：2026-08-07。用户授权后已从父 commit `6d0db56b` 创建服务器独立 worktree
  `/root/autodl-tmp/RLinf_ogpo_pi0_robotwin`、branch `codex/ogpo-pi0-robotwin`；主体实现和正式
  smoke 前定向测试完成。公共 RLinf dirty worktree、DSRL/RLT/QAM 历史分支均未合并或修改。
- `EmbodiedRunner.run` 不是零 diff：只为 OGPO 增加按累计 primitive rows 的 eval/checkpoint/stop 与
  resume 阈值薄分支；`RoboTwinEnv`、`embodied_io_struct.py` 仍未改。CTX-0010 的“runner 不改”由本条
  覆盖。
- 真实 π0 两卡 `FULL_SHARD + use_orig_params=true` 的 B4×G8 candidate microbatch 已通过，flat
  32/rank、allocated/reserved 约 33.19/36.26 GiB/卡、same-chain score delta=0；CTX-0012 的
  “microbatch 待 probe”由本条覆盖。
- QAM 只在真实出现完全同构的 tied PaliGemma `embed_tokens/lm_head` FSDP ownership 错误后窄读；
  复用了其已验证底层修复形态，未迁移 QAM 的 AM/VJP、目标、参数或失败解释。错误的 target
  embed-pruning 假设已回滚，不进入当前上下文。
- replay row-size 已实测：250k 全局 ring 约 334.98 GiB RAM、满 sidecar 约 322.79 GiB。当前
  250k capacity 与 50k checkpoint interval 的组合不能在保留全部完整 sidecar 时进入 formal；该资源
  选择是 smoke 包前唯一明确未决项，不扩展成多套 runtime。
- `clean50` 历史标签已从 OGPO config fingerprint 移除；当前只记录已验证的 π0 RoboTwin SFT
  checkpoint 路径和 norm SHA。服务器未安装新依赖、未下载 OGPO clone，真实 RoboTwin smoke/
  pilot/formal 仍未运行。
- 详细文件、命令、失败、修复和测试结果只由
  `evidence/IMPLEMENTATION_LOG.md` 维护；本账本以后不复制成长篇实施流水。

## CTX-0014：提交后上下文收口（取代“full update/commit/push 待完成”状态）

- 日期：2026-08-07。主体实现已提交为
  `5d5c84e3ac4efa1713a4139a05ac1b776e634ed3` 并推送到
  `personal/codex/ogpo-pi0-robotwin`；upstream 0/0、worktree clean。公共 RLinf worktree 未改，未安装
  依赖、未下载模型/OGPO clone、未运行真实 RoboTwin smoke/pilot/formal。
- 真实两卡 full update 曾在 PPO microbatches 后切到旧 full-model success-BC forward 时暴露
  sharded `q_proj` view；当前唯一实现改为 frozen cached prefix + online action-expert velocity，保持
  完整 H50 noisy input、只对 C10×D32 做 flow-matching loss。该修复没有改 FSDP internals、关闭 BC
  或引入备用路线。
- 最终真实 B64/G8、flat32/rank、rank0 有 success/rank1 无 success 的 production
  `target action -> actor+BC -> critic` 已通过：约 12.75 秒/paired credit，每卡峰值约
  35.31/39.57 GiB allocated/reserved；ratio=1，两个 rank 的 actor/critic counters 均前进一步。
- 最终统一回归为 25-file syntax、26 pytest、9 config locks、OGPO/PPO Hydra compose、Ruff/diff
  全通过。完整 precommit patch 为 26 文件、6,121+/57-，secret/实际凭据/冲突标记检查均通过。
- CTX-0012/0013 中“full update timing 待测、commit/push 待完成”由本条覆盖。full timing 同时新增一个
  formal 预算事实：按当前 UTD1 原样执行约 230k credits，update-only 线性投影约 33.9 wall-days/
  1,629 A800 GPU-hours。因此 formal 前不再只有 replay/checkpoint 容量问题；250k/UTD1 的计算预算也
  必须显式批准或基于来源重新讨论。smoke 仍需先展示完整 resolved config、命令、输出、预算和停止条件。

## CTX-0015：首次真实 smoke 收口（取代“smoke 未运行/资源只来自 probe”状态）

- 日期：2026-08-07 19:41–20:10（Asia/Shanghai）。用户明确批准 smoke 后，先刷新服务器身份、
  GPU/RAM/磁盘/进程/Git，再生成并展示 resolved packet。实际运行保持 2 GPU、train 8 env、B64/G8、
  flat32/rank，只把串行预算缩为 80 primitive rows、1 paired update、一次 4-env C10 eval 和一个
  checkpoint；没有自动 fallback 或第二套算法配置。
- 生产链路 exit 0：8 条 train trajectory 各 C10、global rows 80、两 rank replay 各 40；
  `updates_run=actor_updates=critic_updates=policy_version=1`，pending credits 为 0，ratio 1；train rollout
  54.05 秒、paired update 12.62 秒、sync 12.49 秒、eval rollout 14.86 秒。driver/monitor 正常退出，
  GPU 回到 0 MiB，无 Ray/train 残留。
- 207 个资源样本覆盖 271 秒，实际 median/max cadence 1/2 秒。GPU0/1 物理显存峰值为
  50,701/51,311 MiB、util 峰值均 100%；cgroup current 峰值 60,136,706,048 bytes，主机最低
  MemAvailable 1,003,950,000,000 bytes，OOM/OOM-kill 增量为 0。该全流程峰值取代只用模型 probe
  35.31/39.57 GiB 判断 runner 容量的旧做法，但不直接证明 B/G/env 可翻倍。
- `global_step_1` 的 DCP 两片约 5.51 GB；两 rank sidecar 各约 1.53 GB，manifest 共享同一 snapshot。
  CPU 读取验证 global rows 80、replay 40+40、actor/critic/version 1、pending 0、EMA 126 tensors/rank；
  checkpoint 总落盘使数据盘可用量减少约 13.13 GiB。
- train/eval 均只运行 10/200 primitive steps，reward/success 为 0；`eval/num_trajectories=0` 是没有完整
  episode 终局，而非 eval 未调用。一次 actor update 的 loss/grad norm 也为 0；它证明调用和计数但
  不建立有效 actor 学习信号。是否需要二次小 probe 或直接进入较长 pilot，随并行/UTD/预算一起讨论。
- 从本次 smoke 起新增 `evidence/COMMAND_AND_CHANGE_INDEX.md`，SMK-0001–0018 记录完整无密码
  launcher、command SHA、时间、exit、raw output、服务器副作用、问题/修复/复测。postflight 首次因
  `bash -c` argv 自匹配误报，改为 SHA 锁定的远端文件执行后通过；本地下载首次被 ExecutionPolicy
  拒绝，显式 Bypass 后 21 个小证据文件逐一与 remote SHA 对齐。两次失败都没有重跑训练。

## CTX-0016：一日预算 formal 已启动（取代“formal 未批准/预算未冻结”状态）

- 日期：2026-08-07 23:13–23:26（Asia/Shanghai）。用户批准formal和独立显存/RAM监控，并明确只改
  四项预算：total35k、warmup10k、paired UTD.1、capacity40k；算法、B64/G8/flat32、8 train env、
  4×5 eval、20k eval interval和50k checkpoint interval均不变。
- resolved SHA为`77419258766880fca7b8d15d725dfb9f2fba5b88d2bfe21eb918fd760e9a7bc9`；目标为
  2,500 paired updates、1,280,000 imagined group chains。因50k>35k，中途无checkpoint，runner只在
  exact35k final save。35k replay约46.90 GiB RAM/45.19 GiB sidecar，40k capacity ceiling约
  53.60/51.65 GiB。
- 正式进程于23:13:50后台启动；23:26快照中baseline 20 episodes已完成，online actor/rollout/env已
  分别进入`recv_rollout_trajectories/generate/interact`，两卡8个compute rows，monitor持续写1秒CSV，
  cgroup OOM/oom_kill=0。训练按35k自然运行；后续动态状态必须重新连接刷新。
- 23:32首个online wave用时8分20秒并结算global1,600 rows（800/rank）；success rows0、updates0符合
  10k warmup，第二个wave已开始。该条覆盖“只有活跃进程、尚无replay row证据”的启动中间状态。
- source-aligned 20k/250k/UTD1仍保留为论文/官方近邻参照，但不再是当前运行配置。QAM/DSRL/RLT等
  历史实现未进入本次配置或代码路径。完整命令与问题解决仅由`COMMAND_AND_CHANGE_INDEX.md`的FRM
  系列和`IMPLEMENTATION_LOG.md` IMP-0036/0037维护。
