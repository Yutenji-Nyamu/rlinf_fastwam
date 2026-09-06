# Idea2：π0 DVAC 内部信号采集主索引与实施计划

最后更新：2026-08-30  
当前状态：**推理telemetry、16-env评估smoke和64-query分析已完成。v1 formal停于完整g54，v2 R-only
`[0.5,1.2]`停于完整g51，v3 R-only `[0,2]`停于完整g49。v1 global-z × v3 `[0,2]`的100-step
factorial cell最新完整到g49，随后g50 rollout仅`14/16`；服务器容器在旧observer最后采样后约3分钟重启，
训练随容器重启中断，并非自然完成。当前GPU/RAM已释放；run期无CUDA/NCCL fatal，OOM/OOM-kill=0。
g49 raw/5-step/10-step training-rollout success=`93.359/93.203/93.320%`；共同g1--49相对原GRPO
累计/末5/末10=`+2.081/+2.734/+2.930pp`。g49 weight ESS=`0.897`、有效H=`44.83/50`、系数角=`18.22°`；
保留checkpoint为g10/g20/g30/g40。容器重启触发源不在容器内日志中。轻量收尾包已封装为
`exports/idea2_dvac_global_z_w0to2_stop_g49_20260824.zip`（3,038,577 bytes，SHA256
`b9f9da78...f57e7`）。**
v3终态见[25号收尾分析](25_V3_G49_CLOSEOUT_AND_METHOD_ANALYSIS_20260823.md)，新实验见
[27号启动记录](27_GLOBAL_Z_W0TO2_FORMAL_LAUNCH_20260823.md)，当前训练/方法/资源图见
[30号现场分析](30_GLOBAL_Z_W0TO2_LIVE_ANALYSIS_G44_20260824.md)。RLT teacher-DVAC `[0,2]`已自然完成
`480/480`并exit0；用户随后改为暂不启动单卡control/`[0.5,1.5]`，两条均未启动。最终轻量归档、
完整对比与单卡配置说明见[37号文档](37_RLT_SINGLE_GPU_PAIRED_480_LAUNCH_20260825.md)。
success-executed DVAC-BC matched-width pair已按用户授权停止于共同完整Step476，最新完整checkpoint均为Step475；
累计train success control/method=`55.49/55.75%`，Step475 fixed20=`17/20 vs 15/20`，未形成稳定领先。
轻量收尾包见`exports/rlt_success_bc_matched_width_stopped_step476_high_info_20260827.zip`。
Pure reference-BC的Pure02/05已自然完成480/480、exit0；累计train success=`60.49/59.74%`，最终
MA5/MA10/MA20=`97.5/97.5/95.0%`与`97.5/98.75/93.13%`，最终fixed20=`20/20 vs 17/20`。
轻量包见`exports/rlt_dvac_pure02_pure05_final480_high_info_20260829.zip`；最终四条单卡同轴图见
[单卡RLT四run同轴图](evidence/rlt_four_single_gpu_success_full_axis_20260828/README.md)。Pure03/04只新增薄配置，
源码/配置HEAD=`f0aaf4b7`已推送；strength分别为1.0/1.5，其余采样、batch、replay、schedule、eval、Stage1与
Pure02/05一致。2026-08-29 11:54 CST，GPU0/1两条fresh 480-step formal均compose=0、完整Step1并进入
下一轮rollout；OOM/OOM-kill=0。2026-08-30 19:55按用户授权停止Pure03/Pure04，最终完整Step=`448/446`，
累计train success=`55.11/59.98%`，MA5=`90/95%`、MA10=`86.25/90%`、MA20=`86.88/92.5%`；
Step425 fixed20=`16/20 vs 18/20`。两个owned进程组及pair专用Ray head均退出，GPU显存归零，CUDA OOM与
cgroup OOM/OOM-kill均为0；最新checkpoint均为Step425。轻量ZIP与最终图见
[Pure03/04停止收尾](evidence/rlt_dvac_pure03_pure04_stopped_g448_g446_20260830/README.md)和
[六设置最终同轴图](evidence/rlt_six_single_gpu_success_rows_pure03_g448_pure04_g446_20260830/README.md)。
见[28号实现合同](28_RLT_DVAC_IMPLEMENTATION_AND_RECORDING_PLAN_20260823.md)、
[32号smoke](32_RLT_DVAC_REAL_SMOKE_RESULT_20260824.md)与
[33号正式启动](33_RLT_DVAC_FRESH480_FORMAL_LAUNCH_20260824.md)、
[36号g69现场](36_RLT_DVAC_FRESH480_LIVE_G69_20260824.md)与
[g342快照](evidence/rlt_dvac_formal_live_g334_20260825/README.md)与
[原RLT同轴g357快照](evidence/rlt_dvac_vs_original_live_g350_20260825/README.md)。DSRL暂缓；后续动态step仍以AutoDL现场为准。**

## 1. 这次先解决什么

Idea2 的长期目标是：读取 π0 自己在 flow 去噪过程中的不确定性信号，再用这个信号小幅调整
RL 训练的 credit/weight。第一阶段 active scope **只包含原始 π0 SFT 推理**，先回答一个更基础的
问题：

> π0 在 RoboTwin 推理时，对未来 chunk 中每个 action 的 clean endpoint 估计怎样变化；这些变化能否被
> 无损、可对齐地记录下来？

因此本专题明确分成两步：

1. **先采集**：沿官方评估路径，只增加默认关闭的旁路 telemetry（遥测，即额外记录内部量），不改变
   action、执行 chunk 或环境行为。
2. **后分析**：拿到真实数据并先报告形状、数量、完整性和任务结果，再共同决定阈值、图和训练用法。

第一阶段已经完成。第二阶段已从规划进入实现与smoke：runner step1以`w=1`照常做GRPO，并建立进入
trajectory的4个action-query×h之train-SDE统计；step2开始用最近5个已完成step的global `log V`均值/
标准差生成`[0.8,1.2]`权重。advantage仍决定强化或抑制方向。2-step smoke已验证warmup与apply；
100-step正式训练已由用户授权并运行到完整Global Step 54；随后为释放资源运行v2 smoke而按用户授权停止，
终态轻量材料已封装。

下一版并不覆盖上述v1事实：它把统计改为recent-5、按每个`h`分别建立`log V_L3`的median/MAD位置
基线，只用去位置后的residual生成`[0.5,1.2]`偏重降权。实现、简洁前测与真实两步smoke均已完成。
v3把同一residual映射扩大到`[0,2]`并在完整g49后停止；当前factorial cell则使用v1 global-z统计与v3
`[0,2]`强度，成功GRPO的其余主体参数不变。

## 2. 文档入口

- [信号、张量与数据合同](01_SIGNAL_AND_DATA_CONTRACT.md)：解释 `x/v/z`、action 粒度、`L=2/3/4`、
  chunk/视频边界，以及首轮保存什么。
- [实施与前测流水账](evidence/IMPLEMENTATION_AND_PRETEST_LEDGER.md)：从第一次操作起，逐项记录命令、
  结果、问题、修复和复测；凭据永不入账。
- [AutoDL 运行、Git 与视觉对齐计划](02_AUTODL_RUNTIME_GIT_AND_VISUAL_ALIGNMENT.md)：解释两卡并发、
  fixed-seed shard、独立 worktree，以及现有视频到底能对齐到什么粒度。
- [实现结果与首次 smoke 审阅包](03_IMPLEMENTATION_RESULT_AND_SMOKE_REVIEW.md)：已实施文件、验证边界、
  完整两卡16-env resolved config、精确命令、参数血缘与旁路资源观察。
- [首次 smoke 完整配置](evidence/SMOKE_RESOLVED_2GPU_16ENV_V1.yaml)：已在服务器通过入口
  `--cfg job --resolve`；SHA256为`d4b7393e...bc018`。
- [首次 smoke 逐指令流水账](evidence/SMOKE_2GPU16ENV_EXECUTION_LEDGER.md)：单列准备、上传、验证、
  真实运行、资源与产物核对，完整保留命令、结果、问题和修复。
- [首轮64-query离线分析](04_FIRST_DATA_ANALYSIS.md)：逐图解释L2/L3/L4、per-h结构、success边界、
  future-h位置效应、论文四张图的可复现边界、逐control recorder与32/64并发选择。
- [首轮分析逐指令流水账](evidence/FIRST_DATA_ANALYSIS_LEDGER.md)：服务器CPU分析的每条指令、两次窄修正、
  产物hash、SFTP与视觉后检。
- [训练修改最小增量计划](05_TRAINING_MODIFICATION_PLAN.md)：历史100-step GRPO参数血缘、train-SDE
  telemetry、chunk-level PPO中的per-h梯度挂点、80/20相关工作、记录合同、smoke与受控A/B边界。
- [训练规划流水账](evidence/TRAINING_PLANNING_LEDGER.md)：历史只读调研、设计演进与被后续冻结选择取代的路线。
- [训练实现、前测与smoke逐指令账](evidence/TRAINING_IMPLEMENTATION_AND_SMOKE_LEDGER.md)：服务器worktree、
  逐文件改动、前测、提交、resolved config、精确命令、资源观察与smoke结果。
- [训练实现与2-step smoke结果](06_TRAINING_IMPLEMENTATION_AND_SMOKE_RESULT.md)：方法落点、前测、真实权重、
  grad/clip、资源、细录像、空间估算与正式训练候选。
- [100-step正式训练现场分析](07_FORMAL_TRAINING_LIVE_ANALYSIS_STEP24_20260821.md)：Step 24现场状态、
  Step 1–23完整训练/DVAC/资源快照、历史成功GRPO同轴对照与细录像。
- [Step 39效果与梯度强度分析](08_FORMAL_TRAINING_LIVE_ANALYSIS_STEP39_20260821.md)：g1–39历史GRPO同轴
  对照、各指标读法、global clip语义、`a=.1/.2/.3`与hard top-20%离线反事实及相关工作幅度。
- [GRPO数据流、两层裁剪与重加权教学](09_PPO_DATAFLOW_CLIPPING_AND_REWEIGHTING_DISCUSSION_20260821.md)：
  历史/当前评估边界、reward→advantage→joint PPO→global clip完整链路、图1–5逐项读法、近期方法的实际
  降权/增权幅度，以及`a=.2`为什么主要改变方向而不是最终步长。
- [action权重、位置残差与近期credit文献](10_ACTION_WEIGHT_POSITION_RESIDUAL_AND_RECENT_CREDIT_LITERATURE_20260821.md)：
  精确解释`[0.8,1.2]`改的是哪层梯度、action粒度边界、历史GRPO clip fraction；用two-way/median-polish
  思路拆分chunk位置、query状态和local residual，并对照OAR、THR、ResRL、DynaMO、Stepwise-Flow-GRPO等近期工作。
- [Step 48现场刷新与方法讨论](11_FORMAL_TRAINING_LIVE_REFRESH_STEP48_AND_METHOD_DISCUSSION_20260821.md)：
  Step 48训练/历史GRPO同轴结果、最新DVAC权重与future-h结构、资源同比、recent-5 z-score、两部分位置
  残差和可选two-way细分的当前结论。
- [R-only g23现场分析](12_R_ONLY_FORMAL_LIVE_ANALYSIS_G23_20260822.md)：v2当前训练、权重、位置校准、
  历史GRPO对照、资源与产物快照。
- [梯度修改到效果的中间机制与checkpoint评估计划](13_METHOD_CAUSAL_CHAIN_CHECKPOINT_EVAL_AND_THREE_RUN_COMPARISON_20260822.md)：
  图2逐图教学、`A×w`/ESS/top-k credit离线分析、相关工作证据链、三次训练同轴图、现有checkpoint清单、
  fixed-64/并发32评估设计和RoboTwin control-trace双仓调用关系。
- [RoboTwin control trace实现短说明](14_ROBOTWIN_CONTROL_TRACE_IMPLEMENTATION_NOTE_20260822.md)：两份source
  怎样共同运行、逐control-frame取帧点、配置/产物、TOPP progress到近似h的语义，以及为何不直接套用
  official direct-evaluator recorder。
- [R-only g35现场、信号、credit与证据链](15_R_ONLY_G35_SIGNAL_CREDIT_AND_MECHANISM_CHAIN_20260822.md)：
  g35训练/资源/产物现场，credit/ESS/80/20逐名词教学，五层机制验证的直接文献先例与机器人适配，
  v1 g54/v2 g35的DVAC信号演化和当前强度判断。
- [R-only g50三次训练与资源分析](16_R_ONLY_G50_THREE_RUN_TRAINING_AND_RESOURCE_ANALYSIS_20260822.md)：
  原GRPO/v1/v2同轴success与优化指标、v2权重/ESS/方向变化、GPU/cgroup资源和当前产物。
- [R-only g51收尾与v3区间讨论](17_R_ONLY_G51_CLOSEOUT_AND_V3_RANGE_DISCUSSION_20260822.md)：
  授权停止事实、终态三run/方法/资源图、`[0,2]`真实residual反事实、clip调用链和v3单变量建议。
- [R-only v3 `[0,2]` 100-step formal启动](18_R_ONLY_V3_W0TO2_FORMAL_LAUNCH_20260822.md)：
  单变量配置、resolved/source哈希、完整预算、窄前测、正式PID、首个真实rollout与只读资源快照。
- [R-only v3 Global Step 2早期现场分析](19_R_ONLY_V3_EARLY_LIVE_ANALYSIS_G2_20260822.md)：
  首个真实`[0,2]`更新、四run早期训练对照、权重/ESS/credit强度、GPU/cgroup资源和产物完整性。
- [R-only v3 Global Step 28现场分析](20_R_ONLY_V3_LIVE_ANALYSIS_G28_20260823.md)：
  四run同轴训练、g28权重/ESS/credit、GPU/cgroup主存、checkpoint与轻量现场产物。
- [与当前挂点同构的逐action策略梯度重加权文献](21_PER_ACTION_POLICY_GRADIENT_REWEIGHTING_LITERATURE_20260823.md)：
  从`A_q w(q,h) grad log pi(a_qh)`解释当前straight-through挂点，区分advantage/loss/gradient
  weighting，逐项对照GRAIL、Covariance-Aware GRPO、STEER、A3PO、OAR、THR、Beyond 80/20、
  DelTA、FIPO及stop-gradient实现先例，并整理`signal -> w -> Delta log pi -> behavior`证据链。
- [R-only v3 Global Step 36现场分析](22_R_ONLY_V3_LIVE_ANALYSIS_G36_20260823.md)：四run同轴训练、g36方法
  强度、正负advantage权重、GPU/cgroup资源、g10/g20/g30 checkpoint与轻量产物清单。
- [R-only v3 Global Step 42现场分析](23_R_ONLY_V3_LIVE_ANALYSIS_G42_20260823.md)：四run同轴训练、g42方法
  强度、正负advantage权重、GPU/cgroup资源、g40 checkpoint与84个双rank NPZ清单。
- [R-only v3 Global Step 48现场分析](24_R_ONLY_V3_LIVE_ANALYSIS_G48_20260823.md)：四run同轴训练、g48方法
  强度、正负advantage权重、GPU/cgroup资源、g40 checkpoint与96个双rank NPZ清单。
- [R-only v3 g49终态收尾与方法分析](25_V3_G49_CLOSEOUT_AND_METHOD_ANALYSIS_20260823.md)：停止事实、四run
  g1--49对照、完整原GRPO 100步图、v3权重/credit/资源终态，以及GRAIL/DelTA等同接口参照。
- [v3 g49轻量证据ZIP](../../exports/idea2_dvac_v3_formal_stop_g49_20260823.zip)：4,566,769 bytes、32项；
  含高信息量日志/图/CSV/JSON/双rank代表NPZ/配置/复现脚本，不含checkpoint正文。
- [DVAC迁移到RLT与DSRL的规划与训练语义](26_DVAC_PORT_PLAN_FOR_RLT_AND_DSRL_20260823.md)：按各自真正策略变量
  区分RLT per-h student-Q挂点和DSRL macro latent-Q挂点；补充actor/critic/replay、teacher信号所有权、
  Q/BC或Q/entropy相对作用、连续actor-critic相关依据与observe/apply顺序。
- [global-z 0-to-2 100-step正式启动](27_GLOBAL_Z_W0TO2_FORMAL_LAUNCH_20260823.md)：单变量组合、resolved
  配置、精确命令、预算、commit/push和首个真实rollout现场。
- [RLT × teacher-DVAC实现与记录小计划](28_RLT_DVAC_IMPLEMENTATION_AND_RECORDING_PLAN_20260823.md)：冻结
  π0产生H50 DVAC、student前C10以`[0,2]`只重加权Q分支；冻结initial-replay global-z baseline、replay/
  writer字段、最小文件清单与当前训练隔离边界。
- [RLT-DVAC规划与AutoDL只读现场流水账](evidence/RLT_DVAC_PLANNING_AND_LIVE_LEDGER_20260823.md)：当前
  global-z训练、clean RLT source、SFTP小副本、精确调用链和资源边界的逐项记录。
- [RLT teacher-DVAC实现与smoke检查流水账](evidence/RLT_DVAC_IMPLEMENTATION_LEDGER_20260824.md)：独立
  worktree、逐文件修改、审查修复、ruff/语法/YAML、固定schema后5个CPU-only单测、commit/push与smoke闭环。
- [RLT teacher-DVAC真实smoke结果](32_RLT_DVAC_REAL_SMOKE_RESULT_20260824.md)：v1分布式telemetry schema
  根因、固定sum/count窄修、v2训练/eval/权重/资源/checkpoint闭环，以及正式预算的250→480两段历史合同；
  [逐命令账](evidence/RLT_DVAC_REAL_SMOKE_LEDGER_20260824.md)。
- [RLT teacher-DVAC fresh-480正式启动](33_RLT_DVAC_FRESH480_FORMAL_LAUNCH_20260824.md)：历史成功参数、
  本次单进程480预算、精确命令、输出/source/config hash及g2真实启动现场；逐指令账：
  [RLT_DVAC_FORMAL480_IMPLEMENTATION_AND_LAUNCH_LEDGER_20260824.md](evidence/RLT_DVAC_FORMAL480_IMPLEMENTATION_AND_LAUNCH_LEDGER_20260824.md)。
- [AutoDL内存锯齿3分钟说明](34_AUTODL_MEMORY_SAWTOOTH_AND_SHENZHEN_TRANSFER_20260824.md)：先读入口；
  冻结“run级cgroup为主要新增组件、train/eval nested offload为配套、不改变env并发”的迁移路线。
- [AutoDL内存机制技术附录](35_AUTODL_MEMORY_SAWTOOTH_TECHNICAL_APPENDIX_20260824.md)：按需读取；保存
  env生命周期对照、源码调用链、真实锯齿数字，以及cgroup、offload与Ray各自负责的现象。
- [RLT teacher-DVAC fresh-480 g69现场](36_RLT_DVAC_FRESH480_LIVE_G69_20260824.md)：当前进程、warmup、
  success、checkpoint、GPU/RAM与一张四面板简图。
- [RLT teacher-DVAC fresh-480 g342快照](evidence/rlt_dvac_formal_live_g334_20260825/README.md)：跨过warmup后的
  fixed20、DVAC权重、actor/critic更新、资源与四面板简图。
- [原RLT与RLT teacher-DVAC `[0,2]` g357同轴对照](evidence/rlt_dvac_vs_original_live_g350_20260825/README.md)：
  raw/5步/10步训练曲线、每25-cycle fixed20、逐叶配置一致性、最新资源和单卡容量/拓扑判断。
- [RLT单卡配对fresh-480配置与启动](37_RLT_SINGLE_GPU_PAIRED_480_LAUNCH_20260825.md)：解释两卡→单卡的
  batch/累积、rollout、per-rank replay与checkpoint语义；冻结原RLT control和teacher-DVAC
  `[0.5,1.5]`逐字段合同；按用户最新指示两条均未启动，并记录旧`[0,2]`终态与归档。逐指令账见
  [RLT_SINGLE_GPU_DUAL480_LAUNCH_LEDGER_20260825.md](evidence/RLT_SINGLE_GPU_DUAL480_LAUNCH_LEDGER_20260825.md)。
- [RLT × DVAC方法诊断与下一版设计](38_RLT_DVAC_METHOD_DIAGNOSIS_AND_NEXT_DESIGN_20260825.md)：从当前
  replay/Q/BC真实代码解释逐h Q-gradient挂点为何不同于GRPO，并对齐AWAC/CRR/UWAC、DVAC与近期
  robot rollout后训练工作；给出机制探针、query级Q控制和advantage-gated BC候选。
- [RLT × DVAC成功episode逐action BC实现计划](39_RLT_DVAC_SUCCESS_EPISODE_BC_IMPLEMENTATION_PLAN_20260825.md)：
  冻结恢复原Q、成功episode使用executed-action target、普通/失败episode保留π0-reference target；
  以`success_scale × mean-one DVAC [0,2]`分离成功自模仿总强度与C10内部credit；现已补实现、三次commit、
  10项单测及双单卡smoke终态。
- [RLinf双单卡并发短说明](40_RLINF_DUAL_SINGLE_GPU_CONCURRENCY_NOTE_20260825.md)：解释world size、
  micro/global batch、梯度累积2→4、per-rank replay、单卡变慢原因，以及shared Ray、自动namespace、
  placement、code sync、绝对输出与exact cleanup的最小运行合同。
- [success-episode DVAC-BC双单卡smoke轻量证据](evidence/rlt_success_bc_dual_single_gpu_smoke_v9_20260825/README.md)：
  两条exit0、真实方法指标、checkpoint manifest、资源峰值、resolved逐叶diff及轻量原始材料；完整v1--v9操作见
  [逐指令流水账](evidence/RLT_DVAC_SUCCESS_BC_IMPLEMENTATION_AND_DUAL_SINGLE_GPU_SMOKE_LEDGER_20260825.md)。
- [success-episode DVAC-BC双单卡正式480步启动与代码增量](41_RLT_SUCCESS_BC_FORMAL480_LAUNCH_AND_DELTA_20260826.md)：
  解释新BC方法的必要代码、可后续精简部分、`strength=.25`、单卡参数差异、启动问题与当前两条正式run；
  [逐指令账](evidence/RLT_SUCCESS_BC_FORMAL480_LAUNCH_LEDGER_20260826.md)。
- [RLT双卡迁移与双单卡并发：计算和启动层简明说明](42_RLT_DUAL_TO_SINGLE_GPU_COMPUTE_AND_LAUNCH_STACK_20260826.md)：
  区分显存容量、GPU计算与墙钟；展开`2 rank × 2 microbatch`和`1 rank × 4 microbatch`，并把当前
  shared-Ray启动栈、首次问题、必要边界与后续精简方案收敛到一页主线。
- [AutoDL两张GPU并发两个RLinf任务：问题与解决方法](43_AUTODL_TWO_RLINF_JOBS_CONCURRENT_LAUNCH_GUIDE_20260826.md)：
  独立只讲双job并发的shared Ray、placement、namespace、路径隔离、已遇问题、最终处理和长期最小启动栈；
  不混入DVAC算法与单卡性能教学。
- [单卡RLT control vs success-episode DVAC-BC Step111现场](evidence/rlt_success_bc_formal_live_g111_20260826/README.md)：
  一张成功率/权重/资源图、曲线CSV和摘要JSON；Step100 fixed20=`0/20 vs 4/20`，但单点不足以下效果结论。
- [RLT × DVAC的SAC信号挂点与少数据实验讨论](45_RLT_DVAC_SAC_SIGNAL_AND_LOW_DATA_DISCUSSION_20260827.md)：
  解释当前成功episode DVAC-BC的最小改动、8 episode→macro replay→critic/actor更新的数据循环，整理
  VACO/ACTIVE/GFP/QIPO/AC3/ACE/ReaPER及DACER/DIME等外部信号接口，并把`query质量 u_q`、
  `C10内DVAC分配 c_qh`与总体BC强度分成三层；历史阶段曲线支持的低交互候选为
  `env8→4 + warmup20k→10k + UTD5→10`，以约一半unique interaction保持近似optimizer预算和阶段边界；
  同时澄清Stage1与Stage2-A/B/C/D、fixed20=`4×5`以及fixed32的并发选择。
- [RLT-DVAC-Pure reference-BC实现小计划](46_RLT_DVAC_PURE_REFERENCE_BC_PLAN_20260827.md)：冻结新版本的
  高层语义：保留原RLT的π0 reference target，仅在成功episode内以C10 mean-one DVAC调整BC；Pure采用
  `strength=.5`、非负截断后再归一，历史trace预计ESS约`.929`，并记录可复用数据管线与最小代码增量。
- [RLT-DVAC-Pure双正式启动、单/双卡对照与C10时间线](47_RLT_DVAC_PURE_DUAL_FORMAL_LAUNCH_AND_C10_ANALYSIS_20260828.md)：
  `s0p5/s2p0`已分配GPU0/1并各完成Step1；逐叶参数差异、历史双卡资源/时间/效果对照，以及旧Step475
  329条episode的C10/H50数值模式和图表入口。
- [RLT teacher-DVAC语义、替代切口与单卡耗时拆解](48_RLT_TEACHER_DVAC_SEMANTICS_AND_ALTERNATIVE_ROUTES_20260828.md)：
  解释“高teacher-DVAC为何未必应加强reference-BC”，整理confidence、quality×confidence、成功执行
  target、多样本/集合监督与主动采集等路线，并说明`s0p5/s2p0`命名和单卡`1.72×`耗时来源。
- [SAC × action chunk的future-h credit文献地图](49_SAC_ACTION_CHUNK_PER_H_CREDIT_LITERATURE_MAP_20260828.md)：
  以当前RLT的`student C10 + scalar Q + per-h DVAC`为起点，对齐17篇chunk RL、prefix critic、
  per-step advantage与外部信号加权工作；区分`h/d/denoise-i`三个轴，并收敛出最小BC切口与
  prefix/multi-horizon critic两条主线。
- [RLT/SAC actor侧future-h信号插件](50_RLT_ACTOR_SIDE_PER_H_SIGNAL_PLUGINS_20260828.md)：
  将调研重新收敛到“保留scalar chunk-Q、只在actor侧插外部`[B,H]`信号”；逐层解释当前RLT的
  replay→twin-Q→C10 actor→BC→backward数据流、scalar Q之后真实存在的`[10,14]`梯度接口、SEAR与
  primitive-step advantage，并按actor挂点重排18篇近期或高影响力工作；2026-08-28已更正：其中
  Q-gradient preconditioning与旧RLT-DVAC第一版计算相同，不再列作新首选路线。
- [PRM、dense reward与RLT过程credit](51_PRM_DENSE_REWARD_AND_RLT_PROCESS_CREDIT_20260828.md)：
  逐篇区分PRM评估、replay reward、actor/BC weighting与CFG四种接口；列出真正进入SAC/DSRL的工作，
  解释GFP、ACE、Set-Supervised DP与DAM-VLA的准确挂点，并收敛出
  `process direction × denoising instability`的RLT方法主线。
- [Action-level signal怎样进入chunk-SAC：本轮收敛](52_ACTION_LEVEL_SIGNAL_TO_CHUNK_SAC_CONVERGENCE_20260828.md)：
  用23篇一手工作按reward/replay、actor监督、探索、Q生成方向和prefix critic五个接口重排；精确解释
  DenseReward、Robometer、LLM-as-Verifier、SARM/GVL/TOPReward怎样改下游训练，并收敛出
  “高DVAC释放teacher BC”与“Q给方向、DVAC分配逐h辅助学习量”两个候选。
- [RLT-DVAC-Pure效果机制与样本效率](53_RLT_PURE_EFFECT_MECHANISM_AND_SAMPLE_EFFICIENCY_20260830.md)：
  审计Pure实际只重排成功episode的π0-reference C10 BC；将当前效果收敛为horizon-wise BC/Q调度、
  hard-position distillation与共享MLP梯度重排，并按MA20同阈值给出从训练起点、actor更新后和student接管后
  三套episode/macro-transition样本效率口径。
- [RLT论文设置、Git状态与单卡效率审计](54_RLT_PAPER_CONFIG_GIT_AND_SINGLE_GPU_EFFICIENCY_AUDIT_20260830.md)：
  逐项区分论文核心算法与当前RoboTwin实验协议，确认Pure代码/配置已推送；量化双卡→单卡的wall time、
  GPU-hours、两实验吞吐、GPU/RAM现场，并解释`2×4 env ranks`与`1×8 threaded envs`的差别及RLinf
  placement、env-decoupled、async RLT提速路线。
- [四条单卡RLT成功率最长轴同图](evidence/rlt_four_single_gpu_success_full_axis_20260828/README.md)：
  区分clean、旧success-executed DVAC-BC、Pure02与Pure05；统一Step1--476横轴，提供raw、MA5、MA10、
  MA20及机器可读CSV，Pure两条只画到各自真实终点。
- [RLT matched-width Step403/406现场与Step401同轴图](evidence/rlt_success_bc_matched_width_live_20260827/README.md)：
  历史RLT、当前control与DVAC-BC的逐步/5步/10步train success、fixed20、方法权重、资源和ETA。
- [RLT matched-width Step434/438刷新与共同Step433曲线](evidence/rlt_success_bc_matched_width_live_refresh_20260827/README.md)：
  当前生命周期、逐步/5步/10步train success、Step425 fixed20、DVAC-BC权重、RAM/GPU和错误计数；45号文档第16节展开QIPO/QVPO/GFP/FQL/AC3与当前RLT的逐组件对应。
- [global-z `[0,2]` g35分析与g36现场](29_GLOBAL_Z_W0TO2_LIVE_ANALYSIS_G35_20260824.md)：五run同轴
  success/优化图、global-z权重/ESS/future-h结构、资源与checkpoint/NPZ产物；逐指令账：
  [GLOBAL_Z_W0TO2_LIVE_G35_LEDGER_20260824.md](evidence/GLOBAL_Z_W0TO2_LIVE_G35_LEDGER_20260824.md)。
- [global-z `[0,2]` g44训练、方法、资源与产物分析](30_GLOBAL_Z_W0TO2_LIVE_ANALYSIS_G44_20260824.md)：
  五run同轴曲线、g44权重/credit强度、cgroup max事件、checkpoint/NPZ/control-trace字段说明；逐指令账：
  [GLOBAL_Z_W0TO2_LIVE_G44_LEDGER_20260824.md](evidence/GLOBAL_Z_W0TO2_LIVE_G44_LEDGER_20260824.md)。
- [global-z `[0,2]` g49容器重启收尾与1/5/10步成功率](31_GLOBAL_Z_W0TO2_G49_RESTART_CLOSEOUT_AND_SUCCESS_SMOOTHING_20260824.md)：
  g49完整终点、容器重启证据、五run逐步/5步/10步曲线、优化/方法/资源终态；逐指令账：
  [GLOBAL_Z_W0TO2_G49_STOP_AND_CURVES_LEDGER_20260824.md](evidence/GLOBAL_Z_W0TO2_G49_STOP_AND_CURVES_LEDGER_20260824.md)。
- [global-z `[0,2]` g49轻量证据ZIP](../../exports/idea2_dvac_global_z_w0to2_stop_g49_20260824.zip)：
  3,038,577 bytes（约2.90 MiB），SHA256=`b9f9da78479a2e7406290fcd58e0668df1818ef9e1965b2b5f1b24dc3b0f57e7`；
  28个文件，含README、31号收尾文档、逐指令账、四张图、关键CSV/JSON/分析脚本、训练日志、resolved/命令、
  双rank最新NPZ与资源CSV；不含checkpoint、视频和历史全量NPZ。
- [v3收尾与新实验逐指令账](evidence/V3_CLOSEOUT_AND_GLOBAL_Z_W0TO2_LAUNCH_LEDGER_20260823.md)：从g49
  现场、精确PGID停止、下载分析、配置前测、commit/push到正式启动逐项记录。
- [v3实现与正式启动逐指令账](evidence/V3_FORMAL_IMPLEMENTATION_AND_LAUNCH_LEDGER_20260822.md)：
  每条命令、结果、两次无副作用脚本问题、修正、复测、commit/push和启动证据。
- [2-step smoke轻量证据包](evidence/training_smoke_2step_20260820/README.md)：resolved config、日志、
  rank-local NPZ/CSV、资源CSV、MP4、图和SHA256。
- [v1 formal主动停止后的轻量closeout ZIP](../../exports/idea2_dvac_v1_formal_stop_g54_20260821.zip)：
  4,290,075 bytes、43项；含g1–54日志/曲线、历史GRPO对照、三组代表性双rank NPZ与control trace，
  不含约9.68 GiB checkpoint正文。
- 深圳 H100 专题已暂停，不给本专题提供 source、配置或资源约束；历史 RLT 也不进入当前实现合同。

## 3. 已冻结的首轮决定

### 3.1 任务与 checkpoint

- 首轮只做 `adjust_bottle`。
- 首轮只加载 task-matched 原始 π0 SFT checkpoint；运行时 manifest 必须记录完整权重 revision。
- 不为了凑整数人工构造“100 个离线 state”，而是沿 checked-in official standalone eval 路线采集自然
  policy query。
- 暂不加第二任务、PPO 或 RLT。当前公开权重集合没有第二个同样完备的“RoboTwin task-matched π0 checkpoint +
  official eval contract”；拿 `adjust_bottle` 权重跨任务测会成为 OOD 探针，不是标准评估。

### 3.2 官方语义尽量不动

首轮以 `evaluations/robotwin/robotwin_adjust_bottle_openpi_eval.yaml` 为合同：

- fixed reset IDs，默认 128 个并行评估环境、1 个 eval epoch；
- primitive-step cap 200；
- π0 生成 horizon `H=50`，实际执行 chunk `C=50`；
- active RoboTwin action `D_active=14`，三路相机，clean background；
- π0 默认 flow steps `M=4`，eval 使用 `flow_ode`。

在 episode 走满 200 个 action slots 时，每个环境自然产生 `200/50=4` 个 policy query。official
128-env run 的上界约为 512 个 query-state，但它不是首轮 80/20 数据采集规模。

用户将真实链路smoke收敛为 **两卡、16 个 fixed reset IDs、1 epoch**，预期16 episodes、上界64个
query-state。它保留最小串行重复1，同时用同机旧纯SFT eval跑通过的16-env并行规模，一次验证双rank写盘、
episode join、视觉/视频和资源链路；不把16-ID结果作为official-128 success evaluation。task、checkpoint、
`H/C/M/D`、200-slot cap、三相机与clean环境仍按official语义。两卡placement、历史资源证据和以后如何
覆盖完整128 IDs见[运行/Git/视觉计划](02_AUTODL_RUNTIME_GIT_AND_VISUAL_ALIGNMENT.md)。

### 3.3 信号粒度

- 主信号是每个未来 action 位置的 `V_L(h)`，`h=0..49`；这就是 DVAC 原生粒度。
- 同时保留可离线得到的坐标级 `U_L(h,d)`，用于判断某个关节/夹爪维是否主导方差。
- 首轮不保存也不使用 chunk 汇总作为主指标。`V_L,total=Σ_hV_L(h)` 以后若画 episode 总览，可从
  `V_L(h)` 免费派生；它是有损概览，不参与 DVAC crossing 或首轮训练设计。
- 所有主计算都在 π0 normalized、active 14D action space 中完成，不把 32D padding 或 decode 后混合
  物理单位直接相加。

`V_L(h)` 的计算本身不依赖实际执行 `C`，但 `C` 决定哪些 `h` 真被执行、多久后重新观察，以及如何把
信号放回任务时间轴。DVAC 最终正是用 per-`h` crossing 调执行长度，因此不能说 chunk 长度完全无关；
首轮固定 `C=50`，只观察、不控制。

### 3.4 首轮只存原始 trace 与最小索引

每个 policy query 必须保留下列**信息语义**，具体字段名、shard 命名和 writer 所在 worker 可随现有代码
结构适配：

- `z_endpoint [M,H,14]`：每个 flow step 当下预测的 clean endpoint，这是 DVAC 必需原始量；
- `timesteps [M]`；
- final normalized model action 与实际交给环境的 `[C,14]` action；
- 能把数据放回 rollout 的 query/episode/reset-ID/action-slot 标识；
- policy 实际输入的 head/left-wrist/right-wrist 三张图，或能稳定回取这些图的引用；
- RLinf MP4 的相对路径、tile 和 query 前/后 frame 映射。

现有 sampler 已经持有 `x_chain [M+1,H,14]`；若旁路导出不增加新的模型计算或明显 worker 内存压力，
首版一并保存，以便重建 `v_i=(x_i-z_i)/t_i` 和检查 endpoint。它是推荐的审计数据，不是 DVAC
计算的硬前提。原始 endpoint trace 是唯一指标事实源；`L=2/3/4` 的方差、threshold、crossing 和图
全部在离线分析阶段生成。

首轮产物采用：

```text
run_manifest.json          一次运行的 source/checkpoint/resolved config/shape/seed 合同
query_index_rollout_rankNN.csv  每个 query 一行的时间和 episode 索引
trace_rollout_rankNN.npz        rank-local x_chain/z_endpoint/timesteps/actions
episode_index_env_rankNN.csv    true reset ID与episode outcome join
query_images/rollout_rankNN/    query-level三路policy输入图像
官方 eval log/video/metric  保持原路径与含义
```

不让多个进程追加同一个 CSV/NPZ；每个 rollout rank 独立写 shard，分析阶段再合并。首轮不预先生成
`horizon.csv`、阈值或大批图。

### 3.5 默认无行为影响

telemetry 是 opt-in（显式打开才工作）的旁路：

- 默认关闭时走现有语句和返回行为；
- 打开后不新增 RNG 调用，不改变现有 RNG 调用次序；
- 不改变 noise、timesteps、`M/H/C`、output transform 或 action；
- 不对 `x/v/action` 做 in-place 修改；
- endpoint 只从本来已有的 `x_t_prev/v_t/t_i` detach 后保存；
- CPU 搬运与写盘发生在 action 已经生成之后。

首版不在线改 chunk 长度。换言之，我们先观察 DVAC 信号，不在同一次采集中让 DVAC 反过来改变访问
状态。

## 4. 实施路线与停止点

| 阶段 | 内容 | 当前状态 | 停止/批准边界 |
|---|---|---|---|
| A | 论文、official eval、π0张量、worker出口与视频链核对 | 完成 | 只读 |
| B | 建独立 Idea2 worktree；实现默认关闭的 telemetry 与 rank-local writer | **完成并推送 `61996e15…`** | 历史 baseline worktree未改 |
| C | 服务器前置测试：shape/formula、off/on action 与 RNG 等价、shard 对齐 | **完成** | CPU synthetic/writer/compose证据，不等同于真实CUDA/环境 |
| D | 准备两卡16-env真实 smoke packet | **完成并获批** | 完整config、命令、空output、参数血缘和旁路观察已展示 |
| E | 16-ID `adjust_bottle` 原始 π0 SFT telemetry smoke | **完成，driver rc=0** | 16 episodes/64 queries；14/16只作smoke事实，不报告official success rate |
| F | 32/128-env扩展或第二份collection | 未运行 | 32 env是下一档合理密度；需先锁定是否使用互斥seed shard及是否与recorder分开 |
| G | 报告数据 manifest、实际 query/episode 数、shape、NaN/Inf、文件大小与 success | **完成** | 64/16、全部finite、192 PNG、2 MP4、output约10.1 MB |
| H | 离线分析与第一批图 | **完成（v3）** | 10张图、64-row query表、3,200-row horizon表；原始事实源未改 |
| I | 冻结首版训练映射 | **完成** | recent5全trajectory action-query×h pooled log-V；保留future-h效应；`w=1+0.1 clip(z,-2,2)` |
| J | train-SDE telemetry、per-h梯度挂点与control trace实现/前测 | **完成并commit** | RLinf `052a2ee…`；RoboTwin/wamppo `43696bba…`；默认off保持旧joint ratio/clip |
| K | 两卡16-env、rollout epoch8、2-step apply smoke | **完成，driver rc=0** | step1普通GRPO+`w=1`；step2 apply；observer只读不干预 |
| L | smoke产物分析与100-step formal | **按授权主动停止于54/100** | g50 DCP保留；g1–54轻量closeout已完成；没有训练中held-out eval |
| M | R-only、per-h median/MAD、`[0.5,1.2]` child实现与前测 | **完成并commit `49b83791…`** | 11 passed；two-rank probe和resolved对照通过；v1 run未改 |
| N | R-only真实2-step smoke / formal | **2-step smoke完成；formal按授权停止于完整g51** | g50 DCP保留，三根进程已退出；无held-out eval，g1–51均值低原GRPO 0.92pp，最近5/10步高0.63/0.98pp；cgroup触240 GiB但无OOM |
| O | v3 R-only `[0,2]`单变量100-step formal | **按授权停止于完整g49** | g49 ESS=0.913；g1--49累计success比原GRPO高0.351pp但未稳定领先；g10/20/30/40 DCP保留；旧PGID退出且无OOM |
| P | v1 global-z × v3 `[0,2]` 100-step formal | **容器重启中断于完整g49；g50 rollout 14/16不计** | g1--49 success比原GRPO累计/末5/末10=`+2.081/+2.734/+2.930pp`；g49 ESS=.897；GPU峰30.37/30.22 GiB；cgroup峰240 GiB，OOM/OOM-kill=0；保留g10/20/30/40 DCP |
| Q | RLT teacher-DVAC student-Q实现、smoke与formal | **formal fresh480运行中；10:36 CST完整g357/480** | g136起baseline冻结并apply，g357 update_step=169k；fixed20 g300/g325/g350=90%/80%/95%；L3前C10、global-z `[0,2]`、仅student Q梯度；实时step以AutoDL为准 |
| R | 单卡原RLT vs success-episode DVAC-BC正式480步 | **按授权停于共同完整Step476，已轻量封存** | 最新完整checkpoint Step475；累计`55.49/55.75%`；fixed20=`17/20 vs 15/20`；未形成稳定领先 |
| S | RLT-DVAC-Pure reference-BC | **Pure02/05完成；Pure03/04按授权停止并封存** | Pure03/04最终Step448/446；fixed20 16/20 vs 18/20；轻量ZIP完成，服务器Step425 checkpoint保留 |

## 5. 推理telemetry代码落点（已实施事实）

当前source HEAD为`61996e15cc7f5a32bd6012b61b20893d94636c82`（核心功能提交`73da63f0…`，随后以
fast-forward提交补齐run provenance）；调用链是：

1. 在 π0 `_sample_actions_with_prefix_cache()` 中，在每步已有 `x_t_prev`、`v_t`、`t_i` 的位置收集
   `z_i=x_t_prev-t_i v_t`；不修改通用 `sample_mean_var_val()` 返回签名。
2. 在 `predict_action_batch()` 的 result 中仅在开关打开时附带 telemetry。
3. standalone eval worker在开关打开时收集result，并按rollout rank写NPZ/CSV/PNG；环境仍只接收原
   actions，记录发生在action发送后。
4. EnvWorker从wrapped RoboTwin env读取真实reset ID、elapsed action slots与success，并补一条窄metadata
   通路；不凭batch slot猜seed。每个env rank独立写episode outcome join。
5. query-level 三路输入图像从已经送入 policy 的 observation 旁路保存，不为录像额外执行模型；现有
   MP4 只提供 chunk 前后 head-view 参考。
6. `run_manifest.json`同时记录common base、实际RLinf/RoboTwin commit、checkpoint revision、norm/seed
   hash、完整启动命令、主机/时间，以及运行时读取的`H`、内部`D_model`、active 14D、执行C、M和dtype。

专用配置为`evaluations/robotwin/robotwin_adjust_bottle_openpi_dvac_eval.yaml`：默认
`enabled=false`、保存raw trace、不开阈值控制。plain-SFT之外的DSRL/RLT/NFT/expert/训练模式不进入v1。
逐文件说明与精确验证边界见[实现结果与smoke审阅包](03_IMPLEMENTATION_RESULT_AND_SMOKE_REVIEW.md)。

训练实现authority为RLinf child
`/root/autodl-tmp/RLinf_idea2_dvac_train@052a2ee8902c51595caa997736f1ec76699ad8df`，以及outer wamppo child
`/root/autodl-tmp/idea2_dvac_train_wamppo@43696bbab85fef3dd98074c5ba0ccb90786d0e94`；两者已分别推送到
`personal/codex/idea2-dvac-train-weighting`与`origin/codex/idea2-dvac-control-trace`。调用链、公式和逐操作证据统一路由到[训练修改计划](05_TRAINING_MODIFICATION_PLAN.md)与
[训练实施流水账](evidence/TRAINING_IMPLEMENTATION_AND_SMOKE_LEDGER.md)。

R-only v2/v3/global-z强权重authority为
`/root/autodl-tmp/RLinf_idea2_dvac_residual_downweight@afdaa2e2aa59aa16128e89f47eb4aaf7a64badd8`，
branch `codex/idea2-dvac-residual-downweight`，base=`145fa810…`。当前服务器child已commit且clean；
当前HEAD已成功推送至`personal/codex/idea2-dvac-residual-downweight`；`3061872e…`增加v2 formal配置，
`eb2a091…`只增加获批v3 R-only `[0,2]` formal配置，`afdaa2e2…`只增加global-z `[0,2]` formal配置；
两者均不改变已验证Python实现。它复用既有
RoboTwin/control-trace child，没有新建第二套环境分支。

## 6. 前置测试合同

服务器已完成少量高信息量检查：

1. **默认关闭兼容性**：原 official eval compose/import 路径不受影响。
2. **同 RNG 等价**：重置同一 Python/Torch RNG state，同一 input/noise 分别 telemetry off/on；要求
   actions、现有 chains 以及运行后的 RNG state 相同。
3. **公式与 shape**：`x=[B,5,50,14]`、`z=[B,4,50,14]`、`t=[4]`，数值 finite，抽样验证
   `z=x-t*v`。
4. **写盘与对齐**：两个小 shard 的 query key 唯一，能够回连 episode/reset ID；不并发写同一文件。
5. **真实视觉对齐**：已抽样核对query输入图、MP4 tile、pre/post frame和action-slot起点；两支视频均
   6帧/30 FPS/0.2秒，确认frame5为reset spill，不把30 FPS解释为仿真时间。

前四项通过server CPU synthetic/writer/compose测试；真实checkpoint CUDA forward、RoboTwin episode和
视觉对齐已由16-env真实smoke补齐。两支NPZ shape/finite、64-query/16-episode join、source manifest、
192张PNG和两支MP4均通过后检。

## 7. 当前视觉能回答到哪里

official 配置虽开启 MP4，但当前 RLinf `RecordVideo` 得到的是初始帧和每个 C50 chunk 的结束帧；同一
worker 的多个环境还会被 tile 到一张复合帧。RoboTwin native video 在 VectorEnv 路径中被关闭，且
qpos 的50个 waypoints 会先经轨迹压缩和 TOPP 插值成可变数量的 simulator control steps。

所以首轮采用严格的 query-level 对齐：

```text
DVAC(query q)         <-> q 的三路policy输入图 / MP4 pre frame
执行该C50 chunk之后  <-> MP4 post frame
```

它可以回答哪个**当前状态/query**的 DVAC 高、该 chunk 前后大致发生了什么、success 首次出现在哪个
C50 区间；不能宣称 `h=17` 精确对应某个 SAPIEN physics frame。

训练child现已另加默认关闭的自定义control trace：复用RoboTwin camera API/ffmpeg，在既有control loop
按双臂TOPP进度近似映射h-bin并抽样head frame，不拆C50、不重跑planner/TOPP/policy。它改善query内部
phase/success对齐，但不是official direct-eval recorder，也不声称精确waypoint-h映射，更不会增加独立
DVAC query点。详见[训练修改计划](05_TRAINING_MODIFICATION_PLAN.md)。

## 8. 当前各线状态

- RLT：teacher-DVAC→student-Q实现、审查、固定schema窄修与真实两卡smoke均完成；正式config commit
  `a85b101b...`已push，并已按用户选择以单进程fresh直接启动到480。2026-08-25 10:36 CST现场为完整
  g357/480并持续运行；g136起baseline冻结并apply，g357 update_step=169k；最新fixed20为g350=19/20，
  资源和fatal/memory event正常，动态step以AutoDL刷新为准。
- RLT success-episode DVAC-BC matched-width pair：已按授权停于共同完整Step476；Step475 fixed20=
  `17/20 vs 15/20`，曲线未形成稳定领先；轻量包与最终图已完成。
- RLT-DVAC-Pure：Pure02/05在`a2ae5cbe`上均自然完成480/480并exit0，最终轻量包与四条单卡同轴图已生成。
  Pure03/04薄配置提交`f0aaf4b7`已push；只将strength设为1.0/1.5并更换必要运行身份/placement。2026-08-30
  10:37 CST现场为完整Step316/313且继续运行；累计success=`42.52/47.24%`、Step300 fixed20=`15/20 vs 18/20`；
  DVAC已apply，ESS=`.761/.654`，fatal/OOM/OOM-kill=0。
  teacher-DVAC语义与替代切口见48号文档，本轮方法收敛见52号文档；后续动态状态需现场刷新。
- DSRL：已按用户决定暂缓；26号文档只保留比较背景。
- 深圳 H100：当前搁置，其 source lock、账号、目录、资源和规划不影响 AutoDL Idea2。
- PPO/训练权重：v1训练代码、2-step smoke和离线后检均已完成；formal主动停止于完整g54，g50 DCP保留。
- R-only v2：per-h recent-5 median/MAD residual与`[0.5,1.2]`已实现、前测并完成2-step smoke；formal
  按授权停止于完整g51，g50 DCP保留，轻量closeout与v3 `[0,2]`讨论已完成；fixed-ID评估尚未启动。
- R-only v3：相对v2只把映射区间改为`[0,2]`，已按授权停止于完整g49，终态轻量材料已下载；fixed-ID
  held-out评估尚未启动。
- global-z `[0,2]`：正式训练完整到g49，g50 rollout 14/16时随服务器容器重启中断；未完成step不计。
  g49 raw/5-step/10-step success=`93.359/93.203/93.320%`，g1--49均值比原GRPO高2.081pp；
  cgroup峰240 GiB并新增max事件，但OOM/OOM-kill=0，不能从现有日志断定重启触发源。最新完整DCP为g40。

## 9. 首轮分析完成后的当前下一步

首轮分析见[04_FIRST_DATA_ANALYSIS.md](04_FIRST_DATA_ANALYSIS.md)。已确认：

- `V_L(h)`有明显action粒度结构；L2/L3/L4相关但不等价；
- 50个pre-success query中，`h`与跨query中位`V_L3(h)`相关0.852，78%的query后半h均值更高；
- 14个成功episode在q2 C50期间首次达到success，q3是post-success状态，不能混作普通late phase；
- 两个failure的方差较高只属描述性线索，不能当failure detector或RL正负credit。

逐control recorder已作为默认关闭的抽样旁路实现；32-ID扩样仍是可选推理证据路线，不是当前训练前置。
获批2-step smoke已自然完成：step2非均匀权重、grad/clip、GPU/RAM、checkpoint和111帧录像均通过后检。
用户随后选择100-step formal；v1 run为
`/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_formal_100step_2gpu16env_20260821`，保持16并发与历史完整
GRPO主体参数。2026-08-21 20:21只读刷新时已完成48/100：当步训练rollout success与历史g48同为
85.55%，g1–48均值为86.19% vs 88.00%；当前仍无稳定领先区间，且两者都没有训练中held-out eval。
最新有效位置权重p05/median/p95为0.887/1.016/1.200，高端命中1.2约8.13%、低端命中0.8约0.11%；
future-h后25个位置平均权重比前25个高0.037。两卡显存峰值约30.37/30.22 GiB，cgroup CSV峰值约
207.35 GiB，memory events全0；与历史run相同elapsed的cgroup增长近似。旧
`observe→(h,denoise_ind) artifact→mean-one`路线已被
[训练修改计划](05_TRAINING_MODIFICATION_PLAN.md)中的recent-5 global公式取代。
若只改善6帧视频的观看速度，可设低fps，但不会增加信息。

在上述v1 run之外，用户已冻结v2 R-only公式。新child只替换DVAC统计/映射，成功GRPO的任务、SFT、
2卡16 env、G8、B512/mb32、update2、flow-SDE、chunk reward/logprob、优化器、FSDP和PPO两层clip
均保持不变。服务器`11 passed`、two-rank统计probe、default-off compose和新旧resolved对照均通过；
v1最终完整g54的训练rollout success为88.67%，g1–54均值86.66%，历史GRPO同窗口88.67%，差
`-2.01pp`；两者均无训练中held-out eval。该run为运行v2 smoke而按用户授权停止，服务器保留g50 DCP，
本地closeout ZIP见本文档入口。v2两步smoke于23:06启动、23:35自然完成：step1 warmup全1并建立
512-query逐h history；step2 p05/median/p95约`0.66/1.02/1.20`、rank-local mean约`0.975/0.981`，
grad/ratio/KL均finite，双rank第二步NPZ与g2 DCP存在，memory events全0。v2 100-step formal packet随后
完成冻结并获批；formal于`2026-08-22T00:13:16+08:00`启动：

```text
run     /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
runtime /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_downweight_formal_100step_2gpu16env_20260822
PID     wrapper/driver/observer = 198255/198259/198260
```

2026-08-22 09:54现场刷新时，三进程仍alive，最新完整Global Step为23/100，g10/g20 DCP存在；
g1–23训练rollout success均值为82.52%，历史成功GRPO同step轴为84.97%，两者均非held-out eval。
R-only g2–23权重p05/median/p95约为0.715/1.031/1.198，均值0.9995；g23前后25格平均权重只差
`+0.0034`，说明per-h位置校准已大幅去除固定future-h趋势。GPU峰值30.37/30.22 GiB；cgroup峰值
226.62/240 GiB、live约217.39 GiB，memory events全0。完整产物、训练、DVAC和资源解释见
[12号现场分析](12_R_ONLY_FORMAL_LIVE_ANALYSIS_G23_20260822.md)，后续状态仍以服务器现场为准。

2026-08-22 15:01现场刷新更新上述动态事实：最新完整Global Step为35/100，探针结束时Step36 rollout
`8/16`；三根进程仍alive，g10/g20/g30 DCP存在。g1–35训练rollout success为85.27%，历史GRPO同轴
86.75%；最近10步为91.02% vs 90.70%，仍无稳定领先。g2–35权重p05/median/p95均值为
`0.708/1.030/1.199`，正/负advantage query mean weight为`0.988/1.019`。GPU峰值30.37/30.22 GiB；
cgroup live约223.3/240 GiB、CSV瞬时峰值237.41 GiB，memory events全0；数据盘余约727 GiB。完整图、
产物、v1/v2信号演化和机制解释见[15号文档](15_R_ONLY_G35_SIGNAL_CREDIT_AND_MECHANISM_CHAIN_20260822.md)。

2026-08-22 21:12现场刷新：最新完整Global Step为50/100，三根进程仍alive，后续rollout继续。g50单步
training-rollout success为94.14%；g1–50原GRPO/v1/v2均值为`88.23/86.33/87.26%`，最近5步为
`91.72/88.05/92.19%`，最近10步为`90.47/88.95/90.90%`。v2近期追平并略高于原GRPO，但累计均值仍
低0.97pp，且尚无held-out eval。v2 g2–50权重p05/median/p95均值为`0.692/1.029/1.199`，weight ESS
约0.972、平均系数方向变化约9.1度。GPU峰值30.37/30.22 GiB；cgroup峰值触到239.9999/240 GiB，
`memory.events max=36140`但`oom=oom_kill=0`。完整三run图、方法图和资源图见
[16号文档](16_R_ONLY_G50_THREE_RUN_TRAINING_AND_RESOURCE_ANALYSIS_20260822.md)。

2026-08-22 21:42，v2 formal按用户授权在完整g51后主动停止；未完成的后续step不计入结果。
wrapper/driver/observer均退出，g50 DCP完整保留。g1–51原GRPO/v1/v2 training-rollout success均值为
`88.312/86.512/87.393%`；v2最近5/10步比原GRPO高`0.625/0.977pp`，但累计低`0.919pp`，仍无
held-out结论。g51最终weight ESS=`0.973`、等效H=`48.65`、系数角=`9.0°`。用同批residual映射到
`[0,2]`的反事实为ESS=`0.838`、等效H=`41.9`、top-20% mass=`31.9%`、系数角=`23.0°`；建议v3只改
`weight_min/max=0/2`，其余保持v2。停止后GPU归零、`oom=oom_kill=0`；轻量包与完整讨论见
[17号文档](17_R_ONLY_G51_CLOSEOUT_AND_V3_RANGE_DISCUSSION_20260822.md)。

2026-08-22 22:37:37，用户授权的v3 fresh-SFT 100-step formal启动。相对v2只改
`weight_min/max=0/2`；新配置commit为`eb2a091…`，源码与resolved哈希均已冻结。run/runtime为：

```text
run     /root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
runtime /root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_r_only_v3_w0to2_formal_100step_2gpu16env_20260822
PID     wrapper/driver/observer = 70610/70614/70615
```

22:44:51三根进程及2 actor/2 rollout/2 env worker均alive，driver首个真实rollout已到`2/16`；
两卡约24.0/25.7 GiB，cgroup约134.8 GiB，`oom=oom_kill=0`。Step1仍是
`w=1` warmup，`[0,2]`从Step2开始应用。RoboTwin可选curobo traceback与成功v2启动一致，不影响当前
worker存活与rollout推进。完整启动合同、前测和逐指令证据见[18号文档](18_R_ONLY_V3_W0TO2_FORMAL_LAUNCH_20260822.md)
及其实施账；后续step与资源只能通过服务器现场刷新。

2026-08-22 23:42最终现场刷新：最新完整Global Step=`2/100`，Step3 rollout=`7/16`；三根控制进程和
2 actor/2 rollout/2 env worker均alive，fatal匹配为0，`oom=oom_kill=0`。g2实际权重
p05/median/p95=`0.365/1.171/2.000`，per-query ESS=`0.896`、等效H=`44.8`、系数角=`18.1°`，
确认v3比v2 g2（ESS=`0.985`、角度=`6.7°`）明显更强。g2 success=`69.14%`、KL=`0.052`、
query clip=`15.1%`、pre-clip grad=`47.279`；该success来自第一次加权update之前的rollout，最早的
行为影响要看g3。双卡峰值29.38/28.80 GiB，cgroup峰值166.77/240 GiB，无新增memory event；
当前无checkpoint符合save10合同。图、CSV和解释见[19号文档](19_R_ONLY_V3_EARLY_LIVE_ANALYSIS_G2_20260822.md)。

2026-08-23 10:15锁定g28现场：最新完整Global Step=`28/100`，Step29 rollout已启动；三根控制进程和
六个核心worker均alive，fatal=0、`oom=oom_kill=0`。g28 training-rollout success=`90.63%`；v3
g1–28均值=`85.90%`，与原GRPO同轴`85.57%`接近，最近5/10步高`2.81/4.26pp`，仍不是held-out评估。
g28 weight p05/median/p95=`0.399/1.158/2.000`，ESS=`0.895`、等效H=`44.7/50`、系数角=`18.4°`。
GPU峰值`30.37/30.22 GiB`；cgroup已触`240 GiB`且max event累计新增`38,362`，但无OOM，主压力来自
EnvWorker anonymous memory并叠加file/cache。g10/g20 DCP均约9.7 GiB。完整图、CSV和资源同比见
[20号文档](20_R_ONLY_V3_LIVE_ANALYSIS_G28_20260823.md)。

10:31:33最终只读刷新时，g28仍是最新完整更新，Step29 rollout=`12/16`；全部核心进程仍alive，
GPU约`26.5/28.9 GiB`，cgroup current约`237.2/240 GiB`，`oom=oom_kill=0`，未做任何干预。

2026-08-23 13:31–13:36再次只读刷新：最新完整g36，Step37 rollout=`4/16`；wrapper/driver/observer和
六个核心worker仍alive。g36 success=`95.703%`、KL=`0.060`、query clip=`17.483%`、pre-clip grad=
`33.938`。v3 g1–36相对原GRPO累计/最近5/最近10步success为`+0.543/+0.703/+1.406pp`；仍是训练
rollout。g36权重p05/median/mean/p95=`0.451/1.176/1.196/2.000`、ESS=`0.897`，negative-advantage
mean weight=`1.295`高于positive的`1.163`。GPU峰值`30.368/30.218 GiB`；cgroup current约
234.45 GiB、峰值240 GiB、`oom=oom_kill=0`。g10/g20/g30 checkpoint各约9.7 GiB，72个DVAC NPZ齐。
图表与机器可读材料见[22号文档](22_R_ONLY_V3_LIVE_ANALYSIS_G36_20260823.md)。

2026-08-23 15:56–16:00同口径只读刷新：最新完整g42，下一轮rollout已开始；全部核心进程仍alive。
g42 success/KL/query-clip/pre-clip-grad=`93.359%/0.0733/16.671%/44.244`。v3 g1–42相对原GRPO累计/
最近5/最近10步success为`+0.372/+0.313/+0.352pp`，仍是训练rollout。g42权重
p05/median/mean/p95=`0.328/1.101/1.118/2.000`、ESS=`0.876`、系数角=`20.00°`；negative-advantage
mean weight=`1.222`高于positive的`1.084`。GPU峰值`30.368/30.218 GiB`；cgroup在15:56一度
`239.986/240 GiB`、16:00约233.13 GiB，峰值240 GiB但`oom=oom_kill=0`。g10/g20/g30/g40 checkpoint
各约9.7 GiB，84个DVAC NPZ齐。本次没有干预训练；图表与机器可读材料见
[23号文档](23_R_ONLY_V3_LIVE_ANALYSIS_G42_20260823.md)。

2026-08-23 18:45–18:46同口径只读刷新：最新完整g48，Step49 rollout约`12/16`；全部核心进程仍alive。
g48 success/KL/query-clip/pre-clip-grad=`84.375%/0.0510/16.453%/41.155`。v3 g1–48相对原GRPO累计/
最近5/最近10步success为`+0.326/-0.703/+0.352pp`，仍是训练rollout，未形成稳定领先。g48权重
p05/median/mean/p95=`0.456/1.140/1.167/2.000`、ESS=`0.906`、系数角=`17.37°`；negative-advantage
mean weight=`1.222`高于positive的`1.145`。GPU峰值`30.368/30.218 GiB`；cgroup约232.30 GiB、峰值
240 GiB但`oom=oom_kill=0`。g10/g20/g30/g40 checkpoint各约9.7 GiB，96个DVAC NPZ齐。本次没有
干预训练；图表与机器可读材料见[24号文档](24_R_ONLY_V3_LIVE_ANALYSIS_G48_20260823.md)。

2026-08-23 19:06--19:08最终刷新与收尾：v3完整g49后按用户授权停止，未完成的下一轮rollout不计入。
g49 success=`92.578%`；g1--49相对原GRPO累计/最近5/最近10步为`+0.351/-0.703/+0.313pp`。
g49 weight p05/median/mean/p95=`0.493/1.185/1.209/2.000`、ESS=`0.913`、系数角=`16.72°`；
negative-advantage mean weight=`1.291`高于positive的`1.179`。旧PGID=`70608`已退出，两卡显存归零，
`oom=oom_kill=0`。完整见[25号文档](25_V3_G49_CLOSEOUT_AND_METHOD_ANALYSIS_20260823.md)。

2026-08-23 19:27:49，用户授权的global-z `[0,2]` fresh-SFT 100-step formal启动。相对v1只把
`strength=.1`改为`.5`；相对v3只把per-h residual换回global-z，其余沿用成功GRPO。commit=`afdaa2e2…`，
wrapper/driver/observer=`820640/820644/820645`、PGID=`820638`。19:40首个真实rollout epoch已到`5/16`，
六个核心worker alive，GPU约25.9/25.5 GiB，cgroup约119.7 GiB，`oom=oom_kill=0`。见
[27号启动记录](27_GLOBAL_Z_W0TO2_FORMAL_LAUNCH_20260823.md)。

2026-08-24 16:34--16:41最终现场刷新：global-z `[0,2]`最新完整g49，g50 rollout仅`14/16`。旧driver/
observer分别最后写入`15:59:22/16:00:27`，而当前容器PID 1启动于`16:03:26`，故训练随容器重启中断；
容器内日志不能确定外部重启触发源。g49 raw/5-step/10-step success=`93.359/93.203/93.320%`，共同
g1--49相对原GRPO累计/末5/末10=`+2.081/+2.734/+2.930pp`。g49 weight ESS=`0.897`、有效H=`44.83`、
系数角=`18.22°`；GPU峰30.37/30.22 GiB，cgroup峰240 GiB，run期OOM/OOM-kill=0。当前GPU/RAM已释放，
最新完整DCP为g40。图、CSV、解释与逐指令账见
[31号收尾分析](31_GLOBAL_Z_W0TO2_G49_RESTART_CLOSEOUT_AND_SUCCESS_SMOOTHING_20260824.md)。

2026-08-24 18:09--18:18，RLT teacher-DVAC v2真实smoke自然完成：driver exit0；8 train episodes、20 fixed
eval episodes、8 critic/4 actor updates、两rank8个DVAC trace与完整`global_step_1` checkpoint均闭环。
run级weight p05/median/mean/p95=`0.263/0.959/0.991/1.820`，ESS=.880；GPU峰17.11/17.19 GiB，
cgroup峰57.18 GiB，OOM/OOM-kill=0。v1暴露的rank-local conditional metric key已用固定sum/count schema
窄修，`5 passed`，最终`513dbcb7...`已push。详见[32号结果](32_RLT_DVAC_REAL_SMOKE_RESULT_20260824.md)。
