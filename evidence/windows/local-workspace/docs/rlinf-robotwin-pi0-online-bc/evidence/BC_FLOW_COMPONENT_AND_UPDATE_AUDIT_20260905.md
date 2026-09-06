# 在线成功BC：逐轮流程、继承组件与更新预算

2026-09-05。这是主SSOT下的实现解释与审计，不另立一套实验计划。原审计BC `9876c28d`，SFT包裹修复`5ae809d5`，EnvWorker FD修复`72a92604`；最新证据提交`700b6846`已push。13:39收尾：v6已失败退出，无活跃BC/正式/DVAC；当前共驻留容量问题与资源选择见[本轮结果](BC_RESULT_AND_NEXT_RESOURCE_CHOICE_20260905.md)，下文各时间点保留为定位历史。用户已授权正式，但必须先通过完整容量smoke；不复用失败smoke或诊断权重。

## 1. 图像：原样存，按模型接口读，不等于不预处理

- 保存：采集时三路RGB以uint8保存，240×320；不保存随机裁剪或随机变亮后的副本。图像/状态取本次决策之前，标签取实际提交给RoboTwin的C50×14命令。
- 必需输入变换：通道/布局、resize到模型分辨率、像素范围、状态和动作的原checkpoint归一化、文本token、内部D32的padding。它们是模型的输入语言；去掉会改变输入语义。
- 可选图像增强：OpenPI训练预处理原有裁剪/旋转/亮度变化。我们复用SFT调用链而继承，不是为BC新增方法；已按用户要求显式关闭，仅BC生效，其他方法默认不变。关闭不能保证对泛化/成功率完全无影响。
- FM随机噪声和时间：给动作构造带噪输入，预测噪声到目标的速度；这是π0原生监督目标本身，不是图像增强，不能一起删。每个batch重新抽取，不直接复用采集时的4步去噪轨迹作训练标签。

BF16/FP32均是计算数值格式，与原始图像是否被篡改是两件事。本次兼容错误的确与我们的适配有关：BC起初额外强制FSDP BF16，覆盖了官方π0 DAgger的`precision:null`，使FSDP入口图像/计算参数与原生FP32图像网格、状态投影不匹配。`precision_processor`仅搬设备，并非错误来源。先前只给图像补FP32的修法不完整，已撤掉；现在恢复原生BF16主干/FP32投影的null配方。不是“参考代码本来必然报错”，也不是成功筛选数学要求BF16。

## 2. 每轮准确做什么，来自哪里

| 阶段 | 当前实际行为 | 来源/取舍 |
|---|---|---|
| 初始化一次 | 从既有adjust_bottle SFT载入；冻结VLM，训练action expert及投影；成功池空 | 用户选定起点/训练范围；原生OpenPI factory |
| 发布策略 | actor→rollout同步当前完整权重；每轮开始一次，评估前再同步 | RLinf EmbodiedRunner；更新后必须影响后续采集 |
| 收集 | 单卡32环境、串行1批、episode预算200、C50/M4，最多4个有效query/episode | 并发与串行用户明确；模型/任务沿对应π0配方 |
| 行为采样 | 正常随机初始高斯＋4步ODE，不用GRPO中间SDE探索，也无teacher | BC rollout显式mode=eval；这里eval不是没有初始随机噪声 |
| 结果筛选 | 接到真实任务success时保留此前所有query；失败/超时未成功丢弃；后续终止后query不入池 | 新SuccessEpisodeCollector，避免DAgger仅收专家/干预片段 |
| 对齐边界 | 一张query前图像对应整个提交的C50动作chunk；不是C50张逐动作图像 | RoboTwin只返回chunk后观察，不新增逐动作渲染 |
| 累计数据 | 先独占创建每批成功archive `.pt`，再加入CPU常驻累计成功池；不设近期窗口/淘汰 | 新SuccessReplay；借Hi-ORS成功入池思想，不搬其人工纠正/动作过滤 |
| 训练抽样 | 从累计成功query均匀有放回抽1024；拆32个micro32做梯度累积，step一次；重复2次 | 复用RLinf监督更新；不是shuffle全池两遍 |
| 监督更新 | 用已提交命令，经原norm进FM；只归约有效H50/D14；clip1、AdamW、常数LR | 原生OpenPI SFT + 新mask接点；无PPO比率/adv/Q |
| 空池/示范 | 空池一致跳过；demo_weight=0不加载示范，正值才额外算示范loss并按比例混合 | 新BC适配；混D0开关是用户明确要求，保留 |
| 评估/保存 | 正式fixed32每5轮、每10轮模型/optimizer/scheduler/RNG/累计池/更新计数保存 | RLinf评估/策略保存＋BC sidecar；smoke每轮验证 |
| 下一轮 | 保留模型、Adam和累计池，更新后的策略再采集；不每轮重置SFT | 在线同步分轮设计，不照搬SIME每轮重训语义 |

重要细节：RoboTwin将整块命令交给TOPP插值/压缩；当前mask的全1表示提交目标有效，不声称每个原waypoint都物理执行完毕。训练采样单位是query，长成功episode有更多query，故获得更多抽样概率；不是episode均匀抽样。

源代码：本地镜像`worktrees/pi0-online-bc/rlinf/data/online_bc.py`、`workers/actor/fsdp_online_bc_policy_worker.py`、`workers/actor/fsdp_dagger_policy_worker.py:update_buffer_one_epoch/run_training`、`workers/env/env_worker.py:_run_interact_once`、`runners/embodied_runner.py:run`、`models/embodiment/openpi/openpi_action_model.py:prepare_dagger_sft_batch/sft_forward/sample_actions`（后五个路径均从rlinf/起）。

## 3. 不需要/可精简的继承项，按优先级排序

| 顺序 | 项目 | 当前是否生效 | 结论 |
|---|---|---|---|
| P0 已处理 | 随机图像增强；额外FSDP BF16强制覆盖/逐层cast | 增强off；恢复null、撤图像cast | 遵守用户方法选择并修正接口，勿再带入正式 |
| P0 已窄修，完整smoke中 | 额外`use_orig_params=True`＋expert-only decoder层包裹 | v4后更新导出失败；单改False不足；正确wrap已通过连续2次更新及完整导出/保存读回 | 联合SFT绕过decoder forward；用已有wrap_policy配置包实际调用的投影/MLP/norm，保存格式独立local_shard；不跳过缺失权重 |
| P1 可删工程冗余 | BC之外通用trajectory builder还暂存chain、model_action等，随后整批clear；采样器仍算/存无用logprob/链 | 确实执行，但成功archive已只保留需要字段 | 后续窄减；不可删训练必要obs/action，也不可因去掉ODE中`noise×0`改变随机数序列后声称行为完全相同 |
| P1 可冻结而非删权重 | action expert附带的语言lm_head | 标requires_grad却不参与当前FM；独立probe更新后无Adam状态，生产warmup会为它预建状态 | 可从训练参数中冻结，保留模型/checkpoint键兼容；本轮未与FSDP修复一起改训练范围，不把总requires_grad参数都说成每步收到有效梯度 |
| P1 值得单独优化 | DAgger先把整个GB1024输入搬GPU，再逐MB训练；额外image clone/prepare | 确实执行 | 不改变数学即可改成逐MB搬运；本轮先保留预算可比，不混入新内存优化 |
| P2 文档/配置清理 | adv_type=gae、gamma、value_lr、flow_sde等通用字段 | adv被跳过，无critic；mode=eval不走SDE | 明确“未使用”，按validate_cfg调用依赖再删；写在配置不等于方法生效 |
| P2 无需急拆 | DAgger继承类上的LeRobot/teacher/干预支持、若干锁/成员 | 本BC没启用；setup已替换 | 未加载teacher、不运行LeRobot采集/专家筛选，不能说全套DAgger在训练；为少几个成员复制整个训练器反而扩大维护面 |
| 保留，但记资源代价 | actor/rollout offload，FSDP，train/eval独立环境 | 单卡共置实际需要实测；环境未offload | 属运行设施，不是BC数学；32+32环境共存的完整峰值需smoke覆盖 |
| 必須保留 | resize/norm、FM噪声时间、有效动作mask、成功终止边界、冻结范围、同步、恢复、固定评估 | 生效 | 不因追求少代码而删掉这些语义边界 |

DRQ已false、域随机化off、teacher=null、reward-model/critic/value-head均off；没有继承Q/V/target网络、RLT latent actor、GRPO优势/ratio/clip、RECAP条件标注、Hi-ORS人工纠正/no-op过滤。原生SFT每次关闭模型checkpointing后，Gemma专家forward又强制启用自身gradient checkpointing：这是以重算换显存，不是另一种loss，也不能仅看配置False就断言彻底关闭。

另一个必要而非算法性的原生细节是`warmup_optimizer_state`：用LR=0和零梯度为Adam分配状态，再恢复LR/梯度并把step归零，不学习数据、不改变权重；这样保存/恢复不会因某个尚无梯度的参数缺状态。它不算本轮的一次BC更新。诊断若漏掉它，会制造生产初始化本来已避免的空状态错误。

## 4. BC更新量与依据：不能混淆三个单位

当前每轮：**32新episode尝试 → ≤128新query → 从累计成功池抽2048次query → 64个micro前后向 → 2次AdamW更新**。

`U=2`、`B=1024`、micro32，梯度累积32次；每轮总presentations=`U×B=2048`。100轮上限200次optimizer、204800次query presentations，3200个训练episode；失败/空池时实际更新可少于上限。

若首轮成功池有75–84个query，每个query期望在本轮被抽24.4–27.3次；相同图像/命令配新的FM噪声时间，仍是高数据复用，不是新数据。累计池变大后单记录复用下降。`2/32`是更新/episode比；`2048/新增成功query`是样本呈现/新成功query比；都不能直接和论文不加单位的UTD比较。

| 依据 | 已核实做法 | 支持什么/不支持什么 |
|---|---|---|
| 官方固定RLinf π0 RoboTwin DAgger `dc9b87c` | micro32/global1024、LR2.5e-5、Adam(.9,.95)、clip1；**U1**；全模型训练、4actorGPU；warmup1000/total30000/cosine | 同模型监督FM的优化器/batch起点；不是证明expert-only自成功BC该用U2、常数LR或单卡同样最优 |
| 旧同任务π0两卡GRPO | 每轮256episode、U2、GB1024/MB32 | 解释当时预算沿用从哪来；PPO多遍同轮on-policy数据≠累计replay BC，不能充当最佳BC依据 |
| Hi-ORS固定作者代码 `5fa4b23` | 只SFT、micro16/GPU、成功池阈值10 records、每50learner步发布策略；异步采集/训练 | 直接支持在线成功池FM；不能把`SFT_steps=100`读成每收一轮固定更新100次 |
| Hi-ORS论文§III-D | 约1.5秒一训练迭代、自然UTD约1；持续异步 | 支持根据交互/算力调整比例，不导出本实验U2/GB1024 |
| SIME固定编排 `8318241` | 分轮收集、合并历史、生成监督训练配置；含非纯成功筛选和重置训练 | 只借外循环，不照搬N/U或每轮重新初始化 |

建议：本次正式先不同时更改已讨论的U2/GB1024，作为清楚标注的首个预算，而非“官方BC复现”。后续若讨论更贴近DAgger，最清楚的单项变更是U1（每轮1024呈现），无需改变并发；若目标是适应小成功池，另讨论GB，不能暗中以动态epochs/UTD自调。没有依据保证增加更新必然提高成功率；固定eval比训练FM下降更能回答是否学到任务。

参考：[官方π0 DAgger配置](https://github.com/RLinf/RLinf/blob/dc9b87cc49334c7516487ead68ebeb060fd7c090/examples/embodiment/config/robotwin_adjust_bottle_dagger_openpi.yaml)、[Hi-ORS作者配置](https://github.com/hiors-project/hiors/blob/5fa4b23f80f420dcb340875daec72a051e161fae/config/algo/reject_sampling_pi0.yaml)、[Hi-ORS原文§III-D](https://arxiv.org/html/2510.26406v1#S3.SS4)。源码正文沿用已落地源锁并本轮重读，网页原文本轮刷新；GitHub raw的本轮网页抓取失败，不称抓取成功。

## 5. 资源与恢复的准确边界

v4于11:48退出，采样GPU6峰值61577MiB=60.13GiB；主机MemAvailable最低1047.55GiB，PSI0。这是已经覆盖的采集/训练/同步阶段，不含成功完成eval/save的完整容量验收。历史observer未记录本job的独立RSS峰值，所以不能把整机占用都算成BC内存。

累计query的三路uint8图像约0.659MiB/条，按100轮最多12800条约8.24GiB纯RGB；含动作/元数据/序列化会更多。每10轮checkpoint复制一次累计池时，10代纯RGB上界合计约45.3GiB；另加原archive和模型/Adam快照。短smoke只验容量/接口，不保证100轮环境生命周期稳定或RAM不增长。

## 6. 回答追问：为什么旧π0 GRPO能用，BC这次出问题？

旧已验证π0 GRPO resolved本轮重读：`precision:null / train_expert_only:true / full_shard / use_orig_params:false`。因此不是“换了π0就必须重写保存”，也不是旧配置开全模型、BC冻结才必然失败。新BC的额外True选择没有足够依据，且单卡改no_shard；更关键的是监督forward调用图不同：

- GRPO prefix cache和suffix推理分开，调用完整decoder forward；包在decoder层的FSDP hooks能执行。
- 原生SFT同时处理prefix/suffix，`gemma_pytorch.py::compute_layer_complete`直接读取decoder的内部投影、norm、MLP，**未调用decoder本身forward**；不能假设沿用expert-only decoder包裹就自动兼容。
- FSDP通过hook管理参数视图/注册/梯度生命周期。绕过包裹边界后，可能表现为训练后导出键缺失或参数原地更新后旧view失效。错误出现在state_dict不代表硬盘坏了，也不证明模型实体权重已删除。
- v4失败在训练后评估前的actor→rollout权重导出，还没到checkpoint保存。官方False单更新导出通过只是局部证据；已将诊断加强为原生optimizer初始化、连续2次更新、同步导出、local_shard保存/读回。

最干净的工程组合按职责选：**旧π0已验证运行/同步/local-shard路径＋官方DAgger监督更新＋小的成功采集/replay适配**。不整支merge GRPO，也不整套复制DAgger。必要差异是SFT实际模块调用边界，使用既有wrap_policy配置对齐；不重写FM或checkpoint。

来源：`BC_SFT_WRAP_SOURCE_20260905.txt`保存实际加载的OpenPI源码与RLinf包裹函数，`BC_FSDP_EXACT_20260905.txt`/`BC_OPTIMIZER_CONTRACT_20260905.txt`保存状态导出和原生local_shard；旧resolved为`docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/dual-2gpu-comparison-live-20260827/raw/control/runtime/resolved.yaml`。真实probe见`SFT_SYNC_CHECKPOINT_PROBE_20260905.json`：2次更新、778键集合导出、被扰动的action_out_proj及其Adam读回相等；未逐值验证全部778权重，也不是完整生产worker重启恢复或长程容量通过。v5/v6最终结果见账本与本轮结果，均未完成全流程。

## 7. v5再失败：这次跨过SFT，遇到旧评估环境边界

12:45:06 v5退出255，首错位于`EnvWorker.evaluate → RoboTwinEnv.reset → camera.update_picture`：`vk::Device::getSemaphoreFdKHR: ErrorInitializationFailed`。64次SFT调用已结束并进入评估（对应原监督loop的2次optimizer）；没有重复v4的state_dict断言。26/32成功episode、78个query已保存，但没有完整外轮指标、固定评估或checkpoint，仍未通过smoke。GPU采样峰值60.43GiB、主机available最低1035.07GiB、memory PSI0；非所查OOM。证据`BC_V5_FAILURE_20260905.txt`、`BC_V5_CLOSEOUT_20260905.txt`、`BC_V5_PROGRESS3_20260905.json`。

定向追溯发现**同服务器旧π0 GRPO已有完全同型失败记录**：`docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/GRPO_DUAL_2GPU_FORMAL_LEDGER_20260826.md`第54行。旧两卡fixed64＝32评估环境/卡，首次评估失败；经用户授权改fixed32＝16评估环境/卡后通过，持续跑到Step52。BC此次单卡fixed32却是32评估环境/卡，同时32训练场景未offload。先前只对齐训练32/卡，却遗漏了已验证评估16/卡的容量边界，这是本次运行配置审计不足，不是成功BC需要新的loss组件。

同报错不证明底层唯一原因已经确定；旧记录也没有唯一分离Vulkan资源上限。12:49只读核实SSH和shared Ray的nofile soft1024/hard1048576；死亡v5 worker的瞬时fd数不可补测。GPU6正在做一次32train+32eval的环境级fd定位，零策略查询/零训练更新，只在确认同一失败点时改变隔离进程soft limit，不改共享Ray。不能先把1024文件描述符与此前OIDN的1024 pthread key混为一谈。

若需改成16评估并行，必须先获用户确认；不能直接`rollout_epoch=2`冒充32不同固定状态，因为`use_fixed_reset_state_ids`会复用相同16个状态。应显式分两批覆盖原32个ID。正式尚未启动，原正式配置保留为计划，不作已通过合同。

恢复测试口径补充：隔离probe比较了完整778键集合，以及被主动扰动的`action_out_proj`参数和其Adam状态读回相等；没有逐值比较全部778个权重，也未完成生产worker重启恢复。

13:00定位补充：独立probe的32训练场景渲染后632FD，再加32评估场景触发明确EMFILE（包括读fd目录与写结果文件失败），首个Vulkan报错更早发生在createFenceUnique。13:05代码`72a92604`已push，生产仅EnvWorker init新增17行opt-in资源下限、配置`min_open_files:4096`；10项测试及compose通过。13:10现场EnvWorker616149的soft为4096、rollout616147仍1024，证明设置作用于目标进程，没有抬共享Ray。v6在跑，尚未验收完整两轮；未降训练或评估并发。

13:39结果补充：v6于13:20:25退出，29/32成功、87query、2次更新后首评估`cannot create buffer`，GPU峰值79.17/79.65GiB、主机available最低1739.91GiB。此次无重复FSDP/EMFILE；强烈支持32train＋32eval同时驻留的显存容量/分配边界，不称训练32并行本身装不下。完整评估/checkpoint均0、正式未启动。下一步建议仅启用已有train/eval环境offload、保持各32并行与学习预算，待用户确认，未实施。无需再做相同隔离probe。六次结束smoke轻量证据已推700b6846。
