# EXPO-FT 实现与 SZ2 smoke 的配置依据

**历史范围：主体§1–6记录2026-10-01早期小规模smoke，文内driver、未决点和未验收项均指当时。** 后续已完成正式N1/B64规模、10回合warmup后每40真实动作记一次学习额度、fixed-C10 TD窗口及完整恢复验收；action expert＋原生投影、无LoRA也已确定。当前正式实配与差异以[2026-10-03训练审计](AUDIT_20261003.md)和[熵/并行机制](ENTROPY_PARALLEL_20261003.md)为准；评估25→10执行状态另见[部署记录](EVAL10_DEPLOYMENT_20261003.md)。下文保留早期依据，不把历史smoke改写为正式训练结果。

日期：2026-10-01。本页记录已授权SZ2实现的配置/忠实性合同；真实fresh＋新进程resume smoke已通过，最终数值及资源归还证据见[实施记录](IMPLEMENTATION_LEDGER_20261001.md)。完整调研见本目录既有笔记。静态依据为官方 `023cf9cfcb09dab962b6e806fea47d2954b2b9bb` 的原 `EXPOLearner`、配套OpenPI `46407a41183b037313a383ff679683f2773b766d`，以及已跑通 π0.5 的 PyTorch 接口。

## 1. 继承 control 与新增方法参数

| 类别 | 首版合同 | 依据 / 含义 |
|---|---|---|
| 模型 / 归一化 | π0.5 RoboTwin 已跑通 checkpoint、同配套 norm/assets 与 transforms；明列路径和SHA | 继承Control，不能替换为官方DROID7D checkpoint或norm；启动前核验SZ2实际模型/文件身份 |
| 动作 | 双臂14D canonical，模型padding32D；H50、C10、每episode最多200真实动作 | 继承Control；论文原版H16/C4或8的代码配置属于不同control，不暗改 |
| 推理 | 继承ODE10、图像/指令字段与action decode；normalized14D子空间加delta后仅一次输出transform | 当时已跑通π0.5接口；candidate/Q/replay在同一normalized空间；不继承RLT全动作identity为无界residual |
| 候选 | N8 base，每个1个edit，共16；rollout与TD均hard argmax | 官方方法参数；这里N8是每观测8份提案，和RLT每轮8环境/回合不同 |
| edit | tanh Gaussian，delta scale=.2；不复制DROID xyz/gripper的3:6旋转mask | 原配置默认.2；C10×14=140维；双臂关节与Cartesian索引不同。完整combined action不额外tanh |
| Q / 更新 | 10Q，selection及target随机2取min；actor目标为online10Q均值；UTD20，B64 | 源码合同；UTD是一update call的critic次数，随后各1次base/edit/temperature |
| 小组件优化 | Adam3e-4，γ=.99，τQ=.005，α初始1、entropy_scale1 | 官方默认；target entropy=-140/2=-70；logprob含tanh Jacobian和`-140*log(.2)`scale修正。官方另维护τbase=.001副本但backup不用；当日早期smoke未实现这份无行为用途的base副本，单列差异 |
| 数据 | sparse原生success；Q使用实际执行chunk，base使用演示/成功episode的真实H50动作序列 | 不复制旧DSRL成功0/失败-1奖励；成功标签覆盖整episode，不仅reward=1的末步 |

官方关键位置：[参数](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/configs/model/expo_ft_pi_config.py)、[selection / backup](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L521)、[edit loss](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L694)、[critic TD](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L775)、[update call](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/expo_ft/agents/alg/expo_ft.py#L888)。

细节：官方selection与target value两次随机Q子采样相互独立；edit update条件动作为replay的执行chunk；其共享视觉特征stop-gradient，梯度进入edit而不进入base。官方backup采样live base，虽维护base Polyak copy，上述固定commit的original learner并不使用该copy采样next action。实现可优化计算，但须保留该语义或明确记录差异。

## 2. Base FM 参数组：先明确首版是什么

论文EXPO-FT D.1称LoRA SFT初始化、RL时图像encoder冻结；发布代码的`freeze_pi05_encoder`仅控制候选共享编码，不能证明训练冻结，OpenPI filter静态未排除SigLIP。这是论文与发布源的真实差异；以运行前打印的trainable参数清单及实际更新为准。

本工作区已有PyTorch `Pi0.compute_loss` / `OpenPiPytorchSFTActionModel`，支持真实FM；`freeze_vlm()`的现成策略冻结SigLIP、LLM expert0与embedder，训练action expert1及action/time等projection。首个smoke可采用这一 **action-expert-only EXPO-FT移植**，借用已跑通参数组而不增加新LoRA结构；FM更新仍是真实base更新，但不能称与论文LoRA参数组完全一致。

验收必须记录trainable名字/数量/dtype/optimizer，冻结组无梯度、允许组至少一项参数非零变化；不能用“FM loss有限”替代更新证据。当日早期smoke driver对每个冻结参数固定64个元素的样本作更新前后hash检查，不能称整组全量权重checksum。base FM基于normalized+padded `[B,50,32]` 真动作；使用`use_rlt=False`普通FM，不能混入RLT reconstruction。若视觉/语言prefix冻结，cached prefix跨base更新可复用；action suffix候选必须使用新base版本。

base optimizer可用现有SFT模板 `examples/sft/config/robotwin_sft_openpi_rlinf.yaml` 的AdamW LR2.5e-5、betas(.9,.95)、eps1e-8、weight_decay1e-10、clip1作smoke起点；FP32 master、bf16 compute。这是本地接口依据，不把它称原EXPO-FT论文的base默认。不能直接把小组件3e-4学习率套到数亿base参数。短smoke核对有效LR不因首步warmup为0，并记录master参数Δ，避免bf16舍入掩盖/消除小更新。论文LoRA与当日PyTorch expert-only不同，正式长期实验前需要锁定参数组和base LR；首个闭环smoke不以其学习曲线证明最优配方。

## 3. Smoke 的短预算与方法口径

smoke可串行1环境、critic B4、base B1或更小、少量update calls和小replay，保持H50/C10/14D/200、16候选、10Q等核心语义。若显存/时长要求降低UTD、候选microbatch，逐项写到resolved/receipt；microbatch必须保持8份独立noise，不能把8候选缩为一个复制8次。N/M降为1无法验收多候选edit选优路径，不能用来代替完整smoke。

当日早期smoke driver在每个完整episode结束后只调用1次更新：20次critic、1次真实FM、1次editor/temperature；不是每C10 chunk或每真实动作20次critic。当时正式cadence尚未实现/验收。官方固定commit的示例是至少完成10 episode后、每episode 3次update call；其训练入口另支持按step/episode/batch与采集transition数触发，须和原论文历史协议分开。[官方触发逻辑](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/train_pi_robo.py#L391)、[示例参数](https://github.com/pd-perry/expo-ft/blob/023cf9cfcb09dab962b6e806fea47d2954b2b9bb/scripts/pick/run_server.sh)。因此UTD20通过仅证明一update call内的20次critic执行，不证明正式每采集chunk的更新预算或样本效率。

短预算用于验证接线、梯度与恢复；成功回执可证明一call内UTD20真实执行，不能据其宣称正式B64、采集/更新cadence、八环境规模或论文样本效率已验收。无HIL是允许自主学习的RoboTwin variant（原论文消融支持），不要求为真机干预协议临时部署遥操作。RTC、noise-Q filter不在本次原版方法范围。

## 4. 真正完成 smoke 的最低回执

1. **实际环境**：π0.5生成独立8候选，真实RoboTwin执行选出的C-step canonical动作，原生200上限；记录seed、动作步数、reward/success与select index。随机tensor、离线replay-only不算环境smoke。
2. **TD与边界**：保存实际动作与真实next obs，chunk内成功后的bootstrap=0；truncation不取reset obs当final obs。只从同episode的有效H50窗口做FM；不把未执行的base预测尾部伪作监督。
3. **学习**：Q、edit、temperature各有实际optimizer step和有限参数变化；target-Q更新，edit/Q梯度不穿base；FM至少一次真实允许参数更新。若真实rollout无成功，可用已确认成功演示触发FM，但如实报告on-policy success=0与FM数据来源。
4. **完整恢复**：base可训练权重、Q/edit/temperature、所有optimizers/targets、RNG、replay cursor/数据/成功标签、更新计数与source/model/norm/config指纹保存；新进程resume后校验相等，再实际执行/更新并产出新checkpoint，文件和loss均有限。
5. **资源归还**：唯一run-id与精确PID/boot/start归属；仅按已授权范围暂停SZ2物理4–7的四条RLT，记录完整CP、原实配、精确停止与恢复首轮证据，不动其他用户或共享Ray基础进程。停止/失败路径不能留下占卡子进程。正式验收中的RLT状态由现场刷新证明。

以上全部满足才报告“smoke通过”；其中任一项省略，报告具体已通过的子项。正式启动前仍需根据现场资源列命令、GPU/内存/输出和停止条件，不能把本页建议当作已派发配置。

## 5. 当时需要讨论的未决点（后续决议见页首路由）

- 2026-10-01用户已确定：正式action-expert-only variant，不用LoRA，冻结视觉/语言prefix、训练action expert和原生投影。该项已决；base LR2.5e-5除本地模板亦核对作者所选OpenPI配置。后续[正式讨论方案](FORMAL_DISCUSSION_20261001.md)记录资源、方法预算和必要数据/规模改动，在该讨论截点均尚未派发。
- 正式预算如何按采集chunk/真实动作确定update call频率、warmup与base FM成功数据/demo混合？本smoke的每episode一次、FM优先最后一个成功H50窗口否则单份demo，只验收闭环，不是正式数据分布与cadence。
- 本版已选variable-K TD、三个camera共享torchvision ResNet50/GN、无augmentation。正式对照需明确保留该RoboTwin variant，还是对齐作者fixed-C窗口、Flax ResNetV2/两camera融合及增强；奖励、bootstrap与loss-valid仍须分别定义。

任务/种子选择、串行1环境、B4、有限调用数、checkpoint目录及microbatch属于已授权smoke的常规实施选择，可自行完成并留下实配；无需为每一项再次停下来确认。

## 6. 当日早期smoke实现快审与验收范围（2026-10-01）

审阅文件：`examples/embodiment/train_expo_ft.py`、`rlinf/algorithms/expo_ft/{core,backend,replay}.py`（本机在`local_scripts/expo_ft_20261001/port`下）。当次smoke授权为SZ2物理4–7借卡：原四RLT由冻结helper精确暂停、smoke结束后继续原累计3000预算；smoke使用物理4。fresh1episode+resume1episode，各200动作、1 update call/UTD20，实际均完成并保存新checkpoint；新进程状态哈希一致，全部本次进程释放。没有无限正式训练循环，RLT/Git收尾状态以实施记录为准。

核心方法语义符合上述合同：独立候选noise、normalized base+delta硬选优、selection及TD独立随机Q pair、actor10Q均值、共享当前critic视觉stop-gradient用于next/editor、20critic→1FM→1edit/temp；已执行canonical动作回encode，成功在线FM窗口来源为连续真实H50动作。无成功在线窗口则使用已准备clean50演示，必须在回执注明该来源。

明确变体：torchvision bottleneck ResNet50+GN32、三个view共享同塔后concat，区别于作者Flax ResNetV2与两camera融合；未启用作者的crop/rotation/color-jitter；variable-K折扣与terminal短动作normalized zero尾/valid1，区别于官方fixed-C有效窗口；base仅action expert/projection更新、无LoRA及未使用base EMA副本；短预算每episode一次call，当时正式B64/cadence/并行规模未验。C10×14维在normalized空间加bounded delta=.2，combined action整体不额外tanh，再一次native decode；replay反编码实际已执行canonical命令。

此前静态阻断已在当日smoke源码修正：`digest()`先flatten再`view(uint8)`，支持0D温度/optimizer scalar；backend schema2对starting model实际全部权重文件作稳定SHA manifest并固定norm身份，factory加载后再核对文件身份，resume逐项比较contract。冻结base仍从这些原文件重载，checkpoint保存其可训练子集及optimizer。另已复用成功DV50推理的H100兼容策略，仅关闭cuDNN SDPA，保留flash/efficient/math SDPA及普通cuDNN convolution，并写入runtime contract。fresh/resume回执证明Q、target、edit、temperature、base允许参数均真实更新；冻结参数无梯度且抽样值一致。两个在线episode均失败，FM来自已核验演示，不能把接线成功表述为训练效果提升。
