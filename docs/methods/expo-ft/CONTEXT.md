# EXPO-FT 专题上下文

2026-10-09 EXPO修复验收：2机4/5正式续训，250回合/35720动作/847学习已保存，N4第250回合评估已执行真实动作（全20场仍进行中）。64GiB CPU缓存当前23.99GiB，热调用844–847平均2.41分钟，旧近20次11.0分钟；补齐新建仿真前CUDA闲置显存释放。driver357974/start427782238、observer357981/start427782257，权威deployment-20261008/bc-signal-tau-v1/expo-owner/current.json；RLT4/5权限修复且WAIT_EXTERNAL，SZ2队列owner358370/start427783043，BC6/7原driver92252/92254保持。源码已推1ecb05b44，17份pins及双CP六payload恢复核同；正式验收轻量同步收尾。原60k/B64/Q20/8+8/N1/评估10×20及保存不变。详见REPLAY_CACHE_REVIEW_20261008.md；勿重放旧owner。

2026-10-07 11:52 EXPO只读产物审计：196回合/28227/60000动作/655学习已保存，第656进行中；两卡已连续60次完整学习及180/190两次N4×20评估。固定160/170/180/190为16/13/12/13（/20），近20在线12/20，续训尚无稳定新高；GPU4/5约76.2/65.3GiB、RSS35.8GiB，源17项核同。196回放392文件身份核同，latest/last1目录可读；成功100条有22条短于H50不进FM，近20成功12条中7条被排除。当前协调器已为coexist owner3008973，driver3503441保持；旧RLT等待EXPO+Q优先组结束且待rlt-return-validation.json，不再沿用旧立即归还表述。本轮未启停或调参。[图表、关键帧及讨论](review-20261007/review.html)。

2026-10-06 23:40 EXPO两卡正式续训已验：GPU4单卡B512候选生成OOM；GPU4–5完整N1/N4及B64/Q20/FM1探针通过，正式六payload恢复核同，第596次学习已保存（633.45秒）。当前完整CP为174回合/25697/60000动作/596学习；只改设备数，所有方法/微批/评估/保存参数保持。唯一owner18670/start410221234、driver3503441/start410425820；控制`/data/chenyiteng/projects/expo-ft-sz2-20261001/parallel-trial-20261006`，源码eedc7188、51项锁定文件核同。0–3/6–7无本进程上下文，两路RLT保持WAITING_EXPO_60K且原归还链不变；其他窗口只读新current.json，不重放旧owner。见[两卡试验及续训验收](PARALLEL_TRIAL_20261006.md)。

2026-10-06 11:25发布核对：20k复盘和60k续训已在云端；补充今天曲线、学习日志摘录及CPU准备/回放修复/RLT暂停回执。现场23472/60000动作、155回合、539次学习已保存，固定130/140/150回合为16/20、15/20、14/20；原owner及4–7绑定保持，RLT候补。17份运行源码核同，本次仅补文档和轻量证据。[发布核对与产物](SYNC_20261006.md)。

2026-10-05 18:35 SZ2 EXPO已完整恢复并续训：129回合/20058/60000真实动作/454次学习已保存，第455次学习进行中，尚未提交。新增40k预算、方法/评估10×20/滚动保存保持。唯一owner1367524/start399995223、driver1571275/start400002165，权威控制`/data/chenyiteng/projects/expo-ft-sz2-20261001/continue-60k-20261005`；代码`codex/expo-60k-20261005@061d4995`、17份运行源码核同。GPU4–7实际运行、0–3无C/G；四路nextsix RLT保留CP300/300/325/325并等待。EXPO完成/异常/心跳停滞900秒后，精确清卡再恢复RLT累计3000并验首轮；其他窗口仅观察，勿重放任何旧owner。 [60k续训与验收](CONTINUE_60K_20261005.md)。下方20k终态为历史记录。

2026-10-05 17:05 EXPO终态核验：SZ2 turn_switch已于00:41完整结束，128回合/20000真实动作/454学习，最终固定15/20（初始9/20）；7个场景失败转成功、1个成功转失败。latest/last1及128条回放齐全，源码17/17核同；全程约60h，学习44.84h。原RLT恢复首轮已验，00:52 nextsix接续，17:05四路TRAINING。GPU0现有本人RLT图形535MiB，EXPO已退出。本轮只读审计并整理轻量产物，未启停训练。 当前入口：[最终复盘与产物](FINAL_REVIEW_20261005.md)。下方状态均为各自日期的历史记录；不重放旧owner。

2026-10-03 17:02评估频率25→10已迁移、完整恢复与首个新学习调用保存已验：四条RLT新CP1875/1900/1850/1850严格验收后精确借出GPU4–7，两条已完成Stage1的等待队列改绑新cycle，未重跑Stage1。唯一owner4086166/start381843377、driver1089953/start382155251；16:55:37完整恢复75回合/11677动作/241调用，17:01:59完成call242，17:02:37提交finite checkpoint（core/base242、pending4/carry10），call243执行中。评估输入10/20及17项源码核同；熵设置、采集N1、保存方式与训练配方保持。现场当前路由以原训练目录`active-continuation.json`指向的`eval10-continuation-20261003/current.json`为准，旧current/final属于已结束的前cycle。首包已推远端`codex/sz2-expo-ft-repair-20261002` [1386f943](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/1386f943ab388bdcdc5bf6c16d7557627cfb30bc)；恢复证据随本提交保存，最终对象由所在提交SHA及发布回执标识。详见[部署记录](EVAL10_DEPLOYMENT_20261003.md)与[恢复回执](evidence/20261003-eval10/continuation-resume.json)，解释见[熵/并行机制](ENTROPY_PARALLEL_20261003.md)。以下13:02与更早记录为历史截点。

历史只读审计（2026-10-03 13:02）：同一修复scope仍在运行，69回合、10842/20000真实动作、223次学习调用；固定评估45%→50%→30%，暂无稳定增益。5/10/15/20回合滑动曲线、采样/评估/保存解释、官方差异及GitHub同步缺口见[完整审计](AUDIT_20261003.md)。当前运行17份锁定源码已与修复分支`0bd979f`逐文件核同；在该13:02截点，此次新增审计/图表与CONTEXT更新仍仅本地，当时未变更运行参数或推送。后续状态见页首路由。

当前路由（2026-10-02 12:40）：旧开关初评9/20=45%完成后发生SIGSEGV，零在线动作/零学习。已在不加载policy的N4原生复现中重现；补丁先shutdown线程池再释放场景，三轮N4→N1共六次关闭全部通过。新scope `formal-turn-switch-repair-20261002`、owner197649/start371818311、driver2892740，物理4–7正式完整恢复已验；首回合59真实动作成功并保存checkpoint，第二回合warmup进行中，尚无学习。原配方、20k预算和初评保持。四RLT已精准退出，仅本owner在EXPO结束/故障且GPU释放后续原累计3000。当前唯一入口[修复运行](NATIVE_REPAIR_20261002.md)，源码分支`codex/sz2-expo-ft-repair-20261002`。下面旧scope启动记录保留作历史，不重放旧owner。

最新任务：2026-10-02 SZ2 `turn_switch` EXPO正式已启动，物理4–7、原正式配方/20k真实动作保持。独立scope `formal-turn-switch-20261001`，owner839628/driver942043；四RLT已精准暂停，仅EXPO退出后恢复。见[换任务运行](TURN_SWITCH_FORMAL_20261001.md)。已推[codex/sz2-expo-ft-turn-switch-20261001](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz2-expo-ft-turn-switch-20261001/docs/methods/expo-ft) @67cff0224a55a3387bc09a6ec92484da55bca901，远端SHA核同。以下瓶子正式与smoke为保留历史。

更新：2026-10-01。深圳2四卡B64完整smoke和独立进程恢复已PASS，Q20、FM/editor/temp各1，action expert及小组件实际更新、冻结prefix无梯度。一次call207秒，四卡采样显存峰值约45.0/37.4/37.4/37.4GiB；详见[规模smoke](FORMAL_SCALE_SMOKE_20261001.md)。正式20k真实动作含10回合warmup已启动，原四RLT清理完成，仅EXPO退出后恢复。正式路由与现场进度见[正式运行](FORMAL_RUN_20261001.md)。此前单卡小batch smoke及14:31归还回执仍保留在[运行包](SMOKE_PACKET_20261001.md)，不能混作本次正式结果。

## 已确认

- 后续本轮用户明确：方法预算/Q视觉按论文及作者结构，暂不使用图像增强；要求正式规模短smoke，不进正式训练。新四卡B64候选合同、回放修正、资源路由和验收范围见[正式规模smoke](FORMAL_SCALE_SMOKE_20261001.md)。旧一次smoke已完成回执保持；新scope/新cycle不重放旧guardian。
- 2026-10-01正式讨论：用户明确action expert＋原生投影，不用LoRA；沿现有冻结视觉/语言prefix策略。方法侧优先论文/固定作者源码，RoboTwin基座合同保留。资源、数据/更新预算、TD窗口与Q视觉选择见[正式讨论方案](FORMAL_DISCUSSION_20261001.md)；本轮只读核验/维护文档，未借卡或启动正式训练。
- 原版EXPO-FT arXiv2605.25477v2（2026-08-17）；来源EXPO2507.07986v3；Real-Time2609.18207v1为独立扩展。
- base proposal → bounded action-space edit → 原/edited候选Q选优；rollout和TD下一候选均选优；edit用Q/entropy，base继续FM监督更新。当前官方发布实现默认base用成功episode，原版论文未明确列这个success-only开关。原配方8base＋8edit，不等于我们并行采集N8。
- 官方核心训练闭环公开，DROID/π0.5示例；未找到全部论文8任务demo/norm/SFT/最终checkpoint包，没有现成RoboTwin适配。
- 推荐在现有RLinf PyTorch接口独立实现原版；完整新增范围包括candidate sampler、edit/action-Q、图像/指令/真实执行序列replay、base FM及完整restore。
- 当前RLT Stage2输出完整chunk并冻结VLA；旧已验DSRL port训练32D生成noise。这些可借基础设施，不能直接换维度混称EXPO。

## 来源身份

- 官方主仓库：`pd-perry/expo-ft@023cf9cfcb09dab962b6e806fea47d2954b2b9bb`。
- 原版OpenPI：`pd-perry/openpi`，branch`expo_ft`，`46407a41183b037313a383ff679683f2773b766d`。
- DROID：`pd-perry/droid@04b7551ff8a4901b9914cc4dbca4d066771ae9a7`。
- 只读本地源码：`local_scripts/expo_ft_20261001/official-*`，不发布嵌套clone、依赖或模型；pins在`local_scripts/expo_ft_20261001/source-pins.json`。
- 深圳1实际π0.5源：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-rlt-n8full-sz1-20260929@55c1399a50826d61e8735a64daa2f1742f1b824f`，branch`codex/sz1-pi05-rlt-n8full-20260929`，2026-10-01只读时clean。
- 现役四个Stage2：精确driver身份存活，`rlt_mlp_policy`、identity、C10/14D、8环境/200动作、U5、critic:actor=2、compact replay、bootstrap_on_truncation=true。已结束Stage1的base合同：`pi05_sidney_robotwin`，H50/C10/14D、ODE10、Sidney norm、`robotwin_aloha_canonical_v1`；其resolved无algorithm/env属于Stage1配置，不是部分Stage2 overlay。完整N8调度另见`docs/server-admin/RLT_N8_FULL_RESTORE_20260929.md`。未读取生产feature wrapper的完整调用路径；openpi_rlinf eval/SFT只作为已有接口的移植候选。
- 现场证据：`E:/Codex/home/visualizations/2026/09/27/01a0e2cf-e398-7f90-99e5-7013b133ea33/server-review/sz1/expo-ft-source-20261001-0956.json`；六处源码及SHA在同目录`expo-ft-source-files-20261001-1003.json`。只读脚本：`local_scripts/expo_ft_20261001/target_probe.py`。未记录凭据。

## 未决/后续需验

以下为历史实施问题的定位；B64/四卡、物理步replay/demo-Q、成功FM随机批次、作者结构Q视觉已实施并通过规模smoke；图像增强按用户决定关闭。任务pick_diverse_bottles与20k预算已锁，warmup/cadence的21项服务器CPU检查通过；实际长训和固定评估正在正式scope运行，效果待结果。当前未决按正式运行页，不把旧建议当最新状态。

1. official image trainable filter与论文“冻结视觉”的差异；以显式参数组和FM权重变化证明。
2. early-terminal chunk valid/continuation、timeout/final obs、实际执行K与固定C折扣；公开issue尚未给完整结论。
3. base候选batching/KV共享在当前RLinf π0.5的正确shape与成本；8env×8candidate独立预算。
4. RGB/full-H online replay内存及成功episode储存、base-weight同步与strict恢复范围。
5. method budget：edit scale/active dimensions、10Q/2-min、critic/base/editor cadence、trainable base组、demo/HIL协议；不继承RLT U5/BC-Q课程当EXPO默认，不把DROID pick的xyz/gripper mask下标直接套RoboTwin14D关节。

## 阅读入口

先读[综述](EXPO_FT_REVIEW_20261001.md)，按问题再到[论文笔记](RESEARCH_PAPER_NOTES_20261001.md)、[开源/运行审计](CODE_AUDIT_20261001.md)、[RLinf映射](RLINF_MAPPING_NOTES_20261001.md)。后续实施沿当前已跑通的π0.5模型/动作合同，仅在独立source/config/output/namespace中接方法；运行前列配置差异、命令、资源和停止条件。

当前正式分支：[codex/sz2-expo-ft-formal-20261001](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz2-expo-ft-formal-20261001/docs/methods/expo-ft)，四卡B64源码、完整恢复/正式启动轻量回执已推送并校验远端一致。旧单卡交付分支：[codex/sz2-expo-ft-20261001](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz2-expo-ft-20261001/docs/methods/expo-ft)。server root `/data/chenyiteng/projects/expo-ft-sz2-20261001` 下保留大checkpoint及实际部署合同。
