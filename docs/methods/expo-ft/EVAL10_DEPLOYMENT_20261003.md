# EXPO固定评估25→10：迁移与资源续接记录

**17:02状态：eval10合同迁移、完整恢复、call242完成及新checkpoint保存均已验证；call243执行中。** 2026-10-03，北京时间。本文追加执行事实；[13:02训练审计](AUDIT_20261003.md)和[熵/并行机制讲解](ENTROPY_PARALLEL_20261003.md)中的历史测量保留原时间。

## 授权变更与不变合同

用户授权把固定评估间隔从每25个训练回合改成每10个；保持20个固定评估种子、每回合200动作、N4评估，以及原latest/last1保存方式。在线仍为N1、20,000真实动作总预算、10回合warmup、每40个warmup后动作记一次学习调用、B64、Q20/FM1/editor1/temp1。熵目标、α设置、采集并行度与训练方法未在此变更中调整。

正式driver只改变评估合同检查中的两个25→10字面量；周期调度原本读取`evaluation.every_episodes`。保存函数、优化器、replay、动作计数和运行输出目录保持。迁移另外更新inputs及checkpoint外层合同；六类训练payload（base、core、replay、cadence、rng、progress）要求迁移前后完整digest相等。

## 第一次交接为何中止

15:19，旧driver按SIGTERM在提交边界退出；停止回执与完整checkpoint一致：**75回合、11,677真实动作、241次已完成学习调用，另有5次待执行额度、10动作余数**。checkpoint约8.79GB，有限性检查为true，原件保留。

交接控制器随后立即检查GPU4–7的计算进程列表，仍观察到分配，于`Old driver did not release GPU4-7`门禁中止，没有接受该次eval10合同迁移。进程退出与GPU分配消失不是同一观察时点；本次记录证明门禁触发，不把瞬时列表直接解释成驱动故障或新的训练方法问题。

原唯一资源owner继续原有归还流程。15:29，`RLT_RESTORED`、`gpu_released=true`、`rlt_first_rounds_verified=true`及四卡逐项首轮证明均已取得。旧owner把人为停止的driver退出143记录为formal失败；这不是20k训练自然完成，也不是训练指标退化的证据。[停止边界与归还轻量回执](evidence/20261003-eval10/first-attempt-and-return.json)

首次交接的旧owner/handoff实现不作为本次发布的可执行入口。原现场错误证据保留，不重放旧owner。

随后新cycle的prepare发现：这四条刚归还的RLT run尚未达到25轮保存周期，尚无新的完整checkpoint根目录，因此prepare在借卡前停止，没有再次停止任何RLT。当时处理是只读等待每条新run生成完整checkpoint后再prepare；不使用旧checkpoint回退。等待逻辑已纳入本包并通过服务器CPU检查。16:00快照中唯一新owner为4086166/start381843377，状态`WAITING_FOR_RLT_CHECKPOINTS`；四条原RLT继续运行，尚未新借卡，训练inputs实际评估间隔仍为25。[等待状态轻量回执](evidence/20261003-eval10/waiting-for-fresh-checkpoints.json)

另核对了借卡后、训练driver启动前失败的归还合同：managed进程列表为空时应标记`not_started`；driver已启动后的失败仍标记`failed`。旧CPU等待owner精确退出并归档为`attempt-2-cpu-wait-release-contract`，该次修订没有停止任何训练；当前唯一owner与最终37项检查使用本包同一版本。

## 15:45暂停截点

这组图来自15:45只读快照，**状态为暂停、等待受控续训**；不能作为恢复后推进的证明。已完成75回合、11677动作、241次学习，33/75在线成功；最近5/10/15/20回合分别为40%/60%/53.33%/45%。固定评估仍只有45%→50%→30%这三个点，第75回合评估尚无新增结果。

![15:45暂停截点：逐回合与滑动成功率](figures/20261003-1545/sliding-success.png)

![15:45暂停截点：学习与行为](figures/20261003-1545/behavior-dynamics.png)

[指标摘要](evidence/20261003-1545/summary.json) · [逐回合CSV](evidence/20261003-1545/per-episode.csv) · [行为统计](evidence/20261003-1545/behavior-statistics.json) · [图表核验与SHA](evidence/20261003-1545/visual-qa.json)。13:02两张图仍按原时间保留在原审计中。

## 16:47受控续接进展

16:41前，四条刚归还RLT的新完整checkpoint均经原严格validator验收，GPU4/5/6/7分别使用1875/1900/1850/1850步。四条新CP的历史replay引用需沿既有`repair_latest`重建，重建后再次通过完整性/合同检查；没有改用上一轮旧CP。16:46:45停止回执确认四个精确driver均退出、四个原namespace均空、物理GPU4–7释放。

两条等待队列于16:41:36完成改绑，cycle plan SHA为`3495dcdfce01fc4ba4e1b48c25493f9552f52c332c464b2e3262fc19059548f8`。`place_object_stand`新owner为57232/start382093697，`move_playingcard_away`为57246/start382093758；各自Stage1完整权重仍为10,018,009,814字节，`stage1_restarted=false`、`training_config_changed=false`，clean/combo继续等待本cycle归还。

16:46:49唯一EXPO owner4086166/start381843377进入`MIGRATING_EVALUATION_CONTRACT`；16:47:26采样时迁移完成回执尚未产生，训练inputs仍为eval25，不能据目标配置声称已恢复。[受控续接轻量回执](evidence/20261003-eval10/continuation-resume.json)保留精确身份、四CP合同、队列plan/source SHA、原始快照SHA和当前阶段。熵目标与α设置、采集N1、latest/last1保存方式保持。

源码、机制讲解、历史图表与前期检查已首推[codex/sz2-expo-ft-repair-20261002 @1386f943](https://github.com/Yutenji-Nyamu/rlinf_fastwam/commit/1386f943ab388bdcdc5bf6c16d7557627cfb30bc)，远端已独立核验。本地整理分支为`codex/expo-eval10-20261003`，与实际推送的远端分支分别记录。本段后续执行回执为增量收尾，不把首推时的等待状态当作未同步源码。

## 16:51合同迁移完成与新driver启动

16:51:21，CPU迁移回执`ok=true`：inputs评估合同已改为每10回合、每次20样本；latest与last1的base/core/replay/cadence/rng/progress六类payload迁移前后digest逐项相同，`training_payload_changed=[]`、`save_policy_changed=false`。latest边界仍为75回合/11677动作/241已完成调用、pending5/carry10；last1仍为74回合/11477动作/241调用。replay路径身份及索引SHA保持，截断回合为0。旧/新checkpoint整体SHA因外层合同变化而不同，不等于训练payload变化。

16:51:51，owner4086166/start381843377启动新driver1089953/start382155251。16:51:59采样时两者存活、心跳距采样约1.8秒，driver处于`initializing`；尚无`resume_verified`或新增学习事件。现场17项源码SHA与本轮首推1386f943的Git对象逐字节一致；只有driver评估合同的两处字面量相对原17项合同变化。[迁移和当前身份回执](evidence/20261003-eval10/continuation-resume.json)包含两checkpoint完整六payload前后hash及原始采样SHA。该16:51截点尚无恢复校验；16:55与17:02的后续证据见下。

## 16:55完整恢复校验

16:55:37出现真实`resume_verified`：base/core/replay/cadence/rng/progress六类恢复hash均与16:51迁移后的latest完全一致；完整恢复75回合/11677动作/241已完成调用、pending5/carry10。随后driver进入`learner_started`、call242，按原待执行学习额度继续。16:56:15采样时owner/driver仍活、心跳距采样约1.0秒，两条新队列owner均活且旧owner均已退出；clean/combo保持等待本cycle归还。固定评估输入仍为10/20，17项训练源码持续核同。

该16:56采样只记录到“完整恢复已验、首个新调用执行中”；17:02后续已取得调用完成与提交证据，见下一节。熵设置、采集N1、latest/last1保存策略均未调整。[恢复回执及事件](evidence/20261003-eval10/continuation-resume.json)

## 17:02首调用完成、保存与权威续接路由

17:01:59，call242出现`learner_finished`，耗时381.72秒；Q loss为0.00011394、base FM loss为0.08221274，全部记录指标有限。base采样参数中208项发生变化，冻结参数带梯度数为0，base并行设备数为4。α为1.076185，是继续原自动温度学习得到的值；本轮没有修改熵目标、α学习配置或采集N1。

17:02:37，对应`checkpoint_committed`为`finite=true`，core/base更新计数均为242，latest大小8,786,985,668字节；已完成75回合、11677真实动作保持，pending从5变4、carry仍10。这里先消化恢复前的学习额度，尚未采集第76回合。17:02:46采样时call243执行中，owner/driver均活、心跳距采样约9.0秒，17项源码仍与首推Git对象相同，两条新队列owner活、旧owner已退出，仍等待本cycle归还。调用完成与有限checkpoint提交共同证明恢复后已发生真实学习推进。

原训练目录新增`active-continuation.json`权威指针，SHA为`c337677b947e1b11d1df1f7b289c0d0bedb4ea4375c9f4250517dc691d1fd3db`，指向`eval10-continuation-20261003/current.json`、同级final与新resource cycle；旧训练目录的current/final描述已结束的前cycle。指针含精确owner/driver、输入与迁移回执SHA及resume事件。[恢复、首调用提交和指针回执](evidence/20261003-eval10/continuation-resume.json)附原始采样文件SHA256。

本次修改只改变固定评估间隔25→10；每次评估仍20样本。熵设置、采集N1、训练20k预算及latest/last1保存方式保持。首包1386f943已独立核验；恢复证据随本提交保存，最终发布对象由所在提交SHA及发布回执标识。尚未运行到下一评估点80回合，因此固定评估曲线仍为45%→50%→30%，没有新增评估结果或补造第75回合评估。

## 新控制路径和两条等待队列

训练目录继续使用`formal-turn-switch-repair-20261002`；新资源控制目录为同级`eval10-continuation-20261003`，独立cycle为`rlt-cycle-expo-eval10-20261003-v1`。执行次序为：

1. 核实原owner退出、GPU释放、原RLT四卡首轮全部归还；只读等待四条新run各自出现complete标记，随后由原bridge.prepare严格验证payload/合同并准备新cycle。最多等待两小时，身份变化、失败或超时即关闭本次准备；不回退旧CP。
2. 将`place_object_stand`与`move_playingcard_away`两条已完成Stage1、尚未切换clean/combo的等待队列改绑新cycle。原plan只允许改变`gates`与`old_runs`；原训练runs、源码HEAD、配置、预算和Stage1权重保持。
3. 原等待owner精确退休后，continuation取得同一`owner.lock`，保留原owner身份，另写`owner-identity-continuation.json`。校验Stage1完成、约10.02GB full weights及成功退出回执，复用已完成Stage1，不再launch/wait Stage1。
4. 两条新队列均登记且SHA核同后，新owner才借回4–7。沿既有精确RLT桥接逻辑停止目标，GPU清空后至少隔2秒再次观察为空；默认90秒有界等待，若分配再次出现则重新计空闲时间。
5. 在CPU完成合同迁移并逐payload核对，然后从原完整checkpoint恢复。训练退出后只由新owner释放其进程、续RLT并核首轮；等待队列仍按原gate/stop/launch/watch逻辑行动。

队列owner/wrapper不执行共享Ray重启、驱动操作或宽泛进程清理。单独发布这些脚本是为了复核这次固定路径的执行合同，不代表可以重放已完成的单次操作。

## 发布源码与现场部署映射

| 仓库路径 | 现场映射/用途 |
|---|---|
| [正式driver](../../../examples/embodiment/train_expo_formal.py) | 审核后暂存到新控制目录，迁移时原子替换原source下同路径 |
| [reborrow_owner.py](../../../tools/expo_eval10_20261003/reborrow_owner.py) | 新控制目录`tools/`；唯一资源owner |
| [reborrow_resources.py](../../../tools/expo_eval10_20261003/reborrow_resources.py) | 新控制目录`tools/`；沿原精确RLT bridge准备/借还 |
| [queue_continuation.py](../../../tools/expo_eval10_20261003/queue_continuation.py) | 新控制目录`tools/`；跳过已完成Stage1，继续两条等待队列 |
| [rebind_nextsix.py](../../../tools/expo_eval10_20261003/rebind_nextsix.py) | 新控制目录`tools/`；单次精确改绑与ready回执 |
| [migrate_eval10.py](../../../tools/expo_eval10_20261003/migrate_eval10.py) | 新控制目录根下；CPU合同迁移和完整原件保留 |
| [preflight_migration.py](../../../tools/expo_eval10_20261003/preflight_migration.py) | 新控制目录根下；真实checkpoint只读CPU预检，非控制器运行依赖 |

[工具包README](../../../tools/expo_eval10_20261003/README.md)记录部署映射、三份原字节测试和两份driver必须保持同SHA的约束。

模型、checkpoint、replay、完整日志和凭据不进Git。原始大文件继续保留服务器；轻量回执附原始快照SHA256便于核对。

## 验收状态（17:02）

15:59最终候选版本的服务器CPU检查已经通过：资源续接14项、队列16项、合同迁移7项，共37项，包含新checkpoint等待/超时/身份变化拒绝检查；没有初始化训练CUDA或启动额外训练。[原始CPU检查回执](evidence/20261003-eval10/cpu-tests.json) 发布测试文件与服务器已测字节一致；工具目录另保留同SHA的driver部署补丁/测试fixture，正式examples路径仍是唯一训练源码入口。

15:54另完成真实checkpoint的只读CPU预检：17项现场源码与原合同一致；latest与last1均完整读取、分别验证base/core/replay/cadence/rng/progress六类payload的既存digest，边界分别为75回合/11677动作与74回合/11477动作。latest同时核对停止回执、checkpoint回执、replay索引及零截断回合；两文件读取前后stat保持。driver补丁恰为两处字面替换，未初始化CUDA、未修改训练状态。此时原source/inputs仍为eval25，新控制器继续只读等待四条RLT的新完整CP。[真实CPU预检原始回执](evidence/20261003-eval10/migration-preflight.json)；这项预检通过尚不代表正式迁移或恢复已经发生。

| 验收项 | 17:02已核状态 |
|---|---|
| 原停止checkpoint与stopped计数一致 | 已核75回合/11677动作/241调用，pending5 |
| 原资源owner归还四卡RLT并验首轮 | 已核；不是新EXPO恢复证明 |
| 借卡前真实CPU预检 | 已核17项源码、latest/last1完整六payload与75/74回合边界；未改训练状态 |
| 归还RLT的新完整CP与精确借卡 | 四CP1875/1900/1850/1850已严格验收；16:46:45四driver停、namespace空、GPU4–7释放 |
| 新队列ready、旧队列退休且Stage1未重跑 | 两新owner已登记、ready为true；Stage1重跑与训练配置变化均false |
| eval10实际输入、source和checkpoint合同一致 | 迁移ok；inputs为10/20，17项现场源码与首推Git字节一致，latest/last1新外层合同均已保存 |
| 六类训练payload、replay索引及保存方式保持 | latest/last1六payload前后全量digest相等，replay索引/路径身份保持，save_policy_changed=false |
| 新driver `resume_verified`及首个真实推进 | 16:55:37恢复六hash已核；17:01:59 call242完成，17:02:37有限checkpoint已提交，call243执行中 |
| 首次新间隔固定评估 | 尚未达到下一点80回合；当前仍只有初评/25/50三点，不补造旧评估 |
| 云端commit、远端SHA及部署源码SHA一致 | 首包1386f943已核，现场17项源码逐文件SHA核同；恢复证据随本提交保存，最终对象由所在提交SHA及发布回执标识 |

本次从75回合恢复后先消化原pending_calls；已提交其中1次（call242），尚余4次额度，随后继续原轨迹序列。下一个固定评估整十点为80回合，结果以实际评估事件为准。

本回执已记录新owner/driver、迁移时间、old/new inputs与17项source SHA、latest/last1六payload完整digest、队列ready/plan SHA、四GPU精确借卡、恢复及首调用提交、权威续接指针。后续训练/评估按原20k预算继续，以新current和事件为准；13:02、15:00与15:45历史图保留原采样时间。
