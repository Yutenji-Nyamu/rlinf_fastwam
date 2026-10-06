# U-GROW信号接入BC与RLT-BC · 2026-10-06

本窗口负责深圳1机物理GPU4的BC 4/U5＋U与GPU5的RLT-BC＋U。用户授权整理上下文、实现、检查、smoke、推送和运行；U实验优先，原卡RLT仅在本次事务终止、确认释放后接替。6/7由另一窗口执行DSRL，本窗口不操作其进程、源码或路由。凭据不落盘。

## 当前依据与方法

规划来源：`C:/Users/86136/Documents/seek/outputs/ugrow_training_readiness_2026-10-06/{01_BC,02_RLT,05_U_PRODUCER}.md`；原信号实例为`outputs/dvca_signal_study_2026-10-03/23_UGROW_INSTANCE.md`。

同输入、同完整初始噪声，以确定性ODE分别完整生成10步和5步动作终点。逐坐标总体标准差除以RMS＋1e-8，平均14有效坐标，保留动作位置。复用主10步、prefix cache；旁路5步保存/恢复RNG。主动作与原数据预算保持。U在rollout/reference query计算、随回放保存，不在SGD microbatch重算；BC仅训练采集启用，RLT固定评估也执行额外ODE5。

| 实验 | GPU4：BC＋U | GPU5：RLT-BC＋U |
|---|---|---|
| 历史来源 | Clean01d770db，旧DV ad3da329 | π0.5 N4 pair460008008aea310f83a0b19008c7c105a0a19980 |
| 任务/模型 | move_pillbottle_pad / Sidney π0.5 | adjust_bottle / Sidney π0.5 |
| 预算 | 100轮，N4/U5，global1024/micro32 | 800轮，N4/U5，B512/micro256；回放至少10k，再做15k初始化更新 |
| 动作 | ODE10/H50/D14，200动作上限 | 教师ODE10/H50，学生C10/D14，200动作上限 |
| 信号映射 | 过去5轮log统计，旧bounded_linear[0,5]，首轮1、入成功池冻结，长度过滤off | 成功query上两层exp_mean，τ2.5/dropout0.2/双α按全局R500退火；失败权重1 |
| 更新位置 | 逐H FM误差 | reference BC逐H MSE；原Q/critic保持 |
| 初始化 | 原SFT，空成功池/校准状态 | 原任务专属完整Stage1，新Stage2回放 |
| 固定评估 | 32条，每5轮；每10轮保存 | 20条，每25轮评估/保存 |

这是U-GROW式动作信号的本地训练适配；原论文是state-level rollout-start选择。高U表示两种求解预算分歧较大，按旧映射获得更大训练权重；不等同于因果credit或已证实收益。BC旧映射不保证均值1。RLT历史早期收益并非晚期平台优势。

## 调度边界

权威旧路由实时读取`/data/chenyiteng/deployment-20261004/rlt-after-dojo-n25-v1/active-queue-continuation.json`。旧per-card协调器SIGTERM会影响全部children，禁止用于本次借卡。只对已派发GPU4/5的精确driver身份进行停止；该协调器每角色只派发一次，不自动重启。

借卡前核完整CP、UID/PID/start/boot、namespace、配置、GPU UUID。原RLT是N8候补，恢复时保留其原任务/方法/累计3000轮，与新N4实验区分。smoke至正式之间属于同一租用期；清理完成且本事务明确终止后才恢复原卡RLT。只更新4/5对应watch条目，并使用现有锁。

## 执行状态

12:43北京时间，两条源码已实现并冻结：BC `1c4b3810a1cddcd1ae38312126448c80348f5686`，RLT `bfbc9c889bfe687dc24de7c3dc1d66138d3010df`；本地/服务器13文件各自hash核同、树干净。服务器BC39项、RLT69项、借还7项CPU检查通过。一次准备检查把RLT的`max_epochs=1000/max_steps=800`误读为`max_epochs=800`，检查已按真实max_steps修正，训练实配保持。

原4/5 RLT精确driver已停，原owner确认两角色终态，namespace与C/G均释放后启动本卡U owner：GPU4 owner2432037/start414849667，GPU5 owner2432097/start414849911。两卡保留CP100（回放15581/15557，update0），候补原任务为place_fan clean/combo、N8/3000。停止时CP100之后未保存预采集需归还时重做。

v1独立部署目录`/data/chenyiteng/deployment-20261006/ugrow-bc-rlt-g45-v1`，分`bc`/`rlt`两条生命周期。两路smoke训练均exit0：BC完成两轮，success1/4→2/4，成功池query3→9，第二轮U权重nonunit_fraction=1、std0.7033；旧owner错用BC不存在的`update_step`而误拦。RLT沿旧B32/micro16/U1、两轮/2初始化更新，完整CP2记录update4，U/损失/梯度有限，但0成功；第2轮已切仅训练过一次的student，因此未覆盖成功权重分支，严格验收未过。两路都清理后归还原CP100 RLT，5卡已验返回首轮回放15557→15713；4卡亦已派发再交接。没有把任一路的旧gate失败写成方法收益失败。

13:01:39起，当前权威事务改为`/data/chenyiteng/deployment-20261006/ugrow-bc-rlt-g45-v2`，GPU4 owner2485016/start414961836，GPU5 owner2485099/start414962114。12项新服务器CPU检查（另3子测试）通过。只修独立ops：BC按两轮真实loss/grad/lr/replay验收并复用已完成smoke，直接接fresh formal；RLT只做一次固定4轮补充smoke，将smoke max_steps/save_interval从2改4、warmup_post_collect_updates从2改8，确保四轮沿teacher采集，不筛seed，仍须自然成功+非均匀U权重+有限更新。正式配置保持原预算，均不从smoke续权重/回放。两卡返回driver再次精确停止，当前driver自己的cleanup、namespace/进程/C/G清空实证齐全后启动v2；没有伪造旧owner新确认。

当前源码HEAD仍为上列两条，独立ops位于v2/tools并固定hash。smoke至formal间保持本事务所有权，事务终止清理后才从完整CP接回候补RLT。smoke上限90分钟，BC formal48小时/RLT7天；输出盘低于40GiB、异卡绑定或进程失败停止并归还，不删除历史文件、不reset GPU。

13:22收尾核验：BC正式前3/100轮已写出，success分别1/4、2/4、2/4；第二轮loss0.0086873、grad0.111724、U权重std0.730544/nonunit_fraction1。正式首轮验收回执齐全，继续后续训练。RLT补验13:14:53正常exit0，完整CP4/update8、回放320；四轮实际actor_switch_rate=0，每轮2次critic/1次actor更新，ref BC loss和grad有限。16回合实际0成功，仍未覆盖成功权重，owner因此FAILED并明确不启动正式。U namespace/进程/C/G已清空，13:15恢复旧place_fan combo N8/3000：driver2516932/start415042475，run/namespace `rlt-g5-after-ugrow-1006-v2`，从原完整CP100/回放15557继续；13:21:17恢复首轮已验，step100、success1/8、回放15710（增长153）、update0，沿原20k预采集门槛。此为U验收覆盖不足，不是训练异常或已证明U无效。

当前4卡BC owner/driver有活跃心跳；5卡U owner已完成归还职责，后续是原RLT driver。计算与图形均在各自4/5卡，0–3未见C/G（保留驱动基础显存）；6/7由另窗管理。本轮末次输出盘余量/home约316GiB、/data约443GiB；训练源码两树仍干净、HEAD和pins核同。127项服务器CPU检查（另3子测试）通过；BC真实smoke通过，RLT真实成功权重验收仍缺，两个结论分别记录。

解释覆盖边界的既有证据：2026-09-19历史同一N4 adjust_bottle pair前16轮teacher预采集，Clean2/64、Combo7/64，分别出现连续9轮、6轮零成功；提取文件`local_scripts/ugrow_bc_rlt_20261006/evidence/historical-rlt-teacher-first16.json`含原路径/sha/逐轮scalar。本轮不再扩大smoke或反复挑种子。5卡U后续先补真实成功样本上的独立分支验证，再决定新正式事务；原U采集、回放、有限参数更新和CPU分支/梯度测试证据均保留。

12:43:28两条源码已推到个人GitHub并分别以ls-remote核同：[BC源码](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/1c4b3810a1cddcd1ae38312126448c80348f5686)、[RLT源码](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/bfbc9c889bfe687dc24de7c3dc1d66138d3010df)。这两个源码提交内的`docs/ugrow_CONTEXT_20261006.md`是实施前快照；后续执行文档和轻量证据位于[独立证据分支](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/ugrow-g45-evidence-20261006)，其中`docs/ugrow_CONTEXT_20261006.md`为本文件发布副本，`docs/ugrow_execution_20261006/`含实配、scalar、CP/借还和测试证据，`tools/ugrow_ops_v2/`为修正ops。冻结运行HEAD保持，发布文档不改运行工作树。

本轮工作文件：`local_scripts/ugrow_bc_rlt_20261006/`。其他窗口请保持GPU4/5本窗口事务所有权，不重放旧RLT恢复入口。
