# U-GROW信号接入BC与RLT-BC · 2026-10-06

本窗口负责深圳1机物理GPU4的BC 4/U5＋U与GPU5的RLT-BC＋U。用户授权整理上下文、实现、检查、smoke、推送和运行；U实验优先，原卡RLT仅在本次事务终止、确认释放后接替。6/7由另一窗口执行DSRL，本窗口不操作其进程、源码或路由。凭据不落盘。

## 当前依据与方法

规划来源：`C:/Users/86136/Documents/seek/outputs/ugrow_training_readiness_2026-10-06/{01_BC,02_RLT,05_U_PRODUCER}.md`；原信号实例为`outputs/dvca_signal_study_2026-10-03/23_UGROW_INSTANCE.md`。

同输入、同完整初始噪声，以确定性ODE分别完整生成10步和5步动作终点。逐坐标总体标准差除以RMS＋1e-8，平均14有效坐标，保留动作位置。复用主10步、prefix cache；旁路5步保存/恢复RNG。主动作与原数据预算保持。U在rollout/reference query计算、随回放保存，不在SGD microbatch重算；BC仅训练采集启用，RLT固定评估也执行额外ODE5。

| 实验 | GPU4：BC＋U | GPU5：RLT-BC＋U |
|---|---|---|
| 历史来源 | Clean01d770db，旧DV ad3da329 | π0.5 N4 pair460008008aea310f83a0b19008c7c105a0a19980 |
| 任务/模型 | move_pillbottle_pad / Sidney π0.5 | adjust_bottle / Sidney π0.5 |
| 预算 | 400轮（100后接续），N4/U5，global1024/micro32 | 2000轮（800后接续），N4/U5，B512/micro256；回放至少10k，再做15k初始化更新 |
| 动作 | ODE10/H50/D14，200动作上限 | 教师ODE10/H50，学生C10/D14，200动作上限 |
| 信号映射 | 过去5轮log统计，旧bounded_linear[0,5]，首轮1、入成功池冻结，长度过滤off | 成功query上两层exp_mean，τ2.5/dropout0.2/双α按全局R500退火；失败权重1 |
| 更新位置 | 逐H FM误差 | reference BC逐H MSE；原Q/critic保持 |
| 初始化 | 原SFT，空成功池/校准状态 | 原任务专属完整Stage1，新Stage2回放 |
| 固定评估 | 32条，每5轮；每10轮保存 | 20条，每25轮评估/保存 |

这是U-GROW式动作信号的本地训练适配；原论文是state-level rollout-start选择。高U表示两种求解预算分歧较大，按旧映射获得更大训练权重；不等同于因果credit或已证实收益。BC旧映射不保证均值1。RLT历史早期收益并非晚期平台优势。

## 调度边界

权威旧路由实时读取`/data/chenyiteng/deployment-20261004/rlt-after-dojo-n25-v1/active-queue-continuation.json`。旧per-card协调器SIGTERM会影响全部children，禁止用于本次借卡。只对已派发GPU4/5的精确driver身份进行停止；该协调器每角色只派发一次，不自动重启。

借卡前核完整CP、UID/PID/start/boot、namespace、配置、GPU UUID。原RLT是N8候补，恢复时保留其原任务/方法/累计3000轮，与新N4实验区分。smoke至正式之间属于同一租用期；清理完成且本事务明确终止后才恢复原卡RLT。只更新4/5对应watch条目，并使用现有锁。

## 当前执行 · 2026-10-06 15:05 北京时间

用户将累计预算调整为BC **400轮**、RLT **2000轮**，并明确批准仅本次BC保留最近2个完整断点、接续锚点和最终400步，评估日志全部保留。训练N4/U5、batch、任务、种子、固定评估/保存频率、RLT 10k/15k预热与U权重R500退火均保持。

当前原训练不打断、不修改运行中的源码或实配。已准备并启用原终点完整CP自动接续：BC在CP100后继续至累计400，RLT在CP800后继续至累计2000；不是另加400/2000轮。接续同时保存/恢复模型、优化器、回放、学习计数，以及BC U校准状态/RLT U恢复合同。runner.max_steps和max_epochs均设为目标，避免RLT原1000上限截断。

权威预算入口：`/data/chenyiteng/deployment-20261006/ugrow-budget-400-2000-v1/{bc,rlt}`。BC等待/保留owner2765168/start415701241，RLT等待owner2765178/start415701245；15:05均存活、armed、心跳正常，原RLT归还operation.lock分别由本等待owner排他持有。原BC owner2485016/driver2485173与原RLT-U owner2665562/driver2665696仍在v2/v3正常训练，最近记录21轮/24轮，计算与渲染仍各在4/5；6/7不动。

当前段必须真实exit0、原owner终态COMPLETE、C/G与namespace完全释放、目标CP完整后才接续。当前段失败则不接续，沿原协议归还候补RLT。旧owner终点会在已被接续owner持有的非阻塞lease锁处退出；其可能出现的BlockingIOError属于已登记的CPU归还移交，训练exit0/完整CP和新的handoff.json应独立检查。不要看到旧owner退出就重放归还。接续owner完成/失败清理后才恢复旧RLT，保持U全段优先。

接续输出：BC `/home/chenyiteng/results/rlinf-shenzhen/online-bc/ugrow-bc-g4-400-1006-v1`；RLT `/data/chenyiteng/results/rlinf-rlt/ugrow-rlt-g5-2000-1006-v1`。各自namespace为目录名。候补旧RLT仍沿原lease：4卡v2、5卡v3，原完整CP100保留。

BC每个完整CP实测20,342,681,138字节，约18.95GiB；全留40个会超盘。保留器只在原/接续这两个BC路径内操作：以同步保存后写出的轮次指标确认完成；保留最近两个完整CP、CP100接续锚点及CP400；检查pin、UID、符号链接、打开文件引用后，先写精确文件清单再删除旧CP。未完成CP和所有日志/成功数据不动。当前已启用但尚无待删旧CP，后续看prune-step-N.json。原训练盘低于40GiB保护继续有效。

6项新服务器CPU检查通过：累计预算/恢复路径、成功与失败终态、lease排他、最近2个与未完成保护、接续锚点/日志保护、pin依赖拒删。配置实际diff仅预算、resume、输出路径；BC额外success_data路径是输出项，首次准备白名单漏列时已安全停止并修正，未改变训练方法。尚未到原终点，未来接续不是已完成实跑的结论。

耗时估计：BC总约38.3小时（35–42），15:05起余约36.2小时（33–40）；RLT总约57.6小时（52–68），余约56.8小时（51–67）。BC依据本次普通轮274秒+每5轮350秒评估，以及历史94轮326.95秒/轮；RLT依据历史同π0.5 N4成功配对127轮预采集、18轮初始化、655轮online，将本次采集99.91秒/历史82.26秒的1.215倍用于采集/评估成本，再计原训练/保存。非置信区间；本次RLT尚未开始更新，在线耗时仍是历史迁移估计。推算中心完成时间：BC10月8日03:20，RLT10月8日23:55。

发布仍为`codex/ugrow-g45-evidence-20261006`，本次独立ops在`tools/ugrow_budget_400_2000/`；预算、armed与ETA轻量证据在`docs/ugrow_execution_20261006/budget-*.json`。训练源码BC1c4b3810、RLTbfbc9c88及原pins不动。前序全文在本地`archive/CONTEXT_20261006_1424.md`，发布副本`docs/ugrow_execution_20261006/CONTEXT_1424.md`。
