# RLinf 当前交接入口

## 最新调查与现场（09-06 10:18现场／10:22种子源码，只读）

本轮只授权看图、调查种子/耗时、讨论上下文/存储/Git；不沿用旧授权去改种子、清理、push或重启。
GPU6 BC完整45/100、GPU7 DVAC43/100、GPU4/5 GRPO156/200；fixed最新45=20/32、40=19/32、155=21/32，ckpt40/40/150必要文件在（未恢复测试）；worker/wrapper均在、所查fatal/OOM等0。
新增结论：BC/DVAC环境seed0、表与逻辑offset相同，训练按992种子/32一批顺序轮转，不是GPU编号导致seed不同；rollout动作默认噪声未见显式seed控制，actor1234不能覆盖独立rollout进程。未证明噪声是差距唯一原因，不在当前run中途改。采集约5.3—5.4min、监督更新7.5min，普通轮约13min；补了采集＋fixed图，不虚构Step0评估。
风险仍有环境RAM增长：当前available669.67GiB、PSI/换页短采样0；/data余630.06GiB、/home1246.15GiB。本人/data1695.39GiB，首选讨论清两个新smoke Step1四大文件36.10GiB，未删。RLinf22分支与远端一致，但RLT诊断2dirty、RoboTwin试验1未跟踪、本地docs未跟踪；RoboTwin远端核查失败，不声称已实时验证。
此次主简报／图／种子源码证据：`docs/rlinf-robotwin-pi0-online-bc/evidence/PI05_BC_SEEDS_TIMING_CONTEXT_STORAGE_DISCUSSION_20260906.md`；存储/Git详情唯一在`docs/server-admin/SZ_STORAGE_GIT_REVIEW_20260906.md`。上下文强制阅读约92KiB、HANDOFF约48KiB，建议无损归档入口旧状态，尚未执行裁剪。
BC迁移SSOT仍`02_PI05_ONLINE_BC_PLAN.md`；DVAC方法仍`01_DVAC_DESIGN.md`；GRPO既有heartbeat由原窗口管理。本轮未干预任务/shared Ray/他人，未push或建监控。

最后整理：2026-09-05 21:22 CST heartbeat现场（Sidney完整121/200、122轮采样3/4，原wrapper存活、无所查错误；fixed120=19/32，checkpoint目录最新120，21:12独立窗口已核对双rank/full；/data余913.21GiB。其他窗口21:12记录的授权清理101大文件570.08GiB及独立复查、π0.5 BC仅规划状态保留，heartbeat继续原范围只读）。

本文件只保留当前路由、活动实验和下一步。旧累计时间线已无损归档到
`docs/project-history/HANDOFF_SNAPSHOT_20260903_PRE_WINDOW_HANDOFF.md`。

## 新任务最短阅读顺序

1. 完整读根目录 `AGENTS.md` 与 `PROJECT_CONTEXT.md`。
2. 完整读本文件。
3. 读本轮近期专题交接：
   `docs/window-handoffs/20260903_RLINF_SHENZHEN_WINDOW_HANDOFF.md`。
4. 只按当前问题继续读该交接中指向的一个专题 SSOT；不要默认加载全部历史专题。

动态实验状态必须以服务器只读刷新为准，下面仅是最后记录快照。

## 最新π0.5执行授权：100自然结束后原地续到200并交付轻量ZIP（09-05）

用户本窗口明确要求“盯着pi0.5的实验，结束了，就改为200步，其他不动，原地放下去”，随后追加100步日志/指标/可视化轻量ZIP；已授权等待、完整step100恢复、执行101—200与跟进，无须再问同一批准。旧只读状态不否定该新授权。
唯一执行账本：`docs/rlinf-shenzhen-multitask-pi05/evidence/RESUME100_TO200_EXECUTION_LEDGER_20260905.md`，含完整resolved/精确命令与预算。原100于13:00:18正常exit0，train165/256、MA10=66.91%、fixed19/32；三份checkpoint文件完整。接续13:02:39启动，wrapper602620/observer602621，新runtime为`runtime-resume100-to200`；仅max_steps/resume_dir两叶变化。13:07:43确认从global_step100实际恢复后进入101轮rollout0/4，所查错误0；尚未完整101。勿重复launch，原runtime结束标记不代表新运行结束。
原runtime与checkpoint保留；旧video/eval已归档video/eval_steps001_100，TB旧config也已备份。heartbeat `0-5-100-200`已降每小时；正常安静，完成/失败/需要处理时通知，200完成后暂停。
100步轻量包 `docs/rlinf-shenzhen-multitask-pi05/evidence/formal100-summary-20260905.zip` 已交付，441001 bytes，含指标/配置/关键日志/资源/3PNG/离线HTML，QA通过；冻结100步JSON不覆盖，不混入续训。不接管Fast或BC独立窗口，不动其他用户/shared Ray。

21:22:26小时检查：完整121/200，122轮rollout3/4；train121=161/256=62.89%，MA10=67.73%；fixed120=19/32=59.38%，下一次125。checkpoint目录最新120（本轮未逐文件复核或恢复测试），原wrapper602620存活，无结束标记，所查错误0。/data已回升至913.21GiB，高于本run后续130—200八代预计214.79GiB保存预算；交接记载其他窗口已完成授权清理，先前容量压力已缓解。本heartbeat未参与清理、改参或停止，继续每小时只读并保持安静。证据：专题`evidence/pi05_resume200_heartbeat_20260905_2121.json`；清理依据见本页BC段落及server-admin执行账本。

## 最新π0.5刷新（09-05 12:37 CST）

证据：`docs/rlinf-shenzhen-multitask-pi05/evidence/PI05_STATUS_REFRESH_20260905_1237.md`。
Sidney完整99/100，step100 rollout1/4；训练166/256=64.84%，MA10=67.50%，fixed仍95的22/32，历史最佳70的26/32。原wrapper存活，无结束标记；所查driver fatal/OOM/RuntimeError/Traceback为0；step100 checkpoint尚无文件。
最近10步24.51分钟，条件ETA约13:00（12:55—13:15，含最终评估保存的不确定性）。GPU4/5约54.96/55.56GiB，/data余662.94GiB。仅只读；继续等自然完成与保存，本轮未停止、改参或接续200。

## 最新讨论：π0.5接续与Fast故障/学习审计（09-05 11:34 CST）

本轮唯一主简报：`docs/fastwam-robotwin-rlinf-grpo/evidence/TRAINING_DISCUSSION_PI05_FASTWAM_20260905.md`，链接两项源码专题审阅、恢复语义、现场JSON及参数读数。
Sidney完整96/100，167/256、MA10=66.56%，fixed95=22/32，原driver/wrapper继续，所查fatal/OOM=0；最近10步25.318分钟，条件ETA09-05约13:05（12:50—13:30）。建议自然完成100并保存后再接200，不主动中断回90；constant LR5e-6与模型/Adam/scheduler/actor RNG恢复语义已按部署源核对，未做真实恢复。
Fast仍17步退出。完整栈证实reset已有global_lock；本次是CPython线程局部状态丢失，非“忘释放GIL”或已证实多scene并发析构。实际shim只改render、无Python/TLS导入；此前未处理原生生命周期隐患优先，唯一native caller未捕获。固定SubEnv owner线程是候选，未实施。
Fast学习主链无明显漏update/旧新logprob/H32-C24错接。新确认action1.021B参数与Adam moments全BF16；Step10/40更新后6张量659万元素抽查改变仅0.75%—28.99%、moment全非零，支持实际更新精度为首位嫌疑，未证明唯一根因。MB2 total_loss/ratio的日志口径已区分，不把日志小当梯度少16倍。详细排序与FP32训练主参数候选只放专题。
本窗口仍仅审计/讨论与服务器CPU文件只读；无模型forward/optimizer/新评估、无修复或续训。本轮未接管BC独立实施窗口，后续动态状态再刷新。

## 历史刷新：审计窗口三项实验与整机（09-05 11:05 CST）

唯一简报与原始JSON入口：`docs/server-admin/SZ_BRIEF_AUDIT_20260905.md`。
普通账号chenyiteng已认证；本窗口仅只读审计、讨论与本地证据/交接更新，没有实施修复、smoke/训练、清理或进程干预。
Sidney完整Step95，188/256=73.44%，MA10=67.42%；fixed Step95=22/32，Step90双rank/full仍在，driver继续采样，所查fatal/OOM等0。
Fast仍完整Step17、04:47:11退出255，PyGILState_Release fatal；Step10双distcp/metadata在，未重启。
BC v2已于11:01:04退出255，OpenPI SFT训练图像预处理grid_sample报BFloat16/Float类型不匹配；未完成训练轮，无所查checkpoint，未检出OOM，不能称smoke通过。下方“v2启动”是此前状态。
GPU0其他用户约9.6GiB、4/5 Sidney67.08/66.23GiB；1/2/3/6/7无compute，GPU7预留保持。RAMavailable1104.77GiB、CPUidle94—95%、memory PSI=0、无即时swap进出；/data余731.04GiB（79%已用）。shared Ray原进程继续；本轮未复核Git/SMART/管理员内核日志，不沿用旧快照作当前结论。
后续按用户审计/讨论方向继续；Sidney完成情况需现场再查，BC/Fast故障记录保留，不自动套用其他窗口的修复/重启流程。

## 历史刷新：两项实验与整机（09-05 10:00 CST）

证据与成功率/优化/资源图：`docs/fastwam-robotwin-rlinf-grpo/evidence/CURRENT_TRAINING_HEALTH_20260905_1000.md`。
Fast完整Step17，65/256；fixed Step5/10/15为11/11/13 /32，尚不足以称稳定提升。Step18并发reset期间出现PyGILState_Release线程状态fatal，04:47:11退出255；原driver已不在，GPU6/7空闲。未检出本run OIDN/pthread_key_create/OOM；不能直接认作旧挂帧同因。最新Step10双distcp/metadata在，未恢复测试。旧Fast预计结束时间失效。
Sidney完整Step92，167/256，MA10=63.24%；fixed最新Step90=21/32、最好Step70=26/32，近5次26/21/22/24/21均超过早期19/32；有后期改善但仍波动。Step90双rank/full在，driver继续，所查fatal/OOM等0，未恢复测试。
GPU0其他用户，4/5 Sidney约67GiB/卡；1/2/3/6/7无compute。RAMavailable1103GiB、memory/io PSI=0、无即时swap-in/out；CPU约95%idle；/data余767GiB。三树HEAD/clean核对未变，shared Ray原PID继续。仅只读，不重启Fast、不触碰Sidney/他人任务；BC没有GPU测试。

## 历史刷新：两项实验与整机（09-04 23:48 CST）

证据及三类PNG/交互图：`docs/fastwam-robotwin-rlinf-grpo/evidence/CURRENT_TRAINING_HEALTH_20260904_2348.md`。
Fast完整Step10、46/256；Step5/10 fixed均11/32，暂未见提升；首次双distcp及metadata已在。Sidney完整Step69、178/256、MA10=64.34%；Step60 fixed24/32创新高，Step65回落19/32，Step60双rank/full在。均未恢复测试。
两项原driver及shared Ray继续，所查driver fatal/OOM等均0；GPU1/2/3空闲，4/5 Sidney、6/7 Fast；RAMavailable761.7GiB，较19:11下降约131GiB，memory PSI=0、无即时swap-in/out；/data余1.07TiB。
三树HEAD/clean与Env-local补丁隔离现场核对未变。仅只读，无测试/训练启动、进程干预或新自动化；BC短测仍未获本轮批准。下一次关注Sidney Step70评估/保存、Fast后续fixed及RAM。下方19:19 ETA未重新计算，不作为最新估时。

## 当前讨论：RoboTwin π0 干净在线 BC → 成功＋DVAC

09-06 00:18收尾：π0.5 DVAC源码626825e8及启动证据776ebc98均已push，方法分支clean；训练source-head仍626825e8。只补齐本run小证据，未改正在运行的代码/模型/训练预算。用户no-smoke已遵守；后续状态从00:14启动核验向前刷新，不默认重跑或持续盯守。

00:14:39启动确认补充：新π0.5 DVAC真实start=09-06 00:09:52，driver2223401，Env2224188/Rollout2224184/Actor2224182都绑定GPU7/独立repo/run/原模型。已进入首轮采集、无所查fatal/OOM；基线GPU6 HEAD6a93605d/clean及wrapper2143105、shared Ray321933/322685、Sidney602620均在。源码626825e8已push；未做GPU smoke/未完成本组合首轮更新，不夸大验收。启动确认后结束主动检查。唯一证据`PI05_BC_DVAC_BINDING_20260905.json`与下方专题账本。

π0.5 BC＋DVAC最新（09-06 00:10）：用户授权GPU7同参数正式、权重[0.5,1.5]，并明确不smoke。独立`codex/sz-pi05-online-bc-dvac@626825e8`已push/clean，28项CPU测试与正式resolved对照通过；五基线文件与旧DVAC父版本逐字一致，复用旧736b1416六个方法文件，新增alpha0.125薄配置/模型测试。GPU7已单次launch，wrapper2223376/observer2223377，run `online-bc/pi05-pillbottle-bc-dvac32x1-b1024-u10-m10-w05to15-eval8x4-gpu7-formal100-20260905-v1`，不要重复启动。32×1/micro32/global1024/U10/M10、eval8×4每5/save10/fresh原Sidney SFT＋空池，除DVAC/GPU/路径名字外与GPU6一致；首轮等权、末L3/past5、入池固定w、均值1。唯一方法`docs/rlinf-robotwin-pi0-online-bc/01_DVAC_DESIGN.md`§12，唯一执行账本`evidence/PI05_BC_DVAC_IMPLEMENTATION_LEDGER_20260905.md`。未做GPU smoke、不称长程验证；GPU6基线/4/5 Sidney/shared Ray/其他用户不动，不删产物/升级依赖/加heartbeat。

23:46发布收尾：上述新π0.5 BC formal启动证据`6a93605d`已push、clean（训练source-head912bc690不变）；23:44真实driver2143110命令与三worker GPU6/模型/run路径核对通过，shared Ray原PID及Sidney wrapper均在。启动确认后结束主动轮询，无新heartbeat。唯一证据索引沿下方正式账本。

π0.5 BC正式最新（09-05 23:41:58）：用户明确授权GPU6正式100轮、原eval5/save10、启动后回报不长期盯守；23:38:34从原Sidney SFT/空成功池单次启动。run `online-bc/pi05-pillbottle-bc32x1-b1024-u10-m10-eval8x4-gpu6-formal100-20260905-v1`，wrapper2143105、observer2143106，源912bc690（生产653fe0fb）。23:41三worker在、首轮采集中、无所查fatal/OOM、尚无完整Step1。32×1/micro32/global1024/U10、M10/expert-only、eval8×4；与已过π0.5smoke只4个调度/总量＋7个命名路径叶差异。唯一参数/来源为BC专题`02_PI05_ONLINE_BC_PLAN.md`§10，正式合同/账本`evidence/GPU6_PI05_BC_FORMAL_CONTRACT_20260905.md`/`GPU6_PI05_BC_FORMAL_LAUNCH_LEDGER_20260905.md`，actual状态`PI05_BC_FORMAL_STARTUP_20260905.json`。不要重复launch、不续smoke，不改Sidney/GPU4/5/shared Ray，不清理/升级/加自动轮换或新监控。23:41 /data余875GiB；后续容量不能用smoke保证，下方“formal未授权”均已被本次明确授权覆盖。

π0.5 BC最新验收（09-05 22:42；22:45已push）：独立`codex/sz-pi05-online-bc`源码653fe0fb／证据912bc690已push、clean；13tests＋真实数据/配置检查通过。GPU6两轮smoke于22:41:20 exit0，20Adam、两次fixed32、两代native/full/replay/learner及CPU读回验收通过；train8/18、fixed11/18 /32，峰73.74GiB、FD882，GPU6已释放。原Sidney SFT/pillbottle/M10，继承BC32×1/micro32/global1024/U10/eval8×4/expert-only；两轮不证明稳定提升／100轮容量或整任务resume。目标`online-bc/pi05-pillbottle-smoke32x1-b1024-u10-m10-eval8x4-gpu6-20260905-v1`已结束，不重复launch。唯一计划`docs/rlinf-robotwin-pi0-online-bc/02_PI05_ONLINE_BC_PLAN.md`§9，逐操作`evidence/PI05_BC_IMPLEMENTATION_LEDGER_20260905.md`、完整合同/机器验收均在该专题。**未启动formal；下一步等用户确定正式预算与合同。** 本轮不扩大到Sidney/shared Ray／其他实验、依赖升级或继续清理；下方只研究状态为历史。

最新授权／完成（09-05 21:12）：用户已明确批准上一轮列出的checkpoint大文件清理，覆盖下方旧“禁止清理”；20:49精确删除101文件570.08GiB，21:12独立验证101目标不存在、95保留文件和121小文件不变。6文件14.53GiB因保留点DCP不完整暂缓，未删目录／当前Sidney／DSRL/RLT／新v8／他人数据。唯一执行账本`docs/server-admin/CHENYITENG_CHECKPOINT_PRUNE_LEDGER_20260905.md`；旧清理讨论只作删除前候选快照。BC v7与DVAC smoke第1代大权重已清，Step2保留；v8两代未动。没有新正式训练授权，禁止自动formal或迁移存储。

π0.5在线BC单独上下文：`docs/rlinf-robotwin-pi0-online-bc/02_PI05_ONLINE_BC_PLAN.md`。本轮只研究：基于已跑通BC，借Sidney原SFT／pillbottle实际协议，迁入两处小数据适配与薄配置；不合并GRPO整分支或目标。未建立新训练分支／修改生产代码／测试／launch；具体模型差异、U10讨论、BC实测耗时和checkpoint体积只维护在该专题。21:12Sidney121/200、fixed120=19/32，checkpoint120双rank/full在、无所查错误；GPU6/7无compute，RAM available1342.4GiB，/data余913.22GiB、/home余1246.15GiB。原始`BC_DVAC_SERVER_REFRESH_20260905_LATEST.json`；不接管Sidney原窗口heartbeat。

最新验收（09-05 20:14—20:15）：两项smoke均完成，GPU6/7释放。GPU6原BC eval8×4于20:05:53 exit0，2轮/20Adam/64评估/两代native＋full＋replay＋learner验收通过；峰69.52GiB、FD882，源a8764944/证据2467d997已push、clean；train26/23、fixed24/25 /32。GPU7 DVAC已于19:31:36 exit0，2轮及CPU逐记录权重重建/旧池不变/RNG读回通过，峰77.71GiB、FD1003；源736b1416/证据912808c7已push。见BC专题`evidence/BC_DVAC_IMPLEMENTATION_AND_SMOKE_20260905.md`§3—4和`BC_EVAL8_RESTART_LEDGER_20260905.md`。均不含全worker恢复或长程容量验证；两项评估并发不同，不作严格方法收益比较。用户明确只smoke、暂不formal，禁止自动正式训练、切/home或清理。

BC窗口最终全机只读刷新（09-05 20:17）：Sidney完整118/200、第119轮更新，train169/256、MA10=66.09%，fixed115=21/32、checkpoint110双rank/full在，无所查错误；GPU4/5约74.5/74.9GiB，GPU1/2/3/6/7无compute，RAM available1350.7GiB、CPU97%idle/PSI0。/data余370.04GiB、89%，/home余1246.15GiB。唯一证据与图表见BC专题`evidence/BC_DVAC_IMPLEMENTATION_AND_SMOKE_20260905.md`§5。本窗口不接管另一窗口Sidney进度/heartbeat，不把该快照作为下一轮现场。

最新存储讨论（09-05 19:36目录快照、20:17追问整理）：用户要求解释以前已删与当前残留，**尚未授权删除**。唯一讨论稿`docs/server-admin/SZ_CHEN_STORAGE_CLEANUP_DISCUSSION_20260905.md`§5；9月1日已删146文件1051.64GiB（1.129TB逻辑字节）；当前旧GRPO/PPO残留65DCP＋16full共81文件347.73GiB，不重复。另两个BC组件探针33.40GiB、v7/DVAC各第1代34.38GiB，可讨论合计415.51GiB；不含新v8和Sidney/DSRL/RLT。19:36本人/data1920.8GiB为v8保存前快照，不冒充当前盘面。广义139文件856.35GiB只为初筛，未逐条验证保留点/最佳点/跨run依赖/硬链接/打开状态，不是删除allowlist。当前Sidney、DSRL/RLT、模型/Ray/其他用户不动；未清理/搬家/改磁盘保留比例。全用户分盘表同目录`SZ_STORAGE_BY_USER_20260905.md`。

最新新增授权/并行执行（09-05 19:27）：原BC已OOM后，用户指定评估8×4并要求GPU6与GPU7并行；随后明确“先完成smoke，暂不放正式”，覆盖之前重放formal的计划。GPU6配置仅三个eval叶值及测试改动，a8764944已push、11tests/同32固定种子/真实resolved通过；19:27:23启动两轮smoke，wrapper1597471/observer1597472，唯一接续`evidence/BC_EVAL8_RESTART_LEDGER_20260905.md`。GPU7既有DVAC736b1416继续，第二轮已产生w范围0.602—1.780/均值1，仍待完整验收，未改其16×2配置。禁止自动放formal、切/home、删除或移动旧产物。用户另授权按用户目录只读审计/home与/data；19:24分布见`docs/server-admin/SZ_STORAGE_BY_USER_20260905.md`，/data约422GiB可用、本人目录1903GiB，/home约1246GiB可用。无他人文件内容读取、进程干预或共享服务改动。

最新独立实施（09-05 19:01现场）：用户新增授权BC＋DVAC实现、简测、单卡smoke，未授权新DVAC正式长训。原BC `385d4e75`独立分支`codex/sz-pi0-online-bc-dvac`，实现`736b1416`已push，22tests及真实配置/预算对照通过。GPU7于18:53:14启动2轮smoke，wrapper1512156/observer1512157；19:01首轮采集完成，正在U10更新，未完整验收，不要重复launch。唯一执行入口BC专题`evidence/DVAC_IMPLEMENTATION_LEDGER_20260905.md`，完整合同为`GPU7_DVAC_SMOKE_CONTRACT_20260905.md`，方法见`01_DVAC_DESIGN.md`§11。GPU6原BC已于17:44:40在第6轮rollout推理CUDA OOM退出，完整5轮、fixed5=27/32、无正式checkpoint；未重启/改源。GPU7当前仍按原smoke train32×1/micro32/global1024/U10/eval16×2/save1，不把2轮短测当原第6轮OOM已解决。Sidney/shared Ray/其他用户不动；下方“DVAC仅讨论/BC仍运行”均为历史快照。

最新切入点研究（09-05，Fast/源码只读17:21）：用户澄清先比较相关工作的BC pipeline介入环节，不是质疑旧DVAC信号转换。唯一设计续接为BC专题`01_DVAC_DESIGN.md`§10，作者代码及其他入口比较见`evidence/BC_DVAC_PIPELINE_ENTRY_RELATED_WORK_20260905.md`；倾向逐动作FM监督项加权，仍仅讨论。Fast未发现活动计算进程残留，04:47旧日志未继续增长，不是新报错；后续概览不重复未变化的已结束Fast。未改生产、训练、进程/shared Ray，也未新建DVAC分支或push。下方各实验进度保留原时间，不能当本轮刷新。

BC＋DVAC讨论/全机刷新（09-05 17:01）：BC完整2/100（train25/32→24/32、累计49成功episode），U10更新/同步继续，尚无正式fixed/ckpt；Sidney完整110/200，fixed110=17/32、105=24/32、ckpt110双rank/full在，详见BC专题`evidence/BC_DVAC_REVIEW_20260905.md`与π0.5专题`evidence/pi05-bc-review-20260905/`分栏图。Fast保持17后结束。本轮无生产源码/config/依赖/进程变更，shared Ray/其他用户/GPU7未动。DVAC仅深化`01_DVAC_DESIGN.md`§6—9：建议高V/α0.25/过去5轮/首轮等权/入池固定w，未确认实现；下一步用户确认后再从已测BC建独立worktree。/data余460.4GiB；现两run后续ckpt估计约414GiB，GPU7新正式前需明确空间策略，不能擅自删产物。Git补推PPO/Fast/RLT历史轻量诊断92文件约526KiB，三份evidence-only提交c41b7ff0/0d5daf6f/d3acd650；原RLT dirty及OIDN脚本未改，快照已备份。后续发布结果见`docs/server-admin/SHENZHEN_BC_DVAC_REVIEW_PUBLISH_20260905.txt`与BC讨论账本；Git覆盖不等同所有历史产物完整备份。

BC本轮执行（09-05 16:34）：GPU6正式100轮已于16:32:12启动，wrapper1151769/observer1151770，原SFT/空成功池，16:34已进入首轮采集、尚未完整Step1，无所查错误；不要重复launch。v7完整两轮smoke于16:29:03 exit0，20次Adam/两次fixed32/两代native shard＋full＋replay均验收通过，采样显存峰77.46GiB、FD1003。源码`cb01451f`、轻量配置/证据提交`1d453fcb`均已push；11tests通过。唯一接续账本为BC专题`evidence/GPU6_U10_LAUNCH_LEDGER_20260905.md`，完整合同/输出/固定参数见`GPU6_U10_RUN_CONTRACT_20260905.md`，原始验收及启动JSON分别为`U10_SMOKE_VERIFICATION_20260905.json`/`U10_FORMAL_STARTUP_20260905.json`。GPU7/Sidney/shared Ray未动；不清理旧产物，/data16:34余496GiB、共享后续保存容量需留意。启动确认后即回报，短watch已结束，不新建长期监控；短测不保证100轮原生稳定性或完整worker恢复。下方旧快照保留时间。

BC追问窗口最新（09-05 14:55只读）：用户明确优先降eval并发、必要时可取消中途评估；学习B/U尚待解释确认，未实施新参数/训练。答疑统一在专题`evidence/BC_PARAMETER_QA_AND_UPDATE_BUDGET_20260905.md`§10—12。旧Control现场证实完整96、fixed95=31/32、ckpt90双DCP/full在，32train＋16eval/卡长程通过，但未证明该run正常完成100。BC真实更新/同步已跨过FSDP首错，完整smoke仍阻于评估资源；HEAD700b6846 clean，无活跃BC/正式/DVAC。Sidney完整104/200、105评估中，无所查错误，ckpt100在；GPU6/7空闲，RAMavailable1.59TiB、/data余567.13GiB。继续按单卡eval16×2原32种子优先，非默认开offload；U/global确认后再更新合同/连贯实现与完整smoke，不重做旧组件probe、不动shared Ray/他人/GPU7。

BC讨论窗口最新（09-05 14:14—14:17只读）：逐项问答/一手预算核验见专题`evidence/BC_PARAMETER_QA_AND_UPDATE_BUDGET_20260905.md`。用户新提议训练32不变、评估16×2；当前优先明确分两批覆盖原32固定ID，旧环境offload仅备选，未改生产。U2依据不足重新讨论；候选micro/global32、U100＝3200样本呈现/轮，尚未批准/实施，部署仍GB1024/U2。FM噪声/t必须保留为原生SFT；logprob/chain辅助存储无用、尚未删。新发现HABC虽6k更新但附录声明action-expert不训练，Hi-ORS纯online路由实际batch为配置半批，不能盲搬。服务器HEAD700b6846/clean、五个本地源/config hash一致；BC无活跃run/正式/DVAC，Sidney完整103/200继续，仅只读。下一步先收敛参数再新合同/连贯修改/完整smoke，不直接执行旧wrapper、不重跑旧隔离探针、不动他人/shared Ray/GPU7。

本轮权威接续：专题`evidence/GPU6_SMOKE_LEDGER_20260905.md`与`GPU6_SMOKE_CONTRACT_20260905.md`，完整resolved已保存。用户授权GPU6同并发smoke、本专题Git，随后明确授权smoke通过后的正式100轮；GPU7留后续变体，正式采样32×1＝每轮32条，M4继承官方π0/DAgger配置。以下旧“未运行/未推/正式未授权”状态不覆盖本条。
BC实施窗口13:39补充（与上方其他窗口记录分开）：v6在13:20:25 exit255；29/32成功、87query、64微批/2次更新后，首次评估相机cannot create buffer，完整评估/ckpt均0。显存峰值79.17/79.65GiB，RAMavailable最低1.70TiB、PSI0；当前是32train＋32eval共驻留容量/分配边界，不报“无OOM所以健康”。此前SFT实际调用包裹与EnvWorker FD4096窄修已推72a92604，10项测试通过，v6无重复FSDP断言/EMFILE。六次结束smoke轻量证据已push700b6846，13:39树clean；无活跃BC、无正式/DVAC。下一步推荐既有train/eval交替offload，保持各32并行及全部学习参数，未打开，待用户确认；不再从头跑精度/FSDP/FD探针，不重复旧launch。通过新资源配置完整smoke后才按已有授权从原SFT/空池正式100轮。唯一结果/待选项入口为专题`evidence/BC_RESULT_AND_NEXT_RESOURCE_CHOICE_20260905.md`，流程/预算见`BC_FLOW_COMPONENT_AND_UPDATE_AUDIT_20260905.md`，DVAC仅`01_DVAC_DESIGN.md`规划。现场GPU6/7各11/4MiB，Sidney完整101/200继续、shared Ray原PID在；本窗口仅只读它们，未动其他用户，GPU7仍预留。
用户额外授权只读查看他人评估：证据见`docs/server-admin/LIWENBO_GPU_ROUTING_READONLY_20260905.md`；未停止/迁移/修改对方任务。用户随后要求不再管他们，停止进一步定向检查。共用GPU6已明确获批，之后他人评估自行结束。Fast不重启，Sidney/shared Ray不动。
Git审计见`docs/server-admin/SHENZHEN_GIT_COVERAGE_20260905.md`：23工作树、22codex分支中21远端一致；RLT诊断dirty及OIDN未跟踪脚本不混入算法提交。四个Fast结束run轻量evidence已补推`f3a1689e`（109文件，源产物1.14MiB）。下一轮不能把Git已推等同GPU smoke已通过。

唯一专题 SSOT：`docs/rlinf-robotwin-pi0-online-bc/00_RESEARCH_AND_PLAN.md`。
固定源码、调用链及边界：同目录 `evidence/ONLINE_BC_SOURCE_AUDIT_20260904.md`。
用户确认adjust_bottle π0 SFT、D0混合参数化且默认关闭；GPU6 smoke及通过后的正式100轮已授权，GPU7预留。不得跳过完整smoke验收或复用smoke权重；现役Sidney/shared Ray不动。
本轮已阅读用户附件和seek对应整理，将原广搜归档到专题history；当前SSOT只聚焦干净首版及必要接点。
参考代码优先级：现有RLinf主干 → Hi-ORS成功池/FM → SIME分轮编排 → online LeRobot局部存储；不整套搬SIME探索、筛选和训练重置。
关键发现仍成立：RoboTwin块后obs与逐动作builder不匹配；普通DAgger replay只收intervene片段，不能只删expert/开only_success。
上游dc9b87c至a3795c已独立核对，仅中英文README变化；这些公开源码锁不代表深圳部署版本。
已沿query/chunk记录做累计成功池，复用π0 FM/FSDP；N=32×1、micro32/global1024/U2、M4、D0权重0已写本轮合同，旧256轨迹配置不继承。后续DVAC只在同一监督链上讨论权重，不引入Q/V或旧RLT目标。
旧π0 GRPO为32环境/卡、2卡×32×4＝256条/轮，micro32/global1024/U2，历史峰值约76.64GiB/卡；只是旧容量实测，不能保证BC单卡。09-05早间预算讨论已由本轮GPU6合同覆盖；旧2环境/GPU3/M10/正式32×8建议不执行。

## 最新估时：formal100（19:19 CST）

`docs/fastwam-robotwin-rlinf-grpo/evidence/TWO_RUNS_ETA_20260904_1919.md`。
Sidney最近10步均24.49分钟，预计09-05 12:15左右（11—14时）；Fast前3步均40.68分钟，加尚未实测的评估/保存预留，预计09-07 15:00左右（12—20时）。
两项均条件于持续正常运行；Fast仅3步且未覆盖首次eval/save，误差大，不是长程稳定性保证或完成承诺。只读，无新监控自动化。

## 最新刷新：两项实验及整机（19:11 CST）

证据、成功率/优化/资源PNG和交互图：
`docs/fastwam-robotwin-rlinf-grpo/evidence/CURRENT_TRAINING_HEALTH_20260904_1911.md`。
Fast scene-fence-v3完整Step3、训练88/256，Step4 rollout6/8；Step1—3为65/41/88 /256，尚无fixed或checkpoint。
已跨过旧Step2卡住边界，不代表长程根治；下一次关注Step5 fixed与Step10 DCP。预算32×8不变。
Sidney完整Step58、训练158/256、MA10=60.39%；Step55 fixed19/32追平历史最好，未持续超过；Step50双rank与full在，未恢复测试。
两run所查fatal/OOM/OIDN/Traceback/RuntimeError均0；原driver/shared Ray继续，三树HEAD/clean及补丁隔离核验不变。
GPU0其他用户占用，1/2/3采样时空闲；4/5 Sidney约58GiB/卡，6/7 Fast约56GiB/卡，Fast本run采样峰值63.18GiB。
RAMavailable0.872TiB仍缓降；CPU空闲93—94%，load9.26/128，memory/io PSI=0；/data余1.20TiB。
GPU1可纠正SRAM计数2与17:03相同，不可纠正/row-remap failure为0；本轮未作管理员内核/SMART复核。
本轮只读服务器，本地更新图与证据；无训练改动、测试、进程干预或新自动化。

## 前轮调查：Sidney π0.5 与旧 π0 GRPO 学习速度（17:10）

当前问题只读，专题SSOT为`docs/rlinf-shenzhen-multitask-pi05/00_INDEX_AND_EXECUTION.md`。
完整对比、配置/源码证据与交互/PNG图：
`docs/rlinf-shenzhen-multitask-pi05/evidence/PI05_VS_PI0_LEARNING_REVIEW_20260904.md`。
17:10:52：Sidney完整Step53，训练145/256=56.64%，MA10=58.48%；最新fixed Step50=16/32，
Step10/35最好19/32，未持续改善。所查fatal/OOM/OIDN/Traceback/RuntimeError未检出；Step50双rank与full_weights在，未恢复测试。
比较对象是深圳旧两卡π0 adjust_bottle纯GRPO Control，不是四卡/DVAC或AutoDL；前50步预算均12,800轨迹/100optimizer calls。
训练首10→末10步均值Sidney41.68→57.23%、旧π0 78.36→89.61%；训练百分点增幅和墙钟并不更慢（Step50耗时19.52 vs22.37h），但fixed改善明显不稳。
主要已知差异为任务/SFT、200-action对放置任务的约束、内部delta/absolute与norm、M4/10、state路径；不是已经证实单一根因。
实际embodied actor与更新循环已核对，未发现明显漏更新；没有更改训练/模型/依赖、运行评估或停止任务。
下一步优先同协议SFT/current和失败阶段检查，特别是200-action内放置/松爪是否来得及；需新授权，不直接加LR/noise或改400。

## 前轮执行：Fast scene-fence窄修、256轨迹重启（16:56快照）

该Fast专题SSOT为`docs/fastwam-robotwin-rlinf-grpo/12_SHENZHEN_CURRENT_RLINF_GRPO_PORT_PLAN_20260831.md`。
该轮执行账本/完整配置/精确命令/现场与图入口：
`docs/fastwam-robotwin-rlinf-grpo/evidence/SCENE_FENCE_FIX_RESTART_LEDGER_20260904.md`。
Fast分支`codex/sz-fastwam-current-rlinf-grpo@62526cc95047c8a4a6e948a76be8eeec8a3926de`已推personal；
只补timeline render现有scene-access fence协议，Env-local原生shim，不升级/覆盖shared wheel，旧Python补丁不套回。
真实2场景/128帧/384相机图短测exit0，实际符号binding已核实；不等于本次旧挂帧唯一因果或长程稳定性证明。
16:07仅旧Fast run/namespace清退，旧目录保留、无checkpoint；v2在16:21因全局预加载干扰权重读取退出，0step。
已用plain/preload/late-load对照定位并删除global preload；局部加载的384图+渲染前后权重读取复测通过。
16:38 v3从原SFT启动：
`/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-scene-fence-v3`。
wrapper/PGID1568962，observer1568964，driver1568973，namespace RLinf_1，GPU6/7。
完整resolved仅6个路径/名称叶子变化，32env×8、所有训练/评估参数与旧256/noOIDN合同相同。
生产库`/home/chenyiteng/builds/fastwam-scene-fence-20260904/release-final/librlinf_scene_fence.so`；
新run environment.sh只设置RLINF_SCENE_FENCE_LIBRARY；仅RoboTwin初始化前RTLD_LOCAL加载，不用LD_PRELOAD/LD_LIBRARY_PATH。
16:43核验只有双EnvWorker映射该库，Actor/Rollout/driver/Sidney均未加载，均无LD_PRELOAD；不要使用b60144fd的全局预加载方案。
16:56只读刷新：v3首步采样从1/8推进至2/8，完整0步，无所查异常；仅双Env映射补丁，尚未验证完整更新和旧Step2边界。
Sidney完整Step52（153/256），下一步采样2/4；Step50 fixed16/32，双rank checkpoint与full_weights在；shared Ray/Sidney原进程不动。
RAMavailable约0.98TiB、load1=11.75/128CPU；GPU0/1/2其他用户使用、3空闲，4/5 Sidney、6/7 Fast；本轮无服务器修改。
最新回顾/现场/交互及PNG图：`docs/fastwam-robotwin-rlinf-grpo/evidence/SCENE_FENCE_FOLLOWUP_20260904_1656.md`。
此前修复/启动简报：`docs/fastwam-robotwin-rlinf-grpo/evidence/SCENE_FENCE_RESTART_AND_HEALTH_20260904.md`。
后续仍先现场刷新；不连接新旧Fast曲线、不报未经验证的新ETA。特别关注完整Step1/旧Step2边界、显存峰值和首个DCP。

## 前轮调查：noOIDN可能暴露既有scene fence缺口（15:17—15:25，历史）

最新证据与修复建议：
`docs/fastwam-robotwin-rlinf-grpo/evidence/FASTWAM_NOOIDN_SCENE_FENCE_CAUSAL_REVIEW_20260904.md`。
15:17 Fast仍完整Step1，Step2停滞；GPU6/7=17699/58727MiB、0%，所查fatal/OIDN/OOM/exit_code均无。
Sidney完整Step48继续，RAMavailable约0.955TiB。15:19非阻塞栈仍为左腕相机取图。
新发现：实际svulkan2 timeline render重载漏接已有scene-access fence，binary重载却完整接入；
15:25对故障进程maps指向的库做磁盘反汇编已核对。连续多相机渲染可能过早重用共享场景命令缓冲/TLAS；
OIDN同步execute可能掩盖缺口，关OIDN与触发有具体机制关联。已确认缺口，不等于已证实本次卡死帧的唯一根因。
下一步主修候选改为补齐既有scene fence协议，保持noOIDN、32env×8和训练预算，不先改线程模型/整库升级；
有限等待/GIL属于独立故障传播修复，不再作为“先试它能否恢复训练”的主线。旧Python补丁不自动套回。
本轮仅研究/读取实际库，无生产修改/安装/暂停或停止训练；新GPU回归、构建/替换、重启run需后续执行授权。
前轮等待链、多job隔离、历史与14:50 TB/checkpoint/整机详查见`FASTWAM_STEP2_STALL_INVESTIGATION_20260904.md`；
未重查的项不作为15:17现状。shared Ray/Sidney不动。原50h ETA失效。以下12:49为启动历史。

## 前轮启动合同：clean Fast-WAM / OIDN off / 256轨迹 fresh100（旧v1已被上节替换）

唯一SSOT：`docs/fastwam-robotwin-rlinf-grpo/12_SHENZHEN_CURRENT_RLINF_GRPO_PORT_PLAN_20260831.md`。
当前实施账本、完整resolved、精确命令、预算和逐叶差异：
`docs/fastwam-robotwin-rlinf-grpo/evidence/CLEAN_NOOIDN_FRESH100_LEDGER_20260904.md`。

- 用户明确授权关闭OIDN、干净基线、从原始SFT重新训练；启动前追加采样翻倍、并行不变。
- 新RoboTwin `codex/sz-robotwin-clean-oidn-off@f3e30a83365c`从0008ae6出发仅BaseTask两处开关；
  vector_env与0008ae6相同，旧生命周期补丁不进入新run，但旧两个分支/worktree原样保留。
- 训练32env×rollout8=256轨迹/步，G8，GB1024/MB2/update_epoch2不变，因此实际4次optimizer calls/步。
  LR5e-6、noise0.3、H32/C24/M10、horizon192、fixed32/eval5、DCP/save10及offload全部继承旧33步run。
- GPU6/7，fresh100，resume=null；新run：
  `/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-fresh100-2gpu32x8-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-clean-nooidn-v1`。
- 12:43 wrapper/owned PGID1052625启动，driver1052633、namespace RLinf_1；12:49已进入首步8轮采样。
  两rank代码路径/原始权重已核验，GPU6/7约54,411/54,409MiB；所查OIDN/fatal/OOM/Traceback均0，尚未完整Step1。
  32x4准备packet从未启动；未验证本次完整更新显存/长程稳定性/保存恢复。
- 预算25,600 train episodes、≤204,800新query records、400actor更新、640eval、10代DCP；
  粗估50h/100GPUh，沿用120h硬上限。旧256+OIDN有OOM记录，本次若OOM停止、不自动调参重试。
- 12:34只读：Sidney完整Step42、所查fatal/OOM=0；shared Ray321933/322685与Sidney3176215保持原进程。
  GPU6/7启动前空闲；RAMavailable1.271TB，/data余1.3TiB。以上均有时间戳，不作为下一轮当前现场。

## 前轮：OIDN开关已接通并完成小尝试（12:06—12:11，历史）

唯一当前专题仍为Fast-WAM current port；最新账本/具体参数/图：
`docs/fastwam-robotwin-rlinf-grpo/evidence/OIDN_TOGGLE_TRIAL_LEDGER_20260904.md`。
用户本轮已授权一些尝试，不再是只讨论：在独立RoboTwin worktree中仅两处接入
`ray_tracing_denoiser`参数（默认oidn）；开/关各3个短回合，均exit0，无本次fatal/OOM。
使用原始SFT权重而非Step30恢复；开1/3、关0/3，观测有明显颗粒、动作小幅变化，不能据此定成功率或长期稳定性。
旧Python生命周期补丁保留且hash不变，关闭OIDN不会自动使其失效；没有改共享库、Sidney或shared Ray。
GPU6运行采样约28GB，12:11结束后GPU6/7各5MiB；11:50刷新Sidney到Step40、所查fatal/OOM=0，
Fast仍最后Step33/exit255，Step30双DCP与metadata仍在（未加载）。下面10:44指标是历史快照。

## 历史服务器快照（2026-09-04 10:42—10:44 CST）

最新现场、Fast-WAM图与对照、上游issue、补丁去留判断：
`docs/fastwam-robotwin-rlinf-grpo/evidence/FASTWAM_LEARNING_AND_MINIMAL_FIX_REVIEW_20260904.md`。
该次刷新仅普通账号只读，未重做SMART/内核等整机深审。更早详细健康证据：
`docs/fastwam-robotwin-rlinf-grpo/evidence/OIDN_RECURRENCE_INVESTIGATION_20260904.md`。
前轮修复讨论：`docs/fastwam-robotwin-rlinf-grpo/evidence/OIDN_CLEAN_FIX_DISCUSSION_20260904.md`；
更正clear_cache的语义与实际清理频率，核对OIDN上游GPU释放修复；仅讨论，未实施。
包含Sidney图、整机健康、旧故障/补丁/复发因果与原始证据索引；09:35上一轮刷新仍保留在
`docs/rlinf-shenzhen-multitask-pi05/evidence/TWO_RUNS_READONLY_REFRESH_20260904.md`，不是最新状态。
近期窗口交接仍保留09-03历史快照，不应覆盖本节的更新结果。

前轮讨论与用户选择（本轮授权与执行已由上节更新）：
`docs/fastwam-robotwin-rlinf-grpo/evidence/OIDN_OWNER_AND_ONOFF_FOLLOWUP_20260904.md`。
旧补丁按用户选择先保留，后续删减无必要部分；允许将1—2例有/无OIDN画面与动作对照前置，
评估关闭降噪这个备选的代价，不要求先修复失败数次。当时尚未运行GPU对照。
本次只核对锁定原生源码，未再刷新现场：确认相机→denoiser→device→key所有权链；
若走原生修复线，需同时修底层检查与上层静默退到无降噪的问题；主动关闭OIDN不以该修复为前置。

### GPU 4/5：Sidney 多任务 pi0.5 GRPO

- 任务：`move_pillbottle_pad`，fresh formal100。
- 17:10只读快照：完整Step53/100，训练145/256=`56.64%`、MA10=`58.48%`；wrapper存活。
- fixed32 Step5/10/15/20/25/30/35/40/45/50=
  `10,19,15,14,16,14,19,15,14,16 /32`，尚未持续超过Step10；所查driver fatal/OOM/OIDN/Traceback/RuntimeError=0。
- Step50 local-shard双rank及full_weights均已存在；文件级核验，未执行恢复测试。
- 前轮09:58条件估计为09-05 11:30 CST到Step100；本轮未重算ETA，不将旧估计当承诺。
- 最新Sidney/旧π0对比图：`docs/rlinf-shenzhen-multitask-pi05/evidence/pi05-pi0-comparison-20260904/dashboard.html`；此前独立图为09:58历史快照。
- 主要训练壳：64 train / 32 eval env，rollout4，256 trajectories，G8，
  GB1024/MB32/update2，H50/C50/M10，noise0.5，horizon200，fixed32/eval5/save10，local-shard。
- 相对此前 Sidney `move_stapler_pad` 只改任务名及 run/output 路径；训练、资源、模型叶均相同。
- 运行目录：
  `/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/move-pillbottle-pad-grpo-formal100-2gpu64x4-g8-b1024-u2-m10-noise0p5-h200-fixed32-eval5-phys45-localshard-v1`
- 切换账本：
  `docs/rlinf-shenzhen-multitask-pi05/evidence/SIDNEY_MOVE_PILLBOTTLE_GRPO_CUTOVER_20260903.md`。

### 历史GPU6/7：旧Fast-WAM renderer-life续训已异常退出（新run见顶节）

- 任务：`move_stapler_pad`，从完整 Step10 DCP 恢复到100。
- 最后完整Step33；success/MA5/MA10=`14.06/32.50/28.75%`；本续训fixed32
  Step15/20/25/30=`13/32,9/32,14/32,14/32`。
- Step34训练rollout时于09-03 23:49:18 CST先报OIDN `pthread_key_create failed`，随后
  invalid handle、Python autoTSSkey fatal与通信退出；23:51:49 CST结束，exit=255。
- 已越过旧Step15边界，但长程故障仍复现；两rank fatal栈证明补丁实际加载，文件hash/HEAD均匹配。
  新终止栈在full reset关闭child的scene析构处、尚未执行本轮global clear；补丁调整了cache清理分工/时机。
  OIDN首错对应进程内pthread键分配失败（本机上限1024，非OS线程数上限）；原生资源累积/状态损坏最可疑，
  尚未定位具体未释放对象，不能声称两次唯一根因相同或补丁彻底有效。
- 10:13源码/配置更正：clear_cache只清资源registry，不是强制销毁renderer；旧“确定根因”定性过强。
  实际train/eval清理频率1（非默认8）。最新判断：旧补丁有效性/必要性待证；按用户选择先保留，
  但最终必须逐块审计，删除无必要改动，不因“结构更清楚”永久保留。未整库升级或改变实际RGB；
  有/无OIDN隔离小对照已成为允许讨论的优先下一步，不再将“不得改变RGB”作为绝对候选禁令。
  OIDN2.1.0含GPU释放崩溃修复、2.2.1/2.2.2含后续泄漏修复，但尚未证明本次命中同一bug；不能盲升/混搭库。
- Step30 DCP双distcp与metadata均存在、非零，未做恢复加载；GPU6/7现无该run计算进程。
- 主要训练壳：32 env x rollout4=128 trajectories，G8，GB1024/MB2/update2，
  H32/C24/M10，fixed32/eval5/save10，DCP。
- 本轮图：`docs/fastwam-robotwin-rlinf-grpo/evidence/fastwam-review-20260904/dashboard.html`，4张PNG、7图。
  主线首/末10步训练均值29.14/28.75%，fixed评估末次回到14/32，未证明持续RL增益。
  256轨迹pi0style-v3仅Step5评估18/32，随后OOM；不能视作扩量已有效。
- 运行目录：
  `/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/runs/fastwam-grpo-control-resume10-to100-2gpu32x4-g8-b1024-u2-m10-fixed32-eval5-phys67-dcp-renderlife-v2`
- 修复账本：
  `docs/fastwam-robotwin-rlinf-grpo/evidence/ROBOTWIN_VECTOR_RENDER_LIFECYCLE_FIX_LEDGER_20260903.md`。

### 整机

- 10:42 RAM available约1.165 TiB，GPU4/5约67.07/67.35 GiB，GPU6/7各5 MiB，load5.31；
  shared Ray原进程仍在，未作任何干预。以下PSI/磁盘/内核/服务等为前轮09:44—09:58证据，非本轮重检。
- 前轮RAM available约1.16 TiB，memory/io PSI 10/60/300秒窗口均为0；磁盘余量：`/`222.88 GiB、
  `/home`1.31 TiB、`/data`1.30 TiB。
- GPU0：liwenbo StarVLA约9.5 GiB；GPU1/2/3/6/7采样时无compute进程；
  GPU4/5为Sidney，09:58显存约67.07/67.35 GiB，随阶段波动；两EnvWorker RSS约375/360 GiB，
  整机MemAvailable缓降但不能单凭该曲线确认泄漏。
- CPU采样空闲约96%，load约5/128逻辑CPU；GPU ECC/row remap与NVMe SMART无错误信号，
  管理员只读内核日志未检出本次对应OOM/Xid/I/O/MCE事件，cgroup无oom/pid触限事件。
- shared Ray原gcs_server/raylet存活约11天9小时，未重启；ssh/mihomo active，无failed unit。
  GitHub直连通/代理超时；HF直连失败/代理通（有限时长HEAD探针，不保证持续连通）。
- 不得干预其他用户进程。多任务并发只清理精确 owned PGID 与 exact Ray namespace。

## 当前代码与专题入口

- Sidney多任务 pi0.5 current RLinf：
  `docs/rlinf-shenzhen-multitask-pi05/00_INDEX_AND_EXECUTION.md`。
  - branch：`codex/sz-sidney-pi05-current-rlinf`
  - implementation：`bab221afb8be`
  - 含轻量 evidence 的当前已推 HEAD：`f50e235c5ab1`
- Fast-WAM current RLinf GRPO：
  `docs/fastwam-robotwin-rlinf-grpo/12_SHENZHEN_CURRENT_RLINF_GRPO_PORT_PLAN_20260831.md`。
  - plain implementation/start lock：`codex/sz-fastwam-current-rlinf-grpo@7b2331c5`
  - 16:56 live HEAD：`62526cc95047`，working tree clean；包含当前scene-fence Env-local接入，旧`4faade1d50bf`为修复前历史。
  - DVAC branch/commit：`codex/sz-fastwam-action-dvac-adv@a6ad77ea`
- RoboTwin renderer生命周期修复：
  `codex/sz-robotwin-vector-render-lifecycle-fix@8c7380c1`（`Yutenji-Nyamu/RoboTwin`）。
- OIDN显式开关隔离副本：`codex/sz-robotwin-oidn-toggle@b76d4ed`（已推personal），
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-oidn-toggle-20260904`；
  仅BaseTask两处，默认oidn，旧vector_env补丁不变；本轮诊断脚本仍为该树未跟踪文件，勿误删。
- 新训练RoboTwin：`codex/sz-robotwin-clean-oidn-off@f3e30a83365c`（已推personal），
  `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/robotwin-clean-oidn-off-20260904`；
  无旧vector_env补丁，train/eval显式none；没有升级SAPIEN/OIDN或修改共享venv。
- 深圳模型/算法扩展总入口：
  `docs/rlinf-shenzhen-experiment-expansion/00_INDEX_AND_PLAN.md`。
- 本窗口完整近期交接：
  `docs/window-handoffs/20260903_RLINF_SHENZHEN_WINDOW_HANDOFF.md`。

## 继续工作时必须保留的经验

- 同一 shared Ray 下的不同 RLinf job 必须使用独立 worktree、独立 namespace、显式
  `RLINF_CODE_WORKING_DIR` 与 run-scoped 绝对输出路径。
- 只按 owned PGID 和 exact namespace停止单项任务；不要重启 shared Ray，也不要影响另一项训练。
- 方法对比必须逐叶继承 Control；采样量、并发、batch或update预算变化必须单独说明依据。
- 只做少量高信息检查；动态 step、GPU/RAM、fatal、checkpoint与Git状态在回答前现场刷新。
- 文档只保留证据，聊天必须直接回答用户每个问题并给出对应链接。

## 当前授权边界与下一步

- 最新π0.5/π0学习比较请求为诊断与讨论：仅只读刷新、源码/配置核对、网络资料与本地证据/图更新。没有授权对Sidney改参、增加评估或重启；本轮没有执行这些动作。
- 2026-09-04 15:47后用户新授权：Fast分支隔离修scene fence、简要测试/推送、停止替换当前256/noOIDN停滞run并重启，随后刷新可视化。具体执行以`docs/fastwam-robotwin-rlinf-grpo/evidence/SCENE_FENCE_FIX_RESTART_LEDGER_20260904.md`为本轮账本；保留采样/训练合同，不覆盖shared venv，不动Sidney/shared Ray。下列“只读/不实施”描述属于此前轮次，不再否定本项新授权。
- 当前用户已明确授权本页顶部的clean/noOIDN fresh100，并追加32并行不变、采样4→8轮。
  已展示完整配置、命令、资源、预算与停止条件；仅本项新run可执行，不授权干预Sidney/其他用户、重启shared Ray或升级共享环境。
- 前轮曾以普通账号只读SSH、toom/sudo只读核对内核/SMART/服务，以及本地可视化/证据/交接更新；
  没有更改shared Ray、服务器代码/依赖、训练或其他用户进程，没有运行任何原生渲染诊断或恢复测试。
- 前轮普通账号登录并完成开/关各3短回合；该诊断已结束。本轮新fresh100已启动，未改变其他用户/现役任务。
- 后续查看训练仍须刷新现场，不依据本快照宣称“当前”。
- 旧Fast256/noOIDN v1的Step2阻塞已按本轮授权停止，由顶部scene-fence-v3替换；本轮窄修/短测/推送/重启授权已执行。
  后续不擅自降并行、改GB或循环重启；保持原预算，不把旧50h估计当有效ETA。
  不将新旧run或有/无OIDN的指标无标记拼接；旧OIDN资源调查保留但不是本次修复内容。
  原生修复线仍需隔离计数构造/最终析构与真实key创建/删除，定位具体持有者；只回移确认相关的上游修复。
  错误传播须覆盖构造、执行和上层caller，requested=oidn失败不能静默转none；不以此冒充泄漏根治。
  当前锁2.0.1不整库升级，任何改动不得覆盖shared venv；最终删去无必要旧Python补丁。
  超出本次已授权fresh100的新smoke/训练仍先展示命令/配置/资源/停止条件并获得授权。
  Sidney不作干预，最新快照见顶部；旧fixed32结果保留原时间，后续状态需重新刷新。
  学习诊断优先有效G8组数、valid加权ratio/clip、原始checkpoint同协议baseline与少量失败轨迹；
  不根据MB均值偏小就直接加LR/noise，不把异任务π0/π0.5高成功率当严格模型对照。
