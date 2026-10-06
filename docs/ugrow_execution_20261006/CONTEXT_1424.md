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

## 当前执行 · 14:24:38 北京时间

用户明确要求5卡直接正式。14:19:49启动新事务`/data/chenyiteng/deployment-20261006/ugrow-rlt-g5-formal-v3/rlt`，owner2665562/start415430853，driver2665696/start415431106；namespace/run `ugrow-rlt-g5-formal-1006-v3`，结果根`/data/chenyiteng/results/rlinf-rlt/`。沿原任务Stage1新开Stage2，800轮、N4/U5、B512/micro256、10k预采集/15k初始化更新、每25轮固定20条评估/保存均保持；实配仅6处输出路径/实验名变化。

正式首轮采集与U回执已验，当前回放80条、update0；这是原预热阶段，尚未证实成功样本的非均匀权重更新或训练收益。既有4轮teacher smoke为exit0、完整CP4/update8/回放320、0/16成功；本次按用户新指令直接正式，不把旧成功覆盖缺口改写为smoke通过，不再追加smoke。训练源码仍为RLT `bfbc9c889bfe687dc24de7c3dc1d66138d3010df`，只在独立runtime_v3中加入带授权记录的直接正式入口。

旧5卡RLT driver2516932/start415042475已精确停止，其cleanup/namespace/进程/C与G清空后才启动U；完整CP100/回放15557保留，CP100之后未保存的预采集在日后恢复时重做。新U事务结束或异常并验证释放后，原place_fan combo N8/3000由同owner单次接回`rlt-g5-after-ugrow-1006-v3`。不并行抢占，不重放旧归还入口。

4卡BC＋U仍由v2 owner2485016/start414961836、driver2485173/start414962252运行，当前已记录14/100轮，当前轮成功3/4；该进程未切换。两条源码与pins现场核同、工作树干净；计算/图形均在本卡4/5，无本任务跨卡上下文。磁盘余量/home 297.3GiB、/data 443.4GiB。6/7另窗保持。

原127项服务器CPU检查（另3子测试）和真实smoke证据沿用；本次通过实际身份/配置diff/清理/正式首轮验证，不更改训练实现、不重复扩展测试。RLT owner沿原7天上限，输出盘低于40GiB、异卡绑定或进程失败停止，清理后才接候补RLT。旧全文本地归档`archive/CONTEXT_20261006_1325.md`，发布副本为`docs/ugrow_execution_20261006/CONTEXT_1325.md`。

发布路由：[独立证据分支](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/ugrow-g45-evidence-20261006)；`tools/ugrow_ops_v3/`是本次独立入口，`docs/ugrow_execution_20261006/gpu5-formal-v3.json`含身份、实配差异、原RLT释放与正式首轮实证。训练源码BC `1c4b3810a1cddcd1ae38312126448c80348f5686`、RLT `bfbc9c889bfe687dc24de7c3dc1d66138d3010df`保持已推版本；发布不改运行HEAD。

## 前序记录

v1两条smoke训练均正常退出；BC第二轮真实非1权重，原owner误用不存在的update_step阻断。v2改用BC真实loss/grad/lr/replay并复用原smoke，13:01接100轮正式；RLT固定4轮teacher补验仍0成功，于13:15先归还原RLT，13:21恢复首轮回放15557→15710已验。14:19按用户新指令启动本次5卡正式。完整历史和原始证据保留，禁止把历史路由当当前入口。

本轮工作文件`local_scripts/ugrow_bc_rlt_20261006/`。其他窗口请保持GPU4的v2 BC、GPU5的v3 RLT-BC＋U所有权，读取各自status/terminal/return回执，勿重放owner。
