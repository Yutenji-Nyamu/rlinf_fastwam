# U-GROW信号接入BC与RLT-BC · 2026-10-06

本窗口负责深圳1机物理GPU4的BC 4/U5＋U与GPU5的RLT-BC＋U。用户授权整理上下文、实现、检查、smoke、推送和运行；U实验优先，原卡RLT仅在本次事务终止、确认释放后接替。6/7由另一窗口执行DSRL，本窗口不操作其进程、源码或路由。凭据不落盘。

## 当前依据与方法

规划来源：`C:/Users/86136/Documents/seek/outputs/ugrow_training_readiness_2026-10-06/{01_BC,02_RLT,05_U_PRODUCER}.md`；原信号实例为`outputs/dvca_signal_study_2026-10-03/23_UGROW_INSTANCE.md`。

同输入、同完整初始噪声，以确定性ODE分别完整生成10步和5步动作终点。逐坐标总体标准差除以RMS＋1e-8，平均14有效坐标，保留动作位置。复用主10步、prefix cache；旁路5步保存/恢复RNG。主动作与原数据预算保持。U只在采集计算，随回放保存，不在每个训练microbatch重算。

| 实验 | GPU4：BC＋U | GPU5：RLT-BC＋U |
|---|---|---|
| 历史来源 | Clean01d770db，旧DV ad3da329 | π0.5 N4 pair460008008aea310f83a0b19008c7c105a0a19980 |
| 任务/模型 | move_pillbottle_pad / Sidney π0.5 | adjust_bottle / Sidney π0.5 |
| 预算 | 100轮，N4/U5，global1024/micro32 | 800轮，N4/U5，B512/micro256，10k初池/15k初始化 |
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

已完成历史规划及接点审查；2026-10-06本轮固定host-key与chenyiteng/UID1003/admin身份验证通过，首次只读状态已保存。当前尚未改训练源码、未停止RLT、未启动U实验。后续实配、测试、smoke和发布结果追加到本目录，以实际回执为准。

本轮工作文件：`local_scripts/ugrow_bc_rlt_20261006/`。其他窗口请保持GPU4/5本窗口事务所有权，不重放旧RLT恢复入口。
