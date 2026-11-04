# Dojo5/6、RT2 4/7与FQL汇合 · 2026-11-04

最终用户确认：稳定Dojo卡为5/6，因此5跑Dojo Clean、6跑Dojo＋U；4/7分别RT2 Clean/＋U。4/7先结束的一张接RLT，两路都结束再恢复双卡FQL。

## 实配与依据

| 卡 | 主任务 | 控制依据 | 后续 |
|---|---|---|---|
| 4 | RT2 Clean从完整CP52续到200轮，200动作，N10/H50/C20 | 原6卡同一run，保留模型/优化器/目标Q/温度/回放；未落盘数轮重跑 | 单卡RLT等待7卡 |
| 5 | 原Dojo Clean100轮 | 保持现有运行 | RLT |
| 6 | Dojo扔瓶子DSRL＋U，100轮×8局，700动作，N10/H50/C50 | 5卡Clean＋已验收RT2 U损失/回放 | RLT |
| 7 | RT2 turn_switch DSRL＋U，200轮×8局，200动作，N10/H50/C20 | 原6卡计划U迁至7卡，从原模型开始 | 单卡RLT等待4卡 |
| 4＋7汇合 | 原单任务FQL v2从完整880更新续到2200＝200预算轮 | 同输出、已完成评估、相同数据/参数 | 各卡RLT |

U复用现有同观察/同完整噪声N10与N5差异，真实14维，按实际提交chunk聚合。Dojo仅将U的chunk_length跟随本身C50；RT2仍C20。τ2.5、均值1的chunk权重乘整项SAC actor目标，Q/温度/奖励/回放采样保持。完整1600维噪声及U回放schema4沿用11/04实现。

新U两路先1轮真实短测并恢复到2轮，再从原始模型/空回放正式开始；4卡Clean直接完整续训；短测仅40/50动作、UTD1，不改变正式预算。两路均不继承Clean学到的噪声策略，保证同起点对照。正式保留滚动最新完整恢复点，另保小峰值噪声策略。新产物均写Data。

## 卡位与调度

历史4卡有一次Isaac初始化设备错误，7卡多次设备故障；因此优先4卡Dojo、7卡RT2。Dojo仍使用已验收native_entry、supervisor、精确清场和600秒初始化上限，错误退出接RLT，不无限修改仿真底层。不能事先宣称4卡已修复。

准备时FQL继续运行，先等当前完整点对应评估收尾。新owner在注册前不占GPU；精确暂停原FQL、释放后两路直接领取。唯一既有RLT调度器只改4/7预约，不动其他用户、5/6或共享Ray。

两路释放彼此独立：一张完成或失败，原RLT立即领取该卡；CPU汇合脚本等待两路释放，再调用已有priority_pair_handoff保留RLT最新恢复点并接双卡FQL。汇合失败则保留RLT占卡，不空转。6卡旧队列U用已有defer_formal_to入口指向7卡，避免重复正式训练。

源码：tools/robotwin_dsrl_full/prepare_parallel_u.py、handoff_parallel_u.py、join_fql.py。Dojo owner仅复用已有admission_card过滤已登记邻卡极小图形上下文，仍拒绝任何未登记计算进程。隔离源码：/data/chenyiteng/projects/rlinf-shenzhen/worktrees/parallel-dsrl-u-20261104。

执行回执与GPU实际验收见本次evidence；准备完成不等于已开正式训练。


## 现场换卡与实际故障

19:46，FQL从880点暂停后4卡Dojo在Isaac初始化12.5秒报ERROR_DEVICE_LOST，未进入模型和RL更新。19:46:57精确清理释放，19:46:58原RLT自动接回，间隔约1秒。

用户随后确认使用历史可用的5/6卡。19:51完成新交接：先准备6卡Dojo＋U、4卡RT2完整恢复配置及独立EGL绑定，再从4卡RLT接管、停止6卡RT2的精确owner，5卡与7卡持续运行。4卡沿旧GPU scope实现，仅实例化physical4的manifest/独立marker及profile，其他用户和原scope6/7未改。

汇合对象已改为4卡RT2 Clean和7卡RT2 U；6卡Dojo U独立结束接RLT，不阻塞FQL双卡。源码补充flexible_56_dojo.py，保留初次失败与后续交接回执。
