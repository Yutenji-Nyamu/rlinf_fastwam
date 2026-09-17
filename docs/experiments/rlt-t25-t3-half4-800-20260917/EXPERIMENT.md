# RLT Clean4协议：tau2.5 / tau3，成功倍率1，fresh800

两组从已成功tau2源码固定版本 606fbc3e7fce9d4cf28e44075d5ab93029452880 派生；生产算法代码没有修改。此提交仅记录配置、逐卡切换和真实首轮验收。

| GPU | 方法 | 实际wrapper启动（北京） | 首轮 | driver PID |
| --- | --- | --- | --- | --- |
| 6 | 双tau2.5/scale1 | 2026-09-17T11:30:10+00:00 | R1，update0，replay80.0 | 3343527 |
| 7 | 双tau3.0/scale1 | 2026-09-17T11:30:14+00:00 | R1，update0，replay80.0 | 3362483 |

两组均每轮4环境×rollout_epoch1，B512/micro256、U5、critic:actor=2:1、actor/critic LR1e-4 constant、replay80k、10k池预热/15k初始化/cap800、BC/Q课程10k+25k。Stage1 CP2000及原训练/固定评估种子保持；每25轮fixed20并保存。

方法为success_episode_bc、two_level_batch、exp_mean、双alpha1、成功总倍率1；两组只改变双温度2.5/3，无方向或倍率调度。相对Clean4仅方法字段、600→800终点及独立路径/身份变化；两组resume_dir/ckpt_path均为空，未从旧任务权重恢复。

首轮实测：12个同namespace/同job/本人actors，4环境，fresh R1，更新计数0，teacher收集阶段online0，标量finite，所查driver日志无fatal，实际配置与prepared逐叶diff为空。共享Ray及0–5卡保护对象在逐卡切换回执中保持。无需额外smoke或整套测试。

停止条件：累计R800、用户明确停止或不可恢复错误；无性能截止阈值。

ETA参考：9月17日19:04同协议tau2已到R675，其完整已运行时长加近期R626–675的普通轮速及剩余评估/保存开销，预计fresh800总耗时约22.63小时。平移至本次19:30启动约9月18日18:08完成；新温度和共享负载会改变速度，属于协议迁移估计，不是统计置信区间。

此docs-only提交使Git HEAD比启动时增加一代；运行代码及ops保持不变。后续检查应按生产路径差异/冻结代码内容判断，不能把docs提交视作算法变更。
