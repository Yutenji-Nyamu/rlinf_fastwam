# EXPO 正式运行 · 2026-10-01

用户明确授权：四卡B64 smoke通过后直接正式；RLT为低优先级备用，只在EXPO结束/故障后恢复。预算按论文规模取20,000个在线真实动作，含warmup；评估、demo不计入在线预算。

配置继承[已通过规模smoke](FORMAL_SCALE_SMOKE_20261001.md)：SZ2物理4–7，π0.5 Sidney原权重/norm，action expert＋投影，无LoRA、视觉语言冻结，无随机图像增强；训练1环境，200动作上限、H50/C10/14D、Euler10；每观测8base＋8edit，四卡分batch，全局B64、Q20、FM/editor/温度各1。正式从原基座新建learner，不使用smoke更新后的权重。

调度：前10在线episode仅收集，不积欠更新；随后每40真实动作一个call，在episode末执行，余数持久保存。warmup、episode、实际动作、待执行call、learner counters和全部RNG进入断点；预算在episode中耗尽时保存真实final observation、标明确预算截断，不补动作或余数call。周期checkpoint只保留latest和last1。

固定评估继承Control的pick_diverse_bottles20种子名单、200动作/C10、4环境×5组、success_once/ignore_terminations语义；起点raw π0.5单候选，之后每25在线episode及结束评EXPO策略。EXPO的25episode不能声称与RLT25采样轮采集量相同。eval不进replay、warmup、20k或40动作计数。既有seed表来自其他任务表改键且expert_verified=false，沿用可复现名单，不称本任务专家筛选。

资源/输出计划：`/data/chenyiteng/projects/expo-ft-sz2-20261001/formal-20261001`；正式driver `examples/embodiment/train_expo_formal.py --inputs …/inputs.json --run …/run --max-physical-actions 20000`。独立owner精确清理当前四RLT命名空间及拥有的后代，复用已冻结完整四CP；共享Ray/其他用户保持。EXPO结束/异常/心跳失联后仅清理本EXPO进程，恢复四RLT及首轮回执。启动前固定源码、输入和种子SHA，停止条件为预算完成、非finite、资源身份/配置漂移或进程故障。

现场：2026-10-01已启动正式owner（PID211750）和driver（PID314046），四个原RLT driver/命名空间已停止且4–7卡释放。新cycle保留已验证完整CP1575/1600/1550/1550作为故障恢复点。服务器CPU的11项cadence＋10项driver检查通过；owner独立持锁，聊天/SSH断开不终止训练。当前先初始化原基座，随后进行20种子起点评估；在线动作和learner call均为0，不能把smoke更新计入正式进度。

启动命令含`--enable-evaluation`，输入SHA为`c1e65e1964a9b9ab7994f207d09736ec03db03fae0d231bd6512cdc424b6f384`。owner状态`current.json`、driver状态`run/status.json`，失败`run/failure.json`，最终`run/complete.json`；每call及episode边界保存完整状态。

一次学习call实测207秒；200动作约5calls即17.3分钟学习，另加采集/评估/保存。按20k约450–500calls，学习约26–29小时；仅首call估计，随着数据池增长应再刷新速度与ETA。
