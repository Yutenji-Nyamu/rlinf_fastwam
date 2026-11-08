# 29 · 两路 RT2 训练终点翻倍

用户授权：两个RT2训练终点乘2。2026-11-08已登记同卡完整续训，未停止当前训练。

|槽位|方法|原终点|新终点|总训练局数（16环境）|
|---|---|---:|---:|---:|
|4|RT2 DSRL|200|400|6400|
|7|RT2 RLT|313|626|10016|

5卡Dojo DSRL保持200轮，6卡Dojo RLT保持313轮。UTD、batch、并行数、噪声、动作长度、奖励、Q预热、BC、评估间隔和有效加速配置均保持原样。配置深比较只允许max_steps和resume_dir变化。

运行中的EmbodiedRunner使用启动时创建的range，修改磁盘YAML不能延长现有循环。因此复用既有tools/robotwin_dsrl_full/extend_four_to400.py和continue_owner.py的等待/交接方式：原进程正常跑完200/313轮并保存，CPU等待owner接管同一卡，从该完整检查点继续到400/626。恢复模型、优化器、目标Q、方法计数和回放，不重新warmup。环境会按既有恢复入口重新初始化，这不是进程内热更新或逐位一致的仿真恢复。

新部署根：`/data/chenyiteng/deployment-20261108/rt2-double-budget`，两个目录名仍为`rt2-dsrl-clean-g4`和`rt2-rlt-clean-g7`。源码继续使用上一轮已验证exports；结果/曲线/最佳候选目录保持原路径，轮次从200/313继续。未来终点配置在新部署的config/formal.yaml；当前旧部署配置200/313仍描述正在运行的第一段，不能误报翻倍未生效。

## 队列

- 主路由active-runs.json保留当前正在训练的owner，另记pending_continuation和total_target_steps/episodes；续训真正启动时自动更新当前owner。
- RLT总调度4/7预约到等待owner，external_blockers保留当前训练owner；5/6的调度字段逐项比较不变。
- FQL join已改为等待两个续训owner释放；单卡先完成时由旧RLT填补，双卡延长训练完成后才接FQL。
- 只短暂重启两个CPU调度进程，未停止训练、重置Ray或改其他用户进程。等待owner异常有正常RLT回退；沿用既有watch_owner。

## 验证和边界

两路实际生产配置CPU校验通过；现存完整DSRL CP119、RLT CP100检查通过。DSRL检查local_shard/SAC状态/回放，RLT检查DCP分片/小策略/RLT状态/回放，避免再用DSRL文件名检查RLT。过期marker、缺失DCP及空回放的CPU回归测试通过；Ruff lint/format通过。配置校验初次因Ray节点探针缺少PYTHONPATH失败，补齐已有源码路径后通过，训练进程未受影响。

11:53回读：原两路训练owner和新等待owner均存活；调度和join存活，预约/阻塞/释放路径匹配；5/6仍运行。当前只验收“续训已准备并登记”，尚未到200/313，因此没有宣称本次实际加载未来检查点已完成。预先改到目标轮数不会被原owner使用；必须保留本次登记。

轻量证据：`evidence/rt2-double-budget-20261108/{prepared,armed,verification,package-qa}.json`及两路新配置。`package-check.json`记录登记前源码哈希，`package-qa.json`记录随后仅格式/import清理和dict字面量整理后的发布哈希；未改变队列逻辑。不包含训练环境凭据、权重或回放。
