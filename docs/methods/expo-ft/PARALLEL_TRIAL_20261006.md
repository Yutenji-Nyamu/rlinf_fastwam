# EXPO单卡、两卡容量试验 · 2026-10-06

用户授权按单卡优先、失败再两卡尝试。只改变物理卡数：先GPU4，再GPU4–5；两种都未通过则恢复已验证GPU4–7。保持采集N1、评估N4×5/每10回合、8+8候选、全局B64、所有microbatch、Q20/FM1/editor1/temperature1、ODE10、学习率、60k预算及latest/last1保存机制。

从运行源码061d4995和当前完整断点继续。新源码仅使正式输入检查允许1/2/4卡且base/core一致；候选生成、损失、采样和优化器实现保持。断点仅迁移base合同/core配置的设备数和CUDA RNG设备列表，权重、优化器、回放、进度及全部动作/学习计数逐项验证保持；浮点与多卡随机序列变化不承诺后续轨迹逐位相同。

控制/输出：`/data/chenyiteng/projects/expo-ft-sz2-20261001/parallel-trial-20261006`。入口：`source/tools/expo_parallel_20261006/owner.py`；GPU探针`probe.py --devices 1`或`2`；CPU迁移`migrate.py --devices N`。计算和图形使用同组UUID与独立commname/EGL可见掩码，不占物理0–3。

流程：CPU检查与源文件锁定 → 精确接管旧60k owner → learner边界正常停止并核完整CP → 复制latest/last1及小元数据，不动回放文件 → 独立单卡N1/N4各10真实动作与完整B64学习调用 → 失败清卡后两卡同测 → 通过才迁移正式CP并续训。探针的50个仿真动作及一次更新不进入正式回放、动作预算或模型状态。

每个探针45分钟上限，心跳15分钟上限；OOM/非零退出或超时判本候选失败。每次清理仅对应身份锁定的进程树，计算/图形上下文完全释放才启动下一候选。两者失败沿原四卡配置恢复；正式运行完成或故障，沿原60k借卡周期释放资源后恢复四路RLT并核首轮。其他用户、共享Ray、1/3机不动。

此前只验证过四卡正式B64，单卡旧B4/FM1不是本次验收。本次结果以`trial-N/result.json`、`selection.json`、`migration.json`及正式`resume_verified`/checkpoint增量为准；未有这些回执前不称单卡/两卡已通过。
