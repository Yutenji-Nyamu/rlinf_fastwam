# TOPReward 连续末段奖励接入

本轮授权：自主实现、测试、推送和训练；仅深圳 2 机物理 4/5 卡。

- [01_CONTEXT.md](01_CONTEXT.md)：现有证据、旧链路及此次差异。
- [02_IMPLEMENTATION.md](02_IMPLEMENTATION.md)：奖励公式、代码接口与验证。
- [03_EXECUTION.md](03_EXECUTION.md)：实际配置、命令、资源、停止与恢复。
- [04_RESULTS.md](04_RESULTS.md)：执行结果；未执行的检查明确保留为待验证。

结论边界：旧标定否定了 Qwen 阈值直接决定成功早停，不是否定其排序信号。此轮训练候选使用连续概率，不对现有留出集继续拟合阈值。
