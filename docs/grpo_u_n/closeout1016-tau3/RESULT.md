# GRPO U τ3 pillbottle：授权提前收束

2026-10-16 22:38，用户要求收束并用SZ2物理6/7启动新的EXPO B64 U-BC。旧driver正常SIGTERM退出，exit143、cleanup_error=null；原Ray namespace已无活跃actor，换卡前6/7计算与图形均释放。未改共享Ray和SZ2的GRPO τ2。

- 完成119/200轮；最新固定评估21/32=65.625%（第118轮）。
- 前5次固定评估均值41.25%，末5次61.875%，峰值75%。训练过程已有提升，但同期τ2末5次77.5%更高；该比较是单次运行的描述，缺少等预算Clean，不能得出U净收益结论。
- 最近完整保存点CP110保留；119轮的训练进度不是CP119。未删任何旧检查点。
- primary-logs.tar.gz保留原始driver日志、TensorBoard、实配、请求、退出/清退回执和全量标量；summary.json记录权重精确路径与大小，evaluations.csv保留每次固定评估。不是满200轮完成报告。

新组运行：`/data/chenyiteng/results/expo-ubc-b64-tau2-sz2-g67-60000-1016`；RLT6/7原候补请求保持，现等待新EXPO释放。
