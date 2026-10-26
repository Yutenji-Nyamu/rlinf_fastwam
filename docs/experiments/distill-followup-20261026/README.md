# 2026-10-26 蒸馏对齐、稳定训练与性能讨论

正式训练未改。对照现场配置、数据清单、时间戳、实际attention路径；CPU小样只解码3任务各1轨迹的192窗口，未初始化CUDA。无权重、原始图像、完整视频或系统用户清单。

- [54 对齐检查](../../pi05_distillation/54_ONEDP_ENDPOINT_ALIGNMENT_20261026.md)
- [55 DMD工程来源与建议](../../pi05_distillation/55_DMD_STABILITY_RECIPES_20261026.md)
- [56 时间账与提速优先级](../../pi05_distillation/56_TRAINING_TIME_AND_SPEED_PLAN_20261026.md)
- `alignment-configs.json`：两方法实配，任务manifest内容相同。
- `performance.json`：按队列事件划分训练/评估；加载和更新子项为时间戳估算。
- `cpu-decode-and-attention.json`：独立CPU解码样本及安装代码attention分支，非训练整步profiling。
