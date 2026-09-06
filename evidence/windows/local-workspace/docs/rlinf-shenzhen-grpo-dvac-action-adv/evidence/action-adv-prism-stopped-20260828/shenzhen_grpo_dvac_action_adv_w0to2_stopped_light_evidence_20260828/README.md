# GRPO-DVAC Action-Adv [0,2] 轻量收尾包

最后完整 outer step：`29`。

包含 driver/resource 日志、resolved 与小型运行合同、TensorBoard event/config、逐步核心指标、fixed-32 评估点和三张图。

注意：This stopped v1 used mean reduction over H=50 action losses. Its pre-clip gradient scale was about 45.6x below matched Control. Action-level ratio/clip telemetry also used a Bx1 denominator for BxH values; those old ratio/clip fields are retained only as raw evidence and are not plotted as comparable metrics.

不含 checkpoint、视频、RoboTwin 大数据、Ray 全量日志或逐动作 tensor。
