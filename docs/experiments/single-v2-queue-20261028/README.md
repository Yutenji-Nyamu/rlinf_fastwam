# 2026-10-28：FQL收束、旧MFD续训、三v2正式队列

用户授权的源码和轻量证据；不含权重、原始H5图像、回放、视频或凭据。

- `fql/`：双rank训练指标、10次完整评估摘要、完整恢复点/最佳点回执、训练日志、128条轨迹小数组审计、成功率与Q图、点表。
- `queue/`：旧MFD延长和三v2配置、完整状态检查、准备/登记/等待状态、测试报告。
- `retire-fql-fixed.txt`：精确FQL进程停止和Direct接回回执。
- `register-append.txt`、`prepare-append.txt`、`check-append.txt`：准备、队列登记、32项CPU检查。
- `final-state.json`：21:57队列存活、Direct1320评估中、RLT四路等待、Home/Data余量。

三v2各200预算轮、17次评估、每点128回合。新保存策略只作用于各自输出目录；日志/曲线全留，旧点不清理。

[执行说明](../../small_flow_distillation/23_V2_FORMAL_QUEUE_20261028.md) · [FQL诊断与修复规划](../../small_flow_distillation/24_SINGLE_FQL_DIAGNOSIS_20261028.md)
