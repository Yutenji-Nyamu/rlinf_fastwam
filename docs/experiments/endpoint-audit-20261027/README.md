# 直接蒸馏收敛审计：轻量证据

2026-10-27，标量截至17760步；最新正式评估11130步。

- [现场与实现审计](../../pi05_distillation/67_ENDPOINT_CONVERGENCE_AUDIT_20261027.md)
- [文献与判断](../../pi05_distillation/68_ENDPOINT_LITERATURE_20261027.md)
- [评估点](evaluation-points.csv)、[逐任务比较](task-comparison.csv)、[训练区间](training-bins.csv)、[观察覆盖](data-coverage.csv)、[精简JSON](public-evidence.json)
- [本地交互网页](http://127.0.0.1:8194/endpoint_audit1027/)

原始两rank日志、部署源码、CPU检查点采样回执留本地；此目录不包含权重、视频、凭据、主机/进程/网络资料。未新增GPU工作、未修改训练。梯度统计来自约每10步一次的记录，Adam更新比例来自CPU坐标抽样，均非逐步完整张量分析。4个缺测任务没有计入368有效回合的分母。
