# 2026-11-05 DSRL诊断证据

- `metrics.json`：四组实际TensorBoard标量，重复step取较晚wall_time；没有权重、图像、回放或凭据。
- `summary.json`：15:22动态快照、成绩和最近一次CPU探针的检查点计数。两者采样时间不同，不把检查点计数冒充实时值。
- `series.json` / `index.html`：独立评估、训练20轮平均、Q评分；可勾选曲线，不做插值推断。
- `cpu-probe.json`：每组取最近2048条中均匀256条，保存点BF16权重加载到FP32小网络，检查Q、噪声分布及优化器类型。不是在线策略表现测试。
- `precision-probe.json`：保存Adam状态估算更新舍入，仅作初筛。
- `optimizer-ab.json`：实际一次CPU Q更新。64条真实回放、同一份量化梯度、相同保存Adam历史，比较BF16和FP32；模型与优化器副本全部丢弃。
- `dsrl_*1105.py`：对应只读CPU探针。`CUDA_VISIBLE_DEVICES=''`，Torch两线程；不运行大模型、不创建GPU上下文、不写回保存点。
- `source-manifest.json`：本轮实际读到的服务器源码指纹，主干文件位置固定；将来检查点滚动清理后不能保证脚本仍可直接重跑。

研究解释见[诊断](../../22_DSRL_PLATEAU_AUDIT_20261105.md)和[外部依据](../../23_DSRL_EXTERNAL_EVIDENCE_20261105.md)。本轮未改变训练实现。
