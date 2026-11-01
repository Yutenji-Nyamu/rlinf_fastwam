# Direct＋GAD-OT：2026-11-01轻量记录

原生adjust_bottle π0.5单任务，GAD K4＋同观察整块均匀平衡OT。这里保存参数与短测/排队证据；正式训练尚在等待前驱。

- [实现与参数依据](../../gad_ot/04_IMPLEMENTATION_20261101.md)
- [通俗结果、队列与ETA](../../gad_ot/05_SMOKE_QUEUE_20261101.md)
- [正式配置](config/formal-config.json)：400预算轮/4400更新、每卡224、27评×128。
- [CPU检查](smoke/cpu-checks.json)：13项通过。
- [smoke总结](smoke/gad-ot-smoke-summary.json)：12→14续训、N1导出、显存与耗时；没有闭环成功率。
- [两rank原始指标](smoke/output/metrics-rank0.jsonl)，另一份同目录rank1；保留GAD信号、匹配前后MSE、重配率、梯度、学习率及阶段时间。
- [正式队列登记](queue/registered.json)、[现场执行回执](config/execution-receipt.json)。

`smoke/`内还保留两段原始小日志、两rank原生N1对齐及导出检查、临时权重清理回执。`queue/`保留正式计划及接管前RLT等待目标，便于后续检查队列。源码在同一提交的`rlinf/algorithms/pi05_distillation/ot.py`等文件。

排除模型权重、回放、视频、SSH凭据。临时smoke权重已经删除；正式从原始模型开始，不依赖这些权重。
