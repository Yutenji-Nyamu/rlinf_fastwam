# π0.5 BC dvac-new 4/U10 收尾

按授权于2026-09-10 20:11 CST停止，最后完整训练指标R49；不是完成100轮。wrapper退出码134，目标namespace与GPU6/7已释放。最后现存checkpoint R40；本轮没有删除权重或成功池。

采集MA5 / MA10：25.00% / 45.00%；最新fixed R45：31.25%。

[单列对照图](comparison/index.html)包含每轮、MA5/10、fixed及累计采集，另含GRPO20:07快照。旧BC两组长度过滤off、旧指令环境；新8/U5配对使用修复环境与长度≤3，不混入本包曲线。

raw/runtime含日志、命令、实配、合同和退出记录；raw/tensorboard含原始event；scalars.json/csv是全量训练指标；source及source-lock记录经过校验的源码。checkpoint-inventory仅列元数据。大模型、optimizer、成功数据与视频留在服务器，本ZIP是分析包。

输出：`/home/chenyiteng/results/rlinf-shenzhen/online-bc/pi05-bc-dvac-new-4u10-lenoff-gpu7-formal100-20260910-v1`；源码HEAD `e185219a9856521fcc705a96c9fd4d5eb5252b37`。
