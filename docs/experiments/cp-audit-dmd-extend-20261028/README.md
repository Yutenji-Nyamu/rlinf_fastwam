# CP审计、DMD200轮接续与同族快照

2026-10-28 16:54 CST。源码与轻量证据；不含原始服务器进程信息、凭据、权重、回放或视频。

- [CP填充维/dropout审计](../../small_flow_distillation/14_CP_PADDING_DROPOUT_AUDIT_20261028.md)
- [CP new规划](../../small_flow_distillation/15_CP_NEW_PLAN_20261028.md)：未开始训练。
- [DMD续训](../../small_flow_distillation/16_DMD_EXTEND200_RECEIPT_20261028.md)与[精简回执](queue-receipt.json)：完整恢复1300→2600，固定学习率及5:1保持。
- [原生五方法同族](native-budget.png)；[实际更新数](native.png)。12点评估均128回合，MFD当前11点，FQL未开始。
- [多任务C10](c10.png)、[早期](c10-early.png)、[历史C50](c50.png)。不同协议分图。
- [全部77个实测点](points.csv)。原生原始N5/N1为111/128、98/128；C10参考N10/N1为153/368、75/368。

1机45直接32730/55650，约47–51小时；67 MFD1260/1300，后续FQL→Direct200→CP200→DMD200→RLT，整队列约14–18小时，不含RLT。四路候补有效。2/3机认证前不可达，内部状态未知。

每项ETA含训练、评估、重载，FQL尚按既有smoke估算。折线仅连接原始点，没有平滑后重定义最佳点或虚构多种子误差带。
