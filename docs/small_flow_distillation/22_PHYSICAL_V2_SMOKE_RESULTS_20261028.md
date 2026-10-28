# CP v2 / DMD v2 / MFD v2：短测完成

2026-10-28 18:59（北京时间）核验：三版均在1机物理2/3卡完成12步训练→完整保存→恢复至14步→导出权重原生N1推理。全部通过，短测驱动已退出，2/3卡无计算或图形进程；未启动正式训练。4–7原部署和队列未修改，现场仍有原训练进程。

## 实测

|版本|每卡 / 全局batch|峰值显存/卡|短测稳态秒/更新|续训、原生N1|
|---|---:|---:|---:|---|
|CP v2|80 / 160|74.31 GiB|1.83|通过|
|DMD v2|192 / 384|72.54 GiB|5.83|通过；原batch重测|
|MFD v2|192 / 384|72.59 GiB|10.48|通过|

计时为第3–12步、双rank逐步`data_wait + observation_prefix + auxiliary_updates + student_update`之和的中位数，不含启动/保存；MFD积分已包含在student_update中，不重复相加。MFD每次抽取积分区间不同，本次指导前向4–64次，耗时会波动。

原生N1检查每rank用32个观察，实际加载导出的动作权重，输出`32×50×14`，动作专家调用1次、数值有限；老师训练时保持冻结。只验证训练与推理通路，**没有仿真成功率评估**。

## 目标与稳定性

- CP v2：一致性与专家FM都只评分14维，内部H50×32和EMA回传保留。学生梯度范数0.140–3.200，28条rank更新中2条触发既有阈值1的裁剪。损失/梯度均有限；未加入dropout。
- DMD v2：学生维持14维；辅助epsilon损失改14维。累计14次学生、70次辅助更新；学生无裁剪，辅助平均裁剪比例2.86%。
- MFD v2：学生维持14维；辅助velocity损失改14维。累计14/70次更新；学生无裁剪，辅助平均裁剪比例4.29%。完整32维积分继续保留，本次填充位置绝对最大4.58、速度最大5.28，未见非有限数值；14步不能证明长期不会漂移。

CPU先验检查40项通过；补加运行选项后，相关10项检查和Ruff再通过。旧方法默认评分、原MFD积分、严格完整恢复边界均在检查范围内。

## DMD首试与修复

首试第3步视觉前缀OOM：Torch已分配38.11 GiB、缓存未用36.77 GiB，另申请5.67 GiB失败，符合显存碎片疑点。沿用现有启动器，将**独立DMD v2**配置设为`cuda_allocator_conf=expandable_segments:True`；batch384、学习率、目标均未改，随后12→14和N1检查通过。保留首试日志及配置，未修改旧训练或全局环境。该选项的用途见[PyTorch官方文档](https://docs.pytorch.org/docs/2.14/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)。

## 交付与位置

- 实现与参数：[21](21_PHYSICAL_V2_IMPLEMENTATION_20261028.md)，源码提交`1b8e58ff9`。
- [轻量回执](../experiments/physical-v2-smoke-20261028/README.md)：配置、双rank曲线、续训/导出回执、CPU测试日志及首试失败记录；约300 KiB打包体积，不含权重/数据/视频。
- 源码部署：`/home/chenyiteng/deployment-20261028/physical-v2-sz1-g23/source`。
- 大产物：`/home/chenyiteng/results/pi05-distill/physical-v2-sz1-g23-20261028/smoke-{cp_v2,dmd_v2,mfd_v2}`；每组保留第14步完整点和导出。无正式队列接续。
- 结束时Home余245.5 GiB、Data余155.8 GiB。本轮未进行用户数据清理。

2/3机仅用于选卡前连通性定位：本地TCP通但SSH问候前断开，经1机TCP超时，未进入认证；其内部实验状态未知。
