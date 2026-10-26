# 2026-10-26 OneDP收束、新DMD smoke、FQL接续

`old-onedp/`是旧实验的配置、完整评估点、逐步指标、部分末评记录和日志尾部；最佳1113、末代8904的完整续训文件留在服务器。

`new-dmd/`含新配置、12→14续训、两个rank一步导出与阶段计时。`timing-summary.json`汇总稳态2–12步；`cpu-pair.json`为双CPU进程4/8/16解码线程测试。完整权重、原始图像和视频不在本目录。

新DMD已通过功能smoke，尚无训练后成功率结论；后续卡用于FQL真实数据采集→正式训练/评估→逐卡RLT。详细方法、参数依据和启动导入故障修复见[59号](../../pi05_distillation/59_DMD_NEW_SMOKE_AND_FQL_HANDOFF_20261026.md)。
