# 三个真实动作维度v2短测回执

2026-10-28，1机物理2/3卡，三版各12→14步、原生N1导出检查通过后退出；[汇总说明](../../small_flow_distillation/22_PHYSICAL_V2_SMOKE_RESULTS_20261028.md)。未运行仿真成功率评估。

- `deployment/final-report.json`：耗时、显存、梯度、最终GPU释放回执。
- `deployment/v2-smoke-summary.json`、`smoke-*-accepted.json`：三方法最终验收。
- `cp_v2/`、`dmd_v2/`、`mfd_v2/`：双rank逐步指标、实际配置、checkpoint与原生推理回执。
- `deployment/*_v2.json`：本次可复用配置；大输出路径仍在服务器Home。
- `failure-dmd-nativeallocator/`：DMD首试OOM及原配置；最终DMD用expandable_segments、batch未改。
- CPU检查40项通过，运行选项补充后相关10项通过。首轮驱动的失败总状态是历史记录，最终以`v2-smoke-summary.json`为准。

只保存轻量实验材料。权重、数据、完整视频及环境凭据均不在此目录。
