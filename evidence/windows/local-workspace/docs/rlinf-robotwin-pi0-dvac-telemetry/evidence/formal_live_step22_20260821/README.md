# Formal live snapshot through Global Step 23

这份轻量快照在2026-08-21约10:03–10:09下载。目录名沿用开始检查时的`step22`，但下载完成时
`metrics.log`和两rank `runner_step=22` shard均已齐全，因此完整分析口径是UI Global Step 23。

- `run/`：resolved config、metrics、两rank DVAC artifacts、唯一control trace；
- `runtime/`：截至约10:05的driver/resource/process观察文件；
- `analysis/`：训练、历史GRPO对照、DVAC方法、资源和录像对齐图表；
- 大checkpoint未下载；服务器当时已有global step 10/20，各约9.7 GiB。

主结论见[专题现场分析](../../07_FORMAL_TRAINING_LIVE_ANALYSIS_STEP24_20260821.md)。
