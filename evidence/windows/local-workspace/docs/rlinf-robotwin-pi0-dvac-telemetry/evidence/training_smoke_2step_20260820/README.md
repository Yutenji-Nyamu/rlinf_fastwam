# Idea2 2-step training smoke 轻量证据包

本目录保存AutoDL训练smoke的轻量副本；checkpoint和完整Ray运行目录只留服务器。

权威服务器run：
`/root/autodl-tmp/idea2_dvac_train_runs/idea2_dvac_apply_smoke_2step_2gpu16env_20260820`

权威runtime/log：
`/root/autodl-tmp/idea2_dvac_train_runtime/idea2_dvac_apply_smoke_2step_2gpu16env_20260820`

逐指令与问题/修复记录见上级
`../TRAINING_IMPLEMENTATION_AND_SMOKE_LEDGER.md`。服务器checkpoint与完整Ray目录未下载。

## 结果入口

- [SMOKE_VALIDATION.json](SMOKE_VALIDATION.json)：两rank、两step、shape/finite、warmup/apply、checkpoint、
  control trace的机器后检；SHA256 `02a9498e...05adb`。
- [RESOURCE_SUMMARY.json](RESOURCE_SUMMARY.json)：GPU/RAM/worker RSS与memory events；SHA256
  `3b89aa11...bd8e7`。
- [SMOKE_ANALYSIS_SUMMARY.json](SMOKE_ANALYSIS_SUMMARY.json)：第二步权重和一条success对齐摘要；SHA256
  `cb9c0774...32266`。
- [SMOKE_DVAC_WEIGHT_AND_CONTROL_ALIGNMENT.png](SMOKE_DVAC_WEIGHT_AND_CONTROL_ALIGNMENT.png)：
  512-query `h→weight`分布与reset57 q2曲线；SHA256 `2688c426...2b226`。
- [CONTROL_TRACE_CONTACT_SHEET.png](CONTROL_TRACE_CONTACT_SHEET.png)：111帧视频每10帧抽样；SHA256
  `24c7be31...7fdda`。
- [CONTROL_TRACE_RESET57.mp4](CONTROL_TRACE_RESET57.mp4)：H.264/yuv420p、160×120、10 FPS、111帧；
  SHA256 `6f08555a...ce586`。

## 配置、日志与原始轻量数据

- [RESOLVED_CONFIG.yaml](RESOLVED_CONFIG.yaml)：真实resolved config，SHA256
  `5633cc32b787528b8b6a12ec5d4eee27be732f7312df1f0135b114187f2c324e`；
- [LAUNCH_COMMAND.txt](LAUNCH_COMMAND.txt)、[DRIVER.log](DRIVER.log)、[TRAIN_METRICS.log](TRAIN_METRICS.log)；
- [RESOURCES.csv](RESOURCES.csv)、[PROCESS_RSS.tsv](PROCESS_RSS.tsv)、[OBSERVER_EXIT.txt](OBSERVER_EXIT.txt)；
- `ACTOR_RANK{00,01}_STEP{0000,0001}.npz`：四个rank/step原始shard，共931,890 bytes；
- `ACTOR_RANK{00,01}_STEP_METRICS.csv`、manifest与rolling-stats state；
- [CONTROL_TRACE_RESET57_FRAMES.csv](CONTROL_TRACE_RESET57_FRAMES.csv)和
  [CONTROL_TRACE_RESET57_METADATA.json](CONTROL_TRACE_RESET57_METADATA.json)；
- [WEIGHT_BY_H.csv](WEIGHT_BY_H.csv)：第二步逐`h`分位数与`log10 V_L3`中位数。

本目录26个smoke后新增/下载文件合计3,286,715 bytes；加先前MP4/CSV/metadata与README约3.32 MB。
每个文件的完整hash已在实施账S015中逐项记录；关键入口hash也列于上方。
