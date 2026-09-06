# RLT / DSRL formal 轻量现场证据（2026-08-24）

最终分析口径：RLT Stage 2 v4 图表冻结到完整 Step 85；DSRL v2 完整 Step 200/200、exit code 0。

目录包含：

- 原始 `driver.log`、TensorBoard event、`resource.csv`、resolved YAML 与 command；
- RLT `step1_85`、DSRL `step1_200` 派生指标，以及真实 fixed 点 CSV；
- 四张静态 PNG 和机器可读 `summary.json`。

`step1_74` / `step1_196` 是同轮稍早只读快照，保留用于证明刷新演进；最终结论只读
`step1_85` / `step1_200`。没有下载 checkpoint、replay payload、视频或完整 Ray log。

主解释见
[`../../11_RLT_DSRL_FORMAL_ARTIFACT_AND_METRIC_REFRESH_20260824.md`](../../11_RLT_DSRL_FORMAL_ARTIFACT_AND_METRIC_REFRESH_20260824.md)。
