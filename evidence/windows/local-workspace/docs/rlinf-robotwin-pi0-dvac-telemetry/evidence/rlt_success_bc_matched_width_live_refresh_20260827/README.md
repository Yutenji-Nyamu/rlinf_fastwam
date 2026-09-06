# RLT matched-width 当前快照

时间：2026-08-27 20:07:45 CST  
范围：只读刷新；未修改或停止训练。

## 当前状态

| 项目 | Control | success-episode DVAC-BC |
|---|---:|---:|
| 完整 Step | 434/480 | 438/480 |
| 累计 train success | 52.19% | 53.20% |
| 最近 5 Step | 95.0% | 85.0% |
| 最近 10 Step | 95.0% | 87.5% |
| Step425 fixed20 | 19/20 | 17/20 |
| ETA | 约 3h21m | 约 3h02m |

两条 wrapper 均 alive；fatal、CUDA OOM、worker crash、NCCL error、cgroup OOM/OOM-kill 均为 0。
方法权重 `p05/mean/p95=0.746/1.000/1.252`，ESS=`0.977`。

资源：RAM 当前/峰值约 `230.81/236.40 GiB`；GPU0/GPU1 显存峰值约
`25.17/25.14 GiB`。

## 图与数据

- `RLT_CONTROL_VS_DVAC_BC_SUCCESS_THROUGH_G433.png`：共同完整 Step 1--433 的逐步、5步和10步训练 rollout success。
- `success_curves.csv`：图中逐步数据。
- `summary.json`：共同 Step 433 的聚合结果与最近 fixed20。
- `raw/`：本次只读下载的训练指标与资源 CSV。

当前累计训练 success 略高于 control，但最近窗口和最近两次 fixed20 均低于 control；因此仍不能据此判断方法最终优于 control，等两边完成 480 后再做同一 fixed-ID 评估。
