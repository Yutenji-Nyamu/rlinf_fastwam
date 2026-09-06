# 单卡 RLT control 与旧 DVAC-BC：停止态高信息包

## 终点

- 用户于 `2026-08-27 23:06:39 +08:00`授权停止；两条独立wrapper与Ray head均已退出。
- 两条日志共同完整到 Step 476；两边最新完整checkpoint均为 Step 475，checkpoint正文仍留在服务器，不收入本轻量包。
- fatal、CUDA OOM、worker crash、NCCL error与kernel OOM-kill均为0。

## 主要结果

| 指标（截至共同Step 476） | Control | 旧 DVAC-BC | 方法减Control |
|---|---:|---:|---:|
| train success累计均值 | 55.49% | 55.75% | +0.26 pp |
| 最近5步均值 | 87.50% | 92.50% | +5.00 pp |
| 最近10步均值 | 90.00% | 88.75% | -1.25 pp |
| Step475 fixed-20 | 17/20 | 15/20 | -2/20 |

旧方法末步权重 `p05/mean/p95=0.747/1.000/1.251`，ESS=`0.977`。整体曲线频繁交叉，且最后fixed-20为control更高；这批证据不支持旧方法形成稳定收益。

注意：这里的“旧 DVAC-BC”是 `成功episode改用executed target + DVAC重分配`。它不是随后开发的 Pure 版；Pure版保留原始 π0 reference target。

## 资源

- 运行约34.7小时；两任务合计RAM峰值约239.84 GiB。
- GPU0/GPU1显存峰值约25.17/25.14 GiB。
- 停止并清理旧pair后，两卡均回到0 MiB，cgroup约56.6 GiB。

## 从哪里看

- `analysis/RLT_HISTORICAL_CONTROL_DVAC_BC_THROUGH_G476.png`：raw、5步、10步和fixed-20总览。
- `analysis/RLT_CONTROL_VS_SUCCESS_BC_STOPPED_FINAL.png`：成功率、权重与资源。
- `analysis/comparison_summary.json`、`diagnostics_summary.json`：数值摘要。
- `control/metrics.log`、`method/metrics.log`：完整训练指标日志。
- `method/latest_complete_trace.npz`：最后一份轻量DVAC trace。
- `pair/paired_resources.csv`：5秒资源记录。
- `control|method/resolved.yaml`与`exact_command.txt`：实际参数与命令。
