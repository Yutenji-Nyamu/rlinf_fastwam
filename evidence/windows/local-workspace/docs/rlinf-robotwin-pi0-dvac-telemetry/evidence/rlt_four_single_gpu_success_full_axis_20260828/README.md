# 四条单卡 RLT 成功率同轴图

现场/日志口径：2026-08-28 21:42 CST。横轴统一使用最长真实 run 的 Step 1--476；没有把仍在运行的
Pure 曲线外推到 476。

## 四条实验分别做了什么

| 图中名称 | 相对干净 RLT 的方法变化 | 当前数据终点 |
|---|---|---:|
| Clean single-GPU RLT | 无 DVAC；原 actor-Q、reference-BC、critic、replay 与 schedule | 476，已停止 |
| Old DVAC-BC | 成功 episode 的 BC target 改为 executed action，并用 mean-one DVAC 在 C10 内重分配，strength=.25 | 476，已停止 |
| Pure02 | BC target 仍为原 π0 reference；仅成功 episode 内用 mean-one DVAC 重分配 reference-BC，strength=.5 | 296，运行中 |
| Pure05 | 与 Pure02 相同，但 strength=2.0，重分配更强 | 291，运行中 |

`Pure02/Pure05`是用户侧简称；mean-one 归一后实际最终权重不严格固定在 `[0,2]` / `[0,5]`。

更早还有一组`micro128 / warmup10k / replay50k`的短单卡pair，停在Step129/128；它随后被上表中
`micro256 / warmup20k / replay80k`的matched-width版本替代，所以没有混进本图。

## 共同 Step 291 的平滑成功率

| run | MA5 | MA10 | MA20 |
|---|---:|---:|---:|
| Clean RLT | 72.50% | 76.25% | 75.00% |
| Old DVAC-BC | 75.00% | 73.75% | 73.75% |
| Pure02 | 87.50% | 85.00% | 83.13% |
| Pure05 | 92.50% | 83.75% | 80.63% |

这些是 on-policy training-rollout success；fixed-ID evaluation 是另一条证据轴。

## 文件

- `RLT_FOUR_SINGLE_GPU_SUCCESS_FULL_AXIS_G476.png`：raw、MA5、MA10、MA20 四联图。
- `success_curves.csv`：四条逐 step 曲线；尚未到达的 Pure 后段留空。
- `summary.json`：各 run 真实终点和共同 Step 291 摘要。
- 复现脚本：`tmp/build_rlt_four_single_gpu_success_full_axis_20260828.py`。
