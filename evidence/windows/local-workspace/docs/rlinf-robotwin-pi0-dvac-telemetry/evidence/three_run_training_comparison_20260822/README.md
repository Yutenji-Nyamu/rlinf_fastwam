# 三次GRPO训练同轴对照（截至R-only g23快照）

这组图把三次训练放到同一`Global Step`横轴：

- 历史原始GRPO：100步；
- DVAC v1 global-z `[0.8,1.2]`：主动停止于g54；
- DVAC v2 R-only `[0.5,1.2]`：本快照到g23。

`THREE_RUN_SUCCESS_COMPARISON_G23.png`包含每步训练rollout success、5步滑动平均，以及两个DVAC
run相对同step原始GRPO的5步均值差。它们是on-policy训练数据，不是fixed-ID独立评估。

`THREE_RUN_OPTIMIZATION_COMPARISON_G23.png`包含approx KL、query级PPO clip fraction、global clip前
梯度范数和单步墙钟时间。三条线使用相同纵轴和Global Step语义。

精确逐步数据在`THREE_RUN_METRICS_G23.csv`，汇总数值在`THREE_RUN_SUMMARY_G23.json`，复现脚本为
`build_three_run_comparison.py`。脚本只读取现有本地证据，不访问服务器，也不修改训练。

当前快照的描述性汇总：

| run | 覆盖步数 | success均值 | latest success | latest rolling-5 |
|---|---:|---:|---:|---:|
| 原始GRPO | 100 | 91.71% | 98.44% | 95.94% |
| DVAC v1 | 54 | 86.66% | 88.67% | 90.31% |
| DVAC v2 | 23 | 82.52% | 86.72% | 84.30% |

不能用不同训练长度的全程均值直接判定方法优劣；正式比较应在共同checkpoint步数上使用同一批fixed reset
IDs。这里的用途是看训练轨迹与优化行为，不是替代checkpoint评估。
