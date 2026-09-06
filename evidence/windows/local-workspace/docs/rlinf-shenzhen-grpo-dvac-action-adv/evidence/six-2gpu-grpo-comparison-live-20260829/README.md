# 深圳两卡 GRPO 家族统一曲线（2026-08-29）

主图：[`01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png`](01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png)

统一口径：SZ-H100、两卡、`64 train env × 4 rollout epochs = 256 trajectories/step`、G8、
`B1024/MB32/update2`、fixed-32/eval5。横轴统一到最长 Control 的 Step 96；每条短实验只画到自己的
真实终点或本次现场快照，不外推补线。

| 实验 | 真实终点/快照 | 最新 raw | MA5 | MA10 | fixed-32 累计 | 同步 Control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GRPO Control | 96 | 92.97% | 94.69% | 94.73% | 570/608 | — |
| ST-DVAC `[0,2]` | 52 | 94.53% | 90.31% | 89.34% | 296/320 | 294/320 |
| Prism-RLOO | 55 | 93.36% | 95.08% | 93.13% | 326/352 | 325/352 |
| 旧 Action-Adv（H-mean尺度缺陷） | 29 | 80.08% | 84.84% | 81.13% | 133/160 | 144/160 |
| Action-Adv Fix `[0,2]` | 27 live | 90.63% | 93.75% | 92.81% | 148/160 | 144/160 |
| ST-DVAC `[0.5,1.5]` | 27 live | 89.84% | 89.38% | 90.04% | 143/160 | 144/160 |

颜色与线型双编码：黑=Control、橙=ST `[0,2]`、绿=Prism、灰=旧 Action、洋红=Action Fix、
蓝=ST `[0.5,1.5]`。旧 Action 灰色显示，避免把已确认的 loss-reduction 尺度缺陷误当成有效方法终态。

未纳入：fixed-64 首次评估失败的两卡 v1、Prism v1 checkpoint 卡住版本、所有 smoke、深圳四卡
`[0,2]/[0,5]` 与 AutoDL 实验；它们是失败重跑或预算/机器不同，不能作为额外可比曲线重复计入。

可复现输入与输出：

- `raw/`：本次只读下载的两条当前 driver 日志；
- `curves.csv`：六条 raw/MA5/MA10/fixed-32 对齐表；
- `summary.json`：终点摘要；
- 制图脚本：`C:/Users/86136/Documents/rl/local_scripts/render_sz_six_2gpu_grpo_comparison_20260829.py`。
