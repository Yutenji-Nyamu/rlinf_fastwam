# 深圳两卡 GRPO 家族现场（2026-08-29 18:29 CST）

主图：[`01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png`](01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png)

两条当前实验均正常：wrapper存活、各15个Ray named actors、fatal=0，并均已越过Step40 checkpoint。

| 当前实验 | 完整step | raw | MA5 | MA10 | fixed32累计 | 同步Control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Action-Adv Fix `[0,2]` | 42 | 88.67% | 92.03% | 92.30% | 236/256 | 233/256 |
| ST-DVAC `[0.5,1.5]` | 43 | 90.23% | 91.95% | 92.15% | 232/256 | 233/256 |

按各自共同step窗口，Action Fix相对Control的累计/末5/末10训练差为
`+5.75/+1.80/+3.59 pp`；ST `[0.5,1.5]`为`+4.61/+1.80/+2.38 pp`。
训练曲线目前都高于Control同期，但fixed32只分别为`+3/-1`个成功episode，尚不足以宣称稳定评估提升。

服务器同刻：GPU0--3空闲；Action使用4/5约`63.5/64.3 GiB`，ST使用6/7约`56.8/56.6 GiB`。
MemAvailable约302 GiB，较上午Step26/27现场约482 GiB继续下降；尚未触及Ray 95%阈值，但主存是当前
唯一明显风险。`/`、`/home`、`/data`分别余226 GiB、1.4 TiB、1.8 TiB；failed unit为0，GitHub/HF
代理均HTTP200，其他用户无GPU任务。

复现输入：`raw/`；对齐数据：`curves.csv`；终点摘要：`summary.json`。制图脚本为
`C:/Users/86136/Documents/rl/local_scripts/render_sz_six_2gpu_grpo_comparison_20260829.py`。
