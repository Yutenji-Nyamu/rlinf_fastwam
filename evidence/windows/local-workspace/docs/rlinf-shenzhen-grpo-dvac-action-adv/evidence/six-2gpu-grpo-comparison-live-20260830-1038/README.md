# 深圳两卡 GRPO 家族现场（2026-08-30 10:38 CST）

主图：[`01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png`](01_all_six_2gpu_grpo_raw_ma5_ma10_fixed32.png)

| 当前实验 | 终点/最新完整 step | raw | MA5 | MA10 | fixed32 累计 | 运行状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Action-Adv Fix `[0,2]` | 79 | 87.11% | 91.41% | 92.77% | 449/480 | alive，15 actors，fatal=0 |
| ST-DVAC `[0.5,1.5]` | 53 | 93.75% | 93.83% | 92.07% | 294/320 | exit255，已停止 |

ST-DVAC 不是数值崩溃。2026-08-29 23:05:58 CST，Ray 检测到节点内存
`1926.61/2015.51 GB = 95.5891%`，按内存保护策略杀死四个 job `74000000` worker；其中一个
EnvWorker 当时约 473.19 GB。后续 Ray actor death 与 Gloo 断连均为连锁结果。Action-Adv Fix
未被杀，继续推进至 Step79；其资源记录在本次刷新时显示约 916 GiB host available。

复现输入：`raw/`；对齐曲线：`curves.csv`；终点摘要：`summary.json`；下载清单：
`download_manifest.json`。制图脚本为
`C:/Users/86136/Documents/rl/local_scripts/render_sz_six_2gpu_grpo_comparison_20260829.py`。
