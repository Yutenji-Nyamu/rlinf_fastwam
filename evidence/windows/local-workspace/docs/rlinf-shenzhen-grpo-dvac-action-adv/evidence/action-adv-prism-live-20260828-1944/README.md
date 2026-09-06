# Action-Adv / Prism live snapshot（2026-08-28 19:44 CST）

只读现场：两项训练均存活、`fatal=0`。

| 实验 | 最新完整步 | 当前进度 | raw | MA5 | MA10 | fixed-32累计 | checkpoint | 近期单步 | 粗略完成时间 |
|---|---:|---|---:|---:|---:|---:|---|---:|---|
| Action-Adv `[0,2]`，GPU4/5 | 21 | Step22 rollout 3/4 | 78.91% | 76.80% | 74.49% | 105/128 | 10,20 | 23.2 min | 08-30 02:00 CST |
| Prism-DVAC-RLOO，GPU6/7 | 47 | Step48 rollout 1/4 | 96.09% | 92.81% | 94.26% | 265/288 | 10,20,30,40 | 21.9 min | 08-29 14:55 CST |

GPU4--7现场占用约`67.6/65.5/65.3/66.0 GiB`，主机available RAM约`327.5 GiB`。ETA按近期完整step实测速度线性外推，只作调度估算。

- 图：`01_action_adv_prism_success_live.png`
- 逐步值：`success_curves.csv`
- 汇总：`summary.json`
- 原始日志：`raw/*/runtime/driver.log`
