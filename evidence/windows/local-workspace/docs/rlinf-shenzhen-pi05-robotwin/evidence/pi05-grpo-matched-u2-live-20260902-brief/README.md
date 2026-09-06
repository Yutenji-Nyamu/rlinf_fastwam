# π0.5 matched-update 双实验现场快照

> 采集：2026-09-02 08:34 CST；只读下载日志、resolved与resource CSV，未修改服务器。

| 指标 | Control GPU4/5 | DVAC Action-Adv `[0.5,1.5]` GPU6/7 |
|---|---:|---:|
| 完整step | 27 | 25 |
| 最新raw | 83.59% | 88.67% |
| 5步均值 | 85.00% | 83.36% |
| 10步均值 | 86.84% | 85.27% |
| fixed32累计 | 153/160=95.63% | 142/160=88.75% |

- 两个wrapper存活，`RLinf` / `RLinf_1`各15 actors；fatal和exit marker均为0。
- 配对Step1--25的DVAC-Control raw差为：全程`-3.28 pp`、末5步`-4.06 pp`、末10步`-2.97 pp`。当前没有观察到DVAC领先。
- Control前11步KL/clip均值为`0.0211/0.0925`；旧`GB512/update5`对应为`0.0984/0.233`。`GB1024/update2`已显著降低单轮策略移动强度。
- DVAC最新`warmup=0`、weight mean=`0.997`、ESS=`0.959`，方法实际生效，不是权重关闭。
- 当前GPU约63.7--70.5 GiB/卡；历史单卡峰值Control/DVAC约79.5/80.8 GiB，无OOM。
- 主机available约417 GiB；末1/3/6小时下降约42/61/57 GiB每小时。若该趋势持续，预计早于Step100触到Ray约100 GiB available阈值；本轮未做干预。

主图：[`01_pi05_matched_u2_control_vs_dvac.png`](01_pi05_matched_u2_control_vs_dvac.png)。原始曲线与摘要：[`curves.csv`](curves.csv)、[`summary.json`](summary.json)。
