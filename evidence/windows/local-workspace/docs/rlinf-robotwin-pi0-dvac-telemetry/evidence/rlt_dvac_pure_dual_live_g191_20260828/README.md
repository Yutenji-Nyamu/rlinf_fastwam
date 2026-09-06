# RLT-DVAC-Pure 双实验现场：共同 Step 191

只读快照时间：2026-08-28 14:25 CST。

| 训练rollout success | s0p5 | s2p0 |
|---|---:|---:|
| Step191逐步 | 75.00% | 62.50% |
| 最新5步均值 | 65.00% | 57.50% |
| 最新10步均值 | 51.25% | 51.25% |
| Step1--191累计均值 | 16.56% | 18.59% |
| Step150 fixed20 | 2/20 | 3/20 |
| Step175 fixed20 | 5/20 | 7/20 |

两条都在Step180后快速上升。最新5步s0p5高7.5个百分点，10步相同；较长累计轴和两次最新fixed20则
s2p0较高。当前不能概括为某条稳定领先，下一处高信息量点是Step200 fixed20及其后的student闭环段。

student接管ramp已经开始：Step191 actor switch rate约为`s0p5=.205`、`s2p0=.193`。方法强度仍稳定分开：
s0p5权重p05/mean/p95=`.463/1/1.518`、ESS=`.909`；s2p0为`0/1/2.551`、ESS=`.587`。

两条driver和六个核心worker均alive，Step192 rollout进行中；`oom=0`、`oom_kill=0`。现场RAM约
209.83 GiB，采样峰值211.41 GiB；GPU0/1现场约24.67/24.64 GiB，历史峰值25.17/25.14 GiB。
Step191日志ETA约21小时16分。

- [成功率图](RLT_DVAC_PURE_SUCCESS_THROUGH_G191.png)
- [逐步与滑动均值CSV](success_curves.csv)
- [机器可读摘要](summary.json)
