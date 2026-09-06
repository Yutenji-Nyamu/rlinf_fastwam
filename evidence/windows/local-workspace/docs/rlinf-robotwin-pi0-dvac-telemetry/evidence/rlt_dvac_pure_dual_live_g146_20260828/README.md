# RLT-DVAC-Pure 双实验现场：共同 Step 146

只读快照时间：2026-08-28 10:31 CST。

| 训练rollout success | `[0,2]`简称 / s0p5 | `[0,5]`简称 / s2p0 |
|---|---:|---:|
| 日志最新完整步 | 147 | 146 |
| 最新逐步 | 25.00% | 12.50% |
| 最新5步均值 | 12.50% | 7.50% |
| 最新10步均值 | 15.00% | 10.00% |
| 共同Step1--146累计均值 | 15.33% | 15.15% |

共同Step146处，两者逐步均为12.50%；s0p5的5步/10步为15.00%/16.25%，s2p0为
7.50%/10.00%。每步只有8条episode，因此逐步曲线以12.5个百分点跳动，滑动均值更适合看局部走势。

到Step125为止，两条fixed-20评估均为0/20。最新`actor_switch_rate=0`，训练rollout仍由π0 reference
route采集；因此当前训练rollout曲线还不是student闭环行为效果。DVAC-BC已经在更新student，下一段最有信息量的
是student接管后的训练曲线和Step150/175 fixed-20。

方法强度已明显分开：共同Step146的s0p5权重p05/mean/p95=`0.460/1.000/1.523`、ESS=`0.908`；
s2p0为`0/1.000/2.554`、ESS=`0.586`。这说明两条方法都在生效，且s2p0的C10权重重分配明显更集中。

两条driver均alive；`oom=0`、`oom_kill=0`。现场RAM约186.24 GiB，采样历史峰值190.29 GiB；GPU0/1
现场约16.75/24.64 GiB，历史峰值25.12/25.09 GiB。

- [成功率图](RLT_DVAC_PURE_SUCCESS_THROUGH_G146.png)
- [逐步与滑动均值CSV](success_curves.csv)
- [机器可读摘要](summary.json)

