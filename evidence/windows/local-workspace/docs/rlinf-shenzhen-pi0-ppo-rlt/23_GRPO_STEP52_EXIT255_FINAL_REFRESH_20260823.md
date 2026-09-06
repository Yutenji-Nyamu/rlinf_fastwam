# 深圳 GRPO v2：完整 Step52 后 exit255 的最终现场刷新

现场时间：2026-08-23 20:18–20:49 CST  
操作边界：普通/管理员账号只读；没有停止、重启、改配置、删产物或控制进程。

## 1. 最终训练状态

GRPO v2 已不存活。最新完整训练步为 `52/100`；Step53只出现 rollout `0%`，没有形成训练指标。

| 项 | Step51 | Step52 |
|---|---:|---:|
| train success | `486/512 = 94.92%` | `474/512 = 92.58%` |
| KL | `0.0096` | `0.016` |
| clip fraction | `0.048` | `0.084` |
| grad norm | `15.118` | `15.817` |
| Step52 wall time | — | `1347.2s` |

最近一次完整 fixed-64 是Step50：`62/64 = 96.875%`；Step40为`63/64 = 98.4375%`。最后已确认的
checkpoint为global step50。Step52训练数值仍有限，没有出现发散形态。

全Step1--52已重新提取。训练success全程均值为`89.63%`，最后10步`92.93%`、最后5步`92.66%`；
fixed-64依次为Step10/20/30/40/50的`89.06/93.75/96.88/98.44/96.88%`。同预算PPO只保留到
Step46；在共同Step1--46上，GRPO/PPO训练success均值为`89.17/87.36%`，仅作运行曲线比较，不作
算法显著性结论。

## 2. exit255 已精确定因

完整下载后的`driver.log`尾部明确记录：Step53的4/4 rollout完成后，Ray报告节点内存
`1914.32/2015.51 GB = 94.9793%`，实际用量`2056097792000 B`超过95%阈值
`2055933788160 B`，随后按策略杀4个worker，其中一个被选中的EnvWorker为`438.57 GB`，并打印：

```text
Exiting main process due to a failure upon worker execution.
```

四个EnvWorker当时分别约`528.25/438.57/430.64/406.83 GB`。所以wrapper记录255的链路是：

```text
GRPO EnvWorker主存持续增长
  -> Ray userspace memory monitor越过95%阈值
  -> Ray主动杀4个worker
  -> main process因worker failure退出
  -> wrapper记录255
```

这不是训练数值错误、kernel/cgroup OOM、SSH timeout或SIGKILL/SIGTERM。kernel/cgroup OOM=0并不矛盾：
Ray是在内核OOM前先主动杀worker。原“日志无异常、原因未明”结论由本次完整driver尾部证据取代。

resource observer也闭合了时序：11:58--12:01 UTC，cgroup从`1920.021`升到`1929.455 GiB`，host
available从`123.152`降到`104.192 GiB`；12:01:49左右Ray kill、12:01:56 wrapper exit，12:02:03
available恢复到`1986.929 GiB`。

RLT/DSRL不能解释为代码或组件混淆：三棵独立clean worktree，没有第二个Ray session、模型、RoboTwin或
GPU检查；两个较重CPU检查scope分别在kill前约156秒和56秒已deactivated，无残留。GRPO自身四个
EnvWorker已合计约1.8 TB，并在最后3分钟再增长9.43 GiB。工程上以后在available逼近100 GiB时仍应停止
其他CPU检查，但它们不是本次直接原因。

## 3. 退出后的服务器状态

| 项 | 20:20 CST现场 |
|---|---:|
| GPU0–7 | 全部0 MiB、0% util、无用户compute process |
| RAM | 约2.0 TiB；available约1.9 TiB |
| swap / PSI | 仅用19 MiB / 6 GiB；memory PSI全0 |
| `/` | 余234 GiB，使用18% |
| `/home` | 余2.2 TiB，使用5% |
| `/data` | 余3.0 TiB，使用11% |
| 其他用户 | 无GPU或CPU/RAM重任务；未见异常占盘 |
| 网络 | Mihomo active/7890；代理GitHub/HF均HTTP200；GitHub直连200、HF直连timeout |

服务器健康、卡和主存已经释放。RLT/DSRL current-base两条分支也已push且local/remote `0/0`、worktree
clean；这只说明现在具备准备smoke packet的资源条件，不等于已授权真实smoke或GRPO重启。

管理员安全读取到的供应侧配额已刷新为`500 GiB total / 0.001 GiB used / 499.999 GiB remaining`，到期
`2026-11-23 20:44:40 CST`。Mihomo没有重启、config mtime未变；此前SSL EOF属于订阅拉取/节点瞬态，
不是流量耗尽。

## 4. 相关入口

- [RLT/DSRL实现结果与配置建议](../rlinf-shenzhen-rlt-dsrl-port/04_IMPLEMENTATION_RESULT_AND_CONFIG_RECOMMENDATION_20260823.md)
- [RLT/DSRL smoke前数据、资源与本次退出判断](../rlinf-shenzhen-rlt-dsrl-port/05_PRE_SMOKE_DATA_RESOURCE_AND_GRPO_EXIT_DECISION_20260823.md)
- [RLT/DSRL细粒度实施账](../rlinf-shenzhen-rlt-dsrl-port/evidence/IMPLEMENTATION_LEDGER.md)
- [此前Step42完整曲线与资源图](22_GRPO_STEP42_LIVE_METRICS_AND_RESOURCE_REFRESH_20260823.md)
- [Step1--52 success对比图](evidence/grpo_v2_final_step52_20260823/grpo_vs_ppo_success_step1_52.png)
- [Step1--52优化与墙钟图](evidence/grpo_v2_final_step52_20260823/grpo_vs_ppo_optimization_timing_step1_52.png)
- [至Ray退出的资源图](evidence/grpo_v2_final_step52_20260823/grpo_resource_timeline_through_exit.png)
- [机器可读终态摘要](evidence/grpo_v2_final_step52_20260823/summary_step52.json)
- [轻量可携带ZIP](../../exports/shenzhen_grpo_v2_step52_light_evidence_20260823.zip)：464,603 bytes，
  20成员、展开1,269,859 bytes；含原始日志/配置、两个CSV、三张图、两份说明与绘图脚本，不含checkpoint、
  视频或大Ray日志。
