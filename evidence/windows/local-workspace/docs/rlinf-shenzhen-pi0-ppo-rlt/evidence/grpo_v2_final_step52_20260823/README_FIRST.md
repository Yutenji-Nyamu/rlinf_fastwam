# 深圳 GRPO v2 Step52 轻量证据说明

这是 `grpo-formal100-current-4gpu128train64eval-ppo-matched-v2` 的轻量终态快照；不含checkpoint、视频、
完整Ray日志或约87 GiB运行目录。

## 先看什么

1. `grpo_vs_ppo_success_step1_52.png`：逐步success、5步均值和fixed-64。
2. `grpo_resource_timeline_through_exit.png`：主存逼近Ray 95%阈值、GPU内存与利用率。
3. `grpo_vs_ppo_optimization_timing_step1_52.png`：KL、clip、grad norm与墙钟。
4. `summary_step52.json`：机器可读汇总。
5. `RAY_MEMORY_EXIT_EXCERPT.txt`：exit255的最小原始证据。

## 终态

- 完整Step1--52；Step53完成4/4 rollout，但没有完成optimizer step。
- Step52 train success=`0.92578125`；Step50 fixed64=`62/64`。
- Ray看到节点用量`1914.32/2015.51 GB`越过95%阈值，主动杀4个worker；main process随worker failure
  退出，wrapper记录255。
- 不是数值崩溃、kernel/cgroup OOM、SSH timeout或RLT/DSRL代码混淆。
- 未重启训练。

## 文件职责

- 原始：`driver.log`、`metrics.log`、`resource.csv`、`resolved.yaml`、`launch_manifest.txt`、`driver.exit`。
- 派生：两个CSV、三张PNG、`summary_step52.json`。
- `resource_observer.log`为空是原运行事实；分钟采样数据在`resource.csv`。

PPO比较证据只到Step46，因此图中PPO线在Step46结束；没有用旧值延长到Step52。

