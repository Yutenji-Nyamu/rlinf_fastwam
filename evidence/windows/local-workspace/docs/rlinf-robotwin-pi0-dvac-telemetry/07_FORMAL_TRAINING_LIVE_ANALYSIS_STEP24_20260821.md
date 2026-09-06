# 100-step DVAC-GRPO 正式训练：Step 24 现场与 Step 23 完整快照分析

最后刷新：2026-08-21 10:23:26+08:00  
完整下载快照：Global Step 23；现场追加确认：Global Step 24。

## 1. 一句话结论

训练仍在正常运行，已经完成`24/100`。DVAC权重从step2起持续实际参与反传，PPO训练形态、速度和资源轨迹
都与历史成功GRPO很接近；当前没有数值发散、CUDA/Ray/OOM或显存压力。训练rollout success最近已大致
追平历史曲线，但现在还只是前24步的on-policy训练表现。

## 2. 当前进度

- wrapper / driver / observer三个进程都在，现场没有`driver.exitcode`，说明尚未结束；
- Global Step 24：`success_once=78.52%`、KL=`0.045`、clip fraction=`0.129`、pre-clip grad norm=`33.52`、
  ratio=`1.002`；单步`1512.2s=25.20min`；
- 日志ETA为`31:35:00`；按此前近10步速度估算也约剩32小时；
- `global_step_10`和`global_step_20` checkpoint已存在，完整快照时run约`20 GiB`。

Step 24的success比前几步低，是256条rollout上的单步波动；因此下文完整统计与图固定使用已经下载、
两rank DVAC shard齐全的Global Step 1–23快照。

## 3. 训练指标与历史成功GRPO

| 指标 | DVAC当前快照 | 历史GRPO同step | 解释 |
|---|---:|---:|---|
| g23 rollout success | 87.11% | 85.55% | 当前高1.56个百分点 |
| g1–g23 success均值 | 82.75% | 84.97% | 当前早期低2.23个百分点 |
| 最近5步success均值 | 89.22% | 87.73% | 当前高1.48个百分点 |
| 最近10步success均值 | 86.29% | 86.13% | 基本相同 |
| g23 approx KL | 0.049 | 0.036 | 都在正常波动区间 |
| g23 clip fraction | 0.172 | 0.135 | 当前这一步更高；前23步均值反而低0.009 |
| g23 pre-clip grad norm | 38.41 | 32.05 | 当前高6.37；前23步均值只高1.68 |
| 最近10步耗时 | 1485.47 s | 1491.90 s | DVAC没有可见的持续耗时税 |

当前ratio始终在`0.975–1.009`；除第1步KL=`0.142`外，后续KL最大`0.069`；grad norm范围
`24.33–47.49`。这些形态与旧GRPO接近，说明新增信号和action-h梯度分配没有破坏原来的joint ratio、
PPO clipping或训练节奏。

对应图：

- [DVAC与历史GRPO同step对照](evidence/formal_live_step22_20260821/analysis/BASELINE_COMPARISON.png)
- [当前训练指标总览](evidence/formal_live_step22_20260821/analysis/TRAIN_METRICS.png)
- [逐step对照CSV](evidence/formal_live_step22_20260821/analysis/BASELINE_COMPARISON.csv)

## 4. 我们的方法实际做了什么

Global Step 23的selected signal为：

- current `log V_L3=-4.16985±0.53774`；
- 生成本步权重所用recent-5 history为`-4.22698±0.55717`；
- current比history的几何均值约高`5.88%`；
- loss-mask有效的345个query上，weight p05 / median / mean / p95为
  `0.8896 / 1.0185 / 1.0255 / 1.2000`；
- 只有`0.13%`触及下界0.8，`5.06%`触及上界1.2。

所以它不是把整次学习率统一放大：全部query的apply期平均weight只有`1.0022`，接近1；真正发生的是
同一chunk内部不同future action位置的梯度贡献重新分配。

首版有意保留了future-`h`效应，真实训练也清楚看到了它：`h`与median `V_L3`的Spearman相关为
`0.895`；后25个action比前25个平均多约`0.028`权重，`h=49`的有效query平均weight约`1.096`。
这说明“越靠后通常越不确定”确实进入了本版gradient weighting，而不是被离线校正掉。

另一个稳定现象是：22个apply step里，负advantage位置的平均weight每一步都比正advantage略高，平均差
`0.0295`。高DVAC并不自己决定强化还是惩罚：GRPO advantage仍决定方向；这里表示当前数据中较不确定
的位置更常出现在负向credit一侧。

对应图与表：

- [DVAC方法诊断主图](evidence/formal_live_step22_20260821/analysis/DVAC_FORMAL_STEP23_METHOD_OVERVIEW.png)
- [逐step方法汇总](evidence/formal_live_step22_20260821/analysis/DVAC_STEP_SUMMARY.csv)
- [逐step × future-h汇总](evidence/formal_live_step22_20260821/analysis/DVAC_BY_H.csv)

## 5. 资源与历史GRPO

截至10:05完整资源快照：

- GPU0/1显存峰值`30.37/30.22 GiB`，中位`26.28/26.00 GiB`；两卡80 GiB，显存宽松；
- cgroup峰值`182.69 GiB`，快照末`174.08/240 GiB`，memory low/high/max/oom/oom_kill全部为0；
- 两个EnvWorker RSS合计峰值`126.63 GiB`，仍是主机内存主体；
- host MemAvailable约`834 GiB`，SHM约`120 GiB`，disk available约`795 GiB`。

与旧成功GRPO按各自启动后9.86小时对齐：

- cgroup从各自起点的增长为`161.29 vs 162.97 GiB`，DVAC反而少`1.68 GiB`；
- EnvWorker合计峰值`126.63 vs 127.43 GiB`，几乎相同；
- DVAC每卡显存峰值约多`0.9–1.0 GiB`，但仍远低于80 GiB。

因此当前主机内存上升主要是历史GRPO本身已有的环境worker轨迹，不像DVAC记录带来的额外膨胀。旧run
最终曾到`236.33/240 GiB`并完成100步，所以后半程余量会变小，但当前没有新的资源异常形态。

对应图：

- [前10小时资源对照](evidence/formal_live_step22_20260821/analysis/FORMAL_VS_BASELINE_RESOURCE.png)
- [当前资源全时序](evidence/formal_live_step22_20260821/analysis/FORMAL_RESOURCE_OVERVIEW.png)

## 6. 产物、空间与细录像

- Step 10/20 checkpoint各约`9.7 GiB`；若每10步都保留，100步最终checkpoint约`97 GiB`量级；
- Step 1–23两rank DVAC train artifacts合计约`24.8 MB`，按当前速度到100步约`0.1 GiB`；
- 控制录像只按配置录整个driver的第一条episode，不随100步倍增：MP4 `23,367 bytes`，200帧、
  H.264、160×120、10 FPS；CSV `23,241 bytes`，metadata不足1 KB；
- 这条reset57样例到200步仍失败。q0–q3的`V_L3,total`依次为
  `0.693 / 1.095 / 2.373 / 1.283`，q2最高；它发生在warmup step，所以权重全为1。

可查看：

- [reset57细录像](evidence/formal_live_step22_20260821/run/control_trace/adjust_bottle/worker_000/env_slot_000/recording_0000_episode_0000_reset_57/head_camera.mp4)
- [录像contact sheet](evidence/formal_live_step22_20260821/analysis/CONTROL_TRACE_CONTACT_SHEET.png)
- [录像与DVAC对齐表](evidence/formal_live_step22_20260821/analysis/CONTROL_TRACE_RESET57_DVAC.csv)

## 7. 当前判断

现在最值得保留的三个结论是：

1. 工程链路成立：train-SDE DVAC、recent-5统计、`[0.8,1.2]`action-h梯度和两rank产物已连续运行23个完整step；
2. 训练形态未被破坏：相对历史GRPO，成功率近期追平，KL/clip/ratio/grad和速度都在相近范围；
3. 方法确实在改变credit分配：后段action和负advantage有效query得到稍高权重，而不是只记录了一个无效指标。

完整100步结束后，再用同fixed-ID eval判断最终策略表现；当前无需改变配置或训练进程。

> 证据目录名`formal_live_step22_20260821`来自第一次现场检查时的最新UI step；下载过程中Global Step 23
> 已完成。目录中的metrics与两rank `runner_step=22` shards共同对应完整Global Step 23。
