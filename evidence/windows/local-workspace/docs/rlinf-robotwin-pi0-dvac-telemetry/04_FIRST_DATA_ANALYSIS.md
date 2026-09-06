# Idea2 DVAC 首轮 64-query 离线分析

最后更新：2026-08-20  
原始运行：`idea2_dvac_sft_smoke_2gpu_16env_v1`  
最终派生版本：`first_collection_v3_20260820`  
服务器目录：`/root/autodl-tmp/idea2_dvac_analysis/first_collection_v3_20260820`  
逐指令证据：[FIRST_DATA_ANALYSIS_LEDGER.md](evidence/FIRST_DATA_ANALYSIS_LEDGER.md)

## 1. 先说结论

这 64 个 query 已经证明两件事：

1. **信号确实有 action 粒度结构。** 同一个 query 内的 50 个未来位置 `h` 并非一起等比例变化；失败
   episode 5 的 q0 甚至呈现很清楚的“前段低、从约 h=20 开始整段升高”。这正是以后 first crossing
   能利用的形状，而不是只有一个 chunk scalar。
2. **还不能直接进入训练加权。** 当前最强的共同规律是“越远的未来 action，方差往往越高”：在仍未
   success 的 50 个 query 中，`h` 与跨 query 中位 `V_L3(h)` 的 Spearman 相关为 `0.852`；78% query
   的后半段均值高于前半段，70% query 的最大值落在后半段。因此 signal 明确随 trajectory query/state
   与随机 noise 的组合而变化，同时混有普通的 horizon-distance effect；当前每个状态只采了一次 noise，
   还不能把其中多少单独归因给 task state。若现在直接给高方差 action 更高 RL 权重，很可能主要是在给
   远端 `h` 加权。

首轮最诚实的工作结论是：**DVAC-like endpoint variance 在当前 π0/RoboTwin 链路里可测且有结构，值得
继续分析；下一步应先控制 future-h 基线并补 phase/contact 对齐，而不是立刻改 PPO/RLT objective。**

### 后续训练首版的冻结选择

上面是64-query数据本身支持的结论，不能回写成“高V已被证明更关键”。在后续训练讨论中，用户选择把
raw future-h效应也暂时视作不确定性的一部分，先做一个连续小幅的训练实验：online直接使用recent-5
global `log V_L3`统计生成`[0.8,1.2]`权重，同时落盘`h`，离线继续把raw与per-h residual并排画。这个选择
是在测试一个假设，不改变本节对首轮数据证据强度的判断。训练公式与实现见
[05_TRAINING_MODIFICATION_PLAN.md](05_TRAINING_MODIFICATION_PLAN.md)。

## 2. 论文口径与本次适配

论文 v1 的核心公式是：

\[
z_i=x_i-t_i v_i,
\qquad
V_L(h)=\sum_d \frac1L\sum_{i=M-L}^{M-1}
(z_i(h,d)-\bar z(h,d))^2,
\qquad
V_{total}=\sum_hV_L(h).
\]

本轮严格使用总体方差 `ddof=0`，动作维只取 normalized active 14D。论文默认 `L=5`，当前 π0 只有
`M=4`，所以同时计算 `L=2/3/4`；它们是三个不同观察窗口，绝对数值不能横向当作同一标尺。

另一个容易混淆的边界：论文的 rolling threshold `tau` 是和单个 `V(h)` 比较并寻找 first crossing；
`V_total` 只作分析时的标量诊断，不能把 `V_total` 和 `tau` 放在同一轴上直接比较。论文算法又没有公开
q0 空 rolling-buffer 的初始化规则；当前也没有可验证的官方代码仓。因此首轮没有伪造“exact DVAC
counterfactual N_exec”。完整执行规则是：若首个 crossing 为 `k*`，执行 `max(N_min,k*)`；若没有 crossing，
执行 `N_max`。论文算法默认 `N_min=1`，但其 RoboTwin/π0 实验表覆盖为 `N_min=5,N_max=50`。论文原文与
附录：<https://arxiv.org/html/2606.03847v1>。

## 3. 数据完整性

- 2 个 rollout shards，各 32 query；合计 64 query、16 episode、每 episode 恰好 q0–q3。
- `z_endpoint [64,4,50,14]`、`x_chain [64,5,50,14]`，timesteps 为
  `[1.0,0.75,0.5,0.25]`。
- 所有 trace 数值 finite；64 个 `query_uid` 唯一；query→episode/reset ID join 完整。
- `x_chain[:,-1]` 与 `final_model_action` 最大绝对差为 0。
- 14/16 episode success，2 个 failure 是 ep05/reset 100100081 与 ep09/reset 100100122。
- 原始两份 NPZ SHA256 分别为 `cf4bfd19...f7cc2b` 与 `c7081042...8f911`；完整 hash 在
  [analysis_summary.json](evidence/first_data_analysis_v3/analysis_summary.json)。

派生脚本是 [analyze_dvac_first_collection.py](../../local_scripts/analyze_dvac_first_collection.py)，服务器
最终脚本 SHA256 为 `4d9ec65b095576f62cb979ae4b7169bc793f9f08742cc8d9e5556dce47e7042b`。

## 4. 主要发现

### 4.1 `L` 选择不是无关紧要

64 个 query 的 `V_total` 排名相关：

| 比较 | query-level Spearman | 3,200 个 `query×h` 的描述性 Spearman |
|---|---:|---:|
| L2–L3 | 0.827 | 0.741 |
| L2–L4 | 0.629 | 0.450 |
| L3–L4 | 0.808 | 0.752 |

这说明三个窗口读到相关结构，但远非完全等价。L4 把 `t=1` 的最早 preview 包进来，尺度自然显著更大；
L2 只剩两个 endpoint，估计最接近尾部但样本最少。首轮图把 **L3 作为中间工作视图**，同时保留 L2/L4，
并未宣称 L3 已经是最终超参。

坐标贡献也不是只有 gripper：L3 总方差中 d6/d13 两个夹爪坐标分别占 11.2%/13.4%，合计24.6%；其余
12个关节坐标仍贡献75.4%。这说明保存 action-coordinate 粒度是有价值的，同时也提示以后可以单独检查
gripper/contact grammar。

对应图：[L敏感性与14D贡献](evidence/first_data_analysis_v3/FIG01_L_SENSITIVITY_AND_DIMENSIONS.png)。

### 4.2 future `h` 本身就是一个强基线

只看 `success_before=false` 的 50 个 query：

- `h` 与跨 query 中位 `V_L3(h)`：Spearman `0.852`；
- 后半段 `h=25..49` / 前半段 `h=0..24` 的方差均值比，中位数 `1.274`；
- 78% query 的后半段均值更高；
- 70% query 的最大 `V_L3(h)` 落在后半段。

这不是坏事：DVAC 本来就在找“预测到多远开始不稳定”。但对我们的 80/20→RL 迁移很重要：训练权重
不能把 `V(h)` 原值直接当作 task-critical score，至少要和相同 `h` 的基线比较，或先做 within-query
标准化/残差化。

对应图：[future-h位置效应](evidence/first_data_analysis_v3/FIG10_HORIZON_POSITION_EFFECT.png)、
[全部64×50热图](evidence/first_data_analysis_v3/FIG08_ALL_QUERY_HORIZON_HEATMAP.png)。

### 4.3 episode 时间轴只有4点，但出现了一个可靠的 success 边界

L3 的 `V_total` 中位数为：

| query | action-slot start | 中位 `V_total,L3` | 状态事实 |
|---|---:|---:|---|
| q0 | 0 | 0.695 | 16/16 pre-success |
| q1 | 50 | 0.846 | 16/16 pre-success |
| q2 | 100 | 0.440 | 16/16 pre-success |
| q3 | 150 | 0.368 | 14/16 post-success；2个failure仍pre-success |

也就是说，14 个成功 episode 都是在执行 q2 生成的 C50（action slots 100–149）期间首次使任务 predicate
为真；因为配置 `ignore_terminations=true`，评估仍继续发出 q3。于是 q3 的下降不能简单解释为“后期
moving更稳定”，它大部分已是**任务完成后的另一种状态分布**。

从 head 图做定性对齐，典型成功 episode 大致是 q0 瓶子平放、q1 接近/接触、q2 抓持旋转、q3 已直立；
这和论文的阶段故事相容，但现在只有 query 边界图，不能把它升级成 contact 真值。

对应图：[success边界](evidence/first_data_analysis_v3/FIG09_RECORDED_SUCCESS_BOUNDARY.png)、
[代表性episode图像+曲线](evidence/first_data_analysis_v3/FIG04_REPRESENTATIVE_EPISODE_STORIES.png)、
[16个episode四点时间线](evidence/first_data_analysis_v3/FIG03_ALL_EPISODE_TIMELINES.png)。

### 4.4 两个 failure 有提示性，但远不足以做成败结论

- failure query 的 L3 `log10(V_total)` 中位数为 0.029，success query 为 -0.240；方向上 failure 较高，
  但独立失败样本只有2个。
- ep05 q0 是全体最大值：`V_total,L3=8.413`，明显高于全体75分位0.785；它的 per-h 曲线在约h=20后
  持续升高。
- ep09 的高点出现在 q2，`V_total,L3=1.788`，视觉上已经抓起瓶子但最终未满足成功 predicate。
- 成功 ep06 的 q1 也有较高方差。因此 endpoint variance 不是“失败概率”，更不能自行提供 RL
  advantage 的正负号。

对应图：[代表性4×50热图](evidence/first_data_analysis_v3/FIG05_REPRESENTATIVE_HORIZON_HEATMAPS.png)、
[最低/最高query的L2/L3/L4曲线](evidence/first_data_analysis_v3/FIG06_LOW_HIGH_QUERY_HORIZON_CURVES.png)。

### 4.5 四次“改口”能直接看见

[endpoint收敛图](evidence/first_data_analysis_v3/FIG07_ENDPOINT_CONVERGENCE.png)画的是
`||z_i(h)-x_M(h)||_2`。高方差 query 的远端 action 在 i0/i1 仍离最终 action 很远，低方差 query 的修订
幅度小得多。最后 `i3,t=0.25` 与 final action 距离为0不是数据造假：在当前四等分 Euler 更新里，最后
一次步长正好覆盖剩余0.25，`z_3=x_3-0.25v_3=x_4`。

左侧每列是一个future `h`、每行是一次denoise preview；右侧只抽该query内L3方差最低/最高的两个`h`，
展示模型如何逐步“改口”。上图是64个query中全局最低V的post-success query，下图是全局最高V的failure
q0，所以这里只用于解释机制，不是同状态、同phase的严格成败对照。上下panel还各自autoscale色条/纵轴，
不能直接用相同颜色或线高跨panel读倍数。

## 5. 用户所附四张论文图，我们现在能做到哪一步

先澄清编号：这里的“图1/2/3/4”是用户本轮四张附件的顺序；其中附件图2实际是论文 **Figure 4**。
论文自己的 **Figure 2** 是方法总览：50个 per-`h` 方差组成 `B_s`，历史窗口形成 `tau_s`，first crossing
再决定稳定执行前缀。我们当前的50点 `V(h)` 已支持画它的离线输入与 crossing 形状，但还没有在线改
`N_exec`。

| 用户图 | 论文含义 | 当前数据 |
|---|---|---|
| 图1 / Figure 1(a) | 连续 rollout 的 `V_total`，配 MOVING/OPERATING 帧 | 可做四点 query-level 版本，已见 FIG03/04；不能称 phase-resolved复现 |
| 图2 / Figure 4 | 在线 DVAC 的 `V_total` 与真实自适应 `N_exec` | 当前是固定C50 baseline，没有在线改执行长度；只能先画 `V(h)`，不能声称已验证 `N_exec` |
| 图3 / Figure 7 | 有外部 phase 标签后的 MOVING vs OPERATING 分布 | 当前无phase标签；论文自己也是用Seed-2.0-Pro看前/腕视图标注，不是方差自标注 |
| 图4 / Figure 13 | 每episode inference-step时序并按phase着色 | 已做16个episode×4点与64×50热图；增加曲线点数必须更频繁replan/query；更细帧只改善phase标签对齐，不会新增`V_total`点 |

因此“6帧也能不能先分析”的答案是 **能**：query-level episode故事、50个`h`的action粒度形状、L敏感
性、success边界都能做；但它不支持 C50 内部连续 phase 曲线。

## 6. 为什么每支MP4恰好6帧

当前 `200/C50=4` 次外层 chunk call。`RecordVideo`保存：

```text
1 初始reset帧
+ 3个普通chunk终态
+ 1个最后chunk真实终态
+ 1个auto-reset spill帧
= 6帧
```

`RoboTwinEnv.chunk_step()`把50个 action waypoints 在内部整段执行，只向 wrapper 返回一个终态 observation；
所以不是编码器漏掉49帧。30 FPS 只把6帧以0.2秒播放，并不代表仿真是30 Hz，也不会增加信息。

当前 RLinf 运行使用 `/root/autodl-tmp/RoboTwin_RLinf`，这条 VectorEnv 路径强制关闭 RoboTwin native
`eval_video_log`，本次没有第二套 native MP4。旧 `/root/autodl-tmp/RoboTwin/eval_result` 的确有历史
direct-evaluator视频，但不是这次RLinf评估同步写出的副本。

## 7. 若要更细画面，具体要怎样“开”

原始RLinf/RoboTwin路径**没有一个现成YAML开关能直接得到逐action帧**；单设 `eval_video_log:true`
不够，因为VectorEnv会覆盖它，而且native recorder的入口也不是当前qpos chunk路径。后续训练child已按
本节高层方案实现一个默认关闭的窄自定义旁路：复用camera API/ffmpeg，但不直接调用official direct-eval
recorder。其配置形态为：

```yaml
env.train.task_config.control_trace:
  enabled: false
  camera: head_camera
  output_dir: ${runner.logger.log_path}/control_trace
```

实际首版没有把每个physics step全录下来，而是在RoboTwin qpos control loop的既有render之后，根据双臂
TOPP同步进度只取新h-bin、success或query终点帧；C50每query最多50帧、每episode最多200帧。每个选中env
独占文件，避免多个SubEnv抢同一writer：

```text
control_trace/adjust_bottle/worker_000/env_slot_000/recording_.../
  head_camera.mp4
  frames.csv
  metadata.json
```

`frames.csv`记录query/reset/env/frame/control/physics index、双臂各自control进度、近似
`h_float/h_lo/h_hi/fraction`、success与首次success。这里不能写成“frame=精确执行原chunk第h行”：左右臂
分别经compression/TOPP后按不同进度推进，首版以两臂较慢一侧的normalized progress映射到`0..49`；这是
接近action粒度的phase参考，不是精确waypoint lineage。实现不重复调用`compress_path()`，也不把C50拆成
50次高层`take_action()`，所以没有为录像改变实际轨迹或policy RNG。

训练smoke只确定性选择`worker0/slot0`的一条episode验证帧量、映射、编码与offload重建；没有和32/64
并发压力测试绑在一起。

## 8. 下一次并发：先32，不直接64

- 当前16 env = 8 env/GPU，实测约18.7 GiB/GPU、55.2 GiB cgroup。
- 32 env = 16 env/GPU，正好匹配 official 128 env / 8 GPU 的每rank密度，是合理下一档。粗估约
  25–35 GiB/GPU、75–100 GiB cgroup，需实测。
- 64 env = 32 env/GPU，是official密度两倍；粗估35–55 GiB/GPU、120–170 GiB cgroup，只是单点
  外推，SAPIEN/CPU/RAM比NPZ大小更可能先成为瓶颈。

更重要的是，并发的直接作用是吞吐；它也会让当前 fixed seed bank 的前缀变长，从而增加 episode 数，
但不会让单个 episode 变得更细。按现有两 rank 分区逻辑，16→32 env 会保留当前16个ID并新增16个，
16→64则保留当前16个并新增48个，并非全量重复。若目标是取得**完全无重叠**的增量数据，优先显式制作
两份互斥32-ID shard串行，或多份互斥16-ID shard；不要把“样本量64”绑死成“同时64 env”。只有再次
运行相同16-env前缀，才会重复当前这16个reset IDs。

## 9. 本轮产物

- [query_metrics.csv](evidence/first_data_analysis_v3/query_metrics.csv)：64行，每query的L2/L3/L4
  `V_total`与索引。
- [horizon_metrics.csv](evidence/first_data_analysis_v3/horizon_metrics.csv)：3,200行，每query×h的
  L2/L3/L4 `V(h)`。
- [analysis_summary.json](evidence/first_data_analysis_v3/analysis_summary.json)：hash、校验、统计与证据边界。
- [完整派生产物压缩包](evidence/first_data_analysis_v3/first_collection_v3_20260820_artifacts.tar.gz)：
  3,601,179 bytes，SHA256 `01ec0cab94a28aee064fa2ecf3fac2377200cdf4554e492e84a3a1c63beae4d0`。

本轮没有启动新评估、simulator或GPU进程，没有改模型/chunk/原始run目录，也没有在线阈值或训练权重。
