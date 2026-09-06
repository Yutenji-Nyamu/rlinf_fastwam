# R-only v3 `[0,2]`：Global Step 36训练、方法、资源与产物现场

快照时间：2026-08-23 13:31–13:36 CST。服务器：AutoDL A800容器
`autodl-container-nekaqbwt43-6ce5babb`。本轮只读；没有控制训练、改配置、加载checkpoint或修改远端产物。

## 1. 当前状态

- 最新完整更新：`Global Step 36/100`。
- 13:36:53 CST时Step37 rollout=`4/16`；wrapper、driver、observer仍alive。
- 2 actor、2 rollout、2 env worker均有GPU/process现场证据。
- g36主要值：training-rollout success=`95.703%`、approx KL=`0.060`、joint-query clip=`17.483%`、
  pre-global-clip grad norm=`33.938`。
- 主训练数值全部finite；CUDA OOM、ActorDied、RayActorError、NCCL、cgroup OOM/OOM-kill均为0。
- driver中的4个`Traceback`均来自启动时可选CuRobo/pytorch3d导入提示；与此前成功启动相同，环境worker已
  连续完成36步，当前不是运行时fatal。

## 2. 四次训练同轴结果

所有success均为on-policy训练rollout，不是held-out fixed-ID评估。

| run（g1–36） | 累计success | 最近5步 | 最近10步 | g36 | mean KL | mean clip | mean pre-clip grad | mean step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原GRPO | 86.914% | 91.719% | 91.094% | 92.578% | 0.0467 | 14.808% | 32.241 | 24.601 min |
| v1 global-z `[0.8,1.2]` | 85.059% | 89.766% | 90.078% | 93.359% | 0.0425 | 14.050% | 34.737 | 24.802 min |
| v2 R-only `[0.5,1.2]` | 85.558% | 92.656% | 92.070% | 95.703% | 0.0376 | 13.689% | 32.871 | 24.939 min |
| **v3 R-only `[0,2]`** | **87.457%** | **92.422%** | **92.500%** | **95.703%** | **0.0430** | **15.158%** | **40.415** | **24.731 min** |

同step轴上，v3相对原GRPO：累计`+0.543pp`、最近5步`+0.703pp`、最近10步`+1.406pp`。它已经不再像
早期g1–10那样落后，但当前优势较小且曲线仍会交叉；只能说训练rollout出现轻微正差，不能替代同fixed-ID
checkpoint评估。

v3的mean pre-global-clip grad norm比原GRPO高约25.4%，说明强权重已经进入反向传播。两边随后都会被
`clip_grad=1`统一缩放，因此这个差异不是最终optimizer步长直接大25.4%；更重要的是action贡献重排后的
梯度方向。

![四次训练同轴图](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/v3_formal_live_g36_20260823/analysis/V3_FOUR_RUN_TRAINING_G36.png)

## 3. v3方法实际作用强度

g36最新双rank、179个loss-valid query、8,950个action位置的合并结果：

| 量 | g36 | g2–36平均/解释 |
|---|---:|---:|
| weight p05 / median / mean / p95 | `0.451 / 1.176 / 1.196 / 2.000` | `0.423 / 1.160 / 1.177 / 1.994` |
| 降权 / 增权位置 | `35.43% / 64.57%` | `37.35% / 62.65%` |
| 命中0 / 命中2 | `0.358% / 7.810%` | `0.367% / 7.280%` |
| positive / negative advantage mean weight | `1.163 / 1.295` | `1.124 / 1.286` |
| per-query weight ESS | `0.897` | 等效`44.85/50`个action |
| coefficient angle | `18.23°` | 相对uniform系数向量 |
| top-20% weight mass | `31.24%` | uniform为20% |
| 后25格 − 前25格mean weight | `+0.0557` | 仍有小的相对horizon结构 |

这说明`[0,2]`已经产生实质重排：比v2终态ESS约0.973明显更集中，但仍远弱于hard 80/20的ESS约0.2。
当前并非mean-one：g36所有绝对credit总量约为uniform GRPO的`1.230×`；再加上负advantage query的平均
weight更高，v3目前更强地放大失败侧的负向抑制。这是解释后续效果时需要单独看的机制，不只是“高DVAC
位置获得更多关注”。

g1到g36的raw `V_L3`几何均值增加约62.4%；这是on-policy访问状态和模型同时变化后的曲线，不能只归因于
模型本身变得更不稳定。

![v3方法诊断图](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/v3_formal_live_g36_20260823/analysis/V3_METHOD_DIAGNOSTICS_G36.png)

## 4. 资源

| 资源 | 当前/累计 |
|---|---:|
| GPU0 / GPU1 13:36即时显存 | `24,779 / 24,919 MiB` |
| GPU0 / GPU1累计峰值 | `30.368 / 30.218 GiB` |
| cgroup 13:36 current / limit | `234.453 / 240 GiB` |
| cgroup累计峰值 | `240 GiB` |
| 13:31匿名内存 / file-cache | `153.510 / 77.710 GiB` |
| `memory.events max` | `119,918`；相对启动增加`83,161` |
| OOM / OOM-kill | `0 / 0` |
| host available最低值 | `819.66 GiB` |
| 数据盘可用最低值 | `671.77 GiB`；13:31 `df`约679 GiB |

资源曲线与v2相似，v3在相同elapsed附近的cgroup total约高`8.64 GiB`、anonymous约高`9.03 GiB`。
主存已多次触碰240 GiB ceiling，当前仍能推进且没有OOM；这里只观察，不将资源监控连接到自动停止行为。
GPU显存余量充足，主要压力仍是Env/主机侧内存而非A800显存。

![资源图](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/v3_formal_live_g36_20260823/analysis/V3_RESOURCES_G36.png)

## 5. 当前产物

- run目录约`30 GiB`；runtime约`50 MiB`。
- `global_step_10/20/30` checkpoint目录均存在，各约`9.7 GiB`；本轮没有加载checkpoint正文。
- 72个DVAC NPZ，正好是36个完整step × 2 actor rank；最新为双rank
  `rollout_step0035.npz`。
- 双rank `runner_step_metrics.csv`、rolling-state、manifest均持续更新。
- 1个TensorBoard event；`metrics.log`、driver log、resolved config和资源CSV均存在。
- 抽样control trace仍是1个MP4 + 1个frames CSV：15,569 bytes / 13,172 bytes；它按配置只抽一个case，
  不是每step录像。
- 数据盘约68%使用、inode约2%，当前checkpoint继续按每10步保存仍有空间。

本地轻量快照约8.9 MB，不含checkpoint正文。机器可读结果位于：

- [汇总JSON](evidence/v3_formal_live_g36_20260823/analysis/SUMMARY_G36.json)
- [四run训练CSV](evidence/v3_formal_live_g36_20260823/analysis/FOUR_RUN_TRAINING_G36.csv)
- [DVAC逐step CSV](evidence/v3_formal_live_g36_20260823/analysis/V3_DVAC_STEP_METRICS_G36.csv)
- [资源CSV](evidence/v3_formal_live_g36_20260823/analysis/V3_RESOURCES_G36.csv)

完整远程指令、下载中的活文件大小问题及修正见
[本轮逐指令账](evidence/V3_FORMAL_LIVE_REFRESH_LEDGER_20260823.md)。
