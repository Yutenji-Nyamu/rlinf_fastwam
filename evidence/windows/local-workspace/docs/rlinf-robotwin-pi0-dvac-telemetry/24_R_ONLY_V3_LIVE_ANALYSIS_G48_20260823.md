# R-only v3 `[0,2]`：Global Step 48训练、方法、资源与产物现场

快照时间：2026-08-23 18:45–18:46 CST。服务器：AutoDL A800容器
`autodl-container-nekaqbwt43-6ce5babb`。本轮只读；没有暂停训练、改配置、加载checkpoint或修改远端产物。

## 1. 当前状态

- 最新完整更新：`Global Step 48/100`；18:50:06时Step49 rollout=`15/16`。
- wrapper、driver、observer以及2 actor、2 rollout、2 env worker均alive。
- g48：training-rollout success=`84.375%`、approx KL=`0.0510`、joint-query clip=`16.453%`、
  pre-global-clip grad norm=`41.155`、ratio=`1.0067`。
- 主训练数值全部finite；CUDA OOM、ActorDied、RayActorError、NCCL、cgroup OOM/OOM-kill均为0。
- driver中的4个`Traceback`仍全部是启动时可选CuRobo/pytorch3d导入提示，没有新增训练fatal。

## 2. 四次训练同轴结果

所有success都是on-policy训练rollout，不是held-out fixed-ID评估。

| run（g1–48） | 累计success | 最近5步 | 最近10步 | g48 | mean KL | mean clip | mean pre-clip grad | mean step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原GRPO | 87.996% | 89.141% | 90.391% | 85.547% | 0.0477 | 14.979% | 30.757 | 24.613 min |
| v1 global-z `[0.8,1.2]` | 86.190% | 87.109% | 89.141% | 85.547% | 0.0458 | 14.685% | 34.014 | 24.753 min |
| v2 R-only `[0.5,1.2]` | 87.004% | 90.078% | 90.391% | 90.625% | 0.0387 | 14.208% | 31.076 | 24.893 min |
| **v3 R-only `[0,2]`** | **88.322%** | **88.438%** | **90.742%** | **84.375%** | **0.0428** | **15.383%** | **40.103** | **24.722 min** |

同step轴上，v3相对原GRPO：累计`+0.326pp`、最近5步`-0.703pp`、最近10步`+0.352pp`。g23–29附近
曾出现约5pp的5步均值正差，此后差距收窄并多次过零；到g48没有稳定领先。当前最直接的判断仍是：
v3训练rollout与原GRPO接近，训练曲线本身不足以证明方法效果，需要同fixed-ID checkpoint评估。

v3的mean pre-global-clip grad norm比原GRPO高约30.4%，说明强重加权持续进入反向传播；两者随后都
会被`clip_grad=1`统一缩放，所以该30.4%不是最终optimizer步长的直接增幅。

![四次训练同轴图](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/v3_formal_live_g48_20260823/analysis/V3_FOUR_RUN_TRAINING_G48.png)

## 3. v3方法实际作用强度

g48最新双rank、336个loss-valid query、16,800个action位置的合并结果：

| 量 | g48 | g2–48平均/解释 |
|---|---:|---:|
| weight p05 / median / mean / p95 | `0.456 / 1.140 / 1.167 / 2.000` | `0.414 / 1.160 / 1.177 / 1.994` |
| 降权 / 增权位置 | `38.65% / 61.35%` | `37.53% / 62.47%` |
| 命中0 / 命中2 | `0.286% / 6.405%` | `0.420% / 7.578%` |
| positive / negative advantage mean weight | `1.145 / 1.222` | `1.126 / 1.282` |
| per-query weight ESS | `0.906` | 等效`45.29/50`个action |
| coefficient angle | `17.37°` | 相对uniform系数向量 |
| top-20% weight mass | `31.52%` | uniform为20% |
| weighted credit top-20% mass | `51.13%` | uniform-weight同批为45.94% |
| 后25格 − 前25格mean weight | `+0.0604` | 保留小的相对horizon结构 |

g48的ESS比g42的`0.876`高、系数角比`20.00°`低，表示这一批的action重排稍弱；这类强度随每批residual
分布变化，并非固定在某个角度。g48绝对credit总量约为同批uniform GRPO的`1.193×`。负advantage侧的
平均weight仍更高，因此当前方法继续更强地压制失败侧动作，但正负差距比若干早期step小。

g1到g48的raw `V_L3`几何均值增加约84.9%；它同时混合on-policy访问状态和模型变化，不能仅据此判断
模型自身不确定性增加。

![v3方法诊断图](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/v3_formal_live_g48_20260823/analysis/V3_METHOD_DIAGNOSTICS_G48.png)

## 4. 资源

| 资源 | 当前/累计 |
|---|---:|
| GPU0 / GPU1 18:45即时显存 | `29,497 / 27,825 MiB` |
| GPU0 / GPU1累计峰值 | `30.368 / 30.218 GiB` |
| cgroup current / limit | 18:46约`232.299/240 GiB`；18:50约`231.203/240 GiB` |
| cgroup累计峰值 | `240 GiB` |
| 18:46匿名内存 / file-cache | `160.179 / 70.032 GiB` |
| `memory.events max` | `165,367`；相对observer起点增加`128,610` |
| OOM / OOM-kill | `0 / 0` |
| host available最低值 | `812.70 GiB` |
| 数据盘可用最低值 | `666.08 GiB` |

GPU显存曲线稳定。cgroup总量仍在高位锯齿波动并曾触及240 GiB；相较g42，file/cache下降而anonymous
增加，当前总量略回落。`max`事件继续增加，但截至快照仍无OOM并能正常推进；本轮没有将资源观察连接到
停止行为。

![资源图](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/v3_formal_live_g48_20260823/analysis/V3_RESOURCES_G48.png)

## 5. 当前产物

- run目录约`39 GiB`；runtime约`67 MiB`。
- `global_step_10/20/30/40` checkpoint均存在，各约`9.7 GiB`；下一合同保存点是g50。
- 96个DVAC NPZ，正好是48个完整step × 2 actor rank；最新为双rank`rollout_step0047.npz`。
- 双rank runner CSV、rolling-state、manifest、TensorBoard event、metrics、driver、resolved config与资源CSV齐全。
- control trace仍按配置抽样保存1个MP4和1个frames CSV，不是每step全量录像。

本地轻量快照约11.59 MB，不含checkpoint正文。机器可读结果：

- [汇总JSON](evidence/v3_formal_live_g48_20260823/analysis/SUMMARY_G48.json)
- [四run训练CSV](evidence/v3_formal_live_g48_20260823/analysis/FOUR_RUN_TRAINING_G48.csv)
- [DVAC逐step CSV](evidence/v3_formal_live_g48_20260823/analysis/V3_DVAC_STEP_METRICS_G48.csv)
- [g48 future-h CSV](evidence/v3_formal_live_g48_20260823/analysis/V3_LATEST_HORIZON_G48.csv)
- [资源CSV](evidence/v3_formal_live_g48_20260823/analysis/V3_RESOURCES_G48.csv)

完整远程指令、结果和本地分析过程见
[本轮逐指令账](evidence/V3_FORMAL_LIVE_REFRESH_LEDGER_20260823.md)。
