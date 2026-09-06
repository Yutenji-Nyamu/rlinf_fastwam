# R-only v3 `[0,2]`：Global Step 42训练、方法、资源与产物现场

快照时间：2026-08-23 15:56–16:00 CST。服务器：AutoDL A800容器
`autodl-container-nekaqbwt43-6ce5babb`。本轮只读；没有暂停训练、改配置、加载checkpoint或修改远端产物。

## 1. 当前状态

- 最新完整更新：`Global Step 42/100`；16:05:05时Step43 rollout=`4/16`。
- wrapper、driver、observer以及2 actor、2 rollout、2 env worker均alive。
- g42：training-rollout success=`93.359%`、approx KL=`0.0733`、joint-query clip=`16.671%`、
  pre-global-clip grad norm=`44.244`、ratio=`0.9893`。
- 主训练数值全部finite；CUDA OOM、ActorDied、RayActorError、NCCL、cgroup OOM/OOM-kill均为0。
- driver中的4个`Traceback`仍全部是启动时可选CuRobo/pytorch3d导入提示；不是训练中新增fatal。

## 2. 四次训练同轴结果

所有success都是on-policy训练rollout，不是held-out fixed-ID评估。

| run（g1–42） | 累计success | 最近5步 | 最近10步 | g42 | mean KL | mean clip | mean pre-clip grad | mean step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 原GRPO | 87.788% | 92.656% | 92.148% | 86.719% | 0.0469 | 15.267% | 31.280 | 24.601 min |
| v1 global-z `[0.8,1.2]` | 86.021% | 91.953% | 90.586% | 92.578% | 0.0437 | 14.479% | 34.180 | 24.753 min |
| v2 R-only `[0.5,1.2]` | 86.533% | 91.641% | 92.227% | 90.625% | 0.0375 | 13.917% | 31.918 | 24.907 min |
| **v3 R-only `[0,2]`** | **88.160%** | **92.969%** | **92.500%** | **93.359%** | **0.0446** | **15.419%** | **40.086** | **24.728 min** |

同step轴上，v3相对原GRPO：累计`+0.372pp`、最近5步`+0.313pp`、最近10步`+0.352pp`。g23–29附近
曾出现更大的正差，之后逐渐收窄并短暂交叉；到g42只是轻微正差。因而当前最准确的结论是：v3训练
rollout与原GRPO整体接近，尚未显示稳定领先；最终效果仍要靠同fixed-ID checkpoint评估。

v3的mean pre-global-clip grad norm比原GRPO高约28.2%，说明`[0,2]`重加权确实进入反向传播；随后两者
仍被`clip_grad=1`统一缩放，所以不能把28.2%直接解释为optimizer步长增加28.2%。

![四次训练同轴图](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/v3_formal_live_g42_20260823/analysis/V3_FOUR_RUN_TRAINING_G42.png)

## 3. v3方法实际作用强度

g42最新双rank、281个loss-valid query、14,050个action位置的合并结果：

| 量 | g42 | g2–42平均/解释 |
|---|---:|---:|
| weight p05 / median / mean / p95 | `0.328 / 1.101 / 1.118 / 2.000` | `0.414 / 1.156 / 1.174 / 1.994` |
| 降权 / 增权位置 | `42.33% / 57.67%` | `37.73% / 62.27%` |
| 命中0 / 命中2 | `1.011% / 5.822%` | `0.415% / 7.403%` |
| positive / negative advantage mean weight | `1.084 / 1.222` | `1.123 / 1.280` |
| per-query weight ESS | `0.876` | 等效`43.79/50`个action |
| coefficient angle | `20.00°` | 相对uniform系数向量 |
| top-20% weight mass | `32.54%` | uniform为20% |
| weighted credit top-20% mass | `56.76%` | uniform-weight同批为48.56% |
| 后25格 − 前25格mean weight | `+0.0867` | 仍保留小的相对horizon结构 |

从g36到g42，ESS从`0.897`降到`0.876`、系数角从`18.23°`升到`20.00°`，说明本批action贡献重排稍强。
但它仍远弱于hard 80/20的ESS约0.2。g42绝对credit总量约为同批uniform GRPO的`1.168×`；同时
negative-advantage query的平均weight继续高于positive侧，因此当前机制仍偏向更强地压制失败侧动作。

g1到g42的raw `V_L3`几何均值增加约64.3%；它同时混合on-policy访问状态变化与模型变化，不能仅由这条
曲线判断模型自身不确定性升高。

![v3方法诊断图](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/v3_formal_live_g42_20260823/analysis/V3_METHOD_DIAGNOSTICS_G42.png)

## 4. 资源

| 资源 | 当前/累计 |
|---|---:|
| GPU0 / GPU1 15:59即时显存 | `22,211 / 22,058 MiB` |
| GPU0 / GPU1累计峰值 | `30.368 / 30.218 GiB` |
| cgroup current / limit | 15:56一度`239.986/240 GiB`；16:05约`234.540/240 GiB` |
| cgroup累计峰值 | `240 GiB` |
| 16:00匿名内存 / file-cache | `153.817 / 77.227 GiB` |
| `memory.events max` | `159,401`；相对observer起点增加`122,644` |
| OOM / OOM-kill | `0 / 0` |
| host available最低值 | `819.66 GiB` |
| 数据盘可用最低值 | `666.08 GiB` |

GPU显存稳定且余量充足。主压力仍是Env/主机侧内存：cgroup反复贴近240 GiB上限，`max`事件继续增加，
但截至快照仍能推进且没有OOM。本轮仅观察，没有把资源监控连接到停止行为。

![资源图](/C:/Users/86136/Documents/rl/docs/rlinf-robotwin-pi0-dvac-telemetry/evidence/v3_formal_live_g42_20260823/analysis/V3_RESOURCES_G42.png)

## 5. 当前产物

- run目录约`39 GiB`；runtime约`61 MiB`。
- `global_step_10/20/30/40` checkpoint均存在，各约`9.7 GiB`；本轮没有下载checkpoint正文。
- 84个DVAC NPZ，正好是42个完整step × 2 actor rank；最新为双rank`rollout_step0041.npz`。
- 双rank runner CSV、rolling-state、manifest、TensorBoard event、metrics、driver、resolved config与资源CSV齐全。
- control trace仍按配置抽样保存1个MP4和1个frames CSV，不是每step全量录像。

本地轻量快照约10.16 MB，不含checkpoint正文。机器可读结果：

- [汇总JSON](evidence/v3_formal_live_g42_20260823/analysis/SUMMARY_G42.json)
- [四run训练CSV](evidence/v3_formal_live_g42_20260823/analysis/FOUR_RUN_TRAINING_G42.csv)
- [DVAC逐step CSV](evidence/v3_formal_live_g42_20260823/analysis/V3_DVAC_STEP_METRICS_G42.csv)
- [g42 future-h CSV](evidence/v3_formal_live_g42_20260823/analysis/V3_LATEST_HORIZON_G42.csv)
- [资源CSV](evidence/v3_formal_live_g42_20260823/analysis/V3_RESOURCES_G42.csv)

完整远程指令、结果和本地分析过程见
[本轮逐指令账](evidence/V3_FORMAL_LIVE_REFRESH_LEDGER_20260823.md)。
