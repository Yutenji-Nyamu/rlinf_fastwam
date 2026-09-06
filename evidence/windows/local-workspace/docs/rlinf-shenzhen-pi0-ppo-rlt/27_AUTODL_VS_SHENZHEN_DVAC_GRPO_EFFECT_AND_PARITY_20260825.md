# AutoDL 与深圳 global-z `[0,2]` GRPO-DVAC：效果和实现同口径复核

更新时间：2026-08-25 11:10 CST。

## 1. 结论

1. 深圳 current 实现没有发现公式、统计域、rank 聚合、shuffle 或 ST 梯度挂点错误；它与深圳 matched GRPO 是严格单变量对照。
2. 深圳截至完整 Step31 尚无稳定收益：训练 success 累计差 `-0.50 pp`、最近5步差 `-1.25 pp`；Step10/20/30 fixed64累计两边均为 `179/192`。
3. AutoDL 的训练 rollout 曲线确有优势：g1--49累计/末5/末10为 `+2.081/+2.734/+2.930 pp`；但它没有 fixed held-out eval，也不是同 seed、同代码 `mode=off` 的并行对照，不能直接写成“策略稳定提升”。
4. 当前代码与旧代码的 DVAC 数学一致；更可能的差异来自每步样本/批量/optimizer节奏、RLinf/OpenPI版本、2-rank与4-rank拓扑及单seed随机性，而不是“深圳 DVAC 没接上”。

## 2. 深圳 matched 结果

| 口径 | GRPO-DVAC | matched GRPO | 差值 |
|---|---:|---:|---:|
| Step1--31 train mean | 86.240% | 86.738% | -0.50 pp |
| Step27--31 train mean | 91.836% | 93.086% | -1.25 pp |
| Step30 train | 90.234% | 97.656% | -7.42 pp |
| Step10 fixed64 | 57/64 | 57/64 | 0 |
| Step20 fixed64 | 59/64 | 60/64 | -1/64 |
| Step30 fixed64 | 63/64 | 62/64 | +1/64 |
| fixed累计 | 179/192 | 179/192 | 0 |

Step30单次 fixed 更好，但三次累计完全相同；它不能单独支持“有效”或“无效”。

现场机制不是 no-op：Step31 weight mean/std/ESS=`1.0135/0.4382/0.8425`，upper/lower clip=`3.736%/0.398%`；KL/clip/grad=`0.013/0.055/20.351`，均为有限值。

图：

- [深圳完整 baseline52、当前31、5步均值及fixed64](evidence/dvac-grpo-v4-live-step31-20260825/01_dvac_v4_vs_grpo_v2_success_step1_31_full_baseline52.png)
- [深圳优化与DVAC权重](evidence/dvac-grpo-v4-live-step31-20260825/02_dvac_v4_optimization_and_weights_step1_31.png)
- [深圳资源时间线](evidence/dvac-grpo-v4-live-step31-20260825/03_dvac_v4_resource_timeline_through_step31.png)

## 3. AutoDL 旧结果应怎样表述

AutoDL global-z `[0,2]` 相对其旧 GRPO：

- g1--49累计：`+2.081 pp`；
- 末5步：`+2.734 pp`；
- 末10步：`+2.930 pp`；
- g11--49的39个位置中，5步均值有37个位置高于旧 GRPO。

因此“warm-up后训练 rollout 曲线大部分时间更高”有依据。限制是：

- g1是 `w=1`，两条run已经相差 `-3.125 pp`，说明并非共享随机轨迹；
- 对照不是同seed、同代码 `mode=off`；
- `val_check_interval=-1`，没有 fixed held-out eval；
- 只有一个方法run，没有重复seed。

所以当前最准确的说法是“AutoDL单次run的on-policy训练曲线显示稳定趋势优势”，不是“策略效果已被稳定复现”。

[AutoDL g49收尾和曲线口径](../rlinf-robotwin-pi0-dvac-telemetry/31_GLOBAL_Z_W0TO2_G49_RESTART_CLOSEOUT_AND_SUCCESS_SMOOTHING_20260824.md)；[逐步CSV](../rlinf-robotwin-pi0-dvac-telemetry/evidence/global_z_w0to2_stop_g49_20260824/analysis/FIVE_RUN_SUCCESS_RAW_ROLL5_ROLL10_G49.csv)。

## 4. 实现与参数复核

| 项目 | AutoDL old | 深圳 current | 判断 |
|---|---|---|---|
| DVAC endpoint | `M=4, L=3` | `M=4, L=3` | 等价 |
| DVAC定义 | population variance over last3；14D求和 | 同左 | 等价 |
| 统计域 | 所有rank/query/future-h的 `log(V+1e-12)` | 同左 | 等价 |
| recent history | 前5个已完成outer step | 同左 | 等价 |
| warm-up | Step1全 `w=1` | 同左 | 等价 |
| 映射 | `w=1+0.5 clip(z,-2,2)` | 同左 | 等价 |
| actor挂点 | ST前向恒等，只缩放每个h反向梯度 | 同左 | 等价 |
| rank reduce | 动态world-size `all_reduce`，2 ranks | 同逻辑，4 ranks | 数学等价、拓扑不同 |
| shuffle | old trajectory路径 | current typed `forward_inputs`同shuffle | 已正确适配 |
| resume | recent-5不恢复 | per-rank sidecar严格恢复 | current改进 |

深圳 DVAC 与其 matched baseline 的非DVAC字段逐叶一致：`128×4=512 trajectories`、G8、最多2,048 records、GB2,048/MB32/update2、LR/clip、fixed64/save10、模型和seed内容相同；unexpected difference=`0`。

GitHub 2026-08-25只读核对：旧分支 `codex/idea2-dvac-residual-downweight@afdaa2e2...`、深圳分支 `codex/sz-current-pi0-dvac-grpo@66c863bc...` 与本地审计authority一致。

## 5. 跨机器为什么可能不复现

| 轴 | AutoDL | 深圳 |
|---|---:|---:|
| GPU / actor ranks | 2×A800 / 2 | 4×H100 / 4 |
| trajectories / outer step | 256 | 512 |
| global batch | 512 | 2,048 |
| micro batch / update epochs | 32 / 2 | 32 / 2 |
| optimizer calls / outer step | 约4 | 约2 |
| fixed held-out eval | 无 | 每10步64条 |

所以同一个“Step30”不是同一采样预算或更新节奏。深圳内部DVAC/GRPO仍是公平对照，但不能把AutoDL的效应大小机械搬到深圳。其次，两个实验基于不同RLinf/OpenPI代码代际；数学已复核等价，但训练动力学、模型字节身份和随机调度没有被证明完全相同。

[跨机器同轴图](evidence/dvac-grpo-v4-live-step31-20260825/04_autodl_vs_shenzhen_global_z_effect_same_machine_controls.png)

## 6. 下一观察点

- 当前无需重新上传仓库：代码、commit、resolved和旧曲线已足够完成实现/参数审计。
- 深圳下一高信息量观察点是 Step40 fixed64；在此之前不因单个train step决定停训。
- 若要把AutoDL旧主张升级为效果证据，最缺的是同代码 `mode=off` 与DVAC的matched fixed-ID eval和2--3个fresh seed，而不是更多工程日志。
- 当前主风险仍是128 train + 64 eval常驻环境的主存增长，不是DVAC tensor；应继续只读观察Ray内存边界。

## 7. 现场终点

2026-08-25 11:10 CST：完整Step33/100，Step34 rollout `1/4`；wrapper alive、exit pending、fatal=0。GPU4--7属于本run，约59--61 GiB/card；GPU0--3空闲。host available约318 GiB，memory PSI很低但非零，磁盘 `/`、`/home`、`/data`分别余233 GiB、2.1 TiB、2.7 TiB。
