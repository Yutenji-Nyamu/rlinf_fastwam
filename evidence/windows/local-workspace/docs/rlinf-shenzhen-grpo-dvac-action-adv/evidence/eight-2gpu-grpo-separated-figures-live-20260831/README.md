# 深圳八条两卡 GRPO 正式实验统一图

## 口径

- 只纳入深圳两卡、同一训练预算的正式实验：`64 train env × 4 rollout epochs`、G8、B1024/MB32/update2、fixed32/eval5。
- 不纳入 smoke、失败重跑、四卡实验、AutoDL 实验或不同采样预算实验。
- 横轴统一显示到最长的 Control Step96；每条曲线只画到自身真实最终/最新完整 step，不向后补线。
- Action-Adv Fix `[0.5,1.5]` 与 ST-DVAC `[0.8,1.2]` 使用 2026-08-31 10:43 CST 本地只读快照；其余为封存终态。

## 独立图

- [逐步训练成功率](01_raw_success.png)
- [5 步滑动均值](02_ma5_success.png)
- [10 步滑动均值](03_ma10_success.png)
- [fixed-32 评估](04_fixed32_success.png)
- [桌面交互图](interactive.html)
- [逐步数据](curves.csv)
- [汇总与来源日志](summary.json)

## 当前汇总

| 实验 | 完整 step | 末步 raw | 末 5 步 | 末 10 步 | fixed-32 累计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO Control | 96 | 92.97% | 94.69% | 94.73% | 570/608 = 93.75% |
| ST-DVAC `[0,2]` | 52 | 94.53% | 90.31% | 89.34% | 296/320 = 92.50% |
| Prism-RLOO | 55 | 93.36% | 95.08% | 93.12% | 326/352 = 92.61% |
| Action-Adv old（H-mean） | 29 | 80.08% | 84.84% | 81.13% | 133/160 = 83.12% |
| Action-Adv Fix `[0,2]` | 81 | 94.53% | 93.36% | 92.73% | 480/512 = 93.75% |
| ST-DVAC `[0.5,1.5]` | 53 | 93.75% | 93.83% | 92.07% | 294/320 = 91.88% |
| Action-Adv Fix `[0.5,1.5]` | 59 | 91.80% | 93.75% | 94.57% | 325/352 = 92.33% |
| ST-DVAC `[0.8,1.2]` | 43 | 88.28% | 90.00% | 92.23% | 235/256 = 91.80% |

## 读图结论

- 旧 Action-Adv（H-mean）明显偏低，对应已经确认的 $1/H$ loss-reduction 实现问题；它用于展示修复前后差异，不应作为有效方法结果。
- 修复后的几条方法在训练后段大多进入约 92%--95% 的平滑成功率区间。
- fixed-32 累计没有显示某个新方法稳定、显著地压过 Control；当前是单次轨迹且终点不同，不能用各自“最后一点”直接排总名次。
- 颜色之外同时使用线型与 marker；即使缩小或灰度查看也能区分。

生成脚本：[render_sz_eight_2gpu_grpo_separate_20260831.py](../../../../local_scripts/render_sz_eight_2gpu_grpo_separate_20260831.py)。操作记录见 [OPERATION_LEDGER.md](OPERATION_LEDGER.md)。
