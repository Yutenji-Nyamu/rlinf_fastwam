# 深圳双两卡 GRPO 对照：DVAC `[0,2]` Step 52 终态

## 结论

- DVAC `[0,2]`按用户授权在完整Step52后停止并替换为Prism实验；Control没有停止，继续运行。
- 配对Step1--52的DVAC-Control训练成功率差为：累计`+3.13 pp`、末5步`-1.95 pp`、末10步`-1.33 pp`。
- 十次 fixed-32 累计为 DVAC `296/320`、Control `294/320`，只差 2 条。
- 因此终态是“DVAC早期累计领先、末段短窗略低、held-out近似打平”，不能据此声称稳定收益或退化。

## 配置与实现核对

- 两项同一 source HEAD `0e28ac6f...`。
- 共同预算为 `64 env x 4 epochs = 256 trajectories/step`、G8、最多 1024 records、GB1024/MB32/update2、fixed-32 每 5 步。
- 逐叶比较 `unexpected_differences=[]`。唯一方法差异是 Control `mode=off`、DVAC `mode=apply`；其余只是 GPU placement 和 run-scoped 输出路径。
- DVAC 额外计算 pi0 denoising endpoint variance，使用前 5 个完整 outer step 的全局历史做 z-score，映射到 `[0,2]`，再用 straight-through 只缩放每个 future action 的 log-prob 梯度。reward、GRPO advantage、group filter、ratio、clip、loss 聚合和 optimizer 均未修改。
- DVAC权重与ESS始终有非零变化，可排除method no-op；精确逐步值保存在CSV与轻量包中。

## 跨实验边界

- AutoDL 的 Step 1--49 中，DVAC-Control 累计/末5/末10为 `+2.08/+2.73/+2.93 pp`，但没有 held-out fixed eval、同代码 `mode=off` 对照或多 seed。
- 旧深圳四卡的 DVAC-Control Step 1--41 为 `-0.39/-0.66/-0.33 pp`，fixed-64 累计同为 `242/256`。
- AutoDL 与当前两卡虽然都是 256 trajectories/step，但 AutoDL 是 GB512、约 4 次 optimizer calls/step；当前是 GB1024、2 次 calls/step。跨机器曲线只作背景，不能归因。
- 两个当前 job 也不是共享同一批 simulator trajectory；Step 1 时 DVAC 权重仍全为 1，而两边 rollout success 已不同。因此训练曲线的小差值不能单独视作方法因果，fixed-ID eval 和多 seed 更可靠。

## 产物

- `01_current_2gpu_control_vs_dvac_raw_roll5_roll10.png`：当前两卡 raw、5-step、10-step 与 fixed-32。
- `02_shenzhen_old_grpo_and_autodl_context.png`：旧深圳四卡与 AutoDL 背景曲线。
- `current_2gpu_success_curves.csv`：当前逐步数据。
- `comparison_summary.json`：主要数值摘要。
- `raw/*/runtime/pair_parity.json`：逐叶配置核对。
