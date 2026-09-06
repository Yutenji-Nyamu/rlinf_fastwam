# Global-z `[0,2]` g49容器重启收尾与1/5/10步成功率（2026-08-24）

## 1. 先说当前结果

这次100-step formal没有自然跑完。服务器现场确认：

- 最后完整记录是 **Global Step 49**；下一轮只完成rollout `14/16`，没有形成Global Step 50，故不计入任何训练曲线。
- 旧driver最后写日志于`2026-08-24 15:59:22 +08:00`；资源observer最后采样于`16:00:27`，当时driver仍alive、两卡仍占用显存。
- 当前容器PID 1启动于`16:03:26`。因此旧训练随**容器重启**中断，并非Python自然完成100步。
- driver尾部没有CUDA OOM、NCCL fatal、actor fatal或`KeyboardInterrupt`；run期`event_oom=0`、`event_oom_kill=0`。
- 容器内证据只能确定“重启导致进程消失”，不能确定外部是谁或什么操作触发了重启。
- 当前GPU和训练RAM已经释放；保留完整checkpoint为g10/g20/g30/g40，g49只有训练/方法telemetry，没有g49 checkpoint。

## 2. 用户要看的三种成功率

这里的success均是**训练中的on-policy rollout success**：每个Global Step有256条trajectory，不是held-out fixed-ID评估。

| 口径 | g49 | 含义 |
|---|---:|---|
| 每步原始success | 93.359% | 只看g49本步的256条trajectory；最灵敏，也最抖 |
| trailing 5-step mean | 93.203% | g45--g49五步平均；压掉一部分单步采样波动 |
| trailing 10-step mean | 93.320% | g40--g49十步平均；更稳定，但对新变化反应更慢 |
| g1--49累计平均 | 90.139% | 从训练开始到中断的整体平均，不是末段水平 |

主图：

- [当前run：每步、5步、10步同图](evidence/global_z_w0to2_stop_g49_20260824/analysis/GLOBAL_Z_SUCCESS_RAW_ROLL5_ROLL10_G49.png)
- [原GRPO、v1、v2、v3、当前global-z：三种口径五run同轴](evidence/global_z_w0to2_stop_g49_20260824/analysis/FIVE_RUN_SUCCESS_RAW_ROLL5_ROLL10_G49.png)

计算是“向后看”的算术平均，并包含当前step：

$$
\bar s^{(5)}_t=\frac{1}{5}\sum_{k=t-4}^{t}s_k,
\qquad
\bar s^{(10)}_t=\frac{1}{10}\sum_{k=t-9}^{t}s_k.
$$

前4/9个点因为历史不足，分别使用当时已有的较短窗口；没有补未来值，也没有把任何run延长到其真实终点之后。

## 3. 与已有run的同期比较

只比较共同的g1--49：

| 对照 | 累计平均差 | g49末5步差 | g49末10步差 |
|---|---:|---:|---:|
| 原GRPO | **+2.081 pp** | **+2.734 pp** | **+2.930 pp** |
| v1 global-z `[0.8,1.2]` | +3.850 pp | +5.000 pp | +4.063 pp |
| v2 R-only `[0.5,1.2]` | +3.021 pp | +1.484 pp | +2.695 pp |
| v3 R-only `[0,2]` | +1.730 pp | +3.438 pp | +2.617 pp |

这次末5和末10都在约93.2--93.3%，说明g49不是孤立的单步尖峰；当前global-z cell在中断前的训练rollout曲线上确实处于五个run中较高的一条。

但这还不是最终策略效果结论：原GRPO继续训练到g100后，末10步达到96.25%；当前run在g49中断，且没有同一checkpoint的fixed-ID held-out评估。现阶段只能说“前49步的on-policy success更高”，不能替代checkpoint评估。

## 4. 优化指标有没有一起失控

g1--49平均：

| 指标 | 当前global-z `[0,2]` | 原GRPO | 读法 |
|---|---:|---:|---|
| approx KL | 0.0476 | 0.0472 | 策略相对旧策略的变化量几乎相同 |
| joint-query clip fraction | 14.882% | 15.002% | 被PPO ratio clip的query比例几乎相同 |
| pre-global-clip grad norm | 33.742 | 30.860 | 当前稍高，但两者之后都会做同一个global norm clip |
| wall time / step | 24.667 min | 24.616 min | DVAC记录与权重没有带来可见吞吐损失 |

g49单步为：KL=`0.059`、clip=`9.4%`、pre-clip grad norm=`32.791`、step time=`24.273 min`。单步会波动，均值更适合看训练是否整体偏离原协议。

## 5. 方法本身在g49怎样作用

最新双rank、256个有效query、12,800个action位置：

- 权重p05/median/mean/p95=`0.367/1.038/1.088/2.000`；0.258%位置为0，7.438%位置达到2。
- 46.94%位置被降权，53.06%位置被增权。
- per-query weight ESS=`0.897`，等效`44.83/50`个action发言；系数角约`18.22°`。它比旧`[0.8,1.2]`明显更强，但仍远没有hard 80/20那样稀疏。
- 正advantage query平均权重=`1.041`，负advantage query=`1.220`。公式没有读取正负号；这是当前数据中负advantage样本恰好对应更高global-z，所以本步更强地压制失败方向。
- `weight_back25 - front25=+0.180`，weight与future index的Spearman=`0.929`。这说明raw global-z确实保留了“越远期越不稳定、越被增权”的结构；它与R-only去位置趋势的实验变量不同。
- g1到g49 raw `V_L3`几何均值增加约65.1%。这是on-policy混合量，同时受模型和访问state变化影响。

方法图：[g1--49权重、正负advantage、global-z与future-h结构](evidence/global_z_w0to2_stop_g49_20260824/analysis/GLOBAL_Z_METHOD_DIAGNOSTICS_G49.png)。

## 6. 资源与重启边界

- 记录时长：20.54 h。
- GPU峰值：GPU0=`30.37 GiB`，GPU1=`30.22 GiB`；显存不是瓶颈。
- cgroup内存峰值约`240.00 GiB`；最后采样仍为`220.40 GiB`。
- `memory.events max`相对run首样本增加87,054，表示内存分配多次触及240 GiB限制并被回压；`oom`和`oom_kill`增量均为0。
- 这些数据说明RAM长期很紧，但没有证据把容器重启直接归因于cgroup max事件；重启触发源不在当前容器日志中。

资源图：[GPU、cgroup组成、与v2/v3增长、memory event](evidence/global_z_w0to2_stop_g49_20260824/analysis/GLOBAL_Z_RESOURCES_G49.png)。

## 7. 证据与复现

- [逐指令流水账](evidence/GLOBAL_Z_W0TO2_G49_STOP_AND_CURVES_LEDGER_20260824.md)
- [分析脚本](evidence/global_z_w0to2_stop_g49_20260824/analyze_global_z_g49.py)
- [汇总JSON](evidence/global_z_w0to2_stop_g49_20260824/analysis/SUMMARY_G49.json)
- [当前run逐step/5步/10步CSV](evidence/global_z_w0to2_stop_g49_20260824/analysis/GLOBAL_Z_SUCCESS_RAW_ROLL5_ROLL10_G49.csv)
- [五run完整真实终点CSV](evidence/global_z_w0to2_stop_g49_20260824/analysis/FIVE_RUN_SUCCESS_RAW_ROLL5_ROLL10_G49.csv)

## 8. 轻量收尾ZIP

- 文件：[idea2_dvac_global_z_w0to2_stop_g49_20260824.zip](../../exports/idea2_dvac_global_z_w0to2_stop_g49_20260824.zip)
- 大小：`3,038,577 bytes`（约`2.90 MiB`）
- SHA256：`b9f9da78479a2e7406290fcd58e0668df1818ef9e1965b2b5f1b24dc3b0f57e7`
- 校验旁车：[ZIP SHA256](../../exports/idea2_dvac_global_z_w0to2_stop_g49_20260824.zip.sha256)
- 包内：28个文件，未压缩总量`12,232,071 bytes`；含README、本文、逐指令账、四张高信息量PNG、关键
  CSV/JSON/复现脚本、`metrics.log`、driver/resolved/精确命令、双rank runner/rolling/manifest与各一份最新NPZ、
  一份完整资源CSV。
- 未包含：checkpoint、视频、历史全量逐step NPZ，以及与原始资源CSV重复的约9.9 MB派生资源CSV。

封装后复核：ZIP可正常列举，27个payload文件的SHA256与包内`FILE_SHA256.tsv`全部一致，缺失/不匹配均为0；
凭据与登录端点扫描命中为0。阅读入口是包内`README_FIRST.md`。
