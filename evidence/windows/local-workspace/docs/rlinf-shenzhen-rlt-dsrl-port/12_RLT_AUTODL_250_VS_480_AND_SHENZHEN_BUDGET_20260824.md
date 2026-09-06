# RLT：AutoDL 250/480 与深圳 current 预算一页对齐

最后更新：2026-08-24  
状态：只读配置、历史结果与深圳现场核对。

## 1. “两段”其实有两种含义

RLT 方法本身固定是两阶段：

1. **Stage 1**：离线 demonstration reconstruction，训练 RLT token 表征；
2. **Stage 2**：在线 RoboTwin AC/BC/Q，训练小 actor/critic。

AutoDL Stage 2 后来又分成两个进程：先跑 `1--250`，再从完整 checkpoint 续跑 `251--480`。
这个进程重启不是第三个算法阶段，cycle 250 处也没有改变模型、batch、UTD 或 BC/Q schedule。

## 2. 四条预算不要混淆

| 运行 | Stage 1 | Stage 2 | 结论 |
|---|---:|---:|---|
| AutoDL 旧 pilot | 已用2k artifact | 4 env × 100 cycles = 400 episodes | 低预算 pilot；不是迁移基线 |
| AutoDL 完整 formal | 2×A800，MB16/GB32，2,000 optimizer steps | 8 env × 250 cycles = 2,000 episodes | source-aligned 完整基线；250/250、exit0、final fixed20=18/20 |
| AutoDL 延长续训 | 同一 Stage 1 artifact | 从250续到480，总计3,840 episodes | 同配置延长；final fixed20=17/20，高位平台而非继续稳定上升 |
| 深圳 current | 2×H100，MB16/GB32，2,000 optimizer steps | 8 env × 250 cycles = 2,000 episodes | 复制旧完整250基线，不是100-cycle pilot，也不是最长480延长线 |

因此：

- 如果“完整”指经过验证的正式基线，深圳用的是完整 `8 env × 250`，答案是**是**。
- 如果“长的完整”指 AutoDL 历史最终绝对终点480，深圳当前不是480，答案是**否**。
- 当前没有理由预先把深圳终点改成480：AutoDL cycle250 已是18/20；251--480 的10次 fixed20 合计89%，
  final17/20，主要提升是 train rollout 稳定性，没有证据证明 held-out/fixed eval继续上升。

## 3. 深圳照抄了什么、改了什么

照抄旧完整 formal：

- exact π0、clean-50（50 episodes / 7,188 frames / 三相机 / 14D）和同一 norm；
- Stage 1 仅训 token，π0 frozen，`rlt_train_vla=false`、`alpha=0`；
- Stage 1 MB16/rank、GB32、AdamW 2.5e-5、warmup100+cosine、2k；
- Stage 2 两卡、8 env、H50/C10/D14、50k replay/rank、10k warm-up/rank；
- macro UTD5、critic:actor=2:1、GB512/MB128；
- actor warm-up20k+ramp50k，BC/Q `7/.05 -> 2.5/.45`；
- 每25 cycles做 fixed20 与 checkpoint，共10次。

必须改变的科学项只有 Stage 1 reconstruction：旧仓当时的 official parallel decoder 改为 current 官方
causal AR fix，因此旧 Stage 1 checkpoint不能复用，深圳用相同2k预算 fresh重训。Stage 2 AC/BC/Q主体不换。

current schema/worker、RoboTwin route/adapter、strict resume、persistent Ray和pre-update checkpoint warm-up
是工程适配，不改变上述科学预算。

## 4. 2026-08-24 14:49 CST 深圳现场

- Stage 1：`2000/2000` 已完成并保存 current-AR artifact。
- Stage 2 v4：alive，最新完整 `110/250`。
- min-rank replay=`8160/10000`，global transitions=`16512`。
- `actor_switch_rate=0`、`ready_for_online=0`、`update_step=0`：仍是 reference replay warm-up，尚未开始
  actor/critic update；当前 student fixed20为0不能叫“学习失败”。
- GPU4/5约21.4/21.7 GiB，host available约1.93 TiB；没有资源压力信号。

旧250 formal的真实阶段是：P1 cycles1--135 reference收集；P2 136--154 reference+SAC；P3 155--191
student+ramp；P4 192--250 stable student。深圳目前的 Step110 与这条已验证 schedule 一致，还没走到学习段。

## 5. 判断

深圳没有重新猜参数，也没有重复旧100-cycle pilot；它在复刻旧250正式基线。唯一曾重踩的是 current
FSDP pre-update checkpoint 接缝，而非 RLT 参数：两行 optimizer-state warm-up 修复后，v4 已通过同一
Step25边界并继续运行。

推荐保持 `250` 终点。到250后用 fixed20曲线、student阶段成功率、update预算和数值指标决定是否另立
`250 -> 480` 续训；不要现在把历史最长终点自动等同于必要训练预算。

深度证据：

- [AutoDL Stage 1 2k](../rlinf-robotwin-pi0-rltoken/03_STAGE1_FORMAL_TRAINING_20260729.md)
- [AutoDL Stage 2 250终态](../rlinf-robotwin-pi0-rltoken/15_STAGE2_FORMAL_8ENV250_FINAL_RESULT_20260730.md)
- [AutoDL 250→480终态](../rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md)
- [深圳 resolved formal packet](07_FORMAL_LAUNCH_AND_CONCURRENCY_DECISION_20260823.md)
- [深圳 checkpoint修复](10_RLINF_MULTI_JOB_RUNTIME_AND_RLT_CHECKPOINT_DIAGNOSIS_20260824.md)
