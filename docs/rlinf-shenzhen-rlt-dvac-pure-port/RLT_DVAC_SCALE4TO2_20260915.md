# RLT DVAC new：成功 BC 倍率 4 → 2，GPU6 fresh600

2026-09-15。按本轮授权，将 GPU6 原 Fast-WAM `turn_switch` BC 换为 π0 RLT DVAC new。新组从空 Stage2 回放与更新计数 0 开始，沿用原 **8 env / 600 轮** Clean 的实配、环境和已完成的 current-AR Stage1。GPU7 半量 Clean 及 GPU0–5 保持原状。

## 方法与预算

源码基点为原 scale2 `097420394aaf5a35631f86f3632b48354a0f86ac`，本次生产源码为 `3c60dbddeffe5102253b3b54e460f5cd7dd862f0`。新增 opt-in `algorithm.rlt_dvac.success_scale_schedule`；缺省或禁用时保留旧倍率和旧恢复合同。

| 项目 | 正式设置 |
|---|---|
| DVAC | `apply / two_level_batch / successful_batch`；L3、监督 C10、两个 alpha 均 1；原正向权重 |
| 成功 BC 倍率 | 起始 `success_scale=4`；`anchor=actor_weight_schedule_end`，`final_scale=2` |
| 采集与更新 | 8 env，B512 / micro256，UTD5，critic:actor = 2:1，actor/critic LR 均 1e−4 |
| 初始化 | 最低回放 20,000；初始 critic 30,000 次；每轮更新上限 1,600；回放上限 80,000 |
| 全局 BC/Q 课程 | 前 20,000 次 critic 使用 7 / 0.05，再用 50,000 次过渡到 2.5 / 0.45 |
| 训练与评估 | Stage2 fresh600；固定 20 episode，每 25 轮评估和保存；原 seed 表 |
| Stage1 / norm | 复用 current-AR clean50、训练 2,000 步的原 Stage1 checkpoint；原 `RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50` norm |

成功 query 的 teacher-reference BC 权重为 `W = s × 内层位置权重 × 外层 query 权重`。倍率 4 → 2 在同一 batch 下使成功 BC 损失及其梯度分量减半；成功内部相对分配、失败 BC 和 Q 项保持。若成功比例为 p，全 batch 的平均系数由 `1+3p` 变为 `1+p`；总梯度和 Adam 参数步长不必减半。[原权重机制](RLT_DVAC_NEW_PARAMETERS_EXPLAINED_20260912.md)

## 切点与恢复

`update_step=t` 是当前更新槽之前已完成的 critic 次数。生产课程实际采用 `p=clip((t−20000+1)/50000,0,1)`，因此数学端点为 **t=69999**；actor 只在偶数 t 更新，首个实际使用倍率 2 的 actor 槽是 **t=70000**。该槽先完成第 70001 次 critic，再更新 actor，槽尾计数增至 70001；不会等到 `t>=70001` 才切。

端点的全局 BC/Q 为 2.5 / 0.45，比值约 5.56；初始为 7 / 0.05，比值 140。实现直接读取原课程的实际进度，完整 B512 先计算成功权重再分 microbatch；不另设时间计数或随机流。启用时恢复合同绑定完整 BC/Q 课程、fallback 系数与派生端点，恢复原 `update_step` 即恢复倍率阶段，课程变更会拒绝恢复。

## 验证与状态

深圳 CPU **11/11 通过**（3.70 秒）：覆盖边界、同 batch 成功 BC 梯度减半、失败不变、无随机流消耗、禁用兼容及切点前后恢复/课程变更拒绝。生产 worker 和新增测试是本次两份源码改动。

修正后的 GPU smoke 于 **00:33:56.265044 通过**：使用真实冻结回放 512 query，复用生产 FSDP、loss、优化器与更新循环，注入 t=69998 做 4 次 critic / 2 次 actor，验证倍率 4 → 2。该注入只用于 smoke，正式计数从 0 开始。

**00:44:45.509185 最终启动验收通过**：v2 首轮 **R1=2/8**，随后 **R2=1/8**；日志已记录回放 142、`update_step=0 / online=0`、54 类标量，处于正常 teacher 初池期。driver `2691417 / uid1003 / start229035552` 存活，12 个 actors 正常，8 env、fresh、teacher `apply` 与预期配置一致，无 fatal，GPU0–5/7 保护进程保持。完整对照原 scale2 实配，仅成功倍率 2 → 4、新 schedule 三项及身份/路径变化，`reference_scale2_unexpected={}`；没有额外方法或预算差异。此次验收覆盖启动、实配及边界 smoke，尚未验证正式训练进入成熟 online 阶段，也不构成策略表现结论。

原 Fast BC 于 **00:24:40** 停止，首次 v1 于 00:24:53 启动。v1 配置只移植了 `algorithm.rlt_dvac`，遗漏 teacher 的 **`rollout.rlt_feature_model.openpi.rlt_dvac_mode=apply`**，沿用了 Clean 的 `off`。该开关控制 endpoint 方差记录；冻结回放 smoke 已自带 `teacher_dvac_v`，不能覆盖新采集链路。修复准备曾误写父节点为 `rollout.model`，在停训前因缺少该键而终止，失败记录保留。

随后启用正确的 `rlt_feature_model.openpi` 开关并增加 smoke 配置断言：v1 于 **00:33:43.381982** 精确停止；上述 GPU smoke 通过后，v2 于 **00:33:56.472783** 从空池启动。停 v1 至启动 v2 共约 **13.09 秒**，其中 smoke 约 12.88 秒。源码、Stage1、方法参数及训练预算不变；v1 输出和全部回执保留，没有将缺失 DVAC 字段的旧池带入 v2。

## 精确路由

| 项目 | 路径或标识 |
|---|---|
| 新 root | `/data/chenyiteng/projects/rlinf-shenzhen/worktrees/rlt-dvac-scale4to2-20260915` |
| 新 run | `/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-scale4to2-phys6-fresh600-20260915-v2` |
| namespace | `RLinf_rlt_dvac_scale4to2_single_gpu6_fresh600_20260915_v2` |
| 分支 | `codex/sz-rlt-dvac-scale4to2-20260915` |
| 最终唯一运维阶段 | `/data/chenyiteng/results/server-maintenance-20260915/rlt-scale4to2/teacher-recording-fix` |
| 轻证据 | `docs/experiments/rlt-dvac-scale4to2-20260915` |

命令、完整 resolved、配置差异、环境与合同随轻证据发布。旧 Fast BC 精确目标为 `fastwam-turn-switch-bc8-u10-b1024-phys6-formal200-envresident-20260914-v1`，最后完整采集 **R29**，固定评估 **R25=14/32**，末代 checkpoint R20 留存。39 文件轻 ZIP 为 **1,222,028 bytes**，SHA256 `0ba989962bed829011942afe0b5bf6e651070411c0d551b6087efebed9a7f95b`，CRC 与 manifest 通过，含完整标量及五张独立图；模型与回放保留。[旧组云端归档，bbb2e125](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/bbb2e125aa4c47823883fd6df0d1697675b9a180/docs/experiments/fastwam-bc-turn-switch-closeout-20260915)。已完成的停止、启动和发布回执不可重复执行。
