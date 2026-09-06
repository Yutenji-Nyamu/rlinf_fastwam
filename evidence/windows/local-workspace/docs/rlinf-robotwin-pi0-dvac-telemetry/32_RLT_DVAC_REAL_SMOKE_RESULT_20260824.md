# RLT teacher-DVAC：真实 smoke 结果

日期：2026-08-24  
状态：**真实两卡smoke通过；正式训练尚未启动。**

## 1. 结论先行

RLT teacher-DVAC的真实链路已经闭环：冻结π0 teacher产生DVAC，信号随transition进入replay，冻结global-z
baseline后生成C10逐action权重，student的Q分支用这些权重反传；训练、fixed-20评估、指标归约、资源记录和
`global_step_1`完整保存均自然结束，driver `exit=0`。

这说明实现可以进入正式训练前的参数审阅。它不说明方法已经提升成功率：本次只有一个cycle，fixed-20为0/20
是几乎未训练student的smoke结果。

## 2. 与成功RLT对齐了什么

主体沿用历史成功RLT的2×A800、8 train env、C10、episode上限200、4 eval env×5 epoch=fixed-20、原
actor/critic/BC-Q目标、replay与优化器。新增项只有：

1. 同一次冻结π0 reference ODE旁路保存H50的`V_L2/L3/L4`，不增加teacher forward；
2. 选择L3前C10，冻结global-z后映射为`w=1+0.5 clip(z,-2,2)`，范围`[0,2]`；
3. 只在student action进入Q分支前改变反向倍率，BC和critic TD不加权；
4. 增加DVAC训练指标和低频NPZ。

smoke-only覆盖仅用于在一个cycle内走到apply并保存：`max_steps=1`、`val/save=1`、缩短warmup/update预算。
正式配置里的`warmup_min_size=10000`、正式保存/评估间隔和总预算没有被改写。

## 3. v1为何失败，v2改了什么

v1的模型与加权路径已运行，但新增telemetry按每个rank本地batch是否含正奖励或某种route而条件性增删字段。
RLinf最终把各rank本地字段拼成tensor做all-reduce；一个rank有105项，另一个已经进入下一次1元素barrier，
于是NCCL等待30分钟后超时。

窄修复只把四个条件分组改为始终输出固定`sum/count`，跨rank归约后再求mean。新增一个空组/非空组测试，
相关单测由`4 passed`变成`5 passed`。DVAC公式、`[0,2]`映射、student-Q挂点以及RLT参数均未改变。

代码提交：

- 主体：`275ab4535f8363e15b839cbb64e7d997a056d01e`；
- 固定schema修复：`513dbcb7f31ebb639afa5267dff218028d07187b`；
- branch：`personal/codex/rlt-teacher-dvac-weighting`，已push并与服务器worktree一致。

## 4. v2 smoke实测

| 项 | 结果 |
|---|---:|
| wall | 8分13秒 |
| train rollout | 8 episodes，2成功 |
| fixed-20 eval | 0/20 |
| critic / actor updates | 8 / 4 |
| replay trajectories | 72 |
| actor / critic grad norm | 3.804 / 3.259 |
| baseline count / mean / std | 1440 / -4.8754 / 0.5060 |
| run weight p05 / median / mean / p95 | 0.263 / 0.959 / 0.991 / 1.820 |
| run weight ESS / top-20% mass | 0.880 / 0.298 |
| GPU峰值 | 17.11 / 17.19 GiB |
| cgroup峰值 | 57.18 GiB |
| OOM / OOM-kill | 0 / 0 |
| checkpoint | `global_step_1`完整，约54 MiB |

8个低频trace共320个权重，per-query ESS均值`0.887`、top-20% mass均值`0.295`，并真实命中0与2边界；
因此这不是只有配置生效而权重仍全1。两rank checkpoint都保存`update_step=8`和完全一致的冻结baseline，
complete manifest为`complete=true`。

## 5. 正式预算必须怎样写

用户记得的“成功RLT是480步”是对的，但准确过程是两段：

1. fresh Stage 2先跑`cycle 1--250`，自然`250/250`、exit0；
2. 从完整`global_step_250`严格resume，继续`251--480`，自然`480/480`、exit0。

历史总计3,840 train episodes、57,410 replay rows、215,055 critic updates和107,528 actor updates；续训阶段
fixed-20评估合计`178/200=89%`，最终点`17/20`。所以若正式DVAC实验要完整对齐历史预算，启动合同应明确写成
`fresh 250 -> resume到绝对480`，而不是一次fresh 480，也不是只跑250就称为完整对齐。

## 6. 产物入口

- [v2成功smoke轻量证据](evidence/rlt_dvac_real_smoke_8env1c_20260824_v2/README.md)
- [逐命令流水账](evidence/RLT_DVAC_REAL_SMOKE_LEDGER_20260824.md)
- [实现与记录计划](28_RLT_DVAC_IMPLEMENTATION_AND_RECORDING_PLAN_20260823.md)
- [历史480-cycle正式结果](../rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md)

正式训练尚未启动；下一轮只需审阅正式resolved config、精确命令、输出目录、两段预算与资源观察项。
