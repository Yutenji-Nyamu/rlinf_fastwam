# Prism-style DVAC-RLOO：来源与 current 代码审计

> 日期：2026-08-26  
> 范围：只读论文、公开仓库、本地审计镜像与现有深圳文档；没有服务器写操作或实验操作。

## 1. 外部规划固定

- 原文件：`C:/Users/86136/Documents/seek/0826_prism_style_dvac_grpo_shenzhen_implementation_plan.md`
- 原文件 SHA-256：`60E00AC736D786F4EA26C1E7046D50231496B962BB890B28C109CC475DEDBE73`
- 专题副本：[`../references/0826_prism_style_dvac_grpo_shenzhen_implementation_plan.md`](../references/0826_prism_style_dvac_grpo_shenzhen_implementation_plan.md)
- 副本只修复 3 个损坏的 form-feed `frac` 控制字符，并加入 provenance；外部内容只作参考输入。

## 2. 论文与公开代码

- 论文：[Prism-GRPO, arXiv:2608.17423v1](https://arxiv.org/html/2608.17423)
- 论文公开底座：[PRIME-RL/SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)
- 本轮未在论文、作者链接或公开底座中找到 Prism-GRPO 官方实现；不能假设某个私有实现的接口细节。

论文直接支持：

- $R=success+\lambda q$，$q\in[0,1]$，默认 $\lambda=0.2$；
- 同 scene 的 $G=8$；
- RLOO 不除组内 std；
- quality 应用于所有 group；
- SFT 256 validation scenes 做一次固定 calibration；
- formal 以 adaptive gap-fill 保持固定 retained batch，generated rollouts 单独计数；
- Binary RLOO 是隔离 advantage estimator 的正式 baseline。

论文没有规定：DVAC 如何聚合、方向、$r_0,T$，或是否可用同组 rank 代替固定 calibration。
这些都属于本专题的新设计，不能冒充论文原定义。

## 3. current 源码锁

- official base：`7d07a421...`
- 深圳 current DVAC 分支：`0e28ac6f...`
- 本地只读审计镜像：`references/rlinf_fastwam_audit_20260824/worktrees/sz-current-dvac-grpo-w0to5`
- 镜像 checkout HEAD 为旧提交并有 4 个工作树 diff；被审计生产文件的内容对应已存在的 `0e28ac6f...`。
  真正实施仍必须从服务器 clean `0e28ac6f...` 建新 worktree，不能把审计镜像直接当源工作树。

## 4. 调用链证据

| 位置 | 现场含义 | Prism 落点 |
| --- | --- | --- |
| `openpi_action_model.py:1040-1148` | `sample_actions` 可产出 `z_endpoint[B,4,50,14]` | 不改模型 |
| `huggingface_worker.py:627-732` | 现有 ST apply 时计算 `V_L3[B,H]` | 独立增加 Prism collection trigger |
| `embodied_types.py` + builder | `forward_inputs` 在 typed chain 原样传递 | 不改 schema |
| `embodied_fsdp_actor_worker.py:273-348` | 生成 terminal mask；binary filter 只 AND loss mask | Prism 首版关 binary filter |
| `embodied_fsdp_actor_worker.py:350-385` | advantage 的唯一正确前置入口 | 在这里计算/传入 $u,q$ |
| `embodied_fsdp_actor_worker.py:547-610` | 现有 recent-5/global-z ST，发生在 advantage 之后 | 首版不复用 |
| `advantages.py:89-126` | current GRPO 做 group mean/std 标准化 | 新注册 RLOO，不改旧函数 |
| `metric_utils.py:516-537` | action-level termination mask | executed-$h$ 聚合直接复用 |
| `robotwin_env.py:446-498` | reset seed 按 `group_size` 连续 repeat | 连续 G8 是同 scene |
| `nested_dict_process.py:251+` | 同一递归顺序合并并 shuffle 全字段 | 不建旁路索引 |

## 5. 两卡 v2 control 锁

依据：

- [`../../rlinf-shenzhen-pi0-ppo-rlt/evidence/GRPO_DUAL_2GPU_FORMAL_LEDGER_20260826.md`](../../rlinf-shenzhen-pi0-ppo-rlt/evidence/GRPO_DUAL_2GPU_FORMAL_LEDGER_20260826.md)
- [`../../rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md`](../../rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md)

必须继承：两卡、64 train env、4 rollout epochs、256 trajectories、G8/32 groups、max1024 records、
`B1024/MB32/update2`、fixed32/eval5、save10、formal100。

## 6. 已识别的语义风险及收口

1. 外部 rank 公式只有零基 rank 才能用 $1-rank/(G-1)$；主计划已明确零基与 tie 规则。
2. terminal partial chunk 不能把未执行 future-$h$ 算入 execution quality；复用 action loss mask。
3. success 使用二值 episode success，不使用累计 reward sum。
4. 只关闭 binary group filter；terminal/padding mask继续生效。
5. Binary GRPO 对完整方法同时改变 quality/RLOO/filter；有效后补 Binary RLOO。
6. current 没有 refill；首版不声称复现论文 rollout savings。
7. group-rank stateless；不复用现有 ST recent-5 sidecar。

## 7. 当前边界

本轮只完成规划。任何真实 smoke/formal 前，都必须另行提交 resolved config、命令、路径、预算、资源、
监控与停止条件，并取得用户批准。
