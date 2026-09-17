# 指数 DVAC 的 chunk dropout 与 α 退出

基于已发布 linear controls 分支 `6600987ad6b33f02752b5f205303c4e07ac3f840` 的小改动：同一控制模块现在接受 `linear_centered` 和 `exp_mean`，均要求 `two_level_group`。默认设置、旧 linear recipe、数学映射和 actor/loss 调用顺序保持。

## 本次方法配置

使用 `dvac_grpo=adv_exp_positive_t2_drop02_anneal200`。该 recipe 继承 `adv_exp_positive_t1` 的字段和值，仅双温度改为 2 并启用两项控制：

- `mapping=exp_mean`，双 τ 固定 2；`scope=positive` 只作用于正优势 A>0 的有效 chunk。
- `alpha_local=alpha_chunk=1`；两层独立调度都从 R1 的 1 线性降至 R200 的 0，随后保持 0。版本 `runner.version=0` 对应 R1，版本 199 对应 R200。R100 的双 α 为 100/199，约 0.502513。
- `chunk_dropout.enabled=true`、`probability=0.2`、`seed=42`：每轮约 20% 的适用 chunk 将最终权重恢复为 1，继续原 GRPO 训练；不是屏蔽整段梯度或丢弃样本。

两层先各自用有效 α 映射，乘积得到权重 W，再做整 chunk 的 Bernoulli 回退 `1+m(W-1)`。没有 `1/(1-p)` 补偿，没有 mask 后重新归一。随机 mask 由配置 seed 和绝对轮数决定，先在全局同序 rollout 上生成再切各 rank；同一轮的 U2、microbatch 重用已冻结权重，不消耗策略 RNG。

当双 α 为 0 时，所有 W 精确为 1，同批 loss 和梯度回到 Clean 公式；模型和优化器历史仍是本次训练产生。τ 不退到 0。总预算为 200 轮时，R200 归零表示只有最后一轮完全退出；不能据此评估长期退出后的保持能力。

## 恢复及兼容

保留公开函数名 `linear_controls_contract`、`effective_linear_alphas` 和 checkpoint 字段 `linear_controls`，避免改变已发布 linear checkpoint 契约。关闭控制项不增加契约字段，旧 linear/exp checkpoint 仍按原规则恢复。启用后契约锁定概率、seed、各层启停/起止轮/目标 α，同时原 exp 契约锁定 τ；更改这些字段的恢复会被拒绝。runner 恢复绝对 version 后继续调度和可复现 mask，不重新从 R1 开始。

本 recipe 只覆盖方法配置。正式训练应从 Clean128 的实配继承方法外字段：128 轨迹/轮、相同初始模型与种子、batch/microbatch/update、优化器、评估、保存频率、总 200 轮；新输出身份和物理卡位单列。不得从旧 linear 模型恢复再切换此方法。

## 针对性检查

服务器运行原 DVAC 套件；本次扩展现有测试覆盖两种映射、p=0/1/.2、α 两层独立边界、R1/100/200/201、真实 actor 调用与同批 Clean loss/梯度等价、真实两 rank Gloo、恢复的绝对 version/mask 及 τ/控制契约。旧 linear 用例全部保留。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m pytest -q \
  tests/unit_tests/test_dvac_linear_controls.py \
  tests/unit_tests/test_dvac_two_level.py \
  tests/unit_tests/test_dvac_exp_mean.py \
  tests/unit_tests/test_dvac_adv_new_actor.py \
  tests/unit_tests/test_dvac_adv_new_loss.py
```

该文档描述实现及验收要求；测试、推送、GPU smoke 和正式启动结果以随附现场回执为准。
