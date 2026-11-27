# 在哪里取权重

1 机的 `$DATA_USER`、`$HOME` 分别表示本人 Data 目录和本人 Home。下表列的是已有文件的定位符。11 月 27 日 11:28–11:35 确认下列七个策略文件均存在且非空，两条 DSRL 的最佳策略和完整检查点完成标记也在。此次核查只读文件元数据。

| 实验 | 服务器位置 |
|---|---|
| RT2 GRPO · 移药瓶 Clean 第 140 轮 / 加权第 70 轮 | `$DATA_USER/models/protected-eval/pillbottle/{clean_c17_cp140,grpo_u_tau3_cp70}/full_weights.pt`。同口径 C20 配对复评分别 45/128、77/128；Clean 的 C50 结果 72/128 另列。 |
| RT2 RLT · 按铃加权第 2750 轮 | `$DATA_USER/models/protected-eval/rlt-click-bell-20261126/global_step_2750/rlt-policy.pt` |
| Dojo RLT · 丢瓶子纯模仿第 16 轮 | `$DATA_USER/models/protected-eval/stability-retry-20261110/dojo-rlt-clean-g6/global_step_16/rlt-policy.pt` |
| RT2 RLT · 拨开关第 270、614 轮 | `$DATA_USER/models/protected-eval/dsrl-rlt-retry-20261108/rt2-rlt-clean-g7/global_step_{270,614}/rlt-policy.pt` |
| RT2 RLT · 从头复验拨开关第 62 轮 | `$DATA_USER/models/protected-eval/fresh400-20261112/g7/global_step_62/rlt-policy.pt` |
| Dojo DSRL · 丢瓶子 | `$HOME/models/protected-eval/success-registry-20261119/g5/checkpoints.json` 索引最佳第 18 轮与完整第 97 轮。 |
| RT2 DSRL-NA · 拨开关 | `$HOME/models/protected-eval/success-registry-20261119/g7/checkpoints.json` 索引最佳第 124 轮与完整第 132 轮。 |

RLT、DSRL 的小策略评估须配套同一基础 VLA、任务配置及所需表征。续训另需优化器、训练状态和适用的回放。11 月 21 日的冷归档还登记了两路按铃的评估点 625、1125 和恢复点 775、2075，以及 Dojo 丢瓶子纯模仿恢复点 18、RT2 拨开关恢复点 610。其余任务可直接查[成功实验索引](README.md)中的配置和成绩，原始权重按各自收尾记录定位。

[配对复评存档](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/e0316f2154fd34ccd6f3caa6fcc2b4abec7b5998/docs/pi05_policy_eval/completed-results-20261018/README.md)

