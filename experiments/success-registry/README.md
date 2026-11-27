# 成功训练索引

按实验查实际配置、学习器源码、独立评估和检查点。本文整理深圳系列中已登记的成功对照，覆盖 2026 年 8–11 月本轮核对的记录。链接固定到对应提交。

## RoboTwin 2.0 · π0.5 clean RLT

11 月 26 日覆盖表标记了 8 个成功任务。各行保留原评估局数与训练分段。[50 项 RT2 任务逐评数据](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/experiments/rt2-fastwam-switch-192/microbatch-resume/clean-coverage.json) · [1 机逐轮曲线](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/experiments/rt2-fastwam-switch-192/microbatch-resume/rlt-series.json)

| 任务 | 已登记成绩 | 实现与配置 |
|---|---|---|
| 调瓶子 | 第 800 轮收尾，末五评均 20/20。 | [实配](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/460008008aea310f83a0b19008c7c105a0a19980/examples/embodiment/config/pi05_rlt_clean_20260919.yaml) · [更新代码](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/460008008aea310f83a0b19008c7c105a0a19980/rlinf/workers/actor/fsdp_sac_policy_worker.py) |
| 按铃 | 普通 RLT 末五评均 20/20。 | [N8 起始配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/3f35e89b9a9aa61d66aba2dfead41aebc2ee19cc/examples/embodiment/config/pi05_rlt_click_bell_clean_n8full_20261002.yaml) · [更新代码](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/3f35e89b9a9aa61d66aba2dfead41aebc2ee19cc/rlinf/workers/actor/fsdp_sac_policy_worker.py) · [1 机逐轮曲线](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/experiments/rt2-fastwam-switch-192/microbatch-resume/rlt-series.json) |
| 按订书机 | 历史最好 20/20，末五评平均 98%。 | [实配](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/59b7072f8135522e0aaf378cd8380b0110ac3aff/examples/embodiment/config/pi05_rlt_press_stapler_clean_20260927.yaml) · [更新代码](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/59b7072f8135522e0aaf378cd8380b0110ac3aff/rlinf/workers/actor/fsdp_sac_policy_worker.py) · [原始指标](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/59b7072f8135522e0aaf378cd8380b0110ac3aff/docs/server-admin/sz1-rlt-press_stapler-closeout-20260928/clean-metrics.json.gz) |
| 拨开关 | 9 月 clean 最高 17/20，停训前五评平均 66%。11 月从头复验在第 62、108 轮均为 12/12。 | [9 月配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/fe982ce1c2b9073406eb018158b60152f2279ed1/examples/embodiment/config/pi05_rlt_turn_clean_20260923.yaml) · [更新代码](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/fe982ce1c2b9073406eb018158b60152f2279ed1/rlinf/workers/actor/fsdp_sac_policy_worker.py) · [9 月曲线](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/c153f162ecad4bd53cefa673353ec1b458bc203d/docs/server-admin/closeout-sz1-rlt-turn-20260925/clean-metrics.json) · [11 月 8 日配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/0b604854ede7cb24f4d2644220d9cf037c83bc80/docs/dojo_rl/evidence/recipe-retry-20261108/rt2-rlt-clean-g7/runtime-resolved.yaml) · [历次并行与更新配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/experiments/rt2-fastwam-switch-192/microbatch-resume/rlt-exploration.json) |
| 移罐入锅 | 早期 20 局固定评估最高 12/20，9 月末一个续训段的末五评平均 45%。 | [实配](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/fe982ce1c2b9073406eb018158b60152f2279ed1/examples/embodiment/config/pi05_rlt_can_clean_20260923.yaml) · [更新代码](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/fe982ce1c2b9073406eb018158b60152f2279ed1/rlinf/workers/actor/fsdp_sac_policy_worker.py) · [收尾记录](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/a81eb97f81dd635d7c5af7be1d9418a1d88f3aa4/docs/server-admin/sz1-rlt-can-closeout-20260927/SUMMARY.json) |
| 双手抓瓶 | 9 月 N8 完整配置最高 12/20。11 月 7 卡最高 26/48。 | [实配](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/55c1399a50826d61e8735a64daa2f1742f1b824f/examples/embodiment/config/pi05_rlt_pick_dual_bottles_clean_n8full_20260929.yaml) · [更新代码](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/55c1399a50826d61e8735a64daa2f1742f1b824f/rlinf/workers/actor/fsdp_sac_policy_worker.py) · [1 机逐轮曲线](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/experiments/rt2-fastwam-switch-192/microbatch-resume/rlt-series.json) |
| 移开扑克牌 | 2 机历史最高 19/20，末五评平均 70%。 | [本次补档的原配置](configs/move_playingcard_away.yaml) · [生产父版本的更新代码](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/e5cb9aa4c767490a66cba3c4e155c76f1193f199/rlinf/workers/actor/fsdp_sac_policy_worker.py) |
| 用锤敲积木 | 3 机首段最高 15/20、末五评平均 65%，后续第 375 轮记录到 16/20。 | [实配](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2f484040dbf078fcb6bfcc1da8a300fbc625f87b/examples/embodiment/config/pi05_rlt_beat_block_hammer_clean_n8full_20261002.yaml) · [更新代码](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2f484040dbf078fcb6bfcc1da8a300fbc625f87b/rlinf/workers/actor/fsdp_sac_policy_worker.py) |

移开扑克牌的生产分支为 `codex/sz2-pi05-rlt-next-six-20261002`。本次从 10 月 2 日保存的输入配置与部署差异还原 YAML，原文与部署回执逐字节一致。公开副本仅将机器绝对路径替换为占位路径。锤积木的生产分支为 `codex/sz3-pi05-rlt-next-six-20261002`，配置链接直接指向仍可访问的生产提交。

## DSRL 与单独登记的 RLT 结果

| 实验 | 为什么保留 | 实现、配置与产物 |
|---|---|---|
| Dojo DSRL · 丢瓶子 | 高成功率对照。34 次独评的前、中、后段均值 70.7%、72.4%、72.2%。最佳第 18 轮 27/32，第 72 轮同分。完整恢复点第 97 轮。 | [实现清单](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/6b200dd643d70146abcc324d3ec1a0c6140662d7/experiments/success-registry/2026-11-19/dojo-dsrl-put-bottles/implementation.json) · [起始配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/6b200dd643d70146abcc324d3ec1a0c6140662d7/experiments/success-registry/2026-11-19/dojo-dsrl-put-bottles/initial.yaml) · [登记时配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/6b200dd643d70146abcc324d3ec1a0c6140662d7/experiments/success-registry/2026-11-19/dojo-dsrl-put-bottles/current.yaml) · [曲线](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/6b200dd643d70146abcc324d3ec1a0c6140662d7/experiments/success-registry/2026-11-19/dojo-dsrl-put-bottles/success.json) · [权重登记](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/6b200dd643d70146abcc324d3ec1a0c6140662d7/experiments/success-registry/2026-11-19/dojo-dsrl-put-bottles/experiment.json) |
| RT2 DSRL-NA · 拨开关 | 阶段性提升。同口径三段均值 38.1%、48.2%、47.7%。最佳第 124 轮 21/32。完整恢复点第 132 轮。 | [实现清单](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/6b200dd643d70146abcc324d3ec1a0c6140662d7/experiments/success-registry/2026-11-19/rt2-dsrl-na-turn-switch/implementation.json) · [起始配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/6b200dd643d70146abcc324d3ec1a0c6140662d7/experiments/success-registry/2026-11-19/rt2-dsrl-na-turn-switch/initial.yaml) · [登记时配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/6b200dd643d70146abcc324d3ec1a0c6140662d7/experiments/success-registry/2026-11-19/rt2-dsrl-na-turn-switch/current.yaml) · [曲线](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/6b200dd643d70146abcc324d3ec1a0c6140662d7/experiments/success-registry/2026-11-19/rt2-dsrl-na-turn-switch/success.json) · [权重登记](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/6b200dd643d70146abcc324d3ec1a0c6140662d7/experiments/success-registry/2026-11-19/rt2-dsrl-na-turn-switch/experiment.json) |
| Dojo RLT · 丢瓶子纯模仿 | 第 16 轮学生独立执行 63/64。登记为模仿能力对照，训练阶段 Q 权重为零。 | [训练配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/05bd167274b0ccf08d1b083e7e9464dec8cbcce4/tools/stability_retry_20261110/evidence/formal-g6.yaml) · [实现与训练设置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/05bd167274b0ccf08d1b083e7e9464dec8cbcce4/tools/stability_retry_20261110/README.md) · [权重定位](weights.md) |
| RT2 RLT · 按铃加权版 | 7 卡末八评连续 20/20，第 2750 轮权重受保护。普通按铃单列于上表。 | [1 机逐轮曲线](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/experiments/rt2-fastwam-switch-192/microbatch-resume/rlt-series.json) · [训练记录](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/dbf2e488f41ba9ffe1489b653d1f7a0f0e1ecd83/experiments/rt2-fastwam-switch-192/microbatch-resume/README.md) · [权重定位](weights.md) |

## GRPO 与 PPO 的历史对照

| 实验 | 独立评估 | 实现、实配与证据 |
|---|---|---|
| RT2 π0.5 GRPO · 移药瓶，每轮 128 局 | Clean 第 140 轮，在 C50、200 动作口径的配对复评为 72/128。 | [实配](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/b51f185295e06bf22fbf51e94de05a15e8fef4eb/docs/rlinf-shenzhen-grpo-dvac-action-adv/evidence/grpo-seed42-half-20260911/prepared/resolved.yaml) · [生产源码版本](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/b51f185295e06bf22fbf51e94de05a15e8fef4eb/docs/experiments/pi05-grpo-clean-half128-seed42-closeout-20260912/source-identity.json) · [原训练曲线](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/b51f185295e06bf22fbf51e94de05a15e8fef4eb/docs/experiments/pi05-grpo-clean-half128-seed42-closeout-20260912/metrics.json) · [权重定位](weights.md) |
| RT2 π0.5 GRPO · 移药瓶，每轮 256 局 | 第 70 轮 26/32，第 165 轮 24/32，末十评平均 63.4%。 | [实配](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/1d015a2aa03ec8132d8207ba47a2be3dbe1d9591/evidence/pi05-grpo-closeout-20260906/raw/runtime/resolved.yaml) · [168 轮收尾与原始产物](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/1d015a2aa03ec8132d8207ba47a2be3dbe1d9591/evidence/pi05-grpo-closeout-20260906/README.md) |
| RT2 π0 GRPO / PPO · 调瓶子 | GRPO 第 10、20、40、50 轮依次为 57、60、63、62 / 64。同期 PPO 第 10、20 轮为 58、62 / 64。 | [固定 64 局原记录](early-controls.md) · [PPO 基础配置](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/c41b7ff0008574799636abb37783ffbf16c6be11/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi.yaml) |

## 分支与权重

点上表的文件可直接到具体提交。

| 内容 | 大仓分支或生产版本 |
|---|---|
| 两条 DSRL 的逐实验台账 | [codex/rlt-trials-20261119](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/rlt-trials-20261119) |
| RLT 50 项覆盖表和曲线 | [codex/rt2-fastwam-microbatch-20261126](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/rt2-fastwam-microbatch-20261126) |
| RT2 RLT 的两任务实现 | [codex/sz1-pi05-rlt-two-tasks-dv-20260923](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz1-pi05-rlt-two-tasks-dv-20260923) |
| 11 月拨开关 RLT | [codex/dsrl-rlt-recipe-retry-20261108](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/dsrl-rlt-recipe-retry-20261108) |
| 双手抓瓶 N8 | [codex/sz1-pi05-rlt-n8full-20260929](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz1-pi05-rlt-n8full-20260929) |
| 10 月按铃 N8 | [codex/sz1-pi05-rlt-next-six-20261002](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz1-pi05-rlt-next-six-20261002) |
| Dojo 学生纯模仿 | [codex/stability-retry-20261110](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/stability-retry-20261110) |
| π0.5 GRPO，每轮 128 局 | [codex/sz-pi05-grpo-clean-half-seed42-20260911](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz-pi05-grpo-clean-half-seed42-20260911) |
| π0.5 GRPO，每轮 256 局 | [codex/sz-sidney-pi05-current-rlinf](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz-sidney-pi05-current-rlinf) |

[策略权重和完整恢复点](weights.md) · [1 机 11 月 27 日训练中断记录](../operations/sz1/2026-11-27/README.md)
