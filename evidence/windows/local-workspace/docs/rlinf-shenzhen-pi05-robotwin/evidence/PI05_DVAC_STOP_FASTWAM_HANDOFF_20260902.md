# π0.5 DVAC 停止与 Fast-WAM 讨论交接（2026-09-02）

## 授权动作

用户判断 π0.5 clean GRPO 的有效性仍需继续观察，明确要求先停止 π0.5 GRPO-DVAC；本轮不授权启动
Fast-WAM formal。

## 精确停止结果

- 2026-09-02 08:47:53 CST，只停止 matched-update2 π0.5 DVAC：
  `pi05-grpo-dvac-action-adv-w0p5to1p5-formal100-2gpu64x4-g8-b1024-u2-m5-fixed32-eval5-phys67-localshard-v2`。
- 最后完整步为 Step 26；日志、运行目录和已有 checkpoint 原样保留。
- 停止范围为该 run 的 owned wrapper/process tree 与 exact Ray namespace `RLinf_1`；未重启 shared Ray。
- 停止后 `RLinf_1=0`，GPU6/7 降至约 11 MiB/card。

## 保留的 Control

- π0.5 Control wrapper PID 仍为 `1477572`，Ray job仍为 `3f010000`，namespace `RLinf` 仍有15个 actors。
- 08:49 CST Control 已完整 Step 27并继续运行；GPU4/5约68.3/68.1 GiB。
- 主机 available 从停止前约417 GiB回升到约1.16 TiB；`/data`剩约1.7 TiB。

## Fast-WAM 当前边界

- plain current GRPO 已在分支 `codex/sz-fastwam-current-rlinf-grpo`、HEAD
  `7b2331c55d14397cfb4cb16181470ddc8afae44a` 完成真实两卡 strict-resume smoke。
- 首条 formal 候选仍为 `move_stapler_pad`、两卡、`32 env × rollout4`、G8、128 trajectories、
  1024 query records、GB1024/MB2/update2、fixed32/eval5/save10、DCP。
- 本轮只讨论参数，没有启动 Fast-WAM formal。
