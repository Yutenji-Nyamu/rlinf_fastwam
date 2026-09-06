# Prism Step 10 checkpoint 修复与 formal v2 流水账

日期：2026-08-27。目标：只修复 Prism formal 的 checkpoint 协调边界，保持两卡 Control 和全部科学参数不变。

## 已授权边界

- 精确停止卡死的 Prism v1；不停止 Control、不重启 shared Ray、不碰其他用户。
- 修改、检查并推送 Prism worktree；GPU 6/7 fresh 重启 100-step formal。
- 启动后只确认配置、进程和首轮推进，不持续盯守。

## 问题与选择

- v1 在 Step 10 eval 后进入 DCP/FSDP optimizer-state 收集，两个 rank 未完成会合；目录没有 shard，排除普通慢写盘。
- current RLinf 已有 `dcp` 与 `local_shard` 两条正式保存/恢复路径；`local_shard` 已被现有算法使用，但 FSDP model manager 没有把配置叶子传给 strategy。
- 最小修复：manager 读取 `actor.fsdp_config.checkpoint_format`，校验 `dcp|local_shard`，save/load 对称透传。默认仍是 `dcp`；只有 Prism v2 显式选择 `local_shard`。
- 这绕开本次卡住的 DCP optimizer-state collective，但不声称修复了 PyTorch 内部偶发根因。local shard 严格恢复要求相同 world size/FSDP 拓扑。

## 允许的 v2 差异

相对 Prism v1：

1. `actor.fsdp_config.checkpoint_format=local_shard`；
2. 新 run/experiment/video/save/telemetry 路径与名称。

相对两卡 Control：三项 Prism 方法叶子（`adv_type`、`filter_rewards`、`prism_dvac.enabled`）、上述 checkpoint 工程叶子，以及必要的设备/路径/名称。采样、batch、update、eval、checkpoint 周期和 100-step 预算全部相同。

## 逐条操作记录

1. 只读现场确认：v1 wrapper 与两个 actor 存活但 Step 10 checkpoint 数小时无 shard；Control 同一 shared Ray 上继续保存并推进。
2. 代码审计：确认 `strategy/base.py` 与 `strategy/checkpoint.py` 已完整支持 `local_shard`；不修改底层格式。
3. 已执行：一文件补丁通过 `git diff --check`、`py_compile` 与 Hydra compose；commit
   `306ce2e98a06b6f439a1070d8942e20132e48d49` 已普通 push 到 `codex/sz-prism-dvac-rank-rloo`。
4. 已执行：v2 resolved packet 生成成功。相对 v1 只出现 checkpoint 格式和新路径/名称；相对同代码
   Control 只有3项 Prism 方法叶子；相对历史 Control 的其余差异均在设备、代码路径、Prism schema 与
   checkpoint 工程白名单内，unexpected=0。
5. 已执行：精确停止 v1 owned PGID，清理 Ray job `50000000` 的 `RLinf_1` 共15个 actor；shared Ray、
   GPU4/5 Control 与其他用户不动。v1 最后完整 Step 9，不完整 Step10 目录保留。GPU6/7切换空窗4秒。
6. 已执行：2026-08-28 00:01 CST（resource记录为`2026-08-27T16:01+00:00`），v2 fresh 启动于GPU6/7，wrapper PID=`2485082`；Control PID=`3882371`
   同时存活。resolved 明确含 `checkpoint_format: local_shard`，两个 FSDP actor 与env/rollout actors已创建，
   无fatal；启动检查时模型仍在加载。Step10是否真正写出两份local shard留待后续只读刷新。

## 精确产物

- run：`/data/chenyiteng/results/rlinf-shenzhen/grpo/runs/prism-dvac-rank-rloo-formal100-2gpu64x4-b1024-fixed32-eval5-localshard-phys67-v2`
- packet：同名目录位于 `/data/chenyiteng/results/rlinf-shenzhen/grpo/packets/`，含resolved、same-code control、
  parity、contract和精确命令。
- v2预计checkpoint布局：`actor/local_shard_checkpoint/checkpoint_rank_0.pt` 与
  `checkpoint_rank_1.pt`；同时保留现有full weights。恢复时必须使用同样的两rank/FSDP配置和
  `checkpoint_format=local_shard`。
