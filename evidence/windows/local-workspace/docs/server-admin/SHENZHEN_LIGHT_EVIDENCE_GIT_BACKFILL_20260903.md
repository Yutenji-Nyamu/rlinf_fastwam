# 深圳历史实验轻量 evidence 回填账本 — 2026-09-03

## 合同

- 目标：把已结束深圳实验的轻量配置、命令、指标、TensorBoard、资源 CSV、关键日志、图和 manifest 提交到对应算法分支。
- 排除：checkpoint、模型、数据、视频、NPZ/NPY/HDF5、完整 Ray 日志及任何凭据。
- 活动中的 Sidney π0.5 formal 与 Fast-WAM renderer-fix resume 不进入本轮历史回填。

## 已完成

- `codex/sz-7d07a421-grpo-pi0-robotwin@02b94883...`：4 条 GRPO run，65 个 tracked evidence 文件，约 3 MB。
- `codex/sz-grpo-dvac-action-adv-fix@7006ad20...`：3 条 run，69 个文件，4,576,011 bytes。
- `codex/sz-prism-dvac-rank-rloo@76723092...`：4 条 run，88 个文件，3,223,014 bytes。
- `codex/sz-ppo-dvac-action-adv-fix@036e6bca...`：4 条 run，77 个文件，2,722,701 bytes。
- `codex/sz-pi05-robotwin-rl@ae7e5da7...`：7 条 run，131 个文件，2,743,358 bytes。
- `codex/sz-current-pi0-dvac-grpo-w0to5@ab098849...`：217 个文件，5,079,871 bytes。
- `codex/sz-grpo-dvac-action-adv@9fc8b403...`：旧 pre-fix Action-Adv，52 个文件，801,658 bytes。
- `codex/sz-st-dvac-local-shard@51dbeaeb...`：54 个文件，1,234,259 bytes。
- `codex/sz-ppo-pi0-robotwin@8d00c408...`：plain PPO evidence-only 分支，9 个文件，885,050 bytes。
- `codex/sz-rlt-pi0-robotwin-ar@3adc0d64...`：RLT Stage 1/2，102 个文件，10,478,624 bytes。
- `codex/sz-current-dsrl-pi0-robotwin@4ec52bd8...`：DSRL，61 个文件，4,523,851 bytes。
- `codex/sz-rlt-dvac-pure-single-gpu@30349428...`：单卡 RLT/Pure04，31 个文件，192,823 bytes。
- `codex/sz-fastwam-action-dvac-adv@b816d6d3...`：Fast-WAM Action-DVAC smokes，29 个文件，180,235 bytes。
- `codex/sz-fastwam-current-rlinf-grpo@4faade1d...`：12 条已结束的 plain Fast-WAM smoke/formal attempts，243 个文件，3,033,839 bytes；明确未纳入当前活动 resume。
- `codex/sz-sidney-pi05-current-rlinf@f50e235c...`：Sidney smoke，37 个文件，79,576 bytes。

每次回填均核对目标 worktree/branch/source HEAD、排除的大文件类型与凭据模式，并在 push 后确认远端 HEAD 与本地一致、worktree clean。原始日志按原样保存；日志中的尾随空格不做源码格式化。

## 暂缓

- 当前活动中的 Sidney π0.5 formal：结束后再回填最终轻量产物。
- 当前活动中的 Fast-WAM renderer-fix resume：结束后再回填最终轻量产物。
- `rlt-checkpoint-diagnosis`：存在诊断代码 dirty tree，且不属于算法正式结果，暂不混入 RLT 主分支。
- standalone/telemetry 观测数据不属于本轮“训练历史”批次，按需另行归档。
