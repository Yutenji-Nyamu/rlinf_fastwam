# Dojo π0.5 GRPO · 丢瓶子

任务为 `put_bottles_into_dustbin`。先评估原始 Dojo π0.5 seed 1 / 59999，再从同一权重开始 200 轮 GRPO。两阶段共享已加载的模型和模拟器。

| | 初始评估 | 正式训练 |
|---|---|---|
| 每批 | 32 局，4 批共 128 局 | 64 局，2 批共 128 局/轮 |
| 布局 | 沿用定点评估，seed 1 的固定面板重复采样，逐局计数 | seed 0、2；同一布局 8 局一组 |
| 执行 | 原生 700 步上限，π0.5 每 50 步重新取观测；评估逐步渲染 | 同一步数与动作块；训练仅在动作块边界渲染 |
| 更新 | 固定原模型 | 学习率 5e-6，clip 0.2，每轮两遍更新，微批 32 |

700 步对应 14 个动作块，训练的全局 batch 随每轮总块数从 1024 调到 1792，以维持相同的优化器更新次数。原生成功条件定义在 [RoboDojo 任务类](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/main/task/RoboDojo/tasks/put_bottles_into_dustbin.py)。

四批各自的成功局数记入 `baseline-batches.jsonl`。合计记入 `initial-evaluation.json`，分母为实际执行的 128 局。开始训练后第 2 轮及以后每 10 轮独立评估 32 局。完整恢复点滚动保存，最优模型另存。
