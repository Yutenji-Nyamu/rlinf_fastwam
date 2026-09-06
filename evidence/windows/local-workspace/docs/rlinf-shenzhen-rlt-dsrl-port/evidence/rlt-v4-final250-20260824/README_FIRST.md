# RLT Stage 2 v4 final-250 轻量证据

## 一句话结论

深圳 current-AR RLT Stage 2 在 2×H100、8 train env 下自然完成 `250/250`，`exit_code=0`；最终 train=`7/8`、student-only fixed-20=`18/20`、`update_step=104000`，无 fatal、OOM 或 non-finite。

## 建议先看

1. `01_rlt_final_learning_and_eval.png`：逐步 train、5/20 步滑动平均、10 个 fixed-20 点、replay 与 online update 时间线。
2. `02_rlt_final_optimization_and_timing.png`：actor/critic loss、梯度和 rollout/update 耗时。
3. `03_rlt_final_resources.png`：GPU 4/5 显存与利用率、整机可用主存。
4. `summary.json`：机器可读的最终指标摘要。

## 可复核数据

- `rlt_v4_metrics_step1_250.csv`：从原始 driver 表格解析的 250 个完整 step。
- `rlt_v4_fixed20.csv`：Step 25、50、…、250 的离散 student-only fixed-20。
- `rlt_v4_driver.log`、`rlt_v4_metrics.log`：原始文本日志。
- `rlt_v4_resource.csv`：约 10 秒一次的原始资源采样。
- `rlt_v4_events.out.tfevents.*`：原始 TensorBoard event。
- `rlt_v4_resolved.yaml`、`rlt_v4_command.txt`、`rlt_v4_launch_manifest.txt`：精确配置与启动合同。
- `rlt_v4_started_at.txt`、`rlt_v4_finished_at.txt`、`rlt_v4_exit_code.txt`：运行边界。

## 不在轻量包中的大产物

服务器最终 checkpoint 位于：

`/data/chenyiteng/results/rlinf-rlt/formal-current-ar-stage2-8env250-20260824-v4-warmup-fix/robotwin_adjust_bottle_rlt_stage2_current_ar_8env250_optimizer_warmup_v3/checkpoints/global_step_250`

轻量包不复制 checkpoint 权重、replay、视频或 Ray 全量日志；因此可以快速下载和审阅，但不能单独用于模型恢复。

## 口径提醒

- 每步 train success 是 8 条 reference/student route 训练 rollout，不等价于纯 student 评估。
- fixed-20 是 student-only 评估，图中仅显示真实离散点，不连线插值。
- 在线 AC 从 Step 137 开始；Step 25–150 fixed-20 均为 0，之后为 `9/20 → 18/20 → 19/20 → 18/20`。
