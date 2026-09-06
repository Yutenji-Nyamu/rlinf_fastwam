# DSRL × π0 × RoboTwin 工作材料包

本包对应 2026-07-28 至 2026-07-29 的
`adjust_bottle / π0 / DSRL / H=50 / N=20` 实验。

## 入口

1. `workspace/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_CLOSEOUT_REPORT_STEP198_20260729.md`
   ：最终结论、指标解释、资源与样本效率讨论。
2. `workspace/docs/rlinf-robotwin-pi0-traditional-rl/00_INDEX_AND_IMPLEMENTATION_PLAN.md`
   ：当前设计与实现单一事实源。
3. `workspace/docs/rlinf-robotwin-pi0-traditional-rl/evidence/FORMAL_TRAINING_LOG_20260728.md`
   ：正式训练逐操作流水账与精确命令。
4. `workspace/docs/rlinf-robotwin-pi0-traditional-rl/evidence/IMPLEMENTATION_LOG.md`
   和 `SMOKE_EXECUTION_LOG_20260728.md`：主体实现与 fresh/resume smoke 记录。
5. `workspace/HANDOFF.md`：最终交接状态。

## 组成

- `workspace/`：根规则、最终交接和完整 DSRL 专题目录；
- `historical_source_materials/`：最初提供的七份历史 Markdown；其中两份原文含服务器
  密码，包内副本已把密码值机械替换为 `<REDACTED_SERVER_PASSWORD>`；
- `checksums/`：独立运行包的 SHA-256 指针。

完整运行日志、TensorBoard event、metrics、资源 CSV、Ray logs、Git patch、关键代码
快照和 checkpoint manifest 在独立运行包：

```text
dsrl_pi0_robotwin_formal_v1_runtime_step198_20260729.tar.gz
```

三个真实 DCP payload 约 94 GiB，仅保留在服务器原 run：

```text
/root/autodl-tmp/RLinf_fastwam_rlinf/logs/20260728_dsrl_pi0_robotwin_n20_formal_v1/
robotwin_adjust_bottle_dsrl_openpi_a800_2gpu_formal_v1/checkpoints/
```

恢复应使用完整的 `global_step_195`。step 196–198 的日志和指标存在，但参数更新不在
DCP195 中。
