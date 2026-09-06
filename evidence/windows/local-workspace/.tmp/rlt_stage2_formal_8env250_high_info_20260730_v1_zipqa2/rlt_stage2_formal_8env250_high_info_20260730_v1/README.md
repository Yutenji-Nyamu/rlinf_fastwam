# RLT × π0 × RoboTwin Stage 2 formal250 高信息量材料包

## 1. 这是什么

这是 `adjust_bottle`、`8 env × 250 outer cycles` 正式运行的轻量交付包。运行于
2026-07-30 自然完成，`exit_code=0`；最终 fixed 20-seed deterministic eval 为
`18/20=90%`，稳定 student 阶段 train 成功率为 `375/472=79.45%`。

本包重点保存可审阅、可复算的日志、指标、资源、配置、来源合同、图和规划文档。它不包含
模型权重、replay 或数据集，因此体积小，也**不能独立用于 resume**。

真正的最终恢复点仍在服务器：

```text
/root/autodl-tmp/experiments/rlt_stage2_formal_8env_250c_20260730_v1/
robotwin_adjust_bottle_rlt_stage2_formal_8env_250c_v1/
checkpoints/global_step_250
```

## 2. 推荐阅读顺序

1. `docs/15_STAGE2_FORMAL_8ENV250_FINAL_RESULT_20260730.md`：最终结论与边界；
2. `visuals/rlt-success-cycle250-final.png`：完整成功率和四阶段；
3. `visuals/rlt-optimization-cycle250-final.png` 与 `METRIC_GLOSSARY.md`：
   actor/critic/Q/gradient 的含义；
4. `visuals/rlt-resources-cycle250-final.png`：GPU、RSS、anon/file cache；
5. `config/resolved.yaml`、`config/budget.json` 和
   `config/run_provenance.tsv`：实际训练定义与来源；
6. `runtime/driver.log`、`runtime/tensorboard/` 和 `runtime/resources.csv`：
   完整原始运行证据；
7. `docs/evidence/IMPLEMENTATION_LOG.md`：具体命令、结果、问题与修复。

## 3. 目录说明

| 目录 | 内容 |
|---|---|
| `runtime/` | 完整 driver、TensorBoard event、资源采样和原始启动时间 |
| `config/` | source overlay、resolved config、命令、预算、provenance、seed bank和停止条件 |
| `dependencies/` | Stage 1 accepted artifact manifest；锁住feature/stats/action合同 |
| `results/` | 机器可读的最终统计、结束/退出状态与final checkpoint只读审计 |
| `visuals/` | 成功率、优化、资源三张PNG及轻量交互HTML |
| `docs/` | 唯一主计划、参数依据、smoke、规模设计、启动、最终报告和完整流水账 |
| `tools/` | 本次分析/绘图脚本 |

`CONTENTS_SHA256.txt` 对包内每个文件逐项校验。ZIP 外另有同名 `.sha256`，校验最终压缩包。

## 4. 明确排除

- 十个 full checkpoint 权重和 replay payload；
- rank trainer-state `.pt`；
- Stage 1 的20.56GiB checkpoint；
- clean-50数据集与π0基础模型；
- Ray全量内部日志；
- cycle35/68/100/200重复中间快照和100-cycle历史pilot。

这样保留了正式运行的主要信息，同时避免把轻量审阅包误当成模型交付或完整恢复包。
