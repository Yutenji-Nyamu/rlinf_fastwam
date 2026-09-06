# RLT Stage 1 高信息量证据包

这是 `adjust_bottle`、clean-50、2,000 optimizer steps 的 Stage 1 收尾包。它刻意不包含
20.56 GiB checkpoint，只保留足以审计训练、资源、配置与 artifact 验收的约数 MiB 文件。

## 先看什么

1. `visuals/training_status.png`：loss、学习率、吞吐与 GPU 的训练概览。
2. `visuals/memory_breakdown_desktop.png`：RSS、cgroup anonymous memory、file cache
   与主机 available memory 的区别和曲线。
3. `artifact_acceptance_v2/stage1_artifact_manifest.json`：Stage 2 应绑定的 artifact 身份。
4. `artifact_acceptance_v2/validation.json`：strict reload、π0 delta=0、
   true/shuffled/zero `z_rl` 的原始验收结果。
5. `formal_run/runtime/driver.log` 与 `formal_run/runtime/resources.csv`：完整 2,000 步指标
   和 1,455 个资源采样点。
6. `formal_run/tensorboard/`：7 类、每类 2,000 点的原始 TensorBoard event。

## 关键结论

- 正式训练完成：exit code 0，2,000 steps，约 28.9 分钟。
- 固定真实 batch 上：
  - fresh seed-0 proxy loss：`5.1976585388`
  - endpoint/true-`z_rl` loss：`0.5337553024`
  - shuffled-`z_rl` loss：`1.7118018866`
  - zero-`z_rl` loss：`2.1026575565`
- 所有验收 hard gate 为 true；非 `rlt_module.*` 的 π0 tensor 变化数为 0。
- Stage 2 artifact manifest：
  `robotwin-adjust_bottle-rlt-stage1-clean50-step2000-v1`
- manifest SHA-256：
  `6ca58f26f801e4630f26d6aed36c5084ce1ea3fa93730e54aa69a0f2a3712433`
- full weights SHA-256：
  `7dddc268733b978bf382cda77257371cf9de4155f60ec3094cc8ffcfd6d74bd0`

这些结果能证明 checkpoint 可加载、π0 冻结、重建确实学到了且 decoder 使用
sample-specific bottleneck；它不能证明 Stage 2 控制成功率会提升。

## 内存口径

- matched training-rank RSS peak：`38.51 GiB`
- cgroup anonymous peak：`39.53 GiB`
- cgroup accounted peak：`240.00 GiB`
- 其中 file cache peak：`229.94 GiB`，可在内存紧张时回收
- host available minimum：`928.83 GiB`

因此约 39 GiB 才是训练相关常驻内存的合理量级；240 GiB 主要是容器记账包含文件缓存，
不是 240 GiB 模型泄漏。训练期间无 swap、OOM 或主机内存压力。

## 目录说明

- `formal_run/`：source/resolved config、精确命令、数据清单、完整日志与 TensorBoard。
- `artifact_acceptance_v1/`：因镜像缺少 `/usr/bin/time` 而在模型加载前退出的失败证据。
- `artifact_acceptance_v2/`：使用 Python 标准库资源计数后的成功验收。
- `source/`：验收程序和本地启动包装器。
- `visuals/`：桌面/移动端资源图与训练状态图。
- `CONTENTS_SHA256.txt`：服务器原始 34 个文件的逐文件校验值。
- `LOCAL_ADDITIONS_SHA256.txt`：下载后加入的命令、图表、包装器和本 README 的校验值。

服务器 checkpoint 仍位于：

```text
/root/autodl-tmp/experiments/rlt_stage1_formal_20260729_v1/
robotwin_adjust_bottle_rlt_stage1_clean50_2k_v1/checkpoints/global_step_2000
```
