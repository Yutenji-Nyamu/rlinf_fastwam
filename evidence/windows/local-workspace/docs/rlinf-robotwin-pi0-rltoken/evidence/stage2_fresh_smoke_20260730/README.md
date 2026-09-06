# Stage 2 fresh smoke evidence（2026-07-30）

这是成功 fresh run 的小型高信息量副本；大 checkpoint 仍留服务器。

```text
source HEAD:
  6fd3ee7106fb82f06eda82603c41a09767151709
runtime:
  /root/autodl-tmp/experiment_exports/rlt_stage2_smoke_20260729_v1/fresh_runtime
checkpoint:
  /root/autodl-tmp/experiments/rlt_stage2_smoke_20260729_v1/
  robotwin_adjust_bottle_rlt_stage2_smoke_fresh_v1/checkpoints/global_step_1
exit:
  0
```

核心文件：

- [`resolved.yaml`](resolved.yaml)：真正执行的完整配置；
- [`driver.log`](driver.log)：rollout、update、eval、DCP 和 metric table；
- [`resources.csv`](resources.csv)：2秒粒度 GPU/RAM/disk 曲线；
- [`rlt_trainer_state_complete.json`](rlt_trainer_state_complete.json)：两 rank
  completion、SHA 与保存后 `update_step=8`；
- [`stage1_binding_preflight.json`](stage1_binding_preflight.json)：Stage 1 artifact、
  full weights、stats 与 H/C/D/z/prefix 绑定；
- [`run_provenance.tsv`](run_provenance.tsv)：branch/HEAD/config/artifact identity；
- [`postcheck_summary.json`](postcheck_summary.json)：本次精简 postcheck 结果；
- [`SHA256SUMS.txt`](SHA256SUMS.txt)：11个服务器下载原始文件的校验值。

本轮用户明确省略 resume；DCP 内容完整不等于已经验证新进程恢复继续。
