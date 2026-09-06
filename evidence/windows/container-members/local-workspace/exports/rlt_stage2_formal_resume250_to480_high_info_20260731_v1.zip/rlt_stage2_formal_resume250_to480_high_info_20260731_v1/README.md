# RLT Stage 2 resume250→480 高信息量快照

这是 2026-07-31 对 RoboTwin `adjust_bottle` RLT Stage 2 续训终态的本地只读证据快照。

- 服务器运行：`global_step_250 → 480`，`exit_code=0`；
- 远端下载：50 个小型日志/指标/config/completion/replay-metadata 文件，
  `download_manifest.json` 保存 byte count 与 SHA-256；
- `visuals/`：完整 cycle 1–480 的成功率、优化和资源图，以及机器可读 summary；
- `tools/plot_rlt_complete480.py`：把原 1–250 与续训 251–480 TensorBoard/资源拼接制图。

本目录不含模型权重、replay trajectories 或完整 checkpoint，不能独立 resume。完整可恢复
产物仍位于服务器：

```text
/root/autodl-tmp/experiments/rlt_stage2_formal_resume250_to480_20260730_v1
```

权威中文结论见
[`17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md`](../../docs/rlinf-robotwin-pi0-rltoken/17_STAGE2_FORMAL_RESUME250_TO480_FINAL_RESULT_20260731.md)。
