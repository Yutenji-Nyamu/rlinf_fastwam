# RLT Stage 2 v4：250 终态轻量产物

## 1. 终态

- `250/250` 自然完成，`exit_code=0`，总墙钟 `7h 53m 42s`。
- 最终 train=`7/8`；最近5/20步分别为 `95.0% / 85.0%`。
- student-only fixed-20：Step `175/200/225/250` 分别为 `9/20、18/20、19/20、18/20`。
- 在线 AC 从 Step 137 开始，最终 `update_step=104000`。
- 最终 actor/critic loss=`-0.0900 / 0.0014`，actor/critic grad norm=`2.729 / 0.151`；无 fatal、OOM 或 non-finite。

## 2. 资源

- GPU4/5 显存峰值=`21.4 / 21.7 GiB`，两卡全程平均利用率约 `25.2%`。
- 整机 `MemAvailable` 最低约 `1309 GiB`；cgroup OOM/OOM-kill 均为0。
- 单卡容量足够，但正式 baseline 保留2卡是为了不改变已验证的 world-size、batch与replay/warm-up分片合同。

## 3. 图与数据

- [可直接下载的轻量 ZIP](../../exports/shenzhen_rlt_stage2_v4_final250_light_evidence_20260824.zip)：875,060 bytes、18项；不含checkpoint权重。
- [学习、评估与 replay 时间线](evidence/rlt-v4-final250-20260824/01_rlt_final_learning_and_eval.png)
- [在线优化与耗时](evidence/rlt-v4-final250-20260824/02_rlt_final_optimization_and_timing.png)
- [GPU与主机资源](evidence/rlt-v4-final250-20260824/03_rlt_final_resources.png)
- [轻量证据说明](evidence/rlt-v4-final250-20260824/README_FIRST.md)
- [机器可读摘要](evidence/rlt-v4-final250-20260824/summary.json)

轻量目录同时包含完整原始 driver/metrics/resource/TensorBoard、resolved YAML、精确命令与解析CSV；不包含checkpoint权重。服务器 `global_step_250` 原样保留，可用于后续严格续训到480。
