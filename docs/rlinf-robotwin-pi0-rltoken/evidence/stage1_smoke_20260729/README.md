# Stage 1 smoke evidence

smoke 服务器观察边界：2026-07-29 15:37–16:11（Asia/Shanghai）；同日文档 QA 后只追加
LR scheduler 的源码审计、单配置修复和 CPU scalar contract，没有重跑训练。

## 核心文件

| 文件 | 说明 | SHA-256 |
|---|---|---|
| `source_configs/robotwin_rlt_stage1_sft_openpi.yaml` | smoke-time formal source snapshot；保留历史 absolute `min_lr`，不是修复后的当前 source | `0fa01fa8c6f8624438a3d27288ecb848336cd2857599bc4b1a1d369dfc563cb3` |
| `source_configs/robotwin_rlt_stage1_sft_openpi_a800_2gpu_smoke.yaml` | 服务器实际 S1-A source override | `a6a982b9054521ad17550f5fae96a83f3d89d4dcc68cfda8cec23ffab65bacbd` |
| `s1a_resolved.yaml` | S1-A 实际 resolved config | `2aa7400eb1355bcb1b84cdb431c6110f6f6bde378861379dbffce340befae49d` |
| `s1b_resolved.yaml` | S1-B 实际 resolved config | `5b984a6865df3d0f2aed8e957a4ba8f7f040ef1010ce606b5150714f9811723a` |
| `exact_commands.txt` | 实际命令、输出和停止条件 | `1ea9f65590f211c027641fadc98fea142a0b4cd64b2da3a8cd7472cf1d22dc2b` |
| [`exact_commands_addendum.md`](exact_commands_addendum.md) | reload、上传/状态/postcheck 与 scheduler 命令补录；未重跑 smoke | `d06c5ab04c346072c275329e919c5c3791d8c11257bada0ac2bd0abeacdeb23c` |
| `stage1_postcheck.json` | metric、资源和 checkpoint 汇总 | `8f84e8c5297f1ea8bd6eb55fa6b8c19bd6eea31be66d0a0d8f12fd8c41870acd` |
| `lr_scheduler_contract.json` | 旧 floor 负对照、修复后的 smoke/2k CPU schedule | `e68a7da1457e32538995f39b41f23a21b34d248e2ed3fa37f1177476e7c614df` |

修复后的当前服务器 source config 改为 `min_lr_rate=.1`，SHA-256 为
`8340ef4e953877de510da18548d0a69802104b7b2f8218698cd0fb586b49a8f2`；它由 Git 中的
`examples/sft/config/robotwin_rlt_stage1_sft_openpi.yaml` 直接审阅，不覆盖本目录的历史
snapshot。完整磁盘审计 A–P 命令另见
[`../DISK_AUDIT_COMMANDS_20260729.md`](../DISK_AUDIT_COMMANDS_20260729.md)，SHA-256
`d3d6f031bc9ec5c40ea8ac51aec41d0ab7c71614625dc18add30efa6ce42176c`。

`runtime/` 保存 S1-A、reload-only 和 S1-B 的原始 driver log 与每秒资源 CSV。大 checkpoint
留在服务器，不进入 Git：

```text
/root/autodl-tmp/experiments/rlt_stage1_smoke_20260729_v1/s1a/
robotwin_adjust_bottle_rlt_stage1_s1a_2step_v1/
checkpoints/global_step_2
```

checkpoint 合计 `20.56 GiB`；DCP metadata SHA-256：

```text
09a51c2530d095d838b41eb928729daf15b7d82f233e0e247153927cdf9d590d
```

完整逐操作账本见
[`../IMPLEMENTATION_LOG.md`](../IMPLEMENTATION_LOG.md) A031 起。
