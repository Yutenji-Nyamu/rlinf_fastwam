# EXPO 评估间隔 25→10：2026-10-03 执行源码

此目录保存本次单次续接的可复核源码与服务器已测测试。执行状态与证据见[部署记录](../../docs/methods/expo-ft/EVAL10_DEPLOYMENT_20261003.md)；历史训练审计保留各自采样时间。

正式训练源码入口为 [`examples/embodiment/train_expo_formal.py`](../../examples/embodiment/train_expo_formal.py)。本目录的同名文件是部署补丁和 `test_migrate_eval10.py` 使用的 fixture；两份必须逐字节相同，当前 SHA256 为 `084eb7c01eebd812743c66c185edc8f22b531c95692fcd0f37cfd2bd54f2e7cb`。保留原 493 个 CRLF，只修改评估合同检查中的两处 25→10 字面量。发布暂存须保留这些字节，不能由换行转换改变源码 SHA。

## 现场映射

新控制目录 `CTRL=/data/chenyiteng/projects/expo-ft-sz2-20261001/eval10-continuation-20261003`。原训练输出目录保持 `formal-turn-switch-repair-20261002`。

| 本目录文件 | 现场位置与作用 |
|---|---|
| `reborrow_owner.py`、`reborrow_resources.py` | `CTRL/tools/`；唯一 owner 与原 RLT 精确借还桥接 |
| `queue_continuation.py`、`rebind_nextsix.py` | `CTRL/tools/`；两条已完成 Stage1 的等待队列续接 |
| `test_reborrow.py`、`test_queue_continuation.py` | `CTRL/tools/`；服务器 CPU 检查 |
| `migrate_eval10.py`、`test_migrate_eval10.py`、`train_expo_formal.py` | `CTRL/`；CPU 合同迁移、相邻 fixture 与测试 |
| `preflight_migration.py` | `CTRL/`；借卡前真实 checkpoint 的只读 CPU 预检，非控制器运行依赖 |

这些脚本带本次路径、owner、cycle 和回执约束。它们保存执行依据；已完成的借还或改绑不能据此直接重放。此次发布未包含第一次交接失败的旧 owner/handoff 入口。

## 检查证据

三份测试与服务器已测文件逐字节一致：资源续接 14 项、队列 16 项、合同迁移 7 项，共 37 项通过，见 [CPU 回执](../../docs/methods/expo-ft/evidence/20261003-eval10/cpu-tests.json)。服务器上分别在 `CTRL/tools/` 执行 `python -m unittest -v test_reborrow test_queue_continuation`，在 `CTRL/` 执行 `python -m unittest -v test_migrate_eval10`，均禁用 CUDA；Windows 只维护文件与作静态核验。

[真实 CPU 预检回执](../../docs/methods/expo-ft/evidence/20261003-eval10/migration-preflight.json)验证原 17 项源码合同、补丁范围，以及 latest/last1 两个完整 checkpoint 的六类 payload digest。预检采样时训练输入和现场 source 仍为 eval25，latest 为 75 回合 / 11,677 动作，last1 为 74 回合 / 11,477 动作；它没有修改训练状态。正式迁移、恢复推进和新评估结果以部署记录后续回执为准。
