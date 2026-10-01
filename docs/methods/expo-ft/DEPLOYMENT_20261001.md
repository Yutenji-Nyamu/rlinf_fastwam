# SZ2 EXPO-FT 独立部署与恢复定位

这是一份已授权短smoke的现场路由，不是正式3000步训练配置。方法/配置出处见[实现依据](IMPLEMENTATION_BASIS_20261001.md)，故障与最终验收见[实施记录](IMPLEMENTATION_LEDGER_20261001.md)。

源码从实际SZ2 RLT `e5cb9aa4c767490a66cba3c4e155c76f1193f199` 独立接入，位于 `/data/chenyiteng/projects/expo-ft-sz2-20261001/source`，branch `codex/sz2-expo-ft-20261001`。原RLT源码、共享Ray及其他用户未修改。Python分别使用EXPO parity py311与原RLT Python，精确路径见[运行包](SMOKE_PACKET_20261001.md)。

## 文件与合同

- `rlinf/algorithms/expo_ft/{core,backend,replay}.py`：算法、真实π0.5/RoboTwin桥接及CPU replay。
- `examples/embodiment/train_expo_ft.py`：fresh/resume串行1env真实闭环，每episode一次Q20/FM1/edit1/temp1；模型、norm、输入及9份port源码均固定指纹。
- `tools/expo_smoke_owner.py`：每次unique attempt，物理4/GPU UUID绑定，两个独立进程及精确后代清理。失败attempt日志保留。
- `tools/rlt_guardian.py`、`tools/expo_process.py`：单一资源lease owner与Linux pidfd兼容层。`work-heartbeat`超过15分钟、收到终止信号或smoke成功时按冻结合同归还；可定位失败在有效heartbeat下允许新attempt重试。
- `tools/expo_rlt_switch.py`：本次实际使用的372行bridge原字节；部署时原名为项目根`rlt_switch.py`。固定SHA `c9c20968403c3c861edde1053255e93e3ce16fd788a2d6e6b31dc1517330d8ff`。
- `tools/expo_rlt_cycle.py`：既有已验helper原字节；部署时放`rlt-cycle/rlt_cycle.py`，SHA `083740b44606b858db410d34ea32165de42d395053e41b7bc03c136dae162a2f`。停止/恢复沿用该helper，bridge只冻结当前lineage并修复完整checkpoint副本。

不能另起第二个guardian、覆盖本次冻结bridge/plan或直接运行旧调度器来绕过合同。现场实例输入及唯一owner回执在项目根；后续新借卡需新的prepare与run目录。命令中的实体路径不能当通用机器默认值。

## 大文件与轻量证据

两次完整checkpoint、实际RGB replay及RLT恢复副本留服务器。Git仅提交源码、配置/来源依据、测试和小回执；不提交模型、环境、官方嵌套clone、视频或大型逐文件修复清单。

RLT停止时最终step为1500/1525/1475/1475（物理4/5/6/7）；恢复仍以原累计3000为终点。副本修复保留最新模型/优化器/RNG、全部原索引/计数，同GPU既有完整CP的原payload按exact entry补齐，不删索引、不重置预算。恢复首轮须以`rlt-cycle/rlt-first-round.json`核验，进程出现或占显存本身不能代替继续训练证据。

Smoke的seed字段是向原生环境请求的seed，原生稳定性检查仍可能重试。v2 resume请求131321时出现物体不稳定，原生重试131322；此行为保持原Control，不以日志字段冒充无重试的固定种子。
