# π0.5 clean GRPO：128条/轮基线

2026-09-11，用户确认先运行clean，物理GPU4/5。11:49:07启动；11:53:01两rollout均generate、两env均interact，进入首轮采集（日志明确0/2串行）。job `a0020000`、15个actor ALIVE，driver1072141；实际配置diff为空，无fatal、无退出。健康确认后停止盯跑，尚未取得完整首轮指标。旧DVAC new已由独立收尾流程停止。本实验从原始模型开始，不恢复旧权重；环境保持原GRPO的旧指令实现。

| 项目 | 本次 | 原clean |
|---|---:|---:|
| 并行环境 | 64 | 64 |
| 串行采集次数 | 2 | 4 |
| 每轮轨迹 / G8组数 | 128 / 16 | 256 / 32 |
| global batch / micro batch | 512 / 32 | 1024 / 32 |
| 每轮遍历 / Adam次数 | U2 / 2 | U2 / 2 |
| 每rank梯度累积次数 | 8 | 16 |
| 轮次 / 最多训练采集 | 200 / 25,600 | 200 / 51,200 |
| 学习率 / 去噪 / horizon | 5e-6 / M10、noise0.5 / H50 | 相同 |
| 固定评估 / 保存 | 32条每5轮 / 每10轮 | 相同 |

global batch减半经用户另行明确确认。当前代码每rank要求 `rollout_size % (global_batch_size / world_size) == 0`；半量采集只有全局512个chunk槽位、每rank256个，旧B1024会要求每rank512个，首轮训练必然失败。B512使一次更新仍覆盖整轮数据，U2仍对应每轮两次Adam。过滤无比较信号的二值组仅改变loss mask，不删掉张量槽位，因此不破坏上述整除关系。服务器CPU执行原actor断言已验证B1024失败、B512通过。

独立分支 `codex/sz-pi05-grpo-clean-half-20260911`；基点与生产HEAD均为 `1d015a2aa03ec8132d8207ba47a2be3dbe1d9591`。没有生产源码改动，六份相关源码与原clean逐字节相同。DVAC mode=off，未引入Prism。实际配置与原clean之差为采集次数、已确认batch、原始模型初始化及独立运行路径/名称，缺省值显式化；学习率、G8、原奖励/优势/过滤/clip、训练与评估种子均保持。

运行目录：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-clean-half128-formal200-phys45-20260911-v1`。

源码目录：`/data/chenyiteng/projects/rlinf-shenzhen/worktrees/pi05-grpo-clean-half-20260911`。

namespace：`RLinf_grpo_clean_half128_formal45_20260911`。入口为独立stage `/data/chenyiteng/results/server-maintenance-20260911/grpo-half-clean/ops.py launch`；完整训练argv、环境、wrapper保存在下方prepared证据。正常结束200轮，保留96小时上限；异常或用户停止只清理经过身份核验的本namespace对象。共享Ray及其他任务保持。

存储：每10轮保存，20代checkpoint预计约537GiB。启动前/data约383GiB可用；用户随后明确授权清理本人已结束实验中间checkpoint，11:56:45独立维护回执已完成111个精确大文件、释放1035.75GiB，两盘当前余量/data约964.74GiB、/home约778.80GiB，已覆盖本实验完整保存预算。保存频率未变、未实现自动删除。

准备过程中两次前置核验停止分别是源码manifest文件名笔误、未把关闭DVAC时的输出路径识别为运行身份字段；已修正运维脚本。发生在训练启动前，生产源码未改。最终prepare通过，验证namespace回执保留，未重复训练smoke。

- [配置合同](evidence/grpo-half-clean-20260911/prepared/contract.json)
- [配置差异](evidence/grpo-half-clean-20260911/prepared/config_diff.json)
- [精确命令](evidence/grpo-half-clean-20260911/prepared/command.txt)
- [batch兼容性](evidence/grpo-half-clean-20260911/prepared/batch-compatibility.json)
- [源码核验](evidence/grpo-half-clean-20260911/source-receipt.json)
- [启动回执](evidence/grpo-half-clean-20260911/launch-receipt.json)

- [启动健康确认](evidence/grpo-half-clean-20260911/STARTUP_VERIFIED.json)

本次发布仅正式说明、运维入口与轻量证据；实际训练生产源码保持clean基点不变。后续Git HEAD变化仅代表文档归档提交，生产SHA见源码核验。
