# WMRL 发布盘点 · 2026-10-05

本轮只读核对 Git 远端、已有提交树及本地轻量文件；未连接服务器，未修改 index、分支、运行代码或 HANDOFF。此清单是补推候选，不是本轮新发布成功回执。

## 已推送并重新核到远端

仓库为 `Yutenji-Nyamu/rlinf_fastwam`。本轮 `git ls-remote --heads` 四项均与既有回执一致：

| 范围 | 分支 | 当前远端完整 SHA |
|---|---|---|
| 旧 LIBERO/Wan、评估、OOM 审计及资源监控 | `codex/sz3-wan-goal-20260930` | `21b5d590a9116d94e38c139a3ec4aa9723abbcc3` |
| OpenDW/RoboTwin、正式修复、真 B16 | `codex/robotwin-opendw-wmrl-20261003` | `07faf0636163b8ca6a604f94eb8cea74705a125a` |
| Rynn 接线、Success 诊断 | `codex/rynnvalue-wmrl-release-20261005` | `e9e9faf4826695d552f19fe50366b2dc1969b2bd` |
| Rynn 数值短测、曲线、归还修复及奖励候选调研 | `codex/rynn-numeric-reward-audit-20261005` | `382b2698a1ca8853343e32ae7c0b688a38521933` |

最新数值分支继承 Rynn 接线分支，后者继承 OpenDW B16；无需把这些源码重新各推一遍。数值发布包含22项，前一发布包含58项。旧 LIBERO 分支独立，未声称已合入 main。

本机根仓库仍是无 remote 的初始 `master`，其大量 untracked 不表示文件从未上云。比较使用现有 `worktrees/pi05` 中的已核提交对象，未 fetch/checkout。检查范围为 `docs/world-model`、`local_patches/opendw_smoke_20261003`、`local_patches/rynn_wmrl_20261005`、两处 `local_scripts/rynn_*_20261005`；排除缓存及大于2MB文件。207项与最新 OpenDW/Rynn 或旧 Wan 提交字节一致，最新 OpenDW/Rynn 同路径文件未发现本地修改遗漏。

## 本轮建议补入的精确轻量文件

以下文件尚不在最新数值分支提交树中。发布成功回执通常产生在提交之后，需随下一次发布补入；它们缺席不影响上述发布已经成功的事实。

| 文件 | bytes | 用途 |
|---|---:|---|
| `docs/world-model/publication_batch16_20261004/published.json` | 452 | B16 发布完成回执 |
| `docs/world-model/publication_formal_20261004/published.json` | 452 | 最初正式版本发布回执 |
| `docs/world-model/publication_formal_repair_20261004/acceptance_published.json` | 420 | 正式修复验收发布回执 |
| `docs/world-model/publication_rynn_20261005/published.json` | 423 | Rynn 接线发布完成回执 |
| `docs/world-model/publication_rynn_numeric_20261005/published.json` | 456 | 数值实验发布完成回执 |
| `docs/world-model/publication_rynn_numeric_20261005/return_status_20261005_2030.json` | 1072 | 20:29 RLT 恢复快照；当时首个完整轮次尚未核验 |
| `local_scripts/rynn_wmrl_20261005/owner_implementation_notes.md` | 5734 | 已有 owner 接入审查说明，保留调查时态 |
| `docs/world-model/robotwin_pipeline_20261003/publication_inventory_20261005.md` | 本文件 | 本轮发布覆盖盘点 |

新 Rynn 0/1 测试和 click_bell 切换代码/配置/轻量证据由本轮执行任务另补清单；本盘点没有将尚在开发的文件宣称完成或已推。

## 历史小遗漏与不应覆盖的差异

- `docs/world-model/WAN_GOAL_PAUSE_20261001.md`：本机比旧 Wan 分支多一条01:07恢复/发布记录，3014 bytes；可将该条历史补入旧记录或新的归档说明。
- `docs/world-model/WAN_GOAL_RUNLOG.md`：本机多16:09发布及资源协调记录，41114 bytes；属历史记录小遗漏。
- `local_patches/opendw_smoke_20261003/formal/native_env_probe.py`：7180 bytes，旧原生 reset/关闭探针。没有发现它属于现行运行依赖；若保留历史验收可收入归档，不作为新训练必需文件。
- `local_scripts/rynn_numeric_20261005/make_receipt.py`：2644 bytes，已完成结果的本地轻量回执序列化脚本。结果本身已推，脚本可选补入，不能重放为新实验。
- `publication_rynn_numeric_20261005/rynn-numeric-probe.png` 为359279 bytes；同内容 SVG 已推，本轮无需为覆盖结论重复上传 PNG。
- 旧 Wan 的 `audit_20261003/{REPORT,implementation_audit,reward_audit,training_dynamics}.md` 与 `discussion_20261003/{config_eval,reward_signal}.md` 原始字节不同，逐行核查主要是云端把 Windows 本机路径换成仓库相对链接，另有空行差异。研究结论已覆盖，**不要拿本机版覆盖云端可用链接**。
- `publication_followup_20261004/manifest.json` 的 `pushed=false` 是暂存时刻；其10篇研究/历史文档及两份恢复归档已在最新提交树中，不能据旧字段说它们还没推。

## 继续排除

模型、checkpoint、reset 数据、原始大日志/TensorBoard、生成视频/逐帧图；生成的 RPC payload、SSH/凭据环境、完整运行 plan；已取消草稿、上游源码读取副本、混合多任务旧 HANDOFF 归档。必要的可审阅准备/发布生成器另按精确清单收录。根目录不得批量 `git add -A`。本轮新发布需从精确 allowlist 审查，推送后再核远端 SHA，运行 checkout 保持原状。

上述是 Git 源码/文档/轻量实验备份，不代表模型和原始数据已经有独立异地备份。

## 本次补存的两段 Wan 历史记录

以下摘录来自本机旧日志的追加内容，统一存于本页，避免覆盖云端已经修正的旧文档链接。它们是 **2026-10-01 的历史状态**，不代表当前调度。

- **01:07，RLT 恢复与发布：** 深圳3四组均从完整 CP25 推进到第27轮，真实首轮验证通过，driver 存活；当时三机原4–7共12条 RLT 均运行。文档、源码与轻量证据已推 `codex/sz3-wan-goal-20260930`，提交 `894322d3c731f63ffa1cb8355eac0e67f5bcd71c`，原日志记载远端 SHA 已核。服务器训练独立于本机；本机巡检离线时暂停。来源：本机 `WAN_GOAL_PAUSE_20261001.md` 第21行。
- **16:09，机制及资源快照发布：** w162 退出0，发布 `23acc5eaa2ea749ea5c81bbaec65ac442287d27a`，8新增、2修改、0删除，原日志记载远端 SHA 一致；专题、源码依据、轻量快照、四步回执和只读 helper 入同分支，权重与原始第三方源码日志未发布。16:06 EXPO 窗口收到正式参数规模短 smoke 授权，但当时尚未停 RLT；由该窗口唯一负责新 cycle 的暂停和归还，统一检查暂停深圳2资源变更，仅只读。旧 RESTORED 回执仅属上一轮，深圳3 WM 当时继续。来源：本机 `WAN_GOAL_RUNLOG.md` 第3–6行。

本次发布清单包含以上历史补存，以及当前二值诊断/click_bell 的必要启动、状态查询和准备入口；失败尝试的生成 payload 不纳入。
