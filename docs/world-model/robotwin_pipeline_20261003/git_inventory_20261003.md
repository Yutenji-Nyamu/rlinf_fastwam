# WMRL / OpenDW Git 盘点与更新发布范围

核查时间：2026-10-03 23:10 CST。本轮只读检查 Git 和既有记录，仅新增本文件；没有 commit、push、fetch、清理、服务器操作或修改旧清单。远端核查禁用 credential helper 和交互提示，未读取凭据文件。

## 已推与尚待发布

旧 LIBERO WMRL 已在本轮从 Windows 实时核到远端 `codex/sz3-wan-goal-20260930`：`21b5d590a9116d94e38c139a3ec4aa9723abbcc3`。这与此前记录一致。[远端分支](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz3-wan-goal-20260930)

`git ls-remote --heads origin` 查询 `*opendw*`、`*world*` 未返回分支；预定发布分支 `codex/robotwin-opendw-wmrl-20261003` 尚不存在。本地可见发布脚本只生成/暂存独立发布包，明确 `committed=false,pushed=false`；没有检出对应提交和推送完成回执。因此本轮 OpenDW 实现、实验轻量日志和新多卡实现均须按下方清单完成发布，不能把“已准备发布包”写成“已推”。这不声称检查过所有任意命名分支的全部文件历史。

本次 `ls-remote` 同时读到远端 `codex/sz-pi05-robotwin-rl=ae7e5da72acf4a47a54cecff4f802ebc174b397a`；本地 pi05 分支的远端跟踪引用并非这一分支，不能据其 ahead 数判断云端状态。

## 本机仓库边界

| 本机路径 | 分支 / HEAD | 工作区状态 | 发布处理 |
|---|---|---|---|
| `C:/Users/86136/Documents/rl` | 尚无首个提交的 `master`，无 remote | 快照有27,258个untracked文件，涵盖研究、运维与大批历史资料 | 不在根目录批量 `add -A`，只从精确清单取文件 |
| `worktrees/pi05` | `codex/sz-pi05-robotwin-rl` / `e81383ce5b7013731ebf926269712916de44f873` | clean；上游是旧 `origin/codex/sz-ppo-dvac-action-adv-fix`，本地缓存显示ahead1 | 只作为固定2151源码读取入口，不移动分支或混入本轮文件 |
| `references/rlinf_fastwam_audit_20260824` | `main` / `8138d6700e3838250c1139289ebfba43d48ff7de` | 1,819项staged删除、5项untracked | 保持原状，不能拿这个已有dirty主checkout发布 |

后两者共享 Git worktree 管理，remote均为 `https://github.com/Yutenji-Nyamu/rlinf_fastwam.git`。本地未发现 Wan/OpenDW 的分支或远端跟踪引用。根目录文件untracked不等于远端从未有副本：旧Wan此前通过独立服务器发布checkout提交，已由本轮远端SHA确认。

## 旧发布包的准确状态

[旧JSON清单](publication_allowlist_20261003.json) 的47个本地文件仍全部存在，当前SHA逐项相同，合计515,885 bytes；**它是哈希一致但范围与结论已过时的包**。它还单列2个服务器源码文件及清单自身2文件。不得把旧清单哈希一致等同于包含了本轮最新实现。

旧包遗漏新 `build_multigpu_config.py`、整个 `multigpu/` overlay，以及单卡返回故障、v5兼容修复、用户要求WM优先接管的后续事实。旧状态文档仍以22:32的N8完成/N16保存为最新，并把N16/R1/L384列为下一步；这已被新N64/G8/R8四卡短smoke方案取代。更新旧状态文档由主任务统一处理，本盘点不代写。

发布仍从固定 `2151a08ee1bd75df1bef0d8190e594bd5c7f7977` 建独立 `codex/` 发布checkout。保留原工作区和服务器在跑源码，按精确allowlist暂存，不运行带部署副作用的RPC。

## 最小更新 allowlist 与映射

1. 保留下表A的旧47项作为已实现单卡流程、诊断与方法文档；更新涉及当前结论的文档后重新核SHA。旧清单MD/JSON自身也须刷新后收入。
2. 加入下表B的13项多卡实现/测试/说明，原路径保留。当前默认代码应使用最新多卡兼容env：将B中的两个env模块同时作为A对应规范目标 `rlinf/envs/world_model/` 的来源，不能仍从旧单卡副本覆盖它们。B保留overlay内的两模块是为了其已写测试的相对路径可用；这两份与规范目标须同SHA。
3. 两个服务器规范源码文件仍须取回实际使用字节并核相对2151差异：`rlinf/envs/__init__.py`（仅注册）、`rlinf/runners/embodied_runner.py`（卸载等待）。旧readback为 `a14c4b24…` / `c479190e…`，只代表此前单卡readback，不替代本轮新checkout核验。
4. 多卡lifecycle组合器引用GPU4、GPU5–7两个冻结子模块及其helper。**以新plan实际锁定的模块路径/SHA补齐**，不能只发布组合器却漏子模块；表B已含盘点时出现的GPU5–7子模块，最终仍须核新plan锁定的字节。已在实际返回尝试中使用的v5文件可以在核部署SHA后收入运维历史目录，不能因目录名叫drafts就丢掉已用修复；未使用草稿仍排除。
5. 收入本盘点、刷新后的执行记录和下面列出的脱敏实验摘要；不发布完整环境字典、运行plan、原始日志或服务器RPC。

表中是盘点时的本地快照，部分多卡文件仍在开发。它们是准确待审候选，**不是已冻结、已通过GPU测试或已发布的声明**；发布人需对最终变更逐项审查、更新清单并核部署字节。

### 表A：旧47项，目标分支尚待发布

| 本地来源 → 目标（相同则省略箭头） | bytes | 当前SHA256 |
|---|---:|---|
| `local_patches/opendw_smoke_20261003/build_smoke_config.py` | 8560 | `0cd40d25233849f5fe77e10d6d512e2f3874d60eeb18a8fa3364160d20bcbf45` |
| `local_patches/opendw_smoke_20261003/build_reset_data.py` | 2837 | `8159c784084bc8ffa5e7e826c9a55082fa05007ae80ee56e8771de17a8629aa7` |
| `local_patches/opendw_smoke_20261003/prepare_runner_barrier.py` | 3714 | `96bc3b43f061aad459f1994c85c65aaaa423b3956675f9c02be5abc669839bc0` |
| `local_patches/opendw_smoke_20261003/register_opendw.patch` | 840 | `80dd5d500136464a369c3c94d7bbd041442b6a4451140aeacdaf9daf33b0e7cf` |
| `local_patches/opendw_smoke_20261003/CONFIG_AND_RUNNER.md` | 4167 | `affabb24c728a48ff33173104b11c50c0551d74a9e808866137517830f348117` |
| `local_patches/opendw_smoke_20261003/test_opendw_adapter.py` | 2675 | `93896aa654e90ebd5ed2580c048667e06319696680463d188ed9af74d4a8d500` |
| `local_patches/opendw_smoke_20261003/test_opendw_env.py` | 4551 | `4979d89914470df80fa456f78a21d937176619db4ee770d169d08ab303b430a1` |
| `local_patches/opendw_smoke_20261003/test_opendw_service.py` | 6386 | `ad372cc09d9785bdacd8be3f1e1683c46b154c33f418120a4db98f0b40cd6216` |
| `local_patches/opendw_smoke_20261003/tools/opendw_service.py` | 25571 | `267dcf2531e176dbb1f6a5d82584bcd2742d94e802123ef9ab89a6eddb401aa5` |
| `local_patches/opendw_smoke_20261003/tools/opendw_reward.py` | 4388 | `d5e41b9b89375270c386ec3a2700855ddcc420ae4779c3c72dce509772ab840d` |
| `local_patches/opendw_smoke_20261003/tools/opendw_action_telemetry.py` | 3722 | `72ffdf7252c7dd62968f83ea09734f94b91f5c8e8e0fbec170e8891dcca32079` |
| `local_patches/opendw_smoke_20261003/tools/opendw_smoke_owner.py` | 27143 | `36ddcb97a06f8c5aedbf6d2f9ba4594a7f5789996ef5ce547cb3125b294b50af` |
| `local_patches/opendw_smoke_20261003/tools/test_opendw_owner_pidfd.py` | 3275 | `46071a641e283f65bb995deaaa0c35597bd4c2556cfcbc10443685f1d4c5f5b1` |
| `local_patches/opendw_smoke_20261003/tools/OWNER_PLAN.md` | 4558 | `2f4f294ebd365c7acb346de829203f19752c3091856ed3a501dd2f0572679124` |
| `local_patches/opendw_smoke_20261003/tools/GPU_SCOPE.md` | 12121 | `2fd6f50325b32b847c9781785db8d6f16daf2847ced59b266169bec83f562c7b` |
| `local_patches/opendw_smoke_20261003/tools/gpu_scope_prepare.py` | 8458 | `d4b1d3deff56f9ab80eeb75610e616438d166fe2c954ab03c4a7e9627f2e7162` |
| `local_patches/opendw_smoke_20261003/tools/gpu_scope_runtime.py` | 12221 | `ab897ab43c535f29957118a05ee6d740ac18cfdd558aeaba94fb668b4a6ed4b7` |
| `local_patches/opendw_smoke_20261003/tools/gpu_scope_camera_probe.py` | 11749 | `8ac0937838a63ac5aa838cfaf61c7c6a1646f1b9831ad33fb097db409f2c7713` |
| `local_patches/opendw_smoke_20261003/tools/gpu_scope_diagnose.py` | 4713 | `fe2a16765931dc8617d40139b6c70e62c3c76fdf2623c3992cfc383cbeb790af` |
| `local_patches/opendw_smoke_20261003/tools/gpu_scope_bootstrap/sitecustomize.py` | 393 | `916843c11752ad1ea54a293ea849f9c08533e4bb13f0127c0e069a66f4abc30a` |
| `local_patches/opendw_smoke_20261003/tools/reconstruct_t5_exact_zip.py` | 16537 | `55689bb9a1c6ed5298cf82f4c2b40a54fe9b478c6842aa1d0e524dba4a2b6634` |
| `local_patches/opendw_smoke_20261003/tools/probe_t5_repack_metadata.py` | 16352 | `f20d2d91edbb6de82439264f57f26b9664e5e42ceb167d7dbb88d20177573500` |
| `local_patches/opendw_smoke_20261003/rlinf/envs/world_model/opendw_adapter.py` → `rlinf/envs/world_model/opendw_adapter.py` | 4803 | `12eaff954ab3171fabd10bb3f4988becb6af82082ceb18824316a978bb6c6acc` |
| `local_patches/opendw_smoke_20261003/rlinf/envs/world_model/opendw_robotwin_env.py` → `rlinf/envs/world_model/opendw_robotwin_env.py` | 14248 | `27fc1c336aaf8c74d5eb58d8635321bdaad5015e5cb322ce8d7dd28f0abf28cc` |
| `tools/opendw_smoke_gpu4_cycle.py` | 28818 | `632e63903c5ce804db45f3780db7a2fea2effda43b33de7b8b8d231ca17940c1` |
| `tools/test_opendw_cycle_graphics_scope.py` | 8864 | `e46a60a4da3e51d87f1142a56a6605fb67bf22bd80c55c0bcea7582299f0b57f` |
| `tools/download_opendw_assets.py` | 21234 | `8235bcdc40ddd527376bb617946e7aa0019c973586d8062a9bd499c4d4c3c1bf` |
| `tools/test_download_opendw_assets.py` | 4463 | `4cd7c9266ad57005e767fe544c7b215ee4e03fcaa11a56a67e7aeb91778c117f` |
| `tools/analyze_opendw_smoke.py` | 25179 | `8a774d356d41d059aeceb9060562078921af76875fe8ce51249fd3a58c0a5d43` |
| `tools/test_analyze_opendw_smoke.py` | 5079 | `ecc19c695dc669003de8c85e180d3f50e557e0344c946d3811dae7a6faba4915` |
| `docs/world-model/ROBOTWIN_OPENDW_PIPELINE_CONTEXT.md` | 17421 | `f595d8f09cedb024933c7f84766fc9f86c5474f132407c27ba171bffb75884da` |
| `docs/world-model/robotwin_pipeline_20261003/action_state_followup_20261003.md` | 13263 | `9bc67cdaa828276766fba6424fdc86b99eb150ac046d4b6417ba2122f3ca0a6f` |
| `docs/world-model/robotwin_pipeline_20261003/c32_adapter_decisions.md` | 13386 | `8bd6651d29c43df690583c1de1cb5240cd1169dcaf88bda6607c9389de230929` |
| `docs/world-model/robotwin_pipeline_20261003/execution_20261003.md` | 10342 | `f86901a09b4d7c989e66733596a004bd20bb611f765ba8fe4373fa66d3ae8d64` |
| `docs/world-model/robotwin_pipeline_20261003/general_reward_integration_20261003.md` | 14064 | `961eb20f12ca9d43ce5727919f6b555a8f2e2b17d389152cc5a8596190d3c832` |
| `docs/world-model/robotwin_pipeline_20261003/gpu4_smoke_borrow_restore.md` | 5301 | `106e756df1effa8f7242c7383a821d1158d827832572e34d460ea83e701c9c07` |
| `docs/world-model/robotwin_pipeline_20261003/implementation_review.md` | 5270 | `810482fc8eebb02f17defb82b6d5d05094aeb0e2e588eb41ffaaa33008ef7e0d` |
| `docs/world-model/robotwin_pipeline_20261003/interface_audit.md` | 20175 | `877c7b228bba1cc28944726869e24006bdb7afea10fcd54dbb5bbf72c3ecdce2` |
| `docs/world-model/robotwin_pipeline_20261003/method_map.md` | 11240 | `d2a22c675c698b89627da4c4ee7fc6ab12016b452f858a922c327fed114ab0a9` |
| `docs/world-model/robotwin_pipeline_20261003/reward_options.md` | 13478 | `198d3c27f061823ab48ca67e552309a90d0dbc70e2dbb0b6cef566da8dd77a82` |
| `docs/world-model/robotwin_pipeline_20261003/reward_training_and_candidates.md` | 13102 | `bef6eda68228a2e68cb0432687551c55b783d9567a63524086b31737eba73f6b` |
| `docs/world-model/robotwin_pipeline_20261003/signal_audit_20261003.md` | 6265 | `04892c8f46d23bdbfa8369441b7816b7eb2b796ff1f264e78e292233962a1763` |
| `docs/world-model/robotwin_pipeline_20261003/signal_smoke_384_plan.md` | 8983 | `3d9238cb0bd7b1dccdbdeb915d517b05bef1529e0dd56493e5b3ac787ebfeca7` |
| `docs/world-model/robotwin_pipeline_20261003/smoke_contract.md` | 7837 | `17bf649012426ede0f9ce48ffe22989cafec680e701d39c38b2ce2467a491060` |
| `docs/world-model/robotwin_pipeline_20261003/evidence/publication_preflight_20261003.json` | 2713 | `4c4f5ec875856d3de8a254131f0063e34d5145b21d3d2f28a85716ebcb099017` |
| `local_patches/opendw_smoke_20261003/tools/rlt_checkpoint_lifecycle.py` | 41453 | `009cb0179795d6b4011d35478e8dc962fc5af3603ea6773be976136cf0aeb788` |
| `local_patches/opendw_smoke_20261003/tools/rlt_next_six_ops.py` | 13973 | `1ef808a8296c2672c1c6841432db19f9a25a99f6b434259018184059de633ff7` |

### 表B：新多卡文件，旧清单缺失

| 本地来源 → 目标（相同则省略箭头） | bytes | 当前SHA256 |
|---|---:|---|
| `local_patches/opendw_smoke_20261003/build_multigpu_config.py` | 10062 | `092186f3cdbf3d2b340eb60acf2faf7129471566b02cd758a8be05394bf34d79` |
| `local_patches/opendw_smoke_20261003/multigpu/opendw_multigpu_owner.py` | 32463 | `bce07c3f6fc17b1c7a54b731789fd8e6bbb1357de0b69afce0547fdb1e4a7ddc` |
| `local_patches/opendw_smoke_20261003/multigpu/OWNER_INTERFACE.md` | 5869 | `83825eb5deeaef90c58cc7eb6fe03c5069de27333e6ad5d481cc78e797260bd9` |
| `local_patches/opendw_smoke_20261003/multigpu/rlinf/envs/world_model/opendw_adapter.py` | 4803 | `12eaff954ab3171fabd10bb3f4988becb6af82082ceb18824316a978bb6c6acc` |
| `local_patches/opendw_smoke_20261003/multigpu/rlinf/envs/world_model/opendw_robotwin_env.py` | 15788 | `72883c000970483f4300ca5b1a46b06f2cd79df06942ddf27b47dd78ef08344f` |
| `local_patches/opendw_smoke_20261003/multigpu/rlt_gpu567_cycle.py` | 22594 | `6108464c2bba3a715bbc727b75afa2bcc88e19650e6670a740bf06464dc2eef2` |
| `local_patches/opendw_smoke_20261003/multigpu/rlt_multigpu_cycle.py` | 9112 | `182aec974f22a1cc99d80584ea96d6e4fefa282b07e877f4bdf20006c06f401b` |
| `local_patches/opendw_smoke_20261003/multigpu/ROUTING.md` | 3019 | `714b021075d05e1f0e1ff5c0f1ab38e9fcfb8c1ba87dc2b562e82c12937cce75` |
| `local_patches/opendw_smoke_20261003/multigpu/test_multigpu_routes.py` | 4046 | `2228e016abb69a7686e5ed18b37d1ce9fd9f490b588bb53733844e79e90b7b6b` |
| `local_patches/opendw_smoke_20261003/multigpu/test_opendw_multigpu_owner.py` | 13879 | `3cc9743652cfe124016cdff921d87ea5a6ead108bb628a9cf94bdfed1ce9e9fc` |
| `local_patches/opendw_smoke_20261003/multigpu/tools/opendw_action_telemetry.py` | 3722 | `72ffdf7252c7dd62968f83ea09734f94b91f5c8e8e0fbec170e8891dcca32079` |
| `local_patches/opendw_smoke_20261003/multigpu/tools/opendw_reward.py` | 4388 | `d5e41b9b89375270c386ec3a2700855ddcc420ae4779c3c72dce509772ab840d` |
| `local_patches/opendw_smoke_20261003/multigpu/tools/opendw_service.py` | 28912 | `25ce32aea304014f2f1c1cce9e2c0a3501480541e409f8da9fbc45bb2f2d7204` |


## 已用v5运维修复：按实际冻结依赖补入

| 本地来源 | 状态 / 发布要求 |
|---|---|
| `local_patches/opendw_smoke_20261003/drafts/scope_tools_v5/gpu_scope_runtime.py` | v5全清单CVD→GPU4兼容逻辑，已用于真实fresh-child CPU检查；服务器使用的精确字节核同后收入 |
| `local_patches/opendw_smoke_20261003/drafts/scope_tools_v5/opendw_smoke_gpu4_cycle.py` | 返回修复实际采用的7c019f…冻结cycle；作为历史返回流程收入，不替换正在用的旧文件 |
| `local_patches/opendw_smoke_20261003/drafts/scope_tools_v5/opendw_smoke_gpu4_cycle_preempt.py` | 显式WM优先的prepare选项；只有新GPU4子plan确实使用时才收入，记录first-round实际未确认状态 |
| `local_patches/opendw_smoke_20261003/tools/opendw_return_repair_owner.py` | 本轮实际返回等待owner；核部署字节后收入，结束/接管原因写入轻量摘要 |

上述历史文件须保持版本区分，不能把GPU4专用scope误称为通用4–7隔离实现。多卡子模块、scope helper和运行配置之间的SHA依赖，由新冻结plan完整列出。

## 需要补的实验日志：发布摘要，不发布大日志

| 建议新轻量文件 | 必须据实际回执记录的内容 |
|---|---|
| `evidence/smoke_v3_summary.json` | N8、N16实际配置和退出时间、返回均值、过滤组数、有效loss/grad、checkpoint完整性检查结果；区别工程跑通与有效学习 |
| `evidence/return_scope_v5_summary.json` | v3返回因CPU ChannelWorker全卡掩码被拒绝；v5精确全清单收窄、7个CPU mask案例/fresh-child结果；失败返回与后续WM优先接管事实，不能把未完成归还写成成功 |
| `evidence/multigpu_contract.json` | N64/G8/R8/L32、actor/rollout4,5、env/WM6,7各B1、GB512/micro8/U2、512轨迹/512chunks/2优化器步；服务路由与源码SHA；未启动时明确launched=false |
| `evidence/multigpu_smoke_summary.json` | 真正执行后再生成：每rank/服务计数、资源峰值、有效学习信号、错误、精确清理/归还状态；不得预写通过 |
| `evidence/assets_and_sources.json` | 官方repo revision、相对文件名、大小/SHA、reset来源和数量、policy/WM/RM配置来源；不含签名下载URL、密钥或完整环境变量 |

上述新摘要在盘点时尚未发现于本地evidence目录；只有已有 `publication_preflight_20261003.json`。不能用旧preflight替代实际smoke与返回结果。原生SFT/CP评估尚无本轮完成证据，应写未执行/待验证，不填成功率。

## 明确排除

- 全部 `local_scripts/sz3_status_20261003/*rpc*`、SSH连接/凭据脚本、一次性远程prepare/launch命令以及完整environment/plan回显。
- 模型权重、checkpoint/replay、reset NPZ、视频/图册、TensorBoard事件、原始训练/资源日志、依赖缓存与下载partial。
- 已取消的N16/R1/L384信号路线草稿：`drafts/signal_smoke_mode.patch`、`drafts/test_signal_smoke_mode.py`。现有 `tools/audit_opendw_signal.py` / `test_audit_opendw_signal.py` 只针对旧单卡R1/L384，未适配新合同前不作为多卡验收工具发布。
- 旧T5 legacy/prefix探测脚本、无关仓库dirty、根HANDOFF/PROJECT_CONTEXT全文及与本轮无关的实验文件。
- 尚未实际冻结/采用的其他drafts。不能将整个 `drafts/`、`local_scripts/` 或根目录批量加入。

完成发布的判据是：精确staged列表与差异审过 → 记录commit → push指定codex分支 → `ls-remote`返回同一commit → 独立轻量publication回执记录来源清单SHA、提交SHA和远端SHA。检查点/大日志未放Git，也没有由本轮确认独立对象存储备份；代码已推不等于模型与全部实验数据已异地备份。
