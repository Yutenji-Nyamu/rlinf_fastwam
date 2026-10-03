# WMRL 历史研究与恢复源码补充清单

准备日期：2026-10-04，Asia/Shanghai。此目录是本地待发布清单，**不是已推送回执**。补充范围为10篇已有WMRL研究/历史文档和两份实际使用过的恢复源码；不改变训练配置、资源调度或当前执行代码。

已有发布依据：OpenDW源码及轻量证据提交`0d9e90ae4eb6ecb6103ad10ebace5bbef373adcb`，后续发布/运行快照提交`9c8fe1722af79a6eed459cc4845f91957fd425d1`；旧LIBERO补推提交`21b5d590a9116d94e38c139a3ec4aa9723abbcc3`。本补包固定从`9c8fe1722af79a6eed459cc4845f91957fd425d1`继续，分支仍为`codex/robotwin-opendw-wmrl-20261003`。这些SHA来自已核发布回执，本准备步骤没有重新查询远端。

## 为什么补充

- 新OpenDW发布来源清单中的69个本地来源核SHA一致，没有发现现用多卡实现遗漏。旧清单来源SHA为`1f65520878761ead89232f0898d90f4fea566ad5b1551879d399db2ddf596dc9`，Git内`publication_manifest_v2.json`的SHA为`2142e1053cb553d16ffd40f3f6bfbfa442163e3caa43f2414bcdbc8330b9d8e9`。
- `multiview_20261003/public_options.md`在旧Wan发布后补充了A2World覆盖/C20条件和VLA-MBPO样例范围等实质核验，尚未收入新OpenDW提交。
- 其余9篇研究/历史文档未收入该发布包，其中多篇已被主上下文、接口审计和方法地图引用。本次补齐它们，保留原文，并在顶部明确调查时间与历史状态。
- 两份恢复源码用于10月3日GPU4返回修复尝试；随后等待owner被WM优先接管流程退休。它们作为历史字节归档，不是现行多卡启动入口，也不构成RLT首轮已验收的声明。

## 精确范围

文档保持原目录：

1. `docs/world-model/multiview_20261003/public_options.md`
2. `docs/world-model/OPENDW_PI05_GRPO_CONTEXT.md`
3. `docs/world-model/WORLDARENA_FIRST_REPRODUCTION_PLAN.md`
4. `docs/world-model/robotwin_pipeline_20261003/server_status.md`
5. `docs/world-model/robotwin_pipeline_20261003/discussion_only_20261003.md`
6. `docs/world-model/robotwin_options_20261003/OVERVIEW.md`
7. `docs/world-model/robotwin_options_20261003/a2world.md`
8. `docs/world-model/robotwin_options_20261003/vla_mbpo.md`
9. `docs/world-model/robotwin_options_20261003/robotwin_prior.md`
10. `docs/world-model/robotwin_options_20261003/framework_and_extra_sources.md`

历史代码发布到`local_patches/opendw_smoke_20261003/archive/return_repair_v1_20261003/`，不覆盖当前源码：

| 本机来源 | 归档文件名 | 不变的SHA256 |
|---|---|---|
| `local_patches/opendw_smoke_20261003/tools/opendw_return_repair_owner.py` | `opendw_return_repair_owner.py` | `51f5206a9903e73c4076910131f4e6c0b30d49c19d368fa2964b89990f30c597` |
| `local_patches/opendw_smoke_20261003/drafts/scope_tools_v5/opendw_smoke_gpu4_cycle.py` | `opendw_smoke_gpu4_cycle.py` | `7c019fea53ef7b27790bdbacb8b072cb31c0f2a55b89cc496fa56065582c0761` |

归档脚本含当时的精确目录、进程/回执门禁和模块依赖，供审计，不应对现有运行重放。SHA相同证明与本机留存字节一致；实际使用出处沿已保存返回修复回执追溯，本准备步骤不重验服务器现场。

## 本次文档修改

10篇均仅新增顶部时间/历史说明。讨论暂停页明确旧“最新用户边界”已被后续授权替代；深圳2状态页只代表10月3日18:20–18:30；两份旧路线只代表9月30日方案。其余公开研究均限于10月3日固定源码调查，不声称10月4日重新搜索。

链接只改必要失效项：深圳2状态页的Wan部署链接指向已发布`21b5d590…`的固定文件；本机原始回执、未随包发布的只读脚本及旧基线文档改为明确标注的本机留存路径。保留原路径及原有事实，不将本机路径伪装为可访问的云端材料。

精确字节、来源映射、改动前SHA和补包SHA由同目录`manifest.json`记录。生成器位于本机`local_scripts/sz3_status_20261003/make_opendw_research_followup_rpc.py`，不收入Git。生成的服务器RPC仅验证前置HEAD、按精确清单暂存并写审阅回执，不commit/push，不导入或运行归档脚本。

## 排除与完成标准

继续排除SSH/RPC脚本、环境字典、完整运行plan、密码、模型/检查点、reset数据、原始日志/TensorBoard/媒体、取消的N16草稿和无关工作区文件。本次没有新增训练或真实环境成功率结论，也没有大文件异地备份声明。

完成发布需要另行审阅精确staged列表与diff，commit后推送上述分支，再核远端SHA并写新回执。本清单中的`committed=false,pushed=false`仅描述准备时刻。
