# Rynn 二值短测与 click_bell WMRL 发布准备

本目录维护源码、文档和轻量实验证据的精确发布范围。当前为准备状态；尚无本轮推送完成回执。

- 父提交：`382b2698a1ca8853343e32ae7c0b688a38521933`。
- 新分支：`codex/wmrl-bell-reward-20261005`。
- 独立服务器发布 checkout：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/publication/wmrl-bell-release-v1`。
- 服务器发布事务回执：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/publication/wmrl-bell-release-20261005-v1`。
- 清单生成器：`local_scripts/wmrl_bell_20261005/make_publication_rpc.py`；不执行 SSH、Git 或训练。

`--inventory-only` 列出本地候选与待补文档。默认模式在最终文档及 `light_receipt.json` 就绪后，冻结逐文件 SHA256，生成 `publication_stage_remote.py`、`publication_push_remote.py`。生成 RPC 本身不收入 Git。

stage 仅在新独立 checkout 写文件、精确暂存并产生 staged diff；审阅后由根任务执行 push RPC，提交并核远端 SHA。训练与原生评估 checkout、原 Git index/分支和正在运行进程均不修改。推送不得 force。

## 发布前所需的最终输入

- `docs/world-model/robotwin_pipeline_20261003/rynn_binary_probe_20261005.md`：一批原生成败视频的 Rynn 二值判定结果，明确样本数、标签来源、未知输出与局限。
- `docs/world-model/robotwin_pipeline_20261003/click_bell_execution_20261005.md`：本次任务切换、配置差异、小 RM 核验、smoke/正式真实状态及 RLT 路由。
- `light_receipt.json`：根任务审阅的脱敏轻量证据。不得复制整个 owner plan、environment、原始日志或凭据。
- 实际生成的四份源码与两份配置：取回服务器 `click-bell-v2/generated/{rlinf/envs/world_model/opendw_adapter.py,rlinf/envs/world_model/opendw_robotwin_env.py,owner/opendw_multigpu_owner.py,owner/opendw_formal_owner.py}` 到本机 `local_patches/click_bell_20261005/generated/`，以及 `prepared/{formal.yaml,startup_smoke.yaml}` 到该目录的 `config/`。两个环境模块同时映射到分支规范 `rlinf/envs/world_model/` 路径，避免发布分支继续保留上一实验的 Rynn 环境实现。

轻量回执需含以下字段，值来自实际收集：

```json
{
  "ready_for_publication": true,
  "owner_dir": "/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/runs/click-bell-v2",
  "owner_plan_sha256": "实际64位SHA256",
  "runtime_repositories": {
    "wm": {"path": "/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rlinf-opendw-bell-v1", "head": "2151a08ee1bd75df1bef0d8190e594bd5c7f7977"},
    "native": {"path": "/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/rlinf-rynn-binary-v1", "head": "2151a08ee1bd75df1bef0d8190e594bd5c7f7977"}
  },
  "source_bindings": {
    "本机发布相对路径": {"remote_path": "实际使用的服务器源码路径", "owner_frozen": true}
  },
  "evidence_pins": {"已完成且不再改变的结果回执绝对路径": "实际64位SHA256"}
}
```

`source_bindings` 至少包括 `prechecks.py`、`rynn_binary_probe.py`、`probe_reward.py`、`prepare_formal_remote.py`；其余实际运行源码也应逐项绑定。`owner_frozen=true` 额外核其 SHA 在 owner 的冻结清单中；准备/历史代码可用 false，但仍比较远端真实字节。运行生成文件尚未取回时，不把生成器声称为已经发布了所有生成后的源文件。

当前入口为 v2，沿用 v1 的策略 checkout、RM/起点资产与采样预算；新增两份 `prepare_*_v2.py` 同样绑定实际 SHA。v1 在原生 launcher 的 Ray namespace 接线处失败，其精确清理及 RLT 归还证据写入主轻量回执；`light_receipt_v1_launch.json` 仅保留失败前的启动快照。禁止把该历史快照当作当前状态，禁止重放 v1 启动入口。

轻量回执可另外包含固定指标、检查结果和运行边界，不要求训练结束；应使用已完成的报告及稳定启动回执，不能把持续变化的 `state.json`、`resources.jsonl` 当成冻结证据。

## 包含与排除

包含二值采集/推理、任务配置/适配、有限预检查与资源借还必要源码、必要准备/启动/状态查询入口、对应 CPU 检查、两篇结果文档以及盘点列出的8份历史小遗漏。两段 Wan 历史追加以 `publication_inventory_20261005.md` 中的摘录补存。清单 `allowlist.json` 是准备时快照，`source_manifest.json` 是最终冻结包；发布成功以独立 `published.json` 与远端 SHA 为准。

继续排除本地 `reference/` 环境文件、失败 relay/SSH session helpers、生成 RPC、密码与完整环境字典、模型/检查点/reset 数据、视频、原始大日志、混合其他任务的 HANDOFF。大文件留服务器，不将本次 Git 发布描述为大文件异地备份。

## 按铃结果的同分支补充

首发包含已完成Rynn32结论与按铃22:31进行中记录，不等待按铃全部结束。首发推送核验后，将真实 `published.json` 取回本目录。后续仅补四份明确文件：`click_bell_execution_20261005.md`、本目录的 `bell_reward_result.json`、`light_receipt.json` 和首发 `published.json`。

补齐后运行 `local_scripts/wmrl_bell_20261005/make_followup_publication_rpc.py --parent <已核首发40位SHA>`，只生成续提交stage/push脚本，由根任务审阅执行。它复用独立发布checkout，要求本地HEAD与远端同为已核父提交、工作区全净，并再次核验运行源码及稳定结果哈希；不改训练、不新建调度层、不force push。结果脚本和生成payload目前均未执行。
