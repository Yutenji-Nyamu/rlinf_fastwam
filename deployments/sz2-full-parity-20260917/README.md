# sz2 实验环境入口

已按深圳1的固定版本部署66个实验源码树、137份可选择的运行配置和142份原始实配参考。所有树使用独立的 `codex/sz2-...` 分支；原有GRPO实验源码、环境和Ray保持原状。此准备没有启动训练。

```bash
source ~/experiments-env.sh bc
python "$EXPERIMENT_SETUP" list --tree bc
python "$EXPERIMENT_SETUP" prepare --help
```

可把 `bc` 换为 `grpo`、`grpo-clean`、`bc-dvac`、`rlt`、`rlt-tau`、`stage1`、`fastwam-bc`、`fastwam-rlt`、`sarm`、`iql`、`attena`，也可用registry里的完整树名。selector只设置环境，`prepare`只生成计划；都不会开始训练。没有默认profile的分支需从 `list --tree`/源码配置里明确选配，不能视为正式验收通过。

主Python环境：`/data/chenyiteng/venvs/rlinf-sz1-parity-py311-20260917`。RynnValue独立评分环境：`/data/chenyiteng/venvs/rynnvalue-8b-py310`。使用原机已成功的CUDA/Vulkan入口及内网NO_PROXY；BC/BC-DVAC继承instruction-reset-fix源码，资产仍指向已对齐的共享RoboTwin支持目录。

`registry.json`记录树、HEAD、branch、profile及源码证据等级；`profiles/<id>/template.json`保存已映射到本机的新运行模板，`environment.json`记录对应源码和资产；`reference-configs/<id>/resolved.yaml`保留深圳1实配。评估/训练种子存于各源码树的 `rlinf/envs/robotwin/seeds/`，113份对应输入已逐SHA核对。模型路径在 `required-model-paths.json`，大模型下载/校验结果以兄弟目录 `../models/` 的最终回执为准；独立环境验收以 `../env/ENVIRONMENT_READY.json` 为准。

未来启动先确认本人的GPU空闲，使用新的结果目录、专用Ray地址/namespace，并保留原配置GPU数量和预算；`prepare --dry-run`可打印完整差异。传入GPU只是生成计划，不启动或终止任务。

RLT Stage1源码和原2000步训练配置已具备，但这次不训练Stage1、也不搬历史teacher权重。Stage2必须提供未来 `--stage1-checkpoint` 及 `--stage1-manifest`，恢复profile还需其完整 `--resume-checkpoint`。FastWAM RLT在深圳1只有短smoke，本机已配源码/组件，但无正式默认profile。历史失败/未匹配运行记录的分支仅归档源码，不能称为已可正式训练。

本机family默认选择见 `aliases.json`。每个profile的预算/方法来自原实配；除源码/环境路径和将来显式指定的资源/输出外，不自动更改采样数、batch、更新数、种子或训练步数。
