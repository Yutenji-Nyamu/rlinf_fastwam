# 深圳本人实验Git覆盖复核（2026-09-05）

范围：`/data/chenyiteng/projects/rlinf-shenzhen`的RLinf/RoboTwin共同Git仓库及全部链接工作树，包含项目根外的DSRL树；只涉及chenyiteng，不审计/发布liwenbo。Git远端通过live ls-remote核验，不把本地remote tracking ref当作远端现场。

## 当前结论

- 已遍历23个工作树、22条`codex/*`本地分支。10:50审计时21条与personal远端HEAD完全相同；唯一没有同名远端的是RLT checkpoint诊断分支。
- BC主实现2bb8bd40及后续原生精度/增强off源码`9876c28de25bba429d10e2c8b4c85fb19b24c387`已推/远端一致，不再属于“未推代码”；smoke轻量证据在结束后封存。审计与源文件同步并行，BC瞬时dirty不解释成他人改动；最终状态以最后commit后复核为准。
- 原15个算法分支的轻量evidence仍在：1265个tracked文件，共39.51MiB（文件净大小；旧账本41MiB含目录/统计口径差异）。checkpoint/模型/训练数据/视频及完整Ray日志不应塞Git。
- **不能说所有历史小材料都已备齐**：最新Fast四次结束运行已在本轮补回对应Fast分支（见下节）；Sidney正在跑的pillbottle formal最终证据尚未封存。部分早期PPO一次更新/重载评估、RLT诊断、OIDN诊断脚本等也需按实际源路径另行归档。
- RLT诊断树有`fsdp_model_manager.py`与`strategy/checkpoint.py`两处dirty；OIDN toggle树有未跟踪`oidn_toggle_trial.py`。未把这些未审阅修改冒充验证过的算法实现提交，也没有删除它们。

## 覆盖计数的边界

自动扫描识别87个run/诊断目录，其中57个可通过已tracked的SERVER_RUN_PATH直接映射；剩余30项**不等于30个完全没备份的实验**。有的Sidney证据使用另一种布局/manifest，有的是runtime子目录或仅准备未运行目录。这里只将可直接映射作为正证据，不凭无该marker就断言所有文件都缺失；原始清单保留供精确回填。

本轮不更新正在运行的Sidney源码、参数或进程；不强推、不改remote、不迁移大文件。Fast仅新增轻量evidence，不改变训练代码。

原始[细化审计JSON](SHENZHEN_GIT_COVERAGE_REFINED_20260905.json)；[09-03回填账本](SHENZHEN_LIGHT_EVIDENCE_GIT_BACKFILL_20260903.md)；BC运行[实施账本](../rlinf-robotwin-pi0-online-bc/evidence/GPU6_SMOKE_LEDGER_20260905.md)。

## 本轮已补推的近期Fast证据

`codex/sz-fastwam-current-rlinf-grpo@f3a1689e5ceb65446d3e6feabe055ef2ddffbab4`已推personal，远端SHA一致且工作树clean。仅新增`evidence/completed_runs_20260903_20260905`，109个tracked文件；103份源产物共1,190,314字节，另有manifest与索引。覆盖旧renderer-life续训、clean/noOIDN停滞v1、scene-fence启动失败v2、最新Step17退出v3。

这次没有更改Fast训练代码或重启任务；原实现源锁仍为62526cc。模型/checkpoint/数据/视频/完整Ray日志排除；复制前检查已结束元信息、尺寸和凭据模式。clean/noOIDN v1的wrapper exit0来自明确停止，不把它写成训练成功完成。四次完整原始run仍在服务器，未删任何产物。
