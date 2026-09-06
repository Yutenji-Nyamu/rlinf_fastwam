# 2026-09-06 归档、精确清理与短上下文结果

## 1. 已完成的磁盘清理

本人checkpoint精确文件批次；未删除目录，未触碰其他用户、模型SFT起点、现役训练或shared Ray。

| 类别 | 删除大文件 | 释放GiB | 保留 |
|---|---:|---:|---|
| 历史smoke | 38 | 296.95 | 所有配置、日志、小记录；不再保留这些smoke的大权重 |
| DSRL | 12 | 90.35 | 最后Step200必要权重 |
| Sidney π0.5 GRPO | 39 | 349.03 | 清单时最后Step150；当前续训命令引用的Step100也保留 |
| π0.5 BC | 7 | 58.11 | 清单时最后Step40及之后新生代 |
| π0.5 BC＋DVAC | 7 | 57.92 | 清单时最后Step40及之后新生代 |
| 合计 | **103** | **852.36** | 正式保护清单23个文件前后stat全部通过 |

只删>1GiB常规文件，较小的checkpoint metadata/learner/DVAC记录等未删。大于1GiB的旧代success_replay备份也在授权清单中；原success_data池和最后代回放未动。保留最新完整代不代表本轮重跑了恢复测试。

11:23清理后df可用：/data约1.447TiB，/home约1.217TiB；同时训练继续写文件，最终应以最新df为准。原始记录：`SZ_EXACT_PRUNE_EXECUTION_20260906.txt`、`SZ_PRUNE_STEP10_EXACT_EXECUTION_20260906.txt`。删除不可从Git/日志恢复；仍有历史曲线不等于历史模型还可复评。

## 2. Git的范围

用户补充：一类文件即使每个很小，总量很大的replay/raw数据也不推。按此排除tensor、回放、视频、大权重、编译缓存、凭据；配置、源码、日志、指标、图表、证据清单保留。大文本可无损gzip；旧ZIP/TAR按成员重新过滤，不盲推内含大文件/凭据的容器。

既有个人仓库：`Yutenji-Nyamu/rlinf_fastwam`。独立artifact-only分支：`codex/sz-experiment-archive-20260906`；不把档案混入在跑的算法分支，不修改现有dirty文件。22个既有RLinf分支此前现场远端一致；RoboTwin远端查询HTTP2失败，所以本次同时保存其实际源码快照与HEAD，不假称其remote本轮核验成功。

服务器阶段已收集55939个轻量文件（含26个worktree的大量重复源码，Git会按对象去重），842857396字节；其中results直接收集5852项。两处RLT诊断改动和RoboTwin试验脚本作为“未验证快照/patch”收录，不当作生产修复提交到原分支。

本地及最终commit/push状态在结束前补录；未核验远端前不称已完成发布。

## 3. 上下文裁剪已生效于后续读取规则

原强制入口约64.55KiB；新AGENTS、PROJECT_CONTEXT、HANDOFF合计约6.69KiB，减少约90%。旧根文件逐字复制并SHA256核验，存`docs/project-history/context-20260906-before-trim/`；研究专题和证据不删。

保留用户习惯、模型/方法参数分工、授权边界、source-lock、共享服务器安全、逐项直答和图表偏好。删除根入口内过时过程，取消默认全文读09-03窗口历史；连续追问不重复读未变化文件，专题只取相关章节。

不是439份docs都会自动载入，也不是每条消息都会自动重读全部HANDOFF；旧工作区规则额外要求的手动全文读取才是可直接减掉的负担。依据[OpenAI AGENTS说明](https://learn.chatgpt.com/docs/agent-configuration/agents-md)及本地规则读取审计。本轮用OpenAI Docs技能核对加载边界，避免误把所有文档都当自动上下文。

当前已在对话里的旧文本不会因磁盘裁剪而从本轮历史消失；后续新任务/按需恢复才享受短入口，不声称已经改动应用自动压缩或模型上下文窗口。

## 4. BC审计入口

见`../rlinf-robotwin-pi0-online-bc/evidence/PI05_BC_DVAC_FULL_DIFFERENCE_AUDIT_20260906.md`：17项预期配置差异、28项CPU回归、真实首轮单位权重、动作RNG缺口和U/范围讨论。现役训练本轮未修改。
