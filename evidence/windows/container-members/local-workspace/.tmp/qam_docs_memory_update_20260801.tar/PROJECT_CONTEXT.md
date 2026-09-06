# AutoDL / RLinf 长期工作区上下文

## 入口与文档层次

- 当前工作区：`C:\Users\86136\Documents\rl`
- `AGENTS.md`：稳定执行规则。
- `HANDOFF.md`：唯一动态入口，记录当前主题、已取得授权、最新状态和下一步。
- `docs/project-history/00_INDEX.md`：根目录旧规则、旧交接和跨专题历史索引。
- 各专题目录：该专题唯一计划、参考材料索引、实施账本与证据。具体模型、参数、服务器路径和实验结论不得长期堆在本文件。

## 稳定的事实与安全原则

- 动态事实以服务器现场刷新为准；交接文件和历史记录只提供定位线索。
- 远端密码只存在于当前进程，不写入脚本、文档、命令历史或仓库。密码 SSH 使用 Paramiko，并关闭 key/agent 自动尝试。
- AutoDL 密码 SSH 固定使用已校验 host-key 的低层 Paramiko `Transport`：只对认证前
  banner/EOF/timeout/reset 最多重试 3 次（1/3 秒退避），认证成功后
  `keepalive=30s`；密码认证失败或主机指纹变化立即停止。包含 `$`、引号、管道或循环的
  远程 shell 一律走 UTF-8 command-file，避免本地 PowerShell 提前展开；同一批多条只读
  检查尽量合并在一次连接中，绝不自动重放已经发出的有副作用命令。
- 授权按当前任务和动作范围判断。可逆、只读检查可以主动进行；写代码、基础测试、smoke、训练、进程控制、依赖变更和删除操作分别遵守 `HANDOFF.md` 中的当前边界。
- Windows 本机承担文档、代码和 diff 的保存与审阅；依赖项目运行时的检查通常放在服务器。大模型、数据集、checkpoint、日志和环境均留在数据盘。
- 不把旧快照描述成当前状态；任何恢复、续训或发布动作都先核对仓库、分支、dirty tree、运行进程和目标产物。

## 稳定的开发与记录原则

- 迁移实现必须能追溯到来源 commit/path/symbol，并记录必要适配及其理由；继承行为、确认缺陷、开放风险和新方法改动要明确区分。
- smoke 不得通过缩小模型、改变算法定义或绕开正式调用链来制造通过；只允许调整有明确用途的步数、保存/评估频率等运行预算。
- 任何实验只有在启动前展示并明确批准 `cycles / train episodes / action slots /
  replay transitions / critic-actor updates / eval episodes / checkpoints / wall-clock /
  GPU-hours` 对齐表后才能命名为 formal；未完成规模对齐或仅用于阶段转换观察的运行一律
  命名为 pilot。不能用“跑一晚”代替样本与更新预算，也不能只按另一环境的单一 `step`
  字段声称等规模复刻。
- 优先做端到端连贯实现和集中验收，不为每个小函数建立流程 gate。
- 不把无论文、官方代码或确定接口合同依据的经验阈值升级为正式训练的硬 gate。结构/字段/
  shape 合同错误与 NaN/Inf 等确定性无效状态可以 fail-fast；relative-L2、cosine、梯度大小等
  经验诊断默认只记录和告警，不阻断训练。任何新增硬 gate 必须在启动前说明来源、必要性和
  触发后果，并取得用户明确同意。
- 实施账本从第一条操作开始维护，包含命令、文件、结果、错误、诊断、修复和复测；最终交接引用账本而不是重复全部过程。
- 每个通过最小验证的连贯代码/专题文档批次默认主动 commit 并 push 到当前专题既有云端
  分支，无需重复请求发布授权；不把未验证半成品、dirty tree 或无关用户改动塞进提交。
  GitHub 默认先用无 proxy 的短探针和有界 push；主站/smart-HTTP 明确超时而 API 正常时，
  只在一个子 shell 内临时 `source /etc/network_turbo`，完成 `ls-remote/push/复核` 后
  退出，不持久化 proxy、不改 remote/Git config，也不 force push。
- 专题当前计划是单一事实源；历史版本只读归档并由索引链接。根目录高层文件保持短小、跨项目可复用。

## 稳定的结果呈现原则

- 训练曲线覆盖从 `step 0` 到最新完整 step；真实数据从首个记录 step 开始，不虚构 step 0 数值。
- 需要训练可视化时，同时提供桌面交互图和手机可读的独立 PNG，至少区分成功率、优化指标与资源指标。
- 报告区分策略查询次数、环境/仿真步、transition 数、optimizer update 数、GPU-hours 和 wall-clock，避免用一个“step”混指不同预算。
