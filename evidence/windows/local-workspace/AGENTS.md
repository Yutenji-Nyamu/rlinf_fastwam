# RLinf 工作区规则

- 新任务先读短入口PROJECT_CONTEXT.md与HANDOFF.md；随后只读当前问题所需的专题章节。09-03窗口交接、history、旧快照不再默认全文加载。连续追问沿用已读上下文，未变化文件不重复全文读取。
- 动态训练step、进程、GPU/RAM、fatal、checkpoint、Git状态先刷新服务器；文档/聊天/Memory只作定位线索，必须区分SZ与AutoDL。
- 当前用户请求决定授权。诊断/讨论不自动授权改训练；smoke、正式训练、停止、依赖变更、删除分别按明确授权执行。不得干扰其他用户或重启共享Ray。
- 深圳项目使用本人chenyiteng。密码只在当前进程；使用既有固定host-key Paramiko，禁用自动key/agent尝试，先只读身份探针；认证失败/指纹变化即停。复杂shell用UTF-8命令文件，副作用命令不自动重放。
- Windows用于文档/源码/diff；依赖项目的测试、compose、smoke和训练在服务器。保留用户dirty tree，不擅自覆盖；实现从第一步维护专题账本。
- 比较实验从目标Control实际resolved逐叶继承，方法外的采样/并发/batch/U变化单独确认；不把其他模型默认值当当前协议。真正启动前给出配置、命令、输出、预算、资源及停止条件，按当前授权执行。
- 只做一个连贯实现批次和少量高信息检查；问题驱动窄修，不堆猜测性兜底。发布只含审查过的变更/轻量证据，先排除凭据、视频、模型/大checkpoint；不用force push或全目录盲add。
- 当前计划只维护在专题SSOT；根HANDOFF仅放当前路由/授权/下一步，旧时间线无损归档。结束时更新短HANDOFF；长期习惯才改PROJECT_CONTEXT。使用习惯和呈现要求见该文件。
