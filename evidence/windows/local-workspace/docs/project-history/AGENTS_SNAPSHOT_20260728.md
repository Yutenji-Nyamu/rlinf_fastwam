# AutoDL / RLinf 规则历史快照（截至 2026-07-28）

- 每个新任务开始时，先完整读取本目录的 `PROJECT_CONTEXT.md` 和 `HANDOFF.md`，再按当前主题读取 `HANDOFF.md` 指向的唯一专题文档；外部旧交接只作历史追溯，不再默认完整读取。
- 涉及 Fast-WAM 迁移时，再读取 `docs/fastwam-robotwin-rlinf-grpo/00_INDEX.md`；只按当前问题继续读取它指向的实施计划或技术附录，不默认加载全部文档。
- 动态实验状态的可信顺序：服务器只读现场检查 > 权威交接文件 > 旧任务聊天或 Memories。训练 step、GPU、RAM、日志和 checkpoint 必须现场刷新后才能称为“当前”。
- 默认只读服务器。停止训练、杀 Ray、修改配置或代码、安装依赖、下载模型、删除文件前，必须取得用户明确授权。
- 本机只用于读取、编辑和保存代码、文档与 diff；不在本机运行 Hydra compose、项目 import/compile、测试、smoke 或训练。上述运行全部放到服务器，并继续遵守先展示 diff/完整配置/命令/输出/资源/停止条件、再取得明确授权的边界。
- AutoDL 密码登录的已验证主路径是 Paramiko：用户已提供地址和密码时，把密码注入当前进程，设置 `look_for_keys=False`、`allow_agent=False`，先执行只读身份探针。`ssh -o BatchMode=yes` 的 `Permission denied (publickey,password)` 只表示 OpenSSH 无法回答交互式密码提示，不得据此宣称服务器不可登录，也不得把 AskPass、安装 SSH key 或更换认证方式设为前置条件；只有 Paramiko 密码认证实际失败后才报告认证阻塞。
- 训练可视化默认覆盖横轴 `step 0` 到最新完整 step；真实数据从首个记录 step 开始，不虚构 step 0 数值。每次同时提供桌面交互图和手机可读的独立 PNG：成功率、优化指标、资源指标。
- Fast-WAM 接入是可追溯迁移：每个目标 symbol 要记录来源 commit/path/symbol、必要适配和对应集中验收；新增或偏离参考实现的地方要记录不匹配点与修改理由。不得为了 smoke 萎缩模型、概率定义或 RLinf 常规 rollout/actor/loss/sync/DCP 调用链；train/eval 必须共享 adapter、conditioning、scheduler 和 denoise core，只保留有依据的参数化差异。优先完成一个连贯的主体实现批次，再做少量高信息量检查，不为每个小函数设置独立流程 gate。
- 逐文件讨论 Fast-WAM 接入时，每批先用一句话说明该文件在调用链中的位置、直接上游和直接下游，再讨论来源、适配与设计选择；每批结论及时写回唯一实施计划。
- 任务结束交接时：更新权威交接文件的当前状态、待办、风险和下一步；只有长期规则变化时才更新 `PROJECT_CONTEXT.md`。
- 文档保持单一事实源：`00_INDEX.md` 是迁移总文档，`05_IMPLEMENTATION_PLAN.md` 是唯一实施计划；不新建重复计划或重复交接。
- 实施时每次只推进一个阶段，只做必要操作和必要检查。失败后保留最小证据并讨论一个修正，不预先铺设多套复杂兜底。
