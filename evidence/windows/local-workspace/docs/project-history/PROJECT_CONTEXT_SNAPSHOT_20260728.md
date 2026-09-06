# AutoDL / RLinf 长期上下文历史快照（截至 2026-07-28）

## 项目入口

- 当前 Codex 项目根：`C:\Users\86136\Documents\rl`
- 原始资料与复用工具目录：`C:\Users\86136\Documents\Codex\2026-07-13\n`
- 本目录的 `HANDOFF.md` 是唯一动态入口；它按主题指向当前专题单一事实源。
- `C:\Users\86136\Documents\Codex\2026-07-13\n\autodl-rlinf-fastwam-handoff-20260716.md` 是历史 Fast-WAM 基线，只在追溯旧状态时按需读取，不再称为当前动态权威。
- DSRL 当前唯一计划：`C:\Users\86136\Documents\rl\docs\rlinf-robotwin-pi0-traditional-rl\00_INDEX_AND_IMPLEMENTATION_PLAN.md`；其 `01_FULL_REFERENCE_HISTORY_20260728.md` 只作历史材料，不默认读取。

## 安全边界

- AutoDL 入口：`ssh -p 36406 root@connect.bjb1.seetacloud.com`
- 已验证的非交互登录方式：用 Paramiko 读取当前进程中的 `SEETA_SSH_PASSWORD`，并设置 `look_for_keys=False`、`allow_agent=False`；先运行 `hostname; pwd; id -u` 等只读探针。可复用模式见 `C:\Users\86136\Documents\Codex\2026-07-14\ssh-p-34495-root-connect-bjb2\work\remote_exec.py`，但必须参数化为本机 host/port/user，不能原样调用其中硬编码的旧服务器配置。
- OpenSSH `BatchMode=yes` 无法响应密码提示不等于密码认证不可用。用户已经给出登录信息时，必须先实际尝试上述 Paramiko 主路径；AskPass、安装专用 SSH key 或改造服务器认证都不是默认前置操作。
- 默认只读检查。任何远端写入、中止进程、安装、下载、配置或代码修改都需要明确授权。
- Windows 本机只保存和编辑代码、文档与 diff；不在本机运行 Hydra compose、项目 import/compile、测试、smoke 或训练。所有运行放到服务器，并在执行前展示完整配置、命令、输出目录、资源预期和停止条件。

## 服务器稳定路径

- RLinf：`/root/autodl-tmp/RLinf`
- 旧 wamppo 备份：`/root/autodl-tmp/RLinf_wamppo_backup_20260714_step57_lastdcp40`
- RoboTwin：`/root/autodl-tmp/RoboTwin`
- π0 SFT：`/root/autodl-tmp/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle`
- GRPO 配置：`/root/autodl-tmp/RLinf/examples/embodiment/config/robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline.yaml`
- GRPO run：`/root/autodl-tmp/RLinf/logs/20260715_132507-robotwin_adjust_bottle_grpo_openpi_a800_2gpu_baseline-env16-rollout16-g8-step0-to-100`

## 可复用审计流程

1. 用 `tools/autodl-training-audit.ps1` 只读拉取日志、资源监控、配置和远端快照；脚本位于原始工具目录。
2. 用 `tools/analyze-rlinf-run.py` 生成 `analysis.json`。
3. 可视化覆盖从 `step 0` 到最新完整 step，并同时输出交互图与三张独立 PNG，供手机查看。
4. 重点检查：训练进程与 rollout 阶段、最新完整 step、success_once/5 步均值、KL/clip、step 时间、容器与 EnvWorker RAM、GPU、`memory.events`、OOM、checkpoint 完整性。

## 长期结论

- 正式训练使用官方 RLinf 新仓库；旧 wamppo 目录只作备份，不整体覆盖回来。
- 当前 GRPO 基线为 16 env、16 rollout epoch、group size 8，且 `env.train.enable_offload=true`。
- RoboTwin native/SAPIEN EnvWorker 是主机内存压力主体；不能把高内存简单解释为跨 step 无限增长的 replay pool。
- RoboTwin 仓库含其他实验修改；Fast-WAM 只能选择性复用已核验的 assets/task_config，不能把整个仓库当作干净基线。
- Fast-WAM 是否已安装或开始运行属于动态事实，新任务必须现场核验；未经授权不要安装。

## AutoDL 网络默认

- 学术加速仅临时包裹 GitHub clone；默认清除 `http_proxy`、`https_proxy`、`all_proxy` 的大小写变量，避免影响 pip/Hugging Face。
- 普通 Python 包和 editable install 默认走清华 PyPI 镜像；PyTorch CUDA index 只用于显式安装 Torch/Torchvision 的命令，不设为全局 extra index。镜像只改变传输路径，版本仍由项目官方配置锁定。
- Hugging Face 发布物默认在无学术代理状态下使用 `hf-mirror.com`，cache 放 `/root/autodl-tmp/cache/huggingface`；下载后记录文件身份和 hash。Fast-WAM 是明确例外：`yuanty/fastwam` release checkpoint 属于 Hugging Face，而 loader 重定向的 Wan VAE/T5/tokenizer 属于 ModelScope，必须保留官方默认 `DIFFSYNTH_DOWNLOAD_SOURCE=modelscope`，不能把两类仓库统一强制为 Hugging Face。
- 网络变量只在当前 shell/项目恢复块中设置，不修改全局 pip/conda 配置；某个源现场变慢时再按证据切换，不并行运行多个 pip。

## Fast-WAM 迁移设计入口

- 长期目标：在不破坏现有 π0 基线的前提下完成 Fast-WAM + RoboTwin + RLinf + GRPO/PPO。
- 已建立的稳定隔离布局：原 RLinf `/root/autodl-tmp/RLinf` 保持 π0 基线；Fast-WAM 集成 worktree 为 `/root/autodl-tmp/RLinf_fastwam_rlinf`；联合环境为 `/root/autodl-tmp/conda/envs/FastWAM-RLinf`；π0 venv 完整备份为 `/root/autodl-tmp/backups/RLinf-pi0-venv-golden-20260717`。新任务仍须现场确认这些路径和状态，不能仅凭本条称为当前。
- 锁定源码：RLinf `6d0db56bf26f972cd27fa29535f5eb939e80e5bf`，Fast-WAM `45d8e1458921d83f8ad6cf9ce993d371208dabd0`。GRPO主体和PPO增量均只落在独立worktree；原π0路径通过默认兼容分支保持不变，具体文件与动态状态见`HANDOFF.md`和唯一实施计划。
- 迁移总文档与索引：`C:\Users\86136\Documents\rl\docs\fastwam-robotwin-rlinf-grpo\00_INDEX.md`。
- 唯一实施计划：`C:\Users\86136\Documents\rl\docs\fastwam-robotwin-rlinf-grpo\05_IMPLEMENTATION_PLAN.md`。
- 官方 standalone 阶段 0 命令附录：`C:\Users\86136\Documents\rl\docs\fastwam-robotwin-rlinf-grpo\07_OFFICIAL_STANDALONE_RUNBOOK.md`。
- 新任务先读总文档；只有准备实施或讨论具体接口时，才继续读计划或相应技术附录。
- 当前默认决策是不为版本号先整体升级 RLinf；先从已验证基座派生独立分支/worktree，把适配放在内置 `rlinf/models/embodiment/fastwam/` 并通过官方 registry 注册。官方 Fast-WAM 模型仓库仍保持独立；`RLINF_EXT_MODULE` 只作 RLinf 不可改时的备选，不并行维护。
- 环境保持三份角色：现役 π0 golden venv 只读保留；`FastWAM-official` 只作 standalone oracle，不安装 RLinf；第三个 Fast-WAM×RLinf 联合 venv 在独立 worktree 的最终路径从空创建。联合环境只参考 π0 的 `freeze/list` 清单，不复制或搬迁前两份环境；固定最终 Torch 后重编全部 CUDA extension，π0 备份只作原路径恢复源。
- 迁移遵循“依据明确、适配有理由、语义完整”：最小化的是对 RLinf 核心的侵入，不是 Fast-WAM 模型、ODE/SDE 或训练链。每个目标 symbol 要能追溯到官方 Fast-WAM、当前 π0、社区 Fast-WAM×RLinf、Motus 或 LaWAM 的具体实现；没有可直接照搬来源的兼容改动必须写清不匹配点、修改理由和对应测试。train/eval 只通过参数控制预期差异，不维护两套近似实现。
- Fast-WAM 官方 singleton `infer_action()` 只作 B=1 oracle；生产 batch 只扩展其编排层，复用官方 batch-capable prompt/proprio/VAE/video/action/cache/scheduler，不做异常后逐环境 fallback。官方 scheduler 决定 shifted 网格和 `sigma_shift`，RLinf OpenPI 决定 Flow-SDE mean/std 与首步分母，社区实现只作拼接参考。
- 首版已冻结：内置 `rlinf/models/embodiment/fastwam/`、stochastic `k` 均匀覆盖全部 `0..S-1`、首个非零探索 smoke 使用 `noise_level=0.1`、`N=24` 工程 smoke 使用 192 steps。前三项分别依据 RLinf 内置模型/registry惯例与现有迁移实现、RLinf OpenPI 已验证概率定义、保守的工程稳定性起点；若现场 pin 或 resolved config 出现直接冲突，再记录证据后调整。
- Fast-WAM PPO首版固定从官方release base冷启动，critic使用官方最后一层`video_kv_cache[-1]["v"]`的observation-side token mean；它读取image/text/proprio而不读取action latent、denoise step或SDE noise。critic input detach，head参数跟随BF16模型，value输出转FP32；GRPO默认不创建head属性。
- 训练统一由数据盘联合环境的绝对 Python 路径启动，不依赖 Conda base 是否进入非登录 shell 的 `PATH`。专用 launcher 同步启动 π0 同格式的两秒资源监控，并将命令、resolved config、日志、PID、CSV 和 peak 放进同一 run 目录。
- 实施采用“先完成一个连贯的主体实现批次，再集中做少量高信息量检查”：不为每个小函数设置独立流程 gate；至少保留官方 B=1/真实 batch 数值 parity、一次非零真实 runner smoke、同步/恢复/导出三类验收。`lr=0`只在ratio排障时使用。

## 文档维护规则

- `PROJECT_CONTEXT.md` 只保存稳定背景和长期默认；`HANDOFF.md` 只负责动态状态入口与下一步。
- Fast-WAM 的目标、当前共识、待决策和关键风险统一维护在总文档；执行顺序只维护在实施计划。
- 参考、接口、代码映射、版本和测试文件均为按需附录，不要求每轮全部更新或读取。
- 每次讨论只更新真正发生变化的权威文档，避免同一内容在多份文件中重复和漂移。
- 操作方案优先简洁主路径；先完成相互依赖的主体实现，再做适量必要检查，实际遇到问题后再设计针对性修正。

## 算法移植的长期协作偏好

- 优先解决会改变算法含义或实验结论的设计问题；验收、协议和防御性分支只保留少量高信息量项目，避免计划膨胀。
- 新术语先用项目中的具体张量、次数或直白类比解释；Codex 应用内避免输出可能原样显示的 LaTeX，优先用普通文本和代码形式。
- 用户逐项提出的问题，要么在当轮回复中简答，要么给出权威文档的精确章节入口，并明确本轮更新了哪些文件和章节。
- 历史推导和暂缓路线不删除；移到带索引的非规范性历史文档。当前计划只保留活跃主线、关键依据、待决定项和下一步。
- 真正实施后维护完整实施账本，记录文件增删改、命令、结果、问题、修复、复测和资源峰值；主计划不承载长流水账。
- 任何服务器 smoke 或正式训练前，先展示完整有效配置、精确命令、输出目录、资源预期和停止条件，等待用户明确批准。
- smoke 和正式训练继续使用历史两秒监控方式，记录 GPU 显存/利用率、cgroup/主机 RAM 峰值和 OOM 计数。
- 遇到会改变方法定义、样本预算或主要实验解释的分叉时暂停讨论；普通实现细节在既定范围内自主推进。

## 可视化偏好

- 主要时间序列必须从训练开始展示到当前，而不是默认只显示最近窗口。
- 横轴从 `step 0` 起；没有 step 0 实测值时从 step 1 画线，不伪造零点。
- 默认保留桌面交互式图，并附成功率、GRPO 更新强度、内存/资源三张高分辨率 PNG。
- 静态图与交互图使用同一快照、坐标轴、阈值和注释。
