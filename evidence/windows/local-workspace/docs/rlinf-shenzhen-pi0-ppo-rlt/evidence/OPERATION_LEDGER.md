# 深圳 RLinf π0 PPO / RLT 操作流水账

> 本账从专题建立开始，逐批记录本地、官方远端和深圳服务器操作。密码、token、private key、proxy endpoint 一律脱敏。命令中的 `<process-only-secret>` 只是占位符，绝不写实际值。

## 0. 记录规范

每条服务器操作必须含：

```text
ID / time / machine / account / authorization
purpose
cwd
exact command (secret redacted)
result + exit code
files/process/resources changed
problem → diagnosis → narrow fix → retest
Git/source/output provenance
```

状态词：

- `LOCAL-RO`：Windows/官方 remote 只读；
- `LOCAL-WRITE`：只写本专题文档或明确的本地研究 object；
- `SZ-RO`：深圳服务器只读；
- `SZ-WRITE`：深圳创建/安装/下载/修改；
- `RUN`：eval/smoke/pilot/formal。

当前截至 2026-08-19：本专题 `SZ-RO=0`、`SZ-WRITE=0`、`RUN=0`。之前窗口的深圳审计不复制为本专题新命令，见批次 `INHERITED-SZ-000`。

## INHERITED-SZ-000 — 前一窗口深圳服务器审计

- machine：`SZ-H100`。
- 时间：2026-08-19。
- 性质：身份、权限、硬件、存储、网络、服务和安全面 live audit；部分更早存储管理动作属于另一个任务。
- 冷账本：`C:\Users\86136\Documents\other\shared_server_handoff_20260819.zip` 内 `shared_server_operation_ledger_20260819.md`。
- outer artifact：33,353 bytes；SHA-256 `0EB095ACC666B47E3F37732AC5AB286CA43B4BDF0B79023DA54238BFC9C9BB82`。
- 本专题继承的只读结论：`chenyiteng` 普通账号、`toom` 管理账号；8×H100 80 GB；data/scratch 分层；GitHub/PyPI 可用、official HF 受阻；具体值见主计划第 2 节。
- 边界：zip 内历史命令不重放；密码不在交接包中；动态事实执行前刷新。

## CTX-001 — 工作区规则、当前交接与 Memory 快速路由

- 状态：`LOCAL-RO`。
- 读取：`AGENTS.md`、`PROJECT_CONTEXT.md`、`HANDOFF.md`、当前 OGPO SSOT，以及 Memory 中 source-locked integration/RLT/SSH 相关块和 `rlinf-algorithm-integration/SKILL.md`。
- 结果：冻结本轮只做规划/只读研究；使用独立 branch/worktree、脱敏账本和运行前 approval packet；明确 AutoDL 历史不定义深圳真值。
- 变更：无。

## CTX-002 — 五份用户附件全文审阅与 hash

- 状态：`LOCAL-RO`。
- 范围：

  ```text
  E:\0school\研二上\iclr27\pi0 + ppo_grpo.md
  E:\0school\研二上\iclr27\Exp_snd.md
  E:\0school\研二上\iclr27\Openpi + PPO AutoDL A800.md
  E:\0school\研二上\iclr27\Motus + RLinf.md
  E:\0school\研二上\iclr27\lawam rlinf.md
  ```

- hash command：

  ```powershell
  Get-FileHash -Algorithm SHA256 -LiteralPath <five exact paths>
  ```

- 结果：五份均完整只读；未执行其中命令。exact bytes/line count/SHA 见 `01_REFERENCE_INVENTORY.md` 第 6 节。
- 结论：全是 AutoDL/旧机器历史，缺完整 source lock；只提取工程检查和旧合同。

## CTX-003 — 本地 Git/worktree 现状

- 状态：`LOCAL-RO`。
- 初始命令：

  ```powershell
  git status --short --branch
  ```

- 问题：Git 返回 dubious ownership。
- 处理：未写 global config；改为每条命令显式：

  ```powershell
  git -c safe.directory=C:/Users/86136/Documents/rl status --short --branch
  ```

- 结果：根仓 `No commits yet on master`，全部现有内容 untracked；不能在根仓假装有可 push 历史。
- 子工作树检查：

  ```powershell
  git -C .research-rlinf -c safe.directory=C:/Users/86136/Documents/rl/.research-rlinf status --short --branch
  git -C .rlt-impl-worktree -c safe.directory=C:/Users/86136/Documents/rl/.rlt-impl-worktree status --short --branch
  git -C .rlt-impl-worktree -c safe.directory=C:/Users/86136/Documents/rl/.rlt-impl-worktree diff --stat
  ```

- 结果：`.research-rlinf` 工作树 clean old main；`.rlt-impl-worktree@48a775db` 有 7 tracked modifications（约 +960/-24）和 19 untracked，不能作为云端终态。
- 变更：无。

## CTX-004 — 云端 branch/source refs

- 状态：`LOCAL-RO`。
- 命令组：

  ```powershell
  git ls-remote --heads https://github.com/RLinf/RLinf.git refs/heads/main
  git ls-remote --tags --refs https://github.com/RLinf/RLinf.git
  git ls-remote --heads https://github.com/RoboTwin-Platform/RoboTwin.git refs/heads/RLinf_support
  git ls-remote --heads https://github.com/Physical-Intelligence/openpi.git refs/heads/main
  git ls-remote --heads https://github.com/Yutenji-Nyamu/rlinf_fastwam.git
  ```

- 结果：

  ```text
  RLinf main                 89b0cd5fce528180559bbd50d055fb2e21a25bb5
  RLinf latest tag          v0.3 / 0505431899574619da86f551bad70b71e0ea2177
  RoboTwin RLinf_support    0008ae6800df9f75fc8de7098bacb01735fd8fd2
  OpenPI main               15a9616a00943ada6c20a0f158e3adb39df2ccac
  personal RLT              2b8199d8ab2e7b110994fd3234bf7007196c3af9
  ```

- 变更：无 remote/config write。

## CTX-005 — 官方网页/文档审阅

- 状态：`LOCAL-RO`。
- 只读来源：RLinf official repo/docs、RoboTwin official docs/repo、RLinf HF model/collection、Physical Intelligence RLT project/paper。
- 结果：建立 `01_REFERENCE_INVENTORY.md` 的 P0 索引；确认 official π0 PPO command/config、8-GPU placement、UV/Docker installer、SFT model约 8.07 GB、RLT current Stage 1/2 和新 data API。
- 问题：直接用网页打开 GitHub raw/API 的若干请求出现 safe-open/cache-miss；Windows `curl.exe` 又因 Schannel `SEC_E_NO_CREDENTIALS` 失败。
- 处理：没有绕过认证或写临时下载；使用 Git 的 partial clone/object interface 和已抓取的官方 HTML 文档继续核对。

## CTX-006 — 对本地 partial clone 补取当前 official commit

- 状态：`LOCAL-WRITE`，只写现有 `.research-rlinf/.git/objects`/`FETCH_HEAD`，不更新工作树。
- 下载前检查：

  ```powershell
  Get-PSDrive -Name C | Select-Object Name,Used,Free
  git -C .research-rlinf -c safe.directory=C:/Users/86136/Documents/rl/.research-rlinf count-objects -vH
  ```

- 结果：C: 当时 free 38,223,552,512 bytes；object store 26.58 MiB；repo 是 `blob:none` partial clone。
- 已向用户说明：目标为现有 object store，预计少于 50 MB，不做完整 clone/模型下载。
- exact command：

  ```powershell
  git -C .research-rlinf \
    -c safe.directory=C:/Users/86136/Documents/rl/.research-rlinf \
    fetch --filter=blob:none --no-tags origin \
    89b0cd5fce528180559bbd50d055fb2e21a25bb5
  ```

- 结果：exit 0；`FETCH_HEAD=89b0cd5...`。
- 随后只对目标 paths 执行：

  ```powershell
  git -C .research-rlinf -c safe.directory=... ls-tree -r --name-only 89b0cd5...
  git -C .research-rlinf -c safe.directory=... show 89b0cd5...:<target path>
  git -C .research-rlinf -c safe.directory=... log -1 89b0cd5... -- <target path>
  ```

- 最终 object store 30.74 MiB，较前增加约 4.16 MiB；远低于 50 MB 估计。工作树仍未 checkout/update。
- 已见旧 garbage warning 7 个临时 pack/index、共 9.19 KiB；没有清理，因为不属于本轮授权且不影响审阅。

## CTX-007 — 旧个人 RLT 与新 official 的结构差异

- 状态：`LOCAL-RO` 为主；曾向 partial object store fetch 个人 branch ref/object metadata。
- 个人 branch live command：

  ```powershell
  git ls-remote --heads \
    https://github.com/Yutenji-Nyamu/rlinf_fastwam.git \
    refs/heads/codex/rlt-pi0-robotwin
  ```

- 结果：`2b8199d8ab2e7b110994fd3234bf7007196c3af9`。
- 共同祖先：

  ```powershell
  git -C .research-rlinf -c safe.directory=... merge-base \
    2b8199d8ab2e7b110994fd3234bf7007196c3af9 \
    89b0cd5fce528180559bbd50d055fb2e21a25bb5
  ```

- 结果：`6d0db56bf26f972cd27fa29535f5eb939e80e5bf`。
- 为避免 partial clone 对 rename detection 拉 missing blob，使用：

  ```powershell
  git -C .research-rlinf -c safe.directory=... \
    diff --no-renames --name-status 2b8199d8... 89b0cd5...
  ```

- 结果：all changed 860；task-relevant 253；其中 docs 133、examples 39、`rlinf/` 63、tests 11、toolkits 4、other 3。
- 问题：未加 `--no-renames` 的 diff 会让 partial clone 尝试向 official promisor remote 获取个人仓 blob，报 `not our ref`；一次自定义临时 promisor 尝试也因 GitHub reset/timeout失败。
- 处理：不持久化 remote/config，不继续全量 blob 下载；改用 tree/name-status、云端 branch 和本地 source-locked docs 完成结构判断。
- 结论：禁止整 branch merge；形成 `02_OFFICIAL_SOURCE_AND_PORT_MAP.md` 的复用/重写/废弃矩阵。

## CTX-008 — 服务器交接 zip 只读审阅

- 状态：`LOCAL-RO`。
- commands：

  ```powershell
  Get-Item -LiteralPath C:\Users\86136\Documents\other\shared_server_handoff_20260819.zip
  Get-FileHash -Algorithm SHA256 -LiteralPath <zip>
  tar -tf <zip>
  tar -tvf <zip>
  tar -xOf <zip> <each of five markdown files>
  ```

- 结果：zip 33,353 bytes、outer SHA 与 `INHERITED-SZ-000` 一致；五文件均全文只读，没有解压写盘；职责见材料索引第 4.2 节。

## DOC-001 — 建立深圳专题文档层

- 状态：`LOCAL-WRITE`。
- 目录创建：

  ```powershell
  New-Item -ItemType Directory -Force \
    -Path docs\rlinf-shenzhen-pi0-ppo-rlt\evidence
  ```

- 文件均用 `apply_patch` 创建：

  ```text
  docs/rlinf-shenzhen-pi0-ppo-rlt/00_INDEX_AND_IMPLEMENTATION_PLAN.md
  docs/rlinf-shenzhen-pi0-ppo-rlt/01_REFERENCE_INVENTORY.md
  docs/rlinf-shenzhen-pi0-ppo-rlt/02_OFFICIAL_SOURCE_AND_PORT_MAP.md
  docs/rlinf-shenzhen-pi0-ppo-rlt/evidence/OPERATION_LEDGER.md
  ```

- 内容：任务/机器/授权分离、官方 source lock、路径/环境/Git/network 方案、baseline phases、RLT port matrix、材料索引和本账。
- 同批使用 `apply_patch` 更新根路由：

  ```text
  HANDOFF.md          # 当前专题切到深圳；OGPO 标成 AutoDL 历史停点
  PROJECT_CONTEXT.md  # 多服务器隔离、深圳账号/路径和 network_turbo 机器边界
  ```

- 服务器影响：无。

## CTX-009 — 当前 official installer 静态审阅

- 状态：`LOCAL-RO`；对象为已获取的 `RLinf@89b0cd5...` blobs，未运行 installer。
- exact read-only command family：

  ```powershell
  git -c safe.directory=C:/Users/86136/Documents/rl/.research-rlinf \
    -C .research-rlinf show 89b0cd5...:requirements/sys_deps.sh
  git -c safe.directory=C:/Users/86136/Documents/rl/.research-rlinf \
    -C .research-rlinf grep -n -e install_openpi -e install_robotwin \
    -e rlinf-openpi -e insteadOf -e unset_mirror -e restore_pyproject \
    89b0cd5... -- requirements/install.sh
  git -c safe.directory=C:/Users/86136/Documents/rl/.research-rlinf \
    -C .research-rlinf show 89b0cd5...:requirements/install.sh
  ```

- 结果：
  - normal common-deps 路线调用 `sys_deps.sh`；其要求 `sudo -n true`，并可 apt 安装、写 EGL/Vulkan 配置；
  - official `--no-root` 可跳过该系统脚本，但前提是 live preflight 证明系统依赖已存在；
  - `openpi + robotwin` 安装 `rlinf-openpi==0.1.1`、`pytorch3d@v0.7.9` 和未 pin commit 的 Curobo；
  - installer 对不兼容的既有 `--venv` 会递归删除重建；
  - `--use-mirror` 先写 global Git `insteadOf` cleanup trap，随后 NVIDIA Torch rewrite 覆盖该 trap，因此映射可能残留。
- 处理：计划改为 fresh absent venv；直连优先；mirror 必须用任务专属 `GIT_CONFIG_GLOBAL` 隔离；系统包缺口单独提交管理员审批。详见计划 5.1 和 source map 1.6。
- 服务器影响：无。

## CTX-010 — old/current 关键接缝复核

- 状态：`LOCAL-RO`；未修改任何 research worktree。
- exact checks：对 `89b0cd5...` 与 `2b8199d8...` 使用 `git rev-parse <commit>:<path>`、`git grep -n` 和 `git show <commit>:<path>`。
- 结果：
  - old/current `robotwin_adjust_bottle_ppo_openpi.yaml` 均为 blob `650280ebcd1e65b8be585b23523f060ec0c58574`；YAML 可直接取 current official，runtime 不可回退旧 launcher/runner；
  - current RoboTwin `robotwin_sft_openpi_rlinf.yaml` 顶层 `num_action_chunks=50`；旧 RLT Stage 1 顶层 10、nested `action_horizon=50`，不能照抄；
  - current SFT wrapper 明确 `loss = rlt_loss + rlt_alpha * vla_loss`，官方 RLT example 为 `rlt_alpha=1.0`；旧 config 为 `rlt_train_vla=False`、alpha 0；
  - current transition replay capability 只识别 `MANISKILL_RLT`；route 只有 simulator/real-world 两类，没有旧 RoboTwin full-task route；
  - 首批 port 因而聚焦 route、transition capability/current API、canonical action decode、truncation test；compact replay 与 robust resume后置。
  - official eval guide 记录 `adjust_bottle` 150 success seeds；checked-in default 128 fixed env×1 epoch 仅覆盖约 128 条，formal 分母需另行冻结。
- 问题：一次针对旧 commit 的 glob `git grep` 触发 partial-clone promisor 对缺失 blob 的 `not our ref`，但目标 YAML 行已返回；没有 fetch、config 持久化或工作树修改。
- 服务器影响：无。

## DOC-002 — 文档终检

- 状态：`LOCAL-RO` QA；检查本专题四文件及根 `HANDOFF.md`、`PROJECT_CONTEXT.md`。
- 结果：六文件均 UTF-8 无 BOM、LF、末尾换行、无 trailing whitespace；Markdown fence 数均为偶数。
- credential scan：用户给出的两组密码、常见 password assignment、private-key header 均无命中。
- internal links：本专题入口与三个旧 RLT history targets 均存在；一次人工 QA 清单把 `robotwin` 误写成 `shenzhen` 导致三项 false，随后按文档实际 relative target 复核为 true，没有改文件。
- 服务器影响：无。

## SZ-NEXT 模板 — 下一条深圳操作从这里续写

下一次连接前先追加实际 ID，例如 `SZ-RO-001`，并在执行后立即补结果，不在任务结束时笼统回填。第一批预定只读内容：

```text
identity + time + hostname
mount/path/free-space
GPU/process/RAM
driver/CUDA/Vulkan/EGL
GitHub/PyPI/HF real-file routes
official/personal ls-remote
```

在本账出现明确 `SZ-WRITE` 授权前，不创建目录、clone、install、download、Git auth 或启动 simulator。
