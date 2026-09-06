# 深圳 latest RLinf：source、环境与 π0 模型流水账

## 边界与入口

- 机器/账号：`SZ-H100` / `chenyiteng`；固定 host-key 的低层 Paramiko 密码认证。
- 用户于 2026-08-21 明确要求继续按 latest RLinf 官方仓库/文档跑通 demo；本阶段授权覆盖 live preflight、
  source clone/worktree、独立用户态环境、official 依赖安装、RoboTwin compatibility assets 的独立复用、
  revision-pinned π0 SFT/tokenizer 下载与 load-only 检查。
- fixed-8 simulator eval 与 PPO smoke 启动前仍先展示 resolved config、exact command、output、预算、资源和
  停止条件；不自动扩大到 fixed-128、完整 PPO、pilot/formal、RLT、Git credential/push。
- 凭据不写入本账；所有联网 command file 显式加载深圳 Mihomo profile，且不修改全局 Git/proxy。

## 操作记录

> 从本轮第一条 live identity/source/network/storage probe 起按实际顺序追加。

### RL-SZ-R0-001 — 本地 SSH 执行器入口纠正（未连接服务器）

- 时间：2026-08-21 21:29 CST。
- 首次本地命令使用裸 `python`，命中
  `C:\Users\86136\AppData\Local\Microsoft\WindowsApps\python.exe` 占位符并被系统拒绝启动，exit 1。
- 该失败发生在 Python 进程启动前，没有 SSH 握手、认证或远程命令。
- 窄修复：使用当前 Codex workspace dependencies 返回的真实 Python
  `C:\Users\86136\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`；不安装本地
  Python，不改变 Paramiko/host-key/密码路线。

### RL-SZ-R0-002 — 首次 live preflight 的只读脚本语法失败

- 时间：2026-08-21 21:29 CST；账号/远端目录：`chenyiteng` / `/home/chenyiteng`。
- command file：`local_scripts/remote_commands/shenzhen_rlinf_r0_live_preflight_20260821.sh`，首次 SHA256
  `eecb0c6bc00cf04b26a1859a2ca378da117de78c384a270c04db15f82cc92118`。
- 固定 host-key 与密码认证成功；身份仍为 UID 1003，组 `chenyiteng sudo labdata`。只读 `df` 显示
  `/` 237 GiB、`/home` 约 2.3 TiB、`/data` 约 3.2 TiB 可用，inode 充足。
- 脚本随后 exit 1；原因是把三个 target 作为 `findmnt` 的多个位置参数传入，在 strict shell 下返回非零。
  尚未运行 GPU、target path 或联网 probes，也没有服务器写入。
- 修复：对 `/`、`/home`、`/data` 分别执行 `findmnt --target`；因为本阶段全只读，可安全从头复测。

### RL-SZ-R0-003 — live preflight 主体通过，GitHub REST 末端限流

- 时间：2026-08-21 21:30 CST；修正后 command file SHA256
  `0488d204e9f005d62264ef52fdeac141bdc3668d4a744a2c9d076e381c3414a7`。
- 身份/存储：UID 1003；`/` 237 GiB、`/home` 约 2.3 TiB、`/data` 约 3.2 TiB available；约 2.0 TiB
  RAM available，swap 0 used。
- GPU/进程：8×H100 80GB 均 0 MiB/0%，无 compute process；除本条 shell/grep 外未发现本账号
  RLinf/RoboTwin/Ray/torchrun/python 任务。
- 工具：Git/Git-LFS/CMake/GCC/G++/Python 3.10 存在；普通 shell 未发现 `uv`、`ninja`、`nvcc`。
- 四个 RLinf 目标根均 absent；standalone source 仍为 RoboTwin `30954692...`、installed XPolicyLab
  `c07a096...`，三项 assets 约 11 GiB/901 MiB/4.4 GiB。
- 显式 Mihomo 后三个 official `git ls-remote` 均成功：RLinf main
  `7d07a4212ee6858cc333e1d4fab7a37256d1f839`、RoboTwin `RLinf_support`
  `0008ae6800df9f75fc8de7098bacb01735fd8fd2`、RoboTwin main `30954692...`；与计划锁一致。
- 末端匿名 GitHub REST API 返回 HTTP 403，strict curl 使脚本 exit 1，后续 HF probes 未执行。这不表示
  Git clone 通道失败；`ls-remote` 已闭合 GitHub source 路线。窄处理为改用 pinned raw file + 两个 HF
  revision API，只补未执行的联网检查。

### RL-SZ-R0-004 — pinned raw/HF revision 补充探针

- command file：`local_scripts/remote_commands/shenzhen_rlinf_r0_network_hf_probe_20260821.sh`；SHA256
  `0f9a9c9450f39a72ca3ba88f22233c2238a04dc2f23e5272fc432e1a68063ed8`；exit 0。
- 显式 Mihomo：RLinf pinned `README.md` raw HTTP 200；π0 SFT revision `92684e50...` HTTP 200；
  tokenizer revision `befaa248...` HTTP 200。source/model 的计划 locks 均在线可取。

### RL-SZ-R0-005 — 代理配额安全刷新

- 复用 ACT 阶段已校验的 root-only安全 probe 与 sudo command：
  `shenzhen_mihomo_quota_sanitized_20260821.sh`，SHA256
  `9a0053c17275b32b1d1ce570cc883328a9c3bdb7a1e665214145d34bfadf71c2`；exit 0。
- 输出只含 numeric quota/status，不含 provider URL/token：HTTP 200；total 100 GiB、used 45.284 GiB、
  remaining 54.716 GiB，expiry `2026-08-23 14:01:48 Asia/Shanghai`。足以覆盖 7.514-GiB π0 snapshot
  与环境依赖；仍通过资产本地复用避免重复约14.9-GB下载。

### RL-SZ-R0-006 — official RLinf canonical/worktree 与 RoboTwin compatibility clone

- 时间：2026-08-21 21:33 CST；command file
  `local_scripts/remote_commands/shenzhen_rlinf_r0_clone_sources_20260821.sh`；SHA256
  `aa51dda5c562d76db25655c6178ca085a84159b0f570788e563106588f365acf`；exit 0。
- preflight 已确认整个 `/data/chenyiteng/projects/rlinf-shenzhen` absent；动作按 official public Git
  路线执行，没有 GitHub 登录、credential、global Git/proxy 修改。
- canonical：`/data/chenyiteng/projects/rlinf-shenzhen/RLinf`，detached clean
  `7d07a4212ee6858cc333e1d4fab7a37256d1f839`，约36 MiB。
- PPO worktree：`.../worktrees/ppo-pi0-robotwin`，branch `codex/sz-ppo-pi0-robotwin`，clean same HEAD，
  约22 MiB。
- compatibility tree：`.../RoboTwin-RLinf-support`，按 official `-b RLinf_support` clone 后 detached clean
  `0008ae6800df9f75fc8de7098bacb01735fd8fd2`，约17 MiB。
- `/data` 仍约3.2 TiB available。尚未复用 assets、创建 venv 或安装依赖。

### RL-SZ-R0-007 — 当前 source 内官方安装、PPO 与评测合同核对

- 时间：2026-08-21 21:36 CST；command file
  `local_scripts/remote_commands/shenzhen_rlinf_r0_source_contract_probe_20260821.sh`；SHA256
  `f3b82ac0f8db12349508cab6e7466a32e3f944f27364b39141a594d00254f161`；exit 0。
- 探针直接读取服务器上 clean `7d07a421...` worktree，没有执行安装或修改 source。关键文件 SHA256：
  `requirements/install.sh` = `9b856feb33ef237f35983449a28eb17690c057cea3d6c539694222d256741bc0`；
  RoboTwin train doc = `26e64ff52c4470e52f0100f6ae6ec57dbf5314f579874ac9607653e465572e5e`；
  eval doc = `61e7f53c9d026c1b5fa1c8d4895dafe64d35d61477cde276bb6a81a797a4cde1`；
  official PPO YAML = `7ffd734f1e57cbd830fbafea392b58d0e150d88859967bf4f26bf0740acdc335`；
  official eval YAML = `b1b2293f9c038069ea28e59e0958941428c04e19346d75e9040532c39b06e9d4`。
- 官方安装入口仍是
  `bash requirements/install.sh embodied --model openpi --env robotwin`；installer 支持 absolute `--venv`、
  `--no-root`，默认 Python 3.11.14，并安装 `rlinf-openpi==0.1.1`、OpenPI model requirements、
  RoboTwin environment requirements 与 RLinf 本体。为了不占 root，深圳运行将只增加 absolute user venv 和
  `--no-root`，其余保持官方路径。
- 官方 train doc 仍要求 `RoboTwin -b RLinf_support`、`ROBOT_PLATFORM=ALOHA`，PPO suffix 为
  `robotwin_adjust_bottle_ppo_openpi`；eval 仍要求 `ROBOTWIN_PATH`、`ROBOT_PLATFORM=ALOHA` 并由
  `evaluations/run_eval.sh` 启动。
- official eval 配置默认 128 env、fixed reset IDs、200 steps、video；official PPO 配置默认 train
  256 env × 4 rollout epochs、200 steps、2 update epochs、global batch 2048，π0 action horizon/chunks 50、
  action dimension 14，actor/env/rollout placement 均覆盖 GPU 0–7。这些是 formal defaults；下一阶段的
  fixed-8/one-optimizer-step smoke 只做明确预算缩减，不改算法合同。
- live official refs 与既定锁一致：RLinf `main=7d07a421...`、RoboTwin
  `RLinf_support=0008ae68...`、π0 SFT `main=92684e50...`、tokenizer `main=befaa248...`。因此不因“latest”
  再换 commit；今日 latest 就是当前 source lock。
- 发现 installer 尾部的 OpenPI asset helper 会用浮动 `main` 下载 tokenizer，且没有 `--revision`；若
  expected tokenizer 已存在则 skip。为兑现 revision lock，先精确预置 `befaa248...` 并校验，再运行
  official installer；这是下载锁定，不是修改官方安装代码。

### RL-SZ-R0-008 — tokenizer helper 路径事实与空缓存确认

- command file `shenzhen_rlinf_r0_tokenizer_helper_probe_20260821.sh`，SHA256
  `2fedbfcc73cd4a4f6aeee7a194a0fcebb53a2093c45164c30d5ad1bdcdc92d2e`；exit 0；只读。
- locked source 的 helper 实际在 `$HOME/.cache/openpi/` 检查根层
  `paligemma_tokenizer.model`，而 `hf download`/OpenPI 的真实相对路径是
  `big_vision/paligemma_tokenizer.model`；且下载命令未带 revision。这是 upstream helper 的路径与锁定缺口。
- 现场 `/home/chenyiteng/.cache/openpi` 整体 absent。后续只下载 immutable revision 的真实 nested file，
  并用一个指向同一字节的根层相对 symlink 满足 official helper 的 skip check；不改 RLinf source。

### RL-SZ-R1-001 — `--no-root` 前的系统依赖现场

- 两次 system probe 分别记录 official sys-deps source 与 live package/command 状态。最终 command file SHA256：
  `shenzhen_rlinf_r1_system_dep_preflight_20260821.sh` =
  `fb20ef9ad27a4dcf08307d5996e9d38ea50a17eb7b285f233b18c9e43abf8310`；
  `shenzhen_rlinf_r1_sys_deps_source_probe_20260821.sh` =
  `5882ffd5ca76fdd5e1ef8e0a3bac0bd8b060bce66bf3ffc82d4010f217134082`；
  `shenzhen_rlinf_r1_missing_sys_dep_probe_20260821.sh` =
  `091fbe36c99719274345b9969e1002e0ae9cbd4367ed840a55fec1349eae009a`。
- CUDA/Vulkan/build essentials 已满足：`/usr/local/cuda -> cuda-12.9`，nvcc 12.9.86，cuSPARSE/cuBLAS/
  cuSOLVER headers 与 NVIDIA Vulkan ICD 存在，GCC/G++/CMake/Git-LFS/FFmpeg 可用。
- official generic apt 清单中仍有 `patchelf`、OSMesa/若干 GUI/FFmpeg dev packages 未装；ACT 已在同机完成
  render/collect/train/eval。按用户要求不预装整套泛化系统包：先走 official user-venv installer，只有出现
  精确缺依赖错误时再做窄修复。全部下载/cache/build 放 `/home` 或 `/data`，不把大环境放 root。

### RL-SZ-R1-002 — pinned tokenizer 首次客户端选择失败

- 第一版 `shenzhen_rlinf_r1_preseed_tokenizer_20260821.sh` SHA256
  `628b58e045d264a2d525955e63b0ae0c382c254a05c493c6722b2aa7105a5741`，使用 ACT env Python；
  在联网前因 `ModuleNotFoundError: huggingface_hub` exit 1。
- 没有 payload 下载或 final 发布，只新建空目录
  `/home/chenyiteng/.cache/.partial-openpi-tokenizer-befaa248-20260821`；保留为失败证据，不覆盖它。
- 补充只读 probe 找到现有 RoboTwin env 内 `huggingface_hub==0.25.0`，因此 retry 使用该已存在客户端，
  不向 ACT env 增装无关包。

### RL-SZ-R1-003 — pinned tokenizer 成功发布

- retry command file SHA256
  `5f18c35512d5a95b61d41263fd8ccc215bae2d8ee9fd67adc85703fae333ea48`；exit 0。
- 从 `RLinf/openpi_tokenizer@befaa248e4f82954b625a421658f933dfd1a97a0` 仅下载
  `big_vision/paligemma_tokenizer.model` 到新 partial；实测 4,264,023 B，SHA256
  `8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`，与 pinned LFS oid 一致。
- 校验后原子发布到 `/home/chenyiteng/.cache/openpi/`；根层 relative symlink 指向上述 nested file，
  official installer 会 skip 浮动 main 下载。总逻辑量 4.2 MiB，位于 2.3-TiB `/home`。

### RL-SZ-R1-004 — latest RLinf official OpenPI + RoboTwin 独立环境安装

- 时间：2026-08-21 21:49–22:14 CST；command file
  `shenzhen_rlinf_r1_install_official_20260821.sh`，SHA256
  `544186ff0dfc00e529be769a07a330f52751fccc94fdbaef21685788a7b8f03a`；exit 0；完整日志为服务器
  `/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821/install.log`。
- exact official command：
  `bash requirements/install.sh embodied --model openpi --env robotwin --venv /home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin --no-root`。
  仅额外把现有 Miniforge/`~/.local/bin` 加入 `PATH`，并将 build temp 放在 2.3-TiB `/home`。
- installer 根据 driver 自动选择 CPython 3.11.14、torch `2.11.0+cu129`、torchvision
  `0.26.0+cu129`，GPU probe 为 H100；核心还有 Ray 2.57.0、JAX/JAXlib 0.5.3、
  `rlinf-openpi==0.1.1`、`rlinf-transformer-openpi==4.53.2`、tokenizers 0.21.4、
  SAPIEN 3.0.1、MPLib 0.2.1、PyTorch3D 0.7.9、Warp 1.11.1。
- PyTorch3D official tag 实际解析到 commit `33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba` 并成功编译；
  CuRobo 的未 pin `HEAD` 实际解析为 `8e734f3ced1df898990bcd92de40abce475907db` 并成功构建。二者会在最终
  `direct_url.json` manifest 再固化。
- flash-attn 首候选 Dao release URL 对 2.11 wheel 返回 HTTP 404；这是 installer 内置 fallback 路径，
  随后从 RLinf official release 成功安装 `flash-attn 2.8.3+cu12torch2.11cxx11abiTRUE`，未源码编译。
- pinned tokenizer 的 root-level relative symlink 被 official helper 识别，明确输出 skip；没有再次请求浮动
  tokenizer main。未遇到 `patchelf`/OSMesa 等缺项导致的实际错误，因此未批量改系统包。
- 实测空间：venv 16 GiB、uv cache 2.0 GiB、tokenizer 4.2 MiB、install temp 72 KiB；`/` 仍约237 GiB
  available，`/home` 约2.2 TiB、`/data` 约3.2 TiB available。大环境与 cache 均未写入 root。
- worktree 仍 clean branch `codex/sz-ppo-pi0-robotwin`；installer 生成的 `uv.lock` 被 repo ignore，后续
  final manifest 将另存 hash/freeze。

### RL-SZ-R1-005 — compatibility assets 首选 reflink 不受文件系统支持

- command file `shenzhen_rlinf_r1_reuse_compat_assets_20260821.sh`，SHA256
  `d1df601a91d78db41586fc2138fac2355269fdb4f93c214068a38d47a7cde6de`；exit 非零。
- 只从 standalone 的 `assets/{background_texture,embodiments,objects}` 复制；`/data` 对
  `cp --reflink=always` 返回 `Operation not supported`。命令在 staging 阶段失败，尚未向 compatibility
  `assets/` 发布目录，也未运行 updater；standalone Git status fingerprint 未变。
- 失败 staging 约292 KiB，只含 reflink 失败留下的目录/空壳；先保留到普通 copy 成功，避免覆盖失败现场。
  完整输出在服务器 `.../bootstrap-7d07-20260821/assets_reuse.log`。

### RL-SZ-R1-006 — assets 普通独立 copy、compatibility 路径更新成功

- retry command file `shenzhen_rlinf_r1_reuse_compat_assets_copy_retry1_20260821.sh`，SHA256
  `a7baba1bcc9562f121a480378780c056249563e1a9a79dd981556664f1383af4`；exit 0。
- 使用 `cp -a --reflink=never` 将三类资产复制到新 staging，按相对路径/type/size/symlink target 与源树
  比较一致后，同文件系统 `mv` 发布；没有复制 standalone 的 `envs/`、RoboTwin/XPolicyLab code、数据或
  H100 planner patch，网络量 0。
- 仅在 compatibility cwd 运行 official `script/update_embodiment_config_path.py`；动态找到并成功生成6个
  embodiment YAML（含 ALOHA left/right），全部指向
  `/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support/assets`，无 standalone 路径泄漏。
- standalone Git status SHA256 前后均为
  `ba695dfb57e51b109747fb13efcbf816f5ef957cd7507a96c962ea4129c216d1`，证明原 ACT tree 未污染。
  compatibility source HEAD 仍 `0008ae68...`；资产目录被该仓库 ignore，代码状态不变。
- `/data` used 从 110,683,766,784 B 增至127,343,587,328 B，实际新增16,659,820,544 B（约15.52 GiB）；
  available 仍约3.2 TiB。成功后精确删除本轮产生的292-KiB失败 staging，既有数据未删除。

### RL-SZ-R1-007 — official π0 RoboTwin SFT 完整 snapshot 下载与发布

- 时间：2026-08-21 22:19:51–22:38:33 CST；command file
  `shenzhen_rlinf_r1_download_pi0_sft_20260821.sh`，SHA256
  `634de12edf880f600d7eb62a4a6e23cb35fa6694ff3548db40ba7926e3e35758`；exit 0；服务器日志/manifest：
  `.../bootstrap-7d07-20260821/model_download.log` 与 `model_manifest.json`。
- official `hf download` 使用完整 immutable revision
  `RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50dca1a5f75adc8d332046c4cf4fa7a3d0`、
  single worker；先落同 `/data` partial，全部 gate 通过后原子发布到
  `/data/chenyiteng/models/rlinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle@92684e50`。
- 精确闭合18个 remote files、payload 8,067,741,880 B；两片分别为
  4,284,226,576 B / SHA256 `a6b42e854e78dd59311d7d5121682b5af993778b1868f3210952494e7bab6ab1`，
  3,783,369,156 B / SHA256 `3aea4cd15ae930c25d5ad87912f90bd61cc3f61748fee9e6e75fb01610209446`。
  RoboTwin norm stats 为5,149 B、JSON schema通过，index 只引用上述两片。
- 下载 wall 1,122 秒，端到端平均约7.19 MB/s（约57.5 Mbit/s）；final `du -sh` 7.6 GiB。
  `/data` 仍约3.2 TiB available。尚未启动 model loader、simulator、Ray 或 GPU workload。

### RL-SZ-R1-008 — official final environment 的 metadata conflicts

- 首次 runtime command file SHA256
  `3618125509a89f8d23e2f9b5aeaeba91235338817dc66a3301fd5b15c9d6efe3`；在 `uv pip check` exit 1
  后按 strict shell 停止，未进入 import/compose/model load。
- official installer 的最终375 packages 有6项 metadata incompatibilities：`rlinf-openpi` 仍声明
  torch 2.7.1，LeRobot 声明 torch<2.8/旧 torchvision/torchcodec，而 installer 主动安装 torch
  2.11+cu129；transformers metadata 要 tokenizers>=0.22，而 installer 尾部主动强制 tokenizers
  0.21.4；另有 tensorflow-addons/typeguard 冲突。
- retry command SHA256 `e8896ab0db1b89fb74c9c2cae334afaa1fc83e4ebe19be8e107f4e5fa434e188`
  保留 `pip check` 非零作为证据，但继续高信息量 runtime tests；不擅自降级 official torch/tokenizer。

### RL-SZ-R1-009 — unpinned CuRobo main 已与 RoboTwin compatibility API 断裂

- retry runtime 在 import gate 停于 `ModuleNotFoundError: curobo.types.base`。targeted command
  `shenzhen_rlinf_r1_curobo_contract_probe_20260821.sh`，SHA256
  `99441fbb7e312e76f6fd6409faf0435d852533e7ad43e062f871a0390696d9a9`，闭合如下：
  installer 实际从 unpinned main 安装 `8e734f3ced1df898990bcd92de40abce475907db`；该 upstream tree 已把旧
  `curobo.types.*` 移到新 `_src` API；compatibility `envs/robot/planner.py` 明确仍导入
  `curobo.types.math.Pose` 与 `curobo.types.robot.JointState`。`robotwin.envs.vector_env` 本身可 import。
- source/tag probe SHA256 `70091485de6e303920395852e95a972a35fe7abb8d66815e37b8a4e841d8fa7b`：
  RLinf installer 与 RoboTwin installer 均安装 CuRobo floating main；最后一个 legacy release 是
  `v0.7.8=d64c4b005459db10c5dd867d8b30a87d5bda9bdb`，下一 tag 已是 breaking `v0.8.0`。
- 决策：只把该未 pin 外部依赖锁到 v0.7.8；保持 latest RLinf、RoboTwin compatibility code、
  torch2.11/cu129、PPO配置与模型不变。旧 ACT 的 cu121/LBFGS patch 不搬入本环境。

### RL-SZ-R1-010 — 首次 CuRobo v0.7.8 repair 的 resolver drift 被停止

- 首次 repair command SHA256 `52ea239e38c4aa799255385433ae60f4aacab08b89aaebd7d0cae37deb701b08`
  未带 `--no-deps`；uv resolver 随即准备 torch/CUDA13 与大量 NVIDIA 包，超出“只替换CuRobo”的边界。
- 在尚处 wheel build、installed CuRobo仍为8e734、torch仍2.11+cu129时，精确识别 owned PGID 1267712；
  stop command SHA256 `0e38b3f6defad8ffcccc35d89d54f26a417b283aea5b8329dba2ddb24b68ca37`
  只 TERM 该 process group，exit marker `R1_CUROBO_RESOLVER_DRIFT_STOPPED`。没有 package uninstall/install；
  已下载的候选仅留在2.3-TiB `/home` uv cache。

### RL-SZ-R1-011 — CuRobo v0.7.8 `--no-deps` 精确重编成功

- retry command SHA256 `36232c0cd30b71a44ee8114e04fa74bde36272d7c35fead680cc1cc4fa35a4e2`；
  preflight 断言 torch2.11+cu129 与 current CuRobo8e734；uv 输出 `Resolved 1 package`，只构建 pinned
  `d64c4b...`，并在 postflight 再断言 torch不变/旧API可导入。
- 本地 SSH stream 中断后只读现场确认远端 owned process 已由 PID1 接管但仍在同一 timeout group正常
  运行；没有重复启动。最终该唯一 build 自然完成并替换为 `nvidia-curobo==0.7.8`，`direct_url.json`
  commit 为 `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`。
- postflight 确认 torch仍为 `2.11.0+cu129`，旧 `curobo.types.base/math/robot` 与 compatibility planner
  所需 API 均可导入；没有搬 standalone 的 H100/cu121 planner patch。

### RL-SZ-R1-012 — 补齐 CuRobo 声明的 `scikit-image` 依赖

- command file `shenzhen_rlinf_r1_install_curobo_declared_dependency_20260821.sh`，SHA256
  `8aef5e8c3f5eed641d624c6ded223f883b43aaddbb9ea34835024d60628b4391`；exit 0。
- 按正常 resolver 安装 `scikit-image==0.26.0`，实际新增/更新仅 `lazy-loader==0.5`、
  `scikit-image==0.26.0`、`tifffile==2026.3.3`；torch/CUDA/CuRobo 均保持不变。
- `uv pip check` 回到 official installer 原有的同 6 项 metadata range conflicts，没有新增 CuRobo 缺依赖。

### RL-SZ-R1-013 — final runtime、official compose 与 π0 真实加载闭环

- command file `shenzhen_rlinf_r1_runtime_compose_model_load_20260821.sh`，SHA256
  `33cd6290c7f864cec8a76edf6bf98b29ea9d2c8f4337a3d3f53a92dd9ac7102e`；最终 retry exit 0；完整日志：
  `/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821/runtime_compose_model_load_retry2.log`。
- H100 sm90 的 torch CUDA op、RLinf/Ray/OpenPI/JAX/Orbax、SAPIEN/MPLib/Warp、PyTorch3D `_C`、
  CuRobo legacy API 与 `robotwin.envs.vector_env` 全部 import/load 通过。
- official `robotwin_adjust_bottle_openpi_eval` compose 解析到 pinned model/assets、
  `pi0_aloha_robotwin`、3 images、14D 与 action chunk 50。
- OpenPI loader 真实读取两片 safetensors 与 `physical-intelligence/robotwin/norm_stats.json`；得到
  `OpenPi0ForRLActionPrediction`、3,502,061,329 parameters、CPU load、action horizon/chunk 50、env dim 14。
- venv约16 GiB；失败 resolver 留下的 uv cache 使 `~/.cache/uv` 约4.4 GiB，全部在2.3-TiB `/home`；
  本轮不未经授权清 cache。三棵 source 的预期状态未被 runtime probe 改写。

### RL-SZ-R1-014 — 下载后代理与存储复核

- 复用只输出 numeric quota/status 的安全 sudo probe；HTTP 200。100 GiB total、52.962 GiB used、
  47.038 GiB remaining，expiry `2026-08-23 14:01:48 Asia/Shanghai`。
- 相比 R0 的45.284 GiB，本轮环境/模型阶段代理计量增加约7.678 GiB；π0 snapshot实测有效 payload
  8.068 GB，Git/PyPI/Conda部分请求按Mihomo规则走DIRECT。正常 R2/R3 不再需要大模型下载。
- `/` 仍约237 GiB、`/home`约2.2 TiB、`/data`约3.2 TiB available；大文件均位于 `/home` 或 `/data`。

### RL-SZ-R2-001 — fixed-8 / one-update / reload dry-resolve

- command file `shenzhen_rlinf_r2_compose_fixed8_oneopt_packet_20260821.sh`，SHA256
  `7016f1bb4e41c2d0493121eff820627594d151b0651c7388903f4e3df03505db`；exit 0，marker
  `R2_FIXED8_ONEOPT_PACKET_COMPOSE_OK`。只执行 Hydra `--cfg job --resolve` 与 seed/budget 计算；未启动
  Ray、simulator、GPU workload 或训练。
- evidence root：
  `/data/chenyiteng/results/rlinf-shenzhen/bootstrap-7d07-20260821/resolved-packet-v1/`：
  `sft-fixed8.resolved.yaml=ff9ba684...79834c5`、`ppo-oneopt.resolved.yaml=4453a536...d79ca4`、
  `ppo-reload-fixed8.resolved.yaml=420396d7...ddeae80`、`budget-and-seeds.json=4eb7923b...06787`。
- official seed partition 的 fixed-8 精确为
  `[100100052,100100138,100100025,100100177,100100084,100100172,100100082,100100089]`；pre/reload相同。
- PPO train records=`8×1×(50/50)=8`，global batch=8、update epoch=1，因此精确1次 distributed
  optimizer step。checkpoint 解析为
  `ppo-oneopt-v1/robotwin_ppo_openpi/checkpoints/global_step_1/actor/{dcp_checkpoint,model_state_dict/full_weights.pt}`。
- 三个计划 run root在 compose 时均 absent。完整启动包写入
  `docs/rlinf-shenzhen-pi0-ppo-rlt/09_RLINF_PI0_PPO_EXECUTION_PACKET.md`。

### RL-SZ-R2-002 — 三份真实 launch script 语法校验（未运行）

- SFT fixed-8 script SHA256 `04980104426360cd9f4ad0199e8b603b60ba7263ff94fd44c7dc3270b3e0a7c0`；
  PPO one-update `c1dca941571486d7ffb3c57b1c463b6690449e7e690284fd6601de04d9c4be61`；fresh reload
  `a03394d9d111c3b0e5a6d4bcfa7dd04353b8c2f9d7afc8cc92cd23245b1ec330`。
- 三次均用固定 host-key/password Paramiko，把对应本地 command file流入服务器 `bash -n`；三次 exit 0。
  这只验证 locked server Bash 语法；没有执行脚本正文、没有创建 run root、没有占 GPU。
- Windows WSL `bash -n` 因 sandbox `E_ACCESSDENIED` 未能启动；它不是脚本或服务器故障，随后使用目标服务器
  Bash 完成了更贴近运行现场的语法验证。
