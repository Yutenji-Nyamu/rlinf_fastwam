# 深圳 Fast-WAM official standalone 逐操作流水账（2026-08-22 起）

对应计划：
[`../09_SHENZHEN_OFFICIAL_STANDALONE_PLAN.md`](../09_SHENZHEN_OFFICIAL_STANDALONE_PLAN.md)  
机器：`SZ-H100`  
日常账号：`chenyiteng`  
范围：Fast-WAM official current-HEAD RoboTwin standalone；不含 latest RLinf PPO/RLT 操作。

> 本文件从首条实际操作开始逐项追加。没有执行的命令必须标成 `PLANNED`，不得预填成功结果；动态状态
> 必须由当次服务器输出更新。凭据不写入文档，但这不是另做一份“脱敏版”流水账。

当前完成态（2026-08-22）：source/env/vendor inputs、official HF checkpoint/stats、ModelScope
T5/VAE/tokenizer均完成。`FW-SZ-400`的render通过、expert在MPLib 0.2.1 × NumPy 2.2.6的`Box(...)`
处segfault，因此保留为`PARTIAL / FAIL`；NumPy窄降到1.26.4后没有单独重跑expert。`FW-SZ-500`随后
直接运行official evaluator，run=`fw-sz-500-20260822_045105` exit 0，`adjust_bottle`单集1/1 success。

## 0. 记录合同

每项服务器操作都记录：

```text
ID / CST timestamp / actor / host
purpose / authority / cwd
complete command-file path + local SHA256，或完整 inline command
preconditions / targets
stdout+stderr evidence location / exit code
key result / artifact path + size + hash
problem -> one cause -> one narrow fix -> retest
next safe breakpoint
```

完整命令优先保存为 Windows 工作区中的 UTF-8 command file，账本记录可点击路径和 SHA256；这样可复现
完整命令而不在账本重复几百行。若命令不是 command file，而是在会话中 inline 执行，必须在对应条目中
逐字粘贴。远端密码只注入当前 Paramiko 进程，不属于可复现命令内容。

### 0.1 状态词

| 状态 | 含义 |
|---|---|
| `PLANNED` | 尚未发往服务器 |
| `RUNNING` | 当前 owned command 尚未退出 |
| `PASS` | exit 与所有列出的验收同时通过 |
| `FAIL` | 命令已退出且验收不通过；保留原始证据 |
| `PARTIAL / FAIL` | 阶段内有子项通过，但阶段整体验收失败 |
| `STOPPED` | 经授权停止；不等于代码/环境失败 |
| `SKIPPED` | 有明确 source/live 依据后无需执行 |

## 1. 冻结合同

| 项目 | 冻结值 |
|---|---|
| Fast-WAM official | `7faa71108368fbb3b6885649f112af607427a2d4` |
| official repo | `https://github.com/yuantianyuan01/FastWAM.git` |
| old AutoDL oracle | `45d8e1458921d83f8ad6cf9ce993d371208dabd0`；只作历史问题线索 |
| vendored RoboTwin | `bf44be51cf5717a5595ce59447f2cf5263d2aa95` |
| vendor task_config tree | `fdb995fb05a65f4ee6bdfaf633a89631af93db33` |
| 禁止混入的 RLinf task_config tree | `f114afc...` |
| CuRobo | `v0.7.8@d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |
| HF release revision | `8eaceeb24c3cc92ff2a9c9a9d266a4941b836705` |
| target task | `adjust_bottle / demo_clean / unseen / 1 episode` |
| release semantics | `sigma_shift=5.0 / replan_steps=24` |
| target GPU | physical GPU 3；latest RLinf PPO 使用 physical 4–7 |

### 1.1 固定路径

| 角色 | 路径 |
|---|---|
| source | `/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711` |
| env | `/home/chenyiteng/venvs/fastwam-7faa-py310-cu128` |
| cache/tmp | `/home/chenyiteng/cache/fastwam-7faa` |
| models | `/data/chenyiteng/models/fastwam` |
| results | `/data/chenyiteng/results/fastwam-standalone` |
| read-only asset source | `/data/chenyiteng/projects/robotwin-native/RoboTwin/assets` |
| forbidden compatibility input | `/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support` |

## 2. 同轮前置现场锚点

下表来自本轮 RLinf smoke 后的独立 live preflight；Fast-WAM 每个写入/下载/run 条目仍需引用其最近一次刷新，
不能把本表永久当成当前值。

| 时间 | 事实 | 值 | Fast-WAM 用途 |
|---|---|---|---|
| 2026-08-22 | `/` free | 约 237 GiB | 禁止把大文件放根盘 |
| 2026-08-22 | `/home` free | 约 2.2 TiB | env/cache/build/tmp |
| 2026-08-22 | `/data` free | 约 3.2 TiB | source/assets/models/results |
| 2026-08-22 | GPU | 8×H100 均 idle | target GPU 0 可用，但启动前重查 |
| 2026-08-22 | Ray/project process | 无 | PPO smoke 已退出；启动前重查 |
| 2026-08-22 | proxy | Mihomo `127.0.0.1:7890` active | command shell 显式 source profile |
| 2026-08-22 | quota | 100 GiB total；52.98 GiB used；47.02 GiB remaining | 未观察到充值扩容 |
| 2026-08-22 | expiry | `2026-08-23 14:01:48 CST` | 大下载前再次刷新 |

原始输出的位置与采集命令须由实际执行主账引用：`PENDING`。

## 3. 操作索引

| ID | 阶段 | 状态 | command / evidence | 结果摘要 |
|---|---|---|---|---|
| `FW-SZ-000` | official source/material audit | `PASS` | 本地/官方只读审计；无服务器写入 | current lock、release hash、关键旧→新差异已冻结 |
| `FW-SZ-001` | live preflight + clone | `PASS` | `shenzhen_fastwam_r0_preflight_clone_20260822.sh` | official source 已完成 exact clone@`7faa711...` |
| `FW-SZ-002` | current source contract | `PASS` | `shenzhen_fastwam_r0_source_contract_20260822.sh` | exact source/vendor/config/assets/CUDA 合同已现场读取 |
| `FW-SZ-100` | Python/Torch/Fast-WAM env | `STOPPED` | `shenzhen_fastwam_r1_install_official_env_20260822.sh` | SIGINT让路正式PPO；rc1；仅Python/基础工具 |
| `FW-SZ-101` | resume base env | `PASS` | continuation + conflict probe + narrow cleanup + final acceptance | official Torch/Fast-WAM/CUDA/editable/pip-check 已闭合；physical GPU 3 验收后释放 |
| `FW-SZ-110` | simulation deps + CuRobo | `PASS` | install + MPLib probe + SCM repair + after-inputs final acceptance | CuRobo `0.7.8@d64c4b...` 五个 sm90 `.so`、Warp/H100 与 vendor eager imports通过；仅保留已解释的 MPLib metadata mismatch |
| `FW-SZ-200` | vendor assets/task_config/policy | `PASS` | `shenzhen_fastwam_fw_sz_200_vendor_inputs_draft_20260822.sh` | 20,597 files普通独立复制；vendor task tree、6份updater YAML和policy link均验收 |
| `FW-SZ-300` | HF release download/hash | `PASS` | `shenzhen_fastwam_fw_sz_300_download_release_20260822.sh` | locked revision两文件bytes/SHA256通过；exit 0 |
| `FW-SZ-310` | ModelScope T5/VAE/tokenizer | `PASS` | `shenzhen_fastwam_fw_sz_310_download_modelscope_20260822.sh` | payload 12,792,700,665 bytes；无5B DiT；exit 0 |
| `FW-SZ-400` | imports/render/expert smoke | `PARTIAL / FAIL` | render/expert command + faulthandler repro | official render PASS；expert 在 MPLib `Box(...)` segfault |
| `FW-SZ-500` | official single evaluator | `PASS` | `shenzhen_fastwam_fw_sz_500_single_eval_20260822.sh` | GPU3；adjust_bottle单集1/1 success；exit 0 |
| `FW-SZ-600` | artifact/resource closeout | `SKIPPED` | 不另建服务器命令 | S5已落齐轻量证据并复制一个小视频；大物全留服务器 |

## 4. 已完成：official source/material audit

### FW-SZ-000 — current upstream 与 release 合同

- 状态：`PASS`（只读 source/material audit；没有服务器操作）。
- 时间：2026-08-22。
- Fast-WAM current HEAD：`7faa71108368fbb3b6885649f112af607427a2d4`。
- 旧 AutoDL source：`45d8e1458921d83f8ad6cf9ce993d371208dabd0`。
- compare：ahead 8 commits；37 files；`+2881/-1156`；RoboTwin manager/vendor files不在 compare diff。
- current compatibility boundary：README 保持旧 checkpoint 兼容，但 action shift 默认已从 5 变为 1；
  official old release evaluator必须显式 `EVALUATION.sigma_shift=5.0`。
- current dependency boundary：numpy 2.2.6 与 vendor full requirements 的 scipy 1.10.1 冲突；禁止整装
  vendor requirements，runtime 补包使用 SciPy 1.15.3。
- 直接结果：冻结第 1 节合同与第 5 节模型 manifest；部署仍未由本条执行。

## 5. official model manifest

| 组件 | bytes | SHA256 | 实际服务器验证 |
|---|---:|---|---|
| HF checkpoint | 12,041,813,092 | `776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63` | `PASS FW-SZ-300` |
| HF stats | 88,715 | `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095` | `PASS FW-SZ-300` |
| ModelScope T5 | 11,361,845,432 | `d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23` | `PASS FW-SZ-310` |
| ModelScope VAE | 1,409,401,152 | `0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996` | `PASS FW-SZ-310` |
| ModelScope tokenizer | 21,454,081 | 多文件；计入FW-SZ-310 payload | `PASS FW-SZ-310` |

## 6. 分项执行记录

### FW-SZ-001 — live preflight + clone exact source

- 状态：`PASS`；同轮主操作已确认 source clone 完成且 exact HEAD 为 `7faa711...`。
- 目的：检查 exact targets、space/inode、GPU 0、network、remote main；clone并 detached到 current PIN。
- command file：
  [`../../../local_scripts/remote_commands/shenzhen_fastwam_r0_preflight_clone_20260822.sh`](../../../local_scripts/remote_commands/shenzhen_fastwam_r0_preflight_clone_20260822.sh)
- command-file SHA256：`3812e346ec7f8183693463fe838e6185df1180a9b28a1f985a400922e9955326`。
- SSH transport：authorized fixed-host-key Paramiko；密码仅当前进程；先身份探针。
- expected targets：source/env 必须 absent；models/results/cache 可存在时先列明，不覆盖。
- exit code：成功完成；精确数值和原始输出路径待主操作 closeout 补入，不能据此猜写 clone wall-time。
- stdout/stderr evidence：`PENDING raw-output path`。
- result：`/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711`，
  HEAD=`7faa71108368fbb3b6885649f112af607427a2d4`。
- source size/remote/status原始行：`PENDING raw-output extraction`。
- problem / cause / fix / retest：本次 clone 未收到问题报告；原始输出 closeout 后再确认。
- next breakpoint：base env command 已启动；不再重复 clone。

### FW-SZ-002 — current source contract audit

- 状态：`PASS`；2026-08-22 10:42 CST，`chenyiteng`，fixed-host-key Paramiko，只读，exit 0。
- 目的：记录 current `pyproject.toml`、vendor lock、official configs、现有 asset/config 候选与 CUDA toolchain。
- command file：
  [`../../../local_scripts/remote_commands/shenzhen_fastwam_r0_source_contract_20260822.sh`](../../../local_scripts/remote_commands/shenzhen_fastwam_r0_source_contract_20260822.sh)
- command-file SHA256：`9614cfb36a27dc3ff4c6b944d42fbb79fbdbc4b38cd301f828164e29e57d53fa`。
- 结果：source clean `7faa711...`；current pyproject 明确 Torch/Torchvision cu128 与 numpy 2.2.6；vendor
  RoboTwin lock=`bf44be51...`；native assets tree=`04e19cbb...` 且约 16 GiB；RLinf compatibility
  task_config tree=`f114afc...`，不是 vendor 所需输入；系统 `/usr/local/cuda -> 12.9`。
- 该命令没有写服务器；输出已由本次会话完整捕获，后续 vendor command 只按上述 exact contract 生成。

### FW-SZ-100 — official base env

- 状态：`STOPPED`；用户要求正式 PPO 立即启动后，本轮 owned install 收到精确 SIGINT。
- command file：
  [`../../../local_scripts/remote_commands/shenzhen_fastwam_r1_install_official_env_20260822.sh`](../../../local_scripts/remote_commands/shenzhen_fastwam_r1_install_official_env_20260822.sh)
- command-file SHA256：`ab3117875b5f67051d2ae36f3ada5b100ff354a71551154b0c9df22fff3ba318`。
- precondition：S0 pass；env target absent。
- must record：Python、torch、torch CUDA、torchvision、Fast-WAM editable path、numpy、cache paths、route。
- exact official core：Python 3.10、torch 2.7.1+cu128、torchvision 0.22.1+cu128、current
  `pip install -e .`。
- 本条命令范围：创建单一 env/cache；安装 tooling；安装 torch/torchvision；`pip install -e .`；执行 GPU0
  tensor probe、imports、`pip check`、freeze/hash 与 env/cache size。
- 尚未包含：RoboTwin simulation deps、CuRobo、assets、task_config、HF/ModelScope models、render、expert、
  inference。
- exit：remote rc=`1`；终态为 `ERROR: Operation cancelled by user`。这是用户主线切换后的预期中止，
  不是 pip dependency resolution failure。
- completion marker：没有 `FASTWAM_R1_OFFICIAL_ENV_OK`；没有最终 import/pip-check/freeze/size验收。
- 已完成：env 创建、Python 3.10、基础 packaging/tooling。
- 未完成：torch下载/安装、torchvision、Fast-WAM editable install与GPU/import验证。
- next：不得重跑本原脚本，因为其 `test ! -e "$ENV"` 和 `conda create` 会对现有env fail-fast；改由
  `FW-SZ-101` 从现有 prefix/cache 续装。

### FW-SZ-101 — resume existing base env

- 状态：`PASS`；2026-08-22 10:44–11:01 CST，`chenyiteng`，physical GPU 3，只修改 Fast-WAM 独立
  env/cache；没有停止、改参或写入 PPO。
- command file：
  [`../../../local_scripts/remote_commands/shenzhen_fastwam_r1_resume_official_env_20260822.sh`](../../../local_scripts/remote_commands/shenzhen_fastwam_r1_resume_official_env_20260822.sh)。
- SHA256：`880f7bd974e0d72147fe1de0fe65dc966229d0edaeb81587c7a269ddf82794b6`。
- precondition：确认旧 install PID 已退出；读取现有 env 包清单与 pip cache；不删除任何 partial cache。
- exact continuation：激活 `/home/chenyiteng/venvs/fastwam-7faa-py310-cu128`，恢复同一 cache/route，
  安装 torch 2.7.1+cu128 + torchvision 0.22.1+cu128，再执行 current source `pip install -e .`。
- acceptance：原 FW-SZ-100 末尾的 Python/Torch/Torchvision/Fast-WAM、physical GPU 3 隔离 tensor probe、
  editable root、`pip check`、freeze/hash/size 与前后 GPU 3/4–7 快照全部补齐。该 probe 不是 RoboTwin
  simulator 或 Fast-WAM inference；后两者仍受主存 gate 限制。
- 主 continuation exit 1，但安装主体成功；唯一失败发生在末尾 `pip check`，见 `FW-I-002`。后续窄清理与
  final acceptance exit 0，并输出 `FASTWAM_R1_OFFICIAL_ENV_OK`，因此本阶段最终记为 `PASS`。
- 实际验收：Python `3.10.20`、Torch `2.7.1+cu128`、Torch CUDA `12.8`、Torchvision
  `0.22.1+cu128`、numpy `2.2.6`、packaging `25.0`、Fast-WAM `0.1.0` editable root 精确指向
  `/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711`；physical GPU 3 映射为进程内
  `cuda:0`，H100 capability `(9,0)` tensor probe 通过；`pip check` 为 `No broken requirements found.`。
- freeze SHA256：`2b85ae3b7e111af4e4bffc8edbca9f42b003d80bc739eff074e7c2f3d924f449`；env
  `7.4 GiB`、cache `1.4 GiB`；`/home` 仍余 `2.2 TiB`、`/data` 仍余 `3.1 TiB`。验收后 GPU 3 为
  `4 MiB`，PPO 4–7 仍有 compute load。
- 辅助命令及 SHA256：
  - conflict probe：`shenzhen_fastwam_r1_packaging_conflict_probe_20260822.sh` /
    `2c6d5b8e67dd6fa2272a06a25af159591ac8bea8bd4299d47e7b571ec185eb4d`，exit 0；
  - narrow cleanup/accept：`shenzhen_fastwam_r1_remove_extra_build_tooling_and_accept_20260822.sh` /
    `976398f31eea2458f15c68bd31271310e6051b4048b9917350e99eb8b5203627`，功能验收通过后因错误日志路径
    hash exit 1，见 `FW-I-003`；
  - final acceptance：`shenzhen_fastwam_r1_final_acceptance_20260822.sh` /
    `0c765ba0251bbbee1ef99560da726c96817e19b456e5c46834a9df0242c4a269`，exit 0。
- env 安装后订阅刷新：used `69.126 GiB`、remaining `30.874 GiB`、expiry
  `2026-08-23 14:01:48 CST`；相对 10:28 快照消耗约 `3.615 GiB`。大模型下载前继续按 revision/hash 与
  before/after quota 执行，避免重复下载。
- 后续状态：本条之后的 `FW-SZ-110/200` 已完成，见下节；`FW-SZ-101` 本身没有启动 simulator、render、
  expert 或 Fast-WAM inference。

#### 2026-08-22 10:28 CST 只读恢复点刷新

- source 仍为 detached clean `7faa71108368fbb3b6885649f112af607427a2d4`，约 11 MiB；env 为
  Python 3.10.20、约 228 MiB，只含 packaging/pip/setuptools/wheel；cache 约 4 MiB。
- torch、torchvision、Fast-WAM editable install、model/result 目录仍未完成；没有遗留 install PID。
  所以下一步仍是本条 continuation，不重跑 clone/`conda create`。
- GPU 0–3 无 compute process，GPU 3 从设备隔离角度可用；但同时运行的 PPO cgroup 已达约
  `1.437 TiB`，host available 约 `555.6 GiB`，且 EnvWorker 私有匿名内存仍增长。安装与轻量 source
  probe 不直接争用 PPO 4–7；真实 simulator/model inference 是否并发需先处理主存风险，不能只看空卡。
- Mihomo/GitHub/PyPI/HF/ModelScope 首页均可达。订阅 total `100 GiB`、used `65.511 GiB`、remaining
  `34.489 GiB`、expiry `2026-08-23 14:01:48 CST`；total/expiry 未观察到充值后变化。
- 完整命令/hash、双账号、系统和主存证据见
  [`../../rlinf-shenzhen-pi0-ppo-rlt/evidence/07_SERVER_LIVE_AUDIT_20260822.md`](../../rlinf-shenzhen-pi0-ppo-rlt/evidence/07_SERVER_LIVE_AUDIT_20260822.md)。本条状态仍为 `PLANNED`；本次没有续装或下载。

#### 2026-08-22 source-locked 本地复核与执行包

- 本地 official Git object 已直接读取 exact commit `7faa71108368fbb3b6885649f112af607427a2d4`，不是
  浮动工作树内容；`README.md` blob=`b691cbb230c72a6b0220afc99e9d89992c5ab07b`，
  `pyproject.toml` blob=`71a05cdc856186683a255f23b0f155ed26e4a951`。
- exact README 的环境序列为 Python 3.10 → pip upgrade →
  `torch==2.7.1+cu128 / torchvision==0.22.1+cu128` → `pip install -e .`。exact pyproject 同时声明
  Python `>=3.10`、同一 Torch/Torchvision pin、`numpy==2.2.6` 与 PEP 517
  `setuptools>=68 + wheel`；因此 continuation 没有借用旧 AutoDL 依赖版本。
- command 不运行 `conda create`，只使用现有 prefix；先断言 source HEAD/origin/clean、Python 3.10、
  physical GPU 3 无 compute app，再设置 `CUDA_VISIBLE_DEVICES=3`。GPU 4–7 只做前后 `nvidia-smi`
  只读快照，不向 PPO driver/Ray 发信号或调用其文件。
- proxy 只由该 command process source `/etc/profile.d/mihomo-proxy.sh`；pip/HF/ModelScope/Torch/
  extensions/Triton/CUDA/XDG/tmp 全部显式落到 `/home/chenyiteng/cache/fastwam-7faa`，不改 shell profile、
  pip config 或 Git config。
- editable 验收同时检查 distribution version `0.1.0`、PEP 610 `direct_url.json` 的 editable root 精确
  指向 locked source、Torch CUDA 12.8、单一 visible H100、`numpy==2.2.6`、`pip check`、freeze SHA 和
  source tracked diff。唯一 completion marker 为 `FASTWAM_R1_RESUME_OFFICIAL_ENV_OK`。
- 本包不安装 RoboTwin/CuRobo、不复制 assets、不下载 Fast-WAM/Wan 模型，也不启动 render、expert、
  simulator 或 evaluator；真实执行结果保持 `PENDING`。

### FW-SZ-110 — vendor runtime deps and CuRobo

- 状态：`PASS`；2026-08-22 本轮，`chenyiteng`，只修改 Fast-WAM 独立 env/cache 与 vendor 内 CuRobo
  source；physical GPU 3 只用于 import/CUDA acceptance，没有触碰 PPO 4–7。
- 最终主 command file：
  [`../../../local_scripts/remote_commands/shenzhen_fastwam_r1_install_sim_deps_curobo_v078_20260822.sh`](../../../local_scripts/remote_commands/shenzhen_fastwam_r1_install_sim_deps_curobo_v078_20260822.sh)，
  SHA256=`e68ac7ffdf8c06cd597b19c66c0b41eedfb0d3a810280422c8f3b5b7f9f21f99`。
- 窄探针/修复/最终验收：
  - MPLib × NumPy 2 probe：`shenzhen_fastwam_r1_mplib_numpy2_probe_20260822.sh`，
    SHA256=`9b830d4b3423d090090d4b9b1c3c0b2e3b3d85f7612f36cd6463aab45ef20f44`；
  - CuRobo SCM repair：`shenzhen_fastwam_r1_curobo_scm_metadata_repair_and_accept_20260822.sh`，
    SHA256=`c099fa5043449a8f5c8acdc909801bcb9cc1faead2fd5e41363db42d4843c7d3`；
  - after-inputs final acceptance：`shenzhen_fastwam_fw_sz_110_final_acceptance_after_inputs_20260822.sh`，
    SHA256=`f80b0eabf34f189b24fcdf2489cdf7f262f29d83c9b8eca929458334d2880105`，exit 0，
    completion marker=`FASTWAM_FW_SZ_110_OK`。
- actual runtime：Python `3.10.20`、Torch `2.7.1+cu128`、Torch CUDA `12.8`、Torchvision
  `0.22.1+cu128`、NumPy `2.2.6`、SciPy `1.15.3`、SAPIEN `3.0.0b1`、MPLib `0.2.1`、
  setuptools `80.9.0`、setuptools-scm `9.2.2`、Warp `1.11.1`、CuRobo distribution `0.7.8`。
  其余显式 direct pins 由上述最终主 command file 和 freeze 固定。
- build authority：CuRobo source=`$FW/third_party/RoboTwin/envs/curobo`，HEAD
  `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`；实际 `CUDA_HOME=/usr/local/cuda-12.9`、nvcc
  `12.9`、`TORCH_CUDA_ARCH_LIST=9.0`、`MAX_JOBS=8`。没有复用 RLinf/native-ACT `.so`。
- binary acceptance：`lbfgs_step_cu`、`kinematics_fused_cu`、`line_search_cu`、`tensor_step_cu`、
  `geom_cu` 五个本环境编译扩展全部 import；Warp CUDA/H100 capability `(9,0)` 初始化、CuRobo API、
  vendor evaluator eager imports全部通过。没有安装 PyTorch3D，因为当前 RGB+qpos 路径不调用可选 FPS；
  没有预移植 native ACT fused-LBFGS workaround。
- official compatibility patches：SAPIEN URDF/SRDF UTF-8 与 MPLib remove-`or collide` 只对 exact old line
  应用；命令输出保存了 before/after line、SHA256、control-byte scan和 `py_compile` 结果。
- `pip check` 的唯一输出被精确门控为
  `mplib 0.2.1 has requirement numpy<2.0, but you have numpy 2.2.6.`；当时仅完成 import acceptance，
  Fast-WAM 的 NumPy尚未降级。`FW-SZ-400` 的真实 planner segfault 后，该结论由 `FW-I-007` 更新。
- final footprint：env `9.7 GiB`、cache `2.2 GiB`、CuRobo source/build `239 MiB`；freeze SHA256 已捕获为
  `27e083c...` 前缀（本账不补猜缺失后缀）。physical GPU 3 验收后回到 `4 MiB`。
- warning boundary：import 仍打印 SAPIEN `pkg_resources` deprecated 与 `Vulkan ICD not found` warning；
  本条只证明 import，不把 warning 判为 render 成功。必须在 `FW-SZ-400` 用 `Render Well` 单独闭合。

### FW-SZ-200 — vendor-local assets, task_config and policy

- 状态：`PASS`；2026-08-22 本轮，exit 0，completion marker=`FW_SZ_200_VENDOR_INPUTS_OK`；无网络、
  package install、model download、simulator 或 evaluator 操作。
- pre-status probe：`shenzhen_fastwam_fw_sz_200_status_probe_20260822.sh`，
  SHA256=`d1afecc265f2c336da914e93ef403c3a9514644564c0812e751ee806e37edf2a`，确认三个目标初始 absent、
  Fast-WAM tracked tree 与 CuRobo HEAD保持锁定。
- final command file：
  [`../../../local_scripts/remote_commands/shenzhen_fastwam_fw_sz_200_vendor_inputs_draft_20260822.sh`](../../../local_scripts/remote_commands/shenzhen_fastwam_fw_sz_200_vendor_inputs_draft_20260822.sh)，
  SHA256=`2c84b879fa8434e25b3609c6508c5200fdcd4f7a35b3c5fceeceffe905608ec9`。
- assets authority：native official source HEAD=`30954692d06ba7e89f7a6b76064f4062c488fa81`，HF revision
  `a967b852afa21a9cbf19a198f7e653109042e87c`；只取 `background_texture/embodiments/objects` 三目录。
  source共 `20,597` regular files、apparent `16,608,557,398` bytes、allocated `16,659,820,544` bytes；
  三个结构 manifest SHA256 前缀依次为 `c6caa0d...`、`3e6389...`、`e7ac714...`。
- copy：使用 `cp -a --reflink=never` 向 vendor staging 做普通独立复制，逐目录 structure diff 后原子发布；
  target不是 symlink/reflink复用。`/data` 实际 used delta=`16,659,857,408` bytes。
- task config：从 native 完整 clone 的本地 Git object
  `bf44be51cf5717a5595ce59447f2cf5263d2aa95:task_config` 离线 `git archive`；tree
  `fdb995fb05a65f4ee6bdfaf633a89631af93db33`，7 files、`4,070` payload bytes。RLinf compatibility
  tree `f114afc63a3ffac0ac6f5ef9d780d25c0aee5a14` 被现场核验后明确拒绝。
- updater/policy：运行 locked vendor updater blob=`bf62165597df152077d276f07b6bd74290e5bc3b`，生成6份
  CuRobo embodiment YAML，全部只引用 vendor-local assets，没有 native/RLinf path leakage；policy symlink
  精确指向 current source内
  `/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/experiments/robotwin/fastwam_policy`。
- postflight：Fast-WAM tracked diff/index均为空；唯一 expected untracked entry 为
  `third_party/RoboTwin/policy/fastwam_policy` symlink。assets/task_config 由 current ignore contract覆盖，
  没有覆盖 Fast-WAM tracked vendor source。

### FW-SZ-300 — locked HF release

- 状态：`PASS`；start=`2026-08-22 03:35:54Z`，end=`2026-08-22 04:11:59Z`，
  `download_wall_seconds=2085`，exit 0，marker=`FASTWAM_FW_SZ_300_RELEASE_OK`。
- command file：
  [`../../../local_scripts/remote_commands/shenzhen_fastwam_fw_sz_300_download_release_20260822.sh`](../../../local_scripts/remote_commands/shenzhen_fastwam_fw_sz_300_download_release_20260822.sh)，
  SHA256=`1b791a2fff7d282ce1d09965395cbb6aed0ad0b613b9e763b93ec9cc32fce493`。
- target：`/data/chenyiteng/models/fastwam/release-8eaceeb`，revision
  `8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`。
- checkpoint：`12,041,813,092` bytes，SHA256
  `776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63`；stats：`88,715` bytes，
  SHA256=`7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`。
- 两文件 payload=`12,041,901,807` bytes；target tree=`12,041,918,418` bytes；完成后 `/data` free=
  `3,367,330,164,736` bytes。
- quota：immediately before used/remaining=`69.192/30.808 GiB`；after=`80.458/19.542 GiB`；expiry
  `2026-08-23 06:01:48Z` 未变化。

### FW-SZ-310 — ModelScope inference components

- 状态：`PASS`；start=`2026-08-22 04:12:56Z`，end=`2026-08-22 04:49:40Z`，
  `download_wall_seconds=2119`，exit 0，marker=`FASTWAM_FW_SZ_310_MODELSCOPE_OK`。
- command file：
  [`../../../local_scripts/remote_commands/shenzhen_fastwam_fw_sz_310_download_modelscope_20260822.sh`](../../../local_scripts/remote_commands/shenzhen_fastwam_fw_sz_310_download_modelscope_20260822.sh)，
  SHA256=`cfd89bfda32fec3c4c6cee4e3154bc8eab85873545589b4f23e6b578d623116f`。
- source：`DIFFSYNTH_DOWNLOAD_SOURCE=modelscope`；target base=
  `/data/chenyiteng/models/fastwam/diffsynth`。
- actual payload=`12,792,700,665` bytes；base tree=`12,792,746,372` bytes：
  - T5=`11,361,845,432` bytes，SHA256=`d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23`；
  - VAE=`1,409,401,152` bytes，SHA256=`0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996`；
  - tokenizer=`21,454,081` bytes；`pretrained_5b_dit_files=0`。
- quota post-download：约12:56 CST，sanitized provider API HTTP 200；remaining/used/total=
  `19.531/80.469/100 GiB`，expiry=`2026-08-23 14:01:48 CST`。它与FW-SZ-300后的remaining
  `19.542 GiB`几乎相同，只说明本次ModelScope下载没有显著消耗该订阅额度，不推断未知原因。

### FW-SZ-400 — import/render/expert functional gate

- 状态：`PARTIAL / FAIL`；run id=`fw-sz-400-20260822_042858`；physical GPU 3。
- render：在 vendored RoboTwin cwd 运行 official `script/test_render.py`，exit 0，日志明确包含
  `Render Well`。SAPIEN 的 Vulkan warning 未阻断真实 render。
- expert：运行 official
  `script/collect_data.py adjust_bottle fw-sz-400-20260822_042858_demo_clean_1ep`，进程 segfault，阶段整体验收失败。
- faulthandler 复现：崩溃点为
  `mplib/sapien_utils/conversion.py:311 -> Box(side=shape.half_size*2)`，上游调用链经过 MPLib planner、
  RoboTwin `robot.py` / `_base_task.py`，最终来自 `adjust_bottle.setup_demo`；不是 CuRobo fused-LBFGS 错误。
- 版本对照：失败的 Fast-WAM env 为 NumPy `2.2.6`、MPLib `0.2.1`、SAPIEN `3.0.0b1`；同机已工作的
  native RoboTwin env 为 NumPy `1.26.4`，MPLib/SAPIEN 版本相同。MPLib 0.2.1 官方依赖为 `numpy<2`。
- 唯一窄修复：
  [`../../../local_scripts/remote_commands/shenzhen_fastwam_mplib_numpy126_fix_20260822.sh`](../../../local_scripts/remote_commands/shenzhen_fastwam_mplib_numpy126_fix_20260822.sh)，
  SHA256=`3db1bc4a59e306464030051c0bc8e1229aed6cf021ed2305b9baf6b908cd03e4`；仅以
  `--no-deps --force-reinstall` 将NumPy `2.2.6`降到`1.26.4`，MPLib保持`0.2.1`。
- 按用户纠偏没有独立重跑 expert；`FW-SZ-500` official evaluator直接复测并成功闭合simulator + policy主路径。

### FW-SZ-500 — official Fast-WAM single evaluator

- 状态：`PASS`；run=`fw-sz-500-20260822_045105`；start=`2026-08-22 04:51:05Z`，
  end=`2026-08-22 04:53:19Z`，exit 0，marker=`FASTWAM_FW_SZ_500_SINGLE_EVAL_OK`。
- command file：
  [`../../../local_scripts/remote_commands/shenzhen_fastwam_fw_sz_500_single_eval_20260822.sh`](../../../local_scripts/remote_commands/shenzhen_fastwam_fw_sz_500_single_eval_20260822.sh)，
  SHA256=`f2b6aed2e976186c857cb197c2bfd0996aa5499ed8a97641607014a55ad6a1b2`。
- exact semantic diff：只把 official evaluator规模缩为1 episode并显式锁 release语义；不改模型数学。
- resolved overrides：

```text
task=robotwin_uncond_3cam_384_1e-4
EVALUATION.task_name=adjust_bottle
EVALUATION.task_config=demo_clean
EVALUATION.eval_num_episodes=1
EVALUATION.instruction_type=unseen
EVALUATION.sigma_shift=5.0
EVALUATION.replan_steps=24
EVALUATION.skip_get_obs_within_replan=true
gpu_id=3
```

- 另显式锁定`EVALUATION.action_horizon=null`、`EVALUATION.num_inference_steps=10`、source seed=`42`；
  environment start seed=`4300000`，实际accepted seed=`4300001`。
- result：`1/1 success`；完整 evaluator退出并输出config/log/result/video。peak GPU=`30,274 MiB`，peak host
  used=`1,635,090,612 KiB`。
- run root：
  `/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fw-sz-500-20260822_045105`。
- success video：
  `/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fw-sz-500-20260822_045105/adjust_bottle/episode0_randomized-false_success-true.mp4`；
  H.264、640×480、119 frames、11.9 s、87,954 bytes。
- local light copy：
  [`fastwam_adjust_bottle_official_seed42_20260822.mp4`](fastwam_adjust_bottle_official_seed42_20260822.mp4)；
  copy前C盘free=`38.38 GiB`。checkpoint、Wan组件、assets与env等大物全部留在服务器。
- 边界：1/1只证明accepted seed 4300001的一次成功轨迹，不外推稳定成功率。

### FW-SZ-600 — closeout

- 状态：`SKIPPED`（不另建服务器closeout命令）。
- 原因：`FW-SZ-500`已生成run/log/config/result/resource/video等轻量证据；本地只复制一个87,954-byte MP4，
  大物全部留服务器。本账、09号计划、10号执行包和专题索引直接据实际结果收口；根`HANDOFF.md`由主代理统一路由。

## 7. 问题与修复索引

| 问题 ID | 首次操作 | 现象 | 根因 | 唯一修改 | 复测 | 状态 |
|---|---|---|---|---|---|---|
| `FW-I-001` | `FW-SZ-100` | `ERROR: Operation cancelled by user` | 用户要求立即切换正式PPO主线，精确SIGINT当前owned安装 | 不改包；保留env/cache，后续用continuation续装 | `FW-SZ-101 PASS` | `RESOLVED` |
| `FW-I-002` | `FW-SZ-101` | editable install 后 `pip check` 报 `vcs-versioning` 需要 `packaging>=26.2` | 昨夜启动脚本额外安装了无人消费的 `setuptools-scm -> vcs-versioning`；official Fast-WAM 明确 pin `packaging==25.0` | 先遍历 installed metadata 证明只有 `setuptools-scm` 消费它，再仅卸载这两个 build-only 包，不改 official pin | `pip check` 无 broken requirements；完整 CUDA/editable 验收通过 | `RESOLVED` |
| `FW-I-003` | `FW-SZ-101` | 窄清理脚本功能验收通过后，哈希不存在的预期日志路径而 exit 1 | continuation stdout 由 SSH 会话捕获，原脚本没有创建该远端日志文件 | 不重复安装；另跑只读 final acceptance，去掉不存在日志假设 | exit 0 + `FASTWAM_R1_OFFICIAL_ENV_OK` | `RESOLVED` |
| `FW-I-004` | `FW-SZ-110` | resolver判定 `mplib==0.2.1` 的 `numpy<2.0` 与 Fast-WAM `numpy==2.2.6` 冲突，并在安装前停止 | 当时仅证明 wheel 可 import，尚未证明真实 planner ABI 可用 | exact wheel曾以 `--no-deps` 安装，import通过 | `FW-SZ-400` 的真实 planner 调用后续触发 `FW-I-007`，说明 import 结论不足 | `SUPERSEDED` |
| `FW-I-005` | `FW-SZ-110` | CuRobo首次编译后 distribution version 为 `0.0.0`，未满足预期 `0.7.8` | base-env清理移除了旧SCM build tooling；CuRobo源码版本由 `setuptools-scm` 生成，缺少它时只得到fallback metadata | 安装与 `packaging==25.0` 兼容且不引入 `vcs-versioning` 的 `setuptools-scm==9.2.2`；同一source/torch/toolkit下 `--no-deps --force-reinstall -e` | `nvidia-curobo==0.7.8`；HEAD仍为 `d64c4b...`；五个sm90 `.so`、Warp/H100和CuRobo API imports通过 | `RESOLVED` |
| `FW-I-006` | `FW-SZ-110` | 一次acceptance在completion marker前停止于顶层 `import envs` | 验收顺序过早：vendor assets/task_config/policy 尚未由 `FW-SZ-200` 就绪；不是已观察到的simulator/runtime缺陷 | 不改包或算法；先完成 `FW-SZ-200` exact inputs，再运行 after-inputs final acceptance | exit 0 + `FASTWAM_FW_SZ_110_OK`；vendor eager imports通过 | `RESOLVED` |
| `FW-I-007` | `FW-SZ-400` | official expert 在 `mplib/.../conversion.py:311 Box(...)` segfault | Fast-WAM env 为 NumPy 2.2.6 + MPLib 0.2.1；同机 working native env 为 NumPy 1.26.4，且 MPLib 0.2.1 官方要求 `numpy<2` | 本env以`--no-deps --force-reinstall`将NumPy降到1.26.4，MPLib保持0.2.1 | 不做独立expert retest；`FW-SZ-500` official evaluator exit 0、1/1 success | `RESOLVED` |

历史问题只能作为候选，不预填为本机问题：

- old SAPIEN `pkg_resources` → current env 已锁 setuptools 80.9.0，import通过但仍有deprecated warning；
  Vulkan ICD warning与真实render一起保留到 `FW-SZ-400`，不能由import代替。
- CuRobo 0.7.8 + Warp 1.15 → current env 已固定 Warp 1.11.1且imports通过；本次实际失败不是 CuRobo 路径。
- old `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` 401 → current official component source为ModelScope。
- native ACT H100/cu121 fused LBFGS illegal instruction → current Fast-WAM cu128不预先移植 workaround。
