# 深圳 H100 × Fast-WAM official RoboTwin standalone：current-HEAD 执行计划

更新时间：2026-08-23  
状态：**`PASS`。official source/env/vendor inputs、HF checkpoint、ModelScope inference components 与 NumPy/MPLib 兼容修复均完成；`FW-SZ-500` 已在 physical GPU 3 完成 `adjust_bottle` 单集 official inference，1/1 success。后续default-off DVAC四任务×16全部自然完成：adjust/move/turn/pick=`16/11/10/12` success，总计49/64；GPU3已释放。`FW-SZ-400` 仍按其真实结果保留为 `PARTIAL / FAIL`。**  
范围：只部署 Fast-WAM 官方 RoboTwin 推理，不接入 RLinf、不训练 Fast-WAM、不修改 PPO 代码或运行环境。

实际逐命令记录入口：
[`evidence/SHENZHEN_STANDALONE_OPERATION_LEDGER_20260822.md`](evidence/SHENZHEN_STANDALONE_OPERATION_LEDGER_20260822.md)。

## 0. 2026-08-22 完成态交接

当前完成态：

1. official Fast-WAM 已 clone 到
   `/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711`，精确 HEAD 为
   `7faa71108368fbb3b6885649f112af607427a2d4`。
2. 2026-08-22 10:44–11:01 CST 已从保留 prefix/cache 完成 continuation：Python 3.10.20、Torch
   2.7.1+cu128、Torchvision 0.22.1+cu128、CUDA 12.8、Fast-WAM editable、H100 tensor 与 `pip check`
   全部通过；env/cache 为 7.4/1.4 GiB。唯一 dependency 问题及窄修复见执行账 `FW-I-002`。
3. `FW-SZ-110` 已完成 simulation deps、CuRobo `0.7.8@d64c4b...` 的 H100 sm90 本机编译和最终
   imports；`FW-SZ-200` 已完成 20,597 个 official asset files 的普通独立 copy、vendor task config、
   6份embodiment YAML 与 current policy link。
4. `FW-SZ-310` 已完成 ModelScope T5/VAE/tokenizer 下载，payload 为 `12,792,700,665` bytes，且没有
   下载 5B DiT。随后本独立 env 的 NumPy 由 `2.2.6` 窄降到 `1.26.4`，MPLib 保持 `0.2.1`。
5. `FW-SZ-400` 已在 physical GPU 3 得到
   official `test_render.py` 的 `Render Well`；随后 official `collect_data.py` 在 MPLib 的
   `Box(side=shape.half_size*2)` 处 segfault，因此该阶段仍记为 `PARTIAL / FAIL`；没有单独重跑 expert。
6. `FW-SZ-500` 直接执行 official `eval_robotwin_single.py`，run
   `fw-sz-500-20260822_045105` 于 `04:51:05Z–04:53:19Z` exit 0，marker
   `FASTWAM_FW_SZ_500_SINGLE_EVAL_OK`；`adjust_bottle / demo_clean / unseen / 1 episode` 为
   `1/1 success`，实际 accepted seed=`4300001`。
7. 用户随后授权default-off DVAC观测与直接真实扩量。adjust-bottle sequential-16为`16/16`；P2中的
   move-stapler为`11/16`、turn-switch为`10/16`、pick-diverse-bottles为`12/16`，均自然完成、fatal=0；
   parent/children退出、GPU3释放。精确实现、命令、问题与逐任务终态只以
   [`evidence/SHENZHEN_DVAC_IMPLEMENTATION_LEDGER_20260822.md`](evidence/SHENZHEN_DVAC_IMPLEMENTATION_LEDGER_20260822.md)
   为准。

本项目的计算主线仍是 latest RLinf π0 PPO。2026-08-22 10:28–10:33 CST live 审计确认 physical GPU
0–3 无 compute process、PPO 仅使用 4–7；但 PPO session cgroup 已约 1.437 TiB、host available 约
555.6 GiB，EnvWorker 私有匿名内存仍增长。用户随后允许 Fast-WAM 环境安装和下载在不触碰 PPO 的前提下
继续；这不扩展为 simulator/inference 授权，也不改变 PPO 参数、进程、文件或 GPU placement。

Fast-WAM 是独立辅助线：source/env/models/results 与 PPO 全隔离。`FW-SZ-101/110/400/500` 均固定使用
physical GPU 3，PPO 保持 physical GPU 4–7；本轮没有修改 PPO 参数、代码、环境或 placement。

## 1. 目标、成功定义与非目标

本轮只闭合一条官方 standalone 路径：

```text
Fast-WAM official current source
  -> current official release checkpoint/stats
  -> current Fast-WAM vendored RoboTwin
  -> adjust_bottle / demo_clean / unseen / 1 episode
  -> official eval_robotwin_single.py
  -> exit code + resolved config + log + result + video
```

成功必须同时满足：

1. Fast-WAM source 精确为 `7faa71108368fbb3b6885649f112af607427a2d4`，而不是浮动 `main`。
2. release checkpoint 与 stats 锁到 Hugging Face revision
   `8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`，大小和 SHA256 与第 4 节一致。
3. 使用 Fast-WAM 自带 `third_party/RoboTwin`；不拿深圳 native RoboTwin 或 RLinf compatibility tree
   覆盖它。
4. official single evaluator 完整结束；resolved 配置明确为旧 release 所需的
   `sigma_shift=5.0`、`replan_steps=24`，并保存可复核 log/config/result/video。
5. 记录实际 GPU/RAM、wall-clock、退出码与产物；单 episode 成败只作为该 seed 的结果，不能外推成功率。

本轮不做：

- 不把 Fast-WAM 安装进 latest RLinf venv。
- 不修改 latest RLinf PPO worktree、配置或 checkpoint。
- 不使用 AutoDL 的 `/root/autodl-tmp`、`network_turbo`、旧 conda env 或旧编译产物。
- 不下载 5B DiT、ActionDiT 预处理产物、RoboTwin 全量数据集或 LIBERO 数据。
- 不运行 Fast-WAM 训练、RLinf 集成、GRPO/PPO、IDM 训练或 8-GPU manager。
- 不把旧 AutoDL 1/1 结果当作深圳 current-HEAD 已跑通。

## 2. 来源权威与旧材料边界

本轮事实优先级：

1. `SZ-H100` 当次 live source/runtime/output；
2. official Fast-WAM `7faa711...` 的 README、`pyproject.toml`、configs 和 evaluator；
3. official model revision 与逐文件 hash；
4. 旧 AutoDL `45d8e145...` runbook、`fastwam.md` 和历史成功 run，只作为问题线索。

current official 入口：

- [Fast-WAM official commit `7faa711...`](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4)
- [current README](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/README.md)
- [current `pyproject.toml`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/pyproject.toml)
- [official single RoboTwin evaluator](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/experiments/robotwin/eval_robotwin_single.py)
- [official release revision `8eaceeb...`](https://huggingface.co/yuanty/fastwam/tree/8eaceeb24c3cc92ff2a9c9a9d266a4941b836705)
- [vendored RoboTwin upstream lock `bf44be51...`](https://github.com/RoboTwin-Platform/RoboTwin/tree/bf44be51cf5717a5595ce59447f2cf5263d2aa95)

用户提供的 `E:\0school\研二上\iclr27\fastwam.md` 是历史操作记录，不是本轮命令授权或 current
官方说明。它提供的有效线索包括 setuptools/Warp/ModelScope 三类真实问题；其中的机器路径、代理方式、
版本默认值和命令必须重新核对，不能直接照抄到深圳。

### 2.1 old AutoDL 与 current Shenzhen 的精确分界

| 项目 | AutoDL 历史 oracle | 深圳 current 主路径 | 本轮处理 |
|---|---|---|---|
| Fast-WAM source | `45d8e1458921d83f8ad6cf9ce993d371208dabd0` | `7faa71108368fbb3b6885649f112af607427a2d4` | 只运行 current；旧版不建第二套 runtime |
| 时间/差异 | 2026-07-17 lock | 2026-08-20 commit；ahead 8 commits | 记录 37 files、`+2881/-1156` 的 upstream 演进 |
| RoboTwin vendor | `bf44be51...` | 仍为 `bf44be51...`；manager/vendor 文件未出现在 compare diff | 继续使用 current source 自带 vendor |
| action shift | old release 语义为 5 | current 默认变为 1 | **eval 必须显式覆盖 `EVALUATION.sigma_shift=5.0`** |
| checkpoint | `yuanty/fastwam` release | README 说明旧 release 仍兼容 | 锁 current HF revision 与文件 hash，不依赖浮动仓 |
| Python/Torch | Python 3.10；torch 2.7.1+cu128 | 仍为 Python>=3.10；torch 2.7.1+cu128 | 独立 Python 3.10 prefix |
| 依赖 | numpy 1.26.4 等 | numpy 2.2.6、datasets 4.8.5、torchcodec 0.4 等 | 以 current `pyproject.toml` 为准 |
| 机器 | AutoDL A800 | 深圳 8×H100 | 不继承路径、代理、CUDA_HOME、资源结论 |

`45d8... -> 7faa...` 的 8 个 upstream commit 涉及 acceleration/optional IDM、LeRobot 3、LIBERO
workers、action-shift 默认值、joint compile 与 IDM action-only 等；RoboTwin manager/vendor 没有出现在
该 diff，但核心 model/runtime 已变化。因此本轮选择 current source，同时用显式 `sigma_shift=5.0`
保留这份旧 release checkpoint 的正确推理语义。

### 2.2 current dependency 变化

current `pyproject.toml` 相对旧 lock 的重要变化：

| 包 | old | current |
|---|---:|---:|
| `av` | 16.0.1 | 15.1.0 |
| `datasets` | 3.6 | 4.8.5 |
| `deepspeed` | 0.18.5 | 0.18.7 |
| `numpy` | 1.26.4 | 2.2.6 |
| `pyarrow` | 23 | 24 |
| `torchcodec` | 0.5 | 0.4 |
| `tqdm` | 4.66.5 | 4.68.3 |

vendored RoboTwin 的旧 requirements 仍固定 `scipy==1.10.1`，它与 current `numpy==2.2.6` 不兼容。
因此不能机械运行 vendor 全 requirements；先安装 current Fast-WAM，再只补 standalone evaluator 实际需要的
simulation 包，并使用兼容的 SciPy 1.15.x。最终以 imports、render、expert smoke 和 official evaluator
为验收，不以单独的 `pip check` 代替行为验证。

## 3. 隔离布局与 PPO 并发边界

### 3.1 固定目标

```text
/data/chenyiteng/projects/fastwam-standalone/
└── FastWAM-7faa711/                         # official detached source

/home/chenyiteng/venvs/fastwam-7faa-py310-cu128/
                                                  # standalone-only Python prefix
/home/chenyiteng/cache/fastwam-7faa/             # pip/HF/ModelScope/Torch extension/tmp
/data/chenyiteng/models/fastwam/                 # release + DiffSynth inference components
/data/chenyiteng/results/fastwam-standalone/     # logs/config/results/video/evidence
```

不复用或写入：

```text
/home/chenyiteng/venvs/rlinf-7d07-openpi-robotwin/
/data/chenyiteng/projects/rlinf-shenzhen/
/data/chenyiteng/results/rlinf-shenzhen/
/data/chenyiteng/projects/robotwin-native/RoboTwin/       # 只读资产来源
/data/chenyiteng/projects/rlinf-shenzhen/RoboTwin-RLinf-support/
```

### 3.2 与 PPO 是否冲突

- clone、依赖安装和模型下载不占 PPO 的 4–7 卡；路径、Python prefix、cache、source 与 output 全部隔离。
- Fast-WAM single evaluator 固定 physical GPU 3；当前 PPO 使用 physical GPU 4–7，因此 GPU 集合不重叠。
- 二者仍共享 host RAM、CPU、`/data` I/O、网络和 SAPIEN/Vulkan driver。本轮S5与PPO并发时使用GPU 3，
  PPO保持GPU 4–7；2秒资源记录捕获Fast-WAM peak GPU和整机used，不另造一套环境。
- 不允许用外层 `CUDA_VISIBLE_DEVICES=4,5,6,7` 影响 Fast-WAM；它的 evaluator 命令只见 GPU 3。

## 4. 模型、资产、网络与磁盘预算

### 4.1 official model lock

Hugging Face repo：`yuanty/fastwam`  
revision：`8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`

| 文件 | bytes | SHA256 |
|---|---:|---|
| `robotwin_uncond_3cam_384.pt` | 12,041,813,092 | `776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63` |
| `robotwin_uncond_3cam_384_dataset_stats.json` | 88,715 | `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095` |

official inference 还需要：

| 来源 | 组件 | bytes（已核来源） | 备注 |
|---|---|---:|---|
| ModelScope | Wan T5 | 11,361,845,432 | SHA256 `d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23` |
| ModelScope | Wan2.2 VAE | 1,409,401,152 | SHA256 `0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996` |
| ModelScope | tokenizer | 约 21.5 MB | 小文件集合 |

模型净体积约 24.84 GB（十进制）。`sim_robotwin` 对该 release 设置 skip pretrained DiT，因此不下载
5B DiT，也不做 ActionDiT preprocessing。

### 4.2 assets 与 task config 复用合同

| 输入 | 可以复用什么 | 不能复用什么 | 处理 |
|---|---|---|---|
| native RoboTwin | 已下载的 official assets 内容 | 整棵源码、环境、planner `.so` | 向 Fast-WAM vendor 做独立普通 copy |
| native `task_config` | 仅当 Git tree 精确为 `fdb995fb05a65f4ee6bdfaf633a89631af93db33` | 浮动 current config | tree 不符就从 `bf44be51...` 获取精确目录 |
| RLinf compatibility tree | 无 | 其 `task_config` tree `f114afc...`、代码、assets layout | 明确禁止混入 |
| CuRobo | source revision/问题经验 | 旧 env 编译出的 `.so` | 在最终 Fast-WAM torch 环境重编 |

Fast-WAM vendor 的 assets 不能整树 symlink/hardlink：其 updater 会写 embodiment 绝对路径；深圳 `/data`
现场也不支持 reflink。独立普通 copy 预计新增 `16,659,820,544` bytes，随后在 vendor cwd 执行并核验
official path updater。`task_config` 采用精确 tree 的普通 copy，以便本专题配置不污染来源树。

### 4.3 2026-08-22 同轮容量/网络锚点

这些值来自同轮主操作的 live preflight，执行下载前仍要在流水账重新刷新：

- `/`：约 237 GiB free；不放模型、assets、source 或 cache。
- `/home`：约 2.2 TiB free；只放 env/cache/build/tmp。
- `/data`：约 3.2 TiB free；放 source/assets/models/results。
- Mihomo：`127.0.0.1:7890`，Paramiko command shell 必须显式 source 系统 proxy profile。
- 订阅：100 GiB total，约 52.98 GiB used、47.02 GiB remaining；到期
  `2026-08-23 14:01:48 CST`。本次刷新没有显示扩容或续期。

HF release 约 12.04 GB 会消耗代理流量；ModelScope 约 12.79 GB 是否直连由服务器现行规则决定，不能预先
假定“免费”。每次大下载前记录 route、remaining 和目标 free space；下载后记录实际目录增量。无需把任何
大文件下载到 Windows C:。

## 5. 分阶段执行计划

### S0 — live preflight 与 official source lock

目标：确认 exact targets、GPU 0/PPO 4–7、空间、网络和 current remote HEAD；clone 后 detached 到 PIN。

已准备的 command file：

- `local_scripts/remote_commands/shenzhen_fastwam_r0_preflight_clone_20260822.sh`
- `local_scripts/remote_commands/shenzhen_fastwam_r0_source_contract_20260822.sh`

执行前必须记录 command-file SHA256；执行后必须记录完整 stdout/stderr、exit code、source commit、remote、
status 与 source size。若目标 source/env 已存在，脚本 fail-fast，先审计而不是覆盖。

通过条件：

```text
FastWAM HEAD = 7faa71108368fbb3b6885649f112af607427a2d4
working tree clean
origin = official repo
GPU 0 target state known
/home and /data target capacity known
```

### S1 — standalone Python/CUDA environment

主语义仍遵循 current official README：Python 3.10、torch 2.7.1+cu128、torchvision 0.22.1+cu128、
`pip install -e .`。深圳只增加路径、网络和 vendor compatibility pin。

计划命令语义：

```bash
conda create -p /home/chenyiteng/venvs/fastwam-7faa-py310-cu128 python=3.10 -y
conda activate /home/chenyiteng/venvs/fastwam-7faa-py310-cu128

python -m pip install "setuptools==80.9.0" wheel ninja setuptools_scm
python -m pip install \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -e /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711
```

上述 `conda create` 是首次安装语义，当前不得重跑。现有 prefix 的精确 continuation 已冻结为
`local_scripts/remote_commands/shenzhen_fastwam_r1_resume_official_env_20260822.sh`，SHA256
`880f7bd974e0d72147fe1de0fe65dc966229d0edaeb81587c7a269ddf82794b6`；它已执行同一 README 的
Torch/Torchvision/editable 安装和 base-env 验收，并以 `CUDA_VISIBLE_DEVICES=3` 隔离一条 H100 tensor
probe。完整 precondition、cache、问题/修复与验收见执行账 `FW-SZ-101`；当前为 `PASS`。

simulation 包只按 current vendor imports 补齐，不能机械整装 vendor requirements。初装保留 current
Fast-WAM 的 NumPy 2.2.6；`FW-SZ-400` 的真实 planner segfault 后，才依据 MPLib 0.2.1 的 `numpy<2`
合同实施单一例外：窄降到 NumPy 1.26.4。关键兼容 lock：

```text
setuptools==80.9.0
scipy==1.15.x
CuRobo v0.7.8 @ d64c4b005459db10c5dd867d8b30a87d5bda9bdb
warp-lang==1.11.1
```

CuRobo 必须在最终 torch/CUDA 环境重编。深圳不能硬抄 AutoDL 的 `/usr/local/cuda-12.8`；先记录
`torch.version.cuda`、`nvcc --version` 和实际 `CUDA_HOME`，确认兼容后再编译。

MPLib 的 remove-`or collide` 是 RoboTwin official install 语义。SAPIEN UTF-8 patch 只在 current package
仍命中精确旧行时应用；若上游已修复则只记录，不重复修改。每个 patch 前后保存目标行与 hash。

现场结果（`FW-SZ-110 PASS`）：

- 主安装命令
  `local_scripts/remote_commands/shenzhen_fastwam_r1_install_sim_deps_curobo_v078_20260822.sh`，SHA256
  `e68ac7ffdf8c06cd597b19c66c0b41eedfb0d3a810280422c8f3b5b7f9f21f99`；after-inputs final acceptance
  SHA256=`f80b0eabf34f189b24fcdf2489cdf7f262f29d83c9b8eca929458334d2880105`，exit 0，marker
  `FASTWAM_FW_SZ_110_OK`。
- 实际 build 为 Torch `2.7.1+cu128` / torch CUDA `12.8` / `CUDA_HOME=/usr/local/cuda-12.9` / nvcc
  `12.9` / H100 sm90；CuRobo source仍是 `d64c4b...`，distribution经 `FW-I-005` 窄修复后为 `0.7.8`。
  `lbfgs_step`、`kinematics_fused`、`line_search`、`tensor_step`、`geom` 五个本地 `.so` 与 Warp 1.11.1
  全部import通过；没有借用 RLinf/native-ACT binary，也没有移植ACT LBFGS workaround。
- MPLib 0.2.1 的 wheel metadata 声明 `numpy<2.0`，而 current Fast-WAM固定 NumPy 2.2.6；`FW-I-004`
  当时只证明 exact wheel在 NumPy 2 下可import，因此临时用 `--no-deps`。`FW-SZ-400` 后续证明真实
  planner 调用会 segfault，这条 import-only 结论已被 `FW-I-007` 取代。
- SAPIEN UTF-8 与 MPLib remove-`or collide` 按 exact old line完成，before/after hashes、control-byte
  scan与compile已记录。env/cache/CuRobo最终约 `9.7 GiB / 2.2 GiB / 239 MiB`，GPU 3释放后为 `4 MiB`。
- evaluator eager imports最终通过；SAPIEN仍打印 `pkg_resources` deprecated 和 `Vulkan ICD not found`
  warning。该结果只满足import门，不满足render门，见S4。

### S2 — vendored RoboTwin inputs

顺序：

1. 确认 vendor lock `bf44be51cf5717a5595ce59447f2cf5263d2aa95`。
2. 从 native official assets 向 `$FW/third_party/RoboTwin/assets` 做独立普通 copy，记录来源/目标 bytes、
   file count、关键目录。
3. 在 vendor cwd 运行当前 source 中的 embodiment path updater，并核验路径只指向 vendor copy。
4. 仅复制 tree=`fdb995fb...` 的 `task_config`；禁止复制 RLinf-support 的 `f114afc...`。
5. 建立 official policy link：

```bash
ln -s \
  "$FW/experiments/robotwin/fastwam_policy" \
  "$FW/third_party/RoboTwin/policy/fastwam_policy"
```

现场结果（`FW-SZ-200 PASS`）：

- 命令
  `local_scripts/remote_commands/shenzhen_fastwam_fw_sz_200_vendor_inputs_draft_20260822.sh`，SHA256
  `2c84b879fa8434e25b3609c6508c5200fdcd4f7a35b3c5fceeceffe905608ec9`，exit 0，marker
  `FW_SZ_200_VENDOR_INPUTS_OK`。
- native official三类 assets 共 `20,597` files，apparent `16,608,557,398` bytes、allocated
  `16,659,820,544` bytes；通过 `cp -a --reflink=never` 普通独立 copy，`/data` used实际增加
  `16,659,857,408` bytes。target不是symlink或reflink复用。
- `task_config` 从本地 exact Git object `bf44be51...:task_config` 离线导出，tree=`fdb995fb...`，
  7 files/4,070 bytes；RLinf tree=`f114afc...` 被明确拒绝。
- locked updater生成6份 YAML且只引用 vendor-local assets；policy link只指向 current Fast-WAM policy。
  Fast-WAM tracked diff/index为空，唯一 expected untracked entry 是该policy symlink。

### S3 — official release 与 inference components

HF 命令必须显式 revision：

```bash
huggingface-cli download yuanty/fastwam \
  robotwin_uncond_3cam_384.pt \
  robotwin_uncond_3cam_384_dataset_stats.json \
  --revision 8eaceeb24c3cc92ff2a9c9a9d266a4941b836705 \
  --local-dir /data/chenyiteng/models/fastwam/release-8eaceeb
```

`FW-SZ-300 PASS`：命令
`local_scripts/remote_commands/shenzhen_fastwam_fw_sz_300_download_release_20260822.sh`，SHA256
`1b791a2fff7d282ce1d09965395cbb6aed0ad0b613b9e763b93ec9cc32fce493`；`2026-08-22 03:35:54Z` 开始，
download `2,085 s`，`04:11:59Z` exit 0并输出 `FASTWAM_FW_SZ_300_RELEASE_OK`。checkpoint为
`12,041,813,092` bytes / `776475b2...9e63`，stats为 `88,715` bytes / `7a02c46c...8095`；payload
`12,041,901,807` bytes，target tree `12,041,918,418` bytes。quota remaining从 `30.808 GiB` 变为
`19.542 GiB`，expiry `2026-08-23 06:01:48Z` 不变；完成后 `/data` free=`3,367,330,164,736` bytes。

Wan T5/VAE/tokenizer 使用 current official loader 的 ModelScope 路由，
`DIFFSYNTH_DOWNLOAD_SOURCE=modelscope`；不能沿用旧错误的 Hugging Face source。

`FW-SZ-310 PASS`：命令
`local_scripts/remote_commands/shenzhen_fastwam_fw_sz_310_download_modelscope_20260822.sh`，SHA256
`cfd89bfda32fec3c4c6cee4e3154bc8eab85873545589b4f23e6b578d623116f`；`04:12:56Z–04:49:40Z`，
download wall `2,119 s`，exit 0并输出 `FASTWAM_FW_SZ_310_MODELSCOPE_OK`。payload为
`12,792,700,665` bytes，base tree为 `12,792,746,372` bytes；其中T5为 `11,361,845,432` bytes / SHA256
`d92de679881d38af9c89eff7bb1b6d6c9d96cb2b69831e4027e9ecabdd38eb23`，VAE为
`1,409,401,152` bytes / SHA256 `0e913a2ca571c75fcb63385a8edadcca73454af5842596cb1ad11e4142590996`，
tokenizer为 `21,454,081` bytes；`pretrained_5b_dit_files=0`。

约12:56 CST再次读取sanitized provider API得到HTTP 200、remaining/used/total=
`19.531/80.469/100 GiB`、expiry=`2026-08-23 14:01:48 CST`；与`FW-SZ-300`后的`19.542 GiB`
几乎相同，只说明本次ModelScope下载没有显著消耗该订阅额度，不推断具体路由原因。

### S4 — vendored RoboTwin 最小功能门（`PARTIAL / FAIL`）

- physical GPU 3 上 official `script/test_render.py` exit 0，并明确输出 `Render Well`；此前的 Vulkan
  warning 未造成真实 render 失败。
- official `script/collect_data.py adjust_bottle <1-episode-config>` 随后 segfault；faulthandler 将崩溃点
  定位到 `mplib/sapien_utils/conversion.py` 创建 `Box(side=shape.half_size*2)` 的调用链。
- 失败环境是 NumPy `2.2.6` + MPLib `0.2.1`；同机已工作的 native RoboTwin 环境是 NumPy `1.26.4`，
  且 MPLib 0.2.1 官方依赖明确为 `numpy<2`。
- 唯一窄修复已执行：本 env 以 `--no-deps --force-reinstall` 将 NumPy降到 `1.26.4`，MPLib保持
  `0.2.1`。按用户纠偏没有独立重跑 expert；`FW-SZ-500` official evaluator 已直接成功复测
  simulator + policy 主路径。

### S5 — official single-episode Fast-WAM inference

精确命令：

```bash
cd /data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711

python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=/data/chenyiteng/models/fastwam/release-8eaceeb/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.task_config=demo_clean \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.instruction_type=unseen \
  EVALUATION.sigma_shift=5.0 \
  EVALUATION.replan_steps=24 \
  EVALUATION.skip_get_obs_within_replan=true \
  gpu_id=3
```

这里不做外层 CUDA remap；official single entry 会按 `gpu_id=3` 为子进程选择 physical GPU 3。

`sigma_shift=5.0` 不能省略：current source 默认已经变成 1，而这份 old release checkpoint 的 README
兼容路径要求 5。`replan_steps=24` 同样显式写出，避免其他 YAML 默认值造成歧义。

运行前记录：resolved config、exact command、output absence、GPU 3/4–7、Ray/process、RAM/disk、模型 hash。
运行后记录：exit、wall time、success/fail、seed/step、peak GPU/RAM、log/config/result/video 路径与 bytes/hash。

单 episode 显示 Success 或 Fail 都能证明功能链闭合；只有完整退出且产物齐全才算部署成功。模型加载成功、
`strict=False` 没报错或视频存在中的任一单项都不能单独代替整条验收。

实际结果（`FW-SZ-500 PASS`）：

- run=`fw-sz-500-20260822_045105`，physical GPU 3；start/end=`04:51:05Z/04:53:19Z`，exit 0，marker
  `FASTWAM_FW_SZ_500_SINGLE_EVAL_OK`；
- resolved语义为 `adjust_bottle / demo_clean / unseen / 1 episode`、source seed `42`、environment start seed
  `4300000`、10 inference steps、`sigma_shift=5.0`、`replan_steps=24`、skip-observation=true；
- 结果为 `1/1 success`，accepted seed=`4300001`；peak GPU=`30,274 MiB`，peak host used=
  `1,635,090,612 KiB`；
- 视频：
  `/data/chenyiteng/projects/fastwam-standalone/FastWAM-7faa711/evaluate_results/robotwin/robotwin_uncond_3cam_384/fw-sz-500-20260822_045105/adjust_bottle/episode0_randomized-false_success-true.mp4`，
  H.264、640×480、119 frames、11.9 s、87,954 bytes。

### S6 — 轻量证据与恢复点

轻量收口已完成，服务器保留：

- source HEAD/status/remote；
- env freeze、关键版本、imports/pip-check；
- model size/hash；
- exact command + resolved config；
- driver log、resource summary、result summary、MP4；
- assets/task_config 来源与 copy manifest；
- 问题→原因→窄修复→复测证据。

不把 12 GB checkpoint、12.8 GB Wan 组件、16.66 GB assets 或完整 env 复制到 Windows 文档仓。
本地只复制成功视频到 `evidence/fastwam_adjust_bottle_official_seed42_20260822.mp4`；复制前C盘free=
`38.38 GiB`，所有大物仍留在服务器。

## 6. 停止条件与处理原则

| 观察 | 动作 |
|---|---|
| source/env target 已存在 | 停在只读审计，不覆盖 |
| official source HEAD 或 HF size/hash 不符 | 停止，不进入运行 |
| 代理余量不足或到期状态变化 | 停止大下载，报告 live 值与可选路由 |
| vendor task_config 不是 `fdb995fb...` | 不混用；获取 vendor lock 对应精确目录 |
| 已定位的 MPLib × NumPy 2 segfault | 只降 NumPy 到 1.26.4；不独立重跑 expert，直接由 official evaluator 复测 |
| GPU 3 被占或与目标 PPO placement 重叠 | 不启动 evaluator |
| CUDA OOM/illegal instruction/segfault | 保存版本、栈、GPU/RAM与最小重现；不循环换 seed |
| evaluator 无进展 | 先按 current source 的真实阶段判断；终止只限本轮 owned PID 且遵守当前授权 |

不为尚未出现的问题预装多套 Torch、CuRobo 或 planner fallback；也不把历史 AutoDL 解决方案自动升级为
深圳修改。

## 7. 完成后的下一步

standalone 已通过，本轮在此结束。后续若要把 Fast-WAM 接入 latest RLinf，应另建 current RLinf base 上的
独立 worktree/联合环境，并重新审计 `7faa...` 的 model/runtime 与旧 AutoDL `45d8...` 集成代码差异；
不能直接把现有旧 RLinf Fast-WAM branch 当作深圳 current 实现。
