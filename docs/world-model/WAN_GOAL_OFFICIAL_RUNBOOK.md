# Wan LIBERO Goal 官方安装与资产路线

审计日期：2026-09-30。目标：深圳3 GPU4–7，先官方 Wan Goal + OpenVLA-OFT + GRPO smoke，再准备 Wan Goal + π0.5 + GRPO。先准备源码、独立环境和资产，实际借卡、结束归还原 RLT 由主实施记录负责。

本文维护官方来源与实施路线；实际进展以 `WAN_GOAL_RUNLOG.md` 的带时间回执为准。π0.5 的观测／动作／GRPO语义适配另记；不能将资产清单齐全解释成 π0.5 + Wan 已运行验收。本机网络选择见 `WAN_GOAL_NETWORK_20260930.md`。

## 1. 官方入口与固定版本

官方明确提供 [Wan + LIBERO + OpenVLA-OFT + GRPO](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/wan.html)。本次按当前 RLinf 源码锁定，不混用旧 WorldArena fork 的安装脚本。

| 项目 | 本次核查版本 | 用途 |
|---|---|---|
| [RLinf/RLinf](https://github.com/RLinf/RLinf/tree/d34d4c320d08cb982de034aa9a011f08dc0fa217) | `d34d4c320d08cb982de034aa9a011f08dc0fa217` | 官方 Goal 配方、调度、GRPO、Wan backend；2026-09-30 HEAD |
| [RLinf/diffsynth-studio](https://github.com/RLinf/diffsynth-studio/tree/2a2e05fa1f724828b243f272540989b19a6e54f8) | `2a2e05fa1f724828b243f272540989b19a6e54f8` | Wan DiT、VAE、奖励加载；包名 `diffsynth==1.1.9` |
| [moojink/openvla-oft](https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8) | `e4287e94541f459edc4feabc4e181f537cd569a8` | 当前官方 OFT+Wan installer 的 pip Git 来源；运行模型 `implement_version=rlinf` |
| [rlinf-openpi](https://pypi.org/project/rlinf-openpi/0.1.1/) | `0.1.1` | 当前官方 OpenPI installer 实际指定的 PyPI 包 |
| [rlinf-transformer-openpi](https://pypi.org/project/rlinf-transformer-openpi/4.53.2/) | `4.53.2` | OpenPI 包依赖的 transformers fork |
| [rlinf-libero](https://pypi.org/project/rlinf-libero/0.1.3/) | `0.1.3` | 当前公开包版本；官方 installer 是不带版本的 `rlinf-libero`，实施时应记录最终版本 |

旧 Git OpenPI HEAD `c5dc4b9296a1a4739bf52828f28a579f12dce763` 可追溯，但**不是当前 installer 的默认安装来源，也不能当作 PyPI 0.1.1 的等价提交**。

官方文件：

- [Goal 主配置](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/examples/embodiment/config/wan_libero_goal_grpo_openvlaoft.yaml)：训练预算、资源并行、GRPO、模型与环境路径。
- [Goal 环境配置](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/examples/embodiment/config/env/wan_libero_goal.yaml)：Wan、VAE、KIR/reset、奖励、帧数与图像大小。
- [Wan backend](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/backend/wan.py#L62)：当前入口是 `env_type: world_model`、`backend: wan`。
- [安装器 OFT/Wan](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/requirements/install.sh#L2274)、[OpenPI/LIBERO](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/requirements/install.sh#L2317)、[Wan 依赖](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/requirements/install.sh#L3706)。

## 2. 已公开、能定位的完整资产

以下为固定 revision 的 HF tree 元数据和小配置实际读取结果。大小是十进制 GB，完整大文件的本机 SHA256/可加载性待下载后确认。

| 资产 | 固定官方位置 | 已见内容 | 总大小 |
|---|---|---|---:|
| Wan Goal bundle | [RLinf/RLinf-Wan-LIBERO-Goal](https://huggingface.co/RLinf/RLinf-Wan-LIBERO-Goal/tree/bd395971c3467de3dd19e7e6c7562af48a2894a6) | DiT、VAE、Goal 奖励、普通与KIR初态 | 14.223 GB |
| OFT Goal SFT | [Haozhan72/Openvla-oft-SFT-libero-goal-traj1](https://huggingface.co/Haozhan72/Openvla-oft-SFT-libero-goal-traj1/tree/d20e1d447dfd87c0daa121b0739e2a379f7fe334) | 4片模型、index、tokenizer、processor、config、stats，共20文件 | 15.085 GB |
| π0.5 LIBERO SFT | [RLinf/RLinf-Pi05-LIBERO-SFT](https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT/tree/45ccfcc4e28634f1576ebf78cab0fbe2fd82432d) | `model.safetensors`、LIBERO norm stats，共4文件 | 7.473 GB |
| OpenPI tokenizer | [RLinf/openpi_tokenizer](https://huggingface.co/RLinf/openpi_tokenizer/tree/befaa248e4f82954b625a421658f933dfd1a97a0) | `big_vision/paligemma_tokenizer.model` | 4.264 MB |
| 真实LIBERO评测素材 | [RLinf/LIBERO-assets，dataset库](https://huggingface.co/datasets/RLinf/LIBERO-assets/tree/3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f) | 模型运行外的仿真物件／场景素材，HF列586文件 | 本次未累计 |

前三套合计约36.782 GB／34.26 GiB，未计环境、LIBERO素材、训练checkpoint和视频。Goal-Fullshot 是另外的资产版本，本次不替换原Goal。

Wan bundle具体文件：

| 路径 | 字节数／数量 | 配置使用 |
|---|---:|---|
| `model-00001.safetensors` | 10,050,753,584 | 动作条件DiT |
| `Wan2.2_VAE.pth` | 2,818,839,170 | VAE |
| `taskemb_resnet_rm.pth` | 45,379,978 | `TaskEmbedResnetRewModel`，**不是通用示例中的 `resnet_rm.pth`** |
| `dataset/seed_*_traj_*.npy` | 496文件／97,910,364字节 | 普通reset图像 |
| `dataset/step_*_seed_*_traj_*_kir.npy` | 246文件／1,210,364,396字节 | KIR条件记录 |

该bundle共有747文件，其中742个npy。先保持官方 `enable_kir: true`，完整保留dataset目录；只取普通reset会改变官方采样分布。

OFT `dataset_statistics.json` 实读包含 `libero_goal_no_noops`，与Goal YAML的 `actor.model.unnorm_key`完全一致；action统计7维。不要误用 `libero_10_no_noops`，也不要用已经GRPO训练后的权重替代SFT起点。

本次OFT输入是**单个外部主相机RGB＋语言，不用腕部相机和proprio**。这是所选checkpoint的官方RLinf配方口径，不能推广为整个OFT模型家族都只支持单目：checkpoint自身 `config.json`未记录 `num_images_in_input/use_proprio`；官方 `model/openvla_oft.yaml`指定1／False， [loader把配置覆盖后设置图像数](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openvla_oft/rlinf/__init__.py#L84)。[策略只有图像数大于1才读取腕图](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py#L223)；processor虽然接收state形参，实际仅处理文本和pixels。真实LIBERO wrapper会同时返回主图／腕图／state，但该OFT只消费主图；主图来源是 [agentview_image](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/libero/utils.py#L79)，不严格等同于机器人头部相机。Wan返回主图、`wrist_images=None`，因此可直接适配这个OFT口径。

π0.5 stats路径是 `physical-intelligence/libero/norm_stats.json`，顶层 `norm_stats` 下有 `state`、`actions`。这是RLinf PyTorch权重格式，不能直接当作Dojo JAX checkpoint目录。模型与norm stats都在该库；加载所需的PaliGemma tokenizer另取上表。是否匹配本次Wan观测语义由适配审计负责。

本条Wan加载链 [只传入DiT和VAE](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/backend/wan.py#L81)，不要求另下载通用Wan基座、UMT5编码器或CLIP大权重。固定DiffSynth中 [tokenizer下载已注释](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/diffsynth/pipelines/wan_video_new.py#L597)，[tokenizer路径为空会跳过](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/diffsynth/prompters/wan_prompter.py#L92)，无text encoder的prompt单元会跳过。Goal奖励使用自建ResNet+10任务embedding，直接加载该45MB checkpoint，无T5依赖。

## 3. 最短独立安装路线

只在新实验源码目录和新venv运行，复用下载缓存／现成素材时先核路径；不向共享conda或现有Dojo/RLT环境安装。官方脚本会创建或同步venv，且不匹配Python版本时会重建目标venv，因此 `--venv` 必须是本实验专属路径。

当前脚本默认Python3.11.14，RLinf pyproject默认torch2.11.0；OpenPI0.1.1包metadata则列torch2.7.1，最终解析需以安装日志和freeze为准，不能凭文档猜实际torch。优先使用服务器现成、与本实验依赖匹配的独立环境克隆；否则先走官方命令，出现冲突再按具体错误处理，不预先扩展组件替换。

OFT官方命令形态：

```bash
# 在本实验固定commit的RLinf checkout内；以下路径均为待主实施确定的独立路径。
export HF_HOME=/data/chenyiteng/<experiment>/cache/huggingface
export UV_CACHE_DIR=/data/chenyiteng/<experiment>/cache/uv
export UV_PYTHON_INSTALL_DIR=/data/chenyiteng/<experiment>/cache/uv-python
export TMPDIR=/data/chenyiteng/<experiment>/tmp
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME=/data/chenyiteng/<experiment>/cache/xdg
export PIP_CACHE_DIR=/data/chenyiteng/<experiment>/cache/pip
export TORCH_HOME=/data/chenyiteng/<experiment>/cache/torch
export TRITON_CACHE_DIR=/data/chenyiteng/<experiment>/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/data/chenyiteng/<experiment>/cache/torchinductor
export DOWNLOAD_DIR=/data/chenyiteng/<experiment>/assets
export LIBERO_CONFIG_PATH=/data/chenyiteng/<experiment>/config/libero-oft
export WAN_PATH=/data/chenyiteng/<experiment>/src/diffsynth-studio

# WAN_PATH提前clone并checkout到本文2a2e05fa；官方脚本会复用、不pull。
bash requirements/install.sh embodied --model openvla-oft --env wan \
  --venv /data/chenyiteng/<experiment>/envs/oft-wan --no-root
```

`--no-root`适用于本机系统库已经齐全；如缺库，先定位具体库。调用前创建这些独立目录；已有uv可通过PATH只读复用。OFT+Wan官方分支会附带安装ManiSkill/LIBERO extras，这是上游脚本行为；其中 `download_assets.sh --assets maniskill` 遵循 `DOWNLOAD_DIR`，必须设置以避免默认写入HOME。未核必要前不执行额外全量示教数据下载。

π0.5最短现行路线：**当前安装器没有 `--model openpi --env wan` 分支**，直接调用会返回unsupported。使用官方OpenPI+LIBERO建立另一个独立venv，然后安装同一公开Wan依赖：

```bash
export LIBERO_CONFIG_PATH=/data/chenyiteng/<experiment>/config/libero-pi05
export OPENPI_DATA_HOME=/data/chenyiteng/<experiment>/cache/openpi
bash requirements/install.sh embodied --model openpi --env libero \
  --venv /data/chenyiteng/<experiment>/envs/pi05-wan --no-root
source /data/chenyiteng/<experiment>/envs/pi05-wan/bin/activate
uv pip install -e "$WAN_PATH"
uv pip install -r requirements/embodied/models/wan.txt
```

后两行逐字对应官方 `install_wan_world_model`。不要先向OFT环境覆盖OpenPI的transformers：当前OpenPI installer会覆盖transformers文件并固定 `tokenizers>=0.21,<0.22`，两套venv便于准确复现。

固定OpenPI运行依赖：`jax[cuda12]==0.5.3`、`orbax-checkpoint==0.11.13`、`tyro==1.0.13`；Wan依赖文件包含 `deepspeed==0.18.4` 及 safetensors、sentencepiece、protobuf、modelscope、ftfy、pandas、seaborn。以 [openpi.txt](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/requirements/embodied/models/openpi.txt) 和 [wan.txt](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/requirements/embodied/models/wan.txt) 为准。

独立缓存的两个容易漏项：

- `LIBERO_CONFIG_PATH`须在installer启动前设置；脚本 `reset_libero_config`会写安装路径，默认落在 `~/.libero`。素材可用 `LIBERO_ASSET_PATH` 指向已下载tree，`rlinf-libero0.1.3`的 `libero-download-assets --link` 支持复用；不改已有素材。
- OpenPI实际读取 `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`。installer的 `download_assets.sh` 默认下载到 `$HOME/.cache/openpi`，且其存在性检查少了 `big_vision` 子目录。因此独立缓存应显式按下节下载，检查实际文件，不只检查目录存在。

## 4. 固定下载命令模板

命令仅是待实施模板，`<experiment>` 和 `MODEL_ROOT`由主实施确定。HF snapshot revision已固定；不要把HTTP返回成功或目录创建当作全下载完成。

```bash
MODEL_ROOT=/data/chenyiteng/<experiment>/models
hf download RLinf/RLinf-Wan-LIBERO-Goal \
  --revision bd395971c3467de3dd19e7e6c7562af48a2894a6 \
  --local-dir "$MODEL_ROOT/wan-libero-goal"
hf download Haozhan72/Openvla-oft-SFT-libero-goal-traj1 \
  --revision d20e1d447dfd87c0daa121b0739e2a379f7fe334 \
  --local-dir "$MODEL_ROOT/oft-libero-goal-sft"
hf download RLinf/RLinf-Pi05-LIBERO-SFT \
  --revision 45ccfcc4e28634f1576ebf78cab0fbe2fd82432d \
  --local-dir "$MODEL_ROOT/pi05-libero-sft"
hf download RLinf/openpi_tokenizer \
  --revision befaa248e4f82954b625a421658f933dfd1a97a0 \
  --local-dir "$OPENPI_DATA_HOME"
```

真实LIBERO资产用 **dataset类型**；这不是 `yifengzhu-hf/LIBERO-datasets` 示教轨迹库。官方 `libero-download-assets`可下载到HF缓存并链接入本venv；需要严格固定revision时先用 `hf download --repo-type dataset RLinf/LIBERO-assets --revision 3ba78404b48e8c70fa4ac9782d5aac8e5b46d55f --local-dir <独立asset目录>`，再按上述 `LIBERO_ASSET_PATH`链接。仅有WM初态npy不等于真实LIBERO评测素材齐全。

## 5. 官方运行口径与最小需要替换的路径

官方入口：

```bash
bash examples/embodiment/run_embodiment.sh wan_libero_goal_grpo_openvlaoft LIBERO
```

启动脚本只有配置名和robot platform两个位置参数；其他Hydra参数不会自动透传。在独立checkout复制固定官方Goal为独立主配置，改实际路径和用户批准的smoke预算，保留原官方YAML作对照。不能直接继承完整官方顶层YAML，因为其中 `hydra.searchpath` 只允许主配置定义。三个必须填的模型路径：

```yaml
env:
  train:
    wan_wm_hf_ckpt_path: /data/chenyiteng/<experiment>/models/wan-libero-goal
actor:
  model:
    model_path: /data/chenyiteng/<experiment>/models/oft-libero-goal-sft
rollout:
  model:
    model_path: /data/chenyiteng/<experiment>/models/oft-libero-goal-sft
```

固定官方Goal主要口径：

| 项目 | 官方Goal YAML |
|---|---|
| 训练环境／评测环境 | Wan world_model / 真实 `libero_goal` |
| 图像、动作、条件 | 单图，256×256；7D；C8；condition5；num_frames13；KIR开启 |
| Wan推理 | 5 denoise steps；训练env offload开启 |
| 奖励 | `TaskEmbedResnetRewModel`；相对奖励；coef5 |
| GRPO | G8；action-level reward；token-level logprob；reward filtering 0.5–4.5 |
| 训练规模 | total_num_envs64；rollout_epoch16；每rollout256步；max_epochs1000 |
| actor | micro_batch32；global_batch8192；lr2e-5；FSDP+gradient checkpointing |
| 真环境评测配置 | total_num_envs496；512步；auto_reset=true；G1 |
| 运行器评测／存档 | `val_check_interval=-1`；每5epoch保存；不是自动每轮真环境评测 |

`reward.use_reward_model: false`仅关闭独立reward worker，**不关闭Wan环境内部奖励网络**。smoke的小并行和短预算是本地运行变更，应在resolved-config差异里记录；不要称作完整官方结果。π0.5正式配方需先完成适配与更新验证，不能仅替换OFT model_type后宣布官方π0.5+Wan配方复现。

## 6. 借卡前核查与日志交付

本地已准备保持官方方法项的 [OFT smoke YAML](../../local_scripts/wan_goal_20260930/config/wan_goal_oft_smoke_sz3.yaml)、[官方预算参照 YAML](../../local_scripts/wan_goal_20260930/config/wan_goal_oft_official_budget_sz3.yaml)及[启动／批量推导说明](../../local_scripts/wan_goal_20260930/config/README.md)。smoke为4卡、N32、G8、rollout1、256动作、global1024／micro32、2个runner epoch；C8／denoise5／KIR／reward等方法项不变。21:23 OFT smoke、π05 smoke、π05 formal三套配置均通过服务器Hydra解析，尚未开始GPU校验。

准备阶段可完成：模型目录文件数／字节数／下载完成回执；OFT stats key；π0.5 norm stats和tokenizer；import模块；冻结RLinf/DiffSynth/OFT源码SHA及venv freeze；Hydra最终配置；LIBERO数据路径检查。最终GPU加载、一个Wan chunk、真实参数更新和LIBERO评测在正式借卡后完成。

粗日志至少记录：准备／OFT smoke／π0.5闭环／正式训练／归还RLT各阶段结论、首个错误与修复依据、验收证据位置。细日志保存实际命令与退出码、固定源码、包版本、资产manifest、resolved config及差异、模型加载信息、损失与参数变化、输出视频、GPU进程身份和最终归还回执。具体借卡和恢复清单由主实施唯一控制，避免重复stop/resume。

本次公开审计证据（本地、轻量、可复查）：

`E:/Codex/home/visualizations/2026/09/28/01a0e6c7-bb8c-7421-8697-110ddd91d2f1/wan-goal-official-20260930/`

其中 `asset_inventory.json` 与各 `*__tree.json`保存HF固定revision的文件大小和LFS哈希；官方源码以 `/`替换为 `__`保存；`goal_stats.json`、`pi05_stats.json`、PyPI metadata、OpenPI/LIBERO小wheel源码片段保存实际读取证据。未保存或执行远端模型pickle，未运行GPU。
