# Fast-WAM × RoboTwin × RLinf × GRPO 实施日志（2026-07-17）

## 1. 范围与基线

- 目标：以已跑通的 π0 + RoboTwin + RLinf GRPO 为框架基座，接入官方 Fast-WAM RoboTwin action policy；首版只训练 action expert。
- RLinf 基线：`6d0db56bf26f972cd27fa29535f5eb939e80e5bf`。
- Fast-WAM：`45d8e1458921d83f8ad6cf9ce993d371208dabd0`。
- RLinf 使用的 RoboTwin：`/root/autodl-tmp/RoboTwin_RLinf`，现场 HEAD `481380fbd97cbf9ff830aedfb2279851e1e58969`。
- CuRobo：独立源码 `/root/autodl-tmp/src/curobo-fastwam-rlinf-v0.7.8`，`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`（v0.7.8）。
- 服务器独立 worktree：`/root/autodl-tmp/RLinf_fastwam_rlinf`，分支 `feat/fastwam-robotwin-grpo`；原 `/root/autodl-tmp/RLinf` 和 π0 行为未修改。
- 本地开发 worktree：`C:\Users\86136\Documents\rl\.rlinf-fastwam-worktree`，分支 `codex/fastwam-robotwin-grpo`。

## 2. 实际代码面

共 18 个文件：15 个新增、3 个既有文件小改。

| 类别 | 文件 | 作用与依据 |
|---|---|---|
| 新增 | `rlinf/models/embodiment/fastwam/__init__.py` | 暴露 RLinf `get_model`；沿用内置 embodied model 布局 |
| 新增 | `rlinf/models/embodiment/fastwam/robotwin_adapter.py` | 官方 deploy 的 PIL 三相机、prompt、14D state/action norm/denorm 的严格 batch 版本 |
| 新增 | `rlinf/models/embodiment/fastwam/builder.py` | 组合官方 Hydra、加载官方 checkpoint/stats、规范模块树、只开放 action expert |
| 新增 | `rlinf/models/embodiment/fastwam/fastwam_rl.py` | 官方 scheduler/conditioning/velocity 与 RLinf OpenPI Flow-SDE 的共享 ODE/SDE core |
| 新增 | `rlinf/models/embodiment/fastwam/fastwam_policy.py` | RLinf rollout、tensor-only replay、actor recompute 的 batch policy |
| 新增 | `rlinf/models/embodiment/fastwam/export.py` | RLinf DCP/full state 到官方 deploy checkpoint 的 strict-schema 导出 |
| 新增 | `examples/embodiment/config/model/fastwam_robotwin.yaml` | H/N/S、三相机、BF16、noise、eval seed、官方配置入口 |
| 新增 | `examples/embodiment/config/robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke.yaml` | 16 env/group 8/global 128/micro 2/forward 2、192 steps、GPU bucket |
| 新增 | `examples/embodiment/config/robotwin_adjust_bottle_grpo_fastwam_a800_2gpu.yaml` | 16 env×4 epoch、8 groups、noise0.3/lr5e-6 的 100-step formal 基线 |
| 新增 | `examples/embodiment/run_fastwam_robotwin_grpo.sh` | smoke/train 单入口；直接调用数据盘 joint Python并建立自描述 run 目录 |
| 新增 | `examples/embodiment/monitor_resources.py` | 复用 π0 的 22 列、2 秒资源证据格式；写 CSV/peak 并随 driver 退出 |
| 小改 | `rlinf/models/__init__.py` | lazy 注册 `fastwam_robotwin`；不提前导入 Fast-WAM 重依赖 |
| 小改 | `rlinf/workers/rollout/hf/huggingface_worker.py` | capability flag 传递 train/eval mode；旧模型分支不变 |
| 小改 | `rlinf/hybrid_engines/fsdp/strategy/fsdp2.py` | 新增 `cast_forward_inputs` 配置；默认 `true` 保持全部旧模型行为，Fast-WAM 显式 `false` 保留 FP32 replay |
| 新增 | `tests/models/embodiment/fastwam/test_adapter_builder_parity.py` | adapter、processor、module mode、严格 replay schema |
| 新增 | `tests/models/embodiment/fastwam/test_flow_sde_replay.py` | shifted schedule、OpenPI 首步、ODE/SDE、真实 transition replay、随机性/分块 |
| 新增 | `tests/models/embodiment/fastwam/test_checkpoint_export.py` | strict schema、路径解析、full-state 与 no-dist synthetic DCP |
| 新增 | `tests/models/embodiment/fastwam/test_rlinf_wiring.py` | public registry、worker capability dispatch、FSDP cast 默认兼容性 |

明确零 diff：训练入口、通用配置类、`BasePolicy`、RoboTwin env/reward、trajectory、actor worker、GRPO/PPO loss、advantage、weight syncer、runner、通用 DCP manager 和原有通用 launcher。新增的是 Fast-WAM 专用薄 launcher。

## 3. 已冻结的正式语义

- shape：三相机 composite `[B,3,384,320]`，state/action 14D，生成 H=32、执行 N=24、denoise S=10。
- 官方 scheduler 决定 raw timestep、signed `delta<0` 和 shift=5；OpenPI 决定 Flow-SDE mean/std 和 `t=1` 时使用下一 schedule 点作分母。
- stochastic rollout 每个 logical model batch 共享一个均匀 `k∈[0,9]`；每条轨迹的 initial latent 和 SDE epsilon 独立，且随机量在 resource chunk 前一次生成。
- replay 保存完整 chain `[B,11,32,14]`，old/new logprob 只保留执行前缀 `[B,24,14]`；actor 重放真实 `chain[k]→chain[k+1]`，不从最终 action 反推概率。
- `eval_seed=0` 的 singleton latent broadcast 只用于 eval：它精确等价于官方 evaluator 对每个 B=1 环境重复以同一 seed 调用。train 不使用该分支，因此不是训练萎缩。
- 只训练 canonical `model.mot.mixtures.action.*`；video expert、proprio、VAE、T5 全冻结。官方兼容 alias 保留为非注册属性，state/parameter tree 只有一个 canonical 路径。
- actor/rollout offload=false、train env offload=true、GPU bucket 512 MiB。默认 P2 是一次 fresh `lr=1e-6` 真更新；`lr=0` 仅保留为 ratio 排障覆盖项。

## 4. 环境实施与偏差处理

- 联合环境：`/root/autodl-tmp/conda/envs/FastWAM-RLinf`，Python 3.11、Torch 2.7.1+cu128、Torchvision 0.22.1+cu128。
- Fast-WAM 与独立 RLinf worktree 均 editable 安装；CuRobo 在最终 Torch/CUDA 12.8 环境重新编译。
- Fast-WAM 关键 pin：Transformers 4.49.0、Hydra 1.3.2、OmegaConf 2.3.0、NumPy 1.26.4、torchcodec 0.5、setuptools 80.9.0、warp-lang 1.11.1。
- RLinf 已跑通基线对齐：Ray 2.55.1、timm 1.0.27、SwanLab 0.8.2、TensorBoard 2.20.0。
- Python 3.11 镜像没有 toppra 0.6.8 wheel，采用 RLinf/RoboTwin 可用的 toppra 0.6.3。该适配已经由真实 render、CuRobo/Warp 和 `adjust_bottle.play_once()` 成功验证，不只依赖 `pip check`。
- 首版安装脚本的两条 `sed` 反向引用被脚本生成过程转成控制字符，导致 SAPIEN/MPLib Python 文件语法损坏。现场通过 control-byte scan/`py_compile` 发现；用相同 SAPIEN 3.0.0b1/MPLib 0.2.1 的已跑通 `FastWAM-official` 文件精确恢复，并把安装脚本改成 fail-fast、幂等的 Python exact-string patch。修复后两个文件 hash 与 oracle 环境一致。
- 现役 π0 venv 未修改。已保存 freeze/list/editable/pip-check/关键版本，并完成完整 `rsync -aH` 备份与空 dry-run：`/root/autodl-tmp/backups/RLinf-pi0-venv-golden-20260717`。现役 venv 原有 `pip check` 不为零（OpenPI/Torch、torchcodec、sympy、typeguard 的历史声明差异），仅如实记录，没有为本迁移改包。

## 5. 已执行检查

| 检查 | 结果 |
|---|---|
| joint venv `pip check` | 通过，无 broken requirements |
| Torch/CUDA/Fast-WAM/RLinf/Ray/仿真依赖 import | 通过；2×A800 可见 |
| RoboTwin `script/test_render.py` | `Render Well` |
| joint venv + RoboTwin_RLinf `adjust_bottle` scripted expert | `play_once_complete True` |
| Hydra `--cfg job --resolve` | smoke：epoch1/noise0.1/lr1e-6；formal：epoch4/noise0.3/lr5e-6；共同 16 env、group8、192/24=8、global128、micro2、forward2、cast=false，全断言成功 |
| launcher / 资源监控 | 服务器 `bash -n`、monitor `py_compile` 通过；2 秒监控自测得到 5 samples × 22 fields，并在目标 PID 退出后自动结束 |
| Ruff（仅 Python 文件）/`git diff --check` | 通过；YAML 不交给 Python linter，改由 Hydra 完整解析 |
| synthetic/unit tests | joint venv：`50 passed, 1 warning in 4.90s`；warning 仅为 single-process synthetic DCP 的预期提示 |
| public registry/worker/FSDP cast wiring | 新增 3 项测试均通过 |
| 真实官方 B=1 / B=2 / permutation / replay | 通过；详见下表 |

真实 checkpoint probe 使用 public `rlinf.models.get_model()`、正式 checkpoint/stats 和固定 synthetic RoboTwin observation：

| 指标 | 结果 | 判定 |
|---|---:|---|
| official B=1 max abs | `0.0` | 与官方 singleton oracle 完全一致 |
| B=2 permutation max/mean abs | `0.0 / 0.0` | 无样本串扰、cache 串样或顺序依赖 |
| forward batch 2 vs 两次 batch 1 max/mean abs | `0.008000642 / 0.000861770` | 通过实测 BF16 batch-shape 门限 `max<1e-2, mean<1.5e-3, max/ref<2%`；不是逐环境 fallback |
| rollout old vs actor replay new logprob max/mean abs | `0.0 / 0.0` | 同权重真实 transition 完全一致且 new logprob 保留梯度 |
| train initial latent | 两样本不相等 | 训练随机性逐轨迹独立；eval broadcast 没有进入 train |
| shared denoise index | `k=2` | 同一 logical batch 共享 k；范围仍由 unit test 覆盖全部 `0..9` |
| trainable | `824` tensors / `1,020,900,366` params | 仅 canonical action expert；总参数 `12,406,348,522` |
| model-level peak CUDA | `23.8102 GiB` | 非 FSDP runner 峰值；不能替代 P2 资源记录 |

第一次 probe 直接把 CPU-backed rollout replay 传给 policy，出现 chain 在 CPU、velocity 在 CUDA 的错误。该调用跳过了正式 `FSDPActor.train_micro_batch()` 先执行的递归 `put_tensor_device(micro_batch, self.device)`，因此修正的是 probe，而不是在生产 policy 中加入隐式 device move。正式 replay 仍在 rollout/trajectory 阶段保留 CPU/FP32，actor 边界整体搬到 GPU；`cast_forward_inputs=false` 只阻止 FSDP 把概率链量化为 BF16。

最终静态日志：`/root/autodl-tmp/fastwam-rlinf-setup/final_static_checks_20260717.log`。真实 probe 与日志：`/root/autodl-tmp/fastwam-rlinf-setup/real_fastwam_rlinf_probe.py`、`real_fastwam_rlinf_probe.log`。

2026-07-18 00:31 CST：独立 worktree 仍为 `6d0db56` / `feat/fastwam-robotwin-grpo`，代码面为 3 个 modified + 15 个新增文件。新增配置、launcher、monitor 已同步服务器；未启动真实 runner。

早先 15 文件已做逐一 SHA256/换行归一审计；本轮 formal/launcher/monitor 由 SFTP 直接同步并在服务器解析、自测。服务器侧是最终运行版本。

## 6. 尚未由 Codex 代跑的高成本验收

按用户要求，下列完整 runner 由用户执行，不能把前述 model-level probe 冒充为完成：

1. P2a：全新进程 `lr=0`，1 个 global step，保存 DCP；验收 optimizer 前 ratio≈1、KL/clip≈0、无 NaN/OOM、action-only finite grad、冻结 hash 不变、actor/rollout sync等价和资源峰值。
2. P2b：全新进程 `lr=1e-6`，1 个 global step；验收只更新 action expert且产生非零有限更新。
3. P3：从非零 `global_step_1` 新进程恢复并把绝对 `max_steps` 设为2，再跑一步；验收 model/optimizer/scheduler/RNG恢复、首轮同步、`global_step_2` DCP。
4. 从 DCP 导出官方 `.pt`，strict schema/load 后，用同一固定 fixture 比较恢复 RLinf action 与官方 loader action；官方 episode 成功不能替代数值 parity。

首版明确限制：actor/rollout offload 关闭。若以后开启 model offload，必须先修正并验证官方 `model.device`/`torch_dtype` 属性随 `.to()` 更新；不能直接把当前非 offload 证据外推。

## 7. 历史详细 P2/P3 诊断命令

本节保留用于单独 `lr=0` 和 resume/export 排障；当前 canonical smoke/train 命令见第 8 节，不要求先执行本节的多段命令。

下列命令直接调用官方 RLinf 入口，避免 `run_embodiment.sh` 无法清楚传递 lr/resume/experiment 路径。每段必须是新进程；P2a 通过后再执行 P2b，P2b 通过后再执行 P3。

### 7.1 共用环境

```bash
set -e
set -o pipefail

ROOT=/root/autodl-tmp
REPO="$ROOT/RLinf_fastwam_rlinf"
ENV="$ROOT/conda/envs/FastWAM-RLinf"
MODEL_ROOT="$ROOT/models/fastwam"
ROBOTWIN_PATH="$ROOT/RoboTwin_RLinf"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"
unset OMP_NUM_THREADS
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY all_proxy ALL_PROXY
unset HF_ENDPOINT DIFFSYNTH_SKIP_DOWNLOAD

export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MODELSCOPE_CACHE="$ROOT/cache/modelscope"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_ROOT/diffsynth"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export PYTHONUNBUFFERED=1
export EMBODIED_PATH="$REPO/examples/embodiment"
export REPO_PATH="$REPO"
export ROBOTWIN_PATH
export ROBOT_PLATFORM=ALOHA
export PYTHONPATH="$REPO:$ROBOTWIN_PATH${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0,1

CFG=robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke
SRC="$EMBODIED_PATH/train_embodied_agent.py"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_BASE="$ROOT/fastwam-rlinf-runs/$RUN_TAG"
test ! -e "$RUN_BASE"
mkdir -p "$RUN_BASE"

python "$SRC" \
  --config-path "$EMBODIED_PATH/config" \
  --config-name "$CFG" \
  --cfg job --resolve > "$RUN_BASE/resolved.yaml"
echo "RUN_BASE=$RUN_BASE"
```

### 7.2 P2a：fresh `lr=0`

```bash
P2A_ROOT="$RUN_BASE/p2a_lr0"
mkdir -p "$P2A_ROOT"
python "$SRC" \
  --config-path "$EMBODIED_PATH/config" \
  --config-name "$CFG" \
  runner.logger.log_path="$P2A_ROOT" \
  runner.logger.experiment_name=p2a_lr0 \
  runner.max_steps=1 runner.save_interval=1 \
  runner.resume_dir=null runner.ckpt_path=null \
  actor.optim.lr=0.0 \
  2>&1 | tee "$P2A_ROOT/driver.log"

P2A_CKPT="$P2A_ROOT/p2a_lr0/checkpoints/global_step_1"
test -s "$P2A_CKPT/actor/dcp_checkpoint/.metadata"
```

验收：optimizer 前 ratio≈1、KL/clip≈0，无 NaN/OOM，action expert 梯度有限，冻结模块不变。这里的 FSDP plain-rollout/actor parity 不能用 model-level BF16 action 门限豁免。

### 7.3 P2b：另一个 fresh `lr=1e-6`

P2b 不 resume P2a；先确认前一批 worker 已退出。

```bash
pgrep -af 'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker' || true
P2B_ROOT="$RUN_BASE/p2b_lr1e6"
mkdir -p "$P2B_ROOT"
python "$SRC" \
  --config-path "$EMBODIED_PATH/config" \
  --config-name "$CFG" \
  runner.logger.log_path="$P2B_ROOT" \
  runner.logger.experiment_name=p2b_lr1e6 \
  runner.max_steps=1 runner.save_interval=1 \
  runner.resume_dir=null runner.ckpt_path=null \
  actor.optim.lr=1e-6 \
  2>&1 | tee "$P2B_ROOT/driver.log"

P2B_CKPT="$P2B_ROOT/p2b_lr1e6/checkpoints/global_step_1"
test -s "$P2B_CKPT/actor/dcp_checkpoint/.metadata"
```

验收：只有 action expert 发生非零、有限更新。

### 7.4 P3：fresh resume 到绝对 step 2，再导出

```bash
pgrep -af 'train_embodied_agent.py|MultiStepRolloutWorker|EmbodiedFSDPActor|EnvWorker' || true
test -s "$P2B_CKPT/actor/dcp_checkpoint/.metadata"
P3_ROOT="$RUN_BASE/p3_resume_lr1e6"
mkdir -p "$P3_ROOT"
python "$SRC" \
  --config-path "$EMBODIED_PATH/config" \
  --config-name "$CFG" \
  runner.logger.log_path="$P3_ROOT" \
  runner.logger.experiment_name=p3_resume_lr1e6 \
  runner.max_steps=2 runner.save_interval=1 \
  runner.resume_dir="$P2B_CKPT" runner.ckpt_path=null \
  actor.optim.lr=1e-6 \
  2>&1 | tee "$P3_ROOT/driver.log"

P3_CKPT="$P3_ROOT/p3_resume_lr1e6/checkpoints/global_step_2"
test -s "$P3_CKPT/actor/dcp_checkpoint/.metadata"

BASE_CKPT="$MODEL_ROOT/release/robotwin_uncond_3cam_384.pt"
DEPLOY_CKPT="$P3_ROOT/p3_resume_lr1e6/deploy/fastwam_global_step_2.pt"
test -s "$BASE_CKPT"
test ! -e "$DEPLOY_CKPT"
cd "$REPO"
python -m rlinf.models.embodiment.fastwam.export \
  --checkpoint "$P3_CKPT" \
  --base-checkpoint "$BASE_CKPT" \
  --output "$DEPLOY_CKPT"
test -s "$DEPLOY_CKPT"
sha256sum "$DEPLOY_CKPT"
```

P3 的 `runner.max_steps=2` 是绝对上限，从 `global_step_1` 恢复后仅再执行一步。最终用同一固定 fixture 比较恢复后的 RLinf action 与导出后官方 loader action；单看官方 episode 成功不足以证明恢复/导出等价。

## 8. 当前 canonical 配置与启动

### 8.1 关键参数依据

| 参数 | smoke / formal | 依据 |
|---|---|---|
| 2×A800、actor/env/rollout 共置 | 相同 | 已跑通 π0 的 RLinf/RoboTwin 布局；不改通用 placement |
| 16 env、group8 | 相同 | π0 成功基线与社区 Fast-WAM 都使用 group8；每卡8并发 env |
| rollout epoch | 1 / 4 | smoke 只验链路；formal 为64 trajectories、8 groups，对齐社区 Fast-WAM 每卡32 trajectories |
| H/N/S、episode | 32/24/10、192 | 官方 Fast-WAM；192=8×24，无 200-step 尾段 |
| global/micro/forward batch | 128/2/2 | smoke 的16×8=128 transitions；formal 512 transitions 分4批；2是已测显存起点，待真实 FSDP smoke确认 |
| exploration | 0.1 / 0.3 | 0.1 是工程 smoke；0.3 是社区 Fast-WAM 有真实正向学习结果的较低设置，actor/rollout 同值 |
| lr | 1e-6 / 5e-6 | smoke 保守真更新；formal 接近成功 π0 5.6e-6，低于社区成功 Fast-WAM 2e-5 |
| update epoch、clip | 1、0.2 | 社区 Fast-WAM 正式 GRPO；避免昂贵 replay 重复使用 |
| Adam/WD/grad clip | .9/.95/1e-8、.01、1 | 成功 π0；grad clip1也见社区最终成功实验 |
| reward/logprob/entropy | chunk/chunk/chunk | joint Flow-SDE transition 是概率单位；entropy bonus当前为0 |
| filter | false / true | smoke 不因全成/全败丢批；formal 对齐 π0/社区并用8 groups降低空批概率 |
| KL/entropy/critic | 0/0/false | critic-free GRPO；首版不额外加尚未验证的正则 |
| FSDP | FSDP2 full-shard、no-split、BF16 | 社区 Fast-WAM 的 MoT 跨 expert 调用要求 whole-root 语义；不照搬会破坏调用 |
| FP32 replay | `cast_forward_inputs=false` | Fast-WAM 特有：chain/logprob保持FP32，模型输入内部再转BF16 |
| offload | actor/rollout false、env true | 首个双卡方案；env offload来自成功 π0，模型 offload 尚无 Fast-WAM证据 |
| sync/checkpoint | CPU-staged bucket512MiB、DCP | 社区用 bucket 避免 patch densification；首次本机 smoke 证明 GPU CUDA IPC 被容器拒绝，改用 RLinf 原生 CPU staging；DCP 不变 |
| steps/save/eval | 1/1/-1；100/10/-1 | smoke 必须产出 step1 DCP；formal沿用 π0 100-step/10-step；首轮不常驻 eval env |

兼容字段 `gamma/gae_lambda`、`value_clip/huber_delta` 在 critic-free GRPO 中不主导更新；`kl_penalty`/`entropy_type` 在系数为0时也不改变 loss。它们保留是为了维持 RLinf 常规配置结构，不据此声称启用了 GAE、value loss、KL 或 entropy bonus。

### 8.2 单命令启动与资源证据

```bash
cd /root/autodl-tmp/RLinf_fastwam_rlinf
bash examples/embodiment/run_fastwam_robotwin_grpo.sh smoke
```

通过后再启动 formal：

```bash
cd /root/autodl-tmp/RLinf_fastwam_rlinf
bash examples/embodiment/run_fastwam_robotwin_grpo.sh train
```

launcher 直接调用 `/root/autodl-tmp/conda/envs/FastWAM-RLinf/bin/python`，不依赖 Conda base。每次在 `logs/<timestamp>-<config>/` 保存 `command.txt`、`resolved_config.yaml`、`run_embodiment.log`、`train.pid`、`monitor.pid`；`resource_monitor/` 内保存 π0 同格式的 `resources.csv`、`peak.txt` 和 `monitor.log`。

一步 smoke 能证明初始同步、rollout、真实 replay、backward、optimizer、step1 DCP 和资源峰值；更新后的 actor→rollout 再同步必须由第二步/resume 验证。社区曾出现 plain-BF16 rollout 与 FSDP-BF16 actor 的首轮 ratio 偏移，所以必须现场检查 `actor/ratio`、`ratio_abs`、`approx_kl` 和 clip，不能用 `old=new` 合成值掩盖问题。

## 9. 2026-07-18 首次真实 smoke 与修正

- run：`logs/20260718_004457-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu_smoke`。
- 两卡 rollout/actor 模型成功构建；初始 actor→rollout 广播在 GPU CUDA IPC 报 `pidfd_getfd: Operation not permitted`。尚未进入 rollout、loss、backward 或 DCP，且 `oom=0/oom_kill=0`。
- 高置信本机适配：smoke/formal 的 `bucket_device: cuda` 改为 `cpu`。保留 bucket 算法、同步集合、dtype、512 MiB 分桶和 `load_instant`；两份服务器 Hydra compose 均验证为 `bucket/cpu/536870912`。
- 资源：模型就绪后 GPU0/1=`59063/59583 MiB`，约为同机 π0 GRPO 首步峰值 `2.05×`；cgroup RAM 峰 `237259/245760 MiB`，但基线 `158401 MiB` 且现场大部分是可回收 file cache。因尚未进入真实 rollout/backward，formal 未放行，下一步只重跑同一条 smoke。

## 10. 2026-07-18 成功smoke、正式run与效果审计

- 后续一步smoke已完整通过：192 env steps、真实rollout→actor replay、backward/optimizer、权重同步和DCP均完成；两卡峰值约61.8 GiB，cgroup未发生OOM。正式run的64 trajectories数量关系另见下一条，不把formal规模倒写成smoke规模。
- 正式run：`logs/20260718_020324-robotwin_adjust_bottle_grpo_fastwam_a800_2gpu`；实际resolved配置为4 env×rollout_epoch16、group4、64 trajectories、512 actor transitions、global128/micro2、noise0.3、lr5e-6、rollout offload开启、CPU-staged bucket。
- 09:12 CST 只读现场：step24/100完成、step25进行中；总均值success约97.14%、最近10步96.56%，5/24步全成功且零梯度。GPU0/1峰值63036/62511 MiB；cgroup RAM峰241932/245760 MiB；`oom=0/oom_kill=0`。
- 正式曲线是变化seed且带探索噪声的train rollout，不是fixed deterministic eval。官方论文中`adjust_bottle` clean/randomized均为100%，故本轮不涨首先判定为任务饱和与实验不可辨识，而非runner失败。
- 实现逐层复核未发现P0概率链缺陷：官方B1、B2 permutation、same-weight replay保持零误差；真实runner ratio_abs/KL/clip小且梯度有限。未完验收仍为fresh DCP resume、真实DCP export↔官方loader parity、更新/冻结子树hash。
- 配方差异：当前每步64 trajectories，成功π0与社区Fast-WAM均为256；当前RLinf固定std normalization，社区成功Fast-WAM为no-std；当前无fixed eval。详细比较、任务选择与调用链审计见实施计划第27节。
- 服务器seed清单中，较低基线任务只有`move_stapler_pad`同时具备train/eval成功seed。用户拍板下一轮使用clean、192步，保持当前正式配方不变，只切任务；随后再单变量比较no-std、LR、randomized和proprio。

## 11. 2026-07-18 `move_stapler_pad` 正式配置落地

- 新增env fragment，仅把`task_config.task_name`从`adjust_bottle`改为`move_stapler_pad`。
- 新增formal顶层配置，仅替换train/eval env defaults和experiment name；继续使用192步、4 env×rollout_epoch16=64 trajectories、group4、global128/micro2、noise0.3、lr5e-6、clean随机化、rollout/env offload与CPU-staged bucket。
- launcher语法扩展为`run_fastwam_robotwin_grpo.sh [smoke|train] [config-name]`，默认smoke/train配置不变，旧命令兼容。
- `bash -n`、文件diff和联合环境Hydra compose/resolve通过；train/eval任务均为`move_stapler_pad`，三处step limit均为192，模型YAML、checkpoint/stats、adapter、Flow-SDE、actor/loss/sync/DCP零改动。
- 社区Fast-WAM LIBERO每步32 env×8 rollout_epoch=256 trajectories、group8=32 groups；本次64 trajectories为其1/4，但horizon不同，不能把trajectory比例直接视为总actor样本比例。

## 12. 2026-07-19 GRPO π0对齐配置与Fast-WAM PPO首版实现

- 实现位置：本地镜像`C:\Users\86136\Documents\rl\.rlinf-fastwam-worktree`；服务器工作区`/root/autodl-tmp/RLinf_fastwam_rlinf`。同步前后核心文件SHA256一致；现役旧GRPO PID 915681在全部同步与静态验收后仍存活，未停止、未重启。
- 新GRPO配置：`robotwin_move_stapler_pad_grpo_fastwam_a800_2gpu_pi0_aligned.yaml`。相对旧formal结构化diff严格只有实验名、`rollout_epoch 16→32`、`global_batch_size 128→512`、`update_epoch 1→2`四项；其余包括group4、std-on、lr5e-6、noise0.3、micro2、forward2和save10均保持。数量为128 trajectories、1024 unique transitions、4 optimizer updates、2048 transition presentations。
- PPO配置与入口：新增move-stapler smoke/formal YAML及`run_fastwam_robotwin_ppo.sh`。smoke为`4×4×8=128 transitions/global128/update1`，恰好一次更新；formal为`4×16×8=512/global128/update2`，每RL step八次更新。两者均`resume_dir/ckpt_path=null`并继承官方release checkpoint，属于官方base冷启动。
- PPO critic：`build_action_conditioning()`只在PPO请求时读取官方`video_kv_cache[-1]["v"] [B,120,3072]`，严格检查shape/finite后token mean。该V来自最后一层video cache输入，已经过前29层video self-attention与text/proprio cross-attention，不读取action latent、denoise k或SDE noise。
- PPO wrapper：只在`add_value_head=true`时注册顶层`value_head`；rollout返回CPU FP32 `prev_values[B,1]`，actor replay以同一observation重建feature并返回FP32 `values[B,1]`。critic input显式detach，critic loss只更新head；Flow-SDE、scheduler、H/N/S、chain、真实selected transition和logprob零改动。
- head为RLinf `3072→1024→512→256→1` ReLU MLP。独立审查发现草案若保留FP32参数，会与Fast-WAM的FSDP2 BF16及`cast_forward_inputs=false`产生dtype/old-new路径风险；正式实现改为head参数跟随模型BF16，rollout/actor一致计算，输出再转FP32供GAE/value loss。
- optimizer/FSDP/sync/DCP复用RLinf通用链：顶层`value_head`命名进入独立`value_lr=1.1e-4`、ValueHead FSDP wrap、CPU bucket同步和DCP；GRPO无该属性。官方deploy export显式忽略`value_head.*`，训练DCP仍保留它。
- 现场检查：两个launcher `bash -n`通过；包/测试`compileall`通过；新GRPO、PPO smoke、PPO formal三份配置Hydra `--resolve`通过，resolved actor/rollout head开关与forward2一致；集中Fast-WAM测试`59 passed, 1 warning`，唯一warning为单进程DCP测试未初始化distributed，符合测试意图。
- 未执行：真实两卡PPO模型加载、rollout→GAE→backward、head同步、step1 DCP和resume。这些需要在现役训练结束后由用户单独启动PPO smoke；本轮未把静态/CPU验证冒充真实runner验收。

## 13. 2026-07-19 PPO采样与更新预算对齐修正

- 用户拍板PPO formal应直接参考第12节新增的π0对齐GRPO预算，而不是保留“先小global验证value head”的旧临时方案。formal因此从`4 env×16 epoch/global128/update2`改为`4×32/global512/update2`，并把DCP间隔从20改为10；得到128 trajectories、1024 unique transitions、4 optimizer updates和2048 transition presentations，与新GRPO一致。
- smoke保持formal相同的4路env并发、两卡placement、micro2、model-forward2和offload，只把顺序采样降为`rollout_epoch1`。`4×1×(192/24)=32 transitions`，因此global随数据量设为32、update1，恰好完成一个真实optimizer update；这满足RLinf的`global % (micro×world)=0`以及rollout按global切分合同。
- PPO特有参数不变：`group1`、`gae/actor_critic`、`gamma/gae_lambda=0.99/0.95`、policy/value clip 0.2、value_lr 1.1e-4和observation-side value head。Fast-WAM模型、scheduler、Flow-SDE、H/N/S、head实现、FSDP/sync/DCP代码均未修改。
- 本地`test_training_config_contracts.py`已更新为显式比较PPO formal与π0对齐GRPO的采样、global batch、micro batch和update epoch，并断言smoke为32 transitions/一次更新；又增加两项严格结构diff：formal相对GRPO只能改变PPO算法/head合同，smoke相对formal只能改变六个预算/生命周期字段。本地单文件结果`7 passed`。
- 三个修改文件已同步到服务器独立worktree。联合环境Hydra resolved结果：smoke预算`(4 trajectories, 32 transitions, 1 update, 32 presentations)`；PPO formal和GRPO π0对齐版均为`(128, 1024, 4, 2048)`，head/forward2和global整除合同通过。服务器配置合同`7 passed`，Fast-WAM集中测试`61 passed, 1 warning in 4.45s`；warning仍为预期的单进程DCP提示。现役GRPO PID 915681验证后仍存活，未停止或重启。
