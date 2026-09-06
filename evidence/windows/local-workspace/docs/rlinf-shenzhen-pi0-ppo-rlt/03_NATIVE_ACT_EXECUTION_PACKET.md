# RoboTwin 2.0 `adjust_bottle` ACT 三阶段功能闭环执行包

> 状态：2026-08-21 Gate A/B/C 已全部执行完成；精确 smoke 清理、官方 pickle load 与真实 simulator
> eval 均闭环。本文保留 resolved command 与结果摘要，详细问题—修复—复测见流水账。
> 本轮只证明采集、训练、推理三条链分别可运行；成功率评估只使用锁定的官方 HF ACT 权重。

## 1. 本轮边界与 source lock

| 项目 | Resolved value |
|---|---|
| RoboTwin | `30954692d06ba7e89f7a6b76064f4062c488fa81` |
| XPolicyLab parent pin | `c37109c500be67d0dea6b36bf7337bbd26e763cd` |
| XPolicyLab installed | `c07a09614dd44cc4a67483bcb9a82e7439d99926`；官方 installer 更新到该 revision；本轮 ACT 目标文件与 parent pin 相同 |
| Dataset / checkpoint revision | `TianxingChen/RoboTwin2.0@a967b852afa21a9cbf19a198f7e653109042e87c` |
| Task / embodiment / action | `adjust_bottle` / `aloha_agilex` / `joint`（14D） |
| Simulation env | `/home/chenyiteng/miniforge3/envs/RoboTwin` |
| Policy env | `/home/chenyiteng/miniforge3/envs/act` |
| GPU | `CUDA_VISIBLE_DEVICES=0`；每个 GPU 阶段前刷新空闲状态 |
| Run root | `/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1` |
| Debug train output | `$RUN/06_act_train_smoke_1epoch_not_for_eval/`；绝不用于 simulator 成功率 |
| Official eval leaf | `/data/chenyiteng/models/robotwin2-hf-a967b852/act_ckpt/act-adjust_bottle/demo_clean-50` |

本次集中批准只包含：

1. Gate A：render，采集 1 episode，完整取证后精确删除本轮专用采集根与临时 config；
2. Gate B：official clean-50 上 1 epoch ACT training smoke，正常完成 1 个 validation batch 和 3 个 optimizer updates；
3. Gate C：锁定 official HF checkpoint 的无 simulator debug 与 1 episode simulator eval/video。

不包含完整 6000-epoch 训练、超过 1 episode 的策略评估、系统/driver 修改或 RLinf 安装。

## 2. 已完成的无 simulator 前置闭环

以下阶段已经完成并写入分阶段流水账：

- official assets：16 GiB extracted；
- official `adjust_bottle/demo_clean`：50 HDF5 + 50 videos + 50 instructions，7188 frames，state/action 为 14D；
- ACT preprocess：50/50 processed HDF5，约 19 GiB；
- `TASK_CONFIGS.json` 精确 key：`demo_clean-adjust_bottle-aloha_agilex-joint`；
- `/data` 剩余约 3.2 TiB，空间不是当前 gate。

official checkpoint 只定点下载并校验以下两个文件；它们是 current ACT loader 所需的完整 leaf：

| 文件 | bytes | SHA256 |
|---|---:|---|
| `policy_last.ckpt` | 335,907,442 | `edfb0125103e67465cc2852ea1683acc2ce1060d02b81ba6f1113b5420b40690` |
| `dataset_stats.pkl` | 10,664 | `a79964a7cce7a02cd172fb669ef12c8c2f5cedd2c4bd11adfd86c4d90cb239c4` |

`dataset_stats.pkl` 由 official loader 通过 pickle 读取，因此这是显式信任 TianxingChen official release 的反序列化边界。

## 3. Gate A：render + 1 episode 数据采集 + 精确清理

### 3.1 Live preflight 与 render

先验证 source、GPU、目标不存在；不覆盖任何已有输出：

```bash
ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
CFG="$ROBOTWIN/env_cfg/task_config/sz_collect_smoke_1ep_20260821.yml"
COLLECT_ROOT="$ROBOTWIN/data/sz_collect_smoke_1ep_20260821"

test "$(git -C "$ROBOTWIN" rev-parse HEAD)" = 30954692d06ba7e89f7a6b76064f4062c488fa81
test "$(git -C "$ROBOTWIN/XPolicyLab" rev-parse HEAD)" = c07a09614dd44cc4a67483bcb9a82e7439d99926
test ! -e "$CFG"
test ! -e "$COLLECT_ROOT"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
mkdir -p "$RUN/configs"

source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin
cd "$ROBOTWIN"
set -o pipefail
timeout --signal=INT --kill-after=30s 180s \
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  python scripts/test_render.py 2>&1 | tee "$RUN/01_test_render.log"
```

`test_render.py` 自己捕获异常，故仅有 shell exit 0 不够。通过条件是日志出现 `Render Well` 且不出现
`Render Error`、Vulkan/OIDN/CUDA error。预计 1×H100、约 1 CPU、<4 GiB RAM；180 秒总上限。

### 3.2 精确 config 与采集命令

待上传 config：
[`local_scripts/remote_configs/sz_collect_smoke_1ep_20260821.yml`](../../local_scripts/remote_configs/sz_collect_smoke_1ep_20260821.yml)，
SHA256 `A00159E4FDC7D1FA2927A76B5EE9662A09D71F4DB938E6077D167F8A21F9BFFB`。它是 official
`env_cfg/task_config/demo_clean.yml` 的语义副本，唯一参数 diff：

```diff
-episode_num: 50
+episode_num: 1
```

创建后先记录全文与 SHA256，再执行：

```bash
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin
cd /data/chenyiteng/projects/robotwin-native/RoboTwin
test "$(sha256sum env_cfg/task_config/sz_collect_smoke_1ep_20260821.yml | cut -d' ' -f1)" = \
  a00159e4fdc7d1fa2927a76b5ee9662a09d71f4db938e6077d167f8a21f9bffb
set -o pipefail
timeout --signal=INT --kill-after=60s 1800s \
  env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  bash collect_data.sh adjust_bottle sz_collect_smoke_1ep_20260821 0 2>&1 | \
  tee /data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1/02_collect_1ep.log
```

预计 1×H100、2–8 CPU、<16 GiB RAM；30 分钟总上限。official `collect_data.sh` 没有 `set -e`，且
Python 结束后仍执行 cache cleanup，因此 shell exit 0 不作为成功证据。

### 3.3 成功证据与唯一删除边界

在删除前必须把以下证据写入 `$RUN/03_collect_manifest.txt` 和 Windows 流水账：

- exact tree、各文件 bytes 和 SHA256；
- actual seed；
- HDF5 schema、frame count、14D state/action、三相机首帧可解码；
- instruction JSON 可解析；
- `episode_0000000.mp4` 可由 `ffprobe` 解码。

核心产物位于：

```text
$COLLECT_ROOT/adjust_bottle/aloha_agilex/seed.txt
$COLLECT_ROOT/adjust_bottle/aloha_agilex/data/episode_0000000.hdf5
$COLLECT_ROOT/adjust_bottle/aloha_agilex/instruction/episode_0000000.json
$COLLECT_ROOT/adjust_bottle/aloha_agilex/video/episode_0000000.mp4
```

只有全部成功证据已经持久化，才执行以下精确、不可恢复的清理；无 glob：

```bash
COLLECT_ROOT=/data/chenyiteng/projects/robotwin-native/RoboTwin/data/sz_collect_smoke_1ep_20260821
CFG=/data/chenyiteng/projects/robotwin-native/RoboTwin/env_cfg/task_config/sz_collect_smoke_1ep_20260821.yml
resolved_collect="$(realpath -e -- "$COLLECT_ROOT")"
resolved_cfg="$(realpath -e -- "$CFG")"
test "$resolved_collect" = "$COLLECT_ROOT"
test "$resolved_cfg" = "$CFG"
rm -rf -- "$resolved_collect"
rm -f -- "$resolved_cfg"
test ! -e "$COLLECT_ROOT"
test ! -e "$CFG"
```

若采集失败或 hang，保留 partial tree 取证，不执行清理。H100/SAPIEN 连续 20 分钟无日志、seed 或
frame 进展时，只终止本轮 `timeout` 拥有的 process group（INT，60 秒后 KILL），记录最后 seed/阶段；
不预先修改 renderer、denoiser 或 `pencil`。

## 4. Gate B：clean-50 上的 1-epoch ACT training smoke

不启动 6000 epochs 后强杀。直接调用 official `imitate_episodes.py`，保持模型与 objective 参数，只把
预算显式缩为 1 epoch，并写入隔离的 `not_for_eval` 目录：

| 项目 | Resolved value |
|---|---:|
| train / val episodes | 40 / 10 |
| validation forwards | 1 |
| optimizer updates | 3 |
| batch / chunk | 16 / 50 |
| KL weight | 10 |
| hidden / FFN | 512 / 3200 |
| LR / seed | `1e-5` / 0 |
| cameras | head / right wrist / left wrist |
| GPU / timeout | GPU 0 / 1800 seconds |

实际执行的精确 command file：
[`local_scripts/remote_commands/shenzhen_act_train_smoke_1epoch_20260821.sh`](../../local_scripts/remote_commands/shenzhen_act_train_smoke_1epoch_20260821.sh)，
SHA256 `69BBA215D7C34B81561649C44B15F9E3527211316DE6CF9219CDF3F2117CC401`。它直接调用 official
`imitate_episodes.py`，没有 monkeypatch、hook 或自造 sentinel；只设置 `num_epochs=1`、
`save_freq=1` 与隔离 `ckpt_dir`，其他模型/objective 参数保持 official。

实际 exit 0；official tqdm 1/1 约 2.60 s。产物是 `dataset_stats.pkl` 10,609 B、
`policy_epoch_1_seed_0.ckpt` 335,910,986 B 与 `policy_last.ckpt` 335,907,442 B。该输出只证明训练机制
闭合，不能称为 official baseline，也没有进入 simulator eval。

## 5. Gate C：HF official checkpoint debug + 1-episode sim eval/video

### 5.1 无 simulator debug

```bash
ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
HF_LEAF=/data/chenyiteng/models/robotwin2-hf-a967b852/act_ckpt/act-adjust_bottle/demo_clean-50
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate act
cd "$ROBOTWIN/XPolicyLab/policy/ACT"
set -o pipefail
EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 \
timeout --signal=INT --kill-after=60s 1800s \
  bash eval.sh RoboTwin adjust_bottle "$HF_LEAF" aloha_agilex joint 0 0 0 act RoboTwin 2>&1 | \
  tee "$RUN/07_act_hf_offline_debug.log"
```

通过条件：两文件均从锁定 absolute leaf 加载，action loop 完整结束，日志出现 `[MAIN] eval finished`。

### 5.2 Scheduler dry-run 与 real 1 episode

已上传 eval config：
[`local_scripts/remote_configs/sz_adjust_bottle_act_1ep.yml`](../../local_scripts/remote_configs/sz_adjust_bottle_act_1ep.yml)，
SHA256 `783A69199E287419C084EE004BD077665823DBD6FFA3A7B30EFD630D0F6E6BC9`；remote target 为
`$RUN/configs/sz_adjust_bottle_act_1ep.yml`，不写入 source tree。完整内容：

```yaml
gpu_ids: "0"
jobs_per_gpu: 1
num_workers: 1
enable_remote: false
policy_server_ip: 127.0.0.1
policy_server_port: [18080]
tasks:
  - adjust_bottle
```

除最后的 `--dry-run` 外，dry-run 与 real run 使用同一条精确命令：

```bash
ROBOTWIN=/data/chenyiteng/projects/robotwin-native/RoboTwin
HF_LEAF=/data/chenyiteng/models/robotwin2-hf-a967b852/act_ckpt/act-adjust_bottle/demo_clean-50
RUN=/data/chenyiteng/runs/robotwin-native/adjust_bottle/three_stage_function_smoke_20260821_v1
EVAL_CFG="$RUN/configs/sz_adjust_bottle_act_1ep.yml"
source /home/chenyiteng/miniforge3/etc/profile.d/conda.sh
conda activate RoboTwin
cd "$ROBOTWIN"
test "$(sha256sum "$EVAL_CFG" | cut -d' ' -f1)" = \
  783a69199e287419c084ee004bd077665823dbd6ffa3a7b30efd630d0f6e6bc9
timeout --signal=INT --kill-after=120s 2700s \
  bash scripts/eval_policy.sh multitask \
    --config "$EVAL_CFG" \
    --policy-name ACT \
    --ckpt-name "$HF_LEAF" \
    --env-cfg-type aloha_agilex \
    --policy-conda-env act \
    --eval-env-conda-env RoboTwin \
    --bench-name RoboTwin \
    --action-type joint \
    --seed 0 \
    --task-config demo_clean \
    --test-num 1 \
    --num-workers 1 \
    --expert-check \
    --output-dir "$RUN/08_act_hf_eval_1ep_scheduler" \
    --stream-output
```

先在同一命令末尾追加裸 `--dry-run`；只有 dry-run exit 0 且打印唯一 GPU0/`adjust_bottle` job，才执行
上面的 real 命令。

Scheduler 层必须使用裸 `--expert-check`，不能写 `--expert-check=true` 或 `--expert_check`。它先用专家
筛选可解 seed；`seed=0` 从 candidate 100000 开始，actual seed 可能后移，必须从日志记录。

工程闭环成功定义：scheduler summary 为 1/1 job succeeded、`_result.txt` 存在、ACT action loop 正常
结束、`episode0.mp4` 可由 `ffprobe` 解码。current absolute checkpoint resolver 取 leaf name，预期视频：

```text
/data/chenyiteng/projects/robotwin-native/RoboTwin/eval_result/adjust_bottle/ACT/demo_clean/
  demo_clean-50/<timestamp>/episode0.mp4
```

策略结果可以是 0/1；这仍可能是工程通过，也不能用 1 episode 估计稳定成功率。不会自动扩大到 5 或
更多 episodes。real eval 预计 ACT server 与 SAPIEN client 共用 1×H100、4–8 CPU、<32 GiB RAM；
45 分钟总上限。连续 20 分钟无最近日志进展时只终止本轮 owned process group（INT，120 秒后 KILL），
保留 actual seed、最后 action step、日志和 partial video；不扩大分母或预改渲染栈。

## 6. 集中批准语义

用户已经批准并按 Gate A → B → C 条件完成执行。批准覆盖：

- 创建上述本轮专用 config/run roots；
- Gate A 成功取证后，只删除 preflight 已确认不存在、由本轮新建的 exact collect root 和 exact config；
- 达到总 timeout 或 20 分钟无进展时终止本轮 owned processes；
- official checkpoint 的可信 pickle load。

除上述专用 smoke 产物外，不删除或覆盖任何既有数据；不运行完整 6000 epochs，不扩大策略评估分母，
不修改系统包、driver 或其他用户文件。所有 command file SHA、exit code、关键输出、问题—修复—复测、资源
峰值与 artifact 路径逐条追加到 `evidence/04_ACT_PIPELINE_LEDGER.md`。

实际终态：

- Gate A：render 通过；首次 collect 精确失败于 H100/torch cu121 的 CuRobo fused `lbfgs_step_cu`
  CUDA 715。只在 Hopper/pre-cu126 条件关闭该 fused kernel 后，seed0 1/1 simulation 和 141-row数据采集
  成功；manifest 持久化后约9 MiB专用根与临时config已精确删除。
- Gate B：official direct training 1 epoch正常完成，debug-only output约641 MiB。
- Gate C：official HF checkpoint完成10×20-step offline action loop；real scheduler 1/1 job，actual seed
  100001 在step147成功，return code0。视频 H.264、320×240、148 frames、14.8 s；本地副本为
  [`evidence/act_adjust_bottle_official_hf_seed100001_20260821.mp4`](evidence/act_adjust_bottle_official_hf_seed100001_20260821.mp4)。
- 1/1=100%只报告本次单条事实，不当作稳定成功率或完整baseline复现。

## 7. 官方依据

- [RoboTwin install](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)
- [RoboTwin collect data](https://robotwin-platform.github.io/doc/usage/collect-data.html)
- [RoboTwin H/A/V GPU common issue](https://robotwin-platform.github.io/doc/common-issue/index.html)
- [XPolicyLab ACT README](https://github.com/XPolicyLab/XPolicyLab/blob/c07a09614dd44cc4a67483bcb9a82e7439d99926/policy/ACT/README.md)
- [Official ACT checkpoint leaf](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/a967b852afa21a9cbf19a198f7e653109042e87c/act_ckpt/act-adjust_bottle/demo_clean-50)
