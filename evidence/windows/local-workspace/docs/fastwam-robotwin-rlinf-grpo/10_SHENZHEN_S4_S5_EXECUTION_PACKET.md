# Fast-WAM 深圳 S4/S5 source-locked 执行包

> 状态：**EXECUTED / CLOSED**（2026-08-22）。`FW-SZ-400` 为 `PARTIAL / FAIL`：render通过、expert在MPLib × NumPy 2处segfault；NumPy窄降到1.26.4后，`FW-SZ-500` official single evaluator exit 0并得到1/1 success。

## 1. 固定来源与命令

- Fast-WAM：[`7faa71108368fbb3b6885649f112af607427a2d4`](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4)。
- vendor provenance：RoboTwin [`bf44be51cf5717a5595ce59447f2cf5263d2aa95`](https://github.com/RoboTwin-Platform/RoboTwin/tree/bf44be51cf5717a5595ce59447f2cf5263d2aa95)。
- vendor `task_config` tree：`fdb995fb05a65f4ee6bdfaf633a89631af93db33`；`demo_clean.yml` blob `7e0de0aec1ad1b151570486120c7d25050be01c4`。
- HF release：[`yuanty/fastwam@8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`](https://huggingface.co/yuanty/fastwam/tree/8eaceeb24c3cc92ff2a9c9a9d266a4941b836705)。
- checkpoint：`12041813092` bytes，SHA256 `776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63`。
- stats：`88715` bytes，SHA256 `7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`。

命令文件：

- [FW-SZ-400 render + one expert](../../local_scripts/remote_commands/shenzhen_fastwam_fw_sz_400_render_expert_gate_20260822.sh)，SHA256：`163370bf140a9b35ba00b15bbff3d10e85f45627632807864f07b5f55f88f3aa`。
- [FW-SZ-500 official single evaluator](../../local_scripts/remote_commands/shenzhen_fastwam_fw_sz_500_single_eval_20260822.sh)，SHA256：`f2b6aed2e976186c857cb197c2bfd0996aa5499ed8a97641607014a55ad6a1b2`。

## 2. 源码审计后真正需要执行的内容

### 2.1 S4：imports 不重复；render通过，expert失败

`FW-SZ-110` 已经完成 evaluator eager imports、Torch/Warp/CuRobo/H100 验收，因此S4没有重复imports。实际执行两项行为门：

1. vendor root 下运行 `python script/test_render.py`；
2. vendor root 下运行 `python script/collect_data.py adjust_bottle <config_basename>`，完成一个 clean expert episode。

这两个 CLI 来自 exact vendor source；`test_render.py` 和 `collect_data.py` 分别是 Git blob `97350aa2...`、`6bd9206f...`，与 upstream `bf44be51` 相同。

`test_render.py` 捕获异常后无参数 `exit()`，所以 render 失败也可能是 rc 0。这里唯一额外而必要的判据是：日志含 `Render Well`，且不含 `Render Error`。这是源码真实语义，不是新增诊断树。

official `demo_clean.yml` 是 `episode_num=50 / use_seed=false / save_path=./data / collect_data=true`。命令只生成一个唯一 basename，并把 `episode_num` 改为 1；其余保持 official。保留 `save_path=./data` 是因为 instruction generator 固定写 vendor `./data/<task>/<config>/instructions`，这样所有产物仍在同一隔离目录：

```text
$VRT/data/adjust_bottle/<run_id>_demo_clean_1ep/
├── seed.txt
├── data/episode0.hdf5
├── video/episode0.mp4
├── scene_info.json
└── instructions/episode0.json
```

实际结果：official `test_render.py`输出`Render Well`；expert在
`mplib/sapien_utils/conversion.py:311 -> Box(side=shape.half_size*2)`处segfault。失败env为NumPy 2.2.6 +
MPLib 0.2.1；按MPLib的`numpy<2`合同把NumPy窄降到1.26.4，MPLib保持0.2.1。用户纠偏后没有再建独立
expert门，直接由S5复测完整simulator + policy链。

### 2.2 S5：official single evaluator 的真实语义

Fast-WAM README 的完整 benchmark 入口是 manager；同一 official repo 内的 `eval_robotwin_single.py`（blob `f57fdaba...`）是 manager 实际调用的单任务入口。这里用它缩成一张卡、一个任务、一集，不改 policy 数学：

```text
cwd=$FW
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=<exact release .pt> \
  EVALUATION.dataset_stats_path=<exact stats.json> \
  EVALUATION.task_name=adjust_bottle \
  EVALUATION.task_config=demo_clean \
  EVALUATION.eval_num_episodes=1 \
  EVALUATION.instruction_type=unseen \
  EVALUATION.action_horizon=null \
  EVALUATION.num_inference_steps=10 \
  EVALUATION.sigma_shift=5.0 \
  EVALUATION.replan_steps=24 \
  EVALUATION.skip_get_obs_within_replan=true \
  EVALUATION.output_dir=<unique_leaf> \
  seed=42 gpu_id=<live_selected_physical_gpu>
```

resolved 语义是 `H=32 / replan=24 / 10 denoise steps / 14D qpos / sigma_shift=5.0`。`sigma_shift=5.0` 来自 current README 对 original release checkpoint 的兼容说明，不能省略。

GPU 有一个重要调用细节：single parent 会把 `cfg.gpu_id` 直接写成 child 的 `CUDA_VISIBLE_DEVICES`，所以这里必须传现场选中的**物理卡号**。不能把 physical 3 外层映射为逻辑 0 后再传 `gpu_id=0`。

current source 默认 seed 是 42；vendor expert/environment 起始 seed 为 `100000 * (1 + 42) = 4300000`。若 expert check 换 seed，child log 的 `current seed` 才是实际评测 seed。

`EVALUATION.output_dir` 只取 basename；实际结果固定在：

```text
$FW/evaluate_results/robotwin/robotwin_uncond_3cam_384/<run_leaf>/
├── eval_adjust_bottle_<timestamp>.log
├── eval_config_adjust_bottle.yaml
└── adjust_bottle/
    ├── _result_clean.txt
    └── episode0_randomized-false_success-{true|false}.mp4
```

Fast-WAM-local `eval_policy.py`/`_base_task.py` 与 upstream blob 不同；正是这些 official integration changes 提供单集数、显式 output、视频重命名和三相机 `640x480` eval video。运行权威是 Fast-WAM `7faa711` 的 vendored files，而不是把 upstream 文件覆盖回来。

实际结果：run=`fw-sz-500-20260822_045105`，physical GPU 3，`04:51:05Z–04:53:19Z`，exit 0，
marker=`FASTWAM_FW_SZ_500_SINGLE_EVAL_OK`。resolved值保持上表语义；source seed=42、environment start
seed=4300000、accepted seed=4300001，单集结果为1/1 success。视频为H.264、640×480、119 frames、
11.9 s、87,954 bytes。

## 3. GPU、RAM、output 与停止条件

`09` 曾写 physical 0，后续安装验收及本轮实际运行均使用 physical GPU 3；S5启动前确认该卡没有compute process：

```bash
nvidia-smi
free -h
pgrep -af 'train_embodied_agent.py|eval_embodied_agent.py|raylet|gcs_server|eval_robotwin_single.py|collect_data.py' || true

# 只有本次刷新确认 physical 3 空闲时：
export FASTWAM_PHYSICAL_GPU=3
bash local_scripts/remote_commands/shenzhen_fastwam_fw_sz_400_render_expert_gate_20260822.sh
```

PPO 可以继续使用 physical 4–7；Fast-WAM 只在现场确认所选卡空闲后使用该卡。两者共享主存，所以启动前
记录 `free -h`，运行中以 2 秒采样记录 host/GPU peak；不因 Ray 本身存在而拒绝启动，也不停止外部进程。

`FW-SZ-300/310`先确认release与ModelScope inputs，S5没有重复哈希。S5的2秒资源记录得到peak GPU
`30,274 MiB`、peak host used `1,635,090,612 KiB`；PPO继续使用physical 4–7，Fast-WAM使用physical 3。

两个脚本都要求新 run id 的真实 output 不存在；S4/S5 owned timeout 分别是 3600/7200 秒。停止条件只保留：source/input 不符、GPU 或 RAM baseline 不满足、official command 非零/超时、render 明确失败、核心 result/config/log/video 不齐。失败产物保留，不在同一 run id 上续写。

## 4. 通过标准

| 阶段 | 最小闭环 |
|---|---|
| S4 render | `Render Well` 且无 `Render Error` |
| S4 expert | `PARTIAL / FAIL`；MPLib `Box(...)` segfault，保留原始失败，不把render PASS冒充stage PASS |
| S5 pre-run | `PASS`；resolved config、exact command、output absence、已验收输入路径、GPU/RAM snapshot齐全 |
| S5 result | `PASS`；single rc 0；config/log/result/video齐全；result=1.0、success-true、accepted seed=4300001 |

本次单episode实际成功，只证明accepted seed 4300001的一次成功轨迹，不外推稳定成功率。完整逐命令和产物路径见
[`evidence/SHENZHEN_STANDALONE_OPERATION_LEDGER_20260822.md`](evidence/SHENZHEN_STANDALONE_OPERATION_LEDGER_20260822.md)。
