# Fast-WAM × current RLinf GRPO 实施与 smoke 流水账

日期：2026-08-31  
授权：用户已明确授权实现、推送 Git、必要检查，并在空闲 GPU 上执行真实 smoke。  
目标：official Fast-WAM `7faa711...` × 深圳 current RLinf，`move_stapler_pad`，首条 formal 候选为
两卡 `32 env × rollout4 / G8 / 128 trajectories`；本轮只实现并完成一次目标并发 smoke。

## 操作记录

### 2026-08-31 — 开始前上下文与边界

- 完整读取根 `PROJECT_CONTEXT.md`、`HANDOFF.md` 与专题单一事实源
  `12_SHENZHEN_CURRENT_RLINF_GRPO_PORT_PLAN_20260831.md`。
- 实施边界锁定：独立 branch/worktree/joint runtime；不改公共 dirty tree，不停止现有训练或其他用户进程；
  Fast-WAM 模型专属字段保留 `H32/C24/M10/D14`、192 actions、action-expert only、FP32 denoise chain；
  current 通用 schema/Builder/EnvWorker/GRPO advantage/loss/runner/checkpoint manager 不重写。
- smoke 已由用户授权，精确口径沿已审阅方案：两卡、`32 train env × rollout1 / G8`、192 actions、
  32 trajectories、256 query records、`GB256/MB2/update1`、fixed32、DCP save/reload。
- 下一步：用固定 host-key Paramiko 身份探针和只读现场检查刷新 GPU/RAM/磁盘、运行任务、source/runtime/checkpoint。

### 2026-08-31 17:36 CST — 深圳只读现场刷新

- 固定 host-key Paramiko 以 `chenyiteng` 身份认证成功；远端为 `admin`，UID 1003。
- GPU4--7运行本用户两条现役两卡PPO；GPU1为`liwenbo`任务；GPU0仅少量上下文占用，GPU2/3完全空闲。
  本轮smoke候选锁定physical GPU2/3，不触碰GPU1或GPU4--7。
- 主机约2.0 TiB RAM、available约735 GiB、PSI为0；`/`、`/home`、`/data`分别余226 GiB、1.4 TiB、
  1.2 TiB。shared Ray `172.17.0.1:6389`健康，当前没有Ray资源声明积压。
- current RLinf source lock为clean detached `7d07a421...`；两卡GRPO recipe worktree为`554c6dc8...`。
  official Fast-WAM oracle为`7faa711...`，有1项既有dirty改动，保持只读，不在其上开发。
- runtime确认：current RLinf Python3.11/Torch2.11+cu129；official standalone Python3.10/Torch2.7.1+cu128。
- 本地从`554c6dc8...`建立独立分支/worktree：
  `codex/sz-fastwam-current-rlinf-grpo` / `worktrees/fastwam-current-grpo`；初始clean。

### 2026-08-31 17:45--18:15 CST — current 接口映射与首批实现

- 从服务器只读下载 official `7faa711...` 的8个小型源码/config文件到本地 reference；未复制checkpoint或数据。
- 精确确认 current Fast-WAM 的训练底层为
  `video_expert.prepare -> prefill_video_cache_tensor -> action_expert.prepare ->
  forward_action_with_video_cache_tensor -> action_expert.post`；public `infer_action()` 的B=1限制仅在部署编排层。
- 精确确认 release oracle使用显式 `sigma_shift=5.0`；current源码scheduler默认值不能代替release运行合同。
- 从旧`768e0243...`提取五个模型专属文件后窄改：保留三相机/14D/H32-C24-M10、Flow-SDE、
  完整逻辑batch RNG、chain/k replay与action-only冻结；重写dict KV为current tensor KV；删去首版GRPO
  不需要的PPO value-head/cache pooling/export路径。
- current RLinf只增加三个默认关闭的小接点：注册`fastwam`、capability式rollout mode透传、
  `fsdp_config.cast_forward_inputs`可配置；同时复用已验证的`local_shard` checkpoint format透传修复。
- current RLinf Python3.11/Torch2.11环境加official Fast-WAM source `PYTHONPATH`后，Fast-WAM核心模块与
  `diffusers/transformers/cv2`均直接import成功；没有另建或混用joint venv。后续真实builder首次触发
  official `fastwam.runtime`时发现该venv缺其纯Python运行依赖`boto3`，见下一节的窄补充。
- 本地五个Fast-WAM文件`py_compile`通过，当前diff `git diff --check`通过。下一步补薄配置并送服务器
  做真实模型构建/ratio/梯度检查。

### 2026-08-31 18:10--18:24 CST — server compose、运行依赖与真实模型检查

- 新增三份薄配置：Fast-WAM模型合同、`move_stapler_pad`环境、两卡GRPO主配置。服务器真实Hydra compose
  exit 0；smoke override解析为`32 env × rollout1 / G8 / 32 trajectories / GB256 / MB2 / update1`，
  actor与rollout均为`H32/C24/M10/D14`、`sigma_shift=5.0`、`noise_level=0.3`；FSDP为
  `local_shard + cast_forward_inputs=false`，train/eval产物均为run-scoped绝对路径。
- 第一次真实builder在导入official `fastwam.runtime`时立即报`ModuleNotFoundError: boto3`，尚未加载模型、
  未启动Ray或环境。只读比对official Fast-WAM venv后，将其精确纯Python版本
  `boto3/botocore/s3transfer/jmespath=1.35.99/1.35.99/0.10.4/1.0.1`补入current RLinf venv；没有改动
  Torch/CUDA/Ray/RoboTwin等现役依赖。
- 第二次builder因未继承standalone cache环境而尝试下载并报缺`modelscope`；未下载任何模型。窄修运行环境为
  既有official oracle变量：`MODELSCOPE_CACHE=/home/chenyiteng/cache/fastwam-7faa/modelscope`、
  `DIFFSYNTH_MODEL_BASE_PATH=/data/chenyiteng/models/fastwam/diffsynth`、source=`modelscope`，直接复用已完成资产。
- GPU2真实模型检查exit 0：official 7faa构建与12 GB release checkpoint加载成功；参数树无alias重复、
  仅action expert可训练（824 tensors）；wrapper deterministic B=1与official `infer_action`最大绝对差为0；
  stochastic behavior→replay logprob最大绝对差为0；action expert梯度有限且非零，冻结参数无梯度。
  单模型检查CUDA峰值allocated/reserved=`25.142/25.799 GiB`。
- 下一步：等待最终静态复核，提交并普通push；随后在physical GPU2/3启动唯一1-step真实smoke。

### 2026-08-31 — 提交、推送与首次真实 smoke

- 实现提交并普通 push 到 `Yutenji-Nyamu/rlinf_fastwam`：
  `81076b13e91a3db0b8def6b48dd6e1db2897cb2b`（`feat(embodiment): add current Fast-WAM RoboTwin GRPO`）。
  初始增量为12 files、`+2153/-2`；远端branch为`codex/sz-fastwam-current-rlinf-grpo`。
- physical GPU2/3 的首轮32-env真实smoke完成全部rollout（32 trajectories，约4分53秒），随后在第一个actor
  microbatch的FSDP all-gather处OOM。现场为actor约59.48 GiB、EnvWorker约18.83 GiB，只差80 MiB；不是
  Fast-WAM forward、Flow-SDE、GRPO数值或主机内存故障。
- current `EnvWorker`实际读取`env.train.enable_offload`和`env.eval.enable_offload`；顶层
  `env.enable_offload=true`不会替代两个嵌套叶。只将两个嵌套叶设为true，保持模型、采样、batch和优化预算不变。
  修复提交并push：`f730aff3ab5e11a584c5c8afa74e90f5fb8ffebd`（`fix(embodiment): offload RoboTwin envs for Fast-WAM`）。

### 2026-08-31 — env-offload v2 一步训练闭环

- 同一GPU2/3、32 env、G8、32 trajectories、256 query records、GB256/MB2/update1重新运行；rollout约5分35秒，
  actor update与fixed32评估均完成，进程自然exit 0。train=`13/32`，fixed32=`11/32`；actor grad norm
  `9.357`、policy loss `-0.0011`，均有限。GPU峰约62.15 GiB/卡，未影响GPU4--7任务。
- `global_step_1`确有两份约14.45 GB的rank文件；但fresh-process重载暴露通用格式缺口：Fast-WAM使用
  FSDP2，`local_shard`把DTensor降为裸local Tensor，加载时又直接写回global-shape DTensor，故精确出现
  两卡半宽shape mismatch（如2048↔4096、16↔32）。这是checkpoint格式问题，不是模型参数树或训练数学问题。
- 进一步审计确认optimizer也绕开PyTorch canonical FQN分布式state-dict合同；因此不把“文件写成”误报为
  strict resume成功，也不在Fast-WAM内加特判。PyTorch FSDP2的正式路径是DTensor-aware的
  `get_state_dict/set_state_dict + DCP`：
  [FSDP2教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)、
  [DCP文档](https://docs.pytorch.org/docs/2.11/distributed.checkpoint.html)。

### 2026-08-31 — DCP strict-roundtrip

- 只把checkpoint格式改回`dcp`；FSDP2、wrap、action-only optimizer、模型与全部训练预算不变。这也与旧AutoDL
  Fast-WAM的`FSDP2 + default DCP`路径一致。
- v3曾尝试把评估间隔改为100、保存间隔保持1，以减少重复fixed32；完整rollout完成后由current runner的
  既有配置约束立即拒绝：`save_interval=1 must be divisible by val_check_interval=100`。这是配置断言，
  尚未进入update/save，不是模型、数值或DCP故障；GPU2/3随后释放。
- v4恢复已经跑通的`eval1/save1`，闭环为：fresh Step1 rollout/update/fixed32 → DCP save → 全新进程
  从Step1加载 → 再完成Step2 rollout/update/fixed32并保存DCP。packet：
  `/data/chenyiteng/results/rlinf-shenzhen/fastwam-grpo/packets/fastwam-grpo-smoke1-current-2gpu32x1-g8-b256-u1-m10-phys23-dcp-v4-roundtrip`。
- 12:01 UTC在空闲physical GPU2/3启动；GPU4--7两条PPO未停止或改动。
- fresh Step1已完成32/32 rollout（约5分51秒）、一次真实actor update与fixed32；`grad_norm=26.414`、
  `policy_loss=7.15e-04`，均有限。DCP随后写出`.metadata`和两份约14.45 GB的rank shard，fresh进程
  `exit0`并释放GPU2/3。全新reload进程已明确从`global_step_1`恢复并进入Step2 rollout，未再出现
  local-shard的半宽shape mismatch。
- reload Step2又完成32/32 rollout（约5分34秒）、真实actor update、fixed32和第二次DCP保存；
  `grad_norm=16.908`、`policy_loss=-7.60e-04`，均有限，进程`exit0`。Step2同样具有`.metadata`和两份
  与Step1尺寸一致的约14.45 GB shard。fresh/reload的train success分别为`5/32`、`10/32`，fixed32
  分别为`13/32`、`17/32`；这些仅用于证明真实执行链，不作为两步学习结论。
- 两轮峰值约62.5 GiB/卡；GPU2/3终态为0 MiB任务占用，GPU4--7两条现役PPO全程未停止。外层首次未写
  `SMOKE_OK`只是脚本误要求current runner并不打印的`Training finished!`文案；按真实条件重新验收：
  两次`exit0`、明确resume、Step2完成、两代DCP完整、无traceback/OOM，最终标记
  `FASTWAM_CURRENT_GRPO_DCP_STRICT_ROUNDTRIP_OK`。

### 2026-08-31 — 最终提交与交接

- checkpoint配置的唯一变化为`local_shard -> dcp`；FSDP2、模型、采样、batch、优化器和任务参数均未改。
- 最终修复提交并push：`7b2331c55d14397cfb4cb16181470ddc8afae44a`
  （`fix(embodiment): use DCP for Fast-WAM FSDP2`）。server HEAD、remote HEAD一致，worktree clean。
- 本轮三个提交依次为：`81076b13...`（current adapter）、`f730aff3...`（嵌套env offload）、
  `7b2331c5...`（FSDP2 DCP）。formal没有启动；首条候选仍为两卡
  `32 env × rollout4 / G8 / 128 trajectories / 1024 query records / GB1024/MB2/update2`。
