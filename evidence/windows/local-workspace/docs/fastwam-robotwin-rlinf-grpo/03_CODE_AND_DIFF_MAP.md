# 代码与 Diff 映射

更新时间：2026-07-17

## 1. 目标代码承载方式

当前首选把适配实现作为 RLinf 内置 embodied model，而不是另建外置扩展包：

```text
rlinf/models/embodiment/fastwam/
├── __init__.py              # get_model()；只编排 builder
├── builder.py               # 官方 config/checkpoint/stats、模型树规范化、冻结范围
├── robotwin_adapter.py      # 官方三相机、14D qpos、prompt、norm/denorm
├── fastwam_rl.py            # batch conditioning、ODE/SDE 共核、replay 与 FP32 概率
├── fastwam_policy.py        # BasePolicy mode 分流、RolloutResult/default_forward
└── export.py                # pin原生no-dist DCP model-state → 官方 deploy payload

tests/models/embodiment/fastwam/
├── test_adapter_builder_parity.py      # 一个组合fixture覆盖adapter、构建、模块树
├── test_flow_sde_replay.py             # deterministic/batch/schedule/old-new组合数值检查
└── test_checkpoint_export.py           # 与真实runner的resume/export验收配合

RLinf Fast-WAM 功能分支另外只需要：

rlinf/models/__init__.py
examples/embodiment/config/model/fastwam_robotwin.yaml
examples/embodiment/config/robotwin_<task>_grpo_fastwam_a800_2gpu_smoke.yaml
rlinf/workers/rollout/hf/huggingface_worker.py  # 当前pin必需的opt-in capability shim
```

选择内置路径的依据不是“文件少”：

1. RLinf `new_model_fsdp.rst` 把新 embodied model 放在 `rlinf/models/embodiment/...`，并通过 `register_model` 接入。
2. π0、社区 Fast-WAM、Motus 和 LaWAM 的项目实现都采用该模型内置布局，来源映射最直接。
3. 本项目会维护 RLinf 独立 worktree/分支，而不是把 RLinf 当不可修改的第三方 wheel；因此无需为 driver/Worker 再维护 `extension.py`、专用 launcher 和 `RLINF_EXT_MODULE` 生命周期。
4. 模型本体仍从固定 commit 的独立 Fast-WAM 仓库导入；“内置 RLinf adapter”不等于复制 Fast-WAM 模型源码。

`RLINF_EXT_MODULE` 仍是 RLinf 作为依赖库时的官方方案，但不是本项目首版目标。内置和外置不并行维护。第一版也不新增通用`fsdp.py`、`checkpoint.py`或第二套runner：FSDP2、bucket sync、GRPO loss和DCP继续使用RLinf原实现。FSDP禁用block/expert wrap并shard root，但保留pin对非tied Embedding的既有单独wrap；不把它误写成纯root-only。

| 目标文件 | 第一版实现内容 | 直接来源 |
|---|---|---|
| `rlinf/models/__init__.py` | 延迟导入 Fast-WAM `get_model`，注册唯一 `model_type="fastwam_robotwin"` | RLinf `6d0db56::_register_builtin_models/register_model` |
| `fastwam/__init__.py` | `get_model(cfg, torch_dtype)`，只转交 builder | RLinf 内置 embodied model 约定 |
| `builder.py` | compose 官方 config、加载 checkpoint/stats、canonicalize alias、冻结后只开放 action expert、维持冻结 conditioner 的 eval mode；PPO才创建顶层BF16 value head | 官方 Fast-WAM loader/checkpoint + RLinf FSDP/sync/ValueHead 枚举 |
| `robotwin_adapter.py` | 精确 PIL 三相机 resize/拼图、官方 prompt、14D absolute qpos state norm/action denorm、严格 shape 断言 | 官方 RoboTwin deploy + RLinf pin env obs |
| `fastwam_rl.py` | batch conditioning、官方 shifted schedule、同一个参数化 ODE/SDE loop、完整 action chain、behavior old logprob、actor 同 transition 重算；PPO可从last video-cache V返回observation feature | 官方 Fast-WAM scheduler/model/MoT cache + RLinf OpenPI Flow-SDE + 社区 Fast-WAM 拼接实现 |
| `fastwam_policy.py` | 显式`forward`、`predict_action_batch(train/eval)`、physical/model action 分离、Tensor-only replay、`default_forward`、冻结模块mode；PPO old/new/bootstrap value共用conditioning | RLinf BasePolicy/OpenPI/π0 PPO + 社区同名 policy + Motus 后期 wrapper |
| `export.py` | 复用pin `no_dist=True` DCP抽取、严格canonical prefix/schema、构造官方 `mot`/`proprio_encoder` payload；deploy排除PPO head | RLinf `convert_dcp_to_pt.py` + 官方 `FastWAM.save_checkpoint/load_checkpoint` |
| model YAML | H/N/S、stats/checkpoint、官方文本、`eval_seed=0`、shift参数、action-only trainables；固定eval复用官方singleton latent并沿B broadcast，train独立随机 | standalone resolved config + 官方sim/task config；mode capability是policy类属性，不属于YAML |
| smoke YAML | 复制 π0 RoboTwin/GRPO/placement 生命周期，只替换模型、N、FSDP、bucket transport、已冻结P2资源表和输出路径；本机真实runner使用CPU-staged bucket | π0 100-step resolved config、本机CUDA IPC错误链 |

## 2. 参考代码到目标模块

| 参考 | 重点文件/子图 | 提供什么 | 目标动作 |
|---|---|---|---|
| Fast-WAM 官方 `45d8e145` | `src/fastwam/models/wan22/fastwam.py` | prompt/proprio、VAE/video cache、action velocity、官方 singleton inference | 模型真相；只 batch 化 orchestration，并写 grad-enabled velocity twin |
| Fast-WAM 官方 `45d8e145` | `scheduler_continuous.py`、`configs/model/fastwam.yaml` | shifted schedule、raw timestep、signed delta、默认 infer shift | rollout/replay 共用同一 resolved schedule |
| Fast-WAM 官方 `45d8e145` | `experiments/robotwin/fastwam_policy/deploy_policy.py/yml` | 三相机、qpos、queue、replan、stats | 改写为 batch-native adapter，逐项 parity |
| RLinf server pin `6d0db56` | `rlinf/models/__init__.py`、`new_model_fsdp.rst` | 内置模型目录和 registry | 增加一个延迟 builder 与一条注册；不加旁路 launcher |
| RLinf server pin `6d0db56` | `openpi/openpi_action_model.py::sample_mean_var_val` | 单 stochastic transition、首步 sigma 分母、mean/std、真实 replay | Flow-SDE 概率第一权威；适配 Fast-WAM 的 shifted/signed schedule |
| RLinf server pin `6d0db56` | `base_policy.py`、`fsdp_actor_worker.py` | rollout/actor contract、replay forward、loss、DCP | 原样复用 |
| RLinf server pin `6d0db56` | `huggingface_worker.py::predict/get_bootstrap_values` | 硬编码 train/eval mode 分发、`hasattr(value_head)`行为 | capability shim默认关闭；GRPO不创建head，PPO actor/rollout两侧都创建同构head并复用bootstrap |
| RLinf server pin `6d0db56` | `openpi_action_model.py`、`ValueHead`、FSDP optimizer/GAE/PPO loss | observation/action critic范式、独立value LR、old/new/bootstrap和clipped value loss | PPO复用通用链；只把Fast-WAM feature适配为last video-cache V mean |
| RLinf server pin `6d0db56` | `fsdp/utils.py`、`weight_syncer/*` | root/Embedding wrap、bucket device能力、名称枚举 | `no_split_names=["__none__"]`禁block/expert wrap；保留默认Embedding wrap；当前用RLinf原生CPU-staged bucket；只从canonical action subtree取trainables；保存wrap inventory |
| RLinf server pin `6d0db56` | `envs/robotwin/*` | obs、qpos action、reward、group/reset | 原样复用，不迁移 LaWAM EE planner |
| 社区 Fast-WAM `fd780f0` | `fastwam_rl.py` | Fast-WAM scheduler 与 RLinf Flow-SDE 的已跑通拼接、batch replay | 保留分工和共核；修复漏传 `sigma_shift`、硬 clamp 和 fallback |
| 社区 Fast-WAM `fd780f0` | `fastwam_policy.py/__init__.py` | batched policy、root FSDP、bucket sync | 替换LIBERO adapter；交叉核对RLinf pin的Embedding wrap；不复制catch-all singleton fallback |
| Motus `7a5296a3` | `motus_policy.py`、model-side replay、FSDP/sync/checkpoint | WAM→RLinf 的薄 wrapper、Tensor-only replay、共用概率函数 | 复用工程边界；不复制遗留 singleton helper、critic/debug chain |
| LaWAM `481380fb` | policy、flow replay、value/dtype、RoboTwin planner | dtype/注册/batch/环境失败边界 | 只用教训；不引入 endpose/EE/planner 数据流 |

## 3. 符号级来源账

实现前先锁定 provenance。若目标实现偏离第一来源，diff 必须标成“RoboTwin 适配”“RLinf 兼容性修复”“batch 编排适配”或“实验变量”，并记录理由与集中验收。

| 目标 symbol | 精确第一来源 | 必要适配，不得擅改 | 集中验收 |
|---|---|---|---|
| `_build_fastwam()/register_model(...)` | RLinf `6d0db56:rlinf/models/__init__.py::_register_builtin_models` | 延迟导入 `fastwam.get_model`；沿用 registry，不增加 Fast-WAM 专用 runner/枚举分支 | P1/P2 |
| `get_model()/build_fastwam_policy()` | 官方 `45d8e145:runtime.py::create_fastwam`、`FastWAM.load_checkpoint`；社区 `fd780f0:fastwam/__init__.py::get_model` | compose官方RoboTwin task；与deploy一致强制加载text encoder；显式checkpoint/stats；不复制模型代码、不强制HF源 | P1 |
| `validate_env_obs()` | RLinf `6d0db56:robotwin_env.py::_extract_obs_image` + RoboTwin `0008ae:vector_env.py` | 要求main/wrist/state/text完整；`wrist[:,0]=left,[:,1]=right`；14D有限state；不猜shape或补数据 | P1 |
| `compose_three_camera_image()` | 官方 `deploy_policy.py::_resize_rgb/_build_robotwin_image_tensor` | 使用PIL bilinear；head W320/H256、左右各W160/H128；BCHW；按官方顺序cast后`[-1,1]`；扩展真实batch | P1 |
| `normalize_proprio()/denormalize_actions()/build_prompts()` | 官方 deploy `_normalize_state/_denormalize_action/_infer_action_chunk` +官方`DEFAULT_PROMPT` | 调官方transform/normalizer；singleton key`default`、14D absolute qpos；无π0 delta、LIBERO gripper或LaWAM EE转换 | P1 |
| `prepare_initial_action_latents()` | 官方 `fastwam.py::infer_action` 中 action noise 构造 | 从写死`(1,H,A)`抽成`[B,H,A]`；目标core允许显式latent，官方oracle则用同seed/rand_device重建或捕获对应latent；不改变生产RNG独立性 | P1 |
| `prepare_rollout_randomness()` | RLinf OpenPI共享k语义 + 官方Fast-WAM initial noise构造 | 在逻辑rollout batch分块前一次生成共享`k`、每样本initial latent与SDE epsilon；所有固定子批只取slice | P1 |
| `encode_first_frame_latents()` | 官方 `WanVideoVAE.encode` + `infer_action::_encode_input_image_latents_tensor` | 首版传B个video的list；官方wrapper自己逐项encode并返回stack，目标不二次stack；不绕过wrapper，也不异常后singleton fallback | P1 |
| `build_action_conditioning()` | 官方 `encode_prompt/_append_proprio_to_context/video_expert.pre_dit/mot.prefill_video_cache`；社区同名函数 | prompt直接按list batch；image/context/proprio/cache B先验一致；固定H/图像token长度；PPO opt-in严格池化最后一层cache V，GRPO默认不做 | P1/PPO |
| `resolve_action_schedule()` | 官方 `scheduler_continuous.py::build_inference_schedule`、官方 resolved `sigma_shift` | 必须传`shift_override=sigma_shift`；记录configured/effective shift、raw timestep、normalized`t`、signed`delta` | P1 |
| `predict_action_velocity()` | 官方 `fastwam.py::_predict_action_noise_with_cache`；社区 `_predict_action_velocity` | 调用图逐行同构并去掉官方`@torch.no_grad()`；RL chain为FP32时边界显式cast到model dtype，velocity返回后upcast FP32；eval在外层no-grad | P1 |
| `flow_step_mean_std()` | RLinf `6d0db56:openpi_action_model.py::sample_mean_var_val` + 官方 Fast-WAM scheduler | 断言`delta<0`后令`dt=-delta`；首步分母用下一schedule点；不硬编码`0.98`，不以`abs(delta)`掩盖反向网格 | P1 |
| `gaussian_logprob()/gaussian_entropy()` | RLinf OpenPI/现有 embodied flow policy 的逐元素 Gaussian 定义 | 概率用FP32；第一版std非正/非有限立即失败，不用floor改写density | P1 |
| `flow_sde_rollout(deterministic=...)` | 社区 `fd780f0::flow_sde_rollout` + RLinf OpenPI `_sample_actions_with_prefix_cache` | 一个loop；eval全ODE，train只在共享`k`采样；每个样本initial/SDE noise独立；不保留`simple/full`生产分支 | P1/P2 |
| `recompute_logprob()` | 社区同名函数 + RLinf OpenPI actor replay + Motus 后期 model-side replay | gather真实`x_k/x_(k+1)`；每样本可有不同k；复用同一schedule/conditioning/velocity/mean-std；PPO可同时返回同一conditioning的observation feature | P1/P2/PPO |
| `FastWAMPolicy.forward/predict_action_batch/default_forward/train` | RLinf `6d0db56:base_policy.py/OpenPI PPO`；社区 Fast-WAM policy；Motus `7a5296a3` wrapper | MRO要求显式forward；rollout `mode=train`与actor `.train()`分离；标准physical/model action与Tensor replay；PPO才注册head并返回old/new value；不捕获异常退回singleton；actor train后冻结conditioner仍eval | P1/P2/PPO |
| `_pool_observation_value_features()/ValueHead` | 官方 `mot.py::prefill_video_cache`的`v [B,Sv,H*Dh]`；RLinf π0.5/Motus observation critic；社区Fast-WAM raw-video critic | 取最后层V token mean而非raw pre-DiT或action hidden；input detach；head跟随BF16模型，输出FP32；Flow-SDE零改 | PPO smoke/formal |
| `rlinf_accepts_rollout_mode` + worker `elif` | RLinf `6d0db56:huggingface_worker.py::predict`；社区/Motus/LaWAM硬编码mode分支 | 当前pin必要兼容改动；Fast-WAM opt-in且保留sampling kwargs/DAgger语义；既有硬编码分支不变 | P1/P2 |
| `canonicalize_fastwam_module_tree()` | 官方 alias 树与 RLinf FSDP/sync 冲突推导 | 明确标为兼容性修复；parameter object/state-dict/optimizer/sync唯一且action parity | P1/P2 |
| `load_rlinf_model_state()/extract_official_fastwam_payload()/export_deploy_checkpoint()` | RLinf `6d0db56:convert_dcp_to_pt.py` + 官方 `FastWAM.save_checkpoint/load_checkpoint` | `no_dist=True`读取`fsdp_checkpoint.model`；只接受冻结后的canonical prefix；exact schema后回到官方payload | P3 |

官方 `infer_action()` 只作为 B=1 数值 oracle，因为它写死 image/proprio/noise 的 B=1、返回 `[0]`，并在入口调用 `eval()`/no-grad。生产 B=1/B>1 eval、train rollout 和 actor replay都走上述同一个 batch core。

dtype边界也属于有依据的适配：官方deterministic latent始终是model dtype；Flow-SDE为保存连续Gaussian transition使用FP32 chain。目标仍用一个loop和一个velocity函数，但`deterministic=True`保留官方model-dtype state并调用官方`scheduler.step()`，`False`与actor replay才把chain/mean/std保持FP32。这样既不量化真实SDE样本，也不让FP32 latent未经显式cast进入BF16 action expert。

## 4. Batch 化的边界

Fast-WAM 模型底层已经保留 batch 维，真正需要改的是 singleton orchestration：

```text
[B 条 prompt] ─→ 官方 encode_prompt
[B,14] state ──→ 官方 proprio encoder/token append
[B,3,384,320] ─→ 官方 VAE wrapper（内部逐样本编码并 stack）
                         ↓
          一次 video pre-DiT + batched KV cache
                         ↓
          S 次 batched action velocity/Flow step
                         ↓
                    [B,32,14]
```

首版允许配置固定`model_forward_batch_size`做显存分块；这是同一batch core的资源分块，不是异常fallback。逻辑rollout batch的共享`k`、每样本initial latent和SDE epsilon必须在分块外一次生成，再按slice传给各chunk；chunk内部不得重新采样。任何分块失败都直接停止。后续只有在profiling证明VAE wrapper是瓶颈、且B=1/B>1 parity通过后，才考虑社区直接调用内部batched VAE的优化。

必须固定的 batch 约束：

- 所有 action/video timestep 显式构造 `[B]`，因为训练态 B>1 不接受 singleton timestep。
- 同一 batch 共享 resize 后分辨率、video token length、action horizon 和 rollout `k`；trajectory flatten 后 actor batch允许每样本不同 `k`。
- prompt list、proprio、image、text mask、KV cache 的 B 在 wrapper 边界 fail-fast。
- train生产路径每样本initial latent和SDE noise独立；固定eval使用`eval_seed=0`生成一个官方singleton latent并沿B broadcast，以同时满足B=1 oracle和batch置换等价；该broadcast仅用于eval。
- 相同输入与预生成随机Tensor在unchunked和不同固定分块下必须给出同一`k/chain/action/logprob`（容差内）；batch partition不能成为实验变量。
- 不复制缺失 wrist、不补空 prompt、不用 catch-all exception 逐 env 继续。

## 5. 社区分支必须按提交链阅读

固定范围：`5d75412a..fd780f0`。核心演化：

```text
00db686  Fast-WAM LIBERO eval + SFT
765b7ec  PPO、Flow-SDE、value head
dd4c022  critic-free GRPO、batch
0f0fcf4  actor 侧重算 old logprob 的错误实验方案
26d0e94  t→1 sigma hard clamp
9654a5a  恢复 behavior old logprob、单 stochastic step
f187152  eval 不保存 replay tensor，修 OOM
93284b3  per-rollout debug report
f3d3673  no-std GRPO、bucket sync、eval 修复
fd780f0  LIBERO-130 泛化报告
```

最终代码也不能整段复制：它漏传 scheduler `sigma_shift`，用 `t<=0.98` 代替 RLinf 首步规则，保留 `simple/full` 和 catch-all singleton fallback；proprio 虽在 allowlist，actor replay 却不重跑 encoder，实际没有 RL 梯度。目标首版 action-only、single-step Flow-SDE、fail-fast。

## 6. 目标调用链与文件归属

```text
官方 RLinf train_embodied_agent.py + Hydra config
  → rlinf.models registry → fastwam.get_model()
  → FastWAMPolicy.predict_action_batch(env_obs, mode)
      → robotwin_adapter.py
      → fastwam_rl.py::flow_sde_rollout(deterministic = mode != "train")
        eval: 同一 loop 全部走 ODE，不保存 replay
        train: 同一 loop 仅在选中 k 采样，保存 chain + behavior old logprob
  → RLinf RolloutResult / RoboTwin chunk_step / Trajectory             [原样]
  → FastWAMPolicy.default_forward
      → fastwam_rl.py::recompute_logprob
  → FSDP actor / chunk-level GRPO / optimizer                         [原样]
  → bucket actor→rollout sync / DCP                                   [原样]
```

文件职责保持单向：adapter 不算概率，`fastwam_rl.py` 不做 RoboTwin 预处理，policy 不重写 GRPO loss，builder 只构建/规范化，env 不出现 Fast-WAM 特判；但 eval/train 的 conditioning、schedule、velocity 和 denoise loop 只能有一套。

## 7. 最小 RLinf 核心 diff

第一处是按内置模型既有模式注册：

```diff
--- a/rlinf/models/__init__.py
+++ b/rlinf/models/__init__.py
@@
+ def _build_fastwam(cfg, torch_dtype):
+     from rlinf.models.embodiment.fastwam import get_model
+     return get_model(cfg, torch_dtype)
@@
+ register_model("fastwam_robotwin", _build_fastwam, category="embodied", force=True)
```

这里的 `force=True` 是 `_register_builtin_models()` 对全部内置模型的现有惯例，不是运行时覆盖第三方注册；若现场 pin 的注册结构不同，则严格按该 pin 的内置写法调整。

第二处在当前 server pin 必须增加，因为现场源码已确认worker不能把 `mode` 传给通用opt-in policy：

```diff
--- a/rlinf/workers/rollout/hf/huggingface_worker.py
+++ b/rlinf/workers/rollout/hf/huggingface_worker.py
@@
+ if getattr(self.hf_model, "rlinf_accepts_rollout_mode", False):
+     kwargs = dict(kwargs)
+     loss_type = self.algorithm_cfg.get("loss_type", "actor")
+     kwargs["mode"] = "eval" if loss_type == "embodied_dagger" else mode
```

具体插入位置必须保留worker已选的train/eval sampling params、DAgger强制eval和所有内置旧模型语义；不能以`kwargs={"mode":...}`覆盖原字典。默认capability不存在/为false时走原逻辑。只有实施时目标代码已含等价机制才省略；禁止monkeypatch worker或伪装成π0类型。

## 8. Builder 的必要兼容性处理

builder先以官方`sim_robotwin.yaml`和锁定task compose模型配置，像standalone deploy一样令`load_text_encoder=True`，调用官方runtime、`load_checkpoint`、`load_dataset_stats_from_json`和`processor.set_normalizer_from_stats`；checkpoint/stats必须显式存在。官方processor没有`load_dataset_stats()`。它不修改下载源，不允许随机模型，也不让RLinf YAML静默覆盖官方H/D/scheduler/camera契约。

完整改动面实际为12个新增文件和3个小改，共15个；新增的第15个文件是registry/worker/FSDP wiring回归测试。逐文件上下游、符号与完成定义统一见[05_IMPLEMENTATION_PLAN.md](05_IMPLEMENTATION_PLAN.md)第25节。除registry、worker shim和FSDP2的通用`cast_forward_inputs`开关外，RLinf核心零diff；该开关默认`true`，所以旧模型行为不变。

官方对象把同一专家经 `video_expert`、`action_expert`、`mot`，以及 `dit=mot` 多路径注册；RLinf 同步可能使用 `named_parameters(remove_duplicate=False)`。builder 在官方 checkpoint 加载后、FSDP 前：

1. 以 `mot` 作为唯一注册模块树。
2. `video_expert/action_expert/dit` 只保留非注册兼容访问。
3. 冻结全模型，再只开放 canonical action expert。
4. 断言 parameter object、state-dict、optimizer 和 sync 名称唯一。
5. 变换前后用黄金 fixture 做固定 latent action parity。
6. policy进入train mode后重新把冻结的video/proprio/VAE/T5 conditioner设为eval，仅action expert保持train；避免无梯度模块的dropout/mode漂移污染old/new parity。

这属于 RLinf 兼容性修复，不改 Fast-WAM 数学。非注册alias的具体实现以真实对象树验证为准；不能同时满足模块唯一性和action parity时修正该兼容层，不以保留重复注册或关闭同步绕过。

## 9. 零回归分支布局

```text
保留：/root/autodl-tmp/RLinf
      local/openpi-a800-2gpu-migration  # π0 known-good

新增：独立 git worktree
      feat/fastwam-robotwin-grpo
      rlinf/models/embodiment/fastwam/
      独立 Fast-WAM/RLinf 联合环境
      独立日志与 checkpoint 根目录
```

联合环境以 π0 已跑通清单为基线，但在最终 worktree 路径重新创建；最终 Torch 采用 Fast-WAM `2.7.1+cu128`，之后重编 CuRobo/CUDA extension。既不修改现役 π0 venv，也不污染已跑通的 `FastWAM-official` oracle 环境。

## 10. 实施时保存的最小 diff 证据

1. `server-working-tree-vs-6d0db56.diffstat` 和精确文件列表。
2. `community-fastwam-5d75412a-to-fd780f0.diffstat`，按 Fast-WAM 子图和全局改动分组。
3. `our-fastwam-branch-vs-server-pin.diffstat`，必须能说明 π0 路径/YAML 未改。
