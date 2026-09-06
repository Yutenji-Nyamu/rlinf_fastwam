# 接口契约

## 1. Fast-WAM standalone 黄金契约

2026-07-17 服务器成功 run 的 resolved config 已只读确认以下契约；后续黄金 fixture只用于数值 parity，不再决定这些 shape：

| 项目 | 契约 |
|---|---|
| 相机 | head/main + left wrist + right wrist，共三路 |
| 官方预处理 | PIL bilinear：head W=320/H=256、left/right wrist各W=160/H=128；左右横拼后置于head下，组成H=384/W=320输入，再按官方顺序cast到model dtype并缩放至`[-1,1]` |
| state | 14D absolute qpos，顺序为left arm6、left gripper1、right arm6、right gripper1；使用checkpoint配套dataset stats归一化 |
| action | normalized 14D absolute qpos chunk；官方denorm后原样交RoboTwin qpos，不做delta、gripper或EE转换 |
| action horizon | 32 |
| denoise | 10-step shifted flow Euler |
| replan/executed prefix | 24 |
| scheduler infer shift | 5.0；本次 `sigma_shift=null`，因此使用 scheduler 自身值 |
| eval | deterministic 路径应与官方 adapter 完全一致 |

必须保存一个小型黄金 fixture：三张图、14D state、task text、随机 seed、resolved config 摘要，以及官方 normalized/physical action。fixture 不包含大模型权重。

## 2. RLinf policy 契约

Fast-WAM wrapper 继承 `nn.Module, BasePolicy`。由于该 MRO 会先命中 `nn.Module.forward`，必须显式实现 `forward()`；只接受 `ForwardType.DEFAULT`，不顺带实现 SFT。GRPO无critic，PPO通过同一policy上的可选顶层value head进入RLinf通用actor-critic链。完整外层契约为：

```text
predict_action_batch(env_obs, mode)
  eval  -> actions [B,N,14] physical CPU fp32
           GRPO: result={}
           PPO:  result["prev_values"] [B,1] fp32
  train -> actions [B,N,14] physical CPU fp32
           result["prev_logprobs"] [B,N,14] fp32
           GRPO: result["prev_values"] = None
           PPO:  result["prev_values"] [B,1] fp32
           result["forward_inputs"] = Tensor-only replay

forward(ForwardType.DEFAULT, ...) -> default_forward(...)
default_forward(forward_inputs, compute_logprobs=True,
                compute_values=False, compute_entropy=False)
  -> logprobs [B,N,14] fp32
  -> entropy  [B,N,14] fp32（仅请求时）
  -> values   [B,1] fp32（仅PPO compute_values=True）
```

policy 返回普通 `(actions, result_dict)`，不自行构造 dataclass；`huggingface_worker.py` 再构造 `RolloutResult` 并搬到 CPU。GRPO绝不能定义`value_head`属性，哪怕是`None`：当前worker用`hasattr(model, "value_head")`判断是否额外取bootstrap value。只有PPO actor/rollout两侧都显式`add_value_head=true`时才注册同构head；GRPO train result仍显式返回`prev_values=None`。

Ray driver 和每个 worker 都必须从同一 RLinf 内置 registry 构建模型。目标方式是按 `6d0db56::_register_builtin_models` 的既有模式注册：

```text
def _build_fastwam(cfg, torch_dtype):
    from rlinf.models.embodiment.fastwam import get_model
    return get_model(cfg, torch_dtype)

register_model("fastwam_robotwin", _build_fastwam, category="embodied", force=True)
```

仍使用官方`train_embodied_agent.py`；GRPO/PPO专用launcher只负责固定环境、Hydra配置选择、日志与资源监控，不替代runner。这里的`force=True`只沿用内置registry初始化的统一写法；目标model type必须唯一，不在运行时覆盖第三方实现。`RLINF_EXT_MODULE`只保留为RLinf作为不可改依赖时的备选，不与内置路径并行维护。

## 3. Rollout mode 契约

服务器 pin `6d0db56` 已确认只对既有硬编码模型显式传 `mode=train|eval`；rollout model本身始终保持 eval，不能从 `self.training` 区分训练采样与评估。Fast-WAM 若没有 shim 会误走 deterministic eval并缺少 `prev_logprobs`，因此该 pin 上 shim 是必要兼容改动，不再是条件性猜测。

建议在 Fast-WAM 功能分支加入默认不生效的通用 capability shim：

```python
elif getattr(self.hf_model, "rlinf_accepts_rollout_mode", False):
    kwargs = dict(kwargs)  # 保留已选择的 train/eval sampling params
    loss_type = self.algorithm_cfg.get("loss_type", "actor")
    kwargs["mode"] = "eval" if loss_type == "embodied_dagger" else mode
```

Fast-WAM policy 声明：

```python
rlinf_accepts_rollout_mode = True
```

该 `elif` 放在现有硬编码模型分支之后，比继续增加模型名更小且可复用；默认`False`时，所有π0/既有模型路径逐字保持原行为。不能用新字典覆盖worker此前选好的sampling params，并须保留DAgger强制eval语义。组合检查同时用一个 sentinel sampling kwarg证明Fast-WAM保留旧kwargs、π0仍走原分支。

这里还要区分两种“train”：`predict_action_batch(mode="train")` 是 rollout 的随机采样模式，函数仍在 `no_grad` 下且 rollout module保持 eval；`policy.train(True)` 是 actor 重放/反传时的 module mode。二者不能互相替代，也不能在 rollout 函数里调用会主动 `eval()` 且写死 B=1 的官方 `infer_action()`。

## 4. RoboTwin 环境契约

模型特有逻辑留在 Fast-WAM adapter，RoboTwin env 不新增 Fast-WAM 分支：

```text
RLinf env_obs
  -> Fast-WAM 三相机/state/text adapter
  -> normalized action [B, 32, 14]
  -> 反归一化
  -> 截取实际执行前缀 [B, N, 14]
  -> 现有 RoboTwin qpos chunk_step
```

第一版继承 π0 基座的 env、assets、seed/group/reset、reward 和终止语义。Fast-WAM vendored RoboTwin 只用于 standalone parity 和版本差异对照，不覆盖 RLinf RoboTwin。

完整数据流固定为：

| 阶段 | 数据/shape | 来源 |
|---|---|---|
| RLinf env observation | `main_images uint8 [B,H,W,3]`、`wrist_images uint8 [B,2,Hw,Ww,3]`且`[:,0]=left,[:,1]=right`、`states float [B,14]`、非空任务文本 | π0/RoboTwin 基座 |
| Fast-WAM adapter | 三相机严格拼成 `image [B,3,384,320]`，缩放至 `[-1,1]`；state 用官方 stats 归一化 | 官方 RoboTwin deploy |
| 文本条件 | 精确官方`DEFAULT_PROMPT`包装任务；`text_context [B,128,4096]`、`mask [B,128]`，再追加一个proprio token | 官方 dataset/deploy与`encode_prompt/_append_proprio_to_context` |
| action latent | `x [B,H=32,D=14]`，Flow-SDE chain 为 `[B,S+1,32,14]` | 官方模型 + 社区 RL |
| policy output | normalized `model_action [B,32,14]` | 官方模型 |
| 环境动作 | 官方 denorm 后截取 `actions [B,N,14]` | 官方 deploy + RLinf qpos `chunk_step` |
| trajectory/replay | 先按决策步堆成 `[T_dec,B,...]`，actor 前展平为 `[T_dec*B,...]` | π0/Motus/RLinf 现有数据层 |

### 4.1 Batch 能力边界

官方 `FastWAM.infer_action()` 是 B=1 部署编排器，不等于模型本体只能 B=1。它的 image/proprio 检查、初始 latent `(1,H,D)` 和返回 `[0]` 写死 singleton；但下面这些官方层均保留 batch 维：

- `encode_prompt(Sequence[str])`；
- `_append_proprio_to_context([B,D])`；
- `WanVideoVAE.encode(videos)` 的 batch 返回；
- video/action expert 的 `pre_dit`；
- `MoT.prefill_video_cache` 的 `[B,Sv,*]` KV；
- `forward_action_with_video_cache`；
- scheduler 的共享时间网格和 batched step。

第一版 production B>1 使用官方 VAE wrapper。该 wrapper 内部逐条 VAE encode 再 stack，但随后 video expert、KV cache 和 action expert 都按一个真实 batch 前向；这属于官方 API 的正常实现，不是 policy 异常 fallback。社区直接调用 inner VAE 做并行 3D conv 只作为测出瓶颈后的性能优化，必须另做 B=1/B>1 parity。

允许按显存配置`model_forward_batch_size`将大env batch切成固定子批；这是正常资源分块。共享`k`、每样本initial latent和SDE epsilon必须先对整个逻辑rollout batch一次生成，再把对应slice传给子批，chunk内部不得调用RNG。任一子批失败立即停止，禁止catch-all后改成逐env完整模型推理。

固定eval与train随机性分开定义：train每样本initial latent独立；固定eval已经冻结为用显式seed按官方B=1方式生成一个latent后沿B复制。这样每个样本都等价于独立调用同seed的官方singleton，并保持batch permutation equivariance。该broadcast不得进入train；若未来改为一个generator顺序生成B个latent，验收对象必须改为固定batch，不能再声称逐样本singleton parity。

两路 wrist 必须存在并保持 left/right 顺序；第一版不做“缺腕图时复制主相机”的隐式兜底。state/action stats必须是singleton key`default`、14D、global z-score契约；prompt、图像dtype或batch不符均fail-fast。RoboTwin动作路径保持absolute qpos identity，不新增Fast-WAM env分支，也不迁移π0 delta/关节转换、LIBERO gripper转换或LaWAM EE planner。

## 5. GRPO 契约

- 同一 GRPO group 使用相同 reset state/任务条件。
- 同组不同轨迹使用独立 Gaussian exploration noise。
- 第一版沿用 π0 与社区 Fast-WAM 的已跑通行为：一个 rollout model batch 共用 denoise index `k`，但 Gaussian 样本彼此独立；独立 `k` 是后续算法变量，不混入首次迁移。
- 第一版无 critic，`adv_type=grpo`、`loss_type=actor`。
- reward 与当前基座保持 chunk/trajectory 语义；不能把每个 action token 当独立环境 reward。
- 二值奖励组全 0 或全 1 时 advantage 退化，因此学习任务必须有 headroom。

### 5.1 PPO observation critic 契约

- PPO从官方Fast-WAM release base冷启动，`adv_type=gae`、`loss_type=actor_critic`、`group_size=1`，不从GRPO DCP暖启。
- value feature只取官方`prefill_video_cache()`最后一层`video_kv_cache[-1]["v"] [B,120,3072]`的token mean。它已通过前29层video self-attention及text/proprio cross-attention，但不读取action latent、denoise `k`或SDE noise。
- 取`v`而不取已应用RoPE的`k`；严格检查batch、token数、hidden width和finite。feature不写入replay，actor以同一image/text/proprio重建。
- critic input显式detach。`ValueHead 3072→1024→512→256→1`参数跟随BF16模型，rollout与FSDP actor使用相同计算dtype，输出统一转FP32。
- behavior rollout产生old `prev_values[B,1]`，actor replay产生current `values[B,1]`，final observation沿同一路径产生bootstrap。Flow-SDE与logprob链不因PPO改变。

## 6. Flow-SDE 策略概率契约

初始 `N(0,I)` latent 与 policy 参数无关，不能直接作为 policy logprob。第一版采用单 stochastic transition：

1. 使用官方 scheduler 生成与 deterministic ODE 同构的 denoise chain。
2. 一个 rollout model batch 选择一个 denoise step `k`。
3. 在该 transition 采样：`x_(k+1) ~ Normal(mean_theta, std)`。
4. 首次迁移保留社区与 π0 已验证的完整 action chain、`k`、old logprob 和重建 conditioning 所需内容；不保存 video chain/KV cache。
5. actor 重算完全相同 transition 的 `new_logprob`。
6. SDE 系数、logprob、entropy、ratio、advantage 用 FP32。
7. old logprob 必须来自真实 behavior rollout；不可用 actor 当前 forward 合成 denominator。

时间和概率的权威分工固定为：

- Fast-WAM 官方 scheduler 决定 raw model timestep、shifted normalized `t`、signed `delta` 和有效 `sigma_shift`；rollout/replay 都调用同一 resolved schedule。
- RLinf pin `6d0db56` 的 OpenPI `flow_sde` 决定单 stochastic transition 的 mean/std 与 actor replay 语义。
- 社区 Fast-WAM 只提供这两者的已跑通拼接参考；其遗漏 `sigma_shift` 和 `t<=0.98` 常数不能直接照搬。

Fast-WAM scheduler 的 `delta=t_(k+1)-t_k<0`，实现先断言严格递减，再令 `dt=-delta>0`；不使用 `abs(delta)` 掩盖时间方向错误。`t=1` 的首步 sigma 分母采用 RLinf 官方规则：使用下一 schedule 点代替当前 1，而不是社区硬编码 `0.98`。stochastic probability/chain使用FP32；进入model forward前把latent/raw timestep显式转回官方model dtype，velocity再upcast FP32。deterministic eval保留官方model-dtype state并调用官方scheduler step，但仍在同一个参数化loop内。选中stochastic step的std必须严格正且有限；第一版不以任意floor/clamp改写density。

更新前、同权重的 actor/rollout 首 minibatch 必须满足 `log_ratio≈0`。跨进程 bf16 微小差异经高维求和会被放大，因此这是硬门槛。

第一版 `forward_inputs` 只包含 Tensor；`RolloutResult.__post_init__()` 会统一搬到 CPU，trajectory 再按决策步堆叠/展平：

```text
chains             [B, S+1, 32, 14] fp32
denoise_inds        [B]               int64
image               [B, 3, 384, 320]  model dtype
text_context        [B, 128, 4096]    model dtype，未追加 proprio
text_context_mask   [B, 128]          bool
proprio             [B, 14]           fp32，已按官方 stats 归一化
action              [B, N*14]         fp32，实际执行 physical qpos
model_action        [B, 32*14]        fp32，完整 normalized 输出
```

其中 `action` 供现有 EnvWorker/trajectory 记账，`model_action` 保留模型空间动作语义；actor 概率重算使用 chain，不从 physical action 反推。任务字符串、完整 video chain、KV cache、velocity/debug 张量均不进入 replay。

`chains[:,0]` 是 initial Gaussian；`chains[:,k] -> chains[:,k+1]` 是 behavior policy真实采样的唯一 stochastic transition。chain和`model_action`始终保留完整 H=32；只有环境 action、old/new logprob和loss截前 N=24。policy 内不对 `[B,N,14]` logprob提前求和，RLinf通用 loss按配置完成 chunk-level聚合。

同一个 rollout 调用沿用 RLinf OpenPI 的语义：batch 共享一个均匀采样的 `k`，但每条轨迹的 initial latent 和 SDE Gaussian noise 独立。trajectory 展平后不同样本可以具有不同 `k`，因此 actor replay 的 timestep 必须真正支持 `[B]`，不能只依赖 eval 模式的 scalar broadcast。

## 7. Train/eval 共核契约

分阶段先验 deterministic、再验 stochastic，不代表生产代码拆成两套。生产 B=1/B>1 eval、train rollout 与 actor replay必须共享以下数值原语：

| 环节 | eval | train rollout | actor replay | 必须共享 |
|---|---|---|---|---|
| 环境适配 | 三相机/state/text | 同左 | 使用保存的同语义 Tensor | resize、顺序、prompt、stats |
| conditioning | T5、proprio、VAE、video expert | 同左 | 从保存的 image/base context/proprio 重建 | 同一 `build_action_conditioning()` |
| 时间网格 | 官方 resolved scheduler | 同左 | 使用保存的 `k` 和同一网格 | raw timestep、normalized `t`、`delta`、`sigma_shift` |
| velocity | action expert forward | 同左 | 同左并开启所需梯度 | 同一 `predict_action_velocity()` |
| 去噪 | `flow_sde_rollout(..., deterministic=True)` | 同一函数传 `deterministic=False`，仅选中 `k` 随机 | 不重新采样，只重放真实 `x_k/x_(k+1)` | 同一循环和 mean/std 原语 |
| 后处理 | denorm + 前 N | 同左 | 不从 physical action 反推概率 | 同一 action stats/截断 |

唯一允许差异是：eval 不注入噪声、不保存训练 replay；train rollout 在选中 transition 采样并保存 behavior density；actor replay 对同一 transition 开梯度重算。禁止另写 `deterministic_ode_rollout()` 与 `train_sde_rollout()` 两套生产去噪循环。官方 B=1 `infer_action()` 只作 fixture oracle，不成为生产 eval 路径或 batch fallback。

## 8. Horizon 与执行前缀

第一版只实现社区已跑通的定义：对完整 `H=32` latent 采样，环境只执行前 `N`，old/new logprob 和 loss 也只取前 `N`。这与 π0 当前 `action_horizon/num_action_chunks` 分离方式一致。

该定义的已知限制是尾部 latent 可经 self-attention 影响前缀，却不进入 ratio。`full_latent` 或“只给执行前缀加噪”都会改变策略定义和高维 ratio 尺度，不在首版同时实现；只有首版 parity/学习门通过后才单独比较。

首个工程 smoke 已确定在 `N=24` 时把 train/eval 的 `max_episode_steps`、`max_steps_per_rollout_epoch` 和 `task_config.step_lim` 都设为 192，确保 RLinf 与 RoboTwin 内部终止条件一致并恰好执行 8 个完整 action chunks。正式协议是否继续 192，或以后实现带 mask 的短尾 chunk，后置讨论。

## 9. Trainable、FSDP 与同步契约

GRPO trainable allowlist：

```text
action_expert.*
```

PPO在此基础上额外训练顶层`value_head.*`；RLinf按名称为它使用独立`value_lr`。critic loss不回传到video/text/proprio，actor loss仍只更新action expert。

冻结：proprio encoder、video expert、VAE、T5 及其他视觉/文本组件。该 action-only 选择严格复现社区有效实验的实际梯度路径，也与 Motus 首轮工程链相符。actor调用`policy.train()`后仍必须把这些冻结conditioner恢复为eval，仅canonical action expert保持train；否则即使无梯度，dropout/mode差异也会破坏rollout与actor replay的一致性。

社区把 `proprio_encoder` 标成可训练，但 rollout 缓存的是在 `no_grad` 下生成的最终 context，actor 没有重跑 encoder，因此其成功曲线不能作为“action+proprio 已验证”的证据。第二阶段若启用 proprio，必须用已保存的 normalized proprio 与基础 text context 在 actor 中重跑 encoder，并分别验证 action-context 和 video-context 梯度语义；这是一项新实验，不伪装成首版迁移。

官方模型把同一专家同时注册到 `video_expert`、`action_expert`、`mot`，并令 `dit=mot`；RLinf 同步枚举又使用 `remove_duplicate=False`。因此 builder 在 FSDP 前必须把 `mot` 收敛为唯一注册参数树，其他名称只做非注册兼容 alias。固定 seed 推理前后必须一致，`state_dict`/optimizer/sync 参数名必须无逻辑重复。

社区代码表明 MoT 手工 cross-expert attention 不适合按block/expert自定义wrap。第一轮采用RLinf pin现有FSDP2：禁用transformer/block/expert目标，最后shard root；同时保留`apply_fsdp2_to_model`对非tied `nn.Embedding`的既有单独wrap。该pin的FSDP2路径不读取`use_orig_params`，因此它不是首版配置机制。实施时保存实际FSDPModule/wrap inventory，以双A800 smoke冻结布置。

权重变化通常是稠密的，第一版使用社区已跑通的 bucket sync。trainable 参数名必须只来自 canonical action-expert key 集；通用同步器还会按现有规则携带 persistent buffers，因此验收时分别审计“可训练参数”和“实际传输集合”，不把两者混为一谈。社区成功配置未写`bucket_device`，当前pin默认走GPU；但 2026-07-18 首次双 A800 smoke 在初始 actor→rollout 广播进入 CUDA IPC 时被当前 AutoDL 容器拒绝 `pidfd_getfd`。因此本机 Fast-WAM 配置改用 RLinf 原生 CPU bucket；这只改变 staging/transport，不改变同步集合和 tensor 值，并与本机已跑通 π0 的 CPU transport 方向一致。π0 当前 patch+delta 配置保持不动；Fast-WAM 仍单独使用 bucket，避免社区已经遇到的 sparse patch densify/OOM。

actor/rollout 保留 RLinf 常规初始权重同步；不能沿用 Motus 为减轻初始化负担而临时关闭完整 init sync 的权宜做法。若将来要关闭，必须先在独立进程证明所有模型参数和 persistent buffers 的 hash 一致。FSDP 或 dtype 报错也不能通过临时冻结 action expert 内部有效模块来“过 smoke”；既定 trainable inventory 只能经明确设计决策改变。

## 10. Checkpoint 契约

同时保留：

- RLinf DCP：模型、optimizer、scheduler，用于精确 resume；当前 RLinf 的 global step 由 `global_step_<n>` checkpoint 目录恢复，实施时按现场版本复核。
- Fast-WAM deploy artifact：官方 loader 可读取的里程碑权重。

官方当前 deploy payload 以 `mot` 为主体，并可带 `proprio_encoder`、`step`、`torch_dtype`。目标pin自带`no_dist=True`的DCP model-state抽取器；regular run显式`save_full_model_weights:false`，export从DCP读取canonical完整`mot`树并对原官方checkpoint做exact key/shape/dtype校验，再在新进程使用官方loader。PPO的`value_head.*`必须留在RLinf DCP用于精确resume，但不属于官方deploy schema，export显式忽略该前缀。P3 action parity比较“fresh-process恢复的RLinf模型”与“同一DCP导出的官方模型”，而不是要求非零更新后仍等于原base checkpoint。DCP用于恢复，deploy artifact只保留少量里程碑。
