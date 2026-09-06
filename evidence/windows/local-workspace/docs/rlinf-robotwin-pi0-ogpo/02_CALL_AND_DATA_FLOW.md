# OGPO × π0 × RoboTwin × RLinf：调用流、数据流与接缝图

最后更新：2026-08-08。

本文是 [`00_INDEX_AND_IMPLEMENTATION_PLAN.md`](00_INDEX_AND_IMPLEMENTATION_PLAN.md) 的事实附录，
只维护“基线怎样调用、OGPO 链怎样接、每个接缝抄谁”。设计决定仍以主计划为准。当前实现位于
服务器 `/root/autodl-tmp/RLinf_ogpo_pi0_robotwin`、branch `codex/ogpo-pi0-robotwin`，commit
`5d5c84e3` 已推送；DSRL/RLT/QAM 仍只提供按需的窄工程先例。

## 0. 一屏结构

```text
保留 RLinf 系统骨架
  Hydra config -> placement -> actor/rollout/env workers -> runner -> eval/DCP

保留 OpenPI/RLinf π0 接口
  observation transform -> frozen prefix -> action expert velocity -> action transform

薄改 RoboTwin/RLinf 环境接口（只在 OGPO route）
  π0 predicts H=50 -> execute C=10 via singleton step loop
  -> primitive obs/reward/terminated/truncated/next obs

替换 PPO 算法内核
  不再 GAE/value/current-rollout PPO
  改为 primitive replay -> overlapping 10-step sequences -> Q ensemble TD
       -> replay-state group chains -> whole-chain PPO + success BC + CA -> targets
```

## 1. 当前 RLinf π0 PPO RoboTwin 调用流

### 1.1 入口与 worker 生命周期

```text
examples/embodiment/train_embodied_agent.py::main
  -> Hydra compose / validate_cfg / resolved config print
  -> Cluster + HybridComponentPlacement
  -> actor class selected by algorithm.loss_type
  -> EmbodiedFSDPActor group
  -> MultiStepRolloutWorker group
  -> EnvWorker group
  -> EmbodiedRunner(...)
     -> init_workers()
     -> run()
```

`EmbodiedRunner.init_workers` 先初始化 rollout/env，再初始化 actor；当前 resume 入口只恢复
`resume_dir/global_step_N/actor`。`EmbodiedRunner.run` 的每个 global step 为：

```text
actor/rollout set_global_step
  -> update_rollout_weights
     rollout receiver first -> actor sender -> patch/full weight sync
  -> concurrent:
       EnvWorker.interact
       MultiStepRolloutWorker.generate
  -> actor.recv_rollout_trajectories
  -> actor.compute_advantages_and_returns
  -> actor.run_training
  -> eval / checkpoint / global_step++
```

可复用：Hydra、placement、channel、并发采集、eval、step lifecycle。runner 仍调用同名 actor
接口；新 OGPO worker 让 `compute_advantages_and_returns` 只汇总采集指标、让 `run_training` 执行
replay/Q/actor update，并 override save/load。runner 只在 OGPO route 增加按 primitive online rows 的
eval/checkpoint/stop 与 resume 阈值，不改变旧 PPO 的 global-step 分支。

### 1.2 rollout 与环境内环

```text
EnvWorker.bootstrap_step
  -> reset observation / initial done

MultiStepRolloutWorker.generate_one_epoch
  -> _predict_rollout_actions
  -> HuggingFaceWorker.predict
  -> OpenPi0ForRLActionPrediction.predict_action_batch
     -> obs_processor + input_transform
     -> _sample_actions_with_prefix_cache
        -> _build_prefix_cache
        -> repeated get_velocity / sample_mean_var_val
     -> output_transform
  -> _build_rollout_result
  -> RolloutResult sent to EnvWorker

EnvWorker._run_interact_once
  -> align current action/logp/value with previous env result
  -> env_interact_step
  -> RoboTwinEnv.chunk_step
  -> accumulate EmbodiedRolloutResult
  -> final extra policy query for bootstrap value
  -> to_trajectory
  -> trajectory sent to actor
```

当前 PPO 必须额外查询未执行 action 的 bootstrap value；OGPO 没有 GAE/value head，这个 extra
value query 应在 OGPO route 关闭。环境采集仍由 rollout/env worker 完成。

### 1.3 actor PPO 内核

```text
recv_rollout_trajectories
  -> convert_trajectories_to_batch
  -> _process_received_rollout_batch
  -> process_nested_dict_for_adv
  -> calculate_adv_and_returns
  -> compute_gae_advantages_and_returns
  -> process_nested_dict_for_train
  -> run_training / train_micro_batch
  -> OpenPI default_forward: current logp + value
  -> compute_ppo_actor_critic_loss
  -> optimizer
```

这段只作为“哪里切开”的依据。OGPO 不复用 value head、GAE、return、value clip、current-rollout
minibatch 或 actor-critic loss。

### 1.4 FSDP、同步与 checkpoint

- `FSDPModelManager.setup_model_and_optimizer`：参数分组、FSDP wrap、optimizer、scheduler、RNG；
- `EmbodiedFSDPActor.sync_model_to_rollout` ↔
  `MultiStepRolloutWorker.sync_model_from_actor`：模型权重同步；
- `EmbodiedRunner._save_checkpoint`：当前只保存 `global_step_N/actor`；DCP 内含 model、optimizer、
  scheduler、RNG，并另可导出 full weights。

OGPO 继续复用这些机制；online/EMA expert 作为同一 OpenPI/FSDP model 的显式角色一起同步，
rollout worker 按 train/eval mode 选择。checkpoint 另扩展 Q/target-Q/replay/success/counters。

## 2. 当前 π0 / RoboTwin 数据合同

### 2.1 observation 与 action

当前 `adjust_bottle` π0 PPO 静态配置/源码合同：

| 对象 | 形状/语义 | 来源 |
|---|---|---|
| head image | `[B,Hraw,Wraw,3]` | RoboTwin/RLinf adapter |
| wrist images | `[B,2,Hraw,Wraw,3]` | RoboTwin/RLinf adapter |
| qpos/state | `[B,14]` | RoboTwin ALOHA joint state |
| instruction | `list[str]` | RoboTwin task description |
| processed images | 3 × `[B,224,224,3]` | RLinf/OpenPI transform |
| padded model state | `[B,32]` | OpenPI π0 contract |
| tokens/mask | `[B,48]` | RLinf π0 RoboTwin config |
| model action | `[B,H=50,D_model=32]` | OpenPI default horizon/dim |
| env action | `[B,H=50,D_env=14]` | ALOHA output transform |

14D action 顺序是左臂 6 + 左夹爪 1 + 右臂 6 + 右夹爪 1。OpenPI/RLinf 的 RoboTwin transform
对 12 个 arm joints 使用 delta action、gripper 保持 absolute，再通过 checkpoint norm stats
归一化/反归一化。π0 与 π0.5 的 `adapt_to_pi`、mean/std 与 q01/q99 默认不同，必须随实际
checkpoint/norm stats 一起 source-lock，不能混用。

### 2.2 当前 PPO denoising payload

在静态 `H=50, D_model=32, D_env=14, S=4` 下：

| 字段 | 典型形状 | 说明 |
|---|---|---|
| `chains` | `[B,S+1,50,32]` | 初始 noise + 四步 denoise state |
| `denoise_inds` | `[B,S]` | non-joint 训练实际选择的 denoise index |
| env `action` | `[B,50,14]` | 送入 RoboTwin 的 chunk |
| `model_action` | `[B,50,32]`，运输时可 flatten | normalized model coordinate |
| `prev_logprobs` | `[B,50,14]` | rollout old logp |
| `prev_values` | `[B,1]` | PPO value |

默认 non-joint 路径随机选一个 denoise transition 用 `flow_sde`，其余用 ODE；训练时只对该转移
重算 current logp，`chunk_level` 再把 `50×14` 求和形成一个 macro ratio。打开
`joint_logprob` 会对 denoise 维求均值，但仍不是 OGPO 官方 `[B,G]` normalized chain logp。

已有 PPO 还存在一个不应迁移的细节：rollout value 是四个 denoise-step value 的均值，actor
重算 value 却只取随机 denoise step。OGPO 没有这个 value 路径，直接删除而不是修补继承。

### 2.3 RoboTwin chunk 返回能力

`RoboTwinEnv.chunk_step` 一次把完整 `[C,14]` 传给 vectorized RoboTwin。live 源码确认它只产生
一个块末 observation；`_cal_chunk_rewards` 虽构造 `[B,C]` 张量，真实 sparse reward/done 只合成
到末格，不能恢复块内 state 或提前终止的 waypoint。PPO 因而把整个 chunk 当一个折扣步。

同一 adapter 的 `RoboTwinEnv.step` 已接受 `[B,14]` singleton action。OGPO route 将循环调用它
C=10 次并保留每次返回；这与官方 OGPO `action_queue -> env.step` 同构。原 PPO/GRPO
`chunk_step` 不改，新增 route 需用固定 action/seed 验证最终 simulator state 与原 chunk 路径一致。

## 3. 目标 OGPO 调用流

### 3.1 environment data path

```text
target π0 action expert (rollout copy)
  -> existing predict_action_batch / transforms
  -> full model prediction [B,H_model=50,32]
  -> choose executed prefix [B,C=10,14]
  -> OGPO EnvWorker singleton-step loop
     for j in 0..C-1:
       RoboTwinEnv.step(action[:,j])
       append primitive (obs, action_model[32], action_env[14], reward,
                         next_obs, terminated, truncated, episode/step)
       stop/pad on done
  -> attach obs/action/reward/done/valid trace to existing Trajectory.forward_inputs
  -> actor receives the completed trajectory and batch-inserts valid rows into replay
  -> add one Q credit and one PI credit per inserted primitive row
  -> if episode succeeded, add its row IDs to success view
```

transport 形状固定为：`obs_trace [B,C+1,...]`，`reward/terminated/truncated/valid [B,C]`，
`action_model [B,C,32]`，`action_q/action_env [B,C,14]`；首个 done 后 `valid=false`。
`action_env` 只用于本次执行 transport；长期 ring 保存 `action_model` 与 normalized `action_q`。

singleton primitive loop 只服务训练 replay；eval 不构造 primitive replay，继续使用 RLinf 已有
`chunk_step`，但切换为 online action expert。

这个路径不长期保存 full 50-step prediction、rollout chain/logp/value/forward payload。真实执行
prefix 同时提供 Q action 与 success BC target；actor group update 从 replay observation 重新采样。
首版 train rollout 固定覆盖完整 200-step episode 且 `auto_reset=false`，因此 trajectory 到达 actor 时
成功标签完整，不需要逐 transition 跨 worker streaming 或 pending-episode checkpoint。

### 3.2 critic update path

```text
sample primitive start states + up-to-C consecutive transitions
  -> R_h = Σ gamma^i r_i, true h, bootstrap mask, s[t+h]
  -> four pooled frozen π0 prefix blocks + proprio
  -> target π0 samples next action chunk
  -> take canonical next prefix [C,14]
  -> mean over 10 target-Q heads
  -> y = R_h + gamma^h * bootstrap_mask * Q_target(s[t+h],next_prefix)
  -> regress all 10 online Q heads in FP32
  -> Polyak update target Q
```

sequence return、Q ensemble/aggregation、target order 抄 OGPO；C=10、π0 feature tap 与 FSDP
owner 是适配。首版只采一个 next action；官方 PaliGemma runnable 的额外 8-action Q-target variance
reduction 明确关闭。PPO value head不能替代 action-conditioned `Q(s,a[1:C])`。

### 3.3 actor zeroth-order update path

```text
sample B observations from replay
  -> compute one frozen prefix per observation
  -> expand each state to G candidates (首版建议 G=8)
  -> target action expert samples full stochastic chains
       raw chains:   [B,G,K+1,H_model=50,D_model=32]
       old_chain_score: [B,G]
       projected a:  [B,G,C=10,D_env=14] for Q
  -> target Q ensemble scores terminal action chunks
       Q_full: [B,G,M=10]
  -> CA: per-head group-centered advantages -> all-head sign consensus
  -> current action expert scores the same frozen chains
       current_chain_score: [B,G]
  -> exp(current_score-old_score) + clipped PPO surrogate
     score = joint_logp / ((K+1)*(H_model*D_model)); K4/H50/D32 denominator=8000
  -> success-view π0 flow-matching BC:
       frozen cached prefix + online action-expert velocity on full H50×D32
       loss only on executed C10×D32 prefix
  -> update current action expert
  -> Polyak update target action expert
```

Q 和 advantage 必须 stop-gradient；不计算 `dQ/da`，不把 Q gradient 传入 actor，也不对生成链
BPTT。group chains 属于 replay-state imagination，不进入 EnvWorker。

success BC 不再调用另一套 full-model `sft_forward`。π0 的 target `noise-action` 经
`t_ogpo=1-t_pi0` 和 velocity 反号后写成等价的 `action-noise`；完整 H50 构造 noisy input，最后才裁
C10 loss。这样 PPO same-chain scorer 与 BC 都复用已验证的 cached-prefix/online-suffix FSDP 图。

likelihood chain 保存 post-`mean±3σ`、pre-final-action-bound、pre-env-transform 的 raw latent final
state；Q 使用其 C10×14 normalized canonical projection，环境使用另行 output-transform/14D
projection 后的 execution copy。OpenPI transform 本身不 clip；若执行路径另有 bound，也不写回
raw chain。这样 target old-logp 与 current scorer 始终作用于同一条 raw chain，避免官方 tapered sampler 在饱和动作上“old 按
pre-clip、current 按 post-clip”导致 identity ratio 偏离 1；这也沿用 RLinf π0 PPO 已有的 raw-chain/
env-action 分字段方式。

UTD 计数发生在 actor ingest 后：若本批插入 N 个 primitive rows，官方 `UTD-Q=UTD-PI=1` 对应
累计 N 次 Q updates 和 N 次 actor updates。实现可以批量结算，但不能把 N 行偷换成一次 update。
论文伪代码写 Q-first，固定 commit 的 fused `_update` 则按 actor/EMA-first、critic/target-Q-second
执行；本项目跟发布代码，所以每个 paired credit 的调用顺序是上面的 §3.3，再执行 §3.2。

### 3.4 外层cycle、rollout位置与本轮实测

RLinf中的一个外层`cycle/wave`不是primitive step，也不是一次optimizer update：

```text
sync online+EMA weights
  -> 8 env并行，各生成一条最长200-step episode
       每10步重新看图；π0预测H50但只执行C10
  -> 完整8-env trajectory返回actor
  -> valid primitive rows写replay；成功episode row IDs进success view
  -> pending += warmup后new_rows * UTD
  -> 连续跑floor(pending)个paired updates，保留小数余数
  -> 若跨阈值则eval/checkpoint
  -> runner cycle +1
```

因此rollout固定在每个cycle的前半段、训练update之前；当前没有rollout/update pipeline，也没有
primitive streaming。典型wave写入1,242–1,600 valid rows，UTD`.1`会在随后集中执行124–160次
paired updates。本轮第25 wave写1,107 rows到34,968并跑110次；第26 wave完整模拟8条episode但只
接收quota所需32 rows，再跑4次到2,500。最终为26 cycles、208条实际train episodes、35,000条
replay-valid primitive transitions；35k/200=175只是满长episode-equivalent。

fixed eval在fresh baseline、首次跨20k和final35k触发；每次4 env×5 waves=20条episode，不写replay。
checkpoint interval50k未触发周期save，但runner在final保存`global_step_26`。这里checkpoint名中的26
是runner cycle数，不是row数或optimizer update数。

## 4. 每个接缝抄谁

本文件只维护“怎么流”；逐组件的首要来源、可复用部分和明确适配集中在
[`01_REFERENCE_MATRIX.md`](01_REFERENCE_MATRIX.md#3-每个实现组件主要像谁)。简写为：算法数学抄
OGPO，生命周期抄 RLinf π0 PPO，模型/坐标抄 OpenPI + RoboTwin，C10/canonical action 抄 RLT 的
同系统合同，critic/replay 只复用 QAM 已验证的窄工程接口。

## 5. 已实现的 class / symbol 接缝

实际 class/file 清单集中在主计划
[`00_INDEX_AND_IMPLEMENTATION_PLAN.md` §7.2](00_INDEX_AND_IMPLEMENTATION_PLAN.md#72-已实施的文件级接缝)。
这里补充调用边界：`RoboTwinEnv`、`embodied_io_struct.py` 不改；`EmbodiedRunner.run` 只增加
row-based OGPO 调度。专用 actor worker 实现现有接口，EnvWorker 只增加 OGPO singleton trace，
rollout worker 只跳过 value/bootstrap 请求并继续使用现有 batch inference/sync。

## 6. 最小验收

正式 smoke 前完成三组高信息量检查：OGPO 数学（h-step TD、CA、SDE、same-chain ratio）；真实
checkpoint 单卡 shape/role、真实两卡 B4×G8 candidate backward 与 B64/G8 production
`target-action -> actor+BC -> critic`；EMA shadow 和 checkpoint sidecar save/load/失败原子性。最终为
25-file syntax、26 个定向 pytest、9 个 config locks、两套 Hydra compose、Ruff/diff 全通过。完整 update
约 12.75 秒，allocated/reserved 峰值约 35.31/39.57 GiB/卡；mixed early-done `final_obs` fixture 也已覆盖。

首次真实 smoke 随后把生产调用链完整走通：

```text
8-env reset -> π0 EMA rollout H50 -> 执行 C10 -> primitive trace 80 rows
-> 两 rank replay 各 40 rows -> target next-action
-> OGPO+CA actor update -> critic TD update -> policy_version 1
-> sync online/EMA state 到 rollout -> 4-env C10 eval
-> DCP + two rank sidecars + atomic complete.json -> clean shutdown
```

真实结果为 exit 0、`updates_run=actor_updates=critic_updates=policy_version=1`、pending credits 0、
ratio 1；DCP/sidecar 能由 CPU 重新读取并共享同一 snapshot。它验证 singleton step、C10 trace、channel、
row scheduler、FSDP update、sync、eval 调用和 checkpoint 生命周期。由于 train/eval 都只跑 10/200 步，
没有终局 reward/success；因此没有验证完整 episode success rate，`eval/num_trajectories=0` 也只是短
eval 没有完成 episode。

## 7. 已冻结源码快照与实现期数值

2026-08-07 当前实现已确认：π0 `H_model=50,K=4,D_model=32,D_env=14`；真实 prefix 四块长度
`(256,256,256,48)`；`train_expert_only=true`；B64/G8，flat candidate microbatch=32/rank。tied
PaliGemma head 的 `FULL_SHARD+use_orig_params=true` ownership 已做 OGPO-only 窄适配并真实两卡通过。

参数来源和实现值集中在 `03_PARAMETER_PROVENANCE.md`。尚未落定的是 replay/checkpoint 资源组合：
250k ring 约 334.98 GiB RAM、满 sidecar 约 322.79 GiB，不能在 50k 间隔下无限保留完整快照。
full-update timing 已由模型级 probe 得到；按当前 UTD1 原样外推 230k credits 约需 33.9 wall-days/
1,629 GPU-hours，因此 formal compute budget 也未冻结。首次 8-env smoke 的全流程物理峰值为
`50701/51311 MiB`/卡、cgroup current 约 56.0 GiB；这允许讨论下一次有界并行 probe，但不能仅凭空闲
显存直接宣称 B/G/env 可以翻倍。动态 GPU/进程状态在下次运行前仍需刷新。
