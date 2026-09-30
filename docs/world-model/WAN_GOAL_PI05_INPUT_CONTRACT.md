# Wan LIBERO Goal → π0.5 GRPO 输入契约

更新：2026-09-30。状态：用户已接受一视角 π0.5 适配，要求验证有效更新后启动正式训练；本地已备最小补丁、smoke/formal YAML及服务器CPU检查脚本，尚无本页作者执行的服务器部署或GPU验收。官方 OFT smoke 独立推进。

## 结论

官方 Wan Goal + OpenVLA-OFT 是现成的一视角、无 proprio 路径。选定 π0.5 LIBERO 权重可以沿用 RLinf 的 GRPO 策略实现，但不能原样换模型：其默认 LIBERO 输入需要主相机和腕相机，而 Wan 仅生成主相机。

一个容易混淆的细节是：**本次选定的 `pi05_libero` 配置不把 state 作为策略条件**。它与原 RoboTwin/Sidney π0.5 的输入不能混为一谈。因此不需要编造下一 proprio；真正需要显式决定的是“腕图缺失的一视角 π0.5 适配”。如要求保持原双视角观测，则当前这份单视角 Wan 权重不能满足。

## 固定来源

| 组件 | 固定版本 |
|---|---|
| RLinf | `d34d4c320d08cb982de034aa9a011f08dc0fa217` |
| π0.5 | `RLinf/RLinf-Pi05-LIBERO-SFT@45ccfcc4e28634f1576ebf78cab0fbe2fd82432d` |
| OFT | `Haozhan72/Openvla-oft-SFT-libero-goal-traj1@d20e1d447dfd87c0daa121b0739e2a379f7fe334` |
| Wan | `RLinf/RLinf-Wan-LIBERO-Goal@bd395971c3467de3dd19e7e6c7562af48a2894a6` |
| OpenPI 数据变换依赖 | `rlinf-openpi==0.1.1`，核查 wheel 内源码，未在本地安装 |

小型官方源码与元数据只读副本位于 `E:/Codex/home/visualizations/2026/09/28/01a0e6c7-bb8c-7421-8697-110ddd91d2f1/wan-goal-pi05-contract/`。文件名把 `/` 替换为 `__`。未下载模型权重到此目录。

## 观测：哪些能直接接，哪些不能

| 项目 | 已核事实 | 对适配的含义 |
|---|---|---|
| 主相机 | Wan 输出 `[B,256,256,3]` uint8；π0.5 变换会 resize/pad | 可走原图像预处理 |
| 腕相机 | Wan `wrist_images=None`；`LiberoInputs` 无条件读取 `observation/wrist_image` | 直接替换模型会缺键；必须显式处理 |
| state | Wan 返回 `[B,16]` 零占位；真实 LIBERO state 为 EEF position 3 + axis-angle 3 + gripper qpos 2 | 占位不是预测状态；此模型路径不用它作为条件 |
| 语言 | Wan 返回数据集中的 task description | 可走原 tokenizer |
| 动作 | π0.5 输出前 7 维；LIBERO 数据本身是 delta EEF + gripper | 可沿用原 action 转换，不应按 RoboTwin 绝对关节动作积分/覆盖 state |

state 不进入该策略条件的完整证据链：

1. [`dataconfig/__init__.py:107–115`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openpi/dataconfig/__init__.py#L107-L115) 定义 `pi05_libero`：`action_horizon=10, discrete_state_input=False, extra_delta_transform=False`。
2. `rlinf-openpi==0.1.1` 的 `openpi/training/config.py:125–135` 将 `discrete_state_input` 交给 `TokenizePrompt`；其 `openpi/transforms.py:248–266` 在 false 时明确向 tokenizer 传 `state=None`。
3. [`pi0.py:277–303`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openpi/pi0.py#L277-L303) 仅非 π0.5 分支使用 `state_proj`；π0.5 的 prefix 是图像和语言，suffix 是动作和时间。
4. [`libero_dataconfig.py:78–89`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py#L78-L89) 说明额外 delta 变换只为旧 π0；所选 π0.5 不在输出端加回 state。
5. state 仍提供 batch shape，且经 Normalize/PadStatesAndActions 走通张量接口；不能因此把它称为“真实 proprio 输入”。模型 YAML 的 `use_proprio=True` 不是实际条件路径的充分证据。

Wan 观测证据：[`world_model/env.py:574–585`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L574-L585)。默认双相机证据：[`libero_policy.py:56–88`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openpi/policies/libero_policy.py#L56-L88)。

## 最小、明确的一视角适配

用户现已明确采用“一视角 π0.5 适配”，本地实现位于 `local_scripts/wan_goal_20260930/pi05/`：

1. `LiberoInputs` 增加显式 `wrist_mode`，建议取值 `required` / `disabled`，默认 `required` 保留原行为。`disabled` 不读取腕图，使用同形状零数组占位，并令 `left_wrist_0_rgb` 的 mask 为 false；右腕仍为 false。主图始终有效。
2. `LeRobotLiberoDataConfig` 增加并传递这个字段。现有 `openpi_data` → `dataclasses.replace` 路径可传参，不需要改 GRPO 或 Wan backend。
3. 专用 YAML 显式设 `actor.model.openpi_data.wrist_mode: disabled`，train/eval 同样屏蔽腕相机；这样真实 LIBERO 评测也测的是同一一视角策略。不要仅在 missing 时关闭而让真实 eval 自动恢复双视角。
4. 明确指定该 checkpoint 的 `physical-intelligence/libero/norm_stats.json`，避免缺失统计时只告警、跳过归一化。

实现补充：disabled路径在Normalize之前生成8D零占位state，再由原生Pad扩到32D；明确限定PI05、discrete_state_input=false、extra_delta_transform=false。固定HF stats中state/actions的统计实际均为32维，原Normalize会按输入末维截取，故原Wan16D并不直接导致广播错误；8D规范化是明确契约，不是伪造状态预测。默认required路径不改原state。

**mask 的精确含义：**当前实现仍对零占位图做 SigLIP 计算，并保留 prefix 张量槽位；mask=false 使有效图像、语言及动作不能关注这些 token。它不等于节省掉这部分视觉计算。无需为了这个实验再改网络的 token 拼接。

依据：[`make_attn_mask:39–53`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openpi/pi0.py#L39-L53)、[`embed_prefix:205–219`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openpi/pi0.py#L205-L219)、[`prefix/suffix attention:425–463`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openpi/pi0.py#L425-L463)。`preprocess_observation` 默认要求三类 image key 均存在，故只删腕图 key 会再次报错。

这保持模型参数结构和原权重，但减少输入信息，属于新增实验配置，不是官方双视角 π0.5 原样复现。复制主图到腕图、把黑图 mask 设 true、保留 reset 时的旧腕图，都不能补出真实腕部观测。

## 动作：H10 / C8 / M5

| 设置 | 官方真实 LIBERO Goal π0.5 | Wan π0.5 候选 |
|---|---:|---:|
| 模型生成 horizon H | 10 | **10，保留** |
| 每次执行/计 logprob 的 chunk C | 5 | **8，与 Wan 固定 chunk 一致** |
| flow 步数 M | 5 | 5 |
| 内部动作维数 | 32 | 32 |
| 环境动作维数 | 7 | 7 |
| Wan 条件帧 / 总帧 | 不适用 | 5 / 13 |

[`factory:344–376`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openpi/__init__.py#L344-L376) 已拆开网络 horizon 与执行 chunk；不需要重写 sampler。显式设置 `actor.model.openpi.action_horizon=10`、`actor.model.num_action_chunks=8`、`actor.model.openpi.action_chunk=8`，网络仍生成 10 个动作，只送前 8 个给环境。[`tasks/rl.py:279–300`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/models/embodiment/openpi/tasks/rl.py#L279-L300) 同样只记录前 C×7 的 logprob。

环境 worker 以 `max_steps_per_rollout_epoch // num_action_chunks` 计循环；因此执行 chunk、Wan chunk、输出动作与 logprob 的 C 必须统一为 8。320 步正好为 40 个 chunk；不能保留 C5 而假定 Wan 自行补齐。

`world_model + wm_env_type=libero` 已复用真实 LIBERO 的动作后处理；OpenPI 输出不会被套用 OFT 的额外 gripper 翻转。依据：[`action_utils.py:75–89`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/action_utils.py#L75-L89)、[`329–360`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/action_utils.py#L329-L360)。

## GRPO 配置：保留 π0.5 官方 Goal 参数，单独列环境变更

这是“π0.5 官方 Goal GRPO + Wan 环境 + 一视角适配”的本地专用配置，**不是已有的官方 Wan π0.5 YAML**。已查固定提交的 Wan GRPO YAML 是 OFT Goal/Object/Spatial。

| 类别 | 沿用 π0.5 Goal 的设置 |
|---|---|
| GRPO | G8；normalize advantages=true；update_epoch=1；adv_type=grpo；loss_type=actor；token-mean |
| 颗粒度 | reward_type=chunk_level；logprob_type=chunk_level；entropy_type=token_level |
| 正则/裁剪 | KL beta=0；entropy bonus=0；clip high/low=0.2/0.2；clip c=3；gamma=0.99；gae_lambda=0.95 |
| 奖励尺度/过滤 | reward_coef=1；filter_rewards=true；lower/upper=0.1/0.9 |
| flow 策略 | flow_sde；noise_level=0.3；M5；train_expert_only=true；不加 value head |
| 优化器 | LR=5e-6；Adam beta=.9/.95；eps=1e-8；weight_decay=.01；clip_grad=1 |
| 训练预算参考 | total_num_envs=64；rollout_epoch=8；max_episode_steps=max_steps_per_rollout_epoch=320；seed42 |
| actor batch 参考 | micro_batch_size=128；global_batch_size=2048；no_shard；gradient_checkpointing=false |
| 真实环境评测参考 | LIBERO Goal，500 env，1 rollout，320步，G1，fixed reset IDs；单独显式触发，官方 val_check_interval=-1 |

需要的环境/接口变更：

- train defaults 换 `env/wan_libero_goal`，eval 保留真实 `env/libero_goal`；指定固定 Wan 路径。
- `env.train.chunk=8`、condition_frame_length=5、num_frames=13、image_size=[256,256]、num_inference_steps=5、enable_kir=true。
- `env.train.reward_coef=${algorithm.reward_coef}` 明确保持 1；不混用 OFT 的 reward_coef=5 与过滤 [.5,4.5]。
- `env.train.group_size=8`；总环境和每个进程的环境数都需能容纳完整 G8，组内相同 reset 条件。实际 worker 切分启动前以 resolved config 确认。
- 按 Wan 官方配置开 env/actor/rollout offload 是资源调度选择。micro-batch/并行数如因四卡内存需要改变，需记录与上表的差异，不把显存是否够用写成已验证。

官方参数来源：[`libero_goal_grpo_openpi_pi05.yaml`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/examples/embodiment/config/libero_goal_grpo_openpi_pi05.yaml)、[`pi0_5.yaml`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/examples/embodiment/config/model/pi0_5.yaml)、[`wan_libero_goal.yaml`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/examples/embodiment/config/env/wan_libero_goal.yaml)。

Wan 奖励来自奖励模型对生成帧的分数，默认返回逐帧分数差；max chunk score≥0.9 时估计成功，并在 chunk 末位标 termination，超时单列 truncation。它不是物理仿真成功判定。G8 初始数据集 ID 会 repeat_interleave；当前 backend session seed 接口全部传 0，源码明确尚非每轨迹独立 WM seed。应记录这套原生语义，不额外重写或声称与原物理环境完全等价。依据：[`env.py:232–275`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L232-L275)、[`465–473`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L465-L473)、[`607–666`](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L607-L666)。

## 一次有意义的适配验收

后续在深圳3执行，不在 Windows 跑项目模型：

1. 用同一主图、prompt、初始 noise，把 state 占位值及被 mask 的腕图内容分别改变，π0.5 eval 动作应保持一致；有真腕图且 required 模式应维持原始双视角路径。
2. 记录生成张量 H10×32、环境动作 C8×7、logprob C8×7，以及 actor replay 与 rollout 相同的 mask/normalization 配置；避免只测推理而漏掉训练重放输入。
3. 完成一个含非零有效 advantage 的 GRPO 更新，保存参数变化/梯度与 checkpoint 证据；如果全部组被过滤，应记录为没有有效更新，不能只凭进程退出码称训练成功。
4. 分开报告 WM 奖励模型估计成功与真实 LIBERO 成功率；一视角适配前后使用同一真实评测观察设置。

本地实现仅改两份上游文件的输入适配，采用固定SHA和原文锚点的幂等应用脚本；未改网络、梯度或GRPO。smoke为N32/R1/L320/B1280/MB64/G8、两epoch各一次optimizer更新；formal为N64/R8/L320/B2048/MB128/G8/U1、1000epoch，独立从固定SFT开始。正式每epoch20480个chunk样本、10次optimizer更新。YAML显式物理placement `'4-7'`；RLinf通过NVML枚举整机卡，不能靠Ray四卡参数推断逻辑映射。部署、CPU/GPU验证及有效更新验收由主实施器在服务器完成，结果另写runlog。
