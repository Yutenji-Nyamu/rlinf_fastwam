# RoboTwin WM-RL：三者组装后的方法边界与升级顺序

2026-10-03。范围：实时核对论文与官方代码，讨论工程组合；未操作服务器、下载模型或改变现有WMRL。组件发布物的详细核验见[前一轮记录](../robotwin_options_20261003/robotwin_prior.md)。

## 1. 三者接起来，首先得到什么

**固定OpenDW＋任务奖励模型＋RLinf π0.5-GRPO，首先得到“OpenDW中的π0.5-GRPO基线”。它属于基于世界模型的强化学习，结构接近WMPO/WoVR的固定WM阶段；还不能称为完整WMPO、WoVR或VLA-MBPO复现。**

```text
真实轨迹里的起始三视角、关节状态
        ↓
π0.5提出动作块 → OpenDW预测未来三视角 → 任务RM估计是否成功
        ↑                                      ↓
        └──────── RLinf计算组内优势并更新π0.5 ──┘

此阶段OpenDW与RM冻结；最终收益在原生RoboTwin中另行评估。
```

这里的“MBRL”是大类；WMPO、WoVR、VLA-MBPO、RISE是具体方法。组件名字相同或都用了WM，不等于训练方法相同。

| 来源 | 已有什么 | 本方案借哪几块 |
|---|---|---|
| OpenDW | Robotwin模型包、三视角视频/动作模型、外部动作条件入口、训练代码 | 只用外部π0.5动作预测未来视频的能力，暂不采用OpenDW自己的动作生成器 |
| WorldArena 2.0 | WM评测基准；RLinf衍生的RoboTwin环境、HTTP服务、π0.5/GRPO、两任务reward资产 | 借环境输入输出约定、服务拆分和匹配任务的RM；逐文件适配 |
| 我们的RLinf | 已用的π0.5 actor/rollout、分布式训练、保存及原生评估流程 | 保留训练主线，新增OpenDW环境适配 |

OpenDW的真实外部动作入口是[`rollout_video_with_actions`](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L542-L604)；WorldArena工程与任务reward入口见[官方README](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/README.md)。无需把几个完整RLinf fork合并；依赖版本和训练主干差异会扩大验证范围。

## 2. 与各方法差在哪，之后加什么

下表的投入是相对于“基础接口已验证”的工程判断，不是实际工期或显卡预算。

| 方法/机制 | 真正增加的东西 | 从这条基线出发的投入 |
|---|---|---|
| WMPO | WM先学习目标策略的成功/失败行为；GRPO动态补样，去掉组内全成/全败后继续采到有效批量够数 | 补样中等：改采样循环及批量组织，增加WM调用；策略对齐较大：需采轨迹并微调WM |
| WoVR的KIR＋masked GRPO | 从关键中间帧开始，减少不可靠的长滚动；首次成功后不再训练后续步，并按有效长度归一化 | 相对近：现有RLinf已有相关训练组件，新增OpenDW重置数据与done语义适配仍需验证 |
| WoVR的PACE | 策略训一阶段后回原环境采新行为，再更新WM，然后继续训策略 | 较大：数据导出、WM微调、验证、模型版本切换与阶段调度；不是一个开关 |
| VLA-MBPO | 离线真实状态出发的短分支；多模态WM联合预测下一观察和reward；Flow-Noise PPO＋value/GAE | 较大：分支数据、critic/回报、截断处bootstrap、PPO训练契约都要对齐；换成论文WM另计 |
| RISE | 三视角动力学＋进度/TD value；把估计优势作为策略输入，用优势条件流匹配更新策略 | 较大：先训练value、标注离线数据并预热优势条件策略，再改训练目标与在线离线混合 |

### WMPO：最相近的骨架，但不是已经复现

WMPO把目标策略行为对齐与动态采样作为实际配方：成功和失败轨迹帮助WM覆盖策略会到达的状态；全成/全败的组对GRPO没有区分信号，动态采样会继续补充新组。[论文](https://arxiv.org/html/2511.09515v1)

官方代码确有`while len(valid_batch) < batch_size * n_samples`，过滤后继续采，直到凑满；这与固定批次里把无效组的loss置零不同。[采样与过滤循环](https://github.com/WM-PO/WMPO/blob/c836d74ec6f4525c93fe980d54d0ca870118615a/verl/trainer/ppo/ray_trainer.py#L566-L639)

对我们而言，可以先借补样机制，但应同时记录生成总量、保留组数和耗时；否则“有效batch相同”掩盖了更大的计算预算。OpenDW和π0.5替换了论文模型，适合称“借鉴WMPO”，不直接套其结果。

### WoVR：最近的升级是交互约束，PACE属于数据闭环

KIR选择关键中间状态缩短有效预测深度；masked GRPO屏蔽首次成功后的部分并进行有效长度归一化。完整WoVR还包括WM内部的动作控制与首帧锚定。PACE是在初次策略更新后，用新策略少量真实轨迹再训练WM，论文采用一次或低频更新，并非每个RL iteration都训Wan。[论文§4](https://arxiv.org/html/2602.13977v1#S4)

官方RLinf已提供KIR初始化数据选择、Wan环境与GRPO配方：[Wan文档](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/docs/source-en/rst_source/examples/embodied/wan.rst#L177-L263)、[Goal配方](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/examples/embodiment/config/wan_libero_goal_grpo_openvlaoft.yaml)。这支持复用接口与机制，不代表OpenDW适配完成，也不代表已有自动PACE闭环。

KIR不是任何“从中间开始”都算完成：需要与动作历史、三视角和关节状态对齐的关键帧包。仅保存reset画面，不能替代训练WM的完整动作轨迹。

### VLA-MBPO：短分支可以借，完整方法还换了学习信号

VLA-MBPO的WM用Bagel统一预测下一观察与reward，通过交错多视角解码维持一致性；从离线真实观察出发，只向前滚动少量动作块，减少误差累积。策略采用Flow-Noise PPO和value/GAE。[论文](https://arxiv.org/html/2603.20607v1)

官方实现是OpenPI/JAX：[`train_libero_rl.py`](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/scripts/train_libero_rl.py#L224-L310)同时计算PPO actor和value loss；[`rl_env.py`](https://github.com/LAMDA-RL/VLA-MBPO/blob/5e2b21044f2ef2189c22777643cf10e80a83bc13/src/openpi/training/rl_env.py)有GAE、离线观察起点和WM交互，LIBERO调用传7D动作。不能直接接RoboTwin14D就算跑通。

我们可以独立研究“短分支＋现有GRPO”，但若reward仅在最后成功时给1，分支太短就可能全部为0、再次失去组内优势。需要关键起点、经过验证的密集信号，或value bootstrap；只把horizon调小不能保证解决问题。

### RISE：价值模型能借，完整策略训练变化更大

RISE的value先用时间进度训练，再结合成功/失败轨迹做TD学习；预测未来相对当前的价值改善，用它标注动作块的优势。策略既接观察，也接优势条件，再做流匹配学习；动力学与value在策略自改善阶段保持冻结。[论文§III与附录XI](https://arxiv.org/html/2602.11075v1)

**官方代码的重要边界：**发布配置虽保留`adv_type: gae`、`loss_type: actor_critic`，实际[`train_mode: IL`、`add_value_head: False`](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/examples/embodiment/config/rl_release.yaml#L155-L217)；actor传入`conditional_advantage`后走[`F.mse_loss(u_t, v_t)`](https://github.com/OpenDriveLab/RISE/blob/5fac1e6ab9d50d4cc1ba4daeadf14023c8415955/policy_and_value/policy_online/rlinf/workers/actor/fsdp_actor_worker.py#L829-L868)。因此RISE不能简单描述为“给GRPO换个奖励模型”。

可以先借“独立进度value”的组件，但移植后仍须验证真实与生成图上的进度/失败识别。完整RISE则还包括优势条件输入、预热数据、训练loss和经验混合，宜作为单独实验分支。

## 3. 推荐工程顺序

1. **先接最小基线。** 验证三视角输入输出、14D绝对关节动作归一化、时序、RM与state。OpenDW当前公开推理默认batch=1，默认32动作对应8张未来帧；它不直接预测next proprio，也不返回reward/done。不能把C50、N64或零state默认为已支持。[OpenDW接口及已核缺口](../robotwin_options_20261003/robotwin_prior.md)
2. **先验证世界模型里的信号，再扩训练。** 用同一批真实轨迹比较生成动作响应、成功/失败判断与原生结果；WorldArena的任务RM要校验图像裁剪、归一化、任务标签和阈值。这个阶段验的是环境可信度，不以生成视频看起来流畅作为通过依据。
3. **保持GRPO主线，逐项加交互机制。** 优先复用成功后mask与有效长度处理，再研究KIR、有效组补样。分别量化有效loss比例、WM总调用量及原生成功率；不要同时改变多个机制。
4. **分布偏移有证据后，再做WM行为对齐/PACE。** 这是数据和模型训练工作。收集π0.5的成功与失败动态轨迹、转换OpenDW训练数据并验证后，才有可切换的新WM。
5. **短分支/value PPO和RISE作为不同研究方向。** 二者都可借现成代码，但改变回报、优势或策略训练目标，投入明显高于单纯环境接线；不必为“凑齐名字”同时加入。

**state范围补充：**本轮深圳2 RoboTwin源码审计（HEAD `0008ae6800df9f75fc8de7098bacb01735fd8fd2`）确认，`get_obs()`的`joint_action.vector`经`get_*_arm_jointState()`读取12轴`drive_target`与缓存的夹爪命令，并非实测qpos。因此OpenDW不预测实测next proprio，不等于必须新训state预测器；应先对齐实际执行的目标值、观察时刻及夹爪裁剪，再判断能否按同样控制目标更新state。不能直接把任意动作块的末项视为已验证的下一state。[接口审计](interface_audit.md)

成功RM、PPO critic和RISE进度value不是同一个角色：前者判断当前成败，critic估计未来累计回报以支持bootstrap，RISE的value用于产生条件学习的优势。OpenDW尚未发布完整Value Expert，不妨碍基础GRPO，但不能据此省略后两类方法需要的value训练。

## 4. 与现有LIBERO WMRL的关系

当前已跑通的LIBERO π0.5＋冻结Wan线，与本次RoboTwin/OpenDW组装研究分开。它已有的KIR/GRPO及OOM修复、micro64、每10轮保存、监控、未来C8单双相机评估，沿用既定范围；本文件不启动、暂停或切换任何训练。是否正在训、资源归属与进度，以根agent本轮现场回执为准。

研究命名建议：先用“OpenDW＋任务RM＋π0.5-GRPO”；后续在实验名里逐项标`+KIR`、`+dynamic sampling`、`+policy-aligned WM`。这样每项收益和额外成本可追溯，也避免把组合方法误写成完整论文复现。

## 固定版本

本轮官方API读到：RLinf `c70606f0`（10月2日）；WMPO `c836d74e`（1月4日）；VLA-MBPO `5e2b2104`（8月31日）；RISE `5fac1e6a`（6月3日）。OpenDW/WorldArena模型与源码revision沿用同日已实时核验的[发布物表](../robotwin_options_20261003/robotwin_prior.md)。本文引用均指向官方论文、仓库或固定源码。
