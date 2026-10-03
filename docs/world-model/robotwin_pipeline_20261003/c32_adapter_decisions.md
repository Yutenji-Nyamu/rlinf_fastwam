# OpenDW × Sidney：C32 接线决定与 smoke 合同

更新：2026-10-03。首任务 `adjust_bottle`；执行 C32 已获本轮授权。本文把“接口已经有确定解法”与“模型效果仍需实测”分开。后续用户允许在 smoke 中调整串并行，因此 N 不再写死；G8 与动作语义保持。具体现场资源、命令、测试结果由执行回执记录。

## 1. 可以直接实现的部分

| 部分 | 本轮确定做法 | 是否还需要讨论 |
|---|---|---|
| 策略动作 | Sidney 原模型仍预测 H50；环境仅执行前 C32 | 不需要；两种长度在现有 RLinf 中本来分开 |
| GRPO logprob | rollout 和 actor 重算都使用前32动作、14维；保留完整H50 denoise chain | 不需要新算法；不能只在env里截动作而保留C50 logprob |
| 14D含义 | 左臂6关节＋左夹爪＋右臂6关节＋右夹爪；绝对目标 | 两侧一致，没有末端位姿与关节角冲突 |
| 两套归一化 | policy按Sidney stats输出raw命令；OpenDW按自己bundle stats做z-score；状态另走各自state stats | 正常适配步骤，不是模型冲突 |
| 三路相机 | 头、左腕、右腕；按OpenDW `robotwin_resize`拼图，再拆回三图给Sidney | 布局明确；不用WorldArena单图/零state输入 |
| next state | `state_next = raw_actions[:,31]`，夹爪索引6和13裁剪到[0,1] | 作为当前WM环境命令态约定可直接用，不先训练state预测器 |
| reset/组 | 真实同刻三图＋14D命令态；G8同组共享样本，各环境独立WM随机数和缓存 | 实现确定，不需要增加专家筛选 |
| 首任务奖励 | 复用WorldArena摆瓶子ResNet＋T5对应权重；score→差分reward，阈值0.9 | 模型在生成图上的可靠性需记录，但不前置成一轮额外研究 |
| done | 任一未来帧达阈值记success，整C32结束时done；另记首次命中的4动作粒度位置 | 与参考环境一致；本轮不做chunk内提前截断 |
| 并行 | N个逻辑环境与G8保持；B1视频调用可排队，后续smoke比较多服务/更大batch | 不要求N份模型驻卡，吞吐/显存实测决定 |

新文件已经按该合同准备在 `local_patches/opendw_smoke_20261003/`；“已写代码”不代表已通过真实GPU smoke。不会为保证非零GRPO优势伪造奖励或擅自筛有利初态。

## 2. 为什么 state 可以这样接：一个具体例子

π0.5每次读三张图，以及14个数字。14个数字里，左臂第一关节假设原目标是 `0.20 rad`，新动作要求去 `0.30 rad`。

- **命令目标**是 `0.30`：控制器要求它去哪里。
- **实测关节角**可能是 `0.24`：夹到东西、正在途中时，它现在实际在哪里。
- 已核的RoboTwin `get_obs()` 给本项目Sidney的字段是前者——关节 drive target，不是另一个 real-joint getter 的实测qpos。

所以我们缺的并非“如何从视频精确量出0.24”，而是OpenDW没有替我们返回Sidney下一次要读的命令字段。对于本轮完整执行的32条绝对命令，直接把第32条写回即可。例如末命令为 `[左6关节, 1.2, 右6关节, -0.1]`，下一state为 `[相同左6关节, 1.0, 相同右6关节, 0.0]`。

这是一条**明确的WM状态更新约定**，不是证明世界模型准确模拟了物理控制器。真实RoboTwin的TOPP失败、提前成功、插值噪声可能让返回drive target没有完全抵达请求末点；纯视频WM本来不输出这些物理事件。本轮采用完整chunk、边界done，不凭空构造TOPP失败。若之后实测发现这项近似影响结果，再讨论控制器复现或小型残差状态模型。

可选方案按当前目标排序：

1. **末命令更新（本轮）**：无需数据训练，符合现有policy字段和OpenDW demo惯例。
2. **控制器或残差校正**：若发现目标态偏差，复刻接受/裁剪规则，或用真实转移监督小模型；增加代码和数据，但不换policy契约。
3. **联合预测视觉与state**：如WEAVER思路；现成Franka资产需转RoboTwin14D/三视角，不是当前smoke前置。

不给state、全零state、一直复制旧state都不是当前推荐；Sidney开启了离散state输入，不能随意删掉这个输入。

证据：[RoboTwin固定源码 get_obs](https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/envs/_base_task.py#L512-L519)、[drive target与real qpos两个getter](https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/envs/robot/robot.py#L528-L558)、[夹爪缓存/裁剪](https://github.com/RoboTwin-Platform/RoboTwin/blob/0008ae6800df9f75fc8de7098bacb01735fd8fd2/envs/robot/robot.py#L662-L693)。此前实时链和Sidney直通证据见[接口审计](interface_audit.md)。

## 3. H50保留、C32执行，不需要改模型预测长度

实施基点：`Yutenji-Nyamu/rlinf_fastwam@2151a08ee1bd75df1bef0d8190e594bd5c7f7977`。本轮从本地固定git对象重新检查了以下四条链：

1. `pi05_sidney_robotwin` 保持 `action_horizon=50, discrete_state_input=True`。[dataconfig L454–465](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/models/embodiment/openpi/dataconfig/__init__.py#L454-L465)
2. `actor.model.num_action_chunks=32` 经YAML插值成为 `openpi.action_chunk=32`；不覆盖H50。[pi0_5.yaml L26](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/examples/embodiment/config/model/pi0_5.yaml#L26)
3. 完整动作反归一化后截前32；rollout旧logprob同样截前32/前14维。完整H50 chain留给重算。[输出 L346–361](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/models/embodiment/openpi/openpi_action_model.py#L346-L361)、[rollout logprob L1115–1126](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/models/embodiment/openpi/openpi_action_model.py#L1115-L1126)
4. actor重算也截前32/14，再用既有chunk-level聚合。因此尾18动作不直接计入该chunk的logprob；H50网络内部仍可联合影响其前32输出，这是原模型的正常计算。[actor L738–755](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/models/embodiment/openpi/openpi_action_model.py#L738-L755)、[聚合 L357–364](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/2151a08ee1bd75df1bef0d8190e594bd5c7f7977/rlinf/algorithms/utils.py#L357-L364)

因此模型侧是配置适配，不需要重训H32模型，也不需要修改Flow-SDE公式。完整pipeline还需新增env/backend，这不是只改一行配置就已经存在的环境。

首smoke采用1个C32 chunk时，episode/rollout都是32动作；随后串行长度由smoke配置决定。正式400动作不是32的整数倍，尾16动作需要精确帧选择与有效logprob mask；当前smoke接口明确拒绝部分chunk，避免静默从400跑到416。该边界不妨碍本轮短smoke。

## 4. OpenDW输入是raw14D；图像是直接resize

官方固定源码 `dexmal/opendw@e33befa8005a1585e0140dbf464566e90bc79aa1`，Robotwin bundle `Dexmal/DW05-Robotwin@6ab5f9e2636610cba440d08264663efe70c3f761`：

- bundle的14D stats触发 `zscore_14d` 与 `absolute`，不把动作减当前state；模型内部clip标准化值到±5。[模式选择 L326–349](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L326-L349)
- **同一模式自动选择 `robotwin_resize`**；不是默认函数签名里另一路letterbox。上图256×320，下方左右腕各128×160，总384×320。[图像布局 L145–181](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L145-L181)
- 直接给外部动作 `infer_joint(action=...)`；不用DW动作expert生成的动作覆盖π0.5。[外部动作调用 L577–598](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L577-L598)
- C32→9帧含初始；未来帧对应4、8、…32动作。用第32动作末图回到policy；不在50动作后补成64。[分块逻辑 L542–604](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L542-L604)

两模型的stats本来就不同：它们使用同一raw物理量各自编码。这类似摄氏温度先还原成同一温度，再按另一个仪表格式显示；不能把一个模型的标准化数字直接当另一个模型的关节角。

## 5. 奖励和done的最小接法

WorldArena摆瓶子RM看未来8张头图，输出8个score。每4动作才有新图，不复制4次奖励：把第j个score差分放到该4动作段的最后一个位置，其余置零。chunk-level求和仍是 `coef × (末分数 - 前chunk末分数)`。reset的previous score沿参考环境为0。

阈值0.9作用于原score。8帧中任一达到就记success；本轮在动作32结束时标done。`first_success_action`单独记录4/8/…/32，作为诊断，不悄悄改变chunk内执行或policy logprob。停止后不再调用WM，后续padding奖励为0。

该设计与WorldArena的分数差分和chunk末done同类；不保证score是真实成功。smoke记录分数、帧和有效组数；若全组同分、被filter全部过滤，报告“接口通过/无有效优势”，不能报告有效策略改善。[奖励差分与阈值源码 L229–258](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/envs/world_model/world_model_wan_env.py#L229-L258)

## 6. 进程隔离与HTTP服务合同

旧RLinf负责策略/GRPO；OpenDW和WorldArena RM放独立Python环境，避免升级旧RLinf的torch/transformers。每个服务仅加载一份WM/RM，多逻辑env可B1排队；env对象只存CPU canvas/state/RNG，不按N复制模型。

`POST /infer`：请求/响应均为NPZ二进制，`allow_pickle=False`。

| 方向 | 字段 |
|---|---|
| 请求 | `images uint8[B,384,320,3]`、`actions float32[B,32,14]`、`states float32[B,14]`、`seeds int64[B]`、`instructions Unicode[B]` |
| 诊断字段 | `env_indices[B]`、`reset_ids[B]`、`request_id`；服务应原样用于回执，不用来改变推理 |
| 响应 | `next_images uint8[B,384,320,3]`、`scores float32[B,8]`；score不乘coef |

`POST /onload`、`POST /offload`请求JSON `{}`，响应必须是 `{ok:true,is_offloaded:false/true}`。offload在模型移CPU、CUDA同步/释放缓存完成后回复，runner等待该回执再进入actor训练。入口限定loopback；不向公网暴露服务。

reset文件为NPZ：`main_images[K,H,W,3]`、`wrist_images[K,2,H,W,3]`、`states[K,14]`、`instructions[K]`；全部来自同刻真实RoboTwin观察，不从单图LIBERO reset补零造三相机。首个policy观察直接采用原三图，不先经过WM拼图的低分辨率再放大；只有生成图才从canvas拆分。G8重复同一个样本索引；policy随机性和WM seed分别留档。每环境只保留自己的末图、末命令和RNG；OpenDW本轮接口本身不需要额外长历史队列。

## 7. 还需实测，而非让用户继续替接口做决定的事

- 资产是否完整、模型是否能在当前授权卡上装入和卸下；首次B1时延与N条总时延。
- 根据真实显存/时延选择串行WM请求、多服务或batch实现；不能把逻辑N说成同一时刻N个视频推理。
- 固定seed下动作变化是否改变生成图；RM是否输出正常分数；是否有明显假成功。
- runner能否收到三图14state、rewards/dones、old/new logprob并完成既定优化调用。
- 1chunk下可能没有有效GRPO组；这属于学习信号结果，不靠改标签、加采样或变奖励掩盖。

本轮已准备5项针对CPU adapter的检查（视角身份、gripper/raw命令、差分总量和done时刻、32动作预算、非法输入）、4项模拟HTTP环境合同检查（原reset图不降采样、G8及独立RNG、状态/终止/padding/快照、worker到服务路由）及4项服务CPU合同检查（NPZ、非法输入、设备字段与VAE缓存、HTTP健康/推理/同步卸载）。测试应在服务器独立checkout执行。真实GPU结果由主执行回执补录。

## 8. 本轮服务实现

新增 `tools/opendw_service.py` 与仅含必要推理网络的 `tools/opendw_reward.py`。WM沿官方`DW05RobotWinPolicy`和`infer_joint`，固定10次推理迭代；RM保留ResNet18＋T5投影＋8头cross-attention＋原head和图像预处理，并以strict方式装完整任务权重。ResNet初始化不下载ImageNet权重，T5/tokenizer仅使用明确本地路径；不合并整个WorldArena框架。

服务先在CPU装载。`onload`移动WM、VAE、文本编码器与RM，并显式更新DW缓存的`model.device`；官方`.to()`本身未更新该字段。`offload`先调用VAE官方`clear_cache()`清掉可能因异常残留的普通列表特征，再将模型移CPU、同步CUDA、释放cache，核对参数/缓冲没有残留CUDA，最后返回完成回执。

生成前2条轨迹默认保留input/final PNG、9帧GIF、动作/state/score NPZ和元数据；每条推理及每10秒记录phase、GPU/NVML、torch显存峰值、RSS/PSS。`GET /health`在长推理期间仍能返回当前phase。该服务限定loopback、单个授权物理4–7之一；不自行新增卡或安装依赖。
