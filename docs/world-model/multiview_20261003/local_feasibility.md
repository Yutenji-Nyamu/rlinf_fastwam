# 双相机 WMRL：现有 RLinf / Wan 的具体接口与可行路线

2026-10-03。本文件只做源码与公开资料核查；未 SSH、未改训练代码、未运行训练。当前实配事实复用 [实现审计](../audit_20261003/implementation_audit.md)；公开 main 本轮重新核为 `c70606f08cdca259b8dec03d4430926b5b8fac9d`。本轮范围不改变已批准的单图 WMRL 修复、N/G/R/C 或预算。

## 结论

**π0.5 本来能接双图；缺的是一个能根据同一段动作，同步生成主相机和腕相机未来画面的 WM。** 当前 RLinf action-Wan 的数据、生成及环境返回都是单图。Wan 这一模型家族没有“天生只能单相机”的限制；RLinf 已有基于 Wan 的 DreamZero 多图拼接先例，但那是训练动作策略的 WAM，尚不是当前 π0.5 外挂的动作条件模拟环境。

**自己接并训练是现实的研究扩展。最接近当前链条的方案是同步双图拼接、联合预测、输出再拆回两路。** 需要成对动作轨迹和 WM 微调，不能只在推理时把第二张图塞进去。生成的两路画面还必须描述同一个物理状态：看起来各自逼真，不代表可供策略可靠学习。

## 当前具体卡在哪里

|层级|直接源码事实|改双图要做什么|
|---|---|---|
|策略 π0.5|`pi0.py` 按 image keys 遍历各相机并编码，当前腕图是零填充后 mask=false|保留原主/腕图键、resize/翻转/归一化与mask语义；把正确腕图送回原槽位|
|reset 数据|dataset wrapper 虽有 `camera_names` 参数，但 WM env 默认构造不传；条件窗口只取 `frame["image"]`|成对读取每一时刻主/腕图；initial、KIR参考帧与最近4帧也要成对|
|Wan调用|`input_image` 是长度B的batch，`input_image4` 是 B×4 时间历史，不是相机维|不能把两相机当batch后声称联合多视角；要定义联合表示或专门view维|
|生成分辨率|Wan backend `_pipe_kwargs` 直接写 `height=256,width=256`|改成实际联合画面几何；不能只改YAML的image_size|
|动作注入|DiffSynth active five-frame分支把动作特征空间扩展写成 `repeat(1,1,64,1)`|按真实patch网格计算空间token数；256×512时不能继续64|
|环境返回|`current_obs` 中V轴实际为1；返回 `main_images=full_image,wrist_images=None`|联合图输出后正确裁切；或系统性扩展view轴、history和reset逻辑|
|奖励|当前ResNet按单主图训练；接口接一个RGB tensor|可以先仅给它主图裁切并保持原预处理；不能把整张双图直接喂旧RM当作无变化|

固定证据：

- [Wan backend：batch语义、256高宽与单视频输出](https://github.com/RLinf/RLinf/blob/c70606f08cdca259b8dec03d4430926b5b8fac9d/rlinf/envs/sim/world_model/backend/wan.py#L162-L224)
- [固定运行版env：单图生成、评分、wrist=None](https://github.com/RLinf/RLinf/blob/d34d4c320d08cb982de034aa9a011f08dc0fa217/rlinf/envs/sim/world_model/env.py#L503-L587)
- [action空间广播常数64：active分支L1830、1842](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/diffsynth/pipelines/wan_video_new.py#L1822-L1844)
- [DiT：action MLP、通用patchify及网格重建](https://github.com/RLinf/diffsynth-studio/blob/2a2e05fa1f724828b243f272540989b19a6e54f8/diffsynth/models/wan_video_dit.py#L398-L472)

本轮 latest WanBackend SHA256仍为 `72271ececc307211196a4c3f7a595f48cea21e8d3440e502a6dd7a893c058d14`，与固定运行版相同。通用env虽发生整理，仍仅返回单图。当前 OpenSora backend也走同一单路视频返回契约，换OpenSora并不自动补腕图。

## RLinf 里面有没有可借的双图实现

**有：DreamZero的数据管线。** 官方文档明确 `libero_sim` 为双视角，`concat_multiview_video` 把view维合为一张画面；`target_video_height/width` 是拼图后的尺寸。此管线有Wan2.2 5B权重与LIBERO评估。它证明多视角拼图+Wan在这个框架内已是可实现路径；但当前支持的是WAM策略训练与评估，不是将外部π0.5任意动作作为已知控制量的WM backend。[官方DreamZero说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sft_dreamzero.html)

可借的是相机布局、数据读写、resize与输出接口做法；不能直接把DreamZero权重改路径加载成当前WoVR Wan。两者action注入、预测目标、时间块与状态条件不同。要改成环境，还要有动作钳制/条件预测、rollout history、reset和奖励接线，并验证任意策略动作下的预测。

## 四条路线的工程判断

|路线|会发生什么|相对判断|
|---|---|---|
|继续单图Wan，π0.5补单图SFT|让策略适应缺腕图|工程最小，但不保留原双图使用方式；当前基线全零原因须先解释|
|**双图拼接，微调当前action-Wan**|主图与腕图作为固定左右布局的同一视频联合去噪，输出再拆开|**最接近当前实现，优先讨论的工程路线**；不用从零训练视频基础模型|
|共享DiT、两路latent/视角编码、跨视角attention|每路保留完整图像坐标，视角间交换信息|表达更明确，但训练、checkpoint、attention mask及并行都要新增，工程更高|
|两个独立Wan共用动作|每路都按同一动作生成|接线表面简单，但共享action不保证同一物体位置、接触和遮挡；不宜当作默认可靠方案|

同一随机种子也不能保证两个独立模型的状态一致。把主图复制/裁成腕图、固定使用初始腕图，或根据单张主图做简单几何变换，都不能提供随末端运动变化且包含遮挡后信息的真实腕视角。单图可用于学习视角补全，但要另训模型及处理不可观测信息，不是免费的相机替代。

这些是基于模型结构的工程推断，不是已跑出的比较结果。拼接法同样不能保证几何一致，只是让同一模型有机会利用联合注意力和成对数据学到一致性。

## 双图拼接方案具体训练什么

1. **数据先对齐。** 在原生LIBERO从同一物理状态同步获取主/腕两图及实际执行的7D动作、时间索引、任务和原生成功标签。成对专家演示可以作启动材料，后续应包含目标π0.5策略的成败轨迹，避免只看到理想专家行为。旧742个单图reset文件不能凭空补出隐藏腕视角；若没有可恢复的完整模拟器状态，就需要新采成对reset/KIR。
2. **从现有action-Wan微调。** 固定布局拼接双图；VAE先冻结；训练DiT与动作条件部分，或者研究LoRA+明确可训练的action MLP。现有LoRA默认q/k/v/o/ffn列表并不会自动覆盖action MLP，checkpoint导出与合并也要接好。训练策略π0.5不是获得双图WM的必要前置步骤。
3. **输出恢复原π0.5协议。** Wan生成联合视频后拆回 `main_images` 与 `wrist_images`，分别走原图像处理，保留原相机顺序和mask；不要把拼图作为π0.5的一张主图，后者会再次改变其输入分布。当前π0.5版本并不实际使用proprio条件，不必为了双图自动扩展成预测机器人state的新研究项。
4. **奖励可保持单主图。** 只用生成主图裁切给现有RM，保持原主图分辨率与预处理。若将RM改为双图输入，需要重新训练/校验；双图WM本身不要求立即更换奖励算法。
5. **再接策略RL与PACE。** 新WM先要说明动作响应和两路状态是否可信；之后才用它训练π0.5，策略变化后可沿同一成对轨迹格式做一次PACE更新。

这里的“保持π0.5双图”只保证输入契约相同，不能保证生成图像分布与真实相机完全相同或最终成功率不退化。双图WMRL应视为新方法版本，不把它暗中混入本次只获准micro64/屏障/监控/频率的修复。

## 分辨率与资源代价

若每路都保留256×256，左右拼图为高256、宽512：

- 模型参数量和主干权重显存不因加宽自动翻倍；每段视频像素、VAE latent及DiT空间token约翻倍。
- 当前5B下，每时刻空间token由64增至128，因此上述action广播64必须修正。
- 许多激活随token数近似线性增长；全局attention的乘加量随序列长度平方增长，翻倍token时该部分约4倍。FlashAttention不会等比例保存完整注意力矩阵，因此**不能说总显存必然4倍、总耗时必然4倍**。
- 如果把双图压回总256×256，代价是每视角有效像素减少或发生拉伸；腕图的接触细节正是想补回的信息，不能忽略这个损失。
- 当前head-only适配已对黑腕图做视觉编码后再mask，恢复真实腕图不必然新增整路actor计算；主要新增负担在WM联合生成。实际tensor keys和profile仍需核对。
- LoRA减少可训练梯度/优化器状态，不会消除视频激活；当前4卡冻结WM能跑，不能直接推导4卡双图WM全参数训练容量。公开action-Wan全参数脚本8进程只是示例，不是最低卡数证明。

资源与可行性的区别是：**代码改造有明确落点，模型不需从零学；达到可用动力学和跨视角一致性仍依赖数据与训练。** 目前没有足够证据给出可靠工期、最低卡数或保证所需轨迹数。

## 本轮轻量证据

- 读取脚本：`.tmp/wm_audit_20261003/fetch_multiview_local.py`
- 原始公开源码与SHA清单：`.tmp/wm_audit_20261003/multiview_source/manifest.json`
- source包含latest main提交、Wan backend/env、固定DiffSynth DiT/pipeline/camera模块；未下载模型权重。

`wan_video_camera_controller.py`里的camera control用于控制生成视频的相机轨迹，也不等于同步产生主/腕两路机器人相机。这类关键词需与真正多视角动作条件环境区分。
