# 闭环数据合同与代码切口

2026-10-08；规划，不是已经存在的接口。本轮收敛：保留现用OpenDW单时刻三视角条件、chunk级学习，不加KIR/多时刻历史/新GRPO归约；每10轮原生补采并分别更新WM/RM。数值建议以[PLAN](PLAN.md)为准。

## 1. 先补原生轨迹，而不是拿MP4训练

本地`local_scripts/rynn_binary_20261005/native_binary_recorder.py`的reward_native模式只保存main_images原图、每C32边界action_steps和成功标签；没有三视角同步动作数组。`task_reward_20261006`的数据验证/切分/RM训练可复用，但旧奖励NPZ不足以直接微调WM。

“训练过程中自动保留”可以实现，但记录点要分清：平时每轮512条是WM生成轨迹，不能充当纠正WM的原生真值；在阶段性原生采集时保存完整轨迹。可复用原生评估执行器另跑采集seed，固定评估seed仍留出。新实验从原始π0.5开始，首批数据由它采集；不需要从已有RL断点起步，也不重置OpenDW预训练权重。

在原生RoboTwin逐动作执行处记录，策略仍每32动作调用一次，保持原控制逻辑。基础数据至少有：

| 字段 | 约定 |
|---|---|
| identity | task、actual/requested seed、episode、policy checkpoint/hash、场景/相机配置 |
| clock | 第t条动作前obs_t、执行action_t后obs_(t+1)；记录实际执行长度，禁止跨auto-reset拼接 |
| views | head/left/right的RGB uint8；至少每4个原生动作同步一组三视角，含起点与终点 |
| action_cmd | 每个实际提交的14D绝对控制目标；另记录提交前/裁剪后差别，若二者不同 |
| policy_state | 策略可见的14D command state；另存actual_qpos（如果采集可得），两者明确命名 |
| label | 每动作原生success、termination、truncation；首次成功时刻，不能用RM分数充当真值 |
| validity | 图像/动作时间索引、padding mask、有效步数、异常/中途失败记录 |

保存PNG/无损帧或等价无损数组，加动作/元数据；MP4只是预览。可以只保存每4步图像降低体积，但33个原始时间位置的JSONL转换器必须正确解析稀疏图像索引，不能把缺失帧静默复制为真实观测。

## 2. OpenDW真正的时间与数值合同

**易误读点：** 官方DWDataset的`num_frames=33`指覆盖32个动作的时间窗口；内部以stride4采图，输出是9帧，action_horizon为32。它与当前推理的“9帧/C32”正好对齐。不要把训练参数误设成9，那会变成8个动作的窗口。

### 必须修正：发布bundle与默认训练是两份配方

| 项目 | 当前已部署服务，续训必须继承 | 官方默认训练dataset/model |
|---|---|---|
| action/state维度 | 14/14；服务有显式断言 | 原始14→排列16→添加终止位17→padding32；模型默认32/32 |
| 动作语义 | 14D绝对命令 | `delta_first_frame`，夹爪保留绝对值 |
| 归一化 | action/state各自mean/std，z-score裁剪±5 | q01/q99分位数变换 |
| 图像 | `robotwin_resize`：主图256×320，下排左右腕各128×160，直接resize | robotwin的letterbox缩放/补黑边；未来图也走letterbox |
| 初始化 | 读取现用bundle权重及形状，再构建同形模型 | 默认构造32D模型后，trainer才加载resume权重 |

依据：现有`local_scripts/lift_two_gpu_20261006/reference/lift-pot-v1__generated__service__opendw_service_batched.py:330–341,477–487`，及官方policy/dataset源码；[10-03接口审计](../robotwin_pipeline_20261003/action_state_followup_20261003.md)已记录该区别。本规划初稿误把默认32D训练配方当作当前bundle合同，现纠正。不能仅设`data_config.action_type=state`，该参数在DWDataset构造时被丢弃；也不能只改model的维数，却保留32D数据变换。

**推荐切口：独立的14D dataset adapter，复用官方trainer/model/loss。** 直接读取实际提交的`action_cmd`和对应policy command state，调用与发布服务相同的统计解析、z-score和图像拼接；去掉默认的排列/终止位/delta/quantile/pad32。动作真值显式接入，避免绕道把action伪装成下一行state。模型配置明确`action_dim=14, proprio_dim=14`、相同backbone，再weights-only加载；这仍是现用WM续训，不改策略接口。

目标张量：video `[B,3,9,384,320]`，action `[B,32,14]`；proprio按官方训练器需要提供对应时间序列，实际条件使用起点14D command state。未来图像取`o[t+4],o[t+8],...,o[t+32]`，动作取`a[t:t+32]`。H50仅执行前32；策略自己的归一化与WM统计分开。

文字使用现有服务的`format_prompt`与同一T5编码器/有效mask，提前缓存即可；关闭或对齐默认prompt改写/子任务混选，缺embedding报错，不能静默全零。所有图像时刻沿用同一拼接和缩放，避免首图与未来图走不同预处理。

若选择继续沿用官方JSONL/AddAction路径，它使用下一行state作为当前action，必须逐项证明`state[t+1]`等于记录的已提交命令；不能用滞后的实测qpos代替。采用显式action adapter更直接。

稀疏存图时只从已保存的4步边界起窗，或保存逐步图像；不能让默认按任意行起窗的数据集读取不存在的t+1/t+5图像。C32是命令数，不是32个物理tick；逐条记录TOPP实际执行长度/失败，保持原生控制器不变。短尾按有效mask处理，不伪造完整32步。

## 3. 数据划分与失败轨迹

按actual seed/episode切分，再切window；同一条轨迹、同seed不同策略版本不能分到训练和验证两侧。现有原生评估32个seed永不进入WM/RM训练。保留成功与失败完整轨迹，不能只拿专家成功片段训练一个“什么都会成功”的模拟器。

DW05OutputBuilder在`robot_task_success=False`时把has_action设False；当前模型保留动作条件，主要用该标志屏蔽动作模仿loss，video loss仍存在。转换必须传真实标签，不能使用默认success=1把失败动作当专家行为。新适配还需一组CPU合同样本验证这一分支。`robot_task_success`是回合级结果，用于动作模仿loss开关；逐帧RM标签/首次成功时间另存，不能混用。失败回合早期的某个片段不等于该动作本身完全错误；首版保留官方整回合mask语义。

## 4. 微调和导出接点

- 官方`playground/example_dw_exp.py --task train` → DW05Exp → DW05Trainer可复用；须先使用上述14D dataset/model配置。随后`trainer_config.resume=<现用bundle/model.pt>`走weights-only：新建优化器，不恢复旧optimizer。默认32D模型并不会因设置resume自动变成14D。
- trainer生成`weights/<step>.pt`并单独存训练恢复状态；现有DW05Policy调用同类`model.load_checkpoint`。导出新bundle，仅将模型权重和明确变更的配置版本化；VAE/text/norm复用引用，不重复拷贝巨型文件。
- 官方默认训练video+action联合loss，且可训练范围包括MOT/backbone；**不能声称默认只训一个轻量adapter**。第一版保持已发布训练目标，未来若只训视频/LoRA属于额外方法变量，应单独列对照。OpenDW内部action expert参与MoT；“外部动作由π0.5给”不等于该内部模块可以直接删除。此处联合loss更新的是OpenDW，不是把原生数据用于π0.5的SFT。
- `DW05DataConfig._build_dataset`不直接依is_val分流；若使用单份annotations/recipe_entries，两边可能一样。要用真正不同的train/val recipe，或最小覆盖_build_dataset分别传清单，不能只设val_as_train=False就认为隔离了。
- 官方`DW05Trainer.evaluate`每次只随机选一个样本；`_to_batched_eval_sample`未保留`has_action/action_is_pad/image_is_pad/action_dim_mask`，所以其val_loss与训练有效区间/失败mask并非完全同义。需要窄改验证适配，传齐并batch化这些字段，再用固定留出清单汇总video loss及生成误差；该loss单独不证明物理正确。

## 5. 回接服务：按当前chunk行为，不新增模型机制

现有服务提供onload/offload/infer；在策略保存边界排空请求、释放旧服务、加载新bundle，记录策略/WM/RM版本。一个GRPO批次内不混WM或RM版本。阶段切换期间仍属于优先实验，RLT不插入。

成功所在C32 chunk及当前loss归约保持；后续chunk按现有终止逻辑停止。不增加块内first_success_action截断或论文式轨迹长度归约。保留这些字段供解释即可，不因其存在新增控制分支。

本轮不加KIR，不做回合参考图/近期历史改造。OpenDW的单条件图是一时刻的三视角拼图；完整真实轨迹存盘后，切成多个现模型支持的单起点/C32样本即可。保存完整轨迹不要求模型改成多历史帧结构。

WM与RM使用同一批原生采集分别训练。WM续训配置按当前14D发布模型，RM沿已有主图分类器；逐时刻成功标签不能替换成回合总成功。训练结束后在下个策略阶段统一使用新模型。只复用原有owner/候补边界，不新建常驻调度器。

## 6. 实施前仅需三组针对性验证（本轮未执行）

1. 同一真实C32窗口：adapter与现有服务的首图、prompt、归一化action/state一致；记录第4/8/…/32条动作后的图像索引。固定样本即可，不需要GPU长测。
2. 一成功一失败加一个短尾：成功/失败都保留动作条件和有效视频loss；失败动作模仿loss为0，padding不进入loss，train/val实际seed不交叉。
3. 一个完整训练step（包括optimizer.step）与一次导出回载：测真实峰值，确认14D权重能被现有服务读取。不要以“前向能跑”替代完整更新显存。

## 7. 数据吞吐与预算说明

采集每条实际命令及同步状态/标签，图像可按stride4保存三视角。episode结束写清有效长度，切窗口时使用索引；不反复整轨迹加载校验，也不加未经必要性/端到端代价证明的缓存限制。一个小样本验输入和一次完整WM更新同时测数据等待/计算耗时即可。

训练epoch不是采集回合数。窗口可重叠也可不重叠，必须在清单里明确起点步幅并记录实际样本量；不能用按非重叠C32估算的步数去描述stride4重叠训练。首批150、每次100新＋可选50旧、5epoch均为本轮讨论建议，不是既有正式实配。
