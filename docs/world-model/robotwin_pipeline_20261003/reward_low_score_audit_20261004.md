# OpenDW → WorldArena摆瓶子RM：低分审计

2026-10-04。本轮只读本地源码、此前保存的官方源码和smoke画面；未SSH，未改冻结服务、奖励或训练配置。新N64短测的当前状态由主线程现场核验，本页不代替实时日志。

## 结论

**尚未发现可以解释系统低分的明确接线错误。首32动作低分本身不能判为bug。** WorldArena摆瓶子RM是成功分类器，不是每向瓶子靠近一点就必然涨分的进度模型。旧smoke保存的 `sample-000/predicted-final.png` 里，瓶子仍横躺、夹爪刚接近/夹住；指令要求“用右臂把绿瓶竖直”。其末分 `3.6793e-6` 与“尚未完成”并不矛盾。此前4张原生初态CPU分数也仅约 `1.7e-5～9.4e-5`。

已确定的过滤机制：初始previous score为0，每帧差分，整条回报近似等于最后有效画面的分数；G8**组内平均总回报**需处于 `[0.1,0.9]`。因此末分都约 `1e-6` 时，数值即使互不相等仍整组过滤，优势/梯度为0。增加N会扩大采样覆盖，不能自动解除门限；把轨迹从32延至384，才有机会看到任务真正完成后的分数。若长轨迹仍全低，先定位RM真阳性或策略/WM轨迹问题，不能先放大微小分数制造学习信号。

**00:14服务器CPU补证（由主线程执行，本页读取回执）：**同一decoder/resize256/指令/RM对clean50前10条演示首、中、末共30图打分，26.08秒完成，CUDA未初始化。首帧均<1e-4，中帧均<5e-6；末帧4/10达到0.9、8/10达到0.1，episode3/8约0.00315/0.00244，其余末帧约0.267～0.999。它能在部分完整演示末帧给出高分，故“管线把所有输入都压到1e-6”已不成立；中帧低分也支持成功判别式而非平滑进度式信号。HDF5 attrs均为空，本批只能称**预期正例演示末帧**，不能报告原生真阳性recall。episode3/8应优先看画面与任务标签，再用相同图对照不同指令；不据这10条直接改0.9阈值。

补证服务器路径：`/data/chenyiteng/projects/opendw-robotwin-smoke-20261003/analysis/reward-demo-calibration-20261004-v1.json`；本地回执 `E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/sz3-status-20261003/opendw-reward-demo-calibration1.out`。RM源码SHA `d5e41b9b89375270c386ec3a2700855ddcc420ae4779c3c72dce509772ab840d`。

## 源码已核对的链路

| 环节 | 当前实现 | 审计结果 |
|---|---|---|
| 视角 | OpenDW画布H384×W320；取未来8帧的上方256行作为主相机，排除腕图 | 与官方 `robotwin_resize` 上主图、下两腕布局一致；未把拼图整体交给单图RM |
| 图像范围/颜色 | PIL `.convert("RGB")` → uint8 HWC → /255 | 没有额外BGR交换，没有把WM的[-1,1]tensor直接送入RM |
| resize/归一化 | 主图256×320直接双线性224×224，`align_corners=False`；ImageNet mean/std | 与WorldArena `BaseImageRewardModel.preprocess_images` 默认值一致；没有额外策略256中间缩放 |
| 模型 | ResNet18去pool/fc；T5-base；512维投影、8头cross-attn、LayerNorm、512→256→1 | 与固定官方类一致；完整state dict strict load；此前服务器实计122,382,017参数 |
| 推理模式 | 全模块eval、冻结、FP32；sigmoid一次 | 未发现重复sigmoid/round、dropout训练态或缺失随机初始化层；`weights=None`不会遗失已strict加载的视觉权重 |
| 文本 | 同reset的seen[0]指令；padding/truncate64；attention mask进入T5与cross-attn | 与官方带指令分支一致；发布RM训练时的具体文本分布仍不清楚 |
| reward/done | 8个分数差分放在动作4/8/…/32；任一帧≥0.9在块末done | 未重复4次放大；沿WorldArena块末成功语义。曾高分后回落可能“success=true、末回报低”，需单独统计 |

“未发现源码差异”不等于已经做过官方类和本地类逐样本数值等价测试；本轮没有执行该测试。完整strict加载证明结构/权重键可用，也不能证明生成域上的准确率。

## 还需实证的问题，按最低成本排序

1. **RM能否认出我们现有原生数据中的真成功？** 先从现成clean50取约10条演示各首/中/末三帧，带该episode的指令，在服务器CPU一次加载、批量打分。末帧应结合已有成功元数据/原生标签；只有“演示末帧”而无标签时不要宣称真阳性率。与已保存生成图一并记录原图和分数，无需新采样、无需GPU。
2. **预处理/文本域是否影响校准？** 在同一小批已核成功帧上比较：原始主图→RM；经过当前reset的256方图→OpenDW尺寸→RM的纯resize链；必要时再比较原文与固定摆瓶子描述。RGB/BGR交换只作有标签的离线敏感性对照，不能凭更高分改在线链路。现有reset按旧RoboTwin JPEG约定 `cv2.imdecode` 后不换色；生成帧是明确PIL RGB。两端必须结合原始生产端代码或已知色标确认，不能仅凭cv2惯例推断要换色。
3. **排除独立实现差异。** 对2～4张同图、同指令，用本地保存的官方类与mini类、同checkpoint、FP32/eval比较preprocess tensor、logits、概率最大误差。这可以与上一步在CPU合并执行。
4. **长轨迹实际有没有摆正瓶子？** 复用L384输出，优先看每条最高分与末分、首次≥0.9帧，以及分数最高/最低的少量轨迹。如果画面没有成功，低分可能正确；若明显成功仍低，再把生成域偏差列为强嫌疑。查看累积多块后是否出现瓶子消失、姿态跳变或三视角不一致，不把所有失败都归RM。
5. **过滤实际丢了什么？** 从现有日志重建每条末分、每G8均值、有效组数；分开统计 `max_frame_score≥0.9` 与 `final_score`。若大量组是有可信进展但全部被门限丢弃，再讨论门限/信号方案，作为下一轮明确配置变更。

不建议现在做的事：将全部分数乘大、把1e-6下界改成可通过、改成功阈值、替换大型VLM或重训RM。以上都无法先回答“原生真成功帧是否认得”。

## 核对来源

- 当前服务：[opendw_service.py](../../../local_patches/opendw_smoke_20261003/multigpu/tools/opendw_service.py)，`infer()`中的frames[1:]、head截取和RM调用；[opendw_reward.py](../../../local_patches/opendw_smoke_20261003/multigpu/tools/opendw_reward.py)。
- [差分/动作/结束适配](../../../local_patches/opendw_smoke_20261003/multigpu/rlinf/envs/world_model/opendw_adapter.py)；[reset转换](../../../local_patches/opendw_smoke_20261003/build_reset_data.py)；[配置门限断言](../../../local_patches/opendw_smoke_20261003/build_multigpu_config.py)。
- 固定官方[WorldArena模型](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/models/embodiment/reward/robotwin_reward_model.py)、[预处理](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/models/embodiment/reward/base_image_reward_model.py)、[环境奖励语义](https://github.com/WorldArena2/WorldArena-2.0/blob/5978ce5c81e55b8c8358f4f5966a13ce385ff155/RL_env_benchmark/rlinf/envs/world_model/world_model_wan_env.py)。本轮读取此前存于 `E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/opendw-smoke-20261003/sources` 的对应源码。
- 官方[OpenDW图像布局](https://github.com/dexmal/opendw/blob/e33befa8005a1585e0140dbf464566e90bc79aa1/dexbotic/policy/dw05_policy.py#L145-L182)；[既有smoke回执摘要](execution_20261003.md)。GRPO过滤公式同时核对本地RLinf actor源码，实际新运行结果仍由主线程日志验收。
