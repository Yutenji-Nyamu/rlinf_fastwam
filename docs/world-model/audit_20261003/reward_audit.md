# Reset集合奖励模型只读核验

2026-10-03。深圳3实际CPU推理完成，21.91秒；batch16、4线程，CUDA未初始化，服务器文件写入0。复核脚本与现场源码后，未发现归一化或`.eval()`模式偏离。

|真正供闭环起步的画面|数量|stored reward=0|分类输出0|Sigmoid最大值|
|---|---:|---:|---:|---:|
|initial，第0帧|496|496|496|2.4694e-20|
|KIR，第24帧|246|246|246|3.4792e-18|
|合计|742|742|742|3.4792e-18|

推盘任务`push the plate to the front of the stove`有49个initial＋30个KIR，79帧全为stored0/pred0；全部742文件中约占10.65%，没有发现该任务reset文件缺失或数量过低。其reward模型内部任务ID为2，与物理评估的task_id5不能混用。

**本检查排除了这742个固定reset观察中“分类器一开始就判成功”的现象。** 它不证明reward模型总体准确、召回充分或经过校准：集合没有正标签，stored reward的标注过程及是否属于训练集仍未核实；极低Sigmoid值只表示模型对这些负例强烈输出负类，不能当准确度、召回率，也不能据此判定模型失效。全部文件内部存储的reward唯一值亦为0。

生成的Wan画面、真实终态成功画面、动作后的首个chunk及chunk内成功→失败翻转没有包含在此检查中，因此这些位置的假阳性/假阴性仍未知。由已近成功起点继续少量动作到成功也不等同于初始误报；本结果不评估KIR的完整因果贡献。

预处理和推理路径逐项核验：

- 所有源图均为uint8、像素范围在0–255，实际推理图为3×256×256。直接使用现场`NpyTrajectoryDatasetWrapper._convert_frame_to_lerobot_format`，HWC→CHW、float32、除255；随后使用从同一env源码提取的`_to_condition_frame`及官方Normalize(mean=.5,std=.5)，得到−1至1。
- initial的1帧不足4个target，实际闭环末槽重复第0帧；KIR的25帧按官方last_n_frames=4取21–24，闭环看第24帧。每文件只评一张起始图，共742张。
- 模型为同一`TaskEmbedResnetRewModel`、同一checkpoint，`.eval().to('cpu')`；现场env也在加载奖励模型后调用`.eval().to(device)`，因此BatchNorm/Dropout模式一致。不是以train模式评价负例。
- 调用官方`predict_rew`，其中clamp(−1,1)、float32、任务嵌入、Sigmoid及torch.round均保留。只读forward hook捕获同一次fusion_layer输出；逐批断言官方二元输出恰等于round(该置信度)，未改阈值、权重或输入画面。
- CPU与GPU浮点执行不承诺逐位一致，但这里最大值远离0.5，不是阈值附近的数值歧义。

奖励checkpoint SHA256：`2cf3da891107c9267d15cad2ad6ab14f64fc851885633ea4d5cb1f07558aba7c`。奖励源码/数据转换/env三份SHA均与本轮现场包一致；完整逐帧数据、SHA及分组统计见[只读结果](E:/Codex/home/visualizations/2026/10/02/01a0fc97-7ba7-7a22-811c-f17d4422e077/wmrl-audit-20261003/sz3/reward-cpu-audit.out)，执行脚本见[reward_probe.py](../../../.tmp/wm_audit_20261003/reward_probe.py)。未追加实验。
