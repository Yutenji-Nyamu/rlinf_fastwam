# 跨领域源码核对与最小切口

2026-10-31。只读下载和阅读，没有执行下载源码。缓存位于 E:/Codex/home/visualizations/2026/10/13/01a1354d-42a1-74a3-adc8-6e9c3421e45a/gad_cross_domain1031/。完整commit与下载文件见cross_domain_sources.json。

## 可以借什么

|仓库/固定版本|实际检查的位置|可借内容与需要修改的部分|
|---|---|---|
|[TFB](https://github.com/Wang-ML-Lab/bayesian-peft/tree/f88b49441d52ea842a40038fc826f4201770a6df)|modelwrappers/tfblora.py:86–140,237–263|参考样本→原模型伪标签→二分搜索噪声幅度；分类NLL需替换为明确的连续动作参考指标|
|[TokUR](https://github.com/Wang-ML-Lab/TokUR/tree/bbf81a3f5d2a07cabee8d8ad81e435cf882754a5)|bayesian_transformer/bayesian_transformer/layers.py:39–78；convert_to_tfb.py:55–82|SVD基方向、Linear旁路扰动；代码每次forward按输入行抽输出噪声，不能当作一份固定ΔW跨老师去噪过程|
|[KNN-OOD](https://github.com/deeplearning-wisc/knn-ood/tree/2afb2bbed60a8d69384dc9b28e5637711345222b)|run_imagenet.py:35–52,73–92|归一化、FAISS IndexFlatL2、第k距离；此脚本K=1000为ImageNet设置，不照搬机器人参考库|
|[PatchCore](https://github.com/amazon-science/patchcore-inspection/tree/fcaa92f124fb1ad74a7acf56726decd4b27cbcad)|src/patchcore/patchcore.py:147–176,203–226|无梯度提特征→缩减参考库→近邻得局部分数；图像patch改动作位置token需要新索引和任务/轨迹排除|
|[UA-MT](https://github.com/yulequan/UA-MT/tree/88ed29ad794f877122e542a7fa9505a76fa83515)|code/train_LA_meanteacher_certainty.py:144–178；code/networks/vnet.py:177,195–196,225–226|8次老师预测→熵mask→有效位置归一化的consistency loss；借mask和detach，保留我们冻结老师而不是引入EMA老师|
|[DDU](https://github.com/omegafragger/DDU/tree/f597744c65df4ff51615ace5e86e82ffefe1cd0f)|utils/gmm_utils.py:79–109|按类别拟合均值/协方差并计算log密度；不是无需标签的任意特征距离，亦不是动作误差概率|
|[SWAG](https://github.com/wjmaddox/swa_gaussian/tree/61a918fa1c3e732dd19fc278ad3a5969f24d5a72)|swag/posteriors/swag.py:59–146,148–178|采集权重均值/二阶矩/低秩偏差，生成扰动模型；需训练过程统计，不能由一个末代checkpoint还原|
|[SPUQ](https://github.com/intuit-ai-research/SPUQ/tree/55d271616c9b0750ad91117071dbd23ae7b5aab9)|spuq.py:29–40；aggregation.py:13–19,22–79,82起|扰动和聚合模块分离；ROUGE/语言自报分支过滤，保留我们的动作RMSE/方差|
|[LLM-Check](https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection/tree/2f3bf9ea6db19e60a416090f77694816c92a9146)|common_utils.py:263–316,319–351|隐藏矩阵的中心化谱统计；注意力分支依赖因果三角结构，不直接用于π0.5动作注意力|

另下载DUE的README/GP入口以及tapnet的训练入口/BootsTAP配置用于可用性判断。当前tapnet所读训练文件主要为supervised_point_prediction，**不宣称已定位并复现BootsTAP论文循环筛选的完整训练代码**。BootsTAP筛选机制依据正式论文。HyperDUM本轮读论文，没有定位可确认的作者实现。

## 不能整段照抄的三个细节

1. **TFB代码与论文口径要分开。** 当前tfblora.py:125在默认mean NLL的相对变化后，又除以参考样本数；:130–137记录越界侧的搜索值。我们若采用论文“性能变化容差”思想，应重新写清归一化和可行侧返回值；不能直接复用数字阈值。这里只记录静态阅读，不据此推断作者所有实验有误。
2. **TokUR局部重参数化不等于固定模型变体。** 它把输入展平后每行抽独立输出噪声，适合其逐token方案。我们的老师生成H50动作要保持一个扰动模型贯穿整段N次前向、各动作位置共享同一ΔW。否则“参数分歧”混入每个去噪步重新加噪，且师生N1/N10随机次数不公平。
3. **参考库分数不能自查。** KNN/PatchCore的原本测试样本不在参考训练库；我们拿训练窗口算优先级时会查询同一数据池。需排除同轨迹，并记录参考模型版本；只排一个精确窗口仍可能被邻近帧泄漏压低。

## 我们已有代码的最小入口

- `rlinf/algorithms/pi05_distillation/losses.py:180`的endpoint_loss已经生成同观察同噪声的老师终点和学生终点，真实14维误差可直接保留为[B,H]，无需另跑一遍正常预测。
- `rlinf/algorithms/pi05_distillation/model.py:59–64`冻结非动作部分并将基础模型设eval；不存在“调用train()即可得到已有MC-dropout”的条件。
- `model.py:103–106`构建共享视觉语言前缀。只扰动动作专家可复用该缓存；扰动VLM必须构建对应的独立缓存，不能拿原缓存假装已经扰动VLM。
- 如果需要U_T、U_S不同，VLM-only **特征距离**不合适；VLM扰动后的**输出动作响应**则仍可能不同，因为两者动作专家不同。不要混淆这两种说法。
- 特征方案优先取动作专家某固定层、某固定读取位置，保存[B,H,d]；N1与N10采用相同语义的读取定义。不能混拿老师第9步中间态和学生唯一一步输入，直接宣称可比。
- 正常有梯度的学生分支保留；所有旁算、分数、权重stop-gradient，不写入optimizer参数，不改变正式生成RNG。
- 当采用加权时，只改loss聚合；优先抽样仍走03文档所列的window_id/task_id/probability接口，单独验证。

## 成本边界

|选项|增加的主要工作|是否新训练网络|
|---|---|---|
|动作专家权重扰动|额外动作前向；老师每个变体仍需其正常N步|否|
|VLM权重/图像扰动|额外视觉语言前缀，再加动作前向|否|
|重叠计划一致性|在线复用前后计划；离线训练要补相邻观察的推理|否|
|特征KNN|建立/刷新参考库，检索|否，但有统计状态|
|MC dropout|多次随机前向；当前基座若无训练dropout还需改变训练|不加网络，但改变训练机制|
|LLM-Check式谱统计|提取内部矩阵、矩阵分解；未必比小batch前向更便宜|否|

没有测量GPU耗时，不用模型前向次数直接换算真实训练减速。这里的源码审阅不构成smoke或端到端验收。
