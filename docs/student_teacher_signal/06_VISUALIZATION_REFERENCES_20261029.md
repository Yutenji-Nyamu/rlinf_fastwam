# 信号关联的可视化参考 · 2026-10-29

从“轨迹局部变化、总体关联、任务结果、排序可用性”分别找图，不能把不同论文的被解释量混为一谈。

|一手来源|具体呈现|可借鉴与边界|
|---|---|---|
|[UA-OPD](https://openreview.net/forum?id=hHUZi8oHfo)，Fig.2及附录|自一致性不确定性与经验reward gap的二维密度，Spearman报告0.683|借密度和相关系数；原文adjust_bottle用近最优参考老师估经验奖励差，我们是弱/强动作距离，数值不可横比|
|[U-GROW](https://arxiv.org/html/2609.22879v1#S4)，Fig.2–4|分布、阈值、成功/失败轨迹的时间信号与阶段画面|借全程曲线和结果分组；前3块只是我们的时间定义|
|[ConformalDAgger](https://arxiv.org/html/2410.08852v2#S5)|在学习者访问状态下看专家动作偏差与时序问题|同观察比较很重要；时序点有相关性，应整轨迹重采样|
|[FIPER](https://arxiv.org/html/2510.09459v2#S6)，Fig.4|按成败、ID/OOD分组的均值和四分位范围|借成败分组；本批没有OOD，不额外造OOD对照|
|[Ilg et al., ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eddy_Ilg_Uncertainty_Estimates_and_ECCV_2018_paper.pdf)，§5.2|按不确定性逐步移除样本，看剩余误差，比较oracle/random|借排序删除曲线；我们的被解释量是师生分歧，不称真实预测误差|
|[Uncertainty Quantification for Flow-Based Generalist Robot Policies](https://arxiv.org/html/2606.18043)，§6|任务/条件校准、Spearman、成功/失败时间轨迹|借跨层次分组；不同信号定义与归一化单独说明|

UA-OPD网页遇验证，使用用户本地完整论文提取文本核对Fig.2及616–629行；未将挑战页当论文内容。其他来源直接核对原文。图册保留上述来源链接。

本批实现九类总体视图：等样本分箱、反向分箱、二维密度、按列归一化条件密度、阶段与逐轨迹、块内去均值、相对/绝对分散度对照、排序删除、成败分布。主结果预先保留原指标，补充定义透明标注；不按哪张更漂亮筛选结论。
