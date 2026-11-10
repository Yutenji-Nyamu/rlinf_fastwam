# 11月10日：稳定性修复已进入正式流程

验收快照：2026-11-10 17:12（北京时间）。4/5/6分别通过两轮实际仿真短测及CP2产物检查，owner均进入formal；7卡原RT2 RLT保持原实验。当前尚无新学习成功率结论。

|卡|正式实验|本次改变|总预算|
|---|---|---|---|
|4|RT2 DSRL / turn_switch|噪声先验KL/d，权重0.5；固定α=0|16环境×400轮|
|5|Dojo DSRL / put_bottles_into_dustbin|同4卡|16×200轮|
|6|Dojo RLT / put_bottles_into_dustbin|Q权重0、BC=1、参考dropout=0；固定20k参考经验，参考/在线各半；参考动作bootstrap|16×313轮|
|7|RT2 RLT / turn_switch|原配方继续，未重启|16×626轮|

两路DSRL保留FP32小网络及Adam、独立1600维噪声/±2、初始2000条、Q-only2000、UTD5及有效加速。β=.5按既有Q优势约1与KL/d约2.39定初值，并非外部论文调好的参数。已有有界actor不能精确复现无界高斯，KL旨在限制偏离，不保证完全保住基座。

Dojo RLT本阶段始终纯BC，没有按轮次自动开放Q。参考VLA实测8/8、解码差0；这不是新学生成绩。实际短测的Q权重0、BC权重1、参考遮挡0均已核；保存的160条短测回放能读回。正式采集阈值20k与参考保留上限一致；短测缩短了预采集，仅用于运行验证，短测权重/回放不用于正式训练。不可将BC loss下降视为闭环模仿成功。

源码、配置、来源见[SOURCES.md](SOURCES.md)与[evidence](evidence)。来源扩展到BRAC、RLT、RLPD、verl-vla CQL、OFN、LP-DS、FRS。当前实现只有噪声先验、纯BC/参考bootstrap和固定参考回放。LP-DS更直接讨论噪声集中，但完整残差actor/双Q蒸馏改变更多；FRS需动作反演参考链路；这两项与CQL、OFN均保留为后续。后续先看小策略接管后正式评估与真实回报/Q关系，再决定开放Q或加入保守项。

当前并行数已是16，保留此前8→16的扩容。本轮连续57次样本，4/5/6/7峰值分别47.2 / 54.5 / 30.8 / 35.7 GiB，每卡约79.6GiB；涵盖短测、初始化与部分正式流程，不是全周期峰值。旧近20轮采集占轮次耗时约79%/92%/83%/89%，因此本轮没有再加并行。下次若试16→32，需要按轨迹/秒判断，并重算总轮次、评估频率与UTD对应更新量；7卡接近既有终点，继续原配方。

新结果根：`/home/chenyiteng/results/stability-retry-20261110`。部署根：`/data/chenyiteng/deployment-20261110/stability-retry`，4卡部署名带`-20261110-v2`，5/6为原名。主active-runs仍在11月5日部署根；运行前必须重新取PID/UID/start身份，不照抄旧PID。代码出口在`/data/chenyiteng/projects/rlinf-shenzhen/exports/stability-retry-20261110/{dsrl,rlt}`。

交接中4卡首次因旧工具目录重名退出，新训练尚未开始，真实RLT回退接回；第二次CPU join日志重名未切GPU；随后使用独立部署目录和唯一join日志，三路成功登记。失败尝试目录保留。不要重跑已有prepare/start/launch；`launch-receipt.json`为唯一成功回执。FQL join已等新4卡owner与未改的7卡续训owner；5/6完结独立回RLT。无共享Ray重启、无其他用户停训。

服务器巡检：新产物在Home；Data仍需留意增长。20TB `/data2`仍挂载，但10:22后出现3次READ I/O错误与USB重置，另有NVIDIA引用计数告警；所查范围未见Xid/OOM。GitHub/HF代理访问均200，常见矿工指标未命中。6个精简RLT候选均可读、FP32有限值、约8.26MiB；旧place_fan两路历史1/20、0/20，不能因文件正常就说效果好。

本地网页：`E:/Codex/home/visualizations/2026/11/05/01a1a973-6ab3-7d40-93bf-6148e45a3710/sz1-1108/index.html`。原始巡检和交接原件在`C:/Users/86136/Documents/seek/outputs/stability_retry_20261110`；不将原始全量过程日志、凭据、权重或回放推送。
