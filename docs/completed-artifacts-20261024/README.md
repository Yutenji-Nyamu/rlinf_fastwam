# 近期实验轻量产物与蒸馏讨论 · 2026-10-24

15处实验源码/图册/收束工作树的已提交内容已确认公开；G图包f9b27b3已在bkp主分支历史中，不因临时分支未推而误判缺失。VLABench、四噪声图册、Dojo精选、G接管图包、N39补采仍放各自对应仓库，没有重复搬库。无关dirty工作树保持原状。

本次补深圳1相对10月20/21归档新增或变化的**25,553份文件**，85个压缩分片，共**1.51 GiB**；跳过23,070份未变化文件。原始日志、配置、事件、逐回合/动作记录分为[Release 1–50](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/completed-artifacts-20261024)和[Release 51–85](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/completed-artifacts-20261024-part2)；release-index.json逐项标明所属入口。活动日志是有界前缀快照，不能把归档完成说成实验已结束。

|结果族|文件数|未压缩文本/事件|
|---|---:|---:|
|pi05-policy-eval|13,787|5455.9 MiB|
|rlinf-rlt|74|20.2 MiB|
|pi05-noise-video|83|10.9 MiB|
|pi05-distill|11,603|2955.0 MiB|
|rlinf-shenzhen|6|2.9 MiB|

包括H低预算接管扫描后续、GRPO收束与RLT增量、多任务直接蒸馏/OneDP、原始N1/N10、N39补采轻量信息；补存27份旧GRPO换任务的本地轻量记录。保留真实输入/输出记录，便于后续绘图。

深圳2(.45)、3(.105)仍无法进入SSH握手，本轮不能新核对它们的服务器文件；`cached-unreachable/`只保存10月22最后可读的我方实验记录，明确标时间。两机新生成且未取回的日志仍是待恢复后补齐项，不能声称三机已全量同步。

权重、优化器、回放、原始大数组、完整视频不上传。单文本文件>8MiB列入排除表；环境/凭据文件排除。公开附件沿用已有筛选范围；本地仅以大小/mtime比较是否变化，上传以GitHub回执为准，没有重复下载或逐文件哈希核验。

## 画图入口

- [匹配8回合训练曲线](plots/matched-success.svg)，[逐任务N10−N1](plots/task-difference.svg)，[精确表](plots/matched-evaluation-points.csv)，[原始逐任务计数](task-n1-n10.csv)。`replot.py`可重画训练曲线。
- [配对分析JSON](distill-analysis.json)、`records/`原评估摘要、`plots/current-training-metrics.json`训练目标；原逐回合文件在Release分片。
- [为何总差距很小/早期是否漏峰](discussion/29_GAP_AND_EARLY_CURVE_20261024.md)、[C扫描规划](discussion/30_CHUNK_GAP_SWEEP_PLAN_20261024.md)、[论文与源码依据](discussion/31_GAP_SOURCE_LEDGER_20261024.md)。本轮未启动新扫描。

## 已有发布

- [10/20轻量全集](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/completed-artifacts-20261020)、[10/21增量](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/completed-artifacts-20261021)。恢复历史时按日期叠加，新版同路径覆盖旧快照。
- [VLABench](https://github.com/Yutenji-Nyamu/vlabench_pi05)、[四噪声图册](https://github.com/Yutenji-Nyamu/tts_consistency_fig)、[图稿与精选轨迹备份](https://github.com/Yutenji-Nyamu/uaca_cv27_draw_bkp)。

边界：仅既有深圳实验目录和明确对应的本地材料，不包括任意临时目录、其他用户材料或AutoDL。发布回执另存`publication.json`。
