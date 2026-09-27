# SZ2 原始实验与轻量源码归档

快照日期：2026-09-27T20:46:53.134753+08:00。源服务器直接生成并发布，不通过本机中转。

本包保留 25 个可识别 TensorBoard 训练目录的原始事件、配置、日志、启动记录和轻量 DV 诊断；同时保留失败/短验收与配套对照，避免按最终效果筛掉必要上下文。CATALOG.csv 是逐目录索引。上涨片段只是筛选线索，不等同于 DVCA 的因果增益或跨种子结论。

原始事件使用 TensorBoard 的零起点 step；success-series.json 与 CATALOG 的轮数为 step+1。续训段独立保留，runner.resume_dir 记录连接点；拼接时应截断被续用检查点之后的旧段。active_snapshot=true 的目录仍在训练，文件按各自读取时刻截取；ARTIFACTS.json 记录是否采集时增长。

源码已提交部分由 SOURCE_INDEX.json 的 commit/branch 定位；相对 RLinf 7d07a421 的代码补丁放在 source/，当前 dirty 或 untracked 仅为未验证快照，不是生产修复。不同工作目录可能共享同一提交，补丁按提交去重；运行时保存的源码版本记录优先于当前工作树快照。

大模型、checkpoint/optimizer、回放池、原始 tensor/NPZ、视频、依赖环境、缓存和凭据不进 Git。单文件上限32 MiB，DV文本目录合计上限64 MiB；超限与敏感文件逐项写入 EXCLUSIONS.json，没有截尾冒充原始日志。日志、JSONL和较大文本无损gzip；原始/发布SHA256及大小见ARTIFACTS.json。该包可重绘和追溯，不包含复评模型所需的全部权重。
