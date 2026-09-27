# SZ2 原始实验与轻量源码归档

2026-09-27，由源服务器直接生成并发布。本包收录 25 个可识别 TensorBoard 训练目录，其中 21 个有固定评估，4 个为运行中快照。失败、短验收、续训与配套对照一并保留；目录数不等于成功实验数。

## 内容与索引

- `CATALOG.csv` / `CATALOG.json`：逐目录路径、任务、固定评估、resume连接与运行状态。
- `runs/`：原生事件、实配、日志、启动记录、成功率序列、DV文本与小型诊断tensor、调试图和相关轻量文件。日志与小型二进制诊断无损gzip；诊断.pt/.npz/.npy按目录识别，保存时不执行反序列化。
- `SOURCE_INDEX.json`、`source/`、`source-untracked/`：源码版本、相对RLinf `7d07a4212ee6858cc333e1d4fab7a37256d1f839` 的已提交补丁，以及未验证的dirty/untracked快照；其他依赖仓库以commit/remote定位。对应运行保存的版本记录优先于当前工作树版本。
- `ARTIFACTS.json` 与 `SUPPLEMENT_ARTIFACTS.json`：数据文件的原始/发布SHA256和大小。`CHECKSUMS.json`覆盖当前树全部文件（自身除外），可直接检查当前下载内容。

## 读取与范围

原生TensorBoard的step从0开始；success-series和目录表使用round=step+1。续训按实际runner.resume_dir对齐，在被续用检查点处截断旧段，不能直接拼接所有目录。活动训练的文件按读取时刻保存，并记录是否在复制中增长。上升片段只是筛选线索，不等于DVCA因果增益或多种子结论。

模型权重、checkpoint/optimizer、回放池、数据集、视频、依赖环境、缓存和凭据不入库；超过32 MiB的单个额外诊断文件也留服务器。独立小型DV tensor属于已收录的原始诊断。`EXCLUSIONS.json`和`SUPPLEMENT_EXCLUSIONS.json`记录具体排除项，`EXCLUDED_DATA_LOCATIONS.json`记录重文件位置与混合目录说明。

范围为本人/data/chenyiteng/results、/home/chenyiteng/results、/data/chenyiteng/runs内可识别TB目录。非TB旧实验和已经删除的文件可能只存在于旧归档分支；复评所需的大模型仍需从原服务器获得。训练与共享Ray未改动。
