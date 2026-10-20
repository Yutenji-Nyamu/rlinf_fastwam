# 10月17–20日实验轻量产物补归档

核对深圳三机既有结果树与本工作区专题。截至10月20日，10条相关方法/实验分支HEAD均已在对应远端；GRPO、EXPO收束源码和证据已推，VLABench的最终结果已在独立vlabench_pi05仓库（59a71a4）。

此前CP180 30阈值扫描只有实现、标定和启动证据，缺完整收尾结果；近期RoboTwin U、四噪声采集及恢复训练的新增日志也不能由“源码已推”代替。本次补齐结果树自10月17日起有更新的轻量文件快照，包含成功率、实际接管率、逐回合/逐动作JSON、配置、事件、评估与失败日志。最新同族CSV/SVG、Dojo覆盖统计和精选图也在此。

源文件54,446份，压缩1.41 GiB；原始轻量包见[Release](https://github.com/Yutenji-Nyamu/rlinf_fastwam/releases/tag/completed-artifacts-20261020)。按原服务器路径保留目录，三个source-manifest压缩文件给出精确文件列表、大小与mtime，不做重复逐文件哈希。部分内容与此前基线有重叠，用于补齐可重画数据。

单文件超过8 MiB的文本在manifest-summary的排除表列明。权重、优化器、回放、视频、原始query NPZ/观察、环境/凭据文件不上传；旧checkpoint目录不扫描。活跃任务仅为采集时快照，不冒充训练结束。范围为深圳已知结果树，VLABench独立项目依赖已发布的完整结果；不声称覆盖任意临时目录或AutoDL。

## 主要索引

- `cloud-audit.json`：对应仓库、分支、HEAD及远端核对。
- `manifest-summary.json` / `*-source-manifest.tar.gz`：归档范围与排除项。
- `monitor-curves.csv` / `plots/`：78条当前与历史训练/评估曲线，以及前两块接管等图。
- [Dojo图册预览](dojo-gallery/PREVIEW.md)：10任务13精选、完整U与同回合真实帧；正式945回合104成功，另有19/25 smoke未混计。
- `records/`：从原始包提取的完成条件汇总，保留源路径；当前任务部分完成点不宣称整个扫描结束。

原始GitHub基线仍为post-iclr-artifacts-20261017与inference-artifacts-20261018；不上传多GiB权重与完整媒体。
