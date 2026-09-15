# π0 RLT · DVAC成功倍率4→2 收尾

用户主动中止并切换RLT温度对照。最终已记录采集R275，原计划600轮；最新固定评估R275：16/20。最近checkpoint目录R275；目录清单并非模型加载/完整恢复验证。runtime退出码为15；终止原因记录为`user_requested_stop_for_rlt_temperature_pair`，原始结束时间2026-09-15T09:05:12+00:00。

原计划Stage2 fresh600，8 env、B512/micro256、UTD5、critic:actor=2:1、LR1e-4；20k回放门槛、30k critic初始化、每轮cap1600，回放80k。原BC/Q课程20k+50k、7/.05→2.5/.45；DVAC two_level_batch，L3/C10，alpha1/1，成功BC倍率4在首个actor槽t70000降2。teacher方差记录apply；每25轮fixed20和保存。只含修正后的v2空池运行，v1未混入。

前期teacher、后期student采集；fixed始终测student。原v1不并入。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[五张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放及环境图像/视频不入包，仍保留在服务器。本次未读取模型或池payload、未删除checkpoint，所有末代文件保留；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-scale4to2-phys6-fresh600-20260915-v2`

生产源码：`3c60dbddeffe5102253b3b54e460f5cd7dd862f0`；归档前HEAD：`dcb2e7ba2a1bf45691876be45942917849dec48e`；分支：`codex/sz-rlt-dvac-scale4to2-20260915`。
