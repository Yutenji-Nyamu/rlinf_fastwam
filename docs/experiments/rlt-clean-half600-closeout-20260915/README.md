# π0 RLT · Clean半量4env 收尾

用户主动中止并切换RLT温度对照。最终已记录采集R549，原计划600轮；最新固定评估R525：19/20。最近checkpoint目录R525；目录清单并非模型加载/完整恢复验证。runtime退出码为15；终止原因记录为`user_requested_stop_for_rlt_temperature_pair`，原始结束时间2026-09-15T09:05:35+00:00。

原计划Stage2 fresh600，4 env、B512/micro256、UTD5、critic:actor=2:1、LR1e-4；10k回放门槛、15k critic初始化、每轮cap800，回放80k。BC/Q课程10k+25k、7/.05→2.5/.45；Clean、DVAC off。Stage1沿用原2000步训练产物，不缩减或重训；每25轮fixed20和保存。与8env同轮不是同采样预算。

前期teacher、后期student采集；fixed始终测student。只含本组fresh R1起历史。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[五张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放及环境图像/视频不入包，仍保留在服务器。本次未读取模型或池payload、未删除checkpoint，所有末代文件保留；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-clean-half4env-fresh600-phys7-20260914-v1`

生产源码：`58f1c51f2d915cdf4ab23a3ced4563b475ee54be`；归档前HEAD：`db59a8166f19a34c3db9b5717093a625f9344a38`；分支：`codex/sz-rlt-clean-half600-20260914`。
