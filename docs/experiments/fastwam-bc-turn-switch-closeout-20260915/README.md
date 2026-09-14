# Fast-WAM BC · turn_switch 收尾

用户主动中止。最终已记录采集R29，原计划200轮；最新固定评估R25：14/32。最近checkpoint目录R20；目录清单并非模型加载/完整恢复验证。runtime退出码为-6。

正式200轮、8 env、U10、B1024/micro2、LR2.5e-5、原Adam及clip1；Fast官方release初始化、空成功池开始，无resume。192动作上限、24步提交、32步预测；仅训练动作专家，不启用成功长度过滤。环境保持驻留，actor/rollout仍卸载；每5轮fixed32、每10轮保存。训练与评估seed直接复用前次编号并改任务键，不表示经专家验证的新任务可解性。

每轮8条采集；fixed每5轮32状态；本包仅此fresh run，不并入前次任务或smoke。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[五张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint和成功池文件目录及本说明。模型、optimizer、回放、图像、视频不入包，仍保留在服务器。本次未读取模型或池payload、未删除checkpoint，所有末代文件保留；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-shenzhen/online-bc/fastwam-turn-switch-bc8-u10-b1024-phys6-formal200-envresident-20260914-v1`

生产源码：`eaaebd8730ebe2d57e67bcd05c0731266dd40e07`；归档前HEAD：`ba29fe7b68655ea375962b2c7331dc877c44ed97`；分支：`codex/sz-fastwam-bc-turn-switch-20260914`。
