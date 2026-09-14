# RLT Clean CP350恢复分支收尾

本包是**从CP350恢复的分支**，不是fresh run。本次记录R351–472，共122轮；目标全局R600。用户要求停止旧分支并启动fresh half-budget新组后归档，精确停机与替换启动回执已附。退出码15，最新fixed为R450=17/20。

恢复checkpoint：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-control-phys6-fresh600-20260912-v1/current-single-gpu-control-phys6-fresh600-20260912-v1/checkpoints/global_step_350`。恢复沿用8env、B512/micro256、UTD5、执行C10、固定评估20状态；本包不是新4env half-budget实验。原始实配与启动命令在raw/runtime和raw/tensorboard/config.yaml。

`metrics.csv/json`仅保存本恢复run的原始标量；采集从R351起。`prefix/metrics-r1-350.csv/json`是原Clean的R1–350背景，来自另存的快照及其SHA。**原Clean的R351–373尾段未并入续训、未入背景。** 五图以CP350为恢复界线，各段独立计算MA，不把不同run伪接成一条连续观测曲线。尾部optimizer标量可能比最后完整采集多，逐tag截止见SUMMARY.json。

[五张宽图](plots/index.html) · [本次标量CSV](metrics.csv) · [独立原Clean前缀CSV](prefix/metrics-r1-350.csv) · [实配](raw/tensorboard/config.yaml) · [日志](raw/runtime/driver.log)

包内是日志、TensorBoard events、标量、实配/命令、源码SHA清单和checkpoint目录清单；没有模型、optimizer、replay内容、图像或视频。服务器大文件不修改、不删除；此轻证据包不能独立恢复训练。最近checkpoint目录R450仅是目录清单，不等同模型加载或完整恢复验证。

旧run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-clean-resume350to600-phys7-20260914-v1`；旧namespace：`RLinf_rlt_clean_resume350to600_phys7_20260914`。

生产源码：`77d0a673055db6210bdcdbf11e5ef317d18b990e`；归档HEAD：`58f1c51f2d915cdf4ab23a3ced4563b475ee54be`；旧分支：`codex/sz-rlt-clean-resume600-20260914`。
