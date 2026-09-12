# π0 Clean RLT · 单卡 fresh600 收尾

按用户要求中止，最终采集完成R373，原计划600轮。最新固定评估R350：18/20，最近checkpoint目录为R350。不把主动中止写为自然完成；runtime退出码为15。

配置：单卡，8 train env × 1 wave，执行C10，B512/micro256、UTD5、critic:actor=2:1、actor/critic LR均1e-4。回放容量80k，20k transitions后开始更新，原30k初始化更新；固定评估每25轮20条，每25轮保存。共用深圳adjust_bottle任务50条示范训练2000步的current causal-AR Stage1，Stage2空池全新启动，历史环境实现；没有DVAC。

采集成功率：R1–154为Teacher，R155起为Student。固定评估始终测试Student。R136首次更新，R148开始原BC/Q课程渐变，R193起整轮固定为2.5/0.45。阶段来自本次clean实测历史，不把Teacher早期采集成功率等同Student性能。

[四张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [完整ZIP（含原始日志和实际配置）](rlt-clean-single-gpu600-R373-closeout-20260912.zip)

本ZIP含完整日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和配置、源码身份及checkpoint文件目录。大模型、optimizer、回放、图像和视频不入包，仍保留在服务器；该ZIP不能独立恢复训练。本次归档没有删除checkpoint。

run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-control-phys6-fresh600-20260912-v1`

生产源码：`30349428c37a008b95342121c1455debfeb4805e`；归档前HEAD：`42560088a017eafbc612644c2edcfe1d8a56314e`；分支：`codex/sz-rlt-control-single-gpu600-20260912`。
