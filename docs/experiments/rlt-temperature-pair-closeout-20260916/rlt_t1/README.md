# π0 RLT · DVAC τ=1 · 成功倍率1 收尾

计划600轮自然完成、退出码0。最终已记录采集R600，原计划600轮；最新固定评估R600：19/20。最近checkpoint目录R600；已核CP600完整标记及7项必要组件元数据，未做模型加载或恢复测试。runtime退出码为0；终止原因记录为`planned_budget_completed`，原始结束时间2026-09-16T02:47:17+00:00。

Stage2 fresh600自然完成，4 env、B512/micro256、UTD5、critic:actor=2:1、LR1e-4；10k回放门槛、15k critic初始化、每轮cap800，回放80k。BC/Q课程10k+25k、7/.05→2.5/.45；DVAC two_level_batch、exp_mean，local/chunk温度均1，成功BC均重倍率1，alpha1/1，L3/C10，teacher方差记录apply。Stage1沿用原current-AR CP2000；每25轮fixed20和保存。仅包含本组fresh R1起历史，无恢复或其他运行拼接。

前期teacher、后期student采集；fixed始终测student（20 episodes）。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[五张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放及环境图像/视频不入包，仍保留在服务器。本次未读取模型或池payload、未删除checkpoint，所有末代文件保留；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t1p0-half4env-fresh600-phys7-20260915-v1`

生产源码：`7c9372c6f7560ca65e6270d08cd4ae29a2a1ed0c`；归档前HEAD：`68d901d2a9ac3cf9604407df28f9a9b137f5f480`；分支：`codex/sz-rlt-dvac-temperature-half4-20260915`。
