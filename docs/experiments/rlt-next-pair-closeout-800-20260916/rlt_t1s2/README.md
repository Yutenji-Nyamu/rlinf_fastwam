# π0 RLT · DVAC τ=1 · 成功倍率2 收尾

用户主动中止；后续按独立新run记录。最终已记录采集R425，原计划600轮；最新固定评估R425：17/20。最近checkpoint目录R425；目录清单并非模型加载/完整恢复验证。runtime退出码为15；终止原因记录为`user_requested_rlt_800_cutover`，原始结束时间2026-09-16T15:52:00+00:00。

原计划Stage2 fresh600，4 env、B512/micro256、UTD5、critic:actor=2:1、LR1e-4；10k回放门槛、15k critic初始化、每轮cap800、回放80k；BC/Q课程10k+25k，7/.05→2.5/.45。DVAC exp_mean双层温度=1.0，成功BC倍率=2.0；每25轮fixed20及保存。Stage1沿用已有CP2000。本包只含停止前旧run历史。

本包为GPU7已停止的τ1/成功倍率2原运行；新实验τ2/倍率1从空Stage2状态fresh训练800轮，新run为 /data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t2-half4env-fresh800-phys7-20260916-v1。两组指标分别保留，不串接曲线；旧checkpoint与回放留存服务器。

前期teacher、后期student采集；fixed始终测student（20 episodes）。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[五张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放及环境图像/视频不入包，仍保留在服务器。本次未读取模型或池payload、未删除checkpoint，所有末代文件保留；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t1s2-half4env-fresh600-phys7-20260916-v1`

生产源码：`68d901d2a9ac3cf9604407df28f9a9b137f5f480`；归档前HEAD：`703408b1f525645b58451e90404973a75cf0e41a`；分支：`codex/sz-rlt-dvac-next-pair-half4-20260916`。
