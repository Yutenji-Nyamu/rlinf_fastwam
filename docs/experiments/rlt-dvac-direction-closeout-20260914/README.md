# π0 RLT · DVAC方向反转 收尾

用户主动中止。最终已记录采集R434，原计划600轮；最新固定评估R425：20/20。最近checkpoint目录R425；目录清单并非模型加载/完整恢复验证。runtime退出码为15。

原计划600轮、8 env、执行C10、B512/micro256、UTD5、critic:actor=2:1、actor/critic LR均1e-4；初始20k transitions后训练、30k critic初始化，原BC/Q课程7/.05→2.5/.45；每25轮fixed20及保存。current causal-AR Stage1、全新Stage2空池开始。DVAC仅成功episode来源reference BC，alpha_local/chunk=1/1、success_scale=1；两层方向在critic actor槽70000由+1硬切为−1，失败query仍均权。

前期teacher采集、后期student采集；fixed始终测student，每25轮20条。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[五张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放、图像、视频不入包，仍保留在服务器。本次未删除checkpoint；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-direction-phys7-fresh600-20260913-v1`

生产源码：`2be9a6d542b0688f3092ea67a4f7b4c07944107e`；归档前HEAD：`4246eaf8580ce49410af18f97dfed8382737975f`；分支：`codex/sz-rlt-dvac-direction-20260913`。
