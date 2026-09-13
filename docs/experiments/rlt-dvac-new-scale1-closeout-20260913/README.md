# π0 RLT · DVAC new scale1 收尾

用户主动中止。最终已记录采集R496，原计划600轮；最新固定评估R475：17/20。最近checkpoint目录R475；目录清单并非模型加载/完整恢复验证。runtime退出码为134。

600轮、8 env × 1 wave，执行C10，B512/micro256、UTD5、critic:actor=2:1、actor/critic LR均1e-4；回放80k、采集20k transitions开始更新、初始化30k critic更新，原BC/Q课程7/0.05→2.5/0.45；每25轮固定评估20条、保存25。current causal-AR Stage1、Stage2空池启动。DVAC new两级线性MinMax中心化，alpha_local/chunk=1/1，success_scale=1；完整512 batch成功query外层，失败query均权。

前期Teacher采集，后期Student采集；固定评估始终Student，每25轮20条。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[四张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放、图像、视频不入包，仍保留在服务器。本次未删除checkpoint；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-new-phys7-fresh600-20260912-v1`

生产源码：`5e5882e60b4364e8fd35bcf2321c201294435d0c`；归档前HEAD：`5e5882e60b4364e8fd35bcf2321c201294435d0c`；分支：`codex/sz-rlt-dvac-new-20260912`。
