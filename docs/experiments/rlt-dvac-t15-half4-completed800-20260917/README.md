# π0 RLT · DVAC τ=1.5 · 成功倍率1 · 累计800轮 收尾

累计800轮正常完成（exit0）。最终已记录采集R800，原计划800轮；最新固定评估R800：19/20。最近checkpoint目录R800；目录清单并非模型加载/完整恢复验证。runtime退出码为0；终止原因记录为`configured_cumulative_endpoint_800`，原始结束时间2026-09-17T02:04:46+00:00。

4 env、B512/micro256、UTD5、critic:actor=2:1、LR1e-4 constant；10k回放门槛、15k critic初始化、每轮cap800、回放80k；BC/Q课程10k+25k，7/.05→2.5/.45。DVAC exp_mean双层温度=1.5，成功BC倍率=1.0；每25轮fixed20及保存。Stage1沿用已有CP2000；续训在CP425恢复模型、优化器、LR、target-Q、回放和RLT计数，到累计800正常结束。

主曲线严格拼接旧R1–425和新R426–800。旧run R426–443保留在segments/initial原始日志和metrics.json中，不重复计入主线。恢复完整训练状态，不等于环境与回放随机数逐位复现。

前期teacher、后期student采集；fixed始终测student（20 episodes）。R425后为续训。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[五张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](segments/resume/raw/runtime/driver.log) · [实际配置](segments/resume/raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放及环境图像/视频不入包，仍保留在服务器。本次未读取模型或池payload、未删除checkpoint，所有末代文件保留；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-exp-t15-half4env-resume425to800-phys6-20260916-v1`

生产源码：`703408b1f525645b58451e90404973a75cf0e41a`；归档前HEAD：`aeb947569e3fe0100c906026ce4690cdbaccc3d7`；分支：`codex/sz-rlt-t2-resume-t15-half4-800-20260916`。
