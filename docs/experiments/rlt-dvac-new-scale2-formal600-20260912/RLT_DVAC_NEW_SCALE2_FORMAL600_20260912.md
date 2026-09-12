# RLT DVAC new success_scale=2 · GPU6 · 600轮

用户授权以本组替换GPU6 clean RLT；GPU7 scale=1继续。先准备、连贯切换，再归档旧组。此文件维护本实验上下文。

## 启动约定

- 独立分支：`codex/sz-rlt-dvac-new-scale2-20260912`。
- 生产源码：`5e5882e60b4364e8fd35bcf2321c201294435d0c`，与GPU7 DVAC new一致，无算法代码变更。
- 从相同current-AR Stage1权重开始全新Stage2，空回放池，不续训clean。
- 保留clean单卡设置：8环境、batch 512、microbatch 256、UTD 5、600轮、每25轮评估/保存；BC/Q课程与优化器不变。
- 方法：DVAC new，两级中心化MinMax；`alpha_local=1`、`alpha_chunk=1`、`success_scale=2`。外层在完整batch成功query内比较，失败BC权重仍为1。
- GPU：6。命名空间：`RLinf_rlt_dvac_new_scale2_single_gpu6_fresh600_20260912`。
- 输出：`/data/chenyiteng/results/rlinf-rlt/current-single-gpu-dvac-new-scale2-phys6-fresh600-20260912-v1`。
- 启动入口：`/data/chenyiteng/results/server-maintenance-20260912/rlt-scale2/formal_ops_scale2.py launch formal`；实际封装命令和resolved配置见下方证据。
- 正常停止：600轮；若启动失败，仅处理本组精确driver/namespace，保持共享Ray及其他实验。

## 必要检查

完整512×10 CPU检查通过：成功权重、成功BC损失及梯度均准确翻倍；失败项不变；全失败batch不变。复用同源码已通过的GPU learner smoke（2次actor、4次critic更新），本轮只调整已有参数。

与GPU7 scale=1实际配置相比，差异仅success_scale及卡号/输出路径。与clean实际配置相比，仅方法及运行身份差异。两份实际运行配置补出的3个false默认字段也已核对。

## 证据

- [配置差异](scale1-config-diff.json)
- [clean配置差异](formal-method-identity-diff.json)
- [resolved配置](resolved.yaml)
- [实际启动命令](command.txt)
- [CPU检查](scale2-cpu-result.json)

## 已执行

2026-09-12 20:10:18 CST向旧clean的唯一driver发出SIGTERM；20:10:23.028确认driver、旧namespace和GPU6进程全部退出。20:10:23.344发起新wrapper，退出确认到新组启动约0.32秒（不含模型加载）。其他GPU进程身份和Ray actor保持。

20:12:44启动检查通过：12个actor存活、rollout进入generate、环境进入interact、实配无意外差异、无fatal/退出码，其他进程保持。到此停止盯跑。新组尚处于原有teacher采集/回放预热阶段；此回执确认健康启动，不声称已完成正式actor更新。

旧clean完整指标至R373；最近固定评估R350为18/20，末代checkpoint R350。用户主动中止（exit 15），并非自然完成600轮。模型和回放保留，本次无checkpoint删除。日志、events、所有标量、配置、源码身份和四张独立图归档于 [旧clean归档](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz-rlt-control-single-gpu600-20260912/docs/experiments/rlt-clean-single-gpu600-closeout-20260912)。

- [启动通过回执](STARTUP_VERIFIED.json)
- [停止旧组](stop-clean-receipt.json) · [启动新组](launch-receipt.json)
- [旧组归档摘要](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz-rlt-control-single-gpu600-20260912/docs/experiments/rlt-clean-single-gpu600-closeout-20260912/SUMMARY.json)
- [旧clean R373主要产物ZIP，3.44 MB](https://github.com/Yutenji-Nyamu/rlinf_fastwam/blob/codex/sz-rlt-control-single-gpu600-20260912/docs/experiments/rlt-clean-single-gpu600-closeout-20260912/rlt-clean-single-gpu600-R373-closeout-20260912.zip)
- [旧clean四张单列图](https://github.com/Yutenji-Nyamu/rlinf_fastwam/tree/codex/sz-rlt-control-single-gpu600-20260912/docs/experiments/rlt-clean-single-gpu600-closeout-20260912/plots)

新旧分支的Git发布回执归入同证据目录；生产算法源码保持不变。
