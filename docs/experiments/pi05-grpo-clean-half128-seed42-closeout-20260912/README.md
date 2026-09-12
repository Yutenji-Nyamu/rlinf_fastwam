# π0.5 Clean GRPO · 128条/轮 · seed42 收尾

按用户要求中止，最终采集完成R141，原计划200轮。最新固定评估R140：18/32，最近checkpoint目录为R140。不把主动中止写为自然完成；runtime退出码为134。

配置：64并行×2串行=128条/轮，G8、U2、B512/micro32、LR5e-6；每5轮固定32条评估、每10轮保存；原模型启动，rollout seed42+逻辑rank，训练/评估RNG隔离，历史环境实现。没有Prism或DVAC。

[四张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

本ZIP含完整日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和配置、源码身份及checkpoint文件目录。大模型、optimizer、回放、图像和视频不入包，仍保留在服务器；该ZIP不能独立恢复训练。本次归档没有删除checkpoint。

run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-clean-half128-seed42-formal200-phys45-20260911-v1`

生产源码：`76c2ac4c40dd211d6e6eaab84022e75a61ea5ed7`；归档前HEAD：`9b4cfd1c985bc8746461aeba00a7d24b5e964848`；分支：`codex/sz-pi05-grpo-clean-half-seed42-20260911`。
