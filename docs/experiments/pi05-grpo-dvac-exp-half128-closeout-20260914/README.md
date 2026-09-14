# π0.5 GRPO128 · DVAC exp_mean 收尾

用户主动中止。最终已记录采集R109，原计划200轮；最新固定评估R105：16/32。最近checkpoint目录R100；目录清单并非模型加载/完整恢复验证。runtime退出码为134。

200轮、64并行×2串行=128条/轮，G8、U2、B512/micro32、LR5e-6；每5轮fixed32、每10轮保存；原模型fresh、rollout seed42+rank。DVAC exp_mean、两层temperature=.5、alpha=1，原优势、mask、同scene group与整chunk裁剪保持。

采集与固定评估分开；每5轮fixed32，曲线按真实轮次。 最后一轮采集与尾部优化标量可能不同，逐tag截止已写入SUMMARY.json，不将部分尾部更新冒充完整训练轮。

[五张单列宽图](plots/index.html) · [完整标量CSV](metrics.csv) · [原始日志](raw/runtime/driver.log) · [实际配置](raw/tensorboard/config.yaml)

ZIP包含完整driver日志、原始TensorBoard events、全部标量CSV/JSON、运行命令和实配、源码版本与文件SHA、checkpoint文件目录及本说明。模型、optimizer、回放、图像、视频不入包，仍保留在服务器。本次未删除checkpoint；此包不用于独立恢复训练。

服务器run：`/data/chenyiteng/results/rlinf-shenzhen/pi05-sidney/runs/pi05-grpo-dvac-exp-half128-seed42-t05-formal200-phys45-20260913-v1`

生产源码：`7f23ac2c8915a714b8e42edf75a21477cfbfec78`；归档前HEAD：`06b2f21e1f1afb6e28ba2341240dbd024df91d8f`；分支：`codex/sz-pi05-grpo-dvac-exp-20260913`。
